#include "vm_c/core/scheduler.hpp"
#include <algorithm>
#include <spdlog/spdlog.h>

namespace vm_c {

Scheduler::Scheduler(const VmCConfig& config,
                     std::shared_ptr<KVCacheManager> kv_cache_mgr)
    : config_(config)
    , kv_cache_mgr_(std::move(kv_cache_mgr))
    , max_num_seqs_(config.server.max_num_seqs > 0 ? config.server.max_num_seqs : 1)
    , max_num_batched_tokens_(config.server.max_num_batched_tokens)
    , enable_chunked_prefill_(config.scheduler.enable_chunked_prefill)
    , long_prefill_token_threshold_(config.scheduler.long_prefill_token_threshold)
    , decode_first_(config.scheduler.scheduling_policy == "decode_first")
    , max_prefill_per_batch_(config.scheduler.max_prefill_per_batch)
    , decode_latency_target_ms_(config.scheduler.decode_latency_target_ms)
    , preempt_prefill_on_latency_(config.scheduler.preempt_prefill_on_latency)
    , decode_reserved_tokens_(config.scheduler.decode_reserved_tokens) {
    // Prefix caching 在 TurboQuant 模式下也启用：
    // TurboQuant 的 KV cache store 路径是独立的（store_kv），
    // prefix caching 的 block hash 匹配只影响 block 分配，不影响数据格式。
    // 参考 vLLM 的实现：TurboQuant 模式下也支持 prefix caching。
    prefix_caching_enabled_ = true;
    if (max_num_batched_tokens_ < max_num_seqs_) {
        spdlog::error("--max-num-batched-tokens ({}) must be >= --max-num-seqs ({}). "
                      "Increase --max-num-batched-tokens or decrease --max-num-seqs.",
                      max_num_batched_tokens_, max_num_seqs_);
        std::exit(1);
    }
}

// ── Prefix Caching 哈希计算 ──
// 使用 FNV-1a 64-bit 哈希，链式计算（每个block的哈希依赖前一个block的哈希）
// 参考 vLLM 的 BlockHashWithGroupId 实现
std::vector<uint64_t> Scheduler::compute_block_hashes(
    const std::vector<int32_t>& token_ids, int num_tokens) const {
    int block_size = kv_cache_mgr_->block_size();
    int num_blocks = num_tokens / block_size;
    std::vector<uint64_t> hashes(num_blocks);

    for (int b = 0; b < num_blocks; ++b) {
        uint64_t hash = 0xcbf29ce484222325ULL;  // FNV-1a offset basis
        if (b > 0) {
            hash = hashes[b - 1];  // 链式哈希：依赖前一个block
        }
        for (int i = 0; i < block_size; ++i) {
            int token_idx = b * block_size + i;
            hash ^= static_cast<uint64_t>(token_ids[token_idx]);
            hash *= 0x100000001b3ULL;  // FNV-1a prime
        }
        hashes[b] = hash;
    }
    return hashes;
}

// ── 检查 decode 延迟是否可能超标，触发 prefill 抢占 ──
bool Scheduler::check_decode_latency_preempt(SchedulerOutput& output) {
    if (!preempt_prefill_on_latency_ || decode_latency_target_ms_ <= 0) {
        return false;
    }

    auto now = std::chrono::steady_clock::now();

    // 遍历 running_queue 中 decode 请求，检查等待时间
    for (auto& req : running_queue_) {
        if (req->status != RequestStatus::RUNNING) continue;
        if (req->is_prefill()) continue;

        // 尚未产出 decode token 时不做延迟抢占（避免把 prefill 耗时误判为 decode 等待）
        if (req->num_output_tokens == 0) continue;

        if (req->last_token_time == std::chrono::steady_clock::time_point{}) {
            continue;
        }

        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - req->last_token_time).count();

        // 如果 decode 等待时间超过目标延迟，触发抢占
        if (elapsed_ms > decode_latency_target_ms_) {
            spdlog::warn(
                "Decode latency target exceeded: req={} wait={}ms target={}ms, "
                "preempting prefill to free budget",
                req->request_id, elapsed_ms, decode_latency_target_ms_);

            output.decode_preempted = true;
            return true;
        }
    }
    return false;
}

// ── decode-first 调度策略 ──
// 1. 先调度所有 decode 请求（每个 1 token），保证 decode 延迟
// 2. 用剩余 budget 调度 prefill chunk
// 3. 如果 decode 延迟超标，触发 preempt_prefill 释放更多 budget
void Scheduler::schedule_decode_first(SchedulerOutput& output) {
    int budget_left = max_num_batched_tokens_;

    // ── 阶段一：调度 running_queue 中所有 decode 请求 ──
    int num_decode_scheduled = 0;
    for (auto& req : running_queue_) {
        if (req->status != RequestStatus::RUNNING) continue;
        if (req->is_prefill()) continue;  // 跳过 prefill，后面处理

        // 每个 decode 请求固定 1 个 token
        int tokens = 1;
        if (tokens > budget_left) {
            // budget 不足时，检查是否应触发 preempt_prefill
            if (budget_left <= 0) {
                // 完全无 budget，跳过（后续可能触发抢占）
                continue;
            }
            // 部分 budget 不足，将 decode 推迟到下一个 schedule
            continue;
        }

        output.scheduled_running_reqs.push_back(req);
        output.num_scheduled_tokens[req->request_id] = tokens;
        output.total_num_scheduled_tokens += tokens;
        budget_left -= tokens;
        num_decode_scheduled++;
    }
    output.num_decode_in_batch = num_decode_scheduled;

    // ── 阶段二：检查 decode 延迟是否超标 → 触发 prefill 抢占 ──
    // 如果 decode 的 token 预算被 prefill 挤占严重，且 decode 等待时间超标，
    // 则不再调度新的 prefill，把全部 budget 留给 decode。
    bool preempt_prefill = check_decode_latency_preempt(output);
    if (preempt_prefill) {
        // 不再调度 prefill，所有后续 budget 留给 decode
        // 当前 batch 中已有的 prefill 继续执行，但不追加新 prefill
        spdlog::debug("Decode latency preemption triggered: reserving remaining budget={} for decode", budget_left);
        return;
    }

    // ── 阶段三：调度 running_queue 中的 prefill chunk ──
    int prefill_budget = budget_left;
    if (decode_reserved_tokens_ > 0) {
        // 为 decode 预留 token 预算
        prefill_budget = std::max(0, budget_left - decode_reserved_tokens_);
    }
    if (max_prefill_per_batch_ > 0) {
        prefill_budget = std::min(prefill_budget, max_prefill_per_batch_);
    }

    int num_prefill_from_running = 0;
    if (prefill_budget > 0) {
        for (auto& req : running_queue_) {
            if (req->status != RequestStatus::RUNNING) continue;
            if (!req->is_prefill()) continue;  // 只处理 prefill

            int remaining = req->num_tokens_to_compute();
            int chunk_size = remaining;
            if (long_prefill_token_threshold_ > 0 && chunk_size > long_prefill_token_threshold_) {
                chunk_size = long_prefill_token_threshold_;
            }
            chunk_size = std::min(chunk_size, prefill_budget);
            if (chunk_size <= 0) continue;

            output.scheduled_running_reqs.push_back(req);
            output.num_scheduled_tokens[req->request_id] = chunk_size;
            output.total_num_scheduled_tokens += chunk_size;
            prefill_budget -= chunk_size;
            budget_left -= chunk_size;
            num_prefill_from_running++;
        }
    }

    // ── 阶段四：从 waiting_queue 调度新请求 ──
    int remaining_tokens = budget_left;
    int preempted_count = 0;
    for (auto& req : running_queue_) {
        if (req->status == RequestStatus::PREEMPTED) preempted_count++;
    }
    int remaining_seqs = max_num_seqs_ - static_cast<int>(running_queue_.size()) + preempted_count;

    auto it = waiting_queue_.begin();
    while (it != waiting_queue_.end() && remaining_tokens > 0 && remaining_seqs > 0) {
        auto& req = *it;
        int num_tokens = req->num_prompt_tokens();

        int64_t max_model_len = config_.model.max_model_len;
        if (num_tokens > max_model_len) {
            int truncate_count = num_tokens - static_cast<int>(max_model_len);
            req->prompt_token_ids.erase(req->prompt_token_ids.begin(),
                                        req->prompt_token_ids.begin() + truncate_count);
            num_tokens = static_cast<int>(max_model_len);
            spdlog::warn("Request {} prompt truncated: {} -> {} tokens (max_model_len={})",
                         req->request_id, num_tokens + truncate_count, num_tokens, max_model_len);
        }

        int cached_tokens = 0;
        std::vector<int64_t> computed_block_ids;
        std::vector<uint64_t> block_hashes;

        if (prefix_caching_enabled_ && num_tokens > 0) {
            int block_size = kv_cache_mgr_->block_size();
            int max_cache_hit_length = num_tokens - 1;
            int max_prefix_blocks = max_cache_hit_length / block_size;

            if (max_prefix_blocks > 0) {
                block_hashes = compute_block_hashes(req->prompt_token_ids, num_tokens);
                prefix_cache_stats_.total_queries++;

                std::vector<int64_t> hit_block_ids;
                int hit_blocks = kv_cache_mgr_->find_longest_prefix_hit(
                    block_hashes, max_prefix_blocks, hit_block_ids);

                if (hit_blocks > 0) {
                    cached_tokens = hit_blocks * block_size;
                    computed_block_ids = std::move(hit_block_ids);
                    prefix_cache_stats_.total_hits++;
                }
            }
        }

        int tokens_to_compute = num_tokens - cached_tokens;

        // 考虑 max_prefill_per_batch 限制
        int new_prefill_budget = remaining_tokens;
        if (max_prefill_per_batch_ > 0) {
            int already_scheduled_prefill = output.total_num_scheduled_tokens - output.num_decode_in_batch;
            new_prefill_budget = std::min(new_prefill_budget, max_prefill_per_batch_ - already_scheduled_prefill);
        }
        if (new_prefill_budget <= 0) break;

        int slots_needed = tokens_to_compute > 0 ? tokens_to_compute : 1;
        if (!can_allocate_slots(slots_needed)) {
            spdlog::warn(
                "[SCHED] waiting req {} cannot allocate {} tokens ({} blocks), free_blocks={}",
                req->request_id, slots_needed,
                (slots_needed + kv_cache_mgr_->block_size() - 1) / kv_cache_mgr_->block_size(),
                kv_cache_mgr_->num_free_gpu_blocks());
            preempt_requests(slots_needed);
            if (!can_allocate_slots(slots_needed)) {
                ++it;
                continue;
            }
        }

        if (req->status == RequestStatus::PREEMPTED) {
            kv_cache_mgr_->swap_back_request_slots(req->request_id);
        }
        req->status = RequestStatus::RUNNING;
        if (cached_tokens == 0) {
            req->arrival_time = std::chrono::steady_clock::now();
        }
        if (cached_tokens > 0) {
            req->num_computed_tokens = cached_tokens;
            output.num_cached_tokens[req->request_id] = cached_tokens;
            output.total_num_cached_tokens += cached_tokens;
            if (!block_hashes.empty()) {
                output.prefix_block_hashes[req->request_id] = std::move(block_hashes);
            }
            output.computed_block_ids[req->request_id] = std::move(computed_block_ids);

            spdlog::debug("Prefix cache hit for request {}: {}/{} tokens cached",
                         req->request_id, cached_tokens, num_tokens);
        }

        int chunk_size = tokens_to_compute;
        if (enable_chunked_prefill_) {
            if (long_prefill_token_threshold_ > 0 && chunk_size > long_prefill_token_threshold_) {
                chunk_size = long_prefill_token_threshold_;
            }
            chunk_size = std::min(chunk_size, new_prefill_budget);
        } else {
            if (chunk_size > new_prefill_budget) {
                ++it;
                continue;
            }
        }
        if (chunk_size <= 0 && cached_tokens > 0) {
            chunk_size = std::min(1, new_prefill_budget);
        }

        if (chunk_size > 0) {
            if (cached_tokens > 0) {
                output.scheduled_resumed_reqs.push_back(req);
            } else {
                output.scheduled_new_reqs.push_back(req);
            }
            output.num_scheduled_tokens[req->request_id] = chunk_size;
            output.total_num_scheduled_tokens += chunk_size;
        }

        running_queue_.push_back(req);
        it = waiting_queue_.erase(it);

        remaining_tokens -= chunk_size;
        remaining_seqs--;
    }

    output.num_prefill_in_batch = output.total_num_scheduled_tokens - output.num_decode_in_batch;
}

// ── 主调度入口 ──
SchedulerOutput Scheduler::schedule() {
    SchedulerOutput output;

    if (decode_first_) {
        schedule_decode_first(output);
    } else {
        // ── 传统调度（先 running_queue 后 waiting_queue，不区分 decode/prefill） ──
        int budget_left = max_num_batched_tokens_;

        for (auto& req : running_queue_) {
            if (req->status != RequestStatus::RUNNING) continue;
            if (budget_left <= 0) break;

            if (req->is_prefill()) {
                int remaining = req->num_tokens_to_compute();
                int chunk_size = remaining;
                if (long_prefill_token_threshold_ > 0 && chunk_size > long_prefill_token_threshold_) {
                    chunk_size = long_prefill_token_threshold_;
                }
                chunk_size = std::min(chunk_size, budget_left);
                if (chunk_size <= 0) continue;

                output.scheduled_running_reqs.push_back(req);
                output.num_scheduled_tokens[req->request_id] = chunk_size;
                output.total_num_scheduled_tokens += chunk_size;
                budget_left -= chunk_size;
            } else {
                int tokens = 1;
                if (tokens > budget_left) continue;
                output.scheduled_running_reqs.push_back(req);
                output.num_scheduled_tokens[req->request_id] = tokens;
                output.total_num_scheduled_tokens += tokens;
                budget_left -= tokens;
            }
        }

        int remaining_tokens = max_num_batched_tokens_ - output.total_num_scheduled_tokens;
        int preempted_count = 0;
        for (auto& req : running_queue_) {
            if (req->status == RequestStatus::PREEMPTED) preempted_count++;
        }
        int remaining_seqs = max_num_seqs_ - static_cast<int>(running_queue_.size()) + preempted_count;

        auto it = waiting_queue_.begin();
        while (it != waiting_queue_.end() && remaining_tokens > 0 && remaining_seqs > 0) {
            auto& req = *it;
            int num_tokens = req->num_prompt_tokens();

            int64_t max_model_len = config_.model.max_model_len;
            if (num_tokens > max_model_len) {
                int truncate_count = num_tokens - static_cast<int>(max_model_len);
                req->prompt_token_ids.erase(req->prompt_token_ids.begin(),
                                            req->prompt_token_ids.begin() + truncate_count);
                num_tokens = static_cast<int>(max_model_len);
            }

            int cached_tokens = 0;
            std::vector<int64_t> computed_block_ids;
            std::vector<uint64_t> block_hashes;

            if (prefix_caching_enabled_ && num_tokens > 0) {
                int block_size = kv_cache_mgr_->block_size();
                int max_cache_hit_length = num_tokens - 1;
                int max_prefix_blocks = max_cache_hit_length / block_size;
                if (max_prefix_blocks > 0) {
                    block_hashes = compute_block_hashes(req->prompt_token_ids, num_tokens);
                    std::vector<int64_t> hit_block_ids;
                    int hit_blocks = kv_cache_mgr_->find_longest_prefix_hit(
                        block_hashes, max_prefix_blocks, hit_block_ids);
                    if (hit_blocks > 0) {
                        cached_tokens = hit_blocks * block_size;
                        computed_block_ids = std::move(hit_block_ids);
                    }
                }
            }

            int tokens_to_compute = num_tokens - cached_tokens;

            int slots_needed = tokens_to_compute > 0 ? tokens_to_compute : 1;
            int64_t free_blocks = kv_cache_mgr_->num_free_gpu_blocks();
            int64_t needed_blocks = (slots_needed + kv_cache_mgr_->block_size() - 1) / kv_cache_mgr_->block_size();
            if (!can_allocate_slots(slots_needed)) {
                spdlog::warn("[SCHED] Cannot allocate {} blocks (free={}) for req {} ({} tokens)",
                             needed_blocks, free_blocks, req->request_id, num_tokens);
                preempt_requests(slots_needed);
                if (!can_allocate_slots(slots_needed)) {
                    spdlog::warn("[SCHED] Still cannot allocate after preemption, skipping req {}",
                                 req->request_id);
                    ++it;
                    continue;
                }
            }

            if (req->status == RequestStatus::PREEMPTED) {
                kv_cache_mgr_->swap_back_request_slots(req->request_id);
            }
            req->status = RequestStatus::RUNNING;
            if (cached_tokens == 0) {
                req->arrival_time = std::chrono::steady_clock::now();
            }
            if (cached_tokens > 0) {
                req->num_computed_tokens = cached_tokens;
                output.num_cached_tokens[req->request_id] = cached_tokens;
                output.total_num_cached_tokens += cached_tokens;
                if (!block_hashes.empty()) {
                    output.prefix_block_hashes[req->request_id] = std::move(block_hashes);
                }
                output.computed_block_ids[req->request_id] = std::move(computed_block_ids);
            }

            int chunk_size = tokens_to_compute;
            if (enable_chunked_prefill_) {
                if (long_prefill_token_threshold_ > 0 && chunk_size > long_prefill_token_threshold_) {
                    chunk_size = long_prefill_token_threshold_;
                }
                chunk_size = std::min(chunk_size, remaining_tokens);
            } else {
                if (chunk_size > remaining_tokens) { ++it; continue; }
            }
            if (chunk_size <= 0 && cached_tokens > 0) {
                chunk_size = std::min(1, remaining_tokens);
            }

            if (chunk_size > 0) {
                if (cached_tokens > 0) {
                    output.scheduled_resumed_reqs.push_back(req);
                } else {
                    output.scheduled_new_reqs.push_back(req);
                }
                output.num_scheduled_tokens[req->request_id] = chunk_size;
                output.total_num_scheduled_tokens += chunk_size;
            }

            running_queue_.push_back(req);
            it = waiting_queue_.erase(it);
            remaining_tokens -= chunk_size;
            remaining_seqs--;
        }

        output.num_decode_in_batch = 0; // 传统模式不统计
        output.num_prefill_in_batch = output.total_num_scheduled_tokens;
    }

    return output;
}

void Scheduler::add_request(RequestPtr req) {
    req->status = RequestStatus::WAITING;
    waiting_queue_.push_back(req);
}

void Scheduler::abort_request(const std::string& request_id) {
    auto it = std::find_if(waiting_queue_.begin(), waiting_queue_.end(),
                           [&](const RequestPtr& r) { return r->request_id == request_id; });
    if (it != waiting_queue_.end()) {
        spdlog::info("Abort waiting request {}", request_id);
        waiting_queue_.erase(it);
        return;
    }
    auto rit = std::find_if(running_queue_.begin(), running_queue_.end(),
                            [&](const RequestPtr& r) { return r->request_id == request_id; });
    if (rit != running_queue_.end()) {
        (*rit)->status = RequestStatus::FINISHED;
        (*rit)->finish_reason = FinishReason::ABORTED;
        kv_cache_mgr_->free_slots(request_id);
        running_queue_.erase(rit);
        spdlog::info("Abort running request {}", request_id);
    }
}

RequestPtr Scheduler::find_request(const std::string& request_id) {
    for (auto& r : waiting_queue_) {
        if (r->request_id == request_id) return r;
    }
    for (auto& r : running_queue_) {
        if (r->request_id == request_id) return r;
    }
    return nullptr;
}

void Scheduler::finish_request(const std::string& request_id, FinishReason reason) {
    auto it = std::find_if(running_queue_.begin(), running_queue_.end(),
                           [&](const RequestPtr& r) { return r->request_id == request_id; });
    if (it != running_queue_.end()) {
        auto& req = *it;
        req->status = RequestStatus::FINISHED;
        req->finish_reason = reason;

        if (prefix_caching_enabled_ && !req->prompt_token_ids.empty()) {
            int block_size = kv_cache_mgr_->block_size();
            int num_full_blocks = static_cast<int>(req->prompt_token_ids.size()) / block_size;
            if (num_full_blocks > 0) {
                auto block_hashes = compute_block_hashes(req->prompt_token_ids, num_full_blocks * block_size);
                int num_cached = 0;
                auto slots_it = kv_cache_mgr_->request_slots().find(req->request_id);
                if (slots_it != kv_cache_mgr_->request_slots().end()) {
                    num_full_blocks = std::min(num_full_blocks, static_cast<int>(slots_it->second.size()));
                }
                kv_cache_mgr_->cache_full_blocks(request_id, block_hashes, num_cached, num_full_blocks);
                prefix_cache_stats_.total_blocks_cached += num_full_blocks;
            }
        }

        kv_cache_mgr_->free_slots(request_id);
        running_queue_.erase(it);
    }
}

void Scheduler::update_from_output(const std::vector<RequestOutput>& outputs) {
    const auto now = std::chrono::steady_clock::now();
    for (auto& out : outputs) {
        auto it = std::find_if(running_queue_.begin(), running_queue_.end(),
                               [&](const RequestPtr& r) { return r->request_id == out.request_id; });
        if (it != running_queue_.end()) {
            auto& req = *it;
            req->output_token_ids.insert(req->output_token_ids.end(),
                                           out.output_token_ids.begin(),
                                           out.output_token_ids.end());
            req->num_output_tokens = static_cast<int>(req->output_token_ids.size());
            req->num_computed_tokens += out.num_new_tokens;

            if (!out.output_token_ids.empty()) {
                if (req->first_token_time == std::chrono::steady_clock::time_point{}) {
                    req->first_token_time = now;
                }
                req->last_token_time = now;
            }

            if (prefix_caching_enabled_ && req->is_prefill() && out.prefill_complete) {
                int block_size = kv_cache_mgr_->block_size();
                int total_tokens = req->num_computed_tokens;
                int num_full_blocks = total_tokens / block_size;
                if (num_full_blocks > 0) {
                    auto block_hashes = compute_block_hashes(req->prompt_token_ids, num_full_blocks * block_size);
                    int num_cached = 0;
                    auto slots_it = kv_cache_mgr_->request_slots().find(req->request_id);
                    if (slots_it != kv_cache_mgr_->request_slots().end()) {
                        num_full_blocks = std::min(num_full_blocks, static_cast<int>(slots_it->second.size()));
                    }
                    kv_cache_mgr_->cache_full_blocks(req->request_id, block_hashes, num_cached, num_full_blocks);
                    prefix_cache_stats_.total_blocks_cached += num_full_blocks;
                }
            }

            if (out.finished) {
                finish_request(out.request_id, out.finish_reason);
            }
        }
    }
}

int Scheduler::num_waiting_requests() const {
    return static_cast<int>(waiting_queue_.size());
}

int Scheduler::num_running_requests() const {
    return static_cast<int>(running_queue_.size());
}

bool Scheduler::can_allocate_slots(int num_tokens) const {
    int64_t blocks_needed = (num_tokens + kv_cache_mgr_->block_size() - 1) / kv_cache_mgr_->block_size();
    return kv_cache_mgr_->num_free_gpu_blocks() >= blocks_needed;
}

void Scheduler::preempt_requests(int num_tokens_needed) {
    while (!running_queue_.empty() && !can_allocate_slots(num_tokens_needed)) {
        auto& victim = running_queue_.back();
        spdlog::debug("Preempting request {} (computed_tokens={})", victim->request_id, victim->num_computed_tokens);
        victim->status = RequestStatus::PREEMPTED;
        victim->num_computed_tokens = 0;
        // Swap KV cache to CPU instead of discarding it (preserves computed tokens)
        kv_cache_mgr_->swap_request_slots(victim->request_id);
        waiting_queue_.push_front(victim);
        running_queue_.pop_back();
    }
}

}
