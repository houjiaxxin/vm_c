#include "vm_c/memory/expert_cache.hpp"
#include "vm_c/cuda/gpu_arch.hpp"
#include <spdlog/spdlog.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include <cstring>

namespace vm_c {

static cudaEvent_t as_cuda_event(void* p) { return static_cast<cudaEvent_t>(p); }

ExpertCacheManager::ExpertCacheManager(const ExpertCacheConfig& config,
                                       int num_experts,
                                       int num_experts_per_tok,
                                       int hidden_size,
                                       int moe_intermediate_size)
    : config_(config)
    , num_experts_(num_experts)
    , num_experts_per_tok_(num_experts_per_tok)
    , hidden_size_(hidden_size)
    , moe_intermediate_size_(moe_intermediate_size)
    , cpu_weights_(num_experts)
    , access_freq_ema_(num_experts, 0.0f)
    , prediction_scores_(num_experts, 0.0f)
    , gate_proj_bytes_per_expert_(num_experts, 0)
    , up_proj_bytes_per_expert_(num_experts, 0)
    , down_proj_bytes_per_expert_(num_experts, 0)
{
    if (config_.gpu_cache_capacity <= 0) {
        config_.gpu_cache_capacity = std::max(num_experts_per_tok, 8);
    }
    if (config_.gpu_cache_capacity > num_experts) {
        config_.gpu_cache_capacity = num_experts;
    }
    if (config_.update_period_tokens <= 0) {
        config_.update_period_tokens = ExpertCacheConfig::DEFAULT_UPDATE_PERIOD_TOKENS;
    }
    if (config_.history_window_tokens <= 0) {
        config_.history_window_tokens = config_.update_period_tokens * 2;
    }
    if (config_.min_residency_tokens <= 0) {
        config_.min_residency_tokens = config_.update_period_tokens * 2;
    }
    if (config_.min_update_period <= 0) {
        config_.min_update_period = config_.update_period_tokens / 2;
    }
    if (config_.max_update_period <= 0) {
        config_.max_update_period = config_.update_period_tokens * 4;
    }
    if (config_.lazy_eviction_grace_tokens <= 0) {
        config_.lazy_eviction_grace_tokens = config_.update_period_tokens;
    }
}

ExpertCacheManager::~ExpertCacheManager() {
    for (auto& slot : gpu_slots_) {
        if (slot.weights.gate_proj) CUDA_CHECK(cudaFree(slot.weights.gate_proj));
        if (slot.weights.up_proj) CUDA_CHECK(cudaFree(slot.weights.up_proj));
        if (slot.weights.down_proj) CUDA_CHECK(cudaFree(slot.weights.down_proj));
    }
    for (auto ev : sync_events_) CUDA_CHECK(cudaEventDestroy(as_cuda_event(ev)));
    sync_events_.clear();
    if (prefetch_stream_) CUDA_CHECK(cudaStreamDestroy(prefetch_stream_));
    if (prefetch_event_) CUDA_CHECK(cudaEventDestroy(as_cuda_event(prefetch_event_)));
}

void ExpertCacheManager::allocate_gpu_cache(
    size_t gate_proj_bytes_per_expert,
    size_t up_proj_bytes_per_expert,
    size_t down_proj_bytes_per_expert)
{
    for (int i = 0; i < num_experts_; ++i) {
        gate_proj_bytes_per_expert_[i] = gate_proj_bytes_per_expert;
        up_proj_bytes_per_expert_[i] = up_proj_bytes_per_expert;
        down_proj_bytes_per_expert_[i] = down_proj_bytes_per_expert;
    }

    gpu_slots_.resize(config_.gpu_cache_capacity);
    for (auto& slot : gpu_slots_) {
        slot.expert_id = -1;
        slot.loading = false;
        slot.pending_eviction = false;
        slot.eviction_grace_remaining = 0;
        if (gate_proj_bytes_per_expert > 0) {
            CUDA_CHECK(cudaMalloc(&slot.weights.gate_proj, gate_proj_bytes_per_expert));
            slot.weights.gate_proj_bytes = gate_proj_bytes_per_expert;
        }
        if (up_proj_bytes_per_expert > 0) {
            CUDA_CHECK(cudaMalloc(&slot.weights.up_proj, up_proj_bytes_per_expert));
            slot.weights.up_proj_bytes = up_proj_bytes_per_expert;
        }
        if (down_proj_bytes_per_expert > 0) {
            CUDA_CHECK(cudaMalloc(&slot.weights.down_proj, down_proj_bytes_per_expert));
            slot.weights.down_proj_bytes = down_proj_bytes_per_expert;
        }
        slot.tokens_since_insertion = 0;
    }

    CUDA_CHECK(cudaStreamCreateWithFlags(&prefetch_stream_, cudaStreamNonBlocking));
    cudaEvent_t ev;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    prefetch_event_ = ev;

    spdlog::info("ExpertCache: allocated {} GPU slots, gate={}MB up={}MB down={}MB per expert",
                 config_.gpu_cache_capacity,
                 gate_proj_bytes_per_expert / (1024*1024),
                 up_proj_bytes_per_expert / (1024*1024),
                 down_proj_bytes_per_expert / (1024*1024));
}

void ExpertCacheManager::set_cpu_expert_weights(
    int expert_id,
    void* cpu_gate_proj, size_t gate_bytes,
    void* cpu_up_proj, size_t up_bytes,
    void* cpu_down_proj, size_t down_bytes)
{
    if (expert_id < 0 || expert_id >= num_experts_) return;
    auto& cw = cpu_weights_[expert_id];
    cw.gate_proj = cpu_gate_proj;
    cw.up_proj = cpu_up_proj;
    cw.down_proj = cpu_down_proj;
    cw.gate_proj_bytes = gate_bytes;
    cw.up_proj_bytes = up_bytes;
    cw.down_proj_bytes = down_bytes;
}

bool ExpertCacheManager::is_gpu_cached(int expert_id) const {
    auto it = expert_to_slot_.find(expert_id);
    if (it == expert_to_slot_.end()) return false;
    return !gpu_slots_[it->second].loading;
}

bool ExpertCacheManager::is_loading(int expert_id) const {
    auto it = expert_to_slot_.find(expert_id);
    if (it == expert_to_slot_.end()) return false;
    return gpu_slots_[it->second].loading;
}

const ExpertCacheManager::ExpertGPUWeights& ExpertCacheManager::gpu_weights(int expert_id) const {
    auto it = expert_to_slot_.find(expert_id);
    if (it != expert_to_slot_.end()) {
        return gpu_slots_[it->second].weights;
    }
    static const ExpertGPUWeights empty{};
    return empty;
}

void ExpertCacheManager::record_expert_access(int expert_id, int num_tokens) {
    if (expert_id < 0 || expert_id >= num_experts_) return;

    total_accesses_.fetch_add(1, std::memory_order_relaxed);
    if (is_gpu_cached(expert_id)) {
        gpu_hits_.fetch_add(1, std::memory_order_relaxed);
    }

    recent_accesses_.push_back({expert_id, num_tokens});
    recent_tokens_ += num_tokens;

    while (recent_tokens_ > config_.history_window_tokens && !recent_accesses_.empty()) {
        recent_tokens_ -= recent_accesses_.front().second;
        recent_accesses_.pop_front();
    }
}

void ExpertCacheManager::on_step_complete(int num_tokens_generated) {
    tokens_since_update_ += num_tokens_generated;

    for (auto& slot : gpu_slots_) {
        if (slot.expert_id >= 0) {
            slot.tokens_since_insertion += num_tokens_generated;
        }
    }

    process_lazy_evictions();
    maybe_update_cache();
}

void ExpertCacheManager::maybe_update_cache() {
    bool cold_start = (total_accesses_.load(std::memory_order_relaxed) > 0) &&
                      expert_to_slot_.empty();
    if (!cold_start && tokens_since_update_ < adaptive_update_period()) return;
    tokens_since_update_ = 0;
    adapt_parameters();
    do_evict_and_load(select_topk_experts(compute_expert_scores()));
}

std::vector<float> ExpertCacheManager::compute_expert_scores() {
    std::vector<int> raw_counts(num_experts_, 0);
    for (auto& [eid, nt] : recent_accesses_) {
        if (eid >= 0 && eid < num_experts_) {
            raw_counts[eid] += nt;
        }
    }

    float total = static_cast<float>(std::accumulate(raw_counts.begin(), raw_counts.end(), 0));
    std::vector<float> observed(num_experts_, 0.0f);
    for (int i = 0; i < num_experts_; ++i) {
        observed[i] = (total > 0.0f) ? raw_counts[i] / total : 0.0f;
        access_freq_ema_[i] = config_.ema_alpha * observed[i] +
                              (1.0f - config_.ema_alpha) * access_freq_ema_[i];
    }

    std::vector<float> scores(num_experts_, 0.0f);
    float pw = config_.prediction_weight;
    for (int i = 0; i < num_experts_; ++i) {
        scores[i] = (1.0f - pw) * access_freq_ema_[i] + pw * prediction_scores_[i];
    }
    return scores;
}

void ExpertCacheManager::update_prediction_scores(const std::vector<int>& current_experts) {
    std::vector<float> batch_freq(num_experts_, 0.0f);
    if (!current_experts.empty()) {
        float inv = 1.0f / static_cast<float>(current_experts.size());
        for (int eid : current_experts) {
            if (eid >= 0 && eid < num_experts_) {
                batch_freq[eid] += inv;
            }
        }
    }

    float decay = config_.prediction_decay;
    for (int i = 0; i < num_experts_; ++i) {
        prediction_scores_[i] = decay * batch_freq[i] + (1.0f - decay) * prediction_scores_[i];
    }
    last_batch_experts_ = current_experts;
}

std::vector<int> ExpertCacheManager::select_topk_experts(const std::vector<float>& scores) {
    std::vector<int> indices(num_experts_);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(),
                       indices.begin() + std::min(config_.gpu_cache_capacity, num_experts_),
                       indices.end(),
                       [&](int a, int b) { return scores[a] > scores[b]; });
    indices.resize(std::min(config_.gpu_cache_capacity, num_experts_));
    std::sort(indices.begin(), indices.end());
    return indices;
}

void ExpertCacheManager::do_evict_and_load(const std::vector<int>& new_topk) {
    std::unordered_set<int> new_set(new_topk.begin(), new_topk.end());

    for (auto it = expert_to_slot_.begin(); it != expert_to_slot_.end(); ) {
        int eid = it->first;
        int slot_idx = it->second;
        if (new_set.count(eid) == 0) {
            auto& slot = gpu_slots_[slot_idx];
            if (slot.tokens_since_insertion < config_.min_residency_tokens) {
                ++it;
                continue;
            }
            if (slot.loading) {
                ++it;
                continue;
            }
            slot.pending_eviction = true;
            slot.eviction_grace_remaining = config_.lazy_eviction_grace_tokens;
        }
        ++it;
    }

    for (int eid : new_topk) {
        if (expert_to_slot_.count(eid) > 0) {
            auto& slot = gpu_slots_[expert_to_slot_[eid]];
            slot.pending_eviction = false;
            slot.eviction_grace_remaining = 0;
            continue;
        }

        int slot_idx = -1;
        for (int s = 0; s < static_cast<int>(gpu_slots_.size()); ++s) {
            if (gpu_slots_[s].expert_id == -1 && !gpu_slots_[s].loading) {
                slot_idx = s;
                break;
            }
        }

        if (slot_idx == -1) {
            for (int s = 0; s < static_cast<int>(gpu_slots_.size()); ++s) {
                if (gpu_slots_[s].pending_eviction &&
                    gpu_slots_[s].eviction_grace_remaining <= 0 &&
                    !gpu_slots_[s].loading) {
                    slot_idx = s;
                    break;
                }
            }
            if (slot_idx == -1) {
                int victim_slot = find_victim_slot(new_topk);
                if (victim_slot < 0) continue;
                int victim_eid = gpu_slots_[victim_slot].expert_id;
                float new_freq = access_freq_ema_[eid];
                if (victim_eid >= 0 && new_freq <= access_freq_ema_[victim_eid] * config_.replace_threshold) {
                    continue;
                }
                expert_to_slot_.erase(victim_eid);
                slot_idx = victim_slot;
            }
        }

        if (slot_idx < 0) continue;

        async_load_expert_to_slot(eid, slot_idx);
    }

    current_topk_experts_ = new_topk;
}

void ExpertCacheManager::async_load_expert_to_slot(int expert_id, int slot_idx) {
    auto& slot = gpu_slots_[slot_idx];
    const auto& cw = cpu_weights_[expert_id];

    slot.loading = true;
    slot.pending_eviction = false;
    slot.eviction_grace_remaining = 0;

    if (cw.gate_proj && slot.weights.gate_proj && cw.gate_proj_bytes > 0) {
        CUDA_CHECK(cudaMemcpyAsync(slot.weights.gate_proj, cw.gate_proj, cw.gate_proj_bytes,
                                  cudaMemcpyDefault, prefetch_stream_));
    }
    if (cw.up_proj && slot.weights.up_proj && cw.up_proj_bytes > 0) {
        CUDA_CHECK(cudaMemcpyAsync(slot.weights.up_proj, cw.up_proj, cw.up_proj_bytes,
                                  cudaMemcpyDefault, prefetch_stream_));
    }
    if (cw.down_proj && slot.weights.down_proj && cw.down_proj_bytes > 0) {
        CUDA_CHECK(cudaMemcpyAsync(slot.weights.down_proj, cw.down_proj, cw.down_proj_bytes,
                                  cudaMemcpyDefault, prefetch_stream_));
    }

    CUDA_CHECK(cudaEventRecord(as_cuda_event(prefetch_event_), prefetch_stream_));

    slot.expert_id = expert_id;
    slot.tokens_since_insertion = 0;
    expert_to_slot_[expert_id] = slot_idx;
    pending_prefetch_.insert(expert_id);
    prefetch_in_flight_ = true;
}

void ExpertCacheManager::process_lazy_evictions() {
    for (int s = 0; s < static_cast<int>(gpu_slots_.size()); ++s) {
        auto& slot = gpu_slots_[s];
        if (!slot.pending_eviction) continue;

        slot.eviction_grace_remaining -= recent_tokens_;
        if (slot.eviction_grace_remaining <= 0) {
            int eid = slot.expert_id;
            if (eid >= 0) {
                expert_to_slot_.erase(eid);
            }
            slot.expert_id = -1;
            slot.tokens_since_insertion = 0;
            slot.pending_eviction = false;
            slot.loading = false;
        }
    }
}

int ExpertCacheManager::find_victim_slot(const std::vector<int>& new_topk) {
    int victim = -1;
    float min_freq = 1e30f;
    std::unordered_set<int> new_set(new_topk.begin(), new_topk.end());
    for (int s = 0; s < static_cast<int>(gpu_slots_.size()); ++s) {
        int eid = gpu_slots_[s].expert_id;
        if (eid < 0) continue;
        if (gpu_slots_[s].loading) continue;
        if (new_set.count(eid) > 0) continue;
        if (gpu_slots_[s].tokens_since_insertion < config_.min_residency_tokens) continue;
        if (access_freq_ema_[eid] < min_freq) {
            min_freq = access_freq_ema_[eid];
            victim = s;
        }
    }
    return victim;
}

void ExpertCacheManager::predict_and_prefetch(const std::vector<int>& current_expert_ids,
                                               cudaStream_t compute_stream) {
    if (!config_.enabled) return;

    update_prediction_scores(current_expert_ids);

    if (prefetch_in_flight_) {
        cudaError_t err = cudaEventQuery(as_cuda_event(prefetch_event_));
        if (err == cudaSuccess) {
            for (int eid : pending_prefetch_) {
                auto it = expert_to_slot_.find(eid);
                if (it != expert_to_slot_.end()) {
                    gpu_slots_[it->second].loading = false;
                }
            }
            pending_prefetch_.clear();
            prefetch_in_flight_ = false;
        } else {
            return;
        }
    }

    int budget = adaptive_prefetch_budget();
    int prefetched = 0;

    std::vector<std::pair<float, int>> candidates;
    for (int eid : current_expert_ids) {
        if (eid < 0 || eid >= num_experts_) continue;
        if (is_gpu_cached(eid)) continue;
        if (is_loading(eid)) continue;
        if (pending_prefetch_.count(eid) > 0) continue;
        candidates.push_back({prediction_scores_[eid], eid});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    for (auto& [score, eid] : candidates) {
        if (prefetched >= budget) break;

        int free_slot = -1;
        for (int s = 0; s < static_cast<int>(gpu_slots_.size()); ++s) {
            if (gpu_slots_[s].expert_id == -1 && !gpu_slots_[s].loading) {
                free_slot = s;
                break;
            }
        }
        if (free_slot < 0) break;

        async_load_expert_to_slot(eid, free_slot);
        prefetched++;
    }
}

void ExpertCacheManager::wait_prefetch(cudaStream_t compute_stream) {
    if (prefetch_in_flight_) {
        // Check for errors on the prefetch stream before waiting
        cudaError_t pf_err = cudaStreamQuery(prefetch_stream_);
        if (pf_err != cudaSuccess && pf_err != cudaErrorNotReady) {
            throw std::runtime_error(
                std::string("[EXPERT-CACHE] prefetch_stream_ error: ") +
                cudaGetErrorString(pf_err) + " — expert weights may be corrupted");
        }
        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, as_cuda_event(prefetch_event_), 0));
        for (int eid : pending_prefetch_) {
            auto it = expert_to_slot_.find(eid);
            if (it != expert_to_slot_.end()) {
                gpu_slots_[it->second].loading = false;
            }
        }
        pending_prefetch_.clear();
        prefetch_in_flight_ = false;
    }
}

float ExpertCacheManager::hit_rate() const {
    int64_t total = total_accesses_.load(std::memory_order_relaxed);
    if (total == 0) return 0.0f;
    return static_cast<float>(gpu_hits_.load(std::memory_order_relaxed)) / static_cast<float>(total);
}

size_t ExpertCacheManager::gpu_cache_memory_bytes() const {
    size_t per_expert = 0;
    if (!gpu_slots_.empty()) {
        per_expert = gpu_slots_[0].weights.gate_proj_bytes +
                     gpu_slots_[0].weights.up_proj_bytes +
                     gpu_slots_[0].weights.down_proj_bytes;
    }
    return per_expert * gpu_slots_.size();
}

int ExpertCacheManager::adaptive_update_period() const {
    float hr = hit_rate();
    if (hr < config_.low_hit_rate_threshold) {
        return config_.min_update_period;
    } else if (hr > config_.high_hit_rate_threshold) {
        return config_.max_update_period;
    }
    float t = (hr - config_.low_hit_rate_threshold) /
              (config_.high_hit_rate_threshold - config_.low_hit_rate_threshold);
    return static_cast<int>(config_.min_update_period +
                            t * (config_.max_update_period - config_.min_update_period));
}

int ExpertCacheManager::adaptive_prefetch_budget() const {
    float hr = hit_rate();
    if (hr < config_.low_hit_rate_threshold) {
        return config_.max_prefetch_per_step * 2;
    } else if (hr > config_.high_hit_rate_threshold) {
        return 0;
    }
    return config_.max_prefetch_per_step;
}

void ExpertCacheManager::adapt_parameters() {
    float hr = hit_rate();
    if (hr > 0.01f) {
        spdlog::debug("ExpertCache: hit_rate={:.1f}%, update_period={}, prefetch_budget={}",
                      hr * 100.0f, adaptive_update_period(), adaptive_prefetch_budget());
    }
}

void ExpertCacheManager::get_cache_stats(int64_t& total_accesses, int64_t& gpu_hits,
                                          int& cached_count, float& current_hit_rate) const {
    total_accesses = total_accesses_.load(std::memory_order_relaxed);
    gpu_hits = gpu_hits_.load(std::memory_order_relaxed);
    current_hit_rate = hit_rate();
    cached_count = 0;
    for (const auto& slot : gpu_slots_) {
        if (slot.expert_id >= 0 && !slot.loading) cached_count++;
    }
}

}
