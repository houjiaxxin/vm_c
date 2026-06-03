#pragma once

#include <vector>
#include <deque>
#include <memory>
#include <string>
#include <optional>
#include <unordered_map>
#include <chrono>

#include "vm_c/core/request.hpp"
#include "vm_c/core/config.hpp"
#include "vm_c/memory/kv_cache_manager.hpp"

namespace vm_c {

struct SchedulerOutput {
    std::vector<RequestPtr> scheduled_new_reqs;
    std::vector<RequestPtr> scheduled_running_reqs;
    std::vector<RequestPtr> scheduled_resumed_reqs;
    std::vector<std::string> preempted_req_ids;
    std::unordered_map<std::string, int> num_scheduled_tokens;
    std::unordered_map<std::string, int> num_cached_tokens;
    std::unordered_map<std::string, std::vector<uint64_t>> prefix_block_hashes;
    std::unordered_map<std::string, std::vector<int64_t>> computed_block_ids;
    int total_num_scheduled_tokens = 0;
    int total_num_cached_tokens = 0;

    // ── 混合 batch 统计 ──
    int num_decode_in_batch = 0;   // 本轮 batch 中 decode token 数
    int num_prefill_in_batch = 0;  // 本轮 batch 中 prefill token 数
    bool decode_preempted = false; // 本轮是否触发了 decode 优先抢占
};

class Scheduler {
public:
    Scheduler(const VmCConfig& config,
              std::shared_ptr<KVCacheManager> kv_cache_mgr);
    ~Scheduler() = default;

    SchedulerOutput schedule();

    void add_request(RequestPtr req);
    void abort_request(const std::string& request_id);
    RequestPtr find_request(const std::string& request_id);
    void finish_request(const std::string& request_id, FinishReason reason);

    void update_from_output(const std::vector<RequestOutput>& outputs);

    int num_waiting_requests() const;
    int num_running_requests() const;

    void enable_prefix_caching(bool enable) { prefix_caching_enabled_ = enable; }
    bool prefix_caching_enabled() const { return prefix_caching_enabled_; }

    // ── Prefix Caching 统计信息 ──
    struct PrefixCacheStats {
        int64_t total_queries = 0;
        int64_t total_hits = 0;
        int64_t total_blocks_cached = 0;
        int64_t total_blocks_evicted = 0;
    };

    const PrefixCacheStats& prefix_cache_stats() const { return prefix_cache_stats_; }

private:
    bool can_allocate_slots(int num_tokens) const;
    void preempt_requests(int num_tokens_needed);

    // ── 混合 batch 调度 ──
    // 优先调度所有 decode 请求（保证 decode 延迟），剩余 budget 给 prefill
    void schedule_decode_first(SchedulerOutput& output);
    // 当 decode 延迟超标时，抢占 prefill token budget
    bool check_decode_latency_preempt(SchedulerOutput& output);

    std::vector<uint64_t> compute_block_hashes(const std::vector<int32_t>& token_ids,
                                                int num_tokens) const;

    VmCConfig config_;
    std::shared_ptr<KVCacheManager> kv_cache_mgr_;

    std::deque<RequestPtr> waiting_queue_;
    std::vector<RequestPtr> running_queue_;

    int max_num_seqs_;
    int max_num_batched_tokens_;
    bool enable_chunked_prefill_ = true;
    int long_prefill_token_threshold_ = 0;
    bool prefix_caching_enabled_ = true;
    PrefixCacheStats prefix_cache_stats_;

    // ── 混合 batch 调度配置 ──
    bool decode_first_ = true;
    int max_prefill_per_batch_ = 0;
    int decode_latency_target_ms_ = 100;
    bool preempt_prefill_on_latency_ = true;
    int decode_reserved_tokens_ = 0;
};

}
