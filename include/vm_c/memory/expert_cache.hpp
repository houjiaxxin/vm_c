#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <mutex>
#include <atomic>
#include <cstdint>
#include "vm_c/core/tensor.hpp"

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace vm_c {

struct ModelConfig;

struct ExpertCacheConfig {
    static constexpr int DEFAULT_UPDATE_PERIOD_TOKENS = 256;

    int gpu_cache_capacity = 0;
    int update_period_tokens = 0;
    int history_window_tokens = 0;
    float ema_alpha = 0.1f;
    float replace_threshold = 1.2f;
    int min_residency_tokens = 0;
    int max_prefetch_per_step = 2;
    bool enabled = false;

    float prediction_weight = 0.3f;
    float prediction_decay = 0.5f;
    float low_hit_rate_threshold = 0.7f;
    float high_hit_rate_threshold = 0.9f;
    int min_update_period = 0;
    int max_update_period = 0;
    int lazy_eviction_grace_tokens = 0;
};

class ExpertCacheManager {
public:
    ExpertCacheManager(const ExpertCacheConfig& config,
                       int num_experts,
                       int num_experts_per_tok,
                       int hidden_size,
                       int moe_intermediate_size);
    ~ExpertCacheManager();

    ExpertCacheManager(const ExpertCacheManager&) = delete;
    ExpertCacheManager& operator=(const ExpertCacheManager&) = delete;

    struct ExpertGPUWeights {
        void* gate_proj = nullptr;
        void* up_proj = nullptr;
        void* down_proj = nullptr;
        size_t gate_proj_bytes = 0;
        size_t up_proj_bytes = 0;
        size_t down_proj_bytes = 0;
        DataType dtype = DataType::BFLOAT16;
    };

    struct CPUExpertWeights {
        void* gate_proj = nullptr;
        void* up_proj = nullptr;
        void* down_proj = nullptr;
        size_t gate_proj_bytes = 0;
        size_t up_proj_bytes = 0;
        size_t down_proj_bytes = 0;
        DataType output_dtype = DataType::BFLOAT16;
    };

    void allocate_gpu_cache(size_t gate_proj_bytes_per_expert,
                            size_t up_proj_bytes_per_expert,
                            size_t down_proj_bytes_per_expert);

    void set_cpu_expert_weights(int expert_id,
                               void* cpu_gate_proj, size_t gate_bytes,
                               void* cpu_up_proj, size_t up_bytes,
                               void* cpu_down_proj, size_t down_bytes);

    bool is_gpu_cached(int expert_id) const;

    bool is_loading(int expert_id) const;

    const ExpertGPUWeights& gpu_weights(int expert_id) const;

    void record_expert_access(int expert_id, int num_tokens);

    void on_step_complete(int num_tokens_generated);

    void maybe_update_cache();

    void predict_and_prefetch(const std::vector<int>& current_expert_ids,
                              cudaStream_t compute_stream);

    void wait_prefetch(cudaStream_t compute_stream);

    float hit_rate() const;

    int gpu_cache_capacity() const { return config_.gpu_cache_capacity; }
    bool is_enabled() const { return config_.enabled; }

    size_t gpu_cache_memory_bytes() const;

    std::vector<CPUExpertWeights>& get_cpu_weights_mut() { return cpu_weights_; }

    int adaptive_update_period() const;

    int adaptive_prefetch_budget() const;

    void get_cache_stats(int64_t& total_accesses, int64_t& gpu_hits,
                         int& cached_count, float& current_hit_rate) const;

private:
    struct CacheSlot {
        int expert_id = -1;
        ExpertGPUWeights weights;
        int tokens_since_insertion = 0;
        bool loading = false;
        bool pending_eviction = false;
        int eviction_grace_remaining = 0;
    };

    ExpertCacheConfig config_;
    int num_experts_;
    int num_experts_per_tok_;
    int hidden_size_;
    int moe_intermediate_size_;

    std::vector<CPUExpertWeights> cpu_weights_;

    std::vector<CacheSlot> gpu_slots_;
    std::unordered_map<int, int> expert_to_slot_;

    std::vector<float> access_freq_ema_;
    std::deque<std::pair<int, int>> recent_accesses_;
    int recent_tokens_ = 0;

    std::vector<float> prediction_scores_;
    std::vector<int> last_batch_experts_;

    std::vector<int> current_topk_experts_;

    std::atomic<int64_t> total_accesses_{0};
    std::atomic<int64_t> gpu_hits_{0};

    int tokens_since_update_ = 0;

    cudaStream_t prefetch_stream_ = nullptr;
    void* prefetch_event_ = nullptr;
    std::vector<void*> sync_events_;
    std::unordered_set<int> pending_prefetch_;
    bool prefetch_in_flight_ = false;

    std::vector<size_t> gate_proj_bytes_per_expert_;
    std::vector<size_t> up_proj_bytes_per_expert_;
    std::vector<size_t> down_proj_bytes_per_expert_;

    std::vector<float> compute_expert_scores();

    std::vector<int> select_topk_experts(const std::vector<float>& scores);

    void do_evict_and_load(const std::vector<int>& new_topk);

    int find_victim_slot(const std::vector<int>& new_topk);

    void async_load_expert_to_slot(int expert_id, int slot_idx);

    void process_lazy_evictions();

    void update_prediction_scores(const std::vector<int>& current_experts);

    void adapt_parameters();
};

}
