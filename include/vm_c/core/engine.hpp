#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>

#include "vm_c/core/config.hpp"
#include "vm_c/core/request.hpp"
#include "vm_c/core/scheduler.hpp"
#include "vm_c/model/model_loader.hpp"
#include "vm_c/memory/kv_cache_manager.hpp"
#include "vm_c/distributed/tp_runtime.hpp"
#include "vm_c/speculative/speculative_engine.hpp"

struct llama_vocab;

namespace vm_c {

class Engine {
public:
    explicit Engine(const VmCConfig& config);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    void initialize();
    void shutdown();

    std::string submit_request(const std::vector<int32_t>& prompt_token_ids,
                               const SamplingParams& params,
                               const std::string& model_name = "",
                               std::function<void(const std::vector<int32_t>&, bool, FinishReason)> stream_cb = nullptr);
    void abort_request(const std::string& request_id);

    std::vector<RequestOutput> step();

    bool is_running() const { return running_; }

    const VmCConfig& config() const { return config_; }

    /// 返回 llama.cpp 官方词汇表对象，用于 tokenize/detokenize
    const struct llama_vocab* vocab() const;

    int num_running_requests() const { return scheduler_ ? scheduler_->num_running_requests() : 0; }
    int num_waiting_requests() const { return scheduler_ ? scheduler_->num_waiting_requests() : 0; }

private:
    void resolve_model_config();
    std::vector<RequestOutput> run_forward(const SchedulerOutput& sched_output);

    VmCConfig config_;
    bool running_ = false;
    bool initialized_ = false;

    std::unique_ptr<Scheduler> scheduler_;
    std::shared_ptr<KVCacheManager> kv_cache_mgr_;
    std::unique_ptr<ModelLoader> model_loader_;
    std::unique_ptr<TPRuntime> tp_runtime_;
    std::unique_ptr<SpeculativeEngine> speculative_engine_;

    std::thread engine_thread_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::queue<RequestPtr> pending_requests_;

    int gpu_device_ = 0;
    int tp_rank_ = 0;
    int tp_size_ = 1;
    std::vector<bool> seq_id_in_use_;       // 跟踪哪些 seq_id 正在被使用
    std::mutex seq_id_mtx_;                 // seq_id pool 的互斥锁

    int32_t alloc_seq_id();
    void release_seq_id(int32_t sid);
    cudaStream_t compute_stream_ = nullptr;
    bool compute_stream_owned_ = true;

    struct EngineBuffers {
        int capacity = 0;
        void* d_input_ids = nullptr;
        void* d_position_ids = nullptr;
        void* d_hidden_states = nullptr;
        void* d_seq_lens = nullptr;
        void* d_slot_mapping = nullptr;
        void* d_token_to_seq = nullptr;
        void* d_token_positions = nullptr;
        void* d_block_tables = nullptr;
        int block_tables_size = 0;
        void* d_logits = nullptr;
        int logits_size = 0;
        void* d_lm_gather = nullptr;
        int lm_gather_size = 0;
        void* d_output_ids = nullptr;
        void* d_output_logprobs = nullptr;
        void* d_temperatures = nullptr;
        void* d_top_k = nullptr;
        void* d_top_p = nullptr;
        void* d_min_p = nullptr;
        int sample_buf_size = 0;
        uint8_t* d_prompt_mask = nullptr;
        int32_t* d_output_bin_counts = nullptr;
        float* d_rep_penalties = nullptr;
        float* d_freq_penalties = nullptr;
        float* d_pres_penalties = nullptr;
        int penalty_buf_size = 0;
        void* d_pre_norm_hidden = nullptr;

        // Staging buffer for MTP verify sampling（与 d_logits 分离，避免覆盖 verify logits）
        void* d_verify_staging = nullptr;
        int verify_staging_size = 0;

        void ensure(int num_tokens, int max_num_seqs, int hidden_size,
                    int vocab_size, int max_blocks_per_req, int num_reqs,
                    int gpu_device, int min_lm_gather_rows = 0);
        void free();
    } engine_bufs_;
};

}  // namespace vm_c
