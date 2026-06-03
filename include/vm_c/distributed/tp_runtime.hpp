#pragma once

// TPRuntime — column-TP 多 GPU 并行推理运行时
//
// 架构：
//   1. NcclComm 管理 allreduce（NCCL + internal CUDA kernel fallback）
//   2. 每个 rank 拥有独立 CUDA backend + llama_context
//   3. 图构建中插入 GGML_OP_VM_C_TP_ALLREDUCE 节点
//   4. 列式切分：Wo、FFN down、lm_head 为 ROW 切分，输出需 allreduce
//
// 与旧版差异：
//   - 不再使用 CustomAR / TensorParallel（已删除）
//   - 不再依赖 custom_ar.hpp
//   - allreduce 由 NcclComm 管理，回调由 ggml-cuda 后端在计算时调用

#include <vector>
#include <memory>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

#include "nccl_comm.h"

#include "vm_c/core/config.hpp"
#include "vm_c/core/tensor.hpp"
#include "vm_c/memory/kv_cache_manager.hpp"

#if !defined(VM_C_USE_LLAMA_GRAPH) || !VM_C_USE_LLAMA_GRAPH
#error "vm_c requires VM_C_USE_LLAMA_GRAPH=1 (libllama official forward path)"
#endif

#include "vm_c/official/llama_runtime.hpp"
#include "vm_c/official/turboquant_kv_bridge.hpp"

namespace vm_c {

struct TPRankContext {
    int rank = 0;
    int gpu_device = 0;
    cudaStream_t stream = nullptr;
    bool stream_owned = true;
    std::shared_ptr<KVCacheManager> kv_cache_mgr;

    std::vector<void*> uva_buffers;

    /// 暂存至 initialize_llama_graph 时 move 进 LlamaRuntime::bound_weights_
    std::unordered_map<std::string, GpuTensor> llama_weight_snapshot;
    std::unique_ptr<official::LlamaRuntime> llama_runtime;
    std::unique_ptr<official::TurboQuantKvBridge> tq_bridge;
};

class TPRuntime {
public:
    TPRuntime(int tp_size, int base_gpu_device);
    ~TPRuntime();

    TPRuntime(const TPRuntime&) = delete;
    TPRuntime& operator=(const TPRuntime&) = delete;

    void initialize(const VmCConfig& config);
    void shutdown();

    void profile_gpu_memory(const VmCConfig& config);
    void initialize_kv_cache(const VmCConfig& config);
    void initialize_llama_graph(const VmCConfig& config);

    void verify_gpu_memory_after_llama(const VmCConfig& config);

    void llama_main_seq_trim(int32_t seq_id, int64_t n_keep);

    int64_t kv_cache_blocks() const { return kv_cache_blocks_; }

    void forward_llama(
        const std::vector<int32_t>& tokens,
        const std::vector<int32_t>& positions,
        const std::vector<int32_t>& seq_ids,
        const std::vector<int8_t>& logits_flags,
        const int64_t* d_slot_mapping,
        const int32_t* d_block_tables,
        const int32_t* d_seq_lens,
        const int32_t* d_token_to_seq,
        const int32_t* d_token_positions,
        int num_tokens,
        int num_reqs,
        int max_num_blocks_per_req,
        int64_t block_size,
        bool is_prefill,
        int num_prefill_tokens,
        int num_decode_tokens,
        cudaStream_t primary_stream);

    void forward_llama_mtp_draft(
        const float* h_embd_row,
        int32_t token,
        int64_t position,
        float* output_pre_norm,
        float* output_logits,
        cudaStream_t primary_stream);

    void llama_mtp_seq_rollback(int32_t seq_id, int64_t n_past);

    void llama_mtp_seq_clear(int32_t seq_id);

    official::LlamaRuntime& llama_primary() { return *ranks_[0].llama_runtime; }

    // 返回 NcclComm（替代旧的 TensorParallel）
    // 提供 tensor_parallel() 兼容访问（旧调用方仍可编译）
    class TensorParallelCompat {
        vm_c::tp::NcclComm * comm_;
    public:
        explicit TensorParallelCompat(vm_c::tp::NcclComm * c) : comm_(c) {}
        int world_size() const { return comm_ ? comm_->world_size() : 1; }
    };
    TensorParallelCompat tensor_parallel() {
        return TensorParallelCompat(comm_.get());
    }

    TPRankContext& rank_ctx(int r) { return ranks_[r]; }
    TPRankContext& primary_ctx() { return ranks_[0]; }
    int tp_size() const { return tp_size_; }

private:
    void sync_all_ranks();
    void start_worker_threads();
    void stop_worker_threads();

    /// 向所有 rank worker 派发工作，等待全部完成后返回
    void run_on_workers(std::function<void(int rank, TPRankContext& ctx)> work_fn);

    int tp_size_;
    int base_gpu_device_;
    std::unique_ptr<vm_c::tp::NcclComm> comm_;
    std::vector<TPRankContext> ranks_;
    int64_t kv_cache_blocks_ = 0;
    bool initialized_ = false;

    // --- 持久化 worker 线程（避免每次 forward 创建/销毁线程） ---
    struct Worker {
        std::thread thread;
        std::mutex mtx;
        std::condition_variable cv;
        std::function<void()> work;
        bool work_ready  = false;
        bool work_done   = false;
        bool stop        = false;

        Worker() = default;
        Worker(Worker&&) = delete;
        Worker& operator=(Worker&&) = delete;
    };
    std::vector<std::unique_ptr<Worker>> workers_;
};

}  // namespace vm_c
