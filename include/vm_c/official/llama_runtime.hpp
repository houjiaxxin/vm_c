#pragma once

#include "vm_c/core/config.hpp"
#include "vm_c/core/tensor.hpp"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

struct llama_model;
struct llama_context;
struct llama_batch;

namespace vm_c::official {

class TurboQuantKvBridge;

/// llama.cpp 官方 compute 路径：model load + main/MTP context + llama_decode。
/// 目标：替换 models.cu imperative forward 与 ggml_runner mini-graph slot。
/// TurboQuant KV / vLLM 调度仍由 vm_c Engine 负责（P4：llama_memory 适配）。
class LlamaRuntime {
public:
    LlamaRuntime() = default;
    ~LlamaRuntime();

    LlamaRuntime(const LlamaRuntime&) = delete;
    LlamaRuntime& operator=(const LlamaRuntime&) = delete;

    /// 初始化 ggml/llama 后端（进程级，幂等）。须在 load 前调用。
    static void ensure_backend();

    /// 从 GGUF 解析图结构并绑定 vm_c ModelLoader 分片权重（column TP，所有权移入 runtime）。
    void load(const VmCConfig& config, int gpu_device,
              std::unordered_map<std::string, GpuTensor> vm_c_weights);

    void set_turboquant_bridge(TurboQuantKvBridge* bridge);

    /// column-TP：传入 vm_c::TensorParallel* 与 world size（须在 load 前设置）。
    void set_tensor_parallel(void* tp, int tp_size);

    /// 主 context decode（对齐 llama-context.cpp process_ubatch → llama_decode）。
    /// @param mtp false=ctx_main, true=ctx_mtp（须已 load MTP context）
    void decode(const llama_batch& batch, bool mtp = false);

    /// 取 decode 后第 i 个 logits 行（仅 batch.logits[i]!=0 的 token 有有效 logits）。
    float* logits_ith(int index, bool mtp = false) const;

    /// MTP：final norm 前 hidden（对齐 llama_get_embeddings_pre_norm_ith）。
    float* embeddings_pre_norm_ith(int index, bool mtp = false) const;

    /// llama_memory 序列操作（MTP accept/trim 对齐 speculative.cpp）。
    void memory_clear(bool data);
    void memory_seq_rm(int32_t seq_id, int32_t p0, int32_t p1);
    void memory_seq_add(int32_t seq_id, int32_t p0, int32_t p1, int32_t shift);

    /// MTP context 序列裁剪（draft 后回滚 ctx_mtp KV）。
    void memory_seq_rm_mtp(int32_t seq_id, int32_t p0, int32_t p1);

    int n_embd() const;
    int n_vocab() const;

    bool loaded() const { return model_ != nullptr && ctx_main_ != nullptr; }
    bool mtp_loaded() const { return ctx_mtp_ != nullptr; }

    llama_model* model() const { return model_; }
    llama_context* context_main() const { return ctx_main_; }
    llama_context* context_mtp() const { return ctx_mtp_; }

private:
    void free_all();
    llama_context* select_ctx(bool mtp) const;

    static bool backend_initialized_;

    llama_model* model_ = nullptr;
    llama_context* ctx_main_ = nullptr;
    llama_context* ctx_mtp_ = nullptr;
    TurboQuantKvBridge* tq_bridge_ = nullptr;
    void* vm_c_tp_ = nullptr;
    int vm_c_tp_size_ = 1;
    /// 持有 rebind 后的 vm_c 权重（GgmlWeightStorage 生命周期须覆盖 llama model）。
    std::unordered_map<std::string, GpuTensor> bound_weights_;
};

/// 构建 llama_batch（对齐 llama.cpp server update_batch 字段布局）。
/// 调用方须 llama_batch_free(batch)。
struct LlamaBatchBuilder {
    static llama_batch build(
        const std::vector<int32_t>& tokens,
        const std::vector<int32_t>& positions,
        const std::vector<int32_t>& seq_ids,
        const std::vector<int8_t>& logits_flags);

    static llama_batch build_single(int32_t token, int32_t position, int32_t seq_id, bool want_logits);

    /// MTP draft：单 token batch（所有 rank 同步参与 graph 构建）
    static llama_batch build_mtp_draft(
        int32_t token, int32_t position,
        const float* embd_rows, int n_embd);

    /// 释放由 build / build_mtp_draft 创建的 batch。
    /// 不能直接用 llama_batch_free：fill_batch_common 覆盖了 seq_id[i] 指向
    /// thread_local g_seq_scratch，llama_batch_free 会试图 free 栈地址。
    static void free(llama_batch& batch);
};

}  // namespace vm_c::official
