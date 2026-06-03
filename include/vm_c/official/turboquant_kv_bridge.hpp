#pragma once

#include "vm_c/core/tensor.hpp"
#include "vm_c/cuda/kernels_turboquant.h"
#include "vm_c/memory/kv_cache_manager.hpp"

#include "ggml.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_model;
struct llama_context;

namespace vm_c::official {

/// TurboQuant KV 与 llama graph 的桥接（P4）。
/// 由 Engine 在每步 forward 前更新 batch 元数据；graph 内 GGML_OP_VM_C_TURBOQUANT_ATTN 读取。
class TurboQuantKvBridge {
public:
    TurboQuantKvBridge() = default;
    ~TurboQuantKvBridge();

    TurboQuantKvBridge(const TurboQuantKvBridge&) = delete;
    TurboQuantKvBridge& operator=(const TurboQuantKvBridge&) = delete;

    void initialize(const VmCConfig& config,
                    KVCacheManager* kv_cache_mgr,
                    int gpu_device,
                    int num_layers,
                    int tp_size = 1);

    /// 每步 decode 前由 Engine 调用，写入 slot/block table 等 device 指针。
    void set_batch_metadata(
        const int64_t* d_slot_mapping,
        const int32_t* d_block_tables,
        const int32_t* d_seq_lens,
        int num_tokens,
        int num_reqs,
        int max_num_blocks_per_req,
        int64_t block_size,
        bool is_prefill,
        int num_prefill_tokens,
        int num_decode_tokens,
        const int32_t* d_token_to_seq,
        const int32_t* d_token_positions,
        cudaStream_t stream);

    /// 标记为 decode-only 模式（防止 MTP 验证等场景误触发 prefill attention 路径）
    void mark_decode_mode(int num_tokens);

    bool is_layer_turboquant(int layer) const {
        return kv_cache_mgr_ && kv_cache_mgr_->is_layer_turboquant(layer);
    }

    ggml_tensor* kv_cache_tensor(int layer) const;

    const TQConfig& tq_config() const { return tq_config_; }
    TQBuffers& tq_buffers() { return tq_buffers_; }
    TQWorkspace& tq_workspace() { return tq_workspace_; }
    int gpu_device() const { return gpu_device_; }
    int num_attention_heads() const { return num_attention_heads_; }
    int num_kv_heads() const { return num_kv_heads_; }
    int head_dim() const { return head_dim_; }

    /// 已启用 TurboQuant 的层数（用于启动诊断）
    int tq_layer_enabled_count() const;

    const int64_t* slot_mapping() const { return d_slot_mapping_; }
    const int32_t* block_tables() const { return d_block_tables_; }
    const int32_t* seq_lens() const { return d_seq_lens_; }
    int num_tokens() const { return num_tokens_; }
    int num_reqs() const { return num_reqs_; }
    int max_num_blocks_per_req() const { return max_num_blocks_per_req_; }
    int64_t block_size() const { return block_size_; }
    bool is_prefill() const { return is_prefill_; }
    int num_prefill_tokens() const { return num_prefill_tokens_; }
    int num_decode_tokens() const { return num_decode_tokens_; }
    const int32_t* token_to_seq() const { return d_token_to_seq_; }
    const int32_t* token_positions() const { return d_token_positions_; }
    cudaStream_t stream() const { return stream_; }

    void* kv_cache_ptr(int layer) const;

private:
    KVCacheManager* kv_cache_mgr_ = nullptr;
    int gpu_device_ = 0;
    int num_layers_ = 0;
    int num_attention_heads_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;

    TQConfig tq_config_;
    TQBuffers tq_buffers_;
    TQWorkspace tq_workspace_;

    std::vector<void*> kv_ptrs_;
    std::vector<ggml_tensor*> kv_tensors_;
    ggml_context* tensor_ctx_ = nullptr;

    const int64_t* d_slot_mapping_ = nullptr;
    const int32_t* d_block_tables_ = nullptr;
    const int32_t* d_seq_lens_ = nullptr;
    int num_tokens_ = 0;
    int num_reqs_ = 0;
    int max_num_blocks_per_req_ = 0;
    int64_t block_size_ = 0;
    bool is_prefill_ = false;
    int num_prefill_tokens_ = 0;
    int num_decode_tokens_ = 0;
    const int32_t* d_token_to_seq_ = nullptr;
    const int32_t* d_token_positions_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

/// 注册 ggml-cuda TurboQuant 算子回调（须在 llama_decode 前调用一次）。
void register_ggml_turboquant_attn_kernels();

/// 注册 ggml-cuda column-TP allreduce 回调（须在 llama_decode 前调用一次）。
void register_ggml_tp_allreduce_kernels();

/// 将 vm_c ModelLoader 分片权重绑定到 llama 模型 tensor（column TP）。
void bind_vm_c_weights_to_llama_model(
    llama_model* model,
    const std::unordered_map<std::string, GpuTensor>& gpu_weights);

}  // namespace vm_c::official
