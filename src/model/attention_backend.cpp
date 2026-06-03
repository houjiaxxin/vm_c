#include "vm_c/model/attention_backend.hpp"
#include "vm_c/model/attention_metadata.hpp"
#include "vm_c/model/models.hpp"
#include "vm_c/cuda/kernels_cache.h"
#include "vm_c/cuda/kernels_attention.h"
#include "vm_c/cuda/kernels_flash_attn.h"
#include "vm_c/cuda/kernels_turboquant.h"
#include "vm_c/cuda/vllm_kernel_ops.h"
#include "vm_c/cuda/gpu_arch.hpp"
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

namespace vm_c {

void PagedAttentionBackend::store_kv(
    const void* key, const void* value,
    void* key_cache, void* value_cache,
    const int64_t* slot_mapping,
    int num_tokens, int num_kv_heads, int head_dim,
    int64_t block_size, int64_t cache_stride,
    cudaStream_t stream) {
    // 使用项目自身的 reshape_and_cache 以支持 HND 布局
    // 相比 vllm 版本，增加了 layout 参数支持
    reshape_and_cache(
        key, value, key_cache, value_cache, slot_mapping,
        num_tokens, num_kv_heads, head_dim,
        block_size, cache_stride,
        gpu().supports_bf16() ? "bf16" : "fp16",
        nullptr, nullptr,
        gpu().compute_dtype(), stream,
        kv_cache_layout_);
}

void PagedAttentionBackend::forward_prefill(
    void* output, const void* query,
    const void* key, const void* value,
    const void* key_cache, const void* value_cache,
    const AttentionMetadata& meta,
    int num_heads, int num_kv_heads, int head_dim,
    float scale, int64_t block_size,
    ScalarType dtype, cudaStream_t stream) {
    // 使用高性能 Flash Attention v2 kernel（warp 级并行 softmax + shared memory tiling）
    // 替代旧版串行 kernel，预估 prefill 阶段 3-10x 加速
    if (kv_cache_layout_ == "hnd") {
        // HND 布局下使用 flash_attention_prefill_v2（已适配 HND）
        flash_attention_prefill_v2(
            output, query,
            key_cache, value_cache,
            meta.num_tokens, num_heads, num_kv_heads, head_dim,
            scale, block_size, meta.max_num_blocks_per_req,
            meta.block_tables, meta.seq_lens,
            meta.token_to_seq, meta.token_positions,
            meta.num_reqs, "auto", dtype, stream,
            kv_cache_layout_);
    } else {
        flash_attention_prefill_v2(
            output, query,
            key_cache, value_cache,
            meta.num_tokens, num_heads, num_kv_heads, head_dim,
            scale, block_size, meta.max_num_blocks_per_req,
            meta.block_tables, meta.seq_lens,
            meta.token_to_seq, meta.token_positions,
            meta.num_reqs, "auto", dtype, stream);
    }
}

void PagedAttentionBackend::forward_decode(
    void* output, const void* query,
    const void* key_cache, const void* value_cache,
    const AttentionMetadata& meta,
    int num_heads, int num_kv_heads, int head_dim,
    float scale, int64_t block_size,
    ScalarType dtype, cudaStream_t stream) {
    int dev = 0;
    cudaGetDevice(&dev);

    // 使用项目自身的 paged_attention_v1 以支持 HND 布局
    AttentionParams params;
    params.batch_size = meta.num_reqs;
    params.num_tokens = meta.num_tokens;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.block_size = block_size;
    params.max_num_blocks_per_req = meta.max_num_blocks_per_req;
    params.max_seq_len = meta.max_num_blocks_per_req * block_size;
    params.scale = scale;
    params.dtype = dtype;
    params.tp_rank = 0;

    vm_c::paged_attention_v1(
        output, query,
        key_cache, value_cache,
        params,
        meta.block_tables, meta.seq_lens,
        nullptr, "auto",
        nullptr, nullptr, stream,
        kv_cache_layout_);
}

TurboQuantBackend::TurboQuantBackend(int head_dim, const std::string& quant_method,
                                     int num_attention_heads, int num_kv_heads,
                                     int gpu_device)
    : gpu_device_(gpu_device) {
    CUDA_CHECK(cudaSetDevice(gpu_device_));
    config_ = TQConfig::from_method(head_dim, quant_method.c_str(), gpu_device_);
    buffers_.init(head_dim, config_.key_quant_bits, gpu_device);
    spdlog::info("TurboQuantBackend: initialized with method={}, slot_size={}, "
                 "key_packed={}, val_packed={}, fp8_e4b15={}",
                 quant_method, config_.slot_size_aligned,
                 config_.key_packed_size, config_.value_packed_size,
                 config_.fp8_e4b15);
}

TurboQuantBackend::~TurboQuantBackend() {
    buffers_.free_buf();
    workspace_.free_buf();
}

void TurboQuantBackend::ensure_workspace(int max_num_batched_tokens, int max_num_seqs,
                                          int num_attention_heads, int num_kv_heads,
                                          int head_dim, int gpu_device) {
    max_num_batched_tokens_ = max_num_batched_tokens;
    max_num_seqs_ = max_num_seqs;
    num_attention_heads_ = num_attention_heads;
    num_kv_heads_ = num_kv_heads;
    head_dim_ = head_dim;

    workspace_.ensure_decode(max_num_seqs * num_attention_heads,
                             head_dim,
                             config_.max_num_kv_splits,
                             gpu_device);
    workspace_.ensure_store(num_kv_heads,
                            head_dim,
                            gpu_device);
}

void TurboQuantBackend::store_kv(
    const void* key, const void* value,
    void* key_cache, void* value_cache,
    const int64_t* slot_mapping,
    int num_tokens, int num_kv_heads, int head_dim,
    int64_t block_size, int64_t cache_stride,
    cudaStream_t stream) {
    turboquant_store(
        key, value,
        key_cache, slot_mapping,
        num_tokens, num_kv_heads, head_dim,
        block_size, config_, buffers_, workspace_,
        stream);
}

void TurboQuantBackend::forward_prefill(
    void* output, const void* query,
    const void* key, const void* value,
    const void* key_cache, const void* value_cache,
    const AttentionMetadata& meta,
    int num_heads, int num_kv_heads, int head_dim,
    float scale, int64_t block_size,
    ScalarType dtype, cudaStream_t stream) {
    // Prefill 阶段使用已有的 FlashAttention kernel（与 PagedAttentionBackend 相同）
    // 原因：TurboQuant 的 prefill kernel 有两个严重问题：
    //   1. scores[64] 固定大小数组在 seq_len > 64 时越界
    //   2. 完全串行化（只有 threadIdx.x==0 执行计算），性能极差
    // Prefill 阶段不需要从压缩 KV cache 读取（只有新增 token 的 KV 需要被量化存储），
    // 所以直接使用标准 FlashAttention kernel 是正确且最优的选择。
    //
    // 注意：TurboQuant 后端使用打包的 KV cache 格式（key_cache 指向打包缓冲区，
    // value_cache 为 nullptr）。标准 paged_attention_prefill 从分页 KV cache 读取，
    // 无法直接处理 value_cache=nullptr 的情况。因此，当 value_cache 为 nullptr 时，
    // 改用局部连续 K/V 数组的 flash_attention_prefill_local kernel。
    //
    // 局限：纯 prefill（num_decode_tokens == 0）时 local K/V 包含全部 token，
    // 可以正常工作。mixed decode+prefill 场景需要额外从 cache 解包，暂未支持。
    if (value_cache == nullptr && meta.num_decode_tokens == 0) {
        // TurboQuant 打包缓存 + 纯 prefill：从局部连续 K/V 数组读取
        flash_attention_prefill_local(
            output, query, key, value,
            meta.num_tokens, num_heads, num_kv_heads, head_dim,
            scale,
            meta.token_to_seq, meta.token_positions,
            dtype, stream);
    } else if (value_cache != nullptr) {
        flash_attention_prefill_v2(
            output, query,
            key_cache, value_cache,
            meta.num_tokens, num_heads, num_kv_heads, head_dim,
            scale, block_size, meta.max_num_blocks_per_req,
            meta.block_tables, meta.seq_lens,
            meta.token_to_seq, meta.token_positions,
            meta.num_reqs, "auto", dtype, stream);
    } else {
        throw std::runtime_error(
            "TurboQuant prefill with value_cache=nullptr requires num_decode_tokens==0 "
            "(mixed decode+prefill must split metadata before calling forward_prefill)");
    }
}

void TurboQuantBackend::forward_decode(
    void* output, const void* query,
    const void* key_cache, const void* value_cache,
    const AttentionMetadata& meta,
    int num_heads, int num_kv_heads, int head_dim,
    float scale, int64_t block_size,
    ScalarType dtype, cudaStream_t stream) {
    (void)dtype;
    turboquant_decode_attention(
        output, query,
        key_cache,
        meta.block_tables, meta.seq_lens,
        meta.num_reqs, num_heads, num_kv_heads, head_dim,
        scale, block_size, meta.max_num_blocks_per_req,
        config_, buffers_, workspace_,
        config_.max_num_kv_splits, stream);
}

size_t TurboQuantBackend::workspace_size() const {
    if (max_num_batched_tokens_ == 0) return 0;

    size_t total = 0;
    int nh = num_kv_heads_;
    int hd = head_dim_;
    int T = max_num_batched_tokens_;
    int num_heads = num_attention_heads_;
    int max_seqs = max_num_seqs_;
    int num_kv_splits = config_.max_num_kv_splits;

    size_t store_size = nh * hd * sizeof(float) + nh * sizeof(float) +
                        nh * hd * sizeof(float) + nh * hd * sizeof(float);

    int bhq = max_seqs * num_heads;
    size_t decode_size = bhq * hd * sizeof(float) +
                         bhq * hd * sizeof(float) +
                         bhq * num_kv_splits * (hd + 1) * sizeof(float);

    total = store_size + decode_size;
    return total;
}

}
