// SPDX-License-Identifier: Apache-2.0
// vllm_kernel_ops.cpp — 纯 C++ 桥接层
// 将 vm_c::Tensor 接口转换为原生 CUDA 内核调用（无 PyTorch 依赖）

#include "vm_c/cuda/vllm_kernel_ops.h"
#include <stdexcept>
#include "vm_c/cuda/gpu_arch.hpp"
#include "vm_c/cuda/kernels_attention.h"
#include "vm_c/cuda/kernels_norm.h"
#include "vm_c/cuda/kernels_moe.h"
#include "vm_c/cuda/kernels_cache.h"
#include "vm_c/cuda/kernels_pos_encoding.h"
#include "vm_c/cuda/vm_c_tensor.h"

#include <cuda_runtime.h>

namespace vm_c {
namespace vllm_kernels {

// ── 辅助函数 ────────────────────────────────────────────────────────────────
namespace {

// 提取原生指针（输出 tensor 需去 const）
inline void*       ptr(const Tensor& t) { return const_cast<void*>(t.data()); }
inline const void* cptr(const Tensor& t) { return t.data(); }

inline int64_t dim(const Tensor& t, int i) {
    return i < (int)t.shape().dims.size() ? t.shape().dims[i] : 1;
}
inline int64_t num_tokens(const Tensor& t)  { return dim(t, 0); }
inline int64_t hidden(const Tensor& t)      { return dim(t, (int)t.shape().ndim() - 1); }

inline void set_dev(const Tensor& t) {
    // device>=0 时才切换；临时 view（device=-1）由调用方的 CudaDeviceGuard 保证当前 GPU
    if (t.device() < 0) return;
    int cur = -1;
    CUDA_CHECK(cudaGetDevice(&cur));
    if (cur != t.device()) {
        CUDA_CHECK(cudaSetDevice(t.device()));
    }
}
inline void set_dev(int d) {
    if (d < 0) return;
    int cur = -1;
    CUDA_CHECK(cudaGetDevice(&cur));
    if (cur != d) {
        CUDA_CHECK(cudaSetDevice(d));
    }
}

} // anonymous namespace

// ── Attention ────────────────────────────────────────────────────────────────
void paged_attention_v1(
    Tensor& out, const Tensor& query,
    const Tensor& key_cache, const Tensor& value_cache,
    int num_kv_heads, float scale,
    const Tensor& block_tables, const Tensor& seq_lens,
    int64_t block_size, int64_t max_seq_len,
    std::optional<Tensor> alibi_slopes,
    const std::string& kv_cache_dtype,
    Tensor k_scale, Tensor v_scale,
    int tp_rank,
    int blocksparse_local_blocks, int blocksparse_vert_stride,
    int blocksparse_block_size, int blocksparse_head_sliding_step,
    cudaStream_t stream) {

    set_dev(out);
    AttentionParams p;
    p.batch_size                       = (int)num_tokens(query);
    p.num_tokens                       = (int)num_tokens(query);
    p.num_heads                        = (int)(dim(query, 1));
    p.num_kv_heads                     = num_kv_heads;
    p.head_dim                         = (int)(dim(query, 2));
    p.block_size                       = (int)block_size;
    p.max_seq_len                      = (int)max_seq_len;
    p.scale                            = scale;
    p.tp_rank                          = tp_rank;
    p.blocksparse_local_blocks         = blocksparse_local_blocks;
    p.blocksparse_vert_stride          = blocksparse_vert_stride;
    p.blocksparse_block_size           = blocksparse_block_size;
    p.blocksparse_head_sliding_step    = blocksparse_head_sliding_step;

    vm_c::paged_attention_v1(
        ptr(out), cptr(query), cptr(key_cache), cptr(value_cache), p,
        (const int32_t*)cptr(block_tables),
        (const int32_t*)cptr(seq_lens),
        alibi_slopes.has_value() ? (const float*)cptr(*alibi_slopes) : nullptr,
        kv_cache_dtype,
        cptr(k_scale), cptr(v_scale), stream);
}

void paged_attention_v2(
    Tensor& out, Tensor& exp_sums, Tensor& max_logits, Tensor& tmp_out,
    const Tensor& query, const Tensor& key_cache, const Tensor& value_cache,
    int num_kv_heads, float scale,
    const Tensor& block_tables, const Tensor& seq_lens,
    int64_t block_size, int64_t max_seq_len,
    std::optional<Tensor> alibi_slopes,
    const std::string& kv_cache_dtype,
    Tensor k_scale, Tensor v_scale,
    int tp_rank,
    int blocksparse_local_blocks, int blocksparse_vert_stride,
    int blocksparse_block_size, int blocksparse_head_sliding_step,
    cudaStream_t stream) {

    set_dev(out);
    AttentionParams p;
    p.batch_size                       = (int)num_tokens(query);
    p.num_tokens                       = (int)num_tokens(query);
    p.num_heads                        = (int)(dim(query, 1));
    p.num_kv_heads                     = num_kv_heads;
    p.head_dim                         = (int)(dim(query, 2));
    p.block_size                       = (int)block_size;
    p.max_seq_len                      = (int)max_seq_len;
    p.scale                            = scale;
    p.tp_rank                          = tp_rank;
    p.blocksparse_local_blocks         = blocksparse_local_blocks;
    p.blocksparse_vert_stride          = blocksparse_vert_stride;
    p.blocksparse_block_size           = blocksparse_block_size;
    p.blocksparse_head_sliding_step    = blocksparse_head_sliding_step;
    p.dtype                            = vm_c::ScalarType::FLOAT16; // 默认 FP16，实际 dtype 由 kv_cache 决定

    vm_c::paged_attention_v2(
        ptr(out), ptr(exp_sums), ptr(max_logits), ptr(tmp_out),
        cptr(query), cptr(key_cache), cptr(value_cache), p,
        (const int32_t*)cptr(block_tables),
        (const int32_t*)cptr(seq_lens),
        alibi_slopes.has_value() ? (const float*)cptr(*alibi_slopes) : nullptr,
        kv_cache_dtype,
        cptr(k_scale), cptr(v_scale), stream);
}

// ── Cache ────────────────────────────────────────────────────────────────────
void reshape_and_cache(
    const Tensor& key, const Tensor& value,
    Tensor& key_cache, Tensor& value_cache,
    const Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    Tensor k_scale, Tensor v_scale,
    cudaStream_t stream) {

    set_dev(key);
    int nt   = (int)num_tokens(key);
    int nkv  = (int)(dim(key, 1));
    int hd   = (int)(dim(key, 2));
    int64_t blk_sz = dim(key_cache, 1);       // [num_blocks, block_size, num_kv_heads, head_dim]
    int64_t c_stride = dim(key_cache, 1) * dim(key_cache, 2) * dim(key_cache, 3);

    vm_c::reshape_and_cache(
        cptr(key), cptr(value),
        ptr(key_cache), ptr(value_cache),
        (const int64_t*)cptr(slot_mapping),
        nt, nkv, hd, blk_sz, c_stride,
        kv_cache_dtype,
        cptr(k_scale), cptr(v_scale),
        key.dtype(), stream);
}

void reshape_and_cache_flash(
    const Tensor& key, const Tensor& value,
    Tensor& key_cache, Tensor& value_cache,
    const Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    Tensor k_scale, Tensor v_scale,
    cudaStream_t stream) {

    set_dev(key);
    int nt   = (int)num_tokens(key);
    int nkv  = (int)(dim(key, 1));
    int hd   = (int)(dim(key, 2));
    int64_t blk_sz  = dim(key_cache, 1);
    int64_t c_stride = dim(key_cache, 1) * dim(key_cache, 2) * dim(key_cache, 3);

    vm_c::reshape_and_cache_flash(
        cptr(key), cptr(value),
        ptr(key_cache), ptr(value_cache),
        (const int64_t*)cptr(slot_mapping),
        nt, nkv, hd, blk_sz, c_stride,
        kv_cache_dtype,
        cptr(k_scale), cptr(v_scale),
        key.dtype(), stream);
}

void swap_blocks(
    Tensor& src, Tensor& dst,
    int64_t block_size_in_bytes,
    const Tensor& block_mapping,
    cudaStream_t stream) {

    set_dev(src);
    vm_c::swap_blocks(
        ptr(src), ptr(dst),
        block_size_in_bytes,
        (const int64_t*)cptr(block_mapping),
        block_mapping.numel(),
        src.device(), dst.device(), stream);
}

void convert_fp8(
    Tensor& dst_cache, const Tensor& src_cache,
    double scale, const std::string& kv_cache_dtype,
    cudaStream_t stream) {

    set_dev(dst_cache);
    vm_c::convert_fp8(
        ptr(dst_cache), cptr(src_cache),
        scale, src_cache.numel(),
        kv_cache_dtype,
        dst_cache.dtype(), stream);
}

// ── Norm ────────────────────────────────────────────────────────────────────
void rms_norm(
    Tensor& out, const Tensor& input, const Tensor& weight,
    double epsilon,
    cudaStream_t stream) {

    set_dev(out);
    vm_c::rms_norm(
        ptr(out), cptr(input), cptr(weight),
        (int)num_tokens(input), (int)hidden(input), (int64_t)hidden(input),
        (float)epsilon,
        input.dtype(), stream);
}

void fused_add_rms_norm(
    Tensor& input, Tensor& residual, const Tensor& weight,
    double epsilon,
    cudaStream_t stream) {

    set_dev(input);
    vm_c::fused_add_rms_norm(
        ptr(input), ptr(residual), cptr(weight),
        (int)num_tokens(input), (int)hidden(input), (int64_t)hidden(input),
        (float)epsilon,
        input.dtype(), stream);
}

void silu_and_mul(
    Tensor& out, const Tensor& input,
    cudaStream_t stream) {

    set_dev(out);
    vm_c::silu_and_mul(
        ptr(out), cptr(input),
        num_tokens(input), hidden(out),
        input.dtype(), stream);
}

// ── Element-wise ─────────────────────────────────────────────────────────────
void add_tensor(
    Tensor& out, const Tensor& a, const Tensor& b,
    cudaStream_t stream) {
  set_dev(out);
  int64_t n = a.numel();
  vm_c::add_tensor(ptr(out), cptr(a), cptr(b), n, a.dtype(), stream);
}

void silu_mul(
    Tensor& out, const Tensor& gate, const Tensor& up,
    cudaStream_t stream) {
  set_dev(out);
  int nt = (int)num_tokens(gate);
  int hs = (int)hidden(gate);
  vm_c::silu_mul(ptr(out), cptr(gate), cptr(up), nt, hs, gate.dtype(), stream);
}

// ── Pos Encoding ─────────────────────────────────────────────────────────────
void rotary_embedding(
    Tensor& positions, Tensor& query,
    std::optional<Tensor> key,
    int64_t head_size, Tensor& cos_sin_cache,
    bool is_neox, int64_t rope_dim_offset, bool inverse,
    cudaStream_t stream) {

    (void)head_size;
    set_dev(query);
    int nt     = (int)num_tokens(query);
    int nh     = (int)(dim(query, 1));
    int nkv    = key.has_value() ? (int)(dim(*key, 1)) : nh;
    int head_sz = (int)(dim(query, 2));
    int rot_dim = (int)(dim(cos_sin_cache, 1)) * 2;
    int64_t q_stride = (int64_t)nh * head_sz;
    int64_t k_stride = key.has_value() ? (int64_t)nkv * head_sz : 0;

    vm_c::rotary_embedding(
        (const int64_t*)cptr(positions),
        ptr(query),
        key.has_value() ? ptr(*key) : nullptr,
        nt, nh, nkv, head_sz, rot_dim,
        cptr(cos_sin_cache),
        is_neox, rope_dim_offset, inverse,
        q_stride, k_stride, (int64_t)head_sz,
        query.dtype(), stream);
}

void fused_qk_norm_rope(
    Tensor& qkv, int64_t num_heads_q, int64_t num_heads_k,
    int64_t num_heads_v, int64_t head_dim, double eps,
    Tensor& q_weight, Tensor& k_weight,
    Tensor& cos_sin_cache, bool is_neox,
    Tensor& position_ids,
    int64_t forced_token_heads_per_warp,
    cudaStream_t stream) {
    (void)qkv;
    (void)num_heads_q;
    (void)num_heads_k;
    (void)num_heads_v;
    (void)head_dim;
    (void)eps;
    (void)q_weight;
    (void)k_weight;
    (void)cos_sin_cache;
    (void)is_neox;
    (void)position_ids;
    (void)forced_token_heads_per_warp;
    (void)stream;
    throw std::runtime_error(
        "fused_qk_norm_rope(qkv) removed; use fused_attn_kernels fused_qk_norm_rope(q,k)");
}

// ── MoE ──────────────────────────────────────────────────────────────────────
void topk_softmax(
    Tensor& topk_weights, Tensor& topk_indices, Tensor& token_expert_indices,
    const Tensor& gating_output, bool renormalize,
    std::optional<Tensor> bias,
    cudaStream_t stream) {

    set_dev(topk_weights);
    int nt   = (int)num_tokens(gating_output);
    int ne   = (int)hidden(gating_output);
    int tk   = (int)hidden(topk_weights);

    if (tk != (int)hidden(topk_indices) || tk != (int)hidden(token_expert_indices)) {
        throw std::runtime_error("topk_softmax: topk tensor shape mismatch");
    }
    if (nt != (int)num_tokens(topk_weights) || nt != (int)num_tokens(topk_indices)) {
        throw std::runtime_error("topk_softmax: num_tokens mismatch across tensors");
    }

    vm_c::topk_softmax(
        ptr(topk_weights), ptr(topk_indices), ptr(token_expert_indices),
        cptr(gating_output),
        nt, ne, tk, renormalize,
        bias.has_value() ? cptr(*bias) : nullptr,
        gating_output.dtype(), stream);
}

void topk_softplus_sqrt(
    Tensor& topk_weights, Tensor& topk_indices, Tensor& token_expert_indices,
    const Tensor& gating_output, bool renormalize, double routed_scaling_factor,
    std::optional<Tensor> correction_bias,
    std::optional<Tensor> input_ids, std::optional<Tensor> tid2eid,
    cudaStream_t stream) {

    set_dev(topk_weights);
    int nt  = (int)num_tokens(gating_output);
    int ne  = (int)hidden(gating_output);
    int tk  = (int)hidden(topk_weights);

    vm_c::topk_softplus_sqrt(
        ptr(topk_weights), ptr(topk_indices), ptr(token_expert_indices),
        cptr(gating_output),
        nt, ne, tk, renormalize, routed_scaling_factor,
        correction_bias.has_value() ? cptr(*correction_bias) : nullptr,
        input_ids.has_value() ? cptr(*input_ids) : nullptr,
        tid2eid.has_value() ? cptr(*tid2eid) : nullptr,
        gating_output.dtype(), stream);
}

void moe_align_block_size(
    const Tensor& topk_ids, int64_t num_experts, int64_t block_size,
    Tensor& sorted_token_ids, Tensor& experts_ids,
    Tensor& num_tokens_post_pad,
    std::optional<Tensor> maybe_expert_map,
    cudaStream_t stream) {

    (void)block_size; (void)maybe_expert_map;
    set_dev(topk_ids);
    int nt = (int)num_tokens(topk_ids);
    int tk = (int)(dim(topk_ids, 1));

    vm_c::moe_align_block_size(
        (const int32_t*)cptr(topk_ids),
        nt, (int)num_experts, tk,
        (int32_t*)ptr(sorted_token_ids),
        (int32_t*)ptr(experts_ids),
        (int32_t*)ptr(num_tokens_post_pad),
        stream);
}

// ── Sampler ──────────────────────────────────────────────────────────────────
void top_k_per_row_prefill(
    const Tensor& logits, const Tensor& row_starts, const Tensor& row_ends,
    Tensor& indices, int64_t num_rows, int64_t stride0, int64_t stride1,
    int64_t top_k,
    cudaStream_t stream) {

    (void)logits; (void)row_starts; (void)row_ends; (void)indices;
    (void)num_rows; (void)stride0; (void)stride1; (void)top_k; (void)stream;
}

void top_k_per_row_decode(
    const Tensor& logits, int64_t next_n, const Tensor& seq_lens,
    Tensor& indices, int64_t num_rows, int64_t stride0, int64_t stride1,
    int64_t top_k,
    cudaStream_t stream) {

    (void)logits; (void)next_n; (void)seq_lens; (void)indices;
    (void)num_rows; (void)stride0; (void)stride1; (void)top_k; (void)stream;
}

void persistent_topk(
    const Tensor& logits, const Tensor& lengths,
    Tensor& output, Tensor& workspace,
    int64_t k, int64_t max_seq_len,
    cudaStream_t stream) {

    (void)logits; (void)lengths; (void)output; (void)workspace;
    (void)k; (void)max_seq_len; (void)stream;
}

// ── Quant ─────────────────────────────────────────────────────────────────────
void static_scaled_int8_quant(
    Tensor& out, const Tensor& input, const Tensor& scale,
    std::optional<Tensor> azp,
    cudaStream_t stream) {

    (void)out; (void)input; (void)scale; (void)azp; (void)stream;
}

void dynamic_scaled_int8_quant(
    Tensor& out, const Tensor& input, Tensor& scales,
    std::optional<Tensor> azp,
    cudaStream_t stream) {

    (void)out; (void)input; (void)scales; (void)azp; (void)stream;
}

// ── AllReduce ────────────────────────────────────────────────────────────────
int64_t custom_all_reduce_init(
    const std::vector<int64_t>& fake_ipc_ptrs, Tensor& rank_data,
    int64_t rank, bool fully_connected,
    cudaStream_t stream) {

    (void)fake_ipc_ptrs; (void)rank_data; (void)rank;
    (void)fully_connected; (void)stream;
    return 0;
}

void custom_all_reduce(int64_t handle, Tensor& input, Tensor& output,
                       int64_t reg_buffer, int64_t reg_buffer_sz_bytes,
                       cudaStream_t stream) {

    (void)handle; (void)input; (void)output;
    (void)reg_buffer; (void)reg_buffer_sz_bytes; (void)stream;
}

void custom_all_reduce_dispose(int64_t handle) {
    (void)handle;
}

Tensor minimax_allreduce_rms(
    const Tensor& input, const Tensor& norm_weight,
    Tensor workspace, int64_t rank, int64_t nranks, double eps,
    cudaStream_t stream) {

    (void)input; (void)norm_weight; (void)workspace;
    (void)rank; (void)nranks; (void)eps; (void)stream;
    return Tensor{};
}

} // namespace vllm_kernels
} // namespace vm_c
