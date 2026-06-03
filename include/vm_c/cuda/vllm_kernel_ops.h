#pragma once

#include "vm_c/cuda/vm_c_tensor.h"
#include <optional>
#include <string>
#include <vector>

namespace vm_c {
namespace vllm_kernels {

// ── Attention ────────────────────────────────────────────────────────────
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
    cudaStream_t stream = 0);

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
    cudaStream_t stream = 0);

// ── Cache ────────────────────────────────────────────────────────────────
void reshape_and_cache(
    const Tensor& key, const Tensor& value,
    Tensor& key_cache, Tensor& value_cache,
    const Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    Tensor k_scale, Tensor v_scale,
    cudaStream_t stream = 0);

void reshape_and_cache_flash(
    const Tensor& key, const Tensor& value,
    Tensor& key_cache, Tensor& value_cache,
    const Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    Tensor k_scale, Tensor v_scale,
    cudaStream_t stream = 0);

void swap_blocks(
    Tensor& src, Tensor& dst,
    int64_t block_size_in_bytes,
    const Tensor& block_mapping,
    cudaStream_t stream = 0);

void convert_fp8(
    Tensor& dst_cache, const Tensor& src_cache,
    double scale, const std::string& kv_cache_dtype,
    cudaStream_t stream = 0);

// ── Norm ─────────────────────────────────────────────────────────────────
void rms_norm(
    Tensor& out, const Tensor& input, const Tensor& weight,
    double epsilon,
    cudaStream_t stream = 0);

void fused_add_rms_norm(
    Tensor& input, Tensor& residual, const Tensor& weight,
    double epsilon,
    cudaStream_t stream = 0);

void silu_and_mul(
    Tensor& out, const Tensor& input,
    cudaStream_t stream = 0);

// ── Pos Encoding ─────────────────────────────────────────────────────────
void rotary_embedding(
    Tensor& positions, Tensor& query,
    std::optional<Tensor> key,
    int64_t head_size, Tensor& cos_sin_cache,
    bool is_neox, int64_t rope_dim_offset, bool inverse,
    cudaStream_t stream = 0);

void fused_qk_norm_rope(
    Tensor& qkv, int64_t num_heads_q, int64_t num_heads_k,
    int64_t num_heads_v, int64_t head_dim, double eps,
    Tensor& q_weight, Tensor& k_weight,
    Tensor& cos_sin_cache, bool is_neox,
    Tensor& position_ids,
    int64_t forced_token_heads_per_warp,
    cudaStream_t stream = 0);

// ── MoE ──────────────────────────────────────────────────────────────────
void topk_softmax(
    Tensor& topk_weights, Tensor& topk_indices, Tensor& token_expert_indices,
    const Tensor& gating_output, bool renormalize,
    std::optional<Tensor> bias,
    cudaStream_t stream = 0);

void topk_softplus_sqrt(
    Tensor& topk_weights, Tensor& topk_indices, Tensor& token_expert_indices,
    const Tensor& gating_output, bool renormalize, double routed_scaling_factor,
    std::optional<Tensor> correction_bias,
    std::optional<Tensor> input_ids, std::optional<Tensor> tid2eid,
    cudaStream_t stream = 0);

void moe_align_block_size(
    const Tensor& topk_ids, int64_t num_experts, int64_t block_size,
    Tensor& sorted_token_ids, Tensor& experts_ids,
    Tensor& num_tokens_post_pad,
    std::optional<Tensor> maybe_expert_map,
    cudaStream_t stream = 0);

// ── Sampler ──────────────────────────────────────────────────────────────
void top_k_per_row_prefill(
    const Tensor& logits, const Tensor& row_starts, const Tensor& row_ends,
    Tensor& indices, int64_t num_rows, int64_t stride0, int64_t stride1,
    int64_t top_k,
    cudaStream_t stream = 0);

// ── Element-wise ───────────────────────────────────────────────────────────
void add_tensor(
    Tensor& out, const Tensor& a, const Tensor& b,
    cudaStream_t stream = 0);

void silu_mul(
    Tensor& out, const Tensor& gate, const Tensor& up,
    cudaStream_t stream = 0);

void top_k_per_row_decode(
    const Tensor& logits, int64_t next_n, const Tensor& seq_lens,
    Tensor& indices, int64_t num_rows, int64_t stride0, int64_t stride1,
    int64_t top_k,
    cudaStream_t stream = 0);

void persistent_topk(
    const Tensor& logits, const Tensor& lengths,
    Tensor& output, Tensor& workspace,
    int64_t k, int64_t max_seq_len,
    cudaStream_t stream = 0);

// ── Quant ────────────────────────────────────────────────────────────────
void static_scaled_int8_quant(
    Tensor& out, const Tensor& input, const Tensor& scale,
    std::optional<Tensor> azp,
    cudaStream_t stream = 0);

void dynamic_scaled_int8_quant(
    Tensor& out, const Tensor& input, Tensor& scales,
    std::optional<Tensor> azp,
    cudaStream_t stream = 0);

// ── AllReduce (custom AR wrapper, not vllm's protocol) ────────────────────
// These wrap vllm's init_custom_ar / all_reduce / dispose
int64_t custom_all_reduce_init(
    const std::vector<int64_t>& fake_ipc_ptrs, Tensor& rank_data,
    int64_t rank, bool fully_connected,
    cudaStream_t stream = 0);

void custom_all_reduce(int64_t handle, Tensor& input, Tensor& output,
                       int64_t reg_buffer, int64_t reg_buffer_sz_bytes,
                       cudaStream_t stream = 0);

void custom_all_reduce_dispose(int64_t handle);

Tensor minimax_allreduce_rms(
    const Tensor& input, const Tensor& norm_weight,
    Tensor workspace, int64_t rank, int64_t nranks, double eps,
    cudaStream_t stream = 0);

} // namespace vllm_kernels
} // namespace vm_c
