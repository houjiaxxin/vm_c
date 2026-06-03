#pragma once

#include "vm_c_tensor.h"
#include <cstdint>
#include <string>
#include <optional>

namespace vm_c {

// ── Fused QK Norm + RoPE + KV Cache Store ──
//
// 将以下4个独立操作融合为1个kernel launch：
//   1. Q RMSNorm (q_norm_weight)
//   2. K RMSNorm (k_norm_weight)
//   3. RoPE (cos_sin_cache)
//   4. KV Cache Store (reshape_and_cache)
//
// 参考 llama.cpp 的 ggml_cuda_op_rope_fused 实现：
//   将 RoPE + View + SetRows 融合为一步，避免中间结果写回显存。
//
// 在 vm_c 的 decode 路径中，原始流程是：
//   Q_proj GEMM → memset → fused_add_rms_norm(Q) → memset → fused_add_rms_norm(K) → RoPE → store_kv
//   共 6 个 kernel launch（2个memset + 2个norm + 1个RoPE + 1个store）
//
// 融合后：
//   Q_proj GEMM → K_proj GEMM → V_proj GEMM → fused_qk_norm_rope_store
//   仅 1 个 kernel launch
//
// 参数说明：
//   q_buf:          [num_tokens, num_heads * head_dim] Q投影输出（in-place norm+RoPE）
//   k_buf:          [num_tokens, num_kv_heads * head_dim] K投影输出（in-place norm+RoPE）
//   v_buf:          [num_tokens, num_kv_heads * head_dim] V投影输出（直接store）
//   q_norm_weight:  [num_heads * head_dim] Q的RMSNorm权重（可为nullptr表示不做Q norm）
//   k_norm_weight:  [num_kv_heads * head_dim] K的RMSNorm权重（可为nullptr表示不做K norm）
//   cos_sin_cache:  [max_seq_len, rot_dim] RoPE的cos/sin缓存
//   position_ids:   [num_tokens] 位置ID
//   key_cache:      KV cache key缓冲区
//   value_cache:    KV cache value缓冲区
//   slot_mapping:   [num_tokens] slot映射
//   num_tokens:     token数量
//   num_heads:      Q头数
//   num_kv_heads:   KV头数
//   head_dim:       头维度
//   rms_norm_eps:   RMSNorm epsilon
//   is_neox:        是否使用Neox风格RoPE
//   block_size:     KV cache block大小
//   cache_stride:   KV cache行步长
//   kv_cache_dtype: KV cache数据类型（"auto"表示与compute dtype相同）
//   stream:         CUDA stream

void fused_qk_norm_rope_store(
    void* q_buf,
    void* k_buf,
    const void* v_buf,
    const void* q_norm_weight,
    const void* k_norm_weight,
    const void* cos_sin_cache,
    const int64_t* position_ids,
    void* key_cache,
    void* value_cache,
    const int64_t* slot_mapping,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float rms_norm_eps,
    bool is_neox,
    int64_t block_size,
    int64_t cache_stride,
    const std::string& kv_cache_dtype,
    cudaStream_t stream = 0);

// ── Fused QK Norm + RoPE (不包含 KV Cache Store) ──
//
// 用于 prefill 路径，KV cache store 由 attention kernel 单独处理。
// 将 Q_norm + K_norm + RoPE 融合为1个kernel launch。
//
// 参数同 fused_qk_norm_rope_store，但不包含 key_cache/value_cache/slot_mapping。

void fused_qk_norm_rope(
    void* q_buf,
    void* k_buf,
    const void* q_norm_weight,
    const void* k_norm_weight,
    const void* cos_sin_cache,
    const int64_t* position_ids,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float rms_norm_eps,
    bool is_neox,
    cudaStream_t stream = 0);

} // namespace vm_c