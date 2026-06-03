#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace vm_c {

// ── GDN (GatedDeltaNet) 线性注意力 CUDA kernels ──

// L2 归一化: out = x / sqrt(sum(x^2) + eps)
// x 视为 [num_tokens, heads, head_dim]，对每个 (token, head) 的 head_dim 维度独立归一化
// 当 heads=1 时等价于对所有元素整体归一化
void gdn_l2_norm(void* out, const void* x, int num_tokens, int head_dim,
                 int heads, float eps, int dtype_size, cudaStream_t stream);

// RMSNormGated: out = RMSNorm(x) * SiLU(z)
// x: [num_tokens, dim], z: [num_tokens, dim], weight: [dim], out: [num_tokens, dim]
void gdn_rms_norm_gated(void* out, const void* x, const void* z,
                         const void* weight, int num_tokens, int dim,
                         float eps, int dtype_size, cudaStream_t stream);

// SiLU 激活: out = x * sigmoid(x)
void gdn_silu(void* out, const void* x, int64_t num_elements,
               int dtype_size, cudaStream_t stream);

// Sigmoid: out = 1 / (1 + exp(-x))
void gdn_sigmoid(void* out, const void* x, int64_t num_elements,
                  int dtype_size, cudaStream_t stream);

// Softplus: out = log(1 + exp(x))，数值稳定版本
void gdn_softplus(void* out, const void* x, int64_t num_elements,
                   int dtype_size, cudaStream_t stream);

// 因果卷积更新 (decode 路径，单 token)
// conv_state: [conv_dim, conv_kernel_size-1] per slot，滚动更新
// x: [conv_dim] 输入 token，conv_weight: [conv_dim, conv_kernel_size]
// 更新后 conv_state 并返回卷积输出: [conv_dim]
void gdn_conv1d_update(void* x_out, const void* x_in,
                       void* conv_state, const void* conv_weight,
                       int conv_dim, int conv_kernel_size,
    int slot_offset, int state_stride,
    int dtype_size, cudaStream_t stream);

// 因果卷积 batched decode（多 token 单次 launch）
void gdn_conv1d_update_batched(void* x_out, const void* x_in,
                               void* conv_state, const void* conv_weight,
                               const int32_t* token_to_seq,
                               int num_tokens, int conv_dim, int conv_kernel_size,
                               int conv_time_stride,
                               int dtype_size, cudaStream_t stream);

// 因果卷积 (prefill 路径，多 token)
// conv_state: [conv_dim, conv_kernel_size-1] per slot
// x: [num_tokens, conv_dim]，conv_weight: [conv_dim, conv_kernel_size]
// 输出: [num_tokens, conv_dim]
void gdn_conv1d_prefill(void* x_out, const void* x_in,
                         void* conv_state, const void* conv_weight,
                         int num_tokens, int conv_dim, int conv_kernel_size,
                         int slot_offset, int state_stride,
                         int dtype_size, cudaStream_t stream);

// 类型转换: 将 half/bf16 输入转换为 float 输出
// out: [N] float, x: [N] half/bf16
void gdn_cast_to_float(float* out, const void* x, int64_t num_elements,
                        int dtype_size, cudaStream_t stream);

// 分离 in_proj_ba GEMM 输出: ba [num_tokens, 2*num_v_heads] → beta_raw, alpha_raw
void gdn_split_ba_to_float(float* beta_out, float* alpha_out, const void* ba_in,
                            int num_tokens, int num_v_heads,
                            int dtype_size, cudaStream_t stream);

// 计算 gate 和 sigmoid(beta):
// gate[t, hv] = ssm_a[hv] * softplus(alpha[t, hv] + dt_bias[hv])
//   ssm_a = GGUF blk.{i}.ssm_a = -exp(A_log_raw)（llama.cpp convert 已预计算）
//   recurrent kernel 内 exp(gate) 得到衰减因子
// beta[t, hv] = sigmoid(beta_raw[t, hv])
// A_log/ssm_a: [num_v_heads], dt_bias: [num_v_heads] (广播到每个 token)
// alpha_in: [num_tokens * num_v_heads] float (softplus 输入)
// beta_raw_in: [num_tokens * num_v_heads] float (sigmoid 输入)
// gate_out: [num_tokens * num_v_heads] float
// beta_out: [num_tokens * num_v_heads] float
void gdn_compute_gate_beta(float* gate_out, float* beta_out,
                            const float* alpha_in, const float* beta_raw_in,
                            const void* A_log, const void* dt_bias,
                            int num_tokens, int num_v_heads,
                            int dtype_size, cudaStream_t stream);

// GDN 递推核心 (decode 路径，单 token per request)
// ssm_state: [num_v_heads, head_v_dim, head_k_dim] per slot
// q: [num_k_heads, head_k_dim], k: [num_k_heads, head_k_dim]
// v: [num_v_heads, head_v_dim], gate: [num_v_heads], beta: [num_v_heads]
// A_log: [num_v_heads], dt_bias: [num_v_heads]
// 输出: o [num_v_heads, head_v_dim]
void gdn_recurrent_decode(
    void* o_out, void* ssm_state,
    const void* q, const void* k, const void* v,
    const void* gate, const void* beta,
    const void* A_log, const void* dt_bias,
    int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim,
    float scale_k,
    int slot_offset, int state_stride,
    int dtype_size, cudaStream_t stream);

// GDN 递推核心 (batched decode：多 token 单次 launch)
// token_to_seq: device [num_tokens]，可为 nullptr（等价全 0 slot）
void gdn_recurrent_decode_batched(
    void* o_out, void* ssm_state,
    const void* q, const void* k, const void* v,
    const void* gate, const void* beta,
    const void* A_log, const void* dt_bias,
    const int32_t* token_to_seq,
    int num_tokens, int key_dim, int value_dim,
    int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim,
    float scale_k, int per_head_ssm_stride,
    int dtype_size, cudaStream_t stream);

// GDN 递推核心 (prefill 路径，逐 token 串行处理)
// ssm_state: [num_v_heads, head_v_dim, head_k_dim] per slot
// q: [num_tokens, num_k_heads, head_k_dim]
// k: [num_tokens, num_k_heads, head_k_dim]
// v: [num_tokens, num_v_heads, head_v_dim]
// gate: [num_tokens, num_v_heads], beta: [num_tokens, num_v_heads]
// A_log: [num_v_heads], dt_bias: [num_v_heads]
// 输出: o [num_tokens, num_v_heads, head_v_dim]
void gdn_recurrent_prefill(
    void* o_out, void* ssm_state,
    const void* q, const void* k, const void* v,
    const void* gate, const void* beta,
    const void* A_log, const void* dt_bias,
    int num_tokens, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim,
    float scale_k,
    int slot_offset, int state_stride,
    int dtype_size, cudaStream_t stream,
    float* f32_scratch);

} // namespace vm_c