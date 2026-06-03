#include "vm_c/cuda/kernels_gdn.h"
#include "vm_c/cuda/vm_c_dispatch.h"
#include "vm_c/cuda/gpu_arch.hpp"
#include "vm_c/cuda/kernel_utils.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>
#include <spdlog/spdlog.h>

namespace vm_c {

namespace {

// ── L2 归一化 kernel ──
// 支持 per-head 归一化: 将 x 视为 [rows, heads, head_dim]，
// 对每个 (row, head) 的 head_dim 维度独立归一化。
// 当 heads=1 时等价于原来的行为（单组归一化）。
template <typename scalar_t>
__global__ void gdn_l2_norm_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    int head_dim, int heads, float eps) {
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;

    for (int h = 0; h < heads; ++h) {
        const scalar_t* x_head = x + (row_idx * heads + h) * head_dim;
        scalar_t* out_head = out + (row_idx * heads + h) * head_dim;

        float sum_sq = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            float val = to_float(x_head[i]);
            sum_sq += val * val;
        }

        // warp-level reduction for sum_sq
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
        }

        // lane 0 持有完整的 sum_sq，广播给整个 warp
        float total_sq = __shfl_sync(0xffffffff, sum_sq, 0);

        // 对齐 llama.cpp ggml L2_NORM: scale = rsqrt(max(sum(x^2), eps^2))
        float inv_norm = rsqrtf(fmaxf(total_sq, eps * eps));

        for (int i = tid; i < head_dim; i += blockDim.x) {
            out_head[i] = from_float<scalar_t>(to_float(x_head[i]) * inv_norm);
        }
    }
}

// ── RMSNormGated kernel ──
// out = RMSNorm(x) * SiLU(z)
template <typename scalar_t>
__global__ void gdn_rms_norm_gated_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ z,
    const scalar_t* __restrict__ weight,
    int dim, float eps) {
    const int token_idx = blockIdx.x;
    const scalar_t* x_row = x + token_idx * dim;
    const scalar_t* z_row = z + token_idx * dim;
    scalar_t* out_row = out + token_idx * dim;

    __shared__ float s_variance;
    float variance = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = to_float(x_row[i]);
        variance += val * val;
    }

    __shared__ float s_reduce[1024];
    s_reduce[threadIdx.x] = variance;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(s_reduce[0] / dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float x_val = to_float(x_row[i]);
        float z_val = to_float(z_row[i]);
        float w_val = to_float(weight[i]);
        // SiLU(z) = z * sigmoid(z) = z / (1 + exp(-z))
        float silu_z = z_val / (1.0f + expf(-z_val));
        float normed = x_val * inv_rms * w_val;
        out_row[i] = from_float<scalar_t>(normed * silu_z);
    }
}

// ── SiLU kernel ──
template <typename scalar_t>
__global__ void gdn_silu_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    float val = to_float(x[idx]);
    out[idx] = from_float<scalar_t>(val / (1.0f + expf(-val)));
}

// ── Sigmoid kernel ──
template <typename scalar_t>
__global__ void gdn_sigmoid_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    float val = to_float(x[idx]);
    out[idx] = from_float<scalar_t>(1.0f / (1.0f + expf(-val)));
}

// ── Softplus kernel (数值稳定) ──
template <typename scalar_t>
__global__ void gdn_softplus_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    float val = to_float(x[idx]);
    // softplus(x) = log(1 + exp(x))
    // 数值稳定: x > 20 时 softplus(x) ≈ x
    float result = (val > 20.0f) ? val : logf(1.0f + expf(val));
    out[idx] = from_float<scalar_t>(result);
}

// ── 因果卷积更新 kernel (decode, 单 token) ──
// conv_state 布局: [conv_dim, kernel_size-1]，列主序
// conv_weight 布局: [conv_dim, kernel_size]，行主序 (PyTorch 格式)
// 更新: 将 x_in 移入 conv_state，滚动最旧的列出去
// 输出: conv_state 与 conv_weight 的因果卷积结果
template <typename scalar_t>
__global__ void gdn_conv1d_update_kernel(
    scalar_t* __restrict__ x_out,
    const scalar_t* __restrict__ x_in,
    scalar_t* __restrict__ conv_state,
    const scalar_t* __restrict__ conv_weight,
    int conv_dim, int kernel_size,
    int state_stride) {
    // 单线程处理整个 conv_dim（因为只有 1 个 token）
    // conv_state 布局: [conv_dim * (kernel_size-1)]，按列主序
    // 即 conv_state[d * (kernel_size-1) + t] = state[d][t]
    const int k1 = kernel_size - 1;

    // 1. 滚动 conv_state: 将旧状态左移一位，腾出最后一位给新输入
    //    state[d][0..k1-2] = state[d][1..k1-1]
    //    state[d][k1-1] = x_in[d]
    for (int d = threadIdx.x; d < conv_dim; d += blockDim.x) {
        for (int t = 0; t < k1 - 1; ++t) {
            conv_state[d * state_stride + t] = conv_state[d * state_stride + t + 1];
        }
        conv_state[d * state_stride + k1 - 1] = x_in[d];
    }
    __syncthreads();

    // 2. 因果卷积: out[d] = sum_{t=0}^{k1} weight[d][t] * state[d][t]
    //    其中 state[d][k1] = x_in[d] (最新输入)
    //    weight 布局: weight[d * kernel_size + t]
    for (int d = threadIdx.x; d < conv_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int t = 0; t < k1; ++t) {
            float s = to_float(conv_state[d * state_stride + t]);
            float w = to_float(conv_weight[d * kernel_size + t]);
            sum += w * s;
        }
        // 最新输入 x_in[d] 对应 weight[d * kernel_size + k1]
        sum += to_float(conv_weight[d * kernel_size + k1]) * to_float(x_in[d]);
        // 匹配官方: silu(conv(x)) — silu(x) = x * sigmoid(x)
        float silu_val = sum / (1.0f + expf(-sum));
        x_out[d] = from_float<scalar_t>(silu_val);
    }
}

template <typename scalar_t>
__global__ void gdn_conv1d_update_batched_kernel(
    scalar_t* __restrict__ x_out,
    const scalar_t* __restrict__ x_in,
    scalar_t* __restrict__ conv_state,
    const scalar_t* __restrict__ conv_weight,
    const int32_t* __restrict__ token_to_seq,
    int num_tokens, int conv_dim, int kernel_size,
    int state_stride) {
    const int t = blockIdx.x;
    if (t >= num_tokens) {
        return;
    }
    const int slot = token_to_seq ? token_to_seq[t] : 0;
    const int slot_offset = slot * conv_dim;
    const scalar_t* x_t = x_in + static_cast<size_t>(t) * conv_dim;
    scalar_t* x_out_t = x_out + static_cast<size_t>(t) * conv_dim;
    scalar_t* state_ptr = conv_state + slot_offset * state_stride;

    const int k1 = kernel_size - 1;
    for (int d = threadIdx.x; d < conv_dim; d += blockDim.x) {
        for (int ti = 0; ti < k1 - 1; ++ti) {
            state_ptr[d * state_stride + ti] = state_ptr[d * state_stride + ti + 1];
        }
        state_ptr[d * state_stride + k1 - 1] = x_t[d];
    }
    __syncthreads();

    for (int d = threadIdx.x; d < conv_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int ti = 0; ti < k1; ++ti) {
            sum += to_float(state_ptr[d * state_stride + ti])
                 * to_float(conv_weight[d * kernel_size + ti]);
        }
        sum += to_float(conv_weight[d * kernel_size + k1]) * to_float(x_t[d]);
        float silu_val = sum / (1.0f + expf(-sum));
        x_out_t[d] = from_float<scalar_t>(silu_val);
    }
}

// ── 因果卷积 kernel (prefill, 多 token) ──
// conv_state 布局: [conv_dim, kernel_size-1] per slot
// x_in: [num_tokens, conv_dim]，conv_weight: [conv_dim, kernel_size]
// 输出: [num_tokens, conv_dim]
// 对于每个 token t，使用 conv_state 中的历史 + x_in[0..t] 做因果卷积
template <typename scalar_t>
__global__ void gdn_conv1d_prefill_kernel(
    scalar_t* __restrict__ x_out,
    const scalar_t* __restrict__ x_in,
    scalar_t* __restrict__ conv_state,
    const scalar_t* __restrict__ conv_weight,
    int num_tokens, int conv_dim, int kernel_size,
    int state_stride) {
    const int k1 = kernel_size - 1;
    const int d = blockIdx.x;
    if (d >= conv_dim) return;

    // 每个 block 处理一个 conv_dim 维度，逐 token 计算
    for (int t = 0; t < num_tokens; ++t) {
        float sum = 0.0f;
        // 使用 conv_state 中的历史值
        for (int i = 0; i < k1; ++i) {
            float s = to_float(conv_state[d * state_stride + i]);
            float w = to_float(conv_weight[d * kernel_size + i]);
            sum += w * s;
        }
        // 使用当前 token 的输入
        float x_val = to_float(x_in[t * conv_dim + d]);
        sum += to_float(conv_weight[d * kernel_size + k1]) * x_val;

        // 匹配官方: silu(conv(x)) — silu(x) = x * sigmoid(x)
        float silu_val = sum / (1.0f + expf(-sum));
        x_out[t * conv_dim + d] = from_float<scalar_t>(silu_val);

        // 更新 conv_state: 滚动并加入当前 token
        for (int i = 0; i < k1 - 1; ++i) {
            conv_state[d * state_stride + i] = conv_state[d * state_stride + i + 1];
        }
        conv_state[d * state_stride + k1 - 1] = from_float<scalar_t>(x_val);
    }
}

// ── GDN 递推核心 kernel (decode, 单 token per request) ──
// ssm_state 布局: [num_v_heads, head_v_dim, head_k_dim]，列主序
// 即 ssm_state[hv * head_v_dim * head_k_dim + k * head_v_dim + v] = M[v][k]
// 匹配 llama.cpp 官方设计：每列连续（列主序），便于 warp-level 列并行
// q: [num_k_heads, head_k_dim], k: [num_k_heads, head_k_dim]
// v: [num_v_heads, head_v_dim]
// gate: [num_v_heads] (已计算: ssm_a * softplus(alpha + dt_bias)，kernel 内 exp(gate))
// beta: [num_v_heads] (已计算: sigmoid(beta_raw))
template <typename scalar_t>
__global__ void gdn_recurrent_decode_kernel(
    scalar_t* __restrict__ o_out,
    float* __restrict__ ssm_state,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim,
    float scale_k,
    int state_stride) {
    // 每个 block 处理一个 value head
    const int hv = blockIdx.x;
    if (hv >= num_v_heads) return;

    // GQA: 多个 value head 共享 key head
    // num_v_heads / num_k_heads = kv_group_ratio
    const int kv_group = num_v_heads / num_k_heads;
    const int hk = hv / kv_group;  // 对应的 key head

    // 指向当前 value head 的状态矩阵（列主序，匹配 llama.cpp 官方设计）
    // state[hv] 是 [head_v_dim, head_k_dim] 列主序矩阵
    // M[v][k] 在内存中位置 state_hv[k * head_v_dim + v]，每列连续
    float* state_hv = ssm_state + hv * head_v_dim * head_k_dim;

    // 指向当前 key/value head 的 q, k, v
    const scalar_t* q_h = q + hk * head_k_dim;
    const scalar_t* k_h = k + hk * head_k_dim;
    const scalar_t* v_h = v + hv * head_v_dim;

    float g = gate[hv];
    float b = beta[hv];
    float exp_g = expf(g);

    // 使用 shared memory 加速
    // 布局：s_k[head_k_dim], s_q[head_k_dim], s_v[head_v_dim]
    extern __shared__ float smem[];
    float* s_k = smem;
    float* s_q = s_k + head_k_dim;
    float* s_v = s_q + head_k_dim;

    // 加载 k, q, v 到 shared memory
    for (int i = threadIdx.x; i < head_k_dim; i += blockDim.x) {
        s_k[i] = to_float(k_h[i]);   // k 已 L2 归一化
        s_q[i] = to_float(q_h[i]);   // q 已 L2 归一化
    }
    for (int i = threadIdx.x; i < head_v_dim; i += blockDim.x) {
        s_v[i] = to_float(v_h[i]);
    }
    __syncthreads();

    // 统一 compute (匹配官方 gated_delta_net_cuda kernel, KDA=false):
    //   1. kv[v] = sum_k state[v][k] * k[k]  ← 使用原始状态（未衰减）
    //   2. delta[v] = (v[v] - g * kv[v]) * beta  ← g 应用于 kv
    //   3. state_new[v][k] = g * state_old[v][k] + k[k] * delta[v]  ← 融合衰减+更新
    //   4. o[v] = sum_k state_new[v][k] * q[k]

    for (int v = threadIdx.x; v < head_v_dim; v += blockDim.x) {
        // Step 1: kv = sum_k state[v][k] * k[k]
        float kv_val = 0.0f;
        for (int kk = 0; kk < head_k_dim; ++kk) {
            kv_val += state_hv[kk * head_v_dim + v] * s_k[kk];
        }

        // Step 2: delta = (v - g * kv) * beta
        float delta_val = (s_v[v] - exp_g * kv_val) * b;

        // Step 3+4: 融合衰减+更新 + 查询输出
        // state_new[v][k] = g * state[v][k] + k[k] * delta
        // o[v] = sum_k state_new[v][k] * q[k]
        float o_val = 0.0f;
        for (int kk = 0; kk < head_k_dim; ++kk) {
            float old_h = state_hv[kk * head_v_dim + v];
            float new_h = exp_g * old_h + s_k[kk] * delta_val;
            state_hv[kk * head_v_dim + v] = new_h;
            o_val += new_h * s_q[kk];
        }

        // 写回输出（匹配官方：output 乘以 1/sqrt(S_v)）
        const float out_scale = 1.0f / sqrtf((float)head_v_dim);
        o_out[hv * head_v_dim + v] = from_float<scalar_t>(o_val * out_scale);
    }
}

template <typename scalar_t>
__global__ void gdn_recurrent_decode_batched_kernel(
    scalar_t* __restrict__ o_out,
    float* __restrict__ ssm_state,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const int32_t* __restrict__ token_to_seq,
    int num_tokens, int key_dim, int value_dim,
    int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim,
    float scale_k, int per_head_ssm_stride) {
    const int t = blockIdx.x;
    const int hv = blockIdx.y;
    if (t >= num_tokens || hv >= num_v_heads) {
        return;
    }

    const int slot = token_to_seq ? token_to_seq[t] : 0;
    const int kv_group = num_v_heads / num_k_heads;
    const int hk = hv / kv_group;

    const scalar_t* q_h = q + static_cast<size_t>(t) * key_dim + hk * head_k_dim;
    const scalar_t* k_h = k + static_cast<size_t>(t) * key_dim + hk * head_k_dim;
    const scalar_t* v_h = v + static_cast<size_t>(t) * value_dim + hv * head_v_dim;
    const float* gate_h = gate + static_cast<size_t>(t) * num_v_heads + hv;
    const float* beta_h = beta + static_cast<size_t>(t) * num_v_heads + hv;
    scalar_t* o_h = o_out + static_cast<size_t>(t) * value_dim + hv * head_v_dim;

    float* state_hv = ssm_state
        + static_cast<size_t>(slot * num_v_heads + hv) * per_head_ssm_stride;

    float g = gate_h[0];
    float b = beta_h[0];
    float exp_g = expf(g);

    extern __shared__ float smem[];
    float* s_k = smem;
    float* s_q = s_k + head_k_dim;
    float* s_v = s_q + head_k_dim;

    for (int i = threadIdx.x; i < head_k_dim; i += blockDim.x) {
        s_k[i] = to_float(k_h[i]);
        s_q[i] = to_float(q_h[i]);
    }
    for (int i = threadIdx.x; i < head_v_dim; i += blockDim.x) {
        s_v[i] = to_float(v_h[i]);
    }
    __syncthreads();

    for (int v = threadIdx.x; v < head_v_dim; v += blockDim.x) {
        float kv_val = 0.0f;
        for (int kk = 0; kk < head_k_dim; ++kk) {
            kv_val += state_hv[kk * head_v_dim + v] * s_k[kk];
        }
        float delta_val = (s_v[v] - exp_g * kv_val) * b;
        float o_val = 0.0f;
        for (int kk = 0; kk < head_k_dim; ++kk) {
            float old_h = state_hv[kk * head_v_dim + v];
            float new_h = exp_g * old_h + s_k[kk] * delta_val;
            state_hv[kk * head_v_dim + v] = new_h;
            o_val += new_h * s_q[kk];
        }
        const float out_scale = 1.0f / sqrtf(static_cast<float>(head_v_dim));
        o_h[v] = from_float<scalar_t>(o_val * out_scale);
    }
}

// ── GDN 递推核心 kernel (prefill) ──
// 使用 warp-level 并行 + 寄存器状态（参考 llama.cpp gated_delta_net_cuda）
// 每个 warp 处理 state 矩阵的一列，state 存储在 lane 的寄存器中
// 对 token 循环时 state 保持在寄存器中，无需全局内存访问
template <int S_v>
__global__ void gdn_prefill_warp_kernel(
    float* __restrict__ o_out,          // [num_tokens, num_v_heads, S_v] 输出
    float* __restrict__ ssm_state,      // [num_v_heads, S_v, S_v] 状态（FP32 变体）
    const float* __restrict__ q_all,    // [num_tokens, num_k_heads, S_v]
    const float* __restrict__ k_all,    // [num_tokens, num_k_heads, S_v]
    const float* __restrict__ v_all,    // [num_tokens, num_v_heads, S_v]
    const float* __restrict__ gate_all, // [num_tokens, num_v_heads]
    const float* __restrict__ beta_all, // [num_tokens, num_v_heads]
    int num_tokens, int num_k_heads, int num_v_heads,
    float scale_k) {
    // Grid: (num_v_heads, 1, (S_v+3)/4)
    // Block: (warpSize, 4, 1)   — 每个 block 有 4 个 warp，每 warp 处理一列
    const int hv = blockIdx.x;
    if (hv >= num_v_heads) return;
    const int kv_group = num_v_heads / num_k_heads;
    const int hk = hv / kv_group;

    const int warp_id = threadIdx.y;
    const int lane = threadIdx.x;
    constexpr int warp_size = 32;
    // 每个 lane 处理的行数：S_v < 32 时每 lane 处理 1 行（多出的 lane 为 padding）
    constexpr int rows_per_lane = (S_v < warp_size) ? 1 : (S_v / warp_size);

    // 当前 warp 处理的列: col = (blockIdx.z * 4 + warp_id)
    const int col = blockIdx.z * 4 + warp_id;
    if (col >= S_v) return;

    // 当前 value head 的 state 矩阵: [S_v, S_v]，列主序
    // 列 col 连续存储：M[row][col] = state_hv[col * S_v + row]
    // 当前 warp 处理列 col，指向该列起始位置
    float* state_col = ssm_state + hv * S_v * S_v + col * S_v;

    // 加载状态列到寄存器（每个 lane 持有 rows_per_lane 个元素）
    // row < S_v 时才有效，否则 padding 0
    float s_shard[rows_per_lane];
    #pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        int row = r * warp_size + lane;
        s_shard[r] = (row < S_v) ? state_col[row] : 0.0f;
    }

    // 逐 token 递推
    for (int t = 0; t < num_tokens; t++) {
        const float* q_t = q_all + (t * num_k_heads + hk) * S_v;
        const float* k_t = k_all + (t * num_k_heads + hk) * S_v;
        const float* v_t = v_all + (t * num_v_heads + hv) * S_v;
        float g_val = expf(gate_all[t * num_v_heads + hv]);
        float b_val = beta_all[t * num_v_heads + hv];

        // 加载 k 和 q 到寄存器（无需 scale，匹配官方 CUDA kernel）
        float k_reg[rows_per_lane], q_reg[rows_per_lane];
        #pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            int row = r * warp_size + lane;
            k_reg[r] = (row < S_v) ? k_t[row] : 0.0f;
            q_reg[r] = (row < S_v) ? q_t[row] : 0.0f;
        }

        // kv[col] = sum_i g * S[i][col] * k[i]
        float kv_shard = 0.0f;
        #pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard += s_shard[r] * k_reg[r];
        }
        // warp-level reduction: 所有 lane 的 kv_shard 求和
        for (int offset = 16; offset > 0; offset /= 2)
            kv_shard += __shfl_xor_sync(0xffffffff, kv_shard, offset);
        // lane 0 持有完整的 kv[col]
        float kv_col = kv_shard;

        // delta[col] = (v[col] - g * kv[col]) * beta
        float delta_col = (v_t[col] - g_val * kv_col) * b_val;

        // 融合: S[i][col] = g * S[i][col] + k[i] * delta[col]
        //       attn_partial = sum_i S[i][col] * q[i]
        float attn_shard = 0.0f;
        #pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
            attn_shard += s_shard[r] * q_reg[r];
        }
        // warp-level reduction for attn
        for (int offset = 16; offset > 0; offset /= 2)
            attn_shard += __shfl_xor_sync(0xffffffff, attn_shard, offset);

        // lane 0 写回 attn 输出
        float* o_h = o_out + (t * num_v_heads + hv) * S_v;
        if (lane == 0) {
            o_h[col] = attn_shard * (1.0f / sqrtf((float)S_v));
        }
    }

    // 写回最终状态列（仅有效行）
    #pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        int row = r * warp_size + lane;
        if (row < S_v) {
            state_col[row] = s_shard[r];
        }
    }
}

// ── 类型转换 kernel: half/bf16 -> float ──
template <typename scalar_t>
__global__ void gdn_cast_to_float_kernel(
    float* __restrict__ out,
    const scalar_t* __restrict__ x,
    int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    out[idx] = to_float(x[idx]);
}

// ba_buf [num_tokens, 2*num_v_heads] → beta_raw / alpha_raw（各自 [num_tokens, num_v_heads]）
template <typename scalar_t>
__global__ void gdn_split_ba_kernel(
    float* __restrict__ beta_out,
    float* __restrict__ alpha_out,
    const scalar_t* __restrict__ ba_in,
    int num_tokens, int num_v_heads) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(num_tokens) * num_v_heads;
    if (idx >= total) return;
    int t = static_cast<int>(idx / num_v_heads);
    int h = static_cast<int>(idx % num_v_heads);
    const scalar_t* row = ba_in + static_cast<int64_t>(t) * (2 * num_v_heads);
    beta_out[idx] = to_float(row[h]);
    alpha_out[idx] = to_float(row[num_v_heads + h]);
}

// ── 计算 gate 和 sigmoid(beta) kernel ──
// 对齐 llama.cpp qwen35moe + convert/qwen.py:
//   GGUF ssm_a (vm_c A_log) 已是 -exp(A_log_raw)，gate = ssm_a * softplus(alpha + dt_bias)
//   gated_delta_net CUDA kernel 内对 gate 做 exp(gate) 得到衰减因子
// beta[t, hv] = sigmoid(beta_raw[t, hv])
template <typename scalar_t>
__global__ void gdn_compute_gate_beta_kernel(
    float* __restrict__ gate_out,
    float* __restrict__ beta_out,
    const float* __restrict__ alpha_in,
    const float* __restrict__ beta_raw_in,
    const scalar_t* __restrict__ A_log,
    const scalar_t* __restrict__ dt_bias,
    int num_tokens, int num_v_heads) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(num_tokens) * num_v_heads;
    if (idx >= total) return;

    int hv = idx % num_v_heads;

    // ssm_a per-head 常量（GGUF 中已预计算为 -exp(A_log_raw)）
    float ssm_a = to_float(A_log[hv]);
    float dt_bias_val = to_float(dt_bias[hv]);

    float alpha_val = alpha_in[idx] + dt_bias_val;
    float sp_val = (alpha_val > 20.0f) ? alpha_val : logf(1.0f + expf(alpha_val));
    gate_out[idx] = ssm_a * sp_val;

    // sigmoid(beta_raw)
    float b_raw = beta_raw_in[idx];
    beta_out[idx] = 1.0f / (1.0f + expf(-b_raw));
}

} // anonymous namespace

// ── Host launchers ──

void gdn_l2_norm(void* out, const void* x, int num_tokens, int head_dim,
                 int heads, float eps, int dtype_size, cudaStream_t stream) {
    dim3 grid(num_tokens);
    dim3 block(std::min(head_dim, 32));
    if (dtype_size == 2) {
        VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "gdn_l2_norm", {
            gdn_l2_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
                static_cast<scalar_t*>(out), static_cast<const scalar_t*>(x),
                head_dim, heads, eps);
        });
    } else {
        gdn_l2_norm_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<float*>(out), static_cast<const float*>(x),
            head_dim, heads, eps);
    }
}

void gdn_rms_norm_gated(void* out, const void* x, const void* z,
                         const void* weight, int num_tokens, int dim,
                         float eps, int dtype_size, cudaStream_t stream) {
    dim3 grid(num_tokens);
    dim3 block(std::min(dim, 1024));
    if (dtype_size == 2) {
        VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "gdn_rms_norm_gated", {
            gdn_rms_norm_gated_kernel<scalar_t><<<grid, block, 0, stream>>>(
                static_cast<scalar_t*>(out),
                static_cast<const scalar_t*>(x),
                static_cast<const scalar_t*>(z),
                static_cast<const scalar_t*>(weight),
                dim, eps);
        });
    } else {
        gdn_rms_norm_gated_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<float*>(out), static_cast<const float*>(x),
            static_cast<const float*>(z), static_cast<const float*>(weight),
            dim, eps);
    }
}

void gdn_silu(void* out, const void* x, int64_t num_elements,
               int dtype_size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    if (dtype_size == 2) {
        VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "gdn_silu", {
            gdn_silu_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                static_cast<scalar_t*>(out), static_cast<const scalar_t*>(x),
                num_elements);
        });
    } else {
        gdn_silu_kernel<float><<<blocks, threads, 0, stream>>>(
            static_cast<float*>(out), static_cast<const float*>(x), num_elements);
    }
}

void gdn_sigmoid(void* out, const void* x, int64_t num_elements,
                  int dtype_size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    if (dtype_size == 2) {
        VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "gdn_sigmoid", {
            gdn_sigmoid_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                static_cast<scalar_t*>(out), static_cast<const scalar_t*>(x),
                num_elements);
        });
    } else {
        gdn_sigmoid_kernel<float><<<blocks, threads, 0, stream>>>(
            static_cast<float*>(out), static_cast<const float*>(x), num_elements);
    }
}

void gdn_softplus(void* out, const void* x, int64_t num_elements,
                   int dtype_size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    if (dtype_size == 2) {
        VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "gdn_softplus", {
            gdn_softplus_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                static_cast<scalar_t*>(out), static_cast<const scalar_t*>(x),
                num_elements);
        });
    } else {
        gdn_softplus_kernel<float><<<blocks, threads, 0, stream>>>(
            static_cast<float*>(out), static_cast<const float*>(x), num_elements);
    }
}

void gdn_cast_to_float(float* out, const void* x, int64_t num_elements,
                        int dtype_size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    if (dtype_size == 2) {
        VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "gdn_cast_to_float", {
            gdn_cast_to_float_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                out, static_cast<const scalar_t*>(x), num_elements);
        });
    } else {
        // float -> float: 直接 memcpy
        CUDA_CHECK(cudaMemcpyAsync(out, x, num_elements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }
}

void gdn_split_ba_to_float(float* beta_out, float* alpha_out, const void* ba_in,
                            int num_tokens, int num_v_heads,
                            int dtype_size, cudaStream_t stream) {
    int64_t total = static_cast<int64_t>(num_tokens) * num_v_heads;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    if (dtype_size == 2) {
        VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "gdn_split_ba", {
            gdn_split_ba_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                beta_out, alpha_out, static_cast<const scalar_t*>(ba_in),
                num_tokens, num_v_heads);
        });
    } else {
        gdn_split_ba_kernel<float><<<blocks, threads, 0, stream>>>(
            beta_out, alpha_out, static_cast<const float*>(ba_in),
            num_tokens, num_v_heads);
    }
    CHECK_KERNEL_LAUNCH();
}

void gdn_compute_gate_beta(float* gate_out, float* beta_out,
                            const float* alpha_in, const float* beta_raw_in,
                            const void* A_log, const void* dt_bias,
                            int num_tokens, int num_v_heads,
                            int dtype_size, cudaStream_t stream) {
    int64_t total = static_cast<int64_t>(num_tokens) * num_v_heads;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (dtype_size == 2) {
        VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "gdn_compute_gate_beta", {
            gdn_compute_gate_beta_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                gate_out, beta_out, alpha_in, beta_raw_in,
                static_cast<const scalar_t*>(A_log), static_cast<const scalar_t*>(dt_bias),
                num_tokens, num_v_heads);
        });
    } else {
        gdn_compute_gate_beta_kernel<float><<<blocks, threads, 0, stream>>>(
            gate_out, beta_out, alpha_in, beta_raw_in,
            static_cast<const float*>(A_log), static_cast<const float*>(dt_bias),
            num_tokens, num_v_heads);
    }
}

void gdn_conv1d_update(void* x_out, const void* x_in,
                       void* conv_state, const void* conv_weight,
                       int conv_dim, int conv_kernel_size,
                       int slot_offset, int state_stride,
                       int dtype_size, cudaStream_t stream) {
    dim3 block(std::min(conv_dim, 1024));
    dim3 grid(1);
    if (dtype_size == 2) {
        VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "gdn_conv1d_update", {
            scalar_t* state_ptr = static_cast<scalar_t*>(conv_state) + slot_offset * state_stride;
            gdn_conv1d_update_kernel<scalar_t><<<grid, block, 0, stream>>>(
                static_cast<scalar_t*>(x_out),
                static_cast<const scalar_t*>(x_in),
                state_ptr,
                static_cast<const scalar_t*>(conv_weight),
                conv_dim, conv_kernel_size, state_stride);
        });
    } else {
        float* state_ptr = static_cast<float*>(conv_state) + slot_offset * state_stride;
        gdn_conv1d_update_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<float*>(x_out),
            static_cast<const float*>(x_in),
            state_ptr,
            static_cast<const float*>(conv_weight),
            conv_dim, conv_kernel_size, state_stride);
    }
}

void gdn_conv1d_update_batched(void* x_out, const void* x_in,
                               void* conv_state, const void* conv_weight,
                               const int32_t* token_to_seq,
                               int num_tokens, int conv_dim, int conv_kernel_size,
                               int conv_time_stride,
                               int dtype_size, cudaStream_t stream) {
    (void)conv_time_stride;
    dim3 block(std::min(conv_dim, 1024));
    dim3 grid(num_tokens);
    if (dtype_size == 2) {
        VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "gdn_conv1d_update_batched", {
            gdn_conv1d_update_batched_kernel<scalar_t><<<grid, block, 0, stream>>>(
                static_cast<scalar_t*>(x_out),
                static_cast<const scalar_t*>(x_in),
                static_cast<scalar_t*>(conv_state),
                static_cast<const scalar_t*>(conv_weight),
                token_to_seq,
                num_tokens, conv_dim, conv_kernel_size, conv_time_stride);
        });
    } else {
        gdn_conv1d_update_batched_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<float*>(x_out),
            static_cast<const float*>(x_in),
            static_cast<float*>(conv_state),
            static_cast<const float*>(conv_weight),
            token_to_seq,
            num_tokens, conv_dim, conv_kernel_size, conv_time_stride);
    }
}

void gdn_conv1d_prefill(void* x_out, const void* x_in,
                         void* conv_state, const void* conv_weight,
                         int num_tokens, int conv_dim, int conv_kernel_size,
                         int slot_offset, int state_stride,
                         int dtype_size, cudaStream_t stream) {
    dim3 block(1);
    dim3 grid(conv_dim);
    if (dtype_size == 2) {
        VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "gdn_conv1d_prefill", {
            scalar_t* state_ptr = static_cast<scalar_t*>(conv_state) + slot_offset * state_stride;
            gdn_conv1d_prefill_kernel<scalar_t><<<grid, block, 0, stream>>>(
                static_cast<scalar_t*>(x_out),
                static_cast<const scalar_t*>(x_in),
                state_ptr,
                static_cast<const scalar_t*>(conv_weight),
                num_tokens, conv_dim, conv_kernel_size, state_stride);
        });
    } else {
        float* state_ptr = static_cast<float*>(conv_state) + slot_offset * state_stride;
        gdn_conv1d_prefill_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<float*>(x_out),
            static_cast<const float*>(x_in),
            state_ptr,
            static_cast<const float*>(conv_weight),
            num_tokens, conv_dim, conv_kernel_size, state_stride);
    }
}

void gdn_recurrent_decode(
    void* o_out, void* ssm_state,
    const void* q, const void* k, const void* v,
    const void* gate, const void* beta,
    const void* A_log, const void* dt_bias,
    int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim,
    float scale_k,
    int slot_offset, int state_stride,
    int dtype_size, cudaStream_t stream) {
    dim3 grid(num_v_heads);
    dim3 block(std::min(head_v_dim * head_k_dim, 512));
    size_t smem_size = (head_k_dim * 2 + head_v_dim * 4) * sizeof(float);

    float* state_ptr = static_cast<float*>(ssm_state) + slot_offset * state_stride;
    if (dtype_size == 2) {
        VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "gdn_recurrent_decode", {
            gdn_recurrent_decode_kernel<scalar_t><<<grid, block, smem_size, stream>>>(
                static_cast<scalar_t*>(o_out),
                state_ptr,
                static_cast<const scalar_t*>(q),
                static_cast<const scalar_t*>(k),
                static_cast<const scalar_t*>(v),
                static_cast<const float*>(gate),
                static_cast<const float*>(beta),
                static_cast<const float*>(A_log),
                static_cast<const float*>(dt_bias),
                num_k_heads, num_v_heads,
                head_k_dim, head_v_dim,
                scale_k, state_stride);
        });
    } else {
        gdn_recurrent_decode_kernel<float><<<grid, block, smem_size, stream>>>(
            static_cast<float*>(o_out),
            state_ptr,
            static_cast<const float*>(q),
            static_cast<const float*>(k),
            static_cast<const float*>(v),
            static_cast<const float*>(gate),
            static_cast<const float*>(beta),
            static_cast<const float*>(A_log),
            static_cast<const float*>(dt_bias),
            num_k_heads, num_v_heads,
            head_k_dim, head_v_dim,
            scale_k, state_stride);
    }
}

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
    int dtype_size, cudaStream_t stream) {
    (void)A_log;
    (void)dt_bias;
    (void)scale_k;

    dim3 grid(num_tokens, num_v_heads);
    dim3 block(std::min(head_v_dim * head_k_dim, 512));
    size_t smem_size = (head_k_dim * 2 + head_v_dim * 4) * sizeof(float);

    if (dtype_size == 2) {
        VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "gdn_recurrent_decode_batched", {
            gdn_recurrent_decode_batched_kernel<scalar_t><<<grid, block, smem_size, stream>>>(
                static_cast<scalar_t*>(o_out),
                static_cast<float*>(ssm_state),
                static_cast<const scalar_t*>(q),
                static_cast<const scalar_t*>(k),
                static_cast<const scalar_t*>(v),
                static_cast<const float*>(gate),
                static_cast<const float*>(beta),
                static_cast<const float*>(A_log),
                static_cast<const float*>(dt_bias),
                token_to_seq,
                num_tokens, key_dim, value_dim,
                num_k_heads, num_v_heads,
                head_k_dim, head_v_dim,
                scale_k, per_head_ssm_stride);
        });
    } else {
        gdn_recurrent_decode_batched_kernel<float><<<grid, block, smem_size, stream>>>(
            static_cast<float*>(o_out),
            static_cast<float*>(ssm_state),
            static_cast<const float*>(q),
            static_cast<const float*>(k),
            static_cast<const float*>(v),
            static_cast<const float*>(gate),
            static_cast<const float*>(beta),
            static_cast<const float*>(A_log),
            static_cast<const float*>(dt_bias),
            token_to_seq,
            num_tokens, key_dim, value_dim,
            num_k_heads, num_v_heads,
            head_k_dim, head_v_dim,
            scale_k, per_head_ssm_stride);
    }
}

// FP32 → half/bf16 批量转换 kernel（gdn_cast_to_float 的逆操作）
template <typename scalar_t>
__global__ void gdn_cast_from_float_kernel(
    scalar_t* __restrict__ out,
    const float* __restrict__ x,
    int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    out[idx] = from_float<scalar_t>(x[idx]);
}

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
    float* f32_scratch) {
    if (head_k_dim != head_v_dim) {
        spdlog::error("gdn_recurrent_prefill: head_k_dim({}) != head_v_dim({}) not supported",
                       head_k_dim, head_v_dim);
        return;
    }
    if (dtype_size != 4 && !f32_scratch) {
        spdlog::error("gdn_recurrent_prefill: f32_scratch required for half/bf16 activations");
        return;
    }
    const int S_v = head_k_dim;

    // ssm_state 由 engine 分配为 FP32，与 activation dtype 无关
    float* f32_state = static_cast<float*>(ssm_state) + slot_offset * state_stride;

    // 如果 activation 已是 FP32，直接调用 warp kernel
    if (dtype_size == 4) {
        constexpr int num_warps = 4;
        dim3 grid(num_v_heads, 1, (S_v + num_warps - 1) / num_warps);
        dim3 block(32, num_warps, 1);
        switch (S_v) {
            case 16: gdn_prefill_warp_kernel<16><<<grid, block, 0, stream>>>(
                (float*)o_out, f32_state, (const float*)q, (const float*)k, (const float*)v,
                (const float*)gate, (const float*)beta, num_tokens, num_k_heads, num_v_heads, scale_k); break;
            case 32: gdn_prefill_warp_kernel<32><<<grid, block, 0, stream>>>(
                (float*)o_out, f32_state, (const float*)q, (const float*)k, (const float*)v,
                (const float*)gate, (const float*)beta, num_tokens, num_k_heads, num_v_heads, scale_k); break;
            case 64: gdn_prefill_warp_kernel<64><<<grid, block, 0, stream>>>(
                (float*)o_out, f32_state, (const float*)q, (const float*)k, (const float*)v,
                (const float*)gate, (const float*)beta, num_tokens, num_k_heads, num_v_heads, scale_k); break;
            case 128: gdn_prefill_warp_kernel<128><<<grid, block, 0, stream>>>(
                (float*)o_out, f32_state, (const float*)q, (const float*)k, (const float*)v,
                (const float*)gate, (const float*)beta, num_tokens, num_k_heads, num_v_heads, scale_k); break;
            default: spdlog::error("gdn_recurrent_prefill: unsupported S_v={}", S_v);
        }
        return;
    }

    // FP16/bf16 activation：仅 q/k/v/output 做 dtype 转换，state 保持 FP32
    int total_q = num_tokens * num_k_heads * S_v;
    int total_k = num_tokens * num_k_heads * S_v;
    int total_v = num_tokens * num_v_heads * S_v;
    int total_out = num_tokens * num_v_heads * S_v;
    int total = total_q + total_k + total_v + total_out;
    float* fp32_buf = f32_scratch;
    if (!fp32_buf) {
        spdlog::error("gdn_recurrent_prefill: scratch too small (need {} floats)", total);
        return;
    }
    float* f32_q = fp32_buf;
    float* f32_k = f32_q + total_q;
    float* f32_v = f32_k + total_k;
    float* f32_out = f32_v + total_v;

    VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "gdn_prefill_to_f32", {
        gdn_cast_to_float_kernel<<<(total_q+255)/256, 256, 0, stream>>>(f32_q, (const scalar_t*)q, total_q);
        gdn_cast_to_float_kernel<<<(total_k+255)/256, 256, 0, stream>>>(f32_k, (const scalar_t*)k, total_k);
        gdn_cast_to_float_kernel<<<(total_v+255)/256, 256, 0, stream>>>(f32_v, (const scalar_t*)v, total_v);
    });

    constexpr int num_warps = 4;
    dim3 grid(num_v_heads, 1, (S_v + num_warps - 1) / num_warps);
    dim3 block(32, num_warps, 1);

    switch (S_v) {
        case 16: gdn_prefill_warp_kernel<16><<<grid, block, 0, stream>>>(
            f32_out, f32_state, f32_q, f32_k, f32_v,
            (const float*)gate, (const float*)beta, num_tokens, num_k_heads, num_v_heads, scale_k); break;
        case 32: gdn_prefill_warp_kernel<32><<<grid, block, 0, stream>>>(
            f32_out, f32_state, f32_q, f32_k, f32_v,
            (const float*)gate, (const float*)beta, num_tokens, num_k_heads, num_v_heads, scale_k); break;
        case 64: gdn_prefill_warp_kernel<64><<<grid, block, 0, stream>>>(
            f32_out, f32_state, f32_q, f32_k, f32_v,
            (const float*)gate, (const float*)beta, num_tokens, num_k_heads, num_v_heads, scale_k); break;
        case 128: gdn_prefill_warp_kernel<128><<<grid, block, 0, stream>>>(
            f32_out, f32_state, f32_q, f32_k, f32_v,
            (const float*)gate, (const float*)beta, num_tokens, num_k_heads, num_v_heads, scale_k); break;
        default: spdlog::error("gdn_recurrent_prefill: unsupported S_v={}", S_v);
    }

    // 仅 output 转回 FP16/bf16；state 已在 FP32 buffer 内原地更新
    VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "gdn_prefill_from_f32", {
        gdn_cast_from_float_kernel<<<(total_out+255)/256, 256, 0, stream>>>((scalar_t*)o_out, f32_out, total_out);
    });
}

} // namespace vm_c