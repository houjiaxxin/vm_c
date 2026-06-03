#include "vm_c/cuda/kernels_moe.h"
#include "vm_c/cuda/vm_c_dispatch.h"
#include "vm_c/cuda/gpu_arch.hpp"
#include "vm_c/cuda/kernel_utils.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace vm_c {
namespace {

template <typename scalar_t>
__device__ __forceinline__ void atomicAdd_half_type(scalar_t* addr, float val);

template <>
__device__ __forceinline__ void atomicAdd_half_type(__nv_bfloat16* addr, float val) {
    unsigned short* uaddr = reinterpret_cast<unsigned short*>(addr);
    unsigned short old = *uaddr;
    unsigned short assumed;
    do {
        assumed = old;
        float sum = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&assumed)) + val;
        __nv_bfloat16 bf = __float2bfloat16(sum);
        old = atomicCAS(uaddr, assumed, *reinterpret_cast<unsigned short*>(&bf));
    } while (assumed != old);
}

template <>
__device__ __forceinline__ void atomicAdd_half_type(half* addr, float val) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(addr, __float2half(val));
#else
    unsigned short* uaddr = reinterpret_cast<unsigned short*>(addr);
    unsigned short old = *uaddr;
    unsigned short assumed;
    do {
        assumed = old;
        float sum = __half2float(*reinterpret_cast<const half*>(&assumed)) + val;
        half h = __float2half(sum);
        old = atomicCAS(uaddr, assumed, *reinterpret_cast<unsigned short*>(&h));
    } while (assumed != old);
#endif
}

#define DEFINE_TOPKGATING(scalar_t, NUM_EXPERTS)                          \
  __launch_bounds__(128) __global__                                        \
  void topkGating_kernel_##NUM_EXPERTS##_##scalar_t(                       \
      const scalar_t* __restrict__ input,                                  \
      float* __restrict__ output,                                          \
      int32_t* __restrict__ indices,                                       \
      int32_t* __restrict__ source_rows,                                   \
      const int num_rows,                                                  \
      const int k,                                                         \
      const bool renormalize) {                                            \
    constexpr int VPT = NUM_EXPERTS / 32;                                  \
    constexpr int ROWS_PER_CTA = 4;                                        \
    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;                    \
    const int thread_row = cta_base_row + threadIdx.y;                     \
    if (thread_row >= num_rows) return;                                    \
    const scalar_t* row_ptr = input + thread_row * NUM_EXPERTS;            \
    float rc[VPT];                                                         \
    for (int i = 0; i < VPT; ++i)                                          \
      rc[i] = to_float(row_ptr[i * 32 + threadIdx.x]);                     \
    float tmax = rc[0];                                                    \
    for (int i = 1; i < VPT; ++i) tmax = fmaxf(tmax, rc[i]);              \
    for (int mask = 16; mask > 0; mask >>= 1)                              \
      tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, mask));         \
    float rsum = 0;                                                         \
    for (int i = 0; i < VPT; ++i) {                                        \
      rc[i] = expf(rc[i] - tmax); rsum += rc[i];                           \
    }                                                                      \
    for (int mask = 16; mask > 0; mask >>= 1)                              \
      rsum += __shfl_xor_sync(0xffffffff, rsum, mask);                     \
    float inv_rsum = 1.0f / rsum;                                          \
    for (int i = 0; i < VPT; ++i) rc[i] = rc[i] * inv_rsum;               \
    for (int i = 0; i < VPT; ++i)                                          \
      if (isnan(rc[i]) || isinf(rc[i])) rc[i] = 0.0f;                      \
    int start_col = threadIdx.x;                                           \
    float sel_sum = 0.0f;                                                  \
    for (int kidx = 0; kidx < k; ++kidx) {                                 \
      float mv = rc[0];                                                    \
      int expert = start_col;                                              \
      for (int i = 0; i < VPT; ++i) {                                      \
        int col = start_col + i * 32;                                      \
        if (rc[i] > mv) { mv = rc[i]; expert = col; }                      \
      }                                                                    \
      for (int mask = 16; mask > 0; mask >>= 1) {                          \
        float om = __shfl_xor_sync(0xffffffff, mv, mask);                   \
        int oe = __shfl_xor_sync(0xffffffff, expert, mask);                 \
        if (om > mv || (om == mv && oe < expert)) { mv = om; expert = oe; } \
      }                                                                    \
      __syncwarp();                                                        \
      if (threadIdx.x == 0) {                                              \
        const int idx = thread_row * k + kidx;                              \
        output[idx] = mv;                                                  \
        indices[idx] = expert;                                             \
        source_rows[idx] = kidx * num_rows + thread_row;                    \
        if (renormalize) sel_sum += mv;                                     \
      }                                                                    \
      if (kidx + 1 < k) {                                                   \
        int ck = expert / 32;                                              \
        int tc = expert % 32;                                              \
        if (threadIdx.x == tc) rc[ck] = -10000.0f;                         \
        __syncwarp();                                                      \
      }                                                                    \
    }                                                                      \
    if (renormalize && threadIdx.x == 0) {                                  \
      float isum = (sel_sum > 0.0f) ? (1.0f / sel_sum) : 1.0f;            \
      for (int kidx = 0; kidx < k; ++kidx)                                  \
        output[thread_row * k + kidx] *= isum;                              \
    }                                                                      \
  }

DEFINE_TOPKGATING(__nv_bfloat16, 32)
DEFINE_TOPKGATING(__nv_bfloat16, 64)
DEFINE_TOPKGATING(__nv_bfloat16, 128)
DEFINE_TOPKGATING(__nv_bfloat16, 256)
DEFINE_TOPKGATING(half, 32)
DEFINE_TOPKGATING(half, 64)
DEFINE_TOPKGATING(half, 128)
DEFINE_TOPKGATING(half, 256)

#undef DEFINE_TOPKGATING

template <typename scalar_t, int TPB>
__launch_bounds__(TPB) __global__
void moeSoftmax_kernel(
    const scalar_t* __restrict__ input,
    float* __restrict__ output,
    const int num_cols) {

    const int thread_row_offset = blockIdx.x * num_cols;
    __shared__ float float_max;
    __shared__ float normalizing_factor;
    __shared__ float s_buf[TPB / 32];

    float threadData = -FLT_MAX;
    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        const float val = to_float(input[thread_row_offset + ii]);
        threadData = fmaxf(val, threadData);
    }

    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    int num_warps = TPB / 32;

    for (int offset = 16; offset > 0; offset /= 2)
        threadData = fmaxf(threadData, __shfl_down_sync(0xffffffff, threadData, offset));
    if (lane == 0) s_buf[wid] = threadData;
    __syncthreads();

    if (wid == 0) {
        float my_val = (lane < num_warps) ? s_buf[lane] : -FLT_MAX;
        for (int offset = 16; offset > 0; offset /= 2)
            my_val = fmaxf(my_val, __shfl_down_sync(0xffffffff, my_val, offset));
        if (lane == 0) float_max = my_val;
    }
    __syncthreads();

    threadData = 0.0f;
    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        const float val = to_float(input[thread_row_offset + ii]);
        threadData += expf(val - float_max);
    }
    for (int offset = 16; offset > 0; offset /= 2)
        threadData += __shfl_down_sync(0xffffffff, threadData, offset);
    if (lane == 0) s_buf[wid] = threadData;
    __syncthreads();

    if (wid == 0) {
        float my_val = (lane < num_warps) ? s_buf[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2)
            my_val += __shfl_down_sync(0xffffffff, my_val, offset);
        if (lane == 0) normalizing_factor = 1.0f / my_val;
    }
    __syncthreads();

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        const float val = to_float(input[thread_row_offset + ii]);
        float softmax_val = expf(val - float_max) * normalizing_factor;
        if (isnan(softmax_val) || isinf(softmax_val)) softmax_val = 0.0f;
        output[thread_row_offset + ii] = softmax_val;
    }
}

template <int TPB>
__launch_bounds__(TPB) __global__
void moeTopK_kernel(
    const float* __restrict__ inputs_after_softmax,
    float* __restrict__ output,
    int32_t* __restrict__ indices,
    int32_t* __restrict__ source_rows,
    const int num_experts,
    const int k,
    const bool renormalize) {

    const int block_row = blockIdx.x;
    const int thread_read_offset = blockIdx.x * num_experts;
    __shared__ int32_t s_best_idx;
    __shared__ float s_best_val;
    __shared__ float s_val_buf[TPB / 32];
    __shared__ int32_t s_idx_buf[TPB / 32];

    float selected_sum = 0.0f;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
        float thread_best_val = -1.0f;
        int32_t thread_best_idx = 0;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
            bool already_selected = false;
            for (int prior_k = 0; prior_k < k_idx; ++prior_k)
                if (indices[k * block_row + prior_k] == expert) { already_selected = true; break; }
            if (already_selected) continue;
            float val = inputs_after_softmax[thread_read_offset + expert];
            if (val > thread_best_val) { thread_best_val = val; thread_best_idx = expert; }
        }
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;
        int num_warps = TPB / 32;
        float warp_best_val = thread_best_val;
        int32_t warp_best_idx = thread_best_idx;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, warp_best_val, offset);
            int32_t other_idx = __shfl_down_sync(0xffffffff, warp_best_idx, offset);
            if (other_val > warp_best_val) { warp_best_val = other_val; warp_best_idx = other_idx; }
        }
        if (lane == 0) { s_val_buf[wid] = warp_best_val; s_idx_buf[wid] = warp_best_idx; }
        __syncthreads();
        if (wid == 0) {
            float my_val = (lane < num_warps) ? s_val_buf[lane] : -1.0f;
            int32_t my_idx = (lane < num_warps) ? s_idx_buf[lane] : 0;
            for (int offset = 16; offset > 0; offset /= 2) {
                float other_val = __shfl_down_sync(0xffffffff, my_val, offset);
                int32_t other_idx = __shfl_down_sync(0xffffffff, my_idx, offset);
                if (other_val > my_val) { my_val = other_val; my_idx = other_idx; }
            }
            if (lane == 0) { s_best_val = my_val; s_best_idx = my_idx; }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const int32_t expert = s_best_idx;
            const int idx = k * block_row + k_idx;
            output[idx] = inputs_after_softmax[thread_read_offset + expert];
            indices[idx] = expert;
            source_rows[idx] = k_idx * gridDim.x + block_row;
            if (renormalize) selected_sum += inputs_after_softmax[thread_read_offset + expert];
        }
        __syncthreads();
    }
    if (renormalize && threadIdx.x == 0) {
        float inv_sum = (selected_sum > 0.0f) ? (1.0f / selected_sum) : 1.0f;
        for (int k_idx = 0; k_idx < k; ++k_idx)
            output[k * block_row + k_idx] *= inv_sum;
    }
}

__global__ void exclusive_scan_kernel(int32_t* data, int n) {
    __shared__ int32_t s_data[256];
    int tid = threadIdx.x;
    if (tid < n) s_data[tid] = data[tid];
    else s_data[tid] = 0;
    __syncthreads();

    for (int offset = 1; offset < n; offset <<= 1) {
        int32_t tmp = 0;
        if (tid >= offset) tmp = s_data[tid - offset];
        __syncthreads();
        if (tid >= offset) s_data[tid] += tmp;
        __syncthreads();
    }

    if (tid < n) {
        int32_t val = (tid == 0) ? 0 : s_data[tid - 1];
        data[tid] = val;
    }
}

template <typename scalar_t>
__global__ void gather_tokens_kernel(
    scalar_t* __restrict__ expert_input,
    const scalar_t* __restrict__ hs_ptr,
    const int32_t* __restrict__ sorted_token_ids,
    int total_tokens, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_tokens * hidden_size;
    for (int i = idx; i < total; i += gridDim.x * blockDim.x) {
        int ti = i / hidden_size;
        int h  = i % hidden_size;
        int t  = sorted_token_ids[ti];
        expert_input[ti * hidden_size + h] = hs_ptr[t * hidden_size + h];
    }
}

__global__ void gather_weights_kernel(
    float* __restrict__ sorted_weights,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ topk_ids,
    const int32_t* __restrict__ expert_offsets,
    int num_entries, int top_k, int num_experts) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_entries; i += gridDim.x * blockDim.x) {
        int e = topk_ids[i];
        if (e >= 0 && e < num_experts) {
            int slot = atomicAdd(const_cast<int32_t*>(&expert_offsets[e]), 1);
            sorted_weights[slot] = topk_weights[i];
        }
    }
}
__global__ void count_tokens_per_expert_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ expert_counts,
    int num_entries, int num_experts) {
    extern __shared__ int32_t s_cnt[];
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) s_cnt[i] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_entries; i += gridDim.x * blockDim.x) {
        int e = topk_ids[i];
        if (e >= 0 && e < num_experts) atomicAdd(&s_cnt[e], 1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x)
        if (s_cnt[i] > 0) atomicAdd(&expert_counts[i], s_cnt[i]);
}

__global__ void fill_sorted_token_ids_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_counters,
    const int32_t* __restrict__ expert_offsets,
    int num_entries, int top_k, int num_experts) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_entries; i += gridDim.x * blockDim.x) {
        int token = i / top_k;
        int e = topk_ids[i];
        if (e >= 0 && e < num_experts) {
            int slot = atomicAdd(&expert_counters[e], 1);
            sorted_token_ids[expert_offsets[e] + slot] = token;
        }
    }
}

__global__ void compute_total_kernel(
    int32_t* total,
    const int32_t* expert_offsets,
    const int32_t* expert_counts,
    int num_experts) {
    *total = expert_offsets[num_experts - 1] + expert_counts[num_experts - 1];
}

template <typename scalar_t>
__global__ void expert_scale_kernel(
    scalar_t* __restrict__ scaled_out,
    const scalar_t* __restrict__ expert_out,
    const float* __restrict__ weights,
    int total_tokens, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_tokens * hidden_size;
    for (int i = idx; i < total; i += gridDim.x * blockDim.x) {
        int ti = i / hidden_size;
        float w = weights[ti];
        float v = to_float(expert_out[i]);
        scaled_out[i] = from_float<scalar_t>(v * w);
    }
}

template <typename scalar_t>
__global__ void moe_scatter_add_kernel(
    float* __restrict__ moe_output_fp32,
    const scalar_t* __restrict__ expert_out,
    const float* __restrict__ weights,
    const int32_t* __restrict__ token_ids,
    int total_tokens, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_tokens * hidden_size;
    for (int i = idx; i < total; i += gridDim.x * blockDim.x) {
        int ti = i / hidden_size;
        int h  = i % hidden_size;
        int t  = token_ids[ti];
        float w = weights[ti];
        float v = to_float(expert_out[i]);
        atomicAdd(&moe_output_fp32[t * hidden_size + h], v * w);
    }
}

template <typename scalar_t>
__global__ void fp32_to_half_kernel(
    scalar_t* __restrict__ dst,
    const float* __restrict__ src,
    int64_t total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t i = idx; i < total; i += gridDim.x * blockDim.x) {
        dst[i] = from_float<scalar_t>(src[i]);
    }
}

template <typename scalar_t>
__global__ void expert_scale_add_kernel(
    scalar_t* __restrict__ moe_output,
    const scalar_t* __restrict__ expert_out,
    const float* __restrict__ weights,
    const int32_t* __restrict__ token_ids,
    int total_tokens, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_tokens * hidden_size;
    for (int i = idx; i < total; i += gridDim.x * blockDim.x) {
        int ti = i / hidden_size;
        int h  = i % hidden_size;
        int t  = token_ids[ti];
        float w = weights[ti];
        float v = to_float(expert_out[i]);
        atomicAdd_half_type(&moe_output[t * hidden_size + h], v * w);
    }
}

template <typename scalar_t, int TPB>
__launch_bounds__(TPB) __global__ void grouped_topk_kernel(
    float* __restrict__ topk_weights_out,
    int32_t* __restrict__ topk_indices_out,
    const scalar_t* __restrict__ scores,
    int num_experts, int n_group, int topk_group, int top_k,
    bool renormalize, double routed_scaling_factor,
    int scoring_func) {

    const int row = blockIdx.x;
    const int experts_per_group = num_experts / n_group;
    const scalar_t* row_scores = scores + row * num_experts;

    __shared__ float s_group_max[64];
    __shared__ int32_t s_group_topk_ids[64 * 8];

    for (int g = threadIdx.x; g < n_group; g += TPB) {
        float gmax = -FLT_MAX;
        for (int j = 0; j < experts_per_group; ++j) {
            int e = g * experts_per_group + j;
            float s = to_float(row_scores[e]);
            if (scoring_func == 1) s = sqrtf(s);
            if (s > gmax) gmax = s;
        }
        s_group_max[g] = gmax;
    }
    __syncthreads();

    for (int g = threadIdx.x; g < n_group; g += TPB) {
        float gmax = s_group_max[g];
        for (int kg = 0; kg < topk_group && kg < experts_per_group; ++kg) {
            float best_val = -FLT_MAX;
            int32_t best_idx = -1;
            for (int j = 0; j < experts_per_group; ++j) {
                int e = g * experts_per_group + j;
                bool already = false;
                for (int p = 0; p < kg; ++p) {
                    if (s_group_topk_ids[g * topk_group + p] == e) { already = true; break; }
                }
                if (already) continue;
                float s = to_float(row_scores[e]);
                if (scoring_func == 1) s = sqrtf(s);
                if (s > best_val) { best_val = s; best_idx = e; }
            }
            s_group_topk_ids[g * topk_group + kg] = best_idx;
        }
    }
    __syncthreads();

    __shared__ int32_t s_candidate_ids[64 * 8];
    __shared__ float s_candidate_scores[64 * 8];
    int n_candidates = n_group * topk_group;

    for (int i = threadIdx.x; i < n_candidates; i += TPB) {
        int g = i / topk_group;
        int kg = i % topk_group;
        int e = s_group_topk_ids[g * topk_group + kg];
        s_candidate_ids[i] = e;
        if (e >= 0) {
            float s = to_float(row_scores[e]);
            if (scoring_func == 1) s = sqrtf(s);
            s_candidate_scores[i] = expf(s - s_group_max[g]);
        } else {
            s_candidate_scores[i] = 0.0f;
        }
    }
    __syncthreads();

    float sel_sum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        float best_val = -FLT_MAX;
        int32_t best_idx = -1;
        for (int i = 0; i < n_candidates; ++i) {
            bool already = false;
            for (int p = 0; p < k; ++p) {
                if (topk_indices_out[row * top_k + p] == s_candidate_ids[i]) { already = true; break; }
            }
            if (already) continue;
            if (s_candidate_scores[i] > best_val) { best_val = s_candidate_scores[i]; best_idx = i; }
        }

        if (threadIdx.x == 0) {
            int out_idx = row * top_k + k;
            topk_weights_out[out_idx] = static_cast<float>(routed_scaling_factor) * best_val;
            topk_indices_out[out_idx] = s_candidate_ids[best_idx];
            if (renormalize) sel_sum += best_val;
        }
        __syncthreads();
    }

    if (renormalize && threadIdx.x == 0 && sel_sum > 0.0f) {
        float inv = 1.0f / sel_sum;
        for (int k = 0; k < top_k; ++k)
            topk_weights_out[row * top_k + k] *= inv;
    }
}

template <typename scalar_t, int TPB>
__launch_bounds__(TPB) __global__
void topk_softplus_sqrt_kernel(
    float* __restrict__ topk_weights_out,
    int32_t* __restrict__ topk_indices_out,
    int32_t* __restrict__ source_rows,
    const scalar_t* __restrict__ gating_output,
    int num_experts, int top_k,
    bool renormalize, double routed_scaling_factor) {

    const int row = blockIdx.x;

    __shared__ float s_scores[256];
    for (int e = threadIdx.x; e < num_experts; e += TPB) {
        float val = to_float(gating_output[row * num_experts + e]);
        val = fmaxf(val, 0.0f);
        val = sqrtf(val);
        val = logf(1.0f + expf(val));
        s_scores[e] = val;
    }
    __syncthreads();

    float sel_sum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        float best_val = -FLT_MAX;
        int32_t best_idx = 0;
        for (int e = threadIdx.x; e < num_experts; e += TPB) {
            bool already = false;
            for (int p = 0; p < k; ++p) {
                if (topk_indices_out[row * top_k + p] == e) { already = true; break; }
            }
            if (already) continue;
            if (s_scores[e] > best_val) { best_val = s_scores[e]; best_idx = e; }
        }

        __shared__ float s_val_buf[TPB / 32];
        __shared__ int32_t s_idx_buf[TPB / 32];
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;
        int num_warps = TPB / 32;

        for (int offset = 16; offset > 0; offset /= 2) {
            float ov = __shfl_down_sync(0xffffffff, best_val, offset);
            int32_t oi = __shfl_down_sync(0xffffffff, best_idx, offset);
            if (ov > best_val) { best_val = ov; best_idx = oi; }
        }
        if (lane == 0) { s_val_buf[wid] = best_val; s_idx_buf[wid] = best_idx; }
        __syncthreads();

        if (wid == 0) {
            float my_val = (lane < num_warps) ? s_val_buf[lane] : -FLT_MAX;
            int32_t my_idx = (lane < num_warps) ? s_idx_buf[lane] : 0;
            for (int offset = 16; offset > 0; offset /= 2) {
                float ov = __shfl_down_sync(0xffffffff, my_val, offset);
                int32_t oi = __shfl_down_sync(0xffffffff, my_idx, offset);
                if (ov > my_val) { my_val = ov; my_idx = oi; }
            }
            if (lane == 0) {
                int out_idx = row * top_k + k;
                topk_weights_out[out_idx] = static_cast<float>(routed_scaling_factor) * my_val;
                topk_indices_out[out_idx] = my_idx;
                source_rows[out_idx] = k * gridDim.x + row;
                if (renormalize) sel_sum += my_val;
            }
        }
        __syncthreads();
    }

    if (renormalize && threadIdx.x == 0 && sel_sum > 0.0f) {
        float inv = 1.0f / sel_sum;
        for (int k = 0; k < top_k; ++k)
            topk_weights_out[row * top_k + k] *= inv;
    }
}

template <typename scalar_t>
__global__ void moe_sum_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    int num_tokens, int top_k, int64_t hidden_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(num_tokens) * hidden_size;
    for (int64_t i = idx; i < total; i += gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            sum += to_float(input[k * num_tokens * hidden_size + i]);
        }
        output[i] = from_float<scalar_t>(sum);
    }
}

} // anonymous namespace

void topk_softmax(
    void* topk_weights, void* topk_indices, void* token_expert_indices,
    const void* gating_output,
    int num_tokens, int num_experts, int top_k,
    bool renormalize, const void* bias,
    ScalarType dtype, cudaStream_t stream,
    void* softmax_workspace) {
    (void)bias;

    if (num_tokens <= 0) {
        return;
    }
    if (!topk_weights || !topk_indices || !token_expert_indices || !gating_output) {
        throw std::runtime_error("topk_softmax: null buffer pointer");
    }
    if (num_experts <= 0 || top_k <= 0 || top_k > num_experts) {
        throw std::runtime_error(
            "topk_softmax: invalid shape num_tokens=" + std::to_string(num_tokens)
            + " num_experts=" + std::to_string(num_experts)
            + " top_k=" + std::to_string(top_k));
    }

    auto* output = static_cast<float*>(topk_weights);
    auto* indices = static_cast<int32_t*>(topk_indices);
    auto* source_rows = static_cast<int32_t*>(token_expert_indices);

    bool is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    bool use_topk_gating = is_pow_2 && num_experts >= 32 && num_experts <= 256
        && dtype != ScalarType::FLOAT32;

    if (use_topk_gating) {
        constexpr int ROWS_PER_CTA = 4;
        int num_blocks = (num_tokens + ROWS_PER_CTA - 1) / ROWS_PER_CTA;
        dim3 block_dim(32, ROWS_PER_CTA);

        if (dtype == ScalarType::BFLOAT16) {
            auto* gating = static_cast<const __nv_bfloat16*>(gating_output);
            switch (num_experts) {
                case 32:  topkGating_kernel_32___nv_bfloat16<<<num_blocks, block_dim, 0, stream>>>(gating, output, indices, source_rows, num_tokens, top_k, renormalize); break;
                case 64:  topkGating_kernel_64___nv_bfloat16<<<num_blocks, block_dim, 0, stream>>>(gating, output, indices, source_rows, num_tokens, top_k, renormalize); break;
                case 128: topkGating_kernel_128___nv_bfloat16<<<num_blocks, block_dim, 0, stream>>>(gating, output, indices, source_rows, num_tokens, top_k, renormalize); break;
                case 256: topkGating_kernel_256___nv_bfloat16<<<num_blocks, block_dim, 0, stream>>>(gating, output, indices, source_rows, num_tokens, top_k, renormalize); break;
                default:
                    throw std::runtime_error(
                        "topk_softmax: unsupported num_experts=" + std::to_string(num_experts));
            }
        } else if (dtype == ScalarType::FLOAT16) {
            auto* gating = static_cast<const half*>(gating_output);
            switch (num_experts) {
                case 32:  topkGating_kernel_32_half<<<num_blocks, block_dim, 0, stream>>>(gating, output, indices, source_rows, num_tokens, top_k, renormalize); break;
                case 64:  topkGating_kernel_64_half<<<num_blocks, block_dim, 0, stream>>>(gating, output, indices, source_rows, num_tokens, top_k, renormalize); break;
                case 128: topkGating_kernel_128_half<<<num_blocks, block_dim, 0, stream>>>(gating, output, indices, source_rows, num_tokens, top_k, renormalize); break;
                case 256: topkGating_kernel_256_half<<<num_blocks, block_dim, 0, stream>>>(gating, output, indices, source_rows, num_tokens, top_k, renormalize); break;
                default:
                    throw std::runtime_error(
                        "topk_softmax: unsupported num_experts=" + std::to_string(num_experts));
            }
        } else {
            throw std::runtime_error(
                "topk_softmax: fast path requires FLOAT16/BFLOAT16 gating, got dtype="
                + std::to_string(static_cast<int>(dtype)));
        }
        CHECK_KERNEL_LAUNCH();
        return;
    }
    // Slow path
    {
        size_t ws = num_tokens * num_experts * sizeof(float);
        float* w = static_cast<float*>(softmax_workspace);
        bool own = (w == nullptr);
        if (own) {
            CUDA_CHECK(cudaMalloc(&w, ws));
        }
        constexpr int TPB = 256;
        VMC_DISPATCH_HALF_TYPES(dtype, "moe_softmax_topk",
            auto* gating = static_cast<const scalar_t*>(gating_output);
            moeSoftmax_kernel<scalar_t, TPB><<<num_tokens, TPB, 0, stream>>>(gating, w, num_experts);
        );
        moeTopK_kernel<TPB><<<num_tokens, TPB, 0, stream>>>(w, output, indices, source_rows, num_experts, top_k, renormalize);
        CHECK_KERNEL_LAUNCH();
        if (own) CUDA_CHECK(cudaFree(w));
    }
}

void topk_softplus_sqrt(
    void* topk_weights, void* topk_indices, void* token_expert_indices,
    const void* gating_output,
    int num_tokens, int num_experts, int top_k,
    bool renormalize, double routed_scaling_factor,
    const void* correction_bias, const void* input_ids, const void* tid2eid,
    ScalarType dtype, cudaStream_t stream) {
    auto* output = static_cast<float*>(topk_weights);
    auto* indices = static_cast<int32_t*>(topk_indices);
    auto* source_rows = static_cast<int32_t*>(token_expert_indices);
    constexpr int TPB = 256;
    VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "topk_softplus_sqrt",
        auto* gating = static_cast<const scalar_t*>(gating_output);
        topk_softplus_sqrt_kernel<scalar_t, TPB><<<num_tokens, TPB, 0, stream>>>(
            output, indices, source_rows, gating,
            num_experts, top_k, renormalize, routed_scaling_factor);
    );
}

void moe_align_block_size(
    const int32_t* topk_ids,
    int num_tokens, int num_experts, int top_k,
    int32_t* sorted_token_ids,
    int32_t* expert_ids,
    int32_t* total_tokens,
    cudaStream_t stream,
    int32_t* prealloc_temp_counters) {

    const int num_entries = num_tokens * top_k;
    const int threads = 256;
    CUDA_CHECK(cudaMemsetAsync(expert_ids, 0, num_experts * sizeof(int32_t), stream));
    int shared = num_experts * sizeof(int32_t);
    int blocks = std::min(256, (num_entries + threads - 1) / threads);
    count_tokens_per_expert_kernel<<<blocks, threads, shared, stream>>>(
        topk_ids, expert_ids, num_entries, num_experts);

    exclusive_scan_kernel<<<1, num_experts, 0, stream>>>(expert_ids, num_experts);

    int32_t* temp_counters = prealloc_temp_counters;
    bool owns_temp = false;
    if (!temp_counters) {
        CUDA_CHECK(cudaMallocAsync(&temp_counters, num_experts * sizeof(int32_t), stream));
        owns_temp = true;
    }
    CUDA_CHECK(cudaMemsetAsync(temp_counters, 0, num_experts * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(sorted_token_ids, 0, num_entries * sizeof(int32_t), stream));
    fill_sorted_token_ids_kernel<<<blocks, threads, 0, stream>>>(
        topk_ids, sorted_token_ids, temp_counters, expert_ids, num_entries, top_k, num_experts);

    if (total_tokens) {
        int32_t* d_total;
        CUDA_CHECK(cudaMallocAsync(&d_total, sizeof(int32_t), stream));
        compute_total_kernel<<<1, 1, 0, stream>>>(
            d_total, expert_ids, temp_counters, num_experts);
        CUDA_CHECK(cudaMemcpyAsync(total_tokens, d_total, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaFreeAsync(d_total, stream));
    }
    if (owns_temp) {
        CUDA_CHECK(cudaFreeAsync(temp_counters, stream));
    }
}

void moe_sum(
    const void* input, void* output,
    int num_tokens, int num_experts, int top_k, int64_t hidden_size,
    ScalarType dtype,
    cudaStream_t stream) {
    int threads = 256;
    int64_t total = static_cast<int64_t>(num_tokens) * hidden_size;
    int blocks = (total + threads - 1) / threads;
    VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "moe_sum",
        moe_sum_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            static_cast<scalar_t*>(output),
            static_cast<const scalar_t*>(input),
            num_tokens, top_k, hidden_size);
    );
    CHECK_KERNEL_LAUNCH();
    CHECK_KERNEL_LAUNCH();
}

void grouped_topk(
    void* topk_weights_out, void* topk_indices_out,
    const void* scores, ScalarType score_dtype,
    int num_tokens, int num_experts,
    int n_group, int topk_group, int top_k,
    bool renormalize, double routed_scaling_factor,
    const void* bias, int scoring_func,
    cudaStream_t stream) {
    (void)bias; // bias not used in grouped_topk kernel
    auto* weights_out = static_cast<float*>(topk_weights_out);
    auto* indices_out = static_cast<int32_t*>(topk_indices_out);
    constexpr int TPB = 256;
    VMC_DISPATCH_FLOATING_TYPES(score_dtype, "grouped_topk",
        grouped_topk_kernel<scalar_t, TPB><<<num_tokens, TPB, 0, stream>>>(
            weights_out, indices_out,
            static_cast<const scalar_t*>(scores),
            num_experts, n_group, topk_group, top_k,
            renormalize, routed_scaling_factor, scoring_func);
    );
}

void dsv3_router_gemm(
    void* output, const void* mat_a, const void* mat_b,
    int num_tokens, int num_experts, int hidden_dim,
    ScalarType dtype,
    cudaStream_t stream) {
}

void moe_gather_tokens(
    void* expert_input, const void* hs_ptr,
    const int32_t* sorted_token_ids,
    int total_tokens, int hidden_size,
    cudaStream_t stream) {
    int threads = 256;
    int blocks = std::min(256, (total_tokens * hidden_size + threads - 1) / threads);
    VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "moe_gather_tokens",
        gather_tokens_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            static_cast<scalar_t*>(expert_input),
            static_cast<const scalar_t*>(hs_ptr),
            sorted_token_ids, total_tokens, hidden_size);
    );
    CHECK_KERNEL_LAUNCH();
}

void moe_gather_weights(
    float* sorted_weights, const float* topk_weights,
    const int32_t* topk_ids, const int32_t* expert_offsets,
    int num_entries, int top_k, int num_experts,
    cudaStream_t stream) {
    int threads = 256;
    int blocks = std::min(256, (num_entries + threads - 1) / threads);
    gather_weights_kernel<<<blocks, threads, 0, stream>>>(
        sorted_weights, topk_weights, topk_ids,
        expert_offsets, num_entries, top_k, num_experts);
}

template <typename scalar_t>
void moe_scale_add_impl(
    void* moe_output, const void* expert_out,
    const float* weights, const int32_t* token_ids,
    int total_tokens, int num_tokens, int hidden_size,
    float* moe_output_fp32_workspace,
    cudaStream_t stream) {
    int threads = 256;
    int64_t total_elements = static_cast<int64_t>(total_tokens) * hidden_size;
    int blocks = (total_elements + threads - 1) / threads;

    if (moe_output_fp32_workspace) {
        CUDA_CHECK(cudaMemsetAsync(moe_output_fp32_workspace, 0,
                        total_elements * sizeof(float), stream));
        moe_scatter_add_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            moe_output_fp32_workspace,
            static_cast<const scalar_t*>(expert_out),
            weights, token_ids, total_tokens, hidden_size);
        // 只拷贝 num_tokens*hidden_size 个元素到 moe_output，
        // 而非 total_tokens*hidden_size（FP32 workspace 比输出大 topk 倍）
        fp32_to_half_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            static_cast<scalar_t*>(moe_output),
            moe_output_fp32_workspace,
            static_cast<int64_t>(num_tokens) * hidden_size);
    } else {
        expert_scale_add_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            static_cast<scalar_t*>(moe_output),
            static_cast<const scalar_t*>(expert_out),
            weights, token_ids, total_tokens, hidden_size);
    }
}

void moe_scale_add(
    void* moe_output, const void* expert_out,
    const float* weights, const int32_t* token_ids,
    int total_tokens, int num_tokens, int hidden_size,
    float* moe_output_fp32_workspace,
    ScalarType dtype, cudaStream_t stream) {
    VMC_DISPATCH_HALF_TYPES(dtype, "moe_scale_add",
        moe_scale_add_impl<scalar_t>(
            moe_output, expert_out, weights, token_ids,
            total_tokens, num_tokens, hidden_size,
            moe_output_fp32_workspace, stream);
    );
}

namespace {

template <typename scalar_t>
__global__ void fused_moe_silu_and_mul_kernel(
    scalar_t* __restrict__ gate_up_buf,
    int total_elements,
    int intermediate_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int token_idx = idx / intermediate_size;
    int dim_idx = idx % intermediate_size;

    float gate = to_float(gate_up_buf[token_idx * 2 * intermediate_size + dim_idx]);
    float up = to_float(gate_up_buf[token_idx * 2 * intermediate_size + intermediate_size + dim_idx]);
    float silu_gate = gate / (1.0f + expf(-gate));
    gate_up_buf[token_idx * intermediate_size + dim_idx] = from_float<scalar_t>(silu_gate * up);
}

}

void fused_moe_experts(
    const void* hidden_states,
    const void* w1_gate,
    const void* w1_up,
    const void* w2_down,
    void* output,
    const int32_t* sorted_token_ids,
    const int32_t* expert_offsets,
    const float* sorted_weights,
    int num_tokens,
    int num_experts,
    int top_k,
    int hidden_size,
    int intermediate_size,
    vm_c::ScalarType dtype,
    cudaStream_t stream,
    void* workspace) {
    (void)sorted_token_ids;
    (void)expert_offsets;
    (void)sorted_weights;
    (void)num_experts;
    (void)top_k;
    (void)workspace;

    size_t elem_size = vm_c::dtype_size(dtype);
    CUDA_CHECK(cudaMemsetAsync(output, 0, num_tokens * hidden_size * elem_size, stream));

    int total_elements = num_tokens * intermediate_size;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    VMC_DISPATCH_FLOATING_TYPES(dtype, "fused_moe_silu",
        fused_moe_silu_and_mul_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            static_cast<scalar_t*>(output),
            total_elements,
            intermediate_size);
    );
}

} // namespace vm_c
