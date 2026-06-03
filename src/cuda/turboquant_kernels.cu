#include "vm_c/cuda/kernels_turboquant.h"
#include "vm_c/cuda/vm_c_dispatch.h"
#include "vm_c/cuda/gpu_arch.hpp"
#include "vm_c/cuda/kernel_utils.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace vm_c {

namespace {

template <typename scalar_t>
__device__ __forceinline__ void store_half_bits(float val, uint8_t* ptr) {
    scalar_t h = from_float<scalar_t>(val);
    uint16_t bits;
    memcpy(&bits, &h, 2);
    ptr[0] = (uint8_t)(bits & 0xFF);
    ptr[1] = (uint8_t)((bits >> 8) & 0xFF);
}

template <typename scalar_t>
__device__ __forceinline__ float load_half_bits(const uint8_t* ptr) {
    uint16_t bits = (uint16_t)ptr[0] | ((uint16_t)ptr[1] << 8);
    scalar_t h;
    memcpy(&h, &bits, 2);
    return to_float(h);
}

__device__ __forceinline__ int64_t compute_slot_byte_offset(
    int64_t blk, int64_t off, int head_idx,
    int64_t block_size, int num_kv_heads, int slot_size_aligned) {
    return blk * block_size * num_kv_heads * slot_size_aligned
         + off * num_kv_heads * slot_size_aligned
         + head_idx * slot_size_aligned;
}

__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint32_t sign = (bits >> 31) & 1;
    int32_t exp_f32 = (int32_t)((bits >> 23) & 0xFF);
    uint32_t mant_f32 = bits & 0x7FFFFF;

    if (exp_f32 == 0 && mant_f32 == 0) return (uint8_t)(sign << 7);

    if (exp_f32 == 255) return (uint8_t)((sign << 7) | 0x7C);

    int32_t exp_unbiased = exp_f32 - 127;
    int32_t exp_fp8 = exp_unbiased + 7;

    if (exp_fp8 < 1) {
        int32_t shift = 1 - exp_fp8;
        if (shift > 24) return (uint8_t)(sign << 7);
        uint32_t mant = (0x800000 | mant_f32) >> shift;
        uint32_t mant_fp8 = (mant + 4) >> 4;
        if (mant_fp8 > 7) mant_fp8 = 7;
        return (uint8_t)((sign << 7) | (1 << 3) | mant_fp8);
    }

    if (exp_fp8 > 15) {
        return (uint8_t)((sign << 7) | 0x7C | 0x07);
    }

    uint32_t mant_fp8 = (mant_f32 + 0x40000) >> 19;
    if (mant_fp8 > 7) {
        mant_fp8 = 0;
        exp_fp8++;
        if (exp_fp8 > 15) {
            return (uint8_t)((sign << 7) | 0x7C | 0x07);
        }
    }
    return (uint8_t)((sign << 7) | ((exp_fp8 & 0xF) << 3) | (mant_fp8 & 0x7));
}

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t bits) {
    uint32_t sign = (bits >> 7) & 1;
    uint32_t exp = (bits >> 3) & 0xF;
    uint32_t mant = bits & 0x7;

    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;

    float value = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
    return sign ? -value : value;
}

__device__ __forceinline__ uint8_t float_to_fp8_e4b15(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint32_t sign = (bits >> 31) & 1;
    int32_t exp_f32 = (int32_t)((bits >> 23) & 0xFF);
    uint32_t mant_f32 = bits & 0x7FFFFF;

    if (exp_f32 == 0 && mant_f32 == 0) return (uint8_t)(sign << 7);

    if (exp_f32 == 255) return (uint8_t)((sign << 7) | 0x78);

    int32_t exp_unbiased = exp_f32 - 127;
    int32_t exp_fp8 = exp_unbiased + 15;

    if (exp_fp8 < 1) {
        int32_t shift = 1 - exp_fp8;
        if (shift > 24) return (uint8_t)(sign << 7);
        uint32_t mant = (0x800000 | mant_f32) >> shift;
        uint32_t mant_fp8 = (mant + 4) >> 4;
        if (mant_fp8 > 7) mant_fp8 = 7;
        return (uint8_t)((sign << 7) | (1 << 3) | mant_fp8);
    }

    if (exp_fp8 > 15) {
        return (uint8_t)((sign << 7) | 0x78 | 0x07);
    }

    uint32_t mant_fp8 = (mant_f32 + 0x40000) >> 19;
    if (mant_fp8 > 7) {
        mant_fp8 = 0;
        exp_fp8++;
        if (exp_fp8 > 15) {
            return (uint8_t)((sign << 7) | 0x78 | 0x07);
        }
    }
    return (uint8_t)((sign << 7) | ((exp_fp8 & 0xF) << 3) | (mant_fp8 & 0x7));
}

__device__ __forceinline__ float fp8_e4b15_to_float(uint8_t bits) {
    uint32_t sign = (bits >> 7) & 1;
    uint32_t exp = (bits >> 3) & 0xF;
    uint32_t mant = bits & 0x7;

    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;

    float value = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 15);
    return sign ? -value : value;
}

template <typename scalar_t>
__global__ void __launch_bounds__(256)
half_to_float_norm_kernel(
    const scalar_t* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ norms,
    int NH, int head_dim) {
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= NH) return;
    float norm_sq = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        float v = to_float(input[pid * head_dim + d]);
        output[pid * head_dim + d] = v;
        norm_sq += v * v;
    }
    norms[pid] = sqrtf(norm_sq);
}

__global__ void __launch_bounds__(256)
normalize_kernel(
    const float* __restrict__ input,
    const float* __restrict__ norms,
    float* __restrict__ output,
    int NH, int head_dim) {
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= NH) return;
    float inv_norm = 1.0f / (norms[pid] + 1e-8f);
    for (int d = 0; d < head_dim; ++d) {
        output[pid * head_dim + d] = input[pid * head_dim + d] * inv_norm;
    }
}

template <typename scalar_t>
__global__ void __launch_bounds__(256)
half_to_float_kernel(
    const scalar_t* __restrict__ input,
    float* __restrict__ output,
    int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    output[idx] = to_float(input[idx]);
}

template <typename scalar_t>
__global__ void __launch_bounds__(256)
half_to_fp8_store_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    uint8_t* __restrict__ kv_cache,
    const int64_t* __restrict__ slot_mapping,
    int num_tokens, int num_kv_heads, int head_dim,
    int64_t block_size, int key_packed_size,
    int val_data_bytes, int slot_size_aligned,
    bool use_e4b15) {
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_tokens * num_kv_heads;
    if (pid >= total) return;

    const int token_idx = pid / num_kv_heads;
    const int head_idx = pid % num_kv_heads;

    const int64_t slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const int64_t blk = slot / block_size;
    const int64_t off = slot % block_size;
    const int64_t slot_byte = compute_slot_byte_offset(
        blk, off, head_idx, block_size, num_kv_heads, slot_size_aligned);

    uint8_t* slot_ptr = kv_cache + slot_byte;
    const scalar_t* k_ptr = key + pid * head_dim;
    const scalar_t* v_ptr = value + pid * head_dim;

    for (int d = 0; d < head_dim; ++d) {
        float kv = to_float(k_ptr[d]);
        if (use_e4b15) {
            slot_ptr[d] = float_to_fp8_e4b15(kv);
        } else {
            slot_ptr[d] = float_to_fp8_e4m3(kv);
        }
    }

    float v_float[256];
    float v_min = FLT_MAX, v_max = -FLT_MAX;
    for (int d = 0; d < head_dim; ++d) {
        v_float[d] = to_float(v_ptr[d]);
        v_min = fminf(v_min, v_float[d]);
        v_max = fmaxf(v_max, v_float[d]);
    }
    float v_scale = (v_max - v_min) / 15.0f;
    v_scale = fmaxf(v_scale, 1e-8f);

    for (int d = 0; d < head_dim; d += 2) {
        int q0 = min(max((int)roundf((v_float[d] - v_min) / v_scale), 0), 15);
        int q1 = min(max((int)roundf((v_float[d + 1] - v_min) / v_scale), 0), 15);
        slot_ptr[key_packed_size + d / 2] = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
    }

    store_half_bits<scalar_t>(v_scale, slot_ptr + key_packed_size + val_data_bytes);
    store_half_bits<scalar_t>(v_min, slot_ptr + key_packed_size + val_data_bytes + 2);
}

template <typename scalar_t, int MSE_BITS, int VQB, bool NORM_CORRECTION>
__global__ void __launch_bounds__(256)
tq_store_mse_kernel(
    const float* __restrict__ y_rotated,
    const float* __restrict__ norms,
    const scalar_t* __restrict__ value,
    uint8_t* __restrict__ kv_cache,
    const int64_t* __restrict__ slot_mapping,
    const float* __restrict__ midpoints,
    int num_tokens, int num_kv_heads, int head_dim,
    int64_t block_size, int mse_bytes, int key_packed_size,
    int val_data_bytes, int n_centroids, int slot_size_aligned) {
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_tokens * num_kv_heads;
    if (pid >= total) return;

    const int token_idx = pid / num_kv_heads;
    const int head_idx = pid % num_kv_heads;

    const int64_t slot = slot_mapping[token_idx];
    if (slot < 0) return;

    const int64_t blk = slot / block_size;
    const int64_t off = slot % block_size;
    const int64_t slot_byte = compute_slot_byte_offset(
        blk, off, head_idx, block_size, num_kv_heads, slot_size_aligned);

    uint8_t* slot_ptr = kv_cache + slot_byte;
    const float* y_ptr = y_rotated + pid * head_dim;
    const scalar_t* v_ptr = value + pid * head_dim;

    if constexpr (MSE_BITS == 4) {
        for (int d = 0; d < head_dim; d += 2) {
            float y0 = y_ptr[d];
            float y1 = y_ptr[d + 1];
            int idx0 = 0, idx1 = 0;
            for (int c = n_centroids - 2; c >= 0; --c) {
                if (y0 >= midpoints[c]) { idx0 = c + 1; break; }
            }
            for (int c = n_centroids - 2; c >= 0; --c) {
                if (y1 >= midpoints[c]) { idx1 = c + 1; break; }
            }
            idx0 = min(idx0, n_centroids - 1);
            idx1 = min(idx1, n_centroids - 1);
            slot_ptr[d / 2] = (uint8_t)((idx0 & 0xF) | ((idx1 & 0xF) << 4));
        }
    } else if constexpr (MSE_BITS == 3) {
        for (int g = 0; g < head_dim / 8; ++g) {
            int packed_24 = 0;
            for (int i = 0; i < 8; ++i) {
                int d = g * 8 + i;
                float y_val = y_ptr[d];
                int idx = 0;
                for (int c = n_centroids - 2; c >= 0; --c) {
                    if (y_val >= midpoints[c]) { idx = c + 1; break; }
                }
                idx = min(idx, n_centroids - 1);
                packed_24 |= (idx & 0x7) << (i * 3);
            }
            slot_ptr[g * 3] = (uint8_t)(packed_24 & 0xFF);
            slot_ptr[g * 3 + 1] = (uint8_t)((packed_24 >> 8) & 0xFF);
            slot_ptr[g * 3 + 2] = (uint8_t)((packed_24 >> 16) & 0xFF);
        }
    }

    store_half_bits<scalar_t>(norms[pid], slot_ptr + mse_bytes);

    float v_float[256];
    float v_min = FLT_MAX, v_max = -FLT_MAX;
    for (int d = 0; d < head_dim; ++d) {
        v_float[d] = to_float(v_ptr[d]);
        v_min = fminf(v_min, v_float[d]);
        v_max = fmaxf(v_max, v_float[d]);
    }
    float v_scale = (v_max - v_min);
    if constexpr (VQB == 4) {
        v_scale /= 15.0f;
    } else {
        v_scale /= 7.0f;
    }
    v_scale = fmaxf(v_scale, 1e-8f);

    if constexpr (VQB == 4) {
        for (int d = 0; d < head_dim; d += 2) {
            int q0 = min(max((int)roundf((v_float[d] - v_min) / v_scale), 0), 15);
            int q1 = min(max((int)roundf((v_float[d + 1] - v_min) / v_scale), 0), 15);
            slot_ptr[key_packed_size + d / 2] = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
        }
    } else if constexpr (VQB == 3) {
        for (int g = 0; g < head_dim / 8; ++g) {
            int packed_24 = 0;
            for (int i = 0; i < 8; ++i) {
                int d = g * 8 + i;
                int q = min(max((int)roundf((v_float[d] - v_min) / v_scale), 0), 7);
                packed_24 |= (q & 0x7) << (i * 3);
            }
            slot_ptr[key_packed_size + g * 3] = (uint8_t)(packed_24 & 0xFF);
            slot_ptr[key_packed_size + g * 3 + 1] = (uint8_t)((packed_24 >> 8) & 0xFF);
            slot_ptr[key_packed_size + g * 3 + 2] = (uint8_t)((packed_24 >> 16) & 0xFF);
        }
    }

    store_half_bits<scalar_t>(v_scale, slot_ptr + key_packed_size + val_data_bytes);
    store_half_bits<scalar_t>(v_min, slot_ptr + key_packed_size + val_data_bytes + 2);
}

template <int MSE_BITS, int VQB, bool NORM_CORRECTION, int BLOCK_KV, bool USE_E4B15 = false, bool USE_BF16 = false>
__global__ void __launch_bounds__(256)
tq_decode_stage1_kernel(
    const float* __restrict__ q_rot,
    const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    const float* __restrict__ centroids,
    float* __restrict__ mid_o,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    int64_t block_size, int64_t max_num_blocks_per_req,
    int num_kv_splits, float attn_scale,
    int mse_bytes, int key_packed_size,
    int val_data_bytes, int n_centroids, int slot_size_aligned) {
    const int bid = blockIdx.x;
    const int hid = blockIdx.y;
    const int sid = blockIdx.z;
    if (bid >= batch_size) return;

    const int kv_group = num_heads / num_kv_heads;
    const int kv_head = hid / kv_group;
    const int seq_len = seq_lens[bid];
    if (seq_len <= 0) return;

    const int split_len = (seq_len + num_kv_splits - 1) / num_kv_splits;
    const int split_start = split_len * sid;
    const int split_end = min(split_start + split_len, seq_len);
    if (split_start >= split_end) return;

    const float* q = q_rot + bid * num_heads * head_dim + hid * head_dim;

    extern __shared__ float smem[];
    float* acc = smem;
    float* s_scores = smem + head_dim;
    float* s_values = s_scores + BLOCK_KV;

    for (int d = 0; d < head_dim; ++d) acc[d] = 0.0f;
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;

    for (int start_n = split_start; start_n < split_end; start_n += BLOCK_KV) {
        int end_n = min(start_n + BLOCK_KV, split_end);
        int kv_count = end_n - start_n;

        for (int n = 0; n < kv_count; ++n) {
            const int pos = start_n + n;
            const int page_idx = pos / block_size;
            const int page_off = pos % block_size;
            const int32_t blk = block_tables[bid * max_num_blocks_per_req + page_idx];
            if (blk < 0) continue;
            const int64_t slot_byte = compute_slot_byte_offset(
                blk, page_off, kv_head, block_size, num_kv_heads, slot_size_aligned);
            const uint8_t* sp = kv_cache + slot_byte;

            float k_recon[256];
            if constexpr (MSE_BITS == 8) {
                for (int d = 0; d < head_dim; ++d) {
                    if constexpr (USE_E4B15) {
                        k_recon[d] = fp8_e4b15_to_float(sp[d]);
                    } else {
                        k_recon[d] = fp8_e4m3_to_float(sp[d]);
                    }
                }
            } else if constexpr (MSE_BITS == 4) {
                for (int d = 0; d < head_dim; d += 2) {
                    uint8_t packed = sp[d / 2];
                    int idx0 = packed & 0xF;
                    int idx1 = (packed >> 4) & 0xF;
                    k_recon[d] = centroids[idx0];
                    k_recon[d + 1] = centroids[idx1];
                }
            } else if constexpr (MSE_BITS == 3) {
                for (int g = 0; g < head_dim / 8; ++g) {
                    int b0 = sp[g * 3];
                    int b1 = sp[g * 3 + 1];
                    int b2 = sp[g * 3 + 2];
                    int packed = b0 | (b1 << 8) | (b2 << 16);
                    for (int i = 0; i < 8; ++i) {
                        int idx = (packed >> (i * 3)) & 0x7;
                        k_recon[g * 8 + i] = centroids[idx];
                    }
                }
            }

            if constexpr (NORM_CORRECTION) {
                float c_norm_sq = 0.0f;
                for (int d = 0; d < head_dim; ++d) c_norm_sq += k_recon[d] * k_recon[d];
                float c_inv_norm = 1.0f / sqrtf(c_norm_sq + 1e-16f);
                for (int d = 0; d < head_dim; ++d) k_recon[d] *= c_inv_norm;
            }

            float vec_norm;
            if constexpr (USE_BF16) {
                vec_norm = load_half_bits<__nv_bfloat16>(sp + mse_bytes);
            } else {
                vec_norm = load_half_bits<half>(sp + mse_bytes);
            }
            for (int d = 0; d < head_dim; ++d) k_recon[d] *= vec_norm;

            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) dot += q[d] * k_recon[d];
            s_scores[n] = dot * attn_scale;

            const uint8_t* vp = sp + key_packed_size;
            float v_scale, v_zero;
            if constexpr (USE_BF16) {
                v_scale = load_half_bits<__nv_bfloat16>(vp + val_data_bytes);
                v_zero = load_half_bits<__nv_bfloat16>(vp + val_data_bytes + 2);
            } else {
                v_scale = load_half_bits<half>(vp + val_data_bytes);
                v_zero = load_half_bits<half>(vp + val_data_bytes + 2);
            }

            if constexpr (VQB == 4) {
                for (int d = 0; d < head_dim; d += 2) {
                    uint8_t packed = vp[d / 2];
                    s_values[n * head_dim + d] = (float)(packed & 0xF) * v_scale + v_zero;
                    s_values[n * head_dim + d + 1] = (float)((packed >> 4) & 0xF) * v_scale + v_zero;
                }
            } else if constexpr (VQB == 3) {
                for (int g = 0; g < head_dim / 8; ++g) {
                    int b0 = vp[g * 3];
                    int b1 = vp[g * 3 + 1];
                    int b2 = vp[g * 3 + 2];
                    int packed = b0 | (b1 << 8) | (b2 << 16);
                    for (int i = 0; i < 8; ++i) {
                        int idx = (packed >> (i * 3)) & 0x7;
                        s_values[n * head_dim + g * 8 + i] = (float)idx * v_scale + v_zero;
                    }
                }
            }
        }

        float block_max = -FLT_MAX;
        for (int n = 0; n < kv_count; ++n) block_max = fmaxf(block_max, s_scores[n]);
        float new_max = fmaxf(m_prev, block_max);
        float re_scale = expf(m_prev - new_max);

        for (int d = 0; d < head_dim; ++d) acc[d] *= re_scale;
        l_prev *= re_scale;

        for (int n = 0; n < kv_count; ++n) {
            float p = expf(s_scores[n] - new_max);
            l_prev += p;
            for (int d = 0; d < head_dim; ++d) acc[d] += p * s_values[n * head_dim + d];
        }
        m_prev = new_max;
    }

    float* out_ptr = mid_o + (bid * num_heads + hid) * num_kv_splits * (head_dim + 1) + sid * (head_dim + 1);
    float safe_l = (l_prev > 0.0f) ? l_prev : 1.0f;
    for (int d = 0; d < head_dim; ++d) out_ptr[d] = acc[d] / safe_l;
    out_ptr[head_dim] = m_prev + logf(safe_l);
}

template <typename scalar_t>
__global__ void tq_decode_stage2_kernel(
    scalar_t* __restrict__ output,
    const float* __restrict__ mid_o,
    int batch_size, int num_heads, int head_dim,
    int num_kv_splits) {
    const int bid = blockIdx.x;
    const int hid = blockIdx.y;
    if (bid >= batch_size) return;

    const float* base = mid_o + (bid * num_heads + hid) * num_kv_splits * (head_dim + 1);

    float max_lse = -FLT_MAX;
    for (int s = 0; s < num_kv_splits; ++s) {
        max_lse = fmaxf(max_lse, base[s * (head_dim + 1) + head_dim]);
    }

    extern __shared__ float s_merge_acc[];
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        s_merge_acc[d] = 0.0f;
    }
    __syncthreads();

    float sum_exp = 0.0f;
    for (int s = 0; s < num_kv_splits; ++s) {
        float lse = base[s * (head_dim + 1) + head_dim];
        float weight = expf(lse - max_lse);
        sum_exp += weight;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            s_merge_acc[d] += weight * base[s * (head_dim + 1) + d];
        }
    }
    __syncthreads();

    float inv_sum = 1.0f / (sum_exp + 1e-8f);
    scalar_t* out = output + bid * num_heads * head_dim + hid * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out[d] = from_float<scalar_t>(s_merge_acc[d] * inv_sum);
    }
}

}

void build_hadamard_on_host(float* out, int d) {
    std::vector<float> H = {1.0f};
    // 逐次加倍：1→2→4→...→d（Sylvester 构造法）
    for (int k = 1; k < d; k *= 2) {
        // H 当前为 k×k，展开为 2k×2k
        std::vector<float> H2(4 * k * k);
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                float v = H[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(j)];
                H2[static_cast<size_t>(i) * 2 * static_cast<size_t>(k) + static_cast<size_t>(j)] = v;          // 左上
                H2[static_cast<size_t>(i) * 2 * static_cast<size_t>(k) + static_cast<size_t>(j) + static_cast<size_t>(k)] = v; // 右上
                H2[(static_cast<size_t>(i) + static_cast<size_t>(k)) * 2 * static_cast<size_t>(k) + static_cast<size_t>(j)] = v; // 左下
                H2[(static_cast<size_t>(i) + static_cast<size_t>(k)) * 2 * static_cast<size_t>(k) + static_cast<size_t>(j) + static_cast<size_t>(k)] = -v; // 右下
            }
        }
        H = std::move(H2);
    }
    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(d));
    for (int i = 0; i < d * d; ++i) out[i] = H[static_cast<size_t>(i)] * inv_sqrt_d;
}

namespace {
float gaussian_pdf(float x, float sigma2) {
    return (1.0f / sqrtf(2.0f * 3.14159265358979f * sigma2)) * expf(-x * x / (2.0f * sigma2));
}
}

void solve_lloyd_max_centroids(float* centroids_out, int d, int bits) {
    int n_levels = 1 << bits;
    float sigma2 = 1.0f / d;
    float sigma = sqrtf(sigma2);
    float lo = -3.5f * sigma, hi = 3.5f * sigma;

    std::vector<float> centroids(n_levels);
    for (int i = 0; i < n_levels; ++i)
        centroids[i] = lo + (hi - lo) * (i + 0.5f) / n_levels;

    for (int iter = 0; iter < 200; ++iter) {
        std::vector<float> boundaries(n_levels - 1);
        for (int i = 0; i < n_levels - 1; ++i)
            boundaries[i] = (centroids[i] + centroids[i + 1]) * 0.5f;

        std::vector<float> edges(n_levels + 1);
        edges[0] = lo * 3;
        for (int i = 0; i < n_levels - 1; ++i) edges[i + 1] = boundaries[i];
        edges[n_levels] = hi * 3;

        std::vector<float> new_centroids(n_levels);
        bool converged = true;
        for (int i = 0; i < n_levels; ++i) {
            int n_trap = 200;
            float a = edges[i], b = edges[i + 1];
            float step = (b - a) / n_trap;
            float num = 0.0f, den = 0.0f;
            for (int j = 0; j <= n_trap; ++j) {
                float x = a + j * step;
                float pdf_val = gaussian_pdf(x, sigma2);
                float w = (j == 0 || j == n_trap) ? 0.5f : 1.0f;
                den += w * pdf_val;
                num += w * x * pdf_val;
            }
            num *= step;
            den *= step;
            new_centroids[i] = (den > 1e-15f) ? num / den : centroids[i];
            if (fabsf(new_centroids[i] - centroids[i]) > 1e-10f) converged = false;
        }
        centroids = new_centroids;
        if (converged) break;
    }

    std::sort(centroids.begin(), centroids.end());
    for (int i = 0; i < n_levels; ++i) centroids_out[i] = centroids[i];
}

template <typename scalar_t>
void init_hadamard_fp16_impl(TQBuffers& buffers, int d2, int gpu_device) {
    std::vector<float> h_hadamard(d2);
    build_hadamard_on_host(h_hadamard.data(), buffers.head_dim);

    std::vector<uint16_t> h_hadamard_packed(d2);
    if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
        for (int i = 0; i < d2; ++i) {
            __nv_bfloat16 h = __float2bfloat16(h_hadamard[i]);
            memcpy(&h_hadamard_packed[i], &h, 2);
        }
    } else {
        for (int i = 0; i < d2; ++i) {
            half h = __float2half(h_hadamard[i]);
            memcpy(&h_hadamard_packed[i], &h, 2);
        }
    }

    CUDA_CHECK(cudaSetDevice(gpu_device));
    CUDA_CHECK(cudaMalloc(&buffers.hadamard_fp16, d2 * 2));
    CUDA_CHECK(cudaMemcpy(buffers.hadamard_fp16, h_hadamard_packed.data(), d2 * 2, cudaMemcpyHostToDevice));
}

void TQBuffers::init(int hd, int cbits, int gpu_device) {
    if (initialized) return;
    head_dim = hd;
    centroid_bits = cbits;

    int n_centroids = 1 << cbits;
    int d2 = hd * hd;

    std::vector<float> h_hadamard(d2);
    build_hadamard_on_host(h_hadamard.data(), hd);

    std::vector<float> h_centroids(n_centroids);
    solve_lloyd_max_centroids(h_centroids.data(), hd, cbits);

    std::vector<float> h_midpoints(n_centroids - 1);
    for (int i = 0; i < n_centroids - 1; ++i)
        h_midpoints[i] = (h_centroids[i] + h_centroids[i + 1]) * 0.5f;

    std::vector<uint16_t> h_hadamard_packed(d2);
    if (gpu().supports_bf16()) {
        for (int i = 0; i < d2; ++i) {
            __nv_bfloat16 bf = __float2bfloat16(h_hadamard[i]);
            memcpy(&h_hadamard_packed[i], &bf, 2);
        }
    } else {
        for (int i = 0; i < d2; ++i) {
            half bf = __float2half(h_hadamard[i]);
            memcpy(&h_hadamard_packed[i], &bf, 2);
        }
    }
    CUDA_CHECK(cudaSetDevice(gpu_device));
    CUDA_CHECK(cudaMalloc(&hadamard_fp16, d2 * 2));
    CUDA_CHECK(cudaMemcpy(hadamard_fp16, h_hadamard_packed.data(), d2 * 2, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaSetDevice(gpu_device));
    CUDA_CHECK(cudaMalloc(&hadamard, d2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(hadamard, h_hadamard.data(), d2 * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&centroids, n_centroids * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(centroids, h_centroids.data(), n_centroids * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&midpoints, (n_centroids - 1) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(midpoints, h_midpoints.data(), (n_centroids - 1) * sizeof(float), cudaMemcpyHostToDevice));

    initialized = true;
    spdlog::info("TQBuffers initialized: head_dim={}, centroid_bits={}, n_centroids={}", hd, cbits, n_centroids);
}

void TQBuffers::free_buf() {
    if (hadamard) { CUDA_CHECK(cudaFree(hadamard)); hadamard = nullptr; }
    if (hadamard_fp16) { CUDA_CHECK(cudaFree(hadamard_fp16)); hadamard_fp16 = nullptr; }
    if (centroids) { CUDA_CHECK(cudaFree(centroids)); centroids = nullptr; }
    if (midpoints) { CUDA_CHECK(cudaFree(midpoints)); midpoints = nullptr; }
    initialized = false;
}

void TQWorkspace::ensure_store(int nh, int head_dim_, int gpu_device) {
    if (nh <= store_capacity) return;
    auto t0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaSetDevice(gpu_device));
    auto t1 = std::chrono::steady_clock::now();
    size_t float_size = nh * head_dim_ * sizeof(float);
    size_t norm_size = nh * sizeof(float);
    if (store_k_float) CUDA_CHECK(cudaFree(store_k_float));
    if (store_norms) CUDA_CHECK(cudaFree(store_norms));
    if (store_x_hat) CUDA_CHECK(cudaFree(store_x_hat));
    if (store_y_rotated) CUDA_CHECK(cudaFree(store_y_rotated));
    auto t2 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMalloc(&store_k_float, float_size));
    CUDA_CHECK(cudaMalloc(&store_norms, norm_size));
    CUDA_CHECK(cudaMalloc(&store_x_hat, float_size));
    CUDA_CHECK(cudaMalloc(&store_y_rotated, float_size));
    auto t3 = std::chrono::steady_clock::now();
    store_capacity = nh;
    auto ms_setdev = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto ms_free = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    auto ms_malloc = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    spdlog::info("[TQ_ENSURE] nh={} float_sz={} setdev={}ms free={}ms malloc={}ms",
                 nh, float_size, ms_setdev, ms_free, ms_malloc);
}

void TQWorkspace::ensure_decode(int bhq, int head_dim_, int num_kv_splits, int gpu_device) {
    if (bhq <= decode_capacity) return;
    CUDA_CHECK(cudaSetDevice(gpu_device));
    size_t float_size = bhq * head_dim_ * sizeof(float);
    size_t mid_o_size = bhq * num_kv_splits * (head_dim_ + 1) * sizeof(float);
    if (decode_q_float) CUDA_CHECK(cudaFree(decode_q_float));
    if (decode_q_rotated) CUDA_CHECK(cudaFree(decode_q_rotated));
    if (decode_mid_o) CUDA_CHECK(cudaFree(decode_mid_o));
    CUDA_CHECK(cudaMalloc(&decode_q_float, float_size));
    CUDA_CHECK(cudaMalloc(&decode_q_rotated, float_size));
    CUDA_CHECK(cudaMalloc(&decode_mid_o, mid_o_size));
    decode_capacity = bhq;
}

void TQWorkspace::free_buf() {
    if (store_k_float) { CUDA_CHECK(cudaFree(store_k_float)); store_k_float = nullptr; }
    if (store_norms) { CUDA_CHECK(cudaFree(store_norms)); store_norms = nullptr; }
    if (store_x_hat) { CUDA_CHECK(cudaFree(store_x_hat)); store_x_hat = nullptr; }
    if (store_y_rotated) { CUDA_CHECK(cudaFree(store_y_rotated)); store_y_rotated = nullptr; }
    store_capacity = 0;
    if (decode_q_float) { CUDA_CHECK(cudaFree(decode_q_float)); decode_q_float = nullptr; }
    if (decode_q_rotated) { CUDA_CHECK(cudaFree(decode_q_rotated)); decode_q_rotated = nullptr; }
    if (decode_mid_o) { CUDA_CHECK(cudaFree(decode_mid_o)); decode_mid_o = nullptr; }
    decode_capacity = 0;
}

static cublasHandle_t get_tq_cublas(int device) {
    static thread_local std::unordered_map<int, cublasHandle_t> handles;
    auto it = handles.find(device);
    if (it != handles.end()) return it->second;
    cublasHandle_t h;
    CUBLAS_CHECK(cublasCreate(&h));
    handles[device] = h;
    return h;
}

#define TQ_DISPATCH_STORE_MSE(scalar_t, y_rotated, norms, value, kv_cache, slot_mapping, midpoints, num_tokens, num_kv_heads, head_dim, block_size, config, blocks, threads, stream) \
    if (config.key_quant_bits == 4 && config.value_quant_bits == 4 && config.norm_correction) { \
        tq_store_mse_kernel<scalar_t, 4, 4, true><<<blocks, threads, 0, stream>>>( \
            y_rotated, norms, value, kv_cache, slot_mapping, midpoints, \
            num_tokens, num_kv_heads, head_dim, block_size, \
            config.mse_bytes, config.key_packed_size, \
            config.val_data_bytes, config.n_centroids, config.slot_size_aligned); \
    } else if (config.key_quant_bits == 4 && config.value_quant_bits == 4) { \
        tq_store_mse_kernel<scalar_t, 4, 4, false><<<blocks, threads, 0, stream>>>( \
            y_rotated, norms, value, kv_cache, slot_mapping, midpoints, \
            num_tokens, num_kv_heads, head_dim, block_size, \
            config.mse_bytes, config.key_packed_size, \
            config.val_data_bytes, config.n_centroids, config.slot_size_aligned); \
    } else if (config.key_quant_bits == 3 && config.value_quant_bits == 4 && config.norm_correction) { \
        tq_store_mse_kernel<scalar_t, 3, 4, true><<<blocks, threads, 0, stream>>>( \
            y_rotated, norms, value, kv_cache, slot_mapping, midpoints, \
            num_tokens, num_kv_heads, head_dim, block_size, \
            config.mse_bytes, config.key_packed_size, \
            config.val_data_bytes, config.n_centroids, config.slot_size_aligned); \
    } else if (config.key_quant_bits == 3 && config.value_quant_bits == 3 && config.norm_correction) { \
        tq_store_mse_kernel<scalar_t, 3, 3, true><<<blocks, threads, 0, stream>>>( \
            y_rotated, norms, value, kv_cache, slot_mapping, midpoints, \
            num_tokens, num_kv_heads, head_dim, block_size, \
            config.mse_bytes, config.key_packed_size, \
            config.val_data_bytes, config.n_centroids, config.slot_size_aligned); \
    }

void turboquant_store(
    const void* key, const void* value,
    void* kv_cache, const int64_t* slot_mapping,
    int num_tokens, int num_kv_heads, int head_dim,
    int64_t block_size, const TQConfig& config,
    const TQBuffers& buffers, TQWorkspace& workspace,
    cudaStream_t stream) {
    if (head_dim <= 0 || head_dim > TQConfig::kMaxSupportedHeadDim) {
        throw std::runtime_error(
            "turboquant_store: unsupported head_dim=" + std::to_string(head_dim) +
            " (max " + std::to_string(TQConfig::kMaxSupportedHeadDim) + ")");
    }
    const int NH = num_tokens * num_kv_heads;
    int gpu_device = 0;
    CUDA_CHECK(cudaGetDevice(&gpu_device));

    VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "turboquant_store", [&] {
        if (config.key_fp8) {
            const int threads = 256;
            const int blocks = (NH + threads - 1) / threads;
            half_to_fp8_store_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                static_cast<const scalar_t*>(key),
                static_cast<const scalar_t*>(value),
                static_cast<uint8_t*>(kv_cache),
                slot_mapping,
                num_tokens, num_kv_heads, head_dim, block_size,
                config.key_packed_size, config.val_data_bytes,
                config.slot_size_aligned, config.fp8_e4b15);
            return;
        }

        workspace.ensure_store(NH, head_dim, gpu_device);

        float* k_float = static_cast<float*>(workspace.store_k_float);
        float* norms = static_cast<float*>(workspace.store_norms);
        float* x_hat = static_cast<float*>(workspace.store_x_hat);
        float* y_rotated = static_cast<float*>(workspace.store_y_rotated);

        int grid_sz = (NH + 255) / 256;
        half_to_float_norm_kernel<scalar_t><<<grid_sz, 256, 0, stream>>>(
            static_cast<const scalar_t*>(key), k_float, norms, NH, head_dim);

        normalize_kernel<<<grid_sz, 256, 0, stream>>>(
            k_float, norms, x_hat, NH, head_dim);

        cublasHandle_t handle = get_tq_cublas(gpu_device);
            CUBLAS_CHECK(cublasSetStream(handle, stream));

        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            head_dim, NH, head_dim,
            &alpha,
            static_cast<const float*>(buffers.hadamard), head_dim,
            x_hat, head_dim,
            &beta,
            y_rotated, head_dim);

        const int threads = 256;
        const int blocks = (NH + threads - 1) / threads;

        TQ_DISPATCH_STORE_MSE(scalar_t,
            y_rotated, norms, static_cast<const scalar_t*>(value),
            static_cast<uint8_t*>(kv_cache), slot_mapping,
            static_cast<const float*>(buffers.midpoints),
            num_tokens, num_kv_heads, head_dim, block_size,
            config, blocks, threads, stream);
    });
}

#define TQ_DISPATCH_DECODE_STAGE1(q_rot, kv_cache, block_tables, seq_lens, centroids, mid_o, B, Hq, num_kv_heads, D, block_size, max_num_blocks_per_req, num_kv_splits, scale, config, grid, block, smem_size, stream, use_bf16) \
    if (config.key_fp8 && config.value_quant_bits == 4) { \
        if (config.fp8_e4b15) { \
            if (use_bf16) { \
                tq_decode_stage1_kernel<8, 4, false, 4, true, true><<<grid, block, smem_size, stream>>>( \
                    q_rot, kv_cache, block_tables, seq_lens, centroids, mid_o, B, Hq, num_kv_heads, D, \
                    block_size, max_num_blocks_per_req, num_kv_splits, scale, \
                    config.mse_bytes, config.key_packed_size, config.val_data_bytes, config.n_centroids, config.slot_size_aligned); \
            } else { \
                tq_decode_stage1_kernel<8, 4, false, 4, true, false><<<grid, block, smem_size, stream>>>( \
                    q_rot, kv_cache, block_tables, seq_lens, centroids, mid_o, B, Hq, num_kv_heads, D, \
                    block_size, max_num_blocks_per_req, num_kv_splits, scale, \
                    config.mse_bytes, config.key_packed_size, config.val_data_bytes, config.n_centroids, config.slot_size_aligned); \
            } \
        } else { \
            if (use_bf16) { \
                tq_decode_stage1_kernel<8, 4, false, 4, false, true><<<grid, block, smem_size, stream>>>( \
                    q_rot, kv_cache, block_tables, seq_lens, centroids, mid_o, B, Hq, num_kv_heads, D, \
                    block_size, max_num_blocks_per_req, num_kv_splits, scale, \
                    config.mse_bytes, config.key_packed_size, config.val_data_bytes, config.n_centroids, config.slot_size_aligned); \
            } else { \
                tq_decode_stage1_kernel<8, 4, false, 4, false, false><<<grid, block, smem_size, stream>>>( \
                    q_rot, kv_cache, block_tables, seq_lens, centroids, mid_o, B, Hq, num_kv_heads, D, \
                    block_size, max_num_blocks_per_req, num_kv_splits, scale, \
                    config.mse_bytes, config.key_packed_size, config.val_data_bytes, config.n_centroids, config.slot_size_aligned); \
            } \
        } \
    } else if (config.key_quant_bits == 4 && config.value_quant_bits == 4 && config.norm_correction) { \
        if (use_bf16) { \
            tq_decode_stage1_kernel<4, 4, true, 4, false, true><<<grid, block, smem_size, stream>>>( \
                q_rot, kv_cache, block_tables, seq_lens, centroids, mid_o, B, Hq, num_kv_heads, D, \
                block_size, max_num_blocks_per_req, num_kv_splits, scale, \
                config.mse_bytes, config.key_packed_size, config.val_data_bytes, config.n_centroids, config.slot_size_aligned); \
        } else { \
            tq_decode_stage1_kernel<4, 4, true, 4, false, false><<<grid, block, smem_size, stream>>>( \
                q_rot, kv_cache, block_tables, seq_lens, centroids, mid_o, B, Hq, num_kv_heads, D, \
                block_size, max_num_blocks_per_req, num_kv_splits, scale, \
                config.mse_bytes, config.key_packed_size, config.val_data_bytes, config.n_centroids, config.slot_size_aligned); \
        } \
    } else if (config.key_quant_bits == 4 && config.value_quant_bits == 4) { \
        if (use_bf16) { \
            tq_decode_stage1_kernel<4, 4, false, 4, false, true><<<grid, block, smem_size, stream>>>( \
                q_rot, kv_cache, block_tables, seq_lens, centroids, mid_o, B, Hq, num_kv_heads, D, \
                block_size, max_num_blocks_per_req, num_kv_splits, scale, \
                config.mse_bytes, config.key_packed_size, config.val_data_bytes, config.n_centroids, config.slot_size_aligned); \
        } else { \
            tq_decode_stage1_kernel<4, 4, false, 4, false, false><<<grid, block, smem_size, stream>>>( \
                q_rot, kv_cache, block_tables, seq_lens, centroids, mid_o, B, Hq, num_kv_heads, D, \
                block_size, max_num_blocks_per_req, num_kv_splits, scale, \
                config.mse_bytes, config.key_packed_size, config.val_data_bytes, config.n_centroids, config.slot_size_aligned); \
        } \
    } else if (config.key_quant_bits == 3 && config.value_quant_bits == 4 && config.norm_correction) { \
        if (use_bf16) { \
            tq_decode_stage1_kernel<3, 4, true, 4, false, true><<<grid, block, smem_size, stream>>>( \
                q_rot, kv_cache, block_tables, seq_lens, centroids, mid_o, B, Hq, num_kv_heads, D, \
                block_size, max_num_blocks_per_req, num_kv_splits, scale, \
                config.mse_bytes, config.key_packed_size, config.val_data_bytes, config.n_centroids, config.slot_size_aligned); \
        } else { \
            tq_decode_stage1_kernel<3, 4, true, 4, false, false><<<grid, block, smem_size, stream>>>( \
                q_rot, kv_cache, block_tables, seq_lens, centroids, mid_o, B, Hq, num_kv_heads, D, \
                block_size, max_num_blocks_per_req, num_kv_splits, scale, \
                config.mse_bytes, config.key_packed_size, config.val_data_bytes, config.n_centroids, config.slot_size_aligned); \
        } \
    } else if (config.key_quant_bits == 3 && config.value_quant_bits == 3 && config.norm_correction) { \
        if (use_bf16) { \
            tq_decode_stage1_kernel<3, 3, true, 4, false, true><<<grid, block, smem_size, stream>>>( \
                q_rot, kv_cache, block_tables, seq_lens, centroids, mid_o, B, Hq, num_kv_heads, D, \
                block_size, max_num_blocks_per_req, num_kv_splits, scale, \
                config.mse_bytes, config.key_packed_size, config.val_data_bytes, config.n_centroids, config.slot_size_aligned); \
        } else { \
            tq_decode_stage1_kernel<3, 3, true, 4, false, false><<<grid, block, smem_size, stream>>>( \
                q_rot, kv_cache, block_tables, seq_lens, centroids, mid_o, B, Hq, num_kv_heads, D, \
                block_size, max_num_blocks_per_req, num_kv_splits, scale, \
                config.mse_bytes, config.key_packed_size, config.val_data_bytes, config.n_centroids, config.slot_size_aligned); \
        } \
    }

void turboquant_decode_attention(
    void* output, const void* query, const void* kv_cache,
    const int32_t* block_tables, const int32_t* seq_lens,
    int batch_size, int num_heads, int num_kv_heads, int head_dim,
    float scale, int64_t block_size, int64_t max_num_blocks_per_req,
    const TQConfig& config, const TQBuffers& buffers,
    TQWorkspace& workspace, int max_num_kv_splits,
    cudaStream_t stream) {
    if (head_dim <= 0 || head_dim > TQConfig::kMaxSupportedHeadDim) {
        throw std::runtime_error(
            "turboquant_decode_attention: unsupported head_dim=" + std::to_string(head_dim) +
            " (max " + std::to_string(TQConfig::kMaxSupportedHeadDim) + ")");
    }
    const int num_kv_splits = max_num_kv_splits;
    const int B = batch_size;
    const int Hq = num_heads;
    const int D = head_dim;
    int gpu_device = 0;
    CUDA_CHECK(cudaGetDevice(&gpu_device));

    const int BHQ = B * Hq;
    workspace.ensure_decode(BHQ, D, num_kv_splits, gpu_device);

    float* q_float = static_cast<float*>(workspace.decode_q_float);
    float* q_rotated = static_cast<float*>(workspace.decode_q_rotated);
    float* mid_o = static_cast<float*>(workspace.decode_mid_o);

    VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "tq_decode", [&] {
        int grid_sz = (BHQ * D + 255) / 256;
        half_to_float_kernel<scalar_t><<<grid_sz, 256, 0, stream>>>(
            static_cast<const scalar_t*>(query), q_float, BHQ * D);

        if (!config.key_fp8) {
            cublasHandle_t handle = get_tq_cublas(gpu_device);
        CUBLAS_CHECK(cublasSetStream(handle, stream));
            const float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                D, BHQ, D,
                &alpha,
                static_cast<const float*>(buffers.hadamard), D,
                q_float, D,
                &beta,
                q_rotated, D);
        } else {
            CUDA_CHECK(cudaMemcpyAsync(q_rotated, q_float, BHQ * D * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }

        dim3 grid(B, Hq, num_kv_splits);
        dim3 block(D > 128 ? 256 : 128);
        size_t smem_size = D * sizeof(float) + 4 * sizeof(float) + 4 * D * sizeof(float);

        TQ_DISPATCH_DECODE_STAGE1(
            q_rotated, static_cast<const uint8_t*>(kv_cache),
            block_tables, seq_lens,
            static_cast<const float*>(buffers.centroids),
            mid_o, B, Hq, num_kv_heads, D,
            block_size, max_num_blocks_per_req,
            num_kv_splits, scale, config,
            grid, block, smem_size, stream, gpu().supports_bf16());

        dim3 grid2(B, Hq);
        dim3 block2(min(D, 1024));
        const size_t stage2_smem = static_cast<size_t>(D) * sizeof(float);
        tq_decode_stage2_kernel<scalar_t><<<grid2, block2, stage2_smem, stream>>>(
            static_cast<scalar_t*>(output), mid_o,
            B, Hq, D, num_kv_splits);
    });
}

}
