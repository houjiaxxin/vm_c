// Fused QK Norm + RoPE + KV Cache Store kernel
// 参考 llama.cpp 的 ggml_cuda_op_rope_fused 实现
// 将 Q_norm + K_norm + RoPE + store_kv 融合为1个kernel launch
//
// 原始流程（6个kernel launch）：
//   memset(Q_residual) → fused_add_rms_norm(Q) → memset(K_residual) → fused_add_rms_norm(K) → RoPE → store_kv
//
// 融合后（1个kernel launch）：
//   fused_qk_norm_rope_store(Q, K, V, ...)
//
// 性能提升：减少5次kernel launch + 2次memset + 4次global memory round-trip

#include "vm_c/cuda/kernels_fused_attn.h"
#include "vm_c/cuda/vm_c_dispatch.h"
#include "vm_c/cuda/gpu_arch.hpp"
#include "vm_c/cuda/kernels_cache.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>

namespace vm_c {

namespace {

// ── Fused QK Norm + RoPE + KV Cache Store kernel ──
//
// 每个 block 处理一个 token 的所有 head。
// 线程布局：blockDim.x = head_dim，gridDim = (num_tokens,)
//
// 步骤：
//   1. 加载 Q[k] 并做 RMSNorm（如果 q_norm_weight != nullptr）
//   2. 加载 K[k] 并做 RMSNorm（如果 k_norm_weight != nullptr）
//   3. 对 Q 和 K 应用 RoPE
//   4. 将 K 和 V 写入 KV cache（如果 key_cache != nullptr）

template <typename scalar_t, bool HAS_Q_NORM, bool HAS_K_NORM, bool DO_STORE_KV>
__global__ void fused_qk_norm_rope_store_kernel(
    // Q buffer: [num_tokens, num_heads, head_dim]
    scalar_t* __restrict__ q_buf,
    // K buffer: [num_tokens, num_kv_heads, head_dim]
    scalar_t* __restrict__ k_buf,
    // V buffer: [num_tokens, num_kv_heads, head_dim] (only used if DO_STORE_KV)
    const scalar_t* __restrict__ v_buf,
    // Q norm weight: [num_heads * head_dim] or nullptr
    const scalar_t* __restrict__ q_norm_weight,
    // K norm weight: [num_kv_heads * head_dim] or nullptr
    const scalar_t* __restrict__ k_norm_weight,
    // RoPE cos/sin cache: [max_seq_len, rot_dim] (rot_dim = head_dim)
    const scalar_t* __restrict__ cos_sin_cache,
    // Position IDs: [num_tokens]
    const int64_t* __restrict__ position_ids,
    // KV cache (only used if DO_STORE_KV)
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    // Dimensions
    int num_heads, int num_kv_heads, int head_dim,
    float rms_norm_eps, bool is_neox,
    int64_t block_size, int64_t cache_stride) {

    const int token_idx = blockIdx.x;
    if (token_idx >= gridDim.x) return;

    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

    const int64_t pos = position_ids[token_idx];
    const int64_t q_token_offset = static_cast<int64_t>(token_idx) * num_heads * head_dim;
    const int64_t k_token_offset = static_cast<int64_t>(token_idx) * num_kv_heads * head_dim;

    // ── Step 1: Q RMSNorm (in-place) ──
    if constexpr (HAS_Q_NORM) {
        // 对每个 head 独立做 RMSNorm
        for (int h = 0; h < num_heads; ++h) {
            const int64_t q_head_offset = q_token_offset + h * head_dim;

            // 计算 Q 的平方和
            float q_var = 0.0f;
            for (int d = tid; d < head_dim; d += bdim) {
                float val = static_cast<float>(q_buf[q_head_offset + d]);
                q_var += val * val;
            }

            // Warp reduce
            for (int mask = bdim / 2; mask > 0; mask >>= 1) {
                q_var += __shfl_xor_sync(0xffffffff, q_var, mask);
            }

            float q_inv_rms = 1.0f / sqrtf(q_var / static_cast<float>(head_dim) + rms_norm_eps);

            // 应用 RMSNorm: q[i] = q[i] * inv_rms * weight[i]
            for (int d = tid; d < head_dim; d += bdim) {
                float val = static_cast<float>(q_buf[q_head_offset + d]);
                float w = static_cast<float>(q_norm_weight[h * head_dim + d]);
                q_buf[q_head_offset + d] = static_cast<scalar_t>(val * q_inv_rms * w);
            }
        }
    }

    // ── Step 2: K RMSNorm (in-place) ──
    if constexpr (HAS_K_NORM) {
        for (int h = 0; h < num_kv_heads; ++h) {
            const int64_t k_head_offset = k_token_offset + h * head_dim;

            float k_var = 0.0f;
            for (int d = tid; d < head_dim; d += bdim) {
                float val = static_cast<float>(k_buf[k_head_offset + d]);
                k_var += val * val;
            }

            for (int mask = bdim / 2; mask > 0; mask >>= 1) {
                k_var += __shfl_xor_sync(0xffffffff, k_var, mask);
            }

            float k_inv_rms = 1.0f / sqrtf(k_var / static_cast<float>(head_dim) + rms_norm_eps);

            for (int d = tid; d < head_dim; d += bdim) {
                float val = static_cast<float>(k_buf[k_head_offset + d]);
                float w = static_cast<float>(k_norm_weight[h * head_dim + d]);
                k_buf[k_head_offset + d] = static_cast<scalar_t>(val * k_inv_rms * w);
            }
        }
    }

    // ── Step 3: RoPE (in-place on Q and K) ──
    {
        const int rot_dim = head_dim; // Qwen3.5 uses full rotary dim
        const int embed_dim = rot_dim / 2;
        const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;
        const scalar_t* cos_ptr = cache_ptr;
        const scalar_t* sin_ptr = cache_ptr + embed_dim;

        // Apply RoPE to Q
        for (int h = 0; h < num_heads; ++h) {
            const int64_t q_head_offset = q_token_offset + h * head_dim;
            for (int d = tid; d < embed_dim; d += bdim) {
                float q_x = static_cast<float>(q_buf[q_head_offset + d]);
                float q_y = static_cast<float>(q_buf[q_head_offset + d + embed_dim]);
                float cos_f = static_cast<float>(cos_ptr[d]);
                float sin_f = static_cast<float>(sin_ptr[d]);

                if (is_neox) {
                    // Neox style: [x0, x1, ..., x_{d/2-1}, y0, y1, ..., y_{d/2-1}]
                    q_buf[q_head_offset + d] = static_cast<scalar_t>(q_x * cos_f - q_y * sin_f);
                    q_buf[q_head_offset + d + embed_dim] = static_cast<scalar_t>(q_y * cos_f + q_x * sin_f);
                } else {
                    // GPT-J style: [x0, y0, x1, y1, ..., x_{d/2-1}, y_{d/2-1}]
                    q_buf[q_head_offset + 2 * d] = static_cast<scalar_t>(q_x * cos_f - q_y * sin_f);
                    q_buf[q_head_offset + 2 * d + 1] = static_cast<scalar_t>(q_y * cos_f + q_x * sin_f);
                }
            }
        }

        // Apply RoPE to K
        for (int h = 0; h < num_kv_heads; ++h) {
            const int64_t k_head_offset = k_token_offset + h * head_dim;
            for (int d = tid; d < embed_dim; d += bdim) {
                float k_x = static_cast<float>(k_buf[k_head_offset + d]);
                float k_y = static_cast<float>(k_buf[k_head_offset + d + embed_dim]);
                float cos_f = static_cast<float>(cos_ptr[d]);
                float sin_f = static_cast<float>(sin_ptr[d]);

                if (is_neox) {
                    k_buf[k_head_offset + d] = static_cast<scalar_t>(k_x * cos_f - k_y * sin_f);
                    k_buf[k_head_offset + d + embed_dim] = static_cast<scalar_t>(k_y * cos_f + k_x * sin_f);
                } else {
                    k_buf[k_head_offset + 2 * d] = static_cast<scalar_t>(k_x * cos_f - k_y * sin_f);
                    k_buf[k_head_offset + 2 * d + 1] = static_cast<scalar_t>(k_y * cos_f + k_x * sin_f);
                }
            }
        }
    }

    // ── Step 4: Store K and V into KV cache ──
    if constexpr (DO_STORE_KV) {
        if (key_cache != nullptr && slot_mapping != nullptr) {
            const int64_t slot = slot_mapping[token_idx];
            if (slot >= 0) {
                // Store K: [num_kv_heads, head_dim] → cache[slot * cache_stride + h * head_dim * block_size + d]
                for (int h = 0; h < num_kv_heads; ++h) {
                    const int64_t k_head_offset = k_token_offset + h * head_dim;
                    const int64_t physical_block = slot / block_size;
                    const int64_t block_offset = slot % block_size;
                    const int64_t cache_offset = physical_block * cache_stride +
                                                 h * head_dim * block_size +
                                                 block_offset * head_dim;
                    for (int d = tid; d < head_dim; d += bdim) {
                        key_cache[cache_offset + d] = k_buf[k_head_offset + d];
                    }
                }

                // Store V: [num_kv_heads, head_dim] → cache[slot * cache_stride + h * head_dim * block_size + d]
                const int64_t v_token_offset = static_cast<int64_t>(token_idx) * num_kv_heads * head_dim;
                for (int h = 0; h < num_kv_heads; ++h) {
                    const int64_t v_head_offset = v_token_offset + h * head_dim;
                    const int64_t physical_block = slot / block_size;
                    const int64_t block_offset = slot % block_size;
                    const int64_t cache_offset = physical_block * cache_stride +
                                                 h * head_dim * block_size +
                                                 block_offset * head_dim;
                    for (int d = tid; d < head_dim; d += bdim) {
                        value_cache[cache_offset + d] = v_buf[v_head_offset + d];
                    }
                }
            }
        }
    }
}

} // anonymous namespace

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
    cudaStream_t stream) {

    int threads = (head_dim + 31) / 32 * 32; // 向上对齐到32的倍数
    if (threads < 64) threads = 64;
    if (threads > 512) threads = 512;

    bool has_q_norm = (q_norm_weight != nullptr);
    bool has_k_norm = (k_norm_weight != nullptr);
    bool do_store_kv = (key_cache != nullptr && slot_mapping != nullptr);

    // 根据模板参数选择kernel变体，避免运行时分支
    #define LAUNCH_FUSED_KERNEL(HAS_Q, HAS_K, DO_STORE) \
        VMC_DISPATCH_HALF_TYPES(gpu().compute_dtype(), "fused_qk_norm_rope_store", \
            fused_qk_norm_rope_store_kernel<scalar_t, HAS_Q, HAS_K, DO_STORE> \
                <<<num_tokens, threads, 0, stream>>>( \
                    static_cast<scalar_t*>(q_buf), \
                    static_cast<scalar_t*>(k_buf), \
                    static_cast<const scalar_t*>(v_buf), \
                    static_cast<const scalar_t*>(q_norm_weight), \
                    static_cast<const scalar_t*>(k_norm_weight), \
                    static_cast<const scalar_t*>(cos_sin_cache), \
                    position_ids, \
                    static_cast<scalar_t*>(key_cache), \
                    static_cast<scalar_t*>(value_cache), \
                    slot_mapping, \
                    num_heads, num_kv_heads, head_dim, \
                    rms_norm_eps, is_neox, \
                    block_size, cache_stride); \
        )

    // 8种组合的模板实例化
    if (has_q_norm && has_k_norm && do_store_kv) {
        LAUNCH_FUSED_KERNEL(true, true, true);
    } else if (has_q_norm && has_k_norm && !do_store_kv) {
        LAUNCH_FUSED_KERNEL(true, true, false);
    } else if (has_q_norm && !has_k_norm && do_store_kv) {
        LAUNCH_FUSED_KERNEL(true, false, true);
    } else if (has_q_norm && !has_k_norm && !do_store_kv) {
        LAUNCH_FUSED_KERNEL(true, false, false);
    } else if (!has_q_norm && has_k_norm && do_store_kv) {
        LAUNCH_FUSED_KERNEL(false, true, true);
    } else if (!has_q_norm && has_k_norm && !do_store_kv) {
        LAUNCH_FUSED_KERNEL(false, true, false);
    } else if (!has_q_norm && !has_k_norm && do_store_kv) {
        LAUNCH_FUSED_KERNEL(false, false, true);
    } else {
        LAUNCH_FUSED_KERNEL(false, false, false);
    }

    #undef LAUNCH_FUSED_KERNEL

    CHECK_KERNEL_LAUNCH();
}

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
    cudaStream_t stream) {

    // Prefill 路径：不做 KV cache store，只做 QK Norm + RoPE
    fused_qk_norm_rope_store(
        q_buf, k_buf, nullptr,
        q_norm_weight, k_norm_weight,
        cos_sin_cache, position_ids,
        nullptr, nullptr, nullptr,
        num_tokens, num_heads, num_kv_heads, head_dim,
        rms_norm_eps, is_neox,
        0, 0, "auto", stream);
}

} // namespace vm_c