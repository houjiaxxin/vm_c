#include "vm_c/cuda/kernels_attention.h"
#include "vm_c/cuda/vm_c_dispatch.h"
#include "vm_c/cuda/gpu_arch.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace vm_c {

namespace {

// block level reduction sum (warp shuffle + shared memory)
__device__ float block_reduce_sum(float val, float* shared_buf) {
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    if (lane == 0) shared_buf[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (lane < num_warps) ? shared_buf[lane] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1)
            val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    __syncthreads();
    return val;
}

// ── paged_attention_v1_kernel ──
template <typename scalar_t>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key_cache,
    const scalar_t* __restrict__ value_cache,
    const int num_heads, const int num_kv_heads, const int head_dim,
    const float scale,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    const int64_t block_size, const int64_t max_num_blocks_per_req,
    const int batch_size,
    const bool hnd_layout) {
  const int seq_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  if (seq_idx >= batch_size) return;

  const int seq_len = seq_lens[seq_idx];
  if (seq_len == 0) return;

  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;

  const scalar_t* q = query + seq_idx * num_heads * head_dim + head_idx * head_dim;
  scalar_t* o = out + seq_idx * num_heads * head_dim + head_idx * head_dim;

  float max_val = -FLT_MAX;
  float sum_val = 0.0f;

  __shared__ float s_acc[256];

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    s_acc[d] = 0.0f;
  }
  __syncthreads();

  int total_blocks = (seq_len + block_size - 1) / block_size;
  for (int block = 0; block < total_blocks; ++block) {
    const int32_t block_id = block_tables[seq_idx * max_num_blocks_per_req + block];
    if (block_id < 0) continue;
    const scalar_t* k_block = key_cache + block_id * block_size * num_kv_heads * head_dim;
    const scalar_t* v_block = value_cache + block_id * block_size * num_kv_heads * head_dim;

    int block_start = block * block_size;
    int block_end = min(block_start + (int)block_size, seq_len);

    for (int token = block_start; token < block_end; ++token) {
      float dot = 0.0f;
      for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float k_val;
        if (hnd_layout) {
          k_val = static_cast<float>(k_block[kv_head_idx * block_size * head_dim + token * head_dim + d]);
        } else {
          k_val = static_cast<float>(k_block[token * num_kv_heads * head_dim + kv_head_idx * head_dim + d]);
        }
        dot += static_cast<float>(q[d]) * k_val;
      }
      float score = dot * scale;
      float new_max = fmaxf(max_val, score);
      float exp_diff = expf(max_val - new_max);
      float exp_score = expf(score - new_max);
      sum_val = sum_val * exp_diff + exp_score;
      max_val = new_max;
      for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float v_val;
        if (hnd_layout) {
          v_val = static_cast<float>(v_block[kv_head_idx * block_size * head_dim + token * head_dim + d]);
        } else {
          v_val = static_cast<float>(v_block[token * num_kv_heads * head_dim + kv_head_idx * head_dim + d]);
        }
        s_acc[d] = s_acc[d] * exp_diff + exp_score * v_val;
      }
    }
  }

  if (sum_val > 0.0f) {
    float inv_sum = 1.0f / sum_val;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      o[d] = static_cast<scalar_t>(s_acc[d] * inv_sum);
    }
  }
}

// ── v1 launch helper ──
template <typename scalar_t>
void launch_paged_attn_v1(
    void* out, const void* query, const void* key_cache, const void* value_cache,
    const vm_c::AttentionParams& params,
    const int32_t* block_tables, const int32_t* seq_lens,
    dim3 grid, dim3 block, cudaStream_t stream, bool hnd) {
  paged_attention_v1_kernel<scalar_t>
      <<<grid, block, 0, stream>>>(
          static_cast<scalar_t*>(out),
          static_cast<const scalar_t*>(query),
          static_cast<const scalar_t*>(key_cache),
          static_cast<const scalar_t*>(value_cache),
          params.num_heads, params.num_kv_heads, params.head_dim,
          params.scale, block_tables, seq_lens,
          params.block_size, params.max_num_blocks_per_req, params.batch_size,
          hnd);
}

// ── paged_attention_v2_kernel: two-pass with exp_sums ──
template <typename scalar_t>
__global__ void paged_attention_v2_kernel(
    scalar_t* __restrict__ out,
    float* __restrict__ exp_sums,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key_cache,
    const scalar_t* __restrict__ value_cache,
    const int num_heads, const int num_kv_heads, const int head_dim,
    const float scale,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    const int64_t block_size, const int64_t max_num_blocks_per_req,
    const int batch_size,
    const bool hnd_layout) {
  const int seq_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  if (seq_idx >= batch_size) return;

  const int seq_len = seq_lens[seq_idx];
  if (seq_len == 0) return;

  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;

  const scalar_t* q = query + seq_idx * num_heads * head_dim + head_idx * head_dim;
  scalar_t* o = out + seq_idx * num_heads * head_dim + head_idx * head_dim;

  float max_val = -FLT_MAX;
  float sum_val = 0.0f;

  __shared__ float s_acc[256];
  __shared__ float s_block_sum;
  __shared__ float s_block_max;

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    s_acc[d] = 0.0f;
  }
  __syncthreads();

  int total_blocks = (seq_len + block_size - 1) / block_size;
  for (int block = 0; block < total_blocks; ++block) {
    const int32_t block_id = block_tables[seq_idx * max_num_blocks_per_req + block];
    if (block_id < 0) continue;
    const scalar_t* k_block = key_cache + block_id * block_size * num_kv_heads * head_dim;
    const scalar_t* v_block = value_cache + block_id * block_size * num_kv_heads * head_dim;

    int block_start = block * block_size;
    int block_end = min(block_start + (int)block_size, seq_len);

    float block_max = -FLT_MAX;
    float block_sum = 0.0f;

    for (int token = block_start; token < block_end; ++token) {
      float dot = 0.0f;
      for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float k_val;
        if (hnd_layout) {
          k_val = static_cast<float>(k_block[kv_head_idx * block_size * head_dim + token * head_dim + d]);
        } else {
          k_val = static_cast<float>(k_block[token * num_kv_heads * head_dim + kv_head_idx * head_dim + d]);
        }
        dot += static_cast<float>(q[d]) * k_val;
      }
      float score = dot * scale;
      float new_max = fmaxf(block_max, score);
      float exp_diff = expf(block_max - new_max);
      float exp_score = expf(score - new_max);
      block_sum = block_sum * exp_diff + exp_score;
      block_max = new_max;

      for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float v_val;
        if (hnd_layout) {
          v_val = static_cast<float>(v_block[kv_head_idx * block_size * head_dim + token * head_dim + d]);
        } else {
          v_val = static_cast<float>(v_block[token * num_kv_heads * head_dim + kv_head_idx * head_dim + d]);
        }
        s_acc[d] = s_acc[d] * exp_diff + exp_score * v_val;
      }
    }

    if (threadIdx.x == 0) {
      float new_max = fmaxf(max_val, block_max);
      float exp_diff = expf(max_val - new_max);
      float exp_score = expf(block_max - new_max);
      sum_val = sum_val * exp_diff + exp_score * block_sum;
      max_val = new_max;
      s_block_sum = sum_val;
      s_block_max = max_val;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      sum_val = s_block_sum;
      max_val = s_block_max;
    }
  }

  if (threadIdx.x == 0) {
    exp_sums[seq_idx * num_heads + head_idx] = sum_val;
    if (sum_val > 0.0f) {
      float inv_sum = 1.0f / sum_val;
      for (int d = 0; d < head_dim; ++d) {
        o[d] = static_cast<scalar_t>(s_acc[d] * inv_sum);
      }
    }
  }
}

// ── v2 launch helper ──
template <typename scalar_t>
void launch_paged_attn_v2(
    void* out, void* exp_sums,
    const void* query, const void* key_cache, const void* value_cache,
    const vm_c::AttentionParams& params,
    const int32_t* block_tables, const int32_t* seq_lens,
    dim3 grid, dim3 block, cudaStream_t stream, bool hnd) {
  paged_attention_v2_kernel<scalar_t>
      <<<grid, block, 0, stream>>>(
          static_cast<scalar_t*>(out),
          static_cast<float*>(exp_sums),
          static_cast<const scalar_t*>(query),
          static_cast<const scalar_t*>(key_cache),
          static_cast<const scalar_t*>(value_cache),
          params.num_heads, params.num_kv_heads, params.head_dim,
          params.scale, block_tables, seq_lens,
          params.block_size, params.max_num_blocks_per_req, params.batch_size,
          hnd);
}

// ── flash_attention_prefill_kernel (old prefill, kept for launch_flash_prefill) ──
template <typename scalar_t, int HEAD_DIM, int Br, bool HND_LAYOUT = false>
__global__ void flash_attention_prefill_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key_cache,
    const scalar_t* __restrict__ value_cache,
    const int num_heads, const int num_kv_heads, const int head_dim,
    const float scale,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ token_to_seq,
    const int32_t* __restrict__ token_positions,
    const int64_t block_size, const int64_t max_num_blocks_per_req,
    const int num_reqs, const int num_tokens) {

  const int token_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;

  const int seq_idx = token_to_seq[token_idx];
  const int position = token_positions[token_idx];
  if (position < 0) return;

  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;

  const scalar_t* q_ptr = query + token_idx * num_heads * head_dim + head_idx * head_dim;
  scalar_t* o_ptr = out + token_idx * num_heads * head_dim + head_idx * head_dim;

  __shared__ float s_q[HEAD_DIM];
  __shared__ float s_acc[HEAD_DIM];
  __shared__ float K_tile[Br][HEAD_DIM];
  __shared__ float V_tile[Br][HEAD_DIM];
  __shared__ float scores[Br];

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    s_q[d] = static_cast<float>(q_ptr[d]);
    s_acc[d] = 0.0f;
  }
  __syncthreads();

  float max_val = -FLT_MAX;
  float sum_val = 0.0f;

  const int total_kv = position + 1;
  for (int tile_start = 0; tile_start < total_kv; tile_start += Br) {
    int tile_size = min(Br, total_kv - tile_start);

    for (int i = threadIdx.x; i < tile_size * head_dim; i += blockDim.x) {
      int kv_idx = tile_start + i / head_dim;
      int d = i % head_dim;
      int block_id_idx = kv_idx / static_cast<int>(block_size);
      int block_offset = kv_idx % static_cast<int>(block_size);
      int32_t phys_block = block_tables[seq_idx * max_num_blocks_per_req + block_id_idx];
      if (phys_block < 0) { K_tile[i / head_dim][d] = 0.f; V_tile[i / head_dim][d] = 0.f; continue; }

      const scalar_t* k_ptr = key_cache + phys_block * block_size * num_kv_heads * head_dim;
      const scalar_t* v_ptr = value_cache + phys_block * block_size * num_kv_heads * head_dim;

      if constexpr (HND_LAYOUT) {
        K_tile[i / head_dim][d] = static_cast<float>(k_ptr[kv_head_idx * block_size * head_dim + block_offset * head_dim + d]);
        V_tile[i / head_dim][d] = static_cast<float>(v_ptr[kv_head_idx * block_size * head_dim + block_offset * head_dim + d]);
      } else {
        K_tile[i / head_dim][d] = static_cast<float>(k_ptr[block_offset * num_kv_heads * head_dim + kv_head_idx * head_dim + d]);
        V_tile[i / head_dim][d] = static_cast<float>(v_ptr[block_offset * num_kv_heads * head_dim + kv_head_idx * head_dim + d]);
      }
    }
    __syncthreads();

    for (int j = threadIdx.x; j < tile_size; j += blockDim.x) {
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        dot += s_q[d] * K_tile[j][d];
      }
      scores[j] = dot * scale;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      for (int j = 0; j < tile_size; ++j) {
        float score = scores[j];
        float new_max = fmaxf(max_val, score);
        float exp_diff = expf(max_val - new_max);
        float exp_score = expf(score - new_max);

        for (int d = 0; d < head_dim; ++d) {
          s_acc[d] = s_acc[d] * exp_diff + exp_score * V_tile[j][d];
        }
        sum_val = sum_val * exp_diff + exp_score;
        max_val = new_max;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && sum_val > 0.0f) {
    float inv_sum = 1.0f / sum_val;
    for (int d = 0; d < head_dim; ++d) {
      o_ptr[d] = static_cast<scalar_t>(s_acc[d] * inv_sum);
    }
  }
}

template <typename scalar_t>
void launch_flash_prefill(
    scalar_t* out, const scalar_t* query, const scalar_t* key_cache, const scalar_t* value_cache,
    int num_tokens, int num_heads, int num_kv_heads, int head_dim,
    float scale, int64_t block_size, int64_t max_num_blocks_per_req,
    const int32_t* block_tables, const int32_t* seq_lens,
    const int32_t* token_to_seq, const int32_t* token_positions,
    int num_reqs, cudaStream_t stream, bool hnd_layout = false) {
  dim3 grid(num_heads, num_tokens);
  dim3 block(256);

  #define LAUNCH_OLD_PREFILL(HD, BR) \
    if (hnd_layout) { \
      flash_attention_prefill_kernel<scalar_t, HD, BR, true> \
          <<<grid, block, 0, stream>>>( \
              out, query, key_cache, value_cache, \
              num_heads, num_kv_heads, head_dim, scale, \
              block_tables, seq_lens, token_to_seq, token_positions, \
              block_size, max_num_blocks_per_req, num_reqs, num_tokens); \
    } else { \
      flash_attention_prefill_kernel<scalar_t, HD, BR, false> \
          <<<grid, block, 0, stream>>>( \
              out, query, key_cache, value_cache, \
              num_heads, num_kv_heads, head_dim, scale, \
              block_tables, seq_lens, token_to_seq, token_positions, \
              block_size, max_num_blocks_per_req, num_reqs, num_tokens); \
    }

  if (head_dim <= 64) {
    LAUNCH_OLD_PREFILL(64, 64)
  } else if (head_dim <= 128) {
    LAUNCH_OLD_PREFILL(128, 32)
  } else {
    LAUNCH_OLD_PREFILL(256, 16)
  }

  #undef LAUNCH_OLD_PREFILL
}

} // anonymous namespace

// ── Public API ──

void paged_attention_prefill(
    void* out, const void* query, const void* key_cache, const void* value_cache,
    int num_tokens, int num_heads, int num_kv_heads, int head_dim,
    float scale, int64_t block_size, int64_t max_num_blocks_per_req,
    const int32_t* block_tables, const int32_t* seq_lens,
    const int32_t* token_to_seq, const int32_t* token_positions,
    int num_reqs, const std::string& kv_cache_dtype,
    ScalarType dtype, cudaStream_t stream, const std::string& layout) {
  const bool hnd = (layout == "hnd");
  VMC_DISPATCH_FLOATING_TYPES(dtype, "paged_attention_prefill",
    launch_flash_prefill<scalar_t>(
        static_cast<scalar_t*>(out),
        static_cast<const scalar_t*>(query),
        static_cast<const scalar_t*>(key_cache),
        static_cast<const scalar_t*>(value_cache),
        num_tokens, num_heads, num_kv_heads, head_dim, scale,
        block_size, max_num_blocks_per_req,
        block_tables, seq_lens, token_to_seq, token_positions,
        num_reqs, stream, hnd);
  );
}

void paged_attention_v1(
    void* out, const void* query, const void* key_cache, const void* value_cache,
    const vm_c::AttentionParams& params,
    const int32_t* block_tables, const int32_t* seq_lens,
    const float* alibi_slopes, const std::string& kv_cache_dtype,
    const void* k_scale, const void* v_scale, cudaStream_t stream,
    const std::string& layout) {
  dim3 grid(params.num_heads, params.batch_size);
  dim3 block(std::min(params.head_dim, 1024));
  bool hnd = (layout == "hnd");
  VMC_DISPATCH_FLOATING_TYPES(params.dtype, "paged_attention_v1",
    launch_paged_attn_v1<scalar_t>(
        out, query, key_cache, value_cache,
        params, block_tables, seq_lens,
        grid, block, stream, hnd);
  );
}

void paged_attention_v2(
    void* out, void* exp_sums, void* max_logits, void* tmp_out,
    const void* query, const void* key_cache, const void* value_cache,
    const vm_c::AttentionParams& params,
    const int32_t* block_tables, const int32_t* seq_lens,
    const float* alibi_slopes, const std::string& kv_cache_dtype,
    const void* k_scale, const void* v_scale, cudaStream_t stream,
    const std::string& layout) {
  dim3 grid(params.num_heads, params.batch_size);
  dim3 block(std::min(params.head_dim, 1024));
  bool hnd = (layout == "hnd");
  VMC_DISPATCH_FLOATING_TYPES(params.dtype, "paged_attention_v2",
    launch_paged_attn_v2<scalar_t>(
        out, exp_sums, query, key_cache, value_cache,
        params, block_tables, seq_lens,
        grid, block, stream, hnd);
  );
}

} // namespace vm_c
