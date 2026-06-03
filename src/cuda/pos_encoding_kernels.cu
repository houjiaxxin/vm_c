#include "vm_c/cuda/kernels_pos_encoding.h"
#include "vm_c/cuda/vm_c_dispatch.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace vm_c {

namespace {

template <typename scalar_t, typename cache_t, bool IS_NEOX, bool IS_BF16_CACHE = false>
__global__ void rotary_embedding_kernel(
    const int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const cache_t* __restrict__ cos_sin_cache,
    const int rot_dim, const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int num_heads, const int num_kv_heads,
    const int head_size, const int64_t rope_dim_offset, const bool inverse) {
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const cache_t* cache_ptr = cos_sin_cache + pos * rot_dim;
  const int embed_dim = rot_dim / 2;
  const cache_t* cos_ptr = cache_ptr;
  const cache_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head =
        token_idx * query_stride + head_idx * head_stride + rope_dim_offset;
    int x_index, y_index;
    float cos_f, sin_f;
    if constexpr (IS_NEOX) {
      x_index = i % embed_dim;
      y_index = embed_dim + (i % embed_dim);
      cos_f = static_cast<float>(cos_ptr[i % embed_dim]);
      sin_f = static_cast<float>(sin_ptr[i % embed_dim]);
    } else {
      x_index = 2 * (i % embed_dim);
      y_index = 2 * (i % embed_dim) + 1;
      cos_f = static_cast<float>(cos_ptr[(i % embed_dim)]);
      sin_f = static_cast<float>(sin_ptr[(i % embed_dim)]);
    }
    if (inverse) sin_f = -sin_f;
    float x_f = static_cast<float>(query[token_head + x_index]);
    float y_f = static_cast<float>(query[token_head + y_index]);
    query[token_head + x_index] = static_cast<scalar_t>(x_f * cos_f - y_f * sin_f);
    query[token_head + y_index] = static_cast<scalar_t>(y_f * cos_f + x_f * sin_f);
  }

  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
      const int head_idx = i / embed_dim;
      const int64_t token_head =
          token_idx * key_stride + head_idx * head_stride + rope_dim_offset;
      int x_index, y_index;
      float cos_f, sin_f;
      if constexpr (IS_NEOX) {
        x_index = i % embed_dim;
        y_index = embed_dim + (i % embed_dim);
        cos_f = static_cast<float>(cos_ptr[i % embed_dim]);
        sin_f = static_cast<float>(sin_ptr[i % embed_dim]);
      } else {
        x_index = 2 * (i % embed_dim);
        y_index = 2 * (i % embed_dim) + 1;
        cos_f = static_cast<float>(cos_ptr[(i % embed_dim)]);
        sin_f = static_cast<float>(sin_ptr[(i % embed_dim)]);
      }
      if (inverse) sin_f = -sin_f;
      float x_f = static_cast<float>(key[token_head + x_index]);
      float y_f = static_cast<float>(key[token_head + y_index]);
      key[token_head + x_index] = static_cast<scalar_t>(x_f * cos_f - y_f * sin_f);
      key[token_head + y_index] = static_cast<scalar_t>(y_f * cos_f + x_f * sin_f);
    }
  }
}

}

void rotary_embedding(
    const int64_t* positions,
    void* query, void* key,
    int num_tokens, int num_heads, int num_kv_heads,
    int head_size, int rot_dim,
    const void* cos_sin_cache,
    bool is_neox, int64_t rope_dim_offset, bool inverse,
    int64_t query_stride, int64_t key_stride, int64_t head_stride,
    ScalarType dtype, cudaStream_t stream) {
  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * rot_dim / 2, 1024));
  if (is_neox) {
    VMC_DISPATCH_FLOATING_TYPES(dtype, "rotary_embedding_neox",
      rotary_embedding_kernel<scalar_t, scalar_t, true>
          <<<grid, block, 0, stream>>>(
              positions,
              static_cast<scalar_t*>(query),
              static_cast<scalar_t*>(key),
              static_cast<const scalar_t*>(cos_sin_cache),
              rot_dim, query_stride, key_stride, head_stride,
              num_heads, num_kv_heads, head_size, rope_dim_offset, inverse);
    );
  } else {
    VMC_DISPATCH_FLOATING_TYPES(dtype, "rotary_embedding_gptj",
      rotary_embedding_kernel<scalar_t, scalar_t, false>
          <<<grid, block, 0, stream>>>(
              positions,
              static_cast<scalar_t*>(query),
              static_cast<scalar_t*>(key),
              static_cast<const scalar_t*>(cos_sin_cache),
              rot_dim, query_stride, key_stride, head_stride,
              num_heads, num_kv_heads, head_size, rope_dim_offset, inverse);
    );
  }
}

namespace {

template <typename scalar_t, typename cache_t, bool IS_NEOX>
__global__ void dual_chunk_rope_kernel(
    const int64_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const cache_t* __restrict__ q_cache,
    const cache_t* __restrict__ qc_cache,
    const cache_t* __restrict__ k_cache,
    const cache_t* __restrict__ qc_no_clamp_cache,
    const cache_t* __restrict__ q_inter_cache,
    const int rot_dim, const int chunk_size, const int local_size,
    const int64_t query_stride, const int64_t key_stride,
    const int64_t head_stride, const int num_heads, const int num_kv_heads,
    const int head_size, const int64_t rope_dim_offset) {
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const int embed_dim = rot_dim / 2;

  int offset_in_chunk = pos % chunk_size;
  bool is_local = (offset_in_chunk < local_size);

  const cache_t* q_cos_ptr;
  const cache_t* q_sin_ptr;
  const cache_t* k_cos_ptr;
  const cache_t* k_sin_ptr;

  if (is_local) {
    q_cos_ptr = q_inter_cache + offset_in_chunk * rot_dim;
    q_sin_ptr = q_inter_cache + offset_in_chunk * rot_dim + embed_dim;
    k_cos_ptr = k_cache + pos * rot_dim;
    k_sin_ptr = k_cache + pos * rot_dim + embed_dim;
  } else {
    int qc_idx = offset_in_chunk - local_size;
    q_cos_ptr = qc_cache + qc_idx * rot_dim;
    q_sin_ptr = qc_cache + qc_idx * rot_dim + embed_dim;
    k_cos_ptr = k_cache + pos * rot_dim;
    k_sin_ptr = k_cache + pos * rot_dim + embed_dim;
  }

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int64_t token_head =
        token_idx * query_stride + head_idx * head_stride + rope_dim_offset;
    int x_index, y_index;
    float cos_f, sin_f;
    if constexpr (IS_NEOX) {
      x_index = i % embed_dim;
      y_index = embed_dim + (i % embed_dim);
      cos_f = static_cast<float>(q_cos_ptr[i % embed_dim]);
      sin_f = static_cast<float>(q_sin_ptr[i % embed_dim]);
    } else {
      x_index = 2 * (i % embed_dim);
      y_index = 2 * (i % embed_dim) + 1;
      cos_f = static_cast<float>(q_cos_ptr[(i % embed_dim)]);
      sin_f = static_cast<float>(q_sin_ptr[(i % embed_dim)]);
    }
    float x_f = static_cast<float>(query[token_head + x_index]);
    float y_f = static_cast<float>(query[token_head + y_index]);
    query[token_head + x_index] = static_cast<scalar_t>(x_f * cos_f - y_f * sin_f);
    query[token_head + y_index] = static_cast<scalar_t>(y_f * cos_f + x_f * sin_f);
  }

  if (key != nullptr) {
    const int nk = num_kv_heads * embed_dim;
    for (int i = threadIdx.x; i < nk; i += blockDim.x) {
      const int head_idx = i / embed_dim;
      const int64_t token_head =
          token_idx * key_stride + head_idx * head_stride + rope_dim_offset;
      int x_index, y_index;
      float cos_f, sin_f;
      if constexpr (IS_NEOX) {
        x_index = i % embed_dim;
        y_index = embed_dim + (i % embed_dim);
        cos_f = static_cast<float>(k_cos_ptr[i % embed_dim]);
        sin_f = static_cast<float>(k_sin_ptr[i % embed_dim]);
      } else {
        x_index = 2 * (i % embed_dim);
        y_index = 2 * (i % embed_dim) + 1;
        cos_f = static_cast<float>(k_cos_ptr[(i % embed_dim)]);
        sin_f = static_cast<float>(k_sin_ptr[(i % embed_dim)]);
      }
      float x_f = static_cast<float>(key[token_head + x_index]);
      float y_f = static_cast<float>(key[token_head + y_index]);
      key[token_head + x_index] = static_cast<scalar_t>(x_f * cos_f - y_f * sin_f);
      key[token_head + y_index] = static_cast<scalar_t>(y_f * cos_f + x_f * sin_f);
    }
  }
}

}

void dual_chunk_rotary_embedding(
    const int64_t* positions,
    void* query, void* key,
    int num_tokens, int num_heads, int num_kv_heads,
    int head_size, int rot_dim,
    const void* q_cache,
    const void* qc_cache,
    const void* k_cache,
    const void* qc_no_clamp_cache,
    const void* q_inter_cache,
    int chunk_size, int local_size,
    bool is_neox, int64_t rope_dim_offset,
    int64_t query_stride, int64_t key_stride, int64_t head_stride,
    ScalarType dtype, cudaStream_t stream) {
  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * rot_dim / 2, 1024));
  if (is_neox) {
    VMC_DISPATCH_FLOATING_TYPES(dtype, "dual_chunk_rope_neox",
      dual_chunk_rope_kernel<scalar_t, scalar_t, true>
          <<<grid, block, 0, stream>>>(
              positions,
              static_cast<scalar_t*>(query),
              static_cast<scalar_t*>(key),
              static_cast<const scalar_t*>(q_cache),
              static_cast<const scalar_t*>(qc_cache),
              static_cast<const scalar_t*>(k_cache),
              static_cast<const scalar_t*>(qc_no_clamp_cache),
              static_cast<const scalar_t*>(q_inter_cache),
              rot_dim, chunk_size, local_size,
              query_stride, key_stride, head_stride,
              num_heads, num_kv_heads, head_size, rope_dim_offset);
    );
  } else {
    VMC_DISPATCH_FLOATING_TYPES(dtype, "dual_chunk_rope_gptj",
      dual_chunk_rope_kernel<scalar_t, scalar_t, false>
          <<<grid, block, 0, stream>>>(
              positions,
              static_cast<scalar_t*>(query),
              static_cast<scalar_t*>(key),
              static_cast<const scalar_t*>(q_cache),
              static_cast<const scalar_t*>(qc_cache),
              static_cast<const scalar_t*>(k_cache),
              static_cast<const scalar_t*>(qc_no_clamp_cache),
              static_cast<const scalar_t*>(q_inter_cache),
              rot_dim, chunk_size, local_size,
              query_stride, key_stride, head_stride,
              num_heads, num_kv_heads, head_size, rope_dim_offset);
    );
  }
}

}
