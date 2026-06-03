#include "vm_c/cuda/kernels_cache.h"
#include "vm_c/cuda/vm_c_dispatch.h"
#include "vm_c/cuda/gpu_arch.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstring>
#include <algorithm>

namespace vm_c {
namespace {

template <typename scalar_t, typename cache_t, bool IS_FP8_KV_CACHE, bool HND_LAYOUT = false>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    cache_t* __restrict__ key_cache,
    cache_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    const int num_tokens, const int num_kv_heads, const int head_dim,
    const int64_t block_size, const int64_t cache_stride,
    const float k_scale, const float v_scale) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot = slot_mapping[token_idx];
  if (slot < 0) return;

  const int64_t block_idx = slot / block_size;
  const int64_t block_offset = slot % block_size;

  for (int h = threadIdx.x; h < num_kv_heads; h += blockDim.x) {
    const int64_t src_idx = token_idx * num_kv_heads * head_dim + h * head_dim;
    // NHD: [block_idx][block_offset][h][d]
    // HND: [block_idx][h][block_offset][d]
    int64_t dst_idx;
    if constexpr (HND_LAYOUT) {
      dst_idx = block_idx * cache_stride + h * block_size * head_dim + block_offset * head_dim;
    } else {
      dst_idx = block_idx * cache_stride + block_offset * num_kv_heads * head_dim + h * head_dim;
    }

    for (int d = 0; d < head_dim; ++d) {
      if constexpr (IS_FP8_KV_CACHE) {
        key_cache[dst_idx + d] = static_cast<cache_t>(
            static_cast<float>(key[src_idx + d]) / k_scale);
        value_cache[dst_idx + d] = static_cast<cache_t>(
            static_cast<float>(value[src_idx + d]) / v_scale);
      } else {
        key_cache[dst_idx + d] = static_cast<cache_t>(key[src_idx + d]);
        value_cache[dst_idx + d] = static_cast<cache_t>(value[src_idx + d]);
      }
    }
  }
}

}

void swap_blocks(
    void* src, void* dst,
    int64_t block_size_in_bytes,
    const int64_t* block_mapping, int64_t num_blocks,
    int src_device, int dst_device,
    cudaStream_t stream) {
  cudaMemcpyKind memcpy_type;
  if (src_device >= 0 && dst_device >= 0) {
    memcpy_type = cudaMemcpyDeviceToDevice;
  } else if (src_device >= 0 && dst_device < 0) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else if (src_device < 0 && dst_device >= 0) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    return;
  }

  char* src_ptr = static_cast<char*>(src);
  char* dst_ptr = static_cast<char*>(dst);

  for (int64_t i = 0; i < num_blocks; i++) {
    int64_t src_block_number = block_mapping[i * 2];
    int64_t dst_block_number = block_mapping[i * 2 + 1];
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    CUDA_CHECK(cudaMemcpyAsync(dst_ptr + dst_offset, src_ptr + src_offset,
                    block_size_in_bytes, memcpy_type, stream));
  }
}

void reshape_and_cache(
    const void* key, const void* value,
    void* key_cache, void* value_cache,
    const int64_t* slot_mapping,
    int num_tokens, int num_kv_heads, int head_dim,
    int64_t block_size, int64_t cache_stride,
    const std::string& kv_cache_dtype,
    const void* k_scale, const void* v_scale,
    ScalarType dtype, cudaStream_t stream,
    const std::string& layout) {
  dim3 grid(num_tokens);
  dim3 block(std::min(num_kv_heads, 1024));

  bool is_fp8 = (kv_cache_dtype == "fp8_e4m3" || kv_cache_dtype == "fp8_e5m2");
  float k_s = 1.0f, v_s = 1.0f;
  if (k_scale) k_s = *static_cast<const float*>(k_scale);
  if (v_scale) v_s = *static_cast<const float*>(v_scale);

  const bool hnd = (layout == "hnd");

  if (is_fp8) {
    if (hnd) {
      VMC_DISPATCH_FLOATING_TYPES(dtype, "reshape_and_cache_fp8_hnd",
        reshape_and_cache_kernel<scalar_t, __nv_fp8_e4m3, true, true>
            <<<grid, block, 0, stream>>>(
                static_cast<const scalar_t*>(key),
                static_cast<const scalar_t*>(value),
                static_cast<__nv_fp8_e4m3*>(key_cache),
                static_cast<__nv_fp8_e4m3*>(value_cache),
                slot_mapping, num_tokens, num_kv_heads, head_dim,
                block_size, cache_stride, k_s, v_s);
      );
    } else {
      VMC_DISPATCH_FLOATING_TYPES(dtype, "reshape_and_cache_fp8_nhd",
        reshape_and_cache_kernel<scalar_t, __nv_fp8_e4m3, true, false>
            <<<grid, block, 0, stream>>>(
                static_cast<const scalar_t*>(key),
                static_cast<const scalar_t*>(value),
                static_cast<__nv_fp8_e4m3*>(key_cache),
                static_cast<__nv_fp8_e4m3*>(value_cache),
                slot_mapping, num_tokens, num_kv_heads, head_dim,
                block_size, cache_stride, k_s, v_s);
      );
    }
  } else {
    if (hnd) {
      VMC_DISPATCH_FLOATING_TYPES(dtype, "reshape_and_cache_hnd",
        reshape_and_cache_kernel<scalar_t, scalar_t, false, true>
            <<<grid, block, 0, stream>>>(
                static_cast<const scalar_t*>(key),
                static_cast<const scalar_t*>(value),
                static_cast<scalar_t*>(key_cache),
                static_cast<scalar_t*>(value_cache),
                slot_mapping, num_tokens, num_kv_heads, head_dim,
                block_size, cache_stride, k_s, v_s);
      );
    } else {
      VMC_DISPATCH_FLOATING_TYPES(dtype, "reshape_and_cache_nhd",
        reshape_and_cache_kernel<scalar_t, scalar_t, false, false>
            <<<grid, block, 0, stream>>>(
                static_cast<const scalar_t*>(key),
                static_cast<const scalar_t*>(value),
                static_cast<scalar_t*>(key_cache),
                static_cast<scalar_t*>(value_cache),
                slot_mapping, num_tokens, num_kv_heads, head_dim,
                block_size, cache_stride, k_s, v_s);
      );
    }
  }
}

void reshape_and_cache_flash(
    const void* key, const void* value,
    void* key_cache, void* value_cache,
    const int64_t* slot_mapping,
    int num_tokens, int num_kv_heads, int head_dim,
    int64_t block_size, int64_t cache_stride,
    const std::string& kv_cache_dtype,
    const void* k_scale, const void* v_scale,
    ScalarType dtype, cudaStream_t stream,
    const std::string& layout) {
  reshape_and_cache(key, value, key_cache, value_cache, slot_mapping,
                    num_tokens, num_kv_heads, head_dim,
                    block_size, cache_stride, kv_cache_dtype,
                    k_scale, v_scale, dtype, stream, layout);
}

template <typename scalar_t, typename fp8_t>
__global__ void convert_fp8_kernel(
    fp8_t* __restrict__ dst_cache,
    const scalar_t* __restrict__ src_cache,
    float scale,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= num_elements) return;
  float val = static_cast<float>(src_cache[idx]);
  dst_cache[idx] = static_cast<fp8_t>(val * scale);
}

void convert_fp8(
    void* dst_cache, const void* src_cache,
    double scale, int64_t num_elements,
    const std::string& kv_cache_dtype,
    ScalarType dtype,
    cudaStream_t stream) {
  if (num_elements <= 0) return;
  const void* src = src_cache;
  float s = static_cast<float>(scale);
  bool is_e4m3 = (kv_cache_dtype == "fp8_e4m3");

  if (is_e4m3) {
    VMC_DISPATCH_FLOATING_TYPES(dtype, "convert_fp8_e4m3",
      convert_fp8_kernel<scalar_t, __nv_fp8_e4m3><<<(int)((num_elements + 255) / 256), 256, 0, stream>>>(
          static_cast<__nv_fp8_e4m3*>(dst_cache), static_cast<const scalar_t*>(src), s, num_elements);
    );
  } else {
    VMC_DISPATCH_FLOATING_TYPES(dtype, "convert_fp8_e5m2",
      VMC_DISPATCH_FP8_TYPES(ScalarType::FLOAT8_E5M2, "convert_fp8_e5m2",
        convert_fp8_kernel<scalar_t, __nv_fp8_e5m2><<<(int)((num_elements + 255) / 256), 256, 0, stream>>>(
            static_cast<__nv_fp8_e5m2*>(dst_cache), static_cast<const scalar_t*>(src), s, num_elements);
      );
    );
  }
}

}
