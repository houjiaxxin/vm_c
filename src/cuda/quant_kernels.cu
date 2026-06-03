#include "vm_c/cuda/kernels_quant.h"
#include "vm_c/cuda/vm_c_dispatch.h"
#include "vm_c/cuda/kernel_utils.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cub/cub.cuh>
#include <cmath>
#include <algorithm>

namespace vm_c {

namespace {

template <typename input_t, typename output_t>
__global__ void static_scaled_fp8_quant_kernel(
    output_t* __restrict__ out,
    const input_t* __restrict__ input,
    const float scale,
    int64_t num_elements) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    out[idx] = static_cast<output_t>(static_cast<float>(input[idx]) / scale);
  }
}

template <typename input_t, typename output_t>
__global__ void dynamic_scaled_fp8_quant_kernel(
    output_t* __restrict__ out,
    const input_t* __restrict__ input,
    float* __restrict__ scale,
    int64_t num_elements) {
  float absmax = 0.0f;
  for (int64_t idx = threadIdx.x; idx < num_elements; idx += blockDim.x) {
    absmax = fmaxf(absmax, fabsf(static_cast<float>(input[idx])));
  }
  __shared__ float s_absmax;
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
   absmax = BlockReduce(temp_storage).Reduce(absmax, VMC_MAX_OP);
  if (threadIdx.x == 0) {
    s_absmax = absmax;
    *scale = absmax / 448.0f;
  }
  __syncthreads();

  float inv_scale = (s_absmax > 0.0f) ? 448.0f / s_absmax : 0.0f;
  for (int64_t idx = threadIdx.x; idx < num_elements; idx += blockDim.x) {
    out[idx] = static_cast<output_t>(static_cast<float>(input[idx]) * inv_scale);
  }
}

template <typename input_t, typename output_t>
__global__ void static_scaled_int8_quant_kernel(
    output_t* __restrict__ out,
    const input_t* __restrict__ input,
    const float scale,
    int64_t num_elements) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    out[idx] = static_cast<output_t>(roundf(static_cast<float>(input[idx]) / scale));
  }
}
template <typename input_t, typename output_t>
__global__ void dynamic_scaled_int8_quant_kernel(
    output_t* __restrict__ out,
    const input_t* __restrict__ input,
    float* __restrict__ scales,
    int64_t num_elements) {
  float absmax = 0.0f;
  for (int64_t idx = threadIdx.x; idx < num_elements; idx += blockDim.x) {
    absmax = fmaxf(absmax, fabsf(static_cast<float>(input[idx])));
  }
  __shared__ float s_absmax;
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
   absmax = BlockReduce(temp_storage).Reduce(absmax, VMC_MAX_OP);
  if (threadIdx.x == 0) {
    s_absmax = absmax;
    scales[0] = (absmax > 0.0f) ? absmax / 127.0f : 1.0f;
  }
  __syncthreads();

  float scale = scales[0];
  for (int64_t idx = threadIdx.x; idx < num_elements; idx += blockDim.x) {
    out[idx] = static_cast<output_t>(roundf(static_cast<float>(input[idx]) / scale));
  }
}

template <typename input_t, typename output_t>
__global__ void per_token_group_quant_kernel(
    output_t* __restrict__ out,
    float* __restrict__ scales,
    const input_t* __restrict__ input,
    int64_t num_tokens,
    int64_t num_groups,
    int64_t group_size) {
  int64_t token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;

  const input_t* token_input = input + token_idx * (num_groups * group_size);
  output_t* token_out = out + token_idx * (num_groups * group_size);
  float* token_scales = scales + token_idx * num_groups;

  for (int g = threadIdx.x; g < num_groups; g += blockDim.x) {
    float absmax = 0.0f;
    const input_t* group_ptr = token_input + g * group_size;
    for (int i = 0; i < group_size; ++i) {
      absmax = fmaxf(absmax, fabsf(static_cast<float>(group_ptr[i])));
    }
    float scale = (absmax > 0.0f) ? absmax / 127.0f : 1.0f;
    token_scales[g] = scale;
    for (int i = 0; i < group_size; ++i) {
      token_out[g * group_size + i] = static_cast<output_t>(roundf(static_cast<float>(group_ptr[i]) / scale));
    }
  }
}

template <typename input_t>
__global__ void per_token_group_quant_fp8_kernel(
    __nv_fp8_e4m3* __restrict__ out,
    float* __restrict__ scales,
    const input_t* __restrict__ input,
    int64_t num_tokens,
    int64_t num_groups,
    int64_t group_size) {
  int64_t token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;

  const input_t* token_input = input + token_idx * (num_groups * group_size);
  __nv_fp8_e4m3* token_out = out + token_idx * (num_groups * group_size);
  float* token_scales = scales + token_idx * num_groups;

  for (int g = threadIdx.x; g < num_groups; g += blockDim.x) {
    float absmax = 0.0f;
    const input_t* group_ptr = token_input + g * group_size;
    for (int i = 0; i < group_size; ++i) {
      absmax = fmaxf(absmax, fabsf(static_cast<float>(group_ptr[i])));
    }
    float scale = (absmax > 0.0f) ? absmax / 448.0f : 1.0f;
    token_scales[g] = scale;
    for (int i = 0; i < group_size; ++i) {
      token_out[g * group_size + i] = static_cast<__nv_fp8_e4m3>(static_cast<float>(group_ptr[i]) / scale);
    }
  }
}


}

void static_scaled_fp8_quant(
    void* out, const void* input, const void* scale,
    int64_t num_elements, ScalarType input_dtype, cudaStream_t stream) {
  float s = *static_cast<const float*>(scale);
  int64_t block_size = std::min(num_elements, int64_t(1024));
  int64_t grid_size = (num_elements + block_size - 1) / block_size;
  VMC_DISPATCH_FLOATING_TYPES(input_dtype, "static_scaled_fp8_quant",
    static_scaled_fp8_quant_kernel<scalar_t, __nv_fp8_e4m3>
        <<<grid_size, block_size, 0, stream>>>(
            static_cast<__nv_fp8_e4m3*>(out),
            static_cast<const scalar_t*>(input),
            s, num_elements);
  );
}

void dynamic_scaled_fp8_quant(
    void* out, const void* input, void* scale,
    int64_t num_elements, ScalarType input_dtype, cudaStream_t stream) {
  int64_t block_size = std::min(num_elements, int64_t(1024));
  VMC_DISPATCH_FLOATING_TYPES(input_dtype, "dynamic_scaled_fp8_quant",
    dynamic_scaled_fp8_quant_kernel<scalar_t, __nv_fp8_e4m3>
        <<<1, block_size, 0, stream>>>(
            static_cast<__nv_fp8_e4m3*>(out),
            static_cast<const scalar_t*>(input),
            static_cast<float*>(scale),
            num_elements);
  );
}

void dynamic_per_token_scaled_fp8_quant(
    void* out, const void* input, void* scale,
    int64_t num_tokens, int64_t hidden_size,
    ScalarType input_dtype, cudaStream_t stream) {
  for (int64_t t = 0; t < num_tokens; ++t) {
    VMC_DISPATCH_FLOATING_TYPES(input_dtype, "per_token_fp8_quant",
      const scalar_t* token_input = static_cast<const scalar_t*>(input) + t * hidden_size;
      __nv_fp8_e4m3* token_out = static_cast<__nv_fp8_e4m3*>(out) + t * hidden_size;
      float* token_scale = static_cast<float*>(scale) + t;
      dynamic_scaled_fp8_quant_kernel<scalar_t, __nv_fp8_e4m3>
          <<<1, std::min(hidden_size, int64_t(1024)), 0, stream>>>(
              token_out, token_input, token_scale, hidden_size);
    );
  }
}

void static_scaled_int8_quant(
    void* out, const void* input, const void* scale,
    int64_t num_elements, ScalarType input_dtype,
    const void* azp, cudaStream_t stream) {
  float s = *static_cast<const float*>(scale);
  int64_t block_size = std::min(num_elements, int64_t(1024));
  int64_t grid_size = (num_elements + block_size - 1) / block_size;
  VMC_DISPATCH_FLOATING_TYPES(input_dtype, "static_scaled_int8_quant",
    static_scaled_int8_quant_kernel<scalar_t, int8_t>
        <<<grid_size, block_size, 0, stream>>>(
            static_cast<int8_t*>(out),
            static_cast<const scalar_t*>(input),
            s, num_elements);
  );
}

void dynamic_scaled_int8_quant(
    void* out, const void* input, void* scales,
    int64_t num_elements, ScalarType input_dtype,
    void* azp, cudaStream_t stream) {
  int64_t block_size = std::min(num_elements, int64_t(1024));
  VMC_DISPATCH_FLOATING_TYPES(input_dtype, "dynamic_scaled_int8_quant",
    dynamic_scaled_int8_quant_kernel<scalar_t, int8_t>
        <<<1, block_size, 0, stream>>>(
            static_cast<int8_t*>(out),
            static_cast<const scalar_t*>(input),
            static_cast<float*>(scales),
            num_elements);
  );
}

void per_token_group_quant_fp8(
    void* output, const void* input,
    int64_t num_tokens, int64_t num_groups, int64_t group_size,
    ScalarType input_dtype, cudaStream_t stream) {
  int block_size = static_cast<int>(std::min(num_groups, static_cast<int64_t>(256)));
  VMC_DISPATCH_FLOATING_TYPES(input_dtype, "per_token_group_quant_fp8",
    per_token_group_quant_fp8_kernel<scalar_t>
        <<<num_tokens, block_size, 0, stream>>>(
            static_cast<__nv_fp8_e4m3*>(output),
            static_cast<float*>(nullptr),
            static_cast<const scalar_t*>(input),
            num_tokens, num_groups, group_size);
  );
}

void per_token_group_quant_int8(
    void* output, const void* input,
    int64_t num_tokens, int64_t num_groups, int64_t group_size,
    ScalarType input_dtype, cudaStream_t stream) {
  int block_size = static_cast<int>(std::min(num_groups, static_cast<int64_t>(256)));
  VMC_DISPATCH_FLOATING_TYPES(input_dtype, "per_token_group_quant_int8",
    per_token_group_quant_kernel<scalar_t, int8_t>
        <<<num_tokens, block_size, 0, stream>>>(
            static_cast<int8_t*>(output),
            static_cast<float*>(nullptr),
            static_cast<const scalar_t*>(input),
            num_tokens, num_groups, group_size);
  );
}

}
