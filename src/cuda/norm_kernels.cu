#include "vm_c/cuda/kernels_norm.h"
#include "vm_c/cuda/vm_c_dispatch.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace vm_c {

// ── rms_norm_kernel（对齐 vLLM layernorm_kernels.cu，无 residual）──────────
template <typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t input_stride,
    const scalar_t* __restrict__ weight,
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const scalar_t x = input[blockIdx.x * input_stride + idx];
    const float xf = static_cast<float>(x);
    variance += xf * xf;
  }

  __shared__ float s_reduce[1024];
  s_reduce[threadIdx.x] = variance;
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    if (threadIdx.x < stride) {
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(s_reduce[0] / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = static_cast<float>(input[blockIdx.x * input_stride + idx]);
    const float w = static_cast<float>(weight[idx]);
    output[blockIdx.x * hidden_size + idx] =
        static_cast<scalar_t>(x * s_variance * w);
  }
}

template <typename scalar_t>
void launch_rms_norm(
    scalar_t* output, const scalar_t* input, int64_t input_stride,
    const scalar_t* weight, float epsilon, int num_tokens, int hidden_size,
    cudaStream_t stream) {
  dim3 grid(num_tokens);
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  rms_norm_kernel<<<grid, block, 0, stream>>>(
      output, input, input_stride, weight, epsilon, num_tokens, hidden_size);
}

// ── fused_add_rms_norm_kernel ──────────────────────────────────────────
template <typename scalar_t>
__global__ void fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,
    const int64_t input_stride,
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    const float epsilon, const int num_tokens, const int hidden_size) {
  float variance = 0.0f;
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * input_stride + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }

  __shared__ float s_reduce[1024];
  s_reduce[threadIdx.x] = variance;
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    if (threadIdx.x < stride) {
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    }
  }
  __syncthreads();
  float s_variance = rsqrtf(s_reduce[0] / hidden_size + epsilon);

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    float w = (float)weight[idx];
    input[blockIdx.x * input_stride + idx] = (scalar_t)(x * s_variance * w);
  }
}

template <typename scalar_t>
void launch_fused_add_rms_norm(
    scalar_t* input, int64_t input_stride,
    scalar_t* residual, const scalar_t* weight,
    float epsilon, int num_tokens, int hidden_size,
    cudaStream_t stream) {
  dim3 grid(num_tokens);
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  fused_add_rms_norm_kernel<<<grid, block, 0, stream>>>(
      input, input_stride, residual, weight,
      epsilon, num_tokens, hidden_size);
}

// ── silu_and_mul kernel ────────────────────────────────────────────────
template <typename scalar_t, float (*ACT_FN)(const float&), bool act_first>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ input,
    int64_t num_tokens, int64_t hidden_size) {
  const int64_t token_idx = blockIdx.x;
  const scalar_t* gate_up = input + token_idx * 2 * hidden_size;
  scalar_t* dst = out + token_idx * hidden_size;

  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float gate = (float)gate_up[idx];
    float up = (float)gate_up[hidden_size + idx];
    float result;
    if constexpr (act_first) {
      result = ACT_FN(gate) * up;
    } else {
      result = gate * ACT_FN(up);
    }
    dst[idx] = (scalar_t)result;
  }
}

__device__ __forceinline__ float silu_fn(const float& x) {
  return x / (1.0f + expf(-x));
}

template <typename scalar_t, float (*ACT_FN)(const float&), bool act_first>
void launch_act_and_mul(
    scalar_t* out, const scalar_t* input,
    int64_t num_tokens, int64_t hidden_size,
    cudaStream_t stream) {
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, int64_t(1024)));
  act_and_mul_kernel<scalar_t, ACT_FN, act_first><<<grid, block, 0, stream>>>(
      out, input, num_tokens, hidden_size);
}

// ── Host launchers ─────────────────────────────────────────────────────

void rms_norm(void* output, const void* input, const void* weight,
              int num_tokens, int hidden_size, int64_t input_stride,
              float epsilon, ScalarType dtype, cudaStream_t stream) {
  VMC_DISPATCH_FLOATING_TYPES(dtype, "rms_norm_kernel",
    launch_rms_norm<scalar_t>(
        static_cast<scalar_t*>(output),
        static_cast<const scalar_t*>(input),
        input_stride,
        static_cast<const scalar_t*>(weight),
        epsilon, num_tokens, hidden_size, stream);
  );
}

void fused_add_rms_norm(void* input, void* residual, const void* weight,
                         int num_tokens, int hidden_size, int64_t input_stride,
                         float epsilon, ScalarType dtype, cudaStream_t stream) {
  VMC_DISPATCH_FLOATING_TYPES(dtype, "fused_add_rms_norm_kernel",
    launch_fused_add_rms_norm<scalar_t>(
        static_cast<scalar_t*>(input), input_stride,
        static_cast<scalar_t*>(residual),
        static_cast<const scalar_t*>(weight),
        epsilon, num_tokens, hidden_size, stream);
  );
}

void silu_and_mul(void* out, const void* input,
                   int64_t num_tokens, int64_t hidden_size,
                   ScalarType dtype, cudaStream_t stream) {
  VMC_DISPATCH_FLOATING_TYPES(dtype, "silu_and_mul",
    launch_act_and_mul<scalar_t, silu_fn, true>(
        static_cast<scalar_t*>(out),
        static_cast<const scalar_t*>(input),
        num_tokens, hidden_size, stream);
  );
}

// ── add_tensor kernel ──────────────────────────────────────────────
template <typename scalar_t>
__global__ void add_tensor_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    int64_t num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; idx < num_elements; idx += stride) {
    out[idx] = static_cast<scalar_t>(
        static_cast<float>(a[idx]) + static_cast<float>(b[idx]));
  }
}

template <typename scalar_t>
void launch_add_tensor(
    scalar_t* out, const scalar_t* a, const scalar_t* b,
    int64_t num_elements, cudaStream_t stream) {
  int block = 256;
  int grid = static_cast<int>(std::min<int64_t>(
      (num_elements + block - 1) / block, 65535));
  add_tensor_kernel<<<grid, block, 0, stream>>>(out, a, b, num_elements);
}

void add_tensor(void* out, const void* a, const void* b,
                 int64_t num_elements,
                 ScalarType dtype, cudaStream_t stream) {
  VMC_DISPATCH_FLOATING_TYPES(dtype, "add_tensor",
    launch_add_tensor<scalar_t>(
        static_cast<scalar_t*>(out),
        static_cast<const scalar_t*>(a),
        static_cast<const scalar_t*>(b),
        num_elements, stream);
  );
}

// ── silu_mul kernel ─────────────────────────────────────────────────────
template <typename scalar_t>
__global__ void silu_mul_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    int64_t num_tokens, int64_t hidden_size) {
  int64_t token_idx = blockIdx.x;
  int64_t base = token_idx * hidden_size;
  for (int64_t idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float g = static_cast<float>(gate[base + idx]);
    float u = static_cast<float>(up[base + idx]);
    float s = g / (1.0f + expf(-g));
    out[base + idx] = static_cast<scalar_t>(s * u);
  }
}

template <typename scalar_t>
void launch_silu_mul(
    scalar_t* out, const scalar_t* gate, const scalar_t* up,
    int64_t num_tokens, int64_t hidden_size,
    cudaStream_t stream) {
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, int64_t(1024)));
  silu_mul_kernel<<<grid, block, 0, stream>>>(
      out, gate, up, num_tokens, hidden_size);
}

void silu_mul(void* out, const void* gate, const void* up,
               int64_t num_tokens, int64_t hidden_size,
               ScalarType dtype, cudaStream_t stream) {
  VMC_DISPATCH_FLOATING_TYPES(dtype, "silu_mul",
    launch_silu_mul<scalar_t>(
        static_cast<scalar_t*>(out),
        static_cast<const scalar_t*>(gate),
        static_cast<const scalar_t*>(up),
        num_tokens, hidden_size, stream);
  );
}

} // namespace vm_c
