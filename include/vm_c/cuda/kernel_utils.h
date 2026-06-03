#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ── to_float / from_float ──────────────────────────────────────────────
// 统一转换函数，消除 turboquant_kernels.cu / moe_kernels.cu 中的重复定义
// 使用与原始代码相同的模板+特化模式，确保调用方 `to_float(v)` / `from_float<scalar_t>(v)` 无需修改

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t v) {
    return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float to_float(half v) {
    return __half2float(v);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float v) {
    return static_cast<scalar_t>(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float(float v) {
    return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ half from_float(float v) {
    return __float2half(v);
}

// ── VMC_MAX_OP ─────────────────────────────────────────────────────────
// CUDA 12/13 兼容版本，消除 quant_kernels.cu / sampler_kernels.cu 中的重复定义

#if CUDA_VERSION >= 13000
#include <cuda/std/utility>
#define VMC_MAX_OP cuda::maximum<>{}
#else
struct VmcMaxOp {
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const {
        return (a > b) ? a : b;
    }
};
#define VMC_MAX_OP VmcMaxOp{}
#endif
