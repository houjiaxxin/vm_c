#pragma once

#include "vm_c_tensor.h"
#include <cstdint>

namespace vm_c {

void rms_norm(
    void* output, const void* input, const void* weight,
    int num_tokens, int hidden_size, int64_t input_stride,
    float epsilon, ScalarType dtype, cudaStream_t stream);

void fused_add_rms_norm(
    void* input, void* residual, const void* weight,
    int num_tokens, int hidden_size, int64_t input_stride,
    float epsilon,
    ScalarType dtype,
    cudaStream_t stream = 0);

void silu_and_mul(
    void* out, const void* input,
    int64_t num_tokens, int64_t hidden_size,
    ScalarType dtype,
    cudaStream_t stream = 0);

// 元素级加法: out[i] = a[i] + b[i]
void add_tensor(
    void* out, const void* a, const void* b,
    int64_t num_elements,
    ScalarType dtype,
    cudaStream_t stream = 0);

// silu_mul: out[i] = silu(gate[i]) * up[i]  (gate 和 up 是独立的指针)
void silu_mul(
    void* out, const void* gate, const void* up,
    int64_t num_tokens, int64_t hidden_size,
    ScalarType dtype,
    cudaStream_t stream = 0);

}
