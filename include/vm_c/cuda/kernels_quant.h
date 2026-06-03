#pragma once

#include "vm_c_tensor.h"
#include <cstdint>
#include <optional>

namespace vm_c {

void static_scaled_fp8_quant(
    void* out, const void* input, const void* scale,
    int64_t num_elements,
    ScalarType input_dtype,
    cudaStream_t stream = 0);

void dynamic_scaled_fp8_quant(
    void* out, const void* input, void* scale,
    int64_t num_elements,
    ScalarType input_dtype,
    cudaStream_t stream = 0);

void dynamic_per_token_scaled_fp8_quant(
    void* out, const void* input, void* scale,
    int64_t num_tokens, int64_t hidden_size,
    ScalarType input_dtype,
    cudaStream_t stream = 0);

void static_scaled_int8_quant(
    void* out, const void* input, const void* scale,
    int64_t num_elements,
    ScalarType input_dtype,
    const void* azp,
    cudaStream_t stream = 0);

void dynamic_scaled_int8_quant(
    void* out, const void* input, void* scales,
    int64_t num_elements,
    ScalarType input_dtype,
    void* azp,
    cudaStream_t stream = 0);

void per_token_group_quant_fp8(
    void* output, const void* input,
    int64_t num_tokens, int64_t num_groups, int64_t group_size,
    ScalarType input_dtype,
    cudaStream_t stream = 0);

void per_token_group_quant_int8(
    void* output, const void* input,
    int64_t num_tokens, int64_t num_groups, int64_t group_size,
    ScalarType input_dtype,
    cudaStream_t stream = 0);

}
