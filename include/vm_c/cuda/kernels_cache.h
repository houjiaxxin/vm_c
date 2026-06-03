#pragma once

#include "vm_c_tensor.h"
#include <cstdint>
#include <string>
#include <optional>

namespace vm_c {

void swap_blocks(
    void* src, void* dst,
    int64_t block_size_in_bytes,
    const int64_t* block_mapping, int64_t num_blocks,
    int src_device, int dst_device,
    cudaStream_t stream = 0);

void reshape_and_cache(
    const void* key, const void* value,
    void* key_cache, void* value_cache,
    const int64_t* slot_mapping,
    int num_tokens, int num_kv_heads, int head_dim,
    int64_t block_size, int64_t cache_stride,
    const std::string& kv_cache_dtype,
    const void* k_scale, const void* v_scale,
    ScalarType dtype,
    cudaStream_t stream = 0,
    const std::string& layout = "nhd");

void reshape_and_cache_flash(
    const void* key, const void* value,
    void* key_cache, void* value_cache,
    const int64_t* slot_mapping,
    int num_tokens, int num_kv_heads, int head_dim,
    int64_t block_size, int64_t cache_stride,
    const std::string& kv_cache_dtype,
    const void* k_scale, const void* v_scale,
    ScalarType dtype,
    cudaStream_t stream = 0,
    const std::string& layout = "nhd");

void convert_fp8(
    void* dst_cache, const void* src_cache,
    double scale, int64_t num_elements,
    const std::string& kv_cache_dtype,
    ScalarType dtype,
    cudaStream_t stream = 0);

}
