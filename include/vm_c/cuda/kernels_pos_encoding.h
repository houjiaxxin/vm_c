#pragma once

#include "vm_c_tensor.h"
#include <cstdint>

namespace vm_c {

void rotary_embedding(
    const int64_t* positions,
    void* query, void* key,
    int num_tokens, int num_heads, int num_kv_heads,
    int head_size, int rot_dim,
    const void* cos_sin_cache,
    bool is_neox, int64_t rope_dim_offset, bool inverse,
    int64_t query_stride, int64_t key_stride, int64_t head_stride,
    ScalarType dtype,
    cudaStream_t stream = 0);

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
    ScalarType dtype,
    cudaStream_t stream = 0);

}
