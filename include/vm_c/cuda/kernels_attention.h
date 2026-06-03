#pragma once

#include "vm_c_tensor.h"
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <optional>

namespace vm_c {

struct AttentionParams {
    int batch_size = 0;
    int num_tokens = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int block_size = 0;
    int max_seq_len = 0;
    int max_num_blocks_per_req = 0;
    float scale = 1.0f;
    ScalarType dtype = ScalarType::FLOAT16;
    int tp_rank = 0;
    int blocksparse_local_blocks = 0;
    int blocksparse_vert_stride = 0;
    int blocksparse_block_size = 0;
    int blocksparse_head_sliding_step = 0;
};

void paged_attention_v1(
    void* out,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    const AttentionParams& params,
    const int32_t* block_tables,
    const int32_t* seq_lens,
    const float* alibi_slopes,
    const std::string& kv_cache_dtype,
    const void* k_scale,
    const void* v_scale,
    cudaStream_t stream = 0,
    const std::string& layout = "nhd");

void paged_attention_prefill(
    void* out,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    int num_tokens, int num_heads, int num_kv_heads, int head_dim,
    float scale, int64_t block_size, int64_t max_num_blocks_per_req,
    const int32_t* block_tables,
    const int32_t* seq_lens,
    const int32_t* token_to_seq,
    const int32_t* token_positions,
    int num_reqs,
    const std::string& kv_cache_dtype,
    ScalarType dtype,
    cudaStream_t stream = 0,
    const std::string& layout = "nhd");

void paged_attention_v2(
    void* out,
    void* exp_sums,
    void* max_logits,
    void* tmp_out,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    const AttentionParams& params,
    const int32_t* block_tables,
    const int32_t* seq_lens,
    const float* alibi_slopes,
    const std::string& kv_cache_dtype,
    const void* k_scale,
    const void* v_scale,
    cudaStream_t stream = 0,
    const std::string& layout = "nhd");

}
