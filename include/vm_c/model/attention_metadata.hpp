#pragma once

#include <cstdint>

namespace vm_c {

/// vLLM 风格 batch 元数据，供 TurboQuant bridge / llama forward 调度使用。
struct AttentionMetadata {
    int32_t* block_tables = nullptr;
    int32_t* seq_lens = nullptr;
    int64_t* slot_mapping = nullptr;
    int32_t* token_to_seq = nullptr;
    const int32_t* host_token_to_seq = nullptr;
    int32_t* token_positions = nullptr;
    const int32_t* cu_seqlens = nullptr;
    int num_reqs = 0;
    int num_tokens = 0;
    int max_num_blocks_per_req = 0;
    int block_size = 0;
    int num_prev_tokens = 0;
    int num_computed_tokens = 0;
    int num_prefills = 0;
    int num_prefill_tokens = 0;
    int num_decodes = 0;
    int num_decode_tokens = 0;
    bool is_prefill = false;
    float decode_threshold = 0.0f;
};

}  // namespace vm_c
