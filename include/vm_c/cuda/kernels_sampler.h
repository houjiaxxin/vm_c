#pragma once

#include "vm_c_tensor.h"
#include <cstdint>

namespace vm_c {

void apply_penalties(
    void* logits,
    const void* prompt_mask,
    const void* output_bin_counts,
    const void* repetition_penalties,
    const void* frequency_penalties,
    const void* presence_penalties,
    int num_seqs, int vocab_size,
    ScalarType dtype,
    cudaStream_t stream = 0);

void gpu_greedy_sampling(
    const void* logits,
    const int* logit_offsets,
    int32_t* output_ids,
    float* output_logprobs,
    int batch_size, int vocab_size,
    ScalarType dtype,
    cudaStream_t stream = 0);

void gpu_temperature_sampling(
    const void* logits,
    const int* logit_offsets,
    int32_t* output_ids,
    float* output_logprobs,
    const float* temperatures,
    int batch_size, int vocab_size,
    ScalarType dtype,
    unsigned long long seed,
    cudaStream_t stream = 0);

void gpu_filtered_sampling(
    const void* logits,
    const int* logit_offsets,
    int32_t* output_ids,
    float* output_logprobs,
    const float* temperatures,
    const int* top_k,
    const float* top_p,
    const float* min_p,
    int batch_size, int vocab_size,
    ScalarType dtype,
    unsigned long long seed,
    cudaStream_t stream = 0);

void top_k_per_row_prefill(
    const void* logits,
    const void* row_starts, const void* row_ends,
    void* indices,
    int64_t num_rows, int64_t stride0, int64_t stride1,
    int64_t top_k,
    cudaStream_t stream = 0);

void top_k_per_row_decode(
    const void* logits, int64_t next_n,
    const void* seq_lens,
    void* indices,
    int64_t num_rows, int64_t stride0, int64_t stride1,
    int64_t top_k,
    cudaStream_t stream = 0);

void persistent_topk(
    const void* logits, const void* lengths,
    void* output, void* workspace,
    int64_t k, int64_t max_seq_len,
    int64_t num_elements,
    cudaStream_t stream = 0);

}
