#pragma once

#include "vm_c_tensor.h"
#include <cstdint>
#include <optional>

namespace vm_c {

struct MoEParams {
    int num_tokens = 0;
    int num_experts = 0;
    int top_k = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t block_size = 0;
    bool renormalize = false;
};

void topk_softmax(
    void* topk_weights, void* topk_indices, void* token_expert_indices,
    const void* gating_output,
    int num_tokens, int num_experts, int top_k,
    bool renormalize,
    const void* bias,
    ScalarType dtype,
    cudaStream_t stream = 0,
    void* softmax_workspace = nullptr);

void topk_softplus_sqrt(
    void* topk_weights, void* topk_indices, void* token_expert_indices,
    const void* gating_output,
    int num_tokens, int num_experts, int top_k,
    bool renormalize, double routed_scaling_factor,
    const void* correction_bias,
    const void* input_ids, const void* tid2eid,
    ScalarType dtype,
    cudaStream_t stream = 0);

void moe_align_block_size(
    const int32_t* topk_ids,
    int num_tokens, int num_experts, int top_k,
    int32_t* sorted_token_ids,
    int32_t* expert_ids,
    int32_t* total_tokens,
    cudaStream_t stream = 0,
    int32_t* temp_counters = nullptr);

void moe_sum(
    const void* input, void* output,
    int num_tokens, int num_experts, int top_k, int64_t hidden_size,
    ScalarType dtype,
    cudaStream_t stream = 0);

void grouped_topk(
    void* topk_weights_out, void* topk_indices_out,
    const void* scores, ScalarType score_dtype,
    int num_tokens, int num_experts,
    int n_group, int topk_group, int top_k,
    bool renormalize, double routed_scaling_factor,
    const void* bias, int scoring_func,
    cudaStream_t stream = 0);

void dsv3_router_gemm(
    void* output, const void* mat_a, const void* mat_b,
    int num_tokens, int num_experts, int hidden_dim,
    ScalarType dtype,
    cudaStream_t stream = 0);

void moe_gather_tokens(
    void* expert_input, const void* hs_ptr,
    const int32_t* sorted_token_ids,
    int total_tokens, int hidden_size,
    cudaStream_t stream = 0);

void moe_gather_weights(
    float* sorted_weights, const float* topk_weights,
    const int32_t* topk_ids, const int32_t* expert_offsets,
    int num_entries, int top_k, int num_experts,
    cudaStream_t stream = 0);

void moe_scale_add(
    void* moe_output, const void* expert_out,
    const float* weights, const int32_t* token_ids,
    int total_tokens, int num_tokens, int hidden_size,
    float* moe_output_fp32_workspace,
    ScalarType dtype, cudaStream_t stream = 0);

void fused_moe_experts(
    const void* hidden_states,
    const void* w1_gate,
    const void* w1_up,
    const void* w2_down,
    void* output,
    const int32_t* sorted_token_ids,
    const int32_t* expert_offsets,
    const float* sorted_weights,
    int num_tokens,
    int num_experts,
    int top_k,
    int hidden_size,
    int intermediate_size,
    ScalarType dtype,
    cudaStream_t stream = 0,
    void* workspace = nullptr);

}


