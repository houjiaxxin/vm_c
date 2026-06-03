#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include "vm_c/model/attention_backend.hpp"

namespace vm_c {

class TensorParallel;

struct BufferSizeInfo {
    size_t residual_bytes = 0;
    size_t q_bytes = 0;
    size_t k_bytes = 0;
    size_t v_bytes = 0;
    size_t attn_out_bytes = 0;
    size_t mlp_gate_bytes = 0;
    size_t mlp_up_bytes = 0;
    size_t mlp_gate_up_bytes = 0;
    size_t mlp_down_bytes = 0;

    size_t deq_q_bytes = 0;
    size_t deq_k_bytes = 0;
    size_t deq_v_bytes = 0;
    size_t deq_o_bytes = 0;

    size_t moe_router_logits_bytes = 0;
    size_t moe_topk_weights_bytes = 0;
    size_t moe_sorted_weights_bytes = 0;
    size_t moe_topk_indices_bytes = 0;
    size_t moe_token_expert_indices_bytes = 0;
    size_t moe_output_bytes = 0;
    size_t moe_expert_input_bytes = 0;
    size_t moe_expert_gate_up_bytes = 0;
    size_t moe_expert_mid_bytes = 0;
    size_t moe_expert_output_bytes = 0;
    size_t moe_weighted_output_bytes = 0;
    size_t moe_shared_gate_buf_bytes = 0;
    size_t moe_shared_up_buf_bytes = 0;
    size_t moe_shared_gate_up_bytes = 0;
    size_t moe_shared_out_bytes = 0;
    size_t moe_shared_gate_logits_bytes = 0;
    size_t moe_softmax_workspace_bytes = 0;
    size_t moe_output_fp32_bytes = 0;
    size_t moe_expert_offsets_bytes = 0;
    size_t moe_weight_offsets_bytes = 0;
    size_t moe_temp_counters_bytes = 0;

    // GDN (GatedDeltaNet) 线性注意力缓冲区大小
    size_t gdn_ssm_state_bytes = 0;
    size_t gdn_conv_state_bytes = 0;
    size_t gdn_qkvz_buf_bytes = 0;
    size_t gdn_ba_buf_bytes = 0;
    size_t gdn_q_buf_bytes = 0;
    size_t gdn_k_buf_bytes = 0;
    size_t gdn_v_buf_bytes = 0;
    size_t gdn_z_buf_bytes = 0;
    size_t gdn_gate_buf_bytes = 0;
    size_t gdn_beta_buf_bytes = 0;
    size_t gdn_alpha_buf_bytes = 0;
    size_t gdn_conv_out_buf_bytes = 0;
    size_t gdn_core_out_buf_bytes = 0;
    size_t gdn_normed_out_buf_bytes = 0;
    size_t gdn_prefill_f32_scratch_bytes = 0;

    size_t total_bytes() const {
        return residual_bytes + q_bytes + k_bytes + v_bytes + attn_out_bytes +
               mlp_gate_bytes + mlp_up_bytes + mlp_gate_up_bytes + mlp_down_bytes +
               deq_q_bytes + deq_k_bytes + deq_v_bytes + deq_o_bytes +
               moe_router_logits_bytes + moe_topk_weights_bytes + moe_sorted_weights_bytes + moe_topk_indices_bytes +
               moe_token_expert_indices_bytes + moe_output_bytes + moe_expert_input_bytes +
               moe_expert_gate_up_bytes + moe_expert_mid_bytes + moe_expert_output_bytes +
               moe_weighted_output_bytes + moe_shared_gate_buf_bytes + moe_shared_up_buf_bytes +
                moe_shared_gate_up_bytes + moe_shared_out_bytes + moe_shared_gate_logits_bytes +
               moe_softmax_workspace_bytes + moe_output_fp32_bytes + moe_expert_offsets_bytes +
                moe_weight_offsets_bytes + moe_temp_counters_bytes +
                gdn_ssm_state_bytes + gdn_conv_state_bytes +
               gdn_qkvz_buf_bytes + gdn_ba_buf_bytes +
               gdn_q_buf_bytes + gdn_k_buf_bytes + gdn_v_buf_bytes + gdn_z_buf_bytes +
               gdn_gate_buf_bytes + gdn_beta_buf_bytes + gdn_alpha_buf_bytes +
               gdn_conv_out_buf_bytes + gdn_core_out_buf_bytes + gdn_normed_out_buf_bytes
               + gdn_prefill_f32_scratch_bytes;
    }

    /// @brief 峰值显存需求 = 持久态 + max(各阶段). 利用 attention/moe/dense/gdn 不重叠的特性.
    /// 注意: gdn_ssm/conv_state 由外部单独分配（不在 ForwardContext arena 内），不包含在此。
    size_t peak_bytes() const {
        // ── 持久态（整个 forward 生命周期都存活）──
        size_t persistent = residual_bytes;

        // ── Attention 阶段 ──
        // q/k/v/attn_out 在 flash attention 中同时存活; deq buffer 顺序使用取最大
        size_t attn = q_bytes + k_bytes + v_bytes + attn_out_bytes
                    + std::max({deq_q_bytes, deq_k_bytes, deq_v_bytes, deq_o_bytes});

        // ── MoE 阶段 ──
        // 与 ForwardContext::allocate() 一致：MoE 子区按序分配全部 buffer（非子阶段 alias）
        size_t moe = moe_router_logits_bytes + moe_softmax_workspace_bytes
                   + moe_topk_weights_bytes + moe_sorted_weights_bytes
                   + moe_topk_indices_bytes + moe_token_expert_indices_bytes
                   + moe_output_bytes + moe_expert_input_bytes
                   + moe_expert_gate_up_bytes + moe_expert_mid_bytes
                   + moe_expert_output_bytes + moe_weighted_output_bytes
                   + moe_shared_gate_buf_bytes + moe_shared_up_buf_bytes
                   + moe_shared_gate_up_bytes + moe_shared_out_bytes
                   + moe_shared_gate_logits_bytes + moe_output_fp32_bytes
                   + moe_expert_offsets_bytes + moe_weight_offsets_bytes
                   + moe_temp_counters_bytes;

        // ── Dense FFN 阶段 ──
        size_t dense = mlp_gate_bytes + mlp_up_bytes + mlp_gate_up_bytes + mlp_down_bytes;

        // ── GDN 临时 buffer 阶段 ──
        size_t gdn = gdn_qkvz_buf_bytes + gdn_ba_buf_bytes + gdn_q_buf_bytes
                   + gdn_k_buf_bytes + gdn_v_buf_bytes + gdn_z_buf_bytes
                   + gdn_gate_buf_bytes + gdn_beta_buf_bytes + gdn_alpha_buf_bytes
                   + gdn_conv_out_buf_bytes + gdn_core_out_buf_bytes + gdn_normed_out_buf_bytes
                   + gdn_prefill_f32_scratch_bytes;

        return persistent + std::max({attn, moe, dense, gdn});
    }
};

struct ForwardContext {
    void* buf_residual = nullptr;
    void* buf_q = nullptr;
    void* buf_k = nullptr;
    void* buf_v = nullptr;
    void* buf_attn_out = nullptr;
    void* buf_mlp_gate = nullptr;
    void* buf_mlp_up = nullptr;
    void* buf_mlp_gate_up = nullptr;
    void* buf_mlp_down = nullptr;

    void* deq_q = nullptr;   // Q projection dequant temp buffer (IQ4_XS→FP16)
    void* deq_k = nullptr;   // K projection dequant temp buffer
    void* deq_v = nullptr;   // V projection dequant temp buffer
    void* deq_o = nullptr;   // O projection dequant temp buffer

    void* moe_router_logits = nullptr;
    void* moe_topk_weights = nullptr;
    void* moe_sorted_weights = nullptr;
    void* moe_topk_indices = nullptr;
    void* moe_token_expert_indices = nullptr;
    void* moe_output = nullptr;
    void* moe_expert_input = nullptr;
    void* moe_expert_gate_up = nullptr;
    void* moe_expert_mid = nullptr;
    void* moe_expert_output = nullptr;
    void* moe_weighted_output = nullptr;
    void* moe_shared_gate_buf = nullptr;
    void* moe_shared_up_buf = nullptr;
    void* moe_shared_gate_up = nullptr;
    void* moe_shared_out = nullptr;
    void* moe_shared_gate_logits = nullptr;
    void* moe_softmax_workspace = nullptr;
    float* moe_output_fp32 = nullptr;
    int32_t* moe_expert_offsets = nullptr;
    int32_t* moe_weight_offsets = nullptr;
    int32_t* moe_temp_counters = nullptr;

    // GDN (GatedDeltaNet) 线性注意力状态缓冲区
    // ssm_states[layer]: 每序列 GDN SSM [num_seq_slots, num_v_heads, head_v_dim, head_k_dim]
    // conv_states[layer]: 每序列卷积状态 [num_seq_slots, conv_dim, conv_kernel_size-1]
    std::vector<void*> gdn_ssm_states;    // per GDN layer
    std::vector<void*> gdn_conv_states;   // per GDN layer
    int gdn_state_slots = 0;               // max_num_seqs × (1+spec)，与 max_model_len 无关

    // GDN 临时计算缓冲区
    void* gdn_qkvz_buf = nullptr;
    void* gdn_ba_buf = nullptr;
    void* gdn_q_buf = nullptr;
    void* gdn_k_buf = nullptr;
    void* gdn_v_buf = nullptr;
    void* gdn_z_buf = nullptr;
    void* gdn_gate_buf = nullptr;
    void* gdn_beta_buf = nullptr;
    void* gdn_alpha_buf = nullptr;
    void* gdn_conv_out_buf = nullptr;
    void* gdn_core_out_buf = nullptr;
    void* gdn_normed_out_buf = nullptr;
    void* gdn_prefill_f32_scratch = nullptr;

    void* pre_norm_hidden = nullptr;
    bool capture_pre_norm = false;

    AttentionBackend* attn_backend = nullptr;
    TensorParallel* tensor_parallel = nullptr;
    cudaStream_t stream = nullptr;
    int gpu_device = -1;

    void* arena_ = nullptr;  // 单一 arena 基址（替代逐个 buffer 的 cudaMalloc）

    int max_num_tokens = 0;
    int hidden_size = 0;
    size_t activation_elem_size = 0;
    size_t residual_capacity_bytes = 0;

    size_t activation_bytes(int num_tokens) const;
    void clear_residual(int num_tokens, cudaStream_t stream) const;
    void copy_pre_norm_hidden(void* dst, const void* src, int num_tokens,
                              cudaStream_t stream) const;

    void allocate(const BufferSizeInfo& sizes, int max_num_tokens, int hidden_size,
                  int gpu_device_);
    void free();
};

}
