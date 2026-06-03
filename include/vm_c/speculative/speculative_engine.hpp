#pragma once

#include <cstdint>
#include <vector>
#include <memory>

#include "vm_c/core/config.hpp"
#include "vm_c/core/request.hpp"

#include <cuda_runtime.h>

namespace vm_c {

class KVCacheManager;
class TPRuntime;

/// MTP 推测解码（libllama ctx_mtp + 主 ctx verify，对齐 llama.cpp speculative）
class SpeculativeEngine {
public:
    SpeculativeEngine(const ModelConfig& config,
                      int gpu_device = 0, int tp_rank = 0, int tp_size = 1);
    ~SpeculativeEngine();

    SpeculativeEngine(const SpeculativeEngine&) = delete;
    SpeculativeEngine& operator=(const SpeculativeEngine&) = delete;

    bool enabled() const {
        return config_.spec_method == "mtp" && config_.mtp_predict_layers > 0;
    }
    int mtp_predict_layers() const { return mtp_predict_layers_; }
    int spec_width() const { return spec_width_; }

    void allocate_buffers(int max_batch_tokens);

    void reset_pending(int seq_id, cudaStream_t stream);

    struct DraftOutput {
        std::vector<int32_t> draft_tokens;
        bool success = false;
    };

    /// draft 前记录 ctx_mtp 回滚点（对齐 llama ctx_dft n_past）
    void begin_draft_kv_checkpoint(int64_t n_past);

    /// draft 后回滚 ctx_mtp KV
    void rollback_draft_kv(TPRuntime* tp_runtime);

    /// 从主 ctx forward 后的 h_tgt 更新 pending_h。
    /// 对齐 llama.cpp common/speculative.cpp:632-633：
    ///   pending_h[seq_id] = h_tgt[seq_id][n_tokens-1]
    /// 必须在 MTP draft 之前调用，draft 的首次 MTP forward 用 pending_h 作为 embd。
    /// @param n_tokens 主 ctx forward 处理的 token 数（prefill: >1, decode: 1）
    /// @param host_tgt_pre_norm host pointer to pre-norm embeddings (n_tokens × hidden_size, float32)
    void update_pending_h(int n_tokens, const float* host_tgt_pre_norm);

    /// 从主 ctx verify forward 后的 h_tgt 更新 verify_h 整段。
    /// 对齐 llama.cpp common/speculative.cpp:627-630：
    ///   verify_h[seq_id][i] = h_tgt[seq_id][i]  for i in [0, n_tokens)
    /// accept() 用 verify_h 决定下次 MTP draft 起始的 h。
    /// @param n_tokens verify 处理的 token 数（= 1 + n_draft）
    /// @param host_tgt_pre_norm host pointer to pre-norm embeddings (n_tokens × hidden_size, float32)
    void update_verify_h(int n_tokens, const float* host_tgt_pre_norm);

    /// MTP draft（pending_h float + ctx_mtp 单步 decode）
    /// @param h_pending_h host pointer to pending hidden state (hidden_size, float32)
    DraftOutput draft(const float* h_pending_h, int32_t prev_token, int64_t start_position,
                      TPRuntime* tp_runtime, cudaStream_t stream);

    struct VerifyOutput {
        int n_draft_accepted = 0;
        std::vector<int32_t> accepted_tokens;
        std::vector<float> accepted_logprobs;
        bool all_drafts_accepted = false;
        bool has_bonus = false;
    };

    /// MTP verify + sample：直接在 GPU 上读取 logits 进行采样，避免 D2H→H2D 往返
    /// @param d_verify_logits GPU 上的 verify logits（n_logits_rows × vocab_size，float32）
    VerifyOutput sample_and_accept(
        const std::vector<int32_t>& draft_tokens,
        const float* d_verify_logits,  // GPU pointer
        int num_logits_rows,
        int vocab_size,
        const SamplingParams& sampling,
        const std::vector<int32_t>& prompt_tokens,
        const std::vector<int32_t>& prior_output_tokens,
        void* d_single_logits,  // staging buffer for sampling
        void* d_output_id,
        void* d_output_logprob,
        void* d_prompt_mask,
        void* d_output_bin_counts,
        void* d_rep_penalties,
        void* d_freq_penalties,
        void* d_pres_penalties,
        cudaStream_t stream);

    void accept(int seq_id, int n_draft_accepted, cudaStream_t stream);

    void trim_draft_kv_after_accept(int64_t n_past, int num_committed,
                                    TPRuntime* tp_runtime);

    /// 返回 host pointer to pending hidden state for seq_id
    const float* pending_h(int seq_id = 0) const;

private:
    ModelConfig config_;
    int gpu_device_ = 0;
    int tp_rank_ = 0;
    int tp_size_ = 1;

    int mtp_predict_layers_ = 0;
    int spec_width_ = 0;
    int spec_n_min_ = 0;
    float spec_p_min_ = 0.0f;
    int hidden_size_ = 0;
    int vocab_size_ = 0;
    int max_spec_tokens_ = 0;

    void* hidden_chain_buf_ = nullptr;
    void* logits_buf_ = nullptr;
    
    // Host buffers for pending hidden state and verify embeddings
    // (对齐 llama.cpp speculative.cpp 的 pending_h 和 verify_h)
    std::vector<std::vector<float>> host_pending_h_;  // [n_seq][hidden_size]
    std::vector<std::vector<float>> host_verify_h_;   // [n_seq][n_tokens * hidden_size]
    std::vector<int> host_verify_h_rows_;             // [n_seq] — verify_h 当前有效行数 (accept() 用作上界)

    // GPU staging buffers for draft greedy sampling (avoid D2H logits transfer per step)
    int32_t* d_draft_sampled_id_ = nullptr;
    float* d_draft_sampled_lp_ = nullptr;

    // Pinned (page-locked) host memory for draft sample result D2H.
    // 只有 4+4 字节/步骤，避免传输整个 vocab 维度的 logits。
    // 必须用 cudaMallocHost/cudaHostAlloc 分配，否则 cudaMemcpyAsync
    // 在不可锁页内存上会被 runtime 隐式同步，反而比同步 cudaMemcpy 更慢。
    int32_t* pinned_draft_id_ = nullptr;
    float*   pinned_draft_lp_  = nullptr;

    int64_t draft_mtp_n_past_ = 0;

    // MTP 熔断: 连续 N 次 0 accept 就跳过 draft (回退标准 decode)。
    // 当 MTP 模型权重加载异常 / shifted MTP 假设不成立时, accept rate 持续为 0,
    // 每步跑 5 个 MTP forward + 1 主 verify 全是浪费。熔断器在第 2 次 0-accept 后
    // 让后续 step 走标准 decode, 32-token 请求耗时从 O(32 * 7fwd) 降到 O(32 * 1fwd)。
    int consecutive_zero_accepts_ = 0;
    static constexpr int kCircuitBreakerThreshold = 2;
};

}  // namespace vm_c
