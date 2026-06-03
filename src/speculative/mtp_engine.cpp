#include "vm_c/speculative/speculative_engine.hpp"
#include "vm_c/distributed/tp_runtime.hpp"
#include "vm_c/cuda/gpu_arch.hpp"
#include "vm_c/cuda/kernels_sampler.h"
#include "vm_c/cuda/convert_kernels.h"
#include "vm_c/core/ggml_dequant.hpp"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <cmath>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace vm_c {

SpeculativeEngine::SpeculativeEngine(const ModelConfig& config,
                                     int gpu_device, int tp_rank, int tp_size)
    : config_(config),
      gpu_device_(gpu_device), tp_rank_(tp_rank), tp_size_(tp_size) {

    mtp_predict_layers_ = config_.mtp_predict_layers;
    if (mtp_predict_layers_ > 0 && config_.mtp_spec_width <= 0) {
        throw std::runtime_error(
            "MTP enabled but mtp_spec_width <= 0; set --spec-draft-n-max");
    }
    spec_width_ = config_.mtp_spec_width;
    spec_n_min_ = config_.mtp_spec_n_min;
    spec_p_min_ = config_.mtp_spec_p_min;
    hidden_size_ = config_.hidden_size;
    vocab_size_ = config_.vocab_size;

    if (mtp_predict_layers_ > 0) {
        spdlog::info(
            "SpeculativeEngine: MTP enabled, predict_layers={}, spec_width={}, n_min={}, p_min={}",
            mtp_predict_layers_, spec_width_, spec_n_min_, spec_p_min_);
    }
}

SpeculativeEngine::~SpeculativeEngine() {
    CudaDeviceGuard guard(gpu_device_);
    if (hidden_chain_buf_) { cudaFree(hidden_chain_buf_); hidden_chain_buf_ = nullptr; }
    if (logits_buf_) { cudaFree(logits_buf_); logits_buf_ = nullptr; }
    // host_pending_h_ 和 host_verify_h_ 是 std::vector，自动释放
    host_pending_h_.clear();
    host_verify_h_.clear();
    host_verify_h_rows_.clear();
    if (d_draft_sampled_id_) {
        cudaFree(d_draft_sampled_id_);
        d_draft_sampled_id_ = nullptr;
    }
    if (d_draft_sampled_lp_) {
        cudaFree(d_draft_sampled_lp_);
        d_draft_sampled_lp_ = nullptr;
    }
    // 释放真正 pinned (page-locked) host memory
    if (pinned_draft_id_) {
        cudaFreeHost(pinned_draft_id_);
        pinned_draft_id_ = nullptr;
    }
    if (pinned_draft_lp_) {
        cudaFreeHost(pinned_draft_lp_);
        pinned_draft_lp_ = nullptr;
    }
}

void SpeculativeEngine::allocate_buffers(int max_batch_tokens) {
    if (mtp_predict_layers_ <= 0) return;

    CudaDeviceGuard guard(gpu_device_);
    const int chain_slots = spec_width_ + 1;
    max_spec_tokens_ = std::max({1, max_batch_tokens, chain_slots});
    const size_t row_bytes = static_cast<size_t>(hidden_size_) * sizeof(float);
    const size_t chain_bytes = static_cast<size_t>(chain_slots) * row_bytes;
    const size_t logits_bytes =
        static_cast<size_t>(spec_width_) * static_cast<size_t>(vocab_size_) * sizeof(float);

    if (hidden_chain_buf_) cudaFree(hidden_chain_buf_);
    CUDA_CHECK(cudaMalloc(&hidden_chain_buf_, chain_bytes));

    if (logits_buf_) cudaFree(logits_buf_);
    CUDA_CHECK(cudaMalloc(&logits_buf_, logits_bytes));

    // Host buffers for pending hidden state and verify embeddings
    // (对齐 llama.cpp speculative.cpp 的 pending_h 和 verify_h)
    host_pending_h_.resize(1);
    host_pending_h_[0].resize(hidden_size_, 0.0f);
    
    host_verify_h_.resize(1);
    host_verify_h_[0].resize(static_cast<size_t>(max_spec_tokens_) * hidden_size_, 0.0f);
    host_verify_h_rows_.resize(1, 0);

    // GPU staging buffers for draft greedy sampling
    if (d_draft_sampled_id_) cudaFree(d_draft_sampled_id_);
    CUDA_CHECK(cudaMalloc(&d_draft_sampled_id_, sizeof(int32_t)));
    if (d_draft_sampled_lp_) cudaFree(d_draft_sampled_lp_);
    CUDA_CHECK(cudaMalloc(&d_draft_sampled_lp_, sizeof(float)));

    // 真正 pinned (page-locked) host memory，避免 cudaMemcpyAsync
    // 在普通堆内存上被 runtime 隐式同步
    if (pinned_draft_id_) {
        cudaFreeHost(pinned_draft_id_);
        pinned_draft_id_ = nullptr;
    }
    CUDA_CHECK(cudaMallocHost(&pinned_draft_id_, sizeof(int32_t)));
    if (pinned_draft_lp_) {
        cudaFreeHost(pinned_draft_lp_);
        pinned_draft_lp_ = nullptr;
    }
    CUDA_CHECK(cudaMallocHost(&pinned_draft_lp_, sizeof(float)));

    spdlog::info(
        "SpeculativeEngine: buffers allocated chain={:.1f}MB logits={:.1f}MB (float embd)",
        chain_bytes / (1024.0 * 1024.0),
        logits_bytes / (1024.0 * 1024.0));
}

void SpeculativeEngine::reset_pending(int seq_id, cudaStream_t stream) {
    if (seq_id < 0 || seq_id >= static_cast<int>(host_pending_h_.size())) {
        return;
    }
    // Host buffer: 直接清零，无需 stream
    std::fill(host_pending_h_[seq_id].begin(), host_pending_h_[seq_id].end(), 0.0f);
}

const float* SpeculativeEngine::pending_h(int seq_id) const {
    if (seq_id < 0 || seq_id >= static_cast<int>(host_pending_h_.size())) return nullptr;
    return host_pending_h_[seq_id].data();
}

void SpeculativeEngine::begin_draft_kv_checkpoint(int64_t n_past) {
    draft_mtp_n_past_ = n_past;
}

void SpeculativeEngine::rollback_draft_kv(TPRuntime* tp_runtime) {
    if (!tp_runtime) {
        throw std::runtime_error("SpeculativeEngine::rollback_draft_kv: TPRuntime required");
    }
    tp_runtime->llama_mtp_seq_rollback(0, draft_mtp_n_past_);
}

void SpeculativeEngine::accept(int seq_id, int n_draft_accepted, cudaStream_t stream) {
    if (n_draft_accepted < 0) return;
    if (seq_id < 0 || seq_id >= static_cast<int>(host_pending_h_.size())) return;
    if (host_verify_h_rows_[seq_id] <= 0) return;
    
    // Host buffer: 直接 memcpy，无需 stream
    const int i_h = std::min(n_draft_accepted, host_verify_h_rows_[seq_id] - 1);
    const float* src = host_verify_h_[seq_id].data()
        + static_cast<size_t>(i_h) * hidden_size_;
    std::memcpy(host_pending_h_[seq_id].data(), src,
                static_cast<size_t>(hidden_size_) * sizeof(float));

    // MTP 熔断器: 跟踪连续 0-accept 步数 (MTP 权重错位/采样不一致时, accept 持续为 0)
    if (n_draft_accepted == 0) {
        ++consecutive_zero_accepts_;
    } else {
        consecutive_zero_accepts_ = 0;
    }
}

void SpeculativeEngine::trim_draft_kv_after_accept(
    int64_t n_past, int num_committed, TPRuntime* tp_runtime) {
    if (!tp_runtime || num_committed <= 0) {
        return;
    }
    tp_runtime->llama_mtp_seq_rollback(0, n_past + num_committed);
}

void SpeculativeEngine::update_pending_h(
    int n_tokens, const float* host_tgt_pre_norm) {
    if (n_tokens <= 0 || !host_tgt_pre_norm) {
        throw std::runtime_error("SpeculativeEngine::update_pending_h: invalid args");
    }
    if (host_pending_h_.empty()) {
        throw std::runtime_error("SpeculativeEngine::update_pending_h: pending_h not allocated");
    }
    // 对齐 llama.cpp common/speculative.cpp:632-633
    //   pending_h[seq_id] = verify_h[seq_id][n_rows - 1] = h_tgt[seq_id][batch_in.n_tokens - 1]
    const int seq_id = 0;
    const float* last_row = host_tgt_pre_norm
        + static_cast<size_t>(n_tokens - 1) * static_cast<size_t>(hidden_size_);
    spdlog::info("[MTP-PENDING-WRITE] n_tokens={} src_last_row first8=[{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}] src_first_row first8=[{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}]",
        n_tokens,
        last_row[0], last_row[1], last_row[2], last_row[3], last_row[4], last_row[5], last_row[6], last_row[7],
        host_tgt_pre_norm[0], host_tgt_pre_norm[1], host_tgt_pre_norm[2], host_tgt_pre_norm[3],
        host_tgt_pre_norm[4], host_tgt_pre_norm[5], host_tgt_pre_norm[6], host_tgt_pre_norm[7]);
    std::memcpy(host_pending_h_[seq_id].data(), last_row,
                static_cast<size_t>(hidden_size_) * sizeof(float));
}

void SpeculativeEngine::update_verify_h(
    int n_tokens, const float* host_tgt_pre_norm) {
    if (n_tokens <= 0 || !host_tgt_pre_norm) {
        throw std::runtime_error("SpeculativeEngine::update_verify_h: invalid args");
    }
    if (n_tokens > max_spec_tokens_) {
        throw std::runtime_error(
            "SpeculativeEngine::update_verify_h: n_tokens=" + std::to_string(n_tokens)
            + " exceeds max batch " + std::to_string(max_spec_tokens_));
    }
    if (host_verify_h_.empty()) {
        throw std::runtime_error("SpeculativeEngine::update_verify_h: verify_h not allocated");
    }
    // 对齐 llama.cpp common/speculative.cpp:627-630
    //   verify_h[seq_id][i] = h_tgt[seq_id][i_batch_beg[seq_id] + i]  for i in [0, n_rows)
    // vm_c 是单序列 (seq_id=0), i_batch_beg=0, 所以直接 memcpy n_tokens 行即可
    const int seq_id = 0;
    const size_t verify_size = static_cast<size_t>(n_tokens) * static_cast<size_t>(hidden_size_);
    if (host_verify_h_[seq_id].size() < verify_size) {
        host_verify_h_[seq_id].resize(verify_size);
    }
    std::memcpy(host_verify_h_[seq_id].data(), host_tgt_pre_norm,
                verify_size * sizeof(float));
    host_verify_h_rows_[seq_id] = n_tokens;
}

SpeculativeEngine::DraftOutput SpeculativeEngine::draft(
    const float* h_pending_h, int32_t prev_token, int64_t start_position,
    TPRuntime* tp_runtime, cudaStream_t stream) {

    DraftOutput out;
    if (mtp_predict_layers_ <= 0 || !hidden_chain_buf_ || !logits_buf_ || !h_pending_h || !tp_runtime) {
        return out;
    }
    if (!stream) {
        throw std::runtime_error("SpeculativeEngine::draft: stream is null");
    }

    // MTP 熔断: 连续 kCircuitBreakerThreshold 次 0-accept, 本步跳过 draft 回退标准 decode
    if (consecutive_zero_accepts_ >= kCircuitBreakerThreshold) {
        return out;  // empty DraftOutput: engine.cpp 走 standard decode 路径
    }

    CudaDeviceGuard guard(gpu_device_);
    const size_t row_bytes = static_cast<size_t>(hidden_size_) * sizeof(float);

    // Host buffer for current hidden state (对齐 llama.cpp speculative.cpp)
    std::vector<float> h_cur(hidden_size_);
    std::memcpy(h_cur.data(), h_pending_h, row_bytes);

    spdlog::info("[MTP-DRAFT-IN] prev_token={} start_pos={} h_pending_h first8=[{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}] h_cur first8=[{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}]",
        prev_token, (long long)start_position,
        h_pending_h[0], h_pending_h[1], h_pending_h[2], h_pending_h[3],
        h_pending_h[4], h_pending_h[5], h_pending_h[6], h_pending_h[7],
        h_cur[0], h_cur[1], h_cur[2], h_cur[3],
        h_cur[4], h_cur[5], h_cur[6], h_cur[7]);

    int32_t cur_token = prev_token;
    int64_t cur_pos = start_position;

    for (int i = 0; i < spec_width_; ++i) {
        int step = i;
        float* out_hs = reinterpret_cast<float*>(hidden_chain_buf_)
            + static_cast<size_t>(i) * static_cast<size_t>(hidden_size_);
        float* logits_out = static_cast<float*>(logits_buf_)
            + static_cast<size_t>(i) * static_cast<size_t>(vocab_size_);

        tp_runtime->forward_llama_mtp_draft(
            h_cur.data(), cur_token, cur_pos,
            out_hs, logits_out, stream);

        // GPU 端贪心采样：直接在 GPU 上 argmax，仅 D2H 传输 4+4 bytes（token id + logprob）
        // 注：官方 llama.cpp/common/speculative.cpp:215-220 在 MTP draft 时使用
        // top_k=10 + multinomial 采样（COMMON_SAMPLER_TYPE_TOP_K），本实现保留
        // 纯 greedy（argmax）。两者对比：greedy 接受率更高（与主模型 top-1
        // 一致），但草稿多样性低；top_k 接受率略低但探索性更好。在接受率
        // 优先的 workload 下，本设计是合理选择。
        gpu_greedy_sampling(logits_out, nullptr,
                            d_draft_sampled_id_, d_draft_sampled_lp_,
                            1, vocab_size_, ScalarType::FLOAT32, stream);

        CUDA_CHECK(cudaMemcpyAsync(pinned_draft_id_, d_draft_sampled_id_,
                                   sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(pinned_draft_lp_, d_draft_sampled_lp_,
                                   sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (spec_p_min_ > 0.0f && *pinned_draft_lp_ < spec_p_min_) {
            break;
        }

        out.draft_tokens.push_back(*pinned_draft_id_);

        cur_token = *pinned_draft_id_;
        ++cur_pos;
        
        // Copy output hidden state to host for next iteration
        CUDA_CHECK(cudaMemcpyAsync(h_cur.data(), out_hs, row_bytes,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (static_cast<int>(out.draft_tokens.size()) < spec_n_min_) {
        out.draft_tokens.clear();
    }

    out.success = !out.draft_tokens.empty();
    return out;
}

namespace {

bool sampling_uses_penalties(const SamplingParams& p) {
    return p.repetition_penalty != 1.0f || p.frequency_penalty != 0.0f || p.presence_penalty != 0.0f;
}

}  // namespace

SpeculativeEngine::VerifyOutput SpeculativeEngine::sample_and_accept(
    const std::vector<int32_t>& draft_tokens,
    const float* d_verify_logits,  // GPU pointer
    int num_logits_rows,
    int vocab_size,
    const SamplingParams& sampling,
    const std::vector<int32_t>& prompt_tokens,
    const std::vector<int32_t>& prior_output_tokens,
    void* d_single_logits,
    void* d_output_id,
    void* d_output_logprob,
    void* d_prompt_mask,
    void* d_output_bin_counts,
    void* d_rep_penalties,
    void* d_freq_penalties,
    void* d_pres_penalties,
    cudaStream_t stream) {

    VerifyOutput result;
    if (draft_tokens.empty() || !d_verify_logits || num_logits_rows <= 0 || vocab_size <= 0) {
        return result;
    }
    if (!d_single_logits || !d_output_id || !d_output_logprob || !stream) {
        throw std::runtime_error("SpeculativeEngine::sample_and_accept: missing device buffers");
    }

    const int n_draft = static_cast<int>(draft_tokens.size());
    if (num_logits_rows < n_draft + 1) {
        throw std::runtime_error(
            "SpeculativeEngine::sample_and_accept: need at least n_draft+1 logits rows");
    }

    CudaDeviceGuard guard(gpu_device_);
    auto* d_logits = static_cast<float*>(d_single_logits);
    auto* d_tok = static_cast<int32_t*>(d_output_id);
    auto* d_lp = static_cast<float*>(d_output_logprob);

    const bool use_penalties = sampling_uses_penalties(sampling);

    if (use_penalties) {
        std::vector<uint8_t> host_prompt_mask(static_cast<size_t>(vocab_size), 0);
        std::vector<int32_t> host_output_counts(static_cast<size_t>(vocab_size), 0);
        for (int32_t tid : prompt_tokens) {
            if (tid >= 0 && tid < vocab_size) {
                host_prompt_mask[static_cast<size_t>(tid)] = 1;
            }
        }
        for (int32_t tid : prior_output_tokens) {
            if (tid >= 0 && tid < vocab_size) {
                host_output_counts[static_cast<size_t>(tid)]++;
            }
        }
        const float rep = sampling.repetition_penalty;
        const float freq = sampling.frequency_penalty;
        const float pres = sampling.presence_penalty;

        CUDA_CHECK(cudaMemcpyAsync(d_prompt_mask, host_prompt_mask.data(),
                                   vocab_size * sizeof(uint8_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_output_bin_counts, host_output_counts.data(),
                                   vocab_size * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_rep_penalties, &rep, sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_freq_penalties, &freq, sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pres_penalties, &pres, sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
    }

    // ── 逐行验证 + 采样（直接在 GPU 上读取 logits，避免 D2H→H2D 往返） ──
    std::vector<int32_t> accepted_context = prior_output_tokens;

    auto sample_one_row = [&](int row_idx, int32_t& out_id, float& out_lp) {
        // 直接从 GPU 上的 verify logits 读取（无需 H2D 拷贝）
        const float* src_row = d_verify_logits
            + static_cast<size_t>(row_idx) * static_cast<size_t>(vocab_size);

        // 将该行 logits 复制到 staging buffer（用于 penalties 修改和采样）
        CUDA_CHECK(cudaMemcpyAsync(d_logits, src_row,
                                   static_cast<size_t>(vocab_size) * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));

        if (use_penalties) {
            apply_penalties(d_logits, d_prompt_mask, d_output_bin_counts,
                            d_rep_penalties, d_freq_penalties, d_pres_penalties,
                            1, vocab_size, ScalarType::FLOAT32, stream);
        }

        gpu_greedy_sampling(d_logits, nullptr, d_tok, d_lp,
                            1, vocab_size, ScalarType::FLOAT32, stream);

        CUDA_CHECK(cudaMemcpyAsync(&out_id, d_tok, sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&out_lp, d_lp, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    int n_draft_accepted = 0;
    for (int i = 0; i < n_draft; ++i) {
        int32_t sampled = 0;
        float lp = 0.0f;
        // logits row i = predict(batch[i]) = 用于验证 draft_tokens[i]
        // row 0 = predict(id_last) → draft[0], row 1 = predict(draft[0]) → draft[1], 以此类推
        sample_one_row(i, sampled, lp);

        result.accepted_tokens.push_back(sampled);
        result.accepted_logprobs.push_back(lp);
        accepted_context.push_back(sampled);

        if (sampled != draft_tokens[static_cast<size_t>(i)]) {
            result.n_draft_accepted = n_draft_accepted;
            result.all_drafts_accepted = false;
            return result;
        }
        ++n_draft_accepted;
    }

    // bonus token: logits row n_draft = predict(draft[n_draft-1])
    int32_t bonus = 0;
    float bonus_lp = 0.0f;
    sample_one_row(n_draft, bonus, bonus_lp);

    result.accepted_tokens.push_back(bonus);
    result.accepted_logprobs.push_back(bonus_lp);
    result.n_draft_accepted = n_draft_accepted;
    result.all_drafts_accepted = true;
    result.has_bonus = true;
    return result;
}

}  // namespace vm_c
