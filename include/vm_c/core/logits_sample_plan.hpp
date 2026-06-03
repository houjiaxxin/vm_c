#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "vm_c/core/request.hpp"

namespace vm_c {

// 对齐 vllm logits_indices / cu_num_logits 语义：
// - hidden_batch_index：从 d_hidden_states[total_tokens, hidden] 取 lm_head 输入行
// - logits_row：紧凑 d_logits[num_logits, vocab] 中的行号（gather+matmul 后恒为 0..num_logits-1）
struct LogitsSampleSlot {
    int req_index = -1;
    int hidden_batch_index = -1;
    int logits_row = -1;
};

struct LogitsSamplePlan {
    std::vector<LogitsSampleSlot> slots;

    int num_logits() const { return static_cast<int>(slots.size()); }
    bool empty() const { return slots.empty(); }

    // 紧凑 logits 布局下采样 kernel 用 seq_idx 作行号，无需 device logit_offsets
    static constexpr bool kCompactLogitsLayout = true;
};

LogitsSamplePlan build_logits_sample_plan(
    const std::vector<RequestPtr>& batch_reqs,
    const std::unordered_map<std::string, int>& num_scheduled_tokens);

void validate_logits_sample_plan(
    const LogitsSamplePlan& plan,
    int total_batch_tokens,
    int max_logits_rows);

void gather_hidden_for_logits_plan(
    const LogitsSamplePlan& plan,
    const void* d_hidden_states,
    void* d_gather,
    int hidden_size,
    std::size_t hidden_elem_size,
    cudaStream_t stream);

}  // namespace vm_c
