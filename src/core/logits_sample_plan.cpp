#include "vm_c/core/logits_sample_plan.hpp"
#include "vm_c/cuda/gpu_arch.hpp"

namespace vm_c {

LogitsSamplePlan build_logits_sample_plan(
    const std::vector<RequestPtr>& batch_reqs,
    const std::unordered_map<std::string, int>& num_scheduled_tokens) {
    LogitsSamplePlan plan;
    int batch_offset = 0;
    const int num_reqs = static_cast<int>(batch_reqs.size());

    for (int ri = 0; ri < num_reqs; ++ri) {
        const auto& req = batch_reqs[ri];
        auto it = num_scheduled_tokens.find(req->request_id);
        const int num_sched = (it != num_scheduled_tokens.end()) ? it->second : 1;
        if (num_sched <= 0) {
            throw std::runtime_error(
                "build_logits_sample_plan: non-positive num_sched for request " +
                req->request_id);
        }

        const bool prefill_done =
            !req->is_prefill() ||
            (req->num_computed_tokens + num_sched >= req->num_prompt_tokens());

        if (prefill_done) {
            LogitsSampleSlot slot;
            slot.req_index = ri;
            slot.hidden_batch_index = batch_offset + num_sched - 1;
            slot.logits_row = plan.num_logits();
            plan.slots.push_back(slot);
        }
        batch_offset += num_sched;
    }

    return plan;
}

void validate_logits_sample_plan(
    const LogitsSamplePlan& plan,
    int total_batch_tokens,
    int max_logits_rows) {
    if (plan.empty()) {
        return;
    }
    if (total_batch_tokens <= 0) {
        throw std::runtime_error(
            "validate_logits_sample_plan: total_batch_tokens must be positive");
    }
    if (max_logits_rows <= 0) {
        throw std::runtime_error(
            "validate_logits_sample_plan: max_logits_rows must be positive");
    }
    if (plan.num_logits() > max_logits_rows) {
        throw std::runtime_error(
            "validate_logits_sample_plan: num_logits=" +
            std::to_string(plan.num_logits()) +
            " exceeds max_logits_rows=" + std::to_string(max_logits_rows));
    }

    for (int si = 0; si < plan.num_logits(); ++si) {
        const auto& slot = plan.slots[static_cast<std::size_t>(si)];
        if (slot.req_index < 0) {
            throw std::runtime_error(
                "validate_logits_sample_plan: invalid req_index at slot " +
                std::to_string(si));
        }
        if (slot.hidden_batch_index < 0 ||
            slot.hidden_batch_index >= total_batch_tokens) {
            throw std::runtime_error(
                "validate_logits_sample_plan: hidden_batch_index=" +
                std::to_string(slot.hidden_batch_index) +
                " out of range [0, " + std::to_string(total_batch_tokens) + ")");
        }
        if (slot.logits_row != si) {
            throw std::runtime_error(
                "validate_logits_sample_plan: compact layout requires "
                "logits_row==slot index, got logits_row=" +
                std::to_string(slot.logits_row) + " at slot " + std::to_string(si));
        }
    }
}

void gather_hidden_for_logits_plan(
    const LogitsSamplePlan& plan,
    const void* d_hidden_states,
    void* d_gather,
    int hidden_size,
    const std::size_t hidden_elem_size,
    const cudaStream_t stream) {
    const int num_logits = plan.num_logits();
    if (num_logits <= 0) {
        return;
    }
    const auto row_bytes = static_cast<std::size_t>(hidden_size) * hidden_elem_size;
    const auto* src_base = static_cast<const char*>(d_hidden_states);
    auto* dst_base = static_cast<char*>(d_gather);

    for (int si = 0; si < num_logits; ++si) {
        const int hidden_idx = plan.slots[static_cast<std::size_t>(si)].hidden_batch_index;
        CUDA_CHECK(cudaMemcpyAsync(
            dst_base + static_cast<std::size_t>(si) * row_bytes,
            src_base + static_cast<std::size_t>(hidden_idx) * row_bytes,
            row_bytes,
            cudaMemcpyDeviceToDevice,
            stream));
    }
}

}  // namespace vm_c
