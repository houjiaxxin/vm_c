#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <optional>
#include <functional>

#include "vm_c/core/tensor.hpp"

namespace vm_c {

enum class RequestStatus {
    WAITING,
    RUNNING,
    FINISHED,
    PREEMPTED,
};

enum class FinishReason {
    STOP,
    LENGTH,
    ERROR,
    ABORTED,
};

struct SamplingParams {
    int max_tokens = 16;
    int top_k = 40;              // 对齐 llama.cpp 默认值
    float top_p = 0.95f;
    float temperature = 0.80f;
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    std::vector<int> stop_token_ids;
    std::vector<std::string> stop_strings;
    bool ignore_eos = false;
    int min_tokens = 0;
    float min_p = 0.0f;
    bool enable_thinking = true;
};

struct Request {
    std::string request_id;
    std::vector<int32_t> prompt_token_ids;
    SamplingParams sampling_params;
    std::string model_name;

    RequestStatus status = RequestStatus::WAITING;
    std::optional<FinishReason> finish_reason;
    std::vector<int32_t> output_token_ids;
    std::vector<float> output_logprobs;

    int num_computed_tokens = 0;
    int num_output_tokens = 0;

    std::chrono::steady_clock::time_point arrival_time;
    std::chrono::steady_clock::time_point first_token_time;
    std::chrono::steady_clock::time_point last_token_time;

    // 该请求在 llama memory module 中占用的 sequence ID（由 engine 分配，全局唯一）
    int32_t seq_id_ = -1;

    std::function<void(const std::vector<int32_t>&, bool, FinishReason)> stream_callback;

    Request();

    bool is_prefill() const {
        return num_computed_tokens < static_cast<int>(prompt_token_ids.size());
    }

    int num_prompt_tokens() const {
        return static_cast<int>(prompt_token_ids.size());
    }

    int num_tokens_to_compute() const {
        return num_prompt_tokens() - num_computed_tokens;
    }
};

using RequestPtr = std::shared_ptr<Request>;

struct RequestOutput {
    std::string request_id;
    int32_t seq_id_ = -1;  // 对应 llama memory module 中的 sequence ID
    std::vector<int32_t> output_token_ids;
    std::vector<float> output_logprobs;
    FinishReason finish_reason;
    bool finished;
    bool prefill_complete = false;
    int num_new_tokens = 1;
};

RequestOutput create_request_output(
    const std::string& request_id,
    const std::vector<int32_t>& output_token_ids,
    const std::vector<float>& output_logprobs,
    FinishReason finish_reason,
    bool finished,
    int num_new_tokens = 1);

bool is_request_finished(const Request& req);
bool request_has_new_tokens(const Request& req);

}
