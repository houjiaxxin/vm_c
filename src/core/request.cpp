#include "vm_c/core/request.hpp"
#include <chrono>

namespace vm_c {

Request::Request() : status(RequestStatus::WAITING), num_computed_tokens(0), num_output_tokens(0),
                     arrival_time(std::chrono::steady_clock::now()) {}

RequestOutput create_request_output(
    const std::string& request_id,
    const std::vector<int32_t>& output_token_ids,
    const std::vector<float>& output_logprobs,
    FinishReason finish_reason,
    bool finished,
    int num_new_tokens) {
    RequestOutput out;
    out.request_id = request_id;
    out.output_token_ids = output_token_ids;
    out.output_logprobs = output_logprobs;
    out.finish_reason = finish_reason;
    out.finished = finished;
    out.num_new_tokens = num_new_tokens;
    return out;
}

bool is_request_finished(const Request& req) {
    return req.status == RequestStatus::FINISHED || req.status == RequestStatus::PREEMPTED;
}

bool request_has_new_tokens(const Request& req) {
    return req.num_output_tokens > 0 && req.status == RequestStatus::RUNNING;
}

}
