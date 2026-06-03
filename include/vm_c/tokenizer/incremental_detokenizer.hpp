#pragma once

#include <string>
#include <vector>
#include <optional>
#include "vm_c/tokenizer/tokenizer.hpp"
#include "vm_c/core/request.hpp"

namespace vm_c {

class IncrementalDetokenizer {
public:
    IncrementalDetokenizer(const Tokenizer* tokenizer,
                           const std::vector<int32_t>& prompt_token_ids,
                           const SamplingParams& params);

    struct UpdateResult {
        std::string delta_text;
        std::string delta_reasoning;
        bool stop_hit = false;
        std::string stop_string;
    };

    UpdateResult update(int32_t new_token_id, bool stop_terminated);

    int num_output_tokens() const;

    const std::vector<int32_t>& output_token_ids() const { return output_token_ids_; }

private:
    std::string decode_next(int32_t token_id);
    std::optional<std::pair<std::string, int>> check_stop_strings(
        const std::string& output_text, int new_char_count);
    void split_reasoning_delta(const std::string& delta);

    const Tokenizer* tokenizer_;
    std::vector<int32_t> token_ids_;
    std::vector<int32_t> output_token_ids_;
    int prompt_len_;

    std::string output_text_;
    int prefix_offset_ = 0;
    int read_offset_ = 0;
    int last_output_text_offset_ = 0;

    std::string reasoning_text_;
    int last_reasoning_text_offset_ = 0;
    bool enable_thinking_ = true;
    bool in_reasoning_ = true;
    std::string reasoning_buffer_;

    std::vector<std::string> stop_strings_;
    int stop_buffer_length_ = 0;
    int min_tokens_ = 0;
};

}
