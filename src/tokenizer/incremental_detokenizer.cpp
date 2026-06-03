#include "vm_c/tokenizer/incremental_detokenizer.hpp"
#include <algorithm>
#include <cstddef>

namespace vm_c {

static constexpr int kInitialDetokOffset = 5;
// 与官方 llama.cpp chat.cpp THINK_END 一致（Qwen3 使用 </think>）
static const std::string kThinkEnd = "</think>";

// Check if the end of `text` has an incomplete UTF-8 multi-byte sequence.
// Returns the byte offset where valid UTF-8 ends (i.e., the position before
// any incomplete trailing character).  If the whole text is valid, returns
// text.size().
// Mirrors llama.cpp's validate_utf8() in tools/server/server-common.cpp.
static size_t utf8_valid_end_offset(const std::string& text) {
    size_t len = text.size();
    if (len == 0) return 0;

    // Walk backwards from the end (max 4 bytes) looking for a multi-byte
    // leading byte that is missing its continuation bytes.
    for (size_t i = 1; i <= 4 && i <= len; ++i) {
        unsigned char c = static_cast<unsigned char>(text[len - i]);

        if ((c & 0xE0) == 0xC0) {
            // 2-byte sequence start (110xxxxx)
            if (i < 2) return len - i;  // only 1 byte → incomplete
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte sequence start (1110xxxx)
            if (i < 3) return len - i;  // only 1–2 bytes → incomplete
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte sequence start (11110xxx)
            if (i < 4) return len - i;  // only 1–3 bytes → incomplete
        }
        // Continuation bytes (10xxxxxx) and ASCII (0xxxxxxx) are skipped —
        // they don't start a multi-byte sequence.
    }
    return len;
}

// If output_text_ ends with an incomplete UTF-8 sequence, truncate
// result.delta_text / result.delta_reasoning so the incomplete bytes are
// held back.  Those bytes remain in output_text_ / reasoning_text_ and will
// be included in the next call when the next token completes the character.
// Mirrors llama.cpp's process_token() approach.
static void trim_incomplete_utf8(  const std::string& accumulated,
                                    size_t           prev_size,
                                    std::string&     delta) {
    size_t valid_end = utf8_valid_end_offset(accumulated);
    if (valid_end >= accumulated.size()) return;  // all clean

    // Some trailing bytes are incomplete.  Trim delta.
    if (valid_end <= prev_size) {
        // All new bytes are part of an incomplete sequence → send nothing.
        delta.clear();
    } else {
        // Partial: only the bytes before valid_end are complete.
        delta = accumulated.substr(prev_size, valid_end - prev_size);
    }
}

IncrementalDetokenizer::IncrementalDetokenizer(
    const Tokenizer* tokenizer,
    const std::vector<int32_t>& prompt_token_ids,
    const SamplingParams& params)
    : tokenizer_(tokenizer),
      prompt_len_(static_cast<int>(prompt_token_ids.size())),
      min_tokens_(params.min_tokens),
      enable_thinking_(params.enable_thinking),
      in_reasoning_(params.enable_thinking) {

    token_ids_ = prompt_token_ids;

    for (const auto& s : params.stop_strings) {
        if (!s.empty()) stop_strings_.push_back(s);
    }

    if (!stop_strings_.empty()) {
        int max_len = 0;
        for (const auto& s : stop_strings_) {
            max_len = std::max(max_len, static_cast<int>(s.size()));
        }
        stop_buffer_length_ = max_len - 1;
    }

    if (prompt_len_ > 0) {
        int check_len = std::min(prompt_len_, kInitialDetokOffset + 2);
        read_offset_ = prompt_len_ - check_len;
        prefix_offset_ = std::max(read_offset_ - kInitialDetokOffset, 0);
    }
}

void IncrementalDetokenizer::split_reasoning_delta(const std::string& delta) {
    if (!enable_thinking_ || delta.empty()) return;

    reasoning_buffer_ += delta;

    if (in_reasoning_) {
        auto pos = reasoning_buffer_.find(kThinkEnd);
        if (pos != std::string::npos) {
            std::string reasoning_delta = reasoning_buffer_.substr(0, pos);
            reasoning_text_ += reasoning_delta;

            std::string after = reasoning_buffer_.substr(pos + kThinkEnd.size());
            while (!after.empty() && (after[0] == '\n' || after[0] == '\r')) {
                after = after.substr(1);
            }
            output_text_ += after;

            in_reasoning_ = false;
            reasoning_buffer_.clear();
        } else {
            // 保留最后 kThinkEnd.size()-1 字节以备跨边界匹配 </think>，
            // 但截断时必须保持 UTF-8 字符完整性。
            size_t keep = (kThinkEnd.size() > 1) ? (kThinkEnd.size() - 1) : 0;
            if (reasoning_buffer_.size() > keep) {
                size_t send_len = reasoning_buffer_.size() - keep;
                // 回退到上一个完整的 UTF-8 字符边界
                while (send_len > 0 && utf8_valid_end_offset(reasoning_buffer_.substr(0, send_len)) < send_len) {
                    --send_len;
                }
                if (send_len > 0) {
                    std::string reasoning_delta = reasoning_buffer_.substr(0, send_len);
                    reasoning_text_ += reasoning_delta;
                    reasoning_buffer_ = reasoning_buffer_.substr(send_len);
                }
            }
        }
    } else {
        output_text_ += reasoning_buffer_;
        reasoning_buffer_.clear();
    }
}

std::string IncrementalDetokenizer::decode_next(int32_t token_id) {
    token_ids_.push_back(token_id);
    int cur_len = static_cast<int>(token_ids_.size());

    std::vector<int32_t> prefix_ids(token_ids_.begin() + prefix_offset_,
                                    token_ids_.begin() + read_offset_);
    std::string prefix_text = tokenizer_->decode(prefix_ids, true);

    std::vector<int32_t> new_ids(token_ids_.begin() + prefix_offset_,
                                 token_ids_.end());
    std::string new_text = tokenizer_->decode(new_ids, true);

    if (static_cast<int>(new_text.size()) <= static_cast<int>(prefix_text.size())) {
        prefix_offset_ = read_offset_;
        read_offset_ = cur_len;
        return "";
    }

    std::string delta = new_text.substr(prefix_text.size());
    prefix_offset_ = read_offset_;
    read_offset_ = cur_len;
    return delta;
}

std::optional<std::pair<std::string, int>> IncrementalDetokenizer::check_stop_strings(
    const std::string& text, int new_char_count) {
    if (new_char_count <= 0 || stop_strings_.empty()) return std::nullopt;
    for (const auto& stop_str : stop_strings_) {
        int stop_len = static_cast<int>(stop_str.size());
        int search_start = std::max(0, static_cast<int>(text.size()) - new_char_count - stop_len);
        auto pos = text.find(stop_str, search_start);
        if (pos != std::string::npos) {
            return std::make_pair(stop_str, static_cast<int>(pos));
        }
    }
    return std::nullopt;
}

IncrementalDetokenizer::UpdateResult IncrementalDetokenizer::update(
    int32_t new_token_id, bool stop_terminated) {
    UpdateResult result;

    output_token_ids_.push_back(new_token_id);

    std::string delta;
    if (stop_terminated) {
        delta = "";
    } else {
        delta = decode_next(new_token_id);
    }

    if (enable_thinking_) {
        size_t content_before = output_text_.size();
        split_reasoning_delta(delta);
        std::string content_delta = output_text_.substr(content_before);

        // Trim incomplete UTF-8 from content delta
        trim_incomplete_utf8(output_text_, content_before, content_delta);

        std::string reasoning_delta;
        int reasoning_len = static_cast<int>(reasoning_text_.size());
        if (last_reasoning_text_offset_ < reasoning_len) {
            reasoning_delta = reasoning_text_.substr(last_reasoning_text_offset_);
            // Trim incomplete UTF-8 from reasoning delta
            trim_incomplete_utf8(reasoning_text_,
                                 static_cast<size_t>(last_reasoning_text_offset_),
                                 reasoning_delta);
            last_reasoning_text_offset_ = static_cast<int>(utf8_valid_end_offset(reasoning_text_));
        }

        result.delta_text = content_delta;
        result.delta_reasoning = reasoning_delta;
    } else {
        size_t prev_size = output_text_.size();
        output_text_ += delta;
        result.delta_text = output_text_.substr(prev_size);
        // Trim incomplete UTF-8 from content delta
        trim_incomplete_utf8(output_text_, prev_size, result.delta_text);
        result.delta_reasoning = "";
    }

    int stop_check_offset = static_cast<int>(output_text_.size());

    if (min_tokens_ > 0 && static_cast<int>(output_token_ids_.size()) <= min_tokens_) {
        stop_check_offset = static_cast<int>(output_text_.size());
    }

    if (!stop_strings_.empty() &&
        (min_tokens_ <= 0 || static_cast<int>(output_token_ids_.size()) > min_tokens_)) {
        int new_chars = static_cast<int>(output_text_.size()) - stop_check_offset;
        auto stop_result = check_stop_strings(output_text_, new_chars);
        if (stop_result) {
            result.stop_hit = true;
            result.stop_string = stop_result->first;
            output_text_ = output_text_.substr(0, stop_result->second);
        }
    }

    return result;
}

int IncrementalDetokenizer::num_output_tokens() const {
    return static_cast<int>(output_token_ids_.size());
}

}
