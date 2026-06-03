#pragma once

#include <string>
#include <vector>
#include <functional>
#include <memory>

#include <nlohmann/json.hpp>

#include "vm_c/core/config.hpp"
#include "vm_c/core/request.hpp"
#include "vm_c/tokenizer/jinja_template.hpp"

namespace vm_c {

struct ChatMessage {
    std::string role;
    std::string content;
    std::optional<std::string> name;
    std::optional<std::string> tool_call_id;
    std::vector<std::string> image_urls;
    std::vector<std::string> video_urls;
};

struct StreamOptions {
    bool include_usage = false;
    bool continuous_usage_stats = false;
};

struct ChatCompletionRequest {
    std::string model;
    std::vector<ChatMessage> messages;
    float temperature = 0.80f;     // 对齐 llama.cpp 默认值
    float top_p = 0.95f;
    int top_k = 40;
    int max_tokens = -1;
    int max_completion_tokens = -1;
    int n = 1;
    bool stream = false;
    std::vector<std::string> stop;
    std::vector<int> stop_token_ids;
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    float min_p = 0.0f;
    int seed = -1;
    int min_tokens = 0;
    bool ignore_eos = false;
    bool echo = false;
    StreamOptions stream_options;
    bool add_generation_prompt = true;
    bool enable_thinking = true;
};

struct ChatCompletionChoice {
    int index = 0;
    ChatMessage message;
    std::optional<std::string> finish_reason;
};

struct UsageInfo {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int total_tokens = 0;
};

struct ChatCompletionResponse {
    std::string id;
    std::string object = "chat.completion";
    int64_t created = 0;
    std::string model;
    std::vector<ChatCompletionChoice> choices;
    UsageInfo usage;
};

struct CompletionRequest {
    std::string model;
    std::string prompt;
    float temperature = 0.80f;     // 对齐 llama.cpp 默认值
    float top_p = 0.95f;
    int top_k = 40;
    int max_tokens = 16;
    int n = 1;
    bool stream = false;
    bool echo = false;
    std::vector<std::string> stop;
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    float min_p = 0.0f;
    int min_tokens = 0;
    StreamOptions stream_options;
};

struct GenerateResult {
    std::string text;
    std::string reasoning_content;
    std::string finish_reason = "stop";
    int prompt_tokens = 0;
    int completion_tokens = 0;
};

class HttpServer {
public:
    explicit HttpServer(const VmCConfig& config);
    ~HttpServer();

    void start();
    void stop();
    bool is_running() const;

    using GenerateFn = std::function<GenerateResult(const std::string& prompt,
                                                     const SamplingParams& params,
                                                     const std::string& model)>;
    using StreamFn = std::function<int(const std::string& prompt,
                                       const SamplingParams& params,
                                       const std::string& model,
                                       std::function<void(const std::string& content, const std::string& reasoning, bool finished, const std::string&)>)>;

    using AbortFn = std::function<void(const std::string& request_id)>;
    using TokenizeFn = std::function<std::vector<int32_t>(const std::string&)>;
    using DetokenizeFn = std::function<std::string(const std::vector<int32_t>&, bool spaces_between_special_tokens)>;

    void set_generate_fn(GenerateFn fn) { generate_fn_ = std::move(fn); }
    void set_stream_fn(StreamFn fn) { stream_fn_ = std::move(fn); }
    void set_abort_fn(AbortFn fn) { abort_fn_ = std::move(fn); }
    void set_chat_template(const std::string& tmpl);
    void set_tokenize_fn(TokenizeFn fn) { tokenize_fn_ = std::move(fn); }
    void set_detokenize_fn(DetokenizeFn fn) { detokenize_fn_ = std::move(fn); }

private:
    void setup_routes();
    void handle_chat_completions(const std::string& body, std::string& response,
                                 std::string& content_type);
    void handle_chat_completions_stream(const std::string& body,
                                        std::function<bool(const char*, size_t)> write_fn);
    void handle_completions(const std::string& body, std::string& response,
                            std::string& content_type);
    void handle_completions_stream(const std::string& body,
                                   std::function<bool(const char*, size_t)> write_fn);
    void handle_models(std::string& response, std::string& content_type);
    void handle_health(std::string& response, std::string& content_type);

    SamplingParams build_sampling_params(const ChatCompletionRequest& req);
    SamplingParams build_sampling_params(const CompletionRequest& req);

    ChatCompletionRequest parse_chat_request(const nlohmann::json& j) const;
    CompletionRequest parse_completion_request(const nlohmann::json& j) const;
    std::string build_chat_prompt(const std::vector<ChatMessage>& messages,
                                  bool add_generation_prompt,
                                  bool enable_thinking) const;
    static std::string finish_reason_str(FinishReason r);

    VmCConfig config_;
    struct Impl;
    std::unique_ptr<Impl> impl_;

    GenerateFn generate_fn_;
    StreamFn stream_fn_;
    AbortFn abort_fn_;
    TokenizeFn tokenize_fn_;
    DetokenizeFn detokenize_fn_;
    std::string chat_template_;
    JinjaTemplate jinja_template_;
};

}
