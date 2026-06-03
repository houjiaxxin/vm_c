#include "vm_c/server/http_server.hpp"
#include <nlohmann/json.hpp>
#include <httplib.h>
#include <spdlog/spdlog.h>
#include <chrono>
#include <random>
#include <sstream>

namespace vm_c {

using json = nlohmann::json;

namespace {

// Safe JSON serialization: replace invalid UTF-8 sequences instead of throwing.
static std::string safe_dump(const json& j) {
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

}

static std::string gen_id(const std::string& prefix) {
    static std::atomic<int64_t> counter{0};
    return prefix + "-" + std::to_string(counter.fetch_add(1));
}

static int64_t now_epoch_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static json make_error_json(const std::string& message, const std::string& type,
                            int code, const std::string& param = "") {
    json err;
    err["error"]["message"] = message;
    err["error"]["type"] = type;
    err["error"]["code"] = code;
    if (!param.empty()) {
        err["error"]["param"] = param;
    }
    return err;
}

struct HttpServer::Impl {
    std::unique_ptr<httplib::Server> server;
    std::thread server_thread;
    bool running = false;
    std::atomic<bool> listening{false};
};

HttpServer::HttpServer(const VmCConfig& config) : config_(config) {
    impl_ = std::make_unique<Impl>();
}

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::start() {
    impl_->server = std::make_unique<httplib::Server>();
    impl_->server->new_task_queue = [this] { return new httplib::ThreadPool(config_.server.thread_pool_size); };
    setup_routes();

    impl_->running = true;
    impl_->listening.store(false);

    // 先做 bind（同步、立即返回），成功则开启 listen 线程
    auto bound = impl_->server->bind_to_port(config_.server.host.c_str(),
                                              config_.server.port, 0);
    if (!bound) {
        impl_->running = false;
        spdlog::error("Failed to bind HTTP server to {}:{}",
                      config_.server.host, config_.server.port);
        throw std::runtime_error(
            "Failed to bind HTTP server to " + config_.server.host
            + ":" + std::to_string(config_.server.port));
    }
    impl_->listening.store(true);
    spdlog::info("HTTP server bound to {}:{}, entering accept loop",
                 config_.server.host, config_.server.port);

    impl_->server_thread = std::thread([this]() {
        impl_->server->listen_after_bind();
        impl_->listening.store(false);
        impl_->running = false;
        spdlog::info("HTTP server stopped");
    });
}

void HttpServer::stop() {
    if (impl_->server) {
        impl_->server->stop();
    }
    if (impl_->server_thread.joinable()) {
        impl_->server_thread.join();
    }
    impl_->running = false;
    impl_->listening.store(false);
}

bool HttpServer::is_running() const {
    return impl_->running;
}

void HttpServer::set_chat_template(const std::string& tmpl) {
    chat_template_ = tmpl;
    if (!tmpl.empty()) {
        jinja_template_.load(tmpl);
    }
}

SamplingParams HttpServer::build_sampling_params(const ChatCompletionRequest& req) {
    SamplingParams params;
    int max_out = req.max_completion_tokens > 0 ? req.max_completion_tokens
                  : req.max_tokens > 0 ? req.max_tokens : config_.server.default_max_tokens;
    params.max_tokens = max_out;
    params.temperature = req.temperature;
    params.top_p = req.top_p;
    params.top_k = req.top_k;
    params.repetition_penalty = req.repetition_penalty;
    params.frequency_penalty = req.frequency_penalty;
    params.presence_penalty = req.presence_penalty;
    params.min_p = req.min_p;
    params.min_tokens = req.min_tokens;
    params.ignore_eos = req.ignore_eos;
    params.stop_strings = req.stop;
    params.stop_token_ids = req.stop_token_ids;
    params.enable_thinking = req.enable_thinking;
    return params;
}

SamplingParams HttpServer::build_sampling_params(const CompletionRequest& req) {
    SamplingParams params;
    params.max_tokens = req.max_tokens;
    params.temperature = req.temperature;
    params.top_p = req.top_p;
    params.top_k = req.top_k;
    params.repetition_penalty = req.repetition_penalty;
    params.frequency_penalty = req.frequency_penalty;
    params.presence_penalty = req.presence_penalty;
    params.min_p = req.min_p;
    params.min_tokens = req.min_tokens;
    params.stop_strings = req.stop;
    return params;
}

std::string HttpServer::finish_reason_str(FinishReason r) {
    switch (r) {
        case FinishReason::STOP:    return "stop";
        case FinishReason::LENGTH:  return "length";
        case FinishReason::ERROR:   return "error";
        case FinishReason::ABORTED: return "abort";
    }
    return "stop";
}

ChatCompletionRequest HttpServer::parse_chat_request(const json& j) const {
    ChatCompletionRequest req;
    req.model = j.value("model", "");
    req.temperature = j.value("temperature", 1.0f);
    req.top_p = j.value("top_p", 1.0f);
    req.top_k = j.value("top_k", -1);
    req.max_tokens = j.value("max_tokens", -1);
    req.max_completion_tokens = j.value("max_completion_tokens", -1);
    req.n = j.value("n", 1);
    req.stream = j.value("stream", false);
    req.repetition_penalty = j.value("repetition_penalty", 1.0f);
    req.frequency_penalty = j.value("frequency_penalty", 0.0f);
    req.presence_penalty = j.value("presence_penalty", 0.0f);
    req.min_p = j.value("min_p", 0.0f);
    req.seed = j.value("seed", -1);
    req.min_tokens = j.value("min_tokens", 0);
    req.ignore_eos = j.value("ignore_eos", false);
    req.echo = j.value("echo", false);
    req.add_generation_prompt = j.value("add_generation_prompt", true);
    req.enable_thinking = j.value("enable_thinking", true);

    if (j.contains("stop")) {
        if (j["stop"].is_array()) {
            for (auto& s : j["stop"]) req.stop.push_back(s.get<std::string>());
        } else if (j["stop"].is_string()) {
            req.stop.push_back(j["stop"].get<std::string>());
        }
    }

    if (j.contains("stop_token_ids") && j["stop_token_ids"].is_array()) {
        for (auto& id : j["stop_token_ids"]) req.stop_token_ids.push_back(id.get<int>());
    }

    if (j.contains("stream_options") && j["stream_options"].is_object()) {
        req.stream_options.include_usage = j["stream_options"].value("include_usage", false);
        req.stream_options.continuous_usage_stats = j["stream_options"].value("continuous_usage_stats", false);
    }

    if (j.contains("messages")) {
        for (auto& msg : j["messages"]) {
            ChatMessage cm;
            cm.role = msg.value("role", "");
            if (msg.contains("content")) {
                if (msg["content"].is_string()) {
                    cm.content = msg["content"].get<std::string>();
                } else if (msg["content"].is_array()) {
                    for (auto& part : msg["content"]) {
                        if (part.contains("type") && part["type"] == "text") {
                            cm.content += part.value("text", "");
                        } else if (part.contains("type") && part["type"] == "image_url") {
                            if (part.contains("image_url") && part["image_url"].contains("url")) {
                                cm.image_urls.push_back(part["image_url"]["url"].get<std::string>());
                            }
                        } else if (part.contains("type") && part["type"] == "video_url") {
                            if (part.contains("video_url") && part["video_url"].contains("url")) {
                                cm.video_urls.push_back(part["video_url"]["url"].get<std::string>());
                            }
                        } else if (part.contains("type") && part["type"] == "image") {
                            if (part.contains("image")) {
                                cm.image_urls.push_back(part["image"].get<std::string>());
                            }
                        }
                    }
                }
            }
            if (msg.contains("name") && msg["name"].is_string()) {
                cm.name = msg["name"].get<std::string>();
            }
            if (msg.contains("tool_call_id") && msg["tool_call_id"].is_string()) {
                cm.tool_call_id = msg["tool_call_id"].get<std::string>();
            }
            req.messages.push_back(cm);
        }
    }

    return req;
}

CompletionRequest HttpServer::parse_completion_request(const json& j) const {
    CompletionRequest req;
    req.model = j.value("model", "");
    req.temperature = j.value("temperature", 1.0f);
    req.top_p = j.value("top_p", 1.0f);
    req.top_k = j.value("top_k", -1);
    req.max_tokens = j.value("max_tokens", config_.server.default_completion_max_tokens);
    req.n = j.value("n", 1);
    req.stream = j.value("stream", false);
    req.echo = j.value("echo", false);
    req.repetition_penalty = j.value("repetition_penalty", 1.0f);
    req.frequency_penalty = j.value("frequency_penalty", 0.0f);
    req.presence_penalty = j.value("presence_penalty", 0.0f);
    req.min_p = j.value("min_p", 0.0f);
    req.min_tokens = j.value("min_tokens", 0);

    if (j.contains("prompt")) {
        if (j["prompt"].is_string()) {
            req.prompt = j["prompt"].get<std::string>();
        } else if (j["prompt"].is_array()) {
            for (auto& p : j["prompt"]) {
                if (!req.prompt.empty()) req.prompt += "\n";
                req.prompt += p.get<std::string>();
            }
        }
    }

    if (j.contains("stop")) {
        if (j["stop"].is_array()) {
            for (auto& s : j["stop"]) req.stop.push_back(s.get<std::string>());
        } else if (j["stop"].is_string()) {
            req.stop.push_back(j["stop"].get<std::string>());
        }
    }

    if (j.contains("stream_options") && j["stream_options"].is_object()) {
        req.stream_options.include_usage = j["stream_options"].value("include_usage", false);
        req.stream_options.continuous_usage_stats = j["stream_options"].value("continuous_usage_stats", false);
    }

    return req;
}

std::string HttpServer::build_chat_prompt(const std::vector<ChatMessage>& messages,
                                          bool add_generation_prompt,
                                          bool enable_thinking) const {
    if (jinja_template_.is_loaded()) {
        std::vector<vm_c::ChatMessageForTemplate> tmpl_msgs;
        tmpl_msgs.reserve(messages.size());
        for (auto& msg : messages) {
            tmpl_msgs.push_back({msg.role, msg.content, std::nullopt});
        }
        std::string result = jinja_template_.render(tmpl_msgs, add_generation_prompt, enable_thinking);
        if (!result.empty()) return result;
    }

    std::string prompt;
    for (auto& msg : messages) {
        if (msg.role == "system") {
            prompt += "<|im_start|>system\n" + msg.content + "<|im_end|>\n";
        } else if (msg.role == "user") {
            prompt += "<|im_start|>user\n" + msg.content + "<|im_end|>\n";
        } else if (msg.role == "assistant") {
            prompt += "<|im_start|>assistant\n" + msg.content + "<|im_end|>\n";
        } else if (msg.role == "tool") {
            prompt += "<|im_start|>tool\n" + msg.content + "<|im_end|>\n";
        }
    }
    if (add_generation_prompt) {
        prompt += "<|im_start|>assistant\n";
        if (enable_thinking) {
            // 思考模式：插入 <think>，模型在 </think> 之前输出思考内容
            prompt += "<think>\n";
        } else {
            // 非思考模式：插入 <think>\n\n</think>\n\n 让支持思考的模型跳过思考直接输出
            prompt += "<think>\n\n</think>\n\n";
        }
    }
    return prompt;
}

void HttpServer::setup_routes() {
    auto cors_handler = [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "*");
    };

    impl_->server->Options(".*", [cors_handler](const httplib::Request& req, httplib::Response& res) {
        cors_handler(req, res);
    });

    impl_->server->Post("/v1/chat/completions",
        [this, cors_handler](const httplib::Request& req, httplib::Response& res) {
            cors_handler(req, res);
            spdlog::debug("HTTP POST /v1/chat/completions received, body_size={}", req.body.size());

            try {
                auto j = nlohmann::json::parse(req.body);
                bool is_stream = j.value("stream", false);

                if (is_stream && stream_fn_) {
                    res.set_chunked_content_provider(
                        "text/event-stream; charset=utf-8",
                        [this, body = req.body](size_t, httplib::DataSink& sink) -> bool {
                            auto write_fn = [&sink](const char* data, size_t len) -> bool {
                                sink.write(data, len);
                                return true;
                            };
                            this->handle_chat_completions_stream(body, write_fn);
                            sink.done();
                            return true;
                        });
                } else {
                    std::string response_body, content_type;
                    handle_chat_completions(req.body, response_body, content_type);
                    res.set_content(response_body, content_type);
                }
            } catch (const std::exception& e) {
                res.set_content(make_error_json(e.what(), "invalid_request_error", 400).dump(),
                               "application/json");
            }
        });

    impl_->server->Post("/v1/completions",
        [this, cors_handler](const httplib::Request& req, httplib::Response& res) {
            cors_handler(req, res);
            spdlog::debug("HTTP POST /v1/completions received, body_size={}", req.body.size());

            try {
                auto j = nlohmann::json::parse(req.body);
                bool is_stream = j.value("stream", false);

                if (is_stream && stream_fn_) {
                    res.set_chunked_content_provider(
                        "text/event-stream; charset=utf-8",
                        [this, body = req.body](size_t, httplib::DataSink& sink) -> bool {
                            auto write_fn = [&sink](const char* data, size_t len) -> bool {
                                sink.write(data, len);
                                return true;
                            };
                            this->handle_completions_stream(body, write_fn);
                            sink.done();
                            return true;
                        });
                } else {
                    std::string response_body, content_type;
                    handle_completions(req.body, response_body, content_type);
                    res.set_content(response_body, content_type);
                }
            } catch (const std::exception& e) {
                res.set_content(make_error_json(e.what(), "invalid_request_error", 400).dump(),
                               "application/json");
            }
        });

    impl_->server->Get("/v1/models",
        [this, cors_handler](const httplib::Request& req, httplib::Response& res) {
            cors_handler(req, res);
            std::string response_body, content_type;
            handle_models(response_body, content_type);
            res.set_content(response_body, content_type);
        });

    impl_->server->Get("/health",
        [this, cors_handler](const httplib::Request& req, httplib::Response& res) {
            cors_handler(req, res);
            std::string response_body, content_type;
            handle_health(response_body, content_type);
            res.set_content(response_body, content_type);
        });

    impl_->server->Get("/version",
        [cors_handler](const httplib::Request& req, httplib::Response& res) {
            cors_handler(req, res);
            json j;
            j["version"] = "0.1.0";
            res.set_content(j.dump(), "application/json");
        });

    impl_->server->Post("/tokenize",
        [this, cors_handler](const httplib::Request& req, httplib::Response& res) {
            cors_handler(req, res);
            if (tokenize_fn_) {
                try {
                    auto j = json::parse(req.body);
                    std::string text = j.value("text", "");
                    auto ids = tokenize_fn_(text);
                    json resp;
                    resp["tokens"] = ids;
                    res.set_content(resp.dump(), "application/json");
                } catch (const std::exception& e) {
                    res.set_content(make_error_json(e.what(), "invalid_request_error", 400).dump(), "application/json");
                }
            } else {
                res.set_content(make_error_json("tokenizer not available", "server_error", 503).dump(), "application/json");
            }
        });

    impl_->server->Post("/detokenize",
        [this, cors_handler](const httplib::Request& req, httplib::Response& res) {
            cors_handler(req, res);
            if (detokenize_fn_) {
                try {
                    auto j = json::parse(req.body);
                    std::vector<int32_t> ids;
                    if (j.contains("tokens") && j["tokens"].is_array()) {
                        for (auto& id : j["tokens"]) ids.push_back(id.get<int32_t>());
                    }
                    bool spaces = j.value("spaces_between_special_tokens", false);
                    std::string text = detokenize_fn_(ids, spaces);
                    json resp;
                    resp["text"] = text;
                    res.set_content(resp.dump(), "application/json");
                } catch (const std::exception& e) {
                    res.set_content(make_error_json(e.what(), "invalid_request_error", 400).dump(), "application/json");
                }
            } else {
                res.set_content(make_error_json("detokenizer not available", "server_error", 503).dump(), "application/json");
            }
        });

    spdlog::info("HTTP routes registered: /v1/chat/completions, /v1/completions, /v1/models, /health, /version, /tokenize, /detokenize");
}

void HttpServer::handle_chat_completions(const std::string& body,
                                          std::string& response,
                                          std::string& content_type) {
    content_type = "application/json";
    auto t_start = std::chrono::steady_clock::now();

    try {
        auto j = json::parse(body);
        auto req = parse_chat_request(j);

        if (req.messages.empty()) {
            response = make_error_json("messages is required", "invalid_request_error", 400, "messages").dump();
            return;
        }

        if (req.n != 1) {
            spdlog::warn("n={} requested but only n=1 is supported", req.n);
        }

        spdlog::info("ChatCompletion request: model={}, stream={}, messages={}, max_tokens={}",
                     req.model, req.stream, req.messages.size(),
                     req.max_completion_tokens > 0 ? req.max_completion_tokens
                     : req.max_tokens > 0 ? req.max_tokens : config_.server.default_max_tokens);

        std::string prompt = build_chat_prompt(req.messages, req.add_generation_prompt, req.enable_thinking);
        spdlog::info("[TEMPLATE] prompt_raw ({} bytes): {}", prompt.size(), prompt);
        SamplingParams params = build_sampling_params(req);

        if (req.stream && stream_fn_) {
            content_type = "text/event-stream";
            std::ostringstream ss;
            std::string req_id = gen_id("chatcmpl");
            int64_t created = now_epoch_seconds();

            {
                json chunk;
                chunk["id"] = req_id;
                chunk["object"] = "chat.completion.chunk";
                chunk["created"] = created;
                chunk["model"] = config_.server.served_model_name;
                chunk["choices"] = json::array({
                    {
                        {"index", 0},
                        {"delta", {{"role", "assistant"}, {"content", ""}}},
                        {"finish_reason", nullptr}
                    }
                });
                ss << "data: " << safe_dump(chunk) << "\n\n";
            }

            int token_count = 0;
            auto t_ttft = std::chrono::steady_clock::now();
            int prompt_tokens = 0;

            spdlog::debug("Calling stream_fn_ for request {}", req_id);
        prompt_tokens = stream_fn_(prompt, params, req.model,
            [&](const std::string& content, const std::string& reasoning, bool finished, const std::string& reason) {
                if (token_count == 0) {
                    auto ttft_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t_ttft).count();
                    spdlog::info("TTFT: {:.1f} ms", ttft_ms);
                }
                ++token_count;

                json chunk;
                chunk["id"] = req_id;
                chunk["object"] = "chat.completion.chunk";
                chunk["created"] = created;
                chunk["model"] = config_.server.served_model_name;

                if (finished) {
                    chunk["choices"] = json::array({
                        {
                            {"index", 0},
                            {"delta", json::object()},
                            {"finish_reason", reason}
                        }
                    });
                    if (!config_.server.system_fingerprint.empty()) {
                        chunk["system_fingerprint"] = config_.server.system_fingerprint;
                    }
                    auto total_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t_start).count();
                    double tps = token_count > 0 ? (token_count * 1000.0 / total_ms) : 0;
                    spdlog::info("Stream finished: {} tokens, total {:.0f} ms, TPS {:.1f}, reason={}",
                                 token_count, total_ms, tps, reason);
                } else {
                    json delta;
                    if (!content.empty()) delta["content"] = content;
                    if (!reasoning.empty()) delta["reasoning_content"] = reasoning;
                    if (delta.empty()) delta["content"] = content;
                    chunk["choices"] = json::array({
                        {
                            {"index", 0},
                            {"delta", delta},
                            {"finish_reason", nullptr}
                        }
                    });
                    if (req.stream_options.continuous_usage_stats) {
                        chunk["usage"] = {
                            {"prompt_tokens", prompt_tokens},
                            {"completion_tokens", token_count},
                            {"total_tokens", prompt_tokens + token_count}
                        };
                    }
                }
                ss << "data: " << safe_dump(chunk) << "\n\n";
            });
    spdlog::debug("stream_fn_ returned, prompt_tokens={}", prompt_tokens);

    if (req.stream_options.include_usage) {
        json usage_chunk;
        usage_chunk["id"] = req_id;
        usage_chunk["object"] = "chat.completion.chunk";
        usage_chunk["created"] = created;
        usage_chunk["model"] = config_.server.served_model_name;
        usage_chunk["choices"] = json::array();
        usage_chunk["usage"] = {
            {"prompt_tokens", prompt_tokens},
            {"completion_tokens", token_count},
            {"total_tokens", prompt_tokens + token_count}
        };
        if (!config_.server.system_fingerprint.empty()) {
            usage_chunk["system_fingerprint"] = config_.server.system_fingerprint;
        }
        ss << "data: " << usage_chunk.dump() << "\n\n";
    }

            ss << "data: [DONE]\n\n";
            response = ss.str();
        } else if (generate_fn_) {
            spdlog::debug("Calling generate_fn_ (non-streaming)");
            GenerateResult result = generate_fn_(prompt, params, req.model);
            spdlog::debug("generate_fn_ returned, result_size={}", result.text.size());

            auto total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_start).count();
            spdlog::info("ChatCompletion response: {} chars, total {:.0f} ms",
                         result.text.size(), total_ms);

            json resp_j;
            resp_j["id"] = gen_id("chatcmpl");
            resp_j["object"] = "chat.completion";
            resp_j["created"] = now_epoch_seconds();
            resp_j["model"] = config_.server.served_model_name;
            std::string content = result.text;
            if (req.echo) {
                content = prompt + content;
            }
            json message = {{"role", "assistant"}, {"content", content}};
            if (!result.reasoning_content.empty()) {
                message["reasoning_content"] = result.reasoning_content;
            }
            resp_j["choices"] = json::array({
                {
                    {"index", 0},
                    {"message", message},
                    {"finish_reason", result.finish_reason}
                }
            });
            resp_j["usage"] = {
                {"prompt_tokens", result.prompt_tokens},
                {"completion_tokens", result.completion_tokens},
                {"total_tokens", result.prompt_tokens + result.completion_tokens}
            };
            if (!config_.server.system_fingerprint.empty()) {
                resp_j["system_fingerprint"] = config_.server.system_fingerprint;
            }
            response = safe_dump(resp_j);
        } else {
            response = make_error_json("engine not ready", "server_error", 503).dump();
        }
    } catch (const json::parse_error& e) {
        spdlog::error("ChatCompletion JSON parse error: {}", e.what());
        response = make_error_json(std::string("JSON parse error: ") + e.what(),
                                   "invalid_request_error", 400).dump();
    } catch (const std::exception& e) {
        spdlog::error("ChatCompletion error: {}", e.what());
        response = make_error_json(e.what(), "internal_error", 500).dump();
    }
}

void HttpServer::handle_chat_completions_stream(const std::string& body,
                                                 std::function<bool(const char*, size_t)> write_fn) {
    auto t_start = std::chrono::steady_clock::now();

    auto send_sse = [&](const std::string& data) -> bool {
        std::string msg = "data: " + data + "\n\n";
        return write_fn(msg.c_str(), msg.size());
    };

    try {
        auto j = json::parse(body);
        auto req = parse_chat_request(j);

        if (req.messages.empty()) {
            send_sse(make_error_json("messages is required", "invalid_request_error", 400, "messages").dump());
            return;
        }

        std::string prompt = build_chat_prompt(req.messages, req.add_generation_prompt, req.enable_thinking);
        SamplingParams params = build_sampling_params(req);
        std::string req_id = gen_id("chatcmpl");
        int64_t created = now_epoch_seconds();

        {
            json chunk;
            chunk["id"] = req_id;
            chunk["object"] = "chat.completion.chunk";
            chunk["created"] = created;
            chunk["model"] = config_.server.served_model_name;
            chunk["choices"] = json::array({
                {
                    {"index", 0},
                    {"delta", {{"role", "assistant"}, {"content", ""}}},
                    {"finish_reason", nullptr}
                }
            });
            send_sse(safe_dump(chunk));
        }

        int token_count = 0;
        auto t_ttft = std::chrono::steady_clock::now();
        int prompt_tokens = 0;

        prompt_tokens = stream_fn_(prompt, params, req.model,
            [&](const std::string& content, const std::string& reasoning, bool finished, const std::string& reason) {
                if (token_count == 0) {
                    auto ttft_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t_ttft).count();
                    spdlog::info("TTFT: {:.1f} ms", ttft_ms);
                }
                ++token_count;

                json chunk;
                chunk["id"] = req_id;
                chunk["object"] = "chat.completion.chunk";
                chunk["created"] = created;
                chunk["model"] = config_.server.served_model_name;

                if (finished) {
                    chunk["choices"] = json::array({
                        {
                            {"index", 0},
                            {"delta", json::object()},
                            {"finish_reason", reason}
                        }
                    });
                    if (!config_.server.system_fingerprint.empty()) {
                        chunk["system_fingerprint"] = config_.server.system_fingerprint;
                    }
                    auto total_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t_start).count();
                    double tps = token_count > 0 ? (token_count * 1000.0 / total_ms) : 0;
                    spdlog::info("Stream finished: {} tokens, total {:.0f} ms, TPS {:.1f}, reason={}",
                                 token_count, total_ms, tps, reason);
                } else {
                    json delta;
                    if (!content.empty()) delta["content"] = content;
                    if (!reasoning.empty()) delta["reasoning_content"] = reasoning;
                    if (delta.empty()) delta["content"] = content;
                    chunk["choices"] = json::array({
                        {
                            {"index", 0},
                            {"delta", delta},
                            {"finish_reason", nullptr}
                        }
                    });
                    if (req.stream_options.continuous_usage_stats) {
                        chunk["usage"] = {
                            {"prompt_tokens", prompt_tokens},
                            {"completion_tokens", token_count},
                            {"total_tokens", prompt_tokens + token_count}
                        };
                    }
                }
                send_sse(safe_dump(chunk));
            });

        if (req.stream_options.include_usage) {
            json usage_chunk;
            usage_chunk["id"] = req_id;
            usage_chunk["object"] = "chat.completion.chunk";
            usage_chunk["created"] = created;
            usage_chunk["model"] = config_.server.served_model_name;
            usage_chunk["choices"] = json::array();
            usage_chunk["usage"] = {
                {"prompt_tokens", prompt_tokens},
                {"completion_tokens", token_count},
                {"total_tokens", prompt_tokens + token_count}
            };
            if (!config_.server.system_fingerprint.empty()) {
                usage_chunk["system_fingerprint"] = config_.server.system_fingerprint;
            }
            send_sse(usage_chunk.dump());
        }

        send_sse("[DONE]");
    } catch (const json::parse_error& e) {
        spdlog::error("ChatCompletion stream JSON parse error: {}", e.what());
        json err = make_error_json(std::string("JSON parse error: ") + e.what(),
                                   "invalid_request_error", 400);
        send_sse(err.dump());
    } catch (const std::exception& e) {
        spdlog::error("ChatCompletion stream error: {}", e.what());
        json err = make_error_json(e.what(), "internal_error", 500);
        send_sse(err.dump());
    }
}

void HttpServer::handle_completions(const std::string& body,
                                     std::string& response,
                                     std::string& content_type) {
    content_type = "application/json";
    auto t_start = std::chrono::steady_clock::now();

    try {
        auto j = json::parse(body);
        auto req = parse_completion_request(j);

        if (req.prompt.empty()) {
            response = make_error_json("prompt is required", "invalid_request_error", 400, "prompt").dump();
            return;
        }

        spdlog::info("Completion request: model={}, max_tokens={}, stream={}",
                     req.model, req.max_tokens, req.stream);

        SamplingParams params = build_sampling_params(req);

        if (req.stream && stream_fn_) {
            content_type = "text/event-stream";
            std::ostringstream ss;
            std::string req_id = gen_id("cmpl");
            int64_t created = now_epoch_seconds();
            int token_count = 0;
            int prompt_tokens = 0;

            prompt_tokens = stream_fn_(req.prompt, params, req.model,
                [&](const std::string& content, const std::string& reasoning, bool finished, const std::string& reason) {
                    ++token_count;

                    json chunk;
                    chunk["id"] = req_id;
                    chunk["object"] = "text_completion";
                    chunk["created"] = created;
                    chunk["model"] = config_.server.served_model_name;

                    if (finished) {
                        chunk["choices"] = json::array({
                            {
                                {"index", 0},
                                {"text", ""},
                                {"finish_reason", reason}
                            }
                        });
                        if (!config_.server.system_fingerprint.empty()) {
                            chunk["system_fingerprint"] = config_.server.system_fingerprint;
                        }
                    } else {
                        chunk["choices"] = json::array({
                            {
                                {"index", 0},
                                {"text", content},
                                {"finish_reason", nullptr}
                            }
                        });
                        if (req.stream_options.continuous_usage_stats) {
                            chunk["usage"] = {
                                {"prompt_tokens", prompt_tokens},
                                {"completion_tokens", token_count},
                                {"total_tokens", prompt_tokens + token_count}
                            };
                        }
                    }
                    ss << "data: " << safe_dump(chunk) << "\n\n";
                });

            if (req.stream_options.include_usage) {
                json usage_chunk;
                usage_chunk["id"] = req_id;
                usage_chunk["object"] = "text_completion";
                usage_chunk["created"] = created;
                usage_chunk["model"] = config_.server.served_model_name;
                usage_chunk["choices"] = json::array();
                usage_chunk["usage"] = {
                    {"prompt_tokens", prompt_tokens},
                    {"completion_tokens", token_count},
                    {"total_tokens", prompt_tokens + token_count}
                };
                if (!config_.server.system_fingerprint.empty()) {
                    usage_chunk["system_fingerprint"] = config_.server.system_fingerprint;
                }
                ss << "data: " << usage_chunk.dump() << "\n\n";
            }

            ss << "data: [DONE]\n\n";
            response = ss.str();
        } else if (generate_fn_) {
            GenerateResult result = generate_fn_(req.prompt, params, req.model);

            auto total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_start).count();
            spdlog::info("Completion response: {} chars, total {:.0f} ms",
                         result.text.size(), total_ms);

            json resp;
            resp["id"] = gen_id("cmpl");
            resp["object"] = "text_completion";
            resp["created"] = now_epoch_seconds();
            resp["model"] = config_.server.served_model_name;
            std::string text = result.text;
            if (req.echo) {
                text = req.prompt + text;
            }
            resp["choices"] = json::array({
                {
                    {"index", 0},
                    {"text", text},
                    {"finish_reason", result.finish_reason}
                }
            });
            resp["usage"] = {
                {"prompt_tokens", result.prompt_tokens},
                {"completion_tokens", result.completion_tokens},
                {"total_tokens", result.prompt_tokens + result.completion_tokens}
            };
            if (!config_.server.system_fingerprint.empty()) {
                resp["system_fingerprint"] = config_.server.system_fingerprint;
            }
            response = safe_dump(resp);
        } else {
            response = make_error_json("engine not ready", "server_error", 503).dump();
        }
    } catch (const json::parse_error& e) {
        spdlog::error("Completion JSON parse error: {}", e.what());
        response = make_error_json(std::string("JSON parse error: ") + e.what(),
                                   "invalid_request_error", 400).dump();
    } catch (const std::exception& e) {
        spdlog::error("Completion error: {}", e.what());
        response = make_error_json(e.what(), "internal_error", 500).dump();
    }
}

void HttpServer::handle_completions_stream(const std::string& body,
                                            std::function<bool(const char*, size_t)> write_fn) {
    auto t_start = std::chrono::steady_clock::now();

    auto send_sse = [&](const std::string& data) -> bool {
        std::string msg = "data: " + data + "\n\n";
        return write_fn(msg.c_str(), msg.size());
    };

    try {
        auto j = json::parse(body);
        auto req = parse_completion_request(j);

        if (req.prompt.empty()) {
            send_sse(make_error_json("prompt is required", "invalid_request_error", 400, "prompt").dump());
            return;
        }

        SamplingParams params = build_sampling_params(req);
        std::string req_id = gen_id("cmpl");
        int64_t created = now_epoch_seconds();
        int token_count = 0;
        int prompt_tokens = 0;

        prompt_tokens = stream_fn_(req.prompt, params, req.model,
            [&](const std::string& content, const std::string& reasoning, bool finished, const std::string& reason) {
                ++token_count;

                json chunk;
                chunk["id"] = req_id;
                chunk["object"] = "text_completion";
                chunk["created"] = created;
                chunk["model"] = config_.server.served_model_name;

                if (finished) {
                    chunk["choices"] = json::array({
                        {
                            {"index", 0},
                            {"text", ""},
                            {"finish_reason", reason}
                        }
                    });
                    if (!config_.server.system_fingerprint.empty()) {
                        chunk["system_fingerprint"] = config_.server.system_fingerprint;
                    }
                    auto total_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t_start).count();
                    double tps = token_count > 0 ? (token_count * 1000.0 / total_ms) : 0;
                    spdlog::info("Completion stream finished: {} tokens, total {:.0f} ms, TPS {:.1f}, reason={}",
                                 token_count, total_ms, tps, reason);
                } else {
                    chunk["choices"] = json::array({
                        {
                            {"index", 0},
                            {"text", content},
                            {"finish_reason", nullptr}
                        }
                    });
                    if (req.stream_options.continuous_usage_stats) {
                        chunk["usage"] = {
                            {"prompt_tokens", prompt_tokens},
                            {"completion_tokens", token_count},
                            {"total_tokens", prompt_tokens + token_count}
                        };
                    }
                }
                send_sse(safe_dump(chunk));
            });

        if (req.stream_options.include_usage) {
            json usage_chunk;
            usage_chunk["id"] = req_id;
            usage_chunk["object"] = "text_completion";
            usage_chunk["created"] = created;
            usage_chunk["model"] = config_.server.served_model_name;
            usage_chunk["choices"] = json::array();
            usage_chunk["usage"] = {
                {"prompt_tokens", prompt_tokens},
                {"completion_tokens", token_count},
                {"total_tokens", prompt_tokens + token_count}
            };
            if (!config_.server.system_fingerprint.empty()) {
                usage_chunk["system_fingerprint"] = config_.server.system_fingerprint;
            }
            send_sse(usage_chunk.dump());
        }

        send_sse("[DONE]");
    } catch (const json::parse_error& e) {
        spdlog::error("Completion stream JSON parse error: {}", e.what());
        send_sse(make_error_json(std::string("JSON parse error: ") + e.what(),
                                  "invalid_request_error", 400).dump());
    } catch (const std::exception& e) {
        spdlog::error("Completion stream error: {}", e.what());
        send_sse(make_error_json(e.what(), "internal_error", 500).dump());
    }
}

void HttpServer::handle_models(std::string& response, std::string& content_type) {
    content_type = "application/json";

    int64_t created = now_epoch_seconds();

    json model_card = {
        {"id", config_.server.served_model_name},
        {"object", "model"},
        {"created", created},
        {"owned_by", "vm_c"},
        {"permission", json::array({
            {
                {"id", "modelperm-" + gen_id("mp")},
                {"object", "model_permission"},
                {"created", created},
                {"allow_create_engine", false},
                {"allow_sampling", true},
                {"allow_logprobs", true},
                {"allow_search_indices", false},
                {"allow_view", true},
                {"allow_fine_tuning", false},
                {"organization", "*"},
                {"is_blocking", false}
            }
        })}
    };

    json j = {
        {"object", "list"},
        {"data", json::array({model_card})}
    };
    response = j.dump();
}

void HttpServer::handle_health(std::string& response, std::string& content_type) {
    content_type = "application/json";
    response = R"({"status": "ok"})";
}

}
