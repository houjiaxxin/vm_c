#include "vm_c/tokenizer/jinja_template.hpp"

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <sstream>

namespace vm_c {

// 官方 Qwen3 chat 模板 fallback（取自 llama.cpp/models/templates/Qwen-Qwen3-0.6B.jinja）
// ─────────────────────────────────────────────────────────────────────────────
// 当 GGUF 模型元数据未带 chat_template 时，使用本模板；纯 ASCII 标记 <think>/</think>
// 与 <tool_response>，与模型训练时一致，不会因乱码导致推理崩坏。
// 支持特性：namespace、loop.previtem/nextitem、is not、[:-1]、.startswith()/.split()、tojson 等。
// ─────────────────────────────────────────────────────────────────────────────
static const char* QWEN3_FALLBACK_TEMPLATE = R"jinja({%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set content = message.content %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in message.content %}
                {%- set content = message.content.split('</think>')[-1].lstrip('\n') }}
                {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') }}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %})jinja";

// 声明在 vm_c:: 命名空间内的 jinja_template.hpp
const char* get_qwen3_fallback_template() {
    return QWEN3_FALLBACK_TEMPLATE;
}



JinjaTemplate::JinjaTemplate() = default;
JinjaTemplate::~JinjaTemplate() = default;

bool JinjaTemplate::load(const std::string& template_str) {
    if (template_str.empty()) {
        spdlog::error("JinjaTemplate::load: empty template string");
        loaded_ = false;
        return false;
    }

    try {
        jinja::lexer lexer;
        jinja::lexer_result tokens = lexer.tokenize(template_str);
        // parse_from_tokens 返回 jinja::program prvalue；vm_c 已在 runtime.h 中为
        // jinja::program 显式 = default 移动构造，故可安全 std::move。
        jinja::program parsed = jinja::parse_from_tokens(tokens);
        prog_ = std::make_unique<jinja::program>(std::move(parsed));
        source_ = std::make_shared<std::string>(template_str);
        loaded_ = true;
        spdlog::info("Jinja template parsed successfully ({} chars, {} tokens)",
                     template_str.size(), tokens.tokens.size());
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Jinja parse error: {}", e.what());
        prog_.reset();
        source_.reset();
        loaded_ = false;
        return false;
    }
}

std::string JinjaTemplate::render(
    const std::vector<ChatMessageForTemplate>& messages,
    bool add_generation_prompt,
    bool enable_thinking,
    const std::optional<std::vector<std::string>>& tools
) const {
    if (!loaded_ || !prog_ || !source_) {
        spdlog::error("JinjaTemplate::render called but template not loaded");
        return {};
    }

    // 构造 nlohmann::ordered_json 输入（jinja 引擎通过 global_from_json 接收）
    nlohmann::ordered_json messages_json = nlohmann::ordered_json::array();
    for (const auto& msg : messages) {
        nlohmann::ordered_json m;
        m["role"] = msg.role;
        m["content"] = msg.content;
        if (msg.reasoning_content.has_value()) {
            m["reasoning_content"] = *msg.reasoning_content;
        }
        messages_json.push_back(std::move(m));
    }

    nlohmann::ordered_json params;
    params["messages"] = std::move(messages_json);
    params["add_generation_prompt"] = add_generation_prompt;
    params["enable_thinking"] = enable_thinking;

    if (tools.has_value() && !tools->empty()) {
        nlohmann::ordered_json tools_arr = nlohmann::ordered_json::array();
        for (const auto& t_json : *tools) {
            // tools 元素可能是 JSON 字符串（如 "[{\"type\":\"function\",...}]"），
            // 也可能是裸的 JSON 对象；统一 parse 后传入
            try {
                tools_arr.push_back(nlohmann::ordered_json::parse(t_json));
            } catch (const std::exception& e) {
                spdlog::warn("Failed to parse tool json: {}, fallback to raw string", e.what());
                tools_arr.push_back(t_json);
            }
        }
        params["tools"] = std::move(tools_arr);
    }

    try {
        jinja::context ctx(*source_);
        jinja::global_from_json(ctx, params, /*mark_input=*/false);

        jinja::runtime rt(ctx);
        jinja::value_array results = rt.execute(*prog_);
        // value_array (shared_ptr<value_array_t>) 可隐式转换为 value (shared_ptr<value_t>)
        jinja::value_string parts = jinja::runtime::gather_string_parts(results);
        return parts->as_string().str();
    } catch (const std::exception& e) {
        spdlog::error("Jinja render error: {}", e.what());
        return {};
    }
}

} // namespace vm_c
