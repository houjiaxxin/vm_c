#pragma once

// vm_c 聊天模板包装层
//
// 原实现使用 Jinja2Cpp + 自家降级 ChatML 模板（含乱码字符 ⟁/⟼，UTF-8 被错误转码为 â³/â´）。
// 现改用 llama.cpp 自带的 common/jinja/ 引擎（已随 vm_c_llama 一并编译），
// 命名空间为 jinja::。这才是根因：Jinja2Cpp 解释器不支持 Qwen3 模板中的
// `namespace`/`is not`/`[:-1]`/`.split()`/`.rstrip()` 等高级特性，因此 vm_c 不得不内置
// 一份降级 ChatML 模板，而那份降级模板里就含 ⟁/⟼ 这种非 ASCII 字符；UTF-8 字节
// 被错误当成 Latin-1 再编回 UTF-8 后会变成 "â³"/"â´" 之类的乱码。
//
// 现使用官方 jinja 引擎（src/official/llama/jinja/）和官方 Qwen3 模板
// （llama.cpp/models/templates/Qwen-Qwen3-0.6B.jinja，纯 ASCII 4116 字节），
// 根除模板乱码。
//
// 注意：所有 jinja 头文件已加 jinja_ 前缀（如 jinja_lexer.h），避免与
// 系统 C 头 <string.h>/<lexer.h> 同名冲突。GCC 7.3.0 的 <cstring> 内部
// include <string.h> 时会按 include 路径搜索，若不前缀化，会命中 jinja/string.h
// （C++ 头）而非系统 C <string.h>，导致 ::memcpy 未声明。
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include "jinja_lexer.h"   // jinja 引擎（src/official/llama/jinja/）
#include "jinja_parser.h"
#include "jinja_runtime.h"
#include "jinja_value.h"

namespace vm_c {

struct ChatMessageForTemplate {
    std::string role;
    std::string content;
    std::optional<std::string> reasoning_content;
};

class JinjaTemplate {
public:
    JinjaTemplate();
    ~JinjaTemplate();
    JinjaTemplate(const JinjaTemplate&) = delete;
    JinjaTemplate& operator=(const JinjaTemplate&) = delete;
    JinjaTemplate(JinjaTemplate&&) = default;
    JinjaTemplate& operator=(JinjaTemplate&&) = default;

    // 解析 jinja 模板字符串（支持官方 Qwen3/Hermes/ChatGLM 等所有内置模板）
    bool load(const std::string& template_str);

    // 渲染消息为 prompt 字符串
    std::string render(
        const std::vector<ChatMessageForTemplate>& messages,
        bool add_generation_prompt = true,
        bool enable_thinking = true,
        const std::optional<std::vector<std::string>>& tools = std::nullopt
    ) const;

    bool is_loaded() const { return loaded_; }

private:
    bool loaded_ = false;
    // program 继承自 statement（含 virtual 函数 + unique_ptr 成员），不可直接复制
    // 用 unique_ptr 持有；move 构造由编译器生成
    std::unique_ptr<jinja::program> prog_;
    // 源字符串保存为 shared_ptr，供 jinja::context 共享（context 内部也是 shared_ptr）
    std::shared_ptr<std::string> source_;
};

// 官方 Qwen3 chat 模板 fallback（取自 llama.cpp/models/templates/Qwen-Qwen3-0.6B.jinja）
// 当 GGUF 模型元数据未带 chat_template 时使用此模板；纯 ASCII 标记，与模型训练一致
const char* get_qwen3_fallback_template();

} // namespace vm_c
