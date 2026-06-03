#include "vm_c/tokenizer/gguf_tokenizer.hpp"
#include "vm_c/core/gguf_reader.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <sstream>
#include <unordered_set>
#include <cstring>
#include <cwchar>
#include <clocale>

namespace vm_c {

// ────────────────────────────────────────────────────────────────────────────
// 字节 ↔ Unicode 编码表 (GPT-2 BPE)
// ────────────────────────────────────────────────────────────────────────────
void GgufTokenizer::init_bytes_to_unicode() {
    bytes_to_unicode_.resize(256, -1);
    unicode_to_bytes_.clear();

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (b < 33 || b == 127 || (b > 160 && b < 173) || (b > 173 && b < 256)) {
            bytes_to_unicode_[b] = 256 + n;
            unicode_to_bytes_[256 + n] = static_cast<uint8_t>(b);
            ++n;
        } else {
            bytes_to_unicode_[b] = b;
            unicode_to_bytes_[b] = static_cast<uint8_t>(b);
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 构造器
// ────────────────────────────────────────────────────────────────────────────
GgufTokenizer::GgufTokenizer(const GgufReader& reader) {
    // 1. tokenizer 类型
    model_type_ = reader.get_str(GgufKeys::VOCABULARY_TOKENIZER, "llama");
    spdlog::info("GgufTokenizer: model_type={}", model_type_);

    // 2. 读取 tokens
    int64_t tk_id = reader.find_key(GgufKeys::TOKENIZER_LIST);
    if (tk_id < 0) {
        throw std::runtime_error("GGUF file missing tokenizer.ggml.tokens");
    }
    const auto& tk_val = reader.kv_value(tk_id);
    if (tk_val.is_array && tk_val.array_type == GgufType::STRING) {
        tokens_ = tk_val.strings;
    } else if (!tk_val.is_array && tk_val.type == GgufType::STRING) {
        tokens_.push_back(tk_val.get_string());
    } else {
        throw std::runtime_error("tokenizer.ggml.tokens: unexpected type");
    }
    spdlog::info("GgufTokenizer: loaded {} tokens", tokens_.size());

    // 3. 读取 scores
    int64_t sc_id = reader.find_key(GgufKeys::TOKENIZER_SCORES);
    if (sc_id >= 0) {
        const auto& sc_val = reader.kv_value(sc_id);
        if (sc_val.type == GgufType::ARRAY && sc_val.array_type == GgufType::FLOAT32) {
            auto arr = sc_val.get_array<float>();
            scores_.assign(arr.begin(), arr.end());
        }
    }

    // 4. 读取 token types
    int64_t tt_id = reader.find_key(GgufKeys::TOKENIZER_TOKEN_TYPES);
    if (tt_id >= 0) {
        const auto& tt_val = reader.kv_value(tt_id);
        if (tt_val.type == GgufType::ARRAY && tt_val.array_type == GgufType::INT32) {
            auto arr = tt_val.get_array<int32_t>();
            token_types_.assign(arr.begin(), arr.end());
        }
    }
    token_types_.resize(tokens_.size(), 0);

    // 5. 读取特殊 token ID
    bos_id_ = static_cast<int32_t>(reader.get_i32(GgufKeys::TOKENIZER_BOS_ID, 1));
    eos_id_ = static_cast<int32_t>(reader.get_i32(GgufKeys::TOKENIZER_EOS_ID, 2));
    pad_id_ = static_cast<int32_t>(reader.get_i32(GgufKeys::TOKENIZER_PAD_ID, 0));
    add_bos_ = reader.get_bool(GgufKeys::TOKENIZER_ADD_BOS, true);

    // Chat template (可选)
    if (reader.find_key(GgufKeys::TOKENIZER_CHAT_TEMPLATE) >= 0) {
        chat_template_ = reader.get_str(GgufKeys::TOKENIZER_CHAT_TEMPLATE, "");
    }

    // 6. 收集特殊 token（对齐官方 llama.cpp LLAMA_TOKEN_ATTR_CONTROL | USER_DEFINED | UNKNOWN）
    //    GGUF token_type 编码: 0=normal, 1=unknown, 2=control, 3=user_defined, 4=unused, 5=byte
    for (size_t i = 0; i < token_types_.size(); ++i) {
        int t = token_types_[i];
        if (t == 1 || t == 2 || t == 3) {  // unknown, control, user_defined
            special_ids_.insert(static_cast<int32_t>(i));
        }
    }

    // 7. 初始化 bytes_to_unicode
    init_bytes_to_unicode();

    // 8. 构建 token 映射表
    build_token_map();

    // 9. 读取 BPE merges
    build_bpe_ranks(reader);

    spdlog::info("GgufTokenizer: vocab={}, bos={}, eos={}, pad={}, add_bos={}",
                 tokens_.size(), bos_id_, eos_id_, pad_id_, add_bos_);
}

// ────────────────────────────────────────────────────────────────────────────
// 构建 token 映射
// ────────────────────────────────────────────────────────────────────────────
void GgufTokenizer::build_token_map() {
    token_map_.clear();
    for (size_t i = 0; i < tokens_.size(); ++i) {
        token_map_[tokens_[i]] = static_cast<int32_t>(i);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// BPE merges
// ────────────────────────────────────────────────────────────────────────────
void GgufTokenizer::build_bpe_ranks(const GgufReader& reader) {
    int64_t mg_id = reader.find_key("tokenizer.ggml.merges");
    if (mg_id < 0) return;

    const auto& mg_val = reader.kv_value(mg_id);
    if (!mg_val.is_array || mg_val.array_type != GgufType::STRING) {
        spdlog::warn("tokenizer.ggml.merges unexpected format, skipping BPE");
        return;
    }

    for (size_t i = 0; i < mg_val.strings.size(); ++i) {
        const auto& s = mg_val.strings[i];
        auto space = s.find(' ');
        if (space == std::string::npos) continue;
        std::string l = s.substr(0, space);
        std::string r = s.substr(space + 1);
        auto it_l = token_map_.find(l);
        auto it_r = token_map_.find(r);
        if (it_l != token_map_.end() && it_r != token_map_.end()) {
            bpe_ranks_[{it_l->second, it_r->second}] = static_cast<int>(i);
        }
    }
    spdlog::info("GgufTokenizer: {} BPE merges loaded", bpe_ranks_.size());
}

// ────────────────────────────────────────────────────────────────────────────
// byte → unicode 字符串
// ────────────────────────────────────────────────────────────────────────────
std::string GgufTokenizer::byte_to_unicode(uint8_t b) const {
    int code = bytes_to_unicode_[b];
    char buf[5] = {0};
    if (code < 0x80) {
        buf[0] = static_cast<char>(code);
    } else if (code < 0x800) {
        buf[0] = static_cast<char>(0xC0 | (code >> 6));
        buf[1] = static_cast<char>(0x80 | (code & 0x3F));
    } else {
        buf[0] = static_cast<char>(0xE0 | (code >> 12));
        buf[1] = static_cast<char>(0x80 | ((code >> 6) & 0x3F));
        buf[2] = static_cast<char>(0x80 | (code & 0x3F));
    }
    return buf;
}

// ────────────────────────────────────────────────────────────────────────────
// token 文本 → id（带 byte fallback）
// ────────────────────────────────────────────────────────────────────────────
int32_t GgufTokenizer::token_or_byte(const std::string& piece) const {
    // 1. 直接查找原字符串
    auto it = token_map_.find(piece);
    if (it != token_map_.end()) return it->second;

    // 2. GPT-2 风格字节编码回退：每个字节 → 对应 Unicode 字符（适用于 Qwen3 等 GPT-2 BPE 模型）
    //    例如 \n (0x0A) → Ċ (U+010A)，
    //    空格 (0x20) → Ġ (U+0120)，
    //    !  (0x21) → ġ 等
    if (!piece.empty()) {
        std::string unicode_str;
        for (unsigned char c : piece) {
            unicode_str += byte_to_unicode(c);
        }
        auto it2 = token_map_.find(unicode_str);
        if (it2 != token_map_.end()) return it2->second;
    }

    // 3. <0xXX> 字节回退（适用于 Llama 等其他模型）
    char hex_buf[12];
    std::string bytes_str;
    for (unsigned char c : piece) {
        snprintf(hex_buf, sizeof(hex_buf), "<0x%02X>", c);
        bytes_str += hex_buf;
    }
    auto it3 = token_map_.find(bytes_str);
    if (it3 != token_map_.end()) return it3->second;

    return 0; // unknown
}

// ────────────────────────────────────────────────────────────────────────────
// 预分词
// ────────────────────────────────────────────────────────────────────────────
std::vector<std::string> GgufTokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> pieces;

    // GPT-2 风格预分词（简化实现，对齐官方 regex 效果）：
    // 1. 换行符作为独立片段（保留 prompt 结构）
    // 2. 空白不作为分隔丢弃，而是保留为词的前缀
    // 3. 标点符号单独拆分
    if (text.empty()) return pieces;

    std::string current;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        if (c == '\n' || c == '\r') {
            // 累积连续换行符作为独立片段
            if (!current.empty()) {
                pieces.push_back(current);
                current.clear();
            }
            std::string nl;
            while (i < text.size() && (text[i] == '\n' || text[i] == '\r')) {
                nl += text[i++];
            }
            pieces.push_back(nl);
        } else if (std::isspace(c)) {
            // 空格/制表符：保留为当前词的前缀（GPT-2 风格）
            current += c;
            ++i;
        } else if (std::ispunct(c) && c != '\'') {
            // 标点符号（除单引号外）单独作为片段
            if (!current.empty()) {
                pieces.push_back(current);
                current.clear();
            }
            pieces.push_back(std::string(1, c));
            ++i;
        } else {
            current += c;
            ++i;
        }
    }
    if (!current.empty()) {
        pieces.push_back(current);
    }

    return pieces;
}

// ────────────────────────────────────────────────────────────────────────────
// BPE pair 获取（按 rank 排序）
// ────────────────────────────────────────────────────────────────────────────
std::vector<std::pair<int32_t, int32_t>> GgufTokenizer::get_pairs(
    const std::vector<int32_t>& ids,
    const std::map<std::pair<int32_t, int32_t>, int>& ranks) {

    std::set<std::pair<int32_t, int32_t>> unique;
    for (size_t i = 0; i + 1 < ids.size(); ++i) {
        auto key = std::make_pair(ids[i], ids[i + 1]);
        if (ranks.count(key)) unique.insert(key);
    }
    std::vector<std::pair<int32_t, int32_t>> result(unique.begin(), unique.end());
    std::sort(result.begin(), result.end(),
        [&](const auto& a, const auto& b) { return ranks.at(a) < ranks.at(b); });
    return result;
}

// ────────────────────────────────────────────────────────────────────────────
// BPE encode 单个词
// ────────────────────────────────────────────────────────────────────────────
std::vector<int32_t> GgufTokenizer::bpe_encode(const std::string& word) const {
    // 先尝试完整词查找
    auto it = token_map_.find(word);
    if (it != token_map_.end()) return {it->second};

    if (bpe_ranks_.empty() || word.empty()) {
        return {token_or_byte(word)};
    }

    // 初始化为字符级 token id 序列
    std::vector<int32_t> ids;
    for (size_t i = 0; i < word.size(); ) {
        unsigned char c = static_cast<unsigned char>(word[i]);
        size_t len;
        if (c < 0x80) len = 1;
        else if (c < 0xE0) len = 2;
        else if (c < 0xF0) len = 3;
        else len = 4;
        std::string ch = word.substr(i, len);
        int32_t id = token_or_byte(ch);
        ids.push_back(id);
        i += len;
    }

    if (ids.empty()) return {};

    // BPE merge 循环
    const int MAX_ITER = 100;
    for (int iter = 0; iter < MAX_ITER && ids.size() > 1; ++iter) {
        auto pairs = get_pairs(ids, bpe_ranks_);
        if (pairs.empty()) break;

        auto best = pairs[0];
        std::vector<int32_t> merged;
        for (size_t i = 0; i < ids.size(); ) {
            if (i + 1 < ids.size() && ids[i] == best.first && ids[i+1] == best.second) {
                // 查找合并 token
                std::string combined = tokens_[best.first] + tokens_[best.second];
                auto tok_it = token_map_.find(combined);
                if (tok_it != token_map_.end()) {
                    merged.push_back(tok_it->second);
                } else {
                    merged.push_back(ids[i]);
                    merged.push_back(ids[i+1]);
                }
                i += 2;
            } else {
                merged.push_back(ids[i]);
                i += 1;
            }
        }
        ids = std::move(merged);
    }

    return ids;
}

// ────────────────────────────────────────────────────────────────────────────
// build_special_token_strings — 构建特殊 token 字符串列表（按长度降序）
// 对齐官方 llama.cpp tokenizer_st_partition 的 cache_special_tokens
// ────────────────────────────────────────────────────────────────────────────
void GgufTokenizer::build_special_token_strings() const {
    if (special_strings_built_) return;
    special_token_strings_.clear();

    // 策略：如果 GGUF 提供了 token_types 元数据（CONTROL/USER_DEFINED/UNKNOWN），优先使用
    // 否则，从 token_map_ 中启发式扫描聊天标记特殊 token
    if (!special_ids_.empty()) {
        for (int32_t sid : special_ids_) {
            if (sid >= 0 && static_cast<size_t>(sid) < tokens_.size()) {
                const auto& tok_str = tokens_[sid];
                if (!tok_str.empty()) {
                    special_token_strings_.emplace_back(tok_str, sid);
                }
            }
        }
    } else {
        // 启发式：扫描所有 token，筛选聊天标记/思考标记等特殊 token
        // 匹配规则：以 '<' 开头且以 '>' 结尾（如 <|im_start|>、<s>、</s>、<think>、</think>），
        // 或者包含特殊 Unicode 符号（历史 Qwen3 模型曾使用 U+27C1/U+27FC/U+27F4 作思考标记，
        // 当前官方 Qwen3 已统一为 <think>/</think> ASCII 标记）
        for (const auto& [tok_str, tok_id] : token_map_) {
            if (tok_str.empty()) continue;
            // 规则1: HTML/XML 风格标记：<...>
            if (tok_str.size() >= 3 && tok_str.front() == '<' && tok_str.back() == '>') {
                special_token_strings_.emplace_back(tok_str, tok_id);
                continue;
            }
            // 规则2: 包含 Qwen3 思考标记 Unicode 符号（兼容性扫描）
            //   U+27C1 / U+27FC / U+27F4
            bool has_special_unicode = false;
            for (size_t ci = 0; ci < tok_str.size(); ++ci) {
                unsigned char uc = static_cast<unsigned char>(tok_str[ci]);
                if (uc >= 0xE0) { // 3+ 字节 UTF-8 序列
                    uint32_t cp = 0;
                    if ((uc & 0xF0) == 0xE0 && ci + 2 < tok_str.size()) {
                        cp = ((uint32_t)(uc & 0x0F) << 12) |
                             ((uint32_t)(tok_str[ci+1] & 0x3F) << 6) |
                             ((uint32_t)(tok_str[ci+2] & 0x3F));
                        if (cp == 0x27C1 || cp == 0x27FC || cp == 0x27F4) {
                            has_special_unicode = true;
                            break;
                        }
                    }
                }
            }
            if (has_special_unicode) {
                special_token_strings_.emplace_back(tok_str, tok_id);
            }
        }
        spdlog::debug("GgufTokenizer: heuristic scan found {} special token strings", special_token_strings_.size());
    }

    // 按长度降序排序，确保长 token 优先匹配（如 `<|im_start|>` 优先于 `<s>`）
    std::sort(special_token_strings_.begin(), special_token_strings_.end(),
        [](const auto& a, const auto& b) {
            if (a.first.size() != b.first.size())
                return a.first.size() > b.first.size();
            return a.first > b.first;
        });
    special_strings_built_ = true;
    spdlog::debug("GgufTokenizer: {} special token strings built", special_token_strings_.size());
}

// ────────────────────────────────────────────────────────────────────────────
// encode — 带特殊 token 分区的编码，对齐 llama.cpp tokenizer_st_partition
// ────────────────────────────────────────────────────────────────────────────
std::vector<int32_t> GgufTokenizer::encode(const std::string& text, bool add_special_tokens) const {
    std::vector<int32_t> result;
    if (add_special_tokens && add_bos_ && bos_id_ >= 0) {
        result.push_back(bos_id_);
    }

    // 1. 确保特殊 token 列表已构建
    build_special_token_strings();

    // 2. 按特殊 token 分区：扫描文本，分割出特殊 token 片段和普通文本片段
    //    算法：从左到右扫描，每次找最早出现的特殊 token
    struct Fragment {
        std::string text;
        bool is_special;
        int32_t token_id = -1;
    };
    std::vector<Fragment> fragments;

    std::string remaining = text;
    while (!remaining.empty()) {
        // 在所有特殊 token 中找最早匹配
        size_t best_pos = std::string::npos;
        size_t best_len = 0;
        int32_t best_id = -1;

        for (const auto& [tok_str, tok_id] : special_token_strings_) {
            size_t pos = remaining.find(tok_str);
            if (pos != std::string::npos) {
                if (best_pos == std::string::npos || pos < best_pos ||
                    (pos == best_pos && tok_str.size() > best_len)) {
                    best_pos = pos;
                    best_len = tok_str.size();
                    best_id = tok_id;
                }
            }
        }

        if (best_pos == std::string::npos) {
            // 没有更多特殊 token，剩余文本作为普通片段
            fragments.push_back({remaining, false, -1});
            break;
        }

        // 特殊 token 之前的普通文本
        if (best_pos > 0) {
            fragments.push_back({remaining.substr(0, best_pos), false, -1});
        }
        // 特殊 token 本身
        fragments.push_back({remaining.substr(best_pos, best_len), true, best_id});
        // 继续处理剩余文本
        remaining = remaining.substr(best_pos + best_len);
    }

    // 3. 编码各片段
    for (const auto& frag : fragments) {
        if (frag.is_special) {
            result.push_back(frag.token_id);
        } else {
            auto pieces = pre_tokenize(frag.text);
            for (const auto& piece : pieces) {
                auto ids = bpe_encode(piece);
                result.insert(result.end(), ids.begin(), ids.end());
            }
        }
    }

    return result;
}

// ────────────────────────────────────────────────────────────────────────────
// decode
// ────────────────────────────────────────────────────────────────────────────
std::string GgufTokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
    return decode(ids, skip_special, false);
}

std::string GgufTokenizer::decode(const std::vector<int32_t>& ids, bool skip_special,
                                   bool) const {
    std::string result;
    result.reserve(ids.size() * 4); // 粗略预分配

    for (int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) continue;
        if (skip_special && special_ids_.count(id)) continue;

        const std::string& token = tokens_[id];
        if (token.empty()) continue;

        // 检查是否是连续 <0xXX> 字节序列
        const auto* t = token.c_str();
        size_t tlen = token.size();
        bool all_bytes = (tlen % 6 == 0);
        if (all_bytes) {
            for (size_t i = 0; i < tlen; i += 6) {
                if (t[i] != '<' || t[i+1] != '0' || t[i+2] != 'x' || t[i+5] != '>') {
                    all_bytes = false;
                    break;
                }
            }
        }
        if (all_bytes && tlen > 0) {
            for (size_t i = 0; i < tlen; i += 6) {
                unsigned int b = 0;
                if (sscanf(t + i, "<0x%02x>", &b) == 1) {
                    result.push_back(static_cast<char>(b));
                }
            }
        } else {
            result += token;
        }
    }

    return result;
}

// ────────────────────────────────────────────────────────────────────────────
// 属性
// ────────────────────────────────────────────────────────────────────────────
int32_t GgufTokenizer::eos_token_id() const { return eos_id_; }
int32_t GgufTokenizer::pad_token_id() const { return pad_id_; }
int32_t GgufTokenizer::bos_token_id() const { return bos_id_; }
bool GgufTokenizer::has_bos() const { return add_bos_; }
int GgufTokenizer::vocab_size() const { return static_cast<int>(tokens_.size()); }

// ────────────────────────────────────────────────────────────────────────────
// 工厂函数
// ────────────────────────────────────────────────────────────────────────────
std::unique_ptr<Tokenizer> create_gguf_tokenizer(const GgufReader& reader) {
    return std::make_unique<GgufTokenizer>(reader);
}

} // namespace vm_c
