#pragma once

#include "vm_c/tokenizer/tokenizer.hpp"
#include "vm_c/core/gguf_reader.hpp"
#include <unordered_map>
#include <map>
#include <set>

namespace vm_c {

// GGUF 内嵌 tokenizer — 从 GGUF 元数据直接读取词汇表与 BPE merges
class GgufTokenizer : public Tokenizer {
public:
    // 从已加载的 GgufReader 构建
    explicit GgufTokenizer(const GgufReader& reader);

    ~GgufTokenizer() override = default;

    std::vector<int32_t> encode(const std::string& text, bool add_special_tokens = true) const override;
    std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const override;
    std::string decode(const std::vector<int32_t>& ids, bool skip_special,
                       bool spaces_between_special_tokens) const override;
    int32_t eos_token_id() const override;
    int32_t pad_token_id() const override;
    int32_t bos_token_id() const override;
    bool has_bos() const override;
    int vocab_size() const override;

private:
    // 词汇表
    std::vector<std::string>  tokens_;
    std::vector<float>        scores_;
    std::vector<int32_t>      token_types_;
    int32_t bos_id_ = 1;
    int32_t eos_id_ = 2;
    int32_t pad_id_ = 0;
    bool    add_bos_ = true;

    // tokenizer 类型: "llama", "gpt2", "bert", "qwen2"
    std::string model_type_;

    // BPE 数据
    // merge 排名: pair(left_id, right_id) → priority
    std::map<std::pair<int32_t, int32_t>, int> bpe_ranks_;

    // token string → token_id (用于编码)
    std::unordered_map<std::string, int32_t> token_map_;

    // bytes → unicode 编码表 (GPT-2 BPE)
    std::vector<int32_t> bytes_to_unicode_;
    std::unordered_map<int32_t, uint8_t> unicode_to_bytes_;

    // 特殊 token 集合
    std::set<int32_t> special_ids_;

    // 特殊 token 字符串 → token_id（按长度降序，用于 encode 分区）
    // 对齐 llama.cpp 的 tokenizer_st_partition
    mutable std::vector<std::pair<std::string, int32_t>> special_token_strings_;
    mutable bool special_strings_built_ = false;
    void build_special_token_strings() const;

    // ── 初始化 ──
    void init_bytes_to_unicode();
    void build_token_map();
    void build_bpe_ranks(const GgufReader& reader);

    // ── BPE 编码 ──
    // 对单个"词"应用 BPE merge，返回 token id 列表
    std::vector<int32_t> bpe_encode(const std::string& word) const;

    // 获取 byte 的 unicode 编码
    std::string byte_to_unicode(uint8_t b) const;

    // 预分词
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    // 将 token 文本转换为 id（含 byte-fallback）
    int32_t token_or_byte(const std::string& piece) const;

    // 统计工具
    static std::vector<std::pair<int32_t, int32_t>> get_pairs(
        const std::vector<int32_t>& ids,
        const std::map<std::pair<int32_t, int32_t>, int>& ranks);
};

// 从 GgufReader 构建 GgufTokenizer 的便捷函数
std::unique_ptr<Tokenizer> create_gguf_tokenizer(const GgufReader& reader);

} // namespace vm_c
