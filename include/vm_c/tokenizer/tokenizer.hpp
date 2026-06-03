#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>

namespace vm_c {

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual std::vector<int32_t> encode(const std::string& text, bool add_special_tokens = true) const = 0;
    virtual std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const = 0;
    virtual std::string decode(const std::vector<int32_t>& ids, bool skip_special, bool spaces_between_special_tokens) const = 0;
    virtual int32_t eos_token_id() const = 0;
    virtual int32_t pad_token_id() const = 0;
    virtual int32_t bos_token_id() const = 0;
    virtual bool has_bos() const = 0;
    virtual int vocab_size() const = 0;

    const std::string& chat_template() const { return chat_template_; }
    void set_chat_template(const std::string& tmpl) { chat_template_ = tmpl; }

protected:
    std::string chat_template_;
};

class SentencePieceTokenizer : public Tokenizer {
public:
    explicit SentencePieceTokenizer(const std::string& model_path);
    ~SentencePieceTokenizer() override;

    std::vector<int32_t> encode(const std::string& text, bool add_special_tokens = true) const override;
    std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const override;
    std::string decode(const std::vector<int32_t>& ids, bool skip_special, bool spaces_between_special_tokens) const override;
    int32_t eos_token_id() const override;
    int32_t pad_token_id() const override;
    int32_t bos_token_id() const override;
    bool has_bos() const override;
    int vocab_size() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class BPETokenizer : public Tokenizer {
public:
    explicit BPETokenizer(const std::string& model_dir);
    ~BPETokenizer() override;

    std::vector<int32_t> encode(const std::string& text, bool add_special_tokens = true) const override;
    std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const override;
    std::string decode(const std::vector<int32_t>& ids, bool skip_special, bool spaces_between_special_tokens) const override;
    int32_t eos_token_id() const override;
    int32_t pad_token_id() const override;
    int32_t bos_token_id() const override;
    bool has_bos() const override;
    int vocab_size() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<Tokenizer> create_tokenizer(const std::string& model_dir, const std::string& tokenizer_mode = "auto");

// From GGUF metadata (tokenizer data embedded in the model file)
class GgufReader;
std::unique_ptr<Tokenizer> create_gguf_tokenizer(const GgufReader& reader);

}
