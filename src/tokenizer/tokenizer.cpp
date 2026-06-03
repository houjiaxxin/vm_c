#include "vm_c/tokenizer/tokenizer.hpp"
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <nlohmann/json.hpp>

// 替换 C++17 std::filesystem → POSIX C API（兼容 GCC 7）
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

// 辅助：判断路径是否为目录
static bool fs_is_directory(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
// 辅助：判断路径是否存在
static bool fs_exists(const std::string& path) {
    return access(path.c_str(), F_OK) == 0;
}
// 辅助：列出目录下所有子路径
static std::vector<std::string> fs_list_directory(const std::string& path) {
    std::vector<std::string> entries;
    DIR* dir = opendir(path.c_str());
    if (!dir) return entries;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name(entry->d_name);
        if (name != "." && name != "..")
            entries.push_back(path + "/" + name);
    }
    closedir(dir);
    return entries;
}

namespace vm_c {

static std::unordered_map<uint32_t, uint8_t> build_unicode_to_bytes() {
    std::vector<int> bs;
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);

    std::unordered_map<uint32_t, uint8_t> u2b;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool in_bs = false;
        for (auto v : bs) { if (v == b) { in_bs = true; break; } }
        if (!in_bs) {
            u2b[256 + n] = static_cast<uint8_t>(b);
            n++;
        } else {
            u2b[static_cast<uint32_t>(b)] = static_cast<uint8_t>(b);
        }
    }
    return u2b;
}

static const std::unordered_map<uint32_t, uint8_t>& get_unicode_to_bytes() {
    static auto u2b = build_unicode_to_bytes();
    return u2b;
}

static std::string token_bytes_decode(const std::string& token) {
    std::string bytes;
    size_t i = 0;
    while (i < token.size()) {
        uint32_t cp = 0;
        uint8_t c = static_cast<uint8_t>(token[i]);
        if (c < 0x80) {
            cp = c;
            i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            if (i + 1 > token.size()) break;
            cp = (static_cast<uint32_t>(c & 0x1F) << 6) |
                 (static_cast<uint32_t>(token[i+1] & 0x3F));
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 > token.size()) break;
            cp = (static_cast<uint32_t>(c & 0x0F) << 12) |
                 (static_cast<uint32_t>(token[i+1] & 0x3F) << 6) |
                 (static_cast<uint32_t>(token[i+2] & 0x3F));
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            if (i + 3 > token.size()) break;
            cp = (static_cast<uint32_t>(c & 0x07) << 18) |
                 (static_cast<uint32_t>(token[i+1] & 0x3F) << 12) |
                 (static_cast<uint32_t>(token[i+2] & 0x3F) << 6) |
                 (static_cast<uint32_t>(token[i+3] & 0x3F));
            i += 4;
        } else {
            i += 1;
            continue;
        }

        auto& u2b = get_unicode_to_bytes();
        auto it = u2b.find(cp);
        if (it != u2b.end()) {
            bytes += static_cast<char>(it->second);
        } else {
            if (cp < 0x80) {
                bytes += static_cast<char>(cp);
            } else if (cp < 0x800) {
                bytes += static_cast<char>(0xC0 | (cp >> 6));
                bytes += static_cast<char>(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
                bytes += static_cast<char>(0xE0 | (cp >> 12));
                bytes += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                bytes += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
                bytes += static_cast<char>(0xF0 | (cp >> 18));
                bytes += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                bytes += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                bytes += static_cast<char>(0x80 | (cp & 0x3F));
            }
        }
    }
    return bytes;
}

static std::string json_to_string(const nlohmann::json& val) {
    if (val.is_string()) return val.get<std::string>();
    if (val.is_array() && !val.empty() && val[0].is_string()) return val[0].get<std::string>();
    return "";
}

struct BPETokenizer::Impl {
    struct MergeRule {
        std::string left;
        std::string right;
    };

    std::unordered_map<std::string, int32_t> vocab;
    std::unordered_map<int32_t, std::string> id_to_token;
    std::vector<MergeRule> merges;
    std::unordered_map<std::string, int> merge_rank;

    std::string unk_token;
    int32_t unk_id = 0;
    std::string eos_token;
    int32_t eos_id = -1;
    std::string bos_token;
    int32_t bos_id = -1;
    std::string pad_token;
    int32_t pad_id = -1;
    bool use_end_of_word = false;
    bool has_bos = false;

    std::unordered_map<std::string, std::string> special_tokens;

    std::string byte_fallback_encode(uint8_t b) const {
        char buf[8];
        snprintf(buf, sizeof(buf), "<0x%02X>", b);
        return buf;
    }

    std::vector<std::string> pre_tokenize(const std::string& text) const {
        std::vector<std::string> tokens;
        std::string current;

        auto flush = [&]() {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        };

        for (size_t i = 0; i < text.size(); ) {
            unsigned char c = static_cast<unsigned char>(text[i]);

            if (c == ' ') {
                flush();
                if (!tokens.empty()) {
                    tokens.back() += ' ';
                } else {
                    tokens.push_back(" ");
                }
                i++;
            } else if (c < 0x80 && (std::ispunct(c) || c == '\n' || c == '\r' || c == '\t')) {
                flush();
                tokens.push_back(std::string(1, c));
                i++;
            } else if (c >= 0xC0 && c < 0xE0) {
                flush();
                if (i + 1 < text.size()) {
                    tokens.push_back(text.substr(i, 2));
                    i += 2;
                } else { i++; }
            } else if (c >= 0xE0 && c < 0xF0) {
                flush();
                if (i + 2 < text.size()) {
                    tokens.push_back(text.substr(i, 3));
                    i += 3;
                } else { i++; }
            } else if (c >= 0xF0) {
                flush();
                if (i + 3 < text.size()) {
                    tokens.push_back(text.substr(i, 4));
                    i += 4;
                } else { i++; }
            } else {
                current += text[i];
                i++;
            }
        }
        flush();
        return tokens;
    }

    std::string apply_bpe(const std::string& word, bool use_end_of_word) const {
        if (word.size() <= 1) return word;

        std::vector<std::string> symbols;
        for (size_t i = 0; i < word.size(); i++) {
            symbols.push_back(word.substr(i, 1));
        }
        if (!symbols.empty() && use_end_of_word) {
            symbols.back() += "</w>";
        }

        while (symbols.size() > 1) {
            int best_rank = std::numeric_limits<int>::max();
            int best_idx = -1;

            for (size_t i = 0; i + 1 < symbols.size(); i++) {
                std::string pair = symbols[i] + " " + symbols[i + 1];
                auto it = merge_rank.find(pair);
                if (it != merge_rank.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = static_cast<int>(i);
                }
            }

            if (best_idx < 0) break;

            symbols[best_idx] = symbols[best_idx] + symbols[best_idx + 1];
            symbols.erase(symbols.begin() + best_idx + 1);
        }

        std::string result;
        for (size_t i = 0; i < symbols.size(); i++) {
            if (i > 0) result += " ";
            result += symbols[i];
        }
        return result;
    }
};

BPETokenizer::BPETokenizer(const std::string& model_dir)
    : impl_(std::make_unique<Impl>()) {
    std::string tokenizer_path = model_dir + "/tokenizer.json";

    bool loaded_from_gpt2 = false;

    if (fs_exists(tokenizer_path)) {
        std::ifstream f(tokenizer_path);
        if (!f.is_open()) {
            spdlog::error("Cannot open tokenizer file: {}", tokenizer_path);
            throw std::runtime_error("Cannot open tokenizer: " + tokenizer_path);
        }

        try {
            nlohmann::json j;
            f >> j;

            if (j.contains("model") && j["model"].contains("vocab")) {
                for (auto& [token, id] : j["model"]["vocab"].items()) {
                    if (id.is_number()) {
                        impl_->vocab[token] = id.get<int32_t>();
                        impl_->id_to_token[id.get<int32_t>()] = token;
                    }
                }
            }

            if (j.contains("model") && j["model"].contains("merges")) {
                int rank = 0;
                for (auto& merge : j["model"]["merges"]) {
                    std::string left, right;
                    if (merge.is_array() && merge.size() == 2) {
                        left = merge[0].get<std::string>();
                        right = merge[1].get<std::string>();
                    } else if (merge.is_string()) {
                        std::string merge_str = merge.get<std::string>();
                        size_t space = merge_str.find(' ');
                        if (space == std::string::npos) continue;
                        left = merge_str.substr(0, space);
                        right = merge_str.substr(space + 1);
                    } else {
                        continue;
                    }
                    Impl::MergeRule rule;
                    rule.left = left;
                    rule.right = right;
                    impl_->merges.push_back(rule);
                    impl_->merge_rank[left + " " + right] = rank++;
                }

                impl_->use_end_of_word = false;
                for (auto& mr : impl_->merges) {
                    if (mr.left.find("</w>") != std::string::npos ||
                        mr.right.find("</w>") != std::string::npos) {
                        impl_->use_end_of_word = true;
                        break;
                    }
                }
            }

            if (j.contains("added_tokens")) {
                for (auto& tok : j["added_tokens"]) {
                    std::string content = json_to_string(tok["content"]);
                    if (content.empty()) continue;
                    int32_t id = tok["id"];
                    impl_->vocab[content] = id;
                    impl_->id_to_token[id] = content;
                    impl_->special_tokens[content] = content;
                }
            }

            if (j.contains("model") && j["model"].contains("unk_token")) {
                impl_->unk_token = json_to_string(j["model"]["unk_token"]);
            }

            if (impl_->vocab.count(impl_->eos_token)) {
                impl_->eos_id = impl_->vocab[impl_->eos_token];
            }
            if (impl_->vocab.count(impl_->bos_token)) {
                impl_->bos_id = impl_->vocab[impl_->bos_token];
            }
            if (impl_->vocab.count(impl_->unk_token)) {
                impl_->unk_id = impl_->vocab[impl_->unk_token];
            }

            spdlog::info("BPE tokenizer loaded: vocab_size={}, merges={}",
                         impl_->vocab.size(), impl_->merges.size());

        } catch (const std::exception& e) {
            spdlog::warn("Partial tokenizer.json parse: {} (continuing with {} tokens)", e.what(), impl_->vocab.size());
        }
    } else {
        std::string vocab_path = model_dir + "/vocab.json";
        std::string merges_path = model_dir + "/merges.txt";

        if (!fs_exists(vocab_path) || !fs_exists(merges_path)) {
            spdlog::error("Neither tokenizer.json nor vocab.json+merges.txt found in: {}", model_dir);
            throw std::runtime_error("No BPE tokenizer files found in: " + model_dir);
        }

        {
            std::ifstream vf(vocab_path);
            nlohmann::json vocab_json;
            vf >> vocab_json;

            if (vocab_json.is_object()) {
                for (auto& [token, id] : vocab_json.items()) {
                    if (id.is_number()) {
                        impl_->vocab[token] = id.get<int32_t>();
                        impl_->id_to_token[id.get<int32_t>()] = token;
                    }
                }
            }
        }

        {
            std::ifstream mf(merges_path);
            std::string line;
            int rank = 0;

            if (std::getline(mf, line) && line.find("#version") == 0) {
            }

            while (std::getline(mf, line)) {
                if (line.empty()) continue;
                size_t space = line.find(' ');
                if (space == std::string::npos) continue;
                std::string left = line.substr(0, space);
                std::string right = line.substr(space + 1);

                Impl::MergeRule rule;
                rule.left = left;
                rule.right = right;
                impl_->merges.push_back(rule);
                impl_->merge_rank[left + " " + right] = rank++;
            }

            impl_->use_end_of_word = false;
            for (auto& mr : impl_->merges) {
                if (mr.left.find("</w>") != std::string::npos ||
                    mr.right.find("</w>") != std::string::npos) {
                    impl_->use_end_of_word = true;
                    break;
                }
            }
        }

        loaded_from_gpt2 = true;
        spdlog::info("GPT-2 style BPE tokenizer loaded: vocab_size={}, merges={}",
                     impl_->vocab.size(), impl_->merges.size());
    }

    std::string config_path = model_dir;
    if (fs_is_directory(model_dir)) {
        config_path = model_dir + "/tokenizer_config.json";
    }
    if (fs_exists(config_path)) {
        std::ifstream cf(config_path);
        try {
            nlohmann::json config;
            cf >> config;
            if (config.contains("eos_token")) {
                if (!config["eos_token"].is_null()) {
                    std::string eos = json_to_string(config["eos_token"]);
                    if (!eos.empty()) {
                        if (impl_->vocab.count(eos)) impl_->eos_id = impl_->vocab[eos];
                        impl_->eos_token = eos;
                    }
                }
            }
            if (config.contains("bos_token")) {
                if (!config["bos_token"].is_null()) {
                    std::string bos = json_to_string(config["bos_token"]);
                    if (!bos.empty()) {
                        if (impl_->vocab.count(bos)) impl_->bos_id = impl_->vocab[bos];
                        impl_->bos_token = bos;
                        impl_->has_bos = true;
                    }
                }
            }
            if (config.contains("pad_token")) {
                if (!config["pad_token"].is_null()) {
                    std::string pad = json_to_string(config["pad_token"]);
                    if (!pad.empty()) {
                        if (impl_->vocab.count(pad)) impl_->pad_id = impl_->vocab[pad];
                        impl_->pad_token = pad;
                    }
                }
            }
            if (config.contains("chat_template") && config["chat_template"].is_string()) {
                chat_template_ = config["chat_template"].get<std::string>();
                spdlog::info("Loaded chat_template from tokenizer_config.json ({} chars)", chat_template_.size());
            }
        } catch (...) {}
    }

    if (chat_template_.empty()) {
        std::string jinja_path = model_dir;
        if (fs_is_directory(model_dir)) {
            jinja_path = model_dir + "/chat_template.jinja";
        }
        if (fs_exists(jinja_path)) {
            std::ifstream jf(jinja_path);
            if (jf.is_open()) {
                std::ostringstream ss;
                ss << jf.rdbuf();
                chat_template_ = ss.str();
                spdlog::info("Loaded chat_template from chat_template.jinja ({} chars)", chat_template_.size());
            }
        }
    }

}

BPETokenizer::~BPETokenizer() = default;

std::vector<int32_t> BPETokenizer::encode(const std::string& text, bool add_special_tokens) const {
    std::vector<int32_t> ids;

    if (add_special_tokens && impl_->has_bos && impl_->bos_id >= 0) {
        ids.push_back(impl_->bos_id);
    }

    std::vector<std::pair<std::string, bool>> segments;
    std::string remaining = text;

    while (!remaining.empty()) {
        size_t best_pos = std::string::npos;
        size_t best_len = 0;
        std::string best_token;

        for (auto& [pattern, token_str] : impl_->special_tokens) {
            size_t pos = remaining.find(pattern);
            if (pos != std::string::npos && pattern.size() > 0) {
                if (best_pos == std::string::npos || pos < best_pos ||
                    (pos == best_pos && pattern.size() > best_len)) {
                    best_pos = pos;
                    best_len = pattern.size();
                    best_token = token_str;
                }
            }
        }

        if (best_pos == std::string::npos) {
            segments.emplace_back(remaining, false);
            break;
        }

        if (best_pos > 0) {
            segments.emplace_back(remaining.substr(0, best_pos), false);
        }
        segments.emplace_back(best_token, true);
        remaining = remaining.substr(best_pos + best_len);
    }

    for (auto& [segment, is_special] : segments) {
        if (is_special) {
            auto vit = impl_->vocab.find(segment);
            if (vit != impl_->vocab.end()) {
                ids.push_back(vit->second);
                continue;
            }
        }

        auto words = impl_->pre_tokenize(segment);
        for (auto& word : words) {
            auto it = impl_->vocab.find(word);
            if (it != impl_->vocab.end()) {
                ids.push_back(it->second);
                continue;
            }

            std::string bpe_result = impl_->apply_bpe(word, impl_->use_end_of_word);
            std::istringstream iss(bpe_result);
            std::string token;
            while (iss >> token) {
                auto tit = impl_->vocab.find(token);
                if (tit != impl_->vocab.end()) {
                    ids.push_back(tit->second);
                } else {
                    for (unsigned char c : token) {
                        std::string fb = impl_->byte_fallback_encode(c);
                        auto fit = impl_->vocab.find(fb);
                        if (fit != impl_->vocab.end()) {
                            ids.push_back(fit->second);
                        } else {
                            ids.push_back(impl_->unk_id);
                        }
                    }
                }
            }
        }
    }

    return ids;
}

std::string BPETokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
    return decode(ids, skip_special, false);
}

std::string BPETokenizer::decode(const std::vector<int32_t>& ids, bool skip_special, bool spaces_between_special_tokens) const {
    std::string result;
    bool prev_was_special = false;
    for (auto id : ids) {
        auto it = impl_->id_to_token.find(id);
        if (it == impl_->id_to_token.end()) continue;

        const std::string& token = it->second;
        bool is_special = impl_->special_tokens.count(token) > 0;

        if (skip_special && is_special) {
            prev_was_special = true;
            continue;
        }

        if (spaces_between_special_tokens && prev_was_special && !is_special && !result.empty()) {
            result += " ";
        }
        prev_was_special = is_special;

        if (token.size() == 8 && token.substr(0, 4) == "<0x" && token.back() == '>') {
            unsigned int byte_val;
            if (sscanf(token.c_str(), "<0x%02X>", &byte_val) == 1) {
                result += static_cast<char>(byte_val);
            }
            continue;
        }

        result += token_bytes_decode(token);
    }
    return result;
}

int32_t BPETokenizer::eos_token_id() const { return impl_->eos_id; }
int32_t BPETokenizer::pad_token_id() const { return impl_->pad_id; }
int32_t BPETokenizer::bos_token_id() const { return impl_->bos_id; }
bool BPETokenizer::has_bos() const { return impl_->has_bos; }
int BPETokenizer::vocab_size() const { return static_cast<int>(impl_->vocab.size()); }

struct SentencePieceTokenizer::Impl {
    std::unordered_map<std::string, int32_t> vocab;
    std::unordered_map<int32_t, std::string> id_to_token;
    int32_t unk_id = 0;
    int32_t eos_id = 2;
    int32_t bos_id = 1;
    int32_t pad_id = 0;
    std::string unk_token = "<unk>";
    std::string eos_token = "</s>";
    std::string bos_token = "<s>";
};

SentencePieceTokenizer::SentencePieceTokenizer(const std::string& model_path)
    : impl_(std::make_unique<Impl>()) {
    std::ifstream f(model_path, std::ios::binary);
    if (!f.is_open()) {
        spdlog::error("Cannot open SentencePiece model: {}", model_path);
        throw std::runtime_error("Cannot open SentencePiece model: " + model_path);
    }

    spdlog::info("SentencePiece tokenizer loaded from: {}", model_path);
    spdlog::warn("SentencePiece tokenizer uses stub - link libsentencepiece for full support");
}

SentencePieceTokenizer::~SentencePieceTokenizer() = default;

std::vector<int32_t> SentencePieceTokenizer::encode(const std::string& text, bool add_special_tokens) const {
    std::vector<int32_t> ids;
    if (add_special_tokens) {
        ids.push_back(impl_->bos_id);
    }
    for (char c : text) {
        std::string token(1, c);
        auto it = impl_->vocab.find(token);
        if (it != impl_->vocab.end()) {
            ids.push_back(it->second);
        } else {
            char buf[8];
            snprintf(buf, sizeof(buf), "<0x%02X>", static_cast<unsigned char>(c));
            std::string fb(buf);
            auto fit = impl_->vocab.find(fb);
            if (fit != impl_->vocab.end()) {
                ids.push_back(fit->second);
            } else {
                ids.push_back(impl_->unk_id);
            }
        }
    }
    return ids;
}

std::string SentencePieceTokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
    return decode(ids, skip_special, false);
}

std::string SentencePieceTokenizer::decode(const std::vector<int32_t>& ids, bool skip_special, bool spaces_between_special_tokens) const {
    std::string result;
    bool prev_was_special = false;
    for (auto id : ids) {
        auto it = impl_->id_to_token.find(id);
        if (it == impl_->id_to_token.end()) continue;
        const std::string& token = it->second;
        bool is_special = (token == impl_->bos_token || token == impl_->eos_token);

        if (skip_special && is_special) {
            prev_was_special = true;
            continue;
        }

        if (spaces_between_special_tokens && prev_was_special && !is_special && !result.empty()) {
            result += " ";
        }
        prev_was_special = is_special;

        if (token.size() == 8 && token.substr(0, 4) == "<0x" && token.back() == '>') {
            unsigned int byte_val;
            if (sscanf(token.c_str(), "<0x%02X>", &byte_val) == 1) {
                result += static_cast<char>(byte_val);
            }
            continue;
        }

        std::string decoded = token;
        const char spm_space[] = "\xe2\x96\x81";
        size_t pos = decoded.find(spm_space);
        while (pos != std::string::npos) {
            decoded.replace(pos, 3, " ");
            pos = decoded.find(spm_space, pos + 1);
        }
        if (decoded.size() == 8 && decoded.substr(0, 4) == "<0x" && decoded.back() == '>') {
            unsigned int byte_val;
            if (sscanf(decoded.c_str(), "<0x%02X>", &byte_val) == 1) {
                decoded = std::string(1, static_cast<char>(byte_val));
            }
        }
        result += decoded;
    }
    return result;
}

int32_t SentencePieceTokenizer::eos_token_id() const { return impl_->eos_id; }
int32_t SentencePieceTokenizer::pad_token_id() const { return impl_->pad_id; }
int32_t SentencePieceTokenizer::bos_token_id() const { return impl_->bos_id; }
bool SentencePieceTokenizer::has_bos() const { return true; }
int SentencePieceTokenizer::vocab_size() const { return static_cast<int>(impl_->vocab.size()); }

std::unique_ptr<Tokenizer> create_tokenizer(const std::string& model_dir, const std::string& tokenizer_mode) {
    auto check_file = [&](const std::string& subpath) -> bool {
        return fs_exists(model_dir + "/" + subpath);
    };

    auto try_load_sp = [&]() -> std::unique_ptr<Tokenizer> {
        if (check_file("tokenizer.model")) {
            spdlog::info("Using SentencePiece tokenizer (tokenizer.model)");
            return std::make_unique<SentencePieceTokenizer>(model_dir + "/tokenizer.model");
        }
        return nullptr;
    };

    auto try_load_bpe = [&]() -> std::unique_ptr<Tokenizer> {
        if (check_file("tokenizer.json")) {
            spdlog::info("Using BPE tokenizer (tokenizer.json)");
            return std::make_unique<BPETokenizer>(model_dir);
        }
        return nullptr;
    };

    auto try_load_gpt2 = [&]() -> std::unique_ptr<Tokenizer> {
        if (check_file("vocab.json") && check_file("merges.txt")) {
            spdlog::info("Using GPT-2 style tokenizer (vocab.json + merges.txt)");
            return std::make_unique<BPETokenizer>(model_dir);
        }
        return nullptr;
    };

    if (tokenizer_mode == "sentencepiece") {
        if (auto tok = try_load_sp()) return tok;
        spdlog::error("tokenizer_mode=sentencepiece but no tokenizer.model found in: {}", model_dir);
        throw std::runtime_error(
            "SentencePiece tokenizer not found in: " + model_dir +
            "\n  Expected file: tokenizer.model"
            "\n  Hint: use --tokenizer <path> to specify a different tokenizer directory");
    }

    if (tokenizer_mode == "hf" || tokenizer_mode == "bpe") {
        if (auto tok = try_load_bpe()) return tok;
        if (auto tok = try_load_gpt2()) return tok;
        spdlog::error("tokenizer_mode={} but no tokenizer.json or vocab.json+merges.txt found in: {}", tokenizer_mode, model_dir);
        throw std::runtime_error(
            "BPE tokenizer not found in: " + model_dir +
            "\n  Expected files: tokenizer.json, or vocab.json + merges.txt"
            "\n  Hint: use --tokenizer <path> to specify a different tokenizer directory");
    }

    if (auto tok = try_load_sp()) return tok;
    if (auto tok = try_load_bpe()) return tok;
    if (auto tok = try_load_gpt2()) return tok;

    std::vector<std::string> found_files;
    if (fs_exists(model_dir) && fs_is_directory(model_dir)) {
        for (const auto& full_path : fs_list_directory(model_dir)) {
            struct stat st;
            if (stat(full_path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
                // 提取最后一段文件名
                auto pos = full_path.rfind('/');
                std::string name = (pos != std::string::npos) ? full_path.substr(pos + 1) : full_path;
                if (name.find("tokenizer") != std::string::npos ||
                    name.find("vocab") != std::string::npos ||
                    name.find("merges") != std::string::npos ||
                    name.find("spm") != std::string::npos ||
                    name.find("bpe") != std::string::npos) {
                    found_files.push_back(name);
                }
            }
        }
    }

    std::string hint;
    if (!found_files.empty()) {
        hint = "\n  Found related files (but none matched expected formats):";
        for (auto& f : found_files) {
            hint += "\n    - " + f;
        }
    }

    spdlog::error("No tokenizer found in: {}", model_dir);
    throw std::runtime_error(
        "No tokenizer found in: " + model_dir +
        "\n  Supported formats:"
        "\n    - tokenizer.model  (SentencePiece)"
        "\n    - tokenizer.json   (HuggingFace BPE)"
        "\n    - vocab.json + merges.txt (GPT-2 style BPE)"
        "\n  Hint: use --tokenizer <path> to specify a different tokenizer directory"
        "\n  Hint: use --tokenizer-mode {auto|sentencepiece|hf} to force a specific mode" +
        hint);
}

}
