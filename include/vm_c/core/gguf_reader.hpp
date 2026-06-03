#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <cstring>

#include "ggml.h"

#include "vm_c/core/tensor.hpp"
#include "vm_c/core/config.hpp"

namespace vm_c {

// GGUF format constants
// NOTE: prefixed with VM_C_ to avoid collision with ggml's gguf.h macros (GGUF_MAGIC, GGUF_VERSION, etc.)
constexpr uint32_t VM_C_GGUF_MAGIC = 0x46554747; // "GGUF" little-endian
constexpr uint32_t VM_C_GGUF_VERSION = 3;
constexpr size_t VM_C_GGUF_DEFAULT_ALIGNMENT = 32;

// GGUF value types
enum class GgufType : int32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// 与 third_party/vm_c_official/ggml/include/ggml.h 中 ggml_type 数值一致（GGML_TYPE_BF16=30）
using GgmlType = ggml_type;

// Known GGUF metadata keys
struct GgufKeys {
    // General
    static constexpr const char* GENERAL_ALIGNMENT   = "general.alignment";
    static constexpr const char* GENERAL_NAME         = "general.name";
    static constexpr const char* GENERAL_DESCRIPTION  = "general.description";
    static constexpr const char* GENERAL_FILE_TYPE    = "general.file_type";
    static constexpr const char* GENERAL_SOURCE_HF_REPO = "general.source.huggingface.repository";
    static constexpr const char* GENERAL_ARCHITECTURE = "general.architecture";

    // Context length
    static constexpr const char* CONTEXT_LENGTH = "llama.context_length";

    // Attention
    static constexpr const char* ATTENTION_HEAD_COUNT       = "llama.attention.head_count";
    static constexpr const char* ATTENTION_HEAD_COUNT_KV    = "llama.attention.head_count_kv";
    static constexpr const char* ATTENTION_LAYERNORM_RMS_EPS = "llama.attention.layer_norm_rms_epsilon";
    static constexpr const char* ATTENTION_KEY_LENGTH       = "llama.attention.key_length";

    // RoPE
    static constexpr const char* ROPE_FREQ_BASE  = "llama.rope.freq_base";
    static constexpr const char* ROPE_SCALING_TYPE = "llama.rope.scaling.type";
    static constexpr const char* ROPE_SCALING_FACTOR = "llama.rope.scaling.factor";
    static constexpr const char* ROPE_SCALING_ORIG_CTX = "llama.rope.scaling.original_context_length";
    static constexpr const char* ROPE_DIMENSION_COUNT = "llama.rope.dimension_count";

    // FFN
    static constexpr const char* FEED_FORWARD_LENGTH   = "llama.feed_forward_length";
    static constexpr const char* EXPERT_FEED_FORWARD_LENGTH = "llama.expert_feed_forward_length";

    // MoE
    static constexpr const char* EXPERT_COUNT         = "llama.expert_count";
    static constexpr const char* EXPERT_USED_COUNT    = "llama.expert_used_count";
    static constexpr const char* EXPERT_SHARED_COUNT  = "llama.expert_shared_count";
    static constexpr const char* EXPERT_SHARED_FEED_FORWARD_LENGTH = "llama.expert_shared_feed_forward_length";
    static constexpr const char* EXPERT_WEIGHTS_SCALE = "llama.expert_weights_scale";
    static constexpr const char* EXPERT_GATING_FUNC   = "llama.expert_gating_func";
    static constexpr const char* EXPERT_GROUP_COUNT   = "llama.expert_group_count";
    static constexpr const char* EXPERT_GROUP_USED_COUNT = "llama.expert_group_used_count";
    static constexpr const char* EXPERT_NORM          = "llama.expert.norm";

    // MTP / NextN
    static constexpr const char* NEXTN_PREDICT_LAYERS = "llama.nextn_predict_layers";

    // Full-attention interval (Qwen3.5 hybrid attention)
    static constexpr const char* FULL_ATTENTION_INTERVAL = "llama.full_attention_interval";

    // SSM / Gated Delta Net (Qwen35MoE linear attention)
    static constexpr const char* SSM_CONV_KERNEL    = "llama.ssm.conv_kernel";
    static constexpr const char* SSM_INNER_SIZE      = "llama.ssm.inner_size";
    static constexpr const char* SSM_STATE_SIZE      = "llama.ssm.state_size";
    static constexpr const char* SSM_TIME_STEP_RANK  = "llama.ssm.time_step_rank";
    static constexpr const char* SSM_GROUP_COUNT     = "llama.ssm.group_count";

    // Block count
    static constexpr const char* BLOCK_COUNT = "llama.block_count";

    // Embedding
    static constexpr const char* EMBEDDING_LENGTH = "llama.embedding_length";

    // Vocabulary
    static constexpr const char* VOCABULARY_SIZE     = "llama.vocab_size";
    static constexpr const char* VOCABULARY_TOKENIZER = "tokenizer.ggml.model";
    static constexpr const char* TOKENIZER_LIST       = "tokenizer.ggml.tokens";
    static constexpr const char* TOKENIZER_SCORES     = "tokenizer.ggml.scores";
    static constexpr const char* TOKENIZER_TOKEN_TYPES = "tokenizer.ggml.token_type";
    static constexpr const char* TOKENIZER_BOS_ID     = "tokenizer.ggml.bos_token_id";
    static constexpr const char* TOKENIZER_EOS_ID     = "tokenizer.ggml.eos_token_id";
    static constexpr const char* TOKENIZER_PAD_ID     = "tokenizer.ggml.pad_token_id";
    static constexpr const char* TOKENIZER_ADD_BOS    = "tokenizer.ggml.add_bos_token";
    static constexpr const char* TOKENIZER_ADD_EOS    = "tokenizer.ggml.add_eos_token";
    static constexpr const char* TOKENIZER_CHAT_TEMPLATE = "tokenizer.chat_template";
    static constexpr const char* TOKENIZER_PREFIX_ID  = "tokenizer.ggml.prefix_token_id";
    static constexpr const char* TOKENIZER_SUFFIX_ID  = "tokenizer.ggml.suffix_token_id";
    static constexpr const char* TOKENIZER_MIDDLE_ID  = "tokenizer.ggml.middle_token_id";

    // Lora
    static constexpr const char* LORA_ATTENUATION_FACTOR = "general.lora.attenuation_factor";
};

#include "vm_c/core/iq4xs_types.hpp"

static_assert(sizeof(BlockIQ4XS) == sizeof(uint16_t) + sizeof(uint16_t) + QK_K/64 + QK_K/2,
              "wrong iq4_xs block size/padding");

// Generic GGUF tensor info（GgmlType 与 libllama ggml_type 对齐）
struct GgufTensorInfo {
    std::string name;
    int32_t n_dims;
    int64_t ne[4];
    GgmlType type;
    uint64_t offset;
};

// GGUF KV value
struct GgufKvValue {
    GgufType type;
    bool is_array = false;
    GgufType array_type = GgufType::UINT8;
    std::vector<uint8_t> data;
    std::vector<std::string> strings;

    template<typename T>
    T get_scalar(size_t i = 0) const {
        const T* ptr = reinterpret_cast<const T*>(data.data());
        return ptr[i];
    }

    // 类型安全的 int64 读取：根据实际 GgufType 用正确字节数读取，避免 uint32→int64 缓冲区越界
    int64_t get_int64_safe() const {
        switch (type) {
            case GgufType::UINT8:   return static_cast<int64_t>(get_scalar<uint8_t>());
            case GgufType::INT8:    return static_cast<int64_t>(get_scalar<int8_t>());
            case GgufType::UINT16:  return static_cast<int64_t>(get_scalar<uint16_t>());
            case GgufType::INT16:   return static_cast<int64_t>(get_scalar<int16_t>());
            case GgufType::UINT32:  return static_cast<int64_t>(get_scalar<uint32_t>());
            case GgufType::INT32:   return static_cast<int64_t>(get_scalar<int32_t>());
            case GgufType::UINT64:  return static_cast<int64_t>(get_scalar<uint64_t>());
            case GgufType::INT64:   return get_scalar<int64_t>();
            case GgufType::FLOAT32: return static_cast<int64_t>(get_scalar<float>());
            case GgufType::FLOAT64: return static_cast<int64_t>(get_scalar<double>());
            default:                return 0;
        }
    }

    uint32_t get_uint32_safe() const {
        switch (type) {
            case GgufType::UINT8:   return get_scalar<uint8_t>();
            case GgufType::INT8:    return static_cast<uint32_t>(get_scalar<int8_t>());
            case GgufType::UINT16:  return get_scalar<uint16_t>();
            case GgufType::INT16:   return static_cast<uint32_t>(get_scalar<int16_t>());
            case GgufType::UINT32:  return get_scalar<uint32_t>();
            case GgufType::INT32:   return static_cast<uint32_t>(get_scalar<int32_t>());
            case GgufType::UINT64:  return static_cast<uint32_t>(get_scalar<uint64_t>());
            case GgufType::INT64:   return static_cast<uint32_t>(get_scalar<int64_t>());
            case GgufType::FLOAT32: return static_cast<uint32_t>(get_scalar<float>());
            case GgufType::FLOAT64: return static_cast<uint32_t>(get_scalar<double>());
            default:                return 0;
        }
    }

    float get_float32_safe() const {
        switch (type) {
            case GgufType::UINT8:   return static_cast<float>(get_scalar<uint8_t>());
            case GgufType::INT8:    return static_cast<float>(get_scalar<int8_t>());
            case GgufType::UINT16:  return static_cast<float>(get_scalar<uint16_t>());
            case GgufType::INT16:   return static_cast<float>(get_scalar<int16_t>());
            case GgufType::UINT32:  return static_cast<float>(get_scalar<uint32_t>());
            case GgufType::INT32:   return static_cast<float>(get_scalar<int32_t>());
            case GgufType::UINT64:  return static_cast<float>(get_scalar<uint64_t>());
            case GgufType::INT64:   return static_cast<float>(get_scalar<int64_t>());
            case GgufType::FLOAT32: return get_scalar<float>();
            case GgufType::FLOAT64: return static_cast<float>(get_scalar<double>());
            default:                return 0.0f;
        }
    }

    std::string get_string() const {
        return strings.empty() ? std::string() : strings[0];
    }

    template<typename T>
    std::vector<T> get_array() const {
        size_t n = data.size() / sizeof(T);
        const T* ptr = reinterpret_cast<const T*>(data.data());
        return std::vector<T>(ptr, ptr + n);
    }
};

// GGUF file reader
class GgufReader {
public:
    GgufReader();
    ~GgufReader();

    // Load GGUF file via mmap
    bool load(const std::string& filepath);

    // Access header info
    uint32_t version() const { return version_; }
    int64_t n_tensors() const { return n_tensors_; }
    int64_t n_kv() const { return n_kv_; }
    int64_t n_vocab() const { return n_vocab_; }
    size_t alignment() const { return alignment_; }
    size_t data_offset() const { return data_offset_; }
    const void* data() const { return data_; }
    size_t data_size() const { return data_size_; }

    // KV pair access
    int64_t find_key(const std::string& key) const;
    const std::string& key_name(int64_t id) const { return kv_keys_[id]; }
    const GgufKvValue& kv_value(int64_t id) const { return kv_values_[id]; }

    // Typed KV getters (return default_val if not found)
    uint32_t get_u32(const std::string& key, uint32_t def = 0) const;
    int32_t get_i32(const std::string& key, int32_t def = 0) const;
    uint64_t get_u64(const std::string& key, uint64_t def = 0) const;
    int64_t get_i64(const std::string& key, int64_t def = 0) const;
    float get_f32(const std::string& key, float def = 0.0f) const;
    std::string get_str(const std::string& key, const std::string& def = "") const;
    bool get_bool(const std::string& key, bool def = false) const;

    // Tensor info access
    int64_t find_tensor(const std::string& name) const;
    const GgufTensorInfo& tensor_info(int64_t id) const { return tensor_infos_[id]; }
    const std::vector<GgufTensorInfo>& all_tensors() const { return tensor_infos_; }

    // Get pointer to tensor data
    const void* tensor_data(int64_t tensor_id) const;
    const void* tensor_data_by_name(const std::string& name) const;

    // Get tensor data size in bytes
    size_t tensor_byte_size(const GgufTensorInfo& info) const;

    // Check if file is loaded
    bool is_loaded() const { return data_ != nullptr; }

private:
    bool read_header();
    void read_kv();
    void read_tensor_info();
    void map_file(const std::string& filepath);

    // mmap state
    void* mmap_addr_ = nullptr;
    size_t mmap_size_ = 0;

    // File handle
    int fd_ = -1;

    // GGUF header
    uint32_t version_ = 0;
    int64_t n_tensors_ = 0;
    int64_t n_kv_ = 0;
    int64_t n_vocab_ = 0;
    size_t alignment_ = VM_C_GGUF_DEFAULT_ALIGNMENT;
    size_t data_offset_ = 0;

    // Data pointer (points into mmap)
    const void* data_ = nullptr;
    size_t data_size_ = 0;

    // KV pairs
    std::vector<std::string> kv_keys_;
    std::vector<GgufKvValue> kv_values_;

    // Tensor infos
    std::vector<GgufTensorInfo> tensor_infos_;

    // Offsets for string parsing (positions within mmap)
    const uint8_t* cursor_ = nullptr;
    const uint8_t* end_ = nullptr;

    // Internal read helpers
    template<typename T>
    T read_scalar() {
        T val;
        memcpy(&val, cursor_, sizeof(T));
        cursor_ += sizeof(T);
        return val;
    }

    std::string read_string();
    GgufKvValue read_value(GgufType type);
};

// GGML type traits for quantization formats
struct GgmlTypeTraits {
    size_t type_size;   // bytes per block
    int64_t blck_size;  // elements per block
    std::string name;
};

const GgmlTypeTraits& ggml_type_traits(GgmlType type);

// Dequantization functions for supported types
void dequantize_iq4_xs_cpu(const BlockIQ4XS* x, float* y, int64_t k);

// Convert GGML type to vm_c DataType for unquantized tensors
DataType ggml_type_to_vmc_dtype(GgmlType type);
bool ggml_type_is_quantized(GgmlType type);

// IQ4_XS k-values table (from llama.cpp)
extern const float kvalues_iq4nl[16];

// Parse ModelConfig from GGUF file metadata
ModelConfig parse_gguf_config(const std::string& gguf_path);

} // namespace vm_c
