#include "vm_c/core/gguf_reader.hpp"
#include <spdlog/spdlog.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace vm_c {

// k-values table for IQ4_NL/IQ4_XS dequantization (from llama.cpp)
const float kvalues_iq4nl[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
    1.f,    13.f,   25.f,  38.f,  53.f,  69.f,  89.f,  113.f
};

const GgmlTypeTraits& ggml_type_traits(GgmlType type) {
    static thread_local GgmlTypeTraits traits;
    if (type < 0 || type >= GGML_TYPE_COUNT) {
        static const GgmlTypeTraits kUnknown{0, 0, "unknown"};
        return kUnknown;
    }
    // 对齐 libllama/ggml：block 布局以 ggml_type_size / ggml_blck_size 为唯一来源
    traits.type_size = ggml_type_size(type);
    traits.blck_size = ggml_blck_size(type);
    traits.name = ggml_type_name(type);
    return traits;
}

DataType ggml_type_to_vmc_dtype(GgmlType type) {
    switch (type) {
        case GGML_TYPE_F32:  return DataType::FLOAT32;
        case GGML_TYPE_F16:  return DataType::FLOAT16;
        case GGML_TYPE_BF16: return DataType::BFLOAT16;
        default:             return DataType::FLOAT16; // quantized -> dequant to fp16
    }
}

bool ggml_type_is_quantized(GgmlType type) {
    if (type < 0 || type >= GGML_TYPE_COUNT) {
        return false;
    }
    return ggml_is_quantized(type);
}

// ── GgufReader implementation ──

GgufReader::GgufReader() = default;

GgufReader::~GgufReader() {
    if (mmap_addr_ && mmap_addr_ != MAP_FAILED) {
        munmap(mmap_addr_, mmap_size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

bool GgufReader::load(const std::string& filepath) {
    map_file(filepath);
    if (!mmap_addr_ || mmap_addr_ == MAP_FAILED) {
        spdlog::error("GGUF: failed to mmap file: {}", filepath);
        return false;
    }

    cursor_ = static_cast<const uint8_t*>(mmap_addr_);
    end_ = cursor_ + mmap_size_;

    return read_header();
}

void GgufReader::map_file(const std::string& filepath) {
    fd_ = open(filepath.c_str(), O_RDONLY);
    if (fd_ < 0) {
        perror("open");
        return;
    }

    struct stat st;
    if (fstat(fd_, &st) != 0) {
        perror("fstat");
        close(fd_);
        fd_ = -1;
        return;
    }

    mmap_size_ = st.st_size;
    mmap_addr_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mmap_addr_ == MAP_FAILED) {
        perror("mmap");
        close(fd_);
        fd_ = -1;
    }
}

bool GgufReader::read_header() {
    // Magic
    uint32_t magic = read_scalar<uint32_t>();
    if (magic != VM_C_GGUF_MAGIC) {
        spdlog::error("GGUF: invalid magic: 0x{:08x}, expected 0x{:08x}", magic, VM_C_GGUF_MAGIC);
        return false;
    }

    // Version
    version_ = read_scalar<uint32_t>();
    if (version_ == 0) {
        spdlog::error("GGUF: bad version: 0");
        return false;
    }
    if (version_ == 1) {
        spdlog::error("GGUF: v1 no longer supported");
        return false;
    }
    if (version_ > VM_C_GGUF_VERSION) {
        spdlog::error("GGUF: version {} exceeds max supported {}", version_, VM_C_GGUF_VERSION);
        return false;
    }

    // Check endianness: if version looks big-endian, reject
    if ((version_ & 0x0000FFFF) == 0x00000000) {
        spdlog::error("GGUF: possible endianness mismatch");
        return false;
    }

    // Number of tensors
    n_tensors_ = read_scalar<int64_t>();
    if (n_tensors_ < 0) {
        spdlog::error("GGUF: negative n_tensors: {}", n_tensors_);
        return false;
    }

    // Number of KV pairs
    n_kv_ = read_scalar<int64_t>();
    if (n_kv_ < 0) {
        spdlog::error("GGUF: negative n_kv: {}", n_kv_);
        return false;
    }

    // Read KV pairs
    read_kv();

    // GGUF v3 词汇表数据已通过 KV pairs (tokenizer.ggml.tokens 等) 读取，
    // 无需额外的 vocab_count 字段读取。（llama.cpp 参考实现中 GGUF v3 的
    // 文件头格式为 magic + version + n_tensors + n_kv，之后紧跟 KV pairs，
    // 中间没有 vocab_count 字段。）

    // Read tensor infos
    read_tensor_info();

    // Align to data section
    size_t cur_offset = cursor_ - static_cast<const uint8_t*>(mmap_addr_);
    size_t aligned = (cur_offset + alignment_ - 1) & ~(alignment_ - 1);
    if (aligned < cur_offset) {
        spdlog::error("GGUF: alignment overflow");
        return false;
    }
    data_offset_ = aligned;

    // Data pointer
    data_ = static_cast<const uint8_t*>(mmap_addr_) + data_offset_;
    data_size_ = mmap_size_ - data_offset_;

    spdlog::info("GGUF: loaded (version={}, {} tensors, {} kv pairs, data_offset={}, data_size={})",
                 version_, n_tensors_, n_kv_, data_offset_, data_size_);

    return true;
}

std::string GgufReader::read_string() {
    uint64_t len = read_scalar<uint64_t>();
    if (len > mmap_size_) {
        throw std::runtime_error("GGUF string too long");
    }
    std::string s(reinterpret_cast<const char*>(cursor_), len);
    cursor_ += len;
    return s;
}

GgufKvValue GgufReader::read_value(GgufType type) {
    GgufKvValue val;
    val.type = type;
    val.is_array = false;

    switch (type) {
        case GgufType::UINT8: {
            uint8_t v = read_scalar<uint8_t>();
            val.data.resize(sizeof(v));
            memcpy(val.data.data(), &v, sizeof(v));
            break;
        }
        case GgufType::INT8: {
            int8_t v = read_scalar<int8_t>();
            val.data.resize(sizeof(v));
            memcpy(val.data.data(), &v, sizeof(v));
            break;
        }
        case GgufType::UINT16: {
            uint16_t v = read_scalar<uint16_t>();
            val.data.resize(sizeof(v));
            memcpy(val.data.data(), &v, sizeof(v));
            break;
        }
        case GgufType::INT16: {
            int16_t v = read_scalar<int16_t>();
            val.data.resize(sizeof(v));
            memcpy(val.data.data(), &v, sizeof(v));
            break;
        }
        case GgufType::UINT32: {
            uint32_t v = read_scalar<uint32_t>();
            val.data.resize(sizeof(v));
            memcpy(val.data.data(), &v, sizeof(v));
            break;
        }
        case GgufType::INT32: {
            int32_t v = read_scalar<int32_t>();
            val.data.resize(sizeof(v));
            memcpy(val.data.data(), &v, sizeof(v));
            break;
        }
        case GgufType::FLOAT32: {
            float v = read_scalar<float>();
            val.data.resize(sizeof(v));
            memcpy(val.data.data(), &v, sizeof(v));
            break;
        }
        case GgufType::BOOL: {
            int8_t v = read_scalar<int8_t>();
            val.data.resize(sizeof(v));
            memcpy(val.data.data(), &v, sizeof(v));
            break;
        }
        case GgufType::STRING: {
            val.strings.push_back(read_string());
            break;
        }
        case GgufType::ARRAY: {
            val.is_array = true;
            val.array_type = static_cast<GgufType>(read_scalar<int32_t>());
            uint64_t n = read_scalar<uint64_t>();
            if (val.array_type == GgufType::STRING) {
                for (uint64_t i = 0; i < n; ++i) {
                    val.strings.push_back(read_string());
                }
            } else {
                for (uint64_t i = 0; i < n; ++i) {
                    auto elem = read_value(val.array_type);
                    val.data.insert(val.data.end(), elem.data.begin(), elem.data.end());
                }
            }
            break;
        }
        case GgufType::UINT64: {
            uint64_t v = read_scalar<uint64_t>();
            val.data.resize(sizeof(v));
            memcpy(val.data.data(), &v, sizeof(v));
            break;
        }
        case GgufType::INT64: {
            int64_t v = read_scalar<int64_t>();
            val.data.resize(sizeof(v));
            memcpy(val.data.data(), &v, sizeof(v));
            break;
        }
        case GgufType::FLOAT64: {
            double v = read_scalar<double>();
            val.data.resize(sizeof(v));
            memcpy(val.data.data(), &v, sizeof(v));
            break;
        }
        default:
            throw std::runtime_error("GGUF: unknown type " + std::to_string(static_cast<int>(type)));
    }
    return val;
}

void GgufReader::read_kv() {
    kv_keys_.reserve(n_kv_);
    kv_values_.reserve(n_kv_);

    for (int64_t i = 0; i < n_kv_; ++i) {
        std::string key = read_string();
        int32_t raw_type = read_scalar<int32_t>();
        GgufType type = static_cast<GgufType>(raw_type);

        GgufKvValue val;
        if (type == GgufType::ARRAY) {
            // Array: read element type and count then elements
            int32_t elem_type_raw = read_scalar<int32_t>();
            GgufType elem_type = static_cast<GgufType>(elem_type_raw);
            uint64_t n = read_scalar<uint64_t>();

            val.is_array = true;
            val.array_type = elem_type;
            val.type = GgufType::ARRAY;

            if (elem_type == GgufType::STRING) {
                for (uint64_t j = 0; j < n; ++j) {
                    val.strings.push_back(read_string());
                }
            } else {
                for (uint64_t j = 0; j < n; ++j) {
                    auto elem = read_value(elem_type);
                    val.data.insert(val.data.end(), elem.data.begin(), elem.data.end());
                }
            }
        } else {
            val = read_value(type);
        }

        kv_keys_.push_back(std::move(key));
        kv_values_.push_back(std::move(val));
    }
}

void GgufReader::read_tensor_info() {
    tensor_infos_.reserve(n_tensors_);

    for (int64_t i = 0; i < n_tensors_; ++i) {
        GgufTensorInfo info;

        // Name
        info.name = read_string();

        // Number of dimensions
        info.n_dims = read_scalar<uint32_t>();

        // Shape (ne[0..n_dims-1], remaining = 1)
        for (int j = 0; j < 4; ++j) {
            info.ne[j] = (j < info.n_dims) ? read_scalar<int64_t>() : 1;
        }

        // Type
        info.type = static_cast<GgmlType>(read_scalar<int32_t>());

        // Offset
        info.offset = read_scalar<uint64_t>();

        tensor_infos_.push_back(info);
    }
}

int64_t GgufReader::find_key(const std::string& key) const {
    for (int64_t i = 0; i < static_cast<int64_t>(kv_keys_.size()); ++i) {
        if (kv_keys_[i] == key) return i;
    }
    return -1;
}

uint32_t GgufReader::get_u32(const std::string& key, uint32_t def) const {
    int64_t id = find_key(key);
    if (id < 0) return def;
    return kv_values_[id].get_uint32_safe();
}

int32_t GgufReader::get_i32(const std::string& key, int32_t def) const {
    int64_t id = find_key(key);
    if (id < 0) return def;
    return static_cast<int32_t>(kv_values_[id].get_int64_safe());
}

uint64_t GgufReader::get_u64(const std::string& key, uint64_t def) const {
    int64_t id = find_key(key);
    if (id < 0) return def;
    return static_cast<uint64_t>(kv_values_[id].get_int64_safe());
}

int64_t GgufReader::get_i64(const std::string& key, int64_t def) const {
    int64_t id = find_key(key);
    if (id < 0) return def;
    return kv_values_[id].get_int64_safe();
}

float GgufReader::get_f32(const std::string& key, float def) const {
    int64_t id = find_key(key);
    if (id < 0) return def;
    return kv_values_[id].get_float32_safe();
}

std::string GgufReader::get_str(const std::string& key, const std::string& def) const {
    int64_t id = find_key(key);
    if (id < 0) return def;
    return kv_values_[id].get_string();
}

bool GgufReader::get_bool(const std::string& key, bool def) const {
    int64_t id = find_key(key);
    if (id < 0) return def;
    return kv_values_[id].get_int64_safe() != 0;
}

int64_t GgufReader::find_tensor(const std::string& name) const {
    for (int64_t i = 0; i < static_cast<int64_t>(tensor_infos_.size()); ++i) {
        if (tensor_infos_[i].name == name) return i;
    }
    return -1;
}

const void* GgufReader::tensor_data(int64_t tensor_id) const {
    if (tensor_id < 0 || tensor_id >= static_cast<int64_t>(tensor_infos_.size())) {
        return nullptr;
    }
    const auto& info = tensor_infos_[tensor_id];
    return static_cast<const uint8_t*>(data_) + info.offset;
}

const void* GgufReader::tensor_data_by_name(const std::string& name) const {
    return tensor_data(find_tensor(name));
}

size_t GgufReader::tensor_byte_size(const GgufTensorInfo& info) const {
    const auto& traits = ggml_type_traits(info.type);
    if (traits.blck_size == 0) return 0;

    // n_blocks = total_elements / blck_size
    int64_t n_elems = info.ne[0] * info.ne[1] * info.ne[2] * info.ne[3];
    int64_t n_blocks = (n_elems + traits.blck_size - 1) / traits.blck_size;
    return static_cast<size_t>(n_blocks) * traits.type_size;
}

// ── IQ4_XS CPU dequantization (from llama.cpp ggml-quants.c) ──

void dequantize_iq4_xs_cpu(const BlockIQ4XS* x, float* y, int64_t k) {
    const int64_t nb = k / QK_K;

    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* qs = x[i].qs;
        const float d = kvalues_iq4nl[0]; // placeholder, proper f16 conversion below
        // Proper fp16->fp32
        uint16_t d_bits = x[i].d;
        // Convert fp16 bits to float
        uint32_t sign = (uint32_t)(d_bits & 0x8000) << 16;
        uint32_t exp = (d_bits & 0x7C00) >> 10;
        uint32_t frac = d_bits & 0x03FF;
        uint32_t f;
        if (exp == 0) {
            f = sign | 0x00000000u;
            if (frac != 0) {
                // subnormal
                exp = 0x71;
                while ((frac & 0x0400) == 0) { frac <<= 1; exp--; }
                frac &= 0x03FF;
                f = sign | ((exp + 0x7F - 0x70) << 23) | (frac << 13);
            } else {
                f = sign;
            }
        } else if (exp == 0x1F) {
            f = sign | 0x7F800000 | (frac << 13);
        } else {
            f = sign | ((exp + 0x7F - 0x0F) << 23) | (frac << 13);
        }
        float d_val;
        memcpy(&d_val, &f, sizeof(d_val));

        for (int ib = 0; ib < QK_K / 32; ++ib) {
            int ls = ((x[i].scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf)
                   | (((x[i].scales_h >> (2 * ib)) & 3) << 4);
            float dl = d_val * static_cast<float>(ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j + 0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
    }
}

} // namespace vm_c
