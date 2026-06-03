#include "vm_c/core/gguf_reader.hpp"
#include "vm_c/core/ggml_dequant.hpp"
#include <cstring>
#include "vm_c/model/llama_gguf_tensor_map.hpp"
#include "vm_c/model/model_loader.hpp"
#include "vm_c/cuda/gpu_arch.hpp"
#include "vm_c/cuda/vm_c_tensor.h"
#include "vm_c/cuda/convert_kernels.h"
#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
#include "vm_c/official/ggml_weight.hpp"
#endif
#include <spdlog/spdlog.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace vm_c {

#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
static GpuTensor make_ggml_quant_from_host(
    ggml_type type,
    int gpu_device,
    int64_t ne0, int64_t ne1, int64_t ne2,
    const void* host_data, size_t host_bytes,
    int64_t block_count,
    cudaStream_t stream) {
    auto storage = official::ggml_upload_quant_from_host(
        type, gpu_device, ne0, ne1, ne2, host_data, host_bytes, stream);
    GpuTensor tensor;
    tensor.adopt_ggml_weight(storage, Shape({block_count}), DataType::UINT8, gpu_device);
    return tensor;
}

static GpuTensor make_ggml_q5_k_from_host(
    int gpu_device,
    int64_t ne0, int64_t ne1, int64_t ne2,
    const void* host_data, size_t host_bytes,
    int64_t block_count,
    cudaStream_t stream) {
    auto storage = official::ggml_upload_q5_k_from_host(
        gpu_device, ne0, ne1, ne2, host_data, host_bytes, stream);
    GpuTensor tensor;
    tensor.adopt_ggml_weight(storage, Shape({block_count}), DataType::UINT8, gpu_device);
    return tensor;
}

// K-quant / block-quant TP 分片（按 block 字节 stride，对齐 IQ4 路径）
static int64_t ggml_quant_col_split0_blocks_per_rank(
    int tp_rank, int tp_size, int64_t row_elems, int64_t n_rows, int blk_elems) {
    const int blks_per_row = static_cast<int>((row_elems + blk_elems - 1) / blk_elems);
    const int64_t split_sz = (n_rows + tp_size - 1) / tp_size;
    const int64_t row_off = tp_rank * split_sz;
    const int64_t n_rows_r = std::min(split_sz, n_rows - row_off);
    return n_rows_r > 0 ? n_rows_r * blks_per_row : 0;
}

static int64_t shard_ggml_quant_col_split0(
    void* dst, const void* src, size_t blk_bytes,
    int tp_rank, int tp_size,
    int64_t row_elems, int64_t n_rows, int blk_elems) {
    const int blks_per_row = static_cast<int>((row_elems + blk_elems - 1) / blk_elems);
    const int64_t split_sz = (n_rows + tp_size - 1) / tp_size;
    const int64_t row_off = tp_rank * split_sz;
    const int64_t n_rows_r = std::min(split_sz, n_rows - row_off);
    if (n_rows_r <= 0) {
        return 0;
    }
    const auto* src_b = static_cast<const uint8_t*>(src);
    auto* dst_b = static_cast<uint8_t*>(dst);
    for (int64_t r = 0; r < n_rows_r; ++r) {
        std::memcpy(dst_b + static_cast<size_t>(r * blks_per_row) * blk_bytes,
                    src_b + static_cast<size_t>(row_off + r) * blks_per_row * blk_bytes,
                    static_cast<size_t>(blks_per_row) * blk_bytes);
    }
    return n_rows_r * blks_per_row;
}

static int64_t ggml_quant_row_split1_blocks_per_rank(
    int tp_rank, int tp_size, int64_t ne0, int64_t ne1, int blk_elems) {
    const int blks_per_row_full = static_cast<int>((ne0 + blk_elems - 1) / blk_elems);
    // 按 block 对齐均分：tp_size 个 rank 均分 blks_per_row_full 个 block
    const int blks_lo = (tp_rank * blks_per_row_full) / tp_size;
    const int blks_hi = ((tp_rank + 1) * blks_per_row_full) / tp_size;
    const int blks_per_row_r = blks_hi - blks_lo;
    if (blks_per_row_r <= 0) {
        return 0;
    }
    return static_cast<int64_t>(ne1) * static_cast<int64_t>(blks_per_row_r);
}

static int64_t shard_ggml_quant_row_split1(
    void* dst, const void* src, size_t blk_bytes,
    int tp_rank, int tp_size,
    int64_t ne0, int64_t ne1, int blk_elems) {
    const int blks_per_row_full = static_cast<int>((ne0 + blk_elems - 1) / blk_elems);
    // 按 block 对齐均分
    const int src_blk_off = (tp_rank * blks_per_row_full) / tp_size;
    const int blks_hi = ((tp_rank + 1) * blks_per_row_full) / tp_size;
    const int blks_per_row_r = blks_hi - src_blk_off;
    if (blks_per_row_r <= 0) {
        return 0;
    }
    const auto* src_b = static_cast<const uint8_t*>(src);
    auto* dst_b = static_cast<uint8_t*>(dst);
    int64_t dst_count = 0;
    for (int64_t row = 0; row < ne1; ++row) {
        std::memcpy(dst_b + static_cast<size_t>(dst_count) * blk_bytes,
                    src_b + static_cast<size_t>(row * blks_per_row_full + src_blk_off) * blk_bytes,
                    static_cast<size_t>(blks_per_row_r) * blk_bytes);
        dst_count += blks_per_row_r;
    }
    return dst_count;
}
#endif

// Architecture name mapping
static const std::unordered_map<std::string, std::string> ARCH_ALIASES = {
    {"qwen3moe", "qwen3moe"},
    {"qwen3-moe", "qwen3moe"},
    {"qwen3.5moe", "qwen35moe"},
    {"qwen3.6moe", "qwen35moe"},
};

// ── GDN 权重 TP 分片（对齐 vLLM mamba_v2_sharded_weight_loader / MergedColumnParallelLinear）──
static void gdn_dims_from_config(const ModelConfig& cfg, int tp_size,
                                 int& key_dim, int& value_dim, int& conv_dim) {
    const int head_k = cfg.linear_key_head_dim > 0 ? cfg.linear_key_head_dim : 128;
    const int head_v = cfg.linear_inner_dim > 0
        ? (cfg.linear_inner_dim / std::max(1, cfg.linear_num_value_heads))
        : (cfg.linear_value_head_dim > 0 ? cfg.linear_value_head_dim : 128);
    const int n_k = std::max(1, cfg.linear_num_key_heads);
    const int n_v = std::max(1, cfg.linear_num_value_heads);
    key_dim = head_k * n_k;
    value_dim = head_v * n_v;
    conv_dim = 2 * key_dim + value_dim;
    (void)tp_size;
}

static bool is_gdn_qkv_tensor(const std::string& name) {
    return name.find("linear_attn.in_proj_qkv.weight") != std::string::npos;
}
static bool is_gdn_z_tensor(const std::string& name) {
    return name.find("linear_attn.in_proj_z") != std::string::npos;
}
static bool is_gdn_conv_tensor(const std::string& name) {
    return name.find("linear_attn.conv1d") != std::string::npos;
}
static bool is_gdn_out_proj(const std::string& name) {
    return name.find("linear_attn.out_proj") != std::string::npos;
}
static bool is_gdn_ba_tensor(const std::string& name) {
    return name.find("linear_attn.in_proj_b") != std::string::npos ||
           name.find("linear_attn.in_proj_a") != std::string::npos ||
           name.find("linear_attn.in_proj_ba") != std::string::npos;
}
static bool is_gdn_head_vector(const std::string& name) {
    if (name.find("linear_attn.A_log") != std::string::npos ||
        name.find("linear_attn.dt_bias") != std::string::npos) {
        return true;
    }
    // GGUF 原名 blk.{i}.ssm_a / blk.{i}.ssm_dt(.bias)（映射失败时的 fallback 识别）
    const auto ends_with = [](const std::string& s, const std::string& suf) {
        return s.size() >= suf.size() &&
               s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
    };
    return ends_with(name, ".ssm_a") || ends_with(name, ".ssm_dt.bias") ||
           ends_with(name, ".ssm_dt");
}

// Q/K/V 三段式输出分片: [Q:key_dim | K:key_dim | V:value_dim] × hidden
static void shard_gdn_qkv_rows(
    uint8_t* dst, const uint8_t* src,
    int tp_rank, int tp_size,
    int key_dim, int value_dim, int row_bytes) {
    const int shard_k = key_dim / tp_size;
    const int shard_v = value_dim / tp_size;
    auto copy_block = [&](int dst_row, int src_row, int count) {
        for (int r = 0; r < count; ++r) {
            memcpy(dst + (dst_row + r) * row_bytes,
                   src + (src_row + r) * row_bytes,
                   row_bytes);
        }
    };
    copy_block(0, tp_rank * shard_k, shard_k);
    copy_block(shard_k, key_dim + tp_rank * shard_k, shard_k);
    copy_block(2 * shard_k, 2 * key_dim + tp_rank * shard_v, shard_v);
}

static void shard_gdn_conv_rows(
    uint8_t* dst, const uint8_t* src,
    int tp_rank, int tp_size,
    int key_dim, int value_dim, int row_bytes) {
    shard_gdn_qkv_rows(dst, src, tp_rank, tp_size, key_dim, value_dim, row_bytes);
}

// IQ4_XS GDN 行分片：ggml ne=[hidden, out_rows]，每行 hidden/256 个 block
static int64_t gdn_iq4_head_blocks_per_rank(
    int tp_rank, int tp_size, int64_t n_hidden, int64_t n_out_rows) {
    constexpr int BLK_EL = 256;
    const int blks_per_row = static_cast<int>((n_hidden + BLK_EL - 1) / BLK_EL);
    const int64_t split_sz = (n_out_rows + tp_size - 1) / tp_size;
    const int64_t src_off = tp_rank * split_sz;
    const int64_t n_rows_r = std::min(split_sz, n_out_rows - src_off);
    return n_rows_r > 0 ? n_rows_r * blks_per_row : 0;
}

static int64_t shard_gdn_iq4_head_split0(
    void* dst, const void* src,
    int tp_rank, int tp_size,
    int64_t n_hidden, int64_t n_out_rows) {
    constexpr int BLK_EL = 256;
    const int blks_per_row = static_cast<int>((n_hidden + BLK_EL - 1) / BLK_EL);
    const int64_t split_sz = (n_out_rows + tp_size - 1) / tp_size;
    const int64_t src_off = tp_rank * split_sz;
    const int64_t n_rows_r = std::min(split_sz, n_out_rows - src_off);
    if (n_rows_r <= 0) return 0;
    const auto* src_blk = static_cast<const BlockIQ4XS*>(src);
    auto* dst_blk = static_cast<BlockIQ4XS*>(dst);
    for (int64_t r = 0; r < n_rows_r; ++r) {
        memcpy(dst_blk + r * blks_per_row,
               src_blk + (src_off + r) * blks_per_row,
               static_cast<size_t>(blks_per_row) * sizeof(BlockIQ4XS));
    }
    return n_rows_r * blks_per_row;
}

// IQ4 GDN QKV/conv：out_rows = [Q key_dim | K key_dim | V value_dim]，三段式 TP 分片
static int64_t gdn_iq4_qkv_out_rows_per_rank(int tp_size, int key_dim, int value_dim) {
    const int shard_k = key_dim / tp_size;
    const int shard_v = value_dim / tp_size;
    return 2 * shard_k + shard_v;
}

static int64_t gdn_iq4_qkv_blocks_per_rank(
    int tp_rank, int tp_size, int64_t n_hidden, int key_dim, int value_dim) {
    constexpr int BLK_EL = 256;
    const int blks_per_row = static_cast<int>((n_hidden + BLK_EL - 1) / BLK_EL);
    const int64_t n_out_rows = gdn_iq4_qkv_out_rows_per_rank(tp_size, key_dim, value_dim);
    (void) tp_rank;
    return n_out_rows > 0 ? n_out_rows * blks_per_row : 0;
}

static int64_t shard_gdn_iq4_qkv_rows(
    void* dst, const void* src,
    int tp_rank, int tp_size,
    int64_t n_hidden, int key_dim, int value_dim) {
    constexpr int BLK_EL = 256;
    const int blks_per_row = static_cast<int>((n_hidden + BLK_EL - 1) / BLK_EL);
    const int shard_k = key_dim / tp_size;
    const int shard_v = value_dim / tp_size;
    if (shard_k <= 0 || shard_v <= 0) {
        return 0;
    }
    const auto* src_blk = static_cast<const BlockIQ4XS*>(src);
    auto* dst_blk = static_cast<BlockIQ4XS*>(dst);
    const size_t row_bytes = static_cast<size_t>(blks_per_row) * sizeof(BlockIQ4XS);
    auto copy_section = [&](int dst_row, int src_row, int count) {
        for (int r = 0; r < count; ++r) {
            std::memcpy(dst_blk + static_cast<size_t>(dst_row + r) * blks_per_row,
                        src_blk + static_cast<size_t>(src_row + r) * blks_per_row,
                        row_bytes);
        }
    };
    copy_section(0, tp_rank * shard_k, shard_k);
    copy_section(shard_k, key_dim + tp_rank * shard_k, shard_k);
    copy_section(2 * shard_k, 2 * key_dim + tp_rank * shard_v, shard_v);
    return gdn_iq4_qkv_blocks_per_rank(tp_rank, tp_size, n_hidden, key_dim, value_dim);
}

static int64_t gdn_ggml_quant_qkv_blocks_per_rank(
    int tp_size, int64_t row_elems, int key_dim, int value_dim, int blk_elems) {
    const int blks_per_row = static_cast<int>((row_elems + blk_elems - 1) / blk_elems);
    const int64_t n_out_rows = gdn_iq4_qkv_out_rows_per_rank(tp_size, key_dim, value_dim);
    return n_out_rows > 0 ? n_out_rows * blks_per_row : 0;
}

static int64_t shard_gdn_ggml_quant_qkv_rows(
    void* dst, const void* src, size_t blk_bytes,
    int tp_rank, int tp_size,
    int64_t row_elems, int key_dim, int value_dim, int blk_elems) {
    const int blks_per_row = static_cast<int>((row_elems + blk_elems - 1) / blk_elems);
    const int shard_k = key_dim / tp_size;
    const int shard_v = value_dim / tp_size;
    if (shard_k <= 0 || shard_v <= 0) {
        return 0;
    }
    const auto* src_b = static_cast<const uint8_t*>(src);
    auto* dst_b = static_cast<uint8_t*>(dst);
    const size_t row_bytes = static_cast<size_t>(blks_per_row) * blk_bytes;
    auto copy_section = [&](int dst_row, int src_row, int count) {
        for (int r = 0; r < count; ++r) {
            std::memcpy(dst_b + static_cast<size_t>(dst_row + r) * row_bytes,
                        src_b + static_cast<size_t>(src_row + r) * row_bytes,
                        row_bytes);
        }
    };
    copy_section(0, tp_rank * shard_k, shard_k);
    copy_section(shard_k, key_dim + tp_rank * shard_k, shard_k);
    copy_section(2 * shard_k, 2 * key_dim + tp_rank * shard_v, shard_v);
    return gdn_ggml_quant_qkv_blocks_per_rank(tp_size, row_elems, key_dim, value_dim, blk_elems);
}

// IQ4 row-parallel（o_proj / down_proj）：ggml ne=[ne0, ne1]，沿 ne0 做 TP 分片
static int64_t iq4_row_split1_blocks_per_rank(
    int tp_rank, int tp_size, int64_t ne0, int64_t ne1) {
    constexpr int BLK_EL = 256;
    const int64_t split_sz = (ne0 + tp_size - 1) / tp_size;
    const int64_t src_off = tp_rank * split_sz;
    const int64_t ne0_r = std::min(split_sz, ne0 - src_off);
    if (ne0_r <= 0) {
        return 0;
    }
    const int blks_per_row_r = static_cast<int>((ne0_r + BLK_EL - 1) / BLK_EL);
    return ne1 * blks_per_row_r;
}

static int64_t shard_iq4_row_split1(
    void* dst, const void* src,
    int tp_rank, int tp_size,
    int64_t ne0, int64_t ne1) {
    constexpr int BLK_EL = 256;
    const int blks_per_row_full = static_cast<int>((ne0 + BLK_EL - 1) / BLK_EL);
    const int64_t split_sz = (ne0 + tp_size - 1) / tp_size;
    const int64_t src_off = tp_rank * split_sz;
    const int64_t ne0_r = std::min(split_sz, ne0 - src_off);
    if (ne0_r <= 0) {
        return 0;
    }
    const int src_blk_off = static_cast<int>(src_off / BLK_EL);
    const int blks_per_row_r = static_cast<int>((ne0_r + BLK_EL - 1) / BLK_EL);
    const auto* src_blk = static_cast<const BlockIQ4XS*>(src);
    auto* dst_blk = static_cast<BlockIQ4XS*>(dst);
    int64_t dst_count = 0;
    for (int64_t row = 0; row < ne1; ++row) {
        std::memcpy(dst_blk + dst_count,
                    src_blk + row * blks_per_row_full + src_blk_off,
                    static_cast<size_t>(blks_per_row_r) * sizeof(BlockIQ4XS));
        dst_count += blks_per_row_r;
    }
    return dst_count;
}

// MoE 3D IQ4 gate/up：ggml ne=[n_embd, n_ff, n_expert]，按 n_ff 行做 TP column-split，保留全部 expert
// （对齐 vLLM MergedColumnParallelLinear / FusedMoE，EP=1 时每 rank 持有全部 expert）
static void shard_moe_iq4_gate_up_experts(
    BlockIQ4XS* dst, const BlockIQ4XS* src,
    int64_t n_embd, int64_t n_ff, int64_t n_expert,
    int tp_rank, int tp_size) {
    constexpr int BLK_EL = 256;
    const int blks_per_embd_row = static_cast<int>((n_embd + BLK_EL - 1) / BLK_EL);
    const int64_t ff_r = n_ff / tp_size;
    const int64_t ff_off = tp_rank * ff_r;
    const int64_t blk_per_exp_full = (n_embd * n_ff + BLK_EL - 1) / BLK_EL;
    const int64_t blk_per_exp_r = ff_r * blks_per_embd_row;

    for (int64_t e = 0; e < n_expert; ++e) {
        const BlockIQ4XS* exp_src = src + e * blk_per_exp_full;
        BlockIQ4XS* exp_dst = dst + e * blk_per_exp_r;
        for (int64_t r = 0; r < ff_r; ++r) {
            memcpy(exp_dst + r * blks_per_embd_row,
                   exp_src + (ff_off + r) * blks_per_embd_row,
                   static_cast<size_t>(blks_per_embd_row) * sizeof(BlockIQ4XS));
        }
    }
}

// MoE 3D IQ4/Q5 down：ggml ne=[n_ff, n_embd, n_expert]，按 n_ff 列做 TP row-split
static void shard_moe_down_experts_qblocks(
    void* dst, const void* src, size_t block_bytes,
    int64_t n_ff, int64_t n_embd, int64_t n_expert,
    int tp_rank, int tp_size) {
    constexpr int BLK_EL = 256;
    const int blks_per_hidden_row_full = static_cast<int>((n_ff + BLK_EL - 1) / BLK_EL);
    // block-aligned TP split
    const int blks_lo = (tp_rank * blks_per_hidden_row_full) / tp_size;
    const int blks_hi = ((tp_rank + 1) * blks_per_hidden_row_full) / tp_size;
    const int blks_per_hidden_row_r = blks_hi - blks_lo;
    const int ff_blk_off = blks_lo;

    const int64_t blk_per_exp_full = static_cast<int64_t>(n_embd) * blks_per_hidden_row_full;
    const int64_t blk_per_exp_r = static_cast<int64_t>(n_embd) * blks_per_hidden_row_r;

    auto* dst_b = static_cast<uint8_t*>(dst);
    const auto* src_b = static_cast<const uint8_t*>(src);

    for (int64_t e = 0; e < n_expert; ++e) {
        const uint8_t* exp_src = src_b + e * blk_per_exp_full * block_bytes;
        uint8_t* exp_dst = dst_b + e * blk_per_exp_r * block_bytes;
        for (int64_t h = 0; h < n_embd; ++h) {
            memcpy(exp_dst + h * blks_per_hidden_row_r * block_bytes,
                   exp_src + (h * blks_per_hidden_row_full + ff_blk_off) * block_bytes,
                   static_cast<size_t>(blks_per_hidden_row_r) * block_bytes);
        }
    }
}

static bool is_moe_3d_expert_tensor(const std::string& vmc_name) {
    return vmc_name.find("mlp.experts.gate_proj") != std::string::npos ||
           vmc_name.find("mlp.experts.up_proj") != std::string::npos ||
           vmc_name.find("mlp.experts.down_proj") != std::string::npos;
}

static std::string arch_key(const std::string& arch, const std::string& key) {
    if (key.rfind("llama.", 0) == 0) return arch + "." + key.substr(6);
    return key;
}

// ── 工具函数 ──

// 获取 GGML 量化 block 的元素数 / 字节数（对齐 libllama ggml_blck_size / ggml_type_size）
static int64_t ggml_blk_elems(GgmlType type) {
    return ggml_blck_size(type);
}

static size_t ggml_block_size_bytes(GgmlType type) {
    return ggml_type_size(type);
}

// 对齐 llama.cpp create_tensor：GGUF 原生 F16/BF16/F32 按文件类型存储，不统一转为 compute_dtype
static inline DataType gguf_native_storage_dtype(GgmlType type) {
    switch (type) {
        case GGML_TYPE_F32:  return DataType::FLOAT32;
        case GGML_TYPE_F16:  return DataType::FLOAT16;
        case GGML_TYPE_BF16: return DataType::BFLOAT16;
        default:
            throw std::runtime_error(
                "gguf_native_storage_dtype: type " + std::to_string(static_cast<int>(type))
                + " is not a native dense GGUF dtype");
    }
}

// 判断是否需要在 GPU 上做格式转换
// 对齐 llama.cpp gguf_get_tensor_type：dense F32/F16/BF16 原样保留；仅量化类型需 GPU 转换
static inline bool needs_gpu_convert(GgmlType type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            return false;
        default:
            return true; // Q4/Q5/Q6/Q8 等 → GPU 反量化/转为 compute_dtype
    }
}


// ── ModelConfig 解析 ──

ModelConfig parse_gguf_config(const std::string& gguf_path) {
    GgufReader reader;
    if (!reader.load(gguf_path))
        throw std::runtime_error("Failed to load GGUF file: " + gguf_path);

    ModelConfig cfg;
    cfg.model_dir = gguf_path;
    cfg.dtype_str = "bf16";

    std::string arch = reader.get_str(GgufKeys::GENERAL_ARCHITECTURE, "");
    std::string arch_lower = arch;
    std::transform(arch_lower.begin(), arch_lower.end(), arch_lower.begin(), ::tolower);
    auto it = ARCH_ALIASES.find(arch_lower);
    cfg.arch = (it != ARCH_ALIASES.end()) ? it->second : arch_lower;

    auto ak = [&](const std::string& key) { return arch_key(arch_lower, key); };

    cfg.hidden_size = (int)reader.get_i64(ak(GgufKeys::EMBEDDING_LENGTH), 0);
    cfg.num_hidden_layers = (int)reader.get_i64(ak(GgufKeys::BLOCK_COUNT), 0);
    cfg.num_attention_heads = (int)reader.get_i64(ak(GgufKeys::ATTENTION_HEAD_COUNT), 0);
    cfg.num_key_value_heads = (int)reader.get_i64(ak(GgufKeys::ATTENTION_HEAD_COUNT_KV), cfg.num_attention_heads);
    cfg.vocab_size = (int)reader.get_i64(ak(GgufKeys::VOCABULARY_SIZE), 0);
    if (cfg.vocab_size == 0) {
        int64_t tk_id = reader.find_key(GgufKeys::TOKENIZER_LIST);
        if (tk_id >= 0) {
            const auto& tk_val = reader.kv_value(tk_id);
            if (tk_val.is_array && tk_val.array_type == GgufType::STRING)
                cfg.vocab_size = (int)tk_val.strings.size();
        }
    }
    {
        if (reader.find_key(ak(GgufKeys::CONTEXT_LENGTH)) < 0) {
            throw std::runtime_error(
                "GgufModelLoader: GGUF file missing required key '" +
                std::string(ak(GgufKeys::CONTEXT_LENGTH)) + "'");
        }
        cfg.max_model_len = reader.get_i64(ak(GgufKeys::CONTEXT_LENGTH), 0);
    }
    cfg.rms_norm_eps = reader.get_f32(ak(GgufKeys::ATTENTION_LAYERNORM_RMS_EPS), 1e-6f);
    cfg.rope_theta = reader.get_f32(ak(GgufKeys::ROPE_FREQ_BASE), 10000.0f);
    cfg.head_dim = (int)reader.get_i32(ak(GgufKeys::ATTENTION_KEY_LENGTH), 0);
    if (cfg.head_dim == 0 && cfg.num_attention_heads > 0)
        cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
    if (cfg.head_dim <= 0) {
        throw std::runtime_error(
            "GgufModelLoader: Cannot determine head_dim (ATTENTION_KEY_LENGTH missing "
            "and cannot compute from hidden_size/num_attention_heads)");
    }

    cfg.num_experts = (int)reader.get_i64(ak(GgufKeys::EXPERT_COUNT), 0);
    cfg.num_experts_per_tok = (int)reader.get_i64(ak(GgufKeys::EXPERT_USED_COUNT), 0);
    cfg.moe_intermediate_size = (int)reader.get_i64(ak(GgufKeys::EXPERT_FEED_FORWARD_LENGTH), 0);
    cfg.n_shared_experts = (int)reader.get_i64(ak(GgufKeys::EXPERT_SHARED_COUNT), 0);
    cfg.shared_expert_intermediate_size = (int)reader.get_i64(ak(GgufKeys::EXPERT_SHARED_FEED_FORWARD_LENGTH), 0);
    cfg.num_expert_group = (int)reader.get_i64(ak(GgufKeys::EXPERT_GROUP_COUNT), 1);
    cfg.topk_group = (int)reader.get_i64(ak(GgufKeys::EXPERT_GROUP_USED_COUNT), 1);
    cfg.routed_scaling_factor = reader.get_f32(ak(GgufKeys::EXPERT_WEIGHTS_SCALE), 1.0f);
    cfg.norm_topk_prob = reader.get_bool(ak(GgufKeys::EXPERT_NORM), true);

    int nextn = (int)reader.get_i64(ak(GgufKeys::NEXTN_PREDICT_LAYERS), 0);
    if (nextn > 0) {
        cfg.decoder_sparse_step = nextn;
        cfg.mtp_predict_layers = nextn;
        if (cfg.spec_method == "none") cfg.spec_method = "mtp";
    }

    int full_attn_interval = (int)reader.get_i64(ak(GgufKeys::FULL_ATTENTION_INTERVAL), 0);
    if (full_attn_interval > 0 && cfg.num_hidden_layers > 0) {
        const int n_main = cfg.num_hidden_layers - cfg.mtp_predict_layers;
        cfg.layer_types.clear();
        cfg.layer_types.reserve(cfg.num_hidden_layers);
        for (int i = 0; i < cfg.num_hidden_layers; ++i) {
            if (i >= n_main) {
                // MTP block：dense full-attention + MoE，对齐 llama.cpp load_block_mtp
                cfg.layer_types.push_back("full_attention");
            } else {
                cfg.layer_types.push_back(
                    (i + 1) % full_attn_interval == 0 ? "full_attention" : "linear_attention");
            }
        }
    }

    // SSM / GDN
    {
        int v = (int)reader.get_i64(ak(GgufKeys::SSM_STATE_SIZE), 0);
        if (v > 0) { cfg.linear_key_head_dim = v; cfg.linear_value_head_dim = v; }
        v = (int)reader.get_i64(ak(GgufKeys::SSM_GROUP_COUNT), 0);
        if (v > 0) cfg.linear_num_key_heads = v;
        v = (int)reader.get_i64(ak(GgufKeys::SSM_TIME_STEP_RANK), 0);
        if (v > 0) cfg.linear_num_value_heads = v;
        v = (int)reader.get_i64(ak(GgufKeys::SSM_CONV_KERNEL), 0);
        if (v > 0) cfg.linear_conv_kernel_dim = v;
        v = (int)reader.get_i64(ak(GgufKeys::SSM_INNER_SIZE), 0);
        if (v > 0) cfg.linear_inner_dim = v;
    }

    spdlog::info("[GDN_CFG] ssm_d_state={} ssm_n_group={} ssm_dt_rank={} ssm_d_conv={} ssm_d_inner={}",
                 cfg.linear_key_head_dim, cfg.linear_num_key_heads,
                 cfg.linear_num_value_heads, cfg.linear_conv_kernel_dim,
                 cfg.linear_inner_dim);

    cfg.intermediate_size = (int)reader.get_i64(ak(GgufKeys::FEED_FORWARD_LENGTH), cfg.moe_intermediate_size);
    int rot_dim = (int)reader.get_i32(ak(GgufKeys::ROPE_DIMENSION_COUNT), 0);
    if (rot_dim > 0 && cfg.head_dim > 0)
        cfg.partial_rotary_factor = (float)rot_dim / cfg.head_dim;

    int64_t eos_id = reader.get_i64(GgufKeys::TOKENIZER_EOS_ID, -1);
    if (eos_id >= 0) cfg.eos_token_ids.push_back((int)eos_id);

    spdlog::info("GGUF config: arch={}, layers={}, hidden={}, heads={}/{}",
                 cfg.arch, cfg.num_hidden_layers, cfg.hidden_size,
                 cfg.num_attention_heads, cfg.num_key_value_heads);
    return cfg;
}

// ─────────────────────────────────────────────────────────────
// 新版权重加载函数
//
// 核心优化（对齐 llama.cpp）：
// 1. CPU 不做反量化/格式转换 — 原始格式上传到 GPU，由 GPU kernel 转换
// 2. 使用 cudaMemcpyAsync 异步 H2D 传输
// 3. IQ4_XS 保持量化块格式，直接上传
// ─────────────────────────────────────────────────────────────

void load_gguf_weights(const std::string& gguf_path, ModelWeights& weights,
                       const ModelConfig& config, int gpu_device,
                       int tp_rank, int tp_size) {
    GgufReader reader;
    if (!reader.load(gguf_path))
        throw std::runtime_error("Failed to load GGUF file: " + gguf_path);

    const auto& tensors = reader.all_tensors();
    int num_layers = config.num_hidden_layers;
    size_t dtype_sz = gpu().compute_dtype_size(); // 通常是 2 (FP16)

    spdlog::info("Loading {} GGUF tensors (GPU-first pipeline)...", tensors.size());

    CudaDeviceGuard guard(gpu_device);

    // 创建 CUDA stream 用于所有异步操作
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    int n_converted = 0, n_direct = 0, n_iq4 = 0, n_ggml_quant = 0;
    // 收集需要转换的 tensor 的临时 GPU 缓冲区，同步后统一释放
    std::vector<void*> tmp_gpu_allocs;

    int gdn_key_dim = 0, gdn_value_dim = 0, gdn_conv_dim = 0;
    gdn_dims_from_config(config, tp_size, gdn_key_dim, gdn_value_dim, gdn_conv_dim);

    for (size_t ti = 0; ti < tensors.size(); ++ti) {
        const auto& tinfo = tensors[ti];
        std::string vmc_name = gguf_tensor_name_to_vmc(tinfo.name, num_layers);

        if (tinfo.name.find(".ssm_a") != std::string::npos &&
            tinfo.name.find("ssm_alpha") == std::string::npos &&
            tinfo.name.find("blk.0.") == 0) {
            spdlog::info("GGUF map ssm_a: {} -> {} ne=[{}]", tinfo.name, vmc_name, tinfo.ne[0]);
        }

        // TP sharding decision
        bool is_attn_qkv = vmc_name.find("self_attn.q_proj") != std::string::npos ||
                           vmc_name.find("self_attn.k_proj") != std::string::npos ||
                           vmc_name.find("self_attn.v_proj") != std::string::npos;
        bool is_gate_up = vmc_name.find("gate_proj") != std::string::npos ||
                          vmc_name.find("up_proj") != std::string::npos;
        bool is_output_proj = vmc_name.find("o_proj") != std::string::npos ||
                              vmc_name.find("down_proj") != std::string::npos;

        enum class TpShardMode { None, ColSplit0, RowSplit1, GdnQkv, GdnConv, GdnHeadSplit0 };
        TpShardMode shard_mode = TpShardMode::None;
        if (tp_size > 1) {
            if (is_gdn_qkv_tensor(vmc_name)) {
                shard_mode = TpShardMode::GdnQkv;
            } else if (is_gdn_conv_tensor(vmc_name)) {
                shard_mode = TpShardMode::GdnConv;
            } else if (is_gdn_out_proj(vmc_name)) {
                // 对齐 llama-model.cpp pattern_ssm_out_weight → SPLIT_AXIS_0（沿 value_dim/ne[0]）
                shard_mode = TpShardMode::RowSplit1;
            } else if (is_gdn_z_tensor(vmc_name) ||
                       is_gdn_ba_tensor(vmc_name) || is_gdn_head_vector(vmc_name)) {
                shard_mode = TpShardMode::GdnHeadSplit0;
            } else if (is_attn_qkv || is_gate_up) {
                shard_mode = TpShardMode::ColSplit0;
            } else if (is_output_proj) {
                shard_mode = TpShardMode::RowSplit1;
            }
        }

        int64_t n_elems = tinfo.ne[0] * tinfo.ne[1] * tinfo.ne[2] * tinfo.ne[3];

        // ── IQ4_XS / IQ4_NL：ggml CUDA buffer 分配（对齐 llama.cpp load）──
        if (tinfo.type == GGML_TYPE_IQ4_XS || tinfo.type == GGML_TYPE_IQ4_NL) {
#if !(defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE)
            spdlog::error("IQ4 weights require VM_C_OFFICIAL_GGML=ON");
            continue;
#else
            const ggml_type iq_type = static_cast<ggml_type>(tinfo.type);
            const int BLK_EL = static_cast<int>(ggml_blck_size(iq_type));
            const size_t blk_bytes = ggml_type_size(iq_type);
            int64_t n_total_elems = tinfo.ne[0] * tinfo.ne[1] * tinfo.ne[2] * tinfo.ne[3];
            int64_t n_blocks = (n_total_elems + BLK_EL - 1) / BLK_EL;

            bool is_3d_expert = (tinfo.n_dims == 3 &&
                (vmc_name.find("mlp.experts.gate_proj") != std::string::npos ||
                 vmc_name.find("mlp.experts.up_proj") != std::string::npos ||
                 vmc_name.find("mlp.experts.down_proj") != std::string::npos));

            const void* raw_data = reader.tensor_data(ti);
            if (!raw_data) { spdlog::warn("Null data for {}", vmc_name); continue; }

            int64_t ggml_ne0 = tinfo.ne[0];
            int64_t ggml_ne1 = tinfo.n_dims >= 2 ? tinfo.ne[1] : 1;
            int64_t ggml_ne2 = tinfo.n_dims >= 3 ? tinfo.ne[2] : 1;
            int64_t block_count = n_blocks;
            const void* host_ptr = raw_data;
            size_t host_bytes = static_cast<size_t>(n_blocks) * blk_bytes;
            void* stage_buf = nullptr;

            if (is_3d_expert && tp_size > 1) {
                const bool is_down = vmc_name.find("down_proj") != std::string::npos;
                if (is_down) {
                    const int64_t n_ff = tinfo.ne[0], n_embd = tinfo.ne[1], n_expert = tinfo.ne[2];
                    const int64_t ff_r = n_ff / tp_size;
                    const int blks_per_hidden_row_r = static_cast<int>((ff_r + BLK_EL - 1) / BLK_EL);
                    const int64_t blk_per_exp_r = n_embd * blks_per_hidden_row_r;
                    block_count = blk_per_exp_r * n_expert;
                    host_bytes = static_cast<size_t>(block_count) * blk_bytes;
                    CUDA_CHECK(cudaHostAlloc(&stage_buf, host_bytes, cudaHostAllocDefault));
                    shard_moe_down_experts_qblocks(
                        stage_buf, raw_data, blk_bytes,
                        n_ff, n_embd, n_expert, tp_rank, tp_size);
                    host_ptr = stage_buf;
                    ggml_ne0 = ff_r;
                    ggml_ne1 = n_embd;
                    ggml_ne2 = n_expert;
                } else {
                    const int64_t n_embd = tinfo.ne[0], n_ff = tinfo.ne[1], n_expert = tinfo.ne[2];
                    const int blks_per_embd_row = static_cast<int>((n_embd + BLK_EL - 1) / BLK_EL);
                    const int64_t ff_r = n_ff / tp_size;
                    block_count = ff_r * blks_per_embd_row * n_expert;
                    host_bytes = static_cast<size_t>(block_count) * blk_bytes;
                    CUDA_CHECK(cudaHostAlloc(&stage_buf, host_bytes, cudaHostAllocDefault));
                    shard_moe_iq4_gate_up_experts(
                        static_cast<BlockIQ4XS*>(stage_buf),
                        static_cast<const BlockIQ4XS*>(raw_data),
                        n_embd, n_ff, n_expert, tp_rank, tp_size);
                    host_ptr = stage_buf;
                    ggml_ne0 = n_embd;
                    ggml_ne1 = ff_r;
                    ggml_ne2 = n_expert;
                }
            } else if ((shard_mode == TpShardMode::GdnQkv || shard_mode == TpShardMode::GdnConv)
                       && tp_size > 1 && tinfo.n_dims >= 2) {
                const int64_t n_hidden = tinfo.ne[0];
                block_count = gdn_iq4_qkv_blocks_per_rank(
                    tp_rank, tp_size, n_hidden, gdn_key_dim, gdn_value_dim);
                if (block_count <= 0) {
                    continue;
                }
                host_bytes = static_cast<size_t>(block_count) * blk_bytes;
                CUDA_CHECK(cudaHostAlloc(&stage_buf, host_bytes, cudaHostAllocDefault));
                shard_gdn_iq4_qkv_rows(
                    stage_buf, raw_data, tp_rank, tp_size,
                    n_hidden, gdn_key_dim, gdn_value_dim);
                host_ptr = stage_buf;
                ggml_ne0 = n_hidden;
                ggml_ne1 = gdn_iq4_qkv_out_rows_per_rank(tp_size, gdn_key_dim, gdn_value_dim);
                ggml_ne2 = 1;
            } else if (shard_mode == TpShardMode::GdnHeadSplit0 && tp_size > 1 && tinfo.n_dims >= 2) {
                const int64_t n_hidden = tinfo.ne[0];
                const int64_t n_out_rows = tinfo.ne[1];
                block_count = gdn_iq4_head_blocks_per_rank(
                    tp_rank, tp_size, n_hidden, n_out_rows);
                if (block_count <= 0) continue;
                host_bytes = static_cast<size_t>(block_count) * blk_bytes;
                CUDA_CHECK(cudaHostAlloc(&stage_buf, host_bytes, cudaHostAllocDefault));
                shard_gdn_iq4_head_split0(stage_buf, raw_data, tp_rank, tp_size,
                                          n_hidden, n_out_rows);
                host_ptr = stage_buf;
                const int64_t split_sz = (n_out_rows + tp_size - 1) / tp_size;
                const int64_t row_off = tp_rank * split_sz;
                ggml_ne0 = n_hidden;
                ggml_ne1 = std::min(split_sz, n_out_rows - row_off);
                ggml_ne2 = 1;
            } else if (shard_mode == TpShardMode::ColSplit0 && tp_size > 1 && tinfo.n_dims >= 2) {
                const int64_t n_hidden = tinfo.ne[0];
                const int64_t n_out_rows = tinfo.ne[1];
                block_count = gdn_iq4_head_blocks_per_rank(
                    tp_rank, tp_size, n_hidden, n_out_rows);
                if (block_count <= 0) continue;
                host_bytes = static_cast<size_t>(block_count) * blk_bytes;
                CUDA_CHECK(cudaHostAlloc(&stage_buf, host_bytes, cudaHostAllocDefault));
                shard_gdn_iq4_head_split0(stage_buf, raw_data, tp_rank, tp_size,
                                          n_hidden, n_out_rows);
                host_ptr = stage_buf;
                const int64_t split_sz = (n_out_rows + tp_size - 1) / tp_size;
                const int64_t row_off = tp_rank * split_sz;
                ggml_ne0 = n_hidden;
                ggml_ne1 = std::min(split_sz, n_out_rows - row_off);
                ggml_ne2 = 1;
            } else if (shard_mode == TpShardMode::RowSplit1 && tp_size > 1 && tinfo.n_dims >= 2) {
                const int64_t ne0 = tinfo.ne[0];
                const int64_t ne1 = tinfo.ne[1];
                block_count = iq4_row_split1_blocks_per_rank(tp_rank, tp_size, ne0, ne1);
                if (block_count <= 0) continue;
                host_bytes = static_cast<size_t>(block_count) * blk_bytes;
                CUDA_CHECK(cudaHostAlloc(&stage_buf, host_bytes, cudaHostAllocDefault));
                shard_iq4_row_split1(stage_buf, raw_data, tp_rank, tp_size, ne0, ne1);
                host_ptr = stage_buf;
                const int64_t split_sz = (ne0 + tp_size - 1) / tp_size;
                const int64_t col_off = tp_rank * split_sz;
                ggml_ne0 = std::min(split_sz, ne0 - col_off);
                ggml_ne1 = ne1;
                ggml_ne2 = 1;
            } else if (shard_mode == TpShardMode::GdnHeadSplit0 && tp_size > 1 && tinfo.n_dims <= 1) {
                const int64_t split_sz = (n_elems + tp_size - 1) / tp_size;
                const int64_t src_off = tp_rank * split_sz;
                const int64_t n_r = std::min(split_sz, n_elems - src_off);
                if (n_r <= 0) continue;
                block_count = (n_r + BLK_EL - 1) / BLK_EL;
                host_bytes = static_cast<size_t>(block_count) * blk_bytes;
                CUDA_CHECK(cudaHostAlloc(&stage_buf, host_bytes, cudaHostAllocDefault));
                std::memcpy(stage_buf,
                            static_cast<const uint8_t*>(raw_data)
                                + static_cast<size_t>(src_off / BLK_EL) * blk_bytes,
                            host_bytes);
                host_ptr = stage_buf;
                ggml_ne0 = n_r;
                ggml_ne1 = 1;
                ggml_ne2 = 1;
            }

            GpuTensor gpu_iq4 = make_ggml_quant_from_host(
                iq_type, gpu_device, ggml_ne0, ggml_ne1, ggml_ne2,
                host_ptr, host_bytes, block_count, stream);
            if (stage_buf) {
                CUDA_CHECK(cudaFreeHost(stage_buf));
            }

            weights.add_weight(vmc_name, std::move(gpu_iq4));
            n_iq4++;
            continue;
#endif
        }

        // MoE 3D Q5_K down_exps → ggml CUDA buffer（供 MUL_MAT_ID，与 IQ4 gate/up 同路径）
        if (tinfo.n_dims == 3 && is_moe_3d_expert_tensor(vmc_name) &&
            vmc_name.find("down_proj") != std::string::npos &&
            tinfo.type == GGML_TYPE_Q5_K) {
#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
            const void* raw_data = reader.tensor_data(ti);
            if (!raw_data) { spdlog::warn("Null data for {}", vmc_name); continue; }

            const int64_t n_ff = tinfo.ne[0], n_embd = tinfo.ne[1], n_expert = tinfo.ne[2];
            const size_t blk_bytes = sizeof(block_q5_K);
            const int blks_per_hidden_row_full = static_cast<int>((n_ff + QK_K - 1) / QK_K);
            // block-aligned TP split: 整除时与 n_ff/tp_size 一致, 否则按 block 对齐不均分
            const int blks_lo = tp_size > 1 ? (tp_rank * blks_per_hidden_row_full) / tp_size : 0;
            const int blks_hi = tp_size > 1 ? ((tp_rank + 1) * blks_per_hidden_row_full) / tp_size
                                           : blks_per_hidden_row_full;
            const int64_t ff_r = std::min(
                static_cast<int64_t>(blks_hi) * QK_K, n_ff)
                - static_cast<int64_t>(blks_lo) * QK_K;
            const int blks_per_hidden_row_r = blks_hi - blks_lo;
            const int64_t blk_per_exp_r = n_embd * blks_per_hidden_row_r;
            int64_t block_count = blk_per_exp_r * n_expert;
            size_t host_bytes = static_cast<size_t>(block_count) * blk_bytes;
            const void* host_ptr = raw_data;
            void* stage_buf = nullptr;

            if (tp_size > 1) {
                CUDA_CHECK(cudaHostAlloc(&stage_buf, host_bytes, cudaHostAllocDefault));
                shard_moe_down_experts_qblocks(
                    stage_buf, raw_data, blk_bytes,
                    n_ff, n_embd, n_expert, tp_rank, tp_size);
                host_ptr = stage_buf;
            }

            GpuTensor gpu_q5 = make_ggml_q5_k_from_host(
                gpu_device, ff_r, n_embd, n_expert,
                host_ptr, host_bytes, block_count, stream);
            if (stage_buf) {
                CUDA_CHECK(cudaFreeHost(stage_buf));
            }

            weights.add_weight(vmc_name, std::move(gpu_q5));
            n_iq4++;
            continue;
#else
            spdlog::error("MoE Q5_K down weights require VM_C_OFFICIAL_GGML=ON");
            continue;
#endif
        }

        // ── 其他量化类型：ggml 原生 buffer（对齐 llama.cpp gguf_get_tensor_type，供 bind）──
#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
        if (ggml_type_is_quantized(tinfo.type)) {
            const GgmlTypeTraits& traits = ggml_type_traits(tinfo.type);
            if (traits.type_size == 0 || traits.blck_size <= 0) {
                throw std::runtime_error(
                    "GGUF quant tensor " + vmc_name + " has unsupported ggml type "
                    + std::to_string(static_cast<int>(tinfo.type)));
            }
            const size_t blk_bytes = traits.type_size;
            const int blk_elems = static_cast<int>(traits.blck_size);

            const int64_t full_ne0 = tinfo.ne[0];
            const int64_t full_ne1 = tinfo.n_dims >= 2 ? tinfo.ne[1] : 1;
            const int64_t full_ne2 = tinfo.n_dims >= 3 ? tinfo.ne[2] : 1;

            int64_t ggml_ne0 = full_ne0;
            int64_t ggml_ne1 = full_ne1;
            int64_t ggml_ne2 = full_ne2;

            const void* raw_data = reader.tensor_data(ti);
            if (!raw_data) {
                spdlog::warn("Null data for {}", vmc_name);
                continue;
            }

            const void* host_ptr = raw_data;
            void* stage_buf = nullptr;
            int64_t block_count = 0;
            size_t host_bytes = 0;

            if (shard_mode == TpShardMode::GdnHeadSplit0 && tp_size > 1 && tinfo.n_dims >= 2) {
                block_count = ggml_quant_col_split0_blocks_per_rank(
                    tp_rank, tp_size, full_ne0, full_ne1, blk_elems);
                if (block_count <= 0) {
                    continue;
                }
                host_bytes = static_cast<size_t>(block_count) * blk_bytes;
                CUDA_CHECK(cudaHostAlloc(&stage_buf, host_bytes, cudaHostAllocDefault));
                shard_ggml_quant_col_split0(
                    stage_buf, raw_data, blk_bytes, tp_rank, tp_size,
                    full_ne0, full_ne1, blk_elems);
                host_ptr = stage_buf;
                const int64_t split_sz = (full_ne1 + tp_size - 1) / tp_size;
                const int64_t row_off = tp_rank * split_sz;
                ggml_ne0 = full_ne0;
                ggml_ne1 = std::min(split_sz, full_ne1 - row_off);
            } else if ((shard_mode == TpShardMode::GdnQkv || shard_mode == TpShardMode::GdnConv)
                       && tp_size > 1 && tinfo.n_dims >= 2) {
                block_count = gdn_ggml_quant_qkv_blocks_per_rank(
                    tp_size, full_ne0, gdn_key_dim, gdn_value_dim, blk_elems);
                if (block_count <= 0) {
                    continue;
                }
                host_bytes = static_cast<size_t>(block_count) * blk_bytes;
                CUDA_CHECK(cudaHostAlloc(&stage_buf, host_bytes, cudaHostAllocDefault));
                shard_gdn_ggml_quant_qkv_rows(
                    stage_buf, raw_data, blk_bytes, tp_rank, tp_size,
                    full_ne0, gdn_key_dim, gdn_value_dim, blk_elems);
                host_ptr = stage_buf;
                ggml_ne0 = full_ne0;
                ggml_ne1 = gdn_iq4_qkv_out_rows_per_rank(tp_size, gdn_key_dim, gdn_value_dim);
            } else if (shard_mode == TpShardMode::ColSplit0 && tp_size > 1 && tinfo.n_dims >= 2) {
                block_count = ggml_quant_col_split0_blocks_per_rank(
                    tp_rank, tp_size, full_ne0, full_ne1, blk_elems);
                if (block_count <= 0) {
                    continue;
                }
                host_bytes = static_cast<size_t>(block_count) * blk_bytes;
                CUDA_CHECK(cudaHostAlloc(&stage_buf, host_bytes, cudaHostAllocDefault));
                shard_ggml_quant_col_split0(
                    stage_buf, raw_data, blk_bytes, tp_rank, tp_size,
                    full_ne0, full_ne1, blk_elems);
                host_ptr = stage_buf;
                const int64_t split_sz = (full_ne1 + tp_size - 1) / tp_size;
                const int64_t row_off = tp_rank * split_sz;
                ggml_ne0 = full_ne0;
                ggml_ne1 = std::min(split_sz, full_ne1 - row_off);
            } else if (shard_mode == TpShardMode::RowSplit1 && tp_size > 1 && tinfo.n_dims >= 2) {
                block_count = ggml_quant_row_split1_blocks_per_rank(
                    tp_rank, tp_size, full_ne0, full_ne1, blk_elems);
                if (block_count <= 0) {
                    continue;
                }
                host_bytes = static_cast<size_t>(block_count) * blk_bytes;
                CUDA_CHECK(cudaHostAlloc(&stage_buf, host_bytes, cudaHostAllocDefault));
                shard_ggml_quant_row_split1(
                    stage_buf, raw_data, blk_bytes, tp_rank, tp_size,
                    full_ne0, full_ne1, blk_elems);
                host_ptr = stage_buf;
                // block-aligned ne0: RowSplit1 按 block 对齐分片 (eg. 768→3 blk, TP=2: rank0=256, rank1=512)
                const int total_blks = static_cast<int>((full_ne0 + blk_elems - 1) / blk_elems);
                const int blks_lo = (tp_rank * total_blks) / tp_size;
                const int blks_hi = ((tp_rank + 1) * total_blks) / tp_size;
                ggml_ne0 = std::min(static_cast<int64_t>(blks_hi) * blk_elems, full_ne0)
                         - static_cast<int64_t>(blks_lo) * blk_elems;
                ggml_ne1 = full_ne1;
            } else {
                const int64_t n_elems = full_ne0 * full_ne1 * full_ne2
                    * (tinfo.n_dims >= 4 ? tinfo.ne[3] : 1);
                block_count = (n_elems + blk_elems - 1) / blk_elems;
                host_bytes = static_cast<size_t>(block_count) * blk_bytes;
            }

            GpuTensor gpu_quant = make_ggml_quant_from_host(
                static_cast<ggml_type>(tinfo.type),
                gpu_device, ggml_ne0, ggml_ne1, ggml_ne2,
                host_ptr, host_bytes, block_count, stream);
            if (stage_buf) {
                CUDA_CHECK(cudaFreeHost(stage_buf));
            }

            weights.add_weight(vmc_name, std::move(gpu_quant));
            n_ggml_quant++;
            continue;
        }
#endif

        // ── 非 IQ4_XS 张量 ──

        // (legacy raw Q5 path removed — down_exps must use ggml buffer above)

        // 计算 shape
        Shape shape;
        if (tinfo.n_dims <= 1) {
            shape = Shape({n_elems});
        } else {
            shape = Shape({tinfo.ne[1], tinfo.ne[0]});
        }

        // TP shape 调整
        if (shard_mode == TpShardMode::GdnQkv || shard_mode == TpShardMode::GdnConv) {
            const int shard_k = gdn_key_dim / tp_size;
            const int shard_v = gdn_value_dim / tp_size;
            shape = Shape({2 * shard_k + shard_v, tinfo.ne[0]});
        } else if (shard_mode == TpShardMode::GdnHeadSplit0) {
            int64_t out_dim = shape[0];
            int64_t split_sz = (out_dim + tp_size - 1) / tp_size;
            int64_t split_off = tp_rank * split_sz;
            int64_t actual = std::min(split_sz, out_dim - split_off);
            if (actual <= 0) continue;
            shape.dims[0] = actual;
        } else if (shard_mode == TpShardMode::ColSplit0) {
            int64_t dim_size = shape[0];
            int64_t split_sz = (dim_size + tp_size - 1) / tp_size;
            int64_t split_off = tp_rank * split_sz;
            int64_t actual = std::min(split_sz, dim_size - split_off);
            if (actual <= 0) continue;
            shape.dims[0] = actual;
        } else if (shard_mode == TpShardMode::RowSplit1) {
            int64_t dim_size = shape[1];
            int64_t split_sz = (dim_size + tp_size - 1) / tp_size;
            int64_t split_off = tp_rank * split_sz;
            int64_t actual = std::min(split_sz, dim_size - split_off);
            if (actual <= 0) continue;
            shape.dims[1] = actual;
        }

        int64_t total_elems = shape.numel();
        const void* src_data = reader.tensor_data(ti);
        if (!src_data) { spdlog::warn("Null data for {}", vmc_name); continue; }

        if (!needs_gpu_convert(tinfo.type)) {
            // ── 无需转换：按 GGUF 原生 dtype 直接异步上传（对齐 llama.cpp）──
            const DataType store_dtype = gguf_native_storage_dtype(tinfo.type);
            const size_t elem_sz = dtype_size(store_dtype);
            GpuTensor gpu_tensor(shape, store_dtype, gpu_device);
            size_t nbytes = total_elems * elem_sz;

            if (shard_mode != TpShardMode::None && tp_size > 1) {
                void* stage = nullptr;
                CUDA_CHECK(cudaHostAlloc(&stage, nbytes, cudaHostAllocDefault));
                const int elem_sz_i = static_cast<int>(elem_sz);
                int row_bytes = static_cast<int>(tinfo.ne[0] * elem_sz_i);
                const uint8_t* src = (const uint8_t*)src_data;
                uint8_t* dst = (uint8_t*)stage;

                if (shard_mode == TpShardMode::GdnQkv) {
                    shard_gdn_qkv_rows(dst, src, tp_rank, tp_size,
                                       gdn_key_dim, gdn_value_dim, row_bytes);
                } else if (shard_mode == TpShardMode::GdnConv) {
                    shard_gdn_conv_rows(dst, src, tp_rank, tp_size,
                                        gdn_key_dim, gdn_value_dim, row_bytes);
                } else if (shard_mode == TpShardMode::GdnHeadSplit0) {
                    if (tinfo.n_dims <= 1) {
                        int64_t split_sz = (n_elems + tp_size - 1) / tp_size;
                        int64_t src_off = tp_rank * split_sz;
                        memcpy(dst, src + src_off * elem_sz_i, shape.numel() * elem_sz_i);
                    } else {
                        int64_t full_out = tinfo.ne[1];
                        int64_t split_sz = (full_out + tp_size - 1) / tp_size;
                        int64_t src_off = tp_rank * split_sz;
                        for (int64_t r = 0; r < shape[0]; ++r) {
                            memcpy(dst + r * row_bytes,
                                   src + (src_off + r) * row_bytes,
                                   row_bytes);
                        }
                    }
                } else if (shard_mode == TpShardMode::ColSplit0) {
                    int64_t full_out = tinfo.ne[1];
                    int64_t split_sz = (full_out + tp_size - 1) / tp_size;
                    int64_t src_off = tp_rank * split_sz;
                    for (int64_t r = 0; r < shape[0]; ++r) {
                        memcpy(dst + r * row_bytes,
                               src + (src_off + r) * row_bytes,
                               row_bytes);
                    }
                } else if (shard_mode == TpShardMode::RowSplit1) {
                    int64_t split_sz = (tinfo.ne[0] + tp_size - 1) / tp_size;
                    int64_t src_col_off = tp_rank * split_sz;
                    int dst_stride = static_cast<int>(shape[1] * elem_sz_i);
                    int src_stride = row_bytes;
                    for (int64_t r = 0; r < tinfo.ne[1]; ++r) {
                        memcpy(dst + r * dst_stride,
                               src + r * src_stride + src_col_off * elem_sz_i,
                               dst_stride);
                    }
                }
                CUDA_CHECK(cudaMemcpyAsync(gpu_tensor.data_ptr(), stage, nbytes, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaFreeHost(stage));
            } else {
                // 直接异步上传
                CUDA_CHECK(cudaMemcpyAsync(gpu_tensor.data_ptr(), src_data, nbytes,
                                           cudaMemcpyHostToDevice, stream));
            }

            weights.add_weight(vmc_name, std::move(gpu_tensor));
            n_direct++;
        } else {
#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
            if (ggml_type_is_quantized(tinfo.type)) {
                throw std::runtime_error(
                    "GGUF quant tensor reached convert path (should use ggml buffer): " + vmc_name);
            }
#endif
            // ── 需要 GPU 格式转换（legacy 非 libllama 路径） ──
            // Step 1: 在 GPU 上分配目标张量（compute_dtype 决定 FP16 或 BF16）
            GpuTensor gpu_tensor(shape, gpu().compute_dtype(), gpu_device);
            size_t fp16_bytes = total_elems * dtype_sz;

            // Step 2: 上传原始格式到 GPU（用 pinned staging 暂存）
            // 计算原始格式在 GGUF 文件中占用的字节
            size_t src_bytes;
            if (tinfo.type == GGML_TYPE_F32) {
                src_bytes = total_elems * sizeof(float);
            } else if (tinfo.type == GGML_TYPE_BF16) {
                src_bytes = total_elems * sizeof(uint16_t);
            } else {
                int64_t blk_elems = ggml_blk_elems(tinfo.type);
                int64_t n_blk = (total_elems + blk_elems - 1) / blk_elems;
                src_bytes = n_blk * ggml_block_size_bytes(tinfo.type);
            }

            // 使用 cudaMalloc 分配临时 GPU 缓冲区用于存原始格式
            void* tmp_gpu = nullptr;
            CUDA_CHECK(cudaMalloc(&tmp_gpu, src_bytes));
            tmp_gpu_allocs.push_back(tmp_gpu);

            // Step 2: 异步上传原始格式到 GPU 临时缓冲区
            CUDA_CHECK(cudaMemcpyAsync(tmp_gpu, src_data, src_bytes,
                                       cudaMemcpyHostToDevice, stream));

            // Step 3: 在 GPU 上启动转换 kernel（异步，在 stream 中排队）
            launch_gpu_convert(
                gpu_tensor.data_ptr(),
                tmp_gpu, static_cast<int>(tinfo.type), total_elems,
                gpu().compute_dtype() == DataType::BFLOAT16, stream);

            weights.add_weight(vmc_name, std::move(gpu_tensor));
            n_converted++;
        }
    }

    // 同步所有 GPU 操作（等待所有 async 传输和转换 kernel 完成）
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // 释放所有临时 GPU 缓冲区（转换完成后安全释放）
    for (auto* p : tmp_gpu_allocs) {
        CUDA_CHECK(cudaFree(p));
    }

    spdlog::info(
        "GGUF: loaded {} tensors ({} native-dense + {} ggml-quant [iq4={} other={}] + {} converted)",
        weights.gpu_weights.size(), n_direct, n_iq4 + n_ggml_quant, n_iq4, n_ggml_quant, n_converted);
}

} // namespace vm_c
