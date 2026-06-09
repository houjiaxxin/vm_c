#include "vm_c/core/gguf_reader.hpp"
#include "vm_c/core/ggml_dequant.hpp"
#include <cstring>
#include "vm_c/model/llama_gguf_tensor_map.hpp"
#include "vm_c/model/model_loader.hpp"
#include "vm_c/model/tensor_schema.hpp"
#include "vm_c/cuda/gpu_arch.hpp"
#include "vm_c/cuda/vm_c_tensor.h"
#include "vm_c/cuda/convert_kernels.h"
#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
#include "vm_c/vmc_engine/ggml_weight.hpp"
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
    tensor.adopt_ggml_weight(storage, Shape({ne0, ne1, ne2}), DataType::UINT8, gpu_device);
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
    tensor.adopt_ggml_weight(storage, Shape({ne0, ne1, ne2}), DataType::UINT8, gpu_device);
    return tensor;
}

// ── TP 模式下 token_embd 复制：从 pre-sharded GGUF 加载完整 vocab embedding ──
// GGUF 文件是 pre-sharded 的（token_embd.weight = [shard(n_vocab), n_embd]），
// 每个 rank 加载后需要复制成完整 [n_vocab, n_embd]，因为 TP runtime 会发送
// 全部 token 给所有 rank，每个 rank 都需要完整的 embedding 矩阵。
static void replicate_token_embd_to_full_vocab(
    void* dst, const void* src,
    int64_t n_embd, int64_t n_vocab_shard, int tp_rank, int tp_size,
    int64_t blk_elems, size_t blk_bytes, bool is_quantized,
    size_t elem_size = 2) {  // 默认 F16，非量化路径可指定
    const int64_t n_vocab_full = n_vocab_shard * tp_size;
    if (is_quantized) {
        // 量化类型：需要 block 对齐
        const int64_t total_blks = (n_vocab_full * n_embd + blk_elems - 1) / blk_elems;
        const size_t total_bytes = total_blks * blk_bytes;
        std::memset(dst, 0, total_bytes);
        const int64_t shard_blks = (n_vocab_shard * n_embd + blk_elems - 1) / blk_elems;
        // 将 shard 复制到对应位置（rank * shard_blks）
        std::memcpy(static_cast<uint8_t*>(dst) + static_cast<size_t>(tp_rank * shard_blks) * blk_bytes,
                    src, static_cast<size_t>(shard_blks) * blk_bytes);
    } else {
        const size_t row_bytes = static_cast<size_t>(n_embd) * elem_size;
        const size_t total_bytes = static_cast<size_t>(n_vocab_full) * row_bytes;
        std::memset(dst, 0, total_bytes);
        const size_t shard_bytes = static_cast<size_t>(n_vocab_shard) * row_bytes;
        // 将 shard 复制到对应位置
        std::memcpy(static_cast<uint8_t*>(dst) + static_cast<size_t>(tp_rank) * shard_bytes,
                    src, shard_bytes);
    }
}

// K-quant / block-quant TP 分片（按 block 字节 stride，对齐 IQ4 路径）
// 块计数统一由 ShardContext 计算，此处仅做数据搬运
static int64_t shard_ggml_quant_col_split0(
    void* dst, const void* src, size_t blk_bytes,
    int tp_rank, int tp_size,
    int64_t ne0, int64_t ne1, int64_t ne2, int blk_elems) {
    // ne0 = row width (GGUF ne[0]), ne1 = row count (GGUF ne[1]), ne2 = expert count (GGUF ne[2])
    // 3D 张量：总行数 = ne1 * ne2，沿 ne1 维度分片
    const int64_t total_rows = ne1 * ne2;
    if (tp_size > 1 && (total_rows % tp_size != 0)) {
        throw std::runtime_error("TP Error: total_rows (" + std::to_string(total_rows) + ") must be divisible by tp_size (" + std::to_string(tp_size) + ") for shard_ggml_quant_col_split0");
    }
    const int blks_per_row = static_cast<int>((ne0 + blk_elems - 1) / blk_elems);
    const int64_t split_sz = total_rows / tp_size;
    const int64_t row_off = tp_rank * split_sz;
    if (split_sz <= 0) return 0;
    const auto* src_b = static_cast<const uint8_t*>(src);
    auto* dst_b = static_cast<uint8_t*>(dst);
    for (int64_t r = 0; r < split_sz; ++r) {
        std::memcpy(dst_b + static_cast<size_t>(r * blks_per_row) * blk_bytes,
                    src_b + static_cast<size_t>(row_off + r) * blks_per_row * blk_bytes,
                    static_cast<size_t>(blks_per_row) * blk_bytes);
    }
    return split_sz * blks_per_row;
}

// RowSplit1: 块计数统一由 ShardContext 计算，此处仅做数据搬运
// 支持 3D：ne2 > 1 时，总行数 = ne1 * ne2，沿 ne0 维度分片（列分片）
static int64_t shard_ggml_quant_row_split1(
    void* dst, const void* src, size_t blk_bytes,
    int tp_rank, int tp_size,
    int64_t ne0, int64_t ne1, int64_t ne2, int blk_elems) {
    // ne0 = row width (GGUF ne[0]), ne1 = row count (GGUF ne[1]), ne2 = expert count (GGUF ne[2])
    const int64_t total_rows = ne1 * ne2;
    const int blks_per_row_full = static_cast<int>((ne0 + blk_elems - 1) / blk_elems);
    if (tp_size > 1 && (blks_per_row_full % tp_size != 0)) {
        throw std::runtime_error("TP Error: blks_per_row_full (" + std::to_string(blks_per_row_full) + ") must be divisible by tp_size (" + std::to_string(tp_size) + ") for shard_ggml_quant_row_split1");
    }
    const int blks_per_row_r = blks_per_row_full / tp_size;
    const int src_blk_off = tp_rank * blks_per_row_r;
    if (blks_per_row_r <= 0) return 0;
    const auto* src_b = static_cast<const uint8_t*>(src);
    auto* dst_b = static_cast<uint8_t*>(dst);
    int64_t dst_count = 0;
    for (int64_t row = 0; row < total_rows; ++row) {
        std::memcpy(dst_b + static_cast<size_t>(dst_count) * blk_bytes,
                    src_b + static_cast<size_t>(row * blks_per_row_full + src_blk_off) * blk_bytes,
                    static_cast<size_t>(blks_per_row_r) * blk_bytes);
        dst_count += blks_per_row_r;
    }
    return dst_count;
}


// ── GDN BA TP 分片（ssm_beta_alpha: [beta(n_v_heads) | alpha(n_v_heads)] → [beta_local | alpha_local]）──
// 块计数统一由 ShardContext 计算，此处仅做数据搬运
// ne2 仅用于签名兼容（GDN 张量为 2D，ne2=1）
static int64_t shard_ggml_quant_ba_split0(
    void* dst, const void* src, size_t blk_bytes,
    int tp_rank, int tp_size,
    int64_t ne0, int64_t ne1, int64_t ne2, int blk_elems) {
    (void)ne2;
    // Interleaved split: [beta_heads | alpha_heads] → [beta_local | alpha_local]
    // Each rank gets n_v_heads/tp rows from beta section + n_v_heads/tp rows from alpha section
    const int blks_per_row = static_cast<int>((ne0 + blk_elems - 1) / blk_elems);
    const int64_t n_v_heads = ne1 / 2;
    const int64_t n_v_heads_local = n_v_heads / tp_size;
    if (n_v_heads_local <= 0) return 0;
    const auto* src_b = static_cast<const uint8_t*>(src);
    auto* dst_b = static_cast<uint8_t*>(dst);
    const size_t row_bytes = static_cast<size_t>(blks_per_row) * blk_bytes;
    // Copy beta section (rows 0..n_v_heads-1)
    const int64_t beta_off = tp_rank * n_v_heads_local;
    std::memcpy(dst_b,
                src_b + beta_off * row_bytes,
                static_cast<size_t>(n_v_heads_local) * row_bytes);
    // Copy alpha section (rows n_v_heads..2*n_v_heads-1)
    const int64_t alpha_off = n_v_heads + tp_rank * n_v_heads_local;
    std::memcpy(dst_b + static_cast<size_t>(n_v_heads_local) * row_bytes,
                src_b + alpha_off * row_bytes,
                static_cast<size_t>(n_v_heads_local) * row_bytes);
    return 2 * n_v_heads_local * blks_per_row;
}
#endif

// ── 张量分片决策由 TensorSchema + ShardContext 统一管理 ──

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

// IQ4_XS GDN 行分片：ggml ne=[hidden, out_rows]
// 块计数统一由 ShardContext 计算，此处仅做数据搬运
static int64_t shard_gdn_iq4_head_split0(
    void* dst, const void* src,
    int tp_rank, int tp_size,
    int64_t n_hidden, int64_t n_out_rows) {
    constexpr int BLK_EL = 256;
    if (tp_size > 1 && (n_out_rows % tp_size != 0)) {
        throw std::runtime_error("TP Error: n_out_rows (" + std::to_string(n_out_rows) + ") must be divisible by tp_size (" + std::to_string(tp_size) + ") for shard_gdn_iq4_head_split0");
    }
    const int blks_per_row = static_cast<int>((n_hidden + BLK_EL - 1) / BLK_EL);
    const int64_t split_sz = n_out_rows / tp_size;
    const int64_t src_off = tp_rank * split_sz;
    const int64_t n_rows_r = split_sz;
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

// IQ4 GDN QKV/conv：三段式 TP 分片
static void shard_gdn_iq4_qkv_rows(
    void* dst, const void* src,
    int tp_rank, int tp_size,
    int64_t n_hidden, int key_dim, int value_dim) {
    constexpr int BLK_EL = 256;
    const int blks_per_row = static_cast<int>((n_hidden + BLK_EL - 1) / BLK_EL);
    const int shard_k = key_dim / tp_size;
    const int shard_v = value_dim / tp_size;
    if (shard_k <= 0 || shard_v <= 0) return;
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
}

// GDN GGML quant OJKV 三段式分片：块计数统一由 ShardContext 计算
static void shard_gdn_ggml_quant_qkv_rows(
    void* dst, const void* src, size_t blk_bytes,
    int tp_rank, int tp_size,
    int64_t row_elems, int key_dim, int value_dim, int blk_elems) {
    const int blks_per_row = static_cast<int>((row_elems + blk_elems - 1) / blk_elems);
    const int shard_k = key_dim / tp_size;
    const int shard_v = value_dim / tp_size;
    if (shard_k <= 0 || shard_v <= 0) return;
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
}

// IQ4 row-parallel（o_proj / down_proj）：沿 ne0 做 TP 分片，块计数统一由 ShardContext 计算
// 支持 3D：ne2 > 1 时，总行数 = ne1 * ne2
static int64_t shard_iq4_row_split1(
    void* dst, const void* src,
    int tp_rank, int tp_size,
    int64_t ne0, int64_t ne1, int64_t ne2) {
    constexpr int BLK_EL = 256;
    const int64_t total_rows = ne1 * ne2;
    const int blks_per_row_full = static_cast<int>((ne0 + BLK_EL - 1) / BLK_EL);
    if (tp_size > 1 && (blks_per_row_full % tp_size != 0)) {
        throw std::runtime_error("TP Error: blks_per_row_full (" + std::to_string(blks_per_row_full) + ") must be divisible by tp_size (" + std::to_string(tp_size) + ") for shard_iq4_row_split1");
    }
    const int blks_per_row_r = blks_per_row_full / tp_size;
    const int src_blk_off = tp_rank * blks_per_row_r;
    if (blks_per_row_r <= 0) {
        return 0;
    }
    const auto* src_blk = static_cast<const BlockIQ4XS*>(src);
    auto* dst_blk = static_cast<BlockIQ4XS*>(dst);
    int64_t dst_count = 0;
    for (int64_t row = 0; row < total_rows; ++row) {
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
    if (tp_size > 1 && (blks_per_hidden_row_full % tp_size != 0)) {
        throw std::runtime_error("TP Error: blks_per_hidden_row_full (" + std::to_string(blks_per_hidden_row_full) + ") must be divisible by tp_size (" + std::to_string(tp_size) + ") for shard_moe_down_experts_qblocks");
    }
    const int blks_per_hidden_row_r = blks_per_hidden_row_full / tp_size;
    const int ff_blk_off = tp_rank * blks_per_hidden_row_r;

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
    cfg.arch = normalize_arch_name(arch_lower);

    auto ak = [&](const std::string& key) { return arch_key(arch_lower, key); };

    cfg.hidden_size = (int)reader.get_i64(ak(GgufKeys::EMBEDDING_LENGTH), 0);
    cfg.num_hidden_layers = (int)reader.get_i64(ak(GgufKeys::BLOCK_COUNT), 0);
    cfg.num_attention_heads = (int)reader.get_i64(ak(GgufKeys::ATTENTION_HEAD_COUNT), 0);
    cfg.num_key_value_heads = (int)reader.get_i64(ak(GgufKeys::ATTENTION_HEAD_COUNT_KV), 0);
    if (cfg.num_key_value_heads <= 0) {
        throw std::runtime_error(
            "GgufModelLoader: ATTENTION_HEAD_COUNT_KV missing or zero in GGUF metadata, "
            "cannot determine num_key_value_heads dynamically");
    }
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
    cfg.rms_norm_eps = reader.get_f32(ak(GgufKeys::ATTENTION_LAYERNORM_RMS_EPS), 0.0f);
    if (cfg.rms_norm_eps <= 0.0f) {
        throw std::runtime_error(
            "GgufModelLoader: ATTENTION_LAYERNORM_RMS_EPS missing or zero in GGUF metadata");
    }
    cfg.rope_theta = reader.get_f32(ak(GgufKeys::ROPE_FREQ_BASE), 0.0f);
    if (cfg.rope_theta <= 0.0f) {
        throw std::runtime_error(
            "GgufModelLoader: ROPE_FREQ_BASE missing or zero in GGUF metadata");
    }
    cfg.head_dim = (int)reader.get_i32(ak(GgufKeys::ATTENTION_KEY_LENGTH), 0);
    if (cfg.head_dim <= 0) {
        throw std::runtime_error(
            "GgufModelLoader: ATTENTION_KEY_LENGTH missing or zero in GGUF metadata, "
            "cannot determine head_dim dynamically");
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
        // 不再自动推断 spec_method；由用户 --spec-method 显式控制，默认 "none"
    }

    // 混合注意力层类型：Qwen3.5/3.6 转换脚本（qwen.py）写入的是 full_attention_interval
    {
        int full_attn_interval = (int)reader.get_i64(ak(GgufKeys::FULL_ATTENTION_INTERVAL), 0);
        if (full_attn_interval <= 0 || cfg.num_hidden_layers <= 0) {
            throw std::runtime_error(
                "GgufModelLoader: hybrid attention model requires '" +
                ak(GgufKeys::FULL_ATTENTION_INTERVAL) + "' in GGUF metadata");
        }
        const int n_main = cfg.num_hidden_layers - cfg.mtp_predict_layers;
        cfg.layer_types.clear();
        cfg.layer_types.reserve(cfg.num_hidden_layers);
        for (int i = 0; i < cfg.num_hidden_layers; ++i) {
            if (i >= n_main) {
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

    cfg.intermediate_size = (int)reader.get_i64(ak(GgufKeys::FEED_FORWARD_LENGTH), 0);
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
    // 临时 GPU 缓冲区，同步后统一释放（仅 convert 路径需要）
    std::vector<void*> tmp_gpu_allocs;

    // ── TensorSchema 预计算显存预算（复用已加载的 GGUF reader，不额外分配任何 GPU 资源） ──
    {
        WeightBudget budget;
        const ArchSchema* bs = get_arch_schema(config.arch);
        if (bs) {
            const int bnum_layers = config.num_hidden_layers;
            const auto& btensors = reader.all_tensors();
            budget.items.reserve(btensors.size());
            for (size_t bi = 0; bi < btensors.size(); ++bi) {
                const auto& btinfo = btensors[bi];
                const TensorSchemaEntry* be = bs->find_by_gguf(btinfo.name, bnum_layers);
                if (!be) continue;
                if (be->is_mtp && config.spec_method != "mtp") continue;
                WeightBudget::Item bitem;
                bitem.vmc_name = be->vmc_name;
                bitem.gguf_name = btinfo.name;
                bitem.shard_mode = be->shard_mode;
                for (int i = 0; i < 4; ++i) bitem.gguf_ne[i] = btinfo.ne[i];
                const bool bq = ggml_type_is_quantized(btinfo.type);
                bitem.is_quantized = bq;
                if (bq) {
                    bitem.rank_bytes = compute_quant_alloc_bytes(be->ne, config, 0, tp_size, btinfo);
                    const GgmlTypeTraits& btr = ggml_type_traits(btinfo.type);
                    int64_t belem = btinfo.ne[0] * btinfo.ne[1] * btinfo.ne[2];
                    if (btinfo.n_dims >= 4) belem *= btinfo.ne[3];
                    int64_t bfb = (belem + btr.blck_size - 1) / btr.blck_size;
                    bitem.total_bytes = static_cast<size_t>(bfb) * btr.type_size;
                    // 计算并存储分片后维度（用于 dump 显示）
                    for (int i = 0; i < 4; ++i) {
                        bitem.sharded_ne[i] = be->ne[i].eval(config, tp_size, btinfo.ne);
                    }
                } else {
                    int64_t nev[4];
                    for (int i = 0; i < 4; ++i) {
                        nev[i] = be->ne[i].eval(config, tp_size, btinfo.ne);
                        bitem.sharded_ne[i] = nev[i];
                    }
                    Shape sh;
                    if (btinfo.n_dims <= 1) sh = Shape({nev[0]});
                    else if (btinfo.n_dims == 2) sh = Shape({nev[1], nev[0]});
                    else sh = Shape({nev[2], nev[1], nev[0]});
                    size_t esz = dtype_size(ggml_type_to_vmc_dtype(btinfo.type));
                    bitem.rank_bytes = static_cast<size_t>(sh.numel()) * esz;
                    bitem.total_bytes = bitem.rank_bytes * tp_size;
                }
                bitem.rank_mib = static_cast<double>(bitem.rank_bytes) / (1048576.0);
                budget.total_rank_bytes += bitem.rank_bytes;
                budget.items.push_back(std::move(bitem));
            }
            std::sort(budget.items.begin(), budget.items.end(),
                      [](const WeightBudget::Item& a, const WeightBudget::Item& b) {
                          return a.rank_bytes > b.rank_bytes;
                      });
        }
        budget.total_rank_mib = static_cast<double>(budget.total_rank_bytes) / (1048576.0);
        budget.dump();
        // 查询实际 GPU 显存容量
        cudaDeviceProp props;
        CUDA_CHECK(cudaGetDeviceProperties(&props, gpu_device));
        const size_t vram_total = static_cast<size_t>(props.totalGlobalMem);
        const bool fits = budget.fits_in_vram(vram_total);
        if (!fits) {
            spdlog::warn("═══════════════════════════════════════════════════════════════════");
            spdlog::warn(" WEIGHT BUDGET WARNING: {:.2f} MiB needed, but GPU has {:.2f} MiB",
                         budget.total_rank_mib, vram_total / (1048576.0));
            spdlog::warn(" Consider: --gpu-memory-utilization >0.9 will NOT help (only caps KV cache)");
            spdlog::warn(" Solutions: reduce tp_size, use IQ4_XS quantization, or model with fewer params");
            spdlog::warn("═══════════════════════════════════════════════════════════════════");
        } else {
            spdlog::info("Weight budget check: {:.2f} MiB / {:.2f} MiB GPU — OK",
                         budget.total_rank_mib, vram_total / (1048576.0));
        }
    }

    for (size_t ti = 0; ti < tensors.size(); ++ti) {
        const auto& tinfo = tensors[ti];
        std::string vmc_name = gguf_tensor_name_to_vmc(tinfo.name, num_layers);

        // ── TensorSchema 统一决策：分片模式 + MTP 跳过 ──
        const ArchSchema* arch_schema = get_arch_schema(config.arch);
        const TensorSchemaEntry* schema_entry = nullptr;
        if (arch_schema) {
            schema_entry = arch_schema->find_by_gguf(tinfo.name, num_layers);
        }
        if (schema_entry) {
            // MTP 张量：仅在 spec_method=mtp 时加载
            if (schema_entry->is_mtp && config.spec_method != "mtp") {
                spdlog::debug("SKIP MTP weight (spec_method={}): {} (vmc={})",
                              config.spec_method, tinfo.name, schema_entry->vmc_name);
                continue;
            }
        } else if (!arch_schema) {
            // 架构未注册：仅向后兼容的 MTP 字符串匹配 fallback
            if (config.spec_method != "mtp" && vmc_name.find("nextn.") != std::string::npos) {
                spdlog::debug("SKIP MTP weight (no schema, spec_method={}): {}", config.spec_method, vmc_name);
                continue;
            }
        } else {
            // 架构已注册但该张量无匹配：可能来自不同版本的转换脚本
            spdlog::debug("No TensorSchema entry for GGUF tensor '{}' (vmc='{}') — using default shard",
                         tinfo.name, vmc_name);
        }

        // ── 统一 ShardContext：所有物理路径消费此上下文中的维度信息 ──
        ShardContext ctx;
        if (schema_entry) {
            ctx = ShardContext::compute(*schema_entry, config, tp_rank, tp_size, tinfo);
        } else {
            ctx.shard_mode = TpShardMode::None;
            ctx.vmc_name = vmc_name;
            ctx.n_dims = tinfo.n_dims;
            for (int i = 0; i < 4; ++i) {
                ctx.full_ne[i] = tinfo.ne[i];
                ctx.sharded_ne[i] = tinfo.ne[i];
            }
        }

        // GDN 已知张量保留一下日志便于调试
        if (ctx.shard_mode == TpShardMode::GdnQkv || ctx.shard_mode == TpShardMode::GdnConv ||
            ctx.shard_mode == TpShardMode::GdnBaSplit || ctx.shard_mode == TpShardMode::GdnHeadSplit0) {
            if (tinfo.ne[0] == 32 && tinfo.ne[1] == 32 && tinfo.name.find("blk.0.") == 0) {
                spdlog::info("TensorSchema: {} → shard={} ne=[{},{},{}]",
                             vmc_name,
                             (ctx.shard_mode == TpShardMode::GdnQkv ? "GdnQkv" :
                              ctx.shard_mode == TpShardMode::GdnConv ? "GdnConv" :
                              ctx.shard_mode == TpShardMode::GdnBaSplit ? "GdnBaSplit" : "GdnHeadSplit0"),
                             tinfo.ne[0], tinfo.ne[1], tinfo.ne[2]);
            }
        }

        // ── IQ4_XS / IQ4_NL：ggml CUDA buffer 分配（对齐 llama.cpp load）──
        if (tinfo.type == GGML_TYPE_IQ4_XS || tinfo.type == GGML_TYPE_IQ4_NL) {
#if !(defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE)
            spdlog::error("IQ4 weights require VM_C_OFFICIAL_GGML=ON");
            continue;
#else
            const ggml_type iq_type = static_cast<ggml_type>(tinfo.type);
            const void* raw_data = reader.tensor_data(ti);
            if (!raw_data) { spdlog::warn("Null data for {}", vmc_name); continue; }

            // 使用 ShardContext 的统一计算：block_count / alloc_bytes / sharded_ne 已预计算
            int64_t ggml_ne0 = ctx.sharded_ne[0];
            int64_t ggml_ne1 = ctx.sharded_ne[1];
            int64_t ggml_ne2 = ctx.sharded_ne[2];
            const void* host_ptr = raw_data;
            size_t host_bytes = ctx.alloc_bytes;
            void* stage_buf = nullptr;

            if (ctx.is_3d_expert && tp_size > 1) {
                if (ctx.is_expert_down) {
                    // MoE down: GGUF ne=[n_ff, n_embd, n_expert]
                    const int64_t n_ff = ctx.full_ne[0], n_embd = ctx.full_ne[1], n_expert = ctx.full_ne[2];
                    CUDA_CHECK(cudaHostAlloc(&stage_buf, host_bytes, cudaHostAllocDefault));
                    shard_moe_down_experts_qblocks(
                        stage_buf, raw_data, ctx.blk_bytes,
                        n_ff, n_embd, n_expert, tp_rank, tp_size);
                    host_ptr = stage_buf;
                } else {
                    // MoE gate/up: GGUF ne=[n_embd, n_ff, n_expert]
                    const int64_t n_embd = ctx.full_ne[0], n_ff = ctx.full_ne[1], n_expert = ctx.full_ne[2];
                    CUDA_CHECK(cudaHostAlloc(&stage_buf, host_bytes, cudaHostAllocDefault));
                    shard_moe_iq4_gate_up_experts(
                        static_cast<BlockIQ4XS*>(stage_buf),
                        static_cast<const BlockIQ4XS*>(raw_data),
                        n_embd, n_ff, n_expert, tp_rank, tp_size);
                    host_ptr = stage_buf;
                }
                ggml_ne0 = ctx.sharded_ne[0];
                ggml_ne1 = ctx.sharded_ne[1];
                ggml_ne2 = ctx.sharded_ne[2];
            } else if (ctx.shard_mode != TpShardMode::None && tp_size > 1) {
                CUDA_CHECK(cudaHostAlloc(&stage_buf, host_bytes, cudaHostAllocDefault));

                if (ctx.shard_mode == TpShardMode::GdnQkv || ctx.shard_mode == TpShardMode::GdnConv) {
                    shard_gdn_iq4_qkv_rows(
                        stage_buf, raw_data, tp_rank, tp_size,
                        ctx.full_ne[0], ctx.gdn_key_dim, ctx.gdn_value_dim);
                } else if (ctx.shard_mode == TpShardMode::GdnHeadSplit0 || ctx.shard_mode == TpShardMode::ColSplit0) {
                    // TP 模式下 token_embd / output 复制：加载完整 vocab embedding
                    if ((ctx.vmc_name.find("token_embd") != std::string::npos ||
                         ctx.vmc_name.find("output.weight") != std::string::npos) && tinfo.n_dims >= 2) {
                        const int64_t n_embd = ctx.full_ne[0];
                        const int64_t n_vocab_shard = ctx.full_ne[1];
                        const int64_t n_vocab_full = n_vocab_shard * tp_size;
                        const int64_t total_blks = (n_vocab_full * n_embd + ctx.blk_elems - 1) / ctx.blk_elems;
                        host_bytes = static_cast<size_t>(total_blks) * ctx.blk_bytes;
                        ctx.alloc_bytes = host_bytes;
                        ctx.block_count = total_blks;
                        cudaFreeHost(stage_buf);
                        stage_buf = nullptr;
                        CUDA_CHECK(cudaHostAlloc(&stage_buf, host_bytes, cudaHostAllocDefault));
                        replicate_token_embd_to_full_vocab(
                            stage_buf, raw_data, n_embd, n_vocab_shard, tp_rank, tp_size,
                            ctx.blk_elems, ctx.blk_bytes, true);
                        ggml_ne0 = n_embd;
                        ggml_ne1 = n_vocab_full;
                        spdlog::info("IQ4 REPLICATE token_embd: ggml_ne=[{},{}] block_count={} alloc_bytes={} (tp_rank={})",
                                      ggml_ne0, ggml_ne1, ctx.block_count, ctx.alloc_bytes, tp_rank);
                    } else if (tinfo.n_dims >= 2) {
                        shard_gdn_iq4_head_split0(stage_buf, raw_data, tp_rank, tp_size,
                                                  ctx.full_ne[0], ctx.full_ne[1]);
                    } else {
                        // 1D level: block-aligned element copy
                        const int64_t split_sz = ctx.shard_cols;
                        const int64_t src_off = ctx.shard_col_offset;
                        std::memcpy(stage_buf,
                                    static_cast<const uint8_t*>(raw_data)
                                        + static_cast<size_t>(src_off / ctx.blk_elems) * ctx.blk_bytes,
                                    host_bytes);
                    }
                } else if (ctx.shard_mode == TpShardMode::RowSplit1) {
                    shard_iq4_row_split1(stage_buf, raw_data, tp_rank, tp_size,
                                         ctx.full_ne[0], ctx.full_ne[1], ctx.full_ne[2]);
                }
                host_ptr = stage_buf;
                // 仅当未触发 embedding 复制时才使用 sharded 维度
                if (ctx.vmc_name.find("token_embd") == std::string::npos &&
                    ctx.vmc_name.find("output.weight") == std::string::npos) {
                    ggml_ne0 = ctx.sharded_ne[0];
                    ggml_ne1 = ctx.sharded_ne[1];
                    ggml_ne2 = ctx.sharded_ne[2];
                }
            }

            GpuTensor gpu_iq4 = make_ggml_quant_from_host(
                iq_type, gpu_device, ggml_ne0, ggml_ne1, ggml_ne2,
                host_ptr, host_bytes, ctx.block_count, stream);
            if (stage_buf) {
                CUDA_CHECK(cudaFreeHost(stage_buf));
            }

            weights.add_weight(vmc_name, std::move(gpu_iq4));
            n_iq4++;
            continue;
#endif
        }

        // MoE 3D Q5_K down_exps → ggml CUDA buffer（使用 ShardContext 维度）
        if (ctx.is_3d_expert && ctx.is_expert_down && tinfo.type == GGML_TYPE_Q5_K) {
#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
            const void* raw_data = reader.tensor_data(ti);
            if (!raw_data) { spdlog::warn("Null data for {}", vmc_name); continue; }

            const int64_t n_ff = ctx.full_ne[0], n_embd = ctx.full_ne[1], n_expert = ctx.full_ne[2];
            const int64_t ff_r = ctx.sharded_ne[0];
            void* stage_buf = nullptr;
            const void* host_ptr = raw_data;

            if (tp_size > 1) {
                CUDA_CHECK(cudaHostAlloc(&stage_buf, ctx.alloc_bytes, cudaHostAllocDefault));
                shard_moe_down_experts_qblocks(
                    stage_buf, raw_data, ctx.blk_bytes,
                    n_ff, n_embd, n_expert, tp_rank, tp_size);
                host_ptr = stage_buf;
            }

            GpuTensor gpu_q5 = make_ggml_q5_k_from_host(
                gpu_device, ff_r, n_embd, n_expert,
                host_ptr, ctx.alloc_bytes, ctx.block_count, stream);
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

        // ── 其他量化类型：ggml 原生 buffer（使用 ShardContext 维度）──
#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
        if (ggml_type_is_quantized(tinfo.type)) {
            if (vmc_name.find("token_embd") != std::string::npos && tinfo.ne[0] == 2048) {
                spdlog::info("[GGUF-LOAD] token_embd: type={} shard_mode={} tp_size={} full_ne=[{},{}] blk_elems={} blk_bytes={} sharded_ne=[{},{}]",
                    (int)tinfo.type, (int)ctx.shard_mode, tp_size,
                    ctx.full_ne[0], ctx.full_ne[1], ctx.blk_elems, ctx.blk_bytes,
                    ctx.sharded_ne[0], ctx.sharded_ne[1]);
            }
            const void* raw_data = reader.tensor_data(ti);
            if (!raw_data) { spdlog::warn("Null data for {}", vmc_name); continue; }

            void* stage_buf = nullptr;
            const void* host_ptr = raw_data;
            int64_t ggml_ne0 = ctx.sharded_ne[0];
            int64_t ggml_ne1 = ctx.sharded_ne[1];
            int64_t ggml_ne2 = ctx.sharded_ne[2];

            if (ctx.shard_mode != TpShardMode::None && tp_size > 1 && tinfo.n_dims >= 2) {
                size_t host_bytes = ctx.alloc_bytes;
                CUDA_CHECK(cudaHostAlloc(&stage_buf, host_bytes, cudaHostAllocDefault));

                if (ctx.shard_mode == TpShardMode::GdnBaSplit) {
                    shard_ggml_quant_ba_split0(
                        stage_buf, raw_data, ctx.blk_bytes, tp_rank, tp_size,
                        ctx.full_ne[0], ctx.full_ne[1], ctx.full_ne[2], ctx.blk_elems);
                } else if (ctx.shard_mode == TpShardMode::GdnHeadSplit0 || ctx.shard_mode == TpShardMode::ColSplit0) {
                    // DEBUG: 验证 raw_data + 最大访问偏移在 mmap 范围内
                    const uint8_t* mmap_start = static_cast<const uint8_t*>(reader.data());
                    const size_t mmap_len = reader.data_size();
                    const uint8_t* raw_bytes = static_cast<const uint8_t*>(raw_data);
                    const int64_t total_rows_chk = ctx.full_ne[1] * ctx.full_ne[2];
                    const int64_t blks_per_row_chk = (ctx.full_ne[0] + ctx.blk_elems - 1) / ctx.blk_elems;
                    const int64_t max_off = (total_rows_chk - 1) * blks_per_row_chk * static_cast<int64_t>(ctx.blk_bytes)
                                          + blks_per_row_chk * static_cast<int64_t>(ctx.blk_bytes);
                    if (raw_bytes < mmap_start || (raw_bytes - mmap_start) + max_off > mmap_len) {
                        spdlog::error("CRASH PREDICTED: {} raw_data={} offset={} end={} mmap_start={} mmap_len={} max_access={}",
                            vmc_name, (void*)raw_bytes, (raw_bytes - mmap_start),
                            (raw_bytes - mmap_start) + max_off,
                            (void*)mmap_start, mmap_len, max_off);
                    }
                    // TP 模式下 token_embd / output 复制：加载完整 vocab embedding
                    // GGUF 是 pre-sharded，需要复制成全尺寸供所有 rank 使用
                    if (ctx.vmc_name.find("token_embd") != std::string::npos ||
                        ctx.vmc_name.find("output.weight") != std::string::npos) {
                        const int64_t n_embd = ctx.full_ne[0];
                        const int64_t n_vocab_shard = ctx.full_ne[1];
                        // 计算全 vocab 尺寸
                        const int64_t n_vocab_full = n_vocab_shard * tp_size;
                        const int64_t total_blks = (n_vocab_full * n_embd + ctx.blk_elems - 1) / ctx.blk_elems;
                        host_bytes = static_cast<size_t>(total_blks) * ctx.blk_bytes;
                        ctx.alloc_bytes = host_bytes;
                        ctx.block_count = total_blks;
                        cudaFreeHost(stage_buf);
                        stage_buf = nullptr;
                        CUDA_CHECK(cudaHostAlloc(&stage_buf, host_bytes, cudaHostAllocDefault));
                        replicate_token_embd_to_full_vocab(
                            stage_buf, raw_data, n_embd, n_vocab_shard, tp_rank, tp_size,
                            ctx.blk_elems, ctx.blk_bytes, true);
                        ggml_ne0 = n_embd;
                        ggml_ne1 = n_vocab_full;
                        spdlog::info("REPLICATE token_embd: ggml_ne=[{},{}] block_count={} alloc_bytes={} (tp_rank={})",
                                      ggml_ne0, ggml_ne1, ctx.block_count, ctx.alloc_bytes, tp_rank);
                    } else {
                        shard_ggml_quant_col_split0(
                            stage_buf, raw_data, ctx.blk_bytes, tp_rank, tp_size,
                            ctx.full_ne[0], ctx.full_ne[1], ctx.full_ne[2], ctx.blk_elems);
                    }
                } else if (ctx.shard_mode == TpShardMode::GdnQkv || ctx.shard_mode == TpShardMode::GdnConv) {
                    shard_gdn_ggml_quant_qkv_rows(
                        stage_buf, raw_data, ctx.blk_bytes, tp_rank, tp_size,
                        ctx.full_ne[0], ctx.gdn_key_dim, ctx.gdn_value_dim, ctx.blk_elems);
                } else if (ctx.shard_mode == TpShardMode::RowSplit1) {
                    shard_ggml_quant_row_split1(
                        stage_buf, raw_data, ctx.blk_bytes, tp_rank, tp_size,
                        ctx.full_ne[0], ctx.full_ne[1], ctx.full_ne[2], ctx.blk_elems);
                }
                host_ptr = stage_buf;
            }

            GpuTensor gpu_quant = make_ggml_quant_from_host(
                static_cast<ggml_type>(tinfo.type),
                gpu_device, ggml_ne0, ggml_ne1, ggml_ne2,
                host_ptr, ctx.alloc_bytes, ctx.block_count, stream);
            if (stage_buf) {
                CUDA_CHECK(cudaFreeHost(stage_buf));
            }

            weights.add_weight(vmc_name, std::move(gpu_quant));
            n_ggml_quant++;
            continue;
        }
#endif

        // ── 非 IQ4_XS / 非量化张量（使用 ShardContext 统一 shape）──

        const void* src_data = reader.tensor_data(ti);
        if (!src_data) { spdlog::warn("Null data for {}", vmc_name); continue; }

        if (!needs_gpu_convert(tinfo.type)) {
            // ── 无需转换：按 GGUF 原生 dtype 直接异步上传 ──
            const DataType store_dtype = gguf_native_storage_dtype(tinfo.type);
            const size_t elem_sz = dtype_size(store_dtype);

            // TP 模式下 token_embd/output 复制：预先计算 full vocab shape
            bool is_embd_replicate = false;
            int64_t rep_n_embd = 0, rep_n_vocab_full = 0;
            if (tp_size > 1 && tinfo.n_dims >= 2 &&
                (ctx.vmc_name.find("token_embd") != std::string::npos ||
                 ctx.vmc_name.find("output.weight") != std::string::npos)) {
                is_embd_replicate = true;
                rep_n_embd = ctx.full_ne[0];
                rep_n_vocab_full = ctx.full_ne[1] * tp_size;
            }

            const Shape& alloc_shape = is_embd_replicate ? Shape({rep_n_embd, rep_n_vocab_full}) : ctx.target_shape;
            GpuTensor gpu_tensor(alloc_shape, store_dtype, gpu_device);
            const size_t nbytes = static_cast<size_t>(alloc_shape.numel()) * elem_sz;

            if (ctx.shard_mode != TpShardMode::None && tp_size > 1) {
                void* stage = nullptr;
                CUDA_CHECK(cudaHostAlloc(&stage, nbytes, cudaHostAllocDefault));
                const int elem_sz_i = static_cast<int>(elem_sz);
                const uint8_t* src = (const uint8_t*)src_data;
                uint8_t* dst = (uint8_t*)stage;

                if (ctx.shard_mode == TpShardMode::GdnQkv) {
                    int row_bytes = static_cast<int>(ctx.full_ne[0] * elem_sz_i);
                    shard_gdn_qkv_rows(dst, src, tp_rank, tp_size,
                                       ctx.gdn_key_dim, ctx.gdn_value_dim, row_bytes);
                } else if (ctx.shard_mode == TpShardMode::GdnConv) {
                    int row_bytes = static_cast<int>(ctx.full_ne[0] * elem_sz_i);
                    shard_gdn_conv_rows(dst, src, tp_rank, tp_size,
                                        ctx.gdn_key_dim, ctx.gdn_value_dim, row_bytes);
                } else if (ctx.shard_mode == TpShardMode::GdnBaSplit) {
                    int row_width_bytes = static_cast<int>(ctx.full_ne[0] * elem_sz_i);
                    int64_t n_v_heads = ctx.full_ne[1] / 2;
                    int64_t n_v_heads_local = n_v_heads / tp_size;
                    int64_t beta_src_off = tp_rank * n_v_heads_local;
                    int64_t alpha_src_off = n_v_heads + tp_rank * n_v_heads_local;
                    memcpy(dst, src + beta_src_off * row_width_bytes,
                           static_cast<size_t>(n_v_heads_local) * row_width_bytes);
                    memcpy(dst + static_cast<size_t>(n_v_heads_local) * row_width_bytes,
                           src + alpha_src_off * row_width_bytes,
                           static_cast<size_t>(n_v_heads_local) * row_width_bytes);
                } else if (ctx.shard_mode == TpShardMode::GdnHeadSplit0 || ctx.shard_mode == TpShardMode::ColSplit0) {
                    // TP 模式下 token_embd / output 复制：加载完整 vocab embedding
                    if (is_embd_replicate) {
                        const int64_t n_vocab_shard = rep_n_vocab_full / tp_size;
                        const size_t row_bytes = static_cast<size_t>(rep_n_embd) * elem_sz;
                        const size_t shard_bytes = static_cast<size_t>(n_vocab_shard) * row_bytes;
                        std::memset(stage, 0, nbytes);
                        std::memcpy(static_cast<uint8_t*>(stage) +
                                    static_cast<size_t>(tp_rank) * shard_bytes,
                                    src, shard_bytes);
                        spdlog::debug("Replicate token_embd/output to full vocab: shard={} -> full={} (tp_rank={}/{})",
                                      n_vocab_shard, rep_n_vocab_full, tp_rank, tp_size);
                    } else if (tinfo.n_dims <= 1) {
                        const int64_t split_sz_elems = ctx.full_ne[0] / tp_size;
                        memcpy(dst, src + tp_rank * split_sz_elems * elem_sz_i, split_sz_elems * elem_sz_i);
                    } else {
                        int row_width_bytes = static_cast<int>(ctx.full_ne[0] * elem_sz_i);
                        int64_t rows_per_shard = ctx.full_ne[1] / tp_size;
                        memcpy(dst, src + tp_rank * rows_per_shard * row_width_bytes, rows_per_shard * row_width_bytes);
                    }
                } else if (ctx.shard_mode == TpShardMode::RowSplit1) {
                    if (tinfo.n_dims >= 2) {
                        int64_t cols_per_shard = ctx.full_ne[0] / tp_size;
                        int64_t src_col_off = tp_rank * cols_per_shard;
                        int dst_stride = static_cast<int>(cols_per_shard * elem_sz_i);
                        for (int64_t r = 0; r < ctx.full_ne[1]; ++r) {
                            memcpy(dst + r * dst_stride,
                                   src + (r * ctx.full_ne[0] + src_col_off) * elem_sz_i,
                                   dst_stride);
                        }
                    } else {
                        memcpy(dst, src, nbytes);
                    }
                }
                CUDA_CHECK(cudaMemcpyAsync(gpu_tensor.data_ptr(), stage, nbytes, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaFreeHost(stage));
            } else {
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
            // ── 需要 GPU 格式转换 ──
            GpuTensor gpu_tensor(ctx.target_shape, gpu().compute_dtype(), gpu_device);
            const size_t fp16_bytes = static_cast<size_t>(ctx.target_shape.numel()) * dtype_sz;

            // 计算原始全量张量在 GGUF 文件中的字节
            int64_t full_total_elems = 1;
            for (int i = 0; i < tinfo.n_dims; ++i) full_total_elems *= ctx.full_ne[i];

            size_t full_src_bytes;
            if (tinfo.type == GGML_TYPE_F32) {
                full_src_bytes = full_total_elems * sizeof(float);
            } else if (tinfo.type == GGML_TYPE_BF16) {
                full_src_bytes = full_total_elems * sizeof(uint16_t);
            } else {
                int64_t blk_elems = ggml_blk_elems(tinfo.type);
                int64_t n_blk = (full_total_elems + blk_elems - 1) / blk_elems;
                full_src_bytes = n_blk * ggml_block_size_bytes(tinfo.type);
            }

            const int64_t total_elems = ctx.target_shape.numel();

            if (ctx.shard_mode == TpShardMode::ColSplit0 && tp_size > 1) {
                size_t shard_src_bytes = full_src_bytes / tp_size;
                const uint8_t* offset_src = (const uint8_t*)src_data + tp_rank * shard_src_bytes;
                void* tmp_gpu = nullptr;
                CUDA_CHECK(cudaMalloc(&tmp_gpu, shard_src_bytes));
                tmp_gpu_allocs.push_back(tmp_gpu);
                CUDA_CHECK(cudaMemcpyAsync(tmp_gpu, offset_src, shard_src_bytes, cudaMemcpyHostToDevice, stream));
                launch_gpu_convert(gpu_tensor.data_ptr(), tmp_gpu, static_cast<int>(tinfo.type), total_elems,
                                   gpu().compute_dtype() == DataType::BFLOAT16, stream);
            } else if (ctx.shard_mode == TpShardMode::RowSplit1 && tp_size > 1) {
                void* tmp_gpu_full = nullptr;
                CUDA_CHECK(cudaMalloc(&tmp_gpu_full, full_src_bytes));
                tmp_gpu_allocs.push_back(tmp_gpu_full);
                CUDA_CHECK(cudaMemcpyAsync(tmp_gpu_full, src_data, full_src_bytes, cudaMemcpyHostToDevice, stream));

                void* full_fp16_gpu = nullptr;
                CUDA_CHECK(cudaMalloc(&full_fp16_gpu, full_total_elems * dtype_sz));
                tmp_gpu_allocs.push_back(full_fp16_gpu);
                launch_gpu_convert(full_fp16_gpu, tmp_gpu_full, static_cast<int>(tinfo.type), full_total_elems,
                                   gpu().compute_dtype() == DataType::BFLOAT16, stream);

                const int64_t full_rows = ctx.full_ne[1];
                const int64_t full_cols = ctx.full_ne[0];
                const int64_t cols_per_shard = full_cols / tp_size;
                const int64_t src_col_off = tp_rank * cols_per_shard;
                const size_t row_stride_bytes = static_cast<size_t>(full_cols) * dtype_sz;
                const size_t shard_row_bytes = static_cast<size_t>(cols_per_shard) * dtype_sz;

                for (int64_t r = 0; r < full_rows; ++r) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        static_cast<uint8_t*>(gpu_tensor.data_ptr()) + r * shard_row_bytes,
                        static_cast<uint8_t*>(full_fp16_gpu) + r * row_stride_bytes + src_col_off * dtype_sz,
                        shard_row_bytes, cudaMemcpyDeviceToDevice, stream));
                }
            } else {
                void* tmp_gpu = nullptr;
                CUDA_CHECK(cudaMalloc(&tmp_gpu, full_src_bytes));
                tmp_gpu_allocs.push_back(tmp_gpu);
                CUDA_CHECK(cudaMemcpyAsync(tmp_gpu, src_data, full_src_bytes, cudaMemcpyHostToDevice, stream));
                launch_gpu_convert(gpu_tensor.data_ptr(), tmp_gpu, static_cast<int>(tinfo.type), total_elems,
                                   gpu().compute_dtype() == DataType::BFLOAT16, stream);
            }

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

    // 打印 lm_head.weight 形状（诊断 TP 分片状态）
    for (const auto& [name, tensor] : weights.gpu_weights) {
        if (name == "lm_head.weight" || name == "model.lm_head.weight") {
            auto sh = tensor.shape();
            spdlog::info("[GGUF-DIAG] {}: shape=[{}x{}] dtype={} tp_rank={} tp_size={}",
                         name, sh[0], sh[1], static_cast<int>(tensor.dtype()),
                         tp_rank, tp_size);
            break;
        }
    }

    // ── MTP 专有权重反量化质量检查（仅首次加载，禁止刷屏） ──
    // MTP 小权重（eh_proj=32M, enorm/hnorm=2K）在 GGUF 中可能是 FP16 或量化格式。
    // 记录其 GPU 上的前 4 个值 + L2 范数，用于诊断反量化是否正确。
    // 如果 dequant 损坏，前 4 个值会接近 0 或极端值（如 >1e5 或 NaN）。
    for (const auto& [n, t] : weights.gpu_weights) {
        bool is_nextn = (n.find("nextn.eh_proj") != std::string::npos ||
                         n.find("nextn.en_norm") != std::string::npos ||
                         n.find("nextn.h_norm") != std::string::npos);
        if (!is_nextn) continue;
        auto sh = t.shape();
        int64_t n_elems = sh.numel();
        int64_t probe = std::min<int64_t>(4, n_elems);
        std::vector<float> host(static_cast<size_t>(probe), 0.0f);
        // GPU → host: 用 cudaMemcpy 读取回 CPU 验证
        CUDA_CHECK(cudaMemcpy(host.data(), t.data_ptr(),
                              static_cast<size_t>(probe) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        // 读取 1024 个元素算 L2 范数（如果 tensor 够大；norm 权重直接用 n_elems）
        int64_t l2_n = std::min<int64_t>(n_elems, 1024);
        std::vector<float> host_l2(static_cast<size_t>(l2_n));
        CUDA_CHECK(cudaMemcpy(host_l2.data(), t.data_ptr(),
                              static_cast<size_t>(l2_n) * sizeof(float),
                              cudaMemcpyDeviceToHost));
        double l2 = 0.0;
        for (int64_t i = 0; i < l2_n; ++i) l2 += static_cast<double>(host_l2[i]) * host_l2[i];
        l2 = std::sqrt(l2);
        spdlog::info("[GGUF-MTP-WEIGHT] {} shape=[{}x{}] first4=[{:.6f},{:.6f},{:.6f},{:.6f}] L2(first{})={:.3f} dtype={} tp_rank={}",
                     n, sh[0], sh[1],
                     host[0], host[1], host[2], host[3],
                     l2_n, l2, static_cast<int>(t.dtype()), tp_rank);
    }

    spdlog::info(
        "GGUF: loaded {} tensors ({} native-dense + {} ggml-quant [iq4={} other={}] + {} converted)",
        weights.gpu_weights.size(), n_direct, n_iq4 + n_ggml_quant, n_iq4, n_ggml_quant, n_converted);
}

} // namespace vm_c
