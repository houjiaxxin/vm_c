#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <optional>
#include <list>
#include <cuda_runtime.h>

#include "vm_c/core/tensor.hpp"
#include "vm_c/core/config.hpp"
#include "vm_c/cuda/gpu_arch.hpp"

namespace vm_c {

struct BlockHashKey {
    uint64_t hash;
    int group_id;

    bool operator==(const BlockHashKey& other) const {
        return hash == other.hash && group_id == other.group_id;
    }
};

struct BlockHashKeyHash {
    size_t operator()(const BlockHashKey& k) const {
        size_t h = std::hash<uint64_t>()(k.hash);
        h ^= std::hash<int>()(k.group_id) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct KVCacheBlock {
    int64_t block_id = -1;
    int ref_cnt = 0;
    BlockHashKey block_hash{0, -1};
    bool has_hash = false;
    bool is_null = false;

    KVCacheBlock* prev_free = nullptr;
    KVCacheBlock* next_free = nullptr;

    void set_hash(uint64_t hash, int group_id) {
        block_hash = {hash, group_id};
        has_hash = true;
    }

    void reset_hash() {
        block_hash = {0, -1};
        has_hash = false;
    }
};

class FreeKVCacheBlockQueue {
public:
    explicit FreeKVCacheBlockQueue(std::vector<KVCacheBlock>& blocks);

    KVCacheBlock* popleft();
    std::vector<KVCacheBlock*> popleft_n(int64_t n);
    void remove(KVCacheBlock* block);
    void append(KVCacheBlock* block);
    void append_n(const std::vector<KVCacheBlock*>& blocks);

    int64_t num_free_blocks() const { return num_free_; }

private:
    KVCacheBlock fake_head_;
    KVCacheBlock fake_tail_;
    int64_t num_free_ = 0;
};

class BlockHashToBlockMap {
public:
    KVCacheBlock* get(uint64_t hash, int group_id);
    void insert(uint64_t hash, int group_id, KVCacheBlock* block);
    KVCacheBlock* pop(uint64_t hash, int group_id, int64_t block_id);
    void clear();
    size_t size() const { return cache_.size(); }

private:
    struct MapEntry {
        KVCacheBlock* single = nullptr;
        std::unordered_map<int64_t, KVCacheBlock*> multi;
        bool is_multi = false;
    };

    std::unordered_map<BlockHashKey, MapEntry, BlockHashKeyHash> cache_;
};

struct KVCacheSlot {
    int64_t block_id;
    int slot_offset;
};

struct KVCacheGroupSpec {
    int group_id;
    std::vector<int> layer_indices;
    size_t bytes_per_block;
    bool is_tq;
};

struct TurboQuantLayout {
    int head_dim = 0;
    int key_quant_bits = 4;
    int value_quant_bits = 4;
    bool norm_correction = true;
    bool key_fp8 = false;
    bool fp8_e4b15 = false;

    int mse_bytes = 0;
    int key_packed_size = 0;
    int val_data_bytes = 0;
    int value_packed_size = 0;
    int slot_size = 0;
    int slot_size_aligned = 0;
    int n_centroids = 0;

    static TurboQuantLayout create(int head_dim, const std::string& method, int gpu_device = 0) {
        TurboQuantLayout l;
        l.head_dim = head_dim;
        if (method == "turboquant_4bit_nc") {
            l.key_quant_bits = 4;
            l.value_quant_bits = 4;
            l.norm_correction = true;
            l.key_fp8 = false;
        } else if (method == "turboquant_k8v4") {
            l.key_quant_bits = 8;
            l.value_quant_bits = 4;
            l.norm_correction = false;
            l.key_fp8 = true;
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu_device));
            int sm_ver = prop.major * 10 + prop.minor;
            l.fp8_e4b15 = (sm_ver < 89);
        } else if (method == "turboquant_k3v4_nc") {
            l.key_quant_bits = 3;
            l.value_quant_bits = 4;
            l.norm_correction = true;
            l.key_fp8 = false;
        } else if (method == "turboquant_3bit_nc") {
            l.key_quant_bits = 3;
            l.value_quant_bits = 3;
            l.norm_correction = true;
            l.key_fp8 = false;
        }
        l.n_centroids = 1 << l.key_quant_bits;
        if (l.key_fp8) {
            l.mse_bytes = head_dim;
            l.key_packed_size = head_dim;
        } else {
            l.mse_bytes = (head_dim * l.key_quant_bits + 7) / 8;
            l.key_packed_size = l.mse_bytes + 2;
        }
        l.val_data_bytes = (head_dim * l.value_quant_bits + 7) / 8;
        l.value_packed_size = l.val_data_bytes + 4;
        l.slot_size = l.key_packed_size + l.value_packed_size;
        l.slot_size_aligned = l.slot_size + (l.slot_size % 2);
        return l;
    }

    static std::vector<int> get_boundary_skip_layers(
            int num_layers, int n = 2,
            const std::vector<std::string>& layer_types = {}) {
        if (n <= 0 || num_layers <= 0) return {};
        bool is_hybrid = false;
        if (!layer_types.empty()) {
            for (auto& lt : layer_types) {
                if (lt != "full_attention" && lt != "attention") {
                    is_hybrid = true;
                    break;
                }
            }
        }
        if (is_hybrid) return {};
        n = std::min(n, num_layers / 2);
        std::vector<int> result;
        for (int i = 0; i < n; ++i) result.push_back(i);
        for (int i = num_layers - n; i < num_layers; ++i) result.push_back(i);
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }
};

class KVCacheManager {
public:
    KVCacheManager(const CacheConfig& config, const ModelConfig& model_config,
                   int num_layers, int num_kv_heads,
                   int head_dim, int gpu_device = 0, bool is_mla = false);
    ~KVCacheManager();

    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;

    void set_tq_boundary_layers(const std::vector<int>& skip_layers);

    void reserve_gpu_blocks(int64_t total_blocks);
    void allocate_gpu_blocks_for_group(int group_id, int64_t num_blocks);
    void allocate_cpu_blocks_for_group(int group_id, int64_t num_blocks);

    std::vector<KVCacheSlot> allocate_slots(
        const std::string& request_id,
        int num_tokens,
        const std::vector<uint64_t>& prefix_block_hashes = {},
        int num_new_computed_tokens = 0,
        const std::vector<int64_t>& new_computed_block_ids = {});
    void free_slots(const std::string& request_id);

    /// 将 request 的 block 表截断到 num_tokens 所需块数，释放多余 block（MTP verify partial reject 对齐 llama seq_rm）
    void trim_request_to_tokens(const std::string& request_id, int num_tokens);

    size_t kv_bytes_per_token(int layer) const;
    void read_kv_token(int layer, int64_t global_slot, void* dst, cudaStream_t stream) const;
    void write_kv_token(int layer, int64_t global_slot, const void* src, cudaStream_t stream) const;

    void swap_request_slots(const std::string& request_id);
    void swap_back_request_slots(const std::string& request_id);

    void cache_full_blocks(const std::string& request_id,
                           const std::vector<uint64_t>& block_hashes,
                           int num_cached_blocks,
                           int num_full_blocks);

    int find_longest_prefix_hit(const std::vector<uint64_t>& block_hashes,
                                int max_num_blocks,
                                std::vector<int64_t>& out_block_ids) const;

    int64_t num_free_gpu_blocks() const;
    int64_t num_free_cpu_blocks() const { return num_free_cpu_blocks_; }
    int64_t num_gpu_blocks() const;
    const std::vector<KVCacheBlock>& gpu_blocks() const { return gpu_blocks_; }
    const std::unordered_map<std::string, std::vector<KVCacheSlot>>& request_slots() const { return request_slots_; }

    void* key_cache_ptr(int layer);
    void* value_cache_ptr(int layer);

    void* cpu_key_cache_ptr(int layer);
    void* cpu_value_cache_ptr(int layer);

    int64_t block_size() const { return block_size_; }
    int num_layers() const { return num_layers_; }
    int num_kv_heads() const { return num_kv_heads_; }
    int head_dim() const { return head_dim_; }

    bool is_turboquant() const { return tq_layout_.slot_size > 0; }
    bool is_mla() const { return is_mla_; }
    bool is_layer_turboquant(int layer) const {
        return is_turboquant() && layer >= 0 && layer < num_layers_ && tq_layer_enabled_[layer];
    }
    const TurboQuantLayout& tq_layout() const { return tq_layout_; }

    // ── KV Cache 布局 ──
    // "nhd": [block_id][token][head][dim] (默认)
    // "hnd": [block_id][head][token][dim] (更优的 L2 cache 局部性)
    bool is_hnd_layout() const { return kv_cache_layout_ == "hnd"; }
    const std::string& kv_cache_layout() const { return kv_cache_layout_; }

    int num_kv_cache_groups() const { return static_cast<int>(groups_.size()); }
    const KVCacheGroupSpec& group(int group_id) const { return groups_[group_id]; }
    int group_id_for_layer(int layer) const;
    int64_t group_num_gpu_blocks(int group_id) const;
    int64_t group_num_free_gpu_blocks(int group_id) const;

    bool compress_kv_block(int64_t block_id);
    bool decompress_kv_block(int64_t block_id);

    void offload_to_cpu(int64_t block_id);
    void load_from_cpu(int64_t block_id);

    size_t gpu_kv_cache_bytes() const;
    size_t cpu_kv_cache_bytes() const;

    float usage() const;

    bool reset_prefix_cache();

private:
    void build_groups();
    size_t compute_bytes_per_block(bool tq) const;

    int64_t block_size_;
    int num_layers_;
    int num_kv_heads_;
    int head_dim_;
    int gpu_device_;
    ModelConfig model_config_;
    DataType kv_dtype_;
    std::string kv_cache_dtype_;
    std::string kv_cache_layout_ = "nhd"; // "nhd" 或 "hnd"
    TurboQuantLayout tq_layout_;
    std::vector<bool> tq_layer_enabled_;
    bool is_mla_ = false;
    bool enable_caching_;

    std::vector<KVCacheGroupSpec> groups_;
    std::unordered_map<int, int> layer_to_group_;

    std::vector<int64_t> group_num_gpu_blocks_;
    std::vector<int64_t> group_num_free_gpu_blocks_;
    int64_t num_cpu_blocks_ = 0;
    int64_t num_free_cpu_blocks_ = 0;

    std::vector<void*> gpu_key_caches_;
    std::vector<void*> gpu_value_caches_;
    std::vector<void*> cpu_key_caches_;
    std::vector<void*> cpu_value_caches_;

    std::vector<KVCacheBlock> gpu_blocks_;
    std::vector<KVCacheBlock> cpu_blocks_;
    KVCacheBlock* null_block_ = nullptr;

    std::unique_ptr<FreeKVCacheBlockQueue> free_queue_;
    BlockHashToBlockMap hash_cache_;

    std::unordered_map<std::string, std::vector<KVCacheSlot>> request_slots_;
    std::unordered_map<int64_t, int64_t> gpu_to_cpu_mapping_;

    std::unordered_map<int64_t, int64_t> cpu_to_gpu_mapping_;
};

}
