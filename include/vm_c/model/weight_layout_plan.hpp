#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "vm_c/core/config.hpp"
#include "vm_c/core/tensor.hpp"

namespace vm_c {

enum class ShardStrategy {
    REPLICATED,
    COLUMN_PARALLEL,
    ROW_PARALLEL,
    QKV_PARALLEL,
    FUSED_COLUMN_PARALLEL,
    FUSED_ROW_PARALLEL,
};

struct WeightShardInfo {
    std::string name;
    ShardStrategy strategy = ShardStrategy::REPLICATED;
    int shard_dim = 0;
    int num_experts = 0;
    int num_kv_head_replicas = 1;
    bool offload_to_cpu = false;
    bool skip_for_this_rank = false;
    std::vector<int64_t> output_sizes;
};

struct CpuTensorView {
    const void* cpu_ptr = nullptr;
    Shape shape;
    DataType dtype = DataType::FLOAT32;
    int64_t stride_bytes = 0;

    CpuTensorView() = default;
    CpuTensorView(const void* ptr, Shape s, DataType dt, int64_t stride)
        : cpu_ptr(ptr), shape(std::move(s)), dtype(dt), stride_bytes(stride) {}

    CpuTensorView(void* ptr, Shape s, DataType dt, int64_t stride)
        : cpu_ptr(ptr), shape(std::move(s)), dtype(dt), stride_bytes(stride) {}

    size_t nbytes() const { return shape.numel() * dtype_size(dtype); }
};

class WeightLayoutPlan {
public:
    WeightLayoutPlan(const ModelConfig& model_config, int tp_rank, int tp_size,
                     const CacheConfig& cache_config);

    void build();

    const WeightShardInfo* get_shard_info(const std::string& name) const;
    bool should_skip_weight(const std::string& name) const;
    bool should_offload_weight(const std::string& name) const;

    CpuTensorView narrow_cpu_tensor(const CpuTensor& full,
                                    const WeightShardInfo& info) const;

    std::pair<CpuTensorView, std::unique_ptr<CpuTensor>> narrow_qkv_tensor(const CpuTensor& full,
                                                                            const WeightShardInfo& info) const;

    std::pair<CpuTensorView, std::unique_ptr<CpuTensor>> narrow_fused_expert_tensor(const CpuTensor& full,
                                         const WeightShardInfo& info) const;

    WeightShardInfo derive_sub_tensor_shard_info(const std::string& name,
                                                  const Shape& tensor_shape) const;

    const std::unordered_map<std::string, WeightShardInfo>& shard_map() const {
        return shard_map_;
    }

    int tp_rank() const { return tp_rank_; }
    int tp_size() const { return tp_size_; }

private:
    void register_weight(const std::string& name, ShardStrategy strategy,
                         int shard_dim);
    void register_replicated(const std::string& name);
    void register_column_parallel(const std::string& name, int64_t out_features);
    void register_kv_column_parallel(const std::string& name, int64_t kv_out,
                                     int num_kv_heads);
    void register_row_parallel(const std::string& name, int64_t in_features);
    void register_qkv_parallel(const std::string& name,
                               int64_t q_size, int64_t kv_size,
                               int num_kv_heads);
    void register_expert_weights(const std::string& layer_prefix, int num_experts);
    void register_fused_expert_weights(const std::string& layer_prefix);

    bool match_offload_pattern(const std::string& name) const;
    std::string remap_prefix(const std::string& name) const;

    ModelConfig model_config_;
    CacheConfig cache_config_;
    int tp_rank_;
    int tp_size_;
    std::unordered_map<std::string, WeightShardInfo> shard_map_;
    std::unordered_set<int> local_expert_ids_;
};

}
