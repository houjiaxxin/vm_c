#include "vm_c/model/weight_layout_plan.hpp"
#include <spdlog/spdlog.h>
#include <cstring>
#include <memory>
#include <algorithm>

namespace vm_c {

WeightLayoutPlan::WeightLayoutPlan(const ModelConfig& model_config,
                                   int tp_rank, int tp_size,
                                   const CacheConfig& cache_config)
    : model_config_(model_config)
    , cache_config_(cache_config)
    , tp_rank_(tp_rank)
    , tp_size_(tp_size) {}

void WeightLayoutPlan::build() {
    const auto& mc = model_config_;
    const int hidden = mc.hidden_size;
    const int num_heads = mc.num_attention_heads;
    const int num_kv_heads = mc.num_key_value_heads;
    const int head_dim = mc.head_dim;
    const int intermediate = mc.intermediate_size;
    const int num_layers = mc.num_hidden_layers;
    const int num_experts = mc.num_experts;
    const int se_intermediate = mc.shared_expert_intermediate_size > 0
        ? mc.shared_expert_intermediate_size : intermediate;

    if (tp_size_ > 1 && num_experts > 0) {
        for (int i = 0; i < num_experts; ++i) {
            local_expert_ids_.insert(i);
        }
    }

    register_replicated("model.embed_tokens.weight");
    register_replicated("model.lm_head.weight");
    register_replicated("model.norm.weight");

    for (int i = 0; i < num_layers; ++i) {
        std::string lp = "model.layers." + std::to_string(i) + ".";

        register_replicated(lp + "input_layernorm.weight");
        register_replicated(lp + "post_attention_layernorm.weight");

        {
            int q_out = num_heads * head_dim;
            int kv_out = num_kv_heads * head_dim;
            register_column_parallel(lp + "self_attn.q_proj.weight", q_out);
            register_kv_column_parallel(lp + "self_attn.k_proj.weight", kv_out, num_kv_heads);
            register_kv_column_parallel(lp + "self_attn.v_proj.weight", kv_out, num_kv_heads);
            register_row_parallel(lp + "self_attn.o_proj.weight", num_heads * head_dim);
            register_replicated(lp + "self_attn.q_norm.weight");
            register_replicated(lp + "self_attn.k_norm.weight");
        }

        bool has_shared_expert = mc.n_shared_experts > 0;
        bool is_mlp_only = std::find(mc.mlp_only_layers.begin(), mc.mlp_only_layers.end(), i)
                           != mc.mlp_only_layers.end();
        int sparse_step = mc.decoder_sparse_step > 0 ? mc.decoder_sparse_step : 1;
        bool layer_uses_moe = num_experts > 0 && !is_mlp_only &&
                              ((i + 1) % sparse_step == 0);

        if (layer_uses_moe) {
            register_replicated(lp + "mlp.gate.weight");
            register_replicated(lp + "mlp.router.weight");

            if (has_shared_expert) {
                register_column_parallel(lp + "mlp.shared_expert.gate_proj.weight", se_intermediate);
                register_column_parallel(lp + "mlp.shared_expert.up_proj.weight", se_intermediate);
                register_row_parallel(lp + "mlp.shared_expert.down_proj.weight", se_intermediate);
                register_replicated(lp + "mlp.shared_expert_gate.weight");
            }

            register_expert_weights(lp, num_experts);
            register_fused_expert_weights(lp);
        } else {
            register_column_parallel(lp + "mlp.gate_proj.weight", intermediate);
            register_column_parallel(lp + "mlp.up_proj.weight", intermediate);
            register_row_parallel(lp + "mlp.down_proj.weight", intermediate);
            register_column_parallel(lp + "mlp.gate_up_proj.weight", 2 * intermediate);
        }

        register_replicated(lp + "pre_ff_layernorm.weight");
    }

    for (auto& [name, info] : shard_map_) {
        info.offload_to_cpu = match_offload_pattern(name);
    }

    spdlog::info("[WeightLayoutPlan] Built plan for rank {}/{}: {} weights mapped, {} local experts",
                 tp_rank_, tp_size_, shard_map_.size(), local_expert_ids_.size());
}

void WeightLayoutPlan::register_weight(const std::string& name,
                                        ShardStrategy strategy,
                                        int shard_dim) {
    WeightShardInfo info;
    info.name = name;
    info.strategy = strategy;
    info.shard_dim = shard_dim;

    shard_map_[name] = info;
}

void WeightLayoutPlan::register_replicated(const std::string& name) {
    register_weight(name, ShardStrategy::REPLICATED, 0);
}

void WeightLayoutPlan::register_column_parallel(const std::string& name,
                                                  int64_t) {
    register_weight(name, ShardStrategy::COLUMN_PARALLEL, 0);
}

void WeightLayoutPlan::register_kv_column_parallel(const std::string& name,
                                                     int64_t,
                                                     int num_kv_heads) {
    if (tp_size_ <= 1 || num_kv_heads >= tp_size_) {
        register_column_parallel(name, 0);
        return;
    }

    int num_kv_head_replicas = tp_size_ / num_kv_heads;

    WeightShardInfo info;
    info.name = name;
    info.strategy = ShardStrategy::COLUMN_PARALLEL;
    info.shard_dim = 0;
    info.num_kv_head_replicas = num_kv_head_replicas;

    shard_map_[name] = info;
}

void WeightLayoutPlan::register_row_parallel(const std::string& name,
                                               int64_t) {
    register_weight(name, ShardStrategy::ROW_PARALLEL, 1);
}

void WeightLayoutPlan::register_qkv_parallel(const std::string& name,
                                               int64_t q_size,
                                               int64_t kv_size,
                                               int) {
    WeightShardInfo info;
    info.name = name;
    info.strategy = ShardStrategy::QKV_PARALLEL;
    info.shard_dim = 0;
    info.output_sizes = {q_size, q_size, kv_size};
    shard_map_[name] = info;
}

void WeightLayoutPlan::register_expert_weights(const std::string& layer_prefix,
                                                 int num_experts) {
    for (int e = 0; e < num_experts; ++e) {
        std::string ep = layer_prefix + "mlp.experts." + std::to_string(e) + ".";
        bool is_local = local_expert_ids_.empty() || local_expert_ids_.count(e) > 0;

        auto make_expert_shard = [&](const std::string& n, ShardStrategy strategy,
                                     int shard_dim) {
            WeightShardInfo info;
            info.name = n;
            info.strategy = strategy;
            info.shard_dim = shard_dim;
            info.skip_for_this_rank = !is_local;

            shard_map_[n] = info;
        };

        make_expert_shard(ep + "gate_proj.weight", ShardStrategy::COLUMN_PARALLEL, 0);
        make_expert_shard(ep + "up_proj.weight", ShardStrategy::COLUMN_PARALLEL, 0);
        make_expert_shard(ep + "down_proj.weight", ShardStrategy::ROW_PARALLEL, 1);
    }
}

void WeightLayoutPlan::register_fused_expert_weights(const std::string& layer_prefix) {
    const int num_experts = model_config_.num_experts;

    static const char* expert_prefixes[] = {"mlp.experts.", "mlp.switch_mlp."};
    static const struct { const char* proj; ShardStrategy strategy; int shard_dim; } projs[] = {
        {"gate_proj", ShardStrategy::FUSED_COLUMN_PARALLEL, 0},
        {"up_proj",   ShardStrategy::FUSED_COLUMN_PARALLEL, 0},
        {"down_proj", ShardStrategy::FUSED_ROW_PARALLEL,    1},
    };

    for (auto& ep : expert_prefixes) {
        for (auto& p : projs) {
            WeightShardInfo info;
            info.name = layer_prefix + ep + p.proj + ".weight";
            info.strategy = p.strategy;
            info.shard_dim = p.shard_dim;
            info.num_experts = num_experts;
            shard_map_[info.name] = info;
        }
    }
}

std::string WeightLayoutPlan::remap_prefix(const std::string& name) const {
    // 与 model_loader.cpp 中的 normalize_weight_name 保持一致
    static const std::pair<std::string, std::string> prefix_map[] = {
        {"model.language_model.model.", "model."},
        {"model.language_model.",       "model."},
        {"language_model.model.",       "model."},
        {"language_model.",             "model."},
        {"model.model.",                "model."},
        {"base_model.model.",           "model."},
    };
    for (auto& [alt, std] : prefix_map) {
        if (name.find(alt) == 0) {
            return std + name.substr(alt.size());
        }
    }
    return name;
}

const WeightShardInfo* WeightLayoutPlan::get_shard_info(const std::string& name) const {
    auto it = shard_map_.find(name);
    if (it != shard_map_.end()) return &it->second;

    std::string remapped = remap_prefix(name);
    if (remapped != name) {
        it = shard_map_.find(remapped);
        if (it != shard_map_.end()) return &it->second;
    }

    return nullptr;
}

bool WeightLayoutPlan::should_skip_weight(const std::string& name) const {
    auto it = shard_map_.find(name);
    if (it != shard_map_.end() && it->second.skip_for_this_rank) {
        return true;
    }

    std::string remapped = remap_prefix(name);
    if (remapped != name) {
        it = shard_map_.find(remapped);
        if (it != shard_map_.end() && it->second.skip_for_this_rank) {
            return true;
        }
    }

    // 如果所有 expert 都是本地（当前 TP 配置下不分片），无需检查 expert ID
    if (local_expert_ids_.size() == static_cast<size_t>(model_config_.num_experts)) {
        return false;
    }

    if (!local_expert_ids_.empty()) {
        // 用 string find/digit 检测替代 regex 获取 expert_id
        auto extract_expert_id = [&](const char* prefix) -> int {
            size_t p = name.find(prefix);
            if (p == std::string::npos) return -1;
            size_t start = p + std::strlen(prefix);
            size_t end = name.find('.', start);
            if (end == std::string::npos || end == start) return -1;
            for (size_t k = start; k < end; ++k) {
                if (!std::isdigit(static_cast<unsigned char>(name[k]))) return -1;
            }
            return std::stoi(name.substr(start, end - start));
        };
        int expert_id = extract_expert_id(".experts.");
        if (expert_id < 0) expert_id = extract_expert_id(".switch_mlp.");
        if (expert_id >= 0 && local_expert_ids_.count(expert_id) == 0) {
            return true;
        }
    }

    return false;
}

bool WeightLayoutPlan::should_offload_weight(const std::string& name) const {
    return match_offload_pattern(name);
}

bool WeightLayoutPlan::match_offload_pattern(const std::string& name) const {
    for (const auto& segment : cache_config_.cpu_offload_params) {
        std::string search = "." + segment + ".";
        std::string full_name = "." + name + ".";
        if (full_name.find(search) != std::string::npos) {
            return true;
        }
    }
    return false;
}

CpuTensorView WeightLayoutPlan::narrow_cpu_tensor(const CpuTensor& full,
                                                    const WeightShardInfo& info) const {
    if (tp_size_ <= 1 || info.strategy == ShardStrategy::REPLICATED) {
        return CpuTensorView(full.cpu_ptr(), full.shape(), full.dtype(), 0);
    }

    int dim = info.shard_dim;

    if (dim < 0 || dim >= static_cast<int>(full.shape().ndim())) {
        return CpuTensorView(full.cpu_ptr(), full.shape(), full.dtype(), 0);
    }

    int64_t actual_dim_size = full.shape()[dim];
    int64_t shard_size = actual_dim_size / tp_size_;
    int64_t shard_offset = static_cast<int64_t>(tp_rank_) * shard_size;

    if (info.num_kv_head_replicas > 1) {
        int kv_shard_rank = tp_rank_ / info.num_kv_head_replicas;
        int64_t kv_shard_size = actual_dim_size / (tp_size_ / info.num_kv_head_replicas);
        shard_size = kv_shard_size;
        shard_offset = static_cast<int64_t>(kv_shard_rank) * kv_shard_size;
    }

    Shape new_shape = full.shape();
    new_shape.dims[dim] = shard_size;

    int64_t element_size = static_cast<int64_t>(dtype_size(full.dtype()));

    int64_t dim_stride = element_size;
    for (int d = static_cast<int>(full.shape().ndim()) - 1; d > dim; --d) {
        dim_stride *= full.shape()[d];
    }

    const char* ptr = static_cast<const char*>(full.cpu_ptr()) + shard_offset * dim_stride;

    if (dim == 0) {
        return CpuTensorView(ptr, new_shape, full.dtype(), 0);
    }

    int64_t row_stride = element_size;
    for (int d = static_cast<int>(full.shape().ndim()) - 1; d > 0; --d) {
        row_stride *= full.shape()[d];
    }

    return CpuTensorView(ptr, new_shape, full.dtype(), row_stride);
}

std::pair<CpuTensorView, std::unique_ptr<CpuTensor>> WeightLayoutPlan::narrow_qkv_tensor(const CpuTensor& full,
                                                                                            const WeightShardInfo& info) const {
    const auto& os = info.output_sizes;
    int dim = info.shard_dim;
    if (dim < 0 || dim >= static_cast<int>(full.shape().ndim())) {
        return {CpuTensorView(full.cpu_ptr(), full.shape(), full.dtype(), 0), nullptr};
    }

    int64_t total_output_size = 0;
    for (auto s : os) {
        total_output_size += s;
    }

    int64_t actual_dim_size = full.shape()[dim];
    if (total_output_size != actual_dim_size) {
        throw std::runtime_error(
            "[QKV-PARALLEL] output_sizes sum=" + std::to_string(total_output_size) +
            " != actual dim size=" + std::to_string(actual_dim_size) +
            " for " + info.name + ". Weight layout plan is inconsistent.");
    }

    int64_t total_shard_size = 0;
    for (auto s : os) {
        total_shard_size += s / tp_size_;
    }

    Shape new_shape = full.shape();
    new_shape.dims[dim] = total_shard_size;

    auto owned = std::make_unique<CpuTensor>(new_shape, full.dtype());

    int64_t element_size = static_cast<int64_t>(dtype_size(full.dtype()));
    int64_t dim_stride = element_size;
    for (int d = static_cast<int>(full.shape().ndim()) - 1; d > dim; --d) {
        dim_stride *= full.shape()[d];
    }

    int64_t shard_row_stride = element_size;
    for (int d = static_cast<int>(new_shape.ndim()) - 1; d > 0; --d) {
        shard_row_stride *= new_shape[d];
    }

    const char* src_base = static_cast<const char*>(full.cpu_ptr());
    char* dst_base = static_cast<char*>(owned->cpu_ptr());

    int64_t src_offset = 0;
    int64_t dst_offset = 0;
    for (auto sub_size : os) {
        int64_t sub_shard_size = sub_size / tp_size_;
        int64_t sub_shard_offset = static_cast<int64_t>(tp_rank_) * sub_shard_size;
        int64_t row_bytes = dim_stride;

        const char* sub_src = src_base + (src_offset + sub_shard_offset) * dim_stride;
        char* sub_dst = dst_base + dst_offset * shard_row_stride;

        if (dim == 0 && shard_row_stride == row_bytes) {
            std::memcpy(sub_dst, sub_src, sub_shard_size * row_bytes);
        } else {
            for (int64_t r = 0; r < sub_shard_size; ++r) {
                std::memcpy(sub_dst + r * shard_row_stride,
                           sub_src + r * dim_stride,
                           row_bytes);
            }
        }

        src_offset += sub_size;
        dst_offset += sub_shard_size;
    }

    return {CpuTensorView(dst_base, new_shape, full.dtype(), 0), std::move(owned)};
}

std::pair<CpuTensorView, std::unique_ptr<CpuTensor>> WeightLayoutPlan::narrow_fused_expert_tensor(const CpuTensor& full,
                                                        const WeightShardInfo& info) const {
    if (tp_size_ <= 1) {
        return {CpuTensorView(full.cpu_ptr(), full.shape(), full.dtype(), 0), nullptr};
    }

    const int num_experts = info.num_experts;
    const int64_t elem_size = static_cast<int64_t>(dtype_size(full.dtype()));
    const char* src = static_cast<const char*>(full.cpu_ptr());

    if (full.shape().ndim() == 3) {
        const int64_t dim1 = full.shape()[1];
        const int64_t dim2 = full.shape()[2];

        if (info.strategy == ShardStrategy::FUSED_COLUMN_PARALLEL) {
            const int64_t shard_size = dim1 / tp_size_;
            const int64_t shard_offset = static_cast<int64_t>(tp_rank_) * shard_size;

            Shape new_shape = {static_cast<int64_t>(num_experts), shard_size, dim2};
            auto owned = std::make_unique<CpuTensor>(new_shape, full.dtype());
            char* dst = static_cast<char*>(owned->cpu_ptr());

            int64_t expert_src_stride = dim1 * dim2 * elem_size;
            int64_t shard_copy_bytes = shard_size * dim2 * elem_size;
            for (int e = 0; e < num_experts; ++e) {
                std::memcpy(dst + e * shard_copy_bytes,
                            src + e * expert_src_stride + shard_offset * dim2 * elem_size,
                            shard_copy_bytes);
            }
            return {CpuTensorView(dst, new_shape, full.dtype(), 0), std::move(owned)};
        } else {
            const int64_t shard_size = dim2 / tp_size_;
            const int64_t shard_offset = static_cast<int64_t>(tp_rank_) * shard_size;

            Shape new_shape = {static_cast<int64_t>(num_experts), dim1, shard_size};
            auto owned = std::make_unique<CpuTensor>(new_shape, full.dtype());
            char* dst = static_cast<char*>(owned->cpu_ptr());

            int64_t expert_src_stride = dim1 * dim2 * elem_size;
            int64_t dst_expert_stride = dim1 * shard_size * elem_size;
            int64_t src_row_bytes = dim2 * elem_size;
            int64_t dst_row_bytes = shard_size * elem_size;

            for (int e = 0; e < num_experts; ++e) {
                const char* expert_src = src + e * expert_src_stride;
                char* expert_dst = dst + e * dst_expert_stride;
                for (int64_t r = 0; r < dim1; ++r) {
                    std::memcpy(expert_dst + r * dst_row_bytes,
                                expert_src + r * src_row_bytes + shard_offset * elem_size,
                                dst_row_bytes);
                }
            }
            return {CpuTensorView(dst, new_shape, full.dtype(), 0), std::move(owned)};
        }
    }

    const int64_t expert_rows = full.shape()[0] / num_experts;

    if (info.strategy == ShardStrategy::FUSED_COLUMN_PARALLEL) {
        const int64_t shard_size = expert_rows / tp_size_;
        const int64_t shard_offset = static_cast<int64_t>(tp_rank_) * shard_size;
        const int64_t row_bytes = full.shape()[1] * elem_size;

        Shape new_shape = full.shape();
        new_shape.dims[0] = static_cast<int64_t>(num_experts) * shard_size;

        auto owned = std::make_unique<CpuTensor>(new_shape, full.dtype());
        char* dst = static_cast<char*>(owned->cpu_ptr());
        int64_t expert_src_stride = expert_rows * row_bytes;
        for (int e = 0; e < num_experts; ++e) {
            std::memcpy(dst + e * shard_size * row_bytes,
                        src + e * expert_src_stride + shard_offset * row_bytes,
                        shard_size * row_bytes);
        }
        return {CpuTensorView(dst, new_shape, full.dtype(), 0), std::move(owned)};
    } else {
        const int64_t shard_size = full.shape()[1] / tp_size_;
        const int64_t shard_offset = static_cast<int64_t>(tp_rank_) * shard_size;
        const int64_t row_bytes = full.shape()[1] * elem_size;

        Shape new_shape = full.shape();
        new_shape.dims[1] = shard_size;
        auto owned = std::make_unique<CpuTensor>(new_shape, full.dtype());
        char* dst = static_cast<char*>(owned->cpu_ptr());
        int64_t shard_row_bytes = shard_size * elem_size;

        for (int e = 0; e < num_experts; ++e) {
            const char* expert_src = src + e * expert_rows * row_bytes;
            char* shard_dst = dst + e * expert_rows * shard_row_bytes;
            for (int64_t r = 0; r < expert_rows; ++r) {
                std::memcpy(shard_dst + r * shard_row_bytes,
                            expert_src + r * row_bytes + shard_offset * elem_size,
                            shard_row_bytes);
            }
        }
        return {CpuTensorView(dst, new_shape, full.dtype(), 0), std::move(owned)};
    }
}




WeightShardInfo WeightLayoutPlan::derive_sub_tensor_shard_info(
    const std::string& name, const Shape& tensor_shape) const {

    WeightShardInfo info;
    info.name = name;
    info.strategy = ShardStrategy::REPLICATED;

    if (tp_size_ <= 1) return info;

    static const struct { const char* suffix; const char* replace; } suffix_map[] = {
        {".biases",              ".weight"},
        {".weight_packed",       ".weight"},
        {".weight_scale",        ".weight"},
        {".weight_zero_point",   ".weight"},
    };

    std::string base_name;
    for (auto& entry : suffix_map) {
        std::string suf = entry.suffix;
        if (name.size() > suf.size() &&
            name.compare(name.size() - suf.size(), suf.size(), suf) == 0) {
            base_name = name.substr(0, name.size() - suf.size()) + entry.replace;
            break;
        }
    }

    if (base_name.empty()) return info;

    const auto* dense_info = get_shard_info(base_name);
    bool is_per_expert = false;
    if (!dense_info) {
        // 用 string::find + replace 替代 std::regex_replace，获得 ~10-50x 加速
        std::string stripped = base_name;
        auto do_replace = [&](const char* prefix) -> bool {
            size_t p = stripped.find(prefix);
            if (p == std::string::npos) return false;
            size_t dot2 = stripped.find('.', p + std::strlen(prefix));
            if (dot2 == std::string::npos) return false;
            stripped.replace(p, dot2 - p + 1, ".experts.");
            return true;
        };
        if (do_replace(".experts.") || do_replace(".switch_mlp.")) {
            dense_info = get_shard_info(stripped);
            is_per_expert = true;
        }
    }
    if (!dense_info || dense_info->strategy == ShardStrategy::REPLICATED) return info;

    if (dense_info->skip_for_this_rank) {
        info.skip_for_this_rank = true;
        return info;
    }

    if (is_per_expert) {
        if (dense_info->strategy == ShardStrategy::FUSED_COLUMN_PARALLEL) {
            info.strategy = ShardStrategy::COLUMN_PARALLEL;
            info.shard_dim = 0;
        } else if (dense_info->strategy == ShardStrategy::FUSED_ROW_PARALLEL) {
            info.strategy = ShardStrategy::ROW_PARALLEL;
            info.shard_dim = 1;
        } else {
            info.strategy = dense_info->strategy;
            info.shard_dim = dense_info->strategy == ShardStrategy::COLUMN_PARALLEL ||
                             dense_info->strategy == ShardStrategy::QKV_PARALLEL ? 0 : 1;
        }
        info.num_experts = 0;
        info.num_kv_head_replicas = dense_info->num_kv_head_replicas;
        info.output_sizes = dense_info->output_sizes;
        info.skip_for_this_rank = false;
        return info;
    }

    info.strategy = dense_info->strategy;
    info.num_experts = dense_info->num_experts;
    info.num_kv_head_replicas = dense_info->num_kv_head_replicas;
    info.output_sizes = dense_info->output_sizes;
    info.skip_for_this_rank = false;

    bool is_column = dense_info->strategy == ShardStrategy::COLUMN_PARALLEL ||
                     dense_info->strategy == ShardStrategy::QKV_PARALLEL ||
                     dense_info->strategy == ShardStrategy::FUSED_COLUMN_PARALLEL;

    if (dense_info->strategy == ShardStrategy::FUSED_COLUMN_PARALLEL ||
        dense_info->strategy == ShardStrategy::FUSED_ROW_PARALLEL) {
        if (tensor_shape.ndim() == 3) {
            info.shard_dim = is_column ? 1 : 2;
        } else {
            info.shard_dim = is_column ? 0 : 1;
        }
    } else {
        info.shard_dim = is_column ? 0 : 1;
    }

    return info;
}

}
