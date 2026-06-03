#include <unordered_set>
#include <functional>
#include <mutex>
#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <atomic>

#include "vm_c/core/tensor.hpp"
#include "vm_c/core/config.hpp"
#include "vm_c/model/weight_layout_plan.hpp"

namespace vm_c {

struct WeightTensor {
    std::string name;
    Shape shape;
    DataType dtype;
    int64_t offset;
    size_t nbytes;
    std::string shard_file;
};

class SafeTensorsLoader {
public:
    explicit SafeTensorsLoader(const std::string& model_dir);
    ~SafeTensorsLoader() = default;

    bool load_index();
    std::vector<std::string> tensor_names() const;
    const WeightTensor& tensor_meta(const std::string& name) const;
    bool has_tensor(const std::string& name) const;

    CpuTensor load_tensor(const std::string& name);
    std::vector<std::string> get_shard_files() const { return shard_files_; }
    void load_shard_tensors(const std::string& shard_path,
                            std::function<void(const std::string&, CpuTensor&&)> on_tensor);

    // Get total tensor count from the index
    size_t tensor_count() const { return meta_.size(); }

private:
    std::string model_dir_;
    std::unordered_map<std::string, WeightTensor> meta_;
    std::vector<std::string> shard_files_;
};

struct ModelWeights {
    std::unordered_map<std::string, GpuTensor> gpu_weights;

    GpuTensor* find(const std::string& name) {
        auto it = gpu_weights.find(name);
        return it != gpu_weights.end() ? &it->second : nullptr;
    }

    void add_weight(const std::string& name, GpuTensor tensor) {
        gpu_weights.emplace(name, std::move(tensor));
    }

    // 支持合并专家张量：在已有名称外，额外添加切片后的逐个专家条目
    // 用于 llama.cpp GGUF 格式的单个大张量 → 模型期望的逐个专家格式转换
    void add_expert_slices(const std::string& merged_name, const GpuTensor& tensor,
                           int num_experts, const std::string& expert_prefix);

    void emplace(const std::string& name, GpuTensor&& tensor) {
        gpu_weights.emplace(name, std::move(tensor));
    }
};

class ModelLoader {
public:
    ModelLoader(const ModelConfig& model_config, const CacheConfig& cache_config,
                int gpu_device = 0, int tp_rank = 0, int tp_size = 1);
    ~ModelLoader();

    ModelConfig load_config();
    ModelWeights load_weights();

    // 是否使用 GGUF 格式
    bool is_gguf() const { return is_gguf_; }

    // 转移 UVA 缓冲区所有权（TP 模式下 ModelLoader 会被销毁，
    // 但 UVA 映射内存必须持续到推理结束）
    std::vector<void*> release_uva_buffers() {
        auto result = std::move(uva_buffers_);
        uva_buffers_.clear();
        return result;
    }
    void load_weight_to_gpu(const std::string& name, const CpuTensorView& view,
                            ModelWeights& weights);
    bool load_weight_uva_offload(const std::string& name, const CpuTensorView& view,
                                  ModelWeights& weights);

    const WeightLayoutPlan* layout_plan() const { return layout_plan_.get(); }

    // GGUF 路径在构造时尚无完整 ModelConfig，需在 load_config 之后构建分片计划。
    void ensure_layout_plan(const ModelConfig& config);

private:
    bool should_offload_to_cpu(const std::string& tensor_name) const;

    ModelConfig model_config_;
    CacheConfig cache_config_;
    int gpu_device_;
    int tp_rank_;
    int tp_size_;
    bool is_gguf_ = false;
    std::string gguf_path_;
    std::unique_ptr<SafeTensorsLoader> st_loader_;
    std::unique_ptr<WeightLayoutPlan> layout_plan_;
    size_t cpu_offload_capacity_ = 0;
    std::atomic<size_t> cpu_offload_used_{0};
    std::vector<void*> uva_buffers_;
    cudaStream_t load_stream_ = nullptr;

    // Track loaded tensor names for post-load verification
    std::unordered_set<std::string> loaded_tensor_names_;
    std::atomic<int> expected_tensors_{0};
    std::atomic<int> loaded_tensor_count_{0};

    // Mutex for thread-safe weights.emplace during parallel shard loading
    mutable std::mutex weights_mutex_;
};

}
