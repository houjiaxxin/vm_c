#pragma once

#include "vm_c/core/config.hpp"
#include "vm_c/core/tensor.hpp"

#include <string>
#include <unordered_map>

namespace vm_c {

/// libllama 路径：为 ModelLoader 分片权重建立 ggml 视图，供 bind_vm_c_weights 使用。
void prepare_llama_weight_snapshot(
    std::unordered_map<std::string, GpuTensor>& gpu_weights,
    const ModelConfig& config,
    int gpu_device);

}  // namespace vm_c
