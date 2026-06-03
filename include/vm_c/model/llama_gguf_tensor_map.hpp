#pragma once

#include <string>
#include <unordered_set>
#include <vector>

namespace vm_c {

/// GGUF/llama 张量名 → vm_c ModelLoader 内部名（与 gguf_model_loader 一致）
std::string gguf_tensor_name_to_vmc(const std::string& gguf_name, int num_layers);

/// vm_c 内部名 → llama GGUF 张量名；多候选时在 llama_names 中选存在的项
std::string vmc_tensor_name_to_llama_gguf(
    const std::string& vmc_name,
    int num_layers,
    const std::unordered_set<std::string>* llama_names = nullptr);

}  // namespace vm_c
