#include "vm_c/model/models.hpp"
#include "vm_c/cuda/gpu_arch.hpp"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace vm_c {

namespace {

void wrap_dense_tensor_for_ggml(GpuTensor& tensor, int gpu_device, cudaStream_t stream) {
    if (tensor.has_ggml_weight() || !tensor.data_ptr()) {
        return;
    }
    if (tensor.dtype() != DataType::FLOAT16 && tensor.dtype() != DataType::BFLOAT16
        && tensor.dtype() != DataType::FLOAT32) {
        return;
    }
    int64_t ne0 = 0;
    int64_t ne1 = 1;
    Shape shape = tensor.shape();
    if (shape.ndim() == 1) {
        ne0 = shape[0];
        shape = Shape({1, shape[0]});
    } else if (shape.ndim() == 2) {
        ne0 = shape[1];
        ne1 = shape[0];
    } else {
        throw std::runtime_error("wrap_dense_tensor_for_ggml: unsupported weight rank");
    }
    ggml_type gt = GGML_TYPE_F16;
    if (tensor.dtype() == DataType::BFLOAT16) {
        gt = GGML_TYPE_BF16;
    } else if (tensor.dtype() == DataType::FLOAT32) {
        gt = GGML_TYPE_F32;
    }
    const size_t bytes = static_cast<size_t>(tensor.shape().numel()) * dtype_size(tensor.dtype());
    auto storage = official::ggml_upload_dense_from_device(
        gt, gpu_device, ne0, ne1, 1, tensor.data_ptr(), bytes, stream);
    tensor.adopt_ggml_weight(std::move(storage), shape, tensor.dtype(), gpu_device);
}

void prepare_llama_weight_snapshot_impl(
    std::unordered_map<std::string, GpuTensor>& gpu_weights,
    const ModelConfig& config,
    int gpu_device) {
    cudaStream_t wrap_stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&wrap_stream, cudaStreamNonBlocking));
    for (auto& [name, tensor] : gpu_weights) {
        (void)name;
        wrap_dense_tensor_for_ggml(tensor, gpu_device, wrap_stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(wrap_stream));
    CUDA_CHECK(cudaStreamDestroy(wrap_stream));

    if (config.spec_method != "mtp" || config.mtp_predict_layers <= 0) {
        return;
    }
    if (config.mtp_predict_layers != 1) {
        throw std::runtime_error(
            "MTP: only nextn_predict_layers == 1 is supported (got "
            + std::to_string(config.mtp_predict_layers) + ")");
    }
    const int mtp_il = config.mtp_block_layer_index();
    if (mtp_il < 0) {
        throw std::runtime_error("MTP enabled but mtp_block_layer_index() is invalid");
    }

    auto has_tensor_suffix = [&](const std::string& suffix) -> bool {
        for (const auto& [n, t] : gpu_weights) {
            if (n.size() >= suffix.size()
                && n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0
                && t.data_ptr()) {
                return true;
            }
        }
        return false;
    };

    const std::string layer = ".layers." + std::to_string(mtp_il) + ".";
    if (!has_tensor_suffix(layer + "nextn.eh_proj.weight")
        || !has_tensor_suffix(layer + "nextn.en_norm.weight")
        || !has_tensor_suffix(layer + "nextn.h_norm.weight")) {
        throw std::runtime_error(
            "MTP block layer " + std::to_string(mtp_il)
            + " missing nextn.eh_proj / en_norm / h_norm weights");
    }
    if (!has_tensor_suffix(layer + "self_attn.q_proj.weight")) {
        throw std::runtime_error(
            "MTP block layer " + std::to_string(mtp_il) + " missing attention weights");
    }
    if (!has_tensor_suffix(layer + "mlp.gate.weight")
        && !has_tensor_suffix(layer + "mlp.router.weight")) {
        throw std::runtime_error(
            "MTP block layer " + std::to_string(mtp_il) + " missing MoE router weights");
    }
    spdlog::info("prepare_llama_weight_snapshot: MTP block validated at layer {}", mtp_il);
}

}  // namespace

void prepare_llama_weight_snapshot(
    std::unordered_map<std::string, GpuTensor>& gpu_weights,
    const ModelConfig& config,
    int gpu_device) {
#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
    prepare_llama_weight_snapshot_impl(gpu_weights, config, gpu_device);
#else
    (void)gpu_weights;
    (void)config;
    (void)gpu_device;
    throw std::runtime_error(
        "prepare_llama_weight_snapshot requires VM_C_USE_OFFICIAL_GGML_MOE");
#endif
}

}  // namespace vm_c
