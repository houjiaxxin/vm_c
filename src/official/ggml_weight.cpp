#include "vm_c/official/ggml_weight.hpp"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include "vm_c/cuda/gpu_arch.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

namespace vm_c::official {

GgmlWeightStorage::~GgmlWeightStorage() {
    if (buffer) {
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
    }
    data = nullptr;
}

std::shared_ptr<GgmlWeightStorage> ggml_upload_quant_from_host(
    ggml_type type,
    int gpu_device,
    int64_t ne0, int64_t ne1, int64_t ne2,
    const void* host_data, size_t host_bytes,
    cudaStream_t stream) {
    if (!host_data || host_bytes == 0 || ne0 <= 0) {
        throw std::runtime_error("ggml_upload_quant_from_host: invalid arguments");
    }

    ggml_init_params init_params{};
    init_params.mem_size = 16u << 10;
    init_params.no_alloc = true;
    ggml_context* ctx = ggml_init(init_params);
    if (!ctx) {
        throw std::runtime_error("ggml_upload_quant_from_host: ggml_init failed");
    }

    ggml_tensor* layout = (ne2 > 1)
        ? ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2)
        : ggml_new_tensor_2d(ctx, type, ne0, ne1);

    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(gpu_device);
    const size_t alloc_bytes = ggml_backend_buft_get_alloc_size(buft, layout);
    if (host_bytes > alloc_bytes) {
        ggml_free(ctx);
        throw std::runtime_error(
            "ggml_upload_quant_from_host: host payload exceeds ggml alloc size");
    }

    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, alloc_bytes);
    if (!buf) {
        ggml_free(ctx);
        throw std::runtime_error("ggml_upload_quant_from_host: buffer allocation failed");
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    void* base = ggml_backend_buffer_get_base(buf);
    if (!base) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        throw std::runtime_error("ggml_upload_quant_from_host: null buffer base");
    }

    if (ggml_backend_tensor_alloc(buf, layout, base) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        throw std::runtime_error("ggml_upload_quant_from_host: tensor alloc failed");
    }
    ggml_free(ctx);

    CUDA_CHECK(cudaSetDevice(gpu_device));
    CUDA_CHECK(cudaMemcpyAsync(base, host_data, host_bytes, cudaMemcpyHostToDevice, stream));

    auto storage = std::make_shared<GgmlWeightStorage>();
    storage->buffer = buf;
    storage->data = base;
    storage->type = type;
    storage->device = gpu_device;
    storage->ne0 = ne0;
    storage->ne1 = ne1;
    storage->ne2 = ne2;
    return storage;
}

std::shared_ptr<GgmlWeightStorage> ggml_upload_dense_from_device(
    ggml_type type,
    int gpu_device,
    int64_t ne0, int64_t ne1, int64_t ne2,
    const void* device_src, size_t src_bytes,
    cudaStream_t stream) {
    if (!device_src || src_bytes == 0 || ne0 <= 0) {
        throw std::runtime_error("ggml_upload_dense_from_device: invalid arguments");
    }
    if (type != GGML_TYPE_F16 && type != GGML_TYPE_BF16 && type != GGML_TYPE_F32) {
        throw std::runtime_error("ggml_upload_dense_from_device: unsupported ggml type");
    }

    ggml_init_params init_params{};
    init_params.mem_size = 16u << 10;
    init_params.no_alloc = true;
    ggml_context* ctx = ggml_init(init_params);
    if (!ctx) {
        throw std::runtime_error("ggml_upload_dense_from_device: ggml_init failed");
    }

    ggml_tensor* layout = (ne2 > 1)
        ? ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2)
        : ggml_new_tensor_2d(ctx, type, ne0, ne1);

    ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(gpu_device);
    const size_t alloc_bytes = ggml_backend_buft_get_alloc_size(buft, layout);
    if (src_bytes > alloc_bytes) {
        ggml_free(ctx);
        throw std::runtime_error(
            "ggml_upload_dense_from_device: device payload exceeds ggml alloc size");
    }

    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, alloc_bytes);
    if (!buf) {
        ggml_free(ctx);
        throw std::runtime_error("ggml_upload_dense_from_device: buffer allocation failed");
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    void* base = ggml_backend_buffer_get_base(buf);
    if (!base) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        throw std::runtime_error("ggml_upload_dense_from_device: null buffer base");
    }

    if (ggml_backend_tensor_alloc(buf, layout, base) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        throw std::runtime_error("ggml_upload_dense_from_device: tensor alloc failed");
    }
    ggml_free(ctx);

    CUDA_CHECK(cudaSetDevice(gpu_device));
    CUDA_CHECK(cudaMemcpyAsync(base, device_src, src_bytes, cudaMemcpyDeviceToDevice, stream));

    auto storage = std::make_shared<GgmlWeightStorage>();
    storage->buffer = buf;
    storage->data = base;
    storage->type = type;
    storage->device = gpu_device;
    storage->ne0 = ne0;
    storage->ne1 = ne1;
    storage->ne2 = ne2;
    return storage;
}

GgmlWeightRef ggml_weight_ref(const std::shared_ptr<GgmlWeightStorage>& storage) {
    GgmlWeightRef ref{};
    if (!storage) {
        return ref;
    }
    ref.buffer = storage->buffer;
    ref.data = storage->data;
    ref.type = storage->type;
    ref.ne0 = storage->ne0;
    ref.ne1 = storage->ne1;
    ref.ne2 = storage->ne2;
    return ref;
}

void ggml_sync_tensor_shape_from_weight_ref(
    ggml_tensor* tensor,
    const GgmlWeightRef& weight) {
    if (!tensor) {
        throw std::runtime_error("ggml_sync_tensor_shape_from_weight_ref: null tensor");
    }
    if (weight.ne0 <= 0) {
        throw std::runtime_error("ggml_sync_tensor_shape_from_weight_ref: invalid ne0");
    }

    // 与 ggml_new_tensor_impl 一致：ne/nb 必须与 upload 阶段 layout 相同
    tensor->ne[0] = weight.ne0;
    tensor->ne[1] = weight.ne1 > 0 ? weight.ne1 : 1;
    tensor->ne[2] = weight.ne2 > 0 ? weight.ne2 : 1;
    tensor->ne[3] = 1;

    const size_t type_size = ggml_type_size(tensor->type);
    const int64_t blck_size = ggml_blck_size(tensor->type);
    tensor->nb[0] = type_size;
    tensor->nb[1] = tensor->nb[0] * (tensor->ne[0] / blck_size);
    for (int i = 2; i < GGML_MAX_DIMS; ++i) {
        tensor->nb[i] = tensor->nb[i - 1] * tensor->ne[i - 1];
    }
}

bool ggml_bind_weight_tensor(ggml_tensor* tensor, const GgmlWeightRef& weight) {
    if (!tensor || !weight.valid()) {
        return false;
    }
    return ggml_backend_tensor_alloc(weight.buffer, tensor, weight.data) == GGML_STATUS_SUCCESS;
}

bool ggml_rebind_weight_tensor(ggml_tensor* tensor, const GgmlWeightRef& weight) {
    if (!tensor || !weight.valid()) {
        return false;
    }
    if (tensor->buffer == nullptr && tensor->data == nullptr) {
        return ggml_bind_weight_tensor(tensor, weight);
    }
    if (tensor->data == nullptr && tensor->buffer != nullptr) {
        // 清除 no_alloc dummy buffer，再 bind 到 vm_c WEIGHTS buffer
        tensor->buffer = nullptr;
        return ggml_bind_weight_tensor(tensor, weight);
    }
    if (tensor->type != weight.type) {
        throw std::runtime_error(
            std::string("ggml_rebind_weight_tensor: ggml type mismatch: graph tensor uses ")
            + ggml_type_name(tensor->type) + " but weight is "
            + ggml_type_name(weight.type)
            + " (graph cache must rebuild for different quant types)");
    }
    const int64_t tensor_ne2 = (ggml_n_dims(tensor) >= 3) ? tensor->ne[2] : 1;
    const int64_t weight_ne2 = weight.ne2 > 0 ? weight.ne2 : 1;
    if (tensor->ne[0] != weight.ne0 || tensor->ne[1] != weight.ne1 ||
        tensor_ne2 != weight_ne2) {
        throw std::runtime_error(
            std::string("ggml_rebind_weight_tensor: shape mismatch graph=[")
            + std::to_string(tensor->ne[0]) + "," + std::to_string(tensor->ne[1]) + ","
            + std::to_string(tensor_ne2) + "] weight=[" + std::to_string(weight.ne0) + ","
            + std::to_string(weight.ne1) + "," + std::to_string(weight_ne2) + "]");
    }
    // 图缓存复用：切换 data/buffer 指针（各层权重独立 WEIGHTS buffer）
    tensor->buffer = weight.buffer;
    tensor->data = weight.data;
    return true;
}

}  // namespace vm_c::official
