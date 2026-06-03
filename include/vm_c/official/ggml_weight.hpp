#pragma once

#include "ggml.h"

#include <cstdint>
#include <memory>

struct ggml_backend_buffer;
typedef struct ggml_backend_buffer* ggml_backend_buffer_t;

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace vm_c::official {

/// 加载期分配的 ggml CUDA 权重缓冲（对齐 llama.cpp：ggml_backend_buft_alloc_buffer + WEIGHTS）
struct GgmlWeightStorage {
    ggml_backend_buffer_t buffer = nullptr;
    void* data = nullptr;
    ggml_type type = GGML_TYPE_IQ4_XS;
    int device = 0;
    int64_t ne0 = 0;
    int64_t ne1 = 1;
    int64_t ne2 = 1;

    ~GgmlWeightStorage();
    GgmlWeightStorage() = default;
    GgmlWeightStorage(const GgmlWeightStorage&) = delete;
    GgmlWeightStorage& operator=(const GgmlWeightStorage&) = delete;
    GgmlWeightStorage(GgmlWeightStorage&&) = delete;
    GgmlWeightStorage& operator=(GgmlWeightStorage&&) = delete;
};

struct GgmlWeightRef {
    ggml_backend_buffer_t buffer = nullptr;
    void* data = nullptr;
    ggml_type type = GGML_TYPE_IQ4_XS;
    int64_t ne0 = 0;
    int64_t ne1 = 1;
    int64_t ne2 = 1;

    bool valid() const { return buffer != nullptr && data != nullptr; }
};

/// host staging → ggml CUDA buffer（device 侧 H2D）
std::shared_ptr<GgmlWeightStorage> ggml_upload_quant_from_host(
    ggml_type type,
    int gpu_device,
    int64_t ne0, int64_t ne1, int64_t ne2,
    const void* host_data, size_t host_bytes,
    cudaStream_t stream);

inline std::shared_ptr<GgmlWeightStorage> ggml_upload_iq4_from_host(
    int gpu_device,
    int64_t ne0, int64_t ne1, int64_t ne2,
    const void* host_data, size_t host_bytes,
    cudaStream_t stream) {
    return ggml_upload_quant_from_host(
        GGML_TYPE_IQ4_XS, gpu_device, ne0, ne1, ne2, host_data, host_bytes, stream);
}

inline std::shared_ptr<GgmlWeightStorage> ggml_upload_q5_k_from_host(
    int gpu_device,
    int64_t ne0, int64_t ne1, int64_t ne2,
    const void* host_data, size_t host_bytes,
    cudaStream_t stream) {
    return ggml_upload_quant_from_host(
        GGML_TYPE_Q5_K, gpu_device, ne0, ne1, ne2, host_data, host_bytes, stream);
}

GgmlWeightRef ggml_weight_ref(const std::shared_ptr<GgmlWeightStorage>& storage);

/// device FP16/BF16 权重 → ggml CUDA WEIGHTS buffer（供 ggml_mul_mat / fused gating）
std::shared_ptr<GgmlWeightStorage> ggml_upload_dense_from_device(
    ggml_type type,
    int gpu_device,
    int64_t ne0, int64_t ne1, int64_t ne2,
    const void* device_src, size_t src_bytes,
    cudaStream_t stream);

bool ggml_bind_weight_tensor(
    ggml_tensor* tensor,
    const GgmlWeightRef& weight);

/// 图缓存命中时切换层权重（已分配 tensor 不可再次 ggml_backend_tensor_alloc）
bool ggml_rebind_weight_tensor(
    ggml_tensor* tensor,
    const GgmlWeightRef& weight);

/// TP 分片 bind：同步 ne/nb 与 vm_c ggml buffer 布局（ggml_nbytes 依赖 nb，不能只改 ne）
void ggml_sync_tensor_shape_from_weight_ref(
    ggml_tensor* tensor,
    const GgmlWeightRef& weight);

}  // namespace vm_c::official
