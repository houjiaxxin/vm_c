#include "vm_c/core/tensor.hpp"
#include "vm_c/cuda/gpu_arch.hpp"
#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
#include "vm_c/official/ggml_weight.hpp"
#endif
#include <cuda_runtime.h>
#include <cstring>
#include <spdlog/spdlog.h>

namespace vm_c {

// ── GpuTensor ──

GpuTensor::GpuTensor(Shape shape, DataType dtype, int device)
    : Tensor(nullptr, shape, dtype, device) {
    size_t bytes = shape.numel() * dtype_size(dtype);
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMalloc(&data_, bytes));
}

GpuTensor::~GpuTensor() {
#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
    if (ggml_weight_) {
        ggml_weight_.reset();
        data_ = nullptr;
        return;
    }
#endif
    if (data_ && !is_uva_) {
        CUDA_CHECK(cudaFree(data_));
        data_ = nullptr;
    }
}

GpuTensor::GpuTensor(GpuTensor&& o) noexcept
    : Tensor(o.data_, o.shape_, o.dtype_, o.device_) {
    is_uva_ = o.is_uva_;
#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
    ggml_weight_ = std::move(o.ggml_weight_);
#endif
    o.data_ = nullptr;
    o.is_uva_ = false;
}

GpuTensor& GpuTensor::operator=(GpuTensor&& o) noexcept {
    if (this != &o) {
#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
        ggml_weight_.reset();
#endif
        if (data_ && !is_uva_ && !ggml_weight_) {
            CUDA_CHECK(cudaFree(data_));
        }
        data_ = o.data_;
        shape_ = o.shape_;
        dtype_ = o.dtype_;
        device_ = o.device_;
        is_uva_ = o.is_uva_;
#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
        ggml_weight_ = std::move(o.ggml_weight_);
#endif
        o.data_ = nullptr;
        o.is_uva_ = false;
    }
    return *this;
}

#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
void GpuTensor::adopt_ggml_weight(std::shared_ptr<official::GgmlWeightStorage> storage,
                                  Shape shape, DataType dtype, int device) {
    if (data_ && !is_uva_ && !ggml_weight_) {
        CUDA_CHECK(cudaFree(data_));
    }
    ggml_weight_ = std::move(storage);
    data_ = ggml_weight_ ? ggml_weight_->data : nullptr;
    shape_ = std::move(shape);
    dtype_ = dtype;
    device_ = device;
    is_uva_ = false;
}

official::GgmlWeightRef GpuTensor::ggml_weight_ref() const {
    return official::ggml_weight_ref(ggml_weight_);
}
#endif

std::shared_ptr<GpuTensor> GpuTensor::create(Shape shape, DataType dtype, int device) {
    return std::shared_ptr<GpuTensor>(new GpuTensor(shape, dtype, device));
}

CpuTensor::CpuTensor(Shape shape, DataType dtype)
    : Tensor(nullptr, shape, dtype, -1) {
    size_t bytes = shape.numel() * dtype_size(dtype);
    CUDA_CHECK(cudaHostAlloc(&data_, bytes, cudaHostAllocDefault));
    std::memset(data_, 0, bytes);
}

CpuTensor::~CpuTensor() {
    if (data_) {
        CUDA_CHECK(cudaFreeHost(data_));
        data_ = nullptr;
    }
}

CpuTensor::CpuTensor(CpuTensor&& o) noexcept
    : Tensor(o.data_, o.shape_, o.dtype_, o.device_) {
    o.data_ = nullptr;
}

CpuTensor& CpuTensor::operator=(CpuTensor&& o) noexcept {
    if (this != &o) {
        if (data_) CUDA_CHECK(cudaFreeHost(data_));
        data_ = o.data_;
        shape_ = o.shape_;
        dtype_ = o.dtype_;
        device_ = o.device_;
        o.data_ = nullptr;
    }
    return *this;
}

std::shared_ptr<CpuTensor> CpuTensor::create(Shape shape, DataType dtype) {
    return std::shared_ptr<CpuTensor>(new CpuTensor(shape, dtype));
}

}
