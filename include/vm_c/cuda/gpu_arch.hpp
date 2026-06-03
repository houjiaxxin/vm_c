#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "vm_c/core/tensor.hpp"

namespace vm_c {

// ── llama.cpp 风格的 CUDA_CHECK/CUBLAS_CHECK ──

namespace detail {
[[noreturn]] inline void throw_cuda_error(const char* expr, const char* func,
                                           const char* file, int line,
                                           const char* msg) {
    std::string full = std::string("[CUDA] ") + file + ":" + std::to_string(line) +
                       " in " + func + "(): '" + expr + "' failed: " + msg;
    throw std::runtime_error(full);
}
[[noreturn]] inline void throw_cublas_error(const char* expr, const char* func,
                                             const char* file, int line,
                                             const char* msg) {
    std::string full = std::string("[CUBLAS] ") + file + ":" + std::to_string(line) +
                       " in " + func + "(): '" + expr + "' failed: " + msg;
    throw std::runtime_error(full);
}

inline const char* cublas_error_str(cublasStatus_t err) {
#if CUDART_VERSION >= 12000
    return cublasGetStatusString(err);
#else
    switch (err) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        default: return "unknown cublas error";
    }
#endif
}
} // namespace detail

#define CUDA_CHECK(err)                                                      \
    do {                                                                     \
        cudaError_t _err_ = (err);                                           \
        if (_err_ != cudaSuccess) {                                          \
            ::vm_c::detail::throw_cuda_error(                                \
                #err, __func__, __FILE__, __LINE__,                          \
                cudaGetErrorString(_err_));                                  \
        }                                                                    \
    } while (0)

#define CUBLAS_CHECK(err)                                                    \
    do {                                                                     \
        cublasStatus_t _err_ = (err);                                        \
        if (_err_ != CUBLAS_STATUS_SUCCESS) {                                \
            ::vm_c::detail::throw_cublas_error(                              \
                #err, __func__, __FILE__, __LINE__,                          \
                ::vm_c::detail::cublas_error_str(_err_));                    \
        }                                                                    \
    } while (0)

#define CHECK_KERNEL_LAUNCH()                                                \
    do {                                                                     \
        CUDA_CHECK(cudaGetLastError());                                      \
    } while (0)

inline void set_device_safe(int device) {
    CUcontext cur_ctx = nullptr;
    if (cuCtxGetCurrent(&cur_ctx) == CUDA_SUCCESS && cur_ctx != nullptr) {
        int cur = -1;
        CUDA_CHECK(cudaGetDevice(&cur));
        if (cur == device) return;
    }
    CUDA_CHECK(cudaSetDevice(device));
}

class GpuArch {
public:
    static GpuArch& instance() {
        static GpuArch inst;
        return inst;
    }

    void detect(int device_id = 0) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (detected_) return;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
        sm_major_ = prop.major;
        sm_minor_ = prop.minor;
        sm_int_ = sm_major_ * 10 + sm_minor_;
        sm_bf16_ = (sm_int_ >= 80);
        if (sm_bf16_) {
            compute_dtype_ = DataType::BFLOAT16;
        } else {
            compute_dtype_ = DataType::FLOAT16;
        }
        detected_ = true;
    }

    bool supports_bf16() const { return sm_bf16_; }
    int sm_major() const { return sm_major_; }
    int sm_minor() const { return sm_minor_; }
    int sm_int() const { return sm_int_; }

    DataType compute_dtype() const { return compute_dtype_; }
    size_t compute_dtype_size() const { return dtype_size(compute_dtype_); }

    bool has_tensor_core() const { return sm_int_ >= 70; }

    // 与 ggml GGML_CUDA_CC_AMPERE (sm_80+) 策略一致：Turing 及以下禁用 CUDA Graph
    bool supports_cuda_graph() const { return sm_major_ >= 8; }

private:
    GpuArch() = default;
    std::mutex mtx_;
    bool detected_ = false;
    int sm_major_ = 0;
    int sm_minor_ = 0;
    int sm_int_ = 0;
    bool sm_bf16_ = false;
    DataType compute_dtype_ = DataType::FLOAT16;
};

inline const GpuArch& gpu() { return GpuArch::instance(); }

inline void gpu_detect(int device_id = 0) { GpuArch::instance().detect(device_id); }

inline cublasHandle_t get_cublas_handle(int device_id) {
    static std::unordered_map<int, cublasHandle_t> handles;
    static std::mutex mtx;
    {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = handles.find(device_id);
        if (it != handles.end()) return it->second;
    }
    int prev_dev = -1;
    CUDA_CHECK(cudaGetDevice(&prev_dev));
    if (prev_dev != device_id) {
        CUDA_CHECK(cudaSetDevice(device_id));
    }
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, 0));
    if (prev_dev >= 0 && prev_dev != device_id) {
        CUDA_CHECK(cudaSetDevice(prev_dev));
    }
    {
        std::lock_guard<std::mutex> lk(mtx);
        handles[device_id] = handle;
    }
    return handle;
}

void convert_bf16_to_fp16_gpu(void* data, size_t num_elements, cudaStream_t stream = 0);

}
