#pragma once

#include "vm_c_dispatch.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <cstdio>
#include <stdexcept>

namespace vm_c {

struct TensorDesc {
    void* data = nullptr;
    ScalarType dtype = ScalarType::FLOAT32;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    int device_id = 0;

    TensorDesc() = default;

    TensorDesc(void* data, ScalarType dtype, std::vector<int64_t> shape,
               std::vector<int64_t> strides = {}, int device_id = 0)
        : data(data), dtype(dtype), shape(std::move(shape)),
          strides(std::move(strides)), device_id(device_id) {
        if (this->strides.empty()) {
            this->strides.resize(this->shape.size());
            int64_t s = 1;
            for (int i = static_cast<int>(this->shape.size()) - 1; i >= 0; --i) {
                this->strides[i] = s;
                s *= this->shape[i];
            }
        }
    }

    int64_t ndim() const { return static_cast<int64_t>(shape.size()); }
    int64_t size(int64_t dim) const { return shape[dim]; }
    int64_t stride(int64_t dim) const { return strides[dim]; }
    int64_t numel() const {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
    bool is_contiguous() const {
        int64_t expected = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            if (strides[i] != expected) return false;
            expected *= shape[i];
        }
        return true;
    }
    size_t bytes() const {
        return static_cast<size_t>(numel()) * dtype_size(static_cast<DataType>(dtype));
    }

    template <typename T>
    T* data_ptr() { return static_cast<T*>(data); }

    template <typename T>
    const T* data_ptr() const { return static_cast<const T*>(data); }
};

struct CudaStream {
    cudaStream_t stream = nullptr;
    int device_id = 0;

    CudaStream() = default;
    explicit CudaStream(cudaStream_t s, int dev = 0) : stream(s), device_id(dev) {}

    void synchronize() const {
        cudaSetDevice(device_id);
        cudaStreamSynchronize(stream);
    }

    operator cudaStream_t() const { return stream; }
};

class CudaDeviceGuard {
public:
    explicit CudaDeviceGuard(int device_id) : prev_device_(-1) {
        // cudaGetDevice 在新线程（无 CUDA context）上会打印驱动警告：
        //   "No CUDA context is current to the calling thread"
        //   然后才返回 cudaErrorInvalidContext。
        // 修复：先用 cuCtxGetCurrent 查 context 是否存在（无警告），
        //       存在才用 cudaGetDevice 保存前一个 device 以便析构时恢复。
        CUcontext cur_ctx = nullptr;
        CUresult res = cuCtxGetCurrent(&cur_ctx);
        if (res == CUDA_SUCCESS && cur_ctx != nullptr) {
            // 已有 context，保存当前 device 用于析构恢复
            CUDA_CHECK(cudaGetDevice(&prev_device_));
        }
        // 无 context 时 prev_device_ 保持 -1（析构不恢复）
        if (device_id != prev_device_) {
            CUDA_CHECK(cudaSetDevice(device_id));
        }
    }
    ~CudaDeviceGuard() {
        if (prev_device_ >= 0) {
            CUDA_CHECK(cudaSetDevice(prev_device_));
        }
    }
    CudaDeviceGuard(const CudaDeviceGuard&) = delete;
    CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;
private:
    int prev_device_;
};

inline int get_cuda_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

inline size_t get_cuda_device_memory(int device_id) {
    cudaSetDevice(device_id);
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    return total_mem;
}

inline size_t get_cuda_free_memory(int device_id) {
    cudaSetDevice(device_id);
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

inline int get_cuda_sm_count(int device_id) {
    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    return prop.multiProcessorCount;
}

inline int get_cuda_shared_memory_per_block(int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    return prop.sharedMemPerBlock;
}

inline void* cuda_alloc(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        fprintf(stderr, "cudaMalloc failed: %s (requested %zu bytes = %.1f MB, free = %.1f MB)\n",
                cudaGetErrorString(err), bytes, bytes / (1024.0 * 1024.0),
                free_mem / (1024.0 * 1024.0));
        throw std::runtime_error("CUDA memory allocation failed: " + std::string(cudaGetErrorString(err)));
    }
    return ptr;
}

}
