#include "vm_c/official/ggml_backend_pool.hpp"

#include "vm_c/cuda/gpu_arch.hpp"
#include "vm_c/cuda/vm_c_tensor.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cublas_v2.h>
#include <mutex>
#include <unordered_map>

namespace vm_c::official {

namespace {

std::mutex& device_mutex(int gpu_device) {
    static std::mutex map_mu;
    static std::unordered_map<int, std::unique_ptr<std::mutex>> mu;
    std::lock_guard<std::mutex> lk(map_mu);
    auto& slot = mu[gpu_device];
    if (!slot) {
        slot = std::make_unique<std::mutex>();
    }
    return *slot;
}

struct DeviceGgml {
    ggml_backend_t backend = nullptr;
};

// 每个 device 全局唯一的 backend。注意：CUDA stream 属于创建时所在线程的
// primary context。如果 backend 在主线程初始化，stream 只能在主线程使用。
// Worker 线程调用 LlamaRuntime::decode() 时，ggml 内部会用这些 stream 做
// kernel launch，导致 "CUDA Stream does not belong to the expected context"。
//
// 修复方法：backend 统一延迟到 worker 线程首次使用时初始化，
// 确保 backend 的 stream 创建在 worker 线程自己的 CUDA context 中。
// 模型权重（device memory）在 GPU 上是跨线程共享的，不受影响。
std::unordered_map<int, DeviceGgml> g_devices;

DeviceGgml& device_entry(int gpu_device) {
    set_device_safe(gpu_device);
    auto& entry = g_devices[gpu_device];
    if (!entry.backend) {
        entry.backend = ggml_backend_cuda_init(gpu_device);
        if (!entry.backend) {
            throw std::runtime_error(
                "GgmlBackendPool: ggml_backend_cuda_init failed for device "
                + std::to_string(gpu_device));
        }
    }
    return entry;
}

}  // namespace

GgmlBackendPool& GgmlBackendPool::instance() {
    static GgmlBackendPool pool;
    return pool;
}

void* GgmlBackendPool::backend_for_device(int gpu_device) {
    std::lock_guard<std::mutex> lk(device_mutex(gpu_device));
    return device_entry(gpu_device).backend;
}

cudaStream_t ggml_compute_stream_for_device(int gpu_device) {
    std::lock_guard<std::mutex> lk(device_mutex(gpu_device));
    ggml_backend_t backend = device_entry(gpu_device).backend;
    return static_cast<cudaStream_t>(ggml_backend_cuda_get_stream(backend));
}

cudaStream_t ggml_bind_compute_stream(int gpu_device) {
    set_device_safe(gpu_device);
    cudaStream_t stream = ggml_compute_stream_for_device(gpu_device);
    CUBLAS_CHECK(cublasSetStream(get_cublas_handle(gpu_device), stream));
    return stream;
}

}  // namespace vm_c::official
