#pragma once

#include <cuda_runtime.h>

namespace vm_c::official {

/// 进程级 ggml-cuda backend 池（TP allreduce / llama decode 共用 compute stream）。
class GgmlBackendPool {
public:
    static GgmlBackendPool& instance();
    void* backend_for_device(int gpu_device);
};

cudaStream_t ggml_compute_stream_for_device(int gpu_device);
cudaStream_t ggml_bind_compute_stream(int gpu_device);

}  // namespace vm_c::official
