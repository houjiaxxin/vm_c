#include "vmc-tp-allreduce.h"
#include "vmc-tp-allreduce.cuh"
#include "common.cuh"

#include "ggml-cuda.h"

#include <atomic>

namespace {
std::atomic<ggml_cuda_vm_c_tp_allreduce_fn> g_vm_c_tp_allreduce{nullptr};
}  // namespace

void ggml_cuda_register_vm_c_tp_allreduce(ggml_cuda_vm_c_tp_allreduce_fn fn) {
    g_vm_c_tp_allreduce.store(fn, std::memory_order_release);
}

bool ggml_cuda_vm_c_tp_allreduce_registered(void) {
    return g_vm_c_tp_allreduce.load(std::memory_order_acquire) != nullptr;
}

void ggml_cuda_op_vm_c_tp_allreduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto fn = g_vm_c_tp_allreduce.load(std::memory_order_acquire);
    if (!fn) {
        GGML_ABORT("ggml_cuda_op_vm_c_tp_allreduce: vm_c kernel not registered");
    }

    ggml_cuda_set_device(ctx.device);

    cudaStream_t stream = ctx.stream(ctx.device, ctx.curr_stream_no);
    fn(dst, (void *)stream);
}
