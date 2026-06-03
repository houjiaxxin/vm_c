#include "vmc-turboquant-attn.h"
#include "vmc-turboquant-attn.cuh"

#include "ggml-cuda.h"

#include <atomic>
#include <cstring>

namespace {
std::atomic<ggml_cuda_vm_c_turboquant_attn_fn> g_vm_c_turboquant_attn{nullptr};
}  // namespace

void ggml_cuda_register_vm_c_turboquant_attn(ggml_cuda_vm_c_turboquant_attn_fn fn) {
    g_vm_c_turboquant_attn.store(fn, std::memory_order_release);
}

bool ggml_cuda_vm_c_turboquant_attn_registered(void) {
    return g_vm_c_turboquant_attn.load(std::memory_order_acquire) != nullptr;
}

void ggml_cuda_op_vm_c_turboquant_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    (void) ctx;
    auto fn = g_vm_c_turboquant_attn.load(std::memory_order_acquire);
    if (!fn) {
        GGML_ABORT("ggml_cuda_op_vm_c_turboquant_attn: vm_c kernel not registered");
    }
    fn(dst);
}
