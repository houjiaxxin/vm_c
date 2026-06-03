#include "ggml-cuda/common.cuh"
#include "ggml-backend-impl.h"

// vm_c custom op 的 register/op 函数已拆分到独立文件：
//   vmc-turboquant-attn.cu  → register / op / registered
//   vmc-tp-allreduce.cu     → register / op / registered
// 此文件仅保留无法拆分的辅助函数。

extern "C" void * ggml_backend_cuda_get_stream(ggml_backend_t backend) {
    GGML_ASSERT(backend != nullptr);
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    return (void *) cuda_ctx->stream();
}
