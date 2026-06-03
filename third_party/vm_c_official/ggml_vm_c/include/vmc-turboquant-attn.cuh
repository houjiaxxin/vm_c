#pragma once

#include "ggml-cuda/common.cuh"
#include "ggml.h"

void ggml_cuda_op_vm_c_turboquant_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#ifdef __cplusplus
extern "C" {
#endif
bool ggml_cuda_vm_c_turboquant_attn_registered(void);
#ifdef __cplusplus
}
#endif
