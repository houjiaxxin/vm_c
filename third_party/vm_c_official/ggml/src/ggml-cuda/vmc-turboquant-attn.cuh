#pragma once

#include "common.cuh"
#include "vmc-turboquant-attn.h"

void ggml_cuda_op_vm_c_turboquant_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
