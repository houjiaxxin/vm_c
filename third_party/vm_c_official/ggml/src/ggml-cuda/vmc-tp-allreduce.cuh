#pragma once

#include "common.cuh"
#include "vmc-tp-allreduce.h"

void ggml_cuda_op_vm_c_tp_allreduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
