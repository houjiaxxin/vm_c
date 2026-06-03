#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ggml_cuda_vm_c_turboquant_attn_fn)(struct ggml_tensor * dst);

void ggml_cuda_register_vm_c_turboquant_attn(ggml_cuda_vm_c_turboquant_attn_fn fn);

bool ggml_cuda_vm_c_turboquant_attn_registered(void);

#ifdef __cplusplus
}
#endif
