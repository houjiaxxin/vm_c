#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// 注册的回调接收 dst (allreduce 目标 tensor) 和 stream (当前 CUDA stream)
// stream 以 void* 传递以避免在 CPU 头文件中引入 CUDA runtime 依赖
typedef void (*ggml_cuda_vm_c_tp_allreduce_fn)(struct ggml_tensor * dst, void * stream);

// 注册 allreduce 回调（由 vm_c 的 NcclComm 在启动时调用）
void ggml_cuda_register_vm_c_tp_allreduce(ggml_cuda_vm_c_tp_allreduce_fn fn);

// 查询是否已注册回调
bool ggml_cuda_vm_c_tp_allreduce_registered(void);

#ifdef __cplusplus
}
#endif
