#pragma once

// NcclComm — ggml 图内 allreduce（NCCL backbone + registered callback）
//
// 设计：
//   1. init() 时注册 allreduce 回调到 ggml-cuda backend
//   2. 图构建中插入 GGML_OP_VM_C_TP_ALLREDUCE 节点
//   3. CUDA backend 计算该节点时调用注册的回调，传入 dst tensor 和 stream
//   4. 回调中执行 ncclAllReduce
//
// NCCL 是 Tensor Parallel 的硬性依赖，初始化失败直接终止进程。

#include "ggml.h"

#include <cstdint>
#include <vector>

namespace vm_c::tp {

class NcclComm {
public:
    NcclComm() = default;
    ~NcclComm();

    NcclComm(const NcclComm &) = delete;
    NcclComm & operator=(const NcclComm &) = delete;

    bool init(const std::vector<int> & devices);

    bool allreduce(void * data, int64_t ne, ggml_type type,
                   void * stream, int rank);

    void destroy();

    int world_size() const { return world_size_; }
    bool use_nccl() const { return use_nccl_; }
    const std::vector<int> & devices() const;

private:
    int world_size_ = 0;
    std::vector<int> devices_;
    bool use_nccl_ = false;

    std::vector<void *> nccl_comms_;
};

NcclComm * get_global_nccl_comm();

} // namespace vm_c::tp
