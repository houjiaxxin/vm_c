#pragma once

// vm_c 诊断工具：抓取图构建中指定的 tensor 在图执行后的前 N 个 float 值。
//
// 用法：
//   1. 在模型 graph build 代码（qwen35moe.cpp 等）的关键 cb() 附近调用
//      vm_c::hidden_log_register(name, tensor);
//   2. llama_context::process_ubatch() 在 graph_compute() 之后会自动调用
//      vm_c::hidden_log_dump_all(sched) 把所有已注册的 tensor 的前 8 个 float 值
//      打印到 [HIDDEN-LOG] 日志。
//
// 通过环境变量 VMC_LOG_HIDDEN=1 开启（默认关闭，是 no-op）。

#include "ggml.h"
#include "ggml-backend.h"   // ggml_backend_sched_t 在这里

namespace vm_c {

// 查询是否启用（受 VMC_LOG_HIDDEN=1 控制，进程级缓存）。
bool hidden_log_enabled();

// 注册一个要记录的 tensor（name 用于日志，tensor 是 graph 中的节点）。
// 不开启时是 no-op；tensor/name 为空也跳过。
void hidden_log_register(const char* name, ggml_tensor * tensor);

// 清空已注册的列表（在 process_ubatch graph_rebuild 分支开头调用，
// 防止 reuse 路径上累积旧条目）。
void hidden_log_clear();

// 把所有已注册 tensor 的前 N 个 float 值 dump 到日志（在 graph_compute 之后调用）。
// dump 后不自动清空（reuse 路径还需要再次 dump）。
void hidden_log_dump_all(ggml_backend_sched_t sched);

} // namespace vm_c
