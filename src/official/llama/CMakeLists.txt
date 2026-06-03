# vm_c 内置 llama 推理核心（从 llama.cpp 融合精选）
# 只包含推理核心所需的底层能力文件，独立上层模块已移除
cmake_minimum_required(VERSION 3.14)

if(NOT TARGET ggml)
    message(FATAL_ERROR "vm_c/official/llama: ggml target must exist")
endif()

set(_LLAMA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

# 显式列出模型架构实现（从 D:\vm_c\llama.cpp\src\models\ 拷贝对应文件）
# 当前支持: llama, qwen35moe
set(LLAMA_MODELS_SOURCES
    ${_LLAMA_ROOT}/models/llama.cpp
    ${_LLAMA_ROOT}/models/qwen35moe.cpp
    ${_LLAMA_ROOT}/models/delta-net-base.cpp
)

add_library(vm_c_llama STATIC
    # 核心推理循环
    ${_LLAMA_ROOT}/llama.cpp
    # vm_c 诊断工具：hidden state 探针（VMC_LOG_HIDDEN=1 开启）
    ${_LLAMA_ROOT}/vm_c_hidden_log.cpp
    # 图构建（需修改插入 allreduce）
    ${_LLAMA_ROOT}/llama-graph.cpp
    # 模型定义与加载（需修改支持分片权重）
    ${_LLAMA_ROOT}/llama-model.cpp
    ${_LLAMA_ROOT}/llama-model-loader.cpp
    # 架构/参数/词表/IO
    ${_LLAMA_ROOT}/llama-arch.cpp
    ${_LLAMA_ROOT}/llama-batch.cpp
    ${_LLAMA_ROOT}/llama-context.cpp
    ${_LLAMA_ROOT}/llama-cparams.cpp
    ${_LLAMA_ROOT}/llama-hparams.cpp
    ${_LLAMA_ROOT}/llama-impl.cpp
    ${_LLAMA_ROOT}/llama-io.cpp
    ${_LLAMA_ROOT}/llama-vocab.cpp
    # KV cache
    ${_LLAMA_ROOT}/llama-kv-cache.cpp
    ${_LLAMA_ROOT}/llama-kv-cache-iswa.cpp
    # 记忆类型（底层接口 + 具体实现）
    ${_LLAMA_ROOT}/llama-memory.cpp
    ${_LLAMA_ROOT}/llama-memory-hybrid.cpp
    ${_LLAMA_ROOT}/llama-memory-hybrid-iswa.cpp
    ${_LLAMA_ROOT}/llama-memory-recurrent.cpp
    # 文件映射
    ${_LLAMA_ROOT}/llama-mmap.cpp
    # 适配器（图构建深度集成）
    ${_LLAMA_ROOT}/llama-adapter.cpp
    # Unicode（分词基础）
    ${_LLAMA_ROOT}/unicode.cpp
    ${_LLAMA_ROOT}/unicode-data.cpp
    # Jinja 模板引擎（vm_c::JinjaTemplate 包装使用）
    ${_LLAMA_ROOT}/jinja/lexer.cpp
    ${_LLAMA_ROOT}/jinja/parser.cpp
    ${_LLAMA_ROOT}/jinja/runtime.cpp
    ${_LLAMA_ROOT}/jinja/value.cpp
    ${_LLAMA_ROOT}/jinja/string.cpp
    ${_LLAMA_ROOT}/jinja/caps.cpp
    # 模型架构实现
    ${LLAMA_MODELS_SOURCES}
)

target_include_directories(vm_c_llama
    PUBLIC
        "${_LLAMA_ROOT}"
        "${_LLAMA_ROOT}/jinja"   # 让 jinja/{runtime.h,value.h,...} 可被 #include "runtime.h" 找到
    PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
)
target_compile_features(vm_c_llama PRIVATE cxx_std_17)
target_link_libraries(vm_c_llama
    PUBLIC
        ggml
        nlohmann_json::nlohmann_json   # jinja/{value,caps}.cpp 用 <nlohmann/json.hpp>
)
target_compile_definitions(vm_c_llama PRIVATE VM_C_LLAMA_INTEGRATION=1)
set_target_properties(vm_c_llama PROPERTIES POSITION_INDEPENDENT_CODE ON)

message(STATUS "VmC: llama core fused into project (src/official/llama)")
