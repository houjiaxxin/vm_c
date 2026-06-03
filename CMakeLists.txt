cmake_minimum_required(VERSION 3.14...3.28) # ggml 需要 upper bound 3.28

set(CMAKE_POLICY_VERSION_MINIMUM 3.14 CACHE STRING "" FORCE)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL "" FORCE)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(VmCEnv)
vm_c_select_cuda_host_compiler()
vm_c_select_cuda_architectures()

project(vm_c_server VERSION 0.1.0 LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(WARNING
        "VmC: Debug build (-O0 host, unoptimized CUDA) is unsuitable for inference; "
        "reconfigure with -DCMAKE_BUILD_TYPE=Release")
endif()

# C++17 required for CUDA kernels and model code
# 注：代码中未使用 C++20 特有语法，GCC 7+ 即可编译
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=128")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=177")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=555")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
# CUDA 13.1 CCCL (CUB/Thrust) 需要 C++17，但 nvcc host compiler 检测为 GCC 7.3.0 误报
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCCCL_IGNORE_DEPRECATED_CPP_DIALECT")

# 勿用 -G：device debug 会使 IQ4/MM 等 kernel 慢一个数量级以上
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# ── GCC 7.3.0 (libstdc++) 头文件传递性 include 不足的兼容层 ───────────────
# 该版本 std::strchr 第一个参数是 char* 而非 const char*，导致
# std::regex 内部代码报编译错误。GCC 9+ 已修复，故仅在 GCC < 9 时启用。
# 注：早期版本尝试 -include cstring 强制预编译，但 cstring 头在 GCC 7.3.0
# 依赖复杂预处理保护宏（__CORRECT_ISO_CPP_STRING_H_PROTO），强制 include 会
# 让保护机制失效并触发 strchr/strpbrk/strrchr/strstr 重定义。改为只用 -fpermissive。
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fpermissive")
    message(STATUS "VmC: GCC < 9 detected, applying -fpermissive workaround")
endif()

set(CUDA_MIN_VERSION 12.0)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS ${CUDA_MIN_VERSION})
    message(FATAL_ERROR "CUDA >= ${CUDA_MIN_VERSION} required, found ${CMAKE_CUDA_COMPILER_VERSION}")
endif()

vm_c_apply_cuda_architectures(${CMAKE_CUDA_COMPILER_VERSION})

if(NCCL_FOUND)
    message(STATUS "AllReduce: NCCL (required for tensor parallel)")
else()
    message(STATUS "AllReduce: NCCL not found (will be required by vm_c_distributed)")
endif()

option(VM_C_ENABLE_HTTP "Enable HTTP server" ON)
option(VM_C_ENABLE_TURBOQUANT "Enable TurboQuant KV cache compression" ON)
option(VM_C_ENABLE_UVA "Enable UVA (Unified Virtual Addressing)" ON)
option(VM_C_ENABLE_OFFLOAD "Enable CPU offload" ON)
option(VM_C_STATIC_LINK "Static link libgcc/libstdc++ to reduce glibc dependency" ON)

set(VM_C_THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/third_party")

include(FetchContent)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# ggml_runner 使用 spdlog 卡点日志，须在 vm_c_official 之前可用
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.15.0
)
FetchContent_MakeAvailable(spdlog)

vm_c_setup_official_ggml()
include(LlamaIntegration)
vm_c_setup_llama_graph()

# 将 llama 相关桥接文件加入 vm_c_official 并链接融合的 vm_c_llama
if(TARGET vm_c_official AND TARGET vm_c_llama)
    target_sources(vm_c_official PRIVATE
        ${CMAKE_SOURCE_DIR}/src/official/llama_runtime.cpp
        ${CMAKE_SOURCE_DIR}/src/official/llama_batch_builder.cpp
        ${CMAKE_SOURCE_DIR}/src/official/turboquant_kv_bridge.cpp
    )
    target_include_directories(vm_c_official PRIVATE
        "${CMAKE_SOURCE_DIR}/src/official/llama"
    )
    target_link_libraries(vm_c_official PUBLIC vm_c_llama)
    # VM_C_LLAMA_INTEGRATION 必须在 vm_c_official 及其所有消费者中都定义，
    # 以保证 llama.h 中的 llama_context_params 布局一致（否则 samplers 偏移量错位）。
    target_compile_definitions(vm_c_official PUBLIC VM_C_LLAMA_INTEGRATION=1)
endif()

# 所有编译单元都需要 VM_C_USE_LLAMA_GRAPH（libllama 官方前向路径）
add_compile_definitions(VM_C_USE_LLAMA_GRAPH=1)

# PyTorch 已完全移除 — 全部 CUDA 内核为 pure CUDA

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
    cpp-httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.18.3
)
FetchContent_MakeAvailable(cpp-httplib)

# ── Jinja 模板引擎说明 ───────────────────────────────────────────────────
# 原依赖 Jinja2Cpp + Boost 因不支持 namespace()/is not/[:-1]/.split() 等 Qwen3
# 模板特性，被迫自带降级模板（含乱码字符）。现改用 llama.cpp 自带的
# common/jinja/ 引擎（已随 vm_c_llama 一并编译），完整支持所有 Jinja 特性。
# 参见 src/tokenizer/jinja_template.cpp 包装层 + src/official/llama/jinja/。

# ── vm_c_core (无变更) ───────────────────────────────────────────────────
add_library(vm_c_core STATIC
    src/core/engine.cpp
    src/core/logits_sample_plan.cpp
    src/core/scheduler.cpp
    src/core/request.cpp
    src/core/config.cpp
    src/core/tensor.cpp
    src/core/gguf_reader.cpp
    src/core/ggml_dequant.cpp
)
target_include_directories(vm_c_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(vm_c_core PUBLIC
    Threads::Threads
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    CUDA::cudart
    CUDA::cublas
    CUDA::cuda_driver
)
if(TARGET vm_c_official)
    target_link_libraries(vm_c_core PUBLIC vm_c_official)
endif()
if(TARGET vm_c_llama)
    target_link_libraries(vm_c_core PUBLIC vm_c_llama)
endif()

# ── vm_c_cuda (vm_c 原生纯 CUDA 内核 + 桥接层) ───────────────────────────
# 所有内核均为纯 CUDA 实现 — 无 PyTorch 依赖
# 包含: attention, norm, MoE, cache, sampler, quant, 自定义桥接

add_library(vm_c_cuda STATIC
    # 原生 CUDA 内核
    src/cuda/attention_kernels.cu
    src/cuda/cache_kernels.cu
    src/cuda/convert_kernels.cu            # GGUF 量化→FP16/BF16 GPU 反量化转换
    src/cuda/fused_attn_kernels.cu
    src/cuda/flash_attn_v2_kernels.cu
    src/cuda/gdn_kernels.cu
    src/cuda/moe_kernels.cu
    src/cuda/norm_kernels.cu
    src/cuda/pos_encoding_kernels.cu
    src/cuda/quant_kernels.cu
    src/cuda/sampler_kernels.cu
    src/cuda/turboquant_kernels.cu
    # 基础设施
    src/cuda/kernels.cpp
    src/cuda/vllm_kernel_ops.cpp          # Tensor → CUDA kernel 桥接
)
target_include_directories(vm_c_cuda PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(vm_c_cuda PUBLIC
    vm_c_core
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)
set_target_properties(vm_c_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# ── vm_c_model ──────────────────────────────────────────────────────────
add_library(vm_c_model STATIC
    src/model/model_loader.cpp
    src/model/gguf_model_loader.cpp
    src/model/models.cu
    src/model/forward_context.cpp
    src/model/attention_backend.cpp
    src/model/weight_layout_plan.cpp
    src/model/llama_gguf_tensor_map.cpp
)
target_include_directories(vm_c_model PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(vm_c_model PUBLIC
    vm_c_core
    vm_c_cuda
    CUDA::cudart
    CUDA::cublas
)
if(TARGET vm_c_official)
    target_link_libraries(vm_c_model PUBLIC vm_c_official)
endif()
set_target_properties(vm_c_model PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(vm_c_model PRIVATE -fopenmp)

# ── vm_c_memory ─────────────────────────────────────────────────────────
add_library(vm_c_memory STATIC
    src/memory/kv_cache_manager.cpp
    src/memory/uva_offload.cpp
    src/memory/allocator.cpp
    src/memory/expert_cache.cpp
)
target_include_directories(vm_c_memory PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(vm_c_memory PUBLIC
    vm_c_core
    vm_c_cuda
    CUDA::cudart
    spdlog::spdlog
)

# ── vm_c_tokenizer ──────────────────────────────────────────────────────
add_library(vm_c_tokenizer STATIC
    src/tokenizer/tokenizer.cpp
    src/tokenizer/gguf_tokenizer.cpp
    src/tokenizer/incremental_detokenizer.cpp
    src/tokenizer/jinja_template.cpp
)
target_compile_features(vm_c_tokenizer PUBLIC cxx_std_17)
target_include_directories(vm_c_tokenizer PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(vm_c_tokenizer PUBLIC
    vm_c_core
    vm_c_llama
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

# ── vm_c_distributed ────────────────────────────────────────────────────
# TP AllReduce：ggml 图内 allreduce 节点（NCCL backbone）

# NCCL 是 Tensor Parallel 的硬性依赖，必须在编译时可用
find_package(NCCL REQUIRED)
if(NOT NCCL_FOUND)
    message(FATAL_ERROR
        "NCCL is required for tensor parallel. "
        "Install libnccl-dev or set NCCL_ROOT.")
endif()
message(STATUS "VmC: NCCL found (${NCCL_LIBRARY}), USE_NCCL=ON")

add_library(vm_c_distributed STATIC
    src/distributed/tp_runtime.cpp
    src/distributed/nccl_comm.cu
)
target_include_directories(vm_c_distributed PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NCCL_INCLUDE_DIRS}
)
target_include_directories(vm_c_distributed PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/vm_c_official/ggml/src
    ${CMAKE_SOURCE_DIR}/third_party/vm_c_official/ggml/src/ggml-cuda
)
target_link_libraries(vm_c_distributed PUBLIC
    vm_c_core
    vm_c_cuda
    CUDA::cudart
)
if(TARGET vm_c_official)
    target_link_libraries(vm_c_distributed PUBLIC vm_c_official)
endif()
target_compile_definitions(vm_c_distributed PRIVATE USE_NCCL)
target_link_libraries(vm_c_distributed PRIVATE ${NCCL_LIBRARIES})
set_target_properties(vm_c_distributed PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# ── vm_c_server_lib (optional) ─────────────────────────────────────────
if(VM_C_ENABLE_HTTP)
    add_library(vm_c_server_lib STATIC
        src/server/http_server.cpp
        src/server/openai_api.cpp
    )
    target_include_directories(vm_c_server_lib PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    )
    target_link_libraries(vm_c_server_lib PUBLIC
        vm_c_core
        vm_c_tokenizer
        httplib::httplib
        nlohmann_json::nlohmann_json
    )
    target_compile_definitions(vm_c_server_lib PUBLIC VM_C_ENABLE_HTTP)
endif()

# ── vm_c_speculative (推测解码) ─────────────────────────────────────────
add_library(vm_c_speculative STATIC
    src/speculative/mtp_engine.cpp
)
target_include_directories(vm_c_speculative PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(vm_c_speculative PUBLIC
    vm_c_core
    vm_c_model
    vm_c_cuda
    CUDA::cudart
    CUDA::cublas
)
set_target_properties(vm_c_speculative PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# ── vm_c_server (可执行文件) ────────────────────────────────────────────
add_executable(vm_c_server
    src/main.cpp
)
target_link_libraries(vm_c_server PRIVATE
    vm_c_core
    vm_c_model
    vm_c_cuda
    vm_c_memory
    vm_c_tokenizer
    vm_c_server_lib
    vm_c_distributed
    vm_c_speculative
    -Wl,-Bdynamic
    pthread
    dl
)
target_compile_options(vm_c_server PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2>
)

if(VM_C_STATIC_LINK AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(vm_c_server PRIVATE -static-libgcc -static-libstdc++)
    target_link_libraries(vm_c_server PRIVATE -Wl,--as-needed)
endif()

if(VM_C_STATIC_LINK AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(vm_c_server PROPERTIES
        LINK_FLAGS "-Wl,--no-copy-dt-needed-entries"
    )
endif()
