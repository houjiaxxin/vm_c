# vm_c 构建环境适配（硬件 / 工具链 / vendored ggml）
#
# 用法（CMakeLists.txt）：
#   include(VmCEnv)
#   vm_c_select_cuda_host_compiler()   # 必须在 project() 之前
#   vm_c_select_cuda_architectures()   # 必须在 project() 之前（nvidia-smi）
#   project(...)
#   vm_c_apply_cuda_architectures(${CMAKE_CUDA_COMPILER_VERSION})
#   ...
#   vm_c_setup_official_ggml()         # project() 之后

# ── 本机 GPU 架构（nvidia-smi）──────────────────────────────────────────────
option(VM_C_CUDA_ARCH_AUTO
    "Detect local GPU compute capability via nvidia-smi (recommended)" ON)

function(_vm_c_compute_cap_to_arch _cap _arch_out)
    string(STRIP "${_cap}" _cap)
    string(REPLACE "." "" _arch "${_cap}")
    if(_arch MATCHES "^[0-9]+$")
        set(${_arch_out} "${_arch}" PARENT_SCOPE)
    else()
        set(${_arch_out} "" PARENT_SCOPE)
    endif()
endfunction()

function(_vm_c_query_gpu_cuda_architectures _archs_out _ok_out)
    set(${_archs_out} "" PARENT_SCOPE)
    set(${_ok_out} FALSE PARENT_SCOPE)

    find_program(_VM_C_NVIDIA_SMI NAMES nvidia-smi)
    if(NOT _VM_C_NVIDIA_SMI)
        return()
    endif()

    execute_process(
        COMMAND ${_VM_C_NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE _caps
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE _err
        RESULT_VARIABLE _rc
    )
    if(NOT _rc EQUAL 0 OR _caps STREQUAL "")
        return()
    endif()

    string(REPLACE "\n" ";" _cap_list "${_caps}")
    set(_archs "")
    foreach(_cap IN LISTS _cap_list)
        _vm_c_compute_cap_to_arch("${_cap}" _arch)
        if(NOT _arch STREQUAL "")
            list(APPEND _archs "${_arch}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _archs)

    if(_archs STREQUAL "")
        return()
    endif()

    set(${_archs_out} "${_archs}" PARENT_SCOPE)
    set(${_ok_out} TRUE PARENT_SCOPE)
endfunction()

function(_vm_c_cuda_architectures_portable _cuda_version _archs_out)
    if(${_cuda_version} VERSION_GREATER_EQUAL 13.0)
        set(_archs 75 80 86 89 90 100 120)
    elseif(${_cuda_version} VERSION_GREATER_EQUAL 12.4)
        set(_archs 75 80 86 89 90)
    elseif(${_cuda_version} VERSION_GREATER_EQUAL 12.0)
        set(_archs 75 80 86 89)
    else()
        set(_archs 75)
    endif()
    set(${_archs_out} "${_archs}" PARENT_SCOPE)
endfunction()

function(vm_c_select_cuda_architectures)
    if(NOT VM_C_CUDA_ARCH_AUTO)
        return()
    endif()
    _vm_c_query_gpu_cuda_architectures(_archs _ok)
    if(_ok)
        set(CMAKE_CUDA_ARCHITECTURES "${_archs}" CACHE STRING
            "CUDA architectures (auto-detected from local GPU)" FORCE)
        message(STATUS "VmCEnv: local GPU -> CMAKE_CUDA_ARCHITECTURES=${_archs}")
    endif()
endfunction()

function(vm_c_apply_cuda_architectures _cuda_version)
    if(NOT VM_C_CUDA_ARCH_AUTO)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
            _vm_c_cuda_architectures_portable("${_cuda_version}" _archs)
            set(CMAKE_CUDA_ARCHITECTURES "${_archs}" CACHE STRING
                "CUDA architectures (portable multi-GPU build)" FORCE)
            message(STATUS "VmCEnv: portable CUDA archs: ${_archs}")
        else()
            message(STATUS "VmCEnv: manual CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")
        endif()
        return()
    endif()

    if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        message(STATUS "VmCEnv: using auto-detected CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")
        return()
    endif()

    message(WARNING
        "VmCEnv: nvidia-smi unavailable or no GPU; falling back to portable multi-arch build")
    _vm_c_cuda_architectures_portable("${_cuda_version}" _archs)
    set(CMAKE_CUDA_ARCHITECTURES "${_archs}" CACHE STRING
        "CUDA architectures (portable fallback)" FORCE)
    message(STATUS "VmCEnv: portable CUDA archs: ${_archs}")
endfunction()

# ── CUDA 宿主编译器（NVCC -ccbin）────────────────────────────────────────────
function(_vm_c_cuda_host_version_ok _cxx _ver_out _ok_out)
    execute_process(
        COMMAND ${_cxx} -dumpfullversion -dumpversion
        OUTPUT_VARIABLE _ver
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    set(${_ver_out} "${_ver}" PARENT_SCOPE)
    set(_ok FALSE)
    if(NOT _ver STREQUAL "")
        string(REGEX MATCH "^[0-9]+" _major "${_ver}")
        if(_cxx MATCHES "clang\\+\\+")
            if(_major VERSION_GREATER_EQUAL 10)
                set(_ok TRUE)
            endif()
        elseif(_major VERSION_GREATER_EQUAL 9)
            set(_ok TRUE)
        endif()
    endif()
    set(${_ok_out} ${_ok} PARENT_SCOPE)
endfunction()

function(vm_c_select_cuda_host_compiler)
    if(CMAKE_CUDA_HOST_COMPILER AND NOT CMAKE_CUDA_HOST_COMPILER STREQUAL "")
        _vm_c_cuda_host_version_ok("${CMAKE_CUDA_HOST_COMPILER}" _cur_ver _cur_ok)
        if(_cur_ok)
            message(STATUS "VmCEnv: CUDA host ${CMAKE_CUDA_HOST_COMPILER} (${_cur_ver})")
            return()
        endif()
        message(STATUS "VmCEnv: ${CMAKE_CUDA_HOST_COMPILER} (${_cur_ver}) too old, searching...")
    endif()

    foreach(_cand IN ITEMS
        g++-13 g++-12 g++-11 g++-10 g++-9
        clang++-18 clang++-17 clang++-16 clang++-15 clang++-14 clang++-13 clang++-12 clang++-11 clang++-10
        g++-8 g++
    )
        find_program(_cxx NAMES ${_cand})
        if(NOT _cxx)
            continue()
        endif()
        _vm_c_cuda_host_version_ok("${_cxx}" _ver _ok)
        if(_ok)
            set(CMAKE_CUDA_HOST_COMPILER "${_cxx}" CACHE FILEPATH "CUDA host C++ compiler for NVCC" FORCE)
            message(STATUS "VmCEnv: CUDA host ${CMAKE_CUDA_HOST_COMPILER} (${_ver})")
            return()
        endif()
    endforeach()

    message(STATUS "VmCEnv: no GCC>=9 / Clang>=10 for NVCC; using default host compiler")
endfunction()

# ── vendored llama ggml-cuda（IQ4 / MoE）────────────────────────────────────
option(VM_C_OFFICIAL_GGML "Build vendored llama ggml-cuda for IQ4 MoE / quant matmul" ON)
option(VM_C_CUDA_NCCL "Enable NCCL in ggml-cuda for multi-GPU TP AllReduce" ON)

macro(vm_c_setup_official_ggml)
    if(VM_C_OFFICIAL_GGML)
    set(_VM_C_GGML_ROOT "${CMAKE_SOURCE_DIR}/third_party/vm_c_official/ggml")
    if(NOT EXISTS "${_VM_C_GGML_ROOT}/CMakeLists.txt")
        message(FATAL_ERROR
            "third_party/vm_c_official/ggml missing. Run: bash scripts/vendor_official.sh")
    endif()

    set(GGML_STANDALONE OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(GGML_STATIC ON CACHE BOOL "" FORCE)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
    if(VM_C_CUDA_NCCL)
        set(_VM_C_MAX_ARCH 0)
        foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            string(REGEX REPLACE "[^0-9]" "" _num "${_arch}")
            if(_num GREATER _VM_C_MAX_ARCH)
                set(_VM_C_MAX_ARCH "${_num}")
            endif()
        endforeach()
        if(_VM_C_MAX_ARCH LESS 80)
            message(WARNING
                "VmCEnv: GPU SM${_VM_C_MAX_ARCH} < SM80, NCCL FP8 kernel may fail. "
                "Set NCCL_NVLS_ENABLE=0 at runtime to avoid FP8 probing.")
        endif()
        list(APPEND CMAKE_MODULE_PATH "${_VM_C_GGML_ROOT}/cmake")
        find_package(NCCL QUIET)
        if(NCCL_FOUND)
            set(GGML_CUDA_NCCL ON CACHE BOOL "ggml: use NCCL for multi-GPU" FORCE)
            message(STATUS "VmCEnv: NCCL found (${NCCL_LIBRARY}), GGML_CUDA_NCCL=ON")
        else()
            message(WARNING
                "VmCEnv: VM_C_CUDA_NCCL=ON but NCCL not found "
                "(install libnccl-dev or set NCCL_ROOT)")
            set(GGML_CUDA_NCCL OFF CACHE BOOL "" FORCE)
        endif()
    else()
        set(GGML_CUDA_NCCL OFF CACHE BOOL "" FORCE)
        message(STATUS "VmCEnv: GGML_CUDA_NCCL=OFF (VM_C_CUDA_NCCL=OFF)")
    endif()
    set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)

    # vm_c 仅用 ggml-cuda；ggml-cpu 仅为链接依赖。旧 GCC 无法编 AVX2 repack / charconv。
    set(GGML_CPU_REPACK OFF CACHE BOOL "" FORCE)
    set(GGML_AVX512 OFF CACHE BOOL "" FORCE)
    set(GGML_AVX512_VBMI OFF CACHE BOOL "" FORCE)
    set(GGML_AVX512_VNNI OFF CACHE BOOL "" FORCE)
    set(GGML_AVX512_BF16 OFF CACHE BOOL "" FORCE)
    set(GGML_AVX2 OFF CACHE BOOL "" FORCE)
    set(GGML_AVX OFF CACHE BOOL "" FORCE)
    set(GGML_FMA OFF CACHE BOOL "" FORCE)
    set(GGML_F16C OFF CACHE BOOL "" FORCE)
    set(GGML_BMI2 OFF CACHE BOOL "" FORCE)
    set(GGML_SSE42 ON CACHE BOOL "" FORCE)

    set(_VM_C_GGML_VMC "${CMAKE_SOURCE_DIR}/third_party/vm_c_official/ggml_vm_c")
    if(NOT EXISTS "${_VM_C_GGML_VMC}/src/ggml_vm_c_graph.c")
        message(FATAL_ERROR "third_party/vm_c_official/ggml_vm_c missing (vm_c repo)")
    endif()
    execute_process(
        COMMAND bash "${CMAKE_SOURCE_DIR}/scripts/apply_ggml_vm_c.sh"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        RESULT_VARIABLE _VM_C_GGML_APPLY_RC
    )
    if(NOT _VM_C_GGML_APPLY_RC EQUAL 0)
        message(FATAL_ERROR "apply_ggml_vm_c.sh failed (rc=${_VM_C_GGML_APPLY_RC})")
    endif()

    add_subdirectory("${_VM_C_GGML_ROOT}" "${CMAKE_BINARY_DIR}/vm_c_official/ggml" EXCLUDE_FROM_ALL)

    target_sources(ggml PRIVATE "${_VM_C_GGML_VMC}/src/ggml_vm_c_graph.c")
    target_include_directories(ggml PUBLIC "${_VM_C_GGML_VMC}/include")

    if(TARGET ggml-cuda)
        target_sources(ggml-cuda PRIVATE "${_VM_C_GGML_VMC}/src/ggml_vm_c_cuda.cu")
        target_include_directories(ggml-cuda PRIVATE "${_VM_C_GGML_VMC}/include")
    endif()

    if(TARGET ggml-cuda)
        set(_host "${CMAKE_CUDA_HOST_COMPILER}")
        if(_host STREQUAL "")
            set(_host "${CMAKE_CXX_COMPILER}")
        endif()
        if(_host)
            execute_process(
                COMMAND ${_host} -dumpfullversion -dumpversion
                OUTPUT_VARIABLE _ver
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            string(REGEX MATCH "^[0-9]+" _major "${_ver}")
            set(_old FALSE)
            if(_host MATCHES "clang\\+\\+")
                if(_major VERSION_LESS 10)
                    set(_old TRUE)
                endif()
            elseif(_major VERSION_LESS 9)
                set(_old TRUE)
            endif()
            if(_old)
                message(WARNING
                    "VmCEnv: ggml-cuda host ${_host} (${_ver}) may ICE; using -Xcompiler=-O1")
                target_compile_options(ggml-cuda PRIVATE
                    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-O1>")
            endif()
        endif()
    endif()

    add_library(vm_c_official STATIC
        ${CMAKE_SOURCE_DIR}/src/official/ggml_backend_pool.cpp
        ${CMAKE_SOURCE_DIR}/src/official/ggml_weight.cpp
    )
    target_include_directories(vm_c_official PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${_VM_C_GGML_ROOT}/include
    )
    target_link_libraries(vm_c_official PUBLIC
        ggml
        CUDA::cudart
        CUDA::cublas
        spdlog::spdlog
    )
    target_compile_definitions(vm_c_official PUBLIC VM_C_USE_OFFICIAL_GGML_MOE=1)
    set_target_properties(vm_c_official PROPERTIES POSITION_INDEPENDENT_CODE ON)

    message(STATUS "VmCEnv: ggml-cuda from third_party/vm_c_official/ggml")
    else()
        message(STATUS "VmCEnv: ggml disabled (VM_C_OFFICIAL_GGML=OFF)")
    endif()
endmacro()
