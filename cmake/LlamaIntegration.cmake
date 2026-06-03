# libllama 核心推理引擎（从 llama.cpp 融合到项目中）
# 前置：vm_c_setup_official_ggml() 已 add_subdirectory ggml 并导出 TARGET ggml
#
macro(vm_c_setup_llama_graph)
    if(NOT TARGET ggml)
        message(FATAL_ERROR "vm_c_setup_llama_graph: ggml target must exist (call vm_c_setup_official_ggml first)")
    endif()

    set(_VM_C_LLAMA_ROOT "${CMAKE_SOURCE_DIR}/src/official/llama")
    if(NOT EXISTS "${_VM_C_LLAMA_ROOT}/CMakeLists.txt")
        message(FATAL_ERROR "vm_c_setup_llama_graph: ${_VM_C_LLAMA_ROOT} missing (fused llama core not found)")
    endif()

    add_subdirectory("${_VM_C_LLAMA_ROOT}" "${CMAKE_BINARY_DIR}/vm_c_llama" EXCLUDE_FROM_ALL)

    if(NOT TARGET vm_c_llama)
        message(FATAL_ERROR "vm_c_setup_llama_graph: vm_c_llama target was not created")
    endif()

    message(STATUS "VmC: libllama core fused into project (src/official/llama, VM_C_LLAMA_INTEGRATION=1)")
endmacro()
