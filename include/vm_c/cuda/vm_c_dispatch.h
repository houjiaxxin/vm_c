#pragma once

#include "vm_c/cuda/gpu_arch.hpp"

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <cstdio>
#include <cstdlib>

#ifndef USE_ROCM
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
  #include <cuda_fp8.h>
#else
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>
  using __nv_bfloat16 = __hip_bfloat16;
  using __nv_bfloat162 = __hip_bfloat162;
  using __half = _Float16;
#endif

namespace vm_c {

template <ScalarType ST>
struct ScalarTypeToCType;

template <> struct ScalarTypeToCType<ScalarType::FLOAT32> { using type = float; };
template <> struct ScalarTypeToCType<ScalarType::FLOAT16> { using type = half; };
template <> struct ScalarTypeToCType<ScalarType::BFLOAT16> { using type = __nv_bfloat16; };
template <> struct ScalarTypeToCType<ScalarType::FLOAT8_E4M3> { using type = __nv_fp8_e4m3; };
template <> struct ScalarTypeToCType<ScalarType::FLOAT8_E5M2> { using type = __nv_fp8_e5m2; };
template <> struct ScalarTypeToCType<ScalarType::INT8> { using type = int8_t; };
template <> struct ScalarTypeToCType<ScalarType::INT32> { using type = int32_t; };
template <> struct ScalarTypeToCType<ScalarType::INT64> { using type = int64_t; };
template <> struct ScalarTypeToCType<ScalarType::UINT8> { using type = uint8_t; };
template <> struct ScalarTypeToCType<ScalarType::BOOL> { using type = bool; };

}

namespace vm_c {
namespace detail {
[[noreturn]] inline void dispatch_fallback(const char* name, int type_val) {
    fprintf(stderr, "[FATAL] Unsupported dtype/type %d in %s\n", type_val, name);
    std::abort();
}
}
}

#define VMC_DISPATCH_CASE_FLOATING_TYPES(...) \
    case vm_c::ScalarType::FLOAT32: { using scalar_t = float; __VA_ARGS__; break; } \
    case vm_c::ScalarType::FLOAT16: { using scalar_t = half; __VA_ARGS__; break; } \
    case vm_c::ScalarType::BFLOAT16: { using scalar_t = __nv_bfloat16; __VA_ARGS__; break; }

#define VMC_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
    switch (TYPE) { VMC_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__) \
    default: vm_c::detail::dispatch_fallback(NAME, static_cast<int>(TYPE)); break; }

#define VMC_DISPATCH_CASE_HALF_TYPES(...) \
    case vm_c::ScalarType::FLOAT16: { using scalar_t = half; __VA_ARGS__; break; } \
    case vm_c::ScalarType::BFLOAT16: { using scalar_t = __nv_bfloat16; __VA_ARGS__; break; }

#define VMC_DISPATCH_HALF_TYPES(TYPE, NAME, ...) \
    switch (TYPE) { VMC_DISPATCH_CASE_HALF_TYPES(__VA_ARGS__) \
    default: vm_c::detail::dispatch_fallback(NAME, static_cast<int>(TYPE)); break; }

#define VMC_DISPATCH_CASE_FP8_TYPES(...) \
    case vm_c::ScalarType::FLOAT8_E4M3: { using fp8_t = __nv_fp8_e4m3; __VA_ARGS__; break; }

#define VMC_DISPATCH_FP8_TYPES(TYPE, NAME, ...) \
    switch (TYPE) { VMC_DISPATCH_CASE_FP8_TYPES(__VA_ARGS__) \
    default: vm_c::detail::dispatch_fallback(NAME, static_cast<int>(TYPE)); break; }

#define VMC_DISPATCH_CASE_QUANT_TYPES(...) \
    case vm_c::ScalarType::FLOAT8_E4M3: { using fp8_t = __nv_fp8_e4m3; __VA_ARGS__; break; } \
    case vm_c::ScalarType::INT8: { using fp8_t = int8_t; __VA_ARGS__; break; }

#define VMC_DISPATCH_QUANT_TYPES(TYPE, NAME, ...) \
    switch (TYPE) { VMC_DISPATCH_CASE_QUANT_TYPES(__VA_ARGS__) \
    default: vm_c::detail::dispatch_fallback(NAME, static_cast<int>(TYPE)); break; }

#define VMC_DISPATCH_VEC_SIZE(VEC_SIZE, ...) \
    switch (VEC_SIZE) { \
    case 16: { constexpr int vec_size = 16; __VA_ARGS__; break; } \
    case 8:  { constexpr int vec_size = 8;  __VA_ARGS__; break; } \
    case 4:  { constexpr int vec_size = 4;  __VA_ARGS__; break; } \
    case 2:  { constexpr int vec_size = 2;  __VA_ARGS__; break; } \
    default: { constexpr int vec_size = 1;  __VA_ARGS__; break; } \
    }

#define VMC_DISPATCH_BOOL(expr, const_expr, ...) \
    if (expr) { constexpr bool const_expr = true; __VA_ARGS__(); } \
    else { constexpr bool const_expr = false; __VA_ARGS__(); }

#define VMC_DISPATCH_RANK234(NUM_DIMS, ...) \
    switch (NUM_DIMS) { \
    case 2: { constexpr int tensor_rank = 2; __VA_ARGS__(); break; } \
    case 3: { constexpr int tensor_rank = 3; __VA_ARGS__(); break; } \
    case 4: { constexpr int tensor_rank = 4; __VA_ARGS__(); break; } \
    default: vm_c::detail::dispatch_fallback("VMC_DISPATCH_RANK234", NUM_DIMS); break; \
    }
