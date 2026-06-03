#pragma once
// Shared IQ4_XS type definitions
// This file is always included inside namespace vm_c { } blocks.
// Do NOT add a namespace wrapper here.

#include <cstdint>

// IQ4_XS block size constants (from llama.cpp ggml-common.h)
constexpr int QK_K = 256;
constexpr int QK4_NL = 32;  // IQ4_NL block size

// IQ4_XS block structure (from llama.cpp ggml-common.h)
#pragma pack(push, 1)
struct BlockIQ4XS {
    uint16_t d;              // fp16 scale
    uint16_t scales_h;       // high bits of sub-block scales
    uint8_t  scales_l[QK_K / 64]; // low bits of sub-block scales
    uint8_t  qs[QK_K / 2];   // 4-bit quantized values
};
#pragma pack(pop)
