#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <cuda_runtime.h>
#include "vm_c/core/tensor.hpp"
#include "vm_c/cuda/kernels_attention.h"
#include "vm_c/cuda/kernels_turboquant.h"

namespace vm_c {

struct AttentionMetadata;

class AttentionBackend {
public:
    virtual ~AttentionBackend() = default;

    virtual void store_kv(
        const void* key, const void* value,
        void* key_cache, void* value_cache,
        const int64_t* slot_mapping,
        int num_tokens, int num_kv_heads, int head_dim,
        int64_t block_size, int64_t cache_stride,
        cudaStream_t stream) = 0;

    virtual void forward_prefill(
        void* output, const void* query,
        const void* key, const void* value,
        const void* key_cache, const void* value_cache,
        const AttentionMetadata& meta,
        int num_heads, int num_kv_heads, int head_dim,
        float scale, int64_t block_size,
        ScalarType dtype, cudaStream_t stream) = 0;

    virtual void forward_decode(
        void* output, const void* query,
        const void* key_cache, const void* value_cache,
        const AttentionMetadata& meta,
        int num_heads, int num_kv_heads, int head_dim,
        float scale, int64_t block_size,
        ScalarType dtype, cudaStream_t stream) = 0;

    virtual size_t workspace_size() const = 0;

    virtual void ensure_workspace(int max_num_batched_tokens, int max_num_seqs,
                                  int num_attention_heads, int num_kv_heads,
                                  int head_dim, int gpu_device) = 0;
};

class PagedAttentionBackend : public AttentionBackend {
public:
    PagedAttentionBackend(const std::string& kv_cache_layout = "nhd")
        : kv_cache_layout_(kv_cache_layout) {}

    void store_kv(
        const void* key, const void* value,
        void* key_cache, void* value_cache,
        const int64_t* slot_mapping,
        int num_tokens, int num_kv_heads, int head_dim,
        int64_t block_size, int64_t cache_stride,
        cudaStream_t stream) override;

    void forward_prefill(
        void* output, const void* query,
        const void* key, const void* value,
        const void* key_cache, const void* value_cache,
        const AttentionMetadata& meta,
        int num_heads, int num_kv_heads, int head_dim,
        float scale, int64_t block_size,
        ScalarType dtype, cudaStream_t stream) override;

    void forward_decode(
        void* output, const void* query,
        const void* key_cache, const void* value_cache,
        const AttentionMetadata& meta,
        int num_heads, int num_kv_heads, int head_dim,
        float scale, int64_t block_size,
        ScalarType dtype, cudaStream_t stream) override;

    size_t workspace_size() const override { return 0; }

    void ensure_workspace(int, int, int, int, int, int) override {}

private:
    std::string kv_cache_layout_ = "nhd"; // "nhd" 或 "hnd"
};

class TurboQuantBackend : public AttentionBackend {
public:
    TurboQuantBackend(int head_dim, const std::string& quant_method,
                      int num_attention_heads, int num_kv_heads,
                      int gpu_device);

    ~TurboQuantBackend() override;

    void store_kv(
        const void* key, const void* value,
        void* key_cache, void* value_cache,
        const int64_t* slot_mapping,
        int num_tokens, int num_kv_heads, int head_dim,
        int64_t block_size, int64_t cache_stride,
        cudaStream_t stream) override;

    void forward_prefill(
        void* output, const void* query,
        const void* key, const void* value,
        const void* key_cache, const void* value_cache,
        const AttentionMetadata& meta,
        int num_heads, int num_kv_heads, int head_dim,
        float scale, int64_t block_size,
        ScalarType dtype, cudaStream_t stream) override;

    void forward_decode(
        void* output, const void* query,
        const void* key_cache, const void* value_cache,
        const AttentionMetadata& meta,
        int num_heads, int num_kv_heads, int head_dim,
        float scale, int64_t block_size,
        ScalarType dtype, cudaStream_t stream) override;

    size_t workspace_size() const override;

    void ensure_workspace(int max_num_batched_tokens, int max_num_seqs,
                          int num_attention_heads, int num_kv_heads,
                          int head_dim, int gpu_device) override;

    const TQConfig& config() const { return config_; }

private:
    TQBuffers buffers_;
    TQConfig config_;
    TQWorkspace workspace_;
    int gpu_device_;
    int max_num_batched_tokens_ = 0;
    int max_num_seqs_ = 0;
    int num_attention_heads_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
};

}
