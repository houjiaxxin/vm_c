#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <cuda_runtime.h>

namespace vm_c {

struct TQConfig {
    static constexpr int kMaxSupportedHeadDim = 256;

    int head_dim = 128;
    int key_quant_bits = 4;
    int value_quant_bits = 4;
    bool norm_correction = true;
    bool key_fp8 = false;
    bool fp8_e4b15 = false;

    int mse_bytes = 0;
    int key_packed_size = 0;
    int val_data_bytes = 0;
    int value_packed_size = 0;
    int slot_size = 0;
    int slot_size_aligned = 0;
    int n_centroids = 0;
    int max_num_kv_splits = 32;

    void compute_layout() {
        n_centroids = 1 << key_quant_bits;
        if (key_fp8) {
            mse_bytes = head_dim;
            key_packed_size = head_dim;
        } else {
            mse_bytes = (head_dim * key_quant_bits + 7) / 8;
            key_packed_size = mse_bytes + 2;
        }
        val_data_bytes = (head_dim * value_quant_bits + 7) / 8;
        value_packed_size = val_data_bytes + 4;
        slot_size = key_packed_size + value_packed_size;
        slot_size_aligned = slot_size + (slot_size % 2);
    }

    static TQConfig from_method(int head_dim_, const char* method, int gpu_device = 0) {
        TQConfig cfg;
        cfg.head_dim = head_dim_;
        if (method && std::string(method) == "turboquant_k8v4") {
            cfg.key_quant_bits = 8;
            cfg.value_quant_bits = 4;
            cfg.norm_correction = false;
            cfg.key_fp8 = true;
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, gpu_device);
            int sm_ver = prop.major * 10 + prop.minor;
            cfg.fp8_e4b15 = (sm_ver < 89);
        } else if (method && std::string(method) == "turboquant_k3v4_nc") {
            cfg.key_quant_bits = 3;
            cfg.value_quant_bits = 4;
            cfg.norm_correction = true;
            cfg.key_fp8 = false;
        } else if (method && std::string(method) == "turboquant_3bit_nc") {
            cfg.key_quant_bits = 3;
            cfg.value_quant_bits = 3;
            cfg.norm_correction = true;
            cfg.key_fp8 = false;
        } else {
            cfg.key_quant_bits = 4;
            cfg.value_quant_bits = 4;
            cfg.norm_correction = true;
            cfg.key_fp8 = false;
        }
        cfg.compute_layout();
        return cfg;
    }
};

struct TQBuffers {
    void* hadamard = nullptr;
    void* hadamard_fp16 = nullptr;
    void* centroids = nullptr;
    void* midpoints = nullptr;
    int head_dim = 0;
    int centroid_bits = 0;
    bool initialized = false;

    void init(int head_dim_, int centroid_bits_, int gpu_device);
    void free_buf();
};

struct TQWorkspace {
    void* store_k_float = nullptr;
    void* store_norms = nullptr;
    void* store_x_hat = nullptr;
    void* store_y_rotated = nullptr;
    int store_capacity = 0;

    void* decode_q_float = nullptr;
    void* decode_q_rotated = nullptr;
    void* decode_mid_o = nullptr;
    int decode_capacity = 0;

    void ensure_store(int nh, int head_dim_, int gpu_device);
    void ensure_decode(int bhq, int head_dim_, int num_kv_splits, int gpu_device);
    void free_buf();
};

void turboquant_store(
    const void* key,
    const void* value,
    void* kv_cache,
    const int64_t* slot_mapping,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int64_t block_size,
    const TQConfig& config,
    const TQBuffers& buffers,
    TQWorkspace& workspace,
    cudaStream_t stream = 0);

void turboquant_decode_attention(
    void* output,
    const void* query,
    const void* kv_cache,
    const int32_t* block_tables,
    const int32_t* seq_lens,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    int64_t block_size,
    int64_t max_num_blocks_per_req,
    const TQConfig& config,
    const TQBuffers& buffers,
    TQWorkspace& workspace,
    int max_num_kv_splits,
    cudaStream_t stream = 0);

void build_hadamard_on_host(float* out, int d);

void solve_lloyd_max_centroids(float* centroids_out, int d, int bits);

}
