#pragma once

#include <cstdint>
#include <cstddef>
#include <unordered_map>
#include <list>
#include <vector>
#include <mutex>
#include <string>

#include <cuda_runtime.h>

#include "vm_c/core/tensor.hpp"

namespace vm_c {

class UVAManager {
public:
    UVAManager(int gpu_device = 0);
    ~UVAManager();

    void* allocate_cpu_pinned(size_t bytes);
    void free_cpu_pinned(void* ptr);

    void copy_cpu_to_gpu(void* gpu_dst, const void* cpu_src, size_t bytes,
                         cudaStream_t stream = 0);
    void copy_gpu_to_cpu(void* cpu_dst, const void* gpu_src, size_t bytes,
                         cudaStream_t stream = 0);

    bool is_uva_available() const;

private:
    int gpu_device_;
};

// =========================================================================
// StaticBufferPool
//
// 借鉴 vllm PrefetchOffloader.StaticBufferPool：
// 按 (name, shape, stride, dtype_size) 分组，每组预分配 slot_capacity 个
// GPU buffer，循环复用。零动态分配、零碎片化。
// =========================================================================

struct BufferKey {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    int dtype_size;

    bool operator==(const BufferKey& other) const {
        return name == other.name && shape == other.shape &&
               stride == other.stride && dtype_size == other.dtype_size;
    }
};

struct BufferKeyHash {
    size_t operator()(const BufferKey& k) const {
        size_t h = std::hash<std::string>()(k.name);
        for (auto s : k.shape) h ^= std::hash<int64_t>()(s) + 0x9e3779b9 + (h << 6) + (h >> 2);
        for (auto s : k.stride) h ^= std::hash<int64_t>()(s) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(k.dtype_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct ParamInfo {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    int dtype_size;
    size_t num_bytes;
};

class StaticBufferPool {
public:
    StaticBufferPool(const std::vector<ParamInfo>& params, int slot_capacity, int gpu_device);
    ~StaticBufferPool();

    void* get_buffer(const BufferKey& key, int slot_idx);
    size_t total_bytes() const { return total_bytes_; }

private:
    int slot_capacity_;
    int gpu_device_;
    size_t total_bytes_ = 0;
    std::unordered_map<BufferKey, std::vector<void*>, BufferKeyHash> buffers_;
};

// =========================================================================
// ARCCachePolicy
//
// 自适应替换缓存（Adaptive Replacement Cache）：
// 同时追踪频繁访问（freq_list）和近期访问（recency_list），
// 比纯 LRU 更抗扫描攻击。借鉴 vllm CPUOffloadingManager 的缓存策略。
// =========================================================================

class ARCCachePolicy {
public:
    explicit ARCCachePolicy(size_t capacity);

    void access(int64_t block_id);
    int64_t evict_one();
    bool contains(int64_t block_id) const;
    void remove(int64_t block_id);
    size_t size() const;

private:
    size_t capacity_;
    size_t p_;

    std::list<int64_t> t1_;
    std::unordered_map<int64_t, std::list<int64_t>::iterator> t1_map_;

    std::list<int64_t> t2_;
    std::unordered_map<int64_t, std::list<int64_t>::iterator> t2_map_;

    std::list<int64_t> b1_;
    std::unordered_map<int64_t, std::list<int64_t>::iterator> b1_map_;

    std::list<int64_t> b2_;
    std::unordered_map<int64_t, std::list<int64_t>::iterator> b2_map_;

    void move_to_t2(int64_t block_id);
    int64_t replace(bool in_b2);
};

// =========================================================================
// PrefetchOffloadManager（重构版）
//
// 三级流水线预取引擎：
//   Level 1: StaticBufferPool（GPU 侧静态缓冲区，循环复用）
//   Level 2: 异步预取管线（独立 copy_stream + event-based 同步）
//   Level 3: ARC 缓存策略（CPU 侧智能置换）
//
// 借鉴 vllm PrefetchOffloader + CPUOffloadingManager
// =========================================================================

class PrefetchOffloadManager {
public:
    PrefetchOffloadManager(size_t cpu_offload_bytes, int group_size,
                           int prefetch_step, int gpu_device,
                           const std::vector<ParamInfo>& params = {},
                           int slot_capacity = 2);
    ~PrefetchOffloadManager();

    void offload_group(void* gpu_ptr, size_t bytes, int group_id, cudaStream_t stream = 0);
    void prefetch_group(void* gpu_ptr, size_t bytes, int group_id, cudaStream_t stream = 0);
    void load_group(void* gpu_ptr, size_t bytes, int group_id, cudaStream_t stream = 0);

    void wait_layer(int layer_idx, cudaStream_t compute_stream);
    void start_prefetch(int layer_idx, size_t bytes);

    int group_size() const { return group_size_; }
    int prefetch_step() const { return prefetch_step_; }
    size_t total_capacity() const { return total_capacity_; }

    StaticBufferPool* buffer_pool() { return buffer_pool_.get(); }

private:
    int gpu_device_;
    size_t total_capacity_;
    int group_size_;
    int prefetch_step_;

    void* cpu_pool_ = nullptr;

    cudaStream_t copy_stream_ = nullptr;
    cudaEvent_t copy_done_event_ = nullptr;

    std::unique_ptr<StaticBufferPool> buffer_pool_;
    std::unique_ptr<ARCCachePolicy> cache_policy_;

    struct GroupSlot {
        void* cpu_ptr;
        size_t bytes;
        bool in_use;
        bool prefetched;
    };
    std::unordered_map<int, GroupSlot> group_slots_;

    std::mutex pool_mutex_;
    size_t used_bytes_ = 0;

    void* allocate_from_pool(size_t bytes);
};

}
