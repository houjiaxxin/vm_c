#include "vm_c/memory/uva_offload.hpp"
#include "vm_c/cuda/gpu_arch.hpp"
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

namespace vm_c {

UVAManager::UVAManager(int gpu_device) : gpu_device_(gpu_device) {
    CUDA_CHECK(cudaSetDevice(gpu_device_));
}

UVAManager::~UVAManager() = default;

void* UVAManager::allocate_cpu_pinned(size_t bytes) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, bytes));
    return ptr;
}

void UVAManager::free_cpu_pinned(void* ptr) {
    if (ptr) CUDA_CHECK(cudaFreeHost(ptr));
}

void UVAManager::copy_cpu_to_gpu(void* gpu_dst, const void* cpu_src,
                                  size_t bytes, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(gpu_dst, cpu_src, bytes, cudaMemcpyHostToDevice, stream));
}

void UVAManager::copy_gpu_to_cpu(void* cpu_dst, const void* gpu_src,
                                  size_t bytes, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(cpu_dst, gpu_src, bytes, cudaMemcpyDeviceToHost, stream));
}

bool UVAManager::is_uva_available() const {
    int supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&supported, cudaDevAttrUnifiedAddressing, gpu_device_));
    return supported != 0;
}

// =========================================================================
// StaticBufferPool
// =========================================================================

StaticBufferPool::StaticBufferPool(const std::vector<ParamInfo>& params,
                                   int slot_capacity, int gpu_device)
    : slot_capacity_(slot_capacity), gpu_device_(gpu_device) {

    CUDA_CHECK(cudaSetDevice(gpu_device_));

    for (const auto& param : params) {
        BufferKey key{param.name, param.shape, param.stride, param.dtype_size};
        if (buffers_.count(key) > 0) continue;

        std::vector<void*> slots(slot_capacity_, nullptr);
        for (int s = 0; s < slot_capacity_; ++s) {
            CUDA_CHECK(cudaMalloc(&slots[s], param.num_bytes));
            total_bytes_ += param.num_bytes;
        }
        buffers_[key] = std::move(slots);
    }

    spdlog::info("[StaticBufferPool] allocated {} groups x {} slots = {:.3f} MB on GPU {}",
                 buffers_.size(), slot_capacity_,
                 static_cast<double>(total_bytes_) / (1024. * 1024.),
                 gpu_device_);
}

StaticBufferPool::~StaticBufferPool() {
    CUDA_CHECK(cudaSetDevice(gpu_device_));
    for (auto& [key, slots] : buffers_) {
        for (auto* ptr : slots) {
            if (ptr) CUDA_CHECK(cudaFree(ptr));
        }
    }
}

void* StaticBufferPool::get_buffer(const BufferKey& key, int slot_idx) {
    auto it = buffers_.find(key);
    if (it == buffers_.end()) return nullptr;
    slot_idx = slot_idx % slot_capacity_;
    return it->second[slot_idx];
}

// =========================================================================
// ARCCachePolicy
// =========================================================================

ARCCachePolicy::ARCCachePolicy(size_t capacity)
    : capacity_(capacity), p_(0) {}

void ARCCachePolicy::move_to_t2(int64_t block_id) {
    t2_.push_front(block_id);
    t2_map_[block_id] = t2_.begin();
}

int64_t ARCCachePolicy::replace(bool in_b2) {
    int64_t evicted = -1;
    if (!t1_.empty() && (t1_.size() > p_ || (t1_.size() == p_ && in_b2))) {
        evicted = t1_.back();
        t1_.pop_back();
        t1_map_.erase(evicted);
        b1_.push_front(evicted);
        b1_map_[evicted] = b1_.begin();
    } else if (!t2_.empty()) {
        evicted = t2_.back();
        t2_.pop_back();
        t2_map_.erase(evicted);
        b2_.push_front(evicted);
        b2_map_[evicted] = b2_.begin();
    }
    return evicted;
}

void ARCCachePolicy::access(int64_t block_id) {
    if (t1_map_.count(block_id)) {
        t1_.erase(t1_map_[block_id]);
        t1_map_.erase(block_id);
        move_to_t2(block_id);
        return;
    }

    if (t2_map_.count(block_id)) {
        t2_.erase(t2_map_[block_id]);
        t2_map_.erase(block_id);
        move_to_t2(block_id);
        return;
    }

    if (b1_map_.count(block_id)) {
        p_ = std::min(capacity_, p_ + std::max(size_t(1), b2_.size() / b1_.size()));
        replace(true);
        b1_.erase(b1_map_[block_id]);
        b1_map_.erase(block_id);
        move_to_t2(block_id);
        return;
    }

    if (b2_map_.count(block_id)) {
        p_ = std::max(size_t(0), p_ - std::max(size_t(1), b1_.size() / b2_.size()));
        replace(false);
        b2_.erase(b2_map_[block_id]);
        b2_map_.erase(block_id);
        move_to_t2(block_id);
        return;
    }

    size_t total = t1_.size() + t2_.size();
    if (total >= capacity_) {
        replace(false);
    }
    if (t1_.size() + b1_.size() >= capacity_) {
        if (!b1_.empty()) {
            int64_t old = b1_.back();
            b1_.pop_back();
            b1_map_.erase(old);
        }
    } else if (total + b1_.size() + b2_.size() >= capacity_) {
        if (!b2_.empty()) {
            int64_t old = b2_.back();
            b2_.pop_back();
            b2_map_.erase(old);
        }
    }

    t1_.push_front(block_id);
    t1_map_[block_id] = t1_.begin();
}

int64_t ARCCachePolicy::evict_one() {
    return replace(false);
}

bool ARCCachePolicy::contains(int64_t block_id) const {
    return t1_map_.count(block_id) || t2_map_.count(block_id);
}

void ARCCachePolicy::remove(int64_t block_id) {
    if (t1_map_.count(block_id)) {
        t1_.erase(t1_map_[block_id]);
        t1_map_.erase(block_id);
    } else if (t2_map_.count(block_id)) {
        t2_.erase(t2_map_[block_id]);
        t2_map_.erase(block_id);
    } else if (b1_map_.count(block_id)) {
        b1_.erase(b1_map_[block_id]);
        b1_map_.erase(block_id);
    } else if (b2_map_.count(block_id)) {
        b2_.erase(b2_map_[block_id]);
        b2_map_.erase(block_id);
    }
}

size_t ARCCachePolicy::size() const {
    return t1_.size() + t2_.size();
}

// =========================================================================
// PrefetchOffloadManager
// =========================================================================

PrefetchOffloadManager::PrefetchOffloadManager(size_t cpu_offload_bytes, int group_size,
                                               int prefetch_step, int gpu_device,
                                               const std::vector<ParamInfo>& params,
                                               int slot_capacity)
    : gpu_device_(gpu_device), total_capacity_(cpu_offload_bytes),
      group_size_(group_size), prefetch_step_(prefetch_step) {

    CUDA_CHECK(cudaSetDevice(gpu_device_));

    if (cpu_offload_bytes > 0) {
        CUDA_CHECK(cudaMallocHost(&cpu_pool_, cpu_offload_bytes));
    }

    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&copy_done_event_, cudaEventDisableTiming));

    if (!params.empty()) {
        buffer_pool_ = std::make_unique<StaticBufferPool>(params, slot_capacity, gpu_device_);
    }

    size_t num_cache_blocks = cpu_offload_bytes / (group_size > 0 ? group_size : 1);
    cache_policy_ = std::make_unique<ARCCachePolicy>(num_cache_blocks);

    spdlog::info("PrefetchOffload pool: {} bytes, group_size={}, prefetch_step={}, "
                 "copy_stream=dedicated, cache=ARC",
                 cpu_offload_bytes, group_size_, prefetch_step_);
}

PrefetchOffloadManager::~PrefetchOffloadManager() {
    if (copy_stream_) {
        CUDA_CHECK(cudaSetDevice(gpu_device_));
        CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
        CUDA_CHECK(cudaStreamDestroy(copy_stream_));
        copy_stream_ = nullptr;
    }
    if (copy_done_event_) {
        CUDA_CHECK(cudaEventDestroy(copy_done_event_));
        copy_done_event_ = nullptr;
    }

    buffer_pool_.reset();
    cache_policy_.reset();

    if (cpu_pool_) {
        CUDA_CHECK(cudaSetDevice(gpu_device_));
        CUDA_CHECK(cudaFreeHost(cpu_pool_));
        cpu_pool_ = nullptr;
    }
}

void* PrefetchOffloadManager::allocate_from_pool(size_t bytes) {
    std::lock_guard<std::mutex> lk(pool_mutex_);
    if (used_bytes_ + bytes > total_capacity_) {
        int64_t evicted = cache_policy_->evict_one();
        if (evicted >= 0) {
            auto it = group_slots_.find(static_cast<int>(evicted));
            if (it != group_slots_.end() && it->second.in_use) {
                used_bytes_ -= it->second.bytes;
                it->second.in_use = false;
                it->second.cpu_ptr = nullptr;
            }
        }
    }
    if (used_bytes_ + bytes > total_capacity_) return nullptr;
    void* ptr = static_cast<char*>(cpu_pool_) + used_bytes_;
    used_bytes_ += bytes;
    return ptr;
}

void PrefetchOffloadManager::offload_group(void* gpu_ptr, size_t bytes, int group_id,
                                             cudaStream_t stream) {
    auto& slot = group_slots_[group_id];
    if (!slot.in_use) {
        slot.cpu_ptr = allocate_from_pool(bytes);
        slot.bytes = bytes;
        slot.in_use = true;
        slot.prefetched = false;
    }
    if (!slot.cpu_ptr) {
        spdlog::error("PrefetchOffload: no CPU memory for group {}", group_id);
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(slot.cpu_ptr, gpu_ptr, bytes, cudaMemcpyDeviceToHost, stream));
    cache_policy_->access(group_id);
}

void PrefetchOffloadManager::prefetch_group(void* gpu_ptr, size_t bytes, int group_id,
                                              cudaStream_t stream) {
    auto it = group_slots_.find(group_id);
    if (it != group_slots_.end() && it->second.in_use && !it->second.prefetched) {
        CUDA_CHECK(cudaMemcpyAsync(gpu_ptr, it->second.cpu_ptr, bytes,
                                   cudaMemcpyHostToDevice, copy_stream_));
        CUDA_CHECK(cudaEventRecord(copy_done_event_, copy_stream_));
        it->second.prefetched = true;
    }
}

void PrefetchOffloadManager::load_group(void* gpu_ptr, size_t bytes, int group_id,
                                          cudaStream_t stream) {
    auto it = group_slots_.find(group_id);
    if (it != group_slots_.end() && it->second.in_use) {
        if (it->second.prefetched) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, copy_done_event_));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(gpu_ptr, it->second.cpu_ptr, bytes,
                                       cudaMemcpyHostToDevice, stream));
        }
        it->second.prefetched = true;
        cache_policy_->access(group_id);
    }
}

void PrefetchOffloadManager::wait_layer(int layer_idx, cudaStream_t compute_stream) {
    CUDA_CHECK(cudaStreamWaitEvent(compute_stream, copy_done_event_));
}

void PrefetchOffloadManager::start_prefetch(int layer_idx, size_t bytes) {
    int group_id = layer_idx / group_size_;
    auto it = group_slots_.find(group_id);
    if (it == group_slots_.end() || !it->second.in_use) return;

    void* gpu_dst = buffer_pool_ ? buffer_pool_->get_buffer(
        BufferKey{"layer_" + std::to_string(layer_idx), {}, {}, 0},
        layer_idx % 2) : nullptr;

    if (gpu_dst && !it->second.prefetched) {
        CUDA_CHECK(cudaMemcpyAsync(gpu_dst, it->second.cpu_ptr, bytes,
                                   cudaMemcpyHostToDevice, copy_stream_));
        CUDA_CHECK(cudaEventRecord(copy_done_event_, copy_stream_));
        it->second.prefetched = true;
    }
}

}  // namespace vm_c
