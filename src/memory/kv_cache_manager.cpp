#include "vm_c/memory/kv_cache_manager.hpp"
#include "vm_c/cuda/kernels_cache.h"
#include "vm_c/cuda/gpu_arch.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>
#include <spdlog/spdlog.h>

namespace vm_c {

FreeKVCacheBlockQueue::FreeKVCacheBlockQueue(std::vector<KVCacheBlock>& blocks) {
    num_free_ = static_cast<int64_t>(blocks.size());

    fake_head_.block_id = -1;
    fake_tail_.block_id = -1;

    if (num_free_ > 0) {
        for (int64_t i = 0; i < num_free_; ++i) {
            if (i > 0) blocks[i].prev_free = &blocks[i - 1];
            if (i < num_free_ - 1) blocks[i].next_free = &blocks[i + 1];
        }
        fake_head_.next_free = &blocks[0];
        blocks[0].prev_free = &fake_head_;
        fake_tail_.prev_free = &blocks[num_free_ - 1];
        blocks[num_free_ - 1].next_free = &fake_tail_;
    } else {
        fake_head_.next_free = &fake_tail_;
        fake_tail_.prev_free = &fake_head_;
    }
}

KVCacheBlock* FreeKVCacheBlockQueue::popleft() {
    if (fake_head_.next_free == &fake_tail_) {
        return nullptr;
    }

    KVCacheBlock* first = fake_head_.next_free;
    fake_head_.next_free = first->next_free;
    first->next_free->prev_free = &fake_head_;

    first->prev_free = nullptr;
    first->next_free = nullptr;
    num_free_--;
    return first;
}

std::vector<KVCacheBlock*> FreeKVCacheBlockQueue::popleft_n(int64_t n) {
    if (n <= 0) return {};
    if (num_free_ < n) return {};

    std::vector<KVCacheBlock*> result;
    result.reserve(n);
    num_free_ -= n;

    KVCacheBlock* curr = fake_head_.next_free;
    for (int64_t i = 0; i < n; ++i) {
        result.push_back(curr);
        KVCacheBlock* last = curr;
        curr = curr->next_free;
        last->prev_free = nullptr;
        last->next_free = nullptr;
    }

    if (curr) {
        fake_head_.next_free = curr;
        curr->prev_free = &fake_head_;
    }
    return result;
}

void FreeKVCacheBlockQueue::remove(KVCacheBlock* block) {
    if (!block || !block->prev_free || !block->next_free) return;

    block->prev_free->next_free = block->next_free;
    block->next_free->prev_free = block->prev_free;

    block->prev_free = nullptr;
    block->next_free = nullptr;
    num_free_--;
}

void FreeKVCacheBlockQueue::append(KVCacheBlock* block) {
    if (!block) return;

    KVCacheBlock* last = fake_tail_.prev_free;
    last->next_free = block;
    block->prev_free = last;
    block->next_free = &fake_tail_;
    fake_tail_.prev_free = block;
    num_free_++;
}

void FreeKVCacheBlockQueue::append_n(const std::vector<KVCacheBlock*>& blocks) {
    if (blocks.empty()) return;

    KVCacheBlock* last = fake_tail_.prev_free;
    for (auto* block : blocks) {
        last->next_free = block;
        block->prev_free = last;
        last = block;
    }
    last->next_free = &fake_tail_;
    fake_tail_.prev_free = last;
    num_free_ += static_cast<int64_t>(blocks.size());
}

KVCacheBlock* BlockHashToBlockMap::get(uint64_t hash, int group_id) {
    BlockHashKey key{hash, group_id};
    auto it = cache_.find(key);
    if (it == cache_.end()) return nullptr;

    auto& entry = it->second;
    if (!entry.is_multi) {
        return entry.single;
    }
    if (!entry.multi.empty()) {
        return entry.multi.begin()->second;
    }
    return nullptr;
}

void BlockHashToBlockMap::insert(uint64_t hash, int group_id, KVCacheBlock* block) {
    BlockHashKey key{hash, group_id};
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        cache_[key] = {};
        cache_[key].single = block;
        cache_[key].is_multi = false;
    } else if (!it->second.is_multi) {
        KVCacheBlock* existing = it->second.single;
        it->second.multi = {{existing->block_id, existing}, {block->block_id, block}};
        it->second.single = nullptr;
        it->second.is_multi = true;
    } else {
        it->second.multi[block->block_id] = block;
    }
}

KVCacheBlock* BlockHashToBlockMap::pop(uint64_t hash, int group_id, int64_t block_id) {
    BlockHashKey key{hash, group_id};
    auto it = cache_.find(key);
    if (it == cache_.end()) return nullptr;

    auto& entry = it->second;
    if (!entry.is_multi) {
        KVCacheBlock* block = entry.single;
        if (block && block->block_id == block_id) {
            cache_.erase(it);
            return block;
        }
        return nullptr;
    }

    auto mit = entry.multi.find(block_id);
    if (mit == entry.multi.end()) return nullptr;

    KVCacheBlock* block = mit->second;
    entry.multi.erase(mit);
    if (entry.multi.size() == 1) {
        auto remaining = entry.multi.begin();
        entry.single = remaining->second;
        entry.multi.clear();
        entry.is_multi = false;
    } else if (entry.multi.empty()) {
        cache_.erase(it);
    }
    return block;
}

void BlockHashToBlockMap::clear() {
    cache_.clear();
}

KVCacheManager::KVCacheManager(const CacheConfig& config, const ModelConfig& model_config,
                               int num_layers, int num_kv_heads, int head_dim, int gpu_device,
                               bool is_mla)
    : block_size_(config.block_size)
    , num_layers_(num_layers)
    , num_kv_heads_(num_kv_heads)
    , head_dim_(head_dim)
    , gpu_device_(gpu_device)
    , model_config_(model_config)
    , kv_dtype_([&]() {
        const auto& dtype_str = config.kv_cache_quant_method.empty() ? config.kv_cache_dtype : config.kv_cache_quant_method;
        if (dtype_str.find("turboquant") != std::string::npos) {
            return gpu().supports_bf16() ? DataType::BFLOAT16 : DataType::FLOAT16;
        } else if (dtype_str == "fp8_e4m3") {
            return DataType::FLOAT8_E4M3;
        } else if (dtype_str == "fp8_e5m2") {
            return DataType::FLOAT8_E5M2;
        } else if (dtype_str == "bf16") {
            return DataType::BFLOAT16;
        } else if (dtype_str == "fp16") {
            return DataType::FLOAT16;
        }
        return gpu().supports_bf16() ? DataType::BFLOAT16 : DataType::FLOAT16;
    }())
    , is_mla_(is_mla)
    , enable_caching_(config.enable_prefix_caching)
    , kv_cache_dtype_(config.kv_cache_quant_method.empty() ? config.kv_cache_dtype : config.kv_cache_quant_method)
    , kv_cache_layout_(config.kv_cache_layout.empty() ? "nhd" : config.kv_cache_layout) {
    if (kv_cache_dtype_.find("turboquant") != std::string::npos) {
        tq_layout_ = TurboQuantLayout::create(head_dim, kv_cache_dtype_, gpu_device_);
        tq_layer_enabled_.resize(num_layers, true);
        spdlog::info("TurboQuant KV cache: method={}, slot_size={}, key_packed={}, value_packed={}, fp8_e4b15={}",
                     kv_cache_dtype_, tq_layout_.slot_size_aligned,
                     tq_layout_.key_packed_size, tq_layout_.value_packed_size,
                     tq_layout_.fp8_e4b15);
        // [vm_c] 摘要：head_dim 输入 + TQ 关键尺寸（与 TurboQuantKvBridge 必须一致）
        spdlog::info("[TQ-CACHE-INIT] head_dim={} slot_size={} slot_size_aligned={} key_packed={} value_packed={} num_layers={}",
                     head_dim, tq_layout_.slot_size, tq_layout_.slot_size_aligned,
                     tq_layout_.key_packed_size, tq_layout_.value_packed_size, num_layers);
    }
    build_groups();
}

KVCacheManager::~KVCacheManager() {
    CudaDeviceGuard guard(gpu_device_);
    for (auto* p : gpu_key_caches_) { if (p) CUDA_CHECK(cudaFree(p)); }
    for (auto* p : gpu_value_caches_) { if (p) CUDA_CHECK(cudaFree(p)); }
    for (auto* p : cpu_key_caches_) { if (p) CUDA_CHECK(cudaFreeHost(p)); }
    for (auto* p : cpu_value_caches_) { if (p) CUDA_CHECK(cudaFreeHost(p)); }
}

void KVCacheManager::build_groups() {
    groups_.clear();

    if (is_turboquant()) {
        std::vector<int> tq_layers;
        std::vector<int> std_layers;
        for (int i = 0; i < num_layers_; ++i) {
            if (!model_config_.layer_needs_kv_cache(i)) {
                continue;
            }
            if (tq_layer_enabled_.empty() || tq_layer_enabled_[i]) {
                tq_layers.push_back(i);
            } else {
                std_layers.push_back(i);
            }
        }

        if (!tq_layers.empty()) {
            groups_.push_back({0, tq_layers, compute_bytes_per_block(true), true});
        }
        if (!std_layers.empty()) {
            groups_.push_back({static_cast<int>(groups_.size()), std_layers, compute_bytes_per_block(false), false});
        }
    } else {
        std::vector<int> all_layers;
        for (int i = 0; i < num_layers_; ++i) {
            if (model_config_.layer_needs_kv_cache(i)) {
                all_layers.push_back(i);
            }
        }
        if (!all_layers.empty()) {
            groups_.push_back({0, all_layers, compute_bytes_per_block(false), false});
        }
    }

    layer_to_group_.clear();
    for (auto& g : groups_) {
        for (int layer : g.layer_indices) {
            layer_to_group_[layer] = g.group_id;
        }
    }

    group_num_gpu_blocks_.resize(groups_.size(), 0);
    group_num_free_gpu_blocks_.resize(groups_.size(), 0);
}

size_t KVCacheManager::compute_bytes_per_block(bool tq) const {
    if (tq) {
        return block_size_ * num_kv_heads_ * tq_layout_.slot_size_aligned;
    } else if (is_mla_) {
        return block_size_ * head_dim_ * dtype_size(kv_dtype_);
    } else {
        return block_size_ * num_kv_heads_ * head_dim_ * dtype_size(kv_dtype_);
    }
}

int KVCacheManager::group_id_for_layer(int layer) const {
    auto it = layer_to_group_.find(layer);
    return it != layer_to_group_.end() ? it->second : 0;
}

int64_t KVCacheManager::group_num_gpu_blocks(int group_id) const {
    if (group_id < 0 || group_id >= static_cast<int>(group_num_gpu_blocks_.size())) return 0;
    return group_num_gpu_blocks_[group_id];
}

int64_t KVCacheManager::group_num_free_gpu_blocks(int group_id) const {
    if (group_id < 0 || group_id >= static_cast<int>(group_num_free_gpu_blocks_.size())) return 0;
    return group_num_free_gpu_blocks_[group_id];
}

void KVCacheManager::set_tq_boundary_layers(const std::vector<int>& skip_layers) {
    if (!is_turboquant()) return;
    for (int layer : skip_layers) {
        if (layer >= 0 && layer < num_layers_) {
            tq_layer_enabled_[layer] = false;
        }
    }
    build_groups();
    spdlog::info("TurboQuant boundary protection: {} layers use standard KV cache",
                 skip_layers.size());
}

void KVCacheManager::reserve_gpu_blocks(int64_t total_blocks) {
    gpu_blocks_.reserve(total_blocks);
}

void KVCacheManager::allocate_gpu_blocks_for_group(int group_id, int64_t num_blocks) {
    if (group_id < 0 || group_id >= static_cast<int>(groups_.size())) return;
    if (num_blocks <= 0) return;

    auto& g = groups_[group_id];
    group_num_gpu_blocks_[group_id] = num_blocks;
    group_num_free_gpu_blocks_[group_id] = num_blocks;

    CudaDeviceGuard guard(gpu_device_);

    for (int layer : g.layer_indices) {
        if (g.is_tq) {
            void* kv_ptr = nullptr;
            CUDA_CHECK(cudaMalloc(&kv_ptr, num_blocks * g.bytes_per_block));
            gpu_key_caches_.resize(std::max(static_cast<int>(gpu_key_caches_.size()), layer + 1), nullptr);
            gpu_value_caches_.resize(std::max(static_cast<int>(gpu_value_caches_.size()), layer + 1), nullptr);
            gpu_key_caches_[layer] = kv_ptr;
            gpu_value_caches_[layer] = nullptr;
        } else if (is_mla_) {
            void* k_ptr = nullptr;
            CUDA_CHECK(cudaMalloc(&k_ptr, num_blocks * g.bytes_per_block));
            gpu_key_caches_.resize(std::max(static_cast<int>(gpu_key_caches_.size()), layer + 1), nullptr);
            gpu_value_caches_.resize(std::max(static_cast<int>(gpu_value_caches_.size()), layer + 1), nullptr);
            gpu_key_caches_[layer] = k_ptr;
            gpu_value_caches_[layer] = nullptr;
        } else {
            void* k_ptr = nullptr;
            void* v_ptr = nullptr;
            CUDA_CHECK(cudaMalloc(&k_ptr, num_blocks * g.bytes_per_block));
            CUDA_CHECK(cudaMalloc(&v_ptr, num_blocks * g.bytes_per_block));
            gpu_key_caches_.resize(std::max(static_cast<int>(gpu_key_caches_.size()), layer + 1), nullptr);
            gpu_value_caches_.resize(std::max(static_cast<int>(gpu_value_caches_.size()), layer + 1), nullptr);
            gpu_key_caches_[layer] = k_ptr;
            gpu_value_caches_[layer] = v_ptr;
        }
    }

    int64_t prev_total = static_cast<int64_t>(gpu_blocks_.size());
    int64_t new_total = prev_total + num_blocks;
    gpu_blocks_.resize(new_total);
    for (int64_t i = prev_total; i < new_total; ++i) {
        gpu_blocks_[i].block_id = i;
        gpu_blocks_[i].ref_cnt = 0;
    }

    if (!free_queue_) {
        free_queue_ = std::make_unique<FreeKVCacheBlockQueue>(gpu_blocks_);
    } else {
        for (int64_t i = prev_total; i < new_total; ++i) {
            free_queue_->append(&gpu_blocks_[i]);
        }
    }

    if (!null_block_ && !gpu_blocks_.empty()) {
        null_block_ = free_queue_->popleft();
        null_block_->is_null = true;
        null_block_->ref_cnt = 1;
    }

    spdlog::info("Allocated {} GPU KV cache blocks for group {} ({} layers, {} bytes/block, is_tq={})",
                 num_blocks, group_id, g.layer_indices.size(), g.bytes_per_block, g.is_tq);
}

void KVCacheManager::allocate_cpu_blocks_for_group(int group_id, int64_t num_blocks) {
    if (group_id < 0 || group_id >= static_cast<int>(groups_.size())) return;
    if (num_blocks <= 0) return;

    auto& g = groups_[group_id];
    int64_t prev_cpu_blocks = num_cpu_blocks_;
    num_cpu_blocks_ += num_blocks;
    num_free_cpu_blocks_ += num_blocks;

    for (int layer : g.layer_indices) {
        cpu_key_caches_.resize(std::max(static_cast<int>(cpu_key_caches_.size()), layer + 1), nullptr);
        cpu_value_caches_.resize(std::max(static_cast<int>(cpu_value_caches_.size()), layer + 1), nullptr);

        if (prev_cpu_blocks == 0) {
            if (g.is_tq || is_mla_) {
                void* ptr = nullptr;
                CUDA_CHECK(cudaMallocHost(&ptr, num_blocks * g.bytes_per_block));
                cpu_key_caches_[layer] = ptr;
                cpu_value_caches_[layer] = nullptr;
            } else {
                void* k_ptr = nullptr;
                void* v_ptr = nullptr;
                CUDA_CHECK(cudaMallocHost(&k_ptr, num_blocks * g.bytes_per_block));
                CUDA_CHECK(cudaMallocHost(&v_ptr, num_blocks * g.bytes_per_block));
                cpu_key_caches_[layer] = k_ptr;
                cpu_value_caches_[layer] = v_ptr;
            }
        } else {
            size_t old_size = prev_cpu_blocks * g.bytes_per_block;
            size_t new_size = num_cpu_blocks_ * g.bytes_per_block;

            if (g.is_tq || is_mla_) {
                void* old_ptr = cpu_key_caches_[layer];
                void* new_ptr = nullptr;
                CUDA_CHECK(cudaMallocHost(&new_ptr, new_size));
                if (old_ptr) memcpy(new_ptr, old_ptr, old_size);
                CUDA_CHECK(cudaFreeHost(old_ptr));
                cpu_key_caches_[layer] = new_ptr;
            } else {
                void* old_k = cpu_key_caches_[layer];
                void* old_v = cpu_value_caches_[layer];
                void* new_k = nullptr;
                void* new_v = nullptr;
                CUDA_CHECK(cudaMallocHost(&new_k, new_size));
                CUDA_CHECK(cudaMallocHost(&new_v, new_size));
                if (old_k) memcpy(new_k, old_k, old_size);
                if (old_v) memcpy(new_v, old_v, old_size);
                CUDA_CHECK(cudaFreeHost(old_k));
                CUDA_CHECK(cudaFreeHost(old_v));
                cpu_key_caches_[layer] = new_k;
                cpu_value_caches_[layer] = new_v;
            }
        }
    }

    int64_t prev_size = static_cast<int64_t>(cpu_blocks_.size());
    cpu_blocks_.resize(num_cpu_blocks_);
    for (int64_t i = prev_size; i < num_cpu_blocks_; ++i) {
        cpu_blocks_[i].block_id = i;
        cpu_blocks_[i].ref_cnt = 0;
    }

    spdlog::info("Allocated {} CPU KV cache blocks for group {} (total: {}, pinned memory)",
                 num_blocks, group_id, num_cpu_blocks_);
}

int KVCacheManager::find_longest_prefix_hit(const std::vector<uint64_t>& block_hashes,
                                             int max_num_blocks,
                                             std::vector<int64_t>& out_block_ids) const {
    if (!enable_caching_ || block_hashes.empty()) return 0;

    out_block_ids.clear();
    int num_to_check = std::min(max_num_blocks, static_cast<int>(block_hashes.size()));

    for (int i = 0; i < num_to_check; ++i) {
        KVCacheBlock* cached = const_cast<BlockHashToBlockMap&>(hash_cache_).get(block_hashes[i], 0);
        if (!cached) break;
        out_block_ids.push_back(cached->block_id);
    }

    return static_cast<int>(out_block_ids.size());
}

std::vector<KVCacheSlot> KVCacheManager::allocate_slots(
    const std::string& request_id,
    int num_tokens,
    const std::vector<uint64_t>& prefix_block_hashes,
    int num_new_computed_tokens,
    const std::vector<int64_t>& new_computed_block_ids) {
    int64_t blocks_needed = (num_tokens + block_size_ - 1) / block_size_;
    std::vector<KVCacheSlot> slots;

    auto it = request_slots_.find(request_id);
    if (it != request_slots_.end()) {
        slots = it->second;
    }

    for (int64_t bid : new_computed_block_ids) {
        if (bid >= 0 && bid < static_cast<int64_t>(gpu_blocks_.size())) {
            auto& block = gpu_blocks_[bid];
            if (block.ref_cnt == 0 && !block.is_null) {
                free_queue_->remove(&block);
            }
            block.ref_cnt++;
            slots.push_back({bid, 0});
        }
    }

    int64_t additional_needed = blocks_needed - static_cast<int64_t>(slots.size());

    if (additional_needed > 0) {
        auto allocated = free_queue_->popleft_n(additional_needed);
        if (static_cast<int64_t>(allocated.size()) < additional_needed) {
            for (auto* blk : allocated) {
                free_queue_->append(blk);
            }
            spdlog::warn("Not enough GPU blocks: need {} more, got {} (free={})",
                         additional_needed, allocated.size(), free_queue_->num_free_blocks());
            return {};
        }

        for (auto* blk : allocated) {
            if (enable_caching_ && blk->has_hash) {
                int num_groups = static_cast<int>(groups_.size());
                for (int g = 0; g < num_groups; ++g) {
                    hash_cache_.pop(blk->block_hash.hash, g, blk->block_id);
                }
                blk->reset_hash();
            }
            blk->ref_cnt = 1;
            slots.push_back({blk->block_id, 0});
        }
    }

    request_slots_[request_id] = slots;
    return slots;
}

void KVCacheManager::free_slots(const std::string& request_id) {
    auto it = request_slots_.find(request_id);
    if (it == request_slots_.end()) return;

    std::vector<KVCacheBlock*> blocks_to_free;
    for (auto& slot : it->second) {
        if (slot.block_id >= 0 && slot.block_id < static_cast<int64_t>(gpu_blocks_.size())) {
            auto& block = gpu_blocks_[slot.block_id];
            block.ref_cnt--;
            if (block.ref_cnt <= 0 && !block.is_null) {
                block.ref_cnt = 0;
                blocks_to_free.push_back(&block);
            }
        }
    }

    std::reverse(blocks_to_free.begin(), blocks_to_free.end());
    free_queue_->append_n(blocks_to_free);

    request_slots_.erase(it);
}

void KVCacheManager::swap_request_slots(const std::string& request_id) {
    auto it = request_slots_.find(request_id);
    if (it == request_slots_.end()) return;

    for (auto& slot : it->second) {
        if (slot.block_id >= 0 && slot.block_id < static_cast<int64_t>(gpu_blocks_.size())) {
            offload_to_cpu(slot.block_id);
        }
    }

    // Free GPU blocks but keep the slot mapping (don't erase request_slots_)
    std::vector<KVCacheBlock*> blocks_to_free;
    for (auto& slot : it->second) {
        if (slot.block_id >= 0 && slot.block_id < static_cast<int64_t>(gpu_blocks_.size())) {
            auto& block = gpu_blocks_[slot.block_id];
            if (block.ref_cnt <= 0 && !block.is_null) {
                blocks_to_free.push_back(&block);
            }
        }
    }
    free_queue_->append_n(blocks_to_free);
}
void KVCacheManager::swap_back_request_slots(const std::string& request_id) {
    auto it = request_slots_.find(request_id);
    if (it == request_slots_.end()) return;

    // Save old slot info before freeing
    std::vector<int64_t> old_block_ids;
    for (auto& slot : it->second) {
        old_block_ids.push_back(slot.block_id);
    }

    // Free the old slot entries (but keep the mapping for CPU lookup)
    // Actually, we need to first load from CPU, then free
    // Collect old block IDs that have CPU mappings
    std::vector<int64_t> cpu_block_ids;
    for (int64_t old_bid : old_block_ids) {
        auto map_it = gpu_to_cpu_mapping_.find(old_bid);
        if (map_it != gpu_to_cpu_mapping_.end()) {
            cpu_block_ids.push_back(map_it->second);
        }
    }

    // Free old GPU blocks
    std::vector<KVCacheBlock*> blocks_to_free;
    for (auto& slot : it->second) {
        if (slot.block_id >= 0 && slot.block_id < static_cast<int64_t>(gpu_blocks_.size())) {
            auto& block = gpu_blocks_[slot.block_id];
            if (block.ref_cnt <= 0 && !block.is_null) {
                blocks_to_free.push_back(&block);
            }
        }
    }
    free_queue_->append_n(blocks_to_free);

    // Clear the old slot entries
    request_slots_.erase(it);

    // Re-allocate GPU slots for this request
    // We need to figure out how many tokens this request had
    // The slot count * block_size gives us the token count
    // But we need to know the original token count...
    // For simplicity, allocate one block per old slot
    int num_tokens = static_cast<int>(old_block_ids.size()) * static_cast<int>(block_size_);
    auto new_slots = allocate_slots(request_id, num_tokens, {}, 0, {});

    // Load CPU data into the new GPU slots
    CudaDeviceGuard guard(gpu_device_);
    int slot_idx = 0;
    for (int i = 0; i < static_cast<int>(old_block_ids.size()) && slot_idx < static_cast<int>(new_slots.size()); ++i) {
        int64_t old_bid = old_block_ids[i];
        auto map_it = gpu_to_cpu_mapping_.find(old_bid);
        if (map_it == gpu_to_cpu_mapping_.end()) continue;

        int64_t cpu_block = map_it->second;
        if (cpu_block < 0 || cpu_block >= num_cpu_blocks_) continue;
        if (cpu_blocks_[cpu_block].ref_cnt == 0) continue;

        int64_t new_bid = new_slots[slot_idx].block_id;
        if (new_bid < 0 || new_bid >= static_cast<int64_t>(gpu_blocks_.size())) {
            slot_idx++;
            continue;
        }

        // Copy data from CPU to new GPU block
        for (int layer = 0; layer < num_layers_; layer++) {
            int gid = group_id_for_layer(layer);
            size_t block_bytes = groups_[gid].bytes_per_block;

            auto* gpu_k = static_cast<uint8_t*>(gpu_key_caches_[layer]) + new_bid * block_bytes;
            auto* cpu_k = static_cast<uint8_t*>(cpu_key_caches_[layer]) + cpu_block * block_bytes;
            CUDA_CHECK(cudaMemcpy(gpu_k, cpu_k, block_bytes, cudaMemcpyHostToDevice));

            if (!groups_[gid].is_tq && !is_mla_) {
                auto* gpu_v = static_cast<uint8_t*>(gpu_value_caches_[layer]) + new_bid * block_bytes;
                auto* cpu_v = static_cast<uint8_t*>(cpu_value_caches_[layer]) + cpu_block * block_bytes;
                CUDA_CHECK(cudaMemcpy(gpu_v, cpu_v, block_bytes, cudaMemcpyHostToDevice));
            }
        }

        gpu_blocks_[new_bid].ref_cnt = cpu_blocks_[cpu_block].ref_cnt;
        gpu_blocks_[new_bid].has_hash = cpu_blocks_[cpu_block].has_hash;
        gpu_blocks_[new_bid].block_hash = cpu_blocks_[cpu_block].block_hash;

        // Update CPU block reference count and remove mapping
        cpu_blocks_[cpu_block].ref_cnt = 0;
        gpu_to_cpu_mapping_.erase(old_bid);
        cpu_to_gpu_mapping_.erase(cpu_block);
        num_free_cpu_blocks_++;

        slot_idx++;
    }
}
void KVCacheManager::cache_full_blocks(const std::string& request_id,
                                        const std::vector<uint64_t>& block_hashes,
                                        int num_cached_blocks,
                                        int num_full_blocks) {
    if (!enable_caching_ || block_hashes.empty()) return;

    auto it = request_slots_.find(request_id);
    if (it == request_slots_.end()) return;

    auto& slots = it->second;
    int num_groups = static_cast<int>(groups_.size());

    for (int i = num_cached_blocks; i < num_full_blocks && i < static_cast<int>(slots.size()); ++i) {
        int64_t bid = slots[i].block_id;
        if (bid < 0 || bid >= static_cast<int64_t>(gpu_blocks_.size())) continue;

        auto& block = gpu_blocks_[bid];
        if (block.is_null || block.has_hash) continue;

        if (i < static_cast<int>(block_hashes.size())) {
            block.set_hash(block_hashes[i], 0);
            for (int g = 0; g < num_groups; ++g) {
                hash_cache_.insert(block_hashes[i], g, &block);
            }
        }
    }
}

int64_t KVCacheManager::num_free_gpu_blocks() const {
    return free_queue_ ? free_queue_->num_free_blocks() : 0;
}

int64_t KVCacheManager::num_gpu_blocks() const {
    int64_t total = 0;
    for (auto n : group_num_gpu_blocks_) total += n;
    return total;
}

float KVCacheManager::usage() const {
    int64_t total = num_gpu_blocks();
    if (total <= 1) return 0.0f;
    int64_t free = num_free_gpu_blocks();
    return 1.0f - static_cast<float>(free) / static_cast<float>(total - 1);
}

bool KVCacheManager::reset_prefix_cache() {
    if (!enable_caching_) return false;

    int64_t used = num_gpu_blocks() - num_free_gpu_blocks();
    if (used > 1) {
        spdlog::warn("Cannot reset prefix cache: {} blocks still in use", used - 1);
        return false;
    }

    hash_cache_.clear();
    for (auto& block : gpu_blocks_) {
        block.reset_hash();
    }

    spdlog::info("Prefix cache reset successfully");
    return true;
}

void* KVCacheManager::key_cache_ptr(int layer) {
    return (layer >= 0 && layer < static_cast<int>(gpu_key_caches_.size()))
        ? gpu_key_caches_[layer] : nullptr;
}

void* KVCacheManager::value_cache_ptr(int layer) {
    return (layer >= 0 && layer < static_cast<int>(gpu_value_caches_.size()))
        ? gpu_value_caches_[layer] : nullptr;
}

void* KVCacheManager::cpu_key_cache_ptr(int layer) {
    return (layer >= 0 && layer < static_cast<int>(cpu_key_caches_.size()))
        ? cpu_key_caches_[layer] : nullptr;
}

void* KVCacheManager::cpu_value_cache_ptr(int layer) {
    return (layer >= 0 && layer < static_cast<int>(cpu_value_caches_.size()))
        ? cpu_value_caches_[layer] : nullptr;
}

bool KVCacheManager::compress_kv_block(int64_t block_id) {
    return true;
}

bool KVCacheManager::decompress_kv_block(int64_t block_id) {
    return true;
}

void KVCacheManager::offload_to_cpu(int64_t block_id) {
    if (block_id < 0 || block_id >= static_cast<int64_t>(gpu_blocks_.size())) return;
    if (num_cpu_blocks_ == 0) return;
    if (gpu_to_cpu_mapping_.count(block_id)) return;

    CudaDeviceGuard guard(gpu_device_);

    int64_t cpu_block = -1;
    for (int64_t i = 0; i < num_cpu_blocks_; i++) {
        if (cpu_blocks_[i].ref_cnt == 0) {
            cpu_block = i;
            break;
        }
    }
    if (cpu_block < 0) return;

    for (int layer = 0; layer < num_layers_; layer++) {
        int gid = group_id_for_layer(layer);
        size_t block_bytes = groups_[gid].bytes_per_block;

        auto* gpu_k = static_cast<uint8_t*>(gpu_key_caches_[layer]) + block_id * block_bytes;
        auto* cpu_k = static_cast<uint8_t*>(cpu_key_caches_[layer]) + cpu_block * block_bytes;
        CUDA_CHECK(cudaMemcpy(cpu_k, gpu_k, block_bytes, cudaMemcpyDeviceToHost));

        if (!groups_[gid].is_tq && !is_mla_) {
            auto* gpu_v = static_cast<uint8_t*>(gpu_value_caches_[layer]) + block_id * block_bytes;
            auto* cpu_v = static_cast<uint8_t*>(cpu_value_caches_[layer]) + cpu_block * block_bytes;
            CUDA_CHECK(cudaMemcpy(cpu_v, gpu_v, block_bytes, cudaMemcpyDeviceToHost));
        }
    }

    cpu_blocks_[cpu_block].ref_cnt = gpu_blocks_[block_id].ref_cnt;
    cpu_blocks_[cpu_block].has_hash = gpu_blocks_[block_id].has_hash;
    cpu_blocks_[cpu_block].block_hash = gpu_blocks_[block_id].block_hash;
    gpu_to_cpu_mapping_[block_id] = cpu_block;
    cpu_to_gpu_mapping_[cpu_block] = block_id;
    gpu_blocks_[block_id].ref_cnt = 0;
    num_free_cpu_blocks_--;
}

void KVCacheManager::load_from_cpu(int64_t block_id) {
    if (block_id < 0 || block_id >= static_cast<int64_t>(gpu_blocks_.size())) return;

    auto map_it = gpu_to_cpu_mapping_.find(block_id);
    if (map_it == gpu_to_cpu_mapping_.end()) return;

    int64_t cpu_block = map_it->second;
    if (cpu_block < 0 || cpu_block >= num_cpu_blocks_) return;
    if (cpu_blocks_[cpu_block].ref_cnt == 0) return;

    CudaDeviceGuard guard(gpu_device_);

    for (int layer = 0; layer < num_layers_; layer++) {
        int gid = group_id_for_layer(layer);
        size_t block_bytes = groups_[gid].bytes_per_block;

        auto* gpu_k = static_cast<uint8_t*>(gpu_key_caches_[layer]) + block_id * block_bytes;
        auto* cpu_k = static_cast<uint8_t*>(cpu_key_caches_[layer]) + cpu_block * block_bytes;
        CUDA_CHECK(cudaMemcpy(gpu_k, cpu_k, block_bytes, cudaMemcpyHostToDevice));

        if (!groups_[gid].is_tq && !is_mla_) {
            auto* gpu_v = static_cast<uint8_t*>(gpu_value_caches_[layer]) + block_id * block_bytes;
            auto* cpu_v = static_cast<uint8_t*>(cpu_value_caches_[layer]) + cpu_block * block_bytes;
            CUDA_CHECK(cudaMemcpy(gpu_v, cpu_v, block_bytes, cudaMemcpyHostToDevice));
        }
    }

    gpu_blocks_[block_id].ref_cnt = cpu_blocks_[cpu_block].ref_cnt;
    gpu_blocks_[block_id].has_hash = cpu_blocks_[cpu_block].has_hash;
    gpu_blocks_[block_id].block_hash = cpu_blocks_[cpu_block].block_hash;
    gpu_to_cpu_mapping_.erase(block_id);
    cpu_to_gpu_mapping_.erase(cpu_block);
    cpu_blocks_[cpu_block].ref_cnt = 0;
    num_free_cpu_blocks_++;
}

size_t KVCacheManager::gpu_kv_cache_bytes() const {
    size_t total = 0;
    for (size_t g = 0; g < groups_.size(); ++g) {
        total += group_num_gpu_blocks_[g] * groups_[g].bytes_per_block * groups_[g].layer_indices.size();
        if (!groups_[g].is_tq && !is_mla_) {
            total += group_num_gpu_blocks_[g] * groups_[g].bytes_per_block * groups_[g].layer_indices.size();
        }
    }
    return total;
}

size_t KVCacheManager::cpu_kv_cache_bytes() const {
    size_t total = 0;
    for (size_t g = 0; g < groups_.size(); ++g) {
        total += num_cpu_blocks_ * groups_[g].bytes_per_block * groups_[g].layer_indices.size();
        if (!groups_[g].is_tq && !is_mla_) {
            total += num_cpu_blocks_ * groups_[g].bytes_per_block * groups_[g].layer_indices.size();
        }
    }
    return total;
}

void KVCacheManager::trim_request_to_tokens(const std::string& request_id, int num_tokens) {
    if (num_tokens < 0) {
        throw std::runtime_error("trim_request_to_tokens: num_tokens must be non-negative");
    }
    auto it = request_slots_.find(request_id);
    if (it == request_slots_.end()) {
        return;
    }

    const int64_t blocks_needed =
        num_tokens > 0 ? (static_cast<int64_t>(num_tokens) + block_size_ - 1) / block_size_ : 0;
    if (blocks_needed >= static_cast<int64_t>(it->second.size())) {
        return;
    }

    std::vector<KVCacheBlock*> blocks_to_free;
    for (size_t i = static_cast<size_t>(blocks_needed); i < it->second.size(); ++i) {
        const int64_t bid = it->second[i].block_id;
        if (bid < 0 || bid >= static_cast<int64_t>(gpu_blocks_.size())) {
            continue;
        }
        auto& block = gpu_blocks_[static_cast<size_t>(bid)];
        block.ref_cnt--;
        if (block.ref_cnt <= 0 && !block.is_null) {
            block.ref_cnt = 0;
            blocks_to_free.push_back(&block);
        }
    }
    if (!blocks_to_free.empty()) {
        std::reverse(blocks_to_free.begin(), blocks_to_free.end());
        free_queue_->append_n(blocks_to_free);
    }
    it->second.resize(static_cast<size_t>(blocks_needed));
}

size_t KVCacheManager::kv_bytes_per_token(int layer) const {
    if (layer < 0 || layer >= num_layers_) {
        throw std::runtime_error("kv_bytes_per_token: invalid layer " + std::to_string(layer));
    }
    const int gid = group_id_for_layer(layer);
    const auto& g = groups_[static_cast<size_t>(gid)];
    if (g.is_tq) {
        return static_cast<size_t>(num_kv_heads_) * tq_layout_.slot_size_aligned;
    }
    const size_t elem = (kv_dtype_ == DataType::FLOAT32) ? sizeof(float)
        : ((kv_dtype_ == DataType::BFLOAT16 || kv_dtype_ == DataType::FLOAT16)
           ? 2 : gpu().compute_dtype_size());
    return static_cast<size_t>(num_kv_heads_) * static_cast<size_t>(head_dim_) * elem * 2;
}

static int64_t tq_token_byte_offset(int64_t global_slot, int64_t block_size,
                                    int num_kv_heads, int slot_size_aligned) {
    const int64_t blk = global_slot / block_size;
    const int64_t off = global_slot % block_size;
    return blk * block_size * num_kv_heads * slot_size_aligned
         + off * num_kv_heads * slot_size_aligned;
}

void KVCacheManager::read_kv_token(int layer, int64_t global_slot, void* dst,
                                   cudaStream_t stream) const {
    if (!dst) {
        throw std::runtime_error("read_kv_token: dst is null");
    }
    const size_t token_bytes = kv_bytes_per_token(layer);
    const int gid = group_id_for_layer(layer);
    const auto& g = groups_[static_cast<size_t>(gid)];
    CudaDeviceGuard guard(gpu_device_);

    if (g.is_tq) {
        auto* cache = static_cast<uint8_t*>(gpu_key_caches_[static_cast<size_t>(layer)]);
        const int64_t base = tq_token_byte_offset(
            global_slot, block_size_, num_kv_heads_, tq_layout_.slot_size_aligned);
        CUDA_CHECK(cudaMemcpyAsync(dst, cache + base, token_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
        return;
    }

    const size_t elem = token_bytes / (static_cast<size_t>(num_kv_heads_) * 2);
    const size_t head_stride = static_cast<size_t>(head_dim_) * elem;
    const size_t token_stride = static_cast<size_t>(num_kv_heads_) * head_stride;
    const int64_t blk = global_slot / block_size_;
    const int64_t off = global_slot % block_size_;
    const size_t block_bytes = g.bytes_per_block;
    auto* k_cache = static_cast<uint8_t*>(gpu_key_caches_[static_cast<size_t>(layer)]);
    auto* v_cache = static_cast<uint8_t*>(gpu_value_caches_[static_cast<size_t>(layer)]);
    const size_t k_off = static_cast<size_t>(blk) * block_bytes + static_cast<size_t>(off) * token_stride;
    const size_t v_off = k_off;
    auto* dst_bytes = static_cast<uint8_t*>(dst);
    CUDA_CHECK(cudaMemcpyAsync(dst_bytes, k_cache + k_off, token_stride,
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dst_bytes + token_stride, v_cache + v_off, token_stride,
                               cudaMemcpyDeviceToDevice, stream));
}

void KVCacheManager::write_kv_token(int layer, int64_t global_slot, const void* src,
                                    cudaStream_t stream) const {
    if (!src) {
        throw std::runtime_error("write_kv_token: src is null");
    }
    const size_t token_bytes = kv_bytes_per_token(layer);
    const int gid = group_id_for_layer(layer);
    const auto& g = groups_[static_cast<size_t>(gid)];
    CudaDeviceGuard guard(gpu_device_);

    if (g.is_tq) {
        auto* cache = static_cast<uint8_t*>(gpu_key_caches_[static_cast<size_t>(layer)]);
        const int64_t base = tq_token_byte_offset(
            global_slot, block_size_, num_kv_heads_, tq_layout_.slot_size_aligned);
        CUDA_CHECK(cudaMemcpyAsync(cache + base, src, token_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
        return;
    }

    const size_t elem = token_bytes / (static_cast<size_t>(num_kv_heads_) * 2);
    const size_t head_stride = static_cast<size_t>(head_dim_) * elem;
    const size_t token_stride = static_cast<size_t>(num_kv_heads_) * head_stride;
    const int64_t blk = global_slot / block_size_;
    const int64_t off = global_slot % block_size_;
    const size_t block_bytes = g.bytes_per_block;
    auto* k_cache = static_cast<uint8_t*>(gpu_key_caches_[static_cast<size_t>(layer)]);
    auto* v_cache = static_cast<uint8_t*>(gpu_value_caches_[static_cast<size_t>(layer)]);
    const size_t k_off = static_cast<size_t>(blk) * block_bytes + static_cast<size_t>(off) * token_stride;
    auto* src_bytes = static_cast<const uint8_t*>(src);
    CUDA_CHECK(cudaMemcpyAsync(k_cache + k_off, src_bytes, token_stride,
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(v_cache + k_off, src_bytes + token_stride, token_stride,
                               cudaMemcpyDeviceToDevice, stream));
}

}
