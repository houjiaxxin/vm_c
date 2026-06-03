#include "vm_c/cuda/kernels_sampler.h"
#include "vm_c/cuda/vm_c_dispatch.h"
#include "vm_c/cuda/kernel_utils.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <cfloat>
#include <cmath>
#include <algorithm>

namespace vm_c {
namespace {

static constexpr int kBlockDim = 256;
static constexpr int kHistBins = 2048;
static constexpr int kMaxCandidates = 2048;

template <typename scalar_t>
__device__ __forceinline__ float scaled_logit_prob(
    float logit, float inv_temp, float max_logit) {
  return expf(logit * inv_temp - max_logit);
}

template <typename scalar_t>
__device__ void greedy_sample_seq(
    const scalar_t* seq_logits, int vocab_size, int tid,
    int32_t& out_id, float& out_lp) {
  float best_val = -FLT_MAX;
  int best_idx = 0;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    const float val = static_cast<float>(seq_logits[i]);
    if (val > best_val) {
      best_val = val;
      best_idx = i;
    }
  }
  using BlockReduce = cub::BlockReduce<cub::KeyValuePair<int, float>, kBlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  cub::KeyValuePair<int, float> thread_kvp;
  thread_kvp.key = best_idx;
  thread_kvp.value = best_val;
  const auto result = BlockReduce(temp_storage).Reduce(thread_kvp, cub::ArgMax());
  if (tid == 0) {
    out_id = result.key;
    out_lp = 0.0f;
  }
}

// 全词表 temperature 采样：两遍 softmax + 并行 inverse CDF，无固定候选缓冲上限
template <typename scalar_t>
__device__ void full_multinomial_sample_seq(
    const scalar_t* seq_logits, int vocab_size, int tid, int seq_idx,
    float temperature, float min_p, unsigned long long seed,
    int32_t& out_id, float& out_lp) {
  using BlockReduce = cub::BlockReduce<float, kBlockDim>;
  using BlockScan = cub::BlockScan<float, kBlockDim>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ float s_max;
  __shared__ float s_sum;
  __shared__ float s_min_p_threshold;
  __shared__ float s_r;
  __shared__ int s_found_tid;
  __shared__ int s_out_id;
  __shared__ float s_out_lp;

  if (temperature < 1e-5f) {
    greedy_sample_seq(seq_logits, vocab_size, tid, out_id, out_lp);
    return;
  }

  const float inv_temp = 1.0f / temperature;

  float max_logit = -FLT_MAX;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    max_logit = fmaxf(max_logit, static_cast<float>(seq_logits[i]));
  }
  max_logit = BlockReduce(reduce_storage).Reduce(max_logit, VMC_MAX_OP);
  if (tid == 0) {
    s_max = max_logit;
  }
  __syncthreads();

  float thread_prob_sum = 0.0f;
  float thread_max_prob = 0.0f;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    const float p = scaled_logit_prob<scalar_t>(
        static_cast<float>(seq_logits[i]), inv_temp, s_max);
    thread_prob_sum += p;
    thread_max_prob = fmaxf(thread_max_prob, p);
  }
  const float raw_sum = BlockReduce(reduce_storage).Sum(thread_prob_sum);
  const float global_max_prob = BlockReduce(reduce_storage).Reduce(
      thread_max_prob, VMC_MAX_OP);
  if (tid == 0) {
    s_min_p_threshold = (min_p > 0.0f)
        ? min_p * global_max_prob / fmaxf(raw_sum, 1e-10f)
        : 0.0f;
  }
  __syncthreads();

  thread_prob_sum = 0.0f;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    const float p = scaled_logit_prob<scalar_t>(
        static_cast<float>(seq_logits[i]), inv_temp, s_max);
    if (min_p > 0.0f && p / fmaxf(raw_sum, 1e-10f) < s_min_p_threshold) {
      continue;
    }
    thread_prob_sum += p;
  }
  const float filtered_sum = BlockReduce(reduce_storage).Sum(thread_prob_sum);
  if (tid == 0) {
    s_sum = filtered_sum;
    if (!(s_sum > 0.0f) || !isfinite(s_sum)) {
      s_sum = 0.0f;
    }
  }
  __syncthreads();

  if (s_sum <= 0.0f) {
    greedy_sample_seq(seq_logits, vocab_size, tid, out_id, out_lp);
    return;
  }

  if (tid == 0) {
    curandState rng;
    curand_init(seed + static_cast<unsigned long long>(seq_idx), 0, 0, &rng);
    s_r = curand_uniform(&rng) * s_sum;
  }
  __syncthreads();

  thread_prob_sum = 0.0f;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    const float p = scaled_logit_prob<scalar_t>(
        static_cast<float>(seq_logits[i]), inv_temp, s_max);
    if (min_p > 0.0f && p / fmaxf(raw_sum, 1e-10f) < s_min_p_threshold) {
      continue;
    }
    thread_prob_sum += p;
  }

  float thread_prefix = 0.0f;
  BlockScan(scan_storage).ExclusiveSum(thread_prob_sum, thread_prefix);

  __shared__ float s_prefix[kBlockDim];
  __shared__ float s_thread_sum[kBlockDim];
  s_prefix[tid] = thread_prefix;
  s_thread_sum[tid] = thread_prob_sum;
  __syncthreads();

  if (tid == 0) {
    s_found_tid = blockDim.x - 1;
    for (int t = 0; t < blockDim.x; ++t) {
      if (s_prefix[t] + s_thread_sum[t] > s_r) {
        s_found_tid = t;
        break;
      }
    }
  }
  __syncthreads();

  if (tid == s_found_tid) {
    float cum = s_prefix[tid];
    int chosen = 0;
    float chosen_prob = 1e-10f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
      const float p = scaled_logit_prob<scalar_t>(
          static_cast<float>(seq_logits[i]), inv_temp, s_max);
      if (min_p > 0.0f && p / fmaxf(raw_sum, 1e-10f) < s_min_p_threshold) {
        continue;
      }
      cum += p;
      chosen = i;
      chosen_prob = p / s_sum;
      if (cum > s_r) {
        break;
      }
    }
    s_out_id = chosen;
    s_out_lp = logf(fmaxf(chosen_prob, 1e-10f));
  }
  __syncthreads();

  if (tid == 0) {
    out_id = s_out_id;
    out_lp = s_out_lp;
  }
}

template <typename scalar_t>
__global__ void apply_penalties_kernel(
    scalar_t* __restrict__ logits,
    const uint8_t* __restrict__ prompt_mask,
    const int32_t* __restrict__ output_bin_counts,
    const scalar_t* __restrict__ repetition_penalties,
    const scalar_t* __restrict__ frequency_penalties,
    const scalar_t* __restrict__ presence_penalties,
    const int num_seqs, const int vocab_size, const int tile_size) {
  const int seq_idx = blockIdx.x;
  if (seq_idx >= num_seqs) return;
  const int tile_start = blockIdx.y * tile_size;
  const int tile_end = min(tile_start + tile_size, vocab_size);
  const scalar_t rep_p = repetition_penalties[seq_idx];
  const scalar_t freq_p = frequency_penalties[seq_idx];
  const scalar_t pres_p = presence_penalties[seq_idx];
  const bool use_rep = static_cast<float>(rep_p) != 1.0f;
  const bool use_freq = static_cast<float>(freq_p) != 0.0f;
  const bool use_pres = static_cast<float>(pres_p) != 0.0f;
  if (!use_rep && !use_freq && !use_pres) return;

  for (int vocab_idx = tile_start + threadIdx.x; vocab_idx < tile_end;
       vocab_idx += blockDim.x) {
    const int64_t idx = static_cast<int64_t>(seq_idx) * vocab_size + vocab_idx;
    float logit = static_cast<float>(logits[idx]);
    if (use_rep) {
      const bool in_prompt = prompt_mask[idx];
      const bool in_output = output_bin_counts[idx] > 0;
      if (in_prompt || in_output) {
        logit = (logit > 0.0f) ? logit / static_cast<float>(rep_p)
                               : logit * static_cast<float>(rep_p);
      }
    }
    if (use_freq) {
      logit -= static_cast<float>(freq_p) *
               static_cast<float>(output_bin_counts[idx]);
    }
    if (use_pres && output_bin_counts[idx] > 0) {
      logit -= static_cast<float>(pres_p);
    }
    logits[idx] = static_cast<scalar_t>(logit);
  }
}

template <typename scalar_t>
__global__ void greedy_sampling_kernel(
    const scalar_t* __restrict__ logits, const int* __restrict__ logit_offsets,
    int32_t* __restrict__ output_ids, float* __restrict__ output_logprobs,
    const int vocab_size) {
  const int seq_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int offset = logit_offsets ? logit_offsets[seq_idx] : seq_idx;
  const scalar_t* seq_logits =
      logits + static_cast<int64_t>(offset) * vocab_size;
  int32_t out_id = 0;
  float out_lp = 0.0f;
  greedy_sample_seq(seq_logits, vocab_size, tid, out_id, out_lp);
  if (tid == 0) {
    output_ids[seq_idx] = out_id;
    output_logprobs[seq_idx] = out_lp;
  }
}

template <typename scalar_t>
__global__ void full_multinomial_sampling_kernel(
    const scalar_t* __restrict__ logits, const int* __restrict__ logit_offsets,
    int32_t* __restrict__ output_ids, float* __restrict__ output_logprobs,
    const int vocab_size, const float* __restrict__ temperatures,
    const float* __restrict__ min_p, unsigned long long seed) {
  const int seq_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int offset = logit_offsets ? logit_offsets[seq_idx] : seq_idx;
  const scalar_t* seq_logits =
      logits + static_cast<int64_t>(offset) * vocab_size;
  const float temperature = temperatures[seq_idx];
  const float m = min_p ? min_p[seq_idx] : 0.0f;
  int32_t out_id = 0;
  float out_lp = 0.0f;
  full_multinomial_sample_seq(
      seq_logits, vocab_size, tid, seq_idx, temperature, m, seed, out_id, out_lp);
  if (tid == 0) {
    output_ids[seq_idx] = out_id;
    output_logprobs[seq_idx] = out_lp;
  }
}

// top_k / top_p / min_p 过滤采样：候选集上限 kMaxCandidates
template <typename scalar_t>
__global__ void filtered_sampling_kernel(
    const scalar_t* __restrict__ logits, const int* __restrict__ logit_offsets,
    int32_t* __restrict__ output_ids, float* __restrict__ output_logprobs,
    const int vocab_size, const float* __restrict__ temperatures,
    const int* __restrict__ top_k, const float* __restrict__ top_p,
    const float* __restrict__ min_p, unsigned long long seed) {
  const int seq_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int offset = logit_offsets ? logit_offsets[seq_idx] : seq_idx;
  const scalar_t* seq_logits =
      logits + static_cast<int64_t>(offset) * vocab_size;
  const float temperature = temperatures[seq_idx];
  const int k = top_k ? top_k[seq_idx] : 0;
  const float p = top_p ? top_p[seq_idx] : 1.0f;
  const float m = min_p ? min_p[seq_idx] : 0.0f;

  const bool use_top_k = k > 0 && k < vocab_size;
  const bool use_top_p = p < 1.0f - 1e-6f;
  const bool use_min_p = m > 0.0f;

  if (!use_top_k && !use_top_p && !use_min_p) {
    int32_t out_id = 0;
    float out_lp = 0.0f;
    full_multinomial_sample_seq(
        seq_logits, vocab_size, tid, seq_idx, temperature, 0.0f, seed,
        out_id, out_lp);
    if (tid == 0) {
      output_ids[seq_idx] = out_id;
      output_logprobs[seq_idx] = out_lp;
    }
    return;
  }

  if (temperature < 1e-5f) {
    int32_t out_id = 0;
    float out_lp = 0.0f;
    greedy_sample_seq(seq_logits, vocab_size, tid, out_id, out_lp);
    if (tid == 0) {
      output_ids[seq_idx] = out_id;
      output_logprobs[seq_idx] = out_lp;
    }
    return;
  }

  __shared__ float s_max_logit, s_sum_exp, s_max_prob, s_threshold;
  __shared__ int s_hist[kHistBins];
  __shared__ float s_cumsum[kMaxCandidates];
  __shared__ int s_cand_indices[kMaxCandidates];

  const float inv_temp = 1.0f / temperature;

  float max_logit = -FLT_MAX;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    max_logit = fmaxf(max_logit, static_cast<float>(seq_logits[i]));
  }
  using BlockReduce = cub::BlockReduce<float, kBlockDim>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  max_logit = BlockReduce(temp_storage).Reduce(max_logit, VMC_MAX_OP);
  if (tid == 0) {
    s_max_logit = max_logit;
  }
  __syncthreads();

  float sum_exp = 0.0f;
  float max_prob = 0.0f;
  for (int b = tid; b < kHistBins; b += blockDim.x) {
    s_hist[b] = 0;
  }
  __syncthreads();
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    const float prob = scaled_logit_prob<scalar_t>(
        static_cast<float>(seq_logits[i]), inv_temp, s_max_logit);
    sum_exp += prob;
    max_prob = fmaxf(max_prob, prob);
  }
  sum_exp = BlockReduce(temp_storage).Sum(sum_exp);
  max_prob = BlockReduce(temp_storage).Reduce(max_prob, VMC_MAX_OP);
  if (tid == 0) {
    s_sum_exp = sum_exp;
    s_max_prob = max_prob;
  }
  __syncthreads();

  const float inv_sum = 1.0f / fmaxf(s_sum_exp, 1e-10f);
  const float min_p_threshold = m * s_max_prob * inv_sum;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    const float prob = scaled_logit_prob<scalar_t>(
        static_cast<float>(seq_logits[i]), inv_temp, s_max_logit) * inv_sum;
    if (use_min_p && prob < min_p_threshold) {
      continue;
    }
    const int bin = max(
        0, min(kHistBins - 1, static_cast<int>(prob * (kHistBins - 1))));
    atomicAdd(&s_hist[bin], 1);
  }
  __syncthreads();

  const int requested_k = use_top_k ? k : kMaxCandidates;
  const int effective_k = min(requested_k, kMaxCandidates);

  if (tid == 0) {
    int cum_count = 0;
    int threshold_bin = 0;
    for (int b = kHistBins - 1; b >= 0; --b) {
      cum_count += s_hist[b];
      if (cum_count >= effective_k) {
        threshold_bin = b;
        break;
      }
    }
    s_threshold = (threshold_bin + 0.5f) / static_cast<float>(kHistBins);
  }
  __syncthreads();

  __shared__ int s_thread_counts[kBlockDim];
  __shared__ int s_thread_offsets[kBlockDim + 1];
  int my_count = 0;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    const float prob = scaled_logit_prob<scalar_t>(
        static_cast<float>(seq_logits[i]), inv_temp, s_max_logit) * inv_sum;
    if (use_min_p && prob < min_p_threshold) {
      continue;
    }
    if (prob < s_threshold) {
      continue;
    }
    my_count++;
  }
  s_thread_counts[tid] = my_count;
  __syncthreads();
  if (tid == 0) {
    int off = 0;
    for (int t = 0; t < blockDim.x; ++t) {
      s_thread_offsets[t] = off;
      if (off >= kMaxCandidates) {
        continue;
      }
      const int take = min(s_thread_counts[t], kMaxCandidates - off);
      off += take;
    }
    s_thread_offsets[blockDim.x] = min(off, kMaxCandidates);
  }
  __syncthreads();

  const int my_offset = s_thread_offsets[tid];
  const int my_limit = min(
      s_thread_offsets[tid + 1] - my_offset,
      kMaxCandidates - my_offset);
  const int total_candidates = s_thread_offsets[blockDim.x];
  int wp = my_offset;
  int collected = 0;
  for (int i = tid; i < vocab_size; i += blockDim.x) {
    if (collected >= my_limit || wp >= kMaxCandidates) {
      break;
    }
    const float prob = scaled_logit_prob<scalar_t>(
        static_cast<float>(seq_logits[i]), inv_temp, s_max_logit) * inv_sum;
    if (use_min_p && prob < min_p_threshold) {
      continue;
    }
    if (prob < s_threshold) {
      continue;
    }
    if (wp >= kMaxCandidates) {
      break;
    }
    s_cumsum[wp] = prob;
    s_cand_indices[wp] = i;
    wp++;
    collected++;
  }
  __syncthreads();

  if (total_candidates <= 0) {
    int32_t out_id = 0;
    float out_lp = 0.0f;
    greedy_sample_seq(seq_logits, vocab_size, tid, out_id, out_lp);
    if (tid == 0) {
      output_ids[seq_idx] = out_id;
      output_logprobs[seq_idx] = out_lp;
    }
    return;
  }

  if (tid == 0) {
    const int capped = min(total_candidates, kMaxCandidates);
    for (int i = 1; i < capped; ++i) {
      const float kp = s_cumsum[i];
      const int ki = s_cand_indices[i];
      int j = i - 1;
      while (j >= 0 && s_cumsum[j] < kp) {
        s_cumsum[j + 1] = s_cumsum[j];
        s_cand_indices[j + 1] = s_cand_indices[j];
        --j;
      }
      s_cumsum[j + 1] = kp;
      s_cand_indices[j + 1] = ki;
    }

    float cum = 0.0f;
    int cutoff = capped;
    if (use_top_p) {
      for (int i = 0; i < capped; ++i) {
        cum += s_cumsum[i];
        if (cum >= p && cutoff == capped) {
          cutoff = i + 1;
        }
      }
      if (cutoff == capped && p < 1.0f - 1e-6f) {
        cutoff = max(1, capped);
      }
    } else if (use_top_k) {
      cutoff = min(capped, k);
    }

    float rsum = 0.0f;
    for (int i = 0; i < cutoff; ++i) {
      rsum += s_cumsum[i];
    }
    const float ri = 1.0f / fmaxf(rsum, 1e-10f);
    for (int i = 0; i < cutoff; ++i) {
      s_cumsum[i] *= ri;
    }
    float run = 0.0f;
    for (int i = 0; i < cutoff; ++i) {
      run += s_cumsum[i];
      s_cumsum[i] = run;
    }

    curandState rng;
    curand_init(seed + static_cast<unsigned long long>(seq_idx), 0, 0, &rng);
    const float r = curand_uniform(&rng);
    int fi = s_cand_indices[cutoff - 1];
    float fp = s_cumsum[cutoff - 1] -
               (cutoff > 1 ? s_cumsum[cutoff - 2] : 0.0f);
    for (int i = 0; i < cutoff; ++i) {
      if (r <= s_cumsum[i]) {
        fi = s_cand_indices[i];
        fp = (i > 0) ? (s_cumsum[i] - s_cumsum[i - 1]) : s_cumsum[0];
        break;
      }
    }
    output_ids[seq_idx] = fi;
    output_logprobs[seq_idx] = logf(fmaxf(fp, 1e-10f));
  }
}

template <typename scalar_t>
void launch_apply_penalties(
    scalar_t* logits, const uint8_t* prompt_mask, const int32_t* output_bin_counts,
    const scalar_t* rep_penalties, const scalar_t* freq_penalties,
    const scalar_t* pres_penalties,
    int num_seqs, int vocab_size, int tile_size, cudaStream_t stream) {
  dim3 grid(num_seqs, (vocab_size + tile_size - 1) / tile_size);
  dim3 block(tile_size);
  apply_penalties_kernel<scalar_t><<<grid, block, 0, stream>>>(
      logits, prompt_mask, output_bin_counts, rep_penalties, freq_penalties,
      pres_penalties, num_seqs, vocab_size, tile_size);
}

template <typename scalar_t>
void launch_greedy_sampling(
    const scalar_t* logits, const int* logit_offsets,
    int32_t* output_ids, float* output_logprobs,
    int batch_size, int vocab_size, cudaStream_t stream) {
  greedy_sampling_kernel<scalar_t><<<batch_size, kBlockDim, 0, stream>>>(
      logits, logit_offsets, output_ids, output_logprobs, vocab_size);
}

template <typename scalar_t>
void launch_full_multinomial_sampling(
    const scalar_t* logits, const int* logit_offsets,
    int32_t* output_ids, float* output_logprobs,
    int batch_size, int vocab_size,
    const float* temperatures, const float* min_p,
    unsigned long long seed, cudaStream_t stream) {
  full_multinomial_sampling_kernel<scalar_t><<<batch_size, kBlockDim, 0, stream>>>(
      logits, logit_offsets, output_ids, output_logprobs, vocab_size,
      temperatures, min_p, seed);
}

template <typename scalar_t>
void launch_filtered_sampling(
    const scalar_t* logits, const int* logit_offsets,
    int32_t* output_ids, float* output_logprobs,
    int batch_size, int vocab_size,
    const float* temperatures, const int* top_k, const float* top_p,
    const float* min_p, unsigned long long seed, cudaStream_t stream) {
  filtered_sampling_kernel<scalar_t><<<batch_size, kBlockDim, 0, stream>>>(
      logits, logit_offsets, output_ids, output_logprobs, vocab_size,
      temperatures, top_k, top_p, min_p, seed);
}

}  // anonymous namespace

void apply_penalties(
    void* logits, const void* prompt_mask, const void* output_bin_counts,
    const void* repetition_penalties, const void* frequency_penalties,
    const void* presence_penalties,
    int num_seqs, int vocab_size, ScalarType dtype, cudaStream_t stream) {
  const int tile_size = 256;
  VMC_DISPATCH_FLOATING_TYPES(dtype, "apply_penalties",
    launch_apply_penalties<scalar_t>(
        static_cast<scalar_t*>(logits),
        static_cast<const uint8_t*>(prompt_mask),
        static_cast<const int32_t*>(output_bin_counts),
        static_cast<const scalar_t*>(repetition_penalties),
        static_cast<const scalar_t*>(frequency_penalties),
        static_cast<const scalar_t*>(presence_penalties),
        num_seqs, vocab_size, tile_size, stream);
  );
}

void gpu_greedy_sampling(
    const void* logits, const int* logit_offsets,
    int32_t* output_ids, float* output_logprobs,
    int batch_size, int vocab_size, ScalarType dtype, cudaStream_t stream) {
  VMC_DISPATCH_FLOATING_TYPES(dtype, "gpu_greedy_sampling",
    launch_greedy_sampling<scalar_t>(
        static_cast<const scalar_t*>(logits),
        logit_offsets, output_ids, output_logprobs,
        batch_size, vocab_size, stream);
  );
}

void gpu_temperature_sampling(
    const void* logits, const int* logit_offsets,
    int32_t* output_ids, float* output_logprobs,
    const float* temperatures, int batch_size, int vocab_size,
    ScalarType dtype, unsigned long long seed, cudaStream_t stream) {
  VMC_DISPATCH_FLOATING_TYPES(dtype, "gpu_temperature_sampling",
    launch_full_multinomial_sampling<scalar_t>(
        static_cast<const scalar_t*>(logits),
        logit_offsets, output_ids, output_logprobs,
        batch_size, vocab_size, temperatures, nullptr, seed, stream);
  );
}

void gpu_filtered_sampling(
    const void* logits, const int* logit_offsets,
    int32_t* output_ids, float* output_logprobs,
    const float* temperatures, const int* top_k, const float* top_p,
    const float* min_p, int batch_size, int vocab_size,
    ScalarType dtype, unsigned long long seed, cudaStream_t stream) {
  VMC_DISPATCH_FLOATING_TYPES(dtype, "gpu_filtered_sampling",
    launch_filtered_sampling<scalar_t>(
        static_cast<const scalar_t*>(logits),
        logit_offsets, output_ids, output_logprobs,
        batch_size, vocab_size,
        temperatures, top_k, top_p, min_p, seed, stream);
  );
}

void top_k_per_row_prefill(
    const void* logits, const void* row_starts, const void* row_ends,
    void* indices, int64_t num_rows, int64_t stride0, int64_t stride1,
    int64_t top_k, cudaStream_t stream) {}
void top_k_per_row_decode(
    const void* logits, int64_t next_n, const void* seq_lens,
    void* indices, int64_t num_rows, int64_t stride0, int64_t stride1,
    int64_t top_k, cudaStream_t stream) {}
void persistent_topk(
    const void* logits, const void* lengths,
    void* output, void* workspace,
    int64_t k, int64_t max_seq_len, int64_t num_elements, cudaStream_t stream) {}

}  // namespace vm_c
