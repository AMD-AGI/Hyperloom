// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
// Decode-only (1 query/seq) paged attention that reads TRITON_ATTN's FLASH KV layout directly:
//   k_cache / v_cache : [num_blocks, block_size, num_kv_heads, head_size]  (both, after kv_cache.unbind(1))
// so it can replace unified_attention for a pure-decode step WITHOUT a v0 repack and WITHOUT the heavy
// ROCM_ATTN backend path (v0 reshape_and_cache + chunked_prefill wrapper) that made the ROCM_ATTN
// integration regress e2e. Sliding-window aware. fp16/bf16, kv_cache_dtype "auto" only. gfx1100 wave32.
//
// Design: grid (num_q_heads, num_seqs), NUM_THREADS/block.
//  Phase 1 (QK): each WARP handles tokens t=warp,warp+NUM_WARPS,...; lanes split head_size and read
//    K[t] contiguously (COALESCED across lanes), warp-shuffle reduce the dot -> logits[t] in shared.
//  softmax over logits (block reduce max + sum).
//  Phase 2 (PV): each THREAD owns head dims d=tid,tid+NUM_THREADS,...; loops all tokens, reads V[t]
//    contiguously (COALESCED across threads for a fixed token), acc += p*V. Writes out.
#include <torch/extension.h>
#include <torch/library.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <float.h>
#include <algorithm>

#define WARP 32
#define DIVUP(a, b) (((a) + (b) - 1) / (b))

namespace vllmflash {

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int m = WARP / 2; m >= 1; m >>= 1) v += __shfl_xor(v, m);
  return v;
}
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int m = WARP / 2; m >= 1; m >>= 1) v = fmaxf(v, __shfl_xor(v, m));
  return v;
}
__device__ __forceinline__ float to_f(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_f(__hip_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ void from_f(__half& o, float x) { o = __float2half(x); }
__device__ __forceinline__ void from_f(__hip_bfloat16& o, float x) { o = __float2bfloat16(x); }

template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_flash_kernel(
    scalar_t* __restrict__ out,            // [num_seqs, num_heads, HEAD_SIZE]
    const scalar_t* __restrict__ q,        // [num_seqs, num_heads, HEAD_SIZE]
    const scalar_t* __restrict__ k_cache,  // [num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_SIZE]
    const scalar_t* __restrict__ v_cache,  // [num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_SIZE]
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq, const int q_stride,
    const int64_t kv_block_stride, const int64_t kv_token_stride,
    const int sliding_window) {
  constexpr int NUM_WARPS = NUM_THREADS / WARP;
  const int seq_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
  const int seq_len = seq_lens[seq_idx];
  const int tid = threadIdx.x;
  const int warp = tid / WARP, lane = tid % WARP;

  extern __shared__ char smem[];
  float* logits = reinterpret_cast<float*>(smem);  // [seq_len]
  __shared__ float q_sh[HEAD_SIZE];
  __shared__ float red[NUM_WARPS];
  __shared__ float s_bcast;

  const scalar_t* q_ptr = q + (int64_t)seq_idx * q_stride + head_idx * HEAD_SIZE;
  for (int d = tid; d < HEAD_SIZE; d += NUM_THREADS) q_sh[d] = to_f(q_ptr[d]);
  __syncthreads();

  const int* bt = block_tables + seq_idx * max_num_blocks_per_seq;

  // ---- Phase 1: QK with thread groups of TG (=2 for warp32/block16): each group handles a token,
  // its TG threads read contiguous halves of head_size -> NUM_GROUPS (=64) tokens' K loads in flight
  // per iteration -> memory latency is hidden (the warp-per-token version left only NUM_WARPS in flight).
  constexpr int TG = (WARP >= BLOCK_SIZE) ? (WARP / BLOCK_SIZE) : 1;
  constexpr int NUM_GROUPS = NUM_THREADS / TG;
  constexpr int HD_PER = HEAD_SIZE / TG;
  const int grp = tid / TG;
  const int gl = tid % TG;
  const int d_lo = gl * HD_PER;
  float local_max = -FLT_MAX;
  for (int t = grp; t < seq_len; t += NUM_GROUPS) {
    const int64_t blk = bt[t / BLOCK_SIZE];
    const int off = t % BLOCK_SIZE;
    const scalar_t* k_ptr = k_cache + blk * kv_block_stride + (int64_t)off * kv_token_stride +
                            (int64_t)kv_head_idx * HEAD_SIZE + d_lo;
    float part = 0.f;
    constexpr int VELEM = 16 / sizeof(scalar_t);  // 8 (half/bf16); 16-byte coalesced vector load
    constexpr int NVEC = HD_PER / VELEM;
    const int4* k4 = reinterpret_cast<const int4*>(k_ptr);
#pragma unroll
    for (int v = 0; v < NVEC; v++) {
      int4 raw = k4[v];
      const scalar_t* e = reinterpret_cast<const scalar_t*>(&raw);
#pragma unroll
      for (int j = 0; j < VELEM; j++) part += q_sh[d_lo + v * VELEM + j] * to_f(e[j]);
    }
#pragma unroll
    for (int o = 1; o < TG; o <<= 1) part += __shfl_xor(part, o);  // sum the TG partials
    float qk = part * scale;
    if (sliding_window > 0 && (seq_len - 1 - t) >= sliding_window) qk = -FLT_MAX;
    if (gl == 0) logits[t] = qk;
    local_max = fmaxf(local_max, qk);
  }
  // block-reduce max -> broadcast to ALL threads via shared (not just warp 0)
  local_max = warp_reduce_max(local_max);
  if (lane == 0) red[warp] = local_max;
  __syncthreads();
  if (tid == 0) {
    float m = -FLT_MAX;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; w++) m = fmaxf(m, red[w]);
    s_bcast = m;
  }
  __syncthreads();
  const float qk_max = s_bcast;
  __syncthreads();
  // exp + sum
  float local_sum = 0.f;
  for (int t = tid; t < seq_len; t += NUM_THREADS) {
    float e = __expf(logits[t] - qk_max);
    logits[t] = e;
    local_sum += e;
  }
  local_sum = warp_reduce_sum(local_sum);
  if (lane == 0) red[warp] = local_sum;
  __syncthreads();
  if (tid == 0) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; w++) s += red[w];
    s_bcast = s;
  }
  __syncthreads();
  const float inv = 1.f / (s_bcast + 1e-6f);
  __syncthreads();

  // ---- Phase 2: PV (thread owns head dims, loops tokens; coalesced V read) ----
  constexpr int NE = DIVUP(HEAD_SIZE, NUM_THREADS);
  constexpr int U = 4;  // unroll: issue U V-loads before consuming -> hide memory latency
  float acc[NE];
#pragma unroll
  for (int i = 0; i < NE; i++) acc[i] = 0.f;
  for (int t0 = 0; t0 < seq_len; t0 += U) {
    const scalar_t* vp[U];
    float pp[U];
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int t = t0 + u;
      if (t < seq_len) {
        vp[u] = v_cache + bt[t / BLOCK_SIZE] * kv_block_stride +
                (int64_t)(t % BLOCK_SIZE) * kv_token_stride + (int64_t)kv_head_idx * HEAD_SIZE;
        pp[u] = logits[t] * inv;
      } else {
        vp[u] = nullptr;
      }
    }
#pragma unroll
    for (int i = 0; i < NE; i++) {
      const int d = tid + i * NUM_THREADS;
      if (d < HEAD_SIZE) {
        float vv[U];
#pragma unroll
        for (int u = 0; u < U; u++) vv[u] = vp[u] ? to_f(vp[u][d]) : 0.f;  // U loads in flight
#pragma unroll
        for (int u = 0; u < U; u++) acc[i] += pp[u] * vv[u];
      }
    }
  }
  scalar_t* out_ptr = out + (int64_t)seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = 0; i < NE; i++) {
    const int d = tid + i * NUM_THREADS;
    if (d < HEAD_SIZE) from_f(out_ptr[d], acc[i]);
  }
}

}  // namespace vllmflash

#define LAUNCH(scalar_t, HS)                                                       \
  vllmflash::paged_attention_flash_kernel<scalar_t, HS, 16, 128>                   \
      <<<grid, 128, shmem, stream>>>(                                              \
          reinterpret_cast<scalar_t*>(out.data_ptr()),                            \
          reinterpret_cast<const scalar_t*>(query.data_ptr()),                    \
          reinterpret_cast<const scalar_t*>(key_cache.data_ptr()),                \
          reinterpret_cast<const scalar_t*>(value_cache.data_ptr()),              \
          num_kv_heads, (float)scale, block_tables.data_ptr<int>(),               \
          seq_lens.data_ptr<int>(), max_num_blocks, q_stride, kv_block_stride,    \
          kv_token_stride, (int)sliding_window);

void paged_attention_flash(torch::Tensor& out, torch::Tensor& query,
                           torch::Tensor& key_cache, torch::Tensor& value_cache,
                           int64_t num_kv_heads, double scale,
                           torch::Tensor& block_tables, torch::Tensor& seq_lens,
                           int64_t block_size, int64_t max_seq_len, int64_t sliding_window) {
  const int num_seqs = query.size(0);
  const int num_heads = query.size(1);
  const int head_size = query.size(2);
  const int max_num_blocks = block_tables.size(1);
  const int q_stride = query.stride(0);
  const int64_t kv_block_stride = key_cache.stride(0);
  const int64_t kv_token_stride = key_cache.stride(1);
  TORCH_CHECK(block_size == 16, "flash kernel: block_size must be 16");
  const int padded = DIVUP(max_seq_len, 16) * 16;
  const int shmem = padded * sizeof(float);
  dim3 grid(num_heads, num_seqs, 1);
  // single-GPU (device 0); tensors present as "cuda" (masqueraded HIP) so a c10::hip::HIPGuard rejects
  // the device type -- skip it. getCurrentHIPStream() still returns the active (capture) stream.
  const hipStream_t stream = c10::hip::getCurrentHIPStream(query.device().index());
  if (query.dtype() == torch::kFloat16) {
    if (head_size == 128) { LAUNCH(__half, 128); }
    else if (head_size == 256) { LAUNCH(__half, 256); }
    else { TORCH_CHECK(false, "flash kernel: head_size ", head_size); }
  } else if (query.dtype() == torch::kBFloat16) {
    if (head_size == 128) { LAUNCH(__hip_bfloat16, 128); }
    else if (head_size == 256) { LAUNCH(__hip_bfloat16, 256); }
    else { TORCH_CHECK(false, "flash kernel: head_size ", head_size); }
  } else {
    TORCH_CHECK(false, "flash kernel: dtype must be fp16/bf16");
  }
}

TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "paged_attention_flash(Tensor! out, Tensor query, Tensor key_cache, Tensor value_cache,"
      " int num_kv_heads, float scale, Tensor block_tables, Tensor seq_lens, int block_size,"
      " int max_seq_len, int sliding_window) -> ()");
  m.impl("paged_attention_flash", torch::kCUDA, &paged_attention_flash);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
