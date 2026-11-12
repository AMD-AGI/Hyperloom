// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
// moe_decode_w4 — fused-expert W4A16 MoE decode (M=1) for RDNA3 (gfx1100).
// Strategy: "block-tile-ldsx".
//   Grid (E, ceil(N/TILE)); each block handles a TILE of output rows for ONE expert.
//   threads = TILE  => THREAD-PER-OUTPUT-ROW (each thread owns one out-row's contiguous K/2 bytes).
//   x (kernel A) / act[E,I] (kernel B) staged into LDS once per block, reused by all TILE threads.
//   dwordx4 nontemporal loads, K-loop unrolled to keep loads in flight; fp32 accumulate.
//
// Layout (OUT-MAJOR [N, K//2] uint8, packed along K, 2 nibbles/byte, low nibble = even k):
//   nibble(n,k) = (W[e,n,k>>1] >> ((k&1)*4)) & 0xF
//   value(n,k)  = (nibble(n,k) - bias) * scale[e, n, k>>5]   (G=32, symmetric)
// G=32 == 32 nibbles == exactly one u32x4 (16 bytes) == one scale group: 1 scale load per dwordx4.
//
// Kernel A (gate_up + GeGLU): out-rows n in [0,I). thread computes gate row n and up row n+I
//   against the same x in LDS, then act[e,n] = gelu_tanh(gate) * up.  K = H.
// Kernel B (down + weighted sum): out-rows h in [0,H). thread loops all E experts in-block,
//   acc += topk_w[e] * (W2[e,h] . act[e]).  K = I.  all act[E,I] staged in LDS. writes y[h] directly.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <torch/extension.h>
#include <c10/hip/HIPStream.h>

using u32x4 = __attribute__((ext_vector_type(4))) unsigned int;

#define TILE     256   // output rows per block == threads per block
#define UNROLL_A 3     // K=H=2816 -> 88 groups/row
#define UNROLL_B 2     // K=I=704  -> 22 groups/row

// --------------------------------------------------------------------------------------------------
// One thread computes the dot product of ONE quantized out-row (bytes base) against the fp32 vector
// `xs` (in LDS), K contraction length, NG = K/32 groups. scale_ptr indexed by group g (contiguous).
// Streaming dwordx4 loads, UNROLL-ahead buffering to cover DRAM latency. Symmetric dequant (nib-bias).
// --------------------------------------------------------------------------------------------------
template <int UNROLL>
__device__ __forceinline__ float dot_w4_row(const unsigned char* __restrict__ base,
                                            const __hip_bfloat16* __restrict__ scale_ptr,
                                            const float* __restrict__ xs, int NG, float fbias) {
  float acc = 0.f;
  int g = 0;
  for (; g + UNROLL <= NG; g += UNROLL) {
    u32x4 buf[UNROLL];
    float  sc[UNROLL];
    // issue all UNROLL streaming loads first -> in-order vmcnt drains while ALU runs on buf[0..]
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      buf[u] = __builtin_nontemporal_load((const u32x4*)(base + (size_t)(g + u) * 16));
      sc[u]  = __bfloat162float(scale_ptr[g + u]);
    }
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int   k0   = (g + u) * 32;
      float gsum = 0.f;
      #pragma unroll
      for (int w = 0; w < 4; ++w) {                 // 4 words, 8 nibbles each
        unsigned int word = buf[u][w];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {               // low nibble first == k ascending
          int   nib = (int)((word >> (j * 4)) & 0xF);
          gsum += xs[k0 + w * 8 + j] * ((float)nib - fbias);
        }
      }
      acc += gsum * sc[u];                           // one scale per group
    }
  }
  // tail groups (NG not a multiple of UNROLL). K%32==0 guaranteed, so no partial-group tail.
  for (; g < NG; ++g) {
    u32x4 buf = __builtin_nontemporal_load((const u32x4*)(base + (size_t)g * 16));
    float sc  = __bfloat162float(scale_ptr[g]);
    int   k0  = g * 32;
    float gsum = 0.f;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
      unsigned int word = buf[w];
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        int nib = (int)((word >> (j * 4)) & 0xF);
        gsum += xs[k0 + w * 8 + j] * ((float)nib - fbias);
      }
    }
    acc += gsum * sc;
  }
  return acc;
}

__device__ __forceinline__ float gelu_tanh(float v) {
  float t = 0.7978845608f * (v + 0.044715f * v * v * v);
  return 0.5f * v * (1.f + tanhf(t));
}

// --------------------------------------------------------------------------------------------------
// Kernel A: gate_up + GeGLU.  grid=(E, ceil(I/TILE)), block=TILE.
//   x in LDS (fp32, H elems).  thread owns out-row n in [0,I); computes gate=W13[e,n].x, up=W13[e,n+I].x.
//   act[e,n] = gelu_tanh(gate) * up  (bf16).
// --------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(TILE)
moe_gate_up_kernel(const __hip_bfloat16* __restrict__ x,          // [H]
                   const unsigned char*  __restrict__ w13,        // [E, 2I, H/2]
                   const __hip_bfloat16* __restrict__ w13_scale,  // [E, 2I, H/G]
                   __hip_bfloat16*       __restrict__ act,        // [E, I]
                   int H, int I, float fbias) {
  extern __shared__ float xs[];                     // H fp32
  const int e  = blockIdx.x;
  const int n  = blockIdx.y * TILE + threadIdx.x;   // out-row in [0, I)

  // cooperatively stage x into LDS (fp32), reused by all TILE threads (K=H reads/row)
  for (int i = threadIdx.x; i < H; i += TILE)
    xs[i] = __bfloat162float(x[i]);
  __syncthreads();

  if (n >= I) return;

  const int    NG   = H >> 5;                        // groups per row = H/32
  const size_t rowB = (size_t)(H >> 1);              // bytes per row = H/2
  const size_t sRow = (size_t)NG;                    // scales per row = H/G

  // gate row = n ; up row = n + I. both within expert e's W13 block [2I, H/2].
  const unsigned char*  gBase  = w13 + ((size_t)e * (2 * I) + n)     * rowB;
  const unsigned char*  uBase  = w13 + ((size_t)e * (2 * I) + n + I) * rowB;
  const __hip_bfloat16* gScale = w13_scale + ((size_t)e * (2 * I) + n)     * sRow;
  const __hip_bfloat16* uScale = w13_scale + ((size_t)e * (2 * I) + n + I) * sRow;

  float gate = dot_w4_row<UNROLL_A>(gBase, gScale, xs, NG, fbias);
  float up   = dot_w4_row<UNROLL_A>(uBase, uScale, xs, NG, fbias);

  act[(size_t)e * I + n] = __float2bfloat16(gelu_tanh(gate) * up);
}

// --------------------------------------------------------------------------------------------------
// Kernel B: down + weighted sum.  grid=(1, ceil(H/TILE)), block=TILE.
//   all act[E,I] in LDS (fp32).  thread owns out-row h in [0,H); loops all E experts:
//   y[h] = sum_e topk_w[e] * (W2[e,h] . act[e])   (bf16, direct write, no atomics).
// --------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(TILE)
moe_down_kernel(const __hip_bfloat16* __restrict__ act,          // [E, I]
                const unsigned char*  __restrict__ w2,           // [E, H, I/2]
                const __hip_bfloat16* __restrict__ w2_scale,     // [E, H, I/G]
                const __hip_bfloat16* __restrict__ topk_weights, // [E]
                __hip_bfloat16*       __restrict__ y,            // [H]
                int H, int I, int E, float fbias) {
  extern __shared__ float acts[];                    // E*I fp32
  const int h = blockIdx.y * TILE + threadIdx.x;     // out-row in [0, H)

  // stage all act[E,I] into LDS (fp32), reused across all E-loops of every thread
  const int EI = E * I;
  for (int i = threadIdx.x; i < EI; i += TILE)
    acts[i] = __bfloat162float(act[i]);
  __syncthreads();

  if (h >= H) return;

  const int    NG   = I >> 5;                        // groups per row = I/32
  const size_t rowB = (size_t)(I >> 1);              // bytes per row = I/2
  const size_t sRow = (size_t)NG;                    // scales per row = I/G

  float acc = 0.f;
  #pragma unroll 1
  for (int e = 0; e < E; ++e) {
    const unsigned char*  base   = w2 + ((size_t)e * H + h) * rowB;
    const __hip_bfloat16* scale  = w2_scale + ((size_t)e * H + h) * sRow;
    float d = dot_w4_row<UNROLL_B>(base, scale, acts + (size_t)e * I, NG, fbias);
    acc += __bfloat162float(topk_weights[e]) * d;
  }
  y[h] = __float2bfloat16(acc);
}

// --------------------------------------------------------------------------------------------------
// Host launcher.
// --------------------------------------------------------------------------------------------------
torch::Tensor moe_decode_w4(torch::Tensor x, torch::Tensor w13, torch::Tensor w13_scale,
                            torch::Tensor w2, torch::Tensor w2_scale, torch::Tensor topk_weights,
                            int64_t group_size, int64_t bias) {
  const int H = (int)x.size(0);          // hidden
  const int E = (int)w13.size(0);        // active experts
  const int I = (int)(w13.size(1) / 2);  // intermediate (w13 is [E, 2I, H/2])
  const float fbias = (float)bias;

  TORCH_CHECK(group_size == 32, "moe_decode_w4: kernel assumes group_size==32");
  TORCH_CHECK(H % 32 == 0 && I % 32 == 0, "moe_decode_w4: H and I must be multiples of 32");
  TORCH_CHECK(w2.size(1) == H, "moe_decode_w4: w2 out-dim must equal H");

  auto opts = x.options();  // bf16, same device
  auto act  = torch::empty({E, I}, opts);
  auto y    = torch::empty({H},    opts);

  auto stream = c10::hip::getCurrentHIPStream();

  const auto* xp   = reinterpret_cast<const __hip_bfloat16*>(x.data_ptr<at::BFloat16>());
  const auto* w13p = w13.data_ptr<uint8_t>();
  const auto* w13s = reinterpret_cast<const __hip_bfloat16*>(w13_scale.data_ptr<at::BFloat16>());
  const auto* w2p  = w2.data_ptr<uint8_t>();
  const auto* w2s  = reinterpret_cast<const __hip_bfloat16*>(w2_scale.data_ptr<at::BFloat16>());
  const auto* tkw  = reinterpret_cast<const __hip_bfloat16*>(topk_weights.data_ptr<at::BFloat16>());
  auto* actp = reinterpret_cast<__hip_bfloat16*>(act.data_ptr<at::BFloat16>());
  auto* yp   = reinterpret_cast<__hip_bfloat16*>(y.data_ptr<at::BFloat16>());

  // Kernel A: grid (E, ceil(I/TILE)); x[H] fp32 in LDS.
  dim3 gridA(E, (I + TILE - 1) / TILE);
  dim3 blockA(TILE);
  size_t ldsA = (size_t)H * sizeof(float);
  moe_gate_up_kernel<<<gridA, blockA, ldsA, stream>>>(xp, w13p, w13s, actp, H, I, fbias);

  // Kernel B: grid (1, ceil(H/TILE)); act[E,I] fp32 in LDS.
  dim3 gridB(1, (H + TILE - 1) / TILE);
  dim3 blockB(TILE);
  size_t ldsB = (size_t)E * (size_t)I * sizeof(float);
  moe_down_kernel<<<gridB, blockB, ldsB, stream>>>(actp, w2p, w2s, tkw, yp, H, I, E, fbias);

  return y;
}

// Meta (fake) impl for torch.compile tracing: shape only.
torch::Tensor moe_decode_w4_meta(torch::Tensor x, torch::Tensor w13, torch::Tensor w13_scale,
                                 torch::Tensor w2, torch::Tensor w2_scale, torch::Tensor topk_weights,
                                 int64_t group_size, int64_t bias) {
  return torch::empty({x.size(0)}, x.options());
}

TORCH_LIBRARY(vllm_win_moe, m) {
  m.def("moe_decode_w4(Tensor x, Tensor w13, Tensor w13_scale, Tensor w2, Tensor w2_scale, "
        "Tensor topk_weights, int group_size, int bias) -> Tensor");
  m.impl("moe_decode_w4", torch::kCUDA, &moe_decode_w4);
  m.impl("moe_decode_w4", torch::kMeta, &moe_decode_w4_meta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}  // empty: import triggers the TORCH_LIBRARY static init
