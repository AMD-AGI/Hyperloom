---
title: rmsnorm — overview
kind: operator_overview
operator: rmsnorm
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode, both]
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py
  - /sgl-workspace/aiter/aiter/ops/rmsnorm.py
  - https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# rmsnorm  (`y = x · rsqrt(mean(x²) + ε) · γ`)

## TL;DR
RMSNorm is a **memory-bandwidth-bound** row reduction that runs **2× per transformer layer** (input-norm
+ post-attention-norm). The single most important fact on MI300X: it is **HBM-traffic limited**, so the
only real lever is *traffic* — read `x` once with **128-bit vectorized global loads**, reduce in-register
across the wave64, and **fuse** it with its neighbors (residual-add before, fp8 quant after) so the
1×read+1×write is shared. On the serving path it almost never appears standalone — it is
fused_add_rmsnorm and fused_norm_quant.

## Math contract
For a row `x[N]` (hidden dim N): `rms = sqrt(Σx²/N + ε)`, `y = (x/rms) · γ`. **No mean subtraction** and
**no bias** (unlike layernorm). dtype: bf16/fp16 in, **fp32 accumulate** for `Σx²` (mandatory —
bf16 accumulate over N=8192 loses ~3 bits), bf16/fp16 out (or fp8 + scale in the quant variant). `γ` is
fp32-promoted before the multiply in correct impls (vLLM regression #42325 shows what breaks when it
isn't). Layout: `x[M,N]` row-major, reduce over the **last (contiguous) dim** → coalesced loads.

## Shape regimes (typical LLM, hidden 4096/5120/8192)
- **prefill**: `M = batch·seqlen` (1k–64k rows), `N = hidden ∈ {4096, 5120, 8192}`. Many rows → trivially
  fills 304 CUs with a row-per-program grid.
- **decode**: `M = running batch` (1..256 rows), same N. **Few rows → CU starvation**; a persistent
  grid (`grid = min(M, num_SMs)`) or one-row-per-CU is needed, and the kernel is pure latency.
- The reduction width N decides the strategy: **N ≤ LDS-block** (`65536/elt_size` ≈ 32768 for bf16) →
  single-pass row-in-registers; **N > block** → two-pass (sum-of-squares pass, then normalize pass).

## Where it matters (Amdahl)
RMSNorm is **1–3% of GPU time** on a dense LLM, but it touches the residual stream on *every* layer, so
it is the prime **fusion anchor**: folding residual-add and fp8 quant into it removes whole HBM
round-trips from the adjacent ops. SGLang reports **1–6% e2e latency** improvement from fusing
RMSNorm+FP8-dynamic-quant on Qwen3 MI300X (sglang #18466). Standalone it won't move e2e; fused it can.

## Backend landscape (→ SOTA cards)
| backend | status | card |
|---|---|---|
| aiter | 🟢 sota (live path: CK/asm + triton) | aiter |
| triton | 🟢 sota (authorable; aiter's own triton impl) | [triton.md](triton.md) |
| vllm_kernels | 🟢 sota (vectorized HIP `rms_norm_kernel`) | vllm_kernels |
| hip | 🟢 sota (the reference hand-written kernel) | hip |
| flydsl | 🟡 competitive (via shared wave64 `reduce.py`) | flydsl |

## Fusion neighbors
`+residual add` → fused_add_rmsnorm (the dominant serving form); `+fp8/int8 dynamic quant` →
fused_norm_quant (cross-link quant_dequant_fp8 / quant_int8); `+QK-norm+RoPE+KV-write` (aiter
`fused_qk_norm_rope_cache_quant`, the Qwen3 attention-entry mega-fusion). See [fusion.md](fusion.md).

## Numerics
fp32 accumulate is mandatory; `γ` must be fp32-promoted; reduction order differs across CK/asm/Triton →
re-gate greedy parity after a backend swap. See [numerics.md](numerics.md).

## How to bench
Isolated: `python3 op_tests/test_rmsnorm2d.py` (aiter) at `(M,N,dtype)` for the model's hidden dim;
oracle = fp64 reference; compare median of ≥3 warm reps. e2e: same-session A/B with the fused variant
toggled. See [tuning.md](tuning.md).

## Sources
- aiter Triton rmsnorm (persistent grid `min(rows, num_sms)`, two-pass blocked, `.cg` loads): `/sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py`.
- aiter C++/CK/asm dispatch (`rms_norm`, `rmsnorm2d_fwd_ck`, `module_rmsnorm`): `/sgl-workspace/aiter/aiter/ops/rmsnorm.py`.
- vLLM HIP `rms_norm_kernel` + vectorization: https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu, PR https://github.com/vllm-project/vllm/pull/22602.
- MI300X bandwidth-bound / ≥1024 grid / 128-bit loads: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html.
