---
title: act_and_mul_silu_gelu on aiter — SOTA card
kind: sota_card
operator: act_and_mul_silu_gelu
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, mxfp4]
regimes: [both]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/activation.py
  - ROCm/aiter@b467ce342:csrc/kernels/activation_kernels.cu
  - ROCm/aiter@b467ce342:aiter/ops/triton/activation.py
  - ROCm/aiter@b467ce342:aiter/ops/flydsl/kernels/silu_and_mul_fq.py
---

# act_and_mul_silu_gelu × aiter

## TL;DR
aiter is the live gated-activation path on AMD serving and the SOTA choice. It ships a C++/HIP module
(`module_activation`: `silu_and_mul`, `gelu_and_mul`, `gelu_tanh_and_mul`, `scaled_silu_and_mul`,
`gelu_fast`), Triton act+quant kernels (`act_mul_and_fp8_group_quant`, `act_mul_and_mxfp4_quant`), and the
FlyDSL fused `silu_and_mul_fq` for MoE stage-1. The op is **bandwidth-bound** (read `2d` cols, write `d`),
so the levers are 128-bit loads, packed-FP32 math, and **fusion**: on the serving path the gated activation
is almost never standalone — it's the **epilogue of a GEMM** or **inside fused-MoE stage-1** so the
intermediate `silu(x)·y` never hits HBM. Choose this backend unless you need a quant mode the library lacks.

## SOTA implementation(s)
| impl | source (`repo@commit:path`) | gens / dtypes | measured perf | when it's best |
|---|---|---|---|---|
| `silu_and_mul` / `gelu_and_mul` / `gelu_tanh_and_mul` (C++/HIP) | `aiter/ops/activation.py` (`module_activation`), `csrc/kernels/activation_kernels.cu` | gfx942/950, bf16/fp16 | bandwidth-bound; 128-bit chunked loads + `v_pk_mul_f32` | standalone gated act |
| `scaled_silu_and_mul` (static fp8 scale) | same | gfx942/950, fp8 out | fused output quant (static per-tensor scale) | act → fp8 down-proj |
| `act_mul_and_fp8_group_quant` (Triton) | `aiter/ops/triton/activation.py` | gfx942/950, fp8 out | per-group dynamic quant, `BLOCK_SIZE_N = group_size`, grid `(M, N/2 / group)` | dynamic group-fp8 fusion |
| `act_mul_and_mxfp4_quant` (Triton) | same | gfx950, mxfp4 out | block-32 e8m0 scale; `BLOCK_SIZE_N ×32` | dynamic mxfp4 fusion (gfx950) |
| `silu_and_mul_fq` (FlyDSL, act+quant+scale-sort) | `aiter/ops/flydsl/kernels/silu_and_mul_fq.py` | gfx942/950, fp4/fp8/none out | inside fused-MoE (fused_moe = **88–90% of GPU time** on Kimi-K2.5, ROCm blog) | MoE stage-1 → fused_moe_grouped_gemm |

### What the SOTA kernel actually does (on-box C++)
The C++ `act_and_mul_kernel` is the bandwidth-bound recipe: one block per token, **128-bit chunked loads**
of both gate (`x`) and up (`y`) halves, **fp32 activation**, and a **packed-FP32 multiply** (`v_pk_mul_f32`,
two FP32 lanes per instruction), output type independent of input (so the same kernel emits bf16 or fp8):

```cpp
// ROCm/aiter@b467ce342:csrc/kernels/activation_kernels.cu  (act_and_mul_kernel)
auto const* ptr_x = input + token_idx*2*d;      // gate half
auto const* ptr_y = input + token_idx*2*d + d;  // up half
// load_chunk_bytes resolves to 16 when sizeof(DTYPE_I)*VEC_SIZE_I % 16 == 0  → global_load_dwordx4
x = load_vector_nbytes<...,load_chunk_bytes>(buffer_x, idx);
y = load_vector_nbytes<...,load_chunk_bytes>(buffer_y, idx);
float ax0 = ACT_FN(x[j]);    float ax1 = ACT_FN(x[j+1]);     // SiLU/GeLU in fp32
opus::fp32x2_t a = {ax0, ax1}, b = {y0, y1}, c;
asm volatile("v_pk_mul_f32 %0, %1, %2" : "=v"(c) : "v"(a), "v"(b));   // packed fp32 gate·up
r[j] = cast<DTYPE_O>(c.x);  r[j+1] = cast<DTYPE_O>(c.y);              // store as bf16/fp8
```
`silu_kernel` itself uses the fast hardware reciprocal/exp: `x * __builtin_amdgcn_rcpf(1 + __ocml_exp_f32(-x))`.
The input layout is the **packed gate-up** `[..., 2, d]` (gate then up), and the output is `[..., d]` — half
the columns out, so traffic is `2d` read + `d` write.

### The Triton act+quant variants (fused output)
`act_mul_and_fp8_group_quant` computes the activation in fp32, takes a **per-group** amax over `group_size`
columns, and writes fp8 + an fp32 block-scale — one launch, no bf16 intermediate. Grid is `(M, ceil(N/2 /
group))` with `BLOCK_SIZE_N = group_size`. `act_mul_and_mxfp4_quant` is the MXFP4 sibling: a **fixed
`MXFP4_QUANT_BLOCK_SIZE = 32`** (spec, do not tune), e8m0 block-scale, optional scale preshuffle (M→×256,
N→×8).

## Config space / knobs
| knob | where | values | effect |
|---|---|---|---|
| variant | entrypoint | silu / gelu / gelu_tanh; scaled vs plain; group-fp8 / mxfp4 | the entrypoint IS the config |
| `limit` (silu_and_mul) | arg | 0.0 (off) | `>0` clamps gate/up before the multiply (gpt-oss clamped SwiGLU) |
| `VEC_SIZE_I` | C++ template | resolves chunk to 16 B when aligned | 128-bit `dwordx4` loads/stores |
| `group_size` | arg (fp8 group) | e.g. 128 | scale granularity = `BLOCK_SIZE_N`; match consumer GEMM |
| `MXFP4_QUANT_BLOCK_SIZE` | const | **32 (fixed by spec)** | do not tune |
| `BLOCK_SIZE_N` (mxfp4) | derived | `min(256, next_pow2(d))`, ≥32, ×32 if shuffle | tile width; multiple-of-32 for MX blocks |
| `BLOCK_SIZE_M` (mxfp4) | derived | 8 / 16 (M-dependent) | rows per program |
| `num_warps` (mxfp4) | derived | 1 (small M) / 4 | memory-bound; keep low |
| `shuffle` / `scale_shuffle_padding` | arg | False/True | scale layout for block-scaled GEMM |
| `scaling_mode` | arg | "even" (even_round) | MX scale rounding |
| JIT warm | runtime | — | compiles `module_activation` on first call |

## Numerics / parity
- **fp32 activation compute** (SiLU/GeLU), packed-FP32 gate·up multiply; output cast to bf16/fp8 only at the
  store. There is no reduction here (it's a pointwise gated map), so no two-pass / Welford question — the
  only numerics gates are activation-variant choice and quant rounding.
- **Match the GeLU variant to the checkpoint**: exact erf (`gelu_and_mul`, `M_SQRT1_2` erf form) vs tanh
  approximation (`gelu_tanh_and_mul`, the `0.044715·x³` series). Picking the wrong one is a **silent
  accuracy bug** (vLLM #43326 is the analogous gelu-variant footgun).
- **Correct gated half**: `silu_and_mul` applies the activation to the **first** half (gate) and multiplies
  by the second (up); a flipped layout silently corrupts outputs.
- **fp8 fnuz on gfx942**, fp4/mxfp4 **gfx950-only**. The group-fp8/mxfp4 variants take a per-group amax in
  fp32 → exact scale for the written block. Gate quant with a task metric, not allclose. See
  [numerics.md](numerics.md).

## Integration (rebind seam)
- **vLLM**: `VLLM_ROCM_USE_AITER=1` (+ MoE/linear gates) routes the activation; the MLP/MoE down-proj path
  uses the fused stage-1 rather than a standalone activation kernel.
- **SGLang**: on by default; MoE goes through `rocm_moe_utils` / the FlyDSL `silu_and_mul_fq` stage-1.
- **Verify**: `AITER_LOG_MORE=1` shows e.g. `ACT_MUL_FP8_GROUP_QUANT: ...`; `rocprofv3` shows the activation
  **fused into** the MoE / GEMM epilogue, not a standalone `act_and_mul_kernel` between two GEMMs.

## Pitfalls & anti-patterns
- ⚠ Wrong GeLU variant (exact vs tanh) or flipped gate/up half → silent accuracy bug, no crash.
- ⚠ fp4/mxfp4 on gfx942 → no HW (`VLLM_ROCM_USE_AITER_FP4BMM=0`); fp4 is gfx950.
- ⚠ Running the standalone activation when a MoE/GEMM-fused path exists → the `silu(x)·y` intermediate hits
  HBM (extra `d`-wide write + read) and you pay a launch — exactly what the FlyDSL stage-1 avoids.
- ⚠ Tuning `MXFP4_QUANT_BLOCK_SIZE` off 32 → spec violation, wrong dequant.
- ⚠ `group_size` ≠ consumer GEMM's block → wrong fp8 dequant.
- ⚠ Forgetting the `[..., 2, d]` packed gate-up layout when wiring inputs → reads the wrong halves.

## How to verify (bench + oracle)
- **Isolated**: op test at `(M, 2d)` input → `(M, d)` output; compare to `silu(x)·y` in fp64; for the quant
  variants dequantize (×group scale) and compare to the fp32 activation.
- **Bandwidth**: `(2d·in_elt + d·out_elt)·M / time` vs ~3.5 TB/s effective MI300X HBM.
- **e2e**: `rocprofv3` shows the activation fused into fused-MoE / the down-proj epilogue (not standalone);
  task eval for quant; greedy parity for the activation variant.

## Worked example (Llama-style SwiGLU MLP down-proj input, MI300X)
Intermediate `2d = 28672` (d=14336), decode `M=128`, bf16. `silu_and_mul` reads `128·28672·2 = 7.0 MiB`,
writes `128·14336·2 = 3.5 MiB` → floor ≈ `10.5 MiB / 3.5 TB/s ≈ 3.0 µs`; 128-bit loads give 28672/8 = 3584
`dwordx4` per token, the `v_pk_mul_f32` does the gate·up in 7168 packed ops. **Fused** as the up-proj
epilogue, the `silu(x)·y` stays in registers/LDS and feeds the down-proj GEMM directly — the 3.5 MiB
intermediate write **and** its re-read disappear, plus one launch. In MoE (Kimi-K2.5), `fused_moe`
dominated **88–90% of GPU time** (ROCm blog), so folding the activation into stage-1 via FlyDSL
`silu_and_mul_fq` (which also quantizes to fp4/fp8 and writes scales in the expert-sorted layout) is where
the win is — vendor-reported throughput gains on that path.

## Alternatives / cross-links
triton · hip · vllm_kernels ·
fused_moe_grouped_gemm · gemm_epilogue_fused · backends/aiter/fmoe ·
backends/aiter/flydsl_path · optimization/kernel_fusion_strategy.

## Sources
- aiter activation Python entrypoints: `ROCm/aiter@b467ce342:aiter/ops/activation.py`.
- On-box C++/HIP kernel (128-bit chunked loads, fp32 act, `v_pk_mul_f32` packed multiply, type-independent
  output): `ROCm/aiter@b467ce342:csrc/kernels/activation_kernels.cu`.
- Triton act+quant (per-group fp8, mxfp4 block-32 e8m0, scale preshuffle):
  `ROCm/aiter@b467ce342:aiter/ops/triton/activation.py`.
- FlyDSL fused silu+quant+scale-sort for MoE stage-1 (`quant_mode` fp4/fp8/none, inter_dim%32==0):
  `ROCm/aiter@b467ce342:aiter/ops/flydsl/kernels/silu_and_mul_fq.py`.
- Kimi-K2.5 fused-MoE optimization (fused_moe = 88–90% of GPU time; FlyDSL stage-1; throughput vendor-reported):
  https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html.
- GeLU-variant parity footgun (analogous): https://github.com/vllm-project/vllm/issues/43326.
