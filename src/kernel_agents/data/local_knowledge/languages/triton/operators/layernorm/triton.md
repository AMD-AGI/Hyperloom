---
title: layernorm on triton — SOTA card
kind: sota_card
operator: layernorm
backend: triton
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp32]
regimes: [both, training]
status: sota
updated: 2026-06-08
sources:
  - /sgl-workspace/aiter/aiter/ops/triton/normalization/norm.py
  - https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
  - https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
---

# layernorm × triton

## TL;DR
Triton is the authorable SOTA for LayerNorm on MI300X — aiter ships its own Triton impl
(`ops/triton/normalization/norm.py`) with forward (two-pass mean/var), fused add, fused quant, and a
backward with tiled `dγ,dβ` reduce. Competitive with CK because the op is bandwidth-bound. The one design
choice is two-pass vs blocked.

## SOTA implementation(s)
| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| aiter `_layernorm_forward` (two-pass) | `aiter/ops/triton/normalization/norm.py` | gfx942/950, bf16/fp16/fp32 | bandwidth-bound; `num_warps=min(max(BLOCK//256,1),8)`, blocked when N>BLOCK | the library Triton tier / fused variants |
| `layernorm2d_fwd_with_add` / `_with_dynamicquant` / `_with_add_dynamicquant` | same | gfx942/950, bf16/fp8 out | one read+write incl. residual/quant | fused_norm_quant |
| `_layernorm_backward` + tiled `dwdb` reduce | same (`BLOCK_SIZE_M=128/16, BLOCK_SIZE_N=64`) | gfx942/950 | two-kernel: per-row bwd + cross-row dγ/dβ | training |
| Triton tutorial 05-layer-norm | https://triton-lang.org/.../05-layer-norm.html | generic | Welford reference | learning |

## Config space / knobs
- `BLOCK_SIZE = min(65536//elt, next_pow2(N))`; `USE_BLOCKED = N > BLOCK_SIZE` (blocked clamps BLOCK to
  2048 fp32 / 4096 else to bound LDS).
- `num_warps = min(max(BLOCK_SIZE//256,1), 8)` → 2–4.
- Grid: row-per-program (prefill) / persistent `min(M, num_sms)` (decode).
- `waves_per_eu=3–4`; `.cg` x loads; fp32 μ/σ² accumulate.
- Backward: `dwdb_block_m/n` tile the cross-row γ/β gradient reduce.

## Numerics / parity
Two-pass (no cancellation); fp32 μ/σ²; γ,β fp32-promote; biased variance (/N). Reduction order vs CK/asm
differs → greedy re-gate. See [numerics.md](numerics.md).

## Integration (rebind seam)
- Direct: `from aiter.ops.triton.normalization.norm import layer_norm`.
- torch.compile: Inductor emits Triton for `nn.LayerNorm` under max-autotune; wire AMD knobs via
  `torch._inductor.config`.

## Pitfalls & anti-patterns
- `num_warps=8` from NVIDIA → spill. Start 2–4.
- One-pass `Σx²−μ²` cancellation → negative σ²; use the two-pass / Welford body.
- Backward γ/β register spill beyond ~8k hidden → use the tiled dwdb reduce (aiter does).

## How to verify
`TRITON_PRINT_AUTOTUNING=1`; isolated bench vs aiter CK; ISA `global_load_dwordx4`; greedy parity; σ²≥0.

## Alternatives / cross-links
aiter · hip · vllm_kernels · [miopen.md](miopen.md) ·
triton_levers/patterns §5 · [tuning.md](tuning.md).

## Sources
- aiter Triton layernorm (fwd two-pass, bwd dwdb, fused add/quant): `/sgl-workspace/aiter/aiter/ops/triton/normalization/norm.py`.
- Welford reference / register-spill note: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html.
- Memory-bound knobs: https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html.
