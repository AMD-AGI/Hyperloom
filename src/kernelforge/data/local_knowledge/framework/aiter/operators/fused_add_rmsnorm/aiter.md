---
title: fused_add_rmsnorm on aiter — SOTA card
kind: sota_card
operator: fused_add_rmsnorm
backend: aiter
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: sota
updated: 2026-07-14
sources:
  - ROCm/aiter@b467ce342:aiter/ops/rmsnorm.py
  - ROCm/aiter@b467ce342:csrc/kernels/rmsnorm_quant_kernels.cu
  - ROCm/aiter@b467ce342:aiter/ops/triton/normalization/rmsnorm.py
  - ROCm/aiter@b467ce342:aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py
  - https://github.com/vllm-project/vllm/pull/14959
---

# fused_add_rmsnorm × aiter

## TL;DR
aiter is the live path and the SOTA choice. `rmsnorm2d_fwd_with_add` (HIP fast path ↔ opus) is exactly what
`VLLM_ROCM_USE_AITER_RMSNORM=1` wires in (vLLM PR #14959) — the **residual add + RMSNorm** that occurs
**twice per transformer layer** (post-attn, post-MLP), so it sits on the hot path of every token. Fusing
the residual add into the norm saves a full HBM round-trip of the hidden state and a kernel launch; the
quant-stacked `rmsnorm2d_fwd_with_add_dynamicquant` / `add_rmsnorm_quant` are the SGLang Qwen3 norm+quant
win. Use this card's backend unless a shape has no HIP/opus tune → then triton.

## SOTA implementation(s)
| impl | source (`repo@commit:path`) | gens / dtypes | measured perf | when it's best |
|---|---|---|---|---|
| `rmsnorm2d_fwd_with_add` (HIP `add_rmsnorm` ↔ opus) | `aiter/ops/rmsnorm.py` (`_use_hip_common` → `add_rmsnorm` on `module_rmsnorm_quant`, else `rmsnorm2d_fwd_with_add_opus`) | gfx942/950, bf16/fp16 | live via `VLLM_ROCM_USE_AITER_RMSNORM=1`; one read+write incl. residual | **the serving residual norm** |
| `add_rmsnorm` (HIP fast path) | `aiter/ops/rmsnorm.py` (`module_rmsnorm_quant` `add_rmsnorm_quant_kernel`) | gfx942/950 | fused add+norm, no quant | direct HIP entrypoint (2-D bf16/fp16, `N≤8192`) |
| `rmsnorm2d_fwd_with_add_dynamicquant` / `add_rmsnorm_quant` | `aiter/ops/rmsnorm.py` (opus `..._with_add_dynamicquant_opus`) | gfx942/950, fp8 y | part of Qwen3 **+1–6% e2e** (label-sourced, #18466) | residual+norm+fp8-quant → fused_norm_quant |
| `rmsnorm2d_fwd_with_add_smoothquant` | `aiter/ops/rmsnorm.py` (opus `..._with_add_smoothquant_opus`) | gfx942/950, int8 | residual+norm+per-channel int8 | int8 GEMM consumer |
| Triton `_fused_add_rmsnorm_kernel` | `aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py` | gfx942/950 | persistent grid, single/two-pass | fallback / portability |

### What the SOTA kernel actually does
The fused kernel loads `x` and `residual_in`, **adds them once**, writes `residual_out` (the new residual
stream), then normalizes from that summed value — so the residual is materialized exactly once and the norm
reads it from registers, not from a second HBM load. The on-box Triton body shows the contract precisely:

```python
# ROCm/aiter@b467ce342:aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py  (_fused_add_rmsnorm_kernel)
row    = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
res_in = tl.load(res_in_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
row += res_in
tl.store(res_out_ptrs, row.to(res_out_ptr.type.element_ty), mask=mask)   # residual_out in IO dtype
row = row.to(tl.float32)                                                  # promote AFTER storing residual
g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
norm_factor = tl.math.rsqrt(tl.sum(row*row, axis=-1)/n_cols + epsilon)    # fp32 Σ(x+r)²
tl.store(output_ptrs, (row*norm_factor*g).to(output_ptr.type.element_ty), mask=mask)
```
The add happens in the **IO dtype** (bf16) and `residual_out` is stored in that dtype **before** the fp32
promotion — that ordering is load-bearing for cross-framework parity (the next layer's residual must be the
exact bf16 sum). The blocked path re-reads `res_out` (the just-stored sum) for the normalize pass so the
two passes agree bit-for-bit.

### Python dispatch
```python
# ROCm/aiter@b467ce342:aiter/ops/rmsnorm.py
def rmsnorm2d_fwd_with_add(out, input, residual_in, residual_out, weight, epsilon,
                           gemma_norm=False, use_model_sensitive_rmsnorm=0):
    if _use_hip_common(input, use_model_sensitive_rmsnorm):
        add_rmsnorm(out, input, residual_in, residual_out, weight, epsilon, gemma_norm)  # fast HIP
        return
    rmsnorm2d_fwd_with_add_opus(out, input, residual_in, residual_out, weight, epsilon,
                                use_model_sensitive_rmsnorm, gemma_norm)                  # opus
```
So the common case (2-D bf16/fp16, `N≤8192`, `use_model_sensitive_rmsnorm==0`) ⇒ **HIP fast path**
`add_rmsnorm` (`module_rmsnorm_quant`); otherwise ⇒ **opus**. `gemma_norm=True` selects the Gemma
`(1+weight)` variant.

## Config space / knobs
| knob | where | default | effect |
|---|---|---|---|
| `VLLM_ROCM_USE_AITER` + `_RMSNORM` | env | 0 / 1 | gate + route `forward_hip` → `_with_add` |
| `use_model_sensitive_rmsnorm` | arg | 0 | `!=0` disables HIP fast path → opus (higher-parity reduction) |
| hidden dim N / rank / dtype | data | — | HIP fast path needs 2-D, bf16/fp16, `N≤8192`; else opus |
| `gemma_norm` | arg | False | Gemma `(1+weight)` scaling variant |
| fusion entrypoint | call site | — | `_with_add` / `_with_add_dynamicquant` / `_with_add_smoothquant` — the fusion is the lever |
| Triton `block_size`/`use_blocked`/`NUM_PRGMS` | derived | `65536//elt` / `N>bs` / `min(rows,num_sms)` | single vs two-pass, persistent grid |
| JIT warm | runtime | — | first call compiles `module_rmsnorm_quant` |

## Numerics / parity
- **Add in IO dtype, store `residual_out` in IO dtype, then promote to fp32** for `Σ(x+r)²`; γ fp32-promote;
  ε inside the mean. Mismatching `residual_out` dtype corrupts the residual stream and the error
  **compounds layer-over-layer**.
- **fp32 accumulate, single-pass** for `N ≤ 65536/elt` (Triton tier); two-pass blocked only to fit wide
  rows — no Welford (RMSNorm subtracts no mean). The blocked path re-reads the stored sum so both passes
  see identical values.
- **Reduction order differs** across HIP/opus/Triton → greedy re-gate on a swap; with residual fusion the
  drift compounds, so parity gating is *more* important than for standalone norm.
- **fp8 fused-quant output** (`_with_add_dynamicquant`): per-token scale `row_max/DTYPE_MAX`, fnuz on
  gfx942. See [numerics.md](numerics.md).

## Integration (rebind seam)
- **vLLM**: `VLLM_ROCM_USE_AITER=1` + `VLLM_ROCM_USE_AITER_RMSNORM=1` →
  `vllm/model_executor/layers/layernorm.py` `RMSNorm.forward_hip` calls `rmsnorm2d_fwd_with_add`
  (PR #14959). This is the *default-on* form once the master flag is set.
- **SGLang**: on by default; routes through the fused norm (+quant) kernels.
- **Verify**: `AITER_LOG_MORE=1` prints `RMSNORM_2D_FWD_ADD: ...`; `rocprofv3` shows a single fused
  add+norm kernel (not separate `add` + `rmsnorm` kernels).

## Pitfalls & anti-patterns
- ⚠ Using plain `rms_norm` where the block needs `_with_add` → the residual add becomes a separate kernel:
  extra full read+write of the hidden state + a launch (lose the fusion, ~2× the traffic) **and** you must
  manage the residual yourself.
- ⚠ `residual_out` dtype ≠ the model's residual dtype → corrupt residual stream that compounds across all
  layers (looks like slow accuracy decay, not an obvious crash).
- ⚠ Promoting to fp32 **before** storing `residual_out` → the next layer gets an fp32-rounded residual,
  diverging from the bf16 reference.
- ⚠ fnuz/OCP fp8 mismatch on gfx942 for the quant variant.
- ⚠ Cold bench (JIT compile counted as kernel time).

## How to verify (bench + oracle)
- **Isolated**: `op_tests/test_rmsnorm2d.py` with the `_with_add` path at the model hidden dim; assert
  `residual_out ≈ (x + residual_in)` in IO dtype and `out ≈ rmsnorm(x+r)`.
- **Bandwidth**: `(input + residual_in + residual_out + out) bytes / time` vs ~3.5 TB/s effective HBM.
- **e2e**: greedy parity + `rocprofv3` single fused kernel + `AITER_LOG_MORE=1` dispatch line.

## Worked example (Qwen3 post-MLP residual norm, MI300X)
Hidden `N=5120`, decode batch `M=128`, bf16. `N=5120 ≤ 8192` (2-D, elt==2) ⇒ **HIP fast path** `add_rmsnorm`. Four bf16 streams of
`128·5120·2 = 1.25 MiB` each (input, residual_in, residual_out, out) = 5.0 MiB; floor ≈ `5.0 MiB / 3.5 TB/s
≈ 1.5 µs`. The **unfused** alternative (`add` kernel: read x+r, write r_out = 3.75 MiB + launch; then
`rms_norm`: read r_out, write out = 2.5 MiB + launch) moves ~6.25 MiB across two launches — the fusion
removes ~1.25 MiB and one launch per call, ×2 per layer ×N_layers per token. Stacking the fp8 quant
(`_with_add_dynamicquant`) further drops the GEMM input from bf16 to fp8 → fused_norm_quant.

## Alternatives / cross-links
triton · hip · vllm_kernels · [overview.md](overview.md)
· rmsnorm/backends/aiter · fused_norm_quant · fused_allreduce_rmsnorm ·
optimization/kernel_fusion_strategy.

## Sources
- aiter add+norm(+quant) entrypoints and `_use_hip_common` HIP-vs-opus dispatch:
  `ROCm/aiter@b467ce342:aiter/ops/rmsnorm.py`; HIP kernel `csrc/kernels/rmsnorm_quant_kernels.cu`
  (`add_rmsnorm_quant_kernel`); opus generic `csrc/kernels/rmsnorm/rmsnorm_opus_*.cu`.
- Triton fused-add kernel (add in IO dtype, store residual before fp32 promote, blocked re-read):
  `ROCm/aiter@b467ce342:aiter/ops/triton/_triton_kernels/normalization/rmsnorm.py`,
  `.../normalization/rmsnorm.py`.
- vLLM AITER `with_add` integration: https://github.com/vllm-project/vllm/pull/14959.
- Qwen3 norm+quant e2e (label-sourced): https://github.com/sgl-project/sglang/issues/18466.
