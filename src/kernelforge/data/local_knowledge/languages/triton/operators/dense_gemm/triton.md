---
title: dense_gemm on Triton — SOTA card
kind: sota_card
operator: dense_gemm
backend: triton
gens: [gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: competitive
updated: 2026-06-08
sources:
  - https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
  - https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/triton_kernel_dev.html
  - ROCm/aiter@a6bb4993:aiter/tuned_gemm.py
  - https://arxiv.org/html/2511.08083v1
---

# dense_gemm × Triton

## TL;DR
Triton is the fastest backend to **write/iterate** a GEMM and the one PyTorch-Inductor max-autotune emits —
but on a *plain* dense bf16 GEMM it typically **loses to tuned hipBLASLt/aiter** on MI300X (AMD Triton still
under-lowers buffer loads / register lifetimes). Use it when you need a **fused** epilogue (bias+act+residual)
the library can't do, or as an authored Tier-C candidate to e2e-gate. On sglang the dense path is aiter, not
torch dispatch, so an authored Triton GEMM must be wired via the aiter `triton` libtype seam.

## SOTA implementation
aiter exposes Triton as a dispatch target, but the in-tree `triton_gemm` is a thin shim with hard limits.
From `/sgl-workspace/aiter/aiter/tuned_gemm.py` (`ROCm/aiter@a6bb4993`):

```python
def triton_gemm(inp, weights, solidx, bias=None, otype=None,
                scale_a=None, scale_b=None, scale_c=None, bpreshuffle=False):
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
    assert scale_a is None and scale_b is None and scale_c is None, \
        "Triton gemm_a16w16 does not support scaling yet"
    assert not bpreshuffle, "Triton gemm_a16w16 does not support bpreshuffle yet."
    return gemm_a16w16(inp, weights, bias=bias, dtype=otype)
```

So Triton on the aiter path is bf16/fp16, no scale, no preshuffle. To win you author your own kernel and
register it (or call-site rebind), then e2e-gate.

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| Inductor/Triton mm template (max-autotune) | `triton`, PyTorch Inductor | gfx90a–950; bf16/fp16/fp8 | ~0.7–0.95× of tuned hipBLASLt on plain bf16 GEMM @ MI300X, 2025 | fused-epilogue GEMM; `torch.compile` path |
| Hand-written Triton GEMM (split-K) | author via kernel layer | gfx942/950 | shape-specific; can win decode skinny GEMM with SPLIT_K filling all 304 CUs | skinny/decode or custom fusion |

perf_knowledge e2e validation: an authored Triton GEMM measured **0.99–1.47× isolated** on target shapes but **did
not beat the aiter env at e2e** (2026-06) — the win must survive e2e gating, not just microbench.

## Config space / knobs
| param | range / typical | effect | default |
|---|---|---|---|
| `BLOCK_SIZE_M/N/K` | 128×128×64 / 256×128×64 | output + K tile | 128×128×64 |
| `GROUP_SIZE_M` | 4–8 | L2 re-use via grouped scheduling | 8 |
| `matrix_instr_nonkdim` | 16 / 32 | MFMA shape — **16 (mfma_16x16) > 32 (32x32)** on CDNA | 16 |
| `kpack` | 1 / 2 | K-packing into one MFMA issue | 2 |
| `waves_per_eu` | 1–4 | occupancy hint | 2 |
| `num_warps` | 4 / 8 | warps per block | 4 |
| `num_stages` | 0 / 2 | **0 for a single GEMM on CDNA** (no async pipeline win) | 0 |
| `SPLIT_K` | 2–16 | K split across CUs for skinny/decode | 1 |
| `OPTIMIZE_EPILOGUE` | 0 / 1 | fuse epilogue store | 1 |

Autotune key on `(M,N,K)`. ISA target: want `global_load_dwordx4` (buffer loads) and LDS `_b128` stores.

## Numerics / parity
bf16/fp16 in / **fp32 accumulate** → parity with the library up to tiling rounding. Triton block-scaled fp8/
fp4 is a separate path → operators/scaled_quant_gemm/backends/triton.

## Integration (rebind seam)
On sglang the dense path is aiter, not torch dispatch. An authored Triton GEMM is engaged either via the
aiter `triton` libtype (a CSV row → `triton_gemm`) or a direct call-site rebind in the model's `LinearMethod`,
then e2e-gated through operators/dense_gemm/backends/aiter's verification flow.

## Pitfalls & anti-patterns
- **AMD Triton buffer loads not default**: may fail to reclaim registers or to lower `global_load_dwordx4`
  — always verify with `AMDGCN_ENABLE_DUMP=1` (want `global_load_dwordx4`, LDS `_b128`).
- The **in-aiter Triton GEMM is a shim** (asserts no scale/preshuffle) — treat "author needed" for anything
  beyond plain bf16/fp16.
- **Don't expect to beat tuned hipBLASLt on plain GEMM** — the win is fusion or skinny split-K, and it must
  hold at e2e, not just isolated (isolated 1.47× still lost e2e here).
- `num_stages>0` on a single CDNA GEMM usually *hurts* (no overlap to hide); start at 0.

## How to verify (worked example)
```bash
AMDGCN_ENABLE_DUMP=1 python my_triton_gemm.py 2>&1 | grep -E 'global_load_dwordx4|ds_write_b128'
# isolated bench vs hipBLASLt default on the target shape, then:
# wire via aiter triton libtype CSV row and run the same A/B + parity gate as the aiter card
```

## Alternatives / cross-links
operators/dense_gemm/backends/aiter (dispatch + gate) · operators/dense_gemm/backends/flydsl
(higher ceiling, more knobs) · operators/dense_gemm/backends/hipblaslt (beats plain GEMM) ·
operators/scaled_quant_gemm/backends/triton (block-scaled) · optimization/mfma_scheduling ·
language deep-dive `triton_levers/` (P1).

## Sources
- Optimizing Triton kernels (knobs, ISA verify): https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
- Triton kernel dev notebook: https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/triton_kernel_dev.html
- AMD Triton GEMM under-performance / buffer loads (HipKittens): https://arxiv.org/html/2511.08083v1
- In-aiter Triton shim: `/sgl-workspace/aiter/aiter/tuned_gemm.py` (`triton_gemm`, `ROCm/aiter@a6bb4993`).
- perf_knowledge e2e validation: authored Triton GEMM 0.99–1.47× isolated, did not beat aiter env at e2e (2026-06).
