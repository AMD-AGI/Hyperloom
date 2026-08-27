---
title: grouped_gemm_moe on FlyDSL — SOTA card
kind: sota_card
operator: grouped_gemm_moe
backend: flydsl
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, fp4_e2m1, fp16, bf16]
regimes: [prefill, decode]
status: sota
updated: 2026-06-09
---

# grouped_gemm_moe × FlyDSL

## TL;DR
FlyDSL implements the MoE grouped GEMM as **two compiled stages**: `flydsl_moe_stage1` (fused gate+up
projection, A·W1) and `flydsl_moe_stage2` (down projection, A·W2 with expert-weighted reduction). Both are
JIT-compiled from the FLIR/ROCDL MLIR-Python DSL, keyed on `(model_dim, inter_dim, experts, topk, tile_*,
dtypes)` and cached via `functools.lru_cache`. This is the standard-precision (fp8/fp16/bf16) MoE path; the
fp4 a4w4 block-scaled path is documented in [[operators/fused_moe_grouped_gemm/backends/flydsl]]. Gate on
`is_flydsl_available()` — when FlyDSL is absent these symbols are never imported.

## SOTA implementation
The public API in `moe_kernels.py` allocates the sorted-MoE buffers, packs the args, then calls the cached
`compile_flydsl_moe_stage{1,2}` which dispatches `b_dtype=="fp4"` → mixed kernel else the standard
`moe_gemm_2stage` builder. From `/sgl-workspace/aiter/aiter/ops/flydsl/moe_kernels.py`:

```python
def compile_flydsl_moe_stage1(model_dim, inter_dim, experts, topk,
                              tile_m, tile_n, tile_k, doweight_stage1,
                              a_dtype, b_dtype, out_dtype, act="silu", ...):
    """Compile stage1 kernel (cached via underlying lru_cache)."""
    if b_dtype == "fp4":
        from .kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1, GateMode
        return compile_mixed_moe_gemm1(..., gate_mode=GateMode(gate_mode), ...)
    else:
        from .kernels.moe_gemm_2stage import compile_moe_gemm1
        return compile_moe_gemm1(model_dim=model_dim, inter_dim=inter_dim,
            experts=experts, topk=topk, tile_m=tile_m, tile_n=tile_n,
            tile_k=tile_k, doweight_stage1=doweight_stage1,
            in_dtype=a_dtype, out_dtype=out_dtype)
```

`moe_gemm_2stage.py` builds MFMA preshuffled-B pipelines (`compile_moe_gemm1` / `compile_moe_gemm2`). The
in_dtype list it accepts is, per its docstring, `("fp8","fp16","bf16","int8","int8smooth","int4",
"int4_bf16")`; the aiter `moe_kernels` wrapper plumbs `a_dtype ∈ {fp8, fp16}` (and the fp4 path via the
mixed module).

| impl | source | gens/dtypes | measured perf | when best |
|---|---|---|---|---|
| `compile_moe_gemm1` (gate+up, MFMA preshuffle) | `kernels/moe_gemm_2stage.py::compile_moe_gemm1` | gfx942/950; fp8/fp16/bf16 (int8/int4 paths present) | no isolated number on-box | stage1 of fp8/bf16 MoE |
| `compile_moe_gemm2` (down-proj, atomic/reduce) | `kernels/moe_gemm_2stage.py::compile_moe_gemm2` | gfx942/950; fp8/fp16/bf16 | no isolated number on-box | stage2 expert reduction |
| 2-stage FlyDSL MoE GEMM (vendor) | aiter FusedMoE on FlyDSL | gfx942/950 | A4W4/MXFP4 **1.6× latency @ concurrency 512 (MI355X)**; Kimi-K2.5 MI300X up to **+162% throughput, −69% TPOT, −65% TTFT** (vendor, SGLang+AITER) | full MoE block on supported models |

## Config space / knobs
From the `flydsl_moe_stage1` / `flydsl_moe_stage2` signatures (`moe_kernels.py`). For the standard path the
enumerated tunables come from `get_flydsl_stage1_kernels` / `get_flydsl_stage2_kernels`.

| param | range / typical | effect | default |
|---|---|---|---|
| `tile_m` / `tile_n` / `tile_k` | s1: m∈{32,64,128}, n∈{128,256}, k=256; s2: m∈{32,64,128}, n=128, k=128 | per-workgroup output + K tile | s1 32×256×256, s2 32×128×256 |
| `waves_per_eu` (s1) | 1–4 | occupancy hint | 3 |
| `k_batch` (s1) | 1,2,4,7,14 | split-K depth (k_batch>1 routes partials → fused silu_and_mul) | 1 |
| `gate_mode` (s1) | separated / interleave / mock_gate_only | gate/up B-tile strategy (`GateMode` enum) | separated |
| `mode` (s2) | atomic / reduce | atomic accumulate vs explicit reduce | atomic |
| `persist` (s2) | True / False / None | persistent round-robin grid (auto when m_blocks>256) | None (auto) |
| `b_nt` | 0 / 2 | B non-temporal load hint | s1 0, s2 0 |
| `xcd_swizzle` | 0 / 4 | XCD remap for CDNA multi-die scheduling | 0 |
| `doweight_stage{1,2}` | bool | apply per-expert routing weight in this stage | from `sorted_weights is not None` |

## Numerics / parity
GEMM accumulates in fp32 (MFMA), output bf16/fp16. Stage2 `mode`: `"atomic"` accumulates partials into a
zeroed output via atomics (`accumulate=True`); `"reduce"` writes per-(token,slot) partials then
`torch.sum` over the topk axis. Per the stage2 builder docstring, `out_dtype="f16"` uses fp16 half2 atomics
(fast, can overflow to ±inf for bf16-range workloads) while `out_dtype="f32"` uses fp32 scalar atomics
(slower, overflow-safe). Routing weights folded in-kernel when `doweight_stage*` is set. See
[../numerics.md](../numerics.md).

## Integration (rebind seam)
Direct Python entry points `aiter.ops.flydsl.flydsl_moe_stage1` / `flydsl_moe_stage2`, imported in
`aiter/ops/flydsl/__init__.py` **only when `is_flydsl_available()`** and flydsl ≥ 0.1.3. Inputs are the
moe-sorted tensors (`sorted_token_ids`, `sorted_expert_ids`, `num_valid_ids`) plus pre-shuffled weights;
there is no env-CSV overlay for the MoE path (unlike dense_gemm). Compilation is `lru_cache`-keyed, so the
first call per shape pays a JIT cost. On compile failure `_run_compiled` drains leaked `ir.Context`s before
re-raising, to keep later JitFunction calls on the correct path.

## Pitfalls & anti-patterns
- **Optional dependency**: the stage1/stage2 symbols don't exist if flydsl isn't installed — always guard.
- Weights must be pre-shuffled in the layout the kernel expects (stage1 gate+up, stage2 down); passing
  un-shuffled weights gives wrong results, not an error.
- `tile_m` must match the `block_size` used by moe_sorting unless `sort_block_m` is set explicitly on stage2.
- `out_dtype="f16"` atomics can overflow to ±inf for bf16-magnitude activations — use f32 atomics or bf16
  output if you see inf.
- First-call JIT latency per unique shape; warm up before timing.

## How to verify
```python
from aiter.ops.flydsl.utils import is_flydsl_available
assert is_flydsl_available()
from aiter.ops.flydsl import flydsl_moe_stage1, flydsl_moe_stage2
# build moe-sorted ids + pre-shuffled w1/w2, call stage1 then stage2;
# compare against aiter.fused_moe reference for the same routing.
```

## Alternatives / cross-links
[[operators/fused_moe_grouped_gemm/backends/flydsl]] (fp4 a4w4 fused path) ·
[[operators/grouped_gemm_moe/backends/aiter]] (DB-driven dispatch) ·
[[operators/grouped_gemm_moe/backends/ck]] · [[operators/grouped_gemm_moe/backends/triton]] ·
[[operators/dense_gemm/backends/flydsl]] (the dense FlyDSL hgemm) ·
[[operators/act_and_mul_silu_gelu/backends/flydsl]] (fused split-K post-activation).
