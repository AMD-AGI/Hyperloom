---
title: Triton programming model — @jit, launch grid, autotune/heuristics decorators
kind: api_reference
gens: [gfx942, gfx950]
dtypes: [both]
regimes: [both]
status: sota
updated: 2026-07-09
sources:
  - https://triton-lang.org/main/python-api/triton.html
  - https://triton-lang.org/main/python-api/triton.language.html
---

# Triton programming model & decorators

The host-side surface: how a kernel is declared, specialized, launched, and autotuned. The kernel body
ops are in [language_api.md](language_api.md); AMD knob semantics in
[../skills/optimize/triton_levers/triton_knob_space.md](../skills/optimize/triton_levers/triton_knob_space.md).

## `@triton.jit` and launch
```python
import triton, triton.language as tl

@triton.jit
def kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) * 2.0, mask=m)

grid = (triton.cdiv(N, BLOCK),)          # grid is a tuple or a callable(meta)->tuple
kernel[grid](x, y, N, BLOCK=1024)        # launch via subscript; tensors pass as pointers
```
- **`tl.constexpr`** params are compile-time (tile sizes, flags) — each distinct value is a separate
  specialization (own cache entry + own ISA).
- Non-constexpr scalars/pointers are runtime args; Triton infers alignment/divisibility for vectorization
  (`tl.multiple_of`, `tl.max_contiguous` give hints).
- `grid` may be `lambda meta: (triton.cdiv(N, meta["BLOCK"]),)` to depend on tuned tile sizes.

## Standard launch params (map to HIPOptions on AMD)
| param | meaning | AMD note |
|---|---|---|
| `num_warps` | warps/block | **wave64**: `num_warps=N` → N·64 threads. Start GEMM at 4 (8 spills) |
| `num_stages` | stream-pipeliner depth | single GEMM 2, fused FA 1 (NOT 3–4) |
| `maxnreg` | hard VGPR cap | rarely needed |
AMD-only knobs (`matrix_instr_nonkdim`, `kpack`, `waves_per_eu`, `schedule_hint`) go **inside
`triton.Config({...})` kwargs**, not as bare vars — see knobs.md.

## `@triton.autotune`
```python
@triton.autotune(
    configs=[triton.Config({"BLOCK_M":128,"BLOCK_N":256,"BLOCK_K":64,"GROUP_SIZE_M":8,
                            "matrix_instr_nonkdim":16,"kpack":2,"waves_per_eu":2},
                           num_warps=4, num_stages=2),
             # ... more configs ...],
    key=["M","N","K"],                       # re-tune when these change
    prune_configs_by={"early_config_prune": my_prune},   # e.g. drop configs whose LDS > 64 KB
    warmup=25, rep=100)
@triton.jit
def gemm(...): ...
```
`TRITON_PRINT_AUTOTUNING=1` prints the winner + timing. Bake the winner for the serving hot path (single
`Config`, or a per-shape JSON table) — don't autotune in production (first-call latency + nondeterminism).

## `@triton.heuristics`
```python
@triton.heuristics({"EVEN_K": lambda a: a["K"] % a["BLOCK_K"] == 0})
@triton.jit
def kernel(..., EVEN_K: tl.constexpr): ...     # derive a constexpr from runtime args
```

## AOT / inspection
```python
compiled = kernel.warmup(x, y, N, BLOCK=1024, grid=(1,))   # force compile without launch
# ISA / IR dumps: AMDGCN_ENABLE_DUMP=1, MLIR_ENABLE_DUMP=1, TRITON_ALWAYS_COMPILE=1
```

## Sources
- Triton runtime/JIT/autotune/heuristics API: https://triton-lang.org/main/python-api/triton.html
- Triton language (constexpr, program_id, grid): https://triton-lang.org/main/python-api/triton.language.html
- AMD knob mapping (HIPOptions): [../skills/optimize/triton_levers/triton_knob_space.md](../skills/optimize/triton_levers/triton_knob_space.md)
