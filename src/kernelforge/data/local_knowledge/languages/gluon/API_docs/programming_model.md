---
title: Gluon programming model — @gluon.jit, launch, autotune, the layout-typed value model
kind: api_reference
gens: [gfx942, gfx950]
dtypes: [both]
regimes: [both]
status: experimental
updated: 2026-08-23
sources:
  - https://triton-lang.org/main/gluon/index.html
  - https://triton-lang.org/main/getting-started/tutorials/gluon/intro.html
  - https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html
  - https://triton-lang.org/main/dialects/GluonDialect.html
---

# Gluon programming model

The host-side surface and the value model. Layout objects and their costs are in
[layouts.md](layouts.md); the AMD target namespaces are in [amd_targets.md](amd_targets.md).

## TL;DR
Gluon is the **same Python frontend and JIT as Triton** with one semantic change: every tensor value
carries an explicit **layout** in its type. You declare kernels with `@gluon.jit`, launch them with
the identical `kernel[grid](...)` interface, pass PyTorch tensors the same way, and autotune
`constexpr` hyperparameters with the same `@triton.autotune`. What you no longer get for free is
layout assignment, software pipelining, register budgeting and MFMA selection — those become code you
write. `num_stages` has no meaning here.

## Import and declare

```python
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def copy_scalar_kernel(in_ptr, out_ptr):
    value = gl.load(in_ptr)
    gl.store(out_ptr, value)
```

Launch is Triton's, unchanged — PyTorch tensors become global-memory pointers, the grid is a tuple,
and meta-parameters like `num_warps` are launch kwargs:

```python
grid = (1,)
copy_scalar_kernel[grid](input, output, num_warps=1)
```

`constexpr` arguments work as in Triton, and `triton.cdiv` still computes the grid:

```python
@gluon.jit
def memcpy_kernel(in_ptr, out_ptr, xnumel, XBLOCK: gl.constexpr):
    pid = gl.program_id(0)
    start = pid * XBLOCK
    ...

grid = (triton.cdiv(xnumel, XBLOCK),)
```

**One Gluon "program" is one thread block (CTA)**, exactly as in Triton. A scalar loop over elements
is legal and is what the intro tutorial uses, but it processes one element per CTA per iteration —
moving to tiles is what forces you to pick a layout.

### Autotune still applies
`@triton.autotune` stacks above `@gluon.jit` and tunes `constexpr` hyperparameters (tile sizes,
unroll factors, your own pipeline-depth constants) exactly as it does for Triton. What it can **not**
tune is anything Triton exposed as a compiler knob and Gluon turned into source: pipeline placement,
operand layouts, and MFMA selection are now program text, so they are varied by editing or by a
`constexpr` you introduced for the purpose — not by a `triton.Config` field.

> **Forge note.** A `constexpr` you drive from a host-side Python constant is exactly the shape the
> `FORGE_SWEEP_*` mechanism wants — see `common_methodology/optimization/lever_cheap_sweeps.md`. Pipeline
> depth, unroll factor and tile dims are all cheap-sweepable in Gluon without an edit-and-gate cycle,
> which is the main reason to introduce a named constant rather than a literal.

## The value model: layouts are part of the type

This is the whole difference from Triton. A tensor in Gluon is a tile **plus** a statement of how its
elements are distributed over the thread block, following the GPU hierarchy — thread block → warps →
lanes → registers. Tensors are distributed evenly, so every thread owns the same number of elements,
and because tile dimensions are powers of two, elements-per-thread is a power of two.

You seed a layout on an index tensor and let type inference carry it forward:

```python
layout: gl.constexpr = gl.BlockedLayout(
    size_per_thread=[R], threads_per_warp=[64], warps_per_cta=[num_warps], order=[0]
)
indices = gl.arange(0, XBLOCK, layout=layout)
offsets = start + indices              # layout propagates
mask = offsets < xnumel                # layout propagates
value = gl.load(in_ptr + offsets, mask=mask)
```

In practice you annotate the `gl.arange` and almost nothing else. Where a value must change layout,
you say so explicitly with `gl.convert_layout` — see [layouts.md](layouts.md).

> **CDNA note.** `threads_per_warp` is **64** on CDNA, not 32. Every layout literal copied from an
> NVIDIA Gluon tutorial or from the upstream `02-layouts.py` benchmark has `threads_per_warp=[32]`
> and is wrong here. This is the single most common porting error, and it usually shows up as a
> compile-time layout mismatch rather than as a wrong answer — but do not rely on that.

## What Gluon hands you that Triton keeps

| Concern | Triton | Gluon |
|---|---|---|
| Tile layout | compiler-assigned | **explicit** `gl.BlockedLayout` / shared / MFMA layout objects |
| Shared memory | compiler-allocated | **explicit** `gl.allocate_shared_memory(dtype, shape, layout)` |
| Pipeline depth | `num_stages` knob, stream-pipeliner places it | **explicit** — you author the stages |
| Register pressure | compiler allocates, may spill | **you** budget live values against 512 VGPR/EU |
| Matrix op | `tl.dot`, compiler picks the MFMA | **you** issue the MFMA (incl. CDNA4 scaled) |
| Async global→LDS | compiler may or may not emit it | **explicit** `async_copy` group ops |
| Wave schedule | implicit | hand-authored (ping-pong / interleave) |

## Shared memory

```python
smem = gl.allocate_shared_memory(dtype, shape, layout)   # -> shared_memory_descriptor
```

The returned descriptor is what async copies target and what MFMA operands are read from. The layout
argument is a *shared* layout (`gl.SwizzledSharedLayout` and friends), not a blocked layout — see
[layouts.md](layouts.md) § Shared layouts. Reads and writes are affected by **both** the shared layout
and the register layout of the tensor involved, because LDS is banked and a bank serves one address
per cycle per warp.

## Portability boundaries — what does NOT transfer to AMD

The upstream Gluon tutorial series is mostly written against NVIDIA hardware. These parts have no CDNA
equivalent and must not be copied:

- **`gl.warp_specialize`** — Hopper and newer NVIDIA only. On CDNA, pipelining is the async-copy group
  mechanism plus hand-authored wave scheduling. See [amd_targets.md](amd_targets.md).
- **TMA** (`tcgen05`, `NVMMASharedLayout`, `fence_async_shared`, mbarrier-based descriptor pipelines,
  the `conv-im2col` tutorial) — NVIDIA. The AMD analogue of the direct-to-shared path is
  `async_copy.buffer_load_to_shared`.
- **Multi-CTA / `cga_layout` / cluster fences** — NVIDIA Blackwell.
- **Tensor Memory register layouts** — NVIDIA Blackwell.

What **does** transfer: `@gluon.jit`, the launch surface, `constexpr`, `gl.arange`/`gl.load`/`gl.store`,
`gl.BlockedLayout` / `gl.SliceLayout` / `gl.convert_layout`, `gl.allocate_shared_memory`, the
reduction/scan ops, and the whole layout-as-type mental model.

## Status and stability

Gluon lives under `triton.experimental` and is **not a stabilized API** as of Triton 3.7. It has
shipped release-to-release breakage — see `../skills/optimize/gluon_levers/forge_integration.md`
§ Version traps before you build anything on a specific symbol. Probe the surface you intend to use on
the actual build:

```bash
python -c "
import triton
from triton.experimental import gluon
print('triton', triton.__version__)
print('gluon exports', sorted(getattr(gluon, '__all__', [])))
"
```

## Sources
- Gluon overview (what it is, what it exposes): https://triton-lang.org/main/gluon/index.html
- Introduction to Gluon (`@gluon.jit`, launcher, constexpr, autotune, CTA scope):
  https://triton-lang.org/main/getting-started/tutorials/gluon/intro.html
- Tensor Layouts (distribution hierarchy, `gl.arange(layout=)` seeding, propagation):
  https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html
- Warp specialization is Hopper+:
  https://triton-lang.org/main/getting-started/tutorials/gluon/warp-specialization.html
- `gluon` dialect reference: https://triton-lang.org/main/dialects/GluonDialect.html
