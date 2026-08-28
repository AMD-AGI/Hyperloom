---
title: Gluon layouts — blocked, slice, shared, MFMA; conversion costs and bank conflicts
kind: api_reference
gens: [gfx942, gfx950]
dtypes: [both]
regimes: [both]
status: experimental
updated: 2026-08-23
sources:
  - https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html
  - https://triton-lang.org/main/gluon/index.html
  - https://triton-lang.org/main/dialects/GluonOps.html
  - https://arxiv.org/abs/2505.23819
---

# Gluon layouts

The object Triton hides and Gluon makes you name. Read after
[programming_model.md](programming_model.md).

## TL;DR
A layout states which element is owned by which (register, lane, warp). `BlockedLayout` is the
workhorse and its block shape is the elementwise product of its three size vectors —
**on CDNA `threads_per_warp` must multiply out to 64, not 32**. There is **no canonical layout**, so
two different layout objects can describe the same mapping; `gl.convert_layout(..., assert_trivial=True)`
is how you assert a conversion is free instead of hoping. A non-trivial conversion moves data between
lanes and, across warps, through LDS — so a stray conversion in a hot loop is a real cost, and the ISA
dump is where you find it. Do **not** convert before a reduction: the compiler emits efficient
reductions from any layout, so the conversion is pure overhead.

## The inventory

Read off a live build (Triton 3.6.0, gfx950) — re-probe with
`python -c "from triton.experimental.gluon import language as gl; print([n for n in dir(gl) if 'Layout' in n])"`
rather than trusting this list:

| Layout | Kind | Use |
|---|---|---|
| `BlockedLayout` | distributed | the workhorse; coalesced global access |
| `SliceLayout` | distributed | a parent layout with one dim removed; how you build 1D indices for a 2D tile |
| `CoalescedLayout` | distributed | ask for a coalesced arrangement without spelling out the fields |
| `AutoLayout` | distributed | **let the compiler pick** — the Triton-like escape hatch (see below) |
| `DotOperandLayout` | distributed | matrix-core operand feed |
| `DistributedLinearLayout` | distributed | the fully general form |
| `SwizzledSharedLayout` | shared | address-permuted LDS; the standard bank-conflict fix |
| `PaddedSharedLayout` | shared | padded-stride LDS; trades capacity for conflict-freedom |
| `SharedLinearLayout` | shared | the fully general shared form |
| `NVMMADistributedLayout`, `NVMMASharedLayout` | — | **NVIDIA only**, ignore on CDNA |

Plus the AMD matrix-core layout at `gl.amd.AMDMFMALayout` (and `AMDWMMALayout` for RDNA) — see
[amd_targets.md](amd_targets.md).

> **`AutoLayout` is a legitimate starting point.** A first correct Gluon version does not have to name
> every layout; you can let the compiler choose and then replace the ones the ISA says are costing you.
> That keeps v0 short and makes each later layout an isolated, measurable change — which is exactly the
> rung discipline the ladder wants. What you must not do is leave `AutoLayout` on a hot operand feed
> and then wonder why Gluon is not beating Triton: at that point you have paid Gluon's cost and taken
> none of its benefit.

## `BlockedLayout` — the workhorse

```python
gl.BlockedLayout(
    size_per_thread=[2, 4],     # contiguous subtile each thread owns, in registers
    threads_per_warp=[16, 4],   # product MUST be the wavefront size: 64 on CDNA
    warps_per_cta=[2, 2],       # product should be num_warps
    order=[1, 0],               # dimension tiling order, fastest-varying last
)
```

The **block shape is the elementwise product** of the three vectors — here `[2*16*2, 4*4*2]` =
`[64, 32]`. Within the block the layout is a hierarchy of register tiling, then thread tiling, then
warp tiling, applied in `order`. `size_per_thread=[2, 4]` means each thread holds a contiguous 2×4
subtile in its own registers.

Blocked layouts exist mainly to describe **coalesced global-memory access**. The rule of thumb is
unchanged from Triton: make the fastest-varying dimension of `order` the one that is contiguous in
memory, and give threads enough contiguous elements (`size_per_thread` along that dimension) to widen
the load.

### CDNA arithmetic
- Wavefront is **64 lanes**. `prod(threads_per_warp) == 64`. Every literal in the upstream
  `02-layouts.py` benchmark says 32 — those are NVIDIA warps.
- For a 1D tile at `num_warps=4` the entire space of blocked layouts is
  `gl.BlockedLayout([R], [64], [4], [0])` for power-of-two `R`. `R` is the vectorization width; it is
  the axis the upstream `R_vs_throughput` experiment sweeps, and it is a legitimate cheap sweep.
- Tile dimensions are powers of two, so elements-per-thread is a power of two and the tile must divide
  evenly over `64 * num_warps` lanes.

## `SliceLayout` — building lower-rank indices

`gl.arange` accepts a `SliceLayout` to produce a 1D index tensor consistent with a 2D layout, which is
how you build row and column offsets that combine without a conversion:

```python
coalesced_2d: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [1, gl.num_warps()], [1, 0])
row = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, coalesced_2d))
col = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, coalesced_2d))
offsets = row[:, None] * stride_m + col[None, :] * stride_n     # lands in coalesced_2d
```

`gl.SliceLayout(dim, parent)` is "the parent layout with `dim` removed". Getting the `dim` wrong is a
common error and shows up as an unexpected `convert_layout` rather than a compile failure.

## Conversions and what they cost

There is **no canonical layout representation** — multiple layouts express the same element mapping.
For example these two are equivalent:

```python
gl.BlockedLayout([1], [64], [4], [0])
gl.SliceLayout(1, gl.BlockedLayout([1, 1], [64, 1], [4, 1], [1, 0]))
```

So conversion has three tiers, and you should know which one you are paying for:

| Tier | What moves | How to get it |
|---|---|---|
| Free | nothing (relabel), or register reordering within a thread | `gl.convert_layout(x, layout, assert_trivial=True)` — **raises** if it is not free |
| Lane-crossing | data between lanes of one wave | permutes/shuffles; real but bounded |
| Warp-crossing | data between warps → **through LDS** | the expensive one; round-trips shared memory |

**Always pass `assert_trivial=True` when you believe a conversion is free.** It converts a silent
performance regression into a compile-time error, which is the only way to keep the belief honest as
the kernel changes around it.

### The reduction anti-pattern
The compiler generates efficient reductions and scans **regardless of input layout**. Converting to a
"reduction-friendly" layout and then reducing is therefore typically *more* expensive than reducing in
place. Only prefer a reduction-friendly layout when you have a free choice between layouts of equal
cost elsewhere.

## Shared layouts

`gl.allocate_shared_memory(dtype, shape, layout)` takes a **shared** layout, not a blocked one.

LDS is organized into banks; a bank serves one address per cycle per warp, so two lanes hitting
different addresses in the same bank serialize. The compiler minimizes conflicts, but **the layouts
still decide how many are left** — and both the shared layout and the register layout of the tensor
being read or written matter.

Three shapes are worth knowing on CDNA, and all three are first-class objects — this is exactly the
raw / swizzled / padded comparison the AMD GEMM ladder runs at its LDS rung:

- **`gl.SharedLinearLayout`** (or the naive arrangement) — fine for a first correct version; usually
  conflict-heavy for MFMA operand feeds.
- **`gl.SwizzledSharedLayout`** — permutes addresses so a wave's lanes spread across banks. The
  standard fix; the upstream tutorial's own issue tracker shows it does not always behave as expected,
  so **verify against the instruction stream, do not assume**.
- **`gl.PaddedSharedLayout`** — pads the stride so it is coprime with the bank count. Costs LDS
  capacity (64 KB/CU on CDNA3, 160 KB/CU on CDNA4 — see `hardware/`), so it trades occupancy for
  conflict-freedom, and a padding that fits on CDNA4 may not fit on CDNA3. It carries a
  `with_identity_for(...)` helper for deriving the padded form from an existing layout.

The AMD GEMM ladder picks between exactly these three by **comparing them at the instruction level and
measuring the steady-state `ds_read` issue rate** — not by reasoning about them. Do the same: the
bank-conflict model in `ROCm/gfx950-gluon-tutorials:docs/lds_throughput.md` tells you what to expect,
the ISA dump tells you what you got.

**`gl.amd.*` also exposes a transposing LDS read** (the `ds_read_tr` family) so an MFMA operand can be
fed in transposed order without a separate pass — see [amd_targets.md](amd_targets.md).

## MFMA / dot-operand layouts

Matrix-core operands need layouts the matrix core can consume; on AMD that is
`gl.amd.AMDMFMALayout(version, instr_shape, transposed, warps_per_cta, element_bitwidth=None,
tiles_per_warp=None, ...)`, fed through `gl.DotOperandLayout`. `instr_shape` is where you pick the MFMA
variant and `tiles_per_warp` controls contiguous per-warp tile computation.

Which MFMA shape wins on which arch and dtype is a Triton-substrate question and is **not duplicated
here** — read `../triton/skills/optimize/triton_levers/triton_lowering.md` (the `tl.dot` → MFMA mapping
and layout selection) and the `hardware/` matrix-core cards. The Gluon-specific part is only that you
name the layout instead of receiving it — and that changing `instr_shape` on a *scaled* MFMA also
changes the scale packing order, which is a silent-wrong-answer trap covered in
[amd_targets.md](amd_targets.md) § 4.

## Linear layouts — the escape hatch

Every Gluon layout is representable as a **linear layout**, the most expressive form, which allows
zero-cost splits, joins, reshapes and permutes. They are uncommon and hard to read; reach for one only
when the structured layouts cannot express the mapping you need. Reference:
`include/triton/Tools/LinearLayout.h` and the paper at https://arxiv.org/abs/2505.23819.

## Debugging layouts
- The upstream tutorial repo ships a **layout plotter** (`layout_plot/` in
  `ROCm/gfx950-gluon-tutorials`) that renders blocked, dot and LDS layouts to LaTeX. When a layout
  argument is not obviously right, draw it.
- Unexpected `convert_layout` in the IR is the signal that two adjacent ops disagree. Find the *first*
  one in program order — later ones are usually consequences.
- `AMDGCN_ENABLE_DUMP=1` and the ISA workflow are shared with Triton:
  `../triton/skills/optimize/triton_levers/triton_isa_check.md`. Gluon lowers through the same backend, so
  everything that card says about reading the dump applies unchanged.

## Sources
- Tensor Layouts tutorial (distribution hierarchy, `BlockedLayout` fields and block-shape arithmetic,
  `SliceLayout` for 2D offsets, no-canonical-layout + `assert_trivial`, the reduction anti-pattern,
  LDS banking, linear layouts):
  https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html
- Gluon overview (layouts/shared memory as first-class): https://triton-lang.org/main/gluon/index.html
- GluonOps dialect reference: https://triton-lang.org/main/dialects/GluonOps.html
- `SwizzledSharedLayout` not always behaving as expected (verify, don't assume):
  https://github.com/triton-lang/triton/issues/8149
- LDS bank-conflict model + layout plotter: https://github.com/ROCm/gfx950-gluon-tutorials
- Linear layouts paper: https://arxiv.org/abs/2505.23819
