---
title: Gluon AMD target namespaces — buffer ops, async copy to LDS, scaled MFMA (gl.amd.cdna3 / cdna4)
kind: api_reference
gens: [gfx942, gfx950]
dtypes: [fp16, bf16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp8_e4m3, fp8_e5m2, fp4_e2m1, mxfp4]
regimes: [prefill, training, both]
status: experimental
updated: 2026-08-23
sources:
  - https://triton-lang.org/main/gluon/api/amd.html
  - https://triton-lang.org/main/gluon/api/amd.cdna4.html
  - https://triton-lang.org/main/dialects/TritonAMDGPUOps.html
  - https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
  - https://github.com/ROCm/gfx950-gluon-tutorials
---

# Gluon AMD target namespaces

The CDNA-specific surface: what `gl.amd.cdna3` / `gl.amd.cdna4` add over portable Gluon. This is the
part of the language with no NVIDIA equivalent, and the part that pays for choosing Gluon at all.

> **Read the API listing on YOUR build.** These live under `triton.experimental` and the signatures
> move between releases. The authoritative pages are
> https://triton-lang.org/main/gluon/api/amd.html and
> https://triton-lang.org/main/gluon/api/amd.cdna4.html — but what your interpreter imports is what
> you have. `python -c "from triton.experimental.gluon.language.amd import cdna4; print(dir(cdna4))"`
> costs a second and settles it.
>
> **The listings below were read off a live build: Triton 3.6.0, gfx950, ROCm.** Treat them as a
> concrete example of the shape, not as a spec — re-probe rather than trusting this card's inventory.

## The namespace tree

```
triton.experimental.gluon.language.amd
├── AMDMFMALayout, AMDWMMALayout        # layouts, shared across generations
├── cdna3/  cdna4/                      # CDNA (Instinct)
└── rdna3/  rdna4/  gfx1250/            # RDNA — not covered by this folder
```

**`cdna3` and `cdna4` are not the same size, and the gap is the point.** Observed on Triton 3.6.0:

| | `cdna3` (gfx942) | `cdna4` (gfx950) |
|---|---|---|
| `buffer_load` / `buffer_store` | ✅ | ✅ |
| `buffer_atomic_*` (add/and/max/min/or/xchg/xor) | ✅ | ✅ |
| `mfma` | ✅ | ✅ |
| `async_copy` (→ LDS) | ❌ **absent** | ✅ |
| `mfma_scaled` + `get_mfma_scale_layout` | ❌ **absent** | ✅ |
| `AMDMFMALayout` / `DotOperandLayout` re-export | ❌ (use `gl.amd.AMDMFMALayout`) | ✅ |

So on CDNA3 the Gluon-specific win narrows to buffer ops plus hand-authored pipelining and wave
scheduling; **the async-copy-to-LDS rung and the whole scaled-MFMA route are CDNA4-only at the Gluon
API level.** Plan a CDNA3 kernel accordingly rather than discovering it three rungs in.

## TL;DR
Three AMD mechanisms carry essentially all of the win:

1. **Buffer ops** (both gens) — address global memory as `(scalar base pointer, tensor of offsets)`
   instead of a tensor of pointers, so bounds handling moves into the buffer descriptor and the
   address arithmetic shrinks.
2. **Async copy to LDS** (CDNA4) — `async_copy.buffer_load_to_shared` + `commit_group` / `wait_group`
   moves global → LDS **without staging through registers**. This is the CDNA analogue of NVIDIA's
   `cp.async` / TMA path and it is where the biggest single measured jump in the AMD ladder comes from.
3. **Scaled MFMA** (CDNA4) — one instruction consumes FP8/FP6/FP4 operands plus an E8M0 block scale,
   accumulating in FP32. This is the MXFP4 route and it does not exist on CDNA3.

Plus a transposing LDS read (`ds_read_tr` family) that feeds MFMA operands in the right order without a
separate transpose pass.

## 1. Buffer ops — both generations

```python
from triton.experimental.gluon.language.amd import cdna4   # or cdna3

cdna4.buffer_load(ptr, offsets, mask=None, other=None, cache=None)
cdna4.buffer_store(...)
cdna4.buffer_atomic_add / _and / _max / _min / _or / _xchg / _xor
```

AMD buffer instructions take a **scalar base pointer plus a tensor of offsets**, unlike Triton's usual
tensor-of-pointers addressing. Two consequences:

- **Bounds handling moves into the descriptor.** The buffer descriptor carries the extent, so an
  out-of-range offset returns zero (or is dropped on store) in hardware. `mask=` is still accepted —
  it is optional, not forbidden — but the common case no longer needs it, and that is where the
  branches go away.
- **Address arithmetic shrinks.** One scalar base plus 32-bit offsets instead of a full 64-bit pointer
  per element means far fewer VALU ops and far less register pressure on the address path.

In the AMD GEMM ladder, switching masked loads to buffer loads collapsed **140 control-flow branches
down to 4**. That is the whole content of that rung — it is a mechanical substitution with a large,
reliable payoff, and it is usually the first thing to do after a correct naive version.

Practical constraint: the offsets must fit the buffer descriptor's 32-bit offset space, so a tensor
larger than 4 GiB needs the base pointer advanced per tile rather than a single base for the whole
tensor.

## 2. Async copy to LDS — CDNA4

`cdna4.async_copy` exposes:

```python
async_copy.buffer_load_to_shared(...)   # buffer addressing -> LDS, no register staging
async_copy.global_load_to_shared(...)   # pointer addressing -> LDS
async_copy.commit_group()               # close the current group
async_copy.wait_group(N)                # wait until at most N groups remain outstanding
async_copy.load_shared_relaxed(...)     # relaxed read back out of LDS
```

The commit/wait pair mirrors the TritonGPU-dialect async-group semantics shared with the NVIDIA
`cp.async` path; the AMD-specific piece is `buffer_load_to_shared`, which fuses the buffer addressing
above with a direct global→LDS write.

**There is no `async_copy` in the `cdna3` namespace** — see the table above. A CDNA3 kernel stages
through registers and writes LDS explicitly, which is exactly the cost this rung removes on CDNA4.

**Why it matters, in measured terms.** Staging global data through registers before writing LDS costs
a full register-residency phase: ROCm's own tuning work reports that switching a reference GEMM to
direct L1→LDS copy saved **~100 VGPR per wave**, removed an entire register-movement phase, and moved
the kernel from **697 to 1113 TFLOPS**. In the Gluon ladder the corresponding rung eliminates *every*
`ds_write` in the inner loop.

**How you use it.** Allocate a multi-buffered shared descriptor (one slot per pipeline stage), issue
the copy for stage `i+1`, `commit_group()`, then `wait_group(depth-1)` before reading stage `i`. That
loop *is* the software pipeline — there is no `num_stages` to set.

> **Version trap.** AsyncCopy-by-default for gfx950/gfx1250 was enabled upstream and then **reverted
> on the `release/3.7.x` branch**. So whether async copy happens without you asking differs between
> `main` and a 3.7.x wheel. In Gluon you are issuing it explicitly, which is precisely the point — but
> do not read a Triton-side benchmark as telling you what your Gluon kernel does.

> **Correctness trap.** Triton **3.7.1** fixed a missing fence between a shared-memory store and an
> async `copy_local_to_global`: without it the async copy could read shared memory *before* the store
> completed, producing **silently incorrect results**. If you are on 3.7.0 and you build an async
> shared-memory pipeline, you are exposed. See
> `../skills/optimize/gluon_levers/forge_integration.md` § Version traps.

## 3. Transposing LDS reads

The `ds_read_tr` family reads LDS with a transpose, so an MFMA operand that is stored one way and
consumed the other way does not need a separate transpose pass or a second LDS round-trip. Reach for
it when the natural global layout of one operand disagrees with the MFMA operand layout — which for a
row-major × row-major GEMM is the common case.

## 4. Matrix core: `mfma` and `mfma_scaled`

```python
cdna4.mfma(a, b, acc)
cdna4.mfma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc)   # CDNA4 only
cdna4.get_mfma_scale_layout(...)                                     # constexpr helper
```

`mfma` exists on both generations. `mfma_scaled` is the native scaled matrix-core op and is **CDNA4
only**: one instruction consumes low-precision operands **plus a block scale** and accumulates in FP32,
mapping to `v_mfma_scale_f32_16x16x128_f8f6f4` on the hardware. Note that `a_format` / `b_format` are
explicit arguments — the operand encoding is something you declare, not something inferred from the
tensor dtype.

The operand layout is `gl.amd.AMDMFMALayout`:

```python
AMDMFMALayout(version, instr_shape, transposed, warps_per_cta,
              element_bitwidth=None, tiles_per_warp=None, cga_layout=...)
```

`instr_shape` is where you choose the MFMA variant, and `tiles_per_warp` is the knob for computing
contiguous tiles per warp.

### ⚠️ Scale packing order differs between variants
| Variant | Scale packing order |
|---|---|
| `mfma_scaled_16x16x128` | `op_0, op_2, op_1, op_3` |
| `mfma_scaled_32x32x64`  | `op_0, op_1, op_2, op_3` |

**This is a silent-wrong-answer trap.** The packing order is not symmetric between the two variants,
so swapping the MFMA shape without re-packing the scales compiles, runs, and returns plausible
garbage.

**Use `get_mfma_scale_layout` rather than hand-packing.** It exists precisely so the scale layout is
derived from the chosen instruction rather than transcribed, which is the difference between a variant
change being a one-line edit and being a silent regression. If you do hand-pack, re-run the task's
correctness command after any change to `instr_shape` — before you look at timing at all.

### Data formats
- **fp4 (e2m1)** is packed **two elements per `uint8`**, normally along the reduction (K) dimension.
  The **low 4 bits hold the first element, the high 4 bits the second.**
- **MX scales are e8m0** — 8 exponent bits, 0 mantissa bits — representing powers of two from
  `2**-127` to `2**127`, with `255` reserved as NaN. One scale per group of 32 elements.
- The dialect-level upcast (`TritonAMDGPUOps`) takes fp4-as-i8 and an E8M0 scale encoded as BF16 and
  lowers to `v_cvt_scalef32_*`.

### Gating on the architecture
Detect CDNA4 rather than assuming it — the same source may be compiled for both. The upstream idiom is
an `is_hip_cdna4()`-style helper that checks the backend is `'hip'` **and** the arch matches the gfx
target. Do not branch on the SKU name or on an environment variable a benchmark happened to set.

### fp8 dialect, the other silent-wrong-answer trap
This is inherited from the Triton substrate and it bites just as hard here: fp8 is **FNUZ on CDNA3
(gfx942)** and **OCP on CDNA4 (gfx950)**. A mismatched dialect corrupts the descale and produces wrong
numbers rather than an error. See `../triton/` and the `hardware/` numerics cards.

## 5. Wave scheduling — what is and is not available

**`gl.warp_specialize` is Hopper-and-newer NVIDIA only.** There is no CDNA path through it, and there
should not be: on MI355X wave specialization reaches only **~80% of peak BF16 GEMM**, because AMD's
static register allocation starves the producer waves. The two patterns that do reach peak, both from
HipKittens (arXiv 2511.08083):

- **8-wave ping-pong** — split 8 waves into two groups of 4. Within a group, one wave issues matrix ops
  while another issues memory ops; then the roles swap. Bulk global→LDS→register movement overlaps
  MFMA, coordinated by explicit software barriers.
- **4-wave interleave** — one wave per SIMD, each issuing small tightly-interleaved load/compute
  groups. Gets the full 512-VGPR budget per wave. This is the more robust of the two: no `#pragma
  unroll` tuning, and it holds up better across ROCm releases.

HipKittens reports the same 8-wave schedule delivering **>95% of peak on both CDNA3 and CDNA4** with
only shared-memory-size adjustments — so the pattern generalizes across the two archs even though the
scaled-MFMA route does not.

The primitives underneath are `llvm.amdgcn.sched.group.barrier`, `llvm.amdgcn.sched.barrier` and
`s_setprio` — the same ones the AMD backend's own ping-pong scheduler pass uses. Note the distinction:
**the backend's ping-pong pass is a compiler transform on Triton-style code; hand-authored ping-pong in
Gluon is your code.** They are not the same lever and enabling one says nothing about the other.

## 6. Compiler-side scheduling passes

Two runtime-enabled passes matter enough that the AMD ladder treats them as part of the kernel:

```bash
TRITON_ENABLE_LLIR_SCHED=1     # LLIR-level pass: interleaves MFMA with memory ops from a throughput model
TRITON_ENABLE_AMDGCN_AS=1      # post-assembly peephole
```

- **`llirSched`** interleaves MFMA and memory instructions using a throughput model, and **disables
  LLVM's pre-RA and post-RA machine schedulers** to preserve that ordering. Without it the backend
  clusters all the MFMAs together, which causes register spills and MFMA stalls. With it — and this is
  the important part — **it can expose a register-pressure cliff that was previously hidden**; see the
  v6 regression in `../skills/optimize/gluon_levers/overview.md`.
- **`amdgcnas`** sets `amdgpu-agpr-alloc=256` (reserve AGPRs for MFMA accumulators) and
  `amdgpu-mfma-vgpr-form=false` (keep accumulators out of VGPRs), and runs a post-assembly LICM that
  hoists loop-invariant work such as LDS address computation into the loop prologue.

> **These are environment variables, so inside a forge campaign they are part of the measurement, not
> part of the kernel.** A number measured with them on and a number measured with them off are not
> comparable. Either set them in the source's own launch path so they travel with the candidate, or
> sweep them explicitly as `FORGE_SWEEP_*` knobs — see
> `../skills/optimize/gluon_levers/forge_integration.md`.

## Sources
- Gluon AMD API namespace: https://triton-lang.org/main/gluon/api/amd.html
- Gluon AMD CDNA4 API (buffer load via scalar base + offset tensor, scaled MFMA):
  https://triton-lang.org/main/gluon/api/amd.cdna4.html
- TritonAMDGPUOps (fp4-as-i8 upcast with E8M0-as-BF16 scale → `v_cvt_scalef32_*`, nibble order):
  https://triton-lang.org/main/dialects/TritonAMDGPUOps.html
- Scaled-MFMA variants and their differing scale packing orders; e8m0 range and NaN encoding; fp4
  2-per-uint8 packing along K; `is_hip_cdna4()` gating:
  https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html
- Async copy commit/wait group semantics (TritonGPU dialect):
  https://triton-lang.org/main/dialects/TritonGPUOps.html
- `buffer_load_to_lds` saving ~100 VGPR/wave, 697 → 1113 TFLOPS:
  https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html
- Buffer loads collapsing 140 branches to 4; `llirSched` / `amdgcnas` behavior and flags:
  https://github.com/ROCm/gfx950-gluon-tutorials ·
  https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- Wave specialization at ~80% of peak on MI355X; 8-wave ping-pong / 4-wave interleave; `sched.barrier`
  / `s_setprio` primitives; >95% of peak across CDNA3/CDNA4: https://arxiv.org/abs/2511.08083
- `warp_specialize` is Hopper+:
  https://triton-lang.org/main/getting-started/tutorials/gluon/warp-specialization.html
- AsyncCopy default enabled then reverted on release/3.7.x; 3.7.1 FenceAsync correctness fix:
  https://github.com/triton-lang/triton/releases
