---
title: Gluon on AMD Instinct — when to reach for it, and the measured optimization ladder
kind: language
gens: [gfx942, gfx950]
dtypes: [fp16, bf16, fp8_e4m3, fp8_e5m2, fp4_e2m1, mxfp4]
regimes: [prefill, training, both]
status: experimental
updated: 2026-08-23
sources:
  - https://triton-lang.org/main/gluon/index.html
  - https://github.com/ROCm/gfx950-gluon-tutorials
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
  - https://arxiv.org/abs/2511.08083
---

# Gluon on AMD — authoring overview

## TL;DR
Gluon is the answer to one specific diagnosis: **the compiler's schedule is the bottleneck, not the
hardware.** Triton's model asks the compiler to rediscover pipeline structure from generic IR, and the
last 10–20% is where that breaks down. Gluon keeps Triton's tile-level model and Python frontend but
hands you layouts, shared memory, pipeline stages, register budget and the MFMA itself. AMD's public
GEMM ladder walks one FP16 kernel from **~520 TFLOPS at ~25% MFMA efficiency to ~1489 TFLOPS at
~99%** on MI355X in nine measured steps. The cost is real: you now own the layout, the pipeline, the
register budget and the schedule, and one of those nine steps is a **73% regression** that took two
more steps to unwind.

**Reach for Gluon when Triton's autotune has converged and MFMA efficiency is still low.** That
combination — converged search, idle matrix core — is the signature of a scheduling problem the
compiler cannot see past, and it is the only reliable reason to pay Gluon's price.

## Where it fits

| Reach for Gluon when | Stay on Triton when |
|---|---|
| Autotune converged (top-3 configs within ~2%) but PMC shows MFMA efficiency far below peak | Autotune has not converged — cheaper axes remain |
| You need explicit ping-pong / interleave wave scheduling | Portability across NVIDIA and AMD matters |
| You need CDNA4 native scaled MFMA (MXFP4 / block-scaled a4w4) | You are still exploring shapes or the algorithm |
| The operand layout the compiler picked is provably wrong and you cannot steer it with knobs | The win is fusion the library cannot express — Triton already does that well |
| You have session budget for hand layout + pipeline work | You need the `torch.compile` / Inductor codegen path |

And the honest boundary in the other direction: on a **plain dense GEMM**, tuned hipBLASLt / aiter /
CK / hand asm still generally beat everything compiled from a Python DSL. Hand-tuned AMD kernels
outperform the Triton compiler by **1.3–3.0×** on the shapes HipKittens evaluated. Gluon narrows that
gap a long way — it is how a Python DSL got to 99% MFMA efficiency at all — but "Gluon exists" is not
a reason to write a GEMM that a library call already serves.

## The measured ladder (AMD's a16w16 FP16 GEMM, v0 → v9)

This is the reference progression, and it is worth reading as a **sequence of diagnoses** rather than a
recipe. Each rung isolates one idea and is measured on its own. The same skeleton then carries to BF8
and MXFP4 with a larger `BLOCK_K` and the scaled MFMA.

**Act I — get the basics right (v0–v3)**
1. **`v0_naive`** — a correct FP16 GEMM with explicit layouts. ~520 TFLOPS, ~25% MFMA efficiency. The
   matrix core is idle most of the time. *Start here every time: correct first, with the layouts
   written down.*
2. **`v1_buffer_load`** — masked loads become AMD buffer ops. Out-of-bounds handling moves into
   hardware; **140 control-flow branches collapse to 4**. Mechanical, reliable, do it early.
3. **`v2_async_copy`** — global memory goes **directly to LDS**, eliminating register staging and
   **every `ds_write` in the inner loop**. This is the largest structural win available and the reason
   the AMD namespace exists.
4. **`v3_lds`** — kill LDS bank conflicts by comparing **raw vs swizzled vs padded** shared layouts *at
   the instruction level* and picking whichever hits the steady-state `ds_read` issue rate. Measured,
   not reasoned.

**Act II — hide latency (v4–v5)**
5. **`v4_global_prefetch`** — a two-stage software pipeline so iteration `i+1`'s data is in flight
   while iteration `i` computes. This is what replaces `num_stages`.
6. **`v5` — the LLIR scheduler** (`TRITON_ENABLE_LLIR_SCHED=1`). Interleaves MFMA with memory ops from
   a throughput model and disables LLVM's pre-RA and post-RA machine schedulers to preserve that
   ordering. Without it the backend clusters all MFMAs together, causing spills and MFMA stalls.

**Act III — the v6 regression (the most useful rung)**
7. **`v6` regresses ~73%** with `llirSched` on. The scheduler did not break anything; it **exposed a
   register-pressure problem** that the previous clustering had masked. *A change that makes things
   worse by exposing a real constraint is not a change to revert — it is a diagnosis.* This is the
   rung most worth internalizing, because the instinct inside an optimization loop is to revert and
   move on, and reverting here forfeits everything after it.

**Act IV — recovery and beyond the hot loop (v7–v9)**
8. **`v7` — slicing** resolves the register pressure v6 exposed.
9. **`v8`/`v9`** — the post-assembly peephole (`TRITON_ENABLE_AMDGCN_AS=1`: `amdgpu-agpr-alloc=256`
   to reserve AGPRs for MFMA accumulators, `amdgpu-mfma-vgpr-form=false` to keep accumulators out of
   VGPRs, plus post-assembly LICM hoisting LDS address arithmetic into the prologue) and **XCD-aware
   workgroup remapping**.

**The final shape, for all three dtypes:** M+N slicing, a **3-stage** pipeline, loop unrolling by 2,
`llirSched` and `amdgcnas`.

### The numbers, and their caveat
| kernel | dtype | shape | TFLOPS | MFMA eff |
|---|---|---|---|---|
| a16w16 v0 (naive) | FP16 | — | ~520 | ~25% |
| a16w16 v9 | FP16 | 4096×4096×8192 | ~1489 | ~99% |
| a8w8 | BF8 | 4096×4096×16384 | ~3257 | ~99.7% |
| a4w4 | MXFP4 | 4096×4096×32768 | ~5255 | ~92.4% |

⚠️ **Treat these as orders of magnitude, not as a target to hit.** AMD's own two READMEs disagree
(the repo top-level quotes ~541 → ~1421, the GEMM README ~520 → ~1489), and the blog and the repo pin
**different annotated Triton tags** (`gfx950-tutorial-v0.1` vs `gfx950-tutorial-v0.2`). They are
gfx950 / ROCm 7.0 / AMD-measured on large-K square-ish shapes. **Your baseline is what you measured on
your box**, and inside a forge campaign it is the pristine measurement the loop took — never a number
from this table.

Two more boundaries on that table:
- **MXFP4's lower ceiling is structural**, not a tuning failure: the a4w4 kernel runs a *separate scale
  pipeline* (global-read → LDS-write → LDS-read) alongside the data pipeline, and the resulting LDS
  port contention is what caps it near 92% while BF8 reaches ~99.7%.
- **These are compute-bound regimes** (K = 8192 / 16384 / 32768). Skinny and decode-shaped GEMM is a
  different problem and none of these ceilings apply.

## Wave scheduling: the two patterns that work on CDNA

Do not port NVIDIA's producer/consumer warp specialization. It reaches only ~80% of peak BF16 GEMM on
MI355X because static register allocation starves the producer waves, and `gl.warp_specialize` is
Hopper-and-newer NVIDIA only in any case. The two patterns that reach peak are **8-wave ping-pong** and
**4-wave interleave**; mechanics, primitives and the CDNA3/CDNA4 generality claim are in
[`../../../API_docs/amd_targets.md`](../../../API_docs/amd_targets.md) § 5.

Prefer **4-wave interleave** when you have a choice: it needs no `#pragma unroll` tuning and holds up
better across ROCm releases.

## Method — how to actually work the ladder

1. **Correct first, with layouts written down.** A naive Gluon kernel at 25% MFMA efficiency is a
   *successful* v0. Do not optimize an incorrect kernel.
2. **One rung per measurement.** Every rung above isolates one idea. Two ideas in one candidate and
   you learn nothing from the number.
3. **Read the ISA, not just the clock.** Bank conflicts, spills, branch counts and MFMA clustering are
   all visible in the AMDGCN dump and invisible in wall time until they are large. The workflow is
   shared with Triton: `../../../../triton/skills/optimize/triton_levers/triton_isa_check.md`.
4. **Watch register pressure at every rung.** It is the constraint that binds, it is why v6 regressed,
   and it is the thing a change three rungs earlier silently spends.
5. **A regression that exposes a constraint is information.** Before reverting, establish *what* got
   worse. See v6.
6. **Sweep the constants you introduced, don't argue about them.** Pipeline depth, unroll factor and
   tile dims are `constexpr` in your own source — exactly what `FORGE_SWEEP_*` is for. See
   `common_methodology/optimization/lever_cheap_sweeps.md`.

## Cross-links
- Declare/launch/autotune and the layout-typed value model:
  [`../../../API_docs/programming_model.md`](../../../API_docs/programming_model.md)
- Layout objects, conversion costs, LDS banking:
  [`../../../API_docs/layouts.md`](../../../API_docs/layouts.md)
- Buffer ops, async copy to LDS, scaled MFMA, wave patterns, `llirSched`/`amdgcnas`:
  [`../../../API_docs/amd_targets.md`](../../../API_docs/amd_targets.md)
- **Working inside a forge campaign:** [`forge_integration.md`](forge_integration.md)
- Shared Triton substrate (lowering pipeline, ISA verification): `../../../../triton/`
- Hardware constants (wavefront, LDS, VGPR, MFMA shapes): `../../../../../hardware/`

## Sources
- Gluon overview / why the last 10–20% is hard for a compiler:
  https://triton-lang.org/main/gluon/index.html
- The v0→v9 ladder, per-rung diagnoses, v6 regression, final design, env flags:
  https://github.com/ROCm/gfx950-gluon-tutorials ·
  https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
- Ping-pong / interleave origin and the CDNA scheduling argument:
  https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
- Wave specialization at ~80% of peak; hand-tuned vs Triton 1.3–3.0×; >95% of peak across CDNA3/CDNA4:
  https://arxiv.org/abs/2511.08083
