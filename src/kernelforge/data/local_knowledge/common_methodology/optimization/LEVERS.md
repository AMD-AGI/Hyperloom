---
title: optimization levers — section index
kind: index
scope: common_methodology/optimization
gens: [gfx950]
dtypes: [bf16, fp16, fp8_e4m3, fp8_e5m2, fp6_e2m3, fp4_e2m1, int8]
regimes: [prefill, decode, both]
updated: 2026-08-28
---

# Optimization levers — MI350X · MI355X (gfx950)

One file per technique. Every card follows the same shape so you can jump straight to the part you
need:

**Route here when** (does this apply?) → **gfx950 constants** (numbers inline, no second lookup) →
**what to change, in order** → **Verify** → **Expected magnitude** → **Failure modes** → **Deeper**.

**Start at `lever_bottleneck_class.md`.** Every other lever assumes you already know which roof the
kernel sits under. Pulling one against the wrong bottleneck wastes the iteration and sometimes makes
things worse.

## Apply top-down

| # | Stage | Levers |
|---|---|---|
| 0 | **Classify** | `lever_bottleneck_class.md` |
| 1 | **Algorithm / loop form / fusion** | `lever_fusion.md`, `lever_loop_form.md` |
| 2 | **Feed the matrix cores** | `lever_mfma_sched.md`, `lever_prefetch.md`, `lever_lds_banks.md` |
| 3 | **Parallelism** | `lever_grid_sizing.md`, `lever_occupancy.md`, `lever_xcd_locality.md` |
| 4 | **Memory subsystem** | `lever_coalescing.md`, `lever_prefetch.md`, `lever_xcd_locality.md` |
| 5 | **Search the residual** | `lever_cheap_sweeps.md`, then `lever_autotune.md` |
| 6 | **Keep it correct** | `lever_numerics.md` — a gate, not an option |

**Loop form is decided at stage 1, before any knob below is reachable.** A `while` whose trip count or
addresses come from a tensor *load* silently forfeits both `num_stages` pipelining and direct-to-LDS
async copy — so the stage-2 and stage-4 levers have nothing to act on and their sweeps read flat.

**Cutting across all of it: `lever_edit_surface.md`.** Before pricing a lever as unreachable ("that
would mean patching the framework, not this file"), work out what your permitted files actually reach.
Rebinding an installed symbol, injecting device source through the framework's own hook, a module
constant another module's dispatch reads (an `os.environ` default does not put it off-limits), and a
row in a permitted data file are all edits *inside* the permitted set.

## The cards

| Card | Routes from | Covers |
|---|---|---|
| `lever_bottleneck_class.md` | **entry point** | AI vs ridge, the four classes, gfx950 machine balance, the ~45–55% reality, re-classifying after fusion |
| `lever_mfma_sched.md` | compute-bound | 16×16 vs 32×32 (the 4× C-register argument), 8-wave ping-pong / 4-wave interleave, multiple accumulators, `OPTIMIZE_EPILOGUE`, block-scaled MXFP |
| `lever_occupancy.md` | latency / occupancy-bound | 512 regs/SIMD, 16-granule, AGPR pool, the 160 KiB LDS denominator, `waves_per_eu`, the spill cliff |
| `lever_lds_banks.md` | LDS-bound | **64 banks** (not 32), padding vs XOR swizzle, `ds_*_b128`, read-with-transpose, double-buffer on 160 KiB |
| `lever_prefetch.md` | latency-bound | 128-bit `global_load_lds`, software pipelining, `num_stages` 3–4 on gfx950, the flat-sweep diagnostic |
| `lever_loop_form.md` | **precondition for the two above** | shape-static trip count + addresses, the `while`/indirect-load signature, gather → static range + mask, why 2–4× more nominal work ran ~2× faster |
| `lever_coalescing.md` | bandwidth-bound | 128-bit `dwordx4`, alignment, lane mapping, grid-stride, coalescing ≠ bank conflicts |
| `lever_xcd_locality.md` | bandwidth-bound (re-fetch) | 8 XCDs × 32 CU, per-XCD L2, ≥1024 workgroups, 8-multiple tiles, swizzled CTA order, the 512 B stride cliff |
| `lever_grid_sizing.md` | latency / occupancy-bound | wave64, `num_warps`, `__launch_bounds__`, **256 CUs**, split-K for decode, persistent kernels |
| `lever_fusion.md` | bandwidth-bound; launch-bound | traffic fusion vs launch fusion, donors, when NOT to fuse, `launch_bound_share` and the 0.13 graph discount |
| `lever_cheap_sweeps.md` | stage 5 | `FORGE_SWEEP_<NAME>` + `sweep_const:` echo, one bench command per data point, joint sweeps, keep the knobs through the search |
| `lever_autotune.md` | stage 5 | only aiter's per-shape DB reaches the live path; capture → race → deploy → **prove engagement**; the 10-tuple key |
| `lever_edit_surface.md` | cross-cutting | what an editable file reaches: package rebind, injected device source, module constants (incl. `os.environ` defaults), data/config rows |
| `lever_numerics.md` | **gate on everything** | FP32 accumulate, online softmax, Welford, the **OCP** fp8 trap, MXFP block scaling, the `err_ratio` gate |

## gfx950 facts the cards assume

Stated here once so a card can be read standalone without a hardware lookup:

| | |
|---|---|
| 256 CU (8 XCD × 32), 4 SIMD/CU, 1024 matrix cores | wave64, 8 slots/SIMD → 32 waves/CU |
| 512 regs/SIMD, 16-granule, ≤256 AGPR, unified pool | LDS **160 KiB/CU, 64 banks**, 256 B/clk |
| HBM3E 288 GB @ 8 TB/s · 256 MiB Infinity Cache · **L2 per-XCD** | FP16 2.5 PF / FP8 5 PF / FP6·FP4 10 PF |
| FP16 ridge ≈ **312 FLOP/byte** | tuned GEMM sustains **~45–55% of peak** |
| FP8 is **OCP**, not FNUZ | **TF32 removed** |
| `global_load_lds` up to **128 b/lane** | `mfma_16x16` over `32x32`; ≥1024 WGs; 8-multiple tiles |

Full detail: `hardware/` (one card per subsystem).

## Validated reference points

- **aiter DB tune: +2.23% e2e** @ Qwen3.5-27B, sglang 0.5.11 / aiter, MI300X gfx942, 2026-06-08
  (1548.9 → 1583.5 tok/s, 5 non-overlapping reps, 246 engagement hits). The lookup key is a **10-tuple**
  (`gfx` first); a mismatched `bias` ⇒ 0% engagement. Tuning gate `err_ratio < 0.05`.
- **FP8 GEMM, MI355X / ROCm 7.1.0, M=N=K=8192**: HIP 8-wave ping-pong **3204 TFLOPS** (beats hipBLASLt's
  3130 with no assembly); HipKittens 4-wave interleave **3327 TFLOPS**. NVIDIA-style wave specialization
  caps out ~80% of peak on CDNA.
- **Loop form**: a data-dependent gather rewritten as static range + mask ran **~2× faster while
  visiting 2–4× more blocks** (gfx950, Triton 3.6).
