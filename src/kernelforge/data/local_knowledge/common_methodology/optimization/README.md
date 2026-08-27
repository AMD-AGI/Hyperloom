---
title: optimization — index
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e5m2_fnuz, fp4_e2m1, int8]
regimes: [prefill, decode, training, both]
updated: 2026-08-23
---

# optimization — cross-operator performance techniques (MI300X / MI350X)

Hardware-grounded optimization levers that apply across operators. Each file is one technique,
tied to CDNA3 (gfx942 / MI300X) and CDNA4 (gfx950 / MI350X·MI355X) hardware facts and to AMD docs.
For per-operator application see `[[operators/dense_gemm/tuning.md]]` (the canonical style example) and the
operator cards; for raw hardware numbers see `[[hardware/cdna3_mi300/arch.md]]` and `[[hardware/cdna4_mi350/arch.md]]`.

## The hierarchy of levers (apply top-down)

1. **Pick the right algorithm / loop form / fusion** — `[[optimization/kernel_fusion_strategy.md]]`,
   `[[optimization/loop_form_and_pipelining.md]]`, `[[optimization/roofline_and_bottlenecks.md]]` (classify
   the kernel first; do not tune a bandwidth-bound kernel as if it were compute-bound). Loop *form* is
   decided here, before any knob below is reachable: a `while` whose trip count or addresses come from a
   tensor load silently forfeits `num_stages` pipelining and direct-to-LDS async copy, so the level-2 and
   level-4 levers have nothing to act on.
2. **Feed the matrix cores** — `[[optimization/mfma_scheduling.md]]`,
   `[[optimization/memory_pipelining.md]]`, `[[optimization/lds_and_bank_conflicts.md]]`.
3. **Get parallelism right** — `[[optimization/wave_and_grid_sizing.md]]`,
   `[[optimization/occupancy_and_registers.md]]`, `[[optimization/xcd_l2_locality.md]]`.
4. **Get the memory subsystem right** — `[[optimization/vectorization_and_coalescing.md]]`,
   `[[optimization/memory_pipelining.md]]`, `[[optimization/xcd_l2_locality.md]]`.
5. **Search the residual** — `[[optimization/cheap_sweeps.md]]` (one bench command per data point for a
   single constant; keep the knobs through the search), then `[[optimization/autotuning_methodology.md]]`
   (the only lever that engages the live sglang/vllm path is aiter's per-shape DB; see
   `[[operators/dense_gemm/backends/aiter.md]]`).
6. **Keep it correct** — `[[optimization/numerical_stability.md]]`.

Cutting across all six: `[[optimization/edit_surface_reach.md]]` — before pricing a lever as unreachable
("that would mean patching the framework, not this file"), work out what the files you may edit actually
reach. Rebinding an installed symbol, injecting device source through the framework's own hook, a module
constant another module's dispatch reads (an `os.environ` default does not put it off-limits), and a row
in a permitted data file are all edits *inside* the permitted set.

## Files in this section

| file | what it covers |
|---|---|
| `[[optimization/occupancy_and_registers.md]]` | 512 VGPR/EU, 16-granule alloc, AGPR pool, occupancy vs spilling, waves/EU, `num_warps`/`waves_per_eu` |
| `[[optimization/lds_and_bank_conflicts.md]]` | 64KB (CDNA3) / 160KB (CDNA4) LDS, 32 banks, padding, XOR swizzle, double-buffer |
| `[[optimization/mfma_scheduling.md]]` | 16×16 vs 32×32 MFMA, AGPR accumulators, issue cadence, latency hiding, `OPTIMIZE_EPILOGUE`, 512B Tagram |
| `[[optimization/memory_pipelining.md]]` | `global_load_lds` / async copy, software pipelining, `num_stages`, prefetch, ds_read/ds_write overlap, 128-bit GLOBAL_LOAD_LDS on CDNA4 |
| `[[optimization/loop_form_and_pipelining.md]]` | the precondition for the row above: shape-static trip count + addresses; the `while`/indirect-load signature; rewriting a data-dependent gather as static range + mask; why 2–4× more nominal work ran ~2× faster once |
| `[[optimization/vectorization_and_coalescing.md]]` | 128-bit `dwordx4` loads, alignment, grid-stride, coalesced access |
| `[[optimization/xcd_l2_locality.md]]` | 8-XCD MI300X, XCD-aware grid, L2 partition, ≥1024 workgroups, 8-multiple tiles, swizzled CTA order |
| `[[optimization/wave_and_grid_sizing.md]]` | wave64, workgroup size, `__launch_bounds__`, persistent kernels, CU=304 (MI300X) / 256 (MI350X) |
| `[[optimization/cheap_sweeps.md]]` | `FORGE_SWEEP_<NAME>` + `sweep_const:` echo, one bench command per data point, joint sweeps for coupled constants, keep the knobs (defaulted to the winners) through the search |
| `[[optimization/edit_surface_reach.md]]` | what an editable file reaches outside itself: installed-package rebind, `import_source`/`pragma_import_c`/extern-call injection, module constants read by another module's dispatch (`os.environ` defaults included), rows in permitted data/config files |
| `[[optimization/autotuning_methodology.md]]` | triton autotune, hipBLASLt bench, aiter `AITER_TUNE_GEMM`→gradlib `err_ratio<0.05`→`AITER_CONFIG_GEMM_BF16`, search-space pruning, config caching |
| `[[optimization/roofline_and_bottlenecks.md]]` | compute vs bandwidth bound, ~45% sustained-of-peak reality, arithmetic intensity, classifying a kernel |
| `[[optimization/kernel_fusion_strategy.md]]` | when to fuse (epilogue/prologue, norm+quant, rope+cache, comm+norm), fusion donors, when NOT to fuse |
| `[[optimization/numerical_stability.md]]` | fp32 accumulate, online softmax, Welford, fp8 FNUZ vs OCP 2× trap, scaling |

## On-box validated facts referenced throughout
- `mfma_16x16x16` > `mfma_32x32x8` for LLM N/K on MI300X (occupancy/register pressure).
- aiter GEMM lookup key is a **9-tuple** (m, n, k, dtype_in, dtype_out, bias, scaleAB, bpreshuffle, …);
  mismatched `bias` ⇒ 0% engagement. Tuning gate `err_ratio < 0.05`. Validated **+2.23% e2e** on
  Qwen3.5-27B / sglang 0.5.11 / aiter, 2026-06-08 (see `[[operators/dense_gemm/backends/aiter.md]]`).
- ≥1024 workgroups and 8-multiple tiles for XCD/L2 friendliness.
