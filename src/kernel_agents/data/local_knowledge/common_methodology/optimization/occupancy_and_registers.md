---
title: occupancy and registers (VGPR/AGPR, waves/EU)
kind: technique
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode, training, both]
updated: 2026-06-05
---

# occupancy and registers

## TL;DR
On CDNA3/CDNA4 each SIMD (Execution Unit, EU) has **512 × 32-bit registers**, allocated in
**16-register granules**, split between **architected VGPRs (≤256)** and **accumulation AGPRs (≤256)**.
Occupancy in waves/EU = `floor(512 / round_up(VGPR_used, 16))` and is capped at **8 waves/EU** (32/CU)
by the instruction-buffer slots. The tuning game is: keep enough waves resident to hide MFMA + memory
latency, but **do not push so hard that the compiler spills** (latency cliff). For dense GEMM the right
answer is usually *fewer waves, more registers* (large MFMA tiles); for memory-bound elementwise/norm
it is *more waves*. See `[[hardware/shared/wavefront_simd_vgpr_agpr.md]]` and
`[[hardware/cdna3_mi300/occupancy.md]]`.

## Concepts (the hardware)
- **Register file**: 512 VGPRs/EU, 32-bit each. A single wave can use up to 512 total =
  256 architected VGPRs + 256 AGPRs. When a wave uses <512 total, the VGPR/AGPR split is flexible.
- **Allocation granularity**: VGPRs are reserved per wave in units of **16** (the tuning guide's
  occupancy unit; the ISA states groups of 8 Dwords). So 170 used ⇒ 176 reserved.
- **AGPRs**: accumulation registers, the *only* destinations/sources for MFMA accumulators
  (`v_mfma_*`). They extend usable register space beyond the 256 architected VGPRs and can also be
  loaded directly from memory; the compiler also uses `v_accvgpr_{read,write}` for cheap spill/fill.
  See `[[optimization/mfma_scheduling.md]]`.
- **Wave slots**: 8 wavefront slots per SIMD ⇒ max **8 waves/EU, 32 waves/CU**. Occupancy never
  exceeds this even with tiny register footprints.

## The occupancy formula (worked)
`waves_per_eu = min( 8 , floor(512 / round_up(VGPR_per_thread, 16)) )`

| VGPR/thread (reserved) | waves/EU |
|---|---|
| ≤ 64 | 8 (slot-capped) |
| 96 | 5 |
| 128 | 4 |
| 176 (e.g. 170 used) | **2** (176×3 > 512) |
| 256 | 2 |
| 512 (256 VGPR + 256 AGPR) | 1 |

AGPRs come out of the *same* 512 budget, so a GEMM with a large MFMA accumulator tile in AGPRs is
inherently low-occupancy — and that is fine, because MFMA latency is hidden by the deep pipeline,
not by many waves (see `[[optimization/mfma_scheduling.md]]`).

## The levers
- **`waves_per_eu=N` (triton / `__attribute__((amdgpu_waves_per_eu(N)))`)**: a *hint*; the LLVM
  backend tries to cut VGPR usage so N waves fit. Raise it to force more parallelism on
  latency-bound kernels; it can backfire by inducing spills.
- **`num_warps` (triton)**: warps = wave64 wavefronts per workgroup. More warps = bigger workgroup,
  more LDS/registers consumed per block, fewer blocks/CU. Typical GEMM: 4–8 warps. See
  `[[optimization/wave_and_grid_sizing.md]]`.
- **`__launch_bounds__(maxThreads, minWavesPerEU)` (HIP)**: hard-caps VGPRs the compiler may use so
  the requested occupancy is guaranteed; under-setting it forces spills.
- **MFMA tile size**: `32x32` instructions hold a larger accumulator in AGPRs than `16x16`, raising
  register pressure and dropping occupancy — a key reason `16x16` often wins on MI300X
  (`[[operators/dense_gemm/tuning.md]]`).
- **Reduce live state**: recompute cheap values instead of holding them; shrink `BLOCK_K` accumulation
  scope; move loop-invariants to scalar (SGPR) regs.

## Occupancy vs spilling — the cliff
Spills convert register accesses into **scratch (global) memory** traffic; on a compute-bound GEMM a
single spilled inner-loop value can cost more than the occupancy it buys. Diagnose with the assembler
report and the profiler:
- Look for `scratch` usage / `buffer_store`/`buffer_load` to scratch in the ISA dump
  (`[[languages/triton_amd/isa_verify.md]]`).
- `rocprof` / Omniperf counters: `VALUBusy`, wavefront occupancy, `SQ_WAIT_INST_LDS`, scratch traffic
  (`[[profiling/]]`, `[[hardware/cdna3_mi300/occupancy.md]]`).
- Rule of thumb: prefer **2 waves/EU with no spills** over 3 waves/EU that spill, for GEMM-class kernels.

## Pitfalls
- Treating "more occupancy = faster" as universal. MFMA-bound kernels run great at 1–2 waves/EU.
- Forgetting AGPRs count against the 512 budget — a fat accumulator silently caps occupancy.
- Setting `waves_per_eu` high without checking the ISA dump for spills.
- Assuming CUDA "blocks/SM" math; CDNA granularity is **16 VGPR**, slots are **8/SIMD**, wave is **64**.

## Verify
- ISA/asm: confirm VGPR/AGPR counts and zero scratch (`amdgpu-arch` dump; triton `TRITON_CACHE`/`AMDGCN`).
- Profiler: occupancy and `VALUBusy` from Omniperf; compare across `waves_per_eu` settings.
- A/B: sweep `waves_per_eu ∈ {1,2,3,4}` and `num_warps ∈ {4,8}`, keep the lowest latency with no spill.
