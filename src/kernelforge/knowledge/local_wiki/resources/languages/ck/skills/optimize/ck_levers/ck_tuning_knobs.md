---
title: CK — the knob space, ranked
kind: language
lever: ck_tuning_knobs
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
  - https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/hardware/gemm_optimization.html
  - https://github.com/ROCm/composable_kernel/issues/1727
---

# CK knob space

The config space is large; a handful of knobs dominate. **Tune block tile and pipeline first —
everything else is second-order.** Applies to both front-ends (`ck_frontend_classic.md`,
`ck_frontend_tile.md`).

## Route here when
You have picked a front-end and a working instance, and you are choosing what to sweep. If you have not
picked a front-end yet, go back — the front-end decision dominates every knob on this page.

## Ranked

| Knob | Values | Effect | Priority |
|---|---|---|:---:|
| `MPerBlock × NPerBlock` | 256×256, 256×128, 128×128, 128×64, 64×64 | block tile — more reuse vs fewer blocks (occupancy/tail trade-off) | ★★★★★ |
| `KPerBlock` | 32, 64, 128 | K-loop tile — better MFMA/load overlap, more LDS + VGPR | ★★★★ |
| `BlockGemmPipelineVersion` | v1 / v2 / **v3** / v4 / v5 | hot-loop schedule depth | ★★★★ |
| `BlockGemmPipelineScheduler` | **Intrawave** / Interwave | overlap strategy | ★★★★ |
| `MPerXDL × NPerXDL` | **16×16**, 32×32 | MFMA tile — 16×16 usually wins, see below | ★★★ |
| `MXdlPerWave × NXdlPerWave` | 4×4, 4×2, 2×2 | MFMA tiles **per wave** — drives VGPR and occupancy | ★★★ |
| `AK1 / BK1` | 8 (bf16), 16 (fp8) | global-load vector width — **must be ≥128 bit** | ★★★ |
| `KBatch` (split-K) | 1, 2, 4, 8 | atomic K split — fills CUs for **small-M decode** | ★★★ (decode) |
| `GemmSpecialization` | Default / MNKPadding / MNPadding | pad guards for non-divisible shapes | ★★ |
| CShuffle store vector | 8 (bf16) | coalesced C store width | ★★ |
| LDS swizzle | XOR (`make_xor_transform`) | kills bank conflicts, no extra LDS | ★★ |

## Why 16×16 over 32×32 — two independent reasons

Both point the same way:

1. **Register footprint** — 16×16 carries **4 C-registers/lane**; 32×32 carries **16**. That 4× comes
   out of the 512-register budget and costs occupancy.
2. **Power and clock** — the 32×32 op draws more power, so the part clocks lower and delivers **lower
   max-achievable FLOPs** (ROCm Max-Achievable-FLOPs Part 2).

Default 16×16; test 32×32 only for a specific large square shape.

## The 128-bit-per-load rule

`AK1` / `BK1` must make each lane's global load **≥ 128 bit**:

| dtype | `AK1` | Why |
|---|---:|---|
| bf16 / fp16 | **8** | 8 × 16 bit = 128 bit → `buffer_load_dwordx4` |
| fp8 | **16** | 16 × 8 bit = 128 bit |

**A sub-128-bit load silently halves effective HBM bandwidth.** Pointers must be aligned to the vector
width or `IsSupportedArgument` rejects the instance.

## Shape heuristics (gfx950)

| Regime | Configuration |
|---|---|
| **Prefill** (large M) | 256×256×64, `MPerXDL=NPerXDL=16` (test 32×32), WaveMap 4×4, **v3 Intrawave**, `AK1=BK1=8` |
| **Decode** (M = batch ≪ N,K) | small M tile (16/32 × 256), **split-K `KBatch≥2`** to occupy CUs, **Interwave** often wins, 16×16 MFMA. A 256×256 tile leaves most CUs idle at tiny M. |
| **fp8 weight-only linear** | `*_b_scale` fp8 instance, `AK1/BK1=16`, `bpreshuffle` the static weight into MFMA layout at load. `KPerBlock` can double (K-density doubles). |
| **MoE** | `DeviceGroupedGemm*` (+ `mx` / `b_scale` for low-precision experts) |

**Grid sizing:** aim for `ceil(M/MPerBlock)·ceil(N/NPerBlock) ≈ k·256` — gfx950 has **256 CUs**, not
304. A grid sized for MI300X leaves a quantization tail here.

**Pipeline depth:** gfx950's **160 KiB LDS** (2.5× a 64 KiB part) makes deeper pipelines affordable.
Where v3 used to be the practical ceiling, test v4.

## Verify

| Check | How |
|---|---|
| Instance ranking | offline `ckProfiler` sweep at the exact shape; record top TFLOP/s + GB/s |
| Cross-check | hipBLASLt solidx and the aiter tuned config at the same shape |
| No spills | disassemble — a bigger tile that spills regresses to a smaller-tile class |
| After a bump | re-measure and **append** the new number with a date; do not overwrite |

## Pitfalls
- **Defaulting to 32×32 MFMA** — see above; two reasons it loses.
- **A pinned "winning instance" is build-specific** — tile/pipeline IDs drift across CK/ROCm versions.
  Re-sweep after any bump.
- **Bigger block tile is not free** — VGPR/AGPR pressure → spills → throughput regresses. Verify in
  disassembly, not by reading the config string.
- **Split-K writes through atomics** — extra HBM traffic; only a win when it fills otherwise-idle CUs
  (decode).
- **Sub-128-bit `AK1`/`BK1`** — halves bandwidth silently.

Full list: `ck_traps.md`.

## Sources
- ROCm "Optimizing with Composable Kernel" (instance selection, profiler, knob guidance): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
- A Block GEMM on MI300 (tile/occupancy, LDS sizing): https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/hardware/gemm_optimization.html
- MI300X workload optimization (16×16 vs 32×32, 128-bit load, split-K): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- Issue #1727 (reference instance, v3/Intrawave): https://github.com/ROCm/composable_kernel/issues/1727
