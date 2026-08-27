---
title: L2 / XCD locality, tile swizzle & coalescing (CDNA cross-gen)
kind: hardware
gens: [gfx942, gfx950]
dtypes: []
regimes: [both]
updated: 2026-06-08
---

# L2 / XCD locality, tile swizzle & coalescing

## TL;DR
> On chiplet CDNA (MI300X/MI350X) **L2 is per-XCD**, not global. Two perf rules fall out:
> (1) make GEMM tile counts and grid dims **multiples of 8** so work distributes evenly across the
> 8 XCDs and reuse hits the same XCD's L2; (2) avoid the **512 B-stride "Tagram" hotspot** on the TN
> GEMM case. Always coalesce to **128 B-aligned** `dwordx4` loads.

## Concepts

### Why "multiple of 8"
- MI300X/MI325X = **8 XCDs**; MI350X/355X = **8 XCDs** too. The hardware scheduler round-robins
  workgroups across XCDs in blocks. If the **number of tiles** (or grid blocks) is a multiple of 8,
  each XCD gets an equal share and tiles that should reuse the same operand land on the **same XCD's
  L2** (4 MiB), maximizing L2 hit rate. A non-multiple leaves some XCDs idle in the tail and scatters
  reuse across XCDs (which misses to L3/Fabric).
- This is why the MI300X workload guide recommends **8-multiple tile dimensions** and **≥1024
  workgroups** (≥~3.4/CU on 304 CUs): fills the device and gives the scheduler tail slack.

### Grid/tile rules of thumb
| Rule | Value | Why |
|---|---|---|
| Workgroups per launch | **≥ 1024** | fill 304 CUs (MI300X) / 256 CUs (MI350X) + tail hiding |
| Tile M/N | **multiple of 8** | even XCD distribution, L2 reuse, no straggler XCD |
| Threads/block | 256–1024 (4–16 waves) | fits a CU; occupancy is per-SIMD |
| MFMA shape | 16×16 over 32×32 | see [matrix_core_mfma_smfmac.md](matrix_core_mfma_smfmac.md) |

### The 512 B-stride "Tagram" hotspot
On MI300, a GEMM whose leading-dimension byte stride is an exact **multiple of 512 B** (notably the
**TN** layout — A non-transposed, B transposed) can collide in the L2 **tag RAM (Tagram)**, creating a
perf cliff: many addresses map to the same L2 set, serializing accesses. Mitigations:
- **Pad the leading dimension** off a 512 B multiple (the GEMM analogue of LDS padding).
- Let the tuned library pick a swizzle / split-K that breaks the stride; hipBLASLt/CK solutions
  already encode this in their solution selection.

### `OPTIMIZE_EPILOGUE`
A Composable Kernel / Triton-on-ROCm knob: store the MFMA result **in its native MFMA register
layout** directly to global memory, **skipping the LDS reblock/transpose** in the epilogue. Usually
set **`OPTIMIZE_EPILOGUE=1`** — it removes an LDS round-trip and the associated VGPR/LDS pressure for
GEMMs whose output layout tolerates the MFMA-native order. Trade-off: the global store may be less
coalesced, so verify on the target shape.

### Coalescing (feeds everything above)
- Coalesce so a wave's 64 lanes touch a **contiguous 128 B-aligned** region; emit
  `global_load_dwordx4` (16 B/lane) so each wave fills full 128 B cache lines with fewer instructions.
- `buffer_load`/`buffer_store` with a descriptor (V#) gives bounds-checked OOB handling → cheaper
  guards in tiled GEMM than branchy `global_load`.

## The levers
1. **Make tile/grid counts multiples of 8** for even XCD spread + L2 reuse.
2. **Launch ≥1024 workgroups.**
3. **Break 512 B strides** (pad LD) to dodge the Tagram hotspot on TN GEMM.
4. **`OPTIMIZE_EPILOGUE=1`** to skip the epilogue LDS reblock (verify coalescing on your shape).
5. **`global_load_dwordx4`, 128 B-aligned**; use `buffer_*` for cheap bounds checks.
6. **Keep an operand's reuse on one XCD** (tile schedule) so it stays in that XCD's 4 MiB L2.
7. **Consider CPX/NPS partitioning** to make L2/HBM strictly XCD-local for many-small-kernel workloads
   → see each gen's `xcd_chiplet.md` / `arch.md`.

## Pitfalls
- **Non-8-multiple grids** leave straggler XCDs and scatter L2 reuse — a silent ~10–15% loss.
- **TN GEMM at a 512 B stride** hits the Tagram cliff; symptom is anomalously low L2 hit rate at
  specific N/K.
- **Assuming L2 is global** — cross-XCD reuse does **not** hit the 4 MiB L2; it falls to the 256 MiB L3.

## Verify
- `rocprof-compute`: L2 hit rate per shape; look for cliffs at 512 B-stride N/K; XCD load balance.
- A/B a GEMM with tile dims rounded to a multiple of 8 vs not, at fixed (M,N,K).
- Toggle `OPTIMIZE_EPILOGUE` and compare TFLOP/s + store-coalescing counters.
