---
title: LDS memory model & bank conflicts (CDNA cross-gen)
kind: hardware
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: []
regimes: [both]
updated: 2026-06-08
---

# LDS (Local Data Share) memory model & bank conflicts

## TL;DR
> LDS is the on-CU scratchpad used to stage GEMM/attention tiles. The killer rule:
> a **bank conflict** occurs when ≥2 lanes in the same half-wave hit the **same bank** at
> **different addresses**, serializing into N cycles. Bank = `(byte_address / 4) mod (#banks)`.
> Pad the leading dimension (or swizzle) so column access spreads across all banks.

## Concepts

### Per-generation LDS geometry
| Gen | Product | LDS/CU | Banks | Read BW | Bank index |
|---|---|---|---|---|---|
| CDNA1 gfx908 | MI100 | 64 KiB | 32 × 4 B | up to 128 B/clk | `(addr/4) mod 32` |
| CDNA2 gfx90a | MI250/210 | 64 KiB | 32 × 4 B | up to 128 B/clk | `(addr/4) mod 32` |
| CDNA3 gfx942 | MI300X/325X | 64 KiB | 32 × 4 B | up to 128 B/clk | `(addr/4) mod 32` |
| CDNA4 gfx950 | MI350X/355X | **160 KiB** | **64 × 4 B** (640 entries each) | **256 B/clk** | `(addr/4) mod 64` |

CDNA4 changes are load-bearing: **2.5× capacity** raises tiling/occupancy headroom; **64 banks**
means the classic 32-stride conflict pattern shifts, and `ds_read`/`ds_write` swizzles tuned for 32
banks must be re-checked for 64. (CDNA4 LDS = 64 banks × 640 × 4 B = 163840 B per CU.)

### How LDS is issued
- A wave issues LDS in **half-waves of 32 lanes**. Within a half-wave, lanes hitting the **same
  address** in the same bank is a **broadcast** (free); different addresses in the same bank is an
  N-way conflict. (On CDNA4's 64-bank LDS the access still resolves over the 64-bank structure.)
- Synchronization to LDS uses **`s_waitcnt lgkmcnt`** (count-based), not a fence — wait only for the
  specific outstanding LDS/scalar ops you need, enabling deep prefetch overlap.

### The classic conflict: column access of a 32-wide tile (32-bank LDS)
A `float tile[32][32]` accessed as `tile[k][tid]` has row stride 32 = exactly 32 banks → every lane
in a column maps to the **same bank** → 32-way conflict.
```cpp
__shared__ float tile[32][32];   // BAD: stride 32 -> 32-way column conflict
__shared__ float tile[32][33];   // GOOD: pad +1 -> column lanes spread across all banks
float v = tile[k][threadIdx.x];  // conflict-free
```
For 16-bit data pad so the bank pattern breaks (e.g. `[32][32+4]` for half) or use a swizzled layout.

## The levers
1. **Pad the leading dimension** so the stride is coprime with the bank count (32 on CDNA1–3, 64 on
   CDNA4). The padding amount differs by generation — re-tune when porting.
2. **Use wide LDS ops** — `ds_read_b128` / `ds_write_b128` move 16 B/lane in one instruction, cutting
   issue pressure and conflict opportunities. CDNA4 adds **read-with-transpose `ds` loads** so the
   GEMM B-operand can be transposed for free on the LDS read.
3. **Swizzle the staged A/B tile to match the MFMA lane mapping** so the feeding `ds_read` is
   conflict-free → see [matrix_core_mfma_smfmac.md](matrix_core_mfma_smfmac.md). CK and Triton
   generate these automatically; hand-written kernels mirror the calculator's register map.
4. **Bypass VGPRs with direct global→LDS** (`buffer_load ... lds` / `global_load_lds`): on CDNA3 up
   to 32 b/lane, **on CDNA4 up to 128 b/lane** (DWORD counts 1/2/4/12/16). This eliminates `ds_write`,
   staging VGPRs, and the copy index math — see [memory_hierarchy of each gen] and
   [wavefront_simd_vgpr_agpr.md](wavefront_simd_vgpr_agpr.md). Measured: switching a reference GEMM to
   `buffer_load_to_lds` freed ~100 VGPR/wave and moved it 697 → 1113 TFLOP/s (ROCm Triton guide).
5. **Double-buffer**: while MFMA consumes LDS buffer 0, async-load LDS buffer 1.

### LDS occupancy limit
```
occ_lds (workgroups/CU) = floor(LDS_per_CU / LDS_bytes_per_workgroup)
# CDNA1-3: floor(65536 / L) ; CDNA4: floor(163840 / L)
```
CDNA4 allocates LDS in **320-DWORD blocks** (vs 128-DWORD at 64 KiB). See per-gen `occupancy.md`.

## Pitfalls
- **Porting a 32-bank swizzle to CDNA4 unchanged.** 64 banks changes the conflict-free stride; re-run
  the calculator and re-pad.
- **Scalar/uncoalesced LDS.** A strided `ds_read_b32` per element wastes 4× the issue slots vs `b128`.
- **Forgetting LDS counts against occupancy**, not just VGPRs — an attention kernel is often
  LDS-limited, not register-limited.

## Verify
- ISA dump `.lds_size` / `-Rpass-analysis=kernel-resource-usage` for per-kernel LDS bytes.
- `rocprof-compute` (Omniperf) LDS panel: bank-conflict rate, LDS BW utilization, % stalls on LDS.
- Compare measured LDS read BW against the per-gen ceiling (128 B/clk CDNA1–3, 256 B/clk CDNA4).
