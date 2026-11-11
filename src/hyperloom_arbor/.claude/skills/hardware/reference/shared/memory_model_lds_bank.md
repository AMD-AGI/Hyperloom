---
title: LDS memory model & bank conflicts (CDNA cross-gen)
kind: hardware
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: []
regimes: [both]
updated: 2026-11-07
---

# LDS (Local Data Share) memory model & bank conflicts

> Numbers in brackets point to the Sources at the end.

## TL;DR
> LDS is the on-CU scratchpad used to stage GEMM/attention tiles. The killer rule: a **bank conflict**
> occurs when lanes access the **same bank** at **different addresses**; the accesses are serialized
> [3][5], so an LDS op takes from 2 up to 64 cycles depending on conflicts [3]. Dwords are placed in the
> banks serially [3], so bank = `(byte_address / 4) mod (#banks)`. Pad the leading dimension (or
> swizzle) so column access spreads across all banks.

## Concepts

### Per-generation LDS geometry
| Gen | Product | LDS/CU | Banks | Read BW | Bank index |
|---|---|---|---|---|---|
| CDNA1 gfx908 | MI100 | 64 KiB [1] | 32 × 4 B (512 entries each) [1] | 128 B/clk [5] | `(addr/4) mod 32` |
| CDNA2 gfx90a | MI250/210 | 64 KiB [2] | 32 × 4 B (512 entries each) [2] | 128 B/clk [5] | `(addr/4) mod 32` |
| CDNA3 gfx942 | MI300X/325X | 64 KiB [3] | 32 × 4 B (512 entries each) [3] | 128 B/clk [5][7] | `(addr/4) mod 32` |
| CDNA4 gfx950 | MI350X/355X | **160 KiB** [4] | **64 × 4 B** (640 entries each) [4] | **256 B/clk** [5][6] | `(addr/4) mod 64` |

CDNA4 changes are load-bearing: **2.5× capacity** raises tiling/occupancy headroom; **64 banks** means
the classic 32-stride conflict pattern shifts, and `ds_read`/`ds_write` swizzles tuned for 32 banks
must be re-checked for 64 (both derived from the table). (CDNA4 LDS = 64 banks × 640 × 4 B = 163840 B
per CU.)

### How LDS is issued
- Reads across a wavefront are dispatched over four cycles [3]; one wave can move up to 64 bytes per
  cycle (16 lanes per cycle) [5]. Lanes reading the **same address** get a **broadcast** with no
  conflict; **different addresses in the same bank** are serialized [5].
- Synchronization to LDS uses **`s_waitcnt lgkmcnt`** (count-based), not a fence [3] — wait only for
  the specific outstanding LDS/scalar ops you need, enabling deep prefetch overlap [7].

### The classic conflict: column access of a 32-wide tile (32-bank LDS)
A `float tile[32][32]` accessed as `tile[k][tid]` has row stride 32 = exactly 32 banks → every lane
in a column maps to the **same bank** → 32-way conflict (from the bank formula above).
```cpp
__shared__ float tile[32][32];   // BAD: stride 32 -> 32-way column conflict
__shared__ float tile[32][33];   // GOOD: pad +1 -> column lanes spread across all banks
float v = tile[k][threadIdx.x];  // conflict-free
```
For 16-bit data, two elements share a 4-byte bank word: re-derive the padding from the bank formula or
use a swizzled layout.

## The levers
1. **Pad the leading dimension** so the stride is coprime with the bank count (32 on CDNA1–3, 64 on
   CDNA4). The padding amount differs by generation — re-tune when porting. AMD's Gluon guidance uses
   padded shared layouts to remove bank conflicts [7].
2. **Use wide LDS ops** — `ds_read_b128` / `ds_write_b128` move 16 B/lane in one instruction [3], and
   AMD's guide recommends the `_b128` forms to cut the number of LDS instructions [7]. CDNA4 adds
   **read-with-transpose `ds` loads** (`ds_read_b64_tr_b16/b8/b4`, `ds_read_b96_tr_b6`) that transpose
   16/8/6/4-bit data on the way from LDS [4].
3. **Swizzle the staged A/B tile to match the MFMA lane mapping** so the feeding `ds_read` is
   conflict-free → see [matrix_core_mfma_smfmac.md](matrix_core_mfma_smfmac.md); the matrix
   instruction calculator gives each element's register and lane [8].
4. **Bypass VGPRs with direct global→LDS** (`buffer_load ... lds` / `global_load_lds_*`) [3][4]: on
   CDNA3 the sizes are 1, 2 or 4 bytes per lane (`ubyte`/`sbyte`, `ushort`/`sshort`, `dword`), so up to
   32 b/lane [3]; **CDNA4 adds `dwordx3`/`dwordx4`** — 1, 2, 4, 12 or 16 bytes per lane, up to 128 b/lane
   [4]. This skips the `ds_write`, the staging VGPRs, and the copy index math — see each gen's memory
   card and [wavefront_simd_vgpr_agpr.md](wavefront_simd_vgpr_agpr.md). AMD's guide reports
   `buffer_load_to_lds` saving ~100 VGPRs per wave and taking its gfx950 Gluon reference GEMM from 697
   to 1113 TFLOP/s [7].
5. **Double-buffer**: while MFMA consumes LDS buffer 0, async-load LDS buffer 1 [7].

### LDS occupancy limit
```
occ_lds (workgroups/CU) = floor(LDS_per_CU / LDS_bytes_per_workgroup)
# CDNA1-3: floor(65536 / L) ; CDNA4: floor(163840 / L)      [7]
```
LDS is allocated in blocks of 512 B (128 dwords) on CDNA3 [3] and 1280 B (320 dwords) on CDNA4 [4][9].
See per-gen `occupancy.md`.

## Pitfalls
- **Porting a 32-bank swizzle to CDNA4 unchanged.** 64 banks changes the conflict-free stride; re-derive
  the padding.
- **Scalar/uncoalesced LDS.** A strided `ds_read_b32` per element issues 4× the instructions of `b128`.
- **Forgetting LDS counts against occupancy**, not just VGPRs — an attention kernel can be LDS-limited
  rather than register-limited (see the worked examples in `cdna3_mi300/occupancy.md`).

## Verify
- The kernel's LDS bytes (`.group_segment_fixed_size` [9]) / `-Rpass-analysis=kernel-resource-usage`
  [10].
- `rocprof-compute` LDS panel: bank-conflict rate, LDS BW utilization, % stalls on LDS.
- Compare measured LDS read BW against the per-gen ceiling (128 B/clk CDNA1–3, 256 B/clk CDNA4 [5]).

## Sources
1. AMD, *"AMD Instinct MI100" Instruction Set Architecture* (CDNA1 ISA; LDS) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi100-cdna1-shader-instruction-set-architecture.pdf
2. AMD, *"AMD Instinct MI200" Instruction Set Architecture* (CDNA2 ISA; LDS) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf
3. AMD, *"AMD Instinct MI300" Instruction Set Architecture* (CDNA3 ISA; §2.2.1, §3.6.5, §9.1.9, §11) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
4. AMD, *AMD Instinct CDNA4 Instruction Set Architecture* (§2.2.1, §3.6.5, §9.1.9, §11.4) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
5. HIP documentation, *Hardware implementation* (local data share) — https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
6. AMD, *AMD CDNA 4 Architecture* white paper (LDS) — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
7. ROCm documentation, *AMD Instinct MI300 Series / MI350 Series workload optimization* — https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html
8. ROCm, *AMD Matrix Instruction Calculator* README — https://github.com/ROCm/amd_matrix_instruction_calculator
9. LLVM, *User Guide for AMDGPU Backend* (`GRANULATED_LDS_SIZE`; code object metadata) — https://llvm.org/docs/AMDGPUUsage.html
10. ROCm Blogs, *Register pressure in AMD CDNA2 GPUs* (`-Rpass-analysis=kernel-resource-usage`) — https://rocm.blogs.amd.com/software-tools-optimization/register-pressure/README.html
