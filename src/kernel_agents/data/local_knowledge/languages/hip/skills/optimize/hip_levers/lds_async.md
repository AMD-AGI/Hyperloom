---
title: HIP / C++ — LDS, banks, direct-to-LDS / async copy, barriers
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [both]
status: sota
updated: 2026-07-08
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
  - https://llvm.org/docs/AMDGPUUsage.html
  - https://github.com/iree-org/iree/issues/23765
---

# HIP — LDS, bank conflicts, direct-to-LDS & barriers

## 1. LDS capacity & banks
- **64 KB/CU (CDNA3)**, **160 KB/CU (CDNA4)**. Two blocks per CU on CDNA3 → ≤ 32 KB each. Budget LDS
  so the target occupancy survives.
- **32 banks × 4 B**, 128 B/clk (CDNA3; 256 B/clk CDNA4). A wavefront issues memory for **64 lanes**
  but there are only 32 banks → addresses are serviced in **two phases**; same-bank/different-row
  accesses across the phase **conflict** and serialize.
```cpp
__shared__ float tile[64][64];     // 16 KB — static LDS
__syncthreads();                    // s_barrier — workgroup barrier
// dynamic LDS (3rd <<<>>> arg):
extern __shared__ char smem[];
k<<<grid, block, (BM*BK + BK*BN)*sizeof(half), stream>>>();
```

## 2. Bank-conflict mitigations
- **Pad the inner dim**: `__shared__ float tile[64][64+1];` breaks the stride that maps columns onto
  the same bank for wave64 access.
- **XOR-swizzle** the column index for transpose-heavy / MFMA-staging patterns. This is the standard
  GEMM fix and is **required** for direct-to-LDS — removing the swizzle in one IREE study caused
  **201M bank conflicts, −28% TFLOPS**.
- **128-bit LDS access** (`float4` → `ds_read_b128`/`ds_write_b128`): fewer instructions, higher BW.
```cpp
float4 v = *reinterpret_cast<float4*>(&tile[r][c]);   // -> ds_read_b128
*reinterpret_cast<float4*>(&tile[r][c]) = v;          // -> ds_write_b128
```
LDS throughput per wave: 4-B accesses ~50% peak (8 cycles/64 lanes), 16-B ~80% (20 cycles). **Vectorize.**

## 3. Direct-to-LDS / async copy (skip register staging)
`global_load_lds` / `buffer_load ... lds` moves data **straight from global into LDS**, bypassing
VGPRs — removes the `ds_write` and the staging registers (frees VGPRs → higher occupancy, fewer
instructions in the loop).
```cpp
// each lane contributes one element; the 64-lane group fills a contiguous LDS chunk
__builtin_amdgcn_global_load_lds(
    thread_global_addr,   // per-lane global addr (may be scattered = gather)
    subgroup_lds_addr,    // MUST be coalesced across the subgroup
    /*size*/ 4,           // 1/2/4 bytes per lane (4 preferred); 64 lanes×4B = 256B/call
    /*offset*/ 0, /*aux*/ 0);
asm volatile("s_waitcnt vmcnt(0)");   // wait until it lands
__builtin_amdgcn_s_barrier();          // publish to all lanes
```
- Availability: `global_load_lds` is gated to the **gfx940 family (gfx942)**. The unified
  `llvm.amdgcn.load.to.lds` lowers correctly on **gfx950**; gfx942 uses `global_load_lds`. Scratch→LDS
  exists post-gfx942 only via inline asm.
- The **LDS destination must be coalesced**; global addresses may be scattered. Pair with the swizzle
  (above) — direct-to-LDS without swizzle is the classic bank-conflict regression.
- This is exactly what Triton's `knobs.amd.use_async_copy`, FlyDSL's `rocdl.raw_ptr_buffer_load_lds`,
  and CK's pipelined loaders emit to overlap the next tile's load with the current tile's MFMAs.

## 4. Barriers & wait counters (correctness for async memory)
CDNA memory ops are **asynchronous**; correctness needs explicit counters/barriers.
| Builtin / instr | Meaning |
|---|---|
| `__syncthreads()` → `s_barrier` | workgroup barrier (all waves in block) |
| `__builtin_amdgcn_wave_barrier()` | single-wave barrier |
| `s_waitcnt vmcnt(k)` | ≤ k vector-memory (global/buffer) ops outstanding |
| `s_waitcnt lgkmcnt(k)` | ≤ k LDS/GDS/const/message ops outstanding |
| `__builtin_amdgcn_s_waitcnt(n)` | wait on encoded counters |
| `s_wait_asynccnt` (gfx950) | wait on async-copy completion |

Pattern (direct-to-LDS → compute):
```cpp
__builtin_amdgcn_global_load_lds(g, l, 4, 0, 0);   // async load to LDS
asm volatile("s_waitcnt vmcnt(0)");                 // landed
__builtin_amdgcn_s_barrier();                        // all lanes see it
float4 a = *reinterpret_cast<float4*>(&lds[off]);    // ds_read_b128
asm volatile("s_waitcnt lgkmcnt(0)");                // LDS read complete before use
```
The compiler usually inserts `s_waitcnt`; hand-place them only in microkernels where you also control
scheduling. **`s_waitcnt vmcnt(0)` after *every* load = no overlap** — a common perf bug.

## 5. Double-buffering (the core of LDS pipelining)
Two LDS buffers overlap the next K-tile's load with the current tile's MFMA:
```cpp
__shared__ half As[2][TILE], Bs[2][TILE];
int buf = 0;
load_tile(0, buf); s_waitcnt vmcnt(0); s_barrier();
for (int k = 0; k < KTILES; ++k) {
    int nbuf = buf ^ 1;
    if (k+1 < KTILES) load_tile(k+1, nbuf);   // issue next (overlaps MFMA below)
    /* read As[buf]/Bs[buf] via ds_read_b128, MFMA */
    s_waitcnt vmcnt(0); s_barrier();
    buf = nbuf;
}
```
With `global_load_lds`, the staging VGPRs disappear entirely. Verify in ISA:
`.group_segment_fixed_size` = 2×tile bytes, `ds_*_b128`, no scratch. See [intrinsics.md](intrinsics.md)
for the MFMA inner loop and [patterns.md](patterns.md) for the full GEMM skeleton.

## Sources
- LDS banks / 64 KB-CU / occupancy: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- global_load_lds gating (gfx942 vs gfx950), swizzle requirement, 201M conflicts/−28%: https://github.com/iree-org/iree/issues/23765
- s_waitcnt vmcnt/lgkmcnt / async cnt, ds builtins: https://llvm.org/docs/AMDGPUUsage.html
- CDNA4 LDS 160 KB / 256 B/clk: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
