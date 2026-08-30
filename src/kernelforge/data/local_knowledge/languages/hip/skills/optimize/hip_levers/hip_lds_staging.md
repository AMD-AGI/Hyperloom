---
title: HIP — LDS staging, 64-bank conflicts, direct-to-LDS, barriers
kind: language
lever: hip_lds_staging
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
  - https://llvm.org/docs/AMDGPUUsage.html
  - https://github.com/iree-org/iree/issues/23765
---

# LDS staging

How operands get from HBM into the matrix core, and the three things that go wrong on the way: bank
conflicts, register staging you did not need, and missing wait counters.

## Route here when
- `ds_*` stall cycles are high, or the bank-conflict counter is non-zero.
- You are staging tiles through VGPRs and want to stop.
- Results are non-deterministic across runs (a missing barrier or waitcnt).

## Geometry — gfx950

| Property | Value | vs gfx942 |
|---|---|---|
| Capacity | **160 KiB/CU** | 64 KiB |
| Banks | **64 × 4 B** | 32 |
| Bank index | **`(byte_addr / 4) mod 64`** | `mod 32` |
| Read bandwidth | **256 B/clk** | 128 B/clk |
| Allocation granule | **320 DWORD** | 128 DWORD |
| Direct global→LDS | **1/2/4/12/16 DWORD** (≤128 b/lane) | 1/2/4 (32 b/lane) |
| Read-with-transpose `ds` | **yes** | no |

> **The 32 → 64 bank change invalidates every inherited swizzle.** A padding or XOR pattern tuned for
> 32 banks does *not* guarantee conflict-freedom on 64. Re-derive it.

A wavefront issues memory for **64 lanes** in **half-waves of 32**. Within a half-wave, lanes hitting
the same bank at *different* addresses serialize; same *address* is a free broadcast.

```cpp
__shared__ float tile[64][64];      // 16 KiB — static LDS
__syncthreads();                    // -> s_barrier (workgroup barrier)

extern __shared__ char smem[];      // dynamic LDS (3rd <<<>>> argument)
k<<<grid, block, (BM*BK + BK*BN)*sizeof(__bf16), stream>>>();
```

## Fixing conflicts

**Pad the inner dimension** — `__shared__ float tile[64][64+1];` breaks the stride that maps a column
onto one bank. Choose the pad so `(byte_stride/4) mod 64 != 0`, **and keep 16-byte alignment** so
`ds_read_b128` still fires. A pad that fixes conflicts and breaks vectorization is a net loss.

**XOR-swizzle the column index** for transpose-heavy and MFMA-staging patterns. This is the standard
GEMM fix, costs no extra LDS, and is **required** when using direct-to-LDS. One IREE study measured
removing it: **201 M bank conflicts, −28% TFLOPS**.

**Use 128-bit LDS access.** Throughput per wave: 4-byte accesses reach ~50% of peak (8 cycles/64
lanes); 16-byte reach ~80% (20 cycles). **Vectorize.**

```cpp
float4 v = *reinterpret_cast<float4*>(&tile[r][c]);   // -> ds_read_b128
*reinterpret_cast<float4*>(&tile[r][c]) = v;          // -> ds_write_b128
```

**Use read-with-transpose `ds` loads (gfx950)** to feed the MFMA B operand without a transpose pass.

## Direct-to-LDS — skip the register staging

`global_load_lds` / `buffer_load ... lds` moves data **straight from global into LDS**, bypassing
VGPRs. That removes the `ds_write` **and** the staging registers — freeing VGPRs, raising occupancy,
and cutting instructions in the loop.

```cpp
// each lane contributes; the 64-lane group fills a contiguous LDS chunk
__builtin_amdgcn_global_load_lds(
    thread_global_addr,   // per-lane global addr (may be scattered = gather)
    subgroup_lds_addr,    // MUST be coalesced across the subgroup
    /*size*/ 16,          // gfx950: up to 16 B/lane — 64 lanes x 16 B = 1024 B per call
    /*offset*/ 0, /*aux*/ 0);
asm volatile("s_waitcnt vmcnt(0)");   // wait until it lands
__builtin_amdgcn_s_barrier();          // publish to all lanes
```

Two rules that bite:
- **The LDS destination must be coalesced**; the global addresses may be scattered.
- **Pair it with the swizzle.** Direct-to-LDS *without* a swizzle is the classic bank-conflict
  regression (iree #23765).

On availability: the unified `llvm.amdgcn.load.to.lds` lowers correctly on **gfx950**; gfx942 used
`global_load_lds` gated to the gfx940 family. Scratch→LDS exists only via inline asm.

This is the same mechanism behind Triton's `knobs.amd.use_async_copy`, FlyDSL's
`rocdl.raw_ptr_buffer_load_lds`, and CK's pipelined loaders.

## Barriers and wait counters

CDNA memory is **asynchronous**; correctness needs explicit counters and barriers.

| Builtin / instruction | Meaning |
|---|---|
| `__syncthreads()` → `s_barrier` | workgroup barrier (all waves in the block) |
| `__builtin_amdgcn_wave_barrier()` | single-wave barrier |
| `s_waitcnt vmcnt(k)` | **≤ k** vector-memory ops outstanding |
| `s_waitcnt lgkmcnt(k)` | **≤ k** LDS/GDS/const/message ops outstanding |
| `__builtin_amdgcn_s_waitcnt(n)` | wait on encoded counters |
| `s_wait_asynccnt` (gfx950) | wait on async-copy completion |

**`s_waitcnt <counter>(N)` means "wait until ≤ N outstanding", not "wait N instructions."**

```cpp
__builtin_amdgcn_global_load_lds(g, l, 16, 0, 0);   // async load to LDS
asm volatile("s_waitcnt vmcnt(0)");                  // landed
__builtin_amdgcn_s_barrier();                        // all lanes see it
float4 a = *reinterpret_cast<float4*>(&lds[off]);    // ds_read_b128
asm volatile("s_waitcnt lgkmcnt(0)");                // LDS read complete before use
```

The compiler usually inserts `s_waitcnt` for you; hand-place them only in microkernels where you also
control scheduling. **`s_waitcnt vmcnt(0)` after *every* load means no overlap at all** — a common and
easily-missed perf bug.

## Double-buffering — the core of LDS pipelining

```cpp
__shared__ __bf16 As[2][TILE], Bs[2][TILE];
int buf = 0;
load_tile(0, buf); s_waitcnt vmcnt(0); s_barrier();
for (int k = 0; k < KTILES; ++k) {
    int nbuf = buf ^ 1;
    if (k+1 < KTILES) load_tile(k+1, nbuf);   // issue next — overlaps the MFMA below
    /* read As[buf]/Bs[buf] via ds_read_b128, run MFMA */
    s_waitcnt vmcnt(0); s_barrier();
    buf = nbuf;
}
```

With `global_load_lds` the staging VGPRs disappear entirely.

**gfx950's 160 KiB budget affords 3–4 stages** where a 64 KiB part topped out at 2. If you inherited a
2-stage pipeline, that is now a tuning opportunity, not a ceiling.

## Verify

| Check | Pass |
|---|---|
| `.group_segment_fixed_size` | = stages × tile bytes, after 320-DWORD rounding |
| LDS access width | `ds_read_b128` / `ds_write_b128`, not `b32` |
| Direct-to-LDS emitted | the 12/16-DWORD form, not 1/2/4 |
| Bank conflicts | `rocprof-compute` LDS panel, **over 64 banks** — near zero |
| `scratch_` | none |
| Waitcnt | overlapped, not `vmcnt(0)` after every load |

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Ported kernel slow, "worked before" | 32-bank swizzle on 64 banks | re-derive |
| Fixed conflicts, still slow | pad broke 16-byte alignment → scalar `ds_read` | re-pad preserving alignment |
| Direct-to-LDS made it *worse* | no swizzle — 201 M conflicts, −28% in one study | swizzle is **required** with DGL |
| No overlap despite double-buffering | `s_waitcnt vmcnt(0)` after every load | relax the counts |
| Non-deterministic results | missing `s_barrier` between stages | audit barrier discipline |
| Occupancy dropped after adding stages | LDS × stages over budget | recompute `floor(163840/L)` |

## Sources
- LDS banks / capacity / occupancy: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
- `global_load_lds` gating, swizzle requirement, 201 M conflicts / −28%: https://github.com/iree-org/iree/issues/23765
- `s_waitcnt` vmcnt/lgkmcnt/asynccnt, `ds` builtins: https://llvm.org/docs/AMDGPUUsage.html
- CDNA4 LDS 160 KiB / 256 B/clk: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf
