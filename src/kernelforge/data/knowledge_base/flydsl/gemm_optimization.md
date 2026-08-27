# GEMM Optimization Playbook (FlyDSL on CDNA)

> Distilled from `.claude/skills/gemm-optimization/SKILL.md` in the FlyDSL repo
> and the production `kernels/preshuffle_gemm.py`. Reference: gfx942 (MI300X)
> and gfx950 (MI350/MI355X). For RDNA / gfx1250 patterns see
> `kernels/wmma_gemm_gfx1250.py` and `kernels/gemm_fp8fp4_gfx1250.py`.

## 1. Tiling

Output `C[M,N] = A[M,K] @ B[K,N]^T`. Block-level tile is `(tile_m, tile_n)`
in C; K is reduced in chunks of `tile_k`.

```
Grid:   block_x → M tiles    block_y → N tiles
Block:  256 threads = 4 waves × 64 lanes  (wave64)
  wave_id    = tid // 64      → N partition (4 ways)
  lane_id    = tid %  64      → M+N within wave
  lane_div16 = lane_id // 16  → M direction (4 groups of 16)
  lane_mod16 = lane_id %  16  → N direction within MFMA
```

Derived per-tile:
```
m_repeat   = tile_m // 16
n_per_wave = tile_n // 4
num_acc_n  = n_per_wave // 16
k_unroll   = tile_k_bytes // a_elem_vec_pack // 64    # K64 micro-steps
```

### Tile size constraints
- `tile_m` multiple of 16 (MFMA M dim)
- `tile_n` multiple of 64 (4 waves × 16 N)
- `tile_k * elem_bytes` multiple of 64 (K64-byte micro-step)
- LDS budget: `2 × tile_m × tile_k × elem_bytes` ≤ 64 KB (gfx942) / 160 KB (gfx950)
- B preshuffle requires `tile_k` divides K evenly

### Recommended tile configs
| Scenario | tile_m | tile_n | tile_k | Dtype | Note |
|---|---|---|---|---|---|
| Small batch (M ≤ 32) | 16 | 64-128 | 256-512 | FP8/INT8 | Mem-bound; big tile_k for B reuse |
| Medium | 64 | 256 | 128 | any | balanced |
| Large (M ≥ 4096) | 128 | 256 | 128 | FP8/INT8 | compute-dense; needs async copy |
| FP4 (gfx950) | 32–64 | 128–256 | 256 | FP4 | MFMA_SCALE |

MFMA count per tile:
```
MFMA_per_tile = k_unroll × m_repeat × num_acc_n × 2   # 2× K32 per K64 step

Example: 64×256×128 FP8:
  k_unroll = 2, m_repeat = 4, num_acc_n = 4
  MFMA_per_tile = 2 × 4 × 4 × 2 = 64
```

## 2. LDS Ping-Pong Double Buffer (`lds_stage=2`)

Two independent `SmemAllocator` regions for A; alternate between compute on
PONG / load to PING and vice versa.

```python
allocator_pong = SmemAllocator(None, arch=arch, global_sym_name="smem0")
allocator_ping = SmemAllocator(None, arch=arch, global_sym_name="smem1")
lds_a_pong = allocator_pong.allocate_array(T.i8, tile_m * tile_k)
lds_a_ping = allocator_ping.allocate_array(T.i8, tile_m * tile_k)
```

Main-loop body processes 2 K-tiles per iteration. Each half:
1. Store A(next) → other LDS
2. Prefetch B(next) → VGPR
3. MFMA compute on current LDS + B VGPR
4. `hot_loop_scheduler()` — sched_* hints to interleave VMEM / MFMA / DS
5. `rocdl.s_waitcnt(num_b_loads)` + `gpu.barrier()`
6. Prefetch A0 pack from the other LDS into VGPR (overlaps the next VMEM)

LDS size budget:
```
2-stage A:   2 × tile_m × tile_k × elem_bytes
CShuffle epilogue (optional): tile_m × tile_n × 2 bytes
```

## 3. LDS XOR16 Bank-Conflict Swizzle

A row stored row-major with stride `tile_k` causes bank conflicts when 4-wave
reads hit the same bank for different addresses.

### The trick (`kernels/mfma_preshuffle_pipeline.py:swizzle_xor16`)
```python
def swizzle_xor16(row, col, k_blocks16):
    """XOR-with-row swizzle at 16-byte granularity."""
    rem = row % k_blocks16
    return col ^ (rem * 16)
```

Apply to BOTH write (G→LDS store) and read (LDS→VGPR load) paths — they MUST
match.

### Bank-conflict math
- gfx942 (32 banks): stride multiple of 128 B causes full conflict
- gfx950 (64 banks): stride 256 B causes full; 128 B causes 2-way; XOR16
  distributes evenly

Trade-off vs padding: swizzle is zero-overhead but needs correct mask.
Padding adds bytes but is conceptually simpler. See [lds_optimization.md](lds_optimization.md).

## 4. Data Prefetch Pipeline

### A matrix: Global → LDS
- **Sync** (default): `buffer_load_dwordx4 → VGPR → ds_write → LDS`
- **Async** (`use_async_copy=True`): `raw_ptr_buffer_load_lds` (direct DMA,
  skips VGPR). gfx942 = 4B/DMA, gfx950 = 16B/DMA.

Async is preferred for `tile_m ≥ 128` where it saves ~32 VGPRs of A-tile
buffers (avoiding occupancy regression).

### B matrix: Global → VGPR (Preshuffle)
B is pre-transposed/reshuffled to layout
`(N/16, K/64, 4, 16, kpack_bytes)`. The `4` dim maps to 4 dwords/lane (dwordx4
load); the `16` dim maps to 16 lanes per MFMA. Loaded directly to VGPR — no
VALU shuffle needed.

```python
b_tile = prefetch_b_tile(base_k)   # buffer_load_dwordx4 → VGPR
# Structure: k_unroll × [(packs0[num_acc_n], packs1[num_acc_n])]   (i64 each)
```

### A0 prefetch (cross-tile)
After `gpu.barrier()` (LDS valid), immediately load first A pack into VGPR —
overlaps the next iteration's VMEM:

```python
a0_prefetch = lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_buffer)
```

Hides the first `ds_read` latency (~20–40 cycles).

## 5. `hot_loop_scheduler()` — Instruction Hints

Schedule primitives (`flydsl.expr.rocdl`):
| Hint | Maps to |
|---|---|
| `sched_barrier(0)` | full fence (no reorder) |
| `sched_mfma(N)` | allow N `v_mfma_*` |
| `sched_dsrd(N)` | allow N `ds_read_*` |
| `sched_dswr(N)` | allow N `ds_write_*` |
| `sched_vmem(N)` | allow N `buffer_load_*` / `global_load_*` |

Standard sync-copy schedule (`hot_loop_scheduler` in `preshuffle_gemm.py`):
```python
def hot_loop_scheduler():
    mfma_group = num_acc_n
    mfma_total = (k_unroll * 2) * m_repeat * mfma_group
    mfma_per_iter = 2 * mfma_group
    sche_iters = mfma_total // mfma_per_iter

    # Prologue: pre-load 2 LDS reads, then a couple MFMAs
    rocdl.sched_dsrd(2)
    rocdl.sched_mfma(1)
    rocdl.sched_mfma(1)

    # Main: interleave VMEM + MFMA + ds_read + MFMA, tail-load LDS writes
    dswr_tail = num_a_loads
    dswr_start = max(sche_iters - dswr_tail - 2, 0)
    for sche_i in range_constexpr(sche_iters):
        rocdl.sched_vmem(1)
        rocdl.sched_mfma(mfma_group)
        rocdl.sched_dsrd(1)
        rocdl.sched_mfma(mfma_group)
        if sche_i >= dswr_start - 1:
            rocdl.sched_dswr(1)

    rocdl.sched_barrier(0)
```

For async copy on gfx950 the scheduler uses `_build_scheduler()` to evenly
distribute `sched_dsrd` and `sched_vmem` across all MFMAs.

## 6. MFMA Inner Loop (FP8 K64 Micro-step)

```python
for ku in range_constexpr(k_unroll):
    b_packs0, b_packs1 = b_tile_in[ku]
    col_base = col_offset_base_bytes + ku * 64
    for mi in range_constexpr(m_repeat):
        curr_row_a_lds = row_a_lds + (mi * 16)
        a0, a1 = lds_load_packs_k64(...)         # 2× i64 from LDS
        for ni in range_constexpr(num_acc_n):
            acc[mi * num_acc_n + ni] = mfma_k64_bytes(
                acc[mi * num_acc_n + ni], a0, a1,
                b_packs0[ni], b_packs1[ni]
            )
```

MFMA selection:
| Dtype | K per MFMA | Instruction | Acc |
|---|---|---|---|
| FP8 | 32 | `mfma_f32_16x16x32_fp8_fp8` | f32×4 |
| INT8 | 32 | `mfma_i32_16x16x32_i8` | i32×4 |
| BF16 | 16 | `mfma_f32_16x16x16bf16_1k` | f32×4 |
| FP16 | 16 | `mfma_f32_16x16x16f16` | f32×4 |
| FP4 (gfx950) | 128 | `mfma_scale_f32_16x16x128_f8f6f4` | f32×4 |

On gfx950 the K dimension doubled for f16/bf16/i8: prefer
`mfma_f32_{16,32}×{16,32}×{32,16}_{f16,bf16}` instead of the gfx942
counterparts to halve MFMA count.

## 7. Epilogue Strategies

### Direct store (default)
```python
for mi in range_constexpr(m_repeat):
    for ii in range(4):     # 4 rows per lane_div_16 group
        row = bx_m + mi * 16 + lane_div_16 * 4 + ii
        for ni in range_constexpr(num_acc_n):
            col = by_n + wave_id * n_per_wave + ni * 16 + lane_mod_16
            val = acc[mi * num_acc_n + ni][ii] * scale_a * scale_b
            buffer_store(truncate(val, out_dtype), c_rsrc, row * N + col)
```
Pros: no extra LDS. Cons: non-coalesced for some tile sizes.

### CShuffle epilogue (`kernels/mfma_epilogues.py:c_shuffle_epilog`)
1. Write acc rows to `lds_out` row-major
2. `gpu.barrier()`
3. Shuffle-read: threads remap to (MLane=8, NLane=32)
4. `buffer_store_dwordx2` (4-element vectorized writes)

```python
e_vec = 4 if (tile_n % 128 == 0) else 2
m_reps_shuffle = tile_m // 8
n_reps_shuffle = tile_n // (32 * e_vec)
```

Use when `tile_n ≥ 128` and direct store shows non-coalesced bandwidth.

## 8. Register Budget (gfx942)

```
Accumulators (accum_vgpr file):  m_repeat × num_acc_n × 4
B tile (arch_vgpr):              k_unroll × 2 × num_acc_n × 2  (i64 packs)
A prefetch:                      2 × 2
A tile regs (sync copy only):    num_a_loads × 4  (dwordx4)
Address VGPRs:                   ~10–20
```

Example 64×256×128 FP8 sync:
```
accum_vgpr ≈ 4×4×4 = 64
arch_vgpr  ≈ 32 (B) + 4 (A prefetch) + 32 (A regs) + 16 (addr) ≈ 84
```

Occupancy: 256 arch_vgpr / 256 accum_vgpr per SIMD on gfx942.
- ≤128 arch_vgpr → 2 waves/SIMD (good)
- 129–256 → 1 wave/SIMD (acceptable for compute-bound)
- >256 → spill (regression)

`accum_vgpr` and `arch_vgpr` are **separate register files**; MFMA
accumulators don't compete with prefetch buffers.

## 9. Performance Metrics

```python
# TFLOPS
flops = 2 * M * N * K
tflops = flops / 1e12 / (t_us / 1e6)

# Peak (gfx942 MI300X single GCD):
#   FP8:   ~653 TFLOPS    BF16: ~326 TFLOPS    INT8: ~653 TOPS
# Max-Achievable (typical ~50% of peak): FP8 ~326 TF/s, BF16 ~163 TF/s

# Bandwidth (FP8/INT8):
bytes_moved = (M*K*1) + (N*K*1) + (M*N*2) + (M+N)*4
tbps = bytes_moved / 1e12 / (t_us / 1e6)

# INT4 W4A8:
bytes_moved = M*K + (N*K)//2 + M*N*2 + (M+N)*4

# FP4 (MXFP4):
bytes_moved = (M*K)//2 + (N*K)//2 + M*N*2 + (M+N)*(K//32)
```

Memory- vs compute-bound rule of thumb:
- M ≤ 512 → memory-bound (focus bandwidth)
- M > 512 → compute-bound (focus MFMA utilization)

## 10. ATT Trace Bottleneck Matrix

From `kernel-trace-analysis` + `rocprofv3` ATT data:

| Symptom | Bottleneck | Action |
|---|---|---|
| High `s_waitcnt vmcnt(0)` before MFMA | Global load latency exposed | Improve prefetch overlap; increase `tile_k` |
| High `s_waitcnt lgkmcnt(0)` | LDS latency exposed | Increase write→read distance; check bank conflicts |
| High `s_barrier` stall | Workgroup sync overhead | Check `lds_stage`; reduce barrier count |
| MFMA utilization <50% | Memory-bound | Increase tile size; prefetch more aggressively |
| Many `s_nop` between MFMAs | Pipeline bubbles | Interleave loads between MFMAs; tune `hot_loop_scheduler` |
| High-cycle `buffer_load` | TA-blocked | Reduce concurrent loads; check access coalescing |

## 11. Counting Main-Loop ISA (sanity check)

```bash
FLYDSL_DUMP_IR=1 python my_gemm.py
grep -c "v_mfma"      final_isa.s
grep -c "s_barrier"   final_isa.s
grep -c "buffer_load" final_isa.s
grep -c "ds_read"     final_isa.s
grep -c "ds_write"    final_isa.s
```

Healthy ratios:
- MFMA ratio ≥ 40% → compute-dominant
- 30–40% → acceptable
- < 30% → too much non-MFMA overhead — review scheduler

Target: FlyDSL MFMA count should match the reference (aiter or CK); barrier
count ≤ reference.

## 12. Worked Example: 5120 × 5120 × 8320 FP8 GEMM

```
tile (64, 256, 128)
Grid = 80 × 20 = 1600 blocks
k_unroll=2, m_repeat=4, num_acc_n=4 → 64 MFMA/tile
Total MFMA/block = 64 × (8320/128) = 4160
LDS = 2 × 8 KB = 16 KB (well under 64 KB)
VGPR ~84 arch + 64 accum → 2 waves/SIMD
flops = 2 × 5120 × 5120 × 8320 = 436 GFLOP
Target ~500 TFLOPS → ~0.87 ms
Bytes ≈ 137.6 MB → 158 GB/s (well below HBM peak 5.3 TB/s)
→ Compute-bound. Focus: MFMA utilization, scheduler tuning.
```

## 13. The `_TILE_PRELOAD_TABLE`

The production preshuffle GEMM (`kernels/preshuffle_gemm.py:24-104`) carries an
empirically-tuned table of `(dsrd_preload, dvmem_preload)` per
`(tile_m, tile_n, tile_k)`. Example entries:

```python
(16, 256, 256): (2, 2),
(64, 128, 128): (8, 8),
(64, 256, 128): (8, 8),
(128, 128, 128): (8, 8),
(256, 256, 128): (4, 4),
```

Default fallback `(0, 0)`. If you adopt a new tile shape, run the autotuner
or copy from the nearest sibling shape. These values feed the prologue
prefetch count in `hot_loop_scheduler()`.
