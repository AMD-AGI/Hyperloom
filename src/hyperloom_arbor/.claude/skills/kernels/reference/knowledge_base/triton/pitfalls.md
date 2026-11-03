# Triton Pitfalls (AMD Backend)

## AMD vs NVIDIA Differences

### Warp size
- AMD: wavefront = 64 lanes (not 32)
- Triton's `tl.arange(0, BLOCK)` with BLOCK < 64 wastes lanes
- Prefer BLOCK sizes that are multiples of 64

### Matrix instruction mapping
- `tl.dot()` maps to MFMA instructions on AMD
- Minimum efficient dot size: 16×16 (maps to mfma_f32_16x16x32)
- Preferred: 32×32 (maps to mfma_f32_32x32x16)

### Buffer loads
- `ENABLE_AMD_BUFFER_LOADS=1` can improve memory throughput
- Not always faster — profile both modes

## Autotune Pitfalls

### Exhaustive search is expensive
`triton.autotune` runs ALL configs. For attention kernels with 50+ configs,
this takes 30+ minutes.
- Strategy: pre-filter configs based on problem size, then autotune the shortlist

### Autotune results are cache-dependent
Triton caches compiled kernels in `~/.triton/cache/`. Stale cache entries
can mask code changes.
- FIX: `rm -rf ~/.triton/cache/` when kernel source changes substantially

### Config carryover between shapes
Optimal config for shape A may not be optimal for shape B.
Always autotune on the target shape, not a proxy.

## Numerical Precision

### Accumulator precision
`tl.dot(a, b, acc)` accumulates in f32 by default on AMD.
But intermediate values in attention (softmax) can overflow in f16.
- Always use f32 accumulators for attention
- Convert to f16/bf16 only at store time

### atomic_add ordering
`tl.atomic_add` does NOT guarantee ordering across workgroups.
For attention backward dK/dV accumulation, this can cause non-deterministic
results that pass allclose but fail exact match.

## Performance Traps

### Small block sizes
BLOCK_M=32 or BLOCK_N=32 underutilizes MFMA on gfx950.
Minimum efficient: BLOCK_M=64, BLOCK_N=64.

### Excessive register spill
Triton's register allocator can spill aggressively for complex kernels.
Check with `TRITON_PRINT_AUTOTUNING=1` for occupancy info.
Reduce `num_stages` or `num_warps` if spilling.

### grid() function overhead
Complex grid computations in Python add launch overhead.
Pre-compute grid dimensions outside the kernel.
