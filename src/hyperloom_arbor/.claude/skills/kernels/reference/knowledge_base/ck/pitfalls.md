# CK (Composable Kernel) Pitfalls

## Build System Traps

### Stale .so artifact
`ninja` in the build subdirectory does NOT copy the final .so to the runtime path.
Python keeps loading the stale version.
- FIX: Always `cp build/<module>/build/<module>.so ../../<module>.so` after ninja
- SYMPTOM: rocprof shows old kernel name (e.g., "BlockFmhaBwdDQDKDVPipelineVSA" when
  code says split pipeline)
- COST: ~1 hour to diagnose if not caught

### Header dependency tracking
`ninja` doesn't track changes to `3rdparty/composable_kernel/.../pipeline/*.hpp`.
Editing a pipeline header doesn't trigger recompilation.
- FIX: `rm -f *.cuda.o` before ninja when pipeline headers changed
- SYMPTOM: code edits have no effect on runtime behavior

## Tile Configuration Traps

### Warp tile is NOT an independent knob
Swapping warp tile dimensions without re-tuning LDS descriptors and block dimensions
causes silent regressions. The kernel compiles clean, passes SNR, but wall-clock regresses.
- EVIDENCE: 4-warp (16,16,16) warp tile regressed 11→14 ms (+27%)
- REASON: narrower warp tile = proportionally more MFMA instructions
- RULE: narrower tile needs wider block to amortize (bn0=128, bk0=64 for dense;
  sparse locked at bn0=64, bk0=32 → narrow tiles always lose)

### LDS aliasing locks occupancy
Sparse VSA dq-only reuses K_lds slot for V_lds. De-aliasing saves 16KB but pushes
2-block occupancy over 160KB → occupancy drops from 2 to 1.
Any async-prefetch redesign must prove LDS budget still allows target occupancy.

### block_m=128 tile: same LDS, double occupancy
kM0=128 tile (4 warps) uses identical LDS (27,136 B) as kM0=64 (2 warps) because
K/V prefetch buffers don't depend on kM0. This means block_m=128 gets 8 waves/CU
vs 4 waves/CU for block_m=64 — a free occupancy doubling.

## Register Pressure Traps

### AGPR inline asm bug (clang)
`asm volatile("..." : "+a"(fp32x16))` drops reg_idx=0 from the live set.
Result: first row of MFMA output holds half the correct value → 21 dB SNR.
- FIX: Use `__builtin_amdgcn_mfma_*` with empty-asm `"+v"` barrier:
  ```c
  c = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, c, 0, 0, 0);
  asm volatile("" : "+v"(c));
  ```
- ACHIEVED: 256 VGPR + 170 AGPR + 0 spill at 256 fp32/lane pressure

### Occupancy cliff at 256 VGPR
Reducing VGPR from 296→238 delivered −32% latency in a single change.
The trick: per-iteration Q reload from LDS instead of persistent q_reg.

## Data Corruption Traps

### BLOCK_M=64 with sparse attention (bwd)
BLOCK_M=64 + NUM_WAVES=4 causes wave_q_offset to spill past block boundary.
When workgroups have different LUT values, races corrupt cross-WG writes.
- FIX: BLOCK_M=128 with NUM_WAVES=8 (overlap stays in-bounds)
- SYMPTOM: SNR drops 4-11 dB on some configs, not all

### VBS correction kernel block_q/block_kv coupling (fwd)
VBS correction kernel originally hardcoded `BLOCK=64` for both Q-block and KV-block
indexing. With block_m=128, `q_blk = tid / 64` reads past the q2k_index bounds
(which has Sq/128 rows). Causes GPU memory access fault.
- FIX: Split into independent `block_q` and `block_kv` parameters
- SYMPTOM: GPU memory fault only with block_m=128 + VBS enabled

## Module-Level Traps

### Stale import bindings
`from .kernel import _attention` captures the function at import time.
Monkey-patching `kernel._attention` doesn't update the local binding in `core.py`.
- FIX: Patch BOTH locations
- COST: This hid an entire 10 ms kernel speedup until discovered

### save_for_backward overhead
autograd's `Function.apply` saves 7 tensors, costing ~10 ms per call even under
`torch.no_grad()` on MI355X.
- FIX: Gate on `requires_grad`: `if any(t.requires_grad for t in (q,k,v)): ctx.save_for_backward(...)`

## Block Map Traps

### Merged pattern (union for block_m=128) is slower
Merging adjacent m=64 Q-block rows via `bm_64[:,:,0::2,:] | bm_64[:,:,1::2,:]`
to create m=128 block maps doubles effective topk (~76→145 for 10% density).
The extra K/V blocks negate the K/V reuse benefit of block_m=128.
- RULE: Use natively-generated 128-granularity block maps, not merged m=64 maps
- EVIDENCE: Merged pattern is consistently slower than same-density m=64
