# CK VSA Sparse Attention Forward — Inference Optimization

## Summary
CK-tile block-sparse attention forward for FastVideo on MI355X (gfx950).
2.0–2.7× over Triton at Sq≥8K, bf16, ~10% sparsity.

## Key Architecture
- Pipeline: `BlockGemmPipelineVSA` with delta-encoded LUT
- Block map: 128-token Q blocks × 64-token KV blocks (nq=Sq/128, nkv=Sk/64)
- Tile: kM0=128, kN0=64, kK0=32, 4 warps → 8 waves/CU (LDS=27,136 B unchanged)
- Memory-bound: WAIT_ANY/MFMA ≈ 9–10

## Wrapper Optimizations (cumulative 1.22–1.48× over CK base)
1. **Fused delta LUT**: `map_to_index_and_delta` triton kernel emits abs+delta in one launch
2. **skip_vbs_correction**: skip VBS kernel when all blocks are full (saves ~5 µs)
3. **Pre-computed delta passthrough**: `q2k_delta` kwarg skips HIP abs→delta kernel
4. Combined: single GPU launch (CK attention kernel only)

## block_m=128 Tile (biggest win)
- Doubles K/V reuse per Q token → directly attacks the HBM bandwidth bottleneck
- Same LDS footprint → doubles wave occupancy (4→8 waves/CU)
- PMC: VMEM_INSTS/MFMA dropped from ~7.4 to 0.29
- 1.18× faster than block_m=64 at same density; up to 2.69× over Triton

## When to Use Which block_m
- **block_m=128**: Sq ≥ 8K, uniform blocks, natively-generated 128-granularity block maps
- **block_m=64**: small Sq (<8K), variable block sizes (VBS), 64-token Q granularity required
- **Merged pattern (union of m=64 rows to m=128) is NOT recommended**: doubles effective topk, negating K/V reuse benefit

## VBS Correction
- Split `block_q`/`block_kv` parameters (was single `BLOCK=64`)
- With block_m=128, VBS kernel indexes Q blocks at Sq/128 granularity
- Hardcoded BLOCK=64 caused OOB read with m=128 → GPU memory fault

## Code Locations
- Kernel wrapper: `FastVideo/fastvideo-kernel/csrc/attention/ck_sparse/ck_vsa_fwd.hip`
- Python wrapper: `FastVideo/fastvideo-kernel/python/fastvideo_kernel/ck_sparse_attn.py`
- Delta LUT: `FastVideo/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/index.py`
- CK codegen: `aiter-amd/3rdparty/composable_kernel/example/ck_tile/50_sparse_attn/`
- Tests: `FastVideo/fastvideo-kernel/benchmarks/strict_test_ck_fwd.py`
- Guide: `FastVideo/fastvideo-kernel/docs/ck_block_sparse_attn_guide.md`
- Branches: `feature/ck-block-sparse-attn-clean` (FastVideo), `sla-ck-fwd-release-inference` (aiter-amd)

## Performance (MI355X, D=128, ~10% sparsity)
| Sq | Triton (ms) | CK opt (ms) | Speedup |
|----|-------------|-------------|---------|
| 4,096 | 0.054 | 0.033 | 1.66× |
| 16,384 | 0.632 | 0.258 | 2.45× |
| 49,152 | 5.651 | 2.104 | 2.69× |
| 65,536 | 9.370 | 3.771 | 2.48× |

## Lessons Learned
1. VBS kernel parameter coupling: block_q and block_kv must be independent when tile sizes differ
2. Merged block maps (union for m=128 from m=64 rows) double topk → slower, not faster
3. LUT memory: CK m=128 has same LUT overhead as Triton; CK m=64 is 2× (abs+delta)
4. Runtime peak memory is identical across implementations; only LUT overhead differs
5. CK submodule pin: `042d4f0` on composable_kernel (single commit: VSA sparse attention forward)
