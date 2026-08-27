# CK Tile Programming Model

## Template Parameter Hierarchy
```
Block Tile (MPerBlock, NPerBlock, KPerBlock)
  └── Warp Tile (MPerWarp, NPerWarp, KPerWarp)
      └── MFMA Instruction (e.g., 32x32x16 for bf16)
```

Block tile must be divisible by (num_warps × warp_tile).
Warp tile must be divisible by MFMA instruction dimensions.

## Common CK Types
- `DeviceGemm<F16, F16, F16, Row, Col, Row, ...>` — GEMM
- `DeviceBatchedGemm` — batched GEMM
- `DeviceGroupedGemm` — grouped (variable-size) GEMM
- `DeviceGemmSplitK` — split-K GEMM for tall-skinny shapes

## Pipeline Types
- `BlockGemmPipelineV1` — basic pipeline
- `BlockGemmPipelineV2` — double-buffered async prefetch
- `BlockGemmPipelineVSA` — variable split, used for attention

## Data Layout
- `Row` = row-major (M×K with stride K)
- `Col` = column-major (K×N with stride N)
- Must match the actual tensor layout in memory

## Tile Size Selection Heuristics (gfx950)
For GEMM with M,N,K ≥ 1024:
- Block: 128×128×64 or 256×128×64
- Warp: 32×32×16 (matches MFMA)
- Occupancy target: 2 waves per SIMD

For small M (M < 128):
- Block: M×128×64
- May need split-K for utilization

For attention (softmax in loop):
- Block_Q: 64 or 128
- Block_KV: 64
- Must fit Q + K + V + accumulator in registers
