# FlyDSL DSL Patterns

## Kernel Definition Pattern
```python
@flydsl_kernel
def my_gemm(
    a: Tensor["M", "K", dtype=bf16],
    b: Tensor["K", "N", dtype=bf16],
    c: Tensor["M", "N", dtype=f32],
):
    # Tile dimensions
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    
    # Compute
    acc = mfma_f32_32x32x16_bf16(a_tile, b_tile, acc)
    
    # Store
    store(c, acc)
```

## MFMA Instruction Selection
- `mfma_f32_32x32x16_bf16`: 32×32 output, good for large tiles
- `mfma_f32_16x16x32_bf16`: 16×16 output, 2× K reduction per instruction
- For gfx950: prefer 32×32 variant for better register utilization

## Async DMA (gfx950)
```python
FLYDSL_SLA_FWD_ENABLE_DMA=1  # Enable async buffer_load_dwordx4_lds
```
Mandatory for sparse attention — without it, LDS fill dominates latency.

## Reduction Modes
- `"xor"`: standard cross-lane reduction
- `"ds_bpermute"`: use LDS permute for reduction (tiny gain in isolation,
  but anti-composes with wpe=3)

## Hybrid Strategy
When a subkernel is too complex for pure FlyDSL (register pressure):
1. Use FlyDSL for the compute-heavy part (e.g., preprocess + dQ)
2. Use Triton/CK for the remainder (e.g., dKdV)
3. This captured 80% of the total win for SLA backward
