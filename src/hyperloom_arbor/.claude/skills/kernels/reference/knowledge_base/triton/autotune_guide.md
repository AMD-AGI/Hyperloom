# Triton Autotune Guide (AMD)

## Autotune Configuration
```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
                       num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
                       num_stages=3, num_warps=4),
        # ... more configs
    ],
    key=["M", "N", "K"],  # re-tune when these change
)
@triton.jit
def my_kernel(...):
    ...
```

## Config Selection Strategy for gfx950

### GEMM (M, N, K ≥ 1024)
Start with:
- BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=2
- BLOCK_M=128, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3
Then try:
- BLOCK_K=32 (lower register pressure) vs BLOCK_K=128 (more compute per load)
- num_stages=1 vs 2 vs 3 (tradeoff: prefetch vs register pressure)

### Attention
- BLOCK_Q=64 or 128 (matches head dimension)
- BLOCK_KV=64 (sweet spot for gfx950)
- num_stages=1 (attention has complex control flow, prefetch helps less)
- num_warps=4 or 8

### Reduction (softmax, layernorm)
- BLOCK_SIZE=1024 or 2048
- num_warps=8 (maximize parallelism within block)
- num_stages=1

## Environment Variables
```bash
TRITON_PRINT_AUTOTUNING=1      # Show which config wins
TRITON_ALWAYS_COMPILE=1        # Bypass cache
MLIR_ENABLE_DUMP=1             # Dump MLIR for debugging
AMDGCN_ENABLE_DUMP=1           # Dump ISA
```

## num_warps Selection (AMD-specific)
- gfx950 has 64-lane wavefronts
- num_warps=4 → 256 threads per block
- num_warps=8 → 512 threads per block
- More warps = more parallelism but higher register pressure
- Rule: start with num_warps=8, reduce if spilling

## num_stages Selection
- num_stages=1: no prefetch (simplest, lowest register pressure)
- num_stages=2: double buffer (1 compute + 1 prefetch)
- num_stages=3: triple buffer (rare win on AMD, usually increases reg pressure)
- For attention: usually num_stages=1 wins
- For GEMM: usually num_stages=2 wins
