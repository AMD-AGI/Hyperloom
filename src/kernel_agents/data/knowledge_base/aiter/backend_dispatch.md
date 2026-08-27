# AITER Backend Dispatch System

## Backend Overview

AITER selects among 4 kernel backends per operation:

| Backend | When Used | Strengths |
|---------|-----------|-----------|
| **CK (Composable Kernel)** | Default for GEMM, attention (training), normalization, RoPE | Flexible, CK-tile codegen, splitK support |
| **ASM (precompiled HSACO)** | Attention decode, MoE, activation, topK | Lowest latency, hand-tuned for specific shapes |
| **Triton** | Alternative implementations, fallback | Fast iteration, Python-level tuning |
| **hipBLASLt** | Standard GEMM via `hipb_mm` | Vendor-optimized, easy integration |

## Dispatch Decisions by Operator

### Attention Dispatch
```
if kv_cache_dtype in ("fp8", "int8"):
    → ASM (pa_fwd_asm) with high_precision=2
elif total_heads > 2 * cu_num:
    → ASM (compute-heavy heuristic)
elif is_decode and max_qlen == 1:
    → ASM (PA decode kernel)
elif is_prefill:
    → CK (mha_fwd or fmha_v3_fwd)
else:
    → HIP/CK fallback
```

### GEMM Dispatch (A8W8)
```
1. Exact match: lookup_table[(M, N, K)] → kernel instance
2. Bucketed: lookup_table[(nextPow2(M), N, K)] → kernel instance
3. Heuristic: rowwise_heuristic_dispatch(M, N, K) → default kernel
```
Tuned configs in: `aiter/configs/a8w8_tuned_gemm.csv`
Format: `[cu_num, M, N, K, kernelName, splitK, us]`

### RMSNorm Dispatch
```
if input.size(-1) > 8192 or use_model_sensitive_rmsnorm > 0:
    → CK (module_rmsnorm)
else:
    → ASM (module_rmsnorm, assembly path)
```

### MoE Dispatch
```
if quant_type == "fp8" and blockscale:
    → ASM (fmoe_fp8_blockscale_g1u1)
elif quant_type in ("int8quant", "fp8quant"):
    → ASM (fmoe_g1u1)
elif quant_type == "No":
    → CK 2-stage (ck_moe_stage1/stage2) or ASM (fmoe)
```

## ASM Kernel Metadata

### Directory Structure
```
hsa/
├── gfx942/              # MI300X
│   ├── bf16gemm/        # BF16 GEMM HSACO files
│   ├── f4gemm/          # MXFP4 GEMM
│   ├── fp8gemm_blockscale/
│   ├── i8gemm/          # INT8 GEMM
│   ├── fmha_v3_fwd/     # Flash Attention v3 forward
│   ├── fmha_v3_bwd/     # Flash Attention v3 backward
│   ├── fmoe/            # Fast MoE
│   ├── fmoe_2stages/
│   ├── mla/             # Multi-Latent Attention
│   ├── pa/              # Paged Attention
│   ├── topk_per_row_decode/
│   ├── topk_per_row_prefill/
│   └── topksoftmax/
├── gfx950/              # MI355X (same structure)
└── codegen.py           # Generates config headers from CSV
```

### ASM Config CSV Format
```csv
# Example: fmha_v3_fwd/fmha_fwd.csv
dtype,hdim_q,hdim_v,mask,mode,bf16_cvt,ts_qo,ts_kv,knl_name,co_name
bf16,128,128,0,0,0,256,32,_ZN5aiter24fmha_fwd_hd128_bf16_rtneE,fwd_hd128_bf16_rtne.co
bf16,128,128,2,0,0,256,32,_ZN5aiter28fmha_fwd_hd128_bf16_causal_rtneE,fwd_hd128_bf16_causal_rtne.co
```

### ASM Kernel Loading
```python
# codegen.py generates asm_*_configs.hpp from CSV
# C++ side loads HSACO via:
#   1. Read .co file from hsa/{gfx}/{op}/
#   2. Extract kernel function by mangled name
#   3. Launch via hipModuleLaunchKernel()
```

## CK Kernel Instance System

### Instance Definition (GEMM example)
```python
# csrc/ck_gemm_a8w8/gemm_a8w8_common.py
kernelInstance(
    BLOCK_SIZE=256,
    MPerBLOCK=128, NPerBLOCK=128, KPerBLOCK=128,
    WAVE_TILE_M=32, WAVE_TILE_N=32,
    WAVE_MAP_M=2, WAVE_MAP_N=2,
    ABLOCK_TRANSFER=[8, 32, 1],
    BBLOCK_TRANSFER=[8, 32, 1],
    CBLOCK_TRANSFER=[1, 32, 1, 8],
    CBLOCK_SPV=[8, 8, 1],
    LOOP_SCHED="Intrawave",   # or "Interwave"
    PIPELINE_VERSION=3         # 1-5
)
```

### Instance Name Encoding
```
a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3
         │        │          │     │      │          │          │       │         │       │
    dtype    block dims   wave  wave  A_transfer  B_transfer  C_transfer  C_spv  sched  pipeline
```

### Pipeline Versions
- v1: Basic (no prefetch)
- v2: Single-buffer prefetch
- v3: Double-buffer prefetch (default)
- v4: Async pipeline
- v5: Experimental

## @compile_ops Decorator System

```python
@compile_ops(
    module_name="module_gemm_a8w8",  # Matches optCompilerConfig.json key
    fc_name=None,                     # Function name override
    gen_func=None,                    # Dynamic kernel selection (returns {md_name, blob_gen_cmd})
    gen_fake=None,                    # Placeholder for shape inference
    ffi_type="pybind",               # "pybind" or "ctypes"
    develop=False,                    # Development mode (in-place compile)
)
def gemm_a8w8_ck(XQ, WQ, x_scale, w_scale, Out, bias=None, splitK=0):
    return get_module("module_gemm_a8w8").gemm_a8w8(XQ, WQ, x_scale, w_scale, Out, bias, splitK)
```

### Module Resolution Flow
1. `get_module(md_name)` checks if module is already imported
2. If not: reads `optCompilerConfig.json` for build recipe
3. Runs `blob_gen_cmd` if specified (e.g., ASM config generation)
4. Compiles with hipcc + CK headers
5. Links into .so and loads via pybind11 or ctypes
6. Caches in `~/.aiter/jit/` or `AITER_JIT_DIR`

## Architecture-Specific Routing

| Architecture | GPU | FP8 | MXFP4 | ASM Kernels |
|-------------|-----|-----|-------|-------------|
| gfx90a | MI250/MI250X | FNUZ only | No | Limited |
| gfx942 | MI300X/MI325X | OCP + FNUZ | Yes | Full set |
| gfx950 | MI350X/MI355X | OCP | Yes | Full set |

Detection:
```python
from aiter.jit.utils.chip_info import get_gfx, get_arch
gfx = get_gfx()     # "gfx942", "gfx950"
arch = get_arch()    # "MI300", "MI350"
```

## Tuned Configuration Files

```
aiter/configs/
├── a8w8_tuned_gemm.csv
├── a8w8_bpreshuffle_tuned_gemm.csv
├── a4w4_blockscale_tuned_gemm.csv
├── bf16_tuned_gemm.csv
├── a8w8_tuned_batched_gemm.csv
├── tuned_fmoe.csv
└── ...
```

Format: `cu_num,M,N,K,kernelName,splitK,us,...`
- `cu_num`: GPU compute unit count (architecture filter)
- Kernel selected by closest match to (M, N, K)
- `AITER_LOG_TUNED_CONFIG=1` logs which config was selected
