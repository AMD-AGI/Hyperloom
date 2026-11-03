# Pre-built Kernels Inventory (`kernels/*.py`)

> All production kernels in the FlyDSL repo, with target arch and notable
> features. Use this as a starting catalog before writing a new kernel.

## GEMM Family

| File | Kernel | Arch | Dtypes | Notes |
|---|---|---|---|---|
| `preshuffle_gemm.py` | `compile_preshuffle_gemm_a8` | gfx942 / gfx950 | fp8, int8, int4 (W4A8), fp16, bf16, fp4 | MFMA 16×16, B preshuffle layout `(N/16,K/64,4,16,kpack)`, K64 micro-step, ping-pong LDS (`lds_stage=2`), XOR16 swizzle, CShuffle epilogue option, `_TILE_PRELOAD_TABLE` for tuned prefetch counts |
| `preshuffle_gemm_v2.py` | newer variant | gfx942 / gfx950 | same | iteration on the v1 design |
| `blockscale_preshuffle_gemm.py` | blockscale FP8 GEMM | gfx942 / gfx950 | fp8 with 128×128 block scales | Per-block scale loaded per-row in epilogue; multiplied in-register (NOT LDS round-trip) |
| `hgemm_splitk.py` | FP16 split-K GEMM | gfx942 / gfx950 | fp16 | Split-K reduction with atomic accumulation |
| `rdna_f16_gemm.py` | RDNA fp16 GEMM | gfx120x | fp16 | WMMA atoms, wave32 |
| `rdna_fp8_preshuffle_gemm.py` | RDNA fp8 GEMM | gfx120x | fp8 | WMMA preshuffle |
| `gemm_common_gfx1250.py` | gfx1250 common helpers | gfx1250 | n/a | TDM (Tensor Descriptor Move) async-copy primitives |
| `gemm_fp8fp4_gfx1250.py` | gfx1250 fp8/fp4 GEMM | gfx1250 | fp8, fp4 | WMMA + TDM async |
| `wmma_gemm_gfx1250.py` | gfx1250 WMMA GEMM | gfx1250 | fp16/bf16 | WMMA base |

## MoE Family

| File | Stages | Arch | Notes |
|---|---|---|---|
| `moe_gemm_2stage.py` | gate/up + reduce | gfx942 / gfx950 | 2-stage: stage1 fuses gate+up projections (acc → softmax + dequant in epilogue); stage2 = down projection. Supports int4/int8 inputs and fp8/fp16/bf16 weights. Optional split-K (`k_batch > 1`) for high TP. |
| `moe_blockscale_2stage.py` | gate/up + reduce | gfx942 / gfx950 | Blockscale FP8 MoE |
| `mixed_moe_gemm_2stage.py` | gate/up + reduce | gfx942 / gfx950 | Mixed-precision (int4/int8/fp8/bf16 mix) |
| `moe_gemm_2stage_common_gfx1250.py` | shared MoE helpers | gfx1250 | WMMA + TDM |
| `moe_gemm_2stage_mxscale_gfx1250.py` | MX-scale MoE | gfx1250 | MXFP4/FP6/FP8 scaled |
| `moe_gemm_2stage_wmma_gfx1250.py` | WMMA MoE | gfx1250 | RDNA wave32 |
| `topk_gating_softmax_kernel.py` | top-k + softmax over gate logits | all | feeds MoE 2-stage |
| `silu_and_mul_fq.py` | activation + dequant | all | gate × silu(up) fused with FP quant |

## Attention Family

| File | Kernel | Arch | Notes |
|---|---|---|---|
| `pa_decode_fp8.py` | Paged-Attention decode | gfx942 / gfx950 | FP8 KV cache, paged via `kv_page_indices`. Tile: QUERY_GROUP=16, HEAD=128, KV_BLOCK=1024 phys page, KV_COMPUTE=256 tile. Per-warp 64 tokens × 128 head. MFMA 16×16 K32 for S=Q·K^T. **Status: WIP perf tuning** |
| `flash_attn_func.py` | FlashAttention (forward) | gfx942 / gfx950 | Online softmax in-register per row; GEMM1 K·Q^T, GEMM2 V·P. Persistent grid `(num_SM, 1, 4)` with work-stealing. **Status: WIP perf tuning** |
| `mla_fwd_decode.py` | MLA forward decode | gfx942 / gfx950 | DeepSeek-style multi-latent attention |
| `mla_fwd_decode_m16x8_fp8_fp8.py` | MLA FP8 variant | gfx942 / gfx950 | M=16 N=8 FP8/FP8 |

## Normalization / Activation

| File | Kernel | Arch | Notes |
|---|---|---|---|
| `layernorm_kernel.py` | LayerNorm | gfx942 / gfx950 / RDNA | 2-pass: mean+var → affine. f32/f16/bf16. Fast path when `N == BLOCK_THREADS*VEC_WIDTH*4` (e.g. N=8192). Software RNE pack on gfx942; hardware `cvt_pk_bf16_f32` on gfx950+ |
| `rmsnorm_kernel.py` | RMSNorm | gfx942 / gfx950 / RDNA | 3-pass with LDS row cache: Pass 0 G→LDS; Pass 1 sum-of-squares; Pass 2 normalize+gamma with software-pipelined Gamma prefetch |
| `softmax_kernel.py` | row-wise softmax | gfx942 / gfx950 / RDNA | 6-stage: load → local-max → global-max (XOR shuffle → LDS) → local-exp+sum → global-sum → normalize+store |
| `fused_rope_cache_kernel.py` | RoPE + KV cache write | gfx942 / gfx950 | fused rotation + KV cache store |

## Multi-GPU

| File | Kernel | Notes |
|---|---|---|
| `custom_all_reduce.py` | Multi-GPU all-reduce orchestrator | high-level entry |
| `custom_all_reduce_kernel.py` | reduction kernel | low-level |

## Shared Building Blocks

| File | Provides |
|---|---|
| `kernels_common.py` | `reduce_vec_max`, `reduce_vec_sum`, `make_block_reduce*`, `validate_moe_dtypes`, `dtype_to_elem_type`, `get_warp_size(arch)`, `stream_ptr_to_async_token`, `_if_then` SCF helper |
| `mfma_epilogues.py` | `default_epilog`, `c_shuffle_epilog`, `mfma_epilog` (dispatcher) |
| `mfma_preshuffle_pipeline.py` | `swizzle_xor16`, `make_preshuffle_b_layout`, `load_b_pack_k32`, `tile_chunk_coord_i32`, `buffer_copy_gmem16_dwordx4`, `lds_store_*b_xor16`, `lds_load_packs_k64`, `xcd_remap_bx_by` |
| `pipeline_utils.py` | `make_tail_plan` (tail scheduling), `tdm_epilogue_fence_threshold_bytes` (TDM reuse fence) |
| `layout_utils.py` | lightweight `idx2crd`/`crd2idx` with pow2 shift/mask optimizations |
| `tensor_shim.py` | `GTensor` / `STensor` global+shared tensor view abstractions |
| `fp8_gemm_utils.py` | preshuffle weight reshape helpers (host-side) |
| `fp8_gemm_4wave.py` / `fp8_gemm_8wave.py` | 4-wave vs 8-wave variants of fp8 GEMM |

## Tests

All in `tests/kernels/` (markers `l2_device` + `rocm_lower`, require GPU):
`test_preshuffle_gemm.py`, `test_blockscale_preshuffle_gemm.py`,
`test_hgemm_splitk.py`, `test_moe_gemm.py`, `test_moe_blockscale.py`,
`test_moe_reduce.py`, `test_pa.py`, `test_flash_attn_func.py`,
`test_layernorm.py`, `test_rmsnorm.py`, `test_softmax.py`,
`test_fused_rope_cache.py`, `test_allreduce.py`, `test_rdna_gemm.py`,
`test_gemm_fp8fp4_gfx1250.py`, `test_wmma_gemm_gfx1250.py`,
`test_vec_add.py`, `test_quant.py`.

Reference Torch implementations in `test_pa.py`: `reference_masked_attention()`,
`torch_mha_extend()`, `torch_mha_extend2()`.

## Examples (`examples/*.py`)

Use as templates rather than production code:

| File | Pattern |
|---|---|
| `01-vectorAdd.py` | Basic block/thread tiling, copy_atom_call, single-stage |
| `02-tiledCopy.py` | `zipped_divide` + `TiledCopy` with `partition_S/D` |
| `03-tiledMma.py` | 64×64×8 GEMM via 2×2 MMA layout, retile copy→MMA fragments |
| `04-preshuffle_gemm.py` | Full preshuffle pipeline (2-stage), double-buffer K-loop, scheduler directives, 128×128×64 tile |
