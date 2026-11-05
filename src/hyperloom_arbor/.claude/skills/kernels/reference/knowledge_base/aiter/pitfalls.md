# AITER Pitfalls

## Version & Build Traps

### Version mismatch between aiter-amd package and container
The `aiter-amd` pip package version must match the container's ROCm version.
- SYMPTOM: Missing symbols (e.g., `gated_rmsnorm_fp8_group_quant` not found)
- SYMPTOM: ABI incompatibility → silent wrong results
- FIX: `pip show aiter-amd` and compare with container ROCm version
- FIX: Rebuild from source if mismatched

### JIT compilation hangs on first use
Importing AITER operators triggers JIT compilation if not cached. Can take 30+ seconds.
- SYMPTOM: Python appears to hang at import time
- FIX: Pre-compile: `python -c "from aiter.ops import flash_attn"` before benchmarking
- FIX: Set `AITER_JIT_DIR` to persistent location across runs
- FIX: Use `PREBUILD_KERNELS=1` during install for pre-compilation

### Stale JIT cache masks source changes
AITER's JIT (`aiter.jit.core.build_module`) caches compiled .so files. Editing
CK headers or kernel source won't trigger recompilation.
- FIX: `AITER_REBUILD=1` to force rebuild, `AITER_REBUILD=2` to also delete .so
- FIX: Remove `~/.aiter/jit/` or `AITER_JIT_DIR` contents

### ENABLE_CK=0 silently excludes operators
Setting `ENABLE_CK=0` disables Composable Kernel but doesn't error — operators
that need CK simply become unavailable at runtime.
- SYMPTOM: `ModuleNotFoundError` or `AttributeError` when calling CK-backed ops
- FIX: Check `ENABLE_CK` env var; default is 1

## Attention Traps

### FP8 flash attention restrictions
FP8 mode does NOT support:
- `dropout_p > 0` — silently produces wrong results
- `return_softmax_lse` — not implemented
- `return_attn_probs` — not implemented
- Custom attention bias
- FIX: Always use FP8 attention with dropout_p=0 and no return flags

### Causal + backward produces NaNs
Both causal+fused_backward and causal+dropout produce NaN gradients.
- SYMPTOM: NaN in backward pass, correct forward
- FIX: For causal training, avoid dropout OR avoid fused backward
- This is a known limitation in the CK flash attention implementation

### Paged attention head_size constraint for FP8 KV cache
FP8 KV cache requires `head_size % 16 == 0`.
- SYMPTOM: GPU memory fault or wrong results with odd head sizes
- FIX: Pad head dimension to next multiple of 16

### 32-bit stride overflow with large models
Models with 128+ attention heads (e.g., LLaMA 3 405B) can overflow 32-bit
stride calculations.
- SYMPTOM: Memory access fault on large batch/seq combinations
- FIX: `export AITER_INT64_STRIDES=1`

### PA high_precision parameter
Paged attention `high_precision` controls accuracy vs speed:
- 0 = fast (lowest precision)
- 1 = standard (default)
- 2 = highest (recommended for FP8 KV cache)
- Using 0 with FP8 KV cache causes significant accuracy loss

## GEMM Traps

### Preshuffle format constraints
Weight pre-shuffling (`bpreshuffle=True`) requires:
- `N % 16 == 0`
- `K % 32 == 0`
- Violating these constraints causes silent wrong results
- FIX: Pad weights to required alignment before shuffle

### SplitK selection matters for narrow K
For K < 512, default splitK=0 may choose a slow kernel. The tuned CSV config
selects optimal splitK per (M, N, K) shape.
- FIX: Use tuned configs: `aiter/configs/a8w8_tuned_gemm.csv`
- FIX: Set `AITER_LOG_TUNED_CONFIG=1` to verify config selection

### CK tile GEMM vs legacy CK GEMM
Two CK GEMM backends exist: legacy CK (compile-time instances) and CK-tile
(runtime-generated). They have different shape constraints and performance.
- `gemm_a8w8_blockscale_ck()` — legacy CK
- `gemm_a8w8_blockscale_cktile()` — CK-tile (newer, more flexible)
- FIX: Try both and benchmark for your specific shapes

## MoE Traps

### Block size alignment is mandatory
`moe_align_block_size()` must be called before any fused MoE op. Tokens per
expert are padded to `block_size` boundaries.
- SYMPTOM: CUDA error or wrong results without alignment
- FIX: Always call `moe_align_block_size(topk_ids, num_experts, block_size, ...)`

### MXFP4 MoE BLOCK_SIZE_M=32 is hardcoded
`fmoe_fp8_blockscale_g1u1` has `block_size_M=32` hardcoded by the MX spec.
Do NOT attempt to change this.

### MOE quantization algorithm names are strings, not enums
Tests use string names like `"int8quant"`, `"fp8quant"`, `"int8smoothquant"`,
`"fp8smoothquant"`, `"wint4afp8smoothquant"` — not the QuantType enum.
- Each algorithm has different supported operator variants (g1u0 vs g1u1)
- `"No"` quant only works with g1u0 and ck variants

### Shared experts change routing
When `num_shared_experts > 0`, the gating output and scoring function change.
- `shared_expert_scoring_func` controls how shared expert scores are computed
- Mismatch between gating and scoring causes silent accuracy loss

## Normalization Traps

### RMSNorm fallback threshold
RMSNorm uses ASM kernel for small hidden dimensions but falls back to CK when:
- `input.size(-1) > 8192`
- `use_model_sensitive_rmsnorm > 0`
- The CK fallback may be slower — benchmark both

### Fused residual add is in-place
`fused_add_rms_norm_cu(input, residual_in, weight, epsilon)` modifies BOTH
`input` and `residual_in` tensors in-place.
- SYMPTOM: Downstream ops see modified input
- FIX: `.clone()` inputs if they're needed later (but beware clone stride trap)

## Quantization Traps

### per_1x32_f4_quant pack_dim matters
`per_1x32_f4_quant()` with `pack_dim=-1` packs for LHS (activations),
`pack_dim=0` packs for RHS (weights). Using the wrong pack_dim causes
dimension mismatch in GEMM.

### Scale dtype must match kernel expectation
Some kernels expect FP32 scales, others FP8 scales (`per_1x32_f8_scale_f8_quant`).
Passing wrong scale dtype doesn't error — it produces wrong results.

## Import & Dispatch Traps

### Import-time stale bindings
Similar to CK: importing captures function references. If you monkey-patch
an AITER operator, verify the binding is updated in ALL call sites.
- `from aiter.ops.flash_attn import flash_attn_func` captures at import
- Patching `aiter.ops.flash_attn.flash_attn_func` doesn't update the local binding

### @compile_ops decorator hides import errors
The `@compile_ops` decorator catches compilation failures at decoration time.
If a module fails to compile, the decorated function silently returns None.
- SYMPTOM: `TypeError: 'NoneType' object is not callable`
- FIX: Check `AITER_LOG_LEVEL=DEBUG` for compilation errors

### Multiple backends for same operation
Same operation may exist in CK, ASM, and Triton variants. They have different
performance profiles and shape constraints. When comparing, ensure you're
calling the same variant consistently.
- `gemm_a8w8_ck()` vs `gemm_a8w8_asm()` — different optimal shapes
- `pa_fwd_asm()` vs `paged_attention_v1()` — different KV cache formats

## Benchmarking Traps

### Isolated benchmarks are misleading
AITER operators should be benchmarked IN-CONTEXT (within the full model),
not in isolation. Isolated benchmarks miss:
- Memory allocation patterns from surrounding operators
- Cache effects from the actual data flow
- Occupancy interactions with adjacent kernels

### FP8 tolerance is wide
Standard FP8 test tolerance: atol=0.3, rtol=0.1. Dropout amplifies errors.
A test "passing" at FP8 tolerance doesn't mean results match FP16/BF16.
- For production: verify with cosine similarity ≥ 0.96

### ASM kernel selection is architecture-dependent
ASM kernels are precompiled for specific GPU architectures (gfx942, gfx950).
Running on wrong arch either falls back to slow path or crashes.
- FIX: Verify `get_gfx()` matches expected architecture
