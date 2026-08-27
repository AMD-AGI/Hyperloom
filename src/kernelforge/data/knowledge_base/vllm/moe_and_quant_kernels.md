# MoE and quant kernel inventory (ROCm vLLM)

**Source checkout:** `vllm-amd`. Paths absolute
under that root.

## MoE execution paths — selected at `FusedMoE.__init__`

The dispatch happens inside `FusedMoEMethod` subclasses, then inside
`FusedMoE.forward`. On ROCm, the master switch is the bool
`self.rocm_aiter_fmoe_enabled` set at
`vllm/model_executor/layers/fused_moe/layer.py:362-364`:

```python
self.rocm_aiter_fmoe_enabled = (
    rocm_aiter_ops.is_fused_moe_enabled() and is_act_and_mul
)
self.aiter_fmoe_shared_expert_enabled = (
    rocm_aiter_ops.is_fusion_moe_shared_experts_enabled() and is_act_and_mul
)
```

`is_act_and_mul=False` (non-gated activation) forces the Triton path
because aiter only supports gated (silu/gelu).

### 1. Triton fused_moe (canonical, CUDA + ROCm)

`vllm/model_executor/layers/fused_moe/fused_moe.py` (2363 lines).
Entry points are registered as torch custom ops:

```python
# fused_moe.py:1439-1444
direct_register_custom_op(
    op_name="inplace_fused_experts",
    op_func=inplace_fused_experts,         # body wraps fused_experts_impl(... inplace=True)
    mutates_args=["hidden_states"],
    fake_impl=inplace_fused_experts_fake,
)
# fused_moe.py:1531-1535
direct_register_custom_op(
    op_name="outplace_fused_experts",
    op_func=outplace_fused_experts,        # body returns fused_experts_impl(... inplace=False)
    fake_impl=outplace_fused_experts_fake,
)
```

`inplace_fused_experts` (L1355) and `outplace_fused_experts` (L1447)
both pass through to `fused_experts_impl` (L1381 / L1473) — the
difference is the 6th positional arg (`inplace=True/False`).

Per-shape tuning JSONs:
`vllm/model_executor/layers/fused_moe/configs/E=*,N=*,device_name=*.json`.
The `device_name=` suffix must match `_ROCM_DEVICE_ID_NAME_MAP`
(e.g. `AMD_Instinct_MI300X`); a missed match falls back to a default
config and silently drops perf.

Used when `is_fused_moe_enabled() == False` (master `VLLM_ROCM_USE_AITER=0`
or sub `VLLM_ROCM_USE_AITER_MOE=0` or `is_act_and_mul=False`).

### 2. aiter prebuilt fused_moe (ROCm)

Shim wrapper: `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py`
(507 lines). Entry point `rocm_aiter_fused_experts(...)` at L235 is
called from `FusedMoEMethod.apply` (e.g. L479):

```python
# rocm_aiter_fused_moe.py:479-494
result = rocm_aiter_fused_experts(
    hidden_states=hidden_states,
    w1=w1, w2=w2,
    topk_weights=topk_weights, topk_ids=topk_ids,
    activation=activation,
    apply_router_weight_on_input=apply_router_weight_on_input,
    expert_map=expert_map,
    quant_config=self.quant_config,
    moe_config=self.moe_config,
    a1q_scale=a1q_scale,
    num_local_tokens=num_local_tokens,
    output_dtype=output.dtype,
    moe_sorting_dispatch_policy=envs.VLLM_ROCM_AITER_MOE_DISPATCH_POLICY,
)
```

Underlying torch op `vllm.rocm_aiter_fused_moe` is registered at
`_aiter_ops.py:1505-1511` (see `amd_rocm_code_locations.md`). Its impl
just does:

```python
# _aiter_ops.py:127-153
def _rocm_aiter_fused_moe_impl(hidden_states, w1, w2, topk_weight, topk_ids, ...) -> torch.Tensor:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    activation = ActivationType(activation_method)
    quant_type = QuantType(quant_method)
    # ... call aiter.fused_moe.fused_moe with translated enums
```

Activation enum mapping (used by callers, _aiter_ops.py:1275-1281):

```python
mapping = {"none": ActivationType.No, "no": ActivationType.No,
           "silu": ActivationType.Silu, "gelu": ActivationType.Gelu,
           "swiglu": ActivationType.Swiglu}
```

QuantType mapping at L1306-1313 covers `no`, `per_tensor`, `per_token`,
`per_1x32`, `per_1x128`, `per_128x128`.

### 3. aiter ASM TKW1 MoE (w8a8 path)

`torch.ops.vllm.rocm_aiter_asm_moe_tkw1`, impl at `_aiter_ops.py:206-240`:

```python
def _rocm_aiter_asm_moe_tkw1_impl(hidden_states, w1, w2, topk_weights, topk_ids,
                                  fc1_scale, fc2_scale, fc1_smooth_scale, fc2_smooth_scale,
                                  a16, per_tensor_quant_scale, expert_mask, activation_method):
    from aiter import ActivationType
    from aiter.fused_moe_bf16_asm import asm_moe_tkw1
    activation = ActivationType(activation_method)
    return asm_moe_tkw1(hidden_states, w1, w2, topk_weights, topk_ids,
                        fc1_scale=fc1_scale, fc2_scale=fc2_scale, ...)
```

Public static API `RocmAiterOps.asm_moe_tkw1` at _aiter_ops.py:1824.
Used by specific w8a8 schemes; the `fc1_smooth_scale` / `fc2_smooth_scale`
args carry the per-channel smooth-quant scaling produced at load time.

### 4. aiter mxfp4 / w4a8 MoE experts

`vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py`
— dedicated mxfp4 w4a8 expert kernel, `import aiter` directly.
Selected through the FusedMoEMethod chain when the Quark scheme is
w4a8 mxfp4 (`quark/schemes/quark_w4a8_mxfp4_fp8.py`).

### 5. DeepSeek V4 MegaMoE (FP4, deep_gemm-backed)

`DeepseekV4MegaMoEExperts` at `deepseek_v4.py:391`. Built only when
both gates are satisfied:

```python
# deepseek_v4.py:1229-1237 (in DeepseekV4Model.__init__)
self.use_mega_moe = (
    vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
)
if self.use_mega_moe and not vllm_config.parallel_config.enable_expert_parallel:
    raise NotImplementedError(
        "DeepSeek V4 MegaMoE currently requires expert parallel. "
        ...
    )
```

Forward at L598; requires `expert_dtype="fp4"` (deepseek_v4.py:741-746).
Primarily CUDA — ROCm uses path 1 (Triton) or 2 (aiter).

### 6. Mori EP prepare-finalize

`vllm/model_executor/layers/fused_moe/prepare_finalize/mori.py`
`import aiter` directly for the Mori expert-parallel dispatch/combine.

## Scale-layout decisions — where weights get reshaped

Three layers; **errors here become silent correctness loss** because
no graph capture re-runs the loader.

1. **At weight load** — `FusedMoE.weight_loader` and per-method flags
   like `DeepseekV4MegaMoEExperts.weight_loader.supports_moe_loading`
   (deepseek_v4.py:663). Layout transforms baked into the loader
   write directly into the final shape on device.

2. **Inside the quant scheme module** —
   `vllm/model_executor/layers/quantization/quark/quark_moe.py` and
   `quark/schemes/*.py` decide e.g. preshuffle / block-scale K-tail
   padding for w4a8 mxfp4.

3. **At dispatch time inside `_aiter_ops`** — most aiter ops accept
   the scale as a plain `Tensor` and **assume the layout the loader
   produced**. The wrappers do *not* validate layout.

## Quant scheme catalogue

Schemes accepted on ROCm — registered in `RocmPlatform.supported_quantization`
(rocm.py:406-429):

```python
supported_quantization = [
    "awq", "awq_marlin",   # awq_marlin -> awq
    "gptq", "gptq_marlin", # gptq_marlin -> gptq
    "fp8", "deepseek_v4_fp8", "compressed-tensors", "fbgemm_fp8",
    "gguf", "quark", "mxfp4", "mxfp8", "torchao", "bitsandbytes",
    "modelopt", "modelopt_fp4", "modelopt_mxfp8", "modelopt_mixed",
    "fp8_per_tensor", "fp8_per_block", "online", "gpt_oss_mxfp4",
]
```

Files of interest under `vllm/model_executor/layers/quantization/`:

| File | What |
| --- | --- |
| `fp8.py` | Base FP8 weight + act quant. Extended by `DeepseekV4FP8Config` (`deepseek_v4.py:120`). |
| `mxfp4.py` / `mxfp8.py` | Block-scaled FP4/FP8 |
| `quark/quark_moe.py` | Quark-MoE; `import aiter` direct (uses aiter GEMM) |
| `quark/schemes/quark_w4a8_mxfp4_fp8.py` | w4a8 mxfp4 weights + FP8 acts; pairs with `experts/aiter_mxfp4_w4a8_moe.py` |
| `quark/schemes/quark_ocp_mx.py` | OCP-MX block-scaled scheme |
| `compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py` | w8a8 FP8 path; calls aiter GEMMs via `kernels/linear/scaled_mm/aiter.py` |

## aiter linear / scaled GEMM dispatch

Single integration point: `vllm/model_executor/kernels/linear/scaled_mm/aiter.py`,
`import aiter` for the scaled-mm GEMM kernels. Gated by
`VLLM_ROCM_USE_AITER_LINEAR=True` (default; envs.py:113). All FP8 linears
in attention/QKV/O projections go through here.

Block-scale GEMMs have two variants — both registered as torch ops:

```python
# _aiter_ops.py:613-622 (Triton variant)
def _rocm_aiter_triton_gemm_a8w8_blockscale_impl(A, B, As, Bs, output_dtype):
    from aiter.ops.triton.gemm_a8w8_blockscale import gemm_a8w8_blockscale
    return gemm_a8w8_blockscale(A, B, As, Bs, dtype=output_dtype)

# _aiter_ops.py:638-647 (asm/HIP variant)
def _rocm_aiter_gemm_a8w8_blockscale_impl(A, B, As, Bs, output_dtype):
    from aiter import gemm_a8w8_blockscale
    return gemm_a8w8_blockscale(A, B, As, Bs, dtype=output_dtype)
```

Triton variant gated by `VLLM_ROCM_USE_AITER_TRITON_GEMM=True`
(envs.py:125).

## Hot-kernel registry seed (decode, ROCm DSv4)

Trace-bucketing pass should expect these names:

| Bucket | Underlying symbol(s) |
| --- | --- |
| `moe_gemm` | `rocm_aiter_fused_moe`, `fused_experts` Triton kernels |
| `topk` | `rocm_aiter_biased_grouped_topk`, `persistent_topk` |
| `mla_attn` | `rocm_aiter_mla_decode_fwd`, `triton_mla_decode_*` |
| `sparse_attn_indexer` | `sparse_attn_indexer`, Triton top-k |
| `quant_op` | `rmsnorm2d_fwd_with_add_dynamicquant` (aiter), `dynamic_per_token_fp8_quant` |
| `rmsnorm` | `rocm_aiter_rmsnorm_*`, `fused_q_kv_rmsnorm` |
| `rope` | `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` (HIP) |
| `gemm_dense` | hipBLASLt / aiter `gemm_a8w8` |
| `allreduce_1stage` | aiter `fused_ar_rms` or quickreduce |

## Not investigated

- Per-shape tuning coverage of the JSON config files (which `(E, N,
  device)` triples currently miss a config).
- Whether the deep_gemm MegaMoE path has any active ROCm support
  (raises on ROCm when `moe_backend=deep_gemm_mega_moe`).
- `turboquant` / `humming` quant schemes — present but not audited.
