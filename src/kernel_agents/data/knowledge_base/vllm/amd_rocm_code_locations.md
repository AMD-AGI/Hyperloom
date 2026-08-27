# AMD / ROCm-specific code map in vLLM

**Source checkout:** `vllm-amd` (vllm main,
May 2026). All paths relative to that root.

## Platform layer — `vllm/platforms/rocm.py` (983 lines)

Single source of truth for AMD GPU bringup. Module-level constants are
populated at import via amdsmi (no CUDA init); Ray workers can still
`CUDA_VISIBLE_DEVICES`-set after import.

### Device ID -> name table (L58-74)

```python
_ROCM_DEVICE_ID_NAME_MAP: dict[str, str] = {
    "0x74a0": "AMD_Instinct_MI300A",
    "0x74a1": "AMD_Instinct_MI300X",
    "0x74b5": "AMD_Instinct_MI300X",       # MI300X VF
    "0x74a2": "AMD_Instinct_MI308X",
    "0x74a5": "AMD_Instinct_MI325X",
    "0x74b9": "AMD_Instinct_MI325X",       # MI325X VF
    "0x74a9": "AMD_Instinct_MI300X_HF",    # HBM3e variant
    "0x74bd": "AMD_Instinct_MI300X_HF",
    "0x744c": "AMD_Radeon_RX7900XTX",
    "0x150e": "AMD_Radeon_890M",           # gfx1150, Strix Point
    "0x1586": "AMD_Radeon_8060S",          # gfx1151, Strix Halo
    "0x7550": "AMD_Radeon_RX9070XT",       # gfx1201
    "0x7551": "AMD_Radeon_R9700",          # gfx1201
}
```

This is what the runtime sees as `gpu_props.name`; MoE tuning JSONs
use these exact strings (e.g. `device_name=AMD_Instinct_MI300X.json`).

### GFX arch detection (L150-193)

```python
# rocm.py:185 — resolved once at module load via amdsmi (no CUDA init)
_GCN_ARCH = _get_gcn_arch()                            # e.g. "gfx942"

_ON_GFX1X  = any(arch in _GCN_ARCH for arch in ["gfx11", "gfx12"])
_ON_GFX12X = any(arch in _GCN_ARCH for arch in ["gfx12"])
_ON_MI3XX  = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950"])
_ON_GFX9   = any(arch in _GCN_ARCH for arch in ["gfx90a", "gfx942", "gfx950"])
_ON_GFX90A = "gfx90a" in _GCN_ARCH
_ON_GFX942 = "gfx942" in _GCN_ARCH                     # MI300X/325X
_ON_GFX950 = "gfx950" in _GCN_ARCH                     # MI355X
```

Helpers `on_gfx9()`, `on_gfx942()`, `on_gfx950()`, etc. at L267-292
just return the booleans — fully `torch.compile`/Dynamo-safe.

### Quant whitelist (rocm.py:406-429)

```python
supported_quantization: list[str] = [
    "awq", "awq_marlin",   # awq_marlin will be overwritten with awq
    "gptq", "gptq_marlin", # gptq_marlin will be overwritten with gptq
    "fp8", "deepseek_v4_fp8", "compressed-tensors", "fbgemm_fp8",
    "gguf", "quark", "mxfp4", "mxfp8", "torchao", "bitsandbytes",
    "modelopt", "modelopt_fp4", "modelopt_mxfp8", "modelopt_mixed",
    "fp8_per_tensor", "fp8_per_block", "online", "gpt_oss_mxfp4",
]
```

### Kernel import (rocm.py:431-440)

```python
@classmethod
def import_kernels(cls) -> None:
    super().import_kernels()
    import contextlib
    with contextlib.suppress(ImportError):
        import vllm._rocm_C  # noqa: F401
```

Suppressed on ImportError so a CUDA-only build does not crash —
gated by CMake (`CMakeLists.txt` at repo root, 1254 lines). The
extension is produced from `csrc/rocm/torch_bindings.cpp`.

### Custom all-reduce gate (rocm.py:816-819)

```python
@classmethod
def use_custom_allreduce(cls) -> bool:
    # We only enable custom allreduce for MI300 series
    return any(gfx in _GCN_ARCH for gfx in ["gfx94", "gfx95"])
```

So enabled on MI300X/325X (gfx942) and MI355X (gfx950), disabled on
MI200/gfx90a and RDNA.

## ROCm C++ kernels — `csrc/rocm/`

Compact directory, 4 files:

| File | Contains |
| --- | --- |
| `csrc/rocm/attention.cu` | ROCm paged-attention `ll4mi` / `mfma16` family; bound as `paged_attention_rocm` |
| `csrc/rocm/skinny_gemms.cu` | Skinny (M small, N big) GEMMs for decode hot path |
| `csrc/rocm/torch_bindings.cpp` | `TORCH_LIBRARY` bindings into `vllm._rocm_C` |
| `csrc/rocm/ops.h` | C++ declarations |

Cross-cutting kernels with ROCm conditional paths live in `csrc/`
proper: `dsv3_fused_a_gemm.cu`, `fused_qknorm_rope_kernel.cu`,
`fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`,
`minimax_reduce_rms_kernel.cu`, `layernorm_quant_kernels.cu`.

## aiter integration — `vllm/_aiter_ops.py` (2404 lines)

Wraps every aiter callable as a `torch.library` custom op so it's safe
in cudagraph capture and `torch.compile`.

### Module-level aiter probe (_aiter_ops.py:30-42)

```python
def is_aiter_found() -> bool:
    """Check if aiter library is available (just the library)."""
    return find_spec("aiter") is not None

# `find_spec` is not torch.compile compatible.
# Set once outside forward passes that are torch compiled.
IS_AITER_FOUND = is_aiter_found()
```

Module-level constant, NOT a class attr. Subsequent gates
(`RocmAiterOps.is_enabled()` etc.) cheap-check this bool instead of
re-importing.

### Platform + library gate (_aiter_ops.py:66-86)

```python
def is_aiter_found_and_supported() -> bool:
    if current_platform.is_rocm() and IS_AITER_FOUND:
        from vllm.platforms.rocm import on_mi3xx
        return on_mi3xx()    # gfx942 or gfx950 only
    return False
```

Used by `_get_backend_priorities` to gate `ROCM_AITER_UNIFIED_ATTN`
without checking env vars (explicit backend selection works even with
`VLLM_ROCM_USE_AITER=0`).

### `RocmAiterOps` class — env-var cache + static API (L1180+)

Caches every `VLLM_ROCM_USE_AITER_*` env var as a class attr
(_aiter_ops.py:1205-1221):

```python
class RocmAiterOps:
    _AITER_ENABLED                = envs.VLLM_ROCM_USE_AITER
    _LINEAR_ENABLED               = envs.VLLM_ROCM_USE_AITER_LINEAR
    _FMOE_ENABLED                 = envs.VLLM_ROCM_USE_AITER_MOE
    _MLA_ENABLED                  = envs.VLLM_ROCM_USE_AITER_MLA
    _MHA_ENABLED                  = envs.VLLM_ROCM_USE_AITER_MHA
    _TRITON_UNIFIED_ATTN_ENABLED  = envs.VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION
    _FP8BMM_ENABLED               = envs.VLLM_ROCM_USE_AITER_FP8BMM
    _FP4BMM_ENABLED               = envs.VLLM_ROCM_USE_AITER_FP4BMM
    _MOE_SHARED_EXPERTS_ENABLED   = envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS
    # ... more aliases (_TRITON_ROTARY_EMBED, _SHUFFLE_KV_CACHE_ENABLED, etc.)
```

`is_xxx_enabled` methods AND with `_AITER_ENABLED`:

```python
# _aiter_ops.py:1389-1395
@classmethod
@if_aiter_supported
def is_mla_enabled(cls) -> bool:
    return cls._AITER_ENABLED and cls._MLA_ENABLED
@classmethod
@if_aiter_supported
def is_mha_enabled(cls) -> bool:
    return cls._AITER_ENABLED and cls._MHA_ENABLED
```

So **every sub-toggle is a no-op until `VLLM_ROCM_USE_AITER=1` is
set** (master default is `False`). `@if_aiter_supported` decorator
additionally requires ROCm + gfx9 + library present.

To refresh env vars after monkey-patching (tests):
`RocmAiterOps.refresh_env_variables()` (_aiter_ops.py:1229-1250).

### Representative op wrappers

All are `@staticmethod` thin shims calling `torch.ops.vllm.rocm_aiter_*`:

| Method | Line | Underlying op | Backed by |
| --- | --- | --- | --- |
| `fused_moe` | L1778 | `torch.ops.vllm.rocm_aiter_fused_moe` | `aiter.fused_moe.fused_moe` |
| `asm_moe_tkw1` | L1824 | `torch.ops.vllm.rocm_aiter_asm_moe_tkw1` | `aiter.fused_moe_bf16_asm.asm_moe_tkw1` |
| `topk_softmax` | L1856 | `torch.ops.vllm.rocm_aiter_topk_softmax` | `aiter.topk_softmax` |
| `biased_grouped_topk` | L1890 | `torch.ops.vllm.rocm_aiter_biased_grouped_topk` | aiter |
| `grouped_topk` | L1914 | `torch.ops.vllm.rocm_aiter_grouped_topk` | aiter |

Each `torch.ops.vllm.*` op is registered with `direct_register_custom_op`
(from `vllm/utils/torch_utils.py`) at _aiter_ops.py:1495-1700. Example
registration (L1505-1511):

```python
direct_register_custom_op(
    op_name="rocm_aiter_fused_moe",
    op_func=_rocm_aiter_fused_moe_impl,         # imports `aiter.fused_moe.fused_moe`
    mutates_args=[],
    fake_impl=_rocm_aiter_fused_moe_fake,
    dispatch_key=current_platform.dispatch_key,  # "CUDA" on ROCm
)
```

The fake impl returns an empty-shape tensor so cudagraph capture and
torch.compile shape inference works without launching the real kernel.

## Files that `import aiter` directly (bypass the shim)

```
vllm/_aiter_ops.py                                                  (main wrapper)
vllm/model_executor/kernels/linear/scaled_mm/aiter.py                (FP8 linear)
vllm/model_executor/layers/utils.py
vllm/model_executor/layers/attention/mla_attention.py                (MLA layer)
vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py
vllm/model_executor/layers/fused_moe/prepare_finalize/mori.py        (Mori EP)
vllm/model_executor/layers/quantization/quark/quark_moe.py
vllm/model_executor/layers/mamba/gdn_linear_attn.py
vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py
vllm/model_executor/layers/quantization/quark/utils.py
```

Reproduce: `grep -rln "^[[:space:]]*\(import\|from\) aiter"
vllm/model_executor/ vllm/_aiter_ops.py`.

## ROCm attention backends (V1) — `vllm/v1/attention/backends/`

| File | AttentionBackendEnum name | Use |
| --- | --- | --- |
| `rocm_attn.py` | `ROCM_ATTN` | Default MHA fallback for ROCm |
| `rocm_aiter_fa.py` | `ROCM_AITER_FA` | aiter flash-attention MHA |
| `rocm_aiter_unified_attn.py` | `ROCM_AITER_UNIFIED_ATTN` | aiter unified attn |
| `triton_attn.py` | `TRITON_ATTN` | Triton fallback |
| `turboquant_attn.py` | `TURBOQUANT` | Quantized attn |
| `mla/rocm_aiter_mla.py` | `ROCM_AITER_MLA` | aiter MLA decode |
| `mla/rocm_aiter_mla_sparse.py` | `ROCM_AITER_MLA_SPARSE` | DSv3.2/V4 C4A sparse path |
| `mla/aiter_triton_mla.py` | `ROCM_AITER_TRITON_MLA` | aiter+Triton MLA |
| `mla/triton_mla.py` | `TRITON_MLA` | Triton MLA fallback |

Selection priority defined in `_get_backend_priorities` (rocm.py:357-388);
see `attention_backend_dispatch.md`.

## Triton kernels with ROCm-aware tuning

- `vllm/model_executor/layers/fused_moe/fused_moe.py` (2363 lines) —
  canonical Triton fused-MoE. Tuning configs in
  `vllm/model_executor/layers/fused_moe/configs/`, JSONs named
  `E=...,N=...,device_name=AMD_Instinct_MI300X.json`. Filename
  `device_name=` string must exactly match `_ROCM_DEVICE_ID_NAME_MAP`
  values.
- `vllm/model_executor/layers/sparse_attn_indexer.py` — Triton paths
  for sparse-attn indexer top-k (used when not on CUDA's
  `persistent_topk`; see `dsv4_forward_map.md`).
- `vllm/model_executor/layers/rotary_embedding/` — RoPE.

## ROCm-specific runtime decisions (sample, with code)

```python
# vllm/model_executor/models/deepseek_v4.py:1249-1253
# Disable aux CUDA streams on ROCm to avoid hangs
aux_stream_list = (
    None
    if current_platform.is_rocm()
    else [torch.cuda.Stream() for _ in range(3)]
)
```

```python
# vllm/model_executor/layers/deepseek_v4_attention.py:311-322
# ROCm keeps BF16 reference wo_a path; CUDA uses fused FP8 einsum
if current_platform.is_rocm():
    z = rocm_inv_rope_einsum(
        self.rotary_emb, o, positions,
        self.rope_head_dim, self.n_local_groups,
        self.o_lora_rank, self.wo_a,
    )
    return self.wo_b(z.flatten(1))
# else: fused_inv_rope_fp8_quant + deepseek_v4_fp8_einsum, see L325-352
```

## Not investigated

- `csrc/quickreduce/` — separate AMD all-reduce kernel family.
- `csrc/cutlass_extensions/` — likely NVIDIA-only; not confirmed.
- `vllm/model_executor/layers/quantization/turboquant/` — referenced
  by `TurboQuantAttentionBackend` but quant sub-tree not audited.
