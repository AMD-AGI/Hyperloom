# Attention backend dispatch on ROCm vLLM

**Source checkout:** `vllm-amd` (vllm main,
May 2026). Citations are `<path>:<line>`.

## Two-layer pick: registry resolves the class, platform picks priority

### Layer 1 — backend registry

`AttentionBackendEnum` in `vllm/v1/attention/backends/registry.py:34-89`.
Every member is `NAME = "fully.qualified.ClassPath"`. Relevant ROCm
entries:

```python
class AttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    TRITON_ATTN              = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    ROCM_ATTN                = "vllm.v1.attention.backends.rocm_attn.RocmAttentionBackend"
    ROCM_AITER_MLA           = "vllm.v1.attention.backends.mla.rocm_aiter_mla.AiterMLABackend"
    ROCM_AITER_TRITON_MLA    = "vllm.v1.attention.backends.mla.aiter_triton_mla.AiterTritonMLABackend"
    ROCM_AITER_FA            = "vllm.v1.attention.backends.rocm_aiter_fa.AiterFlashAttentionBackend"
    ROCM_AITER_MLA_SPARSE    = "vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse.ROCMAiterMLASparseBackend"
    TRITON_MLA               = "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend"
    ROCM_AITER_UNIFIED_ATTN  = "vllm.v1.attention.backends.rocm_aiter_unified_attn.RocmAiterUnifiedAttentionBackend"
    TURBOQUANT               = "vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend"
    FLASH_ATTN               = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    TORCH_SDPA               = ""   # ViT-only tag, no class path
```

`enum.get_class()` (L109) calls `resolve_obj_by_qualname` on the value
string, honoring any runtime `register_backend(...)` override.

### Layer 2 — platform priority + validate

`RocmPlatform._get_backend_priorities` (rocm.py:357-388, full body):

```python
def _get_backend_priorities(use_mla: bool, use_sparse: bool) -> list[AttentionBackendEnum]:
    from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops

    if use_sparse:
        return [AttentionBackendEnum.ROCM_AITER_MLA_SPARSE]

    if use_mla:
        if rocm_aiter_ops.is_mla_enabled():
            return [
                AttentionBackendEnum.ROCM_AITER_MLA,
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.ROCM_AITER_TRITON_MLA,
            ]
        else:
            return [AttentionBackendEnum.TRITON_MLA]

    backends = [AttentionBackendEnum.ROCM_ATTN]
    if rocm_aiter_ops.is_mha_enabled():
        backends.append(AttentionBackendEnum.ROCM_AITER_FA)
    if is_aiter_found_and_supported():
        backends.append(AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)
    backends.append(AttentionBackendEnum.TRITON_ATTN)
    backends.append(AttentionBackendEnum.TURBOQUANT)
    return backends
```

Lower index = higher priority. `get_attn_backend_cls` (rocm.py:475-561)
walks the list, calls each backend's `validate_configuration`, and
returns the **lowest-priority valid** one. If `--attention-backend
<NAME>` was passed it validates just that one and skips the walk
(rocm.py:486-505).

`is_aiter_found_and_supported()` requires ROCm AND gfx9 family AND
`find_spec("aiter") is not None` (_aiter_ops.py:66-86).

## Priority tables by mode

| Mode | Priority order |
| --- | --- |
| `use_sparse=True` (C4A indexer) | `[ROCM_AITER_MLA_SPARSE]` only |
| `use_mla=True`, `VLLM_ROCM_USE_AITER_MLA=1` (default) | `[ROCM_AITER_MLA, TRITON_MLA, ROCM_AITER_TRITON_MLA]` |
| `use_mla=True`, `VLLM_ROCM_USE_AITER_MLA=0` | `[TRITON_MLA]` only |
| Non-MLA, aiter enabled | `[ROCM_ATTN, ROCM_AITER_FA, ROCM_AITER_UNIFIED_ATTN, TRITON_ATTN, TURBOQUANT]` |
| Non-MLA, no aiter | `[ROCM_ATTN, TRITON_ATTN, TURBOQUANT]` |

`use_mla` is per-layer (model config); `use_sparse=True` when the
layer has a sparse indexer (DeepSeek V4 C4A path,
`deepseek_v4.py:1033-1046`, gated on `compress_ratio == 4`). **A
single model can use two different backends in one forward** —
e.g. C4A layers on `ROCM_AITER_MLA_SPARSE`, regular MLA layers on
`ROCM_AITER_MLA`.

## CLI flag

`--attention-backend` is wired via `vllm/engine/arg_utils.py:867`:

```python
attention_group.add_argument(
    "--attention-backend", **attention_kwargs["backend"]
)
```

stored as `AttentionConfig.backend` (arg_utils.py:641). When set,
`get_attn_backend_cls` validates just that backend and emits:
```
Using <NAME> backend (selected via --attention-backend).
```
(rocm.py:501-505).

## Auto-pick logging

When no `--attention-backend` is given (rocm.py:544-559):

```python
if invalid_reasons:
    logger.info(
        "Found incompatible backend(s) [%s] with %s. "
        "Overriding with %s out of potential backends: %s.",
        rejected_str, attn_type, selected_backend.name, valid_str,
    )
else:
    logger.info_once(
        "Using %s backend out of potential backends: %s.",
        selected_backend.name, valid_str,
    )
```

Grep server logs for those exact prefixes to confirm which backend
is running before profiling. There is no per-layer log line.

## ViT (image encoder) backends

Distinct path. `RocmPlatform.get_supported_vit_attn_backends`
(rocm.py:563-570):

```python
return [
    AttentionBackendEnum.FLASH_ATTN,
    AttentionBackendEnum.ROCM_AITER_FA,
    AttentionBackendEnum.TRITON_ATTN,
    AttentionBackendEnum.TORCH_SDPA,
]
```

`get_vit_attn_backend` (rocm.py:572-615) picks:

```python
if rocm_aiter_ops.is_enabled() and on_gfx9():
    return AttentionBackendEnum.ROCM_AITER_FA           # L591-593
if on_gfx9() and find_spec("flash_attn") is not None \
        and dtype in (torch.float16, torch.bfloat16):
    return AttentionBackendEnum.FLASH_ATTN              # L595-601
if on_gfx1x() and flash_attn_triton_available() \
        and dtype in (torch.float16, torch.bfloat16):
    return AttentionBackendEnum.FLASH_ATTN              # RDNA3/4 path
return AttentionBackendEnum.TORCH_SDPA                  # final fallback
```

## Env-var toggles (defaults from `vllm/envs.py:111-125`)

| Env | Default | Class cache attr | What it toggles |
| --- | :---: | --- | --- |
| `VLLM_ROCM_USE_AITER` | `False` (L111) | `_AITER_ENABLED` | Master switch for aiter ops |
| `VLLM_ROCM_USE_AITER_MHA` | `True` (L118) | `_MHA_ENABLED` | Enables `ROCM_AITER_FA` for MHA |
| `VLLM_ROCM_USE_AITER_MLA` | `True` (L117) | `_MLA_ENABLED` | Enables `ROCM_AITER_MLA` for MLA |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | `False` (L123) | `_TRITON_UNIFIED_ATTN_ENABLED` | Enables `ROCM_AITER_UNIFIED_ATTN` |
| `VLLM_ROCM_USE_AITER_PAGED_ATTN` | `False` (L112) | — | Paged-attention aiter shim |
| `VLLM_ROCM_USE_AITER_FP8BMM` | `True` (L121) | `_FP8BMM_ENABLED` | FP8 BMM for MLA |
| `VLLM_ROCM_USE_AITER_FP4BMM` | `True` (L122) | `_FP4BMM_ENABLED` | FP4 BMM (gfx950-gated, _aiter_ops.py:1414-1417) |
| `VLLM_ROCM_USE_AITER_MOE` | `True` (L114) | `_FMOE_ENABLED` | Fused MoE aiter path |

Note: master `VLLM_ROCM_USE_AITER=False` by default. Sub-toggles are
`True` by default, but the `is_xxx_enabled` check is ANDed with master
(`return cls._AITER_ENABLED and cls._MLA_ENABLED` etc. — see
_aiter_ops.py:1389, 1394). So **nothing is enabled until master is
turned on**.

Refresh after monkey-patching env vars in tests:
`RocmAiterOps.refresh_env_variables()` (_aiter_ops.py:1230-1250).

## Sparse / MLA gotcha (DeepSeek V4 C4A)

`DeepseekV4Attention.__init__` instantiates a `DeepseekV4Indexer`
only when `compress_ratio == 4` (deepseek_v4.py:1033-1046):

```python
self.indexer = None
if self.compress_ratio == 4:
    self.indexer = DeepseekV4Indexer(...)
```

C4A layers carry the indexer; their attn metadata triggers
`use_sparse=True` and dispatches to `ROCM_AITER_MLA_SPARSE` (the
single-entry priority list). Non-C4A layers in the same model use
the regular MLA priority list. **The backend pick is per-layer, not
per-model.**

## Not investigated

- `TurboQuantAttentionBackend` validation conditions (when it
  becomes the chosen backend).
- `flash_attn_diffkv` backend (CUDA-only by line count).
- Per-backend `validate_configuration` rejection reasons (logged as
  `"<NAME>: [<reason>]"` strings; needs case-by-case audit).
