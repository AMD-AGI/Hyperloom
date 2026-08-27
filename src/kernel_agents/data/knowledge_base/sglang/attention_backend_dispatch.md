# Attention backend selection in sglang

Static analysis of `sglang` @ b4fe0246c.

## Registration table

All backends register at import time via the decorator below
(`python/sglang/srt/layers/attention/attention_registry.py`):

```python
# attention_registry.py:20-28
ATTENTION_BACKENDS = {}
def register_attention_backend(name):
    def decorator(fn):
        ATTENTION_BACKENDS[name] = fn
        return fn
    return decorator
```

| name | factory line | backend class / file |
| ---- | -----------: | -------------------- |
| `flashinfer` | `:31` | `FlashInferAttnBackend` / `FlashInferMLAAttnBackend` (chooses on `runner.use_mla_backend`) |
| `trtllm_mla` | `:56` | `TRTLLMMLABackend` (asserts MLA) |
| `aiter` | `:65` | `AiterAttnBackend` (`aiter_backend.py:112`) |
| `wave` | `:72` | `WaveAttnBackend` |
| `ascend` | `:79` | `AscendAttnBackend` (hardware_backend/npu) |
| `nsa` | `:88` | `NativeSparseAttnBackend` (`nsa_backend.py`) |
| `compressed` | `:95` | `DeepseekV4BackendRadix` or `DeepseekV4Backend` |
| `triton` | `:113` | `TritonAttnBackend` (asserts not encoder/decoder) |
| `torch_native` | `:124` | `TorchNativeAttnBackend` |
| `flex_attention` | `:131` | `TorchFlexAttnBackend` |
| `flashmla` | `:138` | `FlashMLABackend` |
| `fa3` | `:145` | `FlashAttentionBackend` (SM≥80 or MUSA MP≥31) |
| `fa4` | `:171` | `FlashAttentionBackend(runner, fa_impl_ver=4)` |
| `cutlass_mla` | `:178` | `CutlassMLABackend` |
| `trtllm_mha` | `:185` | `TRTLLMHAAttnBackend` (asserts non-MLA) |
| `intel_amx` | `:194` | `IntelAMXAttnBackend` |
| `dual_chunk_flash_attn` | `:201` | `DualChunkFlashAttentionBackend` |
| `intel_xpu` | `:289` | `XPUAttentionBackend` |

`attn_backend_wrapper` (`:210`) wraps the chosen backend with
`HybridLinearAttnBackend` when `runner.mambaish_config` is set (mamba2 /
GDN / KDA / lightning hybrid linear-attn paths).

## Per-model auto-selection (server_args)

`python/sglang/srt/server_args.py:851` calls
`_handle_attention_backend_compatibility`. Model-specific blocks:

DeepseekV4 (`server_args.py:1675-1689`):
```python
if model_arch in ["DeepseekV4ForCausalLM"]:
    if self.is_attention_backend_not_set():
        self.attention_backend = "compressed"
    if self.page_size is None:
        self.page_size = 256
    if self.kv_cache_dtype == "auto":
        self.kv_cache_dtype = "fp8_e4m3"
```

DeepseekV3 / V3.2 / Kimi K25 / Pixtral / MistralLarge3 / GlmMoeDsa
(`:1691-1768`):
```python
if model_arch in [...DSv3 family...]:
    if is_deepseek_nsa(hf_config):          # DSv3.2 / GLM-DSA
        envs.SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.set(
            get_nsa_index_topk(hf_config))  # :1715
        if self.is_attention_backend_not_set():
            self.attention_backend = "nsa"  # :1722
        if not is_npu() and self.enable_nsa_prefill_context_parallel:
            self.enable_dp_attention = True       # :1732
            self.moe_dense_tp_size = 1            # :1733
            self.moe_a2a_backend = "deepep"       # :1734
        if is_hip():
            self.page_size = 1                    # :1762
        else:
            self.page_size = 64                   # :1768
```

User overrides (`server_args.py:499-501, 530`):
```python
attention_backend: Optional[str] = None             # :499
decode_attention_backend: Optional[str] = None      # :500 — split decode
prefill_attention_backend: Optional[str] = None     # :501 — split prefill
speculative_draft_attention_backend: Optional[str] = None  # :530 — draft spec
```

## `compressed` dispatch (DSv4)

```python
# attention_registry.py:95-110
@register_attention_backend("compressed")
def create_compressed_backend(runner):
    from sglang.srt.environ import envs
    if envs.SGLANG_OPT_DPSK_V4_RADIX.get():  # default True (environ.py:578)
        from sglang.srt.layers.attention.deepseek_v4_backend_radix import (
            DeepseekV4BackendRadix)
        return DeepseekV4BackendRadix(runner)
    else:
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4Backend)
        return DeepseekV4Backend(runner)
```

```python
# deepseek_v4_backend.py:62
class DeepseekV4Backend(AttentionBackend, C4IndexerBackend, CompressorBackend):
    ...

# deepseek_v4_backend.py:363-364  (inside .forward)
backend = os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "kernel")
o = flash_mla_with_kvcache_entrypoint(**input_dict, backend=backend)[0]
```

The entrypoint dispatches by platform:
```python
# layers/attention/debug_flash_mla_adapter.py:11-44
def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
    if is_hip():
        from sglang.srt.layers.attention.nsa.tilelang_kernel import (
            dpsk_v4_fp8_attention_fwd)
        backend = os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "tilelang")  # :19
    else:
        import flash_mla
    if backend == "torch":      return flash_mla_with_kvcache_torch(**kwargs)
    if backend == "tilelang":   return dpsk_v4_fp8_attention_fwd(**kwargs)
    if backend == "kernel":     return flash_mla.flash_mla_with_kvcache(**kwargs)
    if backend == "comparison": ...  # runs torch + kernel and asserts close
    raise NotImplementedError(f"unknown backend: {backend!r}")
```

So on AMD the DSv4 "compressed" core attention is always the **TileLang**
`dpsk_v4_fp8_attention_fwd`, NOT aiter PA or flash_mla. The
`SGLANG_HACK_FLASHMLA_BACKEND` env can pick `torch` ref or `comparison`
mode for correctness debugging.

TileLang kernel branches on gfx95
(`layers/attention/nsa/tilelang_kernel.py:2337-2340`):
```python
if _is_gfx95_supported:
    block_I, threads, num_stages, block_per_cu, cu = 64, 512, 0, 2, 256
else:
    block_I, threads, num_stages, block_per_cu, cu = 32, 128, 1, 1, 304
```

## `aiter` backend (other models)

```python
# layers/attention/aiter_backend.py:32-45
try:
    from aiter import (
        flash_attn_varlen_func, get_mla_metadata_info_v1, get_mla_metadata_v1,
        get_ps_metadata_info_v1, get_ps_metadata_v1, mha_batch_prefill_func,
        mla_prefill_ps_asm_fwd, mla_reduce_v1, paged_attention_ragged,
    )
    from aiter.mla import mla_decode_fwd, mla_prefill_fwd
    from aiter.ops.triton.attention.unified_attention import unified_attention
except ImportError: ...
```
FP8-prefill toggle (`aiter_backend.py:66-68`):
```python
_use_fp8_prefill_attn = (
    get_bool_env_var("SGLANG_AITER_FP8_PREFILL_ATTN", "True")
    and is_gfx95_supported())
```

## NSA backend (DSv3.2, GLM-DSA)

`NativeSparseAttnBackend` in `layers/attention/nsa_backend.py`. ROCm
hot-call sites in `layers/attention/nsa/nsa_indexer.py`:
- `:42-43`: `if _use_aiter: from aiter.ops.cache import indexer_k_quant_and_cache`
- `:624-630`: `from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits`
  used for ROCm; CUDA path takes `deep_gemm.fp8_mqa_logits`.

## AMD-specific overrides summary

| env var | default | effect |
| ------- | ------- | ------ |
| `SGLANG_HACK_FLASHMLA_BACKEND` | `"kernel"` (env), `"tilelang"` (HIP override in adapter `:19`) | picks DSv4 core attn impl: `tilelang`/`torch`/`kernel`/`comparison` |
| `SGLANG_AITER_FP8_PREFILL_ATTN` | True | only effective on gfx95 (and only enables fp8 prefill on aiter backend) |
| `SGLANG_USE_AITER` | False (`environ.py:323`) | gates every aiter import via per-file `_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()` |
| `SGLANG_OPT_DPSK_V4_RADIX` | True (`environ.py:578`) | picks `DeepseekV4BackendRadix` vs `DeepseekV4Backend` for `compressed` |

## Not investigated

- Spec-decode draft backend wiring (`speculative_draft_attention_backend`).
- Hybrid linear-attn variants in `hybrid_linear_attn_backend.py`.
