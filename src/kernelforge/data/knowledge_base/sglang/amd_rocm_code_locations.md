# AMD / ROCm code locations in sglang

Static analysis of `sglang` @ b4fe0246c
(`amd/deepseek_v4` branch). Use this as an index for "where on AMD does X
happen".

## Platform detection helpers (`python/sglang/srt/utils/common.py`)

```python
# common.py:105-107
@lru_cache(maxsize=1)
def is_hip() -> bool:
    return torch.version.hip is not None

# common.py:128-130
@lru_cache(maxsize=1)
def is_cuda():
    return torch.cuda.is_available() and torch.version.cuda is not None

# common.py:133-135
@lru_cache(maxsize=1)
def is_cuda_alike():
    return is_cuda() or is_hip()

# common.py:3481-3490  (note: a near-identical helper sits at :3471 too)
@lru_cache(maxsize=1)
def is_gfx95_supported():
    if torch.version.hip:
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        return any(gfx in gcn_arch for gfx in ["gfx95"])
    return False

# common.py:3493-3496
def get_hip_version():
    if torch.version.hip:
        return tuple(map(int, torch.version.hip.split("-")[0].split(".")))
    return (0, 0, 0)
```

FP8 max constant is platform-gated (`common.py:110-114`):
```python
if is_hip():
    HIP_FP8_E4M3_FNUZ_MAX = 224.0
    FP8_E4M3_MAX = HIP_FP8_E4M3_FNUZ_MAX
else:
    FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
```
`is_fp8_fnuz()` (in `layers/quantization/fp8_kernel.py`) is what consumers
use to pick `torch.float8_e4m3fnuz` vs `fn`
(`debug_flash_mla_adapter.py:8`).

## `_use_aiter` pattern

Every aiter consumer declares the same two-liner near the top of the file:
```python
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
```

| file:line | what flips on |
| --------- | ------------- |
| `layers/layernorm.py:47` | aiter rmsnorm2d_fwd / rmsnorm2d_fwd_with_add (`:73-75`) |
| `layers/communicator.py:85` | aiter rms+mxfp4, rms+fp8-group, rms+fp8-pertoken paths (`:538..620`) |
| `layers/attention/vision.py:73` | vision aiter attn |
| `layers/attention/nsa/nsa_indexer.py:33` | `indexer_k_quant_and_cache` import (`:43`) |
| `layers/moe/rocm_moe_utils.py:15` | `rocm_aiter_asm_moe_tkw1` custom op (`:27-49`) |
| `layers/moe/fused_moe_triton/layer.py:78` | aiter fused_moe runner registration |
| `layers/moe/deepseek_v4_topk.py:30` | DSv4 topk aiter path |
| `layers/moe/moe_runner/deep_gemm.py:45` | (HIP guard for deep_gemm MoE) |
| `layers/moe/topk.py:83` | `aiter_biased_grouped_topk` / `aiter_fused_topk` (`:138-139`) |
| `layers/moe/ep_moe/layer.py:54` | DeepEP `forward_aiter` (`:276`) |
| `models/deepseek_v2.py:148` | `aiter_dsv3_router_gemm` (`:164-165`); `if _use_aiter: logits = aiter_dsv3_router_gemm(...)` (`:354-355`) |

`_use_aiter_gfx95` (mxfp4 gating) is reused as a name in
`models/deepseek_v2.py:149` and `layers/communicator.py:86`
(`_is_gfx95_supported = is_gfx95_supported()`).

Env vars (`python/sglang/srt/environ.py:322-324`):
```python
# AMD & ROCm
SGLANG_USE_AITER = EnvBool(False)
SGLANG_USE_AITER_UNIFIED_ATTN = EnvBool(False)
```

## ROCm-only wrappers (replace NV paths)

| AMD wrapper | what it does (with code) | replaces (NV) |
| ----------- | ------------------------ | ------------- |
| `layers/rocm_linear_utils.py` | `from aiter.tuned_gemm import tgemm` (`:4`); `aiter_dsv3_router_gemm(hs, w)` → `tgemm.mm(hs, w.detach(), otype=hs.dtype)` (`:9-14`). Also reexports `fused_qk_rope_cat`, `fused_qk_rope_cat_and_cache_mla` (`:2-3, :6`). | NV cutlass router GEMM |
| `layers/moe/rocm_moe_utils.py` | `from aiter.fused_moe_bf16_asm import asm_moe_tkw1` (`:45`); `rocm_aiter_asm_moe_tkw1(...)` calls into it; custom op (`:27`). | vLLM rocm fused-moe |
| `layers/moe/moe_runner/aiter.py` | `@register_fused_func("none","aiter")` (`:49`); body `fused_moe(hs, w1, w2, topk_weight, topk_ids, quant_type=getattr(QuantType, quant_info.quant_type.value), activation=getattr(ActivationType, _AITER_ACTIVATIONS.get(activation,"Gelu")), w1_scale=..., w2_scale=...)` (`:75-93`). | NV MoE runner |
| `models/deepseek_common/attention_forward_methods/forward_mla_fused_rope_rocm.py` | One-off `_rocm.py` suffix wrapper (only HIP `*_rocm.py` file in tree). | NV fused-rope MLA forward |
| `layers/attention/aiter_backend.py` | `class AiterAttnBackend(AttentionBackend)` (`:112`). Imports `aiter.mla.{mla_decode_fwd, mla_prefill_fwd}` and `aiter.ops.triton.attention.unified_attention.unified_attention` (`:44-45`). | flashinfer / fa3 / fa4 |
| `layers/attention/nsa/tilelang_kernel.py` (HIP branch) | `dpsk_v4_fp8_attention_fwd` (`:2316`); on gfx95 uses `(block_I=64, threads=512, num_stages=0, block_per_cu=2, cu=256)`, else `(32, 128, 1, 1, 304)` (`:2337-2340`). Routed via `debug_flash_mla_adapter.py:38-39`. | `flash_mla.flash_mla_with_kvcache` (NV) |
| `layers/layernorm.py` | When `_use_aiter`: `from aiter import rmsnorm2d_fwd as rms_norm` (`:74`), `from aiter import rmsnorm2d_fwd_with_add as fused_add_rms_norm` (`:75`), `from aiter import layernorm2d_fwd as layer_norm` (`:73`). | NV layernorm kernels |

## gfx-arch-specific branches (`grep -rn "gfx9" python/sglang/srt`)

```python
# distributed/device_communicators/quick_all_reduce.py:28-38
def qr_rocm_arch_available():
    if not _is_hip:
        return False
    props = torch.cuda.get_device_properties(0)
    gcn_arch = getattr(props, "gcnArchName", "")
    supported_archs = ["gfx94", "gfx95"]
    return any(gfx in gcn_arch for gfx in supported_archs)
```

```python
# distributed/parallel_state.py:396-407  (gfx942+ quick-reduce hookup)
if is_hip():
    try:
        if qr_rocm_arch_available():
            self.qr_comm = QuickAllReduce(group=self.cpu_group, device=self.device)
    except Exception as e:
        logger.warning(f"Failed to initialize QuickAllReduce: {e}")
```

Other branch sites:
- `layers/communicator.py:86` (`_is_gfx95_supported = is_gfx95_supported()`).
  Used at `:538, 548, 583, 593` to pick fused `RMSNorm+mxfp4 quant` and
  fused `RMSNorm+fp8 group quant` paths from aiter
  (`fused_rms_mxfp4_quant`, `fused_rms_fp8_group_quant`).
- `layers/attention/aiter_backend.py:25` import; `:66-68` enables fp8
  prefill attn only on gfx95 (see code in attention_backend_dispatch.md).
- `layers/attention/nsa/tilelang_kernel.py:2337` — DSv4 kernel tile-shape
  branch (snippet above).
- `layers/attention/nsa/nsa_indexer.py:35` — same `_is_gfx95_supported` flag.

## ROCm profiler hooks

`managers/scheduler_profiler_mixin.py:163-189` — RPD profiler integration
gated by `"RPD" in activities`. Snippet inlined in
`profiler_endpoints_and_env.md`. Output filename:
`rpd-<ts>-TP-<rank>.trace.json.gz`.

## What is NOT replaced on AMD (still generic / triton)

- Sampler — `layers/sampler.py` is Triton.
- Quantization scale ops in `layers/quantization/fp8_kernel.py` — Triton.
- Compressor / indexer fall back to Triton when aiter envs unset;
  see `layers/attention/compressed/compressor.py`,
  `layers/attention/compressed/indexer.py`.

## Not investigated

- `csrc/` C++/HIP source (we only mapped Python).
- ROCm-specific build flags (`setup.py` / `sgl-kernel/` directories).
- `python/sglang/srt/lora/` ROCm variants (if any).
