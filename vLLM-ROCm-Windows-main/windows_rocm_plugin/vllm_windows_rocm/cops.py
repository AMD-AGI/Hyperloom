# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Fallback implementations of the vLLM `torch.ops._C.*` fused ops, for the no-native-kernel
(VLLM_TARGET_DEVICE=empty) Windows build.

vLLM's CustomOp layers bind `torch.ops._C.<op>` in __init__/forward_cuda (e.g.
SiluAndMul.__init__ does `self.op = torch.ops._C.silu_and_mul` unconditionally on cuda-alike),
so on an empty build they crash before any forward_native fallback. We register the `_C`
op namespace here with correct schemas (mirrored from csrc/torch_bindings.cpp) and torch-native
implementations, so the model runs end-to-end.

These are CORRECTNESS-first (torch ops). Phase 2 replaces the hot ones with fused Triton/HIP
kernels for speed — same op names, so no vLLM changes needed.

Native-kernel handoff: if the compiled `vllm_win_C` library (built from vLLM's own csrc via
experiments/vllm_c_ext/) is present, we load it FIRST. Its TORCH_LIBRARY(_C) registrations
(silu_and_mul/rms_norm/fused_add_rms_norm/rotary_embedding, real HIP kernels) then take over,
and the fallbacks below are installed only for ops the native lib does NOT provide (e.g.
weak_ref_tensor). Set VLLM_WIN_C_DIR to override the build dir; set VLLM_WIN_C_NATIVE=0 to
force pure fallbacks (for A/B measurement).
"""
import glob
import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)

_INSTALLED = False


def _default_dir(name: str) -> str:
    """Default home of one native-extension build dir: <build root>/<name>, resolved with
    the same logic (VLLM_WIN_BUILD_ROOT, default C:/AI/build, else drive root). Kept in
    tools/winrocm_paths.py; duplicated here inline to avoid importing torch-adjacent tools
    at plugin import time."""
    root = (os.environ.get("VLLM_WIN_BUILD_ROOT") or "").strip()
    if not root:
        for cand in ("C:\\AI\\build", "C:\\"):
            if os.path.isdir(cand):
                root = cand
                break
        root = root or (os.environ.get("LOCALAPPDATA") or os.getcwd())
    return os.path.join(root, name)


def _default_native_dir() -> str:
    return _default_dir("vw_cext_build")


_NATIVE_DIR = os.environ.get("VLLM_WIN_C_DIR") or _default_native_dir()


def _hip_root() -> str:
    """HIP SDK / ROCm install root; honours HIP_PATH first, then probes the usual locations."""
    for var in ("HIP_PATH", "ROCM_PATH", "ROCM_HOME"):
        v = (os.environ.get(var) or "").strip().rstrip("\\/")
        if v and os.path.isdir(v):
            return v
    cands = [r"C:\HIP-SDK"] + sorted(glob.glob(r"C:\Program Files\AMD\ROCm\*"), reverse=True)
    for c in cands:
        if os.path.isdir(os.path.join(c, "bin")):
            return c
    return r"C:\HIP-SDK"


def _load_native() -> str | None:
    """Load the compiled vLLM _C kernels (vllm_win_C.pyd) so its TORCH_LIBRARY(_C) wins."""
    if os.environ.get("VLLM_WIN_C_NATIVE", "1") == "0":
        return None
    cands = sorted(glob.glob(os.path.join(_NATIVE_DIR, "vllm_win_C*.pyd")))
    for p in cands:
        # torch.ops.load_library binds the TORCH_LIBRARY static initializers without needing
        # a Python import; dependent DLLs (c10/torch_hip/amdhip64) are already in-process.
        try:
            torch.ops.load_library(p)
            return p
        except Exception as e:
            logger.warning("native _C load_library warning: %r", e)
            # fallback: import as a module (adds nothing extra, but uses Python's loader path)
            try:
                d = os.path.dirname(p)
                if d not in sys.path:
                    sys.path.insert(0, d)
                import importlib
                importlib.import_module(os.path.splitext(os.path.basename(p))[0])
                return p
            except Exception as e2:
                logger.warning("native _C import warning: %r", e2)
    logger.warning("no vllm_win_C*.pyd in %s: running with the SLOW torch fallbacks "
                   "(build it with experiments/vllm_c_ext/build_run.bat, or point "
                   "VLLM_WIN_C_DIR at your build)", _NATIVE_DIR)
    return None


def _silu_and_mul(result: torch.Tensor, x: torch.Tensor) -> None:
    d = x.shape[-1] // 2
    result.copy_(torch.nn.functional.silu(x[..., :d]) * x[..., d:])


def _gelu_and_mul(result: torch.Tensor, x: torch.Tensor) -> None:
    d = x.shape[-1] // 2
    result.copy_(torch.nn.functional.gelu(x[..., :d]) * x[..., d:])


def _gelu_tanh_and_mul(result: torch.Tensor, x: torch.Tensor) -> None:
    # gemma uses GeGLU with the tanh GELU approximation.
    d = x.shape[-1] // 2
    result.copy_(torch.nn.functional.gelu(x[..., :d], approximate="tanh") * x[..., d:])


def _rms_norm(result: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    orig = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    result.copy_(xf.to(orig) * weight)


def _fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                        eps: float) -> None:
    orig = x.dtype
    added = x + residual
    xf = added.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    out = xf.to(orig) * weight
    x.copy_(out)
    residual.copy_(added)


def _weak_ref_tensor(x: torch.Tensor) -> torch.Tensor:
    # vLLM's CUDA-graph path uses this to hold output buffers without bumping the storage
    # refcount. A view aliasing the same storage is a correct (if slightly less weak) fallback.
    return x.view(x.shape)


def _rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox) -> None:
    # Reuse vLLM's own native RoPE math (handles neox/gptj + partial rotary correctly).
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    rotary_dim = cos_sin_cache.shape[-1]
    q2, k2 = RotaryEmbedding.forward_static(
        positions, query, key, head_size, rotary_dim, cos_sin_cache, is_neox
    )
    query.copy_(q2)
    if key is not None and k2 is not None:
        key.copy_(k2)


def _apply_repetition_penalties(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    penalties = repetition_penalties.unsqueeze(1).expand_as(logits)
    penalties = torch.where(prompt_mask | output_mask, penalties, 1.0)
    logits.mul_(torch.where(logits > 0, 1.0 / penalties, penalties))


def _moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids,
                          experts_ids, num_tokens_post_pad, maybe_expert_map):
    # Torch fallback for torch.ops._moe_C.moe_align_block_size (no _moe_C on Windows).
    # Groups the flattened top-k slot indices by expert, pads each expert's run up to a
    # multiple of block_size, and writes: sorted_token_ids (slot indices, padding=numel
    # sentinel), experts_ids (expert per block, -1 past the end), num_tokens_post_pad.
    # CUDAGRAPH-SAFE: fixed output shapes, no .item()/host sync, no data-dependent sizes,
    # so the FULL_DECODE_ONLY graph can capture the MoE decode path.
    device = topk_ids.device
    flat = topk_ids.reshape(-1).to(torch.long)
    numel = flat.numel()
    sorted_token_ids.fill_(numel)  # sentinel = num valid slots (masked by the gemm kernel)
    counts = torch.zeros(num_experts, dtype=torch.long, device=device)
    counts.scatter_add_(0, flat.clamp(0, num_experts - 1), torch.ones_like(flat))
    padded = ((counts + block_size - 1) // block_size) * block_size
    offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(padded, 0)
    # scatter each slot to its padded destination (fully vectorized, fixed-size)
    order = torch.argsort(flat, stable=True)            # slot indices grouped by expert
    sorted_experts = flat[order]
    cnt_offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=device)
    cnt_offsets[1:] = torch.cumsum(counts, 0)
    ranks = torch.arange(numel, device=device) - cnt_offsets[sorted_experts]
    dst = offsets[sorted_experts] + ranks
    sorted_token_ids[dst] = order.to(sorted_token_ids.dtype)
    # expert id per block over the FULL preallocated experts_ids (fixed size); -1 past total
    nb = experts_ids.shape[0]
    block_starts = torch.arange(nb, device=device) * block_size
    total = offsets[num_experts]
    eids = (torch.searchsorted(offsets, block_starts, right=True) - 1).clamp(0, num_experts - 1)
    eids = torch.where(block_starts < total, eids, torch.full_like(eids, -1))
    if maybe_expert_map is not None:
        mapped = maybe_expert_map[eids.clamp(min=0)]
        eids = torch.where(eids >= 0, mapped, torch.full_like(eids, -1))
    experts_ids.copy_(eids.to(experts_ids.dtype))
    num_tokens_post_pad.copy_(total.reshape(num_tokens_post_pad.shape).to(num_tokens_post_pad.dtype))


def _moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    # output[T, H] = sum over the top_k dim of input[T, top_k, H] (fp32 accumulate).
    output.copy_(input.to(torch.float32).sum(dim=1).to(output.dtype))


def _install_moe_C() -> None:
    """Load native `_moe_C` (built from csrc/moe via build_moe_c.py) if present; otherwise
    register torch fallbacks for the fused-MoE ops the standard fused_experts path needs."""
    # Native first: its TORCH_LIBRARY(_moe_C) (moe_align_block_size/moe_sum/topk_softmax/
    # batched_moe_align_block_size, real HIP kernels) wins over the torch fallbacks below.
    if os.environ.get("VLLM_WIN_MOE_NATIVE", "1") != "0":
        moe_dir = os.environ.get("VLLM_WIN_MOE_DIR") or _default_dir("vw_moe_build")
        for p in sorted(glob.glob(os.path.join(moe_dir, "vllm_win_moe_C*.pyd"))):
            try:
                torch.ops.load_library(p)
                logger.info("loaded native _moe_C from %s", p)
                break
            except Exception as e:  # noqa: BLE001
                logger.warning("native _moe_C load warning: %r", e)
    if hasattr(torch.ops, "_moe_C") and hasattr(torch.ops._moe_C, "moe_align_block_size"):
        return  # native present -> use it, skip torch fallbacks
    if os.environ.get("VLLM_WIN_MOE_NATIVE", "1") != "0":
        logger.warning("no vllm_win_moe_C*.pyd found: MoE models need its topk_softmax and the "
                       "torch fallbacks below cover only moe_align/moe_sum "
                       "(build: experiments/vllm_c_ext/build_moe_run.bat)")
    lib = torch.library.Library("_moe_C", "FRAGMENT")
    ops = [
        ("moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, "
         "Tensor(a!) sorted_token_ids, Tensor(b!) experts_ids, Tensor(c!) num_tokens_post_pad, "
         "Tensor? maybe_expert_map) -> ()", _moe_align_block_size),
        ("moe_sum(Tensor input, Tensor(a!) output) -> ()", _moe_sum),
    ]
    for schema, fn in ops:
        name = schema.split("(", 1)[0]
        try:
            lib.define(schema)
        except Exception:
            pass
        for key in ("CUDA", "CPU"):
            try:
                lib.impl(name, fn, key)
            except Exception:
                pass
    globals()["_MOE_LIB"] = lib


def _unsupported_op(*args, **kwargs):
    raise NotImplementedError(
        "this fused fp8/fp4 _C op is not implemented on the Windows ROCm build. It is "
        "registered only as a stub so vLLM's inductor fusion-pass matchers (which build "
        "torch.ops._C.<op>.default dicts at import time) can load. It is never called in "
        "eager / CompilationMode.NONE."
    )


# Stub schemas (mirrored from csrc/torch_bindings.cpp) for the fused fp8/fp4 quant ops that
# vLLM's fusion-pass matchers reference at module-import time (act_quant_fusion / rms_quant_fusion
# / qk_norm_rope_fusion via matcher_utils, imported for any cuda_alike platform incl. ROCm).
# Defining them makes `torch.ops._C.<op>.default` resolvable so model init (which imports the
# pass manager) succeeds even though no native _C kernels exist. The impl raises if ever called.
_STUB_OPS = [
    "static_scaled_fp8_quant(Tensor(a!) result, Tensor input, Tensor scale, (int, int)? group_shape=None) -> ()",
    "dynamic_scaled_fp8_quant(Tensor(a!) result, Tensor input, Tensor(b!) scale) -> ()",
    "dynamic_per_token_scaled_fp8_quant(Tensor(a!) result, Tensor input, Tensor(b!) scale, Tensor? scale_ub) -> ()",
    "scaled_fp4_quant(Tensor input, Tensor input_scale, bool is_sf_swizzled_layout) -> (Tensor, Tensor)",
    # vLLM's _custom_ops.py register_fake's BOTH scaled_fp4_quant and its .out overload (guarded
    # by hasattr(_C, "scaled_fp4_quant")); defining the base flips that guard True, so the .out
    # overload must exist too.
    ("scaled_fp4_quant.out(Tensor input, Tensor input_scale, bool is_sf_swizzled_layout, *, "
     "Tensor(a!) output, Tensor(b!) output_scale) -> ()"),
    ("per_token_group_fp8_quant(Tensor input, Tensor(a!) output_q, Tensor(b!) output_s, int group_size, "
     "float eps, float fp8_min, float fp8_max, bool scale_ue8m0, bool dummy_is_scale_transposed, "
     "bool dummy_is_tma_aligned) -> ()"),
    "rms_norm_static_fp8_quant(Tensor(a!) result, Tensor input, Tensor weight, Tensor scale, float epsilon) -> ()",
    ("fused_add_rms_norm_static_fp8_quant(Tensor(a!) result, Tensor input, Tensor(b!) residual, Tensor weight, "
     "Tensor scale, float epsilon) -> ()"),
    ("rms_norm_dynamic_per_token_quant(Tensor(a!) result, Tensor input, Tensor weight, Tensor(b!) scale, "
     "float epsilon, Tensor? scale_ub, Tensor(c!)? residual) -> ()"),
    ("rms_norm_per_block_quant(Tensor(a!) result, Tensor input, Tensor weight, Tensor(b!) scale, float epsilon, "
     "Tensor? scale_ub, Tensor(c!)? residual, int group_size, bool is_scale_transposed) -> ()"),
    ("fused_qk_norm_rope(Tensor(a!) qkv, int num_heads_q, int num_heads_k, int num_heads_v, int head_dim, "
     "float eps, Tensor q_weight, Tensor k_weight, Tensor cos_sin_cache, bool is_neox, Tensor position_ids) -> ()"),
    "silu_and_mul_quant(Tensor(a!) result, Tensor input, Tensor scale) -> ()",
    ("silu_and_mul_nvfp4_quant(Tensor(a!) result, Tensor(b!) result_block_scale, Tensor input, "
     "Tensor input_global_scale) -> ()"),
]


_OPS = [
    ("silu_and_mul(Tensor(a!) result, Tensor input) -> ()", _silu_and_mul),
    ("gelu_and_mul(Tensor(a!) result, Tensor input) -> ()", _gelu_and_mul),
    ("gelu_tanh_and_mul(Tensor(a!) result, Tensor input) -> ()", _gelu_tanh_and_mul),
    ("rms_norm(Tensor(a!) result, Tensor input, Tensor weight, float epsilon) -> ()", _rms_norm),
    ("fused_add_rms_norm(Tensor(a!) input, Tensor(b!) residual, Tensor weight, float epsilon) -> ()",
     _fused_add_rms_norm),
    ("rotary_embedding(Tensor positions, Tensor(a!) query, Tensor(b!)? key, int head_size, "
     "Tensor cos_sin_cache, bool is_neox) -> ()", _rotary_embedding),
    ("apply_repetition_penalties_(Tensor(a!) logits, Tensor prompt_mask, Tensor output_mask, "
     "Tensor repetition_penalties) -> ()", _apply_repetition_penalties),
    ("weak_ref_tensor(Tensor(a) input) -> Tensor(a)", _weak_ref_tensor),
]


def _install_cache_C() -> None:
    """Load native `_C_cache_ops` (built from csrc/cache_kernels.cu via build_cache_c.py) so the
    sliding-layer KV write goes through the HIP reshape_and_cache_flash instead of the Triton fallback."""
    # Default OFF pending a CLEAN A/B: native reshape_and_cache_flash is built + correct, but the only
    # measurements so far (native ON 39.9 vs OFF 41.5) were on a DEGRADED GPU (decode had dropped from
    # the real 56.8 baseline with VRAM free -- clock/thermal from process churn, not spill), so the ~2
    # tok/s gap is inconclusive. Re-measure after a GPU/driver reset before flipping this on.
    if os.environ.get("VLLM_WIN_CACHE_NATIVE", "0") != "1":
        return
    cache_dir = os.environ.get("VLLM_WIN_CACHE_DIR") or _default_dir("vw_cache_build")
    for p in sorted(glob.glob(os.path.join(cache_dir, "vllm_win_cache_C*.pyd"))):
        try:
            torch.ops.load_library(p)
            if (hasattr(torch.ops, "_C_cache_ops")
                    and hasattr(torch.ops._C_cache_ops, "reshape_and_cache_flash")):
                logger.info("loaded native _C_cache_ops (reshape_and_cache_flash) from %s", p)
                return
        except Exception as e:  # noqa: BLE001
            logger.warning("native _C_cache_ops load warning: %r", e)


_S5_NATIVE_CALLS = [0]


def _patch_rocm_decode() -> None:
    """Monkeypatch chunked_prefill_paged_decode so the pure-DECODE path for head_size==256 (gemma's
    fp16 sliding layers) calls our native torch.ops._C.paged_attention_v1 (S5, ~3.2x vs the Triton
    kernel_paged_attention_2d it replaces) instead of the Triton decode kernel. Reads the SAME v0 paged
    KV cache (written by ROCM_ATTN's reshape_and_cache v0), and native's sliding mask
    ((seq_len-1-token) >= sw) matches kpa2d's (context_len-seq_offset) < SLIDING_WINDOW exactly. Prefill
    (max_query_len>1), non-256 heads, fp8/alibi/sinks/block-table-ptr all fall through to the original."""
    try:
        import vllm.v1.attention.ops.chunked_prefill_paged_decode as _mod
    except Exception as e:  # noqa: BLE001  (never break plugin init over the patch)
        logger.warning("S5 patch skipped (import failed): %r", e)
        return
    _orig = _mod.chunked_prefill_paged_decode

    def _wrapper(query, key, value, output, kv_cache_dtype, key_cache, value_cache, block_table,
                 query_start_loc, seq_lens, max_seq_len, max_query_len, k_scale, v_scale,
                 alibi_slopes=None, sliding_window=None, sm_scale=None, output_scale=None,
                 sinks=None, is_block_table_ptr=False):
        head_size = query.shape[2]
        # native paged_attention_v1 instantiates head sizes up to 256 (incl. 128, 256). gemma sliding
        # layers are 256; ERNIE-4.5 (and most llama/qwen) are 128 -- both fire the native decode swap.
        if (max_query_len == 1 and head_size in (128, 256) and kv_cache_dtype == "auto"
                and alibi_slopes is None and sinks is None and not is_block_table_ptr
                and output_scale is None and key_cache.dim() == 5):
            scale = sm_scale if sm_scale is not None else 1.0 / (head_size ** 0.5)
            sw = 0 if (sliding_window is None or sliding_window <= 0) else int(sliding_window)
            num_kv_heads = key_cache.shape[1]
            block_size = value_cache.shape[3]
            bt = block_table if block_table.dtype == torch.int32 else block_table.to(torch.int32)
            torch.ops._C.paged_attention_v1(
                output, query, key_cache, value_cache, num_kv_heads, scale,
                bt, seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype,
                k_scale, v_scale, 0, 0, 0, 0, 0, sw)
            _S5_NATIVE_CALLS[0] += 1
            return
        return _orig(query, key, value, output, kv_cache_dtype, key_cache, value_cache, block_table,
                     query_start_loc, seq_lens, max_seq_len, max_query_len, k_scale, v_scale,
                     alibi_slopes, sliding_window, sm_scale, output_scale, sinks, is_block_table_ptr)

    _mod.chunked_prefill_paged_decode = _wrapper
    # Disable the ROCm "custom paged attention" gate: on gfx1x it PASSES for head_size==128 (ERNIE,
    # llama, qwen) and makes chunked_prefill_paged_decode call ops.paged_attention_rocm ==
    # torch.ops._rocm_C.paged_attention -- the gfx9 wave64 MFMA kernel we do NOT have (our _rocm_C has
    # only wvSplitK/LLMM1). That decode-part runs even during PREFILL (filtered per-seq), so it crashes
    # before our wrapper's pure-decode native swap ever matters. Forcing it False routes the non-native
    # decode-part to the Triton kernel_paged_attention_2d (reads the same v0 cache), which exists; our
    # wrapper still swaps the pure-decode (max_query_len==1) to native paged_attention_v1.
    try:
        import vllm.platforms.rocm as _rp
        _rp.use_rocm_custom_paged_attention = lambda *a, **k: False
    except Exception as e:  # noqa: BLE001
        logger.warning("S5 use_rocm_custom disable warning: %r", e)
    # Do NOT import vllm.v1.attention.backends.rocm_attn here: install() can run at plugin/sitecustomize
    # time (before vllm is fully up), and importing the backends module then triggers a circular import
    # that aborts plugin registration (and with it the WNA16 fallback). rocm_attn.py does
    # `from ...chunked_prefill_paged_decode import chunked_prefill_paged_decode` at ITS import time, which
    # is later than this patch (backend is built during engine init) -> it binds the patched _wrapper.
    import sys as _sys
    _ra = _sys.modules.get("vllm.v1.attention.backends.rocm_attn")
    if _ra is not None and hasattr(_ra, "chunked_prefill_paged_decode"):
        _ra.chunked_prefill_paged_decode = _wrapper  # already-imported: rebind in place (no import)
    logger.info("S5 native paged_attention_v1 patched into ROCM_ATTN decode (head_size 128/256)"
          " + use_rocm_custom_paged_attention disabled")


def _install_attn_C() -> None:
    """Load native paged_attention_v1/v2 (built from csrc/attention via build_attn_c.py, registered as
    TORCH_LIBRARY_FRAGMENT(_C) so it coexists with vllm_win_C) and patch the ROCM_ATTN decode path (S5).
    Opt-in: needs VLLM_WIN_ATTN_NATIVE=1 AND ROCM_ATTN backend + kv_cache_dtype=auto for the sliding layers."""
    if os.environ.get("VLLM_WIN_ATTN_NATIVE", "0") != "1":
        return
    attn_dir = os.environ.get("VLLM_WIN_ATTN_DIR") or _default_dir("vw_attn_build")
    for p in sorted(glob.glob(os.path.join(attn_dir, "vllm_win_attn_C*.pyd"))):
        try:
            torch.ops.load_library(p)
            if hasattr(torch.ops._C, "paged_attention_v1"):
                logger.info("loaded native _C.paged_attention_v1 from %s", p)
                # NOTE: do NOT monkeypatch here. install() runs during plugin register() while vllm is
                # mid-import; importing chunked_prefill_paged_decode now triggers a circular import that
                # aborts plugin registration (killing the WNA16 fallback). If the ops module is already
                # loaded (rare at install time) patch now; otherwise the harness/caller must invoke
                # maybe_patch_s5_decode() AFTER `from vllm import LLM` (see below).
                if "vllm.v1.attention.ops.chunked_prefill_paged_decode" in sys.modules:
                    _patch_rocm_decode()
                return
        except Exception as e:  # noqa: BLE001
            logger.warning("native _C attn load warning: %r", e)


def maybe_patch_s5_decode() -> None:
    """Apply the S5 decode monkeypatch. Call this AFTER `from vllm import LLM` and BEFORE LLM(...) so the
    ROCM_ATTN backend (imported during engine init) binds the patched chunked_prefill_paged_decode. Safe:
    no-op unless VLLM_WIN_ATTN_NATIVE=1 and the native op is loaded."""
    if os.environ.get("VLLM_WIN_ATTN_NATIVE", "0") != "1":
        return
    if not (hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "paged_attention_v1")):
        logger.warning("S5 patch skipped (native paged_attention_v1 not loaded)")
        return
    _patch_rocm_decode()


_FLASH_CALLS = [0]
_FLASH_ARGN = ["q", "k", "v", "out", "cu_seqlens_q", "max_seqlen_q", "seqused_k", "max_seqlen_k",
               "softmax_scale", "causal", "window_size", "block_table", "softcap",
               "q_descale", "k_descale", "v_descale"]

_CK_VARLEN = [None]   # loaded ck_fmha_varlen_C module
_CK_CALLS = [0]       # count of forward() calls routed to CK (diagnostics)
_CK_DBG = [0]         # one-shot reject-reason prints under VLLM_WIN_CK_DEBUG=1


def maybe_patch_flash_decode() -> None:
    """FLASH-KERNEL path (the real one): keep TRITON_ATTN's light fused path (rope+KV-write, metadata)
    and swap ONLY unified_attention for a native decode kernel that reads TRITON_ATTN's FLASH KV cache
    directly (no v0 repack, no ROCM_ATTN overhead that made the S5 ROCM_ATTN path regress). Pure decode
    only (max_seqlen_q==1); prefill/chunked fall through to the real unified_attention. Opt-in:
    VLLM_WIN_FLASH_ATTN=1. Call AFTER `from vllm import LLM`, BEFORE LLM(...)."""
    if os.environ.get("VLLM_WIN_FLASH_ATTN", "0") != "1":
        return
    d = os.environ.get("VLLM_WIN_FLASH_DIR") or _default_dir("vw_attnflash_build")
    loaded = hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "paged_attention_flash")
    if not loaded:
        for p in sorted(glob.glob(os.path.join(d, "vllm_win_attn_flash_C*.pyd"))):
            try:
                torch.ops.load_library(p)
                if hasattr(torch.ops._C, "paged_attention_flash"):
                    loaded = True
                    logger.info("loaded native _C.paged_attention_flash from %s", p)
                    break
            except Exception as e:  # noqa: BLE001
                logger.warning("flash load warning: %r", e)
    if not loaded:
        logger.warning("flash patch skipped (paged_attention_flash not loaded)")
        return
    try:
        import vllm.v1.attention.ops.triton_unified_attention as _mod
    except Exception as e:  # noqa: BLE001
        logger.warning("flash patch skipped (import failed): %r", e)
        return
    _orig = _mod.unified_attention

    def _wrap(*args, **kw):
        a = dict(zip(_FLASH_ARGN, args))
        a.update(kw)
        q, k, v, out = a["q"], a["k"], a["v"], a["out"]
        head = q.shape[2]
        # pure-decode fast path only; everything else -> real unified_attention
        if (a.get("max_seqlen_q") == 1 and k.dim() == 4 and head in (128, 256)
                and q.dtype in (torch.float16, torch.bfloat16)
                and k.dtype in (torch.float16, torch.bfloat16)
                and k.shape[1] == 16 and not a.get("softcap")
                and a.get("sinks") is None and a.get("alibi_slopes") is None):
            ws = a.get("window_size")
            sw = int(ws[0]) + 1 if (ws is not None and ws[0] is not None and ws[0] >= 0) else 0
            bt = a["block_table"]
            bt = bt if bt.dtype == torch.int32 else bt.to(torch.int32)
            torch.ops._C.paged_attention_flash(
                out, q, k, v, k.shape[2], float(a["softmax_scale"]), bt,
                a["seqused_k"], 16, int(a["max_seqlen_k"]), sw)
            _FLASH_CALLS[0] += 1
            return
        return _orig(*args, **kw)

    _mod.unified_attention = _wrap
    _ta = sys.modules.get("vllm.v1.attention.backends.triton_attn")
    if _ta is not None and hasattr(_ta, "unified_attention"):
        _ta.unified_attention = _wrap
    logger.info("flash decode patched into unified_attention (head 128/256, pure decode)")


def maybe_patch_ck_prefill() -> None:
    """CK ck_tile FMHA-varlen for PURE PREFILL -- the prefill/TTFT lever (compute-bound WMMA, the one axis
    RDNA3 single-stream decode can't move). Patches TritonAttentionImpl.forward: when the batch is a pure
    prefill with NO prior KV context (per-seq seq_len == query_len), head_size 128, decoder self-attn, no
    sliding-window/softcap/alibi/sinks, and fp16/bf16 -- routes the attention COMPUTE to the CK varlen
    kernel on the CONTIGUOUS key/value (reshape_and_cache runs in its own method, so the decode KV cache is
    unaffected and correct). Decode, mixed prefill+decode, sliding-window, head!=128, and any prior-context
    step all fall through to the stock unified_attention. Opt-in VLLM_WIN_CK_PREFILL=1; call AFTER
    `from vllm import LLM`, BEFORE LLM(...). See memory ck-fmha-gfx1100-windows."""
    if os.environ.get("VLLM_WIN_CK_PREFILL", "0") != "1":
        return
    if _CK_VARLEN[0] is None:
        d = os.environ.get("VLLM_WIN_CKVARLEN_DIR") or _default_dir("vw_ckvarlen_build")
        for hp in (os.path.join(_hip_root(), "bin"), os.path.join(_hip_root(), "lib"), d):
            try:
                os.add_dll_directory(hp)
            except Exception:  # noqa: BLE001
                pass
        if d not in sys.path:
            sys.path.insert(0, d)
        try:
            import ck_fmha_varlen_C as _ckmod
            _CK_VARLEN[0] = _ckmod
            logger.info("loaded CK varlen FMHA from %s", d)
        except Exception as e:  # noqa: BLE001
            logger.warning("CK prefill patch skipped (import failed): %r", e)
            return
    try:
        import vllm.v1.attention.backends.triton_attn as _mod
        _Impl = _mod.TritonAttentionImpl
    except Exception as e:  # noqa: BLE001
        logger.warning("CK prefill patch skipped (import failed): %r", e)
        return
    if getattr(_Impl.forward, "_ck_wrapped", False):
        return
    _orig = _Impl.forward
    _ck = _CK_VARLEN[0]

    def _forward(self, layer, query, key, value, kv_cache, attn_metadata,
                 output=None, output_scale=None, output_block_scale=None):
        m = attn_metadata
        try:
            ok = (m is not None and output is not None and key is not None and value is not None
                  and kv_cache is not None and output_scale is None
                  and getattr(m, "use_cascade", False) is False
                  and getattr(m, "max_query_len", 0) and m.max_query_len > 1
                  and m.max_query_len == getattr(m, "max_seq_len", -1)
                  and getattr(self, "sliding_window", (-1, -1)) == (-1, -1)
                  and not getattr(self, "logits_soft_cap", 0)
                  and getattr(self, "alibi_slopes", None) is None
                  and getattr(self, "sinks", None) is None
                  and query.dim() == 3 and query.shape[2] == 128
                  and query.dtype in (torch.float16, torch.bfloat16)
                  and not self.kv_cache_dtype.startswith("fp8"))
        except Exception:  # noqa: BLE001
            ok = False
        if (os.environ.get("VLLM_WIN_CK_DEBUG") == "1" and _CK_DBG[0] < 8
                and m is not None and getattr(m, "max_query_len", None) and m.max_query_len > 1):
            _CK_DBG[0] += 1
            try:
                print("vllm-win CK-call:", type(self).__name__, "ok=", ok,
                      "out_none=", output is None, "oscale=", (output_scale is not None), dict(
                    mq=getattr(m, "max_query_len", None), ms=getattr(m, "max_seq_len", None),
                    cascade=getattr(m, "use_cascade", None), sw=getattr(self, "sliding_window", None),
                    softcap=getattr(self, "logits_soft_cap", None),
                    alibi=(getattr(self, "alibi_slopes", None) is not None),
                    sinks=(getattr(self, "sinks", None) is not None),
                    out_scale=(output_scale is not None),
                    head=(query.shape[2] if query.dim() == 3 else query.shape),
                    dt=str(query.dtype), kv=getattr(self, "kv_cache_dtype", None)))
            except Exception:  # noqa: BLE001
                pass
        if ok:
            n = m.num_actual_tokens
            cu = m.query_start_loc
            nreq = int(cu.shape[0]) - 1
            # pure-prefill-no-context proof: single req w/ query_len==seq_len is guaranteed (no sync);
            # multi-req -> one GPU verify (prefill step only, off the decode hot path).
            safe = (nreq == 1)
            if not safe and nreq >= 1:
                qlens = cu[1:nreq + 1] - cu[:nreq]
                safe = bool((m.seq_lens[:nreq] == qlens).all().item())
            if safe:
                # k/v come from a fused-QKV split (GQA) -> often non-contiguous; enforce thd contiguity
                # (a small [tokens,H,D] copy, negligible vs the O(S^2) attention it feeds).
                q = query[:n].contiguous(); k = key[:n].contiguous(); v = value[:n].contiguous()
                Hq = q.shape[1]; D = q.shape[2]
                o_ck = torch.empty_like(q)                      # contiguous [n, Hq, D]
                cuq = cu if cu.dtype == torch.int32 else cu.to(torch.int32)
                _ck.ck_fmha_varlen(q, k, v, o_ck, cuq, cuq,
                                   int(m.max_query_len), int(m.max_seq_len), float(self.scale), True)
                output[:n].copy_(o_ck.reshape(output[:n].shape))
                _CK_CALLS[0] += 1
                return output
        return _orig(self, layer, query, key, value, kv_cache, attn_metadata,
                     output, output_scale, output_block_scale)

    _forward._ck_wrapped = True
    _Impl.forward = _forward
    logger.info("CK varlen FMHA patched into TritonAttentionImpl.forward (pure prefill, head 128, fp16/bf16)")


def maybe_patch_attn_2d_decode() -> None:
    """Force TRITON_ATTN decode through the 2D unified-attention kernel, skipping the 3D
    split-softmax kernel + reduce. A bisect switch for KV-cache problems that only show at long
    context (issue #25: fp8 KV collapses at 16k, prefill via the 2D kernel is fine, decode via the
    3D kernel is not): unified_attention takes the 2D kernel whenever num_seqs > seq_threshold_3D,
    and the 2D grid does not depend on seq_len, so cudagraph capture is unaffected. Opt-in
    VLLM_WIN_ATTN_2D_DECODE=1; applied from the platform's check_and_update_config."""
    if os.environ.get("VLLM_WIN_ATTN_2D_DECODE", "0") != "1":
        return
    try:
        import vllm.v1.attention.backends.triton_attn as _mod
        _B = _mod.TritonAttentionMetadataBuilder
    except Exception as e:  # noqa: BLE001
        logger.warning("2D-decode patch skipped (import failed): %r", e)
        return
    if getattr(_B.build, "_2d_wrapped", False):
        return
    _orig_build = _B.build

    def _build(self, *args, **kwargs):
        md = _orig_build(self, *args, **kwargs)
        md.seq_threshold_3D = -1
        return md

    _build._2d_wrapped = True
    _B.build = _build
    logger.info("TRITON_ATTN decode forced onto the 2D unified-attention kernel "
                "(VLLM_WIN_ATTN_2D_DECODE=1)")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    # Load the compiled native kernels first; their TORCH_LIBRARY(_C) ops win over fallbacks.
    native = _load_native()
    lib = torch.library.Library("_C", "FRAGMENT")
    for schema, fn in _OPS:
        name = schema.split("(", 1)[0]
        # Per-op guard: if the native lib already provides this op, don't shadow it (but DO
        # still register the others, e.g. weak_ref_tensor, which the native lib lacks).
        if hasattr(torch.ops, "_C") and hasattr(torch.ops._C, name):
            continue
        try:
            lib.define(schema)
        except Exception:
            pass  # already defined
        for key in ("CUDA", "CPU", "Meta"):
            try:
                lib.impl(name, fn, key)
            except Exception:
                pass
    # Stub-register the fused fp8/fp4 quant ops so the fusion-pass matchers can import.
    for schema in _STUB_OPS:
        name = schema.split("(", 1)[0]
        if hasattr(torch.ops, "_C") and hasattr(torch.ops._C, name):
            continue
        try:
            lib.define(schema)
        except Exception as e:  # noqa: BLE001
            logger.warning("cops stub define warning: %s %r", name, e)
            continue
        # Only register CUDA: vLLM adds its own register_fake (Meta) for the tensor-returning
        # ops (e.g. scaled_fp4_quant); registering Meta here would collide. CUDA-only means an
        # accidental eager call raises our clear NotImplementedError instead of silent garbage.
        try:
            lib.impl(name, _unsupported_op, "CUDA")
        except Exception:
            pass
    # keep a ref so the Library isn't GC'd
    globals()["_LIB"] = lib
    _install_moe_C()  # fused-MoE ops (_moe_C namespace) torch fallbacks
    _install_cache_C()  # native reshape_and_cache_flash (_C_cache_ops), else Triton fallback
    _install_attn_C()  # S5: native paged_attention_v1 decode swap on ROCM_ATTN (head 256), opt-in
    _INSTALLED = True
    if native:
        present = [s.split("(", 1)[0] for s, _ in _OPS
                   if hasattr(torch.ops._C, s.split("(", 1)[0])]
        # logger, not print: this runs at import time in EVERY interpreter (also via
        # sitecustomize), including vLLM/py-cpuinfo subprocesses that write machine-readable
        # JSON to stdout. A bare print there corrupts their output (py-cpuinfo crashed with
        # JSONDecodeError on this exact mechanism).
        logger.info("native _C kernels loaded from %s | ops: %s", native, present)
