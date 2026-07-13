# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared model-config helpers (config.json parsing + arch/type detection).

Leaf module: depends only on the standard library (and the stdlib-only
``hyperloom.common`` base) so both ``cli`` and the orchestrator executors can
import it without a circular dependency.
"""

from __future__ import annotations

import json
import logging
import re
import struct
from pathlib import Path

from hyperloom.common.coerce import to_int as _to_int


def _load_model_config_dict(model_path: str) -> dict | None:
    """Best-effort parse of ``<model_path>/config.json`` into a dict; returns ``None`` on any failure.

    Args:
        model_path: Filesystem path to the model directory.

    Returns:
        The parsed ``config.json`` dict, or ``None`` when the path is empty,
        the file is missing/unreadable, the JSON is invalid, or it is not a
        dict.
    """
    if not model_path:
        return None
    cfg_path = Path(model_path) / "config.json"
    try:
        raw = cfg_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        logging.warning("model_config_unreadable: %s (%s)", cfg_path, exc)
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logging.warning("model_config_invalid_json: %s (%s)", cfg_path, exc)
        return None
    if not isinstance(data, dict):
        logging.warning(
            "model_config_not_a_dict: %s (got %s)",
            cfg_path,
            type(data).__name__,
        )
        return None
    return data


def _config_architectures(config: dict) -> list[str]:
    """Normalise ``config["architectures"]`` to a list of non-empty strings (scalar wrapped; absent -> []).

    Args:
        config: A parsed model config dict.

    Returns:
        The non-empty architecture strings; a lone scalar is wrapped and a
        missing key yields ``[]``.
    """
    arches_raw = config.get("architectures")
    if isinstance(arches_raw, list):
        return [str(a).strip() for a in arches_raw if str(a or "").strip()]
    if isinstance(arches_raw, str) and arches_raw.strip():
        return [arches_raw.strip()]
    return []


# Gemma2 forward builds ``normalizer = torch.tensor(...)`` (a host scalar) on
# every call. The TraceLens kernel_shape_profiler patch activates inside the
# CUDA-graph capture critical section, so that host construct runs during HIP
# stream capture and raises ``hipErrorStreamCaptureUnsupported`` -> capture
# fails -> roofline produces no ceiling. Callers skip shape-discovery for
# Gemma2 to keep CUDA graph while avoiding the crash.
# Single source of truth: ``cli`` reuses these for its preflight checks too.
GEMMA2_MODEL_TYPE = "gemma2"
GEMMA2_ARCHITECTURES = frozenset({"gemma2forcausallm"})


# ``gemma`` then an optional single separator then ``2`` as a standalone token
# (start/separator on the left, separator/end on the right). Matches gemma2 /
# gemma-2 / gemma_2 but not gemma3, gemma25, or notgemma2.
_GEMMA2_PATH_RE = re.compile(r"(?:^|[-_.])gemma[-_.]?2(?:[-_.]|$)")


def _path_looks_like_gemma2(model_path: str) -> bool:
    """Heuristic Gemma2 detection from the path when config.json is absent.

    Word-boundary match on the directory name (gemma2 / gemma-2 / gemma_2),
    so a not-yet-materialized Hub-id style path still gets the workaround
    without false-positives on names like notgemma2 / gemma25.

    Args:
        model_path: Filesystem or Hub-id style path to the model.

    Returns:
        ``True`` when the path's final component looks like a Gemma2 name.
    """
    if not model_path:
        return False
    return _GEMMA2_PATH_RE.search(Path(model_path).name.lower()) is not None


def _config_gemma2_scopes(data: dict) -> list[dict]:
    """Return [top-level, text_config?] scopes for Gemma2 inspection.

    Args:
        data: A parsed model config dict.

    Returns:
        The top-level config plus its nested ``text_config`` when that is a
        dict.
    """
    scopes = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        scopes.append(nested)
    return scopes


def _config_is_gemma2(data: dict) -> bool:
    """True when a parsed config dict declares Gemma2 (top level or text_config).

    Args:
        data: A parsed model config dict.

    Returns:
        ``True`` when any scope declares the Gemma2 ``model_type`` or
        architecture.
    """
    for cfg in _config_gemma2_scopes(data):
        if str(cfg.get("model_type") or "").strip().lower() == GEMMA2_MODEL_TYPE:
            return True
        if any(a.lower() in GEMMA2_ARCHITECTURES for a in _config_architectures(cfg)):
            return True
    return False


def _config_has_model_identity(data: dict) -> bool:
    """True when the config carries any recognizable model_type/architectures.

    Used to decide whether a non-Gemma2 verdict is trustworthy: a config that
    clearly identifies another model (e.g. llama) must NOT fall back to the path
    heuristic, while an empty/residual config (``{}``, no model_type) should.

    Args:
        data: A parsed model config dict.

    Returns:
        ``True`` when any scope carries a non-empty ``model_type`` or
        ``architectures``.
    """
    for cfg in _config_gemma2_scopes(data):
        if str(cfg.get("model_type") or "").strip():
            return True
        if _config_architectures(cfg):
            return True
    return False


# Standard HF FP8 quant_method handled by sglang's Fp8LinearMethod. Only this
# loader honours SGLANG_USE_AITER_FP8_PER_TOKEN; compressed-tensors / other
# formats route through different methods and are intentionally excluded.
_FP8_QUANT_METHOD = "fp8"
# Sanity cap for the safetensors JSON header length (real headers are KB-MB);
# guards against a corrupt/garbage length prefix triggering a huge read.
_SAFETENSORS_HEADER_MAX_BYTES = 100 * 1024 * 1024


def _read_safetensors_header(path: Path) -> dict | None:
    """Parse the JSON header of a ``.safetensors`` file without loading tensor data.

    The safetensors layout is: 8-byte little-endian ``uint64`` header length,
    then that many bytes of UTF-8 JSON mapping tensor name -> ``{dtype, shape,
    data_offsets}``. Only the header is read.

    Args:
        path: Path to a ``.safetensors`` file.

    Returns:
        The parsed header dict, or ``None`` on any read/parse failure.
    """
    try:
        with path.open("rb") as fh:
            raw_len = fh.read(8)
            if len(raw_len) != 8:
                return None
            (header_len,) = struct.unpack("<Q", raw_len)
            if header_len <= 0 or header_len > _SAFETENSORS_HEADER_MAX_BYTES:
                return None
            header_bytes = fh.read(header_len)
            if len(header_bytes) != header_len:
                return None
        header = json.loads(header_bytes)
    except (OSError, ValueError, struct.error) as exc:
        logging.warning("safetensors_header_unreadable: %s (%s)", path, exc)
        return None
    return header if isinstance(header, dict) else None


def _fp8_weight_scale_is_per_channel(model_path: str) -> bool | None:
    """Classify a serialized FP8 checkpoint's weight-scale granularity.

    Reads the first ``*.weight_scale`` tensor found in the model's safetensors
    header(s) and classifies it by element count: a per-channel scale has one
    entry per output channel (numel > 1), while a per-tensor scale is a scalar
    (numel == 1). Granularity is uniform across a checkpoint, so the first
    weight-scale tensor is representative.

    Args:
        model_path: Filesystem path to the model directory.

    Returns:
        ``True`` for per-channel, ``False`` for per-tensor, or ``None`` when it
        cannot be determined (no readable safetensors / no weight-scale tensor).
    """
    if not model_path:
        return None
    files = sorted(Path(model_path).glob("*.safetensors"))
    if not files:
        return None
    for fpath in files:
        header = _read_safetensors_header(fpath)
        if not header:
            continue
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            # ``weight_scale_inv`` is the block-scale tensor (a different path
            # excluded upstream); only the per-channel/per-tensor ``weight_scale``
            # is relevant here.
            if "weight_scale" not in name or "weight_scale_inv" in name:
                continue
            shape = meta.get("shape")
            if not isinstance(shape, list):
                continue
            numel = 1
            for dim in shape:
                if isinstance(dim, int):
                    numel *= dim
            return numel > 1
    return None


def _fp8_is_per_channel_per_token(model_path: str) -> bool:
    """True when a serialized FP8 checkpoint uses per-channel weight + per-token (dynamic) activation.

    This is exactly the scheme that benefits from the aiter CK
    ``gemm_a8w8_bpreshuffle`` fast path in sglang's ``apply_fp8_linear``: with
    ``SGLANG_USE_AITER_FP8_PER_TOKEN=1`` the weights are converted to
    per-channel scales and dynamic activations use per-token scales, routing
    the GEMM to the fused CK kernel instead of the slow unfused
    ``_apply_fallback_scaled_mm``.

    Gated strictly so it is default-safe:

    * ``quantization_config.quant_method == "fp8"`` (the standard HF FP8 format
      that sglang's ``Fp8LinearMethod`` serves), AND
    * NO ``weight_block_size`` — block-scale FP8 takes the
      ``w8a8_block_fp8_linear`` path and is unaffected, AND
    * activation is dynamic (per-token). ``activation_scheme == "static"`` is a
      per-tensor activation scheme that takes the fused per-tensor path; an
      absent scheme defaults to dynamic in sglang's ``Fp8Config``, AND
    * the serialized weight scale is **per-channel**. e2e A/B on MI300X showed a
      per-tensor FP8 checkpoint already serves from the fast fused per-tensor
      ``torch._scaled_mm`` path, so forcing per-channel + bpreshuffle CK *regresses*
      it (~6% lower throughput / higher TPOT). Only per-channel weights hit the
      slow unfused fallback that the env actually rescues. Granularity is read
      from the safetensors header; an undeterminable checkpoint declines (safe).

    Args:
        model_path: Filesystem path to the model directory.

    Returns:
        ``True`` only for non-block FP8 checkpoints with dynamic activation
        whose serialized weight scale is confirmed per-channel.
    """
    data = _load_model_config_dict(model_path)
    if not isinstance(data, dict):
        return False
    qc = data.get("quantization_config")
    if not isinstance(qc, dict):
        return False
    if str(qc.get("quant_method") or "").strip().lower() != _FP8_QUANT_METHOD:
        return False
    # Block-scale FP8 is served by a different kernel path; never touch it.
    if qc.get("weight_block_size") is not None:
        return False
    # Only dynamic (per-token) activation hits the fast path; static is
    # per-tensor and would regress to the unfused fallback if forced.
    activation = str(qc.get("activation_scheme") or "").strip().lower()
    if activation not in ("", "dynamic"):
        return False
    # Per-tensor weight checkpoints already use the fast fused per-tensor path;
    # only confirmed per-channel weights benefit. Undeterminable -> decline.
    return _fp8_weight_scale_is_per_channel(model_path) is True


def _fp8_is_block_scale(model_path: str) -> bool:
    """True when a serialized FP8 checkpoint uses block-scale quantization.

    Block-scale FP8 is exactly the scheme the CK
    ``aiter_w8a8_block_fp8_linear`` / ``gemm_a8w8_blockscale`` fast path
    rewrites: the standard HF FP8 format (``quant_method == "fp8"``, served by
    sglang's ``Fp8LinearMethod``) that additionally declares a non-empty
    ``weight_block_size``. Per-tensor, static and per-channel/per-token FP8
    carry no ``weight_block_size`` and are intentionally excluded — they take
    other GEMM paths the block-scale switch must never touch.

    Args:
        model_path: Filesystem path to the model directory.

    Returns:
        ``True`` only for standard HF FP8 checkpoints that declare a non-empty
        ``weight_block_size``.
    """
    data = _load_model_config_dict(model_path)
    if not isinstance(data, dict):
        return False
    qc = data.get("quantization_config")
    if not isinstance(qc, dict):
        return False
    if str(qc.get("quant_method") or "").strip().lower() != _FP8_QUANT_METHOD:
        return False
    # A present-but-empty weight_block_size (``[]`` / ``0`` / ``None``) does not
    # select the block-scale kernel path; require a non-empty value.
    return bool(qc.get("weight_block_size"))


_MLA_KEYS = ("kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "q_lora_rank")
_MOE_EXPERT_KEYS = ("num_experts", "n_routed_experts", "num_local_experts")
_SHARED_EXPERT_KEYS = ("n_shared_experts", "num_shared_experts", "moe_num_shared_experts")
# Nested text-tower config keys used by multimodal wrappers (priority order).
_TEXT_SCOPE_KEYS = ("text_config", "llm_config", "language_config")
# Base-family tokens for derived/hybrid model_types (longest first).
_FAMILY_TOKENS = ("qwen3", "qwen2", "deepseek", "llama", "gemma", "mistral", "phi", "glm")


def _merge_config_scopes(data: dict) -> dict:
    """Flatten nested text-tower config(s) over the top level (nested wins).

    Multimodal wrappers describe the benchmarkable decoder under
    ``text_config`` / ``llm_config`` / ``language_config``. Merge every present
    scope in priority order so a stub high-priority scope is backfilled by a
    fuller lower-priority one (a field already set by a higher scope is kept).
    """
    merged = dict(data)
    seen: set[str] = set()
    for scope_key in _TEXT_SCOPE_KEYS:
        nested = data.get(scope_key)
        if not isinstance(nested, dict):
            continue
        for k, v in nested.items():
            if v in (None, ""):
                continue
            if k in seen:
                continue
            merged[k] = v
            seen.add(k)
    return merged


def _derive_attention_type(cfg: dict) -> str:
    """Infer attention variant (MLA/MQA/GQA/MHA) from head/lora config fields."""
    if any(cfg.get(k) for k in _MLA_KEYS):
        return "MLA"
    heads = _to_int(cfg.get("num_attention_heads")) or 0
    kv_raw = cfg.get("num_key_value_heads")
    kv = _to_int(kv_raw) if kv_raw is not None else heads
    kv = kv or 0
    if heads <= 0 or kv <= 0:
        return ""
    if kv == 1:
        return "MQA"
    if kv < heads:
        return "GQA"
    return "MHA"


def _derive_quantization(cfg: dict) -> str:
    """Return the weight quant method (e.g. ``fp8``) or '' when unquantized."""
    qc = cfg.get("quantization_config")
    if isinstance(qc, dict):
        return str(qc.get("quant_method") or "").strip()
    return ""


def _derive_model_family(model_type: str, model_path: str) -> str:
    """Infer the base model family with generation (e.g. qwen3, deepseek_v3).

    Collapses same-generation structural variants (moe / next / vl / text)
    into the generation key. Bare ``llama`` derives its generation from the
    path; unknown types fall back to a family prefix. Returns '' when unknown.
    Callers pass the merged (nested-wins) model_type so wrappers already
    resolve to the underlying decoder.
    """
    mt = str(model_type or "").strip().lower()
    name = Path(model_path or "").name.lower()

    # DeepSeek: keep major version (v32 -> v3). Check v3 before v2 since
    # 'deepseek_v32' contains both substrings.
    if mt.startswith("deepseek"):
        if "v4" in mt:
            return "deepseek_v4"
        if "v3" in mt or mt == "deepseek":
            return "deepseek_v3"
        if "v2" in mt:
            return "deepseek_v2"
        return "deepseek"
    # Qwen: collapse the generation's variants.
    if mt.startswith("qwen"):
        if mt.startswith("qwen3"):
            return "qwen3"
        if mt.startswith("qwen2"):
            return "qwen2"
        if mt.startswith("qwen1") or "qwen1.5" in name:
            return "qwen1.5"
        return "qwen"
    # Gemma generations.
    for gen in ("gemma4", "gemma3", "gemma2"):
        if mt.startswith(gen):
            return gen
    if mt == "gemma":
        return "gemma"
    # Mistral vs Mixtral kept distinct.
    if mt.startswith("mixtral"):
        return "mixtral"
    if mt.startswith("mistral"):
        return "mistral"
    # Llama: model_type is bare 'llama'; derive generation from name.
    if mt == "llama" or mt.startswith("llama"):
        if mt == "llama4" or "llama-4" in name or "llama4" in name:
            return "llama4"
        if "llama-3" in name or "llama3" in name or "llama_3" in name:
            return "llama3"
        if "llama-2" in name or "llama2" in name or "llama_2" in name:
            return "llama2"
        return "llama"
    # MiniMax / Nemotron / InternVL families collapse sub-variants.
    if mt.startswith("minimax"):
        return "minimax"
    if mt.startswith("nemotron"):
        return "nemotron"
    if mt.startswith("internvl"):
        return "internvl"
    if mt.startswith("glm"):
        return "glm4" if "4" in mt else "glm"
    if mt.startswith("phi"):
        return "phi3" if mt.startswith("phi3") else "phi"
    if not mt:
        return ""
    # Derived/hybrid types (rwkv6qwen2, llava_qwen2, hybrid_qwen3): map to the
    # base family token embedded in the model_type when present.
    for tok in _FAMILY_TOKENS:
        if tok in mt:
            return tok
    # Generic fallback: family prefix before the first separator.
    return mt.split("_")[0]


def summarize_model_config(model_path: str) -> dict:
    """Best-effort structured summary of a model's ``config.json`` ({} on failure).

    Reads core shape/quant fields plus inferred ``attention_type`` and
    ``is_moe`` so session state can carry model basics without a framework.
    """
    data = _load_model_config_dict(model_path)
    if data is None:
        return {}
    cfg = _merge_config_scopes(data)
    out: dict = {}

    # Prefer the merged (nested text-tower wins) model_type so multimodal
    # wrappers report the real decoder rather than the wrapper shell.
    model_type = str(cfg.get("model_type") or data.get("model_type") or "").strip()
    if model_type:
        out["model_type"] = model_type
    arches = _config_architectures(data) or _config_architectures(cfg)
    if arches:
        out["architectures"] = arches

    family = _derive_model_family(model_type, model_path)
    if family:
        out["model_family"] = family

    heads = _to_int(cfg.get("num_attention_heads")) or 0
    kv_raw = cfg.get("num_key_value_heads")
    kv = (_to_int(kv_raw) if kv_raw is not None else heads) or 0
    head_dim = _to_int(cfg.get("head_dim"))
    hidden = _to_int(cfg.get("hidden_size"))
    if not head_dim and hidden and heads:
        head_dim = hidden // heads

    attn = _derive_attention_type(cfg)
    if attn:
        out["attention_type"] = attn
    if heads:
        out["num_attention_heads"] = heads
    if kv:
        out["num_key_value_heads"] = kv
    if head_dim:
        out["head_dim"] = head_dim

    for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size", "max_position_embeddings"):
        val = _to_int(cfg.get(key))
        if val is not None:
            out[key] = val

    num_experts = 0
    for k in _MOE_EXPERT_KEYS:
        ne = _to_int(cfg.get(k))
        if ne:
            num_experts = ne
            break
    experts_per_tok = _to_int(cfg.get("num_experts_per_tok")) or 0
    out["is_moe"] = num_experts > 0
    if num_experts > 0:
        out["num_experts"] = num_experts
    if experts_per_tok > 0:
        out["num_experts_per_tok"] = experts_per_tok

    # Shared-expert detection: only emit when is_moe is also true to avoid
    # false positives on non-MoE models that happen to carry shared-looking keys.
    if out["is_moe"]:
        num_shared = 0
        for k in _SHARED_EXPERT_KEYS:
            ns = _to_int(cfg.get(k))
            if ns:
                num_shared = ns
                break
        shared_evidence = (
            num_shared > 0
            or bool(cfg.get("shared_expert_intermediate_size"))
            or bool(cfg.get("shared_experts"))
        )
        if shared_evidence:
            out["has_shared_expert"] = True
            if num_shared > 0:
                out["num_shared_experts"] = num_shared

    quant = _derive_quantization(cfg)
    if quant:
        out["quantization"] = quant
    for key in ("torch_dtype", "kv_cache_dtype"):
        val = str(cfg.get(key) or "").strip()
        if val:
            out[key] = val
    return out


def _model_is_gemma2(model_path: str) -> bool:
    """Best-effort detect a Gemma2 model from config.json (top level or text_config).

    Falls back to a path heuristic when config.json is missing/unreadable OR
    present-but-unidentifiable (empty dict / no model_type/architectures). A
    config that clearly identifies a non-Gemma2 model is trusted as-is.

    Args:
        model_path: Filesystem path to the model directory.

    Returns:
        ``True`` when the model is detected as Gemma2 via config.json or the
        path heuristic.
    """
    data = _load_model_config_dict(model_path)
    if data is not None:
        if _config_is_gemma2(data):
            return True
        if _config_has_model_identity(data):
            return False
    return _path_looks_like_gemma2(model_path)
