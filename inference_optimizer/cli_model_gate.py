# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Model-gate helpers for the CLI: arch / config loading + unsupported-model detection.

Extracted from ``cli.py`` (phase 4). Pure helpers that read a model's
config.json and decide whether it is a supported text-generation model on AMD.
Imports stdlib + cli_gpu only; must not import ``cli`` (one-way dependency).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .cli_gpu import _resolve_amd_gpu_type

log = logging.getLogger(__name__)

_SUPPORTED_ARCH_MARKERS = (
    "ForCausalLM",
    "LMHeadModel",
    "ForCausalLMWithValueHead",
)

_SUPPORTED_MODEL_TYPES = frozenset({
    "llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe",
    "gemma", "gemma2", "phi", "phi3", "phimoe",
    "starcoder2", "codellama", "deepseek_v2", "deepseek_v3",
    "falcon", "gpt_neox", "gpt2", "opt", "bloom",
    "internlm", "internlm2", "yi", "baichuan",
    "chatglm", "glm", "glm4",
    "command-r", "cohere", "cohere2", "dbrx",
    "mpt", "olmo", "olmo2", "jamba", "arctic",
    "exaone", "granite", "granitemoeshared",
    "stablelm", "persimmon",
})

_UNSUPPORTED_MODEL_TYPES = frozenset({
    "gemma3",
    "mllama",
    "llava",
    "llava_next",
    "qwen2_vl",
    "qwen2_5_vl",
    "idefics",
    "idefics2",
    "idefics3",
    "paligemma",
    "pixtral",
    "internvl_chat",
    "phi3_v",
})

_UNSUPPORTED_ARCHITECTURES = frozenset({
    # RWKV6/Qwen2 hybrid linear-attention arch: not in sglang's supported list
    # (only plain RwkvForCausalLM is), fails ModelConfig validation at boot.
    "RWKV6Qwen2ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "InternVLChatModel",
    "Phi3VForCausalLM",
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "MllamaForConditionalGeneration",
    "PaliGemmaForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Idefics2ForConditionalGeneration",
    "Idefics3ForConditionalGeneration",
    "PixtralForConditionalGeneration",
})

_UNSUPPORTED_CONFIG_KEYS = (
    "vision_config",
    "image_token_id",
    "image_token_index",
    "mm_config",
    "multi_modal_config",
    "vision_tower",
    "vision_tower_cfg",
    "image_processor_type",
    "projector_config",
    "mm_projector_type",
)

_TEXT_DECODER_CONFIG_KEYS = (
    "text_config",
    "language_config",
    "llm_config",
)

_VERDICT_TEXT_COERCIBLE = "text_coercible"

_VERDICT_VISION_ONLY = "vision_only"

_TEXT_COERCIBLE_MODEL_TYPES = frozenset({
    "kimi_k25",
    "qwen3_5_moe",
    # Gemma-4 ships a vision_config (Gemma4ForConditionalGeneration) but its
    # text decoder (text_config / gemma4_text) is a standard dense causal LM
    # that vLLM serves text-only (both Gemma4ForCausalLM and
    # Gemma4ForConditionalGeneration are registered). Text benchmarks never
    # exercise the vision tower, so route to the degraded text path.
    "gemma4",
})

_MAXPOS_CONFIG_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
)

_ROPE_CONFIG_KEYS = ("rope_scaling", "rope_parameters", "rope_theta")

_AMD_UNSUPPORTED_MODEL_TYPES = frozenset({"deepseek_v32"})

_AMD_UNSUPPORTED_ARCHITECTURES = frozenset({"deepseekv32forcausallm"})

_UNREGISTERED_CUSTOM_CONFIG_TYPES = frozenset({"kimi_k2"})

_PHI3_ROPE_TYPES = frozenset({"su", "longrope"})

_GEMMA2_ARCHITECTURES = frozenset({"gemma2forcausallm"})

_AMD_UNSUPPORTED_QUANT_ALGOS = frozenset({"nvfp4", "fp4"})

_AMD_UNSUPPORTED_QUANT_METHODS = frozenset({"bitsandbytes", "bnb"})

def _load_model_arch(workspace_root: Path, model_name: str) -> dict:
    """Best-effort loader for the advisory ``<workspace_root>/model_arch.json`` profile (prompts only).

    Soft-degrades to ``{}`` (never blocks launch) on missing/unreadable/invalid file. Stale-file guard:
    require ``data["model_name"]`` basename to match launched ``--model`` basename, else WARN + ``{}``.
    """
    arch_path = workspace_root / "model_arch.json"
    try:
        raw = arch_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logging.warning("model_arch_unreadable: %s (%s)", arch_path, exc)
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logging.warning("model_arch_invalid_json: %s (%s)", arch_path, exc)
        return {}
    if not isinstance(data, dict):
        logging.warning(
            "model_arch_not_a_dict: %s (got %s)", arch_path, type(data).__name__
        )
        return {}
    declared = str(data.get("model_name") or "").strip()
    if not declared:
        logging.warning(
            "model_arch_missing_model_name: %s (cannot verify freshness)", arch_path
        )
        return {}
    if Path(declared).name != Path(model_name).name:
        logging.warning(
            "model_arch_stale_or_mismatch: %s declares model_name=%r but "
            "launching %r — ignoring",
            arch_path,
            declared,
            model_name,
        )
        return {}
    return data

def _load_model_config_dict(model_path: str) -> dict | None:
    """Best-effort parse of ``<model_path>/config.json`` into a dict; returns ``None`` on any failure."""
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
            "model_config_not_a_dict: %s (got %s)", cfg_path, type(data).__name__,
        )
        return None
    return data

def _config_architectures(config: dict) -> list[str]:
    """Normalise ``config["architectures"]`` to a list of non-empty strings (scalar wrapped; absent -> [])."""
    arches_raw = config.get("architectures")
    if isinstance(arches_raw, list):
        return [str(a).strip() for a in arches_raw if str(a or "").strip()]
    if isinstance(arches_raw, str) and arches_raw.strip():
        return [arches_raw.strip()]
    return []

def _load_model_config_tags(model_path: str) -> dict:
    """Best-effort loader for KB architecture-identity tags (``architectures`` + ``model_type``) from config.json.

    Soft-degrades to ``{}`` (never blocks launch); normalised fields are omitted when empty so callers can .get().
    """
    data = _load_model_config_dict(model_path)
    if data is None:
        return {}
    out: dict = {}
    arches = _config_architectures(data)
    if arches:
        out["architectures"] = arches
    model_type = str(data.get("model_type") or "").strip()
    if model_type:
        out["model_type"] = model_type
    return out

def _arch_is_supported_text_generation(arch: str) -> bool:
    """True when an architecture class name denotes a supported text-generation
    (decoder-only causal LM) model."""
    a = (arch or "").strip()
    if not a:
        return False
    return any(marker in a for marker in _SUPPORTED_ARCH_MARKERS)

def _config_declares_text_decoder(config: dict, architectures: list[str], model_type_l: str) -> bool:
    """True when config positively identifies a usable text decoder.

    For multimodal wrapper configs the top-level architecture often names the
    wrapper (``*ForConditionalGeneration``), while the benchmarkable decoder is
    described under ``text_config`` / ``language_config``. Treat those nested
    text blocks as capability evidence instead of requiring a per-family
    allowlist entry.
    """
    if model_type_l in _TEXT_COERCIBLE_MODEL_TYPES:
        return True
    if any(_arch_is_supported_text_generation(a) for a in architectures):
        return True

    for key in _TEXT_DECODER_CONFIG_KEYS:
        nested = config.get(key)
        if not isinstance(nested, dict):
            continue
        nested_architectures = _config_architectures(nested)
        if any(_arch_is_supported_text_generation(a) for a in nested_architectures):
            return True

        nested_model_type = str(nested.get("model_type") or "").strip().lower()
        if (
            nested_model_type in _SUPPORTED_MODEL_TYPES
            or nested_model_type in _TEXT_COERCIBLE_MODEL_TYPES
        ):
            return True

        # Some multimodal configs (including newer family wrappers) expose a
        # text_config with decoder dimensions but do not use a model_type that
        # this package has seen yet. Because the evidence is scoped to an
        # explicitly named text block, this does not widen fallback for a
        # top-level mislabeled VLM.
        has_vocab = isinstance(nested.get("vocab_size"), int) and nested["vocab_size"] > 0
        has_decoder_shape = any(
            isinstance(nested.get(field), int) and nested[field] > 0
            for field in ("hidden_size", "num_hidden_layers", "intermediate_size")
        )
        if has_vocab and has_decoder_shape:
            return True

    return False

def _detect_unsupported_model(model_path: str) -> dict | None:
    """Best-effort classify a model's text-serving viability.

    Returns ``None`` for a plain text-generation model (and for an unreadable
    config.json — we don't hard-block on a config we cannot read). Otherwise
    returns ``{"architecture", "model_type", "signal", "verdict"}`` where
    ``verdict`` is one of:

    * ``"vision_only"`` — positively-identified VLM with no usable text path, or
      an unclassifiable config. Caller fail-fasts.
    * ``"text_coercible"`` — multimodal signal present but a text decoder exists
      (e.g. Kimi-K2.6 / Qwen3.6 MoE, or a generic ``ForCausalLM`` arch that
      merely carries a ``vision_config``). Caller proceeds on the text path with
      a degraded-mode warning unless ``--allow-mm-text-fallback`` is off.
    """
    config = _load_model_config_dict(model_path)
    if config is None:
        return None
    architectures = _config_architectures(config)
    # Wrapper models may nest the real arch under text_config; merge so the
    # unsupported-arch blocklist still matches (e.g. RWKV6Qwen2ForCausalLM).
    nested = config.get("text_config")
    if isinstance(nested, dict):
        for a in _config_architectures(nested):
            if a not in architectures:
                architectures.append(a)
    model_type = str(config.get("model_type") or "").strip()
    model_type_l = model_type.lower()

    # Hard denylist wins first: explicit VLM arch / model_type is vision_only
    # even if it also carries a ForCausalLM marker (e.g. Phi3VForCausalLM).
    for arch in architectures:
        if arch in _UNSUPPORTED_ARCHITECTURES:
            return {
                "architecture": arch,
                "model_type": model_type,
                "signal": f"unsupported architecture '{arch}'",
                "verdict": _VERDICT_VISION_ONLY,
            }
    if model_type_l in _UNSUPPORTED_MODEL_TYPES:
        return {
            "architecture": architectures[0] if architectures else "",
            "model_type": model_type,
            "signal": f"unsupported model_type '{model_type}'",
            "verdict": _VERDICT_VISION_ONLY,
        }

    # A multimodal config key (vision_config, image_token_id, …) is only a
    # degrade signal, not a hard block: if a text decoder exists we coerce to
    # the text path with a warning instead of fail-fasting. Routing to
    # text_coercible requires a *positive* text-decoder signal — either an
    # explicitly coercible model_type family, a confirmed text-generation
    # architecture class, or a nested text decoder config. We deliberately do
    # NOT fall back to top-level ``_SUPPORTED_MODEL_TYPES`` here: that allowlist
    # is a last-resort match for a bare model_type with no architectures, and a
    # mislabeled VLM config (e.g. a real vision model carrying model_type="llama"
    # but no decoder evidence) must fail-fast rather than silently degrade to a
    # text run.
    _has_text_decoder = _config_declares_text_decoder(config, architectures, model_type_l)
    for key in _UNSUPPORTED_CONFIG_KEYS:
        if key in config:
            verdict = (
                _VERDICT_TEXT_COERCIBLE if _has_text_decoder
                else _VERDICT_VISION_ONLY
            )
            return {
                "architecture": architectures[0] if architectures else "",
                "model_type": model_type,
                "signal": f"multimodal config key '{key}'",
                "verdict": verdict,
            }

    if any(_arch_is_supported_text_generation(a) for a in architectures):
        return None

    if model_type_l in _SUPPORTED_MODEL_TYPES:
        return None

    if architectures:
        return {
            "architecture": architectures[0],
            "model_type": model_type,
            "signal": (
                f"architecture '{architectures[0]}' does not match any "
                f"supported text-generation pattern "
                f"({', '.join(_SUPPORTED_ARCH_MARKERS)})"
            ),
            "verdict": _VERDICT_VISION_ONLY,
        }

    if model_type:
        return {
            "architecture": "",
            "model_type": model_type,
            "signal": (
                f"model_type '{model_type}' is not in the supported "
                f"text-generation allowlist"
            ),
            "verdict": _VERDICT_VISION_ONLY,
        }

    return {
        "architecture": "",
        "model_type": "",
        "signal": "config.json has neither architectures nor model_type",
        "verdict": _VERDICT_VISION_ONLY,
    }

def _load_model_max_position_embeddings(model_path: str) -> int | None:
    """Best-effort read of max sequence length from config.json (first positive among known keys, incl. nested ``text_config``), or None."""
    if not model_path:
        return None
    cfg_path = Path(model_path) / "config.json"
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    candidates = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested)
    for cfg in candidates:
        for key in _MAXPOS_CONFIG_KEYS:
            val = cfg.get(key)
            if isinstance(val, bool):
                continue
            if isinstance(val, int) and val > 0:
                return val
    return None

def _model_has_dual_chunk_attention(model_path: str) -> bool:
    """Best-effort detect a ``dual_chunk_attention_config`` in config.json.

    Qwen 1M long-context models ship this block; sglang then rejects the
    default aiter attention backend and demands ``dual_chunk_flash_attn``.
    Checks the top level and a nested ``text_config``. Soft-degrades to
    False on any missing / unreadable / invalid config.
    """
    data = _load_model_config_dict(model_path)
    if data is None:
        return False
    if data.get("dual_chunk_attention_config"):
        return True
    nested = data.get("text_config")
    return isinstance(nested, dict) and bool(
        nested.get("dual_chunk_attention_config")
    )

def _model_is_moe(model_path: str) -> bool:
    """Best-effort detect a Mixture-of-Experts model from config.json.

    MoE checkpoints declare an expert count (``num_experts`` /
    ``num_local_experts`` / ``n_routed_experts``), a ``moe_intermediate_size``,
    or carry a ``moe`` marker in ``architectures`` / ``model_type`` (e.g.
    Qwen3MoeForCausalLM / qwen3_moe). On ROCm/aiter, sglang's default
    ``--moe-runner-backend auto`` routes these through aiter's CK 2-stage
    fused-MoE kernel, whose first-request JIT build is broken in some images;
    callers use this to switch to a ROCm-capable MoE runner. Checks the top
    level and a nested ``text_config``. Soft-degrades to False on any missing
    / unreadable / invalid config.
    """
    data = _load_model_config_dict(model_path)
    if data is None:
        return False
    candidates = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested)
    expert_keys = ("num_experts", "num_local_experts", "n_routed_experts")
    for cfg in candidates:
        for key in expert_keys:
            val = cfg.get(key)
            if isinstance(val, bool):
                continue
            if isinstance(val, int) and val > 1:
                return True
        if cfg.get("moe_intermediate_size"):
            return True
        if "moe" in str(cfg.get("model_type") or "").lower():
            return True
        if any("moe" in arch.lower() for arch in _config_architectures(cfg)):
            return True
    return False

def _detect_amd_unsupported_quant(model_path: str) -> str | None:
    """Return a reason when the model ships a quant format unsupported on ROCm.

    Reads both ``config.json:quantization_config`` (standard HF) and the
    separate ``hf_quant_config.json`` (NVIDIA ModelOpt). Returns None when the
    format is ROCm-runnable or absent.
    """
    if not model_path:
        return None
    cfg = _load_model_config_dict(model_path) or {}
    qc = cfg.get("quantization_config")
    if isinstance(qc, dict):
        method = str(qc.get("quant_method") or "").strip().lower()
        if method in _AMD_UNSUPPORTED_QUANT_METHODS:
            return (
                f"quantization_config.quant_method '{method}' ships CUDA-only "
                f"kernels with no ROCm equivalent; it crashes in engine init "
                f"on AMD."
            )
    # NVIDIA ModelOpt writes a separate hf_quant_config.json, not config.json.
    hq_path = Path(model_path) / "hf_quant_config.json"
    if hq_path.is_file():
        try:
            hq = json.loads(hq_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            hq = None
        if isinstance(hq, dict):
            producer = str(
                (hq.get("producer") or {}).get("name") or "",
            ).strip().lower()
            algo = str(
                (hq.get("quantization") or {}).get("quant_algo") or "",
            ).strip().lower()
            if producer == "modelopt" and algo:
                return (
                    f"NVIDIA ModelOpt '{algo.upper()}' quantization "
                    f"(hf_quant_config.json) uses vendor-specific scale packing "
                    f"with no sglang ROCm loader (e.g. 'modelopt_fp8 ... not "
                    f"supported in ROCm'); use an AMD-native (Quark) checkpoint."
                )
            if algo in _AMD_UNSUPPORTED_QUANT_ALGOS:
                return (
                    f"'{algo.upper()}' quantization needs NVIDIA Blackwell "
                    f"hardware; no AMD/ROCm runtime path exists."
                )
    return None

def _detect_phi3_rope_scaling_incompatible(data: dict) -> str | None:
    """Return a reason when a Phi-3 su/longrope config crashes Phi3Config validation.

    Phi3Config._rope_scaling_validation() requires rope_scaling to be a 3-key
    dict, but transformers folds the top-level rope_theta into rope_scaling at
    load, yielding 4 keys and a ValueError. This is hardware-agnostic and the
    su/longrope type triggers it; yarn (the non-longrope path) is left alone.
    """
    model_type = str(data.get("model_type") or "").strip().lower()
    arches = {a.lower() for a in _config_architectures(data)}
    if model_type != "phi3" and "phi3forcausallm" not in arches:
        return None
    rope = data.get("rope_scaling")
    if not isinstance(rope, dict):
        return None
    rope_type = str(rope.get("type") or "").strip().lower()
    if rope_type not in _PHI3_ROPE_TYPES:
        return None
    # The crash only triggers when a top-level rope_theta exists: transformers
    # folds it into rope_scaling, giving 4 keys instead of the required 3.
    # Without rope_theta the 3-key dict passes validation fine.
    if data.get("rope_theta") is None:
        return None
    return (
        f"config.json is a Phi-3 model with rope_scaling.type='{rope_type}' "
        f"and a top-level rope_theta={data['rope_theta']}; "
        "Phi3Config._rope_scaling_validation requires a 3-key rope_scaling, but "
        "transformers folds the top-level rope_theta into it at load, so the "
        "validator sees 4 keys and raises ValueError at "
        "AutoConfig.from_pretrained — before --json-model-override-args can "
        "apply, so the engine crashes in init."
    )

def _detect_gemma2_missing_hidden_act(data: dict) -> str | None:
    """Return a reason when a Gemma2 config omits hidden_act.

    sglang's gemma2 runtime reads config.hidden_act unconditionally; configs
    that only ship hidden_activation crash with AttributeError in engine init.
    """
    model_type = str(data.get("model_type") or "").strip().lower()
    arches = {a.lower() for a in _config_architectures(data)}
    if model_type != "gemma2" and not (arches & _GEMMA2_ARCHITECTURES):
        return None
    scopes = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        scopes.append(nested)
    if any(s.get("hidden_act") for s in scopes):
        return None
    return (
        "config.json is a Gemma2 model but lacks hidden_act (only "
        "hidden_activation may be present); sglang's gemma2 runtime reads "
        "config.hidden_act unconditionally and crashes with AttributeError "
        "in engine init."
    )

def _detect_incompatible_model_config(
    model_path: str, gpu_type: str | None = None,
) -> str | None:
    """Detect a statically-knowable model-config incompatibility.

    Returns a human-readable reason string when the model's ``config.json``
    will crash vLLM/transformers at load time, else ``None``. Two cases,
    both conservative (no false positives on healthy configs):

    * ``config.json`` is present but corrupt / not a JSON object — the loader
      soft-degrades to ``None``, but a present-yet-unparseable file means the
      framework will fail at config load, so block early.
    * the config (top level or ``text_config``) declares a RoPE block but has
      no max-position key at all — the rope init then dereferences a missing
      ``max_position_embeddings``.

    A fully absent ``config.json`` is NOT blocked (kept soft-degrade): the
    upstream submission filter + downstream loader still apply.
    """
    if not model_path:
        return None
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.is_file():
        return None
    data = _load_model_config_dict(model_path)
    if data is None:
        # File exists but did not parse into a dict (corrupt / non-object).
        return (
            f"config.json at {cfg_path} is present but unparseable "
            f"(corrupt JSON or not a JSON object); the framework would crash "
            f"at config load."
        )
    # Reject DSA-like architectures only on AMD/ROCm.
    # The same model can still run on vendor-supported NVIDIA engines.
    if _resolve_amd_gpu_type(gpu_type):
        quant_reason = _detect_amd_unsupported_quant(model_path)
        if quant_reason is not None:
            return quant_reason
        model_type = str(data.get("model_type") or "").strip().lower()
        arches = {a.lower() for a in _config_architectures(data)}
        if (
            model_type in _AMD_UNSUPPORTED_MODEL_TYPES
            or arches & _AMD_UNSUPPORTED_ARCHITECTURES
        ):
            label = model_type or (next(iter(arches), "") if arches else "?")
            return (
                f"model architecture '{label}' has no AMD/ROCm runtime path "
                f"(needs a vendor engine on NVIDIA Hopper/Blackwell, e.g. "
                f"DeepSeek Sparse Attention); it crashes in engine init on "
                f"this hardware."
            )
    scopes = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        scopes.append(nested)
    has_rope = any(
        s.get(k) for s in scopes for k in _ROPE_CONFIG_KEYS
    )
    has_maxpos = any(
        isinstance(s.get(k), int) and not isinstance(s.get(k), bool)
        and s.get(k) > 0
        for s in scopes for k in _MAXPOS_CONFIG_KEYS
    )
    if has_rope and not has_maxpos:
        return (
            "config.json declares a RoPE block "
            f"({', '.join(_ROPE_CONFIG_KEYS)}) but no max-position field "
            f"({', '.join(_MAXPOS_CONFIG_KEYS)}); transformers/vLLM rope "
            "init dereferences a missing max_position_embeddings and crashes "
            "in engine init (DeepSeek-V3.2-Exp class)."
        )
    # Phi-3 longrope/su rope_scaling with non-canonical keys: hardware-agnostic
    # (it is a transformers-layer Phi3Config validation, not a ROCm gap).
    phi3_reason = _detect_phi3_rope_scaling_incompatible(data)
    if phi3_reason is not None:
        return phi3_reason
    # Gemma2 missing hidden_act: also hardware-agnostic.
    gemma2_reason = _detect_gemma2_missing_hidden_act(data)
    if gemma2_reason is not None:
        return gemma2_reason
    # Custom AutoConfig with unregistered model_type: sglang/vLLM fall
    # back to PreTrainedConfig (no max_position_embeddings attr) → crash.
    auto_map = data.get("auto_map")
    model_type = str(data.get("model_type") or "").strip().lower()
    if (
        isinstance(auto_map, dict)
        and auto_map.get("AutoConfig")
        and model_type in _UNREGISTERED_CUSTOM_CONFIG_TYPES
    ):
        return (
            f"model_type '{model_type}' ships a custom AutoConfig "
            f"({auto_map['AutoConfig']}) but is not registered in sglang/"
            f"vLLM's config mapping; the engine falls back to "
            f"PreTrainedConfig which lacks key attributes "
            f"(max_position_embeddings) and crashes in init."
        )
    # Dual-chunk attention on AMD/ROCm: sglang hard-requires
    # dual_chunk_flash_attn (sm90+ only) and rejects all other backends.
    if _resolve_amd_gpu_type(gpu_type) and _model_has_dual_chunk_attention(
        model_path
    ):
        return (
            "model declares dual_chunk_attention_config but sglang requires "
            "the dual_chunk_flash_attn backend which only builds on sm90+ "
            "(NVIDIA Hopper); no compatible backend exists for AMD/ROCm."
        )
    return None

