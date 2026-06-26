# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Model / GPU gate for the CLI: GPU-type resolution, arch / config loading,
unsupported-model detection, and the pre-flight gates that run before a session
is born.

Extracted from ``cli.py`` (phase 4) and consolidated in phase 6D: the former
``cli_gpu.py`` (GPU-type resolution) and ``cli_preflight.py`` (context-window +
model-config compatibility gates) were folded back in — they answer the same
"can this model run on this hardware?" question. Imports stdlib only; must not
import ``cli`` (one-way dependency, mirroring cli_kb / cli_backends).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
from pathlib import Path

from .model_config_utils import (  # noqa: F401 - re-exported for callers/tests
    GEMMA2_ARCHITECTURES as _GEMMA2_ARCHITECTURES,
    _config_architectures,
    _load_model_config_dict,
)

# Re-exported from model_config_utils for callers/tests that import these via
# ``cli_model_gate``; declared here so the re-export is intentional rather than
# a flagged unused import.
__all__ = ["_GEMMA2_ARCHITECTURES", "_config_architectures", "_load_model_config_dict"]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU-type resolution (folded back from cli_gpu.py; phase 6D). Pure helpers
# that detect / normalize the AMD GPU type from args, env, and ``rocm-smi``.
# ---------------------------------------------------------------------------
_AMD_GPU_TYPES = frozenset({"mi300x", "mi308x", "mi325x", "mi355x"})

_GFX_TO_RUNNER: dict[str, str] = {
    # Mirror Magpie/modes/benchmark/image_selector.py:138-140 so we can log resolved value at session start.
    "gfx942":  "mi300x",
    "gfx950":  "mi355x",
}


def _gpu_runner_type(gpu_type: str) -> str:
    """Return the Magpie runner label for a resolved real GPU type.

    MI308X and MI325X share the gfx942 / CDNA3 die with MI300X and reuse
    the same Magpie benchmark scripts (sglang_mi300x.sh / vllm_mi300x.sh).

    Args:
        gpu_type (str): The resolved real GPU type (e.g. ``mi325x``).

    Returns:
        str: The Magpie runner label (``mi325x`` / ``mi308x`` collapse to
            ``mi300x``).
    """
    normalized = str(gpu_type or "").strip().lower()
    if normalized in ("mi325x", "mi308x"):
        return "mi300x"
    return normalized

def _resolve_gpu_type(
    user_specified: str,
    probed: str,
) -> tuple[str, list[str]]:
    """Resolve effective gpu_type from a user hint and a hardware probe; pure for unit testing.

    Probe always wins on disagreement (wrong --gpu-type corrupts baseline+KB rows); user value kept
    only on probe failure. Returns ``(effective_gpu_type, warnings)``; warnings go to stderr to keep
    the ``HYPERLOOM_LAUNCH`` stdout sentinel clean.

    Args:
        user_specified (str): The user-supplied ``--gpu-type`` hint.
        probed (str): The hardware-probed GPU type.

    Returns:
        tuple[str, list[str]]: ``(effective_gpu_type, warnings)`` — the probe
            wins on disagreement; ``warnings`` carries any stderr notes.
    """
    warnings: list[str] = []
    if probed and user_specified and probed != user_specified:
        warnings.append(
            f"WARN: --gpu-type={user_specified!r} disagrees with probed "
            f"{probed!r}; using probed {probed!r}. The probe wins because "
            f"Magpie runner_type + KB recipe rows must match the actual "
            f"hardware to keep baseline numbers comparable across sessions."
        )
        return probed, warnings
    return (probed or user_specified), warnings

def _autodetect_gpu_type() -> str | None:
    """Return mi300x|mi308x|mi325x|mi355x or None if undetectable (rocm-smi then torch gcnArchName, best-effort).

    Returns:
        str | None: The detected GPU type, or ``None`` when undetectable.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=5,
        ).stdout.upper()
        for tag in ("MI355X", "MI325X", "MI308X", "MI300X"):
            if tag in out:
                return tag.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        # rocm-smi missing / slow / not permitted — fall through to the torch
        # gcnArchName probe below (autodetect is best-effort).
        pass
    try:
        import torch
        arch = torch.cuda.get_device_properties(0).gcnArchName
        gfx = arch.split(":", 1)[0].lower()
        return _GFX_TO_RUNNER.get(gfx)
    except Exception:  # noqa: BLE001
        return None

def _resolve_amd_gpu_type(explicit: str | None = None) -> str | None:
    """Resolve the current AMD GPU type, or None when not on AMD/unknown.

    Resolution order (most authoritative first): an explicit ``gpu_type``
    argument, the ``GPU_TYPE`` env, then a best-effort runtime autodetect.
    Returning the resolved value only when it names a known AMD runner lets
    callers gate AMD-specific behaviour on real hardware while still honouring
    a launcher/CI-supplied ``gpu_type`` even if ``rocm-smi``/torch probing is
    unavailable at the call site.

    Args:
        explicit (str | None): An explicit GPU-type hint that takes priority
            over the ``GPU_TYPE`` env and autodetect.

    Returns:
        str | None: The resolved AMD runner type, or ``None`` when not on a
            known AMD GPU.
    """
    for cand in (explicit, os.environ.get("GPU_TYPE")):
        norm = str(cand or "").strip().lower()
        if norm in _AMD_GPU_TYPES:
            return norm
    detected = (_autodetect_gpu_type() or "").strip().lower()
    return detected if detected in _AMD_GPU_TYPES else None

_SUPPORTED_ARCH_MARKERS = (
    "ForCausalLM",
    "LMHeadModel",
    "ForCausalLMWithValueHead",
)

_SUPPORTED_VL_MODEL_TYPES = frozenset({
    # Qwen VL families: carry vision_config but expose a standard text decoder
    # and are fully supported on the text path. Listed here so they bypass the
    # vision_config degrade-signal check and reach _SUPPORTED_MODEL_TYPES cleanly.
    "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe",
})

_SUPPORTED_MODEL_TYPES = frozenset({
    "llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe",
    "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe",
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
    # RWKV6/Qwen2 hybrid can also be identified by model_type alone in some
    # checkpoints; keep this aligned with CI submit filtering.
    "rwkv6qwen2",
    "gemma3",
    "mllama",
    "llava",
    "llava_next",
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
})

_MAXPOS_CONFIG_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
)

_ROPE_CONFIG_KEYS = ("rope_scaling", "rope_parameters", "rope_theta")

# minimax_m1: its lightning-attention kernel needs 128KB LDS (Required: 131072)
# but MI300X's per-CU shared-memory limit is 64KB → "out of resource: shared
# memory" → engine core init failed. Confirmed from MiniMax-M1-80k server.log.
_AMD_UNSUPPORTED_MODEL_TYPES = frozenset({"deepseek_v32", "minimax_m1"})

_AMD_UNSUPPORTED_ARCHITECTURES = frozenset({
    "deepseekv32forcausallm", "minimaxm1forcausallm",
})

_UNREGISTERED_CUSTOM_CONFIG_TYPES = frozenset({"kimi_k2"})

# Architectures Transformers/sglang's ModelConfig does not recognize at all
# (hardware-agnostic). The pydantic ModelConfig validation raises
# "model type `X` but Transformers does not recognize this architecture"
# → ValidationError in engine init regardless of GPU vendor. Matched
# case-insensitively against model_type and architectures.
# glm4_moe_lite: confirmed from zai-org-GLM-4.7-Flash server.log.
# mimo_v2_flash: confirmed from XiaomiMiMo-MiMo-V2-Flash server.log ("model of
# type mimo_v2_flash to instantiate a model of type ." + Unknown attention
# backend TRITON) — the unrecognized arch leaves an empty model type.
# deepseek_v4: confirmed from DeepSeek-V4-Flash server.log ModelConfig
# validation failure.
# glm_moe_dsa: confirmed from zai-org-GLM-5.1 server.log ("model type
# `glm_moe_dsa` but Transformers does not recognize this architecture" →
# ModelConfig ValidationError in engine init).
_UNRECOGNIZED_MODEL_TYPES = frozenset({
    "deepseek_v4", "gemma4", "glm4_moe_lite", "mimo_v2_flash", "glm_moe_dsa",
})
_UNRECOGNIZED_ARCHITECTURES = frozenset({
    "deepseekv4forcausallm", "gemma4forcausallm",
    "gemma4forconditionalgeneration", "glm4moeliteforcausallm",
    "mimov2flashforcausallm", "glmmoedsaforcausallm",
})
# Some model_type values only appear inside nested decoder configs carried by a
# wrapper. A bare top-level type is left to the framework unless we have a direct
# repro, so these are checked only against the nested text_config scope.
#
# ministral3: Mistral3 multimodal wrapper (Surpem-Supertron2 server.log: vLLM
# registry raises KeyError('ministral3') for text_config.model_type).
# qwen3_5_moe_text: Qwen3.6 text decoder wrapper; vLLM/Transformers rejects it
# with "model type `qwen3_5_moe_text` but Transformers does not recognize this
# architecture".
_NESTED_ONLY_UNRECOGNIZED_MODEL_TYPES = frozenset({
    "ministral3",
    "qwen3_5_moe_text",
})

_PHI3_ROPE_TYPES = frozenset({"su", "longrope"})
_STRICT_BOOL_CONFIG_KEYS = ("use_cache",)

# ``_GEMMA2_ARCHITECTURES`` is imported from model_config_utils (single source
# of truth) at module top.

_AMD_UNSUPPORTED_QUANT_ALGOS = frozenset({"nvfp4", "fp4"})

_AMD_UNSUPPORTED_QUANT_METHODS = frozenset({"bitsandbytes", "bnb"})

# Quant methods with a real vLLM/sglang loader. Anything else declared in
# config.json is a private/third-party format (e.g. paroquant) that fails in
# engine init. bitsandbytes/bnb are listed here but separately gated on AMD.
_SUPPORTED_QUANT_METHODS = frozenset({
    "fp8", "mxfp8", "mxfp4", "nvfp4", "blockwise_int8", "modelopt",
    "modelopt_fp8", "modelopt_fp4", "modelopt_mixed", "w8a8_int8", "w8a8_fp8",
    "w4afp8", "awq", "awq_marlin", "gptq", "gptq_marlin", "moe_wna16",
    "compressed-tensors", "compressed_tensors", "qoq", "petit_nvfp4",
    "fbgemm_fp8", "quark", "quark_int4fp8_moe", "auto-round", "modelslim",
    "bitsandbytes", "bnb", "gguf", "torchao",
})
# MLX mx.quantize uses a ``mode: affine/mlx`` block and emits per-tensor
# ``.biases`` / ``.scales`` weights (plural — distinct from a standard ``.bias``).
_MLX_QUANT_MODES = frozenset({"affine", "mlx"})

def _load_model_arch(workspace_root: Path, model_name: str) -> dict:
    """Best-effort loader for the advisory ``<workspace_root>/model_arch.json`` profile (prompts only).

    Soft-degrades to ``{}`` (never blocks launch) on missing/unreadable/invalid file. Stale-file guard:
    require ``data["model_name"]`` basename to match launched ``--model`` basename, else WARN + ``{}``.

    Args:
        workspace_root (Path): Directory containing ``model_arch.json``.
        model_name (str): The launched model name, used for the stale-file
            freshness check.

    Returns:
        dict: The advisory architecture profile, or ``{}`` when missing,
            unreadable, invalid, or stale.
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

def _load_model_config_tags(model_path: str) -> dict:
    """Best-effort loader for KB architecture-identity tags (``architectures`` + ``model_type``) from config.json.

    Soft-degrades to ``{}`` (never blocks launch); normalised fields are omitted when empty so callers can .get().

    Args:
        model_path (str): The local model directory containing ``config.json``.

    Returns:
        dict: Architecture-identity tags (``architectures`` / ``model_type``);
            empty fields are omitted, ``{}`` when the config is unreadable.
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
    (decoder-only causal LM) model.

    Args:
        arch (str): The architecture class name to test.

    Returns:
        bool: ``True`` when ``arch`` contains a supported text-generation
            marker.
    """
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

    Args:
        config (dict): The decoded model ``config.json`` mapping.
        architectures (list[str]): The top-level architecture class names.
        model_type_l (str): The lowercased top-level ``model_type``.

    Returns:
        bool: ``True`` when the config positively identifies a usable text
            decoder.
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

    Args:
        model_path (str): The local model directory containing ``config.json``.

    Returns:
        dict | None: ``None`` for a plain text-generation model (or an
            unreadable config), otherwise a dict with ``architecture``,
            ``model_type``, ``signal``, and ``verdict``
            (``vision_only`` / ``text_coercible``).
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
    nested_model_type = ""
    nested_model_type_l = ""
    if isinstance(nested, dict):
        nested_model_type = str(nested.get("model_type") or "").strip()
        nested_model_type_l = nested_model_type.lower()

    # Registry/config incompatibilities are handled by the model-config gate so
    # they get the precise model_config_incompatible stop reason. Do not emit a
    # misleading multimodal text-fallback warning for wrappers such as Gemma4.
    if _detect_unrecognized_architecture(config) is not None:
        return None

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
    if nested_model_type_l in _UNSUPPORTED_MODEL_TYPES:
        return {
            "architecture": architectures[0] if architectures else "",
            "model_type": model_type,
            "signal": f"unsupported text_config.model_type '{nested_model_type}'",
            "verdict": _VERDICT_VISION_ONLY,
        }

    # Confirmed text-generation arch or explicitly supported VL model type: both
    # bypass the vision_config degrade check below. A supported arch (ForCausalLM /
    # LMHeadModel) or a known VL family (qwen2_vl / qwen3_vl_moe / …) that carries
    # vision_config by design is fully viable on the text-serving path.
    if any(_arch_is_supported_text_generation(a) for a in architectures):
        return None
    if model_type_l in _SUPPORTED_VL_MODEL_TYPES:
        return None

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
    """Best-effort read of max sequence length from config.json (first positive among known keys, incl. nested ``text_config``), or None.

    Args:
        model_path (str): The local model directory containing ``config.json``.

    Returns:
        int | None: The first positive max-sequence-length value found, or
            ``None`` when unavailable.
    """
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

    Args:
        model_path (str): The local model directory containing ``config.json``.

    Returns:
        bool: ``True`` when a ``dual_chunk_attention_config`` block is present.
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

    Args:
        model_path (str): The local model directory containing ``config.json``.

    Returns:
        bool: ``True`` when the config carries a Mixture-of-Experts signal.
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

    Args:
        model_path (str): The local model directory containing the quant
            config files.

    Returns:
        str | None: A human-readable reason when the quant format is
            unsupported on ROCm, else ``None``.
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


def _detect_mlx_quant_weights(model_path: str) -> str | None:
    """Detect MLX (mx.quantize) checkpoints by their ``.biases``/``.scales``
    tensors in the safetensors index. Only call this when no standard
    quant_method is declared; standard quant formats also ship scale tensors.
    """
    idx = Path(model_path) / "model.safetensors.index.json"
    if not idx.is_file():
        return None
    try:
        wm = (json.loads(idx.read_text(encoding="utf-8")) or {}).get(
            "weight_map",
        ) or {}
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if any(k.endswith(".biases") or k.endswith(".scales") for k in wm):
        return (
            "checkpoint ships MLX mx.quantize weights (per-tensor '.biases'/"
            "'.scales'); no vLLM/sglang loader handles this private format, so "
            "weights fail to map in engine init (JANG/MTPLX class)."
        )
    return None


def _detect_gguf_only_checkpoint(model_path: str) -> str | None:
    """Detect a GGUF-only checkpoint with no HF-loadable weight files."""
    d = Path(model_path)
    if not d.is_dir() or not any(d.glob("*.gguf")):
        return None
    has_hf_weights = (
        any(d.glob("model*.safetensors"))
        or any(d.glob("*.safetensors.index.json"))
        or any(d.glob("pytorch_model*.bin"))
    )
    if has_hf_weights:
        return None
    return (
        "checkpoint ships only GGUF weights (llama.cpp, e.g. TQ3_4S ternary) "
        "with no HF safetensors/pytorch_model.bin weights; the default "
        "vLLM/sglang loader finds no model weights and fails in engine init."
    )


def _detect_private_quant(model_path: str, data: dict) -> str | None:
    """Reject private/third-party quantization with no vLLM/sglang loader."""
    qc = data.get("quantization_config")
    declared_supported = False
    if isinstance(qc, dict):
        raw_method = qc.get("quant_method")
        method = str(raw_method or "").strip().lower()
        if "quant_method" in qc and not method:
            return (
                "quantization_config.quant_method is empty; sglang/vLLM treats "
                "the checkpoint as quantized but cannot select a loader and "
                "fails engine init with \"Unknown quantization method: ''\"."
            )
        if method and method not in _SUPPORTED_QUANT_METHODS:
            return (
                f"quantization_config.quant_method '{method}' is a private/"
                f"third-party format with no vLLM/sglang loader; it fails in "
                f"engine init (e.g. 'Unknown quantization method')."
            )
        wfmt = str(qc.get("weight_format") or "").strip().lower()
        if "mxtq" in wfmt or "mxtq" in str(qc.get("method") or "").lower():
            return (
                "quantization_config 'mxtq' weight format (MLX/JANGTQ) has no "
                "vLLM/sglang loader; weights fail to map in engine init."
            )
        if str(qc.get("mode") or "").strip().lower() in _MLX_QUANT_MODES and not method:
            return (
                "quantization_config 'mode: affine/mlx' with no quant_method is "
                "an MLX (mx.quantize) checkpoint with no vLLM/sglang loader."
            )
        # quantization_config carries real quant params (bits/group_size/...) but
        # declares no quant_method: sglang can't pick a loader and raises
        # "Unknown quantization method: ''" in engine init.
        if not method and any(
            qc.get(k) is not None
            for k in ("bits", "group_size", "weight_format", "weight_bits")
        ):
            return (
                "quantization_config declares quant params (e.g. bits/group_size) "
                "but no quant_method; sglang/vLLM cannot select a loader and "
                "fails engine init with \"Unknown quantization method: ''\"."
            )
        declared_supported = bool(method)
    # A declared supported quant_method (awq/gptq/compressed-tensors/...)
    # legitimately ships '.scales'/'.biases'; the MLX weight-index tell only
    # applies to checkpoints with NO quant_method declared.
    if not declared_supported:
        mlx_reason = _detect_mlx_quant_weights(model_path)
        if mlx_reason is not None:
            return mlx_reason
    return _detect_gguf_only_checkpoint(model_path)


def _detect_phi3_rope_scaling_incompatible(data: dict) -> str | None:
    """Return a reason when a Phi-3 su/longrope config crashes Phi3Config validation.

    Phi3Config._rope_scaling_validation() requires rope_scaling to be a 3-key
    dict, but transformers folds the top-level rope_theta into rope_scaling at
    load, yielding 4 keys and a ValueError. This is hardware-agnostic and the
    su/longrope type triggers it; yarn (the non-longrope path) is left alone.

    Args:
        data (dict): The decoded model ``config.json`` mapping.

    Returns:
        str | None: A human-readable reason when the Phi-3 rope_scaling config
            would crash validation, else ``None``.
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

    Args:
        data (dict): The decoded model ``config.json`` mapping.

    Returns:
        str | None: A human-readable reason when a Gemma2 config omits
            ``hidden_act``, else ``None``.
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


def _detect_diffusers_pipeline_model(model_path: str) -> str | None:
    """Return a reason when the directory is a Diffusers pipeline, not an LLM.

    Diffusers repos such as FLUX.1-dev ship ``model_index.json`` at the root and
    no causal-LM ``config.json``. Without this guard they can reach baseline and
    fail only after server health checks time out.
    """
    idx = Path(model_path) / "model_index.json"
    if not idx.is_file():
        return None
    try:
        data = json.loads(idx.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    class_name = str(data.get("_class_name") or "").strip()
    if class_name.endswith("Pipeline") or class_name in {
        "FluxPipeline",
        "StableDiffusionPipeline",
        "DiffusionPipeline",
    }:
        return (
            f"model_index.json declares Diffusers pipeline '{class_name or '?'}', "
            "not a decoder-only causal LM; Hyperloom text-generation benchmarks "
            "cannot serve this model."
        )
    return None


def _detect_null_strict_bool_config(data: dict) -> str | None:
    """Return a reason for config fields that strict HF validators require bool.

    Some checkpoints serialize ``use_cache: null``. The loader path then fails
    before useful work with ``StrictDataclassFieldValidationError: field
    'use_cache' expected bool, got NoneType``.
    """
    scopes = [("config", data)]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        scopes.append(("text_config", nested))
    for scope_name, scope in scopes:
        for key in _STRICT_BOOL_CONFIG_KEYS:
            if key in scope and scope.get(key) is None:
                return (
                    f"{scope_name}.{key} is null, but HuggingFace/vLLM strict "
                    "config validation expects a bool and raises "
                    "StrictDataclassFieldValidationError before server init."
                )
    return None


# Local tokenizer artifacts sglang/HF need to build a real tokenizer. A
# checkpoint shipping only weights + config (no tokenizer) loads a degraded
# fallback whose warmup encodes an empty prompt → empty (M=0) batch → aiter
# rotary_embedding SIGFPE on MI300X. Confirmed by repro: adding the official
# Qwen2.5 tokenizer to such a model removes the M:0 crash entirely.
_TOKENIZER_ARTIFACT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "spiece.model",
)


def _detect_unrecognized_architecture(data: dict) -> str | None:
    """Return a reason when the architecture is unknown to Transformers/sglang.

    Hardware-agnostic: the ModelConfig pydantic validation rejects the unknown
    model_type with a ValidationError in engine init on any GPU vendor.

    Args:
        data (dict): The decoded model ``config.json`` mapping.

    Returns:
        str | None: A human-readable reason when the architecture is
            unrecognized by Transformers/sglang/vLLM, else ``None``.
    """
    scopes = [(data, False)]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        scopes.append((nested, True))
    for scope, is_nested in scopes:
        model_type = str(scope.get("model_type") or "").strip().lower()
        arches = {a.lower() for a in _config_architectures(scope)}
        unrecognized_types = _UNRECOGNIZED_MODEL_TYPES
        if is_nested:
            unrecognized_types = (
                _UNRECOGNIZED_MODEL_TYPES | _NESTED_ONLY_UNRECOGNIZED_MODEL_TYPES
            )
        if (
            model_type in unrecognized_types
            or arches & _UNRECOGNIZED_ARCHITECTURES
        ):
            label = model_type or (next(iter(arches), "") if arches else "?")
            return (
                f"model type '{label}' is not recognized by Transformers/"
                f"sglang/vLLM's ModelConfig or model registry; engine init "
                f"raises a validation/registry error. Needs a framework "
                f"upgrade or a registered architecture mapping."
            )
    return None


_VOCAB_WEIGHT_NAMES = (
    "embed_tokens.weight",
    "wte.weight",
    "word_embeddings.weight",
    "lm_head.weight",
)
_FULL_BASE_WEIGHT_NAMES = (
    "model.embed_tokens.weight",
    "embed_tokens.weight",
    "transformer.wte.weight",
    "wte.weight",
    "word_embeddings.weight",
    "lm_head.weight",
)
_PEFT_ADAPTER_WEIGHT_MARKERS = (
    ".lora_A.",
    ".lora_B.",
    ".lora_embedding_A",
    ".lora_embedding_B",
    ".modules_to_save.",
    ".base_layer.",
)
_SAFETENSORS_HEADER_LIMIT = 64 * 1024 * 1024


def _read_safetensors_header(path: Path) -> dict | None:
    """Read only the safetensors JSON header; never materialize tensor data.

    Args:
        path (Path): The ``*.safetensors`` shard to inspect.

    Returns:
        dict | None: The parsed JSON header, or ``None`` when it cannot be
            read / parsed within the header size limit.
    """
    try:
        with path.open("rb") as f:
            raw_len = f.read(8)
            if len(raw_len) != 8:
                return None
            header_len = struct.unpack("<Q", raw_len)[0]
            if header_len <= 0 or header_len > _SAFETENSORS_HEADER_LIMIT:
                return None
            header = json.loads(f.read(header_len))
    except (OSError, json.JSONDecodeError, ValueError, struct.error):
        return None
    return header if isinstance(header, dict) else None


def _detect_vocab_weight_shape_mismatch(model_path: str, data: dict) -> str | None:
    """Return a reason when the checkpoint has FEWER vocab rows than config.

    Best-effort and safetensors-only: reads just the JSON header (never tensor
    data) of ``*.safetensors`` shards. Legacy ``*.bin``/``pytorch_model.bin``
    checkpoints are not inspected; truncated/corrupt headers are skipped
    silently (return None) and left to the downstream loader.

    Only ``actual < config.vocab_size`` is flagged (a genuinely broken /
    truncated checkpoint that cannot serve the full vocab). ``actual >
    config.vocab_size`` is left to the framework: it is commonly a padded
    embedding (rounded up to an alignment / TP boundary while config and
    tokenizer keep the unpadded size), so blocking it here would be a
    false-positive skip of a runnable model.

    Args:
        model_path (str): The local model directory holding the safetensors
            shards.
        data (dict): The decoded model ``config.json`` mapping (supplies
            ``vocab_size``).

    Returns:
        str | None: A human-readable reason when the on-disk vocab dimension is
            smaller than ``config.json`` ``vocab_size``, else ``None``.
    """
    expected = data.get("vocab_size")
    nested = data.get("text_config")
    if not isinstance(expected, int) and isinstance(nested, dict):
        expected = nested.get("vocab_size")
    if isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0:
        return None

    mdir = Path(model_path)
    for st_path in sorted(mdir.glob("*.safetensors")):
        header = _read_safetensors_header(st_path)
        if not header:
            continue
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            if not any(name.endswith(suffix) for suffix in _VOCAB_WEIGHT_NAMES):
                continue
            shape = meta.get("shape")
            if not (
                isinstance(shape, list)
                and shape
                and isinstance(shape[0], int)
                and not isinstance(shape[0], bool)
            ):
                continue
            actual = shape[0]
            # Only block when the checkpoint has FEWER vocab rows than the
            # config declares — that is an unambiguously broken/wrong checkpoint
            # (not enough embeddings for the tokenizer). A larger on-disk
            # dimension (actual > expected) is commonly a padded embedding
            # (rounded up to an alignment / TP boundary while config + tokenizer
            # keep the unpadded value); the framework handles that, so do not
            # pre-empt it here and risk a false-positive skip of a runnable
            # model.
            if actual < expected:
                return (
                    f"config.json vocab_size={expected} but {st_path.name}:"
                    f"{name} has only {actual} vocab rows ({expected - actual} "
                    f"short); the checkpoint cannot serve the full vocab and "
                    f"weight loading will fail."
                )
    return None


def _detect_peft_adapter_only_checkpoint(model_path: str, data: dict) -> str | None:
    """Return a reason when a checkpoint looks like an unmerged PEFT adapter.

    Some repos ship ``config.json`` plus LoRA/PEFT adapter tensors but not the
    corresponding base-model weights. The default vLLM/sglang loaders then try
    to resolve base tensors such as ``base_model.model.lm_head.base_layer.weight``
    and fail during engine init. Keep this conservative: only block when the
    safetensors index explicitly carries adapter-shaped tensor names and lacks
    normal base embedding / LM-head weights.
    """
    mdir = Path(model_path)
    idx = mdir / "model.safetensors.index.json"
    if not idx.is_file():
        return None
    try:
        weight_map = (json.loads(idx.read_text(encoding="utf-8")) or {}).get(
            "weight_map",
        ) or {}
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(weight_map, dict) or not weight_map:
        return None

    keys = {str(k) for k in weight_map}
    has_adapter_tensors = any(
        marker in key for key in keys for marker in _PEFT_ADAPTER_WEIGHT_MARKERS
    )
    has_adapter_manifest = (
        (mdir / "adapter_config.json").is_file()
        or any("adapter" in str(v).lower() for v in weight_map.values())
    )
    if not has_adapter_tensors and not has_adapter_manifest:
        return None

    has_base_weights = any(
        key.endswith(suffix)
        for key in keys
        for suffix in _FULL_BASE_WEIGHT_NAMES
    )
    if has_base_weights:
        return None

    return (
        "checkpoint appears to be an unmerged PEFT/LoRA adapter: "
        "model.safetensors.index.json contains adapter/base_layer tensor names "
        "but no full base embedding or lm_head weights. The default "
        "vLLM/sglang loader cannot reconstruct missing base tensors such as "
        "base_model.model.lm_head.base_layer.weight; merge the adapter into the "
        "base model before running Hyperloom."
    )


def _detect_missing_tokenizer_files(model_path: str, data: dict) -> str | None:
    """Return a reason when a local checkpoint ships no tokenizer artifacts.

    Conservative: only fires when NONE of the known tokenizer files exist AND
    the config carries no custom AutoTokenizer (auto_map) that could supply one.

    Args:
        model_path (str): The local model directory to inspect.
        data (dict): The decoded model ``config.json`` mapping (checked for a
            custom ``auto_map`` AutoTokenizer).

    Returns:
        str | None: A human-readable reason when no tokenizer artifacts are
            present, else ``None``.
    """
    auto_map = data.get("auto_map")
    if isinstance(auto_map, dict) and auto_map.get("AutoTokenizer"):
        return None
    mdir = Path(model_path)
    if any((mdir / f).is_file() for f in _TOKENIZER_ARTIFACT_FILES):
        return None
    return (
        "model directory ships weights + config but no tokenizer artifacts "
        f"({', '.join(_TOKENIZER_ARTIFACT_FILES)}); sglang loads a degraded "
        "fallback tokenizer whose warmup encodes an empty prompt, producing an "
        "empty (M=0) batch that crashes the aiter rotary_embedding kernel with "
        "SIGFPE on MI300X (Gensyn-Swarm fine-tune class)."
    )


def _detect_mistral_common_tokenizer_gap(model_path: str, data: dict) -> str | None:
    """Return a reason for Mistral checkpoints missing Mistral tokenizer files.

    Some Mistral fine-tunes ship ``tokenizer.json`` but omit the files that
    Transformers' MistralCommonBackend accepts. SGLang then fails during server
    init with ``ValueError: No tokenizer file found`` even though the generic
    missing-tokenizer check sees a tokenizer artifact.
    """
    model_type = str(data.get("model_type") or "").strip().lower()
    arches = {str(a or "").strip() for a in _config_architectures(data)}
    if model_type != "mistral" and "MistralForCausalLM" not in arches:
        return None

    auto_map = data.get("auto_map")
    if isinstance(auto_map, dict) and auto_map.get("AutoTokenizer"):
        return None

    mdir = Path(model_path)
    if not (mdir / "tokenizer.json").is_file():
        return None
    mistral_files = (
        "tokenizer.model",
        "tokenizer.model.v3",
        "tekken.json",
        "tokenizer_config.json",
    )
    if any((mdir / f).is_file() for f in mistral_files):
        return None
    return (
        "Mistral checkpoint ships tokenizer.json but none of the tokenizer "
        "metadata/files accepted by Transformers MistralCommonBackend "
        f"({', '.join(mistral_files)}); sglang server init fails with "
        "\"No tokenizer file found\"."
    )


def _detect_llama_sentencepiece_metadata_gap(model_path: str, data: dict) -> str | None:
    """Return a reason for Llama checkpoints with bare SentencePiece tokenizer.

    Some local Llama fine-tunes ship ``tokenizer.model`` but omit
    ``tokenizer_config.json`` / ``tokenizer.json``. SGLang first loads a generic
    tokenizer backend, then retries the declared-class path with the local
    absolute model path. The HF Hub validator rejects that path with
    ``HFValidationError: Repo id must be in the form ...`` before serving starts.
    """
    model_type = str(data.get("model_type") or "").strip().lower()
    arches = {str(a or "").strip() for a in _config_architectures(data)}
    if model_type != "llama" and "LlamaForCausalLM" not in arches:
        return None

    auto_map = data.get("auto_map")
    if isinstance(auto_map, dict) and auto_map.get("AutoTokenizer"):
        return None

    mdir = Path(model_path)
    if not (mdir / "tokenizer.model").is_file():
        return None
    if (mdir / "tokenizer_config.json").is_file() or (mdir / "tokenizer.json").is_file():
        return None
    return (
        "Llama checkpoint ships tokenizer.model but lacks tokenizer_config.json "
        "or tokenizer.json; sglang falls back through a local-path tokenizer "
        "resolution path that raises HFValidationError before server init."
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

    Args:
        model_path (str): The local model directory containing ``config.json``.
        gpu_type (str | None): Optional GPU type; AMD-only checks fire when it
            resolves to a known AMD runner.

    Returns:
        str | None: A human-readable reason when a statically-knowable config
            incompatibility is detected, else ``None``.
    """
    if not model_path:
        return None
    pipeline_reason = _detect_diffusers_pipeline_model(model_path)
    if pipeline_reason is not None:
        return pipeline_reason
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
    null_bool_reason = _detect_null_strict_bool_config(data)
    if null_bool_reason is not None:
        return null_bool_reason
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
    # Unrecognized architecture (e.g. glm4_moe_lite): hardware-agnostic
    # ModelConfig ValidationError in engine init.
    unrecognized_reason = _detect_unrecognized_architecture(data)
    if unrecognized_reason is not None:
        return unrecognized_reason
    # Private/third-party quantization (paroquant, MLX, mxtq, GGUF): no loader
    # exists on any backend; hardware-agnostic engine-init failure.
    private_quant_reason = _detect_private_quant(model_path, data)
    if private_quant_reason is not None:
        return private_quant_reason
    peft_adapter_reason = _detect_peft_adapter_only_checkpoint(model_path, data)
    if peft_adapter_reason is not None:
        return peft_adapter_reason
    vocab_shape_reason = _detect_vocab_weight_shape_mismatch(model_path, data)
    if vocab_shape_reason is not None:
        return vocab_shape_reason
    # Missing tokenizer artifacts: hardware-agnostic; the degraded fallback
    # tokenizer's empty-prompt warmup triggers an aiter M=0 SIGFPE.
    tokenizer_reason = _detect_missing_tokenizer_files(model_path, data)
    if tokenizer_reason is not None:
        return tokenizer_reason
    mistral_tokenizer_reason = _detect_mistral_common_tokenizer_gap(model_path, data)
    if mistral_tokenizer_reason is not None:
        return mistral_tokenizer_reason
    llama_tokenizer_reason = _detect_llama_sentencepiece_metadata_gap(model_path, data)
    if llama_tokenizer_reason is not None:
        return llama_tokenizer_reason
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


# ===========================================================================
# Pre-flight gates (folded back from cli_preflight.py; phase 6D). Validate the
# requested context window + model-config compatibility before a run is born.
# They drive the detectors above (same "can this model run?" concern), so they
# live here too. Each persists a stop reason and returns True when the caller
# should exit.
# ===========================================================================
_CONTEXT_HEADROOM_ENV = "HYPERLOOM_CONTEXT_HEADROOM_TOKENS"

_CONTEXT_HEADROOM_DEFAULT = 512

_MAX_MODEL_LEN_HEADROOM = 4096

def _context_headroom_tokens() -> int:
    """Resolve the context headroom (tokens); env override, else default.

    Returns:
        int: The configured context headroom in tokens (falls back to the
            default for unset / invalid / negative env values).
    """
    raw = os.environ.get(_CONTEXT_HEADROOM_ENV, "").strip()
    if not raw:
        return _CONTEXT_HEADROOM_DEFAULT
    try:
        val = int(raw)
    except ValueError:
        return _CONTEXT_HEADROOM_DEFAULT
    return val if val >= 0 else _CONTEXT_HEADROOM_DEFAULT

def _resolve_max_model_len(isl: int, osl: int, model_path: str) -> int:
    """Resolve ``MAX_MODEL_LEN`` = ISL+OSL+headroom, clamped to ``max_position_embeddings`` (never stretch context).

    Args:
        isl (int): Input sequence length.
        osl (int): Output sequence length.
        model_path (str): The local model directory containing ``config.json``.

    Returns:
        int: The resolved ``MAX_MODEL_LEN``, clamped to the model's native
            max-position window when known.
    """
    desired = int(isl) + int(osl) + _MAX_MODEL_LEN_HEADROOM
    maxpos = _load_model_max_position_embeddings(model_path)
    if maxpos:
        return min(desired, maxpos)
    return desired

def _emit_breakdown_to_langfuse(session_dir: Path) -> None:
    """Best-effort: push the just-written ``session_breakdown.json`` to Langfuse.

    The pre-flight gates below fail-fast *before* ``coordinator.run()``'s
    ``finally`` — the one place a normal session flushes Langfuse and attaches
    the breakdown document. Without this, an early-aborted session writes
    ``session_breakdown.json`` to disk (so the on-disk collector path still
    forwards it downstream) but never emits the trace/observation to Langfuse,
    leaving the session absent from Langfuse entirely. Mirror ``cli``'s
    end-of-session order (flush -> patch -> record) so the attached document
    carries the post-flush receipt counts.

    No-op unless ``HYPERLOOM_LANGFUSE_ENABLE`` + the ``LANGFUSE_*`` connection
    vars are set; never raises (a Langfuse outage must not mask the stop
    reason). Call only *after* ``write_breakdown_json`` has run.

    Args:
        session_dir (Path): The session root directory whose breakdown is
            pushed to Langfuse.
    """
    try:
        from .breakdown import patch_breakdown_langfuse
        from .orchestrator.trace.langfuse_emitter import (
            flush_session,
            record_session_breakdown,
        )

        flush_session(session_dir)
        patch_breakdown_langfuse(session_dir)
        record_session_breakdown(session_dir)
    except Exception as exc:  # noqa: BLE001 — best-effort; never mask the reason
        print(
            f"WARNING: failed to emit session_breakdown to Langfuse on "
            f"fail-fast: {exc!r}",
            file=sys.stderr,
        )


def _preflight_context_window(args: argparse.Namespace, session_dir: Path) -> bool:
    """Fail fast when ``max_position_embeddings < ISL+OSL+headroom`` (no --context-length stretch by policy).

    Persists a stop reason and returns True (caller should exit) when the workload does NOT fit; False
    when it fits or the model's max length is unknown.

    Args:
        args (argparse.Namespace): The parsed CLI namespace (reads
            ``isl`` / ``osl`` / ``model``).
        session_dir (Path): The session root directory for the stop report.

    Returns:
        bool: ``True`` when the workload does not fit (caller should exit),
            ``False`` when it fits or the max length is unknown.
    """
    isl = int(getattr(args, "isl", 0) or 0)
    osl = int(getattr(args, "osl", 0) or 0)
    if isl <= 0 or osl <= 0:
        return False
    maxpos = _load_model_max_position_embeddings(str(getattr(args, "model", "") or ""))
    if not maxpos:
        return False
    headroom = _context_headroom_tokens()
    required = isl + osl + headroom
    if maxpos >= required:
        return False

    reason = (
        f"model max_position_embeddings={maxpos} < required {required} "
        f"(ISL={isl} + OSL={osl} + headroom={headroom}). The workload exceeds "
        f"the model context window; every request would 400. Refusing to run "
        f"(no --context-length override by policy). Lower ISL/OSL for this "
        f"model, or lower {_CONTEXT_HEADROOM_ENV} if the headroom is too "
        f"conservative (it is added to `required`, so raising it makes "
        f"admission stricter, not looser)."
    )
    # Persist the stop reason so CI / the robustness monitor read it from state.json instead of the log.
    try:
        from .orchestrator.shared_state import SharedState
        from .orchestrator.action_executors.report import (
            _build_summary_dict,
            _format_md,
        )
        from .session_paths import reports_dir

        state = SharedState.load_or_init(session_dir)
        # Validated writer keeps the vocab-closed invariant Inv-8.3 (term registered in STOP_REASON_VOCAB).
        state.set_stop_reason("model_context_window_too_small")
        state.closing_phase = True
        state.save(session_dir)
        summary = _build_summary_dict(state, {}, [], external_baseline=None)
        summary["stop_detail"] = reason
        rdir = reports_dir(session_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "final.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
        )
        (rdir / "final.md").write_text(_format_md(summary), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — don't mask the reason on a writer bug
        print(
            f"WARNING: failed to persist context-window stop report: {exc!r}",
            file=sys.stderr,
        )
    # Delivery-artifact parity: emit session_breakdown.json here too since fail-fast exits before
    # coordinator.run()'s finally, so CI's delivery contract sees a clean skip not "Missing artifacts".
    try:
        from .breakdown import write_breakdown_json
        write_breakdown_json(session_dir)
    except Exception as exc:  # noqa: BLE001 — best-effort; never mask the reason
        print(
            f"WARNING: failed to write session_breakdown.json on context "
            f"fail-fast: {exc!r}",
            file=sys.stderr,
        )
    # Langfuse parity: this gate exits before coordinator.run()'s finally, so
    # push the breakdown to Langfuse here too (else the session is on disk for
    # the collector but missing from Langfuse).
    _emit_breakdown_to_langfuse(session_dir)
    print(f"ERROR: {reason}", file=sys.stderr)
    return True

def _preflight_model_config_compat(
    args: argparse.Namespace, session_dir: Path,
) -> bool:
    """Fail fast when the model config is statically known to be incompatible.

    Catches configs that crash vLLM/transformers at load (corrupt config.json,
    or a RoPE block without any max-position field) so we persist a clear stop
    reason instead of booting a server that dies cryptically in engine init.

    Returns True when incompatible (caller should exit); False otherwise.

    Args:
        args (argparse.Namespace): The parsed CLI namespace (reads ``model``
            and ``gpu_type``).
        session_dir (Path): The session root directory for the stop report.

    Returns:
        bool: ``True`` when the config is incompatible (caller should exit),
            ``False`` otherwise.
    """
    model = str(getattr(args, "model", "") or "")
    detail = _detect_incompatible_model_config(
        model, str(getattr(args, "gpu_type", "") or "") or None,
    )
    if detail is None:
        return False
    name = Path(model).name or model
    reason = (
        f"Model '{name}' has an incompatible config: {detail} Refusing to run "
        f"before the heavy server bring-up. Upgrade the framework/transformers "
        f"to a version that supports this model, or skip it on this hardware."
    )
    try:
        from .orchestrator.shared_state import SharedState
        from .orchestrator.action_executors.report import (
            _build_summary_dict,
            _format_md,
        )
        from .session_paths import reports_dir

        state = SharedState.load_or_init(session_dir)
        state.set_stop_reason("model_config_incompatible")
        state.closing_phase = True
        state.save(session_dir)
        summary = _build_summary_dict(state, {}, [], external_baseline=None)
        summary["stop_detail"] = reason
        rdir = reports_dir(session_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "final.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
        )
        (rdir / "final.md").write_text(_format_md(summary), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — don't mask the reason on a writer bug
        print(
            f"WARNING: failed to persist model-config stop report: {exc!r}",
            file=sys.stderr,
        )
    try:
        from .breakdown import write_breakdown_json
        write_breakdown_json(session_dir)
    except Exception as exc:  # noqa: BLE001 — best-effort; never mask the reason
        print(
            f"WARNING: failed to write session_breakdown.json on config "
            f"fail-fast: {exc!r}",
            file=sys.stderr,
        )
    # Langfuse parity: this gate exits before coordinator.run()'s finally, so
    # push the breakdown to Langfuse here too (else the session is on disk for
    # the collector but missing from Langfuse).
    _emit_breakdown_to_langfuse(session_dir)
    print(f"ERROR: {reason}", file=sys.stderr)
    return True

def _preflight_unsupported_model_arch(
    args: argparse.Namespace, session_dir: Path,
) -> bool:
    """Gate multimodal/vision models before expensive bring-up.

    Best-effort (an unreadable config.json is not a hard block). Three outcomes:

    * plain text model → returns False (run proceeds normally).
    * ``text_coercible`` (multimodal signal but a text decoder exists) →
      when ``--allow-mm-text-fallback`` is on (default), records a degraded-mode
      warning on SharedState, emits a loud stderr/log warning, and returns False
      so the run proceeds on the text path. When the flag is off, falls through
      to fail-fast.
    * ``vision_only`` (true VLM / unclassifiable) → persists
      ``stop_reason=unsupported_model_arch`` and returns True (caller exits).

    Args:
        args (argparse.Namespace): The parsed CLI namespace (reads ``model``
            and ``allow_mm_text_fallback``).
        session_dir (Path): The session root directory for any stop report /
            degraded-mode marker.

    Returns:
        bool: ``True`` when the model is vision-only (caller should exit),
            ``False`` for plain text or coercible-with-fallback models.
    """
    model = str(getattr(args, "model", "") or "")
    hit = _detect_unsupported_model(model)
    if hit is None:
        return False

    name = Path(model).name or model
    arch = hit.get("architecture") or "<unknown>"
    mt = hit.get("model_type") or "<unknown>"
    verdict = str(hit.get("verdict") or _VERDICT_VISION_ONLY)
    allow_fallback = bool(getattr(args, "allow_mm_text_fallback", True))

    if verdict == _VERDICT_TEXT_COERCIBLE and allow_fallback:
        warning = (
            f"DEGRADED MODE: model '{name}' carries a multimodal signal "
            f"({hit.get('signal', 'multimodal config')}; architecture '{arch}', "
            f"model_type '{mt}') but exposes a text-generation path. Hyperloom "
            f"is running it on the TEXT path only — any image/audio inputs are "
            f"ignored, so benchmark numbers reflect the text decoder alone. "
            f"Pass --no-allow-mm-text-fallback to fail-fast instead."
        )
        print(f"WARNING: {warning}", file=sys.stderr)
        log.warning(warning)
        try:
            from .orchestrator.shared_state import SharedState

            state = SharedState.load_or_init(session_dir)
            state.degraded_mode = True
            state.model_warnings = list(state.model_warnings or []) + [{
                "kind": "multimodal_text_fallback",
                "model_name": name,
                "architecture": arch,
                "model_type": mt,
                "signal": str(hit.get("signal") or ""),
                "detail": warning,
            }]
            state.save(session_dir)
        except Exception as exc:  # noqa: BLE001 — never block the run on advisory write
            print(
                f"WARNING: failed to persist degraded-mode marker: {exc!r}",
                file=sys.stderr,
            )
        return False

    reason = (
        f"Unsupported model '{name}': architecture '{arch}' (model_type "
        f"'{mt}') is not a supported text-generation model. Hyperloom only "
        f"supports decoder-only causal LM models (architectures containing "
        f"ForCausalLM or LMHeadModel). Rejected because: "
        f"{hit.get('signal', 'unknown architecture')}. Submit a "
        f"text-generation checkpoint instead."
    )
    # Persist the stop reason so CI / the robustness monitor read it from state.json instead of the log.
    try:
        from .orchestrator.shared_state import SharedState
        from .orchestrator.action_executors.report import (
            _build_summary_dict,
            _format_md,
        )
        from .session_paths import reports_dir

        state = SharedState.load_or_init(session_dir)
        # Validated writer keeps the vocab-closed invariant Inv-8.3 (term registered in STOP_REASON_VOCAB).
        state.set_stop_reason("unsupported_model_arch")
        state.closing_phase = True
        state.save(session_dir)
        summary = _build_summary_dict(state, {}, [], external_baseline=None)
        summary["stop_detail"] = reason
        rdir = reports_dir(session_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "final.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
        )
        (rdir / "final.md").write_text(_format_md(summary), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — don't mask the reason on a writer bug
        print(
            f"WARNING: failed to persist unsupported-model stop report: {exc!r}",
            file=sys.stderr,
        )
    # Delivery-artifact parity: emit session_breakdown.json here too since fail-fast exits before
    # coordinator.run()'s finally, so CI's delivery contract sees a clean skip not "Missing artifacts".
    try:
        from .breakdown import write_breakdown_json
        write_breakdown_json(session_dir)
    except Exception as exc:  # noqa: BLE001 — best-effort; never mask the reason
        print(
            f"WARNING: failed to write session_breakdown.json on unsupported-"
            f"model fail-fast: {exc!r}",
            file=sys.stderr,
        )
    # Langfuse parity: this gate exits before coordinator.run()'s finally, so
    # push the breakdown to Langfuse here too (else the session is on disk for
    # the collector but missing from Langfuse).
    _emit_breakdown_to_langfuse(session_dir)
    print(f"ERROR: {reason}", file=sys.stderr)
    return True
