from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

log = logging.getLogger("optimize-submit")

from . import config as _config

globals().update({k: v for k, v in vars(_config).items() if not k.startswith("__")})

from . import hf_client as _hf_client

globals().update({k: v for k, v in vars(_hf_client).items() if not k.startswith("__")})

# ── Auto-detection ──────────────────────────────────────────────────────────────


@dataclass
class DetectedConfig:
    """Auto-detected launch configuration for a model.

    Attributes:
        arch (str): HF ``architectures[0]`` class name.
        framework (str): Chosen serving framework (``sglang`` / ``vllm``).
        precision (str): Detected precision tag (e.g. ``FP8`` / ``INT4``).
        tp (int): Tensor-parallel size.
        concurrency (int): Benchmark concurrency.
        image (str): Container image to run.
        params_b (float): Parameter count in billions.
    """

    arch: str
    framework: str
    precision: str
    tp: int
    concurrency: int
    image: str
    params_b: float
    max_context_tokens: int
    # Raw config.json max_position_embeddings (0 when absent).
    max_position_embeddings: int = 0
    # Raw config.json dict, used by the shared model_compat pre-flight.
    raw_config: dict = field(default_factory=dict)


def _quant_type(config: dict) -> str:
    """Read the quantization tag from HF config.json (vendors disagree on the
    field name). Priority: quant_algo > quant_type > quantization_type >
    quant_method > method. First non-empty wins, lowercased.

    Args:
        config (dict): A HF ``config.json`` dict.

    Returns:
        str: The lowercased quantization tag, or ``""`` when none is present.
    """
    quant = config.get("quantization_config") or {}
    raw = (
        quant.get("quant_algo")
        or quant.get("quant_type")
        or quant.get("quantization_type")
        or quant.get("quant_method")
        or quant.get("method")
        or ""
    )
    return raw.lower() if isinstance(raw, str) else ""


def detect_framework(config: dict) -> str:
    """Choose the serving framework for a model from its config.

    vLLM is selected for architectures that require it or for quantization
    types it handles better; SGLang is used for known-good architectures;
    otherwise vLLM is the broader-support fallback.

    Args:
        config (dict): A HF ``config.json`` dict.

    Returns:
        str: ``"vllm"`` or ``"sglang"``.
    """
    arch = (config.get("architectures") or [""])[0]
    qt = _quant_type(config)
    if arch in VLLM_REQUIRED_ARCHS:
        return "vllm"
    if any(q in qt for q in VLLM_QUANT_TYPES):
        return "vllm"
    if arch in SGLANG_ARCHS:
        return "sglang"
    log.warning("unknown architecture %r — defaulting to vllm (broader support)", arch)
    return "vllm"


def detect_precision(config: dict) -> str:
    """Detect the serving precision from a model's quantization tag.

    Args:
        config (dict): A HF ``config.json`` dict.

    Returns:
        str: One of ``FP8`` / ``FP4`` / ``INT4``, defaulting to ``FP8`` for
            unquantized models on MI300X.
    """
    qt = _quant_type(config)
    if "fp8" in qt:
        return "FP8"
    if "mxfp4" in qt:
        return "FP4"
    if "nvfp4" in qt:
        return "FP4"
    if "int4" in qt:
        return "INT4"
    if "gptq" in qt:
        return "INT4"
    if "awq" in qt:
        return "INT4"
    return "FP8"  # unquantized default for MI300X


def detect_param_count(hf_info: dict, config: dict) -> float:
    """Estimate a model's parameter count in billions.

    Prefers the exact ``safetensors.total`` count from HF metadata; otherwise
    approximates from hidden size, layer count, and vocab size.

    Args:
        hf_info (dict): The HF model-info JSON.
        config (dict): The HF ``config.json`` dict.

    Returns:
        float: Parameter count in billions, or 0.0 when it cannot be estimated.
    """
    total = (hf_info.get("safetensors") or {}).get("total", 0)
    if total:
        return total / 1e9
    h = config.get("hidden_size", 0)
    n = config.get("num_hidden_layers", 0)
    vocab = config.get("vocab_size", 0)
    if h and n:
        return (12 * h * h * n + vocab * h) / 1e9
    return 0.0


def detect_max_context_tokens(config: dict) -> int:
    """Return the model context length from HF config.json when present.

    Args:
        config (dict): A HF ``config.json`` dict.

    Returns:
        int: The smallest positive context-length field found, or ``0`` when
        none is present.
    """
    candidates = []
    for key in ("max_position_embeddings", "max_sequence_length", "n_positions", "seq_length"):
        value = config.get(key)
        if isinstance(value, (int, float)) and value > 0:
            candidates.append(int(value))
    return min(candidates) if candidates else 0


def context_too_short(
    max_context_tokens: int,
    isl: int,
    osl: int,
    reserve_tokens: int = DEFAULT_CONTEXT_RESERVE_TOKENS,
) -> bool:
    """Return whether the model context cannot fit the requested workload.

    Args:
        max_context_tokens: Model's maximum context length; ``<= 0`` means
            unknown, in which case the check is skipped.
        isl: Input sequence length.
        osl: Output sequence length.
        reserve_tokens: Headroom kept free beyond input + output.

    Returns:
        ``True`` when the context is known and smaller than
        ``isl + osl + reserve_tokens``; otherwise ``False``.
    """
    if max_context_tokens <= 0:
        return False
    return max_context_tokens < (isl + osl + reserve_tokens)


def detect_tp(params_b: float, precision: str = "BF16", gpu_type: str | None = None) -> int:
    """Pick tensor parallelism from param count and GPU profile. precision is
    kept for API compatibility but unused.

    Args:
        params_b (float): Parameter count in billions.
        precision (str): Kept for API compatibility; unused.
        gpu_type (str | None): GPU type used to select the profile thresholds.

    Returns:
        int: The tensor-parallel size (1, 2, 4, or 8).
    """
    if params_b <= 0:
        return 1
    profile_key = normalize_gpu_profile(gpu_type) or DEFAULT_GPU_PROFILE
    profile = GPU_PROFILES[profile_key]
    tp1_max, tp2_max, tp4_max = profile["tp_thresholds_b"]
    if params_b <= tp1_max:
        return 1
    if params_b <= tp2_max:
        return 2
    if params_b <= tp4_max:
        return 4
    return 8


def detect_concurrency(tp: int, framework: str) -> int:
    """Pick a benchmark concurrency from tensor-parallel size and framework.

    CI policy is a single fixed concurrency of 64 across every framework and TP
    size, so benchmark load stays comparable between models. ``tp`` /
    ``framework`` are kept for API compatibility.

    Args:
        tp (int): Tensor-parallel size.
        framework (str): Serving framework (``vllm`` / ``sglang``).

    Returns:
        int: The chosen concurrency level (always ``64``).
    """
    return 64


def _sglang_image_for(repo_id: str = "") -> str:
    """Pick the sglang image, honoring per-model baseline-arch needs.

    Returns the default v0.5.12 profilerfix image.

    Args:
        repo_id (str): Model repo id, matched on its basename for overrides.

    Returns:
        str: The default sglang image. ``repo_id`` is accepted for future
        per-model overrides.
    """
    return _default_sglang_image()


def _vllm_image_for(repo_id: str = "") -> str:
    """Pick the vLLM image, honoring per-model baseline-arch needs.

    Default is the standard vLLM image; Gemma-4 needs the dedicated gemma4 image
    since the stock build does not serve the gemma-4 arch. Matched on the repo
    basename.

    Args:
        repo_id (str): Model repo id, matched on its basename for overrides.

    Returns:
        str: The gemma4 vLLM image for gemma-4 repos, else the default vLLM image.
    """
    basename = (repo_id or "").split("/")[-1].lower()
    if "gemma-4" in basename or "gemma4" in basename:
        return "harbor.core42.example-internal-host.invalid/sync/vllm-openai-rocm:gemma4"
    return _default_vllm_image()


def detect_image(framework: str, repo_id: str = "") -> str:
    """Select the server image for a framework and model.

    Args:
        framework: Serving framework (``vllm`` / ``sglang``).
        repo_id: Model repo id, used to honor per-model image overrides.

    Returns:
        The vLLM image chosen by :func:`_vllm_image_for` for ``vllm`` (gemma-4
        gets a dedicated image); otherwise the SGLang image chosen by
        :func:`_sglang_image_for`.
    """
    return _vllm_image_for(repo_id) if framework == "vllm" else _sglang_image_for(repo_id)


def auto_detect(hf: HuggingFaceClient, repo_id: str, gpu_type: str | None = None) -> DetectedConfig | None:
    """Derive a benchmark configuration from a model's HF metadata.

    Fetches model info and ``config.json`` and infers framework, precision,
    tensor parallelism, concurrency, image, and context limits.

    Args:
        hf: Hugging Face client used to fetch metadata.
        repo_id: Model repo id to inspect.
        gpu_type: Target GPU type, used for TP/profile selection.

    Returns:
        A :class:`DetectedConfig`, or ``None`` when the HF metadata cannot be
        fetched.
    """
    log.info("[%s] fetching HF metadata", repo_id)
    try:
        info = hf.model_info(repo_id)
        config = hf.model_config(repo_id)
    except Exception as e:
        log.error("[%s] HF fetch failed: %s", repo_id, e)
        return None

    arch = (config.get("architectures") or ["unknown"])[0]

    # Refuse non-generative repos: sglang/vllm won't serve them.
    if not is_generative_arch(arch):
        log.error(
            "[%s] arch=%s is not a generative LM "
            "(expected ForCausalLM / ForConditionalGeneration / LMHeadModel / ForSeq2SeqLM "
            "suffix). Skipping — pass an actual causal-LM repo, or override "
            "with --manual --framework vllm if you really want to try.",
            repo_id,
            arch,
        )
        return None

    framework = detect_framework(config)
    precision = detect_precision(config)
    params_b = detect_param_count(info, config)
    max_context_tokens = detect_max_context_tokens(config)
    mpe_raw = config.get("max_position_embeddings")
    max_position_embeddings = int(mpe_raw) if isinstance(mpe_raw, (int, float)) and mpe_raw > 0 else 0
    tp = detect_tp(params_b, precision, gpu_type)
    conc = detect_concurrency(tp, framework)
    image = detect_image(framework, repo_id)

    cfg = DetectedConfig(
        arch=arch,
        framework=framework,
        precision=precision,
        tp=tp,
        concurrency=conc,
        image=image,
        params_b=params_b,
        max_context_tokens=max_context_tokens,
        max_position_embeddings=max_position_embeddings,
        raw_config=config,
    )
    log.info(
        "[%s] arch=%s params=%.1fB context=%d framework=%s precision=%s gpu=%s tp=%d conc=%d",
        repo_id,
        arch,
        params_b,
        max_context_tokens,
        framework,
        precision,
        canonical_gpu_type(gpu_type),
        tp,
        conc,
    )
    return cfg
