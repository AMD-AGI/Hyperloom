# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared model-config helpers (config.json parsing + arch/type detection).

Leaf module: depends only on the standard library so both ``cli`` and the
orchestrator executors can import it without a circular dependency.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path


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
      absent scheme defaults to dynamic in sglang's ``Fp8Config``.

    Args:
        model_path: Filesystem path to the model directory.

    Returns:
        ``True`` only for non-block FP8 checkpoints with dynamic activation.
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
    return activation in ("", "dynamic")


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
