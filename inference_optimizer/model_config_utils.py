# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Shared model-config helpers (config.json parsing + arch/type detection).

Leaf module: depends only on the standard library so both ``cli`` and the
orchestrator executors can import it without a circular dependency.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path


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


# Gemma2 forward builds ``normalizer = torch.tensor(...)`` (a host scalar) on
# every call. The TraceLens kernel_shape_profiler patch activates inside the
# CUDA-graph capture critical section, so that host construct runs during HIP
# stream capture and raises ``hipErrorStreamCaptureUnsupported`` -> capture
# fails -> roofline produces no ceiling. Callers skip shape-discovery for
# Gemma2 to keep CUDA graph while avoiding the crash.
# Single source of truth: ``cli`` reuses these for its preflight checks too.
GEMMA2_MODEL_TYPE = "gemma2"
GEMMA2_ARCHITECTURES = frozenset({"gemma2forcausallm"})


def _path_looks_like_gemma2(model_path: str) -> bool:
    """Heuristic Gemma2 detection from the path when config.json is absent.

    Matches a ``gemma-2`` / ``gemma2`` directory name while excluding Gemma3, so
    a not-yet-materialized Hub-id style path still gets the workaround.
    """
    if not model_path:
        return False
    compact = (
        Path(model_path).name.lower()
        .replace("-", "").replace("_", "").replace(".", "")
    )
    if "gemma3" in compact:
        return False
    return "gemma2" in compact


def _model_is_gemma2(model_path: str) -> bool:
    """Best-effort detect a Gemma2 model from config.json (top level or text_config).

    Falls back to a path heuristic when config.json is missing/unreadable.
    """
    data = _load_model_config_dict(model_path)
    if data is None:
        return _path_looks_like_gemma2(model_path)
    candidates = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested)
    for cfg in candidates:
        if str(cfg.get("model_type") or "").strip().lower() == GEMMA2_MODEL_TYPE:
            return True
        if any(a.lower() in GEMMA2_ARCHITECTURES for a in _config_architectures(cfg)):
            return True
    return False
