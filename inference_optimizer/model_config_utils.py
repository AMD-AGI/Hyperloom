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


# ``gemma`` then an optional single separator then ``2`` as a standalone token
# (start/separator on the left, separator/end on the right). Matches gemma2 /
# gemma-2 / gemma_2 but not gemma3, gemma25, or notgemma2.
_GEMMA2_PATH_RE = re.compile(r"(?:^|[-_.])gemma[-_.]?2(?:[-_.]|$)")


def _path_looks_like_gemma2(model_path: str) -> bool:
    """Heuristic Gemma2 detection from the path when config.json is absent.

    Word-boundary match on the directory name (gemma2 / gemma-2 / gemma_2),
    so a not-yet-materialized Hub-id style path still gets the workaround
    without false-positives on names like notgemma2 / gemma25.
    """
    if not model_path:
        return False
    return _GEMMA2_PATH_RE.search(Path(model_path).name.lower()) is not None


def _config_gemma2_scopes(data: dict) -> list[dict]:
    """Return [top-level, text_config?] scopes for Gemma2 inspection."""
    scopes = [data]
    nested = data.get("text_config")
    if isinstance(nested, dict):
        scopes.append(nested)
    return scopes


def _config_is_gemma2(data: dict) -> bool:
    """True when a parsed config dict declares Gemma2 (top level or text_config)."""
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
    """
    for cfg in _config_gemma2_scopes(data):
        if str(cfg.get("model_type") or "").strip():
            return True
        if _config_architectures(cfg):
            return True
    return False


def _model_is_gemma2(model_path: str) -> bool:
    """Best-effort detect a Gemma2 model from config.json (top level or text_config).

    Falls back to a path heuristic when config.json is missing/unreadable OR
    present-but-unidentifiable (empty dict / no model_type/architectures). A
    config that clearly identifies a non-Gemma2 model is trusted as-is.
    """
    data = _load_model_config_dict(model_path)
    if data is not None:
        if _config_is_gemma2(data):
            return True
        if _config_has_model_identity(data):
            return False
    return _path_looks_like_gemma2(model_path)
