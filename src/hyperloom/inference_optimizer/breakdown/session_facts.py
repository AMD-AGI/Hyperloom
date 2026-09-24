# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Session facts shared by the recorder and the collector fallback.

Architecture and recovery are written at author time and re-projected at
export when the fragment is missing. The two paths used to copy the field
lists; this module is the one list they both read.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

ARCHITECTURE_FIELDS = (
    "model_family",
    "model_type",
    "architectures",
    "attention_type",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_size",
    "intermediate_size",
    "max_position_embeddings",
    "vocab_size",
    "torch_dtype",
    "kv_cache_dtype",
    "quantization",
    "is_moe",
    "num_experts",
    "num_experts_per_tok",
    "has_shared_expert",
    "num_shared_experts",
)


def _field(state: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping or from an object attribute."""
    if isinstance(state, Mapping):
        return state.get(name, default)
    return getattr(state, name, default)


def architecture_block(model_info: Any, *, model_class: str = "") -> dict[str, Any]:
    """The structural model block, or ``{}`` when nothing is known."""
    info = dict(model_info or {}) if isinstance(model_info, Mapping) else {}
    resolved_class = str(model_class or "").strip()
    if not resolved_class and info:
        resolved_class = "moe" if bool(info.get("is_moe")) else "dense"
    if not info and not resolved_class:
        return {}
    architecture: dict[str, Any] = {"model_class": resolved_class}
    for field in ARCHITECTURE_FIELDS:
        if field in info:
            architecture[field] = info[field]
    return architecture


def recovery_block(state: Any) -> dict[str, Any]:
    """Crash / interruption / resume history from live state or ``state.json``."""
    crash_count = int(_field(state, "crash_count", 0) or 0)
    crash_timestamps: list[str] = []
    for raw in _field(state, "crash_timestamps", None) or []:
        try:
            crash_timestamps.append(datetime.fromtimestamp(float(raw), tz=timezone.utc).isoformat())
        except (TypeError, ValueError, OSError, OverflowError):
            continue
    last_exception: dict[str, Any] | None = None
    raw_exception = _field(state, "last_tick_exception", None)
    if isinstance(raw_exception, Mapping) and raw_exception:
        last_exception = {
            "tick": raw_exception.get("tick"),
            "ts": raw_exception.get("ts"),
            "stage": raw_exception.get("stage"),
            "agent": raw_exception.get("agent"),
            "type": raw_exception.get("type"),
            "message": (str(raw_exception.get("message") or "")[:500] or None),
        }
    resume_pending = bool(_field(state, "resume_pending_revalidation", False))
    return {
        "recovered": bool(crash_count > 0 or crash_timestamps or resume_pending or last_exception),
        "crash_count": crash_count,
        "crash_timestamps": crash_timestamps,
        "degraded_mode": bool(_field(state, "degraded_mode", False)),
        "resume_pending_revalidation": resume_pending,
        "last_tick_exception": last_exception,
    }


def grading_block(state: Any) -> dict[str, Any]:
    """The axis this session was configured to grade on, and the band it grades under.

    Prefers ``state.grading`` recorded at seed. A mapping snapshot with no
    recorded axis is absence, not the exporting process's environment. A live
    object without a recorded axis still uses :func:`resolved_grading`.
    """
    from hyperloom.common.perf_metric import GRADED_INTVTY, GRADED_OUTPUT
    from hyperloom.orchestrator.state.shared_state import resolved_grading

    recorded = _field(state, "grading", None)
    recorded = recorded if isinstance(recorded, dict) else {}
    objective = str(recorded.get("objective") or "").strip()
    if objective:
        noise_pct = recorded.get("noise_pct")
        on_intvty = objective == GRADED_INTVTY
        noise = float(noise_pct) if isinstance(noise_pct, (int, float)) else None
    elif isinstance(state, Mapping):
        return {}
    else:
        on_intvty, noise = resolved_grading(state)
    return {
        "benchmark_mode": str(_field(state, "benchmark_mode", "") or "").strip() or "synthetic",
        "objective": GRADED_INTVTY if on_intvty else GRADED_OUTPUT,
        "tput_guard": {"enabled": on_intvty, "noise_pct": noise},
    }


def workload_signature(config: Mapping[str, Any]) -> str:
    """The workload contract digest for ``config``, empty when it is unknown."""
    fields = {name: config.get(name) for name in ("conc", "isl", "osl", "precision", "tp")}
    if not any(str(value or "").strip() for value in fields.values()):
        return ""
    from hyperloom.orchestrator.actions.executors._canonical_fingerprint import (
        workload_signature as digest,
    )

    return digest(**{name: value for name, value in fields.items() if value is not None})
