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
