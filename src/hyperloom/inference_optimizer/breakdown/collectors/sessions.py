# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic collectors for ``session_breakdown.json``."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom_common.coerce import first_str, to_unix
from hyperloom.common.timeutil import iso_z, now_iso

from ._common import _to_int
from ..session_facts import recovery_block


log = logging.getLogger(__name__)


def _leg_start_ts(state: dict[str, Any], start_ts: str) -> str:
    """When the session's current run leg began."""
    resumed_ts = str(state.get("resumed_ts") or "")
    dated = [(to_unix(ts), ts) for ts in (start_ts, resumed_ts)]
    parseable = [(at, ts) for at, ts in dated if at is not None]
    if not parseable:
        return start_ts
    return max(parseable)[1]


def _close_phase_stop_reason(state: dict[str, Any], *, leg_start_ts: str) -> tuple[str, str]:
    """Recover terminal reason/time from the current leg's CLOSE transition (next-best when ``state.stop_reason`` wasn't mirrored)."""
    history = state.get("phase_history") or []
    if not isinstance(history, list):
        return "", ""
    leg_start = to_unix(leg_start_ts)
    for row in reversed(history):
        if not isinstance(row, dict):
            continue
        if str(row.get("to_phase") or "").strip().upper() != "CLOSE":
            continue
        reason = str(row.get("reason") or "").strip()
        ts = str(row.get("ts") or "").strip()
        closed_at = to_unix(ts)
        if leg_start is not None and closed_at is not None and closed_at < leg_start:
            continue
        return reason, ts
    return "", ""


def _first_recorded_end(*candidates: Any) -> str:
    """The first candidate that reads as a timestamp, canonicalised to ``...Z``."""
    for value in candidates:
        if to_unix(value) is not None:
            return iso_z(value)
    return ""


def _session_has_ended(stop_reason: Any) -> bool:
    """Whether a stop reason marks the session as no longer running."""
    return bool(str(stop_reason or "").strip())


def _measured_duration_seconds(start_ts: Any, ended_at_utc: Any, stop_reason: Any) -> int | None:
    """Seconds the session ran, or ``None`` when no window can be established."""
    start = to_unix(start_ts)
    if start is None:
        return None
    end = to_unix(ended_at_utc)
    if end is None and not _session_has_ended(stop_reason):
        end = datetime.now(timezone.utc).timestamp()
    if end is None or end <= start:
        return None
    return int(round(end - start))


def session_elapsed_minutes(session_section: dict[str, Any]) -> float:
    """Wall-clock minutes of the leg described by a resolved ``session`` section."""
    duration_s = _measured_duration_seconds(
        session_section.get("start_ts") or session_section.get("created_at_utc"),
        session_section.get("ended_at_utc"),
        session_section.get("stop_reason"),
    )
    return round(duration_s / 60.0, 2) if duration_s is not None else 0.0


def _should_use_close_stop_reason(stop_reason: str, close_stop_reason: str) -> bool:
    """Decide whether the CLOSE-phase stop reason should override the session's."""
    if not close_stop_reason:
        return False
    if not stop_reason:
        return True
    return stop_reason == "time_exhausted" and close_stop_reason != "time_exhausted"


# Session metadata
def collect_session(
    session_dir: Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the session-identification + lifecycle section."""
    start_ts = first_str(state.get("start_ts"), manifest.get("created_at_utc"))
    stop_reason = str(state.get("stop_reason") or "").strip()
    close_stop_reason, close_ts = _close_phase_stop_reason(state, leg_start_ts=_leg_start_ts(state, start_ts))
    if _should_use_close_stop_reason(stop_reason, close_stop_reason):
        stop_reason = close_stop_reason
    ended_at_utc = ""
    if _session_has_ended(stop_reason):
        # ``stop_ts`` is stamped once, when the reason is written, so a re-export of a finished session keeps
        # reporting the same end.
        ended_at_utc = _first_recorded_end(state.get("stop_ts"), close_ts) or now_iso(timespec="seconds")
    image = str(manifest.get("image") or "").strip() or None
    if image is None:
        warnings.append("image: manifest.json records no image")
    section = {
        "session_id": first_str(state.get("session_id"), manifest.get("session_id")),
        "claw_session_id": manifest.get("claw_session_id") or state.get("claw_session_id"),
        "sandbox_user_id": manifest.get("sandbox_user_id") or state.get("sandbox_user_id"),
        "created_at_utc": manifest.get("created_at_utc") or start_ts,
        "start_ts": start_ts,
        "ended_at_utc": ended_at_utc,
        "stop_reason": stop_reason,
        "max_minutes": int(state.get("max_minutes") or manifest.get("max_minutes") or 0),
        "elapsed_minutes": 0.0,
        "host": str(manifest.get("host") or ""),
        "image": image,
        "code_revision": str(manifest.get("code_revision") or ""),
        "pid": int(manifest.get("pid") or 0),
        "session_dir": str(session_dir),
        # USER_DATA_PATH root (the operator-chosen workspace base).
        "user_data_path": first_str(
            manifest.get("user_data_path"), state.get("user_data_path"), os.environ.get("USER_DATA_PATH")
        ),
        "tick_count": int(state.get("tick") or 0),
        # Crash / interruption / resume history.
        "recovery": recovery_block(state),
    }
    section["elapsed_minutes"] = session_elapsed_minutes(section)
    return section


# Workload
def collect_workload(state: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Collect the workload-description section."""
    wl = manifest.get("workload") or {}
    return {
        "framework_name": str(state.get("framework") or manifest.get("framework") or ""),
        "framework_version": str(manifest.get("framework_version") or ""),
        "model_name": str(state.get("model_name") or manifest.get("model_name") or ""),
        "model_path": str(state.get("model_path") or manifest.get("model_path") or ""),
        "model_class": str(state.get("model_class") or ""),
        "gpu_type": str(state.get("gpu_type") or manifest.get("gpu_type") or ""),
        "tp": _to_int(manifest.get("tp")),
        "conc": _to_int(wl.get("conc")),
        "isl": _to_int(wl.get("isl")),
        "osl": _to_int(wl.get("osl")),
        "max_model_len": _to_int(wl.get("max_model_len")),
        "precision": str(wl.get("precision") or ""),
        "objective": dict(manifest.get("objective") or {"kind": "time_only", "value": None}),
    }


# Model basics
def collect_model_info(state: dict[str, Any]) -> dict[str, Any]:
    """Collect the ``model_info`` section (state.model_info passthrough)."""
    info = state.get("model_info")
    return dict(info) if isinstance(info, dict) else {}
