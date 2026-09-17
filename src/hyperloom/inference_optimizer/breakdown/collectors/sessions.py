# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic collectors for ``session_breakdown.json``."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_unix
from hyperloom.common.timeutil import iso_z, now_iso

from ._common import _to_int


log = logging.getLogger(__name__)


def _detect_image_for_session(manifest: dict[str, Any]) -> str | None:
    """Resolve the container image for ``collect_session``."""
    manifest_image = manifest.get("image") if isinstance(manifest, dict) else None
    if isinstance(manifest_image, str) and manifest_image.strip():
        return manifest_image.strip()
    for var in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        val = (os.environ.get(var) or "").strip()
        if val:
            return val
    for marker in ("/etc/podinfo/image", "/etc/hyperloom-image"):
        try:
            p = Path(marker)
            if p.exists():
                txt = p.read_text(encoding="utf-8", errors="replace").strip()
                if txt:
                    return txt
        except OSError:
            continue
    try:
        cgroup = Path("/proc/1/cgroup")
        if cgroup.exists():
            for line in cgroup.read_text(encoding="utf-8", errors="replace").splitlines():
                if "docker" not in line and "containerd" not in line:
                    continue
                m = re.search(r"([0-9a-f]{12,64})", line)
                if m:
                    return f"unknown@{m.group(1)[:12]}"
    except OSError as exc:
        # /proc/1/cgroup may be unreadable; fall through to None.
        log.debug("cgroup-based image detection failed: %r", exc)
    return None


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
        reason = str(row.get("reason") or row.get("stop_reason") or row.get("exit_reason") or "").strip()
        ts = str(row.get("ts") or row.get("entered_ts") or "").strip()
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
def _collect_recovery(state: dict[str, Any]) -> dict[str, Any]:
    """Project SharedState's crash / interruption / resume signals."""
    crash_count = _to_int(state.get("crash_count")) or 0
    crash_ts_iso: list[str] = []
    raw_ts = state.get("crash_timestamps")
    if isinstance(raw_ts, list):
        for t in raw_ts:
            try:
                crash_ts_iso.append(datetime.fromtimestamp(float(t), tz=timezone.utc).isoformat())
            except (TypeError, ValueError, OSError, OverflowError):
                continue

    last_exc: dict[str, Any] | None = None
    lte = state.get("last_tick_exception")
    if isinstance(lte, dict) and lte:
        # Drop the large traceback; keep the compact postmortem header.
        last_exc = {
            "tick": lte.get("tick"),
            "ts": lte.get("ts"),
            "stage": lte.get("stage"),
            "agent": lte.get("agent"),
            "type": lte.get("type"),
            "message": (str(lte.get("message") or "")[:500] or None),
        }

    resume_pending = bool(state.get("resume_pending_revalidation"))
    degraded = bool(state.get("degraded_mode"))
    recovered = bool(crash_count > 0 or crash_ts_iso or resume_pending or last_exc)
    return {
        "recovered": recovered,
        "crash_count": crash_count,
        "crash_timestamps": crash_ts_iso,
        "degraded_mode": degraded,
        "resume_pending_revalidation": resume_pending,
        "last_tick_exception": last_exc,
    }


def collect_session(
    session_dir: Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the session-identification + lifecycle section."""
    start_ts = str(state.get("start_ts") or manifest.get("created_at_utc") or "")
    stop_reason = str(state.get("stop_reason") or "").strip()
    close_stop_reason, close_ts = _close_phase_stop_reason(state, leg_start_ts=_leg_start_ts(state, start_ts))
    if _should_use_close_stop_reason(stop_reason, close_stop_reason):
        stop_reason = close_stop_reason
    ended_at_utc = ""
    if _session_has_ended(stop_reason):
        # ``stop_ts`` is stamped once, when the reason is written, so a re-export of a finished session keeps
        # reporting the same end.
        ended_at_utc = _first_recorded_end(state.get("stop_ts"), close_ts) or now_iso(timespec="seconds")
    image = _detect_image_for_session(manifest)
    if image is None:
        warnings.append("image: not configured (set HYPERLOOM_IMAGE env var)")
    section = {
        "session_id": str(state.get("session_id") or manifest.get("session_id") or ""),
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
        "user_data_path": str(
            manifest.get("user_data_path") or state.get("user_data_path") or os.environ.get("USER_DATA_PATH") or ""
        ),
        "tick_count": int(state.get("tick") or 0),
        # Crash / interruption / resume history.
        "recovery": _collect_recovery(state),
    }
    section["elapsed_minutes"] = session_elapsed_minutes(section)
    return section


# Workload
def collect_workload(
    state: dict[str, Any],
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
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
def collect_model_info(
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the ``model_info`` section (state.model_info passthrough)."""
    info = state.get("model_info")
    return dict(info) if isinstance(info, dict) else {}
