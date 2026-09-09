# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic collectors for ``session_breakdown.json``.

Each ``collect_<section>`` is a pure function over ``session_dir`` /
``state`` / ``manifest`` returning its schema section (see :mod:`.schema`).
Collectors never mutate state, fabricate values, or raise — failures are
recorded in ``warnings`` and the section returns a best-effort partial.
"""

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
    """Resolve the container image for ``collect_session``.

    Prefers the manifest field (the spawn-time image), then falls back to the
    env / mount-point chain the manifest helper uses. Kept separate from
    :func:`manifest._detect_image` to avoid an import cycle.

    Resolution order: manifest ``image`` field → ``HYPERLOOM_IMAGE`` /
    ``CONTAINER_IMAGE`` / ``IMAGE`` env vars → known image marker files →
    a ``unknown@<short-cgroup-id>`` derived from ``/proc/1/cgroup``.

    Args:
        manifest (dict[str, Any]): The parsed ``manifest.json`` dict.

    Returns:
        str | None: The resolved container image reference, or ``None`` when
        no source yields a value.
    """
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
    """When the session's current run leg began.

    ``start_ts`` alone does not answer this. A resume re-anchors it only after
    a crash or a stop with a reason; a resume after a clean stop deliberately
    keeps it, so that ``--max-hours`` still counts from the original start.
    ``state.resumed_ts`` is stamped by every resume, so the later of the two is
    the boundary on both paths.

    Args:
        state (dict[str, Any]): Parsed ``state.json``.
        start_ts (str): The session's resolved start (see
            :func:`collect_session`).

    Returns:
        str: The later of the two timestamps, or whichever one is parseable.
    """
    resumed_ts = str(state.get("resumed_ts") or "")
    dated = [(to_unix(ts), ts) for ts in (start_ts, resumed_ts)]
    parseable = [(at, ts) for at, ts in dated if at is not None]
    if not parseable:
        return start_ts
    return max(parseable)[1]


def _close_phase_stop_reason(state: dict[str, Any], *, leg_start_ts: str) -> tuple[str, str]:
    """Recover terminal reason/time from the current leg's CLOSE transition (next-best when ``state.stop_reason`` wasn't mirrored).

    A resume clears ``state.stop_reason`` and ``stop_ts`` but cannot clear the
    previous leg's CLOSE row, and that row is not evidence about the leg
    running now: honouring it reports a live session as having stopped, for
    the reason it stopped last time. A row from before the leg boundary is
    skipped whole -- reason and timestamp -- because the timestamp is stamped
    as the session's end even when the reason itself is not adopted, and the
    scan carries on so a history written out of order can still be answered
    from a row that does belong to this leg.

    A row is only disqualified on comparable evidence. When either timestamp
    is missing or unparseable the row stands, since the whole point of the
    fallback is a session whose reason never reached the state file.

    Args:
        state (dict[str, Any]): Parsed ``state.json``.
        leg_start_ts (str): Start of the current leg (see
            :func:`_leg_start_ts`); ``""`` when the session recorded none.

    Returns:
        tuple[str, str]: ``(reason, ts)`` from the most recent CLOSE
        transition of the current leg, or ``("", "")`` when there is none.
    """
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
    """The first candidate that reads as a timestamp, canonicalised to ``...Z``.

    A value that does not parse is no more an end time than a missing one:
    passed through it lands in ``ended_at_utc`` verbatim and collapses the
    measured duration to zero, where the next candidate (or the export clock)
    still answers.

    Args:
        *candidates (Any): Recorded end timestamps, best evidence first.

    Returns:
        str: The first parseable candidate, or ``""`` when none is.
    """
    for value in candidates:
        if to_unix(value) is not None:
            return iso_z(value)
    return ""


def _session_has_ended(stop_reason: Any) -> bool:
    """Whether a stop reason marks the session as no longer running.

    Args:
        stop_reason (Any): Raw ``stop_reason`` from a state or session section.

    Returns:
        bool: ``True`` once a non-blank stop reason has been recorded.
    """
    return bool(str(stop_reason or "").strip())


def _measured_duration_seconds(start_ts: Any, ended_at_utc: Any, stop_reason: Any) -> int | None:
    """Seconds the session ran, or ``None`` when no window can be established.

    A finished session is measured to its recorded end; only one still running
    may be measured up to now, since extrapolating a finished session grows its
    duration on every re-export and reads as a plausible number rather than as
    missing evidence.

    Args:
        start_ts (Any): Start of the window (see :func:`collect_session` for
            which start that is across a resume).
        ended_at_utc (Any): Recorded end of the window, if any.
        stop_reason (Any): Terminal reason; a non-blank one means the session
            is no longer running.

    Returns:
        int | None: Whole seconds between start and end, or ``None``.
    """
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
    """Wall-clock minutes of the leg described by a resolved ``session`` section.

    Derived from the section's own timestamps rather than stored, so a section
    assembled from the live recorder's snapshot reports the same elapsed time
    as one built by :func:`collect_session`. ``session_meta`` measures the same
    window from the same fields; the two agree because both producers of the
    section carry those timestamps, not because either reads the other.

    Args:
        session_section (dict[str, Any]): A ``session`` section.

    Returns:
        float: Minutes elapsed, or ``0.0`` when no window can be established.
    """
    duration_s = _measured_duration_seconds(
        session_section.get("start_ts") or session_section.get("created_at_utc"),
        session_section.get("ended_at_utc"),
        session_section.get("stop_reason"),
    )
    return round(duration_s / 60.0, 2) if duration_s is not None else 0.0


def _should_use_close_stop_reason(stop_reason: str, close_stop_reason: str) -> bool:
    """Decide whether the CLOSE-phase stop reason should override the session's.

    Args:
        stop_reason: The session-level stop reason.
        close_stop_reason: The CLOSE-phase stop reason.

    Returns:
        ``True`` when the close reason is more specific — i.e. it is set and the
        session reason is empty, or the session merely timed out while the close
        reason did not.
    """
    if not close_stop_reason:
        return False
    if not stop_reason:
        return True
    return stop_reason == "time_exhausted" and close_stop_reason != "time_exhausted"


# Session metadata
def _collect_recovery(state: dict[str, Any]) -> dict[str, Any]:
    """Project SharedState's crash / interruption / resume signals.

    Folds crash / degraded-mode / pending-revalidation signals into the
    ``session.recovery`` block so a resumed run is not read as a clean monotonic
    one. Pure / best-effort: unparseable fields are skipped, never raised.

    Args:
        state (dict[str, Any]): Parsed ``state.json`` (SharedState-shaped).

    Returns:
        dict[str, Any]: The ``recovery`` block (see schema ``Recovery``).
    """
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
    """Collect the session-identification + lifecycle section.

    Merges identifiers and timing from ``state`` and ``manifest`` (state
    taking precedence on overlapping fields), resolves the container image,
    and stamps ``ended_at_utc`` from the recorded stop timestamp only once a
    ``stop_reason`` is present -- one the state file carries, or one recovered
    from a CLOSE transition belonging to the current leg (see
    :func:`_close_phase_stop_reason`). When no image can be detected a warning
    is appended.

    ``elapsed_minutes`` runs from ``state.start_ts``, the same anchor
    ``--max-hours`` is counted against, to the recorded end (or to now while
    the run is still going), so the two stay comparable. A resume re-anchors
    ``start_ts`` only when the previous leg crashed or stopped for a recorded
    reason; after a clean stop it keeps the original start, and the elapsed
    time then spans the gap between the legs -- as the budget does. The
    manifest's ``created_at_utc`` names the first launch either way, and is
    the fallback start only for a session that never recorded one.

    Args:
        session_dir (Path): Absolute session root.
        state (dict[str, Any]): Parsed ``state.json`` (SharedState-shaped).
        manifest (dict[str, Any]): Parsed ``manifest.json``.
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        dict[str, Any]: The session section (ids, timestamps, stop reason,
        elapsed minutes, host, image, code revision, pid, tick count, etc.).
    """
    start_ts = str(state.get("start_ts") or manifest.get("created_at_utc") or "")
    stop_reason = str(state.get("stop_reason") or "").strip()
    close_stop_reason, close_ts = _close_phase_stop_reason(state, leg_start_ts=_leg_start_ts(state, start_ts))
    if _should_use_close_stop_reason(stop_reason, close_stop_reason):
        stop_reason = close_stop_reason
    ended_at_utc = ""
    if _session_has_ended(stop_reason):
        # ``stop_ts`` is stamped once, when the reason is written, so a re-export
        # of a finished session keeps reporting the same end. The CLOSE
        # transition and the export clock are only next-best guesses.
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
        # USER_DATA_PATH root (the operator-chosen workspace base). Manifest is
        # snapshotted at session start; env is the in-process fallback.
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
    """Collect the workload-description section.

    Merges framework name / model / GPU / parallelism fields from ``state`` and
    ``manifest`` (state preferred) plus the workload knobs (``conc`` / ``isl``
    / ``osl`` / ``max_model_len`` / ``precision``) nested under
    ``manifest.workload``, and the optimization objective.

    Args:
        state (dict[str, Any]): Parsed ``state.json``.
        manifest (dict[str, Any]): Parsed ``manifest.json``.
        warnings (list[str]): Shared warnings list (unused here but kept for a
            uniform collector signature).

    Returns:
        dict[str, Any]: The workload section with coerced numeric knobs and a
        defaulted ``objective`` mapping.
    """
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
    """Collect the ``model_info`` section (state.model_info passthrough).

    The summary is computed once at launch (``cli_bootstrap`` →
    ``summarize_model_config``) and persisted on ``state.model_info``, so the
    breakdown just mirrors it verbatim. Returns ``{}`` when the field is absent
    (sessions whose state predates it) or empty (non-transformers models such
    as diffusion checkpoints, where the config.json could not be parsed); the
    frontend treats an empty object as "model info unavailable".

    Args:
        state (dict[str, Any]): Parsed ``state.json``.
        warnings (list[str]): Shared warnings list (unused here but kept for a
            uniform collector signature).

    Returns:
        dict[str, Any]: The model_info object, or ``{}`` when unavailable.
    """
    info = state.get("model_info")
    return dict(info) if isinstance(info, dict) else {}
