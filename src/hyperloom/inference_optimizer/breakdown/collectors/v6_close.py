# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Session close-out for SBD V6.

``close`` is a top-level V6 key rather than a timeline event: it describes how
the session was wrapped up, not a stage that competes for wall-clock with the
others. That fixed position is the point — a timeline event vanishes when it
does not happen, whereas a session that never closed cleanly is exactly the
case this key has to be able to state.

The CLOSE sequencer records the close-out as it performs it (see
:mod:`..recorder.close_out`), and this collector's job is to put that recording
on the wire. Two things are still done here rather than at author time:
``robustness.signals`` is joined in from ``critic_robustness``, where the
signals are already recorded once and are not worth recording twice, and the
step vocabulary is checked so a producer that starts emitting a new step name
surfaces as a warning instead of passing unnoticed.

**The section is written twice, and the first pass is deliberately partial.**
``session_breakdown`` is itself a step in the middle of the sequence, so when
the breakdown is written the steps after it have not run yet. The recording
reports ``running`` at that point — it is the sequencer's own last act that
records a verdict — and the sequencer then calls
:func:`~..exporter.patch_breakdown_close` to splice the settled section back
in. So ``running`` on a breakdown found on disk means the process died during
its close-out, which is a fact about the session rather than about the record.

A reader wanting to know whether a step genuinely failed must look at
``steps[].status``; the absence of a step is not evidence against it.
``langfuse_flush`` in particular only ever records a step when it fails, so its
silence is success.

The projection below is the fallback for a session with no recorded close
fragments, and is retained only for the length of the migration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..recorder.close_out import ESCALATED_STOP_REASON, SESSION_BREAKDOWN_PATH
from ._common import (
    _dict_rows,
    _mapping,
    _parse_iso_unix,
    _rel,
)


# Steps whose presence with a non-failed status means the close-out is on
# track. ``running`` is not among them: a step still running when the
# breakdown was written never reported an outcome.
_SETTLED_STATUSES = frozenset({"done", "skipped"})

# ``sequencer_started`` is a marker, not a unit of work: the sequencer records
# it once as ``running`` on entry and never revisits it, so it has no terminal
# status to wait for. Treating it as unsettled would make ``succeeded``
# unreachable by construction, no matter how cleanly the session closed.
_MARKER_STEPS = frozenset({"sequencer_started"})

# The step vocabulary the CLOSE sequencer actually emits. ``fact_finalize`` is
# in the runtime but was missing from the V6 field design, which is a gap in
# the contract rather than in the producer; unknown names are passed through
# and warned about so drift surfaces instead of being silently dropped.
_KNOWN_STEPS = frozenset(
    {
        "sequencer_started",
        "geak_rebench_drain",
        "fact_finalize",
        "report",
        "session_breakdown",
        "langfuse_flush",
        "artifact_package",
        "ndjson_drain",
        "done",
    }
)

# The status vocabulary the sequencer actually writes. A word outside it is
# passed through unchanged — inventing ``done`` for something spelled
# differently is the one failure mode this key cannot afford — but it is also
# warned about, because an unrecognized status counts as unsettled and would
# otherwise pin ``close.status`` to ``degraded`` with nothing to explain why.
_KNOWN_STATUSES = frozenset({"running", "done", "failed", "skipped"})

# The section's own status while the sequencer is still working through the
# steps. Not a verdict: it is what stands until the sequencer records one.
_STATUS_RUNNING = "running"


def _close_step(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one recorded close step to the V6 shape."""
    step = str(row.get("step") or "")
    status = str(row.get("status") or "").strip().lower()
    task_id = str(row.get("task_id") or "") or None
    detail = str(row.get("detail") or "") or None
    return {
        "step": step,
        # An unrecognized status is passed through rather than coerced into
        # the enum: inventing ``done`` for something a producer spelled
        # differently would be the one failure mode this key cannot afford.
        "status": status,
        "ts": str(row.get("ts") or ""),
        "task_id": task_id,
        "detail": detail,
    }


def _collect_steps(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Gather ``close_steps`` from every phase_history row, oldest first.

    ``_record_close_step`` appends to ``phase_history[-1].evidence``, so in the
    ordinary case every step is on the CLOSE row. Every row is swept anyway
    because a phase transition landing mid-CLOSE would split the sequence
    across two rows, and half a close-out is worse than a slightly wider scan.
    """
    steps: list[dict[str, Any]] = []
    for row in _dict_rows(state.get("phase_history")):
        evidence = _mapping(row.get("evidence"))
        for raw in _dict_rows(evidence.get("close_steps")):
            steps.append(_close_step(raw))
    steps.sort(key=lambda step: (_parse_iso_unix(step["ts"]) is None, _parse_iso_unix(step["ts"]) or 0.0))
    return steps


def _close_entry_ts(state: dict[str, Any]) -> str:
    """Return the ts of the last transition into CLOSE, or ``""``."""
    for row in reversed(_dict_rows(state.get("phase_history"))):
        if str(row.get("to_phase") or "").strip().upper() == "CLOSE":
            return str(row.get("ts") or "")
    return ""


def _existing_rel(session_dir: Path, path: Path) -> str | None:
    """Return ``path`` relative to the session, or ``None`` when absent."""
    try:
        if not path.exists():
            return None
    except OSError:
        return None
    return _rel(path, session_dir)


def _artifact_package_path(steps: list[dict[str, Any]], session_dir: Path) -> str | None:
    """Read the package location off the ``artifact_package`` step's detail.

    The step reuses ``detail`` for both the path (on success) and the reason
    (on skip/failure), so only a ``done`` row is read.
    """
    row = next(
        (step for step in reversed(steps) if step["step"] == "artifact_package" and step["status"] == "done"),
        None,
    )
    if row is None or not row["detail"]:
        return None
    # The package is written to ``/workspace``, which is normally outside the
    # session; ``_rel`` falls back to the absolute path in that case.
    return _rel(Path(row["detail"]), session_dir)


def collect_v6_close(
    session_dir: Path,
    state: Any,
    critic_robustness: Any,
    warnings: list[str],
    recorded: Any = None,
) -> dict[str, Any]:
    """Build the V6 ``close`` key, preferring the sequencer's own recording.

    Args:
        session_dir (Path): Absolute session root.
        state (Any): The V5 ``state.json`` mapping, read only by the fallback.
        critic_robustness (Any): The V5 ``critic_robustness`` section, whose
            ``robustness_signals`` are already in the V6 signal shape.
        warnings (list[str]): V6 warning sink (mutated in place).
        recorded (Any): The recorder's ``close`` fragment, when present. It is
            authoritative: the sequencer states what it did as it does it, and
            the projection cannot improve on that.

    Returns:
        dict[str, Any]: The ``close`` object. Always a full object — unlike a
        timeline event, ``close`` has a fixed place in the payload, so an
        un-closed session reports ``status: "failed"`` with empty steps rather
        than vanishing.
    """
    session_dir = Path(session_dir)
    signals = _dict_rows(_mapping(critic_robustness).get("robustness_signals"))
    if isinstance(recorded, dict) and recorded:
        return _recorded_close(recorded, signals=signals, warnings=warnings)
    return _projected_close(session_dir, _mapping(state), signals=signals, warnings=warnings)


def _recorded_close(
    recorded: dict[str, Any],
    *,
    signals: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    """Put the recorded close-out on the wire.

    The verdict, the timestamps, the artifact paths and the escalation are all
    read straight through: each was stated by the step that knew it. Only the
    signals are joined in, and only the step vocabulary is checked.
    """
    steps = [_close_step(row) for row in _dict_rows(recorded.get("steps"))]
    _warn_unknown_vocabulary(steps, warnings)
    artifacts = _mapping(recorded.get("artifacts"))
    robustness = _mapping(recorded.get("robustness"))
    write_back = recorded.get("kb_write_back")
    close: dict[str, Any] = {
        # A fragment with no status was opened by a producer that then failed
        # to write one; it is still an un-settled close-out.
        "status": str(recorded.get("status") or "") or _STATUS_RUNNING,
        "start_time": str(recorded.get("start_time") or ""),
        "end_time": str(recorded.get("end_time") or ""),
        "close_sequence_done": bool(recorded.get("close_sequence_done")),
        "steps": steps,
        "robustness": {
            "escalated": bool(robustness.get("escalated")),
            # Recorded alongside the verdict so a reader can check the
            # escalation against the reason it was drawn from.
            "stop_reason": str(recorded.get("stop_reason") or ""),
            "signals": signals,
        },
        "artifacts": {
            "final_json_path": artifacts.get("final_json_path") or None,
            "final_md_path": artifacts.get("final_md_path") or None,
            "session_breakdown_path": artifacts.get("session_breakdown_path") or SESSION_BREAKDOWN_PATH,
            "artifact_package_path": artifacts.get("artifact_package_path") or None,
        },
    }
    # Absent when the session never attempted a publication. Left out rather
    # than emitted empty: the key is the record that it was tried, so an empty
    # one would claim an attempt that never happened.
    if isinstance(write_back, dict) and write_back:
        close["kb_write_back"] = write_back
    # Same rule as the publication above: absent when the close-out never got
    # far enough to snapshot the progress, because an empty curve and a session
    # that made no progress are different claims.
    progress = recorded.get("roofline_progress")
    if isinstance(progress, dict) and progress:
        close["roofline_progress"] = progress
    baseline = recorded.get("baseline_progress")
    if isinstance(baseline, dict) and baseline:
        close["baseline_progress"] = baseline
    return close


def _warn_unknown_vocabulary(steps: list[dict[str, Any]], warnings: list[str]) -> None:
    """Warn about step names and statuses outside the known vocabulary."""
    unknown = sorted({step["step"] for step in steps if step["step"] and step["step"] not in _KNOWN_STEPS})
    if unknown:
        warnings.append(f"v6.close: unrecognized close step(s) {', '.join(unknown)}; passed through unchanged")
    unknown_statuses = sorted(
        {step["status"] for step in steps if step["status"] and step["status"] not in _KNOWN_STATUSES}
    )
    if unknown_statuses:
        warnings.append(
            f"v6.close: unrecognized close step status(es) {', '.join(unknown_statuses)}; "
            "passed through unchanged and counted as unsettled"
        )


def _projected_close(
    session_dir: Path,
    state: dict[str, Any],
    *,
    signals: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    """Derive the close-out from ``state.json`` for a session that recorded none.

    Retained for the length of the migration, and for the ``cli.finally``
    safety net that writes a breakdown without the sequencer having run. It
    cannot tell a step that had not happened yet from one that never will, so
    it reports ``degraded`` for the healthy mid-sequence case; that limitation
    is why the sequencer now records its verdict instead.
    """
    steps = _collect_steps(state)
    sequence_done = bool(state.get("close_sequence_done"))
    _warn_unknown_vocabulary(steps, warnings)

    failed = [step for step in steps if step["status"] == "failed"]
    unsettled = [
        step
        for step in steps
        if step["step"] not in _MARKER_STEPS and step["status"] not in _SETTLED_STATUSES and step["status"] != "failed"
    ]
    if not steps:
        # No close step at all: either the session died before CLOSE, or the
        # breakdown is a cli.finally safety net written outside the sequencer.
        status = "failed"
    elif failed:
        status = "degraded"
    elif sequence_done and not unsettled:
        status = "succeeded"
    else:
        # The expected steady state — see the module docstring. Not a warning:
        # it is the designed behaviour of the write order, and warning on
        # every healthy session would drown the ones that matter.
        status = "degraded"

    start_time = steps[0]["ts"] if steps else _close_entry_ts(state)
    end_time = steps[-1]["ts"] if steps else ""
    stop_reason = str(state.get("stop_reason") or "").strip()

    reports_dir = session_dir / "reports"
    return {
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "close_sequence_done": sequence_done,
        "steps": steps,
        "robustness": {
            "escalated": stop_reason.lower() == ESCALATED_STOP_REASON,
            "stop_reason": stop_reason,
            "signals": signals,
        },
        "artifacts": {
            "final_json_path": _existing_rel(session_dir, reports_dir / "final.json"),
            "final_md_path": _existing_rel(session_dir, reports_dir / "final.md"),
            # The breakdown is being written right now, so its own presence
            # cannot be tested; the path is reported unconditionally.
            "session_breakdown_path": SESSION_BREAKDOWN_PATH,
            "artifact_package_path": _artifact_package_path(steps, session_dir),
        },
    }
