# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Session close-out projection for SBD V6.

``close`` is a top-level V6 key rather than a timeline event: it describes how
the session was wrapped up, not a stage that competes for wall-clock with the
others.

**This collector runs twice, and the first pass is deliberately partial.**
``session_breakdown`` is step 2 of the CLOSE sequencer
(``orchestrator/phases/close.py``), so the breakdown written there can only
describe the close-out as far as itself: ``langfuse_flush`` /
``artifact_package`` / ``ndjson_drain`` / ``done`` have not happened, and
``close_sequence_done`` is still false. That snapshot honestly reports
``degraded``. The sequencer's last act calls
:func:`~..exporter.patch_breakdown_close`, which recomputes this section
against the finished ``state.json`` and splices it back in, so the breakdown a
reader finds on disk describes the whole sequence.

``degraded`` therefore means "the record of the close-out is incomplete",
**not** "the close-out failed" — it is what a session killed before the
sequencer finished leaves behind. A reader wanting to know whether a step
genuinely failed must look at ``steps[].status``; the absence of a step is not
evidence against it. ``langfuse_flush`` in particular only ever records a step
when it fails, so its silence is success.

Two quirks of the producer are worth knowing before reading ``steps``:
``sequencer_started`` is a marker recorded once as ``running`` and never
settled, and ``fact_finalize`` is emitted by the sequencer but absent from the
V6 field design's ``step`` enum. Both are handled below; neither is a defect
in the session being reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

_ESCALATED_STOP_REASON = "robustness_escalated"


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
) -> dict[str, Any]:
    """Project the session close-out into the V6 ``close`` key.

    Args:
        session_dir (Path): Absolute session root.
        state (Any): The V5 ``state.json`` mapping.
        critic_robustness (Any): The V5 ``critic_robustness`` section, whose
            ``robustness_signals`` are already in the V6 signal shape.
        warnings (list[str]): V6 warning sink (mutated in place).

    Returns:
        dict[str, Any]: The ``close`` object. Always a full object — unlike a
        timeline event, ``close`` has a fixed place in the payload, so an
        un-closed session reports ``status: "failed"`` with empty steps rather
        than vanishing.
    """
    state = _mapping(state)
    session_dir = Path(session_dir)
    steps = _collect_steps(state)
    sequence_done = bool(state.get("close_sequence_done"))

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

    reports_dir = session_dir / "reports"
    return {
        "status": status,
        "start_time": start_time,
        "end_time": end_time,
        "close_sequence_done": sequence_done,
        "steps": steps,
        "robustness": {
            "escalated": str(state.get("stop_reason") or "").strip().lower() == _ESCALATED_STOP_REASON,
            "signals": _dict_rows(_mapping(critic_robustness).get("robustness_signals")),
        },
        "artifacts": {
            "final_json_path": _existing_rel(session_dir, reports_dir / "final.json"),
            "final_md_path": _existing_rel(session_dir, reports_dir / "final.md"),
            # The breakdown is being written right now, so its own presence
            # cannot be tested; the path is reported unconditionally. The
            # filename is spelled out rather than imported from ``exporter``,
            # which imports this package.
            "session_breakdown_path": "session_breakdown.json",
            "artifact_package_path": _artifact_package_path(steps, session_dir),
        },
    }
