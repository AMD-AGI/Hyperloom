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
on the wire. One thing is still done here
rather than at author time: the step vocabulary is checked so a producer that
starts emitting a new step name surfaces as a warning instead of passing
unnoticed.

What the agent itself raised is not here. It is recorded per turn by
:mod:`..recorder.robustness_out` and exported as the top-level ``robustness``
key; this block carries only the close-out's own verdict about it.

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

from typing import Any

from ..recorder.close_out import SESSION_BREAKDOWN_PATH
from ._common import (
    _dict_rows,
    _mapping,
)


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


def collect_v6_close(
    warnings: list[str],
    recorded: Any = None,
    robustness: Any = None,
) -> dict[str, Any]:
    """Put the sequencer's own recording of the close-out on the wire.

    Args:
        warnings (list[str]): V6 warning sink (mutated in place).
        recorded (Any): The recorder's ``close`` fragment. The sequencer states
            what it did as it does it, so this is the only account of the
            close-out there is.
        robustness (Any): The assembled robustness view, whose turns join the
            close-out's robustness block. The agent's turns and the verdict
            drawn from them are one account of one thing, and reading them
            required knowing to look in two places.

    Returns:
        dict[str, Any]: The ``close`` object. Always a full object — unlike a
        timeline event, ``close`` has a fixed place in the payload, so an
        un-closed session reports ``status: "failed"`` with empty steps rather
        than vanishing.
    """
    if isinstance(recorded, dict) and recorded:
        return _recorded_close(recorded, warnings=warnings, robustness=robustness)
    return _unclosed(robustness)


def _robustness_block(recorded: dict[str, Any], robustness: Any) -> dict[str, Any]:
    """Assemble the close-out's robustness block from both of its sources.

    Args:
        recorded (dict[str, Any]): The close fragment, holding the escalation
            verdict and the findings read at close time.
        robustness (Any): The assembled robustness view, holding the turns.

    Returns:
        dict[str, Any]: The block. ``findings`` is present only when the ladder
            wrote some, because an empty list would claim a ladder that ran and
            found nothing.
    """
    block = _mapping(recorded.get("robustness"))
    view = _mapping(robustness)
    assembled: dict[str, Any] = {
        "escalated": bool(block.get("escalated")),
        # Recorded alongside the verdict so a reader can check the escalation
        # against the reason it was drawn from.
        "stop_reason": str(recorded.get("stop_reason") or ""),
        "turns": _dict_rows(view.get("turns")),
    }
    if block.get("findings") is not None:
        assembled["findings"] = _dict_rows(block.get("findings"))
        assembled["findings_total"] = int(block.get("findings_total") or 0)
    return assembled


def _recorded_close(
    recorded: dict[str, Any],
    *,
    warnings: list[str],
    robustness: Any = None,
) -> dict[str, Any]:
    """Put the recorded close-out on the wire.

    The verdict, the timestamps, the artifact paths and the escalation are all
    read straight through: each was stated by the step that knew it. Only the
    only the step vocabulary is checked.
    """
    steps = [_close_step(row) for row in _dict_rows(recorded.get("steps"))]
    _warn_unknown_vocabulary(steps, warnings)
    artifacts = _mapping(recorded.get("artifacts"))
    write_back = recorded.get("kb_write_back")
    close: dict[str, Any] = {
        # A fragment with no status was opened by a producer that then failed
        # to write one; it is still an un-settled close-out.
        "status": str(recorded.get("status") or "") or _STATUS_RUNNING,
        "start_time": str(recorded.get("start_time") or ""),
        "end_time": str(recorded.get("end_time") or ""),
        "close_sequence_done": bool(recorded.get("close_sequence_done")),
        "steps": steps,
        "robustness": _robustness_block(recorded, robustness),
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
    candidate = recorded.get("geak_candidate")
    if isinstance(candidate, dict) and candidate:
        close["geak_candidate"] = candidate
    recipe = recorded.get("final_recipe")
    if isinstance(recipe, dict) and recipe:
        close["final_recipe"] = recipe
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


def _unclosed(robustness: Any) -> dict[str, Any]:
    """The close-out of a session that never recorded one.

    Reached by a run that died before CLOSE, and by the ``cli.finally`` safety
    net that writes a breakdown without the sequencer having run. ``close`` has
    a fixed place in the payload, so this reports the absence rather than
    omitting the key: no step happened, so there is nothing to summarize and
    ``failed`` is the whole of what can be said.

    The export used to derive a verdict here by re-reading ``phase_history``
    and probing the reports directory, and could not tell a step that had not
    happened yet from one that never would -- so it called a healthy
    mid-sequence session ``degraded``, which is why the sequencer records its
    own verdict now.

    Args:
        robustness (Any): The assembled robustness view. Its turns are the
            agent's own record and survive the close-out never running; the
            findings do not, because they are read at close time.

    Returns:
        dict[str, Any]: The ``close`` object for a session with no close-out.
    """
    return {
        "status": "failed",
        "start_time": "",
        "end_time": "",
        "close_sequence_done": False,
        "steps": [],
        "robustness": {
            "escalated": False,
            "stop_reason": "",
            "turns": _dict_rows(_mapping(robustness).get("turns")),
        },
        "artifacts": {
            "final_json_path": None,
            "final_md_path": None,
            # The breakdown is being written right now, so its own presence
            # cannot be tested; the path is reported unconditionally.
            "session_breakdown_path": SESSION_BREAKDOWN_PATH,
            "artifact_package_path": None,
        },
    }
