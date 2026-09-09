# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Author-time recording of the SBD v6 ``close`` section.

The CLOSE sequencer states what it did as it does it. The verdict cannot be
derived at export time, because ``session_breakdown`` is itself a step in the
middle of the sequence, so the sequencer's own last act is the only thing that
knows the sequence finished. Until it writes, the section stands at
``running``, leaving three states distinguishable on the wire: no ``close``
fragment (the close-out was never reached), ``running`` (the mid-sequence
snapshot, or a process that died), and any other status (the verdict).

Recording is best-effort: a failure here must never propagate into the
wind-down it is describing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from hyperloom.common.jsonio import read_jsonl
from hyperloom.common.timeutil import now_iso

from .recorder import recorder_for
from .trace import trace_skip

log = logging.getLogger(__name__)

SECTION = "close"
STEP_SECTION = "close_step"
WRITE_BACK_SECTION = "close_write_back"
WRITE_BACK_ATTEMPT_SECTION = "close_write_back_attempt"
PRODUCER = "coordinator"

#: Stable ``result_type`` codes for the Recipe KB publication, set by the
#: publisher at each exit so no reader substring-matches a free-text reason.
RESULT_WRITTEN = "written"
RESULT_KB_DISABLED = "kb_disabled"
RESULT_AGENTX_BLOCKED = "agentx_blocked"
RESULT_INVALID_SCOPE = "invalid_scope"
RESULT_CONFIGURATION_FAILED = "configuration_failed"
RESULT_TRANSPORT_FAILED = "transport_failed"
RESULT_NO_NEW_KEEP = "no_new_keep_or_pure_warm_replay"
RESULT_INVALID_THROUGHPUT = "invalid_throughput"
RESULT_MISSING_THROUGHPUT = "missing_throughput"
RESULT_EMPTY_REPLAY_MATERIAL = "empty_replay_material"
RESULT_NOT_BETTER = "not_better_than_champion"
RESULT_CHAMPION_NOT_PROMOTED = "champion_not_promoted"
RESULT_BUNDLE_BUILD_FAILED = "bundle_build_failed"
RESULT_SKIPPED_OTHER = "skipped_other"

#: An attempt opened and never settled. Not coerced to a failure: the KB never
#: refused anything, the process died before it could answer.
STATUS_PENDING = "pending"

#: Named rather than probed: the file is being written as this is assembled.
SESSION_BREAKDOWN_PATH = "session_breakdown.json"

#: The ``stop_reason`` meaning a robustness critic escalated to the close.
ESCALATED_STOP_REASON = "robustness_escalated"

#: Where the robustness ladder persists findings. Read rather than mirrored
#: through a recorder: a second copy could only be poorer than the ladder's.
ROBUSTNESS_FINDINGS_SUBDIR = "agents/robustness/findings"

#: How many findings the close-out carries; the newest, being the ones the stop
#: reason was drawn from. ``findings_total`` reports the untruncated count.
_FINDINGS_LIMIT = 50

#: The share of the theoretical ceiling the session aims at, a roofline
#: ceiling not being reachable in practice.
ROOFLINE_TARGET_RATIO = 0.70

#: A genuine failure, as against a step with no terminal row, which was
#: merely interrupted.
_FAILED = "failed"


def _stamp(ts: str) -> str:
    """``ts`` if the caller supplied one, else now, at microsecond precision:
    consecutive steps routinely settle inside the same second."""
    return str(ts).strip() or now_iso()


def _write(session_dir: Path | str | None, payload: Mapping[str, Any]) -> None:
    """Deep-merge ``payload`` into the ``close`` singleton. Never raises."""
    if not session_dir:
        trace_skip(reason="no session_dir", section=SECTION)
        return
    if not payload:
        trace_skip(reason="empty payload", section=SECTION)
        return
    try:
        recorder_for(session_dir, producer=PRODUCER).record_upsert_singleton(SECTION, dict(payload))
    except Exception as exc:  # noqa: BLE001 — the wind-down outranks its own record
        log.debug("record close failed", exc_info=True)
        trace_skip(reason="writer raised", section=SECTION, error=exc)


def record_close_opened(session_dir: Path | str | None, *, ts: str = "") -> None:
    """Open the section as the sequencer is entered.

    Writes ``status: "running"``, which stands until :func:`record_close_settled`
    replaces it, so a session that dies mid-close reports that state rather
    than having it mistaken for a verdict.
    """
    _write(
        session_dir,
        {
            "status": "running",
            "start_time": _stamp(ts),
            "close_sequence_done": False,
            "artifacts": {"session_breakdown_path": SESSION_BREAKDOWN_PATH},
        },
    )


def record_close_step(
    session_dir: Path | str | None,
    *,
    step: str,
    status: str,
    ts: str = "",
    task_id: str = "",
    detail: str = "",
) -> None:
    """Append one settled close step.

    Rows are append-only rather than keyed by step name: a resumed session
    runs the sequence again, and its rows describe a second close-out attempt
    instead of correcting the first. ``detail`` is free text for a human.
    """
    if not session_dir:
        trace_skip(reason="no session_dir", section=STEP_SECTION)
        return
    row: dict[str, Any] = {
        "step": str(step or ""),
        "status": str(status or "").strip().lower(),
        "ts": _stamp(ts),
    }
    if task_id:
        row["task_id"] = str(task_id)
    if detail:
        row["detail"] = str(detail)
    try:
        recorder_for(session_dir, producer=PRODUCER).record_item(STEP_SECTION, row)
    except Exception as exc:  # noqa: BLE001
        log.debug("record close step failed", exc_info=True)
        trace_skip(reason="writer raised", section=STEP_SECTION, error=exc)


def record_close_artifacts(
    session_dir: Path | str | None,
    *,
    final_json_path: Path | str | None = None,
    final_md_path: Path | str | None = None,
    artifact_package_path: Path | str | None = None,
) -> None:
    """Name the artifacts the close-out produced, at the step that produced them.

    Only the arguments supplied are written: the singleton merges leaf-by-leaf
    with no notion of an empty value, so a path this caller does not know would
    overwrite one an earlier caller did.
    """
    if not session_dir:
        trace_skip(reason="no session_dir", section=SECTION)
        return
    artifacts: dict[str, Any] = {}
    root = Path(session_dir)
    for key, value in (
        ("final_json_path", final_json_path),
        ("final_md_path", final_md_path),
        ("artifact_package_path", artifact_package_path),
    ):
        if value is not None:
            artifacts[key] = _rel(Path(value), root)
    if not artifacts:
        trace_skip(reason="empty payload", section=SECTION)
        return
    _write(session_dir, {"artifacts": artifacts})


def record_baseline_progress(
    session_dir: Path | str | None,
    *,
    failure_streak: Any = None,
    total_failures: Any = None,
    arg_error_streak: Any = None,
) -> None:
    """Record the session's final tally of baseline failures.

    No baseline event can hold these: an event closes when its measurement
    ends, but the counters are advanced afterwards by the write-back.
    ``arg_error_streak`` is apart because a rejected server arg is a
    configuration error the session can correct.
    """
    _write(
        session_dir,
        {
            "baseline_progress": {
                "failure_streak": _to_int(failure_streak),
                "total_failures": _to_int(total_failures),
                "arg_error_streak": _to_int(arg_error_streak),
            }
        },
    )


def record_final_recipe(
    session_dir: Path | str | None,
    *,
    throughput: Any = None,
    ttft_mean_ms: Any = None,
    e2el_mean_ms: Any = None,
    action_path: Any = None,
    extra_server_args: str = "",
    extra_envs: Mapping[str, Any] | None = None,
) -> None:
    """Record the configuration the session ended on. Never raises.

    No event holds the terminal recipe: a revert takes a layer off the stack
    without retracting the ledger row that adopted it. ``action_path`` is the
    surviving layers in promotion order, each ``action`` or ``action:variant``.
    """
    _write(
        session_dir,
        {
            "final_recipe": {
                "throughput": _to_float(throughput),
                "ttft_mean_ms": _to_float(ttft_mean_ms),
                "e2el_mean_ms": _to_float(e2el_mean_ms),
                "action_path": [str(step) for step in (action_path or []) if str(step)],
                "extra_server_args": str(extra_server_args or ""),
                "extra_envs": {str(key): str(value) for key, value in dict(extra_envs or {}).items()},
            }
        },
    )


def record_geak_candidate(
    session_dir: Path | str | None,
    *,
    pending: Mapping[str, Any] | None = None,
    revalidation_pending: Any = None,
) -> None:
    """Record where the GEAK candidate stood when the session wound down.

    A slot awaiting a rebench settles after the kernel event that ran its
    attempts has closed. ``pending`` is empty both when the session had no
    candidate and when the slot was released; ``status`` tells the two apart.
    """
    if not session_dir:
        trace_skip(reason="no session_dir", section=SECTION)
        return
    slot = dict(pending or {})
    _write(
        session_dir,
        {
            "geak_candidate": {
                "revalidation_pending": bool(revalidation_pending),
                "status": str(slot.get("status") or ""),
                "revalidation_error": str(slot.get("revalidation_error") or "") or None,
                "revalidation_error_class": str(slot.get("revalidation_error_class") or "") or None,
                "self_reported_gain_pct": _to_float(slot.get("self_reported_gain_pct")),
                "self_reported_tput": _to_float(slot.get("self_reported_tput")),
                "self_reported_basis": str(slot.get("self_reported_basis") or ""),
            }
        },
    )


def record_roofline_progress(
    session_dir: Path | str | None,
    *,
    baseline_tput: Any = None,
    baseline_ts: str = "",
    optimization_stack: Any = None,
    latest_snapshot: Mapping[str, Any] | None = None,
    current_best_tput: Any = None,
    cumulative_gain_pct: Any = None,
    failure_streak: Any = None,
) -> None:
    """Record how far the session got against its roofline ceiling.

    The close-out is the first moment the ceiling, the trajectory and the
    streak are all final. ``latest_snapshot`` supplies the ceiling and is
    absent when no analysis completed, which is why the ceiling fields are
    nullable rather than zero.
    """
    if not session_dir:
        trace_skip(reason="no session_dir", section=SECTION)
        return
    baseline = _to_float(baseline_tput) or 0.0
    trajectory = _trajectory(baseline, baseline_ts, optimization_stack)
    snapshot = dict(latest_snapshot or {})

    ceiling = _to_float(snapshot.get("theoretical_peak_tok_per_sec"))
    ceiling_available = ceiling is not None and ceiling > 0
    target = round(ceiling * ROOFLINE_TARGET_RATIO, 4) if ceiling_available else None
    best = _to_float(trajectory[-1]["tput"]) if trajectory else 0.0

    payload: dict[str, Any] = {
        "ceiling_kind": "throughput" if ceiling_available else "none",
        "ceiling_tok_per_sec": ceiling,
        "target_tok_per_sec": target,
        "ceiling_ratio_target": ROOFLINE_TARGET_RATIO,
        "ceiling_available": ceiling_available,
        "latency_ceiling_ms": None,
        "achieved_latency_ms": None,
        "latency_ceiling_available": False,
        "current_best_pct_of_latency_ceiling": None,
        "trajectory": trajectory,
        "baseline_tput": baseline,
        "current_best_tput": best,
        "cumulative_gain_pct": round(_to_float(cumulative_gain_pct) or 0.0, 4),
        "current_best_pct_of_ceiling": (round(best / ceiling * 100.0, 4) if ceiling_available and best > 0 else None),
        "current_best_pct_of_target": (round(best / target * 100.0, 4) if target and target > 0 and best > 0 else None),
        "roofline_failure_streak": _to_int(failure_streak),
        "latest_snapshot_id": _to_int(snapshot.get("snapshot_id")) or None,
    }

    # Diffusion (xDiT) image models decode no tokens: their roofline is the
    # ideal per-image compute floor against measured latency. ``ceiling_kind``
    # keeps a reader from taking the null tok/s fields for a failed analysis.
    if not ceiling_available:
        ideal_ms = _to_float(snapshot.get("roofline_ideal_ms"))
        measured_ms = _to_float(snapshot.get("e2e_mean_ms"))
        if ideal_ms and ideal_ms > 0 and measured_ms and measured_ms > 0:
            payload["ceiling_kind"] = "latency"
            payload["latency_ceiling_ms"] = round(ideal_ms, 4)
            payload["achieved_latency_ms"] = round(measured_ms, 4)
            payload["latency_ceiling_available"] = True
            # Ideal over measured: higher is nearer the floor.
            payload["current_best_pct_of_latency_ceiling"] = round(ideal_ms / measured_ms * 100.0, 4)

    # A tail disagreeing with the session's own best means a promotion never
    # made it onto the stack, as when a resume interrupted a mid-promote.
    declared = _to_float(current_best_tput)
    if declared and declared > 0 and best > 0 and abs(declared - best) / max(declared, 1.0) > 0.001:
        payload["trajectory_incomplete"] = True
        payload["current_best_tput_declared"] = declared
    else:
        payload["trajectory_incomplete"] = False

    _write(session_dir, {"roofline_progress": payload})


def _trajectory(baseline: float, baseline_ts: str, stack: Any) -> list[dict[str, Any]]:
    """Build the throughput curve, oldest first; a non-positive ``baseline``
    yields no baseline point, a curve having to start somewhere real."""
    points: list[dict[str, Any]] = []
    if baseline > 0:
        points.append(
            {
                "ts": str(baseline_ts or ""),
                "tput": baseline,
                "label": "baseline",
                "action": "baseline",
                "gain_pct": 0.0,
                "flags": "",
                "extra_envs": {},
            }
        )
    entries = stack if isinstance(stack, list) else []
    # Sorted by timestamp rather than trusted in list order: a legacy prepend
    # puts the newest first and would silently invert the curve.
    for entry in sorted((e for e in entries if isinstance(e, Mapping)), key=lambda e: str(e.get("ts") or "")):
        tput = _to_float(entry.get("tput"))
        if tput is None or tput <= 0:
            continue
        points.append(
            {
                "ts": str(entry.get("ts") or ""),
                "tput": tput,
                "label": str(entry.get("variant_name") or entry.get("action") or ""),
                "action": str(entry.get("action") or ""),
                "gain_pct": round((tput - baseline) / baseline * 100.0, 4) if baseline > 0 else 0.0,
                "flags": str(entry.get("candidate_extra_server_args") or ""),
                "extra_envs": dict(entry.get("extra_envs") or {}),
            }
        )
    return points


def _to_float(value: Any) -> float | None:
    """``value`` as a float, or ``None`` when it is not a finite number."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and result not in (float("inf"), float("-inf")) else None


def _to_int(value: Any) -> int:
    """``value`` as an int, or ``0`` when it is not one."""
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def record_close_settled(
    session_dir: Path | str | None,
    *,
    stop_reason: str = "",
    ts: str = "",
) -> None:
    """Record the sequencer's verdict as its last act.

    ``degraded`` when any recorded step reported ``failed``, ``succeeded``
    otherwise: reaching this function is itself the evidence that the sequence
    ran to the end, so an un-settled step does not count against the verdict.
    ``stop_reason`` is recorded so the escalation verdict stays auditable.
    """
    if not session_dir:
        trace_skip(reason="no session_dir", section=SECTION)
        return
    reason = str(stop_reason or "").strip()
    _write(
        session_dir,
        {
            "status": "degraded" if _any_step_failed(session_dir) else "succeeded",
            "end_time": _stamp(ts),
            "close_sequence_done": True,
            "stop_reason": reason,
            "robustness": {
                "escalated": reason.lower() == ESCALATED_STOP_REASON,
                **_robustness_findings(session_dir),
            },
        },
    )


def _robustness_findings(session_dir: Path | str) -> dict[str, Any]:
    """Read what the robustness ladder found over the session.

    Returns ``findings`` and ``findings_total``, or an empty mapping when the
    ladder never wrote: an empty list would claim it ran and found nothing.
    """
    directory = Path(session_dir) / ROBUSTNESS_FINDINGS_SUBDIR
    rows: list[dict[str, Any]] = []
    try:
        for path in sorted(directory.glob("*.jsonl")):
            rows.extend(read_jsonl(path) or [])
    except Exception:  # noqa: BLE001 — a findings log we cannot read is not a close-out failure
        log.debug("close: cannot read robustness findings under %s", directory, exc_info=True)
        return {}
    if not rows:
        return {}
    # A row whose clock is unreadable sorts to the front rather than raising,
    # so one malformed line cannot cost the close-out every finding beside it.
    rows.sort(key=lambda row: (_to_float(row.get("timestamp_unix")) or 0.0, _to_int(row.get("tick_index"))))
    return {
        "findings": [_finding_row(row) for row in rows[-_FINDINGS_LIMIT:]],
        "findings_total": len(rows),
    }


def _finding_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project one persisted finding onto what the close-out reports.

    ``intents`` is reduced to the intent types: one badly-degraded session's
    worth of their payloads would dwarf the rest of the breakdown.
    """
    return {
        "tick_index": _to_int(row.get("tick_index")),
        "timestamp_unix": _to_float(row.get("timestamp_unix")),
        "symptom_name": str(row.get("symptom_name") or ""),
        "severity": str(row.get("severity") or ""),
        "summary": str(row.get("summary") or ""),
        "rca_text": str(row.get("rca_text") or ""),
        "intents": [
            str(intent.get("intent_type") or intent.get("type") or "")
            for intent in row.get("intents") or []
            if isinstance(intent, dict)
        ],
        "evidence": row.get("evidence") if isinstance(row.get("evidence"), dict) else {},
    }


def record_write_back_opened(
    session_dir: Path | str | None,
    *,
    attempt: int,
    source: str,
    ts: str = "",
) -> None:
    """Open one Recipe KB publication attempt, before the write is tried.

    Opening before the write is what makes a mid-publish death visible: the row
    stands at ``pending`` until :func:`record_write_back_settled` replaces it,
    so a session killed here reports an unsettled attempt rather than a refusal
    the KB never issued. ``attempt`` is 1-based and keys the row.
    """
    _write_attempt(
        session_dir,
        attempt=attempt,
        row={
            "attempt": int(attempt),
            "source": str(source or ""),
            "status": STATUS_PENDING,
            "opened_at": _stamp(ts),
        },
    )


def record_write_back_settled(
    session_dir: Path | str | None,
    *,
    attempt: int,
    source: str,
    status: str,
    result_type: str = "",
    raw_reason: str = "",
    error_class: str = "",
    backend: str = "",
    canonical_id: str = "",
    session_id: str = "",
    scope: Mapping[str, Any] | None = None,
    optimized_throughput: float | None = None,
    validated_gain_pct: float | None = None,
    ts: str = "",
) -> None:
    """Settle one publication attempt and the arc it belongs to.

    ``result_type`` is one of this module's ``RESULT_*`` codes, stated by the
    publisher so no reader has to read it out of ``raw_reason``, which is kept
    verbatim. ``error_class`` is separate because the publisher reports
    transport failures as the bare exception class name.
    """
    if not session_dir:
        trace_skip(reason="no session_dir", section=WRITE_BACK_SECTION)
        return
    settled = _stamp(ts)
    verdict = str(status or "").strip().lower()
    code = str(result_type or "").strip()
    reason = str(raw_reason or "").strip()
    error = str(error_class or "").strip()

    attempt_row: dict[str, Any] = {
        "attempt": int(attempt),
        "source": str(source or ""),
        "status": verdict,
        "settled_at": settled,
    }
    if code:
        attempt_row["result_type"] = code
    if reason:
        attempt_row["raw_reason"] = reason
    if error:
        attempt_row["error_class"] = error
    _write_attempt(session_dir, attempt=attempt, row=attempt_row)

    arc: dict[str, Any] = {
        "status": verdict,
        "end_time": settled,
        "queue": _queue_depth(session_dir),
    }
    for key, value in (
        ("result_type", code),
        ("raw_reason", reason),
        ("backend", str(backend or "").strip()),
        ("canonical_id", str(canonical_id or "").strip()),
        ("session_id", str(session_id or "").strip()),
    ):
        # Only what this exit knows: the singleton merges leaf-by-leaf, so a
        # blank would erase an earlier attempt's answer.
        if value:
            arc[key] = value
    if scope:
        arc["scope"] = dict(scope)
    if optimized_throughput is not None:
        arc["optimized_throughput"] = float(optimized_throughput)
    if validated_gain_pct is not None:
        arc["validated_gain_pct"] = float(validated_gain_pct)
    if error:
        arc["failure"] = {"error_class": error, "error": reason or error}
    _write_arc(session_dir, arc)


def _write_arc(session_dir: Path | str, payload: Mapping[str, Any]) -> None:
    """Deep-merge ``payload`` into the write-back singleton. Never raises."""
    try:
        recorder_for(session_dir, producer=PRODUCER).record_upsert_singleton(
            WRITE_BACK_SECTION,
            dict(payload),
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("record write-back failed", exc_info=True)
        trace_skip(reason="writer raised", section=WRITE_BACK_SECTION, error=exc)


def _write_attempt(session_dir: Path | str | None, *, attempt: int, row: Mapping[str, Any]) -> None:
    """Upsert one attempt row, keyed by its number. Never raises."""
    if not session_dir:
        trace_skip(reason="no session_dir", section=WRITE_BACK_ATTEMPT_SECTION)
        return
    try:
        recorder_for(session_dir, producer=PRODUCER).record_upsert_item(
            WRITE_BACK_ATTEMPT_SECTION,
            dict(row),
            key=str(int(attempt)),
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("record write-back attempt failed", exc_info=True)
        trace_skip(reason="writer raised", section=WRITE_BACK_ATTEMPT_SECTION, error=exc)


def _queue_depth(session_dir: Path | str) -> dict[str, int]:
    """Line counts of the local KB write queues, as of this settlement.

    Snapshotted here because the queues keep moving: a depth read at export
    describes when the export ran, not when the publication settled.
    """
    from ...session.session_paths import (
        recipe_kb_dead_letter_ndjson,
        recipe_kb_flushed_ndjson,
        recipe_kb_pending_ndjson,
    )

    root = Path(session_dir)
    return {
        "pending_lines": _count_lines(recipe_kb_pending_ndjson(root)),
        "flushed_bookmarks": _count_lines(recipe_kb_flushed_ndjson(root)),
        "dead_letter_lines": _count_lines(recipe_kb_dead_letter_ndjson(root)),
    }


def _count_lines(path: Path) -> int:
    """Non-blank line count for ``path``, or ``0`` when it cannot be read."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _any_step_failed(session_dir: Path | str) -> bool:
    """Whether any close step this session recorded reported a failure."""
    try:
        # Deferred: the assembler imports this package's recorder module.
        from .assembler import close_steps

        return any(str(row.get("status") or "") == _FAILED for row in close_steps(session_dir))
    except Exception:  # noqa: BLE001 — an unreadable spool must not block the close
        log.debug("close verdict: step readback failed", exc_info=True)
        return False


def _rel(path: Path, session_dir: Path) -> str:
    """Express ``path`` relative to the session, or absolute when outside it."""
    try:
        return path.resolve().relative_to(session_dir.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)


__all__ = [
    "ESCALATED_STOP_REASON",
    "RESULT_AGENTX_BLOCKED",
    "RESULT_BUNDLE_BUILD_FAILED",
    "RESULT_CHAMPION_NOT_PROMOTED",
    "RESULT_CONFIGURATION_FAILED",
    "RESULT_EMPTY_REPLAY_MATERIAL",
    "RESULT_INVALID_SCOPE",
    "RESULT_INVALID_THROUGHPUT",
    "RESULT_KB_DISABLED",
    "RESULT_MISSING_THROUGHPUT",
    "RESULT_NOT_BETTER",
    "RESULT_NO_NEW_KEEP",
    "RESULT_SKIPPED_OTHER",
    "RESULT_TRANSPORT_FAILED",
    "RESULT_WRITTEN",
    "ROOFLINE_TARGET_RATIO",
    "SESSION_BREAKDOWN_PATH",
    "STATUS_PENDING",
    "record_baseline_progress",
    "record_close_artifacts",
    "record_close_opened",
    "record_close_settled",
    "record_close_step",
    "record_geak_candidate",
    "record_roofline_progress",
    "record_write_back_opened",
    "record_write_back_settled",
]
