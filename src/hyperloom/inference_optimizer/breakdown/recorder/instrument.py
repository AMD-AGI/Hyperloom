# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Author-time instrumentation for ``session_breakdown.json``.

These helpers are called from the producing code (the Coordinator's
``SharedState``) to record breakdown facts where they are born, instead of
having the exporter re-walk artifacts later.

Every helper is best-effort: all failures are swallowed (logged at debug).
Payloads are shaped to the matching ``schema.py`` TypedDict.

What is left in this module after the entity streams were retired:

* Coordinator state snapshots -- ``session`` / ``metadata`` /
  ``explore_search`` / ``roofline`` sections, plus one ``phase_timeline``
  event per recorded action attempt.
* Backend build provenance from a kernel-agent result, which reaches the
  optimizer through nothing else, mirrored alongside the per-backend
  attempts onto the kernel timeline event.
* A generic singleton-section writer for producer-owned summaries.

The authoritative public surface is the re-export list in
``recorder/__init__``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_float
from hyperloom.common.timeutil import iso_z

from .session_metadata import snapshot_metadata
from . import tool_versions
from .trace import trace_skip

log = logging.getLogger(__name__)

PRODUCER_COORDINATOR = "coordinator"
PRODUCER_KERNEL_AGENT = "kernel-agent"

_FAILED_STATUSES = frozenset({"failed", "error", "crashed", "timeout"})


def _recorder(session_dir: Path | str, producer: str):
    """Return the process-cached recorder for ``session_dir`` and ``producer``.

    Returns:
        The process-cached recorder for the ``(session_dir, producer)`` pair.
    """
    from .recorder import recorder_for

    return recorder_for(session_dir, producer=producer)


def snapshot_state_sections(
    session_dir: Path | str | None,
    state: Any,
    *,
    producer: str = PRODUCER_COORDINATOR,
) -> None:
    """Snapshot every state-owned breakdown section from a live ``SharedState``."""
    if not session_dir or state is None:
        trace_skip(reason="no session_dir" if not session_dir else "no state", section="session")
        return
    rec = None
    try:
        rec = _recorder(session_dir, producer)
    except Exception as exc:  # noqa: BLE001
        log.debug("recorder unavailable", exc_info=True)
        trace_skip(reason="writer raised", section="session", error=exc)
        return

    for name, fn in (
        ("session", _snapshot_session),
        ("metadata", snapshot_metadata),
    ):
        try:
            fn(rec, state)
        except Exception as exc:  # noqa: BLE001
            log.debug("snapshot section %s failed", name, exc_info=True)
            trace_skip(reason="writer raised", section=name, error=exc)


def _snapshot_session(rec, st: Any) -> None:
    """Snapshot the ``session`` singleton from ``st`` (no-op without a session id)."""
    session_id = str(getattr(st, "session_id", "") or "")
    if not session_id:
        return
    stop_reason = str(getattr(st, "stop_reason", "") or "")
    rec.record_singleton(
        "session",
        {
            "session_id": session_id,
            "claw_session_id": getattr(st, "claw_session_id", "") or "",
            "sandbox_user_id": getattr(st, "sandbox_user_id", "") or "",
            "start_ts": str(getattr(st, "start_ts", "") or ""),
            # A resumed run clears its reason but not necessarily the stale timestamp, so the pair is only ever
            # emitted together.
            "ended_at_utc": iso_z(getattr(st, "stop_ts", "")) if stop_reason else "",
            "stop_reason": stop_reason,
            "max_minutes": int(getattr(st, "max_minutes", 0) or 0),
            "tick_count": int(getattr(st, "tick", 0) or 0),
            "phase": str(getattr(st, "phase", "") or ""),
        },
    )


def _to_bool(value: Any) -> bool | None:
    """Coerce a loosely-typed truthy/falsy value to ``bool``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "pass", "passed", "ok"):
        return True
    if s in ("false", "0", "no", "fail", "failed"):
        return False
    return None


def _mirror_backend_attempts_to_kernel_timeline(result: dict[str, Any]) -> None:
    """Mirror the result's backend attempts into the open KERNEL timeline event.

    The legacy ``kernel_backend_result`` fragment is session-wide; the V6 kernel
    event is visit-scoped. This copies each attempt as its own
    ``kernel_rewrites[]`` row while a KERNEL visit recorder is active.

    Every backend the kernel agent dispatched lands here, GEAK included: the
    lane is about a kernel having been rewritten, not about which backend did
    it, and each row names its own. ``ext.geak`` stays reserved for the
    delegated optimizer's own campaign, which is a different producer's account
    of a different run.
    """
    from .kernel_event import active_kernel_recorder

    recorder = active_kernel_recorder()
    if recorder is None:
        return
    kid = str(result.get("kernel_id") or "")
    if not kid:
        return
    run_id = str(result.get("run_id") or result.get("session_id") or "")
    verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
    proposal = result.get("proposal") if isinstance(result.get("proposal"), dict) else {}
    attempts = result.get("attempts") if isinstance(result.get("attempts"), list) else []
    all_backends = [
        str(item.get("backend") or "") for item in attempts if isinstance(item, dict) and item.get("backend")
    ]
    adopted_attempt_id = str(verification.get("best_attempt_id") or "")
    kernel_decision = str(proposal.get("decision") or "").upper()
    kernel_artifact = str(verification.get("best_artifact_path") or "")
    task_group = ""
    candidate = result.get("candidate")
    if isinstance(candidate, dict):
        task_group = str(candidate.get("task_group") or "")

    if attempts:
        for att in attempts:
            if not isinstance(att, dict):
                continue
            attempt_id = str(att.get("attempt_id") or att.get("id") or "")
            backend = str(att.get("backend") or "")
            is_adopted = bool(attempt_id) and attempt_id == adopted_attempt_id
            status_lower = str(att.get("status") or "").lower()
            decision = str(att.get("decision") or "").upper()
            if not decision and status_lower in _FAILED_STATUSES:
                decision = "FAILED"
            if is_adopted and kernel_decision:
                decision = kernel_decision
            micro_speedup = to_float(att.get("micro_speedup") or att.get("speedup"))
            if micro_speedup is None and is_adopted:
                micro_speedup = to_float(verification.get("micro_speedup"))
            compile_passed = _to_bool(att.get("compile_passed"))
            correctness_passed = _to_bool(att.get("correctness_passed"))
            if is_adopted and compile_passed is None:
                compile_passed = _to_bool(verification.get("compile_passed"))
            if is_adopted and correctness_passed is None:
                correctness_passed = _to_bool(verification.get("correctness_passed"))
            optimized = att.get("optimized_path") or att.get("optimized_file")
            artifact_path = str(optimized or (kernel_artifact if is_adopted else "") or "")
            recorder.record_kernel_rewrite(
                run_id=attempt_id or f"{run_id}-{backend}",
                kernel_id=kid,
                kernel_name=str(result.get("kernel_name") or result.get("name") or ""),
                status=status_lower or "unknown",
                dispatched=True,
                backends_tried=all_backends or ([backend] if backend else []),
                adopted_backend=backend if is_adopted else "",
                task_group=task_group,
                speedup=micro_speedup,
                compile_status="passed" if compile_passed is True else ("failed" if compile_passed is False else ""),
                correctness=correctness_passed,
                artifact_path=artifact_path,
                micro_decision=decision,
                started_at=str(att.get("started_at") or att.get("created_at") or att.get("ts") or ""),
                ended_at=str(att.get("ended_at") or ""),
                duration_sec=to_float(att.get("duration_sec") or att.get("elapsed_sec") or att.get("elapsed_s")),
                failure_reason=str(att.get("error") or att.get("error_message") or ""),
            )
        return

    status = str(result.get("status") or "").lower()
    err_class = str(result.get("error_class") or "")
    decision = str(proposal.get("decision") or "").upper()
    failed = status in _FAILED_STATUSES or (decision == "REVERT" and bool(err_class))
    skipped = status == "skipped"
    if not failed and not skipped:
        return
    backend = str(result.get("backend") or "").lower() or "unknown"
    recorder.record_kernel_rewrite(
        # ``:`` is the fragment key's own separator, so a synthesized run id
        # must not contain one or the row is dropped on the way to the event.
        run_id=run_id or f"{kid}-predispatch",
        kernel_id=kid,
        status=status or "failed",
        dispatched=False,
        backends_tried=[backend] if backend != "unknown" else [],
        # ``reason`` first: for an undispatched row it is the only field that
        # names *which* gate declined -- below the GPU-share floor, merged into
        # an op-fanout representative, a group already in flight. ``status`` is
        # "skipped" for all of them, so reading it first collapses the
        # distinction this row exists to draw.
        skip_reason=str(result.get("reason") or result.get("skip_reason") or err_class or status or ""),
        micro_decision=decision or ("SKIPPED" if skipped else "FAILED"),
        failure_reason=str(result.get("error") or err_class or ""),
    )


def record_backend_versions_and_timeline(
    session_dir: Path | str | None,
    result: dict[str, Any],
    *,
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Record what a kernel-agent result says about the backends that ran.

    Two facts outlive the entity streams this used to also write: the build of
    each backend, which reaches the optimizer through nothing else, and the
    attempts themselves, which are mirrored onto the kernel timeline event.

    Args:
        session_dir (Path | str | None): the session directory; a falsy value
            is a no-op.
        result (dict[str, Any]): the kernel-agent result carrying the
            per-backend ``attempts`` ladder.
        producer (str): the breakdown producer label.
    """
    if not session_dir or not isinstance(result, dict):
        trace_skip(
            reason="no session_dir" if not session_dir else "result is not a dict",
            section="versions",
        )
        return
    try:
        result_meta = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        attempts = result.get("attempts")
        attempts = attempts if isinstance(attempts, list) else []
        recorded: set[str] = set()
        for att in attempts:
            if not isinstance(att, dict):
                continue
            backend = str(att.get("backend") or "").lower()
            if not backend or backend in recorded:
                continue
            recorded.add(backend)
            att_meta = att.get("metadata") if isinstance(att.get("metadata"), dict) else {}
            tool_versions.record_tool_version(
                session_dir,
                tool=backend,
                root=str(att_meta.get("root_dir") or result_meta.get("root_dir") or "") or None,
                version=str(att_meta.get("version") or result_meta.get("version") or "") or None,
                producer=producer,
            )
        # No attempts means the run failed before any backend launched. The
        # backend the result names is still the one whose build was in play --
        # unless it names none, which is the pre-dispatch gating case that
        # never resolved a build to report.
        if not recorded:
            backend = str(result.get("backend") or "").lower()
            if backend:
                tool_versions.record_tool_version(
                    session_dir,
                    tool=backend,
                    root=str(result_meta.get("root_dir") or "") or None,
                    version=str(result_meta.get("version") or "") or None,
                    producer=producer,
                )
        _mirror_backend_attempts_to_kernel_timeline(result)
    except Exception as exc:  # noqa: BLE001
        log.debug("record_backend_versions_and_timeline failed", exc_info=True)
        trace_skip(reason="writer raised", section="versions", error=exc)


def record_singleton_section(
    session_dir: Path | str | None,
    section: str,
    payload: dict[str, Any],
    *,
    producer: str,
) -> None:
    """Record a producer-owned singleton section (report summaries, etc.)."""
    if not session_dir or not isinstance(payload, dict) or not payload:
        trace_skip(reason="no session_dir" if not session_dir else "empty payload", section=section)
        return
    try:
        _recorder(session_dir, producer).record_singleton(section, payload)
    except Exception as exc:  # noqa: BLE001
        log.debug("record_singleton_section %s failed", section, exc_info=True)
        trace_skip(reason="writer raised", section=section, error=exc)


__all__ = [
    "PRODUCER_COORDINATOR",
    "PRODUCER_KERNEL_AGENT",
    "record_backend_versions_and_timeline",
    "record_singleton_section",
    "snapshot_state_sections",
]
