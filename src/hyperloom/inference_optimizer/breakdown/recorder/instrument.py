# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Author-time instrumentation for ``session_breakdown.json``.

These helpers are called from the producing code (the Coordinator's
``SharedState``) to record breakdown facts where they are born, instead of
having the exporter re-walk artifacts later.

What lives here: the Coordinator's state snapshots, the backend build
provenance carried by a kernel-agent result (which reaches the optimizer
through nothing else).

Every helper is best-effort: spool failures degrade the section and never
propagate into the run they are describing.
Payloads are shaped to the matching ``schema.py`` TypedDict.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.timeutil import iso_z

from . import tool_versions
from .kernel_records import KernelRewriteRun, LaneRunFields
from .session_metadata import snapshot_metadata
from .trace import trace_skip

if TYPE_CHECKING:
    from .kernel_event import KernelEventRecorder

PRODUCER_COORDINATOR = "coordinator"
PRODUCER_KERNEL_AGENT = "kernel-agent"


def _recorder(session_dir: Path | str, producer: str):
    """Return the process-cached recorder for ``session_dir`` and ``producer``."""
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
    rec = _recorder(session_dir, producer)

    _snapshot_session(rec, state)
    snapshot_metadata(rec, state)


def _unset_or_int(st: Any, attr: str) -> int | None:
    """The integer at ``attr``, or ``None`` when the state never set it.

    A budget nobody set and a budget of zero are different facts, and writing
    the first one as ``0`` destroys the difference: the export then has to
    guess, and the only guess available -- treat zero as unset -- throws away
    the sessions that really did run zero ticks.
    """
    value = getattr(st, attr, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _snapshot_session(rec, st: Any) -> None:
    """Snapshot the ``session`` singleton from ``st`` (no-op without a session id)."""
    session_id = str(getattr(st, "session_id", "") or "")
    if not session_id:
        return
    stop_reason = str(getattr(st, "stop_reason", "") or "")
    row = {
        "session_id": session_id,
        "claw_session_id": getattr(st, "claw_session_id", "") or "",
        "sandbox_user_id": getattr(st, "sandbox_user_id", "") or "",
        "start_ts": str(getattr(st, "start_ts", "") or ""),
        # A resumed run clears its reason but not necessarily the stale timestamp, so the pair is only ever
        # emitted together.
        "ended_at_utc": iso_z(getattr(st, "stop_ts", "")) if stop_reason else "",
        "stop_reason": stop_reason,
        "phase": str(getattr(st, "phase", "") or ""),
    }
    # Left off the row entirely when unset, so that a recorded number -- zero
    # included -- always means the state actually carried it.
    for key, attr in (("max_minutes", "max_minutes"), ("tick_count", "tick")):
        value = _unset_or_int(st, attr)
        if value is not None:
            row[key] = value
    rec.record_singleton("session", row)


def _mirror_backend_attempts_to_kernel_timeline(result: dict[str, Any], recorder: "KernelEventRecorder | None") -> None:
    """Mirror the result's backend attempts into the open KERNEL timeline event.

    The legacy ``kernel_backend_result`` fragment is session-wide; the V6 kernel
    event is visit-scoped. This copies each attempt as its own ``kernel_rewrites[]`` row.

    Every backend the kernel agent dispatched lands here, GEAK included: the
    lane is about a kernel having been rewritten, not about which backend did
    it, and each row names its own. ``ext.geak`` stays reserved for the
    delegated optimizer's own campaign, which is a different producer's account
    of a different run.
    """
    if recorder is None:
        return
    kid = str(result.get("kernel_id") or "")
    if not kid:
        return
    verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
    adopted_attempt_id = str(verification.get("best_attempt_id") or "")
    attempts = result.get("attempts") if isinstance(result.get("attempts"), list) else []
    attempts = [att for att in attempts if isinstance(att, dict)]
    all_backends = [att["backend"] for att in attempts if att.get("backend")]
    for att in attempts:
        attempt_id = str(att.get("attempt_id") or "")
        is_adopted = bool(attempt_id) and attempt_id == adopted_attempt_id
        recorder.record_kernel_rewrite(
            KernelRewriteRun(
                lane=LaneRunFields(
                    run_id=attempt_id,
                    status=att.get("status"),
                    micro_decision=att.get("decision"),
                    failure_reason=att.get("error"),
                    error_class=att.get("error_class"),
                ),
                kernel_id=kid,
                kernel_name=result.get("kernel_name"),
                backends_tried=all_backends,
                adopted_backend=att.get("backend") if is_adopted else "",
                speedup=att.get("micro_speedup") or None,
            )
        )


def record_backend_versions_and_timeline(
    session_dir: Path | str | None,
    result: dict[str, Any],
    *,
    recorder: "KernelEventRecorder | None",
    producer: str = PRODUCER_KERNEL_AGENT,
) -> None:
    """Record what a kernel-agent result says about the backends that ran.

    Two facts are recorded: the build of each backend, which reaches the
    optimizer through nothing else, and the attempts themselves, which are
    mirrored onto the kernel timeline event via ``recorder``. A falsy
    ``session_dir``, or a ``result`` that is not a dict, is a no-op.
    """
    if not session_dir or not isinstance(result, dict):
        trace_skip(
            reason="no session_dir" if not session_dir else "result is not a dict",
            section="versions",
        )
        return
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
    _mirror_backend_attempts_to_kernel_timeline(result, recorder)


__all__ = [
    "PRODUCER_COORDINATOR",
    "PRODUCER_KERNEL_AGENT",
    "record_backend_versions_and_timeline",
    "snapshot_state_sections",
]
