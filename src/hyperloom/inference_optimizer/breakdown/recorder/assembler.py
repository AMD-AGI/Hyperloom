# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-side of the breakdown recorder.

Assembles the per-producer fragments written by :class:`~.recorder.Recorder`
into a ``{section: value}`` mapping for the ``session_breakdown.json``
envelope: a ``singleton`` section takes the payload of the latest fragment by
``ts``, a plain ``item`` section concatenates its payloads by ``seq`` then
``ts``. A compose pass runs last and reconciles across fragments. Bad or
partial fragments are skipped and noted in ``warnings``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperloom.common.jsonio import read_json

_UNREADABLE = object()


def parts_dir(session_dir: Path | str) -> Path:
    """Return the breakdown spool directory for ``session_dir``."""
    from ...session.session_paths import breakdown_parts_dir  # local: avoid import cycle

    return breakdown_parts_dir(Path(session_dir))


def has_parts(session_dir: Path | str) -> bool:
    """True iff the spool directory holds at least one ``*.json`` fragment."""
    d = parts_dir(session_dir)
    return d.is_dir() and any(d.glob("*.json"))


def _load(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    """Read and parse one fragment file, noting problems into ``warnings``."""
    rec = read_json(
        path,
        default=_UNREADABLE,
        on_error=lambda exc: warnings.append(f"recorder: failed to read {path.name}: {exc!r}"),
    )
    if rec is _UNREADABLE:
        return None
    if not isinstance(rec, dict):
        warnings.append(f"recorder: {path.name} is not an object")
        return None
    return rec


def assemble_parts(
    session_dir: Path | str,
    *,
    warnings: list[str] | None = None,
    keep_event_rows: bool = False,
) -> dict[str, Any]:
    """Return ``{section: list | dict}`` assembled from the spool directory."""
    warns = warnings if warnings is not None else []
    d = parts_dir(session_dir)
    if not d.is_dir():
        return {}

    items: dict[str, list[dict[str, Any]]] = {}
    singletons: dict[str, dict[str, Any]] = {}
    discarded: dict[str, list[str]] = {}

    for path in sorted(d.glob("*.json")):
        rec = _load(path, warns)
        if rec is None:
            continue
        section = rec.get("section")
        if not isinstance(section, str) or not section:
            warns.append(f"recorder: {path.name} missing 'section'")
            continue
        if rec.get("kind") == "singleton":
            prev = singletons.get(section)
            if prev is None or str(rec.get("ts") or "") >= str(prev.get("ts") or ""):
                if prev is not None:
                    discarded.setdefault(section, []).append(str(prev.get("producer") or "?"))
                singletons[section] = rec
            else:
                discarded.setdefault(section, []).append(str(rec.get("producer") or "?"))
        else:
            items.setdefault(section, []).append(rec)

    # A singleton fragment is named for its producer, so a section with more than one is one two producers claimed.
    for section, producers in discarded.items():
        warns.append(
            f"recorder: {section} was written as a singleton by more than one "
            f"producer; only the newest was kept and "
            f"{sorted(set(producers))} were dropped whole"
        )

    out: dict[str, Any] = {}
    for section, recs in items.items():
        recs.sort(key=lambda r: (int(r.get("seq") or 0), str(r.get("ts") or "")))
        out[section] = [r.get("payload") for r in recs]
    for section, rec in singletons.items():
        out[section] = rec.get("payload")

    _compose_versions(out)
    _compose_critic(out)
    _compose_robustness(out)
    _compose_close(out)
    if not keep_event_rows:
        _drop_event_rows(out)
    return out


def _compose_close(out: dict[str, Any]) -> None:
    """Fold the ``close_step`` substream into ``close.steps``, mutating ``out``.

    Unlike the other compose helpers this one merges into a directly-recorded
    singleton rather than deferring to it: the CLOSE sequencer writes both.
    """
    write_back = _compose_write_back(out)
    rows = out.pop("close_step", None)
    if rows is None and write_back is None:
        return
    close = out.get("close")
    close = dict(close) if isinstance(close, dict) else {}
    if write_back is not None:
        close["kb_write_back"] = write_back
    if rows is None:
        out["close"] = close
        return
    steps = [row for row in rows if isinstance(row, dict)]
    # ``(seq, ts)`` is only the write order within one process, and a resumed
    # session closes in a second whose sequence restarts at zero. Stable, so
    # the write order still breaks same-microsecond ties.
    steps.sort(key=lambda row: str(row.get("ts") or ""))
    close["steps"] = steps
    out["close"] = close


def _compose_write_back(out: dict[str, Any]) -> dict[str, Any] | None:
    """Fold the Recipe KB publication into one ``close.kb_write_back`` block.

    ``None`` when nothing was recorded, which tells a session that never
    attempted a publication apart from one whose attempt never settled: the
    latter has an attempt row still at ``pending``.
    """
    arc = out.pop("close_write_back", None)
    attempts = out.pop("close_write_back_attempt", None)
    if arc is None and attempts is None:
        return None
    block = dict(arc) if isinstance(arc, dict) else {}
    rows = [row for row in attempts if isinstance(row, dict)] if isinstance(attempts, list) else []
    rows.sort(key=lambda row: int(row.get("attempt") or 0))
    block["attempts"] = rows
    if not block.get("status"):
        # Attempts but no settled arc: opened, then the process died before
        # anything answered. Not a failure the store never returned.
        block["status"] = _WRITE_BACK_PENDING
    return block


#: Mirrors ``close_out.STATUS_PENDING``, spelled out to break an import cycle.
_WRITE_BACK_PENDING = "pending"


def close_steps(session_dir: Path | str) -> list[dict[str, Any]]:
    """Read back the close steps recorded for ``session_dir``, oldest first.

    Read through assembly rather than by re-globbing the spool, so the
    sequencer's own verdict sees exactly what the envelope will.
    """
    close = assemble_parts(session_dir, warnings=[]).get("close")
    if not isinstance(close, dict):
        return []
    steps = close.get("steps")
    return [row for row in steps if isinstance(row, dict)] if isinstance(steps, list) else []


def _compose_versions(out: dict[str, Any]) -> None:
    """Fold the ``versions`` item substream into ``metadata.versions.tools``.

    Rows arrive keyed by tool name, so a later row replaces the one before it.
    ``metadata`` is created when missing.
    """
    rows = out.pop("versions", None)
    if not isinstance(rows, list):
        return
    tools: dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("tool") or "").strip()
        if name:
            tools[name] = row
    if not tools:
        return
    metadata = out.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        out["metadata"] = metadata
    versions = metadata.get("versions")
    if not isinstance(versions, dict):
        versions = {}
        metadata["versions"] = versions
    existing = versions.get("tools")
    versions["tools"] = {**existing, **tools} if isinstance(existing, dict) else tools


def _compose_critic(out: dict[str, Any]) -> None:
    """Fold the ``critic_iteration`` substream into ``critic``, ordered as the
    agent ran, mutating ``out`` and popping the raw substream."""
    rows = out.pop("critic_iteration", None)
    if not isinstance(rows, list):
        return

    def _iter_of(row: dict[str, Any]) -> int:
        try:
            return int(row.get("iter") or 0)
        except (TypeError, ValueError):
            return 0

    iterations = sorted(
        (r for r in rows if isinstance(r, dict)),
        key=lambda r: (_iter_of(r), str(r.get("ts") or "")),
    )
    out["critic"] = {"iterations": iterations}


def _compose_robustness(out: dict[str, Any]) -> None:
    """Fold the ``robustness_turn`` substream into ``robustness``, ordered by
    turn, mutating ``out`` and popping the raw substream."""
    rows = out.pop("robustness_turn", None)
    if not isinstance(rows, list):
        return

    def _turn_of(row: dict[str, Any]) -> int:
        try:
            return int(row.get("turn_idx") or 0)
        except (TypeError, ValueError):
            return 0

    turns = sorted(
        (r for r in rows if isinstance(r, dict)),
        key=lambda r: (_turn_of(r), str(r.get("ts") or "")),
    )
    out["robustness"] = {"turns": turns}


KERNEL_EVENT_SECTIONS: tuple[str, ...] = (
    "kernel_event",
    "kernel_lane_run",
    "kernel_rebench_attempt",
    "kernel_trace_analyze",
    "kernel_geak_attempt",
    "kernel_geak_discovery",
    "kernel_geak_acceptance",
    "kernel_discovered",
    "kernel_integrate",
)

#: The roofline substreams. Rows belong to whichever event tagged them: the
#: roofline's own when dispatched, the enclosing phase's when called inline.
ROOFLINE_EVENT_SECTIONS: tuple[str, ...] = (
    "roofline_event",
    "roofline_action",
    "roofline_profile_run",
    "roofline_analysis_run",
    "roofline_kernel",
)

BASELINE_EVENT_SECTIONS: tuple[str, ...] = (
    "baseline_event",
    "baseline_action",
    "baseline_run",
    "baseline_round",
)

CONC_SWEEP_EVENT_SECTIONS: tuple[str, ...] = (
    "conc_sweep_event",
    "conc_sweep_action",
    "conc_sweep_arm",
    "conc_sweep_variant",
    "conc_sweep_pair",
)

ENABLEMENT_EVENT_SECTIONS: tuple[str, ...] = (
    "enablement_event",
    "enablement_attempt",
    "enablement_build",
    "enablement_revalidation",
    "enablement_human_review",
)

PHASE_EVENT_SECTIONS: tuple[str, ...] = (
    "phase_event",
    "phase_segment",
    "phase_action",
    "phase_marker",
    "phase_proposal",
)

STACK_EVENT_SECTIONS: tuple[str, ...] = (
    "stack_event",
    "stack_adoption",
    "stack_validation",
)

WARM_REPLAY_EVENT_SECTIONS: tuple[str, ...] = (
    "warm_replay_event",
    "warm_replay_gate",
)

WARM_START_EVENT_SECTIONS: tuple[str, ...] = (
    "warm_start_event",
    "warm_start_read",
)

FRAMEWORK_EVENT_SECTIONS: tuple[str, ...] = (
    "framework_event",
    "framework_plateau",
    "framework_run",
    "framework_proposal",
    "framework_proposal_step",
    "framework_attempt",
    "framework_attempt_gate",
)

#: Every section holding v6 event rows. Consumed by the timeline rather than
#: the breakdown envelope, so assembly pops them out of the wire shape.
EVENT_SECTIONS: tuple[str, ...] = (
    KERNEL_EVENT_SECTIONS
    + ROOFLINE_EVENT_SECTIONS
    + BASELINE_EVENT_SECTIONS
    + CONC_SWEEP_EVENT_SECTIONS
    + ENABLEMENT_EVENT_SECTIONS
    + PHASE_EVENT_SECTIONS
    + STACK_EVENT_SECTIONS
    + WARM_REPLAY_EVENT_SECTIONS
    + WARM_START_EVENT_SECTIONS
    + FRAMEWORK_EVENT_SECTIONS
)


def event_parts(sections: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    """Read back the event rows of the bound session, keyed by section.

    Raises :exc:`SessionNotBoundError` when no session is bound, as do the
    per-event wrappers below, which only name their own section tuple.
    """
    from ...session.session_binding import bound_session  # local: avoid import cycle

    assembled = assemble_parts(bound_session(), warnings=[], keep_event_rows=True)
    parts: dict[str, list[dict[str, Any]]] = {}
    for section in sections:
        rows = assembled.get(section)
        parts[section] = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    return parts


def kernel_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the KERNEL substreams of the bound session, keyed by section."""
    return event_parts(KERNEL_EVENT_SECTIONS)


def roofline_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the roofline substreams of the bound session, keyed by section."""
    return event_parts(ROOFLINE_EVENT_SECTIONS)


def baseline_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the baseline substreams of the bound session, keyed by section."""
    return event_parts(BASELINE_EVENT_SECTIONS)


def conc_sweep_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the conc-sweep substreams of the bound session, keyed by section."""
    return event_parts(CONC_SWEEP_EVENT_SECTIONS)


def enablement_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the enablement substreams of the bound session, keyed by section."""
    return event_parts(ENABLEMENT_EVENT_SECTIONS)


def phase_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the phase substreams of the bound session, keyed by section."""
    return event_parts(PHASE_EVENT_SECTIONS)


def stack_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the stack-ledger substreams of the bound session, keyed by section."""
    return event_parts(STACK_EVENT_SECTIONS)


def warm_replay_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the warm-replay substreams of the bound session, keyed by section."""
    return event_parts(WARM_REPLAY_EVENT_SECTIONS)


def warm_start_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the warm-start substreams of the bound session, keyed by section."""
    return event_parts(WARM_START_EVENT_SECTIONS)


def framework_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the framework substreams of the bound session, keyed by section."""
    return event_parts(FRAMEWORK_EVENT_SECTIONS)


def _drop_event_rows(out: dict[str, Any]) -> None:
    """Drop the v6 event substreams from the breakdown envelope."""
    for section in EVENT_SECTIONS:
        out.pop(section, None)


__all__ = [
    "assemble_parts",
    "has_parts",
    "parts_dir",
]
