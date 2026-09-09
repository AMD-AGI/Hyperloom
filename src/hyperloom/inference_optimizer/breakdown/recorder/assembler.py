# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-side of the breakdown recorder.

Assembles the per-producer fragments written by :class:`~.recorder.Recorder`
into a ``{section: value}`` mapping ready to drop into the
``session_breakdown.json`` envelope:

* ``singleton`` sections -> the payload of the latest fragment (by ``ts``).
* plain ``item`` sections -> payloads concatenated into a list, ordered by
  ``seq`` then ``ts``.
A compose pass runs last and reconciles across fragments: the critic,
robustness, and close substreams are folded into their composed sections, and
``versions`` collapses into ``metadata.versions.tools`` as a ``{tool: meta}``
map (last row per tool wins). Bad/partial fragments are skipped and noted in
``warnings``.
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
    """True iff at least one record fragment exists for this session.

    Returns:
        ``True`` when the spool directory holds at least one ``*.json``
        fragment.
    """
    d = parts_dir(session_dir)
    return d.is_dir() and any(d.glob("*.json"))


def _load(path: Path, warnings: list[str]) -> dict[str, Any] | None:
    """Read and parse one fragment file, noting problems into ``warnings``.

    Args:
        path (Path): the fragment file to read.
        warnings (list[str]): a list that parse/validation warnings are
            appended to.

    Returns:
        dict[str, Any] | None: the parsed fragment record, or ``None`` when it
            cannot be read or is not a JSON object.
    """
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
    """Return ``{section: list | dict}`` assembled from the spool directory.

    Empty mapping when no fragments exist (caller falls back to collectors).

    Args:
        session_dir: The session root directory.
        warnings: Optional list to append parse/validation warnings to; a
            fresh list is used when not provided.
        keep_event_rows: Retain the :data:`EVENT_SECTIONS` substreams instead
            of dropping them. Only :func:`event_parts` sets this; the breakdown
            envelope never wants them.

    Returns:
        A ``{section: list | dict}`` mapping assembled from the spool
        directory, or ``{}`` when no fragments exist.
    """
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

    # A singleton fragment is named for its producer, so a section with more
    # than one is a section two producers both claimed. Only the newest
    # survives, and the other producer's payload does not merge into it -- it
    # is dropped whole. Nothing downstream can see that it existed.
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
    """Fold the ``close_step`` item substream into ``close.steps``. Pops the raw
    substream so it doesn't leak into the breakdown envelope.

    Unlike the other compose helpers this one merges into a directly-recorded
    singleton rather than deferring to it: the CLOSE sequencer writes both, the
    singleton for the close-out's own facts and one row per step it settles, so
    neither is a substitute for the other.

    Args:
        out: The assembled section mapping mutated in place.
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
    # ``assemble_parts`` has already ordered these by ``(seq, ts)``, which is
    # the write order within one process. A resumed session closes in a second
    # process whose sequence restarts at zero, so the timestamp is what orders
    # the two passes; the sort is stable, leaving the write order to break ties
    # between steps that settled inside the same microsecond.
    steps.sort(key=lambda row: str(row.get("ts") or ""))
    close["steps"] = steps
    out["close"] = close


def _compose_write_back(out: dict[str, Any]) -> dict[str, Any] | None:
    """Fold the Recipe KB publication into one ``close.kb_write_back`` block.

    Returns ``None`` when nothing was recorded, which is how a session that
    never attempted a publication is told apart from one whose attempt never
    settled: the first has no block at all, the second has an attempt row
    still standing at ``pending``.

    Args:
        out: The assembled section mapping; the raw substreams are popped.

    Returns:
        The composed block, or ``None`` when neither substream exists.
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
        # Attempts but no settled arc: the publication was opened and the
        # process died before anything answered. Reported as pending rather
        # than as a failure the store never actually returned.
        block["status"] = _WRITE_BACK_PENDING
    return block


#: Mirrors ``close_out.STATUS_PENDING``; spelled out because that module
#: imports this one to read its own parts back.
_WRITE_BACK_PENDING = "pending"


def close_steps(session_dir: Path | str) -> list[dict[str, Any]]:
    """Read back the close steps recorded for ``session_dir``, in order.

    Assembly folds this substream into ``close.steps``, so the sequencer
    deriving its own verdict reads it through here rather than re-globbing the
    spool.

    Returns:
        The recorded step rows, oldest first; empty when none were recorded.
    """
    close = assemble_parts(session_dir, warnings=[]).get("close")
    if not isinstance(close, dict):
        return []
    steps = close.get("steps")
    return [row for row in steps if isinstance(row, dict)] if isinstance(steps, list) else []


def _compose_versions(out: dict[str, Any]) -> None:
    """Fold the ``versions`` item substream into ``metadata.versions.tools``.

    Rows arrive keyed by tool name, so one row per tool is the normal case and
    a later row for a tool it already saw replaces it. Pops the raw substream
    so it doesn't leak into the breakdown envelope.

    The ``metadata`` section is created when it is missing: the tools are the
    only provenance some sessions record, and dropping them because the
    Coordinator never wrote its own half would lose the whole answer.

    Args:
        out: The assembled section mapping mutated in place.
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
    """Fold the ``critic_iteration`` item substream into the ``critic`` view,
    ordered as the agent ran. Pops the raw substream so it doesn't leak into
    the breakdown envelope.

    Args:
        out: The assembled section mapping mutated in place.
    """
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
    """Fold the ``robustness_turn`` item substream into the ``robustness``
    view, ordered by turn. Pops the raw substream so it doesn't leak into the
    breakdown envelope.

    Args:
        out: The assembled section mapping mutated in place.
    """
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


#: The KERNEL substreams, in the order a reader follows them.
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

#: The roofline substreams. They belong to whichever event their rows are
#: tagged with, which is the roofline's own event when it was dispatched and
#: the enclosing phase's event when it was called inline.
ROOFLINE_EVENT_SECTIONS: tuple[str, ...] = (
    "roofline_event",
    "roofline_action",
    "roofline_profile_run",
    "roofline_analysis_run",
    "roofline_kernel",
)

#: The baseline substreams, in the order a reader follows them: the event, the
#: measurements dispatched into it, each measurement's passes, and each pass's
#: benchmark rounds.
BASELINE_EVENT_SECTIONS: tuple[str, ...] = (
    "baseline_event",
    "baseline_action",
    "baseline_run",
    "baseline_round",
)

#: The conc-sweep substreams, in the order a reader follows them: the event,
#: the sweep dispatched into it, the sweep's two arms, each arm's rungs, and
#: the concurrencies the arms are paired at.
CONC_SWEEP_EVENT_SECTIONS: tuple[str, ...] = (
    "conc_sweep_event",
    "conc_sweep_action",
    "conc_sweep_arm",
    "conc_sweep_variant",
    "conc_sweep_pair",
)

#: The enablement lane's sections: the lane itself with its trigger and
#: terminal, one row per authoring round, the targeted builds it ran, the
#: eval-origin revalidation windows it opened, and the launch failures it could
#: not classify well enough to dispatch a round for.
ENABLEMENT_EVENT_SECTIONS: tuple[str, ...] = (
    "enablement_event",
    "enablement_attempt",
    "enablement_build",
    "enablement_revalidation",
    "enablement_human_review",
)

#: A phase's sections: the span it covered, one row per entry into it, one per
#: action dispatched from it, and one per non-transition marker raised in it.
PHASE_EVENT_SECTIONS: tuple[str, ...] = (
    "phase_event",
    "phase_segment",
    "phase_action",
    "phase_marker",
    "phase_proposal",
)

#: The stack ledger's sections: the ledger itself with the session baseline
#: every contribution is measured against, one row per adoption, and one row
#: per session validation of the stack as a whole.
STACK_EVENT_SECTIONS: tuple[str, ...] = (
    "stack_event",
    "stack_adoption",
    "stack_validation",
)

#: The warm-replay event's sections: the replay's own request, measurement and
#: verdict, and one row per gate it was judged by.
WARM_REPLAY_EVENT_SECTIONS: tuple[str, ...] = (
    "warm_replay_event",
    "warm_replay_gate",
)

#: The warm-start event's sections: the T0 lookup's own request and match, and
#: one row per KB read it made.
WARM_START_EVENT_SECTIONS: tuple[str, ...] = (
    "warm_start_event",
    "warm_start_read",
)

#: The framework event's sections: the phase entry's own policy and exit, its
#: plateau evaluations, and the three links of the proposal chain -- the runs
#: that produced proposals, the proposals themselves with their lifecycle
#: steps, and the attempts they funnelled into with their gates.
FRAMEWORK_EVENT_SECTIONS: tuple[str, ...] = (
    "framework_event",
    "framework_plateau",
    "framework_run",
    "framework_proposal",
    "framework_proposal_step",
    "framework_attempt",
    "framework_attempt_gate",
)

#: Every section holding v6 event rows. They are consumed by the timeline
#: rather than by the breakdown envelope, so assembly pops them here to keep
#: them from leaking into the wire shape.
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

    Assembly pops these sections, so a caller that wants them -- the phase
    closing an event, or finalize recovering one that never closed -- reads
    them through here instead.

    Args:
        sections: The sections to read, e.g. :data:`KERNEL_EVENT_SECTIONS`.

    Returns:
        A ``{section: [payload, ...]}`` mapping holding those sections, each
        defaulting to an empty list.

    Raises:
        SessionNotBoundError: If no session is bound.
    """
    from ...session.session_binding import bound_session  # local: avoid import cycle

    assembled = assemble_parts(bound_session(), warnings=[], keep_event_rows=True)
    parts: dict[str, list[dict[str, Any]]] = {}
    for section in sections:
        rows = assembled.get(section)
        parts[section] = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    return parts


def kernel_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the KERNEL substreams of the bound session, keyed by section.

    Returns:
        A ``{section: [payload, ...]}`` mapping over
        :data:`KERNEL_EVENT_SECTIONS`.

    Raises:
        SessionNotBoundError: If no session is bound.
    """
    return event_parts(KERNEL_EVENT_SECTIONS)


def roofline_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the roofline substreams of the bound session, keyed by section.

    Returns:
        A ``{section: [payload, ...]}`` mapping over
        :data:`ROOFLINE_EVENT_SECTIONS`.

    Raises:
        SessionNotBoundError: If no session is bound.
    """
    return event_parts(ROOFLINE_EVENT_SECTIONS)


def baseline_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the baseline substreams of the bound session, keyed by section.

    Returns:
        A ``{section: [payload, ...]}`` mapping over
        :data:`BASELINE_EVENT_SECTIONS`.

    Raises:
        SessionNotBoundError: If no session is bound.
    """
    return event_parts(BASELINE_EVENT_SECTIONS)


def conc_sweep_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the conc-sweep substreams of the bound session, keyed by section.

    Returns:
        A ``{section: [payload, ...]}`` mapping over
        :data:`CONC_SWEEP_EVENT_SECTIONS`.

    Raises:
        SessionNotBoundError: If no session is bound.
    """
    return event_parts(CONC_SWEEP_EVENT_SECTIONS)


def enablement_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the enablement substreams of the bound session, keyed by section.

    Returns:
        A ``{section: [payload, ...]}`` mapping over
        :data:`ENABLEMENT_EVENT_SECTIONS`.

    Raises:
        SessionNotBoundError: If no session is bound.
    """
    return event_parts(ENABLEMENT_EVENT_SECTIONS)


def phase_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the phase substreams of the bound session, keyed by section.

    Returns:
        A ``{section: [payload, ...]}`` mapping over
        :data:`PHASE_EVENT_SECTIONS`.

    Raises:
        SessionNotBoundError: If no session is bound.
    """
    return event_parts(PHASE_EVENT_SECTIONS)


def stack_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the stack-ledger substreams of the bound session, keyed by section.

    Returns:
        A ``{section: [payload, ...]}`` mapping over
        :data:`STACK_EVENT_SECTIONS`.

    Raises:
        SessionNotBoundError: If no session is bound.
    """
    return event_parts(STACK_EVENT_SECTIONS)


def warm_replay_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the warm-replay substreams of the bound session, keyed by section.

    Returns:
        A ``{section: [payload, ...]}`` mapping over
        :data:`WARM_REPLAY_EVENT_SECTIONS`.

    Raises:
        SessionNotBoundError: If no session is bound.
    """
    return event_parts(WARM_REPLAY_EVENT_SECTIONS)


def warm_start_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the warm-start substreams of the bound session, keyed by section.

    Returns:
        A ``{section: [payload, ...]}`` mapping over
        :data:`WARM_START_EVENT_SECTIONS`.

    Raises:
        SessionNotBoundError: If no session is bound.
    """
    return event_parts(WARM_START_EVENT_SECTIONS)


def framework_event_parts() -> dict[str, list[dict[str, Any]]]:
    """Return the framework substreams of the bound session, keyed by section.

    Returns:
        A ``{section: [payload, ...]}`` mapping over
        :data:`FRAMEWORK_EVENT_SECTIONS`.

    Raises:
        SessionNotBoundError: If no session is bound.
    """
    return event_parts(FRAMEWORK_EVENT_SECTIONS)


def _drop_event_rows(out: dict[str, Any]) -> None:
    """Drop the v6 event substreams from the breakdown envelope.

    These are recorded for the v6 timeline, which assembles them into events
    when the phase or action that produced them ends. They carry no meaning of
    their own in ``session_breakdown.json``, and leaving them in would publish
    ten undocumented sections alongside the events built from them.

    Args:
        out: The assembled section mapping mutated in place.
    """
    for section in EVENT_SECTIONS:
        out.pop(section, None)


__all__ = [
    "assemble_parts",
    "has_parts",
    "parts_dir",
]
