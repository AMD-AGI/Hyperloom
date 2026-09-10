# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The SBD V6 ``stack`` event: what was adopted, and what each adoption bought.

Everything else on the timeline is an attempt; this is the ordered list of
attempts that survived, with the throughput each one moved.

Two invariants carry the ledger. The throughput an adoption beat is recorded at
the moment the lift accepts it, because it is in scope there and nowhere
afterwards. And every contribution is measured against the *session baseline*,
the one denominator they all share: on it the contributions add up exactly, so
``unattributed_gain_pct`` means throughput that appeared between one adoption's
measurement and the next's, which no adoption claims.

The event spans the whole session because the stack does: adoptions arrive from
four phases into one ordered ledger, and scoping it by phase would cut the
chain at every boundary.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from .event_fields import (
    as_list as _as_list,
    float_or_none as _float_or_none,
    now_iso_seconds as _now,
    text_or_none as _text_or_none,
)
from .event_ids import event_id
from .event_rows import rows_for_event, sort_rows, wire_rows
from .event_sink import EventSink, make_sink
from .event_timeline import finish_event, open_event

# Every section a stack event assembles from. Named from the leaf module the
# assembler shares, so this writer reads its parts without an import cycle.
from .sections import STACK_EVENT_SECTIONS
from .recorder_warnings import note_failure

log = logging.getLogger(__name__)

EVENT_TYPE = "stack"
EVENT_KIND = "stack"

EVENT_COMPONENT = "stack"

#: A literal: the stack is one object every phase contributes to, and an id
#: scoped by ``state.phase`` would split the ledger at each boundary.
EVENT_PHASE = "stack"

#: A literal for the same reason: one stack per run.
EVENT_CYCLE = 0

PRODUCER = "orchestrator"

#: The session baseline every contribution is measured against, and the axis.
SECTION_EVENT = "stack_event"

#: One row per adoption, keyed by its position in the stack.
SECTION_ADOPTION = "stack_adoption"

#: One row per session validation, keyed by the stack length it validated, so a
#: second validation at one length supersedes the figure it revises.
SECTION_VALIDATION = "stack_validation"

#: Adoptions whose before or after throughput is missing, so their
#: contribution cannot be measured at all.
GUARD_UNMEASURED = "unmeasured"

#: Adoptions whose ``throughput_before`` does not match the previous adoption's
#: ``throughput_after``: each break is a place the run re-measured its anchor,
#: and their sum is exactly ``unattributed_gain_pct``. A large unattributed
#: figure with no breaks means the arithmetic is wrong.
GUARD_CHAIN_BREAKS = "chain_breaks"

#: Relative tolerance for calling two readings the same anchor: continuity is
#: a question about whether the run re-measured, not about float equality.
CHAIN_TOLERANCE_PCT = 0.001

# The attribution buckets. Recorded alongside the raw ``action``, so a row
# whose bucket turns out wrong can still be re-bucketed.
SOURCE_WARM_REPLAY = "warm_replay"
SOURCE_EXPLORE = "explore"
SOURCE_FRAMEWORK_AGENT = "framework_agent"
SOURCE_KERNEL = "kernel"
SOURCE_UNATTRIBUTED = "unattributed"

SOURCES: tuple[str, ...] = (
    SOURCE_WARM_REPLAY,
    SOURCE_EXPLORE,
    SOURCE_FRAMEWORK_AGENT,
    SOURCE_KERNEL,
    SOURCE_UNATTRIBUTED,
)

#: Action kind to attribution bucket. Deliberately small: it classifies the
#: handful of kinds that can reach the stack, not the action catalogue. An
#: unlisted kind lands in :data:`SOURCE_UNATTRIBUTED` and says so.
_SOURCE_BY_ACTION: dict[str, str] = {
    "replay_warm_recipe": SOURCE_WARM_REPLAY,
    "explore": SOURCE_EXPLORE,
    "conc_sweep": SOURCE_EXPLORE,
    "integrate": SOURCE_KERNEL,
    "gemm_tuning": SOURCE_KERNEL,
    "collective": SOURCE_KERNEL,
    "fusion": SOURCE_KERNEL,
    "geak_e2e": SOURCE_KERNEL,
    "integrate_patch": SOURCE_FRAMEWORK_AGENT,
    "framework_agent": SOURCE_FRAMEWORK_AGENT,
}

#: The kernel backends the per-backend split reports. Anything else is folded
#: into ``unattributed`` rather than minting a bucket per unrecognized string.
BACKENDS: tuple[str, ...] = ("geak", "forge")

STATUS_SUCCEEDED = "succeeded"
STATUS_DEGRADED = "degraded"
STATUS_SKIPPED = "skipped"


def stack_event_id() -> str:
    """Build the stack ledger's event id, ``stack:0:stack``."""
    return event_id(EVENT_PHASE, EVENT_CYCLE, EVENT_COMPONENT)


def source_for(action: str) -> str:
    """Classify an action kind into one of :data:`SOURCES`."""
    return _SOURCE_BY_ACTION.get(str(action or "").strip(), SOURCE_UNATTRIBUTED)


def _sink() -> EventSink | None:
    """The sink rows are written through; ``None`` when no session is bound."""
    try:
        from ...session.session_binding import bound_session_or_none

        if bound_session_or_none() is None:
            return None
        return make_sink(stack_event_id(), producer=PRODUCER)
    except Exception as exc:  # noqa: BLE001 — the stack outranks its own record
        note_failure(section="stack_event", error=exc, detail="stack event: cannot resolve a sink")
        return None


def _open(*, baseline_tput: Any = None, objective: str = "", start_time: str = "") -> int | None:
    """Put the ledger on the timeline, once, however many callers ask.

    ``None`` means the shell write failed; the fragments land either way and
    finalize recovers the event from them.
    """
    shell: dict[str, Any] = {}
    base = _float_or_none(baseline_tput)
    if base is not None and base > 0:
        shell["baseline_tput"] = base
    if objective:
        shell["objective"] = str(objective)
    if shell:
        # Onto the fragment too: assembly rebuilds ``ext`` from fragments and
        # never re-reads the shell.
        sink = _sink()
        if sink is not None:
            sink.record(SECTION_EVENT, dict(shell))
    return open_event(
        event_type=EVENT_TYPE,
        event=stack_event_id(),
        event_section=SECTION_EVENT,
        producer=PRODUCER,
        kind=EVENT_KIND,
        start_time=start_time or _now(),
        ext=shell,
    )


def record_adoption(
    *,
    stack_index: int,
    entry: Mapping[str, Any],
    throughput_before: Any,
    throughput_after: Any,
    baseline_tput: Any,
    objective: str = "",
    degrade_reason: str = "",
) -> None:
    """Record one adoption at the moment it is accepted onto the stack. Never raises.

    Called from the sole ``optimization_stack`` append, where every fact below
    is in scope: ``throughput_before`` in particular is the figure the lift just
    refused to accept the winner without beating, and this same call overwrites
    the anchor.
    """
    try:
        sink = _sink()
        if sink is None:
            return
        _open(baseline_tput=baseline_tput, objective=objective)
        before = _float_or_none(throughput_before)
        after = _float_or_none(throughput_after)
        base = _float_or_none(baseline_tput)
        action = str(entry.get("action") or "")
        row: dict[str, Any] = {
            "stack_index": int(stack_index),
            "recorded_at": _now(),
            "ts": _text_or_none(entry.get("ts")) or _now(),
            "action": action,
            "source": source_for(action),
            "variant_name": _text_or_none(entry.get("variant_name")),
            "lever_kind": _text_or_none(entry.get("lever_kind")),
            "operation_kind": _text_or_none(entry.get("operation_kind")),
            "scope": _text_or_none(entry.get("scope")),
            "backend": _text_or_none(entry.get("backend")),
            "source_phase": _text_or_none(entry.get("source_phase")),
            "task_id": _text_or_none(entry.get("task_id")),
            "kernel_id": _text_or_none(entry.get("kernel_id")),
            "fingerprint": _text_or_none(entry.get("fingerprint")),
            "provenance": _text_or_none(entry.get("provenance")),
            "gap_canonical_id": _text_or_none(entry.get("gap_canonical_id")),
            "objective": str(objective or ""),
            "degrade_reason": str(degrade_reason or ""),
            "throughput_before": before,
            "throughput_after": after,
            "baseline_tput": base,
            # On the session baseline, not ``throughput_before``: one
            # denominator makes the contributions sum to the chain total
            # exactly, so the residual is a fact and not an artifact.
            "contribution_pct": _pct(after, before, base),
            # The step's own gain over the anchor it beat, which is what the
            # promotion decision was actually made on.
            "local_gain_pct": _pct(after, before, before),
            "cumulative_gain_pct": _pct(after, base, base),
            "accuracy": _float_or_none(entry.get("accuracy")),
            "attribution_eligible": (
                bool(entry.get("attribution_eligible")) if "attribution_eligible" in entry else None
            ),
            "accepted_kernels": [str(k) for k in _as_list(entry.get("accepted_kernels")) if str(k)],
        }
        sink.record(SECTION_ADOPTION, row, row_type="adoption", natural_ids=str(int(stack_index)))
    except Exception as exc:  # noqa: BLE001 — an adoption outranks its own record
        note_failure(section="stack_event", error=exc, detail="stack event: adoption record failed")


def record_validation(
    *,
    stack_len: int,
    baseline_tput: Any,
    validated_tput: Any,
    validated_gain_pct: Any,
    source: str = "",
    measurement_basis: str = "",
    ts: str = "",
    ttft_mean_ms: Any = None,
    e2el_mean_ms: Any = None,
    ttft_e2el_source: str = "",
    server_launch_flags: str = "",
    workspace: Any = None,
) -> None:
    """Record one session validation of the stack as a whole. Never raises.

    This is the ledger's only independent check on itself: a figure measured on
    the whole stack, against which the sum of the parts can be reconciled.
    Without it the session total is the sum of the very steps it is meant to be
    checking.

    ``stack_len`` keys the row, so a later validation at one length supersedes
    the earlier. ``measurement_basis`` is ``e2e_rebench`` for a full-stack
    revalidation or ``e2e_decision_round`` for the round a variant was graded
    on. The latency pair and ``server_launch_flags`` are carried here because
    the run that produced ``validated_tput`` resolves them and they cannot be
    recovered afterwards.
    """
    try:
        sink = _sink()
        if sink is None:
            return
        _open(baseline_tput=baseline_tput)
        sink.record(
            SECTION_VALIDATION,
            {
                "stack_len": int(stack_len or 0),
                "ts": str(ts or "") or _now(),
                "baseline_tput": _float_or_none(baseline_tput),
                "validated_tput": _float_or_none(validated_tput),
                "validated_gain_pct": _float_or_none(validated_gain_pct),
                "source": str(source or ""),
                "measurement_basis": str(measurement_basis or ""),
                "ttft_mean_ms": _float_or_none(ttft_mean_ms),
                "e2el_mean_ms": _float_or_none(e2el_mean_ms),
                "ttft_e2el_source": str(ttft_e2el_source or ""),
                "server_launch_flags": str(server_launch_flags or ""),
                "workspace": _text_or_none(workspace),
            },
            row_type="validation",
            natural_ids=str(int(stack_len or 0)),
        )
    except Exception as exc:  # noqa: BLE001
        note_failure(section="stack_event", error=exc, detail="stack event: validation record failed")


def finish(*, end_time: str = "") -> None:
    """Close the ledger on the rows recorded against it. Never raises.

    Called at close, where the stack stops changing. A run that ended without
    reaching close leaves the event open and finalize recovers it as
    ``interrupted``: the ledger was never settled.
    """
    try:
        sink = _sink()
        if sink is None:
            return
        sequence = _open()
        closed = str(end_time or "") or _now()
        sink.record(SECTION_EVENT, {"end_time": closed})

        from .assembler import event_parts

        ext, status = assemble_stack_ext(
            event_parts(STACK_EVENT_SECTIONS, event=stack_event_id()), event=stack_event_id()
        )
        finish_event(
            event_type=EVENT_TYPE,
            event=stack_event_id(),
            sequence=sequence,
            status=status or STATUS_SKIPPED,
            ext=ext,
            kind=EVENT_KIND,
            end_time=closed,
        )
    except Exception as exc:  # noqa: BLE001
        note_failure(section="stack_event", error=exc, detail="stack event: finish failed")


def _pct(value: Any, against: Any, denominator: Any) -> float | None:
    """Express ``value - against`` as a percentage of ``denominator``.

    A shared denominator is what lets several of these be added together.
    ``None`` -- a missing input or a non-positive denominator -- means
    unmeasurable, a different fact from zero.
    """
    lo, hi, base = _float_or_none(against), _float_or_none(value), _float_or_none(denominator)
    if lo is None or hi is None or base is None or base <= 0:
        return None
    return round((hi - lo) / base * 100.0, 6)


#: Every section the stack event assembles from. Duplicated from the assembler
#: so :func:`finish` can read its own parts without an import cycle.
def assemble_stack_ext(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
) -> tuple[dict[str, Any], str]:
    """Assemble the stack ledger's ``ext`` and status out of its recorded rows.

    ``parts`` maps section name to rows read back from the spool.
    """
    header = _header(rows_for_event(parts.get(SECTION_EVENT) or [], event))
    adoptions = wire_rows(
        sort_rows(rows_for_event(parts.get(SECTION_ADOPTION) or [], event), keys=("stack_index",)),
        drop=("event_id",),
    )
    validations = wire_rows(
        sort_rows(rows_for_event(parts.get(SECTION_VALIDATION) or [], event), keys=("stack_len", "ts")),
        drop=("event_id",),
    )
    baseline = _float_or_none(header.get("baseline_tput"))
    measured = [row for row in adoptions if _float_or_none(row.get("contribution_pct")) is not None]
    attributed = round(sum(float(row["contribution_pct"]) for row in measured), 6) if measured else 0.0
    # Not a sum: asking two figures the same question is how the ledger checks
    # itself below.
    chain_total = _last_cumulative(adoptions)
    settled = validations[-1] if validations else {}
    recorded_total = _float_or_none(settled.get("validated_gain_pct"))
    ext: dict[str, Any] = {
        "baseline_tput": baseline,
        "objective": str(header.get("objective") or ""),
        "adoptions": {
            "count": len(adoptions),
            "by_source": _by_source(adoptions),
            "rows": adoptions,
        },
        "validations": {
            "count": len(validations),
            "rows": validations,
            "settled": settled or None,
            # A validation behind the head means the session total is a claim
            # about a shorter stack than the one that shipped.
            "at_head": (
                int(settled.get("stack_len") or 0) >= len(adoptions) if settled and adoptions else bool(settled)
            ),
        },
        "attributed_gain_pct": attributed,
        "chain_total_gain_pct": chain_total,
        # Throughput no adoption claims, the anchor having moved between two
        # measurements. An identity: the sum of the chain breaks below.
        "unattributed_gain_pct": (round(chain_total - attributed, 6) if chain_total is not None and measured else None),
        "validated_total_gain_pct": recorded_total,
        # The ledger against the one figure measured on the whole stack: the
        # parts and the whole disagreeing means one of them is wrong.
        "reconciliation_gap_pct": (
            round(recorded_total - chain_total, 6) if recorded_total is not None and chain_total is not None else None
        ),
        "guards": {
            GUARD_UNMEASURED: len(adoptions) - len(measured),
            GUARD_CHAIN_BREAKS: _chain_breaks(adoptions),
        },
    }
    return ext, _status_for(adoptions, chain_total)


def _by_source(adoptions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum each bucket's contribution, on the denominator they all share.

    Buckets with no adoptions are present and zeroed, so a reader can tell
    "this subsystem earned nothing" from "this subsystem is not reported".
    """
    buckets: dict[str, Any] = {name: {"count": 0, "total_gain_pct": 0.0, "unmeasured": 0} for name in SOURCES}
    backends: dict[str, Any] = {
        name: {"count": 0, "total_gain_pct": 0.0, "unmeasured": 0} for name in (*BACKENDS, SOURCE_UNATTRIBUTED)
    }
    for row in adoptions:
        bucket = buckets.setdefault(
            str(row.get("source") or SOURCE_UNATTRIBUTED),
            {"count": 0, "total_gain_pct": 0.0, "unmeasured": 0},
        )
        share = _float_or_none(row.get("contribution_pct"))
        bucket["count"] += 1
        if share is None:
            bucket["unmeasured"] += 1
        else:
            bucket["total_gain_pct"] = round(float(bucket["total_gain_pct"]) + share, 6)
        if str(row.get("source") or "") != SOURCE_KERNEL:
            continue
        backend = str(row.get("backend") or "")
        slot = backends[backend] if backend in BACKENDS else backends[SOURCE_UNATTRIBUTED]
        slot["count"] += 1
        if share is None:
            slot["unmeasured"] += 1
        else:
            slot["total_gain_pct"] = round(float(slot["total_gain_pct"]) + share, 6)
    buckets[SOURCE_KERNEL]["by_backend"] = backends
    return buckets


def _last_cumulative(adoptions: Sequence[Mapping[str, Any]]) -> float | None:
    """The last recorded ``cumulative_gain_pct``, or ``None`` when no adoption
    measured one."""
    for row in reversed(list(adoptions)):
        value = _float_or_none(row.get("cumulative_gain_pct"))
        if value is not None:
            return value
    return None


def _chain_breaks(adoptions: Sequence[Mapping[str, Any]]) -> int:
    """Count the adoptions whose anchor is not the previous one's reading, i.e.
    how many times the anchor moved between two adoptions."""
    breaks = 0
    previous: float | None = None
    for row in adoptions:
        before = _float_or_none(row.get("throughput_before"))
        if previous is not None and before is not None and previous > 0:
            if abs(before - previous) / previous > CHAIN_TOLERANCE_PCT:
                breaks += 1
        after = _float_or_none(row.get("throughput_after"))
        if after is not None:
            previous = after
    return breaks


def _status_for(adoptions: Sequence[Mapping[str, Any]], chain_total: float | None) -> str:
    """The status the ledger reports: ``skipped`` for a stack that adopted
    nothing, ``degraded`` for one with adoptions and no measurable total."""
    if not adoptions:
        return STATUS_SKIPPED
    return STATUS_SUCCEEDED if chain_total is not None else STATUS_DEGRADED


def _header(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold the event-level fragments into one header."""
    header: dict[str, Any] = {}
    for row in rows:
        if isinstance(row, Mapping):
            header.update({k: v for k, v in row.items() if k != "event_id"})
    return header


__all__ = [
    "BACKENDS",
    "CHAIN_TOLERANCE_PCT",
    "EVENT_COMPONENT",
    "EVENT_KIND",
    "EVENT_TYPE",
    "GUARD_CHAIN_BREAKS",
    "GUARD_UNMEASURED",
    "PRODUCER",
    "SECTION_ADOPTION",
    "SECTION_EVENT",
    "SECTION_VALIDATION",
    "SOURCES",
    "SOURCE_EXPLORE",
    "SOURCE_FRAMEWORK_AGENT",
    "SOURCE_KERNEL",
    "SOURCE_UNATTRIBUTED",
    "SOURCE_WARM_REPLAY",
    "STACK_EVENT_SECTIONS",
    "STATUS_DEGRADED",
    "STATUS_SKIPPED",
    "STATUS_SUCCEEDED",
    "assemble_stack_ext",
    "finish",
    "record_adoption",
    "record_validation",
    "source_for",
    "stack_event_id",
]
