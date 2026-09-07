# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The SBD V6 ``stack`` event: what was adopted, and what each adoption bought.

The optimization stack is the session's answer to the question it was run to
answer. Everything else on the timeline is an attempt; this is the ordered list
of attempts that survived, with the throughput each one moved.

``outcome.validation`` -- the attributed / unattributed / reconciliation figures
and the per-source attribution -- was projected at export from the legacy
``optimizations`` section, which reconstructed the whole ledger from the v4
``operations`` / ``measurements`` / ``adoptions`` streams. Three things went
wrong in that reconstruction, and all three are the same mistake:

**The throughput each adoption beat was derived, not recorded.** ``entries[]``
carries ``throughput_after`` and no ``throughput_before``; the chain's before
figures were inferred by walking the attempt list and pairing rows up. The
figure an adoption actually beat is :attr:`GradedComparison.reference`, which is
in scope at the moment the lift is accepted -- the lift refuses the winner
unless it beats exactly that number -- and was thrown away.

**The step gains were summed on mismatched denominators.** Each step's gain was
computed against the previous step's throughput and the results were added,
which is not how ratios compose. The drift this produced was then published as
``unattributed_gain_pct``, as though it were gain nobody could account for
rather than an artifact of the arithmetic. Here every step's contribution is
measured against the *session baseline*, the one denominator they share, so the
contributions add up exactly and ``unattributed_gain_pct`` means the one thing
it should: throughput that appeared between one adoption's measurement and the
next one's, which no adoption claims.

**Eight guard counts existed to detect the inconsistencies.**
``stale_evidence_count``, ``unclaimed_integration_count``,
``unscored_keep_count`` and the rest were each a check on whether the three v4
streams agreed with one another. A fact recorded once at the moment it becomes
true has nothing to disagree with, so most of them have no referent here. The
two that survive -- :data:`GUARD_UNMEASURED` and :data:`GUARD_CHAIN_BREAKS` --
are about the measurements themselves rather than about the bookkeeping.

The event spans the whole session because the stack does: adoptions arrive from
PRELUDE warm replay, from EXPLORE, from FRAMEWORK_AGENT integrate, and from
KERNEL_AGENT, and they form one ordered ledger. Scoping it by phase would cut
the chain at every phase boundary, which is precisely where the interesting
steps are.
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

log = logging.getLogger(__name__)

EVENT_TYPE = "stack"
EVENT_KIND = "stack"

#: The component segment of the stack event's id.
EVENT_COMPONENT = "stack"

#: The phase segment. A literal, like the enablement lane's: the stack is one
#: object that every phase contributes to, and an id scoped by ``state.phase``
#: would split the ledger at each phase boundary -- exactly where the chain's
#: before / after pairs have to line up for the reconciliation to mean anything.
EVENT_PHASE = "stack"

#: The macro-cycle segment, a literal for the same reason: one stack per run.
EVENT_CYCLE = 0

PRODUCER = "orchestrator"

#: The event-level section: the session baseline every contribution is measured
#: against, and the axis it was graded on.
SECTION_EVENT = "stack_event"

#: One row per adoption, keyed by its position in the stack.
SECTION_ADOPTION = "stack_adoption"

#: One row per session validation, keyed by the stack length it validated. A
#: second validation at one length supersedes the first, which is the right
#: reading: a re-measurement of the same stack replaces the figure it revises.
SECTION_VALIDATION = "stack_validation"

#: Adoptions whose before or after throughput is missing, so their contribution
#: cannot be measured at all. The legacy ledger called these
#: ``unmeasured_keep_count`` and reached them through ``gain_method == missing``.
GUARD_UNMEASURED = "unmeasured"

#: Adoptions whose ``throughput_before`` does not match the previous adoption's
#: ``throughput_after``. Each break is a place the run re-measured its anchor
#: between two adoptions, and the sum of the breaks is exactly
#: ``unattributed_gain_pct``. This is the guard worth reading: a large
#: unattributed figure with no chain breaks would mean the arithmetic is wrong,
#: and with many breaks it means the anchor moved for reasons the ledger cannot
#: attribute.
GUARD_CHAIN_BREAKS = "chain_breaks"

#: Relative tolerance for calling two throughput readings the same anchor.
#: Chain continuity is a question about whether the run re-measured, not about
#: float equality.
CHAIN_TOLERANCE_PCT = 0.001

# The attribution buckets. Named here rather than derived from the action kind
# at read time so the classification lives in one place, and recorded alongside
# the raw ``action`` so a row whose bucket is wrong can still be re-bucketed.
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
#: handful of kinds that can reach the stack at all, not the action catalogue.
#: An unlisted kind lands in :data:`SOURCE_UNATTRIBUTED` and says so, which is
#: what the legacy ledger did and is the honest reading -- a gain nobody claimed
#: is a finding, not a rounding error.
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
    """Build the stack ledger's event id.

    Returns:
        str: ``stack:0:stack``. Both leading segments are literals; see
        :data:`EVENT_PHASE`.
    """
    return event_id(EVENT_PHASE, EVENT_CYCLE, EVENT_COMPONENT)


def source_for(action: str) -> str:
    """Classify an action kind into its attribution bucket.

    Args:
        action (str): The action kind that produced the adoption.

    Returns:
        str: One of :data:`SOURCES`.
    """
    return _SOURCE_BY_ACTION.get(str(action or "").strip(), SOURCE_UNATTRIBUTED)


def _sink() -> EventSink | None:
    """The sink every row here is written through, or ``None`` with no session.

    Returns:
        EventSink | None: The bound session's sink, or ``None`` when nothing is
        bound. Recording is best-effort: an adoption that cannot be recorded is
        still an adoption.
    """
    try:
        from ...session.session_binding import bound_session_or_none

        if bound_session_or_none() is None:
            return None
        return make_sink(stack_event_id(), producer=PRODUCER)
    except Exception:  # noqa: BLE001 — the stack outranks its own record
        log.debug("stack event: cannot resolve a sink", exc_info=True)
        return None


def _open(*, baseline_tput: Any = None, objective: str = "", start_time: str = "") -> int | None:
    """Put the ledger on the timeline, once, however many callers ask.

    Args:
        baseline_tput (Any): The session baseline every contribution is
            measured against.
        objective (str): The axis the run grades on.
        start_time (str): When the ledger opened; defaults to now.

    Returns:
        int | None: The storage sequence to close with, or ``None`` when the
        shell write failed. The fragments land either way and finalize recovers
        the event from them.
    """
    shell: dict[str, Any] = {}
    base = _float_or_none(baseline_tput)
    if base is not None and base > 0:
        shell["baseline_tput"] = base
    if objective:
        shell["objective"] = str(objective)
    if shell:
        # Onto the fragment as well as the shell: assembly rebuilds ``ext`` from
        # the fragments and never re-reads the shell, and the shell write is
        # skipped on every open after the first.
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
    is in scope. ``throughput_before`` in particular is the figure the lift just
    refused to accept the winner without beating, so it is known exactly here
    and nowhere afterwards -- the anchor is overwritten by this same call.

    Args:
        stack_index (int): The adoption's position in the stack, which keys its
            row. Zero-based, so it indexes ``optimization_stack`` directly.
        entry (Mapping[str, Any]): The stack entry being appended, read for the
            adoption's identity and provenance.
        throughput_before (Any): The anchor this adoption beat.
        throughput_after (Any): What it measured.
        baseline_tput (Any): The session baseline, which is the denominator
            every contribution in the ledger shares.
        objective (str): The axis both readings were taken on.
        degrade_reason (str): Why the run's requested axis did not apply, when
            it did not.
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
            # Measured on the session baseline, not on ``throughput_before``.
            # This is the number that makes the ledger add up: contributions on
            # one denominator sum to the chain total exactly, and the residual
            # is then a real fact about the run rather than a rounding artifact.
            "contribution_pct": _pct(after, before, base),
            # The step's own gain over the anchor it beat, kept because it is
            # what the promotion decision was actually made on.
            "local_gain_pct": _pct(after, before, before),
            "cumulative_gain_pct": _pct(after, base, base),
            "accuracy": _float_or_none(entry.get("accuracy")),
            "attribution_eligible": (
                bool(entry.get("attribution_eligible")) if "attribution_eligible" in entry else None
            ),
            "accepted_kernels": [str(k) for k in _as_list(entry.get("accepted_kernels")) if str(k)],
        }
        sink.record(SECTION_ADOPTION, row, row_type="adoption", natural_ids=str(int(stack_index)))
    except Exception:  # noqa: BLE001 — an adoption outranks its own record
        log.debug("stack event: adoption record failed", exc_info=True)


def record_validation(
    *,
    stack_len: int,
    baseline_tput: Any,
    validated_tput: Any,
    validated_gain_pct: Any,
    source: str = "",
    measurement_basis: str = "",
    ts: str = "",
) -> None:
    """Record one session validation of the stack as a whole. Never raises.

    This is the ledger's only independent check on itself: a figure measured on
    the whole stack, against which the sum of the parts can be reconciled.
    Without it the session total is the sum of the very steps it is meant to be
    checking.

    Args:
        stack_len (int): The stack length this figure validates, which keys the
            row. A later validation at one length supersedes the earlier.
        baseline_tput (Any): The baseline the figure was graded against.
        validated_tput (Any): The measured throughput of the whole stack.
        validated_gain_pct (Any): The gain the run recorded for it.
        source (str): Which promotion path produced the figure.
        measurement_basis (str): How the reading was obtained --
            ``e2e_rebench`` for a full-stack revalidation,
            ``e2e_decision_round`` for the round a variant was graded on.
        ts (str): The author-time stamp of the validation.
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
            },
            row_type="validation",
            natural_ids=str(int(stack_len or 0)),
        )
    except Exception:  # noqa: BLE001
        log.debug("stack event: validation record failed", exc_info=True)


def finish(*, end_time: str = "") -> None:
    """Close the ledger on the rows recorded against it. Never raises.

    Called at close, where the stack stops changing. A run that ended without
    reaching close leaves the event open and finalize recovers it as
    ``interrupted``, which is the honest reading: the ledger was never settled.

    Args:
        end_time (str): When the ledger closed; defaults to now.
    """
    try:
        sink = _sink()
        if sink is None:
            return
        sequence = _open()
        closed = str(end_time or "") or _now()
        sink.record(SECTION_EVENT, {"end_time": closed})

        from .assembler import event_parts

        ext, status = assemble_stack_ext(event_parts(STACK_EVENT_SECTIONS), event=stack_event_id())
        finish_event(
            event_type=EVENT_TYPE,
            event=stack_event_id(),
            sequence=sequence,
            status=status or STATUS_SKIPPED,
            ext=ext,
            kind=EVENT_KIND,
            end_time=closed,
        )
    except Exception:  # noqa: BLE001
        log.debug("stack event: finish failed", exc_info=True)


def _pct(value: Any, against: Any, denominator: Any) -> float | None:
    """Express ``value - against`` as a percentage of ``denominator``.

    Args:
        value (Any): The measured figure.
        against (Any): What it is compared to.
        denominator (Any): The base the difference is expressed against, which
            is what lets several of these be added together.

    Returns:
        float | None: The percentage, or ``None`` when any input is missing or
        the denominator is not positive. ``None`` means unmeasurable, which is
        a different fact from zero and is counted as such by
        :data:`GUARD_UNMEASURED`.
    """
    lo, hi, base = _float_or_none(against), _float_or_none(value), _float_or_none(denominator)
    if lo is None or hi is None or base is None or base <= 0:
        return None
    return round((hi - lo) / base * 100.0, 6)


#: Every section the stack event assembles from. Declared here as well as in
#: the assembler so :func:`finish` can read its own parts without importing the
#: assembler's tuple, which would close an import cycle.
STACK_EVENT_SECTIONS: tuple[str, ...] = (
    SECTION_EVENT,
    SECTION_ADOPTION,
    SECTION_VALIDATION,
)


def assemble_stack_ext(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
) -> tuple[dict[str, Any], str]:
    """Assemble the stack ledger's ``ext`` out of its recorded rows.

    Args:
        parts (Mapping[str, list[dict[str, Any]]]): The stack sections as read
            back from the spool, section name to row list.
        event (str): The event id to assemble.

    Returns:
        tuple[dict[str, Any], str]: The ``ext`` payload and the status the
            ledger reports.
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
    # The chain total is the last adoption's own reading against the baseline,
    # not a sum: it is one measurement, and asking two figures the same question
    # is how the ledger gets to check itself below.
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
            # Whether the figure the run last validated covers the stack it
            # ended with. A validation behind the head means the last adoptions
            # were never measured as a whole, so the session total is a claim
            # about a shorter stack than the one that shipped.
            "at_head": (
                int(settled.get("stack_len") or 0) >= len(adoptions) if settled and adoptions else bool(settled)
            ),
        },
        "attributed_gain_pct": attributed,
        "chain_total_gain_pct": chain_total,
        # Throughput the chain gained that no adoption claims: the anchor moved
        # between one adoption's measurement and the next one's. An identity,
        # not an estimate -- it is the sum of the chain breaks below.
        "unattributed_gain_pct": (
            round(chain_total - attributed, 6) if chain_total is not None and measured else None
        ),
        "validated_total_gain_pct": recorded_total,
        # The ledger against the one figure measured on the whole stack. This is
        # the number worth alerting on: the parts and the whole disagreeing
        # means one of them is wrong.
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

    Args:
        adoptions (Sequence[Mapping[str, Any]]): The adoption rows.

    Returns:
        dict[str, Any]: One entry per bucket in :data:`SOURCES`, each with its
        adoption count and summed contribution, plus a per-backend split for
        the kernel bucket. Buckets with no adoptions are present and zeroed, so
        a reader can tell "this subsystem earned nothing" from "this subsystem
        is not reported".
    """
    buckets: dict[str, Any] = {
        name: {"count": 0, "total_gain_pct": 0.0, "unmeasured": 0} for name in SOURCES
    }
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
    """The chain's total, read off the last adoption that measured one.

    Args:
        adoptions (Sequence[Mapping[str, Any]]): The adoption rows, in order.

    Returns:
        float | None: The last recorded ``cumulative_gain_pct``, or ``None``
        when no adoption measured one.
    """
    for row in reversed(list(adoptions)):
        value = _float_or_none(row.get("cumulative_gain_pct"))
        if value is not None:
            return value
    return None


def _chain_breaks(adoptions: Sequence[Mapping[str, Any]]) -> int:
    """Count adoptions whose anchor is not the previous adoption's reading.

    Args:
        adoptions (Sequence[Mapping[str, Any]]): The adoption rows, in order.

    Returns:
        int: How many times the anchor moved between two adoptions.
    """
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
    """The status the ledger reports.

    A stack that adopted nothing is ``skipped`` -- the run tried and kept
    nothing, which its attempt events explain. A stack whose total cannot be
    measured is ``degraded``: there are adoptions on it and no figure for what
    they bought.
    """
    if not adoptions:
        return STATUS_SKIPPED
    return STATUS_SUCCEEDED if chain_total is not None else STATUS_DEGRADED


def _header(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold the event-level fragments into one header.

    Args:
        rows (Sequence[Mapping[str, Any]]): The event-level rows.

    Returns:
        dict[str, Any]: The merged header.
    """
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
