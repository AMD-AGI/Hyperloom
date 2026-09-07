# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The SBD V6 ``warm_replay`` event: replaying a KB recipe, recorded live.

A warm replay takes a recipe another session validated, measures it here, and
either promotes it onto ``current_best`` or rolls it back. The arc is a chain
of gates -- throughput validity, image quality, accuracy, the keep threshold,
the presence of replayable params, and the checkout promotion -- and any one of
them can end it.

Projecting that arc was lossy in two specific ways this recorder removes.

The gate verdicts were not recorded, so ``accuracy.passed`` had to be *guessed
from the terminal status*: a replay rejected on accuracy and one rejected on
quality both surfaced as "not reproduced", and a replay whose eval ran but
returned no usable score was indistinguishable from one that scored and failed.
Each gate now writes its own row when it is evaluated, so the reason the arc
ended is read rather than inferred, and the gates that never ran are absent
rather than false.

The measurement's own anchor was never persisted. ``_promote_warm_replay``
computes the gain against the baseline captured at enqueue time, holds it in a
local, and drops it -- so the event had to back-solve the before-throughput
from the after-throughput and the gain. That is exact only while the session
baseline never moves; after a re-baseline it reconstructs an anchor the replay
was never judged against. The anchor is recorded here at the moment it is used.

The anchor is deliberately *not* the same number the adoption row chains from.
An adoption chains from the recorded session baseline so the ledger and
``cumulative_gain_validated`` stay one number, while this event states what the
replay was actually measured against. Recording both is what lets a reader see
when they diverged, which is precisely the re-baseline case.

The event cites the baseline action that produced its numbers rather than
restating them: a warm replay is measured through the baseline executor, so the
rounds, retries and report paths already live in that event's action, keyed by
the same task id.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

from .event_fields import (
    as_dict as _as_dict,
    failure_row as _failure_row,
    float_or_none as _float_or_none,
    now_iso_seconds as _now,
    text_or_none as _text_or_none,
)
from .event_ids import event_id
from .event_rows import rows_for_event, sort_rows, wire_rows
from .event_sink import RecordSink, make_sink
from .event_timeline import finish_event, open_event

log = logging.getLogger(__name__)

EVENT_TYPE = "warm_replay"
EVENT_KIND = "warm_replay"

#: The component segment of a warm-replay event id. The phase segment is the
#: phase the replay was dispatched in, which is why it is a parameter.
EVENT_COMPONENT = "warm_replay"

PRODUCER = "orchestrator"

#: The event-level section, one fragment per event, holding the request, the
#: measurement, the verdict and the timeline sequence the two writes share.
SECTION_EVENT = "warm_replay_event"

#: One row per gate evaluated, keyed by the gate's name.
SECTION_GATE = "warm_replay_gate"

ROW_GATE = "gate"

# The gates the arc can end on, in the order the settling applies them. Named
# constants rather than prose because assembly selects on them and a consumer
# reading "which gate ended this" must not match on wording.
#
# The historical reproduce bar is not among them. It reads a measured gain
# against a fraction of the claimed one, but it never rejects: a replay that
# cleared the keep threshold is promoted whether or not it reproduced the
# claim. A gate row for it would make ``blocked_by`` name the reason an arc
# that succeeded ended, so the bar and the verdict on it are stated in the
# verdict block instead.
GATE_TPUT_VALID = "tput_valid"
GATE_QUALITY = "quality"
GATE_ACCURACY = "accuracy"
GATE_KEEP_THRESHOLD = "keep_threshold"
GATE_PARAMS_PRESENT = "params_present"
GATE_PROMOTION = "promotion"

# Why a replay never ran. Recorded at the seam that refused it, because the
# reason is a decision the session made and not something a reader can recover
# from the state it left behind: the projection had to bucket these by
# substring-matching prose, which meant rewording a log line silently
# reclassified the skip.
SKIP_DISABLED_BY_FLAG = "disabled_by_flag"
SKIP_NO_WARM_START_RECIPE = "no_warm_start_recipe"
SKIP_RECIPE_NOT_REPLAYABLE = "recipe_not_replayable"
SKIP_RECIPE_READ_FAILED = "recipe_read_failed"
SKIP_CONFIDENCE_BELOW_THRESHOLD = "confidence_below_threshold"
SKIP_BEST_CONFIG_EMPTY = "best_config_empty"
SKIP_WORKLOAD_CONFIG_INCOMPATIBLE = "workload_config_incompatible"
SKIP_FRAMEWORK_ROOT_MISSING = "framework_root_missing"
SKIP_KERNEL_ROOT_MISSING = "kernel_root_missing"
SKIP_KERNEL_PREPARATION_FAILED = "kernel_preparation_failed"
SKIP_ENQUEUE_FAILED = "enqueue_failed"

#: The terminal statuses the event reports, mapped from the outcome status the
#: phase settles on. A replay that reproduced is ``succeeded``; one that was
#: measured and judged not to reproduce is ``rejected``, which is a completed
#: arc rather than a failure; one that never got a usable measurement is
#: ``failed``.
STATUS_BY_OUTCOME: dict[str, str] = {
    "reproduced": "succeeded",
    "quality_failed": "rejected",
    "accuracy_failed": "rejected",
    # The phase's word for "measured, and it came in under the keep
    # threshold". A judged rejection, not a failure: the replay produced a
    # real number and the number lost.
    "drift": "rejected",
    "reproduced_but_no_params": "degraded",
    "promotion_failed": "failed",
    # The replay lost and the attempt to undo it also lost, which leaves the
    # session's trees in a state nothing here vouches for.
    "rollback_failed": "failed",
    "failed": "failed",
    # Refused before a task existed. A skip is a completed decision, not an
    # absence, which is why it closes an event of its own rather than leaving
    # the timeline silent about a replay the session considered and declined.
    "skipped": "skipped",
    "kernel_preparation_failed": "failed",
    "enqueue_failed": "failed",
}

__all__ = [
    "EVENT_COMPONENT",
    "EVENT_KIND",
    "EVENT_TYPE",
    "GATE_ACCURACY",
    "GATE_KEEP_THRESHOLD",
    "GATE_PARAMS_PRESENT",
    "GATE_PROMOTION",
    "GATE_QUALITY",
    "GATE_TPUT_VALID",
    "PRODUCER",
    "SECTION_EVENT",
    "SECTION_GATE",
    "SKIP_BEST_CONFIG_EMPTY",
    "SKIP_CONFIDENCE_BELOW_THRESHOLD",
    "SKIP_DISABLED_BY_FLAG",
    "SKIP_ENQUEUE_FAILED",
    "SKIP_FRAMEWORK_ROOT_MISSING",
    "SKIP_KERNEL_PREPARATION_FAILED",
    "SKIP_KERNEL_ROOT_MISSING",
    "SKIP_NO_WARM_START_RECIPE",
    "SKIP_RECIPE_NOT_REPLAYABLE",
    "SKIP_RECIPE_READ_FAILED",
    "SKIP_WORKLOAD_CONFIG_INCOMPATIBLE",
    "STATUS_BY_OUTCOME",
    "WarmReplayEventRecorder",
    "assemble_warm_replay_ext",
    "make_warm_replay_recorder",
    "warm_replay_event_id",
]


def warm_replay_event_id(phase: str, macro_cycle: Any) -> str:
    """Build the event id of the warm replay one phase ran in one cycle.

    Args:
        phase (str): The coordinator phase the replay was dispatched in.
        macro_cycle (Any): The macro cycle it was dispatched in.

    Returns:
        str: The event id, ``{phase}:{macro_cycle}:warm_replay``.

    Raises:
        ValueError: If either segment is malformed.
    """
    return event_id(phase, macro_cycle, EVENT_COMPONENT)


def _derived_status(outcome: Mapping[str, Any]) -> str:
    """Map a settled outcome onto the status the event closes on.

    Args:
        outcome (Mapping[str, Any]): The settled ``warm_replay_outcome``.

    Returns:
        str: The event status. An outcome status the map does not know closes
            the event ``failed`` rather than inventing a reading for it: a
            status this module has not been taught is a replay whose arc it
            cannot vouch for.
    """
    return STATUS_BY_OUTCOME.get(str(outcome.get("status") or ""), "failed")


def _verdict(settled: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the terminal ruling shared by every way an event can close.

    Args:
        settled (Mapping[str, Any]): The settled ``warm_replay_outcome``.

    Returns:
        dict[str, Any]: The verdict block.
    """
    return {
        "outcome_status": str(settled.get("status") or ""),
        "reason": str(settled.get("reason") or ""),
        "error_class": str(settled.get("error_class") or ""),
        "keep_threshold_pct": _float_or_none(settled.get("keep_threshold_pct")),
        "below_historical_reproduce_pct": bool(settled.get("below_historical_reproduce_pct")),
        "historical_reproduce_bar_pct": _float_or_none(settled.get("historical_reproduce_bar_pct")),
        "settled_at": str(settled.get("settled_at") or ""),
    }


class WarmReplayEventRecorder:
    """Records one warm replay's facts into its own event.

    Holds a sink and no state beyond the timeline bookkeeping: every method
    states the whole of what it knows, so a replay recorded across a resume
    assembles from both halves. Nothing written is read back until
    :meth:`finish`, which assembles the event out of the fragments.
    """

    def __init__(
        self,
        sink: RecordSink,
        *,
        task_id: str = "",
        tier: str = "",
        config_source: str = "",
        config_donor_tier: str = "",
        donor: Mapping[str, Any] | None = None,
        expected_gain_pct: Any = None,
        confidence: Any = None,
        min_reproduce_pct: Any = None,
        session_baseline_tput: Any = None,
        kernel_count: Any = None,
        recipe_suppressed: Any = None,
    ):
        """Bind a recorder to one replay's event.

        A warm recipe has no single id. It is identified by the tier it was
        stamped at, the canonical record the config came from, and the donor
        that record belongs to -- three separate facts, because a replay can
        take its config from one place and its kernel section from another,
        and because a low-confidence config is suppressed while the kernel
        half of the same record still runs.

        Args:
            sink (RecordSink): Where the rows go, which decides the event they
                belong to.
            task_id (str): The dispatched task id. Also the key of the baseline
                action that measured this replay, which is how the event cites
                its own numbers instead of restating them.
            tier (str): The warm-recipe tier stamped at warm start.
            config_source (str): The canonical id of the record the config came
                from. Empty when the config was suppressed.
            config_donor_tier (str): Where that config came from -- ``self``
                when the session's own identity match owned it, the donor's
                tier when it was borrowed, or
                ``suppressed_low_confidence`` when it was withheld and only
                the kernel section was replayed.
            donor (Mapping[str, Any] | None): The donor record's identity, when
                the recipe was borrowed from another session.
            expected_gain_pct (Any): The gain the recipe claimed, which the
                historical reproduce bar is a fraction of.
            confidence (Any): The confidence the recipe was admitted on.
            min_reproduce_pct (Any): The fraction of the claimed gain the
                replay must reproduce to clear the historical bar.
            session_baseline_tput (Any): The session's recorded baseline.
                Recorded beside the enqueue anchor because the adoption chains
                from this one while the replay is judged against that one, and
                a reader comparing the two needs both on record.
            kernel_count (Any): How many kernel entries the replay carried.
            recipe_suppressed (Any): Whether the config half was withheld for
                low confidence, which is what makes an empty ``config_source``
                a decision rather than a missing read.
        """
        self._sink = sink
        self._t0 = time.monotonic()
        self._start_time = _now()
        self._sequence: int | None = None
        self._closed = False
        self._task_id = str(task_id or "")
        # Gates are stamped at seconds precision and several of them rule
        # inside one second, so the row's own timestamp cannot order the arc.
        # The ordinal is assigned once per gate, so a gate that re-rules on
        # better evidence settles in the place it was first evaluated rather
        # than jumping to the end of the arc.
        self._gate_ordinals: dict[str, int] = {}
        self._request = {
            "task_id": self._task_id,
            "baseline_action_ref": self._task_id,
            "tier": str(tier or ""),
            "config_source": str(config_source or ""),
            "config_donor_tier": str(config_donor_tier or ""),
            "donor": _as_dict(donor) or None,
            "expected_gain_pct": _float_or_none(expected_gain_pct),
            "confidence": _float_or_none(confidence),
            "min_reproduce_pct": _float_or_none(min_reproduce_pct),
            "session_baseline_tput": _float_or_none(session_baseline_tput),
            "kernel_count": None if kernel_count is None else int(kernel_count),
            "recipe_suppressed": None if recipe_suppressed is None else bool(recipe_suppressed),
        }
        self._sink.record(SECTION_EVENT, {"request": dict(self._request)})

    @property
    def event_id(self) -> str:
        """str: The event every row this recorder writes is tagged with."""
        return self._sink.event_id

    @property
    def task_id(self) -> str:
        """str: The task id this replay was dispatched under."""
        return self._task_id

    # ---- lifecycle -------------------------------------------------------

    def begin(self) -> None:
        """Put the event on the timeline.

        The request rides on the open shell so an in-flight replay is readable
        as the replay of a named recipe rather than as an anonymous event that
        has not finished yet -- which is the state a session killed mid-replay
        is read in.
        """
        self._sequence = open_event(
            event_type=EVENT_TYPE,
            event=self.event_id,
            event_section=SECTION_EVENT,
            producer=PRODUCER,
            kind=EVENT_KIND,
            start_time=self._start_time,
            ext={"request": dict(self._request)},
        )

    def record_measurement(
        self,
        *,
        before_tput: Any,
        after_tput: Any,
        gain_pct: Any,
        hot_tput: Any = None,
        cold_tput: Any = None,
        accuracy: Any = None,
        baseline_accuracy: Any = None,
        eval_ran: Any = None,
    ) -> None:
        """Record the numbers the replay was judged on, as it is judged.

        ``before_tput`` is the anchor captured at enqueue time -- the number
        the gain was actually computed against. It is recorded here because the
        phase holds it in a local and drops it, which is what forced the event
        to back-solve it from the gain.

        Args:
            before_tput (Any): The anchor the gain was measured against.
            after_tput (Any): The throughput the replay measured.
            gain_pct (Any): The measured gain over the anchor.
            hot_tput (Any): The measured hot pass.
            cold_tput (Any): The discarded cold warmup, kept for audit.
            accuracy (Any): The score the replay was gated on, when one could
                be read.
            baseline_accuracy (Any): The score the gate compared it against.
                Recorded beside it because a replay's score only means
                something next to the reference it was judged against.
            eval_ran (Any): Whether the accuracy eval ran at all, which is what
                separates "scored nothing" from "no score could be read".
        """
        self._sink.record(
            SECTION_EVENT,
            {
                "measurement": {
                    "before_tput": _float_or_none(before_tput),
                    "after_tput": _float_or_none(after_tput),
                    "gain_pct": _float_or_none(gain_pct),
                    "hot_tput": _float_or_none(hot_tput),
                    "cold_tput": _float_or_none(cold_tput),
                    "accuracy": _float_or_none(accuracy),
                    "baseline_accuracy": _float_or_none(baseline_accuracy),
                    "eval_ran": None if eval_ran is None else bool(eval_ran),
                },
            },
        )

    def record_applied(
        self,
        *,
        extra_server_args: str | None = None,
        extra_envs: Mapping[str, Any] | None = None,
        kernel: Mapping[str, Any] | None = None,
    ) -> None:
        """Record the config the replay actually ran with.

        The projection had to recover this from the stack entry the promotion
        pushed, which meant a replay that measured and lost left no record of
        *what* lost. Recorded here at the moment it is measured, so a rejected
        replay states its config too.

        The config and the kernel disposition are known at different points --
        the config when the replay is judged, the kernel half only once the
        ruling decides whether to keep or revert it. Only the arguments given
        are written, because rows deep-merge: passing a default for a fact this
        call does not know would overwrite what an earlier call did know.

        Args:
            extra_server_args (str | None): The combined server args the replay
                ran. ``None`` leaves any already-recorded value standing.
            extra_envs (Mapping[str, Any] | None): The combined envs it ran.
            kernel (Mapping[str, Any] | None): The kernel half's disposition --
                status, total, kept, reverted.
        """
        applied: dict[str, Any] = {}
        if extra_server_args is not None:
            applied["extra_server_args"] = str(extra_server_args)
        if extra_envs is not None:
            applied["extra_envs"] = {str(key): value for key, value in (_as_dict(extra_envs) or {}).items()}
        if kernel is not None:
            applied["kernel"] = _as_dict(kernel) or None
        if not applied:
            return
        self._sink.record(SECTION_EVENT, {"applied": applied})

    def record_rollback(self, *, ok: Any, errors: Any = None) -> None:
        """Record the attempt to undo a replay that did not survive its gates.

        Args:
            ok (Any): Whether every tree the replay touched was restored. A
                false reading is why a session stops: the trees are in a state
                nothing vouches for.
            errors (Any): What failed to unwind.
        """
        self._sink.record(
            SECTION_EVENT,
            {
                "rollback": {
                    "ok": None if ok is None else bool(ok),
                    "errors": [str(item) for item in (errors or []) if str(item or "")],
                },
            },
        )

    def record_gate(
        self,
        gate: str,
        *,
        passed: bool | None,
        reason: str = "",
        observed: Any = None,
        threshold: Any = None,
    ) -> None:
        """Record one gate's verdict at the moment it is evaluated.

        A gate that was never reached writes no row, which is how assembly
        tells "did not pass" apart from "did not apply" -- the distinction the
        status-guessing projection could not make.

        Args:
            gate (str): The gate's name (a ``GATE_*`` value).
            passed (bool | None): Whether it passed. ``None`` is a gate that
                ran but could not rule, which is a verdict of its own: an
                accuracy eval that ran and returned no usable score neither
                passed nor failed, and recording it as ``False`` would read as
                a failed score that never existed.
            reason (str): Why it ruled that way.
            observed (Any): The value the gate read.
            threshold (Any): The bar it was read against.
        """
        name = str(gate or "")
        ordinal = self._gate_ordinals.setdefault(name, len(self._gate_ordinals) + 1)
        self._sink.record(
            SECTION_GATE,
            {
                "gate": name,
                "ordinal": ordinal,
                "passed": None if passed is None else bool(passed),
                "reason": str(reason or ""),
                "observed": _float_or_none(observed),
                "threshold": _float_or_none(threshold),
                "ts": _now(),
            },
            row_type=ROW_GATE,
            natural_ids=name,
        )

    def record_promotion(
        self,
        *,
        promoted_checkout: str = "",
        replayed_patch_refs: Any = None,
        stack_entry: Mapping[str, Any] | None = None,
    ) -> None:
        """Record what promoting the replay actually changed.

        Args:
            promoted_checkout (str): The framework checkout the replay was
                promoted onto, when the promotion moved one.
            replayed_patch_refs (Any): The patch files the replay applied.
            stack_entry (Mapping[str, Any] | None): The entry pushed onto
                ``optimization_stack``.
        """
        refs = [str(ref) for ref in (replayed_patch_refs or []) if str(ref or "")]
        self._sink.record(
            SECTION_EVENT,
            {
                "promotion": {
                    "promoted_checkout": str(promoted_checkout or ""),
                    "replayed_patch_refs": refs,
                    "stack_entry": _as_dict(stack_entry) or None,
                },
            },
        )

    def finish(self, outcome: Mapping[str, Any] | None) -> None:
        """Close the event on the outcome the phase settled.

        Args:
            outcome (Mapping[str, Any] | None): The settled
                ``warm_replay_outcome``.
        """
        settled = _as_dict(outcome)
        self._close(status=_derived_status(settled), payload={"verdict": _verdict(settled)})

    def finish_skipped(
        self,
        *,
        code: str,
        outcome: Mapping[str, Any] | None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Close an event for a replay that was refused before it ran.

        Args:
            code (str): Why it was refused (a ``SKIP_*`` value). Recorded
                separately from the outcome's prose reason so a consumer
                selects on the code and reads the prose only for detail.
            outcome (Mapping[str, Any] | None): The settled
                ``warm_replay_outcome``, which carries the prose reason and
                any rollback the refusal had to perform.
            details (Mapping[str, Any] | None): What the refusal turned on --
                the roots it could not resolve, the confidence it read against
                its threshold, the workload shape it found incompatible.
        """
        settled = _as_dict(outcome)
        # An unrecognised status on this path is still a refusal: the replay
        # provably never got a measurement, so reading it as ``failed`` would
        # report a run that never happened.
        status = STATUS_BY_OUTCOME.get(str(settled.get("status") or ""), "skipped")
        self._close(
            status=status,
            payload={
                "skip": {
                    "code": str(code or ""),
                    "reason": str(settled.get("reason") or ""),
                    "details": _as_dict(details) or None,
                },
                "verdict": _verdict(settled),
            },
        )

    def finish_crashed(self, exc: BaseException) -> None:
        """Close an event whose replay raised instead of settling an outcome.

        Distinguishes "the replay blew up" from "the session was killed
        mid-replay", which would otherwise both read as a dangling
        ``status="running"`` event.

        Args:
            exc (BaseException): The exception propagating out of the replay.
        """
        if self._closed:
            return
        self._close(
            status="failed",
            payload={
                "failure": _failure_row(
                    phase=EVENT_TYPE,
                    error_class=type(exc).__name__,
                    message=f"warm replay raised: {exc!r}",
                )
            },
        )

    def _close(self, *, status: str, payload: Mapping[str, Any]) -> None:
        """Record the terminal facts and close the event.

        Args:
            status (str): The status the event ended on.
            payload (Mapping[str, Any]): The terminal fields to record.
        """
        if self._closed:
            return
        self._closed = True
        end_time = _now()
        self._sink.record(
            SECTION_EVENT,
            {
                **payload,
                "status": str(status),
                "end_time": end_time,
                "duration_sec": round(time.monotonic() - self._t0, 3),
            },
        )
        from .assembler import warm_replay_event_parts

        ext, derived = assemble_warm_replay_ext(warm_replay_event_parts(), event=self.event_id)
        finish_event(
            event_type=EVENT_TYPE,
            event=self.event_id,
            sequence=self._sequence,
            status=derived or status,
            ext=ext,
            kind=EVENT_KIND,
            start_time=self._start_time,
            end_time=end_time,
        )


def assemble_warm_replay_ext(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
) -> tuple[dict[str, Any], str]:
    """Assemble one warm-replay event's ``ext`` out of its recorded rows.

    Args:
        parts (Mapping[str, list[dict[str, Any]]]): The warm-replay sections as
            read back from the spool, section name to row list.
        event (str): The event id to assemble; rows of every other event in the
            same session are ignored.

    Returns:
        tuple[dict[str, Any], str]: The ``ext`` payload and the status the
            recorded verdict settled on. The status is empty when no write has
            closed the event, which leaves the caller's own reading standing.
    """
    event_rows = rows_for_event(parts.get(SECTION_EVENT) or [], event)
    header = event_rows[0] if event_rows else {}
    gates = wire_rows(
        sort_rows(rows_for_event(parts.get(SECTION_GATE) or [], event), keys=("ordinal", "ts", "gate")),
        drop=("event_id", "ordinal"),
    )
    status = str(header.get("status") or "")
    ext = {
        "request": _as_dict(header.get("request")),
        "measurement": _as_dict(header.get("measurement")),
        "gates": gates,
        # The gate that ended the arc, named rather than left to a consumer to
        # re-derive by scanning for the first non-pass. An arc that succeeded
        # was ended by nothing, so it names nothing: a replay is admitted on an
        # accuracy eval that ran and could not rule, and reporting that gate as
        # the blocker would say a promoted replay was held up by it.
        "blocked_by": None if status == "succeeded" else _text_or_none(_blocking_gate(gates)),
        "applied": _as_dict(header.get("applied")) or None,
        "verdict": _as_dict(header.get("verdict")),
        "promotion": _as_dict(header.get("promotion")) or None,
        "rollback": _as_dict(header.get("rollback")) or None,
        "skip": _as_dict(header.get("skip")) or None,
        "failure": _as_dict(header.get("failure")) or None,
        "duration_sec": header.get("duration_sec"),
    }
    return ext, status


def _blocking_gate(gates: list[dict[str, Any]]) -> str:
    """Name the first gate that did not pass, or ``""`` when all of them did.

    Args:
        gates (list[dict[str, Any]]): The gate rows in evaluation order.

    Returns:
        str: The gate's name. A gate that ruled ``None`` counts as blocking
            only if nothing after it failed outright, so a replay admitted on
            an unscored eval and then rejected on the keep threshold reports
            the threshold rather than the eval.
    """
    unresolved = ""
    for row in gates:
        passed = row.get("passed")
        if passed is False:
            return str(row.get("gate") or "")
        if passed is None and not unresolved:
            unresolved = str(row.get("gate") or "")
    return unresolved


def make_warm_replay_recorder(
    *,
    phase: str,
    macro_cycle: Any = 0,
    task_id: str = "",
    tier: str = "",
    config_source: str = "",
    config_donor_tier: str = "",
    donor: Mapping[str, Any] | None = None,
    expected_gain_pct: Any = None,
    confidence: Any = None,
    min_reproduce_pct: Any = None,
    session_baseline_tput: Any = None,
    kernel_count: Any = None,
    recipe_suppressed: Any = None,
    open_event_on_timeline: bool = True,
) -> WarmReplayEventRecorder | None:
    """Build a recorder, or ``None`` when one cannot be constructed.

    Replay behavior must not depend on the recorder existing, so construction
    failures degrade to "no event" rather than propagating. An unbound session
    declines too: writing the timeline into whatever the working directory
    happens to be is worse than not recording.

    Args:
        phase (str): The phase the replay was dispatched in.
        macro_cycle (Any): The macro cycle it was dispatched in.
        task_id (str): The dispatched task id.
        tier (str): The warm-recipe tier.
        config_source (str): The canonical id the config came from.
        config_donor_tier (str): Where that config came from.
        donor (Mapping[str, Any] | None): The donor record's identity.
        expected_gain_pct (Any): The gain the recipe claimed.
        confidence (Any): The confidence it was admitted on.
        min_reproduce_pct (Any): The fraction of the claim it must reproduce.
        session_baseline_tput (Any): The session's recorded baseline.
        kernel_count (Any): How many kernel entries the replay carried.
        recipe_suppressed (Any): Whether the config half was withheld.
        open_event_on_timeline (bool): Whether to open the event. ``False``
            rebinds to an event a previous tick already opened, which is how
            the promote seam records onto the arc the enqueue seam started
            without opening it a second time.

    Returns:
        WarmReplayEventRecorder | None: The recorder, already opened on the
            timeline, or ``None`` when it could not be built.
    """
    from ...session.session_binding import session_is_bound

    try:
        if not session_is_bound():
            log.warning(
                "warm replay timeline: no session bound; this replay's whole event will be "
                "missing from the breakdown. The coordinator binds at startup, so this means "
                "either that never happened or the replay ran outside the session's context"
            )
            return None
        recorder = WarmReplayEventRecorder(
            make_sink(warm_replay_event_id(phase, macro_cycle), producer=PRODUCER),
            task_id=task_id,
            tier=tier,
            config_source=config_source,
            config_donor_tier=config_donor_tier,
            donor=donor,
            expected_gain_pct=expected_gain_pct,
            confidence=confidence,
            min_reproduce_pct=min_reproduce_pct,
            session_baseline_tput=session_baseline_tput,
            kernel_count=kernel_count,
            recipe_suppressed=recipe_suppressed,
        )
    except Exception:  # noqa: BLE001 — observability cannot change replay behavior
        log.warning(
            "warm replay timeline: recorder construction failed; this replay's whole event "
            "will be missing from the breakdown",
            exc_info=True,
        )
        return None
    if open_event_on_timeline:
        recorder.begin()
    return recorder
