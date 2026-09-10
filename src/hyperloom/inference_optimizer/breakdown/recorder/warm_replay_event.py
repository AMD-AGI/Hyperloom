# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The SBD V6 ``warm_replay`` event: replaying a KB recipe, recorded live.

A warm replay takes a recipe another session validated, measures it here, and
either promotes it onto ``current_best`` or rolls it back. The arc is a chain
of gates -- throughput validity, image quality, accuracy, the keep threshold,
replayable params, and the checkout promotion -- any one of which can end it.
Each gate writes its own row as it is evaluated, so a gate that never ran is
absent rather than false.

The measurement's anchor is the baseline captured at enqueue time, recorded at
the moment it is used because the phase holds it in a local and drops it. It is
deliberately *not* the number the adoption row chains from, which is the
recorded session baseline; recording both is what lets a reader see when they
diverged, the re-baseline case. The event cites the baseline action that
produced its numbers rather than restating them, keyed by the same task id.
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

#: The component segment of a warm-replay event id; the phase segment is the
#: phase the replay was dispatched in, hence a parameter.
EVENT_COMPONENT = "warm_replay"

PRODUCER = "orchestrator"

#: One fragment per event: the request, measurement, verdict and sequence.
SECTION_EVENT = "warm_replay_event"

#: One row per gate evaluated, keyed by the gate's name.
SECTION_GATE = "warm_replay_gate"

ROW_GATE = "gate"

# The gates the arc can end on, in the order the settling applies them.
# Constants because assembly selects on them, so a consumer reading "which gate
# ended this" must not match on wording. The historical reproduce bar is not
# among them: it never rejects, so a gate row for it would make ``blocked_by``
# name the reason a successful arc ended. It lives in the verdict block.
GATE_TPUT_VALID = "tput_valid"
GATE_QUALITY = "quality"
GATE_ACCURACY = "accuracy"
GATE_KEEP_THRESHOLD = "keep_threshold"
GATE_PARAMS_PRESENT = "params_present"
GATE_PROMOTION = "promotion"

# Why a replay never ran, recorded at the seam that refused it: the reason is a
# decision the session made, not something recoverable from the state it left
# behind, and bucketing by substring lets a reworded log line reclassify it.
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

#: The terminal statuses the event reports, keyed by the outcome status the
#: phase settles on. ``rejected`` is a completed arc rather than a failure: the
#: replay was measured and judged not to reproduce.
STATUS_BY_OUTCOME: dict[str, str] = {
    "reproduced": "succeeded",
    "quality_failed": "rejected",
    "accuracy_failed": "rejected",
    # "Measured, and under the keep threshold": a judged rejection, not a
    # failure -- the replay produced a real number and the number lost.
    "drift": "rejected",
    "reproduced_but_no_params": "degraded",
    "promotion_failed": "failed",
    # The replay lost and the undo lost too, leaving the session's trees in a
    # state nothing here vouches for.
    "rollback_failed": "failed",
    "failed": "failed",
    # Refused before a task existed. A completed decision, not an absence, so
    # it closes an event of its own rather than leaving the timeline silent.
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
    """Build ``{phase}:{macro_cycle}:warm_replay``; raises :exc:`ValueError`
    if either segment is malformed."""
    return event_id(phase, macro_cycle, EVENT_COMPONENT)


def _derived_status(outcome: Mapping[str, Any]) -> str:
    """Map a settled ``warm_replay_outcome`` onto the event's status; one the
    map does not know closes ``failed``, being an arc it cannot vouch for."""
    return STATUS_BY_OUTCOME.get(str(outcome.get("status") or ""), "failed")


def _verdict(settled: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the terminal ruling shared by every way an event can close."""
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

    Holds a sink and no state beyond timeline bookkeeping: every method states
    the whole of what it knows, so a replay recorded across a resume assembles
    from both halves.
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

        A warm recipe has no single id: it takes ``tier``, the record
        ``config_source`` came from, and that record's ``donor``. Three facts,
        because a replay can take its config from one place and its kernel
        section from another, and a low-confidence config is suppressed --
        ``recipe_suppressed`` -- while the kernel half still runs, which is
        what makes an empty ``config_source`` a decision. ``config_donor_tier``
        is ``self``, the donor's tier, or ``suppressed_low_confidence``.
        """
        self._sink = sink
        self._t0 = time.monotonic()
        self._start_time = _now()
        self._sequence: int | None = None
        self._closed = False
        self._task_id = str(task_id or "")
        # Gates are stamped at seconds precision and several rule inside one
        # second, so timestamps cannot order the arc. Assigned once per gate,
        # so a gate that re-rules keeps its original place.
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

        The request rides on the open shell, so a session killed mid-replay
        reads as the replay of a named recipe, not an anonymous unfinished one.
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

        ``before_tput`` is the enqueue-time anchor the gain was computed
        against, recorded here because the phase drops it. ``cold_tput`` is the
        discarded warmup, kept for audit; ``eval_ran`` separates "scored
        nothing" from "no score could be read".
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

        Recorded as the config is measured, so a replay that lost still states
        *what* lost. The config is known when the replay is judged and the
        kernel disposition only once the ruling decides whether to revert it,
        so only the arguments given are written: rows deep-merge, and a default
        would overwrite what an earlier call knew.
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

        A false ``ok`` is why a session stops: some tree was not restored.
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

        A gate never reached writes no row, which is how assembly tells "did
        not pass" apart from "did not apply". A ``None`` ``passed`` is a gate
        that ran and could not rule: an eval returning no usable score neither
        passed nor failed, and ``False`` would read as a score that never was.
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
        """Record what promoting the replay actually changed: the checkout it
        moved, the patches it applied, and the ``optimization_stack`` entry."""
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
        """Close the event on the ``warm_replay_outcome`` the phase settled."""
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

        ``code`` is a ``SKIP_*`` value, apart from the outcome's prose reason
        so a consumer selects on the code. ``details`` is what the refusal
        turned on -- an unresolvable root, a confidence, a workload shape.
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
        mid-replay", which both read as a dangling ``running`` event.
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
        """Record the terminal facts and close the event."""
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

        ext, derived = assemble_warm_replay_ext(warm_replay_event_parts(self.event_id), event=self.event_id)
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

    Rows of every other event in the session are ignored. The returned status
    is empty when no write has closed the event, leaving the caller's reading.
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
        # The gate that ended the arc, named rather than re-derived by a
        # consumer. A successful arc names nothing: a replay is admitted on an
        # eval that could not rule, which is no blocker.
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

    A gate that ruled ``None`` blocks only if nothing after it failed, so a
    replay admitted on an unscored eval and then rejected on the keep threshold
    reports the threshold.
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
    failures degrade to "no event", and an unbound session declines too. A
    false ``open_event_on_timeline`` rebinds to an event a previous tick
    opened, which is how the promote seam records onto the enqueue seam's arc.
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
