# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The SBD V6 ``baseline`` event: the reference measurement, recorded live.

Baseline is the measurement every later gain in the session is read against,
and until this recorder existed it was the one stage the timeline could only
project. The projection's limit was not its logic but its evidence: V5 stamps
an action row and an attempt summary when the measurement *completes*, and
nothing anywhere recorded when it began. So the projected event's window
collapsed onto its own end -- a baseline that ran for four minutes was
published as an instant -- and it sorted onto the timeline as though it had
happened at the moment it finished, behind actions that started after it.

Recording it removes the guess rather than improving it. The event opens when
the action starts, which is a fact only the action holds, and closes when it
ends.

The event holds an array of actions for the same reason the roofline event
does: the id is ``{phase}:{macro_cycle}:baseline`` and one phase and cycle can
measure more than once -- a failure streak re-dispatches, enablement
re-validates after fixing an eval, a warm replay is measured through this same
executor. Each of those is an action keyed by its task id, and the event that
holds them reports the worst of their statuses.

Inside one action the structure is two levels deep because the executor retries
at two levels, and flattening them would lose which retry a round belonged to.
A *run* is one pass through the executor's core, of which there can be three:
the first, the salvage retry taken when the accuracy eval is what aborted the
benchmark, and the retry taken when a MoE runner backend killed the server. A
*round* is one Magpie subprocess inside a run, of which there can also be
three: the discarded cold-start warmup, the measured hot pass, and the deferred
accuracy pass. A run that was refused before it booted anything still records
its own row, which is the case a round-only model would drop entirely.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

from .event_fields import (
    as_dict as _as_dict,
    as_list as _as_list,
    failure_row as _failure_row,
    float_or_none as _float_or_none,
    int_or_none as _int_or_none,
    now_iso_seconds as _now_iso,
    worst_status as _worst_status,
)
from .event_ids import event_id
from .event_rows import group_rows, rows_for_event, sort_rows, wire_rows
from .event_sink import RecordSink
from .event_timeline import finish_event, open_event

log = logging.getLogger(__name__)

EVENT_TYPE = "baseline"
EVENT_KIND = "baseline"

#: The component segment of a baseline event id. The phase segment is the phase
#: the measurement was dispatched in, which is why it is a parameter.
EVENT_COMPONENT = "baseline"

PRODUCER = "orchestrator"

#: The event-level section, one fragment per event, holding the timeline
#: sequence the two writes share. Separate from :data:`SECTION_ACTION` because
#: an event has one sequence and may have several actions.
SECTION_EVENT = "baseline_event"

SECTION_ACTION = "baseline_action"
SECTION_RUN = "baseline_run"
SECTION_ROUND = "baseline_round"

ROW_ACTION = "action"
ROW_RUN = "run"
ROW_ROUND = "round"

# Every run row names why it ran, so a baseline that measured three times can
# be read without re-deriving the retry reason from log text.
RUN_INITIAL = "initial"
RUN_AFTER_EVAL_FAILURE = "retry_after_eval_failure"
RUN_AFTER_MOE_RUNNER_FAILURE = "retry_after_moe_runner_failure"

# The round labels the executor reports, mirrored here so a consumer can select
# the measured pass without matching on prose.
ROUND_SINGLE = "single"
ROUND_WARMUP = "warmup"
ROUND_MEASURE = "measure"
ROUND_ACCURACY = "accuracy"

# The failure class the run's own clock raises. A run carrying it that never
# booted a round was refused rather than attempted, which assembly reports as
# ``skipped``.
_BUDGET_ERROR_CLASS = "session_time_exhausted"

# Warnings are prose an operator reads, and a round can accumulate one per
# harvested artifact. The head characterizes the round and the count carries
# the rest.
_MAX_ROUND_WARNINGS = 12

__all__ = [
    "EVENT_COMPONENT",
    "EVENT_KIND",
    "EVENT_TYPE",
    "PRODUCER",
    "ROUND_ACCURACY",
    "ROUND_MEASURE",
    "ROUND_SINGLE",
    "ROUND_WARMUP",
    "RUN_AFTER_EVAL_FAILURE",
    "RUN_AFTER_MOE_RUNNER_FAILURE",
    "RUN_INITIAL",
    "SECTION_ACTION",
    "SECTION_EVENT",
    "SECTION_ROUND",
    "SECTION_RUN",
    "BaselineEventRecorder",
    "assemble_baseline_action",
    "assemble_baseline_actions",
    "assemble_baseline_ext",
    "baseline_event_id",
    "make_baseline_recorder",
]


def baseline_event_id(phase: str, macro_cycle: Any) -> str:
    """Build the event id of the baselines one phase measured in one cycle.

    Args:
        phase (str): The coordinator phase the measurement was dispatched in.
        macro_cycle (Any): The macro cycle it was dispatched in.

    Returns:
        str: The event id, ``{phase}:{macro_cycle}:baseline``.

    Raises:
        ValueError: If either segment is malformed.
    """
    return event_id(phase, macro_cycle, EVENT_COMPONENT)


def _warnings(result: Mapping[str, Any]) -> dict[str, Any]:
    """Project a result's non-fatal warnings into a bounded block.

    Args:
        result (Mapping[str, Any]): The executor result to read.

    Returns:
        dict[str, Any]: The full count and a bounded head of the messages.
    """
    rows = [str(row) for row in _as_list(result.get("nonfatal_warnings")) if str(row or "")]
    return {"count": len(rows), "messages": rows[:_MAX_ROUND_WARNINGS]}


def _measurement(result: Mapping[str, Any]) -> dict[str, Any]:
    """Project the numbers a benchmark round produced.

    Recorded on the round as well as on the action because the two answer
    different questions: the action carries the figure the session went on to
    use, and the rounds carry every figure that was measured -- including the
    cold warmup's, which is deliberately discarded and is the only thing a
    reader can weigh the adopted number against.

    Args:
        result (Mapping[str, Any]): The executor result to read.

    Returns:
        dict[str, Any]: The measurement block, with absent numbers as ``None``.
    """
    return {
        # Named as the V5 section names it, which is what the projected event
        # published and what a consumer already selects on. The executor's own
        # key for it is ``output_throughput``.
        "throughput_tok_s_per_gpu": _float_or_none(result.get("output_throughput")),
        "ttft_mean_ms": _float_or_none(result.get("ttft_mean_ms")),
        "e2el_mean_ms": _float_or_none(result.get("e2el_mean_ms")),
        "tpot_mean_ms": _float_or_none(result.get("tpot_mean_ms")),
        "accuracy": _float_or_none(result.get("accuracy")),
        "accuracy_task": str(result.get("accuracy_task") or ""),
        "accuracy_metric": str(result.get("accuracy_metric") or ""),
        "accuracy_source": str(result.get("accuracy_source") or ""),
        "benchmark_report_path": str(result.get("report_path") or ""),
        "workspace": str(result.get("workspace") or ""),
    }


def _timing(result: Mapping[str, Any]) -> dict[str, Any]:
    """Project the runtimes a benchmark round reported for itself.

    Args:
        result (Mapping[str, Any]): The executor result to read.

    Returns:
        dict[str, Any]: The subprocess wall-clock and its post-ready share.
    """
    return {
        "subprocess_runtime_sec": _float_or_none(result.get("subprocess_runtime_sec")),
        "post_ready_runtime_sec": _float_or_none(result.get("post_ready_runtime_sec")),
    }


def _failure(result: Mapping[str, Any], *, phase: str) -> dict[str, Any] | None:
    """Project a failed result's failure row, or ``None`` when it succeeded.

    Args:
        result (Mapping[str, Any]): The executor result to read.
        phase (str): The sub-step the failure is attributed to.

    Returns:
        dict[str, Any] | None: The failure row, or ``None``.
    """
    if str(result.get("status") or "") == "succeeded":
        return None
    return {
        **_failure_row(
            phase=phase,
            error_class=str(result.get("error_class") or ""),
            message=result.get("error") or "",
        ),
        "returncode": _int_or_none(result.get("returncode")),
        "stderr_log_path": str(result.get("stderr_log_path") or ""),
    }


class BaselineEventRecorder:
    """Records one baseline action's facts into the event it belongs to.

    Holds a sink, a task id, and the counters the round rows are keyed by.
    Every method is total over the executor's exits: a run or a round the
    executor abandoned leaves the row it already wrote, and closing the action
    is what decides the event's status.
    """

    def __init__(
        self,
        sink: RecordSink,
        *,
        task_id: str = "",
        task_kind: str = "",
        reason: str = "",
        framework: str = "",
        establishes_quality_ref: bool = False,
        params: dict[str, Any] | None = None,
    ):
        """Bind a recorder to one action inside one event.

        Args:
            sink (RecordSink): Where the rows go, which decides the event they
                belong to.
            task_id (str): The dispatched task id, which separates this action
                from the others in the same event.
            task_kind (str): The dispatched task's kind. This executor also
                measures ``replay_warm_recipe``, so the action states which
                kind it served rather than leaving a consumer to infer it.
            reason (str): Why the measurement was dispatched, when the
                dispatcher named a reason.
            framework (str): Resolved serving framework.
            establishes_quality_ref (bool): Whether this run defines the
                session's accuracy reference. A measurement that does not is
                held to a different gate, and which gate applied is not
                recoverable from the numbers alone.
            params (dict[str, Any] | None): The task params, read for the
                workspace and the config the round rendered from.
        """
        self._sink = sink
        self._t0 = time.monotonic()
        self._start_time = _now_iso()
        self._sequence: int | None = None
        self._closed = False
        self._run_index = 0
        self._rounds = 0
        params = _as_dict(params)
        self._task_id = str(task_id or "")
        self._action_id = self._task_id or "unnamed"
        self._sink.record(
            SECTION_ACTION,
            {
                "task_id": self._task_id,
                "start_time": self._start_time,
                "request": {
                    "task_id": self._task_id,
                    "task_kind": str(task_kind or ""),
                    "reason": str(reason or ""),
                    "framework": str(framework or ""),
                    "establishes_quality_ref": bool(establishes_quality_ref),
                    "config_path": str(params.get("config_path") or ""),
                    "output_dir": str(params.get("output_dir") or ""),
                    "requested_timeout_sec": _int_or_none(params.get("timeout_sec")),
                },
            },
            row_type=ROW_ACTION,
            natural_ids=self._action_id,
        )

    @property
    def event_id(self) -> str:
        """str: The event this action's rows belong to."""
        return self._sink.event_id

    @property
    def task_id(self) -> str:
        """str: The task id separating this action from others in its event."""
        return self._task_id

    def _record_action(self, payload: Mapping[str, Any]) -> None:
        """Update this action's own row.

        Args:
            payload (Mapping[str, Any]): The fields this call knows.
        """
        self._sink.record(SECTION_ACTION, payload, row_type=ROW_ACTION, natural_ids=self._action_id)

    # ---- lifecycle -------------------------------------------------------

    def begin(self) -> None:
        """Open the event this action belongs to.

        Opening is idempotent, so several baselines dispatched in one phase and
        cycle share one timeline entry rather than each adding their own.
        """
        self._sequence = open_event(
            event_type=EVENT_TYPE,
            event=self.event_id,
            event_section=SECTION_EVENT,
            producer=PRODUCER,
            kind=EVENT_KIND,
            start_time=self._start_time,
        )

    def begin_run(self, *, attempt_reason: str) -> int:
        """Record that a pass through the executor's core has started.

        Args:
            attempt_reason (str): Why this pass ran (a ``RUN_*`` value).

        Returns:
            int: The 1-based run index, which the rounds of this run are
                recorded under.
        """
        self._run_index += 1
        self._sink.record(
            SECTION_RUN,
            {
                "task_id": self._task_id,
                "run_index": self._run_index,
                "attempt_reason": str(attempt_reason),
                "status": "running",
                "start_time": _now_iso(),
            },
            row_type=ROW_RUN,
            natural_ids=(self._action_id, str(self._run_index)),
        )
        self._record_action({"in_flight_run_index": self._run_index})
        return self._run_index

    def end_run(self, *, run_index: int, result: Mapping[str, Any] | None) -> None:
        """Record how a pass through the executor's core ended.

        Args:
            run_index (int): The index :meth:`begin_run` returned.
            result (Mapping[str, Any] | None): The result the pass returned.
        """
        payload = _as_dict(result)
        self._sink.record(
            SECTION_RUN,
            {
                "task_id": self._task_id,
                "run_index": int(run_index),
                "status": str(payload.get("status") or "failed"),
                "end_time": _now_iso(),
                "error_class": str(payload.get("error_class") or ""),
                "warnings": _warnings(payload),
            },
            row_type=ROW_RUN,
            natural_ids=(self._action_id, str(int(run_index))),
        )

    def record_round(
        self,
        *,
        run_index: int,
        label: str,
        started_at: str,
        duration_sec: float | None,
        timeout_sec: Any = None,
        result: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one Magpie benchmark round.

        Args:
            run_index (int): The run this round belonged to.
            label (str): The round's name (a ``ROUND_*`` value).
            started_at (str): ISO start of the round.
            duration_sec (float | None): Wall-clock seconds the round took,
                boot and teardown included. Distinct from the subprocess
                runtime the round reports for itself.
            timeout_sec (Any): The cap the round ran under, so a round that
                timed out can be read against what it was allowed.
            result (Mapping[str, Any] | None): The round's result dict.
        """
        payload = _as_dict(result)
        self._rounds += 1
        self._sink.record(
            SECTION_ROUND,
            {
                "task_id": self._task_id,
                "run_index": int(run_index),
                # The order the rounds ran in, which their start stamps cannot
                # be relied on to give: those are ISO seconds, and a round that
                # failed fast can start and finish inside the same second as
                # the next one. Without it the tie falls through to the label
                # and the rounds come back alphabetically -- ``measure`` ahead
                # of the ``warmup`` that booted the server it re-attached to.
                "ordinal": self._rounds,
                "label": str(label),
                "status": str(payload.get("status") or "failed"),
                "start_time": str(started_at or ""),
                "end_time": _now_iso(),
                "duration_sec": duration_sec,
                "timeout_sec": _int_or_none(timeout_sec),
                "run_eval_disabled": bool(payload.get("run_eval_disabled")),
                "measurement": _measurement(payload),
                "timing": _timing(payload),
                "warnings": _warnings(payload),
                "failure": _failure(payload, phase=f"round_{label}"),
            },
            row_type=ROW_ROUND,
            natural_ids=(self._action_id, str(int(run_index)), str(label)),
        )

    def finish(self, result: Mapping[str, Any] | None) -> None:
        """Close the action on the result the executor returned.

        Args:
            result (Mapping[str, Any] | None): The action's final result dict.
        """
        payload = _as_dict(result)
        dropped = _as_dict(payload.get("measure_round_dropped"))
        self._close(
            status=self._derived_status(payload),
            action={
                "measurement": _measurement(payload),
                "timing": _timing(payload),
                "warnings": _warnings(payload),
                "run_eval_disabled": bool(payload.get("run_eval_disabled")),
                "materialized_config": str(payload.get("materialized_config") or ""),
                "warmup_round_tput": _float_or_none(payload.get("warmup_round_tput")),
                "convergence": _as_dict(payload.get("baseline_convergence")) or None,
                "accuracy_stage": _as_dict(payload.get("accuracy_stage")) or None,
                "cold_anchor": dropped or None,
                "failure": _failure(payload, phase=EVENT_TYPE),
            },
        )

    def _derived_status(self, result: Mapping[str, Any]) -> str:
        """Decide the status the action closes on.

        Args:
            result (Mapping[str, Any]): The action's final result dict.

        Returns:
            str: ``succeeded`` for a measured baseline, ``degraded`` for one
                that stands on its cold warmup because the budget would not
                hold the hot pass -- the number is usable and knowingly
                depressed, and a reader weighing later gains against it needs
                to be told which -- ``skipped`` for a measurement the run's
                clock refused before it booted anything, and ``failed``
                otherwise.
        """
        if str(result.get("status") or "") == "succeeded":
            return "degraded" if _as_dict(result.get("measure_round_dropped")) else "succeeded"
        if self._rounds == 0 and str(result.get("error_class") or "") == _BUDGET_ERROR_CLASS:
            return "skipped"
        return "failed"

    def finish_crashed(self, exc: BaseException) -> None:
        """Close an action whose executor raised instead of returning a result.

        Distinguishes "the executor blew up" from "the session was killed
        mid-baseline", which would otherwise both read as a dangling
        ``status="running"`` event.

        Args:
            exc (BaseException): The exception propagating out of the action.
        """
        if self._closed:
            return
        self._close(
            status="failed",
            action={
                "failure": _failure_row(
                    phase=EVENT_TYPE,
                    error_class=type(exc).__name__,
                    message=f"baseline action raised: {exc!r}",
                )
            },
        )

    def _close(self, *, status: str, action: Mapping[str, Any]) -> None:
        """Record the action's terminal facts and close the event.

        Args:
            status (str): The status the action ended on.
            action (Mapping[str, Any]): The terminal fields to record.
        """
        if self._closed:
            return
        self._closed = True
        end_time = _now_iso()
        self._record_action(
            {
                **action,
                "status": str(status),
                "in_flight_run_index": None,
                "end_time": end_time,
                "duration_sec": round(time.monotonic() - self._t0, 3),
            }
        )
        from .assembler import baseline_event_parts

        ext, derived = assemble_baseline_ext(baseline_event_parts(), event=self.event_id)
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


def assemble_baseline_actions(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
) -> list[dict[str, Any]]:
    """Assemble every baseline action belonging to one event.

    Args:
        parts (Mapping[str, list[dict[str, Any]]]): The baseline sections as
            read back from the spool.
        event (str): The event id whose rows to select.

    Returns:
        list[dict[str, Any]]: The actions, ordered by when they started.
    """
    action_rows = sort_rows(
        rows_for_event(parts.get(SECTION_ACTION) or [], event),
        keys=("start_time", "task_id"),
    )
    runs = group_rows(
        sort_rows(rows_for_event(parts.get(SECTION_RUN) or [], event), keys=("run_index",)),
        "task_id",
    )
    rounds = group_rows(
        sort_rows(
            rows_for_event(parts.get(SECTION_ROUND) or [], event),
            keys=("run_index", "ordinal", "start_time"),
        ),
        "task_id",
    )

    actions: list[dict[str, Any]] = []
    for row in action_rows:
        task = str(row.get("task_id") or "")
        by_run = group_rows(rounds.get(task, []), "run_index")
        run_rows = []
        for run in wire_rows(runs.get(task, []), drop=("event_id", "task_id")):
            index = _int_or_none(run.get("run_index"))
            run["rounds"] = wire_rows(
                by_run.get("" if index is None else str(index), []),
                drop=("event_id", "task_id", "run_index", "ordinal"),
            )
            run_rows.append(run)
        actions.append(
            {
                "task_id": task,
                "status": str(row.get("status") or "running"),
                "start_time": str(row.get("start_time") or ""),
                "end_time": str(row.get("end_time") or ""),
                "duration_sec": row.get("duration_sec"),
                "in_flight_run_index": row.get("in_flight_run_index"),
                "request": _as_dict(row.get("request")),
                "measurement": _as_dict(row.get("measurement")),
                "timing": _as_dict(row.get("timing")),
                "warnings": _as_dict(row.get("warnings")),
                "run_eval_disabled": bool(row.get("run_eval_disabled")),
                "materialized_config": str(row.get("materialized_config") or ""),
                "warmup_round_tput": row.get("warmup_round_tput"),
                "convergence": row.get("convergence"),
                "accuracy_stage": row.get("accuracy_stage"),
                "cold_anchor": row.get("cold_anchor"),
                "runs": run_rows,
                "failure": _as_dict(row.get("failure")) or None,
            }
        )
    return actions


def assemble_baseline_action(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
    task_id: str,
) -> dict[str, Any] | None:
    """Assemble one baseline action out of its recorded rows.

    Args:
        parts (Mapping[str, list[dict[str, Any]]]): The baseline sections as
            read back from the spool.
        event (str): The event id whose rows to select.
        task_id (str): The action to assemble.

    Returns:
        dict[str, Any] | None: The assembled action, or ``None`` when the event
            holds no action with that task id.
    """
    wanted = str(task_id or "")
    for action in assemble_baseline_actions(parts, event=event):
        if str(action.get("task_id") or "") == wanted:
            return action
    return None


def assemble_baseline_ext(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
) -> tuple[dict[str, Any], str]:
    """Assemble one baseline event's ``ext`` out of its recorded rows.

    Args:
        parts (Mapping[str, list[dict[str, Any]]]): The baseline sections as
            read back from the spool.
        event (str): The event id to assemble.

    Returns:
        tuple[dict[str, Any], str]: The ``ext`` payload, holding one entry per
            action the event owns, and the status derived from them.
    """
    actions = assemble_baseline_actions(parts, event=event)
    return {"actions": actions}, _worst_status(action.get("status") for action in actions)


def make_baseline_recorder(
    sink: RecordSink | None,
    *,
    task_id: str = "",
    task_kind: str = "",
    reason: str = "",
    framework: str = "",
    establishes_quality_ref: bool = False,
    params: dict[str, Any] | None = None,
) -> BaselineEventRecorder | None:
    """Build a recorder, or ``None`` when one cannot be constructed.

    Baseline behavior must not depend on the recorder existing, so construction
    failures degrade to "no event" rather than propagating -- as does an absent
    sink, which is what a caller with no session bound has.

    Args:
        sink (RecordSink | None): Where the rows go, or ``None`` to decline.
        task_id (str): Dispatched task id.
        task_kind (str): The dispatched task's kind.
        reason (str): Why the measurement was dispatched.
        framework (str): Resolved serving framework.
        establishes_quality_ref (bool): Whether this run defines the session's
            accuracy reference.
        params (dict[str, Any] | None): The task params.

    Returns:
        BaselineEventRecorder | None: The recorder, or ``None`` when it could
            not be built.
    """
    if sink is None:
        return None
    try:
        recorder = BaselineEventRecorder(
            sink,
            task_id=task_id,
            task_kind=task_kind,
            reason=reason,
            framework=framework,
            establishes_quality_ref=establishes_quality_ref,
            params=params,
        )
    except Exception:  # noqa: BLE001 — observability cannot change baseline behavior
        log.warning(
            "baseline timeline: recorder construction failed; this measurement's facts will be missing from the event",
            exc_info=True,
        )
        return None
    recorder.begin()
    return recorder
