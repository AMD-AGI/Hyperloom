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
from ... import framework_registry
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

# Where the recorded ``framework_args`` came from. The executor reports the
# args it is about to launch under, at the moment it resolves them, so the
# ordinary answer is ``launch_extra_server_args`` -- including when the string
# is empty, which is a baseline running on the framework's own defaults and is
# a fact rather than a gap. The observed label is the fallback for a run whose
# launch report never landed, and ``unavailable`` means neither did.
ARGS_FROM_LAUNCH = "launch_extra_server_args"
ARGS_FROM_OBSERVED = "observed_server_launch_flags"
ARGS_UNAVAILABLE = "unavailable"

# The failure class the run's own clock raises. A run carrying it that never
# booted a round was refused rather than attempted, which assembly reports as
# ``skipped``.
_BUDGET_ERROR_CLASS = "session_time_exhausted"

# Warnings are prose an operator reads, and a round can accumulate one per
# harvested artifact. The head characterizes the round and the count carries
# the rest.
_MAX_ROUND_WARNINGS = 12

__all__ = [
    "ARGS_FROM_LAUNCH",
    "ARGS_FROM_OBSERVED",
    "ARGS_UNAVAILABLE",
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

    Returns:
        str: The event id, ``{phase}:{macro_cycle}:baseline``.

    Raises:
        ValueError: If either segment is malformed.
    """
    return event_id(phase, macro_cycle, EVENT_COMPONENT)


def record_action_decision(
    *,
    phase: str,
    macro_cycle: Any,
    task_id: str,
    decision: str,
) -> None:
    """Record the write-back's verdict on a measurement that already settled.

    The verdict is not the action's own to state. The executor returns a
    measurement and the write-back decides what the session does with it,
    which happens after this event has closed -- so the action row is written
    without it and the verdict merges on afterwards. A closed event still
    accepts row fragments: nothing is assembled until the export reads the
    whole spool.

    The row is only touched when it is already there. The event id is rebuilt
    from live session state, which is the phase and cycle the write-back is
    running in rather than the ones the measurement was dispatched in; the two
    agree on the settle-in-the-same-tick path and can diverge on a resume, and
    an upsert onto an event with no such action would mint a row carrying a
    verdict and no measurement.
    """
    if not str(task_id or "") or not str(decision or ""):
        return
    try:
        from .event_sink import make_sink

        sink = make_sink(baseline_event_id(phase, macro_cycle), producer=PRODUCER)
        if not sink.has_row(SECTION_ACTION, row_type=ROW_ACTION, natural_ids=str(task_id)):
            log.debug(
                "baseline timeline: event %s holds no action %s; the promotion decision is not recorded",
                sink.event_id,
                task_id,
            )
            return
        sink.record(
            SECTION_ACTION,
            {"task_id": str(task_id), "decision": str(decision)},
            row_type=ROW_ACTION,
            natural_ids=str(task_id),
        )
        _republish_closed_event(sink.event_id)
    except Exception:  # noqa: BLE001 — observability cannot change baseline behavior
        log.warning(
            "baseline timeline: could not record the promotion decision for action %s",
            task_id,
            exc_info=True,
        )


def _republish_closed_event(event: str) -> None:
    """Re-assemble a closed event so a fragment written after it is published.

    The export reads the durable timeline rather than re-assembling it, so a
    closed event's published ``ext`` is whatever the close assembled. A row
    that lands afterwards is in the spool but not in the event, and would stay
    that way. Re-assembling and updating the same storage sequence is what
    puts it there -- the same write the close makes, made again.

    An event with an action still running is left alone: that action's own
    close will assemble the row along with everything else, and publishing
    here would show a running measurement as finished.
    """
    from ...session.sbd_v6 import timeline_sequence
    from .assembler import baseline_event_parts

    parts = baseline_event_parts()
    rows = rows_for_event(parts.get(SECTION_EVENT) or [], event)
    header = rows[0] if rows else {}
    action_rows = rows_for_event(parts.get(SECTION_ACTION) or [], event)
    ends = [str(row.get("end_time") or "") for row in action_rows]
    if not ends or not all(ends):
        return
    ext, derived = assemble_baseline_ext(parts, event=event)
    finish_event(
        event_type=EVENT_TYPE,
        event=event,
        sequence=timeline_sequence(header),
        status=derived,
        ext=ext,
        kind=EVENT_KIND,
        start_time=str(header.get("start_time") or ""),
        end_time=max(ends),
    )


def _warnings(result: Mapping[str, Any]) -> dict[str, Any]:
    """Project a result's non-fatal warnings into a bounded block."""
    rows = [str(row) for row in _as_list(result.get("nonfatal_warnings")) if str(row or "")]
    return {"count": len(rows), "messages": rows[:_MAX_ROUND_WARNINGS]}


def _measurement(result: Mapping[str, Any], framework: str) -> dict[str, Any]:
    """Project the numbers a benchmark round produced.

    Recorded on the round as well as on the action because the two answer
    different questions: the action carries the figure the session went on to
    use, and the rounds carry every figure that was measured -- including the
    cold warmup's, which is deliberately discarded and is the only thing a
    reader can weigh the adopted number against.

    Args:
        result (Mapping[str, Any]): The executor result to read.
        framework (str): The serving framework, which decides the throughput
            unit.

    Returns:
        dict[str, Any]: The measurement block, with absent numbers as ``None``.
    """
    return {
        # Named as the V5 section names it, which is what the projected event
        # published and what a consumer already selects on. The executor's own
        # key for it is ``output_throughput``.
        "throughput_tok_s_per_gpu": _float_or_none(result.get("output_throughput")),
        # The field name above is the serving case; an image framework measures
        # img/s through the same key, so the unit has to be stated rather than
        # read off the name.
        "throughput_unit": framework_registry.throughput_unit(framework),
        "ttft_mean_ms": _float_or_none(result.get("ttft_mean_ms")),
        "e2el_mean_ms": _float_or_none(result.get("e2el_mean_ms")),
        "tpot_mean_ms": _float_or_none(result.get("tpot_mean_ms")),
        # Which of the extraction's sources supplied the latency. A benchmark
        # report, a raw InferenceX JSON found in the workspace, and one
        # salvaged out of a leaked path all write the same keys, so after the
        # fact the numbers are indistinguishable -- and they are not equally
        # trustworthy. The executor labels them where it reads them.
        "ttft_e2el_source": str(result.get("ttft_e2el_source") or ""),
        # Separate because TPOT alone can be computed from the other two, and
        # a computed figure must not be read as a measured one.
        "tpot_source": str(result.get("tpot_source") or ""),
        "accuracy": _float_or_none(result.get("accuracy")),
        "accuracy_task": str(result.get("accuracy_task") or ""),
        "accuracy_metric": str(result.get("accuracy_metric") or ""),
        "accuracy_source": str(result.get("accuracy_source") or ""),
        "benchmark_report_path": str(result.get("report_path") or ""),
        "workspace": str(result.get("workspace") or ""),
    }


def _observed_invocation(result: Mapping[str, Any]) -> dict[str, Any]:
    """Project what the server was observed to have launched under.

    The declared half of the invocation is recorded before the launch, by
    :meth:`BaselineEventRecorder.record_invocation`. This is the other half,
    read back out of the launch evidence the executor builds once the round
    has run, and it is kept as its own set of fields rather than merged into
    the declared ones: a server that booted with flags the session did not ask
    for is exactly what this block exists to make visible, and overwriting the
    request with the observation would erase it.

    Returns:
        dict[str, Any]: The observed fields, empty when the result carries no
            launch evidence -- which is every failure path, since the evidence
            is built only once a round has produced a measurement.
    """
    evidence = _as_dict(result.get("launch_evidence"))
    log_path = str(result.get("server_log_path") or evidence.get("actual_server_log_path") or "")
    if not evidence and not log_path:
        return {}
    observed = str(evidence.get("observed_server_launch_flags") or "").strip()
    block: dict[str, Any] = {
        "observed_server_launch_flags": observed,
        "observed_server_identity": _as_dict(evidence.get("observed_server_identity")),
        "server_log_path": log_path,
        "recipe_digest": str(evidence.get("recipe_digest") or ""),
        "warm_reuse": _as_dict(evidence.get("warm_reuse")),
    }
    # The evidence's own view of the requested args, which is read from the
    # materialized YAML rather than from the launch call. Recorded next to the
    # launch report so a disagreement between the two is on the wire instead of
    # having to be re-derived by reading the config back.
    requested = str(evidence.get("requested_server_args") or "").strip()
    if requested:
        block["materialized_server_args"] = requested
    return block


def _timing(result: Mapping[str, Any]) -> dict[str, Any]:
    """Project the runtimes a benchmark round reported for itself."""
    return {
        "subprocess_runtime_sec": _float_or_none(result.get("subprocess_runtime_sec")),
        "post_ready_runtime_sec": _float_or_none(result.get("post_ready_runtime_sec")),
    }


def _failure(result: Mapping[str, Any], *, phase: str) -> dict[str, Any] | None:
    """Project a failed result's failure row, or ``None`` when it succeeded.

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
        failure_streak_before: Any = None,
        total_failures_before: Any = None,
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
            failure_streak_before (Any): How many baselines had failed
                consecutively when this one was dispatched. Read at the
                dispatch rather than at the close because the session's own
                counter is advanced by the write-back, after this event has
                already closed -- so the value at the close would be the one
                this action produced, not the one it was dispatched under. As
                the count going in, it says what a reader wants of an action:
                whether this was the first attempt or the fourth.
            total_failures_before (Any): How many baselines had failed in the
                session, on the same footing.
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
        self._framework = str(framework or "")
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
                    "failure_streak_before": _int_or_none(failure_streak_before),
                    "total_failures_before": _int_or_none(total_failures_before),
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
        """Update this action's own row."""
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

    def record_invocation(
        self,
        *,
        run_index: int,
        framework_args: str = "",
        extra_envs: Mapping[str, Any] | None = None,
        config_path: Any = "",
        framework: str = "",
        model_path: str = "",
        args_mode: str = "",
    ) -> None:
        """Record what this pass is about to launch the server under.

        Called at the launch, by the frame that resolved the args, which is
        the only place and moment they are known as a fact. The projection
        this replaces recovered them at export time by regexing ``server.log``
        for a launch line and falling back to re-parsing the config YAML --
        five ordered guesses deep, and reporting ``unknown`` whenever a
        framework logged its startup in a shape none of them matched.

        Recorded on the run rather than only on the action because the two
        retries this executor takes can change the args: the MoE-runner
        fallback drops a flag and re-launches, so an action-only record would
        publish one invocation for a pass that ran under two.

        Args:
            run_index (int): The pass this launch belongs to.
            framework_args (str): The extra server args the launch resolved,
                after the one-shot eager fallback and the MoE-runner drop have
                had their say. An empty string is a real answer -- the
                framework's own defaults -- and is recorded as one.
            extra_envs (Mapping[str, Any] | None): The env overrides rendered
                into the materialized config.
            config_path (Any): The materialized YAML the round renders from.
            framework (str): The serving framework being launched.
            model_path (str): The resolved model the server serves.
            args_mode (str): How the extra args combine with the config's own
                (``append`` or ``replace``), which decides whether the
                recorded string is the whole of what was requested.
        """
        invocation: dict[str, Any] = {
            "framework_args": str(framework_args or ""),
            "framework_args_source": ARGS_FROM_LAUNCH,
            "args_mode": str(args_mode or ""),
            "extra_envs": {str(key): str(value) for key, value in _as_dict(extra_envs).items()},
            "config_path": str(config_path or ""),
            "framework": str(framework or ""),
            "model_path": str(model_path or ""),
        }
        self._sink.record(
            SECTION_RUN,
            {"task_id": self._task_id, "run_index": int(run_index), "invocation": invocation},
            row_type=ROW_RUN,
            natural_ids=(self._action_id, str(int(run_index))),
        )
        # The action's own copy is the last pass to launch, which is the one
        # its adopted measurement was taken under. Repeated writes deep-merge,
        # so a later pass revises the fields it changed and leaves the rest.
        self._record_action({"invocation": invocation})

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
                "measurement": _measurement(payload, self._framework),
                "timing": _timing(payload),
                # A round is one server launch, so the observed half of the
                # invocation belongs to it. Only the observed fields: the
                # declared half is on the run, which is what decided them.
                "invocation": _observed_invocation(payload),
                "warnings": _warnings(payload),
                "failure": _failure(payload, phase=f"round_{label}"),
            },
            row_type=ROW_ROUND,
            natural_ids=(self._action_id, str(int(run_index)), str(label)),
        )

    def finish(self, result: Mapping[str, Any] | None) -> None:
        """Close the action on the result the executor returned."""
        payload = _as_dict(result)
        dropped = _as_dict(payload.get("measure_round_dropped"))
        action: dict[str, Any] = {
            "measurement": _measurement(payload, self._framework),
            "timing": _timing(payload),
        }
        # Merged onto whatever the launch already declared, which is why it is
        # only written when there is something to write: the singleton merges
        # leaf-by-leaf, and an empty observation would say nothing while a
        # missing one says the round never got far enough to be observed.
        observed = _observed_invocation(payload)
        if observed:
            action["invocation"] = observed
        self._close(
            status=self._derived_status(payload),
            action={
                **action,
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
        """Record the action's terminal facts and close the event."""
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


def _invocation_block(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a row's recorded invocation onto the wire shape.

    The two halves land separately -- the launch declares its args before the
    server boots, the round reports what was observed once it has -- so a row
    can hold either, both, or neither. The source label is settled here, at
    the one point that can see which of them arrived.

    Args:
        row (Mapping[str, Any]): The action or run row to read.

    Returns:
        dict[str, Any]: The invocation block, always carrying a source label.
    """
    block = dict(_as_dict(row.get("invocation")))
    source = str(block.get("framework_args_source") or "")
    if not source:
        # No launch report. An observed flag string is a weaker answer to the
        # same question -- it is the argv the server logged, not the args the
        # session asked for -- so it is promoted into ``framework_args`` only
        # when nothing better exists, and says so.
        observed = str(block.get("observed_server_launch_flags") or "").strip()
        block["framework_args"] = observed
        block["framework_args_source"] = ARGS_FROM_OBSERVED if observed else ARGS_UNAVAILABLE
    return block


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
            run["invocation"] = _invocation_block(run)
            run["rounds"] = wire_rows(
                by_run.get("" if index is None else str(index), []),
                drop=("event_id", "task_id", "run_index", "ordinal"),
            )
            run_rows.append(run)
        actions.append(
            {
                "task_id": task,
                "status": str(row.get("status") or "running"),
                # Absent until the write-back rules on the measurement, which
                # is after this event closed -- so a running action has no
                # verdict rather than an empty one.
                "decision": str(row.get("decision") or ""),
                "start_time": str(row.get("start_time") or ""),
                "end_time": str(row.get("end_time") or ""),
                "duration_sec": row.get("duration_sec"),
                "in_flight_run_index": row.get("in_flight_run_index"),
                "request": _as_dict(row.get("request")),
                "measurement": _as_dict(row.get("measurement")),
                "timing": _as_dict(row.get("timing")),
                "invocation": _invocation_block(row),
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
    failure_streak_before: Any = None,
    total_failures_before: Any = None,
) -> BaselineEventRecorder | None:
    """Build a recorder, or ``None`` when one cannot be constructed.

    Baseline behavior must not depend on the recorder existing, so construction
    failures degrade to "no event" rather than propagating -- as does an absent
    sink, which is what a caller with no session bound has.
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
            failure_streak_before=failure_streak_before,
            total_failures_before=total_failures_before,
        )
    except Exception:  # noqa: BLE001 — observability cannot change baseline behavior
        log.warning(
            "baseline timeline: recorder construction failed; this measurement's facts will be missing from the event",
            exc_info=True,
        )
        return None
    recorder.begin()
    return recorder
