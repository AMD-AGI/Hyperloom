# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The SBD V6 ``conc_sweep`` event: the concurrency curve, recorded live.

The sweep runs the CONC ladder twice -- once on the session's optimized server
args, once on none -- and pairs the two curves into a speedup per concurrency.
``reports/conc_sweep_summary.json`` is a result document, so a ladder that
measured three rungs of eight looks the same whether the budget gate refused
the rest, the server would not boot at those concurrencies, or the session
started closing. The rows here are therefore the sweep's own decisions,
recorded where they are made; the measurement itself is the same flattening
the report writes.

The event is flat rather than an array of actions: the SWEEP phase enqueues
through ``create_or_return_existing``, so one phase and macro cycle hold at
most one sweep. Rows are still keyed by the dispatched task id, so a second
sweep cannot merge into the first; assembly publishes the newest and counts
the rest.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hyperloom.common.coerce import (
    text_or_none as _text_or_none,
    to_float as _float_or_none,
    to_int as _int_or_none,
    to_mapping as _as_dict,
    to_list as _as_list,
)

from .event_fields import (
    bool_or_none as _bool_or_none,
    failure_row as _failure_row,
    now_iso_seconds as _now_iso,
    text as _text,
)
from .event_ids import event_id
from .event_rows import group_rows, rows_for_event, sort_rows, wire_rows
from .event_sink import EventSink
from .event_timeline import EventHeader, finish_event, open_event
from .sections import CONC_SWEEP_EVENT_SECTIONS

log = logging.getLogger(__name__)

EVENT_TYPE = "conc_sweep"
EVENT_KIND = "conc_sweep"

#: The component segment of a conc-sweep event id. The phase segment is the
#: phase the sweep was dispatched in, which is why it is a parameter.
EVENT_COMPONENT = "conc_sweep"

PRODUCER = "orchestrator"

#: The event-level section, one fragment per event, holding the timeline
#: sequence the open and close writes share.
SECTION_EVENT = "conc_sweep_event"

SECTION_ACTION = "conc_sweep_action"
SECTION_ARM = "conc_sweep_arm"
SECTION_VARIANT = "conc_sweep_variant"
SECTION_PAIR = "conc_sweep_pair"

ROW_ACTION = "action"
ROW_ARM = "arm"
ROW_VARIANT = "variant"
ROW_PAIR = "pair"

# The two arms, named here so a consumer selects them without matching prose.
ARM_BASELINE = "baseline"
ARM_OPTIMIZED = "optimized"

# How a rung came to run. An arm boots one server on its first rung and reuses
# it for the rest; a framework with no server lifecycle restarts per rung
# instead. ``boot_attempt`` is a rung whose round failed to bring the server up,
# after which the next rung boots again.
STAGE_BOOT_ATTEMPT = "boot_attempt"
STAGE_BOOT = "boot"
STAGE_REUSE = "reuse"
STAGE_SERVER_RESTART = "server_restart"
STAGE_BUDGET_SKIP = "budget_skip"

# How an arm ran its ladder. The restart path is the one a lifecycle-ineligible
# framework takes, which is why the arm also records the reason.
STRATEGY_SINGLE_SERVER = "single_server_reuse"
STRATEGY_SERVER_RESTART = "server_restart"

# Where the CONC ladder came from: handed to the sweep by the operator's flag
# or the session setting, which arrive as the same argument, or picked by the
# sweep for the workload.
GRID_REQUESTED = "requested"
GRID_MODE_DEFAULT = "mode_default"

# An arm that was refused before it built a server records the gate that
# refused it rather than a strategy it never chose.
STRATEGY_REFUSED = "refused"

# Pair statuses that count as a measured side. Mirrors the sweep's own
# ``successful_pairs`` accounting so assembly never re-rules on a pair.
_OK_POINT_STATUSES = frozenset({"succeeded", "ok", "success"})

__all__ = [
    "ARM_BASELINE",
    "ARM_OPTIMIZED",
    "EVENT_COMPONENT",
    "EVENT_KIND",
    "EVENT_TYPE",
    "GRID_MODE_DEFAULT",
    "GRID_REQUESTED",
    "PRODUCER",
    "ROW_ACTION",
    "ROW_ARM",
    "ROW_PAIR",
    "ROW_VARIANT",
    "SECTION_ACTION",
    "SECTION_ARM",
    "SECTION_EVENT",
    "SECTION_PAIR",
    "SECTION_VARIANT",
    "STAGE_BOOT",
    "STAGE_BOOT_ATTEMPT",
    "STAGE_BUDGET_SKIP",
    "STAGE_REUSE",
    "STAGE_SERVER_RESTART",
    "STRATEGY_REFUSED",
    "STRATEGY_SERVER_RESTART",
    "STRATEGY_SINGLE_SERVER",
    "ArmStrategy",
    "ConcSweepEventRecorder",
    "SweepAnchor",
    "SweepBudget",
    "SweepEnvironment",
    "SweepPlan",
    "SweepRung",
    "assemble_conc_sweep_ext",
    "conc_sweep_event_id",
    "make_conc_sweep_recorder",
]


def conc_sweep_event_id(*, phase: str, macro_cycle: Any) -> str:
    """Build ``{phase}:{macro_cycle}:conc_sweep``. Raises ``ValueError`` if
    either segment is malformed."""
    return event_id(phase, macro_cycle, EVENT_COMPONENT)


def _int_list(values: Any) -> list[int]:
    """Keep the ints a recorded ladder can supply, dropping the rest."""
    return [number for number in (_int_or_none(item) for item in _as_list(values)) if number is not None]


def _env_map(value: Any) -> dict[str, str]:
    """Normalize an environment block to the string pairs it will be read as."""
    return {str(key): str(item) for key, item in _as_dict(value).items()}


def _pair_error(
    row: Mapping[str, Any],
    *,
    baseline_point: Mapping[str, Any] | None,
    optimized_point: Mapping[str, Any] | None,
) -> str | None:
    """Explain why one concurrency produced no speedup, or ``None`` when it did.

    The pairing is an outer join, so a pair fails when an arm errored or has no
    point at that concurrency. Only the arm that did not succeed can say why --
    one status for the pair hands back ``succeeded`` whenever it is the
    optimized side that broke.
    """
    if _float_or_none(row.get("speedup")) is not None:
        return None
    reasons = []
    for arm, point in ((ARM_BASELINE, baseline_point), (ARM_OPTIMIZED, optimized_point)):
        status = _text(row.get(f"{arm}_status")).strip().lower()
        if status in _OK_POINT_STATUSES:
            continue
        fields = _as_dict(point)
        detail = _text_or_none(fields.get("error")) or _text_or_none(fields.get("error_class")) or status
        reasons.append(f"{arm}: {detail or 'no point recorded'}")
    return "; ".join(reasons) or None


@dataclass(frozen=True)
class SweepAnchor:
    """The optimized configuration the sweep was asked to compare.

    The optimized arm is whatever the session's current best was when the
    sweep started. ``action`` is the action kind that promoted it.
    """

    baseline_tput: Any = None
    anchor_tput: Any = None
    tp: Any = None
    variant_id: Any = None
    action: Any = None
    extra_server_args: Any = None
    extra_envs: Any = None

    def block(self) -> dict[str, Any]:
        """The ``input_anchor`` block this record persists as."""
        tput = _float_or_none(self.anchor_tput)
        gpus = _int_or_none(self.tp)
        per_gpu = round(tput / gpus, 4) if tput is not None and gpus else None
        return {
            "base_variant_id": _text_or_none(self.variant_id),
            "base_action": _text_or_none(self.action),
            "input_throughput_tok_s_per_gpu": per_gpu,
            "anchor_tput": tput,
            "baseline_tput": _float_or_none(self.baseline_tput),
            "extra_server_args": _text(self.extra_server_args),
            "extra_envs": _env_map(self.extra_envs),
        }


@dataclass(frozen=True)
class SweepPlan:
    """The ladder the sweep resolved and the order it will run it in.

    ``concs_ordered`` is deduplicated: the session's own CONC first, then the
    rest cheapest first. ``grid_source`` is a ``GRID_*`` value, and
    ``num_prompts_factor`` turns a rung's CONC into its NUM_PROMPTS.
    """

    concs_requested: Any = None
    concs_ordered: Any = None
    grid_source: str = ""
    num_prompts_factor: Any = None
    variant_timeout_sec: Any = None
    arms_order: Sequence[str] = ()

    def block(self) -> dict[str, Any]:
        """The ``plan`` block this record persists as."""
        return {
            "grid_source": _text_or_none(self.grid_source),
            "concs_requested": _int_list(self.concs_requested),
            "concs_ordered": _int_list(self.concs_ordered),
            "num_prompts_factor": _int_or_none(self.num_prompts_factor),
            "variant_timeout_sec": _int_or_none(self.variant_timeout_sec),
            "arms_order": [_text(arm) for arm in self.arms_order],
        }


@dataclass(frozen=True)
class SweepBudget:
    """The budget the ladder ran under.

    ``deadline`` is the sweep's wall-clock budget boundary, not the earlier
    sweep/session monotonic deadline used by the runner.
    """

    declared_total_sec: Any = None
    deadline: Any = None

    def block(self) -> dict[str, Any]:
        """The ``budget`` block this record persists as; the gate is active whenever a total was declared, and
        ``rung_cost_sec`` is null because a rung is held to its arm's deadline rather than priced."""
        return {
            "declared_total_sec": _int_or_none(self.declared_total_sec),
            "rung_cost_sec": None,
            "gate_active": self.declared_total_sec is not None,
            "deadline": _float_or_none(self.deadline),
        }


@dataclass(frozen=True)
class SweepEnvironment:
    """What the sweep resolved to run against, and where it writes its report.

    ``sweep_task_id`` is the sweep's own minted id, distinct from the
    dispatched task id: it names the ``runs/conc_sweep/`` workspace the rung
    artifacts live under.
    """

    sweep_task_id: Any = None
    workspace: Any = None
    model_path: Any = None
    gpu_type: Any = None
    base_config_path: Any = None
    report_json_path: Any = None
    report_csv_path: Any = None

    def payload(self) -> dict[str, Any]:
        """The ``environment`` and ``artifacts`` blocks this record persists as."""
        return {
            "environment": {
                "sweep_task_id": _text(self.sweep_task_id),
                "workspace": _text(self.workspace),
                "model_path": _text(self.model_path),
                "gpu_type": _text(self.gpu_type),
                "base_config_path": _text(self.base_config_path),
            },
            "artifacts": {
                "report_json_path": _text(self.report_json_path),
                "report_csv_path": _text(self.report_csv_path),
            },
        }


@dataclass(frozen=True)
class ArmStrategy:
    """How one arm will run its ladder, and why.

    ``strategy`` is a ``STRATEGY_*`` value and ``reason`` says why, when it was
    not the intended one; ``lifecycle_eligible`` is whether the framework can
    keep a server across rungs.
    """

    strategy: str
    reason: Any = None
    lifecycle_eligible: Any = None
    lifecycle_reason: Any = None
    port: Any = None
    framework: Any = None
    serving_lease_held: Any = None

    def fields(self) -> dict[str, Any]:
        """The arm row fields this record persists as."""
        return {
            "strategy": _text(self.strategy),
            "strategy_reason": _text_or_none(self.reason),
            "lifecycle": {
                "eligible": None if self.lifecycle_eligible is None else bool(self.lifecycle_eligible),
                "reason": _text(self.lifecycle_reason),
                "port": _int_or_none(self.port),
                "framework": _text(self.framework),
            },
            "serving_lease_held": None if self.serving_lease_held is None else bool(self.serving_lease_held),
        }


@dataclass(frozen=True)
class SweepRung:
    """One rung, measured or refused.

    Attributes:
        stage: A ``STAGE_*`` value.
        wall_duration_sec: Wall clock, distinct from the benchmark's own
            ``duration_seconds``, which covers only the measured window.
    """

    stage: str
    conc: Any
    point: Mapping[str, Any] | None = None
    num_prompts: Any = None
    start_time: Any = None
    wall_duration_sec: Any = None
    granted_cap_sec: Any = None
    budget_remaining_sec: Any = None


class ConcSweepEventRecorder:
    """Records one concurrency sweep as it runs.

    One instance per dispatched sweep. Every method records a fact the sweep
    knows when it is called and that the report it writes at the end either
    loses or -- for the arm and rung decisions -- never held."""

    def __init__(
        self,
        sink: EventSink,
        *,
        task_id: str = "",
        task_kind: str = "",
        reason: str = "",
        params: Mapping[str, Any] | None = None,
    ) -> None:
        """Open the action row for one dispatched sweep. ``task_id`` keys this
        sweep's rows apart from any other sweep landing in the same event, and
        ``params`` is read for the knobs the caller overrode."""
        self._sink = sink
        self._t0 = time.monotonic()
        self._start_time = _now_iso()
        self._sequence: int | None = None
        self._closed = False
        self._action_id = _text(task_id) or "unnamed"
        self._arm_ordinal = 0
        self._variant_ordinal = 0
        # The pair table is rebuilt on every flush, so keeping the measurements
        # here avoids reading rows back to say why a rung had no partner.
        self._points: dict[str, dict[int, dict[str, Any]]] = {ARM_BASELINE: {}, ARM_OPTIMIZED: {}}
        # One row per sweep, so a second fault has to be written alongside the
        # first rather than over it.
        self._faults: list[dict[str, Any]] = []
        given = _as_dict(params)
        self._sink.record(
            SECTION_ACTION,
            {
                "task_id": _text(task_id),
                "start_time": self._start_time,
                "status": "running",
                "request": {
                    "task_id": _text(task_id),
                    "task_kind": _text(task_kind),
                    "reason": _text(reason),
                    "requested_concs": _int_list(given.get("concs")),
                    "requested_variant_timeout_sec": _int_or_none(given.get("variant_timeout_sec")),
                    "requested_total_budget_sec": _int_or_none(given.get("total_budget_sec")),
                },
            },
            row_type=ROW_ACTION,
            natural_ids=self._action_id,
        )

    @property
    def event_id(self) -> str:
        """str: The event this sweep's rows belong to."""
        return self._sink.event_id

    def _record_action(self, payload: Mapping[str, Any]) -> None:
        """Update this sweep's own row."""
        self._sink.record(SECTION_ACTION, payload, row_type=ROW_ACTION, natural_ids=self._action_id)

    def _record_arm(self, arm: str, payload: Mapping[str, Any]) -> None:
        """Update one arm's row."""
        self._sink.record(
            SECTION_ARM,
            {"task_id": self._action_id, "arm": _text(arm), **dict(payload)},
            row_type=ROW_ARM,
            natural_ids=(self._action_id, _text(arm)),
        )

    # ---- lifecycle -------------------------------------------------------

    def begin(self) -> None:
        """Open the event this sweep belongs to."""
        self._sequence = open_event(
            EventHeader(EVENT_TYPE, self.event_id, EVENT_KIND, self._start_time),
            event_section=SECTION_EVENT,
            producer=PRODUCER,
        )

    # ---- the plan --------------------------------------------------------

    def record_workload(
        self,
        *,
        session_id: Any = None,
        isl: Any = None,
        osl: Any = None,
        tp: Any = None,
        benchmark_mode: Any = None,
    ) -> None:
        """Record the shape the ladder is swept over. Which axis pair the points
        are drawn on follows from ``benchmark_mode``, so a reader never infers
        it from whether ``e2e_norm_intvty_p90`` happens to be null."""
        self._record_action(
            {
                "workload": {
                    "session_id": _text(session_id),
                    "isl": _int_or_none(isl),
                    "osl": _int_or_none(osl),
                    "tp": _int_or_none(tp),
                    "benchmark_mode": _text(benchmark_mode),
                }
            }
        )

    def record_anchor(self, anchor: SweepAnchor) -> None:
        """Record the optimized configuration the sweep was asked to compare.

        That configuration is a moving target: a sweep dispatched two cycles
        later compares a different configuration under the same event type.
        """
        self._record_action({"input_anchor": anchor.block()})

    def record_plan(self, plan: SweepPlan) -> None:
        """Record the ladder the sweep resolved and the order it will run it in.

        The arm order matters to a reader of the budget: the optimized arm runs
        first as the more informative of the two, so a budget that runs out
        takes the baseline arm with it.
        """
        self._record_action({"plan": plan.block()})

    def record_budget(self, budget: SweepBudget) -> None:
        """Record the budget the ladder was admitted under."""
        self._record_action({"budget": budget.block()})

    def record_environment(self, environment: SweepEnvironment) -> None:
        """Record what the sweep resolved to run against."""
        self._record_action(environment.payload())

    # ---- arms ------------------------------------------------------------

    def open_arm(self, arm: str, *, extra_server_args: Any = None, extra_envs: Any = None) -> None:
        """Record that one arm has started its ladder. ``extra_server_args`` is
        ``""`` on the baseline arm, whose defining property is that it adds
        none."""
        self._arm_ordinal += 1
        self._record_arm(
            arm,
            {
                "ordinal": self._arm_ordinal,
                "status": "running",
                "start_time": _now_iso(),
                "extra_server_args": _text(extra_server_args),
                "extra_envs": _env_map(extra_envs),
            },
        )

    def record_arm_grid(self, arm: str, *, rungs: Sequence[Mapping[str, Any]]) -> None:
        """Record the rungs this arm will run, with the load each carries. A
        rung's NUM_PROMPTS is derived from its CONC and never written down, so
        a run cannot be reproduced from the report alone."""
        self._record_arm(
            arm,
            {
                "grid": [
                    {
                        "name": _text(rung.get("name")),
                        "conc": _int_or_none(rung.get("conc")),
                        "num_prompts": _int_or_none(rung.get("num_prompts")),
                    }
                    for rung in rungs
                    if isinstance(rung, Mapping)
                ]
            },
        )

    def record_arm_strategy(self, arm: str, strategy: ArmStrategy) -> None:
        """Record how this arm will run its ladder, and why.

        The reuse path and the restart path fail differently -- one can fail to
        boot at a concurrency, the other pays a server start per rung -- so a
        curve is not readable without knowing which one produced it.
        """
        self._record_arm(arm, strategy.fields())

    def record_arm_boot(
        self,
        arm: str,
        *,
        succeeded: bool,
        booted_conc: Any = None,
        attempted_concs: Any = None,
        failed_concs: Any = None,
    ) -> None:
        """Record how the arm's boot attempts resolved. ``attempted_concs`` is
        the rungs whose rounds tried to bring the server up, in the order tried."""
        self._record_arm(
            arm,
            {
                "boot": {
                    "succeeded": bool(succeeded),
                    "booted_conc": _int_or_none(booted_conc),
                    "attempted_concs": _int_list(attempted_concs),
                    "failed_concs": _int_list(failed_concs),
                }
            },
        )

    def record_arm_refused(self, arm: str, *, reason: Any, remaining_sec: Any = None) -> None:
        """Record an arm the budget gate refused before it built anything.
        ``reason`` names the gate and ``remaining_sec`` what was left."""
        self._record_arm(
            arm,
            {
                "strategy": STRATEGY_REFUSED,
                "strategy_reason": _text_or_none(reason),
                "refused": {
                    "reason": _text(reason),
                    "remaining_sec": _float_or_none(remaining_sec),
                },
                "status": "skipped",
                "end_time": _now_iso(),
            },
        )

    def finish_arm(self, arm: str, *, status: str) -> None:
        """Record that one arm has finished its ladder."""
        self._record_arm(arm, {"status": _text(status), "end_time": _now_iso()})

    def fail_arm(self, arm: str, exc: BaseException) -> None:
        """Record that one arm's ladder raised instead of ending on its own.

        Such an arm never reports ``succeeded``: whatever it measured is short
        of the ladder it was asked for. Whether it measured anything is read off
        the rungs recorded here rather than taken from the caller, since a
        ladder that raised never handed its results back.
        """
        measured = any(
            _text(point.get("status")).strip().lower() in _OK_POINT_STATUSES
            for point in self._points.get(_text(arm), {}).values()
        )
        self._record_arm(
            arm,
            {
                "status": "degraded" if measured else "failed",
                "end_time": _now_iso(),
                "failure": _failure_row(
                    stage="arm_ladder",
                    error_class=type(exc).__name__,
                    message=exc,
                ),
            },
        )

    # ---- rungs -----------------------------------------------------------

    def record_variant(self, arm: str, rung: SweepRung) -> None:
        """Record one rung, measured or refused.

        Everything around the measurement is what the report has no place for,
        since it carries one ``elapsed_sec`` for the whole sweep.
        """
        self._variant_ordinal += 1
        measurement = _as_dict(rung.point)
        conc = _int_or_none(rung.conc)
        if conc is not None:
            self._points.setdefault(_text(arm), {})[conc] = dict(measurement)
        self._sink.record(
            SECTION_VARIANT,
            {
                "task_id": self._action_id,
                "arm": _text(arm),
                "conc": conc,
                "stage": _text(rung.stage),
                "ordinal": self._variant_ordinal,
                "committed": True,
                "num_prompts": _int_or_none(rung.num_prompts),
                "start_time": _text(rung.start_time),
                "end_time": _now_iso(),
                "wall_duration_sec": _float_or_none(rung.wall_duration_sec),
                "granted_cap_sec": _float_or_none(rung.granted_cap_sec),
                "budget_remaining_sec": _float_or_none(rung.budget_remaining_sec),
                "measurement": measurement,
            },
            row_type=ROW_VARIANT,
            natural_ids=(self._action_id, _text(arm), _text(rung.stage), _text(conc)),
        )

    # ---- pairing ---------------------------------------------------------

    def record_progress(self, *, comparison: Any, summary: Any) -> None:
        """Record the pair table as it stands, on every checkpoint.

        Recorded on the same beat the sweep recomputes it, so an event read
        mid-sweep carries the pairs measured so far. Each pair is keyed by its
        concurrency, so a later pass revises a row rather than adding one, and
        a failure reason is settled here because the arm that broke is the only
        one that can say why.

        ``baseline_value``/``optimized_value`` are on the axis named by
        ``result.metric``, which is an interactivity percentile rather than a
        throughput whenever the session grades on one. The guard axis rides
        along beside them, null off the interactivity objective, so a rung that
        bought interactivity by giving up throughput is visible as such instead
        of reading as a clean win.
        """
        for row in _as_list(comparison):
            if not isinstance(row, Mapping):
                continue
            rung = _int_or_none(row.get("conc"))
            self._sink.record(
                SECTION_PAIR,
                {
                    "task_id": self._action_id,
                    "conc": rung,
                    "baseline_value": _float_or_none(row.get("baseline_value")),
                    "optimized_value": _float_or_none(row.get("optimized_value")),
                    "speedup": _float_or_none(row.get("speedup")),
                    "delta_pct": _float_or_none(row.get("delta_pct")),
                    "baseline_guard": _float_or_none(row.get("baseline_guard")),
                    "optimized_guard": _float_or_none(row.get("optimized_guard")),
                    "guard_holds": _bool_or_none(row.get("guard_holds")),
                    "baseline_status": _text(row.get("baseline_status")),
                    "optimized_status": _text(row.get("optimized_status")),
                    "error": _pair_error(
                        row,
                        baseline_point=self._points.get(ARM_BASELINE, {}).get(rung),
                        optimized_point=self._points.get(ARM_OPTIMIZED, {}).get(rung),
                    ),
                },
                row_type=ROW_PAIR,
                natural_ids=(self._action_id, _text(rung)),
            )
        roll_up = _as_dict(summary)
        self._record_action(
            {
                "result": {
                    "metric": _text(roll_up.get("metric")),
                    "guard_axis": _text(roll_up.get("guard_axis")),
                    "best_conc": _int_or_none(roll_up.get("best_conc")),
                    "best_speedup": _float_or_none(roll_up.get("best_speedup")),
                    "best_conc_guard_holds": _bool_or_none(roll_up.get("best_conc_guard_holds")),
                    "successful_pairs": _int_or_none(roll_up.get("successful_pairs")),
                    "failed_pairs": _int_or_none(roll_up.get("failed_pairs")),
                    "median_speedup": _float_or_none(roll_up.get("median_speedup")),
                    "mean_speedup": _float_or_none(roll_up.get("mean_speedup")),
                }
            }
        )

    # ---- closing ---------------------------------------------------------

    def record_declined(self, payload: Mapping[str, Any] | None) -> None:
        """Close a sweep that declined before it ran anything. The reason -- no
        baseline to compare against, a missing workload shape, no optimization
        yet -- is what a reader wants of a phase that produced no curve.

        Nearly every refusal is a pre-flight condition rather than an error, so
        the failure block is written only when one actually broke: an empty one
        on ``no_baseline_tput`` would claim something went wrong."""
        envelope = _as_dict(payload)
        action: dict[str, Any] = {
            "result": {
                "status": "skipped",
                "skip_reason": _text(envelope.get("skip_reason")),
                "declined": True,
            },
        }
        if envelope.get("error_class") or envelope.get("error"):
            action["failure"] = _failure_row(
                stage="decline",
                error_class=_text(envelope.get("error_class")),
                message=envelope.get("error"),
            )
        self._close("skipped", action)

    def record_fault(
        self,
        *,
        stage: str,
        exc: BaseException | None = None,
        error_class: str = "",
        message: Any = "",
    ) -> None:
        """Record a step that raised while the sweep carried on.

        A sweep that could not resolve a lifecycle, write its report, or clean
        up after itself is not a failed sweep, so none of this belongs in
        ``failure``. It does not belong only in a log either: the step that
        broke is exactly what the report has no field for, and what makes an
        otherwise clean sweep readable when its artifacts are missing.
        """
        self._faults.append(_failure_row(stage=stage, exc=exc, error_class=error_class, message=message))
        self._record_action({"faults": list(self._faults)})

    def finish(self, payload: Mapping[str, Any] | None, *, stop_reason: Any = None) -> None:
        """Close the sweep on the report it produced.

        The payload is the sweep's own final document, so the fields taken off
        it here are the ones it settles only at the end: the ceiling, the
        roll-up over the pairs, and the budget state after both arms have run.

        ``stop_reason`` is set when the session's end cut the sweep short, and
        belongs beside the rest of how the run went rather than in ``failure``:
        a sweep the session stopped measured what it got to, and nothing about
        it failed.
        """
        final = _as_dict(payload)
        status = _text(final.get("status")) or "failed"
        roll_up = _as_dict(final.get("summary"))
        budget_exhausted = final.get("budget_exhausted")
        result = {
            "status": status,
            "metric": _text(roll_up.get("metric")),
            "guard_axis": _text(roll_up.get("guard_axis")),
            "best_conc": _int_or_none(roll_up.get("best_conc")),
            "best_speedup": _float_or_none(roll_up.get("best_speedup")),
            "best_conc_guard_holds": _bool_or_none(roll_up.get("best_conc_guard_holds")),
            "successful_pairs": _int_or_none(roll_up.get("successful_pairs")),
            "failed_pairs": _int_or_none(roll_up.get("failed_pairs")),
            "median_speedup": _float_or_none(roll_up.get("median_speedup")),
            "mean_speedup": _float_or_none(roll_up.get("mean_speedup")),
            "skip_reason": _text(final.get("skip_reason")),
            "was_skipped": bool(final.get("was_skipped")),
            "budget_exhausted": None if budget_exhausted is None else bool(budget_exhausted),
            "declined": False,
        }
        action: dict[str, Any] = {
            "result": result,
            "runtime": {
                "elapsed_sec": _float_or_none(final.get("elapsed_sec")),
                "budget_remaining_sec": _float_or_none(final.get("budget_remaining_sec")),
                "budget_skip_reason": _text(final.get("budget_skip_reason")),
                "stop_reason": _text(stop_reason),
            },
            "roofline_ceiling": _as_dict(final.get("roofline_ceiling")) or None,
            "schema_version": _text(final.get("schema_version")),
        }
        report_path = _text_or_none(final.get("report_json_path"))
        if report_path:
            action["artifacts"] = {
                "report_json_path": report_path,
                "report_csv_path": _text(final.get("report_csv_path")),
            }
        self._close(status, action)

    def finish_crashed(self, exc: BaseException) -> None:
        """Close a sweep whose own execution raised."""
        self._close(
            "failed",
            {
                "result": {"status": "failed", "declined": False},
                "failure": _failure_row(
                    stage=EVENT_TYPE,
                    error_class=type(exc).__name__,
                    message=exc,
                ),
            },
        )

    def _close(self, status: str, action: Mapping[str, Any]) -> None:
        """Write the sweep's terminal row and close its event."""
        if self._closed:
            return
        self._closed = True
        end_time = _now_iso()
        self._record_action(
            {
                **action,
                "status": _text(status),
                "end_time": end_time,
                "duration_sec": round(time.monotonic() - self._t0, 3),
            }
        )
        from .assembler import event_parts
        from .recorder_warnings import RECORDING_ERRORS, note_failure

        try:
            ext, derived = assemble_conc_sweep_ext(
                event_parts(CONC_SWEEP_EVENT_SECTIONS, event=self.event_id), event=self.event_id
            )
            finish_event(
                EventHeader(EVENT_TYPE, self.event_id, EVENT_KIND, self._start_time),
                sequence=self._sequence,
                status=derived or status,
                ext=ext,
                end_time=end_time,
            )
        except RECORDING_ERRORS as exc:
            note_failure(section=SECTION_EVENT, error=exc, detail=f"closing conc_sweep event {self.event_id}")


def _assemble_arm(
    row: Mapping[str, Any],
    *,
    variants: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble one arm out of its row and its rungs. ``points`` is the curve:
    the committed attempts, ascending; an uncommitted row is an older
    session's boot attempt a later rung superseded. ``boot.attempts`` is the
    boot rounds in the order tried."""
    points = [
        {
            **_as_dict(rung.get("measurement")),
            "stage": _text(rung.get("stage")),
            "num_prompts": rung.get("num_prompts"),
            "start_time": _text(rung.get("start_time")),
            "end_time": _text(rung.get("end_time")),
            "wall_duration_sec": rung.get("wall_duration_sec"),
            "granted_cap_sec": rung.get("granted_cap_sec"),
            "budget_remaining_sec": rung.get("budget_remaining_sec"),
        }
        for rung in variants
        if rung.get("committed")
    ]
    points.sort(key=lambda point: (_int_or_none(point.get("conc")) is None, _int_or_none(point.get("conc")) or 0))
    attempts: list[dict[str, Any]] = []
    for rung in variants:
        if _text(rung.get("stage")) not in (STAGE_BOOT_ATTEMPT, STAGE_BOOT):
            continue
        measured = _as_dict(rung.get("measurement"))
        attempts.append(
            {
                "conc": rung.get("conc"),
                "committed": bool(rung.get("committed")),
                "status": _text(measured.get("status")),
                "error_class": _text(measured.get("error_class")),
                "error": _text_or_none(measured.get("error")),
                "start_time": _text(rung.get("start_time")),
                "wall_duration_sec": rung.get("wall_duration_sec"),
            }
        )
    boot = dict(_as_dict(row.get("boot")))
    boot["attempts"] = attempts
    return {
        "arm": _text(row.get("arm")),
        "status": _text(row.get("status")) or "running",
        "start_time": _text(row.get("start_time")),
        "end_time": _text(row.get("end_time")),
        "extra_server_args": _text(row.get("extra_server_args")),
        "extra_envs": _env_map(row.get("extra_envs")),
        "strategy": _text(row.get("strategy")),
        "strategy_reason": row.get("strategy_reason"),
        "lifecycle": _as_dict(row.get("lifecycle")),
        "serving_lease_held": row.get("serving_lease_held"),
        "refused": _as_dict(row.get("refused")) or None,
        "failure": _as_dict(row.get("failure")) or None,
        "grid": _as_list(row.get("grid")),
        "boot": boot,
        "points": points,
    }


def _status_for(row: Mapping[str, Any], ext: Mapping[str, Any]) -> str:
    """The status a sweep event reports, read off its action row and its assembled ``ext``."""
    status = _text(row.get("status")) or "running"
    if status == "cancelled":
        # The timeline has no word for a sweep stopped from outside. It ran and fell short of its ladder without
        # anything failing, which is what degraded says; ``result.status`` keeps the sweep's own word.
        status = "degraded"
    if status == "succeeded" and ext["result"].get("budget_exhausted"):
        # A curve cut short by the time budget still produced usable pairs,
        # but not the ladder that was asked for.
        status = "degraded"
    ladders_run = [arm for arm in ext["arms"].values() if _text(arm.get("start_time"))]
    if status == "succeeded" and any(_text(arm.get("status")) != "succeeded" for arm in ladders_run):
        # A ladder that lost rungs is the same shortfall one level down, and
        # the sweep cannot read as whole while an arm under it does not. An arm
        # with no row never ran, which is not the same as falling short.
        status = "degraded"
    return status


def assemble_conc_sweep_ext(
    parts: Mapping[str, list[dict[str, Any]]],
    *,
    event: str,
) -> tuple[dict[str, Any], str]:
    """Assemble one conc-sweep event's ``ext`` and the sweep's status out of its
    recorded rows. Both are empty when the event holds no sweep, which is what
    a caller assembling an event that was never recorded gets."""
    action_rows = sort_rows(
        rows_for_event(parts.get(SECTION_ACTION) or [], event),
        keys=("start_time", "task_id"),
    )
    if not action_rows:
        return {}, ""
    row = action_rows[-1]
    variants = group_rows(
        sort_rows(
            rows_for_event(parts[SECTION_VARIANT], event),
            keys=("ordinal", "arm", "conc"),
        ),
        "arm",
    )
    arm_rows = {
        _text(arm_row.get("arm")): arm_row
        for arm_row in sort_rows(rows_for_event(parts[SECTION_ARM], event), keys=("ordinal", "arm"))
    }
    pairs = wire_rows(
        sort_rows(rows_for_event(parts[SECTION_PAIR], event), keys=("conc",)),
        drop=("event_id", "task_id", "ordinal"),
    )
    ext: dict[str, Any] = {
        "schema_version": _text(row.get("schema_version")),
        "request": _as_dict(row.get("request")),
        "input_anchor": _as_dict(row.get("input_anchor")),
        "workload": _as_dict(row.get("workload")),
        "plan": _as_dict(row.get("plan")),
        "budget": _as_dict(row.get("budget")),
        "environment": _as_dict(row.get("environment")),
        "arms": {
            arm: _assemble_arm(arm_rows.get(arm, {"arm": arm}), variants=variants.get(arm, []))
            for arm in (ARM_BASELINE, ARM_OPTIMIZED)
        },
        "comparison": pairs,
        "result": _as_dict(row.get("result")),
        "roofline_ceiling": row.get("roofline_ceiling"),
        "runtime": {
            **_as_dict(row.get("runtime")),
            "workspace": _text(_as_dict(row.get("environment")).get("workspace")),
            "duration_sec": row.get("duration_sec"),
        },
        "artifacts": _as_dict(row.get("artifacts")),
        # Steps that raised and were recovered from, in the order they broke.
        # Empty on a sweep where nothing did.
        "faults": _as_list(row.get("faults")),
        # Absent on a sweep that did not fail, the way every other event reads.
        "failure": _as_dict(row.get("failure")) or None,
    }
    if len(action_rows) > 1:
        # One phase and cycle enqueue one sweep, so this cannot happen through
        # the dispatcher; saying so is cheaper than a silent drop if it does.
        ext["superseded_sweeps"] = [_text(other.get("task_id")) for other in action_rows[:-1]]
    return ext, _status_for(row, ext)


def make_conc_sweep_recorder(
    sink: EventSink | None,
    *,
    task_id: str = "",
    task_kind: str = "",
    reason: str = "",
    params: Mapping[str, Any] | None = None,
) -> ConcSweepEventRecorder | None:
    """Build a recorder, or ``None`` when ``sink`` is absent."""
    if sink is None:
        return None
    recorder = ConcSweepEventRecorder(
        sink,
        task_id=task_id,
        task_kind=task_kind,
        reason=reason,
        params=params,
    )
    recorder.begin()
    return recorder
