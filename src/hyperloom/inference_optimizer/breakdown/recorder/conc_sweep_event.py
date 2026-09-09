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
from typing import Any

from .event_fields import (
    as_dict as _as_dict,
    as_list as _as_list,
    clip as _clip,
    float_or_none as _float_or_none,
    int_or_none as _int_or_none,
    now_iso_seconds as _now_iso,
    text_or_none as _text_or_none,
)
from .event_ids import event_id
from .event_rows import group_rows, rows_for_event, sort_rows, wire_rows
from .event_sink import RecordSink
from .event_timeline import finish_event, open_event

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

# How a rung came to run. The sweep prefers to boot one server at the top of
# the ladder and reuse it down; a framework with no server lifecycle, or a
# ladder whose every boot failed, restarts per rung instead. ``boot_attempt``
# is a rung the retry-descend loop could not bring up, kept whether or not it
# was later committed, because that concurrency is itself the finding.
STAGE_BOOT_ATTEMPT = "boot_attempt"
STAGE_BOOT = "boot"
STAGE_REUSE = "reuse"
STAGE_SERVER_RESTART = "server_restart"
STAGE_BUDGET_SKIP = "budget_skip"

# How an arm ran its ladder. The restart path is both the fallback for a
# lifecycle-ineligible framework and the retry after every boot failed, which
# is why the arm also records the reason.
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
    "ConcSweepEventRecorder",
    "assemble_conc_sweep_ext",
    "conc_sweep_event_id",
    "make_conc_sweep_recorder",
]


def conc_sweep_event_id(*, phase: str, macro_cycle: Any) -> str:
    """Build ``{phase}:{macro_cycle}:conc_sweep``. Raises ``ValueError`` if
    either segment is malformed."""
    return event_id(phase, macro_cycle, EVENT_COMPONENT)


def _text(value: Any) -> str:
    """Coerce to a plain string, treating ``None`` as empty."""
    return str(value or "")


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


class ConcSweepEventRecorder:
    """Records one concurrency sweep as it runs.

    One instance per dispatched sweep. Every method records a fact the sweep
    knows when it is called and that the report it writes at the end either
    loses or -- for the arm and rung decisions -- never held."""

    def __init__(
        self,
        sink: RecordSink,
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
            event_type=EVENT_TYPE,
            event=self.event_id,
            event_section=SECTION_EVENT,
            producer=PRODUCER,
            kind=EVENT_KIND,
            start_time=self._start_time,
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
        it from whether ``intvty_p90`` happens to be null."""
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

    def record_anchor(
        self,
        *,
        baseline_tput: Any = None,
        anchor_tput: Any = None,
        tp: Any = None,
        variant_id: Any = None,
        action: Any = None,
        extra_server_args: Any = None,
        extra_envs: Any = None,
    ) -> None:
        """Record the optimized configuration the sweep was asked to compare.

        The optimized arm is whatever the session's current best was when the
        sweep started, and that is a moving target: a sweep dispatched two
        cycles later compares a different configuration under the same event
        type. ``action`` is the action kind that promoted the current best.
        """
        tput = _float_or_none(anchor_tput)
        gpus = _int_or_none(tp)
        per_gpu = round(tput / gpus, 4) if tput is not None and gpus else None
        self._record_action(
            {
                "input_anchor": {
                    "base_variant_id": _text_or_none(variant_id),
                    "base_action": _text_or_none(action),
                    "input_throughput_tok_s_per_gpu": per_gpu,
                    "anchor_tput": tput,
                    "baseline_tput": _float_or_none(baseline_tput),
                    "extra_server_args": _text(extra_server_args),
                    "extra_envs": _env_map(extra_envs),
                }
            }
        )

    def record_plan(
        self,
        *,
        concs_requested: Any = None,
        concs_ordered: Any = None,
        grid_source: str = "",
        num_prompts_factor: Any = None,
        variant_timeout_sec: Any = None,
        arms_order: Sequence[str] = (),
    ) -> None:
        """Record the ladder the sweep resolved and the order it will run it in.

        The arm order matters to a reader of the budget: the optimized arm runs
        first as the more informative of the two, so a budget that runs out
        takes the baseline arm with it. ``concs_ordered`` is deduplicated and
        descending, because a single-server arm boots at the most demanding
        rung so the lower ones need no restart. ``grid_source`` is a ``GRID_*``
        value, and ``num_prompts_factor`` turns a rung's CONC into its
        NUM_PROMPTS.
        """
        self._record_action(
            {
                "plan": {
                    "grid_source": _text_or_none(grid_source),
                    "concs_requested": _int_list(concs_requested),
                    "concs_ordered": _int_list(concs_ordered),
                    "num_prompts_factor": _int_or_none(num_prompts_factor),
                    "variant_timeout_sec": _int_or_none(variant_timeout_sec),
                    "arms_order": [_text(arm) for arm in arms_order],
                }
            }
        )

    def record_budget(
        self,
        *,
        declared_total_sec: Any = None,
        granted_total_sec: Any = None,
        rung_cost_sec: Any = None,
        raised: bool = False,
        gate_active: bool = True,
        deadline: Any = None,
        session_soft_deadline_sec: Any = None,
    ) -> None:
        """Record the budget the ladder was admitted under.

        Both totals are kept because they disagree: the sweep raises its own
        default when that default cannot fund even one rung at the cap the grid
        runner will actually grant, and a sweep that spent three hours on a
        nine-hundred-second budget otherwise reads as a contradiction.
        ``rung_cost_sec`` is that granted cap rather than the declared timeout,
        since pricing at the smaller admits a rung the budget cannot pay for.
        ``deadline`` is a wall-clock epoch.
        """
        self._record_action(
            {
                "budget": {
                    "declared_total_sec": _int_or_none(declared_total_sec),
                    "granted_total_sec": _int_or_none(granted_total_sec),
                    "rung_cost_sec": _float_or_none(rung_cost_sec),
                    "raised": bool(raised),
                    "gate_active": bool(gate_active),
                    "deadline": _float_or_none(deadline),
                    "session_soft_deadline_sec": _float_or_none(session_soft_deadline_sec),
                }
            }
        )

    def record_environment(
        self,
        *,
        sweep_task_id: Any = None,
        workspace: Any = None,
        model_path: Any = None,
        gpu_type: Any = None,
        base_config_path: Any = None,
        report_json_path: Any = None,
        report_csv_path: Any = None,
    ) -> None:
        """Record what the sweep resolved to run against.

        ``sweep_task_id`` is the sweep's own minted id, distinct from the
        dispatched task id: it names the ``runs/conc_sweep/`` workspace the rung
        artifacts live under, and nothing else ever wrote it down.
        """
        self._record_action(
            {
                "environment": {
                    "sweep_task_id": _text(sweep_task_id),
                    "workspace": _text(workspace),
                    "model_path": _text(model_path),
                    "gpu_type": _text(gpu_type),
                    "base_config_path": _text(base_config_path),
                },
                "artifacts": {
                    "report_json_path": _text(report_json_path),
                    "report_csv_path": _text(report_csv_path),
                },
            }
        )

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

    def record_arm_strategy(
        self,
        arm: str,
        *,
        strategy: str,
        reason: Any = None,
        lifecycle_eligible: Any = None,
        lifecycle_reason: Any = None,
        port: Any = None,
        framework: Any = None,
        serving_lease_held: Any = None,
    ) -> None:
        """Record how this arm will run its ladder, and why.

        The reuse path and the restart path fail differently -- one can fail to
        boot at a concurrency, the other pays a server start per rung -- so a
        curve is not readable without knowing which one produced it.
        ``strategy`` is a ``STRATEGY_*`` value and ``reason`` says why, when it
        was not the intended one; ``lifecycle_eligible`` is whether the
        framework can keep a server across rungs.
        """
        self._record_arm(
            arm,
            {
                "strategy": _text(strategy),
                "strategy_reason": _text_or_none(reason),
                "lifecycle": {
                    "eligible": None if lifecycle_eligible is None else bool(lifecycle_eligible),
                    "reason": _text(lifecycle_reason),
                    "port": _int_or_none(port),
                    "framework": _text(framework),
                },
                "serving_lease_held": None if serving_lease_held is None else bool(serving_lease_held),
            },
        )

    def record_arm_boot(
        self,
        arm: str,
        *,
        succeeded: bool,
        booted_conc: Any = None,
        attempted_concs: Any = None,
        failed_concs: Any = None,
    ) -> None:
        """Record how the boot-retry-descend loop resolved. ``attempted_concs``
        is in the order tried, and ``failed_concs`` is the capacity finding the
        sweep produces for free and then discards."""
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

    # ---- rungs -----------------------------------------------------------

    def record_variant(
        self,
        arm: str,
        *,
        stage: str,
        conc: Any,
        point: Mapping[str, Any] | None = None,
        committed: bool = True,
        num_prompts: Any = None,
        start_time: Any = None,
        wall_duration_sec: Any = None,
        granted_cap_sec: Any = None,
        budget_remaining_sec: Any = None,
    ) -> None:
        """Record one rung, measured or refused.

        Everything around the measurement is what the report has no place for,
        since it carries one ``elapsed_sec`` for the whole sweep.

        Args:
            stage: A ``STAGE_*`` value.
            committed: Whether this attempt is part of the published curve. A
                failed boot is recorded uncommitted and promoted by
                :meth:`commit_variant` if a lower rung eventually boots.
            wall_duration_sec: Wall clock, distinct from the benchmark's own
                ``duration_seconds``, which covers only the measured window.
        """
        self._variant_ordinal += 1
        measurement = _as_dict(point)
        rung = _int_or_none(conc)
        if committed and rung is not None:
            self._points.setdefault(_text(arm), {})[rung] = dict(measurement)
        self._sink.record(
            SECTION_VARIANT,
            {
                "task_id": self._action_id,
                "arm": _text(arm),
                "conc": rung,
                "stage": _text(stage),
                "ordinal": self._variant_ordinal,
                "committed": bool(committed),
                "num_prompts": _int_or_none(num_prompts),
                "start_time": _text(start_time),
                "end_time": _now_iso(),
                "wall_duration_sec": _float_or_none(wall_duration_sec),
                "granted_cap_sec": _float_or_none(granted_cap_sec),
                "budget_remaining_sec": _float_or_none(budget_remaining_sec),
                "measurement": measurement,
            },
            row_type=ROW_VARIANT,
            natural_ids=(self._action_id, _text(arm), _text(stage), _text(rung)),
        )

    def commit_variant(self, arm: str, *, stage: str, conc: Any, point: Mapping[str, Any] | None = None) -> None:
        """Promote an attempt recorded before it was known to count. ``point``
        is passed so the pair table can explain a rung this attempt is the only
        record of."""
        rung = _int_or_none(conc)
        if rung is not None and point is not None:
            self._points.setdefault(_text(arm), {})[rung] = dict(_as_dict(point))
        self._sink.record(
            SECTION_VARIANT,
            {"task_id": self._action_id, "committed": True},
            row_type=ROW_VARIANT,
            natural_ids=(self._action_id, _text(arm), _text(stage), _text(rung)),
        )

    # ---- pairing ---------------------------------------------------------

    def record_progress(self, *, comparison: Any, summary: Any) -> None:
        """Record the pair table as it stands, on every checkpoint.

        Recorded on the same beat the sweep recomputes it, so an event read
        mid-sweep carries the pairs measured so far. Each pair is keyed by its
        concurrency, so a later pass revises a row rather than adding one, and
        a failure reason is settled here because the arm that broke is the only
        one that can say why.
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
                    "baseline_throughput": _float_or_none(row.get("baseline_tput")),
                    "optimized_throughput": _float_or_none(row.get("optimized_tput")),
                    "speedup": _float_or_none(row.get("speedup")),
                    "delta_pct": _float_or_none(row.get("delta_pct")),
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
                    "best_conc": _int_or_none(roll_up.get("best_conc")),
                    "best_speedup": _float_or_none(roll_up.get("best_speedup")),
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
        yet -- is what a reader wants of a phase that produced no curve."""
        envelope = _as_dict(payload)
        self._close(
            "skipped",
            {
                "result": {
                    "status": "skipped",
                    "skip_reason": _text(envelope.get("skip_reason")),
                    "declined": True,
                },
                "failure": {
                    "error_class": _text(envelope.get("error_class")),
                    "message": _clip(envelope.get("error"), 2000),
                },
            },
        )

    def finish(self, payload: Mapping[str, Any] | None, *, stop_reason: Any = None) -> None:
        """Close the sweep on the report it produced.

        The payload is the sweep's own final document, so the fields taken off
        it here are the ones it settles only at the end: the ceiling, the
        roll-up over the pairs, and the budget state after both arms have run.
        ``stop_reason`` is set when the session's end cut the sweep short.
        """
        final = _as_dict(payload)
        status = _text(final.get("status")) or "failed"
        roll_up = _as_dict(final.get("summary"))
        budget_exhausted = final.get("budget_exhausted")
        result = {
            "status": status,
            "metric": _text(roll_up.get("metric")),
            "best_conc": _int_or_none(roll_up.get("best_conc")),
            "best_speedup": _float_or_none(roll_up.get("best_speedup")),
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
            },
            "roofline_ceiling": _as_dict(final.get("roofline_ceiling")) or None,
            "schema_version": _text(final.get("schema_version")),
            "failure": {
                "stop_reason": _text(stop_reason),
                "message": _text(final.get("budget_skip_reason")) or _text(final.get("skip_reason")),
            },
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
                "failure": {
                    "error_class": type(exc).__name__,
                    "message": _clip(exc, 2000),
                },
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
        from .assembler import conc_sweep_event_parts

        ext, derived = assemble_conc_sweep_ext(conc_sweep_event_parts(), event=self.event_id)
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


def _assemble_arm(
    row: Mapping[str, Any],
    *,
    variants: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble one arm out of its row and its rungs. ``points`` is the curve:
    the committed attempts, ascending. ``boot.attempts`` is the descend ladder
    in the order tried, carrying the rungs that never produced a point."""
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
        "grid": _as_list(row.get("grid")),
        "boot": boot,
        "points": points,
    }


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
            rows_for_event(parts.get(SECTION_VARIANT) or [], event),
            keys=("ordinal", "arm", "conc"),
        ),
        "arm",
    )
    arm_rows = {
        _text(arm_row.get("arm")): arm_row
        for arm_row in sort_rows(rows_for_event(parts.get(SECTION_ARM) or [], event), keys=("ordinal", "arm"))
    }
    pairs = wire_rows(
        sort_rows(rows_for_event(parts.get(SECTION_PAIR) or [], event), keys=("conc",)),
        drop=("event_id", "task_id", "ordinal"),
    )
    result = _as_dict(row.get("result"))
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
        "result": result,
        "roofline_ceiling": row.get("roofline_ceiling"),
        "runtime": {
            **_as_dict(row.get("runtime")),
            "workspace": _text(_as_dict(row.get("environment")).get("workspace")),
            "duration_sec": row.get("duration_sec"),
        },
        "artifacts": _as_dict(row.get("artifacts")),
        "failure": _as_dict(row.get("failure")),
    }
    if len(action_rows) > 1:
        # One phase and cycle enqueue one sweep, so this cannot happen through
        # the dispatcher; saying so is cheaper than a silent drop if it does.
        ext["superseded_sweeps"] = [_text(other.get("task_id")) for other in action_rows[:-1] if other is not row]
    ext["arms"][ARM_BASELINE]["arm"] = ARM_BASELINE
    ext["arms"][ARM_OPTIMIZED]["arm"] = ARM_OPTIMIZED
    status = _text(row.get("status")) or "running"
    if status == "succeeded" and result.get("budget_exhausted"):
        # A curve cut short by the time budget still produced usable pairs,
        # but not the ladder that was asked for.
        status = "degraded"
    return ext, status


def make_conc_sweep_recorder(
    sink: RecordSink | None,
    *,
    task_id: str = "",
    task_kind: str = "",
    reason: str = "",
    params: Mapping[str, Any] | None = None,
) -> ConcSweepEventRecorder | None:
    """Build a recorder, or ``None`` when one cannot be constructed. Sweep
    behavior must not depend on the recorder existing, so construction failures
    degrade to "no event", as does the absent sink a caller with no session
    bound has."""
    if sink is None:
        return None
    try:
        recorder = ConcSweepEventRecorder(
            sink,
            task_id=task_id,
            task_kind=task_kind,
            reason=reason,
            params=params,
        )
    except Exception:  # noqa: BLE001 — observability cannot change sweep behavior
        log.warning(
            "conc_sweep timeline: recorder construction failed; this sweep's facts will be missing from the event",
            exc_info=True,
        )
        return None
    recorder.begin()
    return recorder
