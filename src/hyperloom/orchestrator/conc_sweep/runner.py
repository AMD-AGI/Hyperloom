# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The concurrency sweep of a finished session: its baseline and its optimized launch over the recipe's concurrencies.

The points are the operator's ``--conc-sweep-concs``, else the concurrencies the InferenceX arm of the session's base
publishes, with the session's own CONC among them. Each arm runs its
points serially, the session's CONC first and then low to high; an arm keeps one server across its points unless the
workload is an AgentX replay, whose points each boot their own (``BenchmarkPoint.sweeps_on_cold_servers``). The report
is rewritten after every point, and a later run measures only the points not yet measured.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from hyperloom.common.gain_math import conc_pair_comparison
from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.common.overlay import is_safe_path_entry, overlay_is_loadable
from hyperloom.common.perf_metric import GRADED_INTVTY, GRADED_OUTPUT, agentx_replay_session, is_agentx_mode
from hyperloom.inference_optimizer.framework_registry import is_scriptable, native_component
from hyperloom.inference_optimizer.grid_server_args import (
    merge_server_args,
    server_args_delta,
)
from hyperloom.inference_optimizer.grading import resolved_grading
from hyperloom.inference_optimizer.session.session_paths import reports_dir, runs_root
from hyperloom.orchestrator.actions.cancel_channel import current_cancel_scope
from hyperloom.orchestrator.actions.executors._proposal_identity import controls_of, is_executable, normalize_proposal
from hyperloom.orchestrator.actions.executors.grid._grid_runner import (
    GridLaunch,
    GridRecipe,
    GridVariant,
    VariantResult,
    run_grid,
    teardown_variant_server,
)
from hyperloom.orchestrator.actions.executors.serving._server_lifecycle import resolve_lifecycle_params
from hyperloom.orchestrator.actions.executors.workload._workload_envs import (
    BenchmarkTarget,
    FrameworkScriptMismatchError,
    MaterializeOptions,
    default_baseline_config,
    num_prompts_per_conc,
    prepare_grid_benchmark_config,
)
from hyperloom.orchestrator.actions.stop_attribution import ORCHESTRATOR_CANCELLED_CLASS
from hyperloom.orchestrator.state.base_launch import frozen_base, launch_change, server_base, server_base_at
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom_common.coerce import to_str_list
from hyperloom_common.timeutil import utc_now_compact
from inference_suite.argv import argv_flag_occurrences, canonical_flag_name, remove_server_args
from inference_suite.contracts import BaseLaunch

from .plot import ConcCurveOptions
from .points import SCHEMA_VERSION, SweepPoints, ladder, launches_missing, sweep_points
from .report import (
    ArmCurves,
    build_roofline_ceiling,
    point_from_variant,
    publish_curve,
    take_earlier_summary,
    run_facts,
    write_report,
)

log = logging.getLogger(__name__)

_SUCCEEDED = "succeeded"


@dataclass(frozen=True)
class Arm:
    """One side of the comparison: the launch every point of its ladder runs with."""

    name: str
    args: str = ""
    envs: dict[str, str] = field(default_factory=dict)
    overlay: str = ""
    controls: Mapping[str, Any] = field(default_factory=dict)


BASELINE = Arm(name="baseline")


def optimized_arm(state: SharedState) -> Arm | None:
    """The arm for the retained config or kernel overlay, or ``None`` when the session retained neither."""
    best = state.current_best or {}
    config = normalize_proposal(best)
    overlay = str(best.get("final_overlay") or "").strip()
    if not (is_executable(config) or overlay):
        return None
    return Arm("optimized", config["extra_args"], config["extra_envs"], overlay, controls_of(config))


def decline_reason(state: SharedState) -> str | None:
    """Why the session's workload gets no sweep, or ``None`` when it does."""
    if state.nodes > 1:
        return "multi_node"
    if is_scriptable(state.framework):
        return "non_serving_workload"
    if native_component(state.framework) is not None:
        # An arm names no build, so both arms would run the build the session kept last.
        return "checkout_built_server"
    if is_agentx_mode(state.benchmark_mode) and not agentx_replay_session(state):
        return "mlperf_backend"
    return None


@dataclass(frozen=True)
class _Layer:
    """A change to the session's composed baseline launch: args set, envs set, args removed and envs unset; ``replace``
    when ``args`` is the server's whole argv rather than an addition to the baseline's."""

    args: str = ""
    envs: Mapping[str, str] = field(default_factory=dict)
    remove_args: tuple[str, ...] = ()
    unset_envs: tuple[str, ...] = ()
    replace: bool = False


def _arm_layer(arm: Arm) -> _Layer:
    return _Layer(
        arm.args,
        dict(arm.envs),
        tuple(to_str_list(arm.controls.get("remove_args"))),
        tuple(to_str_list(arm.controls.get("unset_envs"))),
    )


def _appended(arm: Arm, frozen: BaseLaunch | None) -> _Layer:
    """``arm``'s appended change as what it changes of the base's own launch: a flag or variable it sets to the value
    that launch already has is left to the source's launch at each concurrency."""
    layer = _arm_layer(arm)
    if frozen is None:
        return layer
    change = launch_change(frozen, layer.args, layer.envs)
    return replace(layer, args=change.args, envs=dict(change.env))


def _replacing(arm: Arm, frozen: BaseLaunch | None, conc: int, official: _Layer) -> _Layer:
    """``arm``'s replacing argv moved to ``conc``: what it changes of the base's own launch, applied to the source's
    launch at ``conc``, so a flag it leaves as the base had it takes the value the recipe gives ``conc``."""
    layer = _arm_layer(arm)
    argv = remove_server_args(arm.args, list(layer.remove_args))
    if frozen is not None:
        delta = server_args_delta(server_base(frozen).args, argv)
        argv = merge_server_args(remove_server_args(server_base_at(frozen, conc).args, list(delta.remove)), delta.add)
    envs = _stacked(official, replace(layer, args="", remove_args=()))
    return _Layer(argv, envs.envs, layer.remove_args, envs.unset_envs, replace=True)


def _conc_layer(frozen: BaseLaunch | None, conc: int) -> _Layer:
    """What the source's own launch at ``conc`` changes of the base, which is its launch at the session's CONC."""
    if frozen is None:
        return _Layer()
    primary, seed = server_base(frozen), server_base_at(frozen, conc)
    delta = server_args_delta(primary.args, seed.args)
    envs = {name: value for name, value in seed.env.items() if primary.env.get(name) != value}
    return _Layer(delta.add, envs, delta.remove, tuple(name for name in primary.env if name not in seed.env))


def _stacked(lower: _Layer, upper: _Layer) -> _Layer:
    """``upper`` over ``lower``: what ``upper`` removes or unsets goes even where ``lower`` sets it, and what ``upper``
    sets stays even where ``lower`` removes or unsets it."""
    args = merge_server_args(remove_server_args(lower.args, list(upper.remove_args)), upper.args)
    envs = {name: value for name, value in lower.envs.items() if name not in upper.unset_envs} | dict(upper.envs)
    unset = [name for name in lower.unset_envs if name not in upper.envs] + list(upper.unset_envs)
    upper_flags = {name for name, _value in argv_flag_occurrences(upper.args.split())}
    removed = [spec for spec in lower.remove_args if _flag_name(spec) not in upper_flags] + list(upper.remove_args)
    return _Layer(args, envs, tuple(dict.fromkeys(removed)), tuple(dict.fromkeys(unset)))


def _flag_name(removal: str) -> str:
    return canonical_flag_name(removal.split()[0].partition("=")[0])


@dataclass(frozen=True)
class _Setup:
    """Where the sweep runs and the materialized base config both arms launch from."""

    workspace: Path
    base_yaml_path: Path
    target: BenchmarkTarget


def _prepare(state: SharedState, session_dir: Path) -> _Setup | str:
    """The sweep's workspace and base config, or why the session cannot give one."""
    base_yaml = Path(state.baseline_config_path or default_baseline_config())
    if not base_yaml.is_file():
        return "baseline_config_missing"
    workspace = runs_root(session_dir) / "conc_sweep" / f"conc_sweep_{utc_now_compact()}"
    target = BenchmarkTarget(
        model_path=resolve_session_model_path(state_model_path=state.model_path) or None,
        gpu_type=state.gpu_type.strip().lower() or None,
        benchmark_script=state.accepted_baseline_script(),
    )
    try:
        materialized = prepare_grid_benchmark_config(
            base_yaml,
            workspace,
            target=target,
            options=MaterializeOptions(out_name="conc_sweep_base.with_envs.yaml", grading=state.grading),
        )
    except FrameworkScriptMismatchError:
        log.warning("conc_sweep: the baseline config does not match the session's framework", exc_info=True)
        return "framework_script_mismatch"
    return _Setup(workspace, materialized, target)


def _graded_on(state: SharedState) -> tuple[str, float | None]:
    """The axis the comparison is drawn on and the noise band its guard reads, as the session graded."""
    on_intvty, noise_pct = resolved_grading(state)
    return (GRADED_INTVTY if on_intvty else GRADED_OUTPUT), noise_pct


@dataclass
class _Sweep:
    """One sweep's inputs and the points both arms have measured so far, by arm and concurrency."""

    state: SharedState
    session_dir: Path
    setup: _Setup
    points: SweepPoints
    optimized: Arm
    frozen: BaseLaunch | None = None
    estimated_sec: float = 0.0
    started_at: float = field(default_factory=time.time)
    measured: dict[str, dict[int, dict[str, Any]]] = field(default_factory=lambda: {"baseline": {}, "optimized": {}})
    cancel_reason: str = ""

    @property
    def arms(self) -> tuple[Arm, Arm]:
        return BASELINE, self.optimized

    def pending(self, arm: Arm) -> list[int]:
        done = self.measured[arm.name]
        order = ladder(self.points.concs, self.state.conc)
        return [conc for conc in order if done.get(conc, {}).get("status") != _SUCCEEDED]

    def observe_cancel(self, result: VariantResult | None = None) -> bool:
        """Latch a cancel of the sweep's scope, or of a round it reaped; return whether the sweep is cancelled."""
        scope = current_cancel_scope()
        if not self.cancel_reason and scope is not None and scope.cancelled:
            self.cancel_reason = scope.reason
        if not self.cancel_reason and result is not None and result.error_class == ORCHESTRATOR_CANCELLED_CLASS:
            self.cancel_reason = ORCHESTRATOR_CANCELLED_CLASS
        return bool(self.cancel_reason)

    def layer(self, arm: Arm, conc: int) -> _Layer:
        """What ``arm``'s point at ``conc`` runs over the baseline: the source's launch there, then ``arm``'s change."""
        official = _conc_layer(self.frozen, conc)
        if arm.controls.get("args_mode") == "replace":
            return _replacing(arm, self.frozen, conc, official)
        return _stacked(official, _appended(arm, self.frozen))

    def variant(self, arm: Arm, conc: int) -> GridVariant:
        state = self.state
        layer = self.layer(arm, conc)
        envs = {
            **layer.envs,
            "CONC": str(conc),
            "ISL": str(state.isl),
            "OSL": str(state.osl),
            "NUM_PROMPTS": str(conc * num_prompts_per_conc(state.isl, state.osl)),
            "RUN_EVAL": "false",
        }
        variant = GridVariant(
            name=f"{arm.name}_conc{conc}",
            extra_server_args=layer.args,
            extra_envs=envs,
            note=f"arm={arm.name} conc={conc}",
            remove_args=list(layer.remove_args) or None,
            unset_envs=list(layer.unset_envs) or None,
            args_mode="replace" if layer.replace else "append",
        )
        variant.overlay_pythonpath = arm.overlay
        return variant

    def record(self, arm: Arm, result: VariantResult) -> None:
        point = point_from_variant(result, arm=arm.name)
        self.measured[arm.name][point["conc"]] = point
        write_report(self.payload(), reports_dir(self.session_dir))

    def payload(self, *, final: bool = False) -> dict[str, Any]:
        state = self.state
        sections = {
            arm.name: {
                "extra_server_args": arm.args,
                "extra_envs": arm.envs,
                "overlay": arm.overlay,
                "controls": dict(arm.controls),
                "points": [self.measured[arm.name][conc] for conc in sorted(self.measured[arm.name])],
            }
            for arm in self.arms
        }
        metric_key, noise_pct = _graded_on(state)
        comparison, summary = conc_pair_comparison(
            sections["baseline"]["points"],
            sections["optimized"]["points"],
            metric_key=metric_key,
            guard_noise_pct=noise_pct,
        )
        return {
            **_header(state, self.session_dir),
            "status": self._status(summary) if final else "in_progress",
            "concs": list(self.points.concs),
            "concs_source": self.points.source,
            "baseline": sections["baseline"],
            "optimized": sections["optimized"],
            "comparison": comparison,
            "summary": summary,
            "workspace": self.setup.workspace.as_posix(),
            "estimated_sec": round(self.estimated_sec),
            "elapsed_sec": round(time.time() - self.started_at, 2),
            "cancel_reason": self.cancel_reason,
        }

    def _status(self, summary: Mapping[str, Any]) -> str:
        if self.cancel_reason:
            return "cancelled"
        if not any(self.pending(arm) for arm in self.arms):
            return _SUCCEEDED
        return "partial" if summary["successful_pairs"] else "failed"


def _header(state: SharedState, session_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": state.session_id or session_dir.name,
        "benchmark_mode": state.benchmark_mode,
        "framework": state.framework,
        "precision": state.precision,
        "gpu_type": state.gpu_type,
        "gpu_count": state.tp,
        "isl": state.isl,
        "osl": state.osl,
        "tp": state.tp,
        "ep": state.ep,
        **run_facts(session_dir),
    }


def _skipped(state: SharedState, session_dir: Path, reason: str, **detail: Any) -> dict[str, Any]:
    """The report of a sweep that measured nothing, written where a sweep's report goes."""
    payload = {
        **_header(state, session_dir),
        "status": "skipped",
        "skip_reason": reason,
        **detail,
        "baseline": {"points": []},
        "optimized": {"points": []},
        "summary": {},
    }
    write_report(payload, reports_dir(session_dir))
    log.info("conc_sweep: skipped (%s)", reason)
    return payload


def _resume(sweep: _Sweep) -> None:
    """Keep the points an earlier run of the same sweep measured on the same launch of each arm, so only the rest
    are run."""
    earlier = take_earlier_summary(reports_dir(sweep.session_dir))
    if not earlier:
        return
    for arm in sweep.arms:
        section = earlier[arm.name]
        launch = (
            section.get("extra_server_args"),
            section.get("extra_envs"),
            section.get("overlay"),
            section.get("controls"),
        )
        if launch != (arm.args, arm.envs, arm.overlay, dict(arm.controls)):
            continue
        for point in section["points"]:
            if point["status"] == _SUCCEEDED and point["conc"] in sweep.points.concs:
                sweep.measured[arm.name][point["conc"]] = point


def _log_plan(sweep: _Sweep, per_point_sec: float) -> None:
    rungs = sum(len(sweep.pending(arm)) for arm in sweep.arms)
    log.info(
        "conc_sweep: plan source=%s concs=%s arms=baseline,optimized serial pending_points=%d "
        "estimated=%.1fh (each point priced as the session's measured baseline round of %.0fs)",
        sweep.points.source,
        list(sweep.points.concs),
        rungs,
        sweep.estimated_sec / 3600.0,
        per_point_sec,
    )


async def _run_rung(sweep: _Sweep, arm: Arm, conc: int, launch: GridLaunch) -> VariantResult:
    variant = sweep.variant(arm, conc)
    try:
        (result,) = await run_grid(
            grid=[variant],
            output_root=sweep.setup.workspace,
            recipe=GridRecipe(base_yaml_path=sweep.setup.base_yaml_path, target=sweep.setup.target),
            launch=launch,
        )
    except Exception as exc:  # isolation boundary: a round that raised is its point's failure
        log.warning("conc_sweep: %s raised %r", variant.name, exc, exc_info=True)
        result = VariantResult(
            name=variant.name,
            extra_server_args=variant.extra_server_args,
            extra_envs=dict(variant.extra_envs),
            status="failed",
            error=str(exc),
            error_class=type(exc).__name__,
        )
    return result


async def _run_points(sweep: _Sweep) -> None:
    """Run both arms' pending points: arm by arm on held servers, else each point on its own server, a concurrency's
    two arms one after the other, so a stop leaves pairs measured rather than an arm."""
    lifecycle = resolve_lifecycle_params(sweep.setup.base_yaml_path)
    if not lifecycle.get("eligible") or agentx_replay_session(sweep.state):
        await _run_cold(sweep)
        return
    for arm in sweep.arms:
        if sweep.observe_cancel():
            return
        await _run_held_arm(sweep, arm, lifecycle)


async def _run_cold(sweep: _Sweep) -> None:
    order = ladder(sweep.points.concs, sweep.state.conc)
    queue = [(conc, arm) for conc in order for arm in sweep.arms if conc in sweep.pending(arm)]
    for conc, arm in queue:
        if sweep.observe_cancel():
            return
        result = await _run_rung(sweep, arm, conc, GridLaunch())
        sweep.record(arm, result)
        if sweep.observe_cancel(result):
            return


async def _run_held_arm(sweep: _Sweep, arm: Arm, lifecycle: Mapping[str, Any]) -> None:
    """Run ``arm``'s pending points, a run of points that launch alike on one held server: the points of one launch
    run together, the launch of the first pending point first, each in ladder order."""
    ladder_order = sweep.pending(arm)
    by_conc = {conc: sweep.layer(arm, conc) for conc in ladder_order}
    launches: list[_Layer] = []
    for conc in ladder_order:
        if by_conc[conc] not in launches:
            launches.append(by_conc[conc])
    concs = sorted(ladder_order, key=lambda conc: launches.index(by_conc[conc]))
    pid_dir = sweep.setup.workspace / f"server_{arm.name}"
    pid_dir.mkdir(parents=True, exist_ok=True)
    port = int(lifecycle.get("port") or 8888)
    booted = False
    layers = [by_conc[conc] for conc in concs]
    try:
        for idx, conc in enumerate(concs):
            if sweep.observe_cancel():
                return
            if booted and layers[idx] != layers[idx - 1]:
                teardown_variant_server(pid_dir)
                booted = False
            last = idx == len(concs) - 1 or layers[idx + 1] != layers[idx]
            lifecycle_settings = {"cleanup": booted and last, "pid_dir": str(pid_dir), "port": port}
            launch = GridLaunch(
                server_lifecycle=lifecycle_settings, server_already_ready=booted, warmup_before_measure=False
            )
            result = await _run_rung(sweep, arm, conc, launch)
            if not booted:
                booted = result.status == _SUCCEEDED
                if not booted:
                    teardown_variant_server(pid_dir)
            sweep.record(arm, result)
            if sweep.observe_cancel(result):
                return
    finally:
        if survivors := teardown_variant_server(pid_dir):
            log.error("conc_sweep: arm=%s pids %s outlived SIGKILL; %s keeps them", arm.name, survivors, pid_dir)


async def run_conc_sweep(session_dir: Path, *, concs: Sequence[int] = ()) -> dict[str, Any]:
    """Sweep the finished session in ``session_dir`` over ``concs`` when given, and return the report it wrote to
    ``reports/``."""
    state = SharedState.load_or_init(session_dir)
    reason = decline_reason(state)
    if reason is not None:
        return _skipped(state, session_dir, reason)
    optimized = optimized_arm(state)
    if optimized is None or state.cumulative_gain_validated <= 0:
        return _skipped(state, session_dir, "no_optimization")
    if optimized.overlay and not (is_safe_path_entry(optimized.overlay) and overlay_is_loadable(optimized.overlay)):
        return _skipped(state, session_dir, "optimized_overlay_unavailable")
    frozen = frozen_base(session_dir)
    base = None if frozen is None else frozen.launch
    if (missing := launches_missing(base)) is not None:
        return _skipped(state, session_dir, missing)
    points = sweep_points(base, state.conc, concs or state.conc_sweep_concs)
    if points is None:
        return _skipped(state, session_dir, "no_sweep_points")
    setup = _prepare(state, session_dir)
    if isinstance(setup, str):
        return _skipped(state, session_dir, setup)
    per_point_sec = state.baseline_runtime_sec
    sweep = _Sweep(state, session_dir, setup, points, optimized, base)
    _resume(sweep)
    sweep.estimated_sec = per_point_sec * sum(len(sweep.pending(arm)) for arm in sweep.arms)
    _log_plan(sweep, per_point_sec)
    await _run_points(sweep)
    payload = sweep.payload(final=True)
    ceiling = build_roofline_ceiling(
        state,
        ArmCurves(payload["baseline"]["points"], payload["optimized"]["points"]),
        concs=list(points.concs),
        isl=state.isl,
        osl=state.osl,
    )
    if ceiling is not None:
        payload["roofline_ceiling"] = ceiling
    write_report(payload, reports_dir(session_dir))
    curve = ConcCurveOptions(state.model_name, state.gpu_type, state.tp, state.isl, state.osl)
    publish_curve(payload, session_dir, curve)
    log.info(
        "conc_sweep: done status=%s successful_pairs=%s best_speedup=%s",
        payload["status"],
        payload["summary"].get("successful_pairs"),
        payload["summary"].get("best_speedup"),
    )
    return payload


__all__ = ["BASELINE", "Arm", "decline_reason", "optimized_arm", "run_conc_sweep"]
