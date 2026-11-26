# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ExploreExecutor."""

from __future__ import annotations
from hyperloom.orchestrator.actions.executors.workload._workload_envs import prepare_grid_benchmark_config
from hyperloom.common.benchmark_config import BenchmarkPoint
import asyncio
import functools
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from hyperloom_common.coerce import to_float, to_int, to_str_list
from hyperloom.common.gain_math import accuracy_passed, gain_pct, graded_gate_row
from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.common.perf_metric import (
    GRADED_DURATION,
    GRADED_ERROR_RATE,
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    GRADED_INTVTY_P50_CHECKPOINT,
    GRADED_OUTPUT,
    VERDICT_REVERT,
    GradingAnchor,
    agentx_active,
    constraint_fields,
    graded_axes_of,
    perf_snapshot_from_mapping,
    resolve_grading_anchor_perf,
    stamp_output_per_gpu,
)
from hyperloom_common.timeutil import now_iso
from hyperloom.inference_optimizer.grading import resolved_grading
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from ...state.explore_ledger import land_verdict, ledger_view
from ...state.failure_evidence import (
    FAILURE_STAGE_DECISION,
    FAILURE_STAGE_WARMUP,
    tail_excerpt,
)
from ...state.shared_state import (
    first_positive_tput,
    resolve_anchor_with_drift,
    resolve_graded_comparison,
    stack_base_params,
)
from ..stop_attribution import StoppedByTheRun, stopped_by_the_run_class, variant_budget_stop
from ._accuracy_gate import AccuracyReference, parse_eval_results
from hyperloom.inference_optimizer.canonical_fingerprint import workload_signature
from ._proposal_identity import (
    BaseControls,
    controls_of,
    effective_fingerprint,
    keeps_effective_args,
)
from .explore_prefilter.measured_variants import GAIN_BELOW_THRESHOLD, MeasuredOn, measured_verdict, stack_of_measured
from .grid._grid_base import DEFAULT_KEEP_THRESHOLD_PCT, TS_FAILED, TS_SKIPPED_DEDUP
from .grid._grid_base import _safe, variant_config_of
from .grid._grid_base import apply_base_env
from .workload._agentx_checkpoint import round_checkpoint
from ._subprocess_kill import AGENTX_EARLY_STOP_ERROR_CLASS
from .benchmark_result import double_run_requested
from .benchmark_run.accuracy import accuracy_round_due, record_accuracy_round, run_candidate_accuracy_round
from .explore_prefilter._prefilter import prefilter_round, record_decision_round
from .explore_prefilter._remeasure import STACK_REVALIDATE_SOURCE, is_stack_revalidation
from .explore_prefilter.grid_payload import (
    framework_lever_attributions,
    grid_variants_from_payload,
    resolved_grid_payload,
)
from .explore_prefilter.launch_filters import (
    filter_baseline_noop_variants,
    filter_operator_pinned_envs,
    filter_synthetic_drafts,
    launch_forces_acceptance,
    launch_identity,
    observed_launch_from_state,
)
from .grid._explore_variants import (
    LOSER_FIELDS,
    REJECTED_FIELDS,
    decision_fields,
    outcome_row,
    staged_variant,
    variant_facts,
    variant_fingerprint,
    variant_with_eval_off,
)
from .grid._grid_inputs import SessionDirField, session_grid_bounds
from .grid._grid_inputs import sanitize_result_dir, sanitize_script_name
from .grid._grid_runner import (
    GridLaunch,
    GridRecipe,
    GridStop,
    GridVariant,
    run_grid,
    teardown_variant_server,
)
from .grid._grid_variant_filter import (
    ServingStack,
    apply_aiter_moe_pin_filter,
    apply_compatibility_filter,
    apply_multi_node_invalid_variants,
    apply_user_skip_list,
    reorder_grid_for_multi_node,
    resolve_skip_spec,
)
from hyperloom.inference_optimizer.framework_registry import is_scriptable, server_args_env_name
from hyperloom.inference_optimizer.grid_server_args import compose_server_args
from .explore_prefilter.spec_decode_golden import session_golden_lookup

from .workload._launch_restatement import observed_launch_after
from .serving._server_argv import ConfigUnreadable, config_point
from .serving._server_lifecycle import resolve_lifecycle_params
from .workload._workload_envs import (
    BenchmarkTarget,
    FrameworkScriptMismatchError,
    LaunchLayer,
    MaterializeOptions,
    default_baseline_config,
)


log = logging.getLogger(__name__)


_now_iso = functools.partial(now_iso, "auto")

# ``skipped_dup`` reason for a variant whose launch the session already measured on the same stack.
ALREADY_MEASURED: str = "already_measured"
# ... for one that starts the stack's own server, or a server this round already launched.
BASELINE_NOOP: str = "baseline_noop"
ROUND_DUP: str = "round_dup"

#: The provenance of a grid row that names none: the model proposed it directly.
_DIRECT_PROVENANCE = "llm_direct"


class _RoundRefused(Exception):
    """A round that cannot start, carrying the action result that says why."""

    def __init__(self, error_class: str, error: str, **extra: Any) -> None:
        super().__init__(error)
        self.result: dict[str, Any] = {"status": "failed", "error_class": error_class, "error": error, **extra}


def _materialized_round_config(
    params: dict[str, Any], output_root: Path, target: BenchmarkTarget, grading: Any
) -> Path:
    # Re-materialize the baseline config so variant YAMLs honour the operator's actual workload.
    #
    # CONC / ISL / OSL / TP / MAX_MODEL_LEN / PRECISION all reach the variants through it.
    config_path = Path(params["config_path"])
    if not config_path.exists():
        raise _RoundRefused("missing_config", f"config not found: {config_path}")
    try:
        return prepare_grid_benchmark_config(
            config_path,
            output_root,
            target=BenchmarkTarget(
                model_path=target.model_path or None,
                gpu_type=target.gpu_type or None,
                benchmark_script=target.benchmark_script or None,
            ),
            options=MaterializeOptions(out_name="explore_base.with_envs.yaml", grading=grading),
        )
    except FrameworkScriptMismatchError as exc:
        raise _RoundRefused("framework_script_mismatch", str(exc)) from exc


def _round_point(config_path: Path) -> BenchmarkPoint:
    try:
        return config_point(config_path)
    except ConfigUnreadable as exc:
        raise _RoundRefused("unreadable_config", str(exc)) from exc


def _anchor_the_params(params: dict[str, Any], ss: Any) -> None:
    # Refresh the anchor and the stack it was measured on, which a KEEP landing while this task queued invalidates.
    #
    # Params snapshot the two together, so they are refreshed as a pair. A revalidation
    # reproduces the saved stack, so it never re-anchors.
    snapshot_tput = to_float(params.get("base_tput"), 0.0) or 0.0
    anchor, drifted = (
        (snapshot_tput, False)
        if params.get("source") == STACK_REVALIDATE_SOURCE
        else resolve_anchor_with_drift(snapshot_tput, ss)
    )
    if anchor <= snapshot_tput:
        return
    if drifted:
        log.warning("explore: anchor drift %.1f -> %.1f; re-reading base args", snapshot_tput, anchor)
    params["base_tput"] = anchor
    cb = getattr(ss, "current_best", None)
    # Only current_best carries args; a baseline_tput anchor leaves the params stack (seeded from the baseline
    # record) authoritative.
    if first_positive_tput(cb) > 0:
        params.update(stack_base_params(cb))


def _sanitized_overrides(params: dict[str, Any]) -> tuple[str | None, str | None]:
    try:
        return sanitize_script_name(params.get("benchmark_script")), sanitize_result_dir(params.get("result_dir"))
    except ValueError as exc:
        raise _RoundRefused("bad_param", str(exc)) from exc


class _ExploreRound:
    """One explore round: the stack it accumulates, the variants it measures, and the ledgers it writes."""

    projection: dict[str, Any] | None = None
    prefilter_dropped = 0

    def __init__(self, ctx: Any, params: dict[str, Any], output_root: Path) -> None:
        extra = getattr(ctx, "extra", None) or {}
        self.ctx = ctx
        self.params = params
        self.output_root = output_root
        self.ss = extra.get("shared_state")
        self.phase_deadline = extra.get("phase_deadline")
        script, self.result_dir = _sanitized_overrides(params)
        self.target = BenchmarkTarget(
            model_path=resolve_session_model_path(
                params=params,
                state_model_path=str(getattr(self.ss, "model_path", "") or ""),
            ),
            gpu_type=str(params.get("gpu_type") or "").strip().lower()
            or os.environ.get("GPU_TYPE", "").strip().lower(),
            benchmark_script=script,
        )
        grading = getattr(self.ss, "grading", None)
        self.config_path = _materialized_round_config(params, output_root, self.target, grading)
        _anchor_the_params(params, self.ss)
        self.point = _round_point(self.config_path)
        self.bench = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))["benchmark"]
        self.framework = self.point.framework.lower()
        self.yaml_envs = dict(self.point.envs)
        self.base_args = str(params.get("base_extra_args") or "").strip()
        self.base_envs = dict(params.get("base_extra_envs") or {})
        self.base_remove_args = to_str_list(params.get("base_remove_args"))
        self.base_unset_envs = to_str_list(params.get("base_unset_envs"))
        self.args_mode = str(params.get("base_args_mode") or "append").strip().lower()
        inherited = str(self.yaml_envs.get(server_args_env_name(self.framework)) or "").strip()
        self.inherited_args = "" if self.args_mode == "replace" else inherited
        # ``args`` / ``envs`` carry the running accumulation; after a KEEP they extend with the KEEP'd variant.
        self.args = self.base_args
        self.envs = dict(self.base_envs)
        self.remove_args = list(dict.fromkeys(self.base_remove_args))
        self.unset_envs = list(dict.fromkeys(self.base_unset_envs))
        self.base_tput = to_float(params.get("base_tput"), 0.0) or 0.0
        self.running_tput = self.base_tput
        self.running_perf: Any = None
        self.keep_threshold_pct = float(params["keep_threshold_pct"])
        self.accuracy_reference = AccuracyReference.of_session(self.ss, params)
        # The synthetic benchmark's repetition loops would grade a draft at an acceptance real text does not reach.
        self.refuse_synthetic_drafts = not agentx_active(
            benchmark_mode=getattr(self.ss, "benchmark_mode", "")
        ) and not is_stack_revalidation(params)
        self.ws_sig = workload_signature()
        # Seeded by the Coordinator from the durable cursor: this executor holds no history of its own, so it
        # cannot count the rounds before this one.
        self.round_id = f"explore-{(to_int(params.get('explore_search_cursor'), 0) or 0) + 1:03d}"
        self.observed = observed_launch_from_state(self.ss)
        self.opening_stack = self.launched_stack()
        self._init_round_ledgers()

    def _init_round_ledgers(self) -> None:
        """Start the round's variant lists, verdict ledgers and run-stop record empty."""
        self.lever_payload: list[dict[str, Any]] = []
        self.grid: list[GridVariant] = []
        self.runnable: list[GridVariant] = []
        self.skipped: list[dict[str, Any]] = []
        self.forcing: set[str] = set()  # variants whose launch simulates acceptance: their eval runs on its own
        self.winners: list[dict[str, Any]] = []
        self.losers: list[dict[str, Any]] = []
        self.rejected: list[dict[str, Any]] = []
        self.winners_history: list[dict[str, Any]] = []
        self.tested: dict[str, dict[str, Any]] = {}
        # Servers this round launched, and the variants that reached the measured check and were answered by the
        # stack or the ledger instead (a restatement of the stack, or a verdict the session already holds).
        self.launched: dict[str, str] = {}
        self.judged_without_measuring = 0
        # Set when the loop stops because the run stopped it -- the budget ran out, or the orchestrator cancelled
        # the action -- so the round can say so instead of reporting a bare, unattributed failure: a variant that
        # never ran is not a variant that failed.
        self.run_stop: StoppedByTheRun | None = None
        self.run_stop_detail = ""
        self.untested = 0

    async def run(self, session_dir: Path | None = None) -> dict[str, Any]:
        self.session_dir = session_dir
        self._resolve_grid()
        self._shape_runnable()
        self._open_schedule()
        round_ctx = await prefilter_round(self, session_dir)
        for idx, gv in enumerate(self.runnable):
            if await self._refused(gv, idx):
                if self.run_stop is not None:
                    break
                continue
            slot = self.output_root / f"v{idx:02d}_{_safe(gv.name)}"
            slot.mkdir(parents=True, exist_ok=True)
            run_gv = staged_variant(gv, self.envs, self.unset_envs)
            try:
                measurement = await self._measure(gv, run_gv, idx, slot)
                if self.run_stop is not None:
                    break
                if measurement is not None:
                    row = await self._settle(gv, run_gv, measurement, slot)
                    await record_decision_round(round_ctx, gv, measurement, row)
            finally:
                # The warmup and decision rounds share this slot's server; reap it on every exit path.
                if self.lifecycle_eligible:
                    self._reap_variant_server(slot)
        return self._result()

    def _reap_variant_server(self, slot: Path) -> None:
        if survivors := teardown_variant_server(slot):
            log.error("explore: pids %s outlived SIGKILL; %s keeps their record", survivors, slot)

    # ----- the stack the round accumulates ---------------------------------

    def measured_against(self) -> dict[str, Any]:
        """The stack this variant launched on top of, as it stands now.

        Read at each write rather than captured once: every KEEP advances the
        stack, so a variant later in the round was measured against a different
        one than the round opened with. Carried verbatim for the same reason
        ``base_tput`` is -- a consumer that reconstructs the stack from the
        session's current config gets whichever one is current, not the one
        this variant was judged on.
        """
        return {
            "throughput": self.running_tput,
            "accuracy": self.accuracy_reference.score or None,
            "extra_server_args": self.args,
            "extra_envs": dict(self.envs),
            "remove_args": list(self.remove_args),
            "unset_envs": list(self.unset_envs),
            "args_mode": self.args_mode,
        }

    def _composed_args(self, args: str, remove_args: list[str]) -> str:
        """``args`` as the run launches them: the recipe's inherited args folded in, ``remove_args`` stripped."""
        return compose_server_args(
            inherited_args=self.inherited_args, base_extra_args=args, remove_args=remove_args, args_mode=self.args_mode
        )

    def launched_stack(self) -> dict[str, Any]:
        """The stack as it launches: its args with the recipe's inherited ones and the removals applied.

        A KEEP can switch the stack to ``replace`` mode, which carries those inherited args in its own, so only
        the composed args say which settings a KEEP changed.
        """
        stack = stack_of_measured(self.measured_against())
        stack["extra_server_args"] = self._composed_args(self.args, self.remove_args)
        return stack

    def _recipe(self) -> GridRecipe:
        return GridRecipe(
            base_yaml_path=self.config_path,
            base=LaunchLayer(
                extra_server_args=self.args,
                extra_envs=dict(self.envs),
                remove_args=list(self.remove_args),
                unset_envs=list(self.unset_envs),
                args_mode=self.args_mode,
            ),
            target=self.target,
        )

    def _run_remove_args(self, gv: GridVariant) -> list[str]:
        own = to_str_list(getattr(gv, "remove_args", []))
        return own if self.args_mode == "replace" else list(dict.fromkeys(self.remove_args + own))

    # ----- the grid --------------------------------------------------------

    def _resolve_grid(self) -> None:
        payload, self.lever_payload = resolved_grid_payload(
            self.params,
            shared_state=self.ss,
            framework=self.framework,
            yaml_envs=self.yaml_envs,
        )
        if not payload:
            raise _RoundRefused(
                "empty_grid",
                "explore: params.grid must be a non-empty list of variant "
                "dicts. The Orchestration prompt "
                "should fill this from specialist proposals / default_grid.",
                workspace=self.output_root.as_posix(),
            )
        grid = grid_variants_from_payload(payload)
        if self.framework == "custom":
            grid, pinned = filter_operator_pinned_envs(grid, self.yaml_envs)
            for name, reason in pinned:
                log.warning("explore: dropping variant %s (%s)", name, reason)
        if not grid:
            raise _RoundRefused(
                "empty_grid",
                "explore: params.grid contains no valid variants",
                workspace=self.output_root.as_posix(),
            )
        # Every variant carries its fingerprint, so the result loop needn't recompute it. In-round duplicates are
        # judged at their turn, against what this round has actually launched: a twin that was only skipped (a
        # restatement of the stack) has not claimed the launch.
        for gv in grid:
            gv.canonical_fp = effective_fingerprint(  # type: ignore[attr-defined]
                gv.extra_server_args,
                gv.extra_envs,
                controls=controls_of(vars(gv)),
                base=BaseControls(self.base_remove_args, self.base_unset_envs, self.args_mode),
            )
        self.grid = grid

    def _shape_runnable(self) -> None:
        runnable, invalid = apply_multi_node_invalid_variants(list(self.grid))
        # Honour an operator-pinned SGLANG_USE_AITER=0: drop variants that would re-enable the (hang-prone) aiter
        # MoE runner.
        runnable, aiter_pinned = apply_aiter_moe_pin_filter(runnable)
        # xDiT do-not-set list, plus flags the model class does not support.
        runnable, incompatible = apply_compatibility_filter(
            runnable,
            serving=ServingStack.of_recipe(
                self.bench,
                model_path=self.target.model_path,
                gpu_type=self.target.gpu_type or str(getattr(self.ss, "gpu_type", "") or ""),
                stack_fingerprint=getattr(self.ss, "stack_fingerprint_meta", None),
            ),
            base_server_args=self._composed_args(self.base_args, self.base_remove_args),
            # The recipe's environment under the accepted edits, layered as the launch layers it.
            base_envs=apply_base_env(dict(self.yaml_envs), self.envs, self.unset_envs),
        )
        # Operator-supplied --skip-variants patterns.
        runnable, skipped = apply_user_skip_list(runnable, skip_spec=resolve_skip_spec(self.params))
        for dropped in (*invalid, *aiter_pinned, *incompatible, *skipped):
            self.skipped.append(
                {
                    "name": dropped.get("name", ""),
                    "reason": dropped.get("source", "grid_invalid"),
                    "detail": dropped.get("reason", ""),
                }
            )
        self.runnable = reorder_grid_for_multi_node(runnable)
        self._drop_leading_noops()

    def _drop_leading_noops(self) -> None:
        # Drop the variants ahead of every other one that only restate the stack's observed launch.
        #
        # They change nothing while the stack stays as observed, and can only launch on it, so they go before any
        # variant runs; any other is judged at its turn, against the stack a KEEP has moved.
        before = len(self.runnable)
        if self.runnable and not is_stack_revalidation(self.params):
            kept, dropped = filter_baseline_noop_variants(
                self.runnable,
                observed=self.observed,
                recipe=GridRecipe(
                    base_yaml_path=self.config_path,
                    base=LaunchLayer(
                        extra_server_args=self.base_args,
                        extra_envs=self.base_envs,
                        remove_args=self.base_remove_args,
                        unset_envs=self.base_unset_envs,
                        args_mode=self.args_mode,
                    ),
                    target=self.target,
                ),
            )
            kept_ids = {id(gv) for gv in kept}
            leading = next((i for i, gv in enumerate(self.runnable) if id(gv) in kept_ids), before)
            for gv, (_, reason) in zip(self.runnable[:leading], dropped[:leading]):
                self._decided_without_measuring(gv, BASELINE_NOOP, detail=reason)
            self.runnable = self.runnable[leading:]
        log.info(
            "explore grid: payload=%d → runnable=%d (leading baseline_noop=%d)",
            len(self.grid),
            len(self.runnable),
            before - len(self.runnable),
        )

    def _open_schedule(self) -> None:
        # Single-node server_lifecycle eligibility (multi-node / non-builtin script / profiler-on falls back to a
        # cold decision round instead of one that re-attaches to the warmup's server).
        lifecycle = resolve_lifecycle_params(self.config_path)
        self.lifecycle_eligible = bool(lifecycle.get("eligible"))
        self.lifecycle_port = to_int(lifecycle.get("port"), 0) or 0
        self.use_warm_decision = self.lifecycle_eligible and double_run_requested(self.params, self.config_path)
        # Stop testing further variants once the session wall-clock budget runs out; untested variants stay out of
        # the ledger so a resume can retry them.
        self.session_deadline_sec, session_expected_sec = session_grid_bounds(self.ss)
        # per-variant overtime kill -- anchored on baseline wall-clock, or on the measured WARM duration
        # (client-only) when this round reuses a server.
        warm_sec = to_float(self.params.get("baseline_warm_runtime_sec"), 0.0) or 0.0
        cold_sec = to_float(self.params.get("baseline_runtime_sec"), 0.0) or 0.0
        decision_sec = warm_sec if (self.use_warm_decision and warm_sec > 0) else cold_sec
        self.warmup_expected_sec = (cold_sec if cold_sec > 0 else None) or session_expected_sec
        self.decision_expected_sec = (decision_sec if decision_sec > 0 else None) or session_expected_sec
        # Round-local grading anchor. Variants stack within a round, so a KEEP advances the figure the next
        # variant is graded against; the session anchor would grade every variant against the round's opening.
        self.grade_on_intvty, _ = resolved_grading(self.ss)
        self.running_perf, anchor_reason = resolve_grading_anchor_perf(self.ss) if self.grade_on_intvty else (None, "")
        if self.grade_on_intvty and self.running_perf is None:
            # AgentX grading needs both axes on the round anchor. Without them every variant fails closed below;
            # output figures are diagnostic only and must not become throughput KEEPs.
            log.info("explore: AgentX grading unavailable for this round (%s)", anchor_reason)

    # ----- one variant -----------------------------------------------------

    async def _refused(self, gv: GridVariant, idx: int) -> bool:
        if twin := self.launched.get(launch_identity(self._recipe(), gv)):
            self._decided_without_measuring(gv, ROUND_DUP, duplicate_of=twin)
            return True
        if restated := self._restates_stack(gv):
            self._decided_without_measuring(gv, BASELINE_NOOP, detail=restated)
            self.judged_without_measuring += 1
            return True
        if self._already_measured(gv):
            self.judged_without_measuring += 1
            return True
        return self._out_of_budget(idx) or await self._refuses_synthetic_draft(gv)

    def _decided_without_measuring(
        self, gv: GridVariant, reason: str, *, detail: str = "", duplicate_of: str = ""
    ) -> None:
        """Record ``gv`` as answered by the stack or by a twin this round launched, not measured.

        Its ledger row marks the proposal decided, so the untested-proposal queue moves past it; a change the ledger
        already holds keeps its row, since a measurement outranks a skip.
        """
        fingerprint = variant_fingerprint(gv)
        log.info("explore: skipping %s variant %s (%s)", reason, gv.name, detail or duplicate_of)
        self.skipped.append(
            {"name": gv.name, "fingerprint": fingerprint, "reason": reason, "detail": detail or duplicate_of}
        )
        known = (getattr(self.ss, "explore_search", None) or {}).get("tested") or {}
        if not fingerprint or fingerprint in self.tested or fingerprint in known:
            return
        self.tested[fingerprint] = {
            **variant_facts(gv),
            "outcome": TS_SKIPPED_DEDUP,
            "reason": reason,
            "duplicate_of": duplicate_of,
            "round_id": self.round_id,
            "ts": _now_iso(),
            "provenance": getattr(gv, "provenance", _DIRECT_PROVENANCE),
            "workload_signature": self.ws_sig,
        }

    def _restates_stack(self, gv: GridVariant) -> str:
        """Why ``gv`` only restates the launch the stack runs as it stands now, or ``""`` when it changes it.

        The observed launch is what the server ran when the round opened; a KEEP since then has made the observed
        value of whatever it changed stale, and ``observed_launch_after`` leaves that out. A stack revalidation
        re-measures the stack on purpose, so it never restates it.
        """
        if is_stack_revalidation(self.params):
            return ""
        _, restated = filter_baseline_noop_variants(
            [gv],
            observed=observed_launch_after(*self.observed, self.opening_stack, self.launched_stack(), self.framework),
            recipe=self._recipe(),
        )
        return restated[0][1] if restated else ""

    def _already_measured(self, gv: GridVariant) -> bool:
        """Whether the session ledger already holds a verdict for ``gv`` on the stack it would launch on.

        A KEEP earlier in the round advances that stack, so the check runs per variant rather than once for the
        round. A stack revalidation exists to re-measure, and a kernel overlay changes the launch without changing
        the fingerprint, so neither is ever skipped.
        """
        if self.ss is None or is_stack_revalidation(self.params) or getattr(gv, "overlay_pythonpath", ""):
            return False
        fingerprint = variant_fingerprint(gv)
        prior = measured_verdict(
            getattr(self.ss, "explore_search", None),
            fingerprint,
            variant={"extra_args": gv.extra_server_args, "extra_envs": gv.extra_envs, **controls_of(vars(gv))},
            now=MeasuredOn(stack_of_measured(self.measured_against()), self.ws_sig, self.keep_threshold_pct),
        )
        if prior is None:
            return False
        detail = (
            f"measured as {prior.get('name') or fingerprint} in {prior.get('round_id') or 'an earlier round'} on "
            f"this stack: {prior.get('outcome')}"
        )
        log.info("explore: skipping already-measured variant %s (%s)", gv.name, detail)
        self.skipped.append({"name": gv.name, "fingerprint": fingerprint, "reason": ALREADY_MEASURED, "detail": detail})
        return True

    def _out_of_budget(self, idx: int) -> bool:
        # Whether the run cannot fit another variant, recording the stop when it cannot.
        #
        # A warm-decision variant pays for both rounds, so admitting it on the decision round alone would let it in
        # and then strand it mid-variant with a discarded warmup and no measurement.
        if self.decision_expected_sec is not None:
            needed = float(self.decision_expected_sec) + (
                float(self.warmup_expected_sec or 0.0) if self.use_warm_decision else 0.0
            )
        else:
            from ._subprocess_kill import resolve_benchmark_timeouts

            needed = resolve_benchmark_timeouts()[1]
        self.run_stop = variant_budget_stop(self.session_deadline_sec, self.phase_deadline, needed)
        if self.run_stop is None:
            return False
        self.run_stop_detail = self.run_stop.never_started
        self.untested = len(self.runnable) - idx
        log.warning(
            "explore: cannot fit another variant (%s, needs %.0fs); stopping after %d/%d variant(s)",
            self.run_stop.error_class,
            needed,
            idx,
            len(self.runnable),
        )
        return True

    async def _refuses_synthetic_draft(self, gv: GridVariant) -> bool:
        # Whether the synthetic benchmark refuses this variant's speculative-decoding draft, recording it.
        # Judged against the stack as it stands now: a KEEP earlier in this round may have changed it.
        if not self.refuse_synthetic_drafts:
            return False
        if launch_forces_acceptance(self._recipe(), gv):
            self.forcing.add(gv.name)
        lookup = session_golden_lookup(self.session_dir, self.framework)
        # The lookup runs the staged interpreter, up to its timeout: off the event loop, the verdict recorded on it.
        _, refused = await asyncio.to_thread(filter_synthetic_drafts, [gv], recipe=self._recipe(), lookup=lookup)
        if not refused:
            return False
        log.warning("explore: refusing variant %s (%s)", gv.name, refused[0][2])
        self.skipped.append({"name": gv.name, "reason": refused[0][1], "detail": refused[0][2]})
        return True

    def _stopped_by_the_run(self, result: Any, gv: GridVariant, idx: int, round_label: str) -> bool:
        """Whether the run stopped this round, and record it if it did."""
        stopped = stopped_by_the_run_class(getattr(result, "error_class", "") if result is not None else "")
        if stopped is None:
            return False
        self.run_stop = stopped
        self.run_stop_detail = stopped.interrupted
        self.untested = len(self.runnable) - idx
        log.warning(
            "explore: the %s round of variant %s was stopped by the run (%s); it and the "
            "%d variant(s) after it stay out of the ledger so a resume can retry them",
            round_label,
            gv.name,
            stopped.error_class,
            self.untested - 1,
        )
        return True

    async def _measure(self, gv: GridVariant, run_gv: GridVariant, idx: int, slot: Path) -> Any:
        run_gv = variant_with_eval_off(run_gv) if gv.name in self.forcing else run_gv
        self.launched[launch_identity(self._recipe(), gv)] = variant_fingerprint(gv)
        # The warmup and decision rounds share this slot as the lifecycle pid_dir so the decision round
        # re-attaches to the server the warmup left hot.
        lifecycle = (
            {"cleanup": False, "pid_dir": str(slot), "port": self.lifecycle_port} if self.lifecycle_eligible else None
        )
        warmup = await self._warmup(gv, run_gv, idx, slot, lifecycle) if self.use_warm_decision else None
        if self.use_warm_decision and warmup is None:
            return None
        # The decision round is warm (it re-attaches to the warmup's hot server, client-only) when a warmup
        # preceded it, and skips eval then: it is timed against a throughput-only anchor, and
        # ``parse_eval_results`` falls back to the score the warmup took. Without a warmup there is nothing to
        # fall back to, so a cold decision round keeps its own eval.
        results = await run_grid(
            grid=[variant_with_eval_off(run_gv) if self.use_warm_decision else run_gv],
            output_root=slot,
            recipe=self._recipe(),
            launch=GridLaunch(
                result_dir=self.result_dir,
                server_lifecycle=lifecycle,
                server_already_ready=self.use_warm_decision,
                ready_server_log=warmup.server_log_path if warmup is not None else None,
            ),
            stop=GridStop(
                session_deadline_sec=self.session_deadline_sec,
                variant_expected_sec=self.decision_expected_sec,
                checkpoint=round_checkpoint(self.running_perf),
            ),
        )
        return None if self._stopped_by_the_run(results[0], gv, idx, "decision") else results[0]

    async def _warmup(self, gv: GridVariant, run_gv: GridVariant, idx: int, slot: Path, lifecycle: Any) -> Any:
        warmup_slot = slot / "warmup_round"
        warmup_slot.mkdir(parents=True, exist_ok=True)
        results = await run_grid(
            grid=[run_gv],
            output_root=warmup_slot,
            recipe=self._recipe(),
            launch=GridLaunch(
                result_dir=self.result_dir,
                server_lifecycle=lifecycle,
                warmup_round=True,
            ),
            stop=GridStop(
                session_deadline_sec=self.session_deadline_sec, variant_expected_sec=self.warmup_expected_sec
            ),
        )
        warmup = results[0] if results else None
        if self._stopped_by_the_run(warmup, gv, idx, "warmup"):
            return None
        if getattr(warmup, "status", "") != "succeeded":
            self._warmup_failed(gv, warmup)
            return None
        return warmup

    def _warmup_failed(self, gv: GridVariant, warmup: Any) -> None:
        log.warning(
            "explore: variant %s warmup round failed (%s); skipping decision round.",
            gv.name,
            (getattr(warmup, "error", "") or "")[-200:] if warmup is not None else "no_result",
        )
        self._record(
            gv,
            {
                **variant_facts(gv),
                "outcome": TS_FAILED,
                "status": getattr(warmup, "status", "failed") if warmup is not None else "failed",
                "tput": None,
                "gain_pct": None,
                "base_tput": self.running_tput,
                "round_id": self.round_id,
                "ts": _now_iso(),
                "provenance": getattr(gv, "provenance", _DIRECT_PROVENANCE),
                "workload_signature": self.ws_sig,
                "framework": self.framework,
                "reason": "warmup_failed",
                "error_class": warmup.error_class if warmup is not None else "",
                "server_log_path": warmup.server_log_path if warmup is not None else None,
                "launch_evidence": dict(warmup.launch_evidence or {}) if warmup is not None else {},
                "launch_evidence_path": warmup.launch_evidence_path if warmup is not None else None,
                "stage": FAILURE_STAGE_WARMUP,
                "error_excerpt": tail_excerpt(warmup.error) if warmup is not None else None,
                "workspace": warmup.workspace if warmup is not None else None,
                "raw_result_path": warmup.raw_result_path if warmup is not None else None,
                # It launched on a stack even though it never reached a gate, so no gates are recorded and
                # nothing validated it.
                "measured_against": self.measured_against(),
            },
        )

    # ----- the verdict -----------------------------------------------------

    async def _settle(self, gv: GridVariant, run_gv: GridVariant, r: Any, slot: Path) -> dict[str, Any]:
        # Rule on this variant's measurement and fold a KEEP onto the round's stack.
        #
        # A variant KEEPs when it clears the graded verdict and the accuracy gate. The axes and the threshold floor
        # belong to resolve_graded_comparison, which every lane shares.
        variant_meas = {
            GRADED_OUTPUT: r.output_throughput,
            "input_throughput": r.input_throughput,
            "total_throughput": r.total_token_throughput,
            GRADED_INTVTY: r.intvty_p90,
            "tpot_p90_ms": r.tpot_p90_ms,
            GRADED_INTVTY_P50: r.intvty_p50,
            GRADED_INTVTY_P50_CHECKPOINT: r.intvty_p50_checkpoint,
            GRADED_DURATION: r.duration_seconds,
            GRADED_ERROR_RATE: r.request_error_rate,
            **constraint_fields(r),
        }
        stamp_output_per_gpu(variant_meas, getattr(self.ss, "tp", None))
        row = {
            **variant_facts(gv),
            **self._graded_verdict(gv, r, variant_meas),
            "status": r.status,
            "tput": r.output_throughput,
            **graded_axes_of(variant_meas),
            "base_tput": self.running_tput,
            "round_id": self.round_id,
            "ts": _now_iso(),
            "provenance": getattr(gv, "provenance", _DIRECT_PROVENANCE),
            "scope": str(getattr(gv, "scope", "") or ""),
            "workload_signature": self.ws_sig,
            "framework": self.framework,
            "workspace": r.workspace,
            "error_class": r.error_class or "",
            "server_log_path": r.server_log_path,
            "launch_evidence": dict(r.launch_evidence or {}),
            "launch_evidence_path": r.launch_evidence_path,
            "stage": FAILURE_STAGE_DECISION,
            "measured_against": self.measured_against(),
            "validation_basis": "",
        }
        if row["outcome"] == TS_FAILED and not row["reason"]:
            if accuracy_round_due(self.ss) or (gv.name in self.forcing and self.accuracy_reference.score > 0):
                await self._run_accuracy_round(row, slot, r.workspace)
            self._apply_accuracy_gate(row, slot)
        self._record(gv, row)
        if row["outcome"] == "KEEP":
            self.winners.append(self._keep(gv, run_gv, r, row, variant_meas))
        return row

    def _graded_verdict(self, gv: GridVariant, r: Any, variant_meas: dict[str, Any]) -> dict[str, Any]:
        # How the graded comparison ruled on this measurement, and which gate ended the arc.
        #
        # The gate is recorded where it is known: read off the outcome afterwards, "REVERT" cannot say which gate
        # ruled, and a gate that never ran is indistinguishable from one that ruled against the variant.
        graded = resolve_graded_comparison(
            self.ss,
            variant_meas,
            keep_threshold_pct=self.keep_threshold_pct,
            anchor=GradingAnchor(
                perf=self.running_perf,
                tput=self.running_tput,
            ),
        )
        on_intvty = graded.graded_on_intvty
        gain: float | None = None
        # The graded movement of a variant the verdict refused. ``gain`` stays None on a REVERT so no consumer
        # counts a refused variant's gain as a win, but the ledger row still has to say the axis was measured:
        # rendered with neither, a guard-refused variant reads as unmeasured.
        graded_gain: float | None = None
        outcome = TS_FAILED
        reason = ""
        if r.error_class == AGENTX_EARLY_STOP_ERROR_CLASS:
            outcome, reason = "REVERT", f"{AGENTX_EARLY_STOP_ERROR_CLASS} ({r.error})"
        elif r.status != "succeeded":
            reason = (r.error or "")[-1200:] or "no_measurement"
        elif graded.degrade_reason:
            # Same fail-closed rule as ``lift_to_current_best``: an AgentX session that could not grade on
            # interactivity does not KEEP on output throughput instead.
            outcome = "FAILED"
            reason = graded.degrade_reason
            log.info("explore: variant %r not comparable (%s)", gv.name, graded.degrade_reason)
        elif graded.verdict == VERDICT_REVERT:
            outcome = "REVERT"
            graded_gain = gain_pct(graded.candidate, graded.reference) if graded.reference > 0 else None
            axes = (
                f"intvty {graded.reference:.1f}->{graded.candidate:.1f} "
                f"tput {graded.tput_reference:.1f}->{graded.tput_candidate:.1f}"
            )
            # A vetoed variant is refused a round earlier than the promotion gate would, so it is never folded
            # onto the stack and never becomes the anchor the rest of the batch is graded against.
            reason = graded.veto_reason or (f"{graded.refused_by} ({axes})" if on_intvty else GAIN_BELOW_THRESHOLD)
        else:
            gain = gain_pct(graded.candidate, graded.reference)
        gates = [graded_gate_row(graded, self.keep_threshold_pct, reason)] if r.status == "succeeded" else []
        return {
            "outcome": outcome,
            # Which gate refused the variant. The prompt ledger renders this row, and without the reason a REVERT
            # on the tail guard is indistinguishable from one the benchmark never measured.
            "reason": reason,
            "gain_pct": gain,
            "graded_gain_pct": graded_gain,
            "graded_objective": graded.objective,
            "gates": gates,
        }

    async def _run_accuracy_round(self, row: dict[str, Any], slot: Path, workspace: str) -> None:
        if self.lifecycle_eligible:  # the round boots a server of its own on the variant's GPUs
            self._reap_variant_server(slot)
        launch = variant_config_of(workspace)
        outcome = await run_candidate_accuracy_round(slot, launch, self.ss, session=self.session_dir)
        record_accuracy_round(row, outcome)

    def _apply_accuracy_gate(self, row: dict[str, Any], slot: Path) -> None:
        # Rule on the round's accuracy evidence for a variant no earlier gate refused.
        #
        # A session with no baseline accuracy gates nothing, so its KEEPs rest on throughput alone -- a weaker
        # claim than one an accuracy gate ruled on, and ``validation_basis`` must not read alike for the two.
        scriptable = is_scriptable(self.framework)
        observed: Any = None
        reference: float | None = None
        passed = True
        # Serving still needs a measured baseline to compare against; scriptable compares against a fixed 1.0.
        gated = scriptable or self.accuracy_reference.score > 0
        if gated:
            results = parse_eval_results(
                slot, framework=self.framework, benchmark_mode=str(getattr(self.ss, "benchmark_mode", "") or "")
            )
            observed = results.get("accuracy") if scriptable else self.accuracy_reference.comparable(results)
            if isinstance(observed, (int, float)):
                reference = 1.0 if scriptable else self.accuracy_reference.score
                passed = accuracy_passed(reference, float(observed))
            else:
                passed = False
            row["gates"].append(
                {
                    "gate": "accuracy",
                    "passed": passed,
                    "observed": observed if isinstance(observed, (int, float)) else None,
                    "threshold": reference,
                    "reason": "" if passed else ("accuracy_unavailable" if observed is None else "accuracy_drop"),
                }
            )
        row["outcome"] = "KEEP" if passed else "REVERT"
        row["validation_basis"] = ("accuracy_pass" if gated else "keep_verdict_unscored") if passed else ""
        if not passed:
            row["reason"] = "accuracy_unavailable" if observed is None else "accuracy_drop"

    # ----- the ledgers -----------------------------------------------------

    def _record(self, gv: GridVariant, row: dict[str, Any]) -> None:
        self.tested[row["fingerprint"]] = row
        if row["outcome"] != "KEEP":
            rejected = {**ledger_view(row, REJECTED_FIELDS), **decision_fields(gv)}
            self.rejected.append(rejected)
            self.losers.append(ledger_view(row, LOSER_FIELDS))
            if self.ss is not None and self.session_dir is not None:
                # A preempted round keeps the verdicts it judged; a KEEP lands at settlement.
                outcome = outcome_row(row, row["fingerprint"], str(row.get("reason") or ""), str(self.ctx.task.task_id))
                attributions = framework_lever_attributions([outcome], self.lever_payload)
                land_verdict(
                    self.ss,
                    self.session_dir,
                    {"tested": {row["fingerprint"]: row}, "rejected": [rejected]},
                    attributions,
                )
            return
        self.winners_history.append(
            {
                "round_id": row["round_id"],
                "variant_name": row["name"],
                "fingerprint": row["fingerprint"],
                "gain_pct": row["gain_pct"],
                "extra_args": row["extra_server_args"],
                "extra_envs": row["extra_envs"],
                **controls_of(vars(gv)),
                "provenance": row["provenance"],
                "scope": row["scope"],
                "ts": row["ts"],
            }
        )

    def _next_effective_args(self, gv: GridVariant, run_remove_args: list[str]) -> str:
        """The full server argv string ``gv`` ran with on top of the inherited and accumulated args."""
        return compose_server_args(
            inherited_args=self.inherited_args,
            base_extra_args=self.args,
            variant_extra_args=gv.extra_server_args,
            remove_args=run_remove_args,
            args_mode="replace" if self.args_mode == "replace" else gv.args_mode,
        )

    def _next_stack_args(self, gv: GridVariant) -> str:
        """The accumulated stack args extended with ``gv``'s own delta."""
        return compose_server_args(
            base_extra_args=self.args,
            variant_extra_args=gv.extra_server_args,
            remove_args=gv.remove_args,
            args_mode=gv.args_mode,
        )

    def _keep(self, gv: GridVariant, run_gv: GridVariant, r: Any, row: dict[str, Any], meas: dict[str, Any]) -> dict:
        control_fields = controls_of(vars(gv))
        run_remove_args = self._run_remove_args(gv)
        next_effective_args = self._next_effective_args(gv, run_remove_args)
        next_stack_args = self._next_stack_args(gv)
        effective = dict(control_fields)
        if run_remove_args:
            effective["remove_args"] = list(run_remove_args)
        if run_gv.unset_envs:
            effective["unset_envs"] = list(run_gv.unset_envs)
        persist = keeps_effective_args(
            self.point.agentx, self.args_mode, gv.args_mode, run_remove_args, run_gv.unset_envs
        )
        if persist:
            effective["args_mode"] = "replace"
        # The variant KEEPs on the round that graded it.
        self.args = next_effective_args if persist else next_stack_args
        self.envs = dict(run_gv.extra_envs)
        self.remove_args = list(run_remove_args)
        self.unset_envs = list(run_gv.unset_envs)
        self.args_mode = "replace" if persist else "append"
        if row["tput"] and row["tput"] > 0:
            self.running_tput = row["tput"]
        if self.grade_on_intvty:
            # Only a comparable intvty KEEP advances the round anchor. Degraded measurements never reach here.
            self.running_perf = perf_snapshot_from_mapping(meas)
        return {
            "fingerprint": row["fingerprint"],
            "name": gv.name,
            "candidate_extra_server_args": gv.extra_server_args,
            "candidate_extra_envs": dict(gv.extra_envs or {}),
            "recipe_delta": {
                "extra_server_args": gv.extra_server_args,
                "extra_envs": dict(gv.extra_envs or {}),
                **control_fields,
            },
            "extra_server_args": self.args,
            "effective_extra_server_args": next_effective_args,
            "extra_envs": dict(run_gv.extra_envs),
            **effective,
            **decision_fields(gv),
            "provenance": row["provenance"],
            # Names of the authored kernels this config carried, when an overlay was loaded.
            "accepted_kernels": list(getattr(gv, "accepted_kernels", []) or []),
            "gain_pct": row["gain_pct"],
            "graded_objective": row["graded_objective"],
            # The verdict this KEEP rests on. ``None`` means the variant was not gated (not high-risk, or no
            # baseline) rather than that it scored nothing.
            "accuracy": next((g["observed"] for g in row["gates"] if g["gate"] == "accuracy"), None),
            "tput": row["tput"],
            # The axes this KEEP was graded on travel with it: current_best becomes the next round's anchor, and
            # an anchor without them degrades the session.
            **graded_axes_of(meas),
            # Promotion re-checks the budgets against this row, not the round's measurement.
            **constraint_fields(r),
            "single_workspace": r.workspace,
            "launch_evidence": dict(r.launch_evidence or {}),
            "launch_evidence_path": r.launch_evidence_path,
            "round_id": self.round_id,
            "accepted_at_round": self.round_id,
            "ts": _now_iso(),
        }

    def _per_variant_outcomes(self, reasons: dict[str, str]) -> list[dict[str, Any]]:
        task_id = str(self.ctx.task.task_id)
        outcomes = [
            outcome_row(te, fingerprint, reasons.get(fingerprint, ""), task_id)
            for fingerprint, te in self.tested.items()
            if te.get("outcome") in ("KEEP", "REVERT", TS_FAILED)
        ]
        outcomes.extend(
            {
                "variant_name": str(sd.get("name") or ""),
                "outcome": TS_SKIPPED_DEDUP,
                "fingerprint": str(sd.get("fingerprint") or ""),
                "provenance": "",
                "metrics": {},
                "reason": str(sd.get("reason") or ""),
            }
            for sd in self.skipped
        )
        return outcomes

    def _result(self) -> dict[str, Any]:
        rejected = {row["fingerprint"]: row for row in self.rejected if row["fingerprint"]}
        outcomes = self._per_variant_outcomes({fp: str(row.get("reason") or "") for fp, row in rejected.items()})
        levers = framework_lever_attributions(outcomes, self.lever_payload)
        if levers:
            log.info(
                "explore: attributed %d framework rewrite lever(s): %s",
                len(levers),
                ", ".join(f"{a['switch']}={a['gain_pct']:+.2f}%" for a in levers),
            )
        best = max(self.winners, key=lambda w: to_float(w.get("gain_pct"), 0.0) or 0.0, default=None)
        best_gain_pct = (to_float(best.get("gain_pct"), 0.0) or 0.0) if best else 0.0
        # A round whose every variant a pre-filter dropped, already had its verdict, or only restated the stack is
        # answered, as if measured again.
        offered = len(self.runnable) + self.prefilter_dropped
        answered = bool(offered) and self.judged_without_measuring + self.prefilter_dropped == offered
        measured = any(t.get("outcome") in ("KEEP", "REVERT") for t in self.tested.values())
        status = "succeeded" if measured or self.winners or answered else "failed"
        return {
            "status": status,
            # A round that measured nothing because the run stopped it is not the same as one whose variants
            # failed. As a bare ``failed`` with no error_class nothing downstream could tell the two apart, and
            # the KB would learn that these variants are bad.
            **self._budget_error(status),
            "base_tput": self.base_tput,
            # Each KEEP advances the running anchor, so this is the final stack.
            "output_throughput": float(self.running_tput) if self.winners else None,
            "best_variant": best,
            "best_gain_pct": best_gain_pct,
            "winners": self.winners,
            "losers": self.losers,
            "skipped_dup": self.skipped,
            "per_variant_outcomes": outcomes,
            "framework_lever_attributions": levers,
            "explore_search_update": {
                "tested": self.tested,
                "rejected": list(rejected.values()),
                "winners_history": self.winners_history,
                "last_round": self._last_round_summary(),
            },
            "round_id": self.round_id,
            "workspace": self.output_root.as_posix(),
            "framework": self.framework,
            # gain_pct for the audit trail (best gain of the batch).
            "gain_pct": best_gain_pct,
            "explore_grid_exhausted": not self.runnable and not self.prefilter_dropped,
            "projection": self.projection,
        }

    def _budget_error(self, status: str) -> dict[str, Any]:
        if status != "failed" or self.run_stop is None:
            return {}
        return {
            "error_class": self.run_stop.error_class,
            "error": (
                f"{self.run_stop_detail}; {self.untested} variant(s) went unmeasured "
                "and stay out of the ledger so a resume can retry them"
            ),
        }

    def _last_round_summary(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "base_tput": self.base_tput,
            "base_extra_args": self.base_args,
            "tested": [w["fingerprint"] for w in self.winners] + [lr["fingerprint"] for lr in self.losers],
            "round_winners": [w["fingerprint"] for w in self.winners],
            "skipped_dup": self.skipped,
            "ts": _now_iso(),
        }


class ExploreExecutor:
    """ActionRunner for the merged ``explore`` action."""

    session_dir = SessionDirField()

    def __init__(
        self,
        *,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        keep_threshold_pct: float = DEFAULT_KEEP_THRESHOLD_PCT,
    ):
        """Initialize the explore executor and its gating thresholds."""
        self.default_config_path = Path(default_config_path) if default_config_path else None
        self.session_dir = session_dir
        self.keep_threshold_pct = float(keep_threshold_pct)

    async def __call__(self, ctx) -> dict[str, Any]:
        """Run the merged ``explore`` action for one task."""
        params = dict(ctx.task.params or {})
        params["config_path"] = params.get("config_path") or self.default_config_path or default_baseline_config()
        params.setdefault("keep_threshold_pct", self.keep_threshold_pct)
        extra = getattr(ctx, "extra", None) or {}
        output_root = Path(
            params.get("output_dir")
            or extra.get("workspace")
            or runs_dir(self.session_dir, "explore", ctx.task.task_id)
        )
        output_root.mkdir(parents=True, exist_ok=True)
        try:
            return await _ExploreRound(ctx, params, output_root).run(self.session_dir)
        except _RoundRefused as refusal:
            return refusal.result


explore_executor = ExploreExecutor()


__all__ = [
    "ExploreExecutor",
    "explore_executor",
]
