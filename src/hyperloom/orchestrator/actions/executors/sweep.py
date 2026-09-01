# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real ``sweep`` ActionRunner — full ISL/OSL/CONC Pareto sweep.

Relaunches one Magpie bench per (CONC, ISL, OSL) combo with the optimized
server config.

Inputs (task.params):

* ``config_path``      — base Magpie YAML (defaults to baseline asset)
* ``base_extra_args``  — current best EXTRA_SGLANG_ARGS to layer in
* ``conc_values``      — list of int CONC, default [4, 16, 64]
* ``isl_osl_configs``  — list of "<ISL>:<OSL>" str, default ["1024:1024",
                          "8192:1024", "1024:8192"]
* ``num_prompts_factor`` — multiplier vs CONC (default 5)

Result::

    status:        "succeeded" | "failed"
    sweep_grid:    [{conc, isl, osl, output_throughput, ttft_mean_ms,
                    e2el_mean_ms, status, workspace, error}]
    pareto_front:  subset of sweep_grid that's not dominated
                   (output tput vs e2el, or total tput vs p90 intvty when
                   HYPERLOOM_PERF_METRIC=composite_v1)
    best_for_each_conc: dict[conc → best cell]
                        (highest output tput, or highest composite *S* when
                        the flag is on)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_int
from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.inference_optimizer.session.session_paths import runs_dir
from ._grid_base import best_entry_for_each_conc, select_sweep_pareto
from ._grid_runner import (
    GridVariant,
    VariantResult,
    _num_gpus_for_config,
    _resolve_session_dir,
    apply_multi_node_invalid_variants,
    run_grid,
    sanitize_result_dir,
    sanitize_script_name,
    session_grid_bounds,
)
from ._ray_serving import maybe_serving_lease
from ._workload_envs import (
    FrameworkScriptMismatchError,
    agentx_enabled,
    default_baseline_config,
    materialize_config_with_envs,
)


log = logging.getLogger(__name__)


DEFAULT_CONC_VALUES = [4, 16, 64]
DEFAULT_ISL_OSL = ["1024:1024", "8192:1024", "1024:8192"]
DEFAULT_NUM_PROMPTS_FACTOR = 5


def _build_grid(
    *,
    conc_values: list[int],
    isl_osl_configs: list[str],
    num_prompts_factor: int,
    base_extra_args: str,
    base_remove_args: list[str] | None = None,
    base_unset_envs: list[str] | None = None,
    base_args_mode: str = "append",
    max_model_len: int = 0,
) -> tuple[list[GridVariant], list[dict[str, Any]]]:
    """Fan out CONC × (ISL, OSL) into per-combo Magpie variants (each
    overriding CONC / ISL / OSL envs).

    Combos with ``ISL + OSL`` over a positive ``max_model_len`` are dropped
    up front (the server would reject every request), avoiding a wasted
    launch.

    Args:
        conc_values: Concurrency values to fan out.
        isl_osl_configs: ``"ISL:OSL"`` strings to fan out.
        num_prompts_factor: Multiplier deriving ``NUM_PROMPTS`` from concurrency.
        base_extra_args: Server args applied to every variant.
        base_remove_args: Inherited server flags removed by current_best.
        base_unset_envs: Inherited env names removed by current_best.
        base_args_mode: ``"append"`` or ``"replace"`` for the base args.
        max_model_len: When positive, drops combos whose ``ISL + OSL`` exceeds
            it.

    Returns:
        A ``(runnable_variants, skipped_records)`` tuple so dropped combos
        stay visible in the result.
    """
    out: list[GridVariant] = []
    skipped: list[dict[str, Any]] = []
    for conc in conc_values:
        num_prompts = max(int(conc) * num_prompts_factor, conc)
        for io_cfg in isl_osl_configs:
            try:
                isl_str, osl_str = io_cfg.split(":", 1)
                isl, osl = int(isl_str), int(osl_str)
            except (ValueError, AttributeError) as exc:
                log.warning("sweep: malformed isl_osl=%s: %s — skipping", io_cfg, exc)
                continue
            name = f"conc{conc}_isl{isl}_osl{osl}"
            if max_model_len > 0 and (isl + osl) > max_model_len:
                reason = f"isl+osl={isl + osl} exceeds max_model_len={max_model_len}"
                log.warning(
                    "sweep: skipping variant %s: %s (server would reject every request)",
                    name,
                    reason,
                )
                skipped.append(
                    {
                        "name": name,
                        "conc": conc,
                        "isl": isl,
                        "osl": osl,
                        "status": "skipped",
                        "skip_reason": reason,
                    }
                )
                continue
            variant_envs = {
                "CONC": str(conc),
                "ISL": str(isl),
                "OSL": str(osl),
                "NUM_PROMPTS": str(num_prompts),
            }
            # Accuracy eval is concurrency-invariant, so it is always skipped per
            # sweep point (the accuracy gate still runs on explore / baseline).
            variant_envs["RUN_EVAL"] = "false"
            out.append(
                GridVariant(
                    name=name,
                    extra_server_args=base_extra_args,
                    extra_envs=variant_envs,
                    remove_args=list(base_remove_args or []),
                    unset_envs=list(base_unset_envs or []),
                    args_mode="replace" if str(base_args_mode).strip().lower() == "replace" else "append",
                    note=f"conc={conc} isl={isl} osl={osl}",
                )
            )
    return out, skipped


def _result_dict(v: VariantResult) -> dict[str, Any]:
    """Convert a VariantResult to a dict with conc/isl/osl pulled out.

    Args:
        v (VariantResult): The variant result to serialize.

    Returns:
        dict[str, Any]: ``v.to_dict()`` augmented with int ``conc`` / ``isl``
            / ``osl`` keys extracted from the variant's ``extra_envs``.
    """
    d = v.to_dict()
    # Surface conc/isl/osl from extra_envs.
    envs = v.extra_envs or {}
    d["conc"] = int(envs.get("CONC", 0))
    d["isl"] = int(envs.get("ISL", 0))
    d["osl"] = int(envs.get("OSL", 0))
    return d


# ---------------------------------------------------------------------------
class SweepExecutor:
    """ActionRunner for the ``sweep`` action."""

    def __init__(
        self,
        *,
        default_config_path: Path | str | None = None,
        session_dir: Path | str | None = None,
        default_conc_values: list[int] | None = None,
        default_isl_osl_configs: list[str] | None = None,
        default_num_prompts_factor: int = DEFAULT_NUM_PROMPTS_FACTOR,
        variant_timeout_sec: int = 2400,
    ):
        """Initialize the sweep executor with default sweep parameters.

        Args:
            default_config_path: Benchmark config path; ``None`` resolves
                from ``$FRAMEWORK`` at call time.
            session_dir: Default session directory for outputs.
            default_conc_values: Default concurrency values to sweep.
            default_isl_osl_configs: Default input/output length configs.
            default_num_prompts_factor: Multiplier for prompt count.
            variant_timeout_sec: Per-variant timeout in seconds.
        """
        # None resolves at call time from $FRAMEWORK; explicit fixture wins.
        self.default_config_path = Path(default_config_path) if default_config_path else None
        self.session_dir = Path(session_dir) if session_dir else _resolve_session_dir()
        self.default_conc_values = list(default_conc_values or DEFAULT_CONC_VALUES)
        self.default_isl_osl_configs = list(default_isl_osl_configs or DEFAULT_ISL_OSL)
        self.default_num_prompts_factor = int(default_num_prompts_factor)
        self.variant_timeout_sec = variant_timeout_sec

    async def __call__(self, ctx) -> dict[str, Any]:
        """Run the full CONC × (ISL, OSL) sweep and map the Pareto frontier.

        Materializes the workload config, builds the variant grid, runs it via
        ``run_grid``, and computes the Pareto front plus the best variant per
        concurrency level.

        Args:
            ctx: The runner context carrying ``task.params`` (sweep knobs /
                config) and ``extra`` (workspace).

        Returns:
            dict[str, Any]: A result dict with ``status``, ``grid_size``,
                ``sweep_grid``, ``pareto_front``, ``best_for_each_conc`` and
                ``workspace``.
        """
        params = ctx.task.params or {}
        extra = getattr(ctx, "extra", None) or {}
        shared_state = extra.get("shared_state") or extra.get("state")
        framework = str(getattr(shared_state, "framework", "") or "") if shared_state is not None else ""
        # GEAK reuse path: sweep the optimized server via GEAK's own bench_e2e.sh
        # + the already-built overlay.
        ps_result = params.get("geak_result") or {}
        if ps_result.get("bench_script") and ps_result.get("status") == "ok":
            output_root = Path(
                params.get("output_dir")
                or extra.get("workspace")
                or runs_dir(self.session_dir, "sweep", ctx.task.task_id)
            )
            from ._geak_sweep import sweep_via_geak

            return await sweep_via_geak(
                result=ps_result,
                conc_values=list(params.get("conc_values") or self.default_conc_values),
                isl_osl_configs=list(params.get("isl_osl_configs") or self.default_isl_osl_configs),
                output_root=output_root,
                variant_timeout_sec=int(params.get("variant_timeout_sec", self.variant_timeout_sec)),
                framework=framework,
                state=shared_state,
            )

        config_path = Path(params.get("config_path") or self.default_config_path or default_baseline_config())
        if not config_path.exists():
            return {"status": "failed", "error_class": "missing_config", "error": f"config not found: {config_path}"}
        output_root = Path(
            params.get("output_dir") or extra.get("workspace") or runs_dir(self.session_dir, "sweep", ctx.task.task_id)
        )
        output_root.mkdir(parents=True, exist_ok=True)

        # Workload-contract materialization: sweep overrides CONC/ISL/OSL/
        # NUM_PROMPTS per variant, but TP/MAX_MODEL_LEN/PRECISION/RUN_EVAL/
        # ROCR_VISIBLE_DEVICES still flow from env onto the variant base.
        resolved_model = resolve_session_model_path(
            params=params,
            state_model_path=str(getattr(shared_state, "model_path", "") or "") if shared_state else "",
            for_serving=True,
        )
        resolved_gpu = (
            str(params.get("gpu_type") or "").strip().lower() or os.environ.get("GPU_TYPE", "").strip().lower()
        )
        try:
            override_script = sanitize_script_name(params.get("benchmark_script"))
            override_result_dir = sanitize_result_dir(params.get("result_dir"))
        except ValueError as exc:
            return {
                "status": "failed",
                "error_class": "bad_param",
                "error": str(exc),
            }
        try:
            config_path = materialize_config_with_envs(
                config_path,
                output_root,
                model_path=resolved_model or None,
                gpu_type=resolved_gpu or None,
                benchmark_script=override_script,
                out_name="sweep_base.with_envs.yaml",
            )
        except FrameworkScriptMismatchError as exc:
            return {
                "status": "failed",
                "error_class": "framework_script_mismatch",
                "error": str(exc),
            }

        conc_values = list(params.get("conc_values") or self.default_conc_values)
        isl_osl_configs = list(params.get("isl_osl_configs") or self.default_isl_osl_configs)
        # The AgentX client reads CONC and nothing else -- request shapes come
        # from the trace corpus, so ISL/OSL are inert placeholders there. Fanning
        # out over them would run byte-identical measurements under three
        # different labels and attribute the spread between them to ISL/OSL.
        # Collapse to a single point; the concurrency axis is unaffected.
        # (Also note the ISL+OSL > max_model_len skip below can no longer fire
        # under AgentX: max_model_len is now the model's native window, orders of
        # magnitude above these placeholders -- so nothing else prunes them.)
        if agentx_enabled() and len(isl_osl_configs) > 1:
            log.info(
                "sweep: AgentX collapses %d ISL/OSL points to %s (the agentic client does not read ISL/OSL)",
                len(isl_osl_configs),
                isl_osl_configs[0],
            )
            isl_osl_configs = isl_osl_configs[:1]
        num_prompts_factor = int(params.get("num_prompts_factor", self.default_num_prompts_factor))
        base_extra_args = params.get("base_extra_args", "")
        base_remove_args = [str(v) for v in (params.get("base_remove_args") or []) if str(v).strip()]
        base_unset_envs = [str(v) for v in (params.get("base_unset_envs") or []) if str(v).strip()]
        base_args_mode = str(params.get("base_args_mode") or "append")
        timeout_sec = int(params.get("variant_timeout_sec", self.variant_timeout_sec))

        # Drop ISL+OSL combos over the context window (see _build_grid);
        # resolved from task params, then $MAX_MODEL_LEN.
        max_model_len = to_int(params.get("max_model_len") or os.environ.get("MAX_MODEL_LEN"), default=0)

        grid, skipped_variants = _build_grid(
            conc_values=conc_values,
            isl_osl_configs=isl_osl_configs,
            num_prompts_factor=num_prompts_factor,
            base_extra_args=base_extra_args,
            base_remove_args=base_remove_args,
            base_unset_envs=base_unset_envs,
            base_args_mode=base_args_mode,
            max_model_len=max_model_len,
        )

        # Drop multi-node-invalid variants (cuda-graph-max-bs < CONC). No-op in
        # single-node; keeps CONC order for the Pareto-front computation.
        grid, _ = apply_multi_node_invalid_variants(grid)

        # Ray-managed GPU execution (§12 T1): one held Ray lease (``num_gpus=TP``)
        # spans the whole sweep grid — every variant (and its auto-warmup +
        # measure rounds) runs through the same lease's actor, so no server
        # outlives its GPU lease. ``None`` on the local path (multi-node /
        # RAY_EXEC off / tests) keeps the legacy behaviour.
        sweep_lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(config_path))
        # Stop the sweep grid mid-way when the session wall-clock budget runs out,
        # and cap each variant at what the budget can still pay for. Admission is
        # judged on the expected runtime rather than ``timeout_sec``, which is the
        # catastrophic-hang backstop and would abandon the tail of the budget.
        session_deadline_sec, variant_expected_sec = session_grid_bounds(
            extra.get("shared_state") or extra.get("state")
        )
        try:
            # Pass resolved_model / resolved_gpu so variant servers inherit TP/precision.
            results = await run_grid(
                base_yaml_path=config_path,
                base_extra_args="",  # sweep variants carry args themselves
                grid=grid,
                output_root=output_root,
                variant_timeout_sec=timeout_sec,
                model_path=resolved_model,
                gpu_type=resolved_gpu,
                benchmark_script=override_script,
                result_dir=override_result_dir,
                serving_lease=sweep_lease,
                session_deadline_sec=session_deadline_sec,
                variant_expected_sec=variant_expected_sec,
            )
        finally:
            if sweep_lease is not None:
                sweep_lease.close()

        entries = [_result_dict(v) for v in results]
        # Surface skipped combos so the grid stays complete; they never enter
        # Pareto / best selections.
        entries.extend(skipped_variants)
        front = select_sweep_pareto(entries, framework=framework)
        best_for_each_conc = best_entry_for_each_conc(
            entries,
            framework=framework,
            state=shared_state,
        )

        successful_entries = [e for e in entries if e.get("status") == "succeeded"]

        return {
            "status": "succeeded" if successful_entries else "failed",
            "grid_size": len(entries),
            "sweep_grid": entries,
            "pareto_front": front,
            "best_for_each_conc": best_for_each_conc,
            "workspace": output_root.as_posix(),
        }


sweep_executor = SweepExecutor()


__all__ = [
    "DEFAULT_CONC_VALUES",
    "DEFAULT_ISL_OSL",
    "DEFAULT_NUM_PROMPTS_FACTOR",
    "SweepExecutor",
    "sweep_executor",
]
