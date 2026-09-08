# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Concurrency sweep over the CONC ladder."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Mapping

from hyperloom.common import io as _common_io
from hyperloom.common.gain_math import conc_pair_comparison
from hyperloom.common.model_paths import resolve_session_model_path
from hyperloom.common.perf_metric import graded_metric_key, is_agentx_mode
from hyperloom.common.timeutil import now_iso, utc_now_compact
from hyperloom.inference_optimizer.breakdown.recorder.conc_sweep_event import (
    GRID_MODE_DEFAULT,
    GRID_REQUESTED,
    STAGE_BOOT,
    STAGE_BOOT_ATTEMPT,
    STAGE_BUDGET_SKIP,
    STAGE_REUSE,
    STAGE_SERVER_RESTART,
    STRATEGY_SERVER_RESTART,
    STRATEGY_SINGLE_SERVER,
)
from hyperloom.inference_optimizer.session.session_paths import reports_dir, runs_root
from ..actions.executors._grid_runner import (
    GridVariant,
    VariantResult,
    _kill_stale_servers,
    agentx_variant_timeout_sec,
    run_grid,
    variant_conc,
)
from ..actions.executors._workload_envs import (
    FrameworkScriptMismatchError,
    default_baseline_config,
    materialize_config_with_envs,
)
from .roofline_ceiling import (
    compute_compute_bound_ceiling_tok_per_sec,
    compute_theoretical_peak_output_tok_per_sec,
    load_model_meta,
    select_peak_and_bound,
)
from ..state.shared_state import SharedState
from ..loop.coordinator_helpers import baseline_benchmark_script


log = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"

# Default ladders, one per workload (override via ``--conc-sweep-concs``).
DEFAULT_CONCS: list[int] = [256, 128, 64, 32, 16, 8, 4, 2]
AGENTX_DEFAULT_CONCS: list[int] = [1, 4, 8, 10, 14, 20, 28]


def default_concs_for_mode(benchmark_mode: Any = "") -> list[int]:
    """The ladder a mode sweeps when the operator names none."""
    return list(AGENTX_DEFAULT_CONCS if is_agentx_mode(benchmark_mode) else DEFAULT_CONCS)


# Multiplier applied to each CONC for NUM_PROMPTS.
DEFAULT_NUM_PROMPTS_FACTOR = 5

# Per-variant timeout (seconds); override via ``--conc-sweep-timeout-sec``.
DEFAULT_VARIANT_TIMEOUT_SEC = 1800

# Total wall-clock budget (seconds); override via ``--conc-sweep-total-budget-sec``.
DEFAULT_TOTAL_BUDGET_SEC = 9000

# How many rungs the AgentX floor below buys when the default budget cannot fund even one.
_AGENTX_MIN_FUNDED_RUNGS = 2


def _granted_cap_sec(variant_timeout_sec: int, shared_state: Any = None, conc: int | None = None) -> float:
    """What a variant will actually be granted, for budget arithmetic."""
    return float(agentx_variant_timeout_sec(variant_timeout_sec, shared_state=shared_state, conc=conc))


def _has_optimization(state: SharedState) -> tuple[bool, str, dict[str, str]]:
    """Return ``(has_opt, args, envs)`` for a retained config or kernel overlay."""
    cb = state.current_best or {}
    args = str(cb.get("extra_server_args") or "").strip()
    envs_raw = cb.get("extra_envs") or {}
    envs = {str(k): str(v) for k, v in envs_raw.items()}
    overlay = str(cb.get("final_overlay") or "").strip()
    return bool(args or envs or overlay), args, envs


def _budget_skip_result(variant: GridVariant) -> VariantResult:
    """Synthetic VariantResult for a budget-exhausted variant; ``skipped`` status distinguishes \"out of time\" from \"Magpie crashed\"."""
    return VariantResult(
        name=variant.name,
        extra_server_args=variant.extra_server_args,
        extra_envs=dict(variant.extra_envs),
        status="skipped",
        output_throughput=None,
        request_throughput=None,
        total_token_throughput=None,
        error="conc_sweep total budget exhausted before this variant ran",
        error_class="budget_exhausted",
        note=variant.note,
    )


def _point_from_variant(v: VariantResult, *, arm: str) -> dict[str, Any]:
    """Flatten a ``VariantResult`` into one row of the curve."""
    envs = v.extra_envs or {}
    try:
        conc = int(envs.get("CONC", "0"))
    except (TypeError, ValueError):
        conc = 0
    # aiperf reports the total; the other parsers pass through whatever the framework named, leaving it null on a run
    # that measured both halves.
    total = v.total_token_throughput
    if total is None and v.input_throughput is not None and v.output_throughput is not None:
        total = v.input_throughput + v.output_throughput
    return {
        "arm": arm,
        "conc": conc,
        "status": v.status,
        "output_throughput": v.output_throughput,
        "request_throughput": v.request_throughput,
        "total_token_throughput": total,
        "input_throughput": v.input_throughput,
        "e2e_norm_intvty_p90": v.intvty_p90,
        "tpot_p90_ms": v.tpot_p90_ms,
        "ttft_mean_ms": v.ttft_mean_ms,
        "e2el_mean_ms": v.e2el_mean_ms,
        "duration_seconds": v.duration_seconds,
        "completed_requests": v.completed_requests,
        "error": v.error,
        "error_class": v.error_class,
        "killed_overtime": v.killed_overtime,
        "estimated_output_throughput": v.estimated_output_throughput,
        "workspace": v.workspace,
        "report_path": v.report_path,
    }


def _budget_limited_without_valid_pair(
    *,
    budget_exhausted: bool,
    summary: dict[str, Any],
    baseline_points: list[dict[str, Any]],
    optimized_points: list[dict[str, Any]],
) -> bool:
    """Return true when budget gating, not benchmark failure, prevented all pairs."""
    if not budget_exhausted or int(summary.get("successful_pairs") or 0) > 0:
        return False
    points = baseline_points + optimized_points
    if not points:
        return False
    saw_budget_skip = False
    for point in points:
        status = str(point.get("status") or "").lower()
        error_class = str(point.get("error_class") or "")
        if error_class == "budget_exhausted":
            saw_budget_skip = True
            continue
        if status not in ("succeeded", "skipped"):
            return False
    return saw_budget_skip


def _write_csv(csv_path: Path, points: list[dict[str, Any]]) -> None:
    """One row per (arm, conc) — flat columns for spreadsheet pivots."""
    columns = [
        "arm",
        "conc",
        "status",
        "output_throughput",
        "request_throughput",
        "total_token_throughput",
        "input_throughput",
        "e2e_norm_intvty_p90",
        "tpot_p90_ms",
        "ttft_mean_ms",
        "e2el_mean_ms",
        "duration_seconds",
        "completed_requests",
        "error_class",
        "error",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for p in points:
            writer.writerow({k: p.get(k) for k in columns})


def _build_roofline_ceiling(
    state: SharedState,
    *,
    concs: list[int],
    isl: int,
    osl: int,
    baseline_points: list[dict[str, Any]],
    optimized_points: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Per-conc decode roofline alongside the measured curves."""
    model_path = str(getattr(state, "model_path", "") or "")
    precision = str(getattr(state, "precision", "") or "") or "bf16"
    meta = load_model_meta(model_path, precision_hint=precision)
    if meta is None:
        return None
    gpu_type = str(getattr(state, "gpu_type", "") or "")
    num_gpus = int(getattr(state, "tp", 0) or 0)
    if not gpu_type or num_gpus <= 0:
        return None

    t_cmp = compute_compute_bound_ceiling_tok_per_sec(
        gpu_type=gpu_type,
        num_gpus=num_gpus,
        precision_tag=precision,
        active_weight_bytes=meta.active_weight_bytes,
        weight_bytes=meta.weight_bytes,
        weight_dtype_bytes=meta.weight_dtype_bytes,
    )

    by_conc_b = {p["conc"]: p for p in baseline_points}
    by_conc_o = {p["conc"]: p for p in optimized_points}

    rows: list[dict[str, Any]] = []
    for c in concs:
        t_mem = compute_theoretical_peak_output_tok_per_sec(
            gpu_type=gpu_type,
            num_gpus=num_gpus,
            weight_bytes=meta.weight_bytes,
            active_weight_bytes=meta.active_weight_bytes,
            num_experts=meta.num_experts,
            experts_per_tok=meta.experts_per_tok,
            expert_weight_bytes=meta.expert_weight_bytes,
            num_layers=meta.num_layers,
            num_kv_heads=meta.num_kv_heads,
            head_dim=meta.head_dim,
            kv_dtype_bytes=meta.weight_dtype_bytes,
            isl=isl,
            osl=osl,
            concurrency=c,
        )
        t_peak, bound_kind = select_peak_and_bound(t_mem, t_cmp)
        # Local import avoids a module-level import cycle.
        from .roofline_snapshot import within_roofline_pct

        def _mbu_pct(measured: Any) -> float | None:
            """Express a measured throughput as a percent of peak."""
            if not isinstance(measured, (int, float)) or measured <= 0:
                return None
            return within_roofline_pct(peak=float(t_peak), achieved=float(measured))

        bt = (by_conc_b.get(c) or {}).get("output_throughput")
        ot = (by_conc_o.get(c) or {}).get("output_throughput")
        rows.append(
            {
                "conc": c,
                "t_mem_tok_s": round(t_mem, 2),
                "t_cmp_tok_s": round(t_cmp, 2),
                "t_peak_tok_s": round(t_peak, 2),
                "bound_kind": bound_kind,
                "mbu_baseline_pct": _mbu_pct(bt),
                "mbu_optimized_pct": _mbu_pct(ot),
            }
        )

    return {
        "schema_version": 1,
        "source": "roofline_ceiling.py",
        "gpu_type": gpu_type,
        "precision": precision,
        "tp": num_gpus,
        "isl": isl,
        "osl": osl,
        "model_meta": {
            "weight_bytes": meta.weight_bytes,
            "active_weight_bytes": meta.active_weight_bytes,
            "num_experts": meta.num_experts,
            "experts_per_tok": meta.experts_per_tok,
            "expert_weight_bytes": meta.expert_weight_bytes,
            "num_layers": meta.num_layers,
            "num_kv_heads": meta.num_kv_heads,
            "head_dim": meta.head_dim,
            "weight_dtype_bytes": meta.weight_dtype_bytes,
        },
        "rows": rows,
    }


def _arm_status(results: list[VariantResult]) -> str:
    """The status an arm reports from the rungs it actually ran.

    An arm that measured some rungs and lost others is ``degraded`` rather
    than either extreme: the curve it produced is real and shorter than the
    ladder it was asked for.
    """
    statuses = {str(result.status or "").lower() for result in results}
    if not statuses:
        return "skipped"
    if "succeeded" in statuses:
        return "degraded" if statuses - {"succeeded"} else "succeeded"
    if statuses <= {"skipped"}:
        return "skipped"
    return "failed"


def _record_rung(
    recorder: Any,
    arm_name: str,
    result: VariantResult,
    *,
    stage: str,
    committed: bool = True,
    start_time: str = "",
    wall_duration_sec: float | None = None,
    granted_cap_sec: float | None = None,
    budget_remaining_sec: float | None = None,
) -> None:
    """Record one rung on the sweep's event, if the sweep is being recorded.

    The result is flattened into the same point the report writes, so the
    recorded curve and the written one cannot differ. ``stage`` is a
    ``STAGE_*`` value naming how the rung came to run, ``committed`` says
    whether it is part of the published curve, and the seconds fields are
    wall clock.
    """
    if recorder is None:
        return
    point = _point_from_variant(result, arm=arm_name)
    recorder.record_variant(
        arm_name,
        stage=stage,
        conc=point.get("conc"),
        point=point,
        committed=committed,
        num_prompts=(result.extra_envs or {}).get("NUM_PROMPTS"),
        start_time=start_time,
        wall_duration_sec=wall_duration_sec,
        granted_cap_sec=granted_cap_sec,
        budget_remaining_sec=budget_remaining_sec,
    )


def _order_concs_desc(concs: list[int]) -> list[int]:
    """Return a strictly descending, deduplicated copy of the CONC ladder."""
    return sorted(set(concs), reverse=True)


def _build_arm_grid(
    arm_name: str,
    concs_desc: list[int],
    *,
    isl: int,
    osl: int,
    num_prompts_factor: int,
    arm_args: str,
    arm_envs: dict[str, str],
    overlay_pythonpath: str = "",
) -> list[GridVariant]:
    """Build a single-arm grid in descending CONC order."""
    out: list[GridVariant] = []
    for conc in concs_desc:
        num_prompts = max(int(conc) * int(num_prompts_factor), int(conc))
        envs = dict(arm_envs)
        envs.update(
            {
                "CONC": str(conc),
                "ISL": str(isl),
                "OSL": str(osl),
                "NUM_PROMPTS": str(num_prompts),
            }
        )
        envs["RUN_EVAL"] = "false"
        variant = GridVariant(
            name=f"{arm_name}_conc{conc}",
            extra_server_args=arm_args,
            extra_envs=envs,
            note=f"arm={arm_name} conc={conc} isl={isl} osl={osl}",
        )
        variant.overlay_pythonpath = overlay_pythonpath  # type: ignore[attr-defined]
        out.append(variant)
    return out


async def _sweep_one_arm_single_server(  # noqa: PLR0913
    arm_name: str,
    concs_desc: list[int],
    *,
    isl: int,
    osl: int,
    num_prompts_factor: int,
    arm_args: str,
    arm_envs: dict[str, str],
    base_yaml_path: Path,
    workspace: Path,
    model_path: str,
    gpu_type: str,
    variant_timeout_sec: int,
    soft_deadline_sec: float | None,
    deadline: float | None,
    state: SharedState,
    session_dir: Path,
    json_path: Path,
    csv_path: Path,
    started_at: float,
    total_budget_sec: int | None,
    has_budget: bool,
    opt_args: str,
    opt_envs: dict[str, str],
    _all_results_ref: list[VariantResult],
    _budget_state: dict[str, Any],
    recorder: Any = None,
    benchmark_script: str | None = None,
) -> list[VariantResult]:
    """Sweep one arm across all CONC values reusing a single persistent server.

    Boots the server on the highest CONC (Option A), then reuses it for all
    lower CONCs.  If boot fails, retries with the next lower CONC
    (boot-retry-descend).  Falls back to the legacy per-variant server-restart
    path (Option B) when all boot retries are exhausted.

    ``_all_results_ref`` and ``_budget_state`` are mutated in place: the first
    so incremental flushes see the full cross-arm picture, the second so the
    caller can inspect the final budget status. ``deadline`` is an absolute
    ``time.time()`` epoch (``None`` when unbounded), distinct from
    ``soft_deadline_sec``, which is a duration.
    """
    from ..actions.executors._grid_runner import _num_gpus_for_config
    from ..actions.executors._ray_serving import maybe_serving_lease
    from ..actions.executors._server_lifecycle import (
        resolve_lifecycle_params,
        teardown_lifecycle_server,
    )

    arm_results: list[VariantResult] = []
    overlay = str((state.current_best or {}).get("final_overlay") or "").strip()
    grid = _build_arm_grid(
        arm_name,
        concs_desc,
        isl=isl,
        osl=osl,
        num_prompts_factor=num_prompts_factor,
        arm_args=arm_args,
        arm_envs=arm_envs,
        overlay_pythonpath=overlay if arm_name == "optimized" else "",
    )
    if not grid:
        return arm_results
    if recorder is not None:
        recorder.record_arm_grid(
            arm_name,
            rungs=[
                {
                    "name": variant.name,
                    "conc": variant.extra_envs.get("CONC"),
                    "num_prompts": variant.extra_envs.get("NUM_PROMPTS"),
                }
                for variant in grid
            ],
        )

    # Ray-managed GPU execution: one held Ray lease (``num_gpus=TP``) spans this arm's persistent server — boot +
    # every CONC reuse round, or the Option B per-variant restarts — so the shared server's whole lifetime is covered
    # by a single lease and no GPU process outlives it.
    arm_lease = maybe_serving_lease(num_gpus=_num_gpus_for_config(base_yaml_path))

    # Shared pid_dir for server reuse across all CONC variants in this arm.
    pid_dir = workspace / f"server_{arm_name}"
    pid_dir.mkdir(parents=True, exist_ok=True)

    # Resolve lifecycle params (port, framework) from the materialized config.
    lc_reason = "resolve_failed"
    try:
        lc_params = resolve_lifecycle_params(base_yaml_path)
        port = int(lc_params.get("port") or 8888)
        framework = str(lc_params.get("framework") or "")
        lc_eligible = bool(lc_params.get("eligible"))
        lc_reason = str(lc_params.get("reason") or "")
    except Exception:  # noqa: BLE001
        log.debug("conc_sweep single-server: resolve_lifecycle_params failed", exc_info=True)
        lc_eligible = False
        port = 8888
        framework = ""

    if recorder is not None:
        recorder.record_arm_strategy(
            arm_name,
            strategy=STRATEGY_SINGLE_SERVER if lc_eligible else STRATEGY_SERVER_RESTART,
            reason=None if lc_eligible else "framework_not_lifecycle_eligible",
            lifecycle_eligible=lc_eligible,
            lifecycle_reason=lc_reason,
            port=port,
            framework=framework,
            serving_lease_held=arm_lease is not None,
        )

    if not lc_eligible:
        # Framework does not support server_lifecycle — fall through to Option B (per-variant server restart via
        # normal run_grid).
        log.info(
            "conc_sweep single-server: arm=%s not lifecycle-eligible (%s); using per-variant server restart (Option B)",
            arm_name,
            lc_reason,
        )
        try:
            arm_results = await _sweep_arm_option_b(
                arm_name=arm_name,
                grid=grid,
                base_yaml_path=base_yaml_path,
                workspace=workspace,
                model_path=model_path,
                gpu_type=gpu_type,
                benchmark_script=benchmark_script,
                variant_timeout_sec=variant_timeout_sec,
                soft_deadline_sec=soft_deadline_sec,
                deadline=deadline,
                state=state,
                session_dir=session_dir,
                json_path=json_path,
                csv_path=csv_path,
                started_at=started_at,
                total_budget_sec=total_budget_sec,
                has_budget=has_budget,
                opt_args=opt_args,
                opt_envs=opt_envs,
                _all_results_ref=_all_results_ref,
                _budget_state=_budget_state,
                serving_lease=arm_lease,
                recorder=recorder,
            )
            return arm_results
        finally:
            if arm_lease is not None:
                arm_lease.close()
            if recorder is not None:
                recorder.finish_arm(arm_name, status=_arm_status(arm_results))

    # Boot-retry-descend: try each CONC from highest to lowest until boot succeeds.
    failed_boots: list[VariantResult] = []
    boot_idx = 0
    boot_succeeded = False
    while boot_idx < len(grid):
        boot_variant = grid[boot_idx]
        log.info(
            "conc_sweep single-server: arm=%s boot attempt %d/%d (conc=%s)",
            arm_name,
            boot_idx + 1,
            len(grid),
            boot_variant.extra_envs.get("CONC", "?"),
        )
        server_lifecycle_boot = {
            "cleanup": False,
            "pid_dir": str(pid_dir),
            "port": port,
        }
        boot_started_iso = now_iso("seconds")
        boot_started_at = time.time()
        try:
            boot_results = await run_grid(
                base_yaml_path=base_yaml_path,
                base_extra_args="",
                grid=[boot_variant],
                output_root=workspace,
                variant_timeout_sec=variant_timeout_sec,
                model_path=model_path,
                gpu_type=gpu_type,
                benchmark_script=benchmark_script,
                server_lifecycle=server_lifecycle_boot,
                server_already_ready=False,
                preclean_before_run=True,
                warmup_before_measure=False,
                soft_deadline_sec=soft_deadline_sec,
                serving_lease=arm_lease,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "conc_sweep single-server: arm=%s boot conc=%s raised %r; trying next lower conc",
                arm_name,
                boot_variant.extra_envs.get("CONC", "?"),
                exc,
            )
            boot_results = [
                VariantResult(
                    name=boot_variant.name,
                    extra_server_args=boot_variant.extra_server_args,
                    extra_envs=dict(boot_variant.extra_envs),
                    status="failed",
                    error=f"single_server_boot_exception: {exc}",
                    error_class="single_server_boot_exception",
                )
            ]

        br = boot_results[0] if boot_results else None
        boot_failed = br is None or br.status in {"failed", "skipped"}
        boot_elapsed = round(time.time() - boot_started_at, 3)

        if boot_failed:
            # Ensure server is torn down before retrying at a lower CONC.
            try:
                teardown_lifecycle_server(pid_dir=pid_dir, framework=framework, port=port)
            except Exception:  # noqa: BLE001
                pass
            failed = br or VariantResult(
                name=boot_variant.name,
                extra_server_args=boot_variant.extra_server_args,
                extra_envs=dict(boot_variant.extra_envs),
                status="failed",
                error="single_server_boot_failed",
                error_class="single_server_boot_failed",
            )
            failed_boots.append(failed)
            # Recorded now and uncommitted: a concurrency the server would not
            # come up at is the finding, and whether it counts toward the curve
            # is not known until a lower rung either boots or does not.
            _record_rung(
                recorder,
                arm_name,
                failed,
                stage=STAGE_BOOT_ATTEMPT,
                committed=False,
                start_time=boot_started_iso,
                wall_duration_sec=boot_elapsed,
                granted_cap_sec=_granted_cap_sec(variant_timeout_sec, state, variant_conc(boot_variant)),
            )
            boot_idx += 1
            continue

        # Boot succeeded (br is not None here — boot_failed guarded above).
        assert br is not None
        boot_succeeded = True
        # Commit the higher-CONC failed boots (genuine capacity failures) first.
        for fb in failed_boots:
            arm_results.append(fb)
            _all_results_ref.append(fb)
            if recorder is not None:
                recorder.commit_variant(
                    arm_name,
                    stage=STAGE_BOOT_ATTEMPT,
                    conc=variant_conc(fb),
                    point=_point_from_variant(fb, arm=arm_name),
                )
        arm_results.append(br)
        _all_results_ref.append(br)
        _record_rung(
            recorder,
            arm_name,
            br,
            stage=STAGE_BOOT,
            start_time=boot_started_iso,
            wall_duration_sec=boot_elapsed,
            granted_cap_sec=_granted_cap_sec(variant_timeout_sec, state, variant_conc(boot_variant)),
        )
        if recorder is not None:
            recorder.record_arm_boot(
                arm_name,
                succeeded=True,
                booted_conc=variant_conc(boot_variant),
                attempted_concs=[variant_conc(variant) for variant in grid[: boot_idx + 1]],
                failed_concs=[variant_conc(fb) for fb in failed_boots],
            )
        # Incremental flush after boot point.
        _maybe_flush(
            state=state,
            session_dir=session_dir,
            json_path=json_path,
            csv_path=csv_path,
            all_results=_all_results_ref,
            concs=list(concs_desc),
            isl=isl,
            osl=osl,
            opt_args=opt_args,
            opt_envs=opt_envs,
            workspace=workspace,
            started_at=started_at,
            total_budget_sec=total_budget_sec,
            has_budget=has_budget,
            budget_exhausted=_budget_state.get("budget_exhausted", False),
            budget_skip_reason=_budget_state.get("budget_skip_reason", ""),
            budget_remaining_sec=_budget_state.get("budget_remaining_sec"),
            recorder=recorder,
        )
        break

    if not boot_succeeded:
        # Every CONC failed to boot the persistent server — retry the full grid via Option B (per-variant restart, no
        # lifecycle) which may succeed where persistent reuse could not.
        log.warning(
            "conc_sweep single-server: arm=%s all boot attempts failed; "
            "falling back to Option B (per-variant restart) for the full ladder",
            arm_name,
        )
        if recorder is not None:
            recorder.record_arm_boot(
                arm_name,
                succeeded=False,
                attempted_concs=[variant_conc(variant) for variant in grid],
                failed_concs=[variant_conc(fb) for fb in failed_boots],
            )
            # The whole ladder is retried per-rung, and those results supersede
            # the boot attempts rather than adding to them -- which is why the
            # attempts above stay uncommitted.
            recorder.record_arm_strategy(
                arm_name,
                strategy=STRATEGY_SERVER_RESTART,
                reason="all_boot_attempts_failed",
                lifecycle_eligible=lc_eligible,
                lifecycle_reason=lc_reason,
                port=port,
                framework=framework,
                serving_lease_held=arm_lease is not None,
            )
        ob_results = await _sweep_arm_option_b(
            arm_name=arm_name,
            grid=grid,
            base_yaml_path=base_yaml_path,
            workspace=workspace,
            model_path=model_path,
            gpu_type=gpu_type,
            benchmark_script=benchmark_script,
            variant_timeout_sec=variant_timeout_sec,
            soft_deadline_sec=soft_deadline_sec,
            deadline=deadline,
            state=state,
            session_dir=session_dir,
            json_path=json_path,
            csv_path=csv_path,
            started_at=started_at,
            total_budget_sec=total_budget_sec,
            has_budget=has_budget,
            opt_args=opt_args,
            opt_envs=opt_envs,
            _all_results_ref=_all_results_ref,
            _budget_state=_budget_state,
            serving_lease=arm_lease,
            recorder=recorder,
        )
        arm_results.extend(ob_results)
        if arm_lease is not None:
            arm_lease.close()
        if recorder is not None:
            recorder.finish_arm(arm_name, status=_arm_status(arm_results))
        return arm_results

    # Server is up: sweep remaining CONCs by reuse.
    try:
        reuse_grid = grid[boot_idx + 1 :]
        for r_idx, variant in enumerate(reuse_grid):
            # Check task-level budget before each reuse point.
            _reuse_remaining = (deadline - time.time()) if has_budget and deadline is not None else None
            if has_budget and _reuse_remaining is not None and _reuse_remaining <= 0:
                _budget_state["budget_exhausted"] = True
                _budget_state["budget_skip_reason"] = "total_budget_exhausted"
                _budget_state["budget_remaining_sec"] = max(0.0, float(_reuse_remaining))
                for v in reuse_grid[r_idx:]:
                    skip_r = _budget_skip_result(v)
                    arm_results.append(skip_r)
                    _all_results_ref.append(skip_r)
                    _record_rung(
                        recorder,
                        arm_name,
                        skip_r,
                        stage=STAGE_BUDGET_SKIP,
                        budget_remaining_sec=max(0.0, float(_reuse_remaining)),
                        granted_cap_sec=_granted_cap_sec(variant_timeout_sec, state, variant_conc(v)),
                    )
                break
            if (
                has_budget
                and _reuse_remaining is not None
                and _reuse_remaining < _granted_cap_sec(variant_timeout_sec, state, variant_conc(variant))
            ):
                _budget_state["budget_exhausted"] = True
                _budget_state["budget_skip_reason"] = "insufficient_remaining_for_variant"
                _budget_state["budget_remaining_sec"] = max(0.0, float(_reuse_remaining))
                for v in reuse_grid[r_idx:]:
                    skip_r = _budget_skip_result(v)
                    arm_results.append(skip_r)
                    _all_results_ref.append(skip_r)
                    _record_rung(
                        recorder,
                        arm_name,
                        skip_r,
                        stage=STAGE_BUDGET_SKIP,
                        budget_remaining_sec=max(0.0, float(_reuse_remaining)),
                        granted_cap_sec=_granted_cap_sec(variant_timeout_sec, state, variant_conc(v)),
                    )
                break
            # Check session deadline before each reuse point.
            if getattr(state, "closing_phase", False) or getattr(state, "stop_reason", ""):
                _budget_state["budget_exhausted"] = True
                _budget_state["budget_skip_reason"] = "session_deadline_reserve"
                _budget_state["budget_remaining_sec"] = 0.0
                for v in reuse_grid[r_idx:]:
                    skip_r = _budget_skip_result(v)
                    arm_results.append(skip_r)
                    _all_results_ref.append(skip_r)
                    _record_rung(
                        recorder,
                        arm_name,
                        skip_r,
                        stage=STAGE_BUDGET_SKIP,
                        budget_remaining_sec=0.0,
                        granted_cap_sec=_granted_cap_sec(variant_timeout_sec, state, variant_conc(v)),
                    )
                break

            is_last = r_idx == len(reuse_grid) - 1
            reuse_started_iso = now_iso("seconds")
            reuse_started_at = time.time()
            server_lifecycle_reuse = {
                "cleanup": is_last,
                "pid_dir": str(pid_dir),
                "port": port,
            }
            try:
                reuse_results = await run_grid(
                    base_yaml_path=base_yaml_path,
                    base_extra_args="",
                    grid=[variant],
                    output_root=workspace,
                    variant_timeout_sec=variant_timeout_sec,
                    model_path=model_path,
                    gpu_type=gpu_type,
                    benchmark_script=benchmark_script,
                    server_lifecycle=server_lifecycle_reuse,
                    server_already_ready=True,
                    preclean_before_run=False,
                    warmup_before_measure=False,
                    soft_deadline_sec=soft_deadline_sec,
                    serving_lease=arm_lease,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "conc_sweep single-server: arm=%s reuse conc=%s raised %r",
                    arm_name,
                    variant.extra_envs.get("CONC", "?"),
                    exc,
                )
                reuse_results = [
                    VariantResult(
                        name=variant.name,
                        extra_server_args=variant.extra_server_args,
                        extra_envs=dict(variant.extra_envs),
                        status="failed",
                        error=f"single_server_reuse_exception: {exc}",
                        error_class="single_server_reuse_exception",
                    )
                ]
            reuse_elapsed = round(time.time() - reuse_started_at, 3)
            for rr in reuse_results:
                arm_results.append(rr)
                _all_results_ref.append(rr)
                _record_rung(
                    recorder,
                    arm_name,
                    rr,
                    stage=STAGE_REUSE,
                    start_time=reuse_started_iso,
                    wall_duration_sec=reuse_elapsed,
                    granted_cap_sec=_granted_cap_sec(variant_timeout_sec, state, variant_conc(variant)),
                    budget_remaining_sec=_reuse_remaining,
                )
            # Incremental flush after each reuse point.
            _maybe_flush(
                state=state,
                session_dir=session_dir,
                json_path=json_path,
                csv_path=csv_path,
                all_results=_all_results_ref,
                concs=list(concs_desc),
                isl=isl,
                osl=osl,
                opt_args=opt_args,
                opt_envs=opt_envs,
                workspace=workspace,
                started_at=started_at,
                total_budget_sec=total_budget_sec,
                has_budget=has_budget,
                budget_exhausted=_budget_state.get("budget_exhausted", False),
                budget_skip_reason=_budget_state.get("budget_skip_reason", ""),
                budget_remaining_sec=_budget_state.get("budget_remaining_sec"),
                recorder=recorder,
            )
    finally:
        # Safety teardown — idempotent, no-op if already torn down.
        try:
            teardown_lifecycle_server(pid_dir=pid_dir, framework=framework, port=port)
        except Exception:  # noqa: BLE001
            pass
        if arm_lease is not None:
            arm_lease.close()
        if recorder is not None:
            recorder.finish_arm(arm_name, status=_arm_status(arm_results))

    return arm_results


async def _sweep_arm_option_b(  # noqa: PLR0913
    arm_name: str,
    grid: list[GridVariant],
    *,
    base_yaml_path: Path,
    workspace: Path,
    model_path: str,
    gpu_type: str,
    variant_timeout_sec: int,
    soft_deadline_sec: float | None,
    deadline: float | None,
    state: SharedState,
    session_dir: Path,
    json_path: Path,
    csv_path: Path,
    started_at: float,
    total_budget_sec: int | None,
    has_budget: bool,
    opt_args: str,
    opt_envs: dict[str, str],
    _all_results_ref: list[VariantResult],
    _budget_state: dict[str, Any],
    serving_lease: Any = None,
    recorder: Any = None,
    benchmark_script: str | None = None,
) -> list[VariantResult]:
    """Option B fallback: run each variant with its own server (legacy behaviour).

    Used when ``_sweep_one_arm_single_server`` detects the framework is not
    lifecycle-eligible or all boot retries are exhausted. ``_all_results_ref``
    and ``_budget_state`` are mutated in place; ``deadline`` is an absolute
    ``time.time()`` epoch, ``None`` when budget tracking is off; and
    ``serving_lease`` is ``None`` when the arm runs on the local (non-Ray)
    path.
    """
    arm_results: list[VariantResult] = []
    for variant in grid:
        # Task-level budget checks.
        _ob_rem = (deadline - time.time()) if has_budget and deadline is not None else None
        _ob_cap = _granted_cap_sec(variant_timeout_sec, state, variant_conc(variant))
        if has_budget and _ob_rem is not None and _ob_rem <= 0:
            _budget_state["budget_exhausted"] = True
            _budget_state["budget_skip_reason"] = "total_budget_exhausted"
            _budget_state["budget_remaining_sec"] = max(0.0, float(_ob_rem))
            skip_r = _budget_skip_result(variant)
            arm_results.append(skip_r)
            _all_results_ref.append(skip_r)
            _record_rung(
                recorder,
                arm_name,
                skip_r,
                stage=STAGE_BUDGET_SKIP,
                budget_remaining_sec=max(0.0, float(_ob_rem)),
                granted_cap_sec=_ob_cap,
            )
            continue
        if has_budget and _ob_rem is not None and _ob_rem < _ob_cap:
            _budget_state["budget_exhausted"] = True
            _budget_state["budget_skip_reason"] = "insufficient_remaining_for_variant"
            _budget_state["budget_remaining_sec"] = max(0.0, float(_ob_rem))
            skip_r = _budget_skip_result(variant)
            arm_results.append(skip_r)
            _all_results_ref.append(skip_r)
            _record_rung(
                recorder,
                arm_name,
                skip_r,
                stage=STAGE_BUDGET_SKIP,
                budget_remaining_sec=max(0.0, float(_ob_rem)),
                granted_cap_sec=_ob_cap,
            )
            continue
        if getattr(state, "closing_phase", False) or getattr(state, "stop_reason", ""):
            _budget_state["budget_exhausted"] = True
            _budget_state["budget_skip_reason"] = "session_deadline_reserve"
            _budget_state["budget_remaining_sec"] = 0.0
            skip_r = _budget_skip_result(variant)
            arm_results.append(skip_r)
            _all_results_ref.append(skip_r)
            _record_rung(
                recorder,
                arm_name,
                skip_r,
                stage=STAGE_BUDGET_SKIP,
                budget_remaining_sec=0.0,
                granted_cap_sec=_ob_cap,
            )
            continue
        rung_started_iso = now_iso("seconds")
        rung_started_at = time.time()
        try:
            sub = await run_grid(
                base_yaml_path=base_yaml_path,
                base_extra_args="",
                grid=[variant],
                output_root=workspace,
                variant_timeout_sec=variant_timeout_sec,
                model_path=model_path,
                gpu_type=gpu_type,
                benchmark_script=benchmark_script,
                soft_deadline_sec=soft_deadline_sec,
                serving_lease=serving_lease,
            )
        except Exception as exc:  # noqa: BLE001
            sub = [
                VariantResult(
                    name=variant.name,
                    extra_server_args=variant.extra_server_args,
                    extra_envs=dict(variant.extra_envs),
                    status="failed",
                    error=f"option_b_exception: {exc}",
                    error_class="option_b_exception",
                )
            ]
        rung_elapsed = round(time.time() - rung_started_at, 3)
        for r in sub:
            arm_results.append(r)
            _all_results_ref.append(r)
            _record_rung(
                recorder,
                arm_name,
                r,
                stage=STAGE_SERVER_RESTART,
                start_time=rung_started_iso,
                wall_duration_sec=rung_elapsed,
                granted_cap_sec=_ob_cap,
                budget_remaining_sec=_ob_rem,
            )
        _concs = [int(v.extra_envs["CONC"]) for v in grid if v.extra_envs.get("CONC")]
        _isl = int(next((v.extra_envs["ISL"] for v in grid if v.extra_envs.get("ISL")), "0"))
        _osl = int(next((v.extra_envs["OSL"] for v in grid if v.extra_envs.get("OSL")), "0"))
        _maybe_flush(
            state=state,
            session_dir=session_dir,
            json_path=json_path,
            csv_path=csv_path,
            all_results=_all_results_ref,
            concs=_concs,
            isl=_isl,
            osl=_osl,
            opt_args=opt_args,
            opt_envs=opt_envs,
            workspace=workspace,
            started_at=started_at,
            total_budget_sec=total_budget_sec,
            has_budget=has_budget,
            budget_exhausted=_budget_state.get("budget_exhausted", False),
            budget_skip_reason=_budget_state.get("budget_skip_reason", ""),
            budget_remaining_sec=_budget_state.get("budget_remaining_sec"),
            recorder=recorder,
        )
    return arm_results


def _maybe_flush(  # noqa: PLR0913
    *,
    state: SharedState,
    session_dir: Path,
    json_path: Path,
    csv_path: Path,
    all_results: list[VariantResult],
    concs: list[int],
    isl: int,
    osl: int,
    opt_args: str,
    opt_envs: dict[str, str],
    workspace: Path,
    started_at: float,
    total_budget_sec: int | None,
    has_budget: bool,
    budget_exhausted: bool,
    budget_skip_reason: str,
    budget_remaining_sec: float | None,
    recorder: Any = None,
) -> None:
    """Build a partial payload from *all_results* and flush it via :func:`_flush_partial_conc_sweep_report`.

    A thin convenience wrapper that avoids repeating the argument list at every
    call site.
    """
    _flush_partial_conc_sweep_report(
        results=list(all_results),
        state=state,
        session_dir=session_dir,
        json_path=json_path,
        csv_path=csv_path,
        concs=concs,
        isl=isl,
        osl=osl,
        opt_args=opt_args,
        opt_envs=opt_envs,
        workspace=workspace,
        started_at=started_at,
        total_budget_sec=total_budget_sec,
        has_budget=has_budget,
        budget_exhausted=budget_exhausted,
        budget_skip_reason=budget_skip_reason,
        budget_remaining_sec=budget_remaining_sec,
        recorder=recorder,
    )


def _flush_conc_sweep_report(payload: dict[str, Any], session_dir: Path) -> None:
    """Atomically write the conc-sweep summary JSON + CSV to the reports dir."""
    try:
        rdir = reports_dir(session_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        json_path = Path(payload["report_json_path"])
        csv_path = Path(payload["report_csv_path"])
        _common_io.atomic_write_text(
            json_path,
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        )
        all_points: list[dict[str, Any]] = list((payload.get("baseline") or {}).get("points") or []) + list(
            (payload.get("optimized") or {}).get("points") or []
        )
        _write_csv(csv_path, all_points)
    except Exception:  # noqa: BLE001
        log.debug("conc_sweep: _flush_conc_sweep_report failed", exc_info=True)


def _flush_partial_conc_sweep_report(  # noqa: PLR0913
    *,
    results: list[VariantResult],
    state: SharedState,
    session_dir: Path,
    json_path: Path,
    csv_path: Path,
    concs: list[int],
    isl: int,
    osl: int,
    opt_args: str,
    opt_envs: dict[str, str],
    workspace: Path,
    started_at: float,
    total_budget_sec: int | None,
    has_budget: bool,
    budget_exhausted: bool,
    budget_skip_reason: str,
    budget_remaining_sec: float | None,
    partial: bool = True,
    recorder: Any = None,
) -> None:
    """Build and flush an incremental payload from the results collected so far.

    Extracts partial baseline/optimized points from *results*, builds a minimal
    in-progress payload, sets ``report_json_path`` / ``report_csv_path``, and
    delegates to :func:`_flush_conc_sweep_report`.

    ``partial`` sets the status to ``"in_progress"`` rather than a terminal
    one, distinguishing an incremental checkpoint from a final write. The pair
    table is recorded on the same beat as this flush, so an event read
    mid-sweep carries the pairs measured so far rather than nothing.
    """
    try:
        b_pts: list[dict[str, Any]] = []
        o_pts: list[dict[str, Any]] = []
        for v in results:
            if v.name.startswith("baseline_"):
                b_pts.append(_point_from_variant(v, arm="baseline"))
            elif v.name.startswith("optimized_"):
                o_pts.append(_point_from_variant(v, arm="optimized"))
        b_pts.sort(key=lambda p: p["conc"])
        o_pts.sort(key=lambda p: p["conc"])

        comparison, summary = conc_pair_comparison(
            b_pts, o_pts, metric_key=graded_metric_key(benchmark_mode=str(getattr(state, "benchmark_mode", "") or ""))
        )
        if recorder is not None:
            recorder.record_progress(comparison=comparison, summary=summary)
        p: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "in_progress" if partial else "unknown",
            "session_id": str(getattr(state, "session_id", "") or session_dir.name),
            "isl": isl,
            "osl": osl,
            "tp": int(getattr(state, "tp", 0) or 0),
            "benchmark_mode": str(getattr(state, "benchmark_mode", "") or ""),
            "concs_requested": concs,
            "baseline": {"extra_server_args": "", "extra_envs": {}, "points": b_pts},
            "optimized": {"extra_server_args": opt_args, "extra_envs": opt_envs, "points": o_pts},
            "comparison": comparison,
            "summary": summary,
            "workspace": workspace.as_posix(),
            "elapsed_sec": round(time.time() - started_at, 2),
            "total_budget_sec": total_budget_sec if has_budget else None,
            "budget_exhausted": budget_exhausted,
            "report_json_path": json_path.as_posix(),
            "report_csv_path": csv_path.as_posix(),
        }
        if budget_exhausted:
            p["budget_skip_reason"] = budget_skip_reason
            if budget_remaining_sec is not None:
                p["budget_remaining_sec"] = round(float(budget_remaining_sec), 2)
        _flush_conc_sweep_report(p, session_dir)
    except Exception:  # noqa: BLE001
        log.debug("conc_sweep: _flush_partial_conc_sweep_report failed", exc_info=True)


def _skip(reason: str, **extras: Any) -> dict[str, Any]:
    """Build a non-fatal skip envelope. Reason is operator-readable."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "skipped",
        "skip_reason": reason,
    }
    payload.update(extras)
    return payload


def _declined(recorder: Any, reason: str, **extras: Any) -> dict[str, Any]:
    """Build a skip envelope and close the sweep's event on it.

    A sweep that declines is still a sweep that was dispatched. Routing every
    pre-flight refusal through here means none can be added later without the
    event learning about it.
    """
    payload = _skip(reason, **extras)
    if recorder is not None:
        recorder.record_declined(payload)
    return payload


def conc_sweep_declined_to_run(record: Mapping[str, Any] | None) -> bool:
    """Whether a conc-sweep record is one that never started a variant."""
    rec = record or {}
    return bool(rec.get("was_skipped")) and not rec.get("budget_exhausted")


async def run_conc_sweep(
    state: SharedState,
    session_dir: Path,
    *,
    concs: list[int] | None = None,
    variant_timeout_sec: int = DEFAULT_VARIANT_TIMEOUT_SEC,
    total_budget_sec: int | None = DEFAULT_TOTAL_BUDGET_SEC,
    num_prompts_factor: int = DEFAULT_NUM_PROMPTS_FACTOR,
    write_reports: bool = True,
    recorder: Any = None,
) -> dict[str, Any]:
    """Run the full conc-sweep SWEEP-phase action end-to-end (always returns a dict; never raises; no files written when skipped).

    ``concs`` of ``None`` uses the default ladder. ``total_budget_sec`` of
    ``None`` runs the ladder unbounded, while ``<=0`` means the caller's clamp
    left no time and the sweep skips immediately. A ``None`` recorder records
    nothing, which is what a direct caller with no session bound wants.
    Returns a skip envelope when prerequisites are unmet.
    """
    session_dir = Path(session_dir)
    # Whether the ladder was handed to the sweep or picked for the workload --
    # a distinction only this line can still see, since the two are the same
    # list one statement later.
    grid_source = GRID_REQUESTED if concs is not None else GRID_MODE_DEFAULT
    # ``None`` → default ladder; an explicit empty list short-circuits below.
    concs = list(concs) if concs is not None else default_concs_for_mode(getattr(state, "benchmark_mode", ""))
    isl = int(getattr(state, "isl", 0) or 0)
    osl = int(getattr(state, "osl", 0) or 0)
    baseline_tput = float(getattr(state, "baseline_tput", 0.0) or 0.0)

    has_opt, opt_args, opt_envs = _has_optimization(state)

    if baseline_tput <= 0:
        return _declined(recorder, "no_baseline_tput")
    if isl <= 0 or osl <= 0:
        return _declined(recorder, "missing_workload_shape", isl=isl, osl=osl)
    if not has_opt:
        return _declined(recorder, "no_optimization_to_compare")
    opt_overlay = str((state.current_best or {}).get("final_overlay") or "").strip()
    if opt_overlay:
        from ..actions.executors._grid_runner import _is_safe_path_entry
        from ..loop.coordinator_helpers import _geak_overlay_is_loadable

        if not _is_safe_path_entry(opt_overlay) or not _geak_overlay_is_loadable(opt_overlay):
            return _declined(recorder, "optimized_overlay_unavailable", final_overlay=opt_overlay)
    if not concs:
        return _declined(recorder, "empty_conc_list")
    # A non-positive budget is "no time left", not "budget gate off": running the
    # ladder here would spend wall-clock the caller already accounted as gone.
    # No variant started, so this is a decline (see conc_sweep_declined_to_run)
    # and must not stamp ``budget_exhausted``.
    if total_budget_sec is not None and int(total_budget_sec) <= 0:
        return _declined(recorder, "no_time_budget_remaining", total_budget_sec=int(total_budget_sec))

    # Prefer the materialized baseline config; fall back to the shipped asset.
    base_yaml_raw = str(getattr(state, "baseline_config_path", "") or "").strip() or str(default_baseline_config())
    base_yaml_path = Path(base_yaml_raw)
    if not base_yaml_path.exists():
        return _declined(recorder, "baseline_config_missing", config_path=base_yaml_raw)

    task_id = f"conc_sweep_{utc_now_compact()}"
    workspace = runs_root(session_dir) / "conc_sweep" / task_id
    workspace.mkdir(parents=True, exist_ok=True)

    # Re-materialize (idempotent) in case we fell back to the shipped asset.
    resolved_model = resolve_session_model_path(
        state_model_path=str(getattr(state, "model_path", "") or ""),
        for_serving=True,
    )
    # Mirror the main flow (baseline/sweep/...): prefer $GPU_TYPE (cli.py canonicalizes mi325x/mi308x -> mi300x), fall
    # back to state.gpu_type, then canonicalize through _gpu_runner_type so the selected Magpie script is a shipped
    # runner (sglang_mi300x.sh), never the unshipped sglang_mi325x.sh.
    from hyperloom.inference_optimizer.gpu_types import _gpu_runner_type

    resolved_gpu = _gpu_runner_type(
        os.environ.get("GPU_TYPE", "").strip().lower() or str(getattr(state, "gpu_type", "") or "").strip().lower()
    )
    benchmark_script = baseline_benchmark_script(state.last_baseline)
    try:
        base_yaml_path = materialize_config_with_envs(
            base_yaml_path,
            workspace,
            model_path=resolved_model or None,
            gpu_type=resolved_gpu or None,
            benchmark_script=benchmark_script,
            out_name="conc_sweep_base.with_envs.yaml",
        )
    except FrameworkScriptMismatchError as exc:
        return _declined(
            recorder,
            "framework_script_mismatch",
            error_class="framework_script_mismatch",
            error=str(exc),
            workspace=str(workspace),
        )

    if recorder is not None:
        cb = state.current_best if isinstance(getattr(state, "current_best", None), dict) else {}
        recorder.record_workload(
            session_id=getattr(state, "session_id", "") or session_dir.name,
            isl=isl,
            osl=osl,
            tp=getattr(state, "tp", 0),
            benchmark_mode=getattr(state, "benchmark_mode", ""),
        )
        recorder.record_anchor(
            baseline_tput=baseline_tput,
            anchor_tput=cb.get("tput"),
            tp=getattr(state, "tp", 0),
            variant_id=cb.get("variant_name"),
            action=cb.get("action"),
            extra_server_args=opt_args,
            extra_envs=opt_envs,
        )
        recorder.record_environment(
            sweep_task_id=task_id,
            workspace=workspace.as_posix(),
            model_path=resolved_model,
            gpu_type=resolved_gpu,
            base_config_path=base_yaml_path.as_posix(),
            report_json_path=(reports_dir(session_dir) / "conc_sweep_summary.json").as_posix(),
            report_csv_path=(reports_dir(session_dir) / "conc_sweep_raw.csv").as_posix(),
        )

    # The module default is synthetic-sized and cannot fund a single AgentX rung.
    # ``_granted_cap_sec`` prices a rung at what ``run_grid`` will actually grant
    # it, which under AgentX is the raised cap (10800s at canonical settings) --
    # larger than DEFAULT_TOTAL_BUDGET_SEC (9000s) on its own, so the whole
    # ladder would be skipped with zero measurements. The CLI already raises
    # this knob for AgentX; a caller reaching ``run_conc_sweep`` directly got
    # the synthetic default. Give it the same floor here, but only when the
    # caller left the default in place -- a number the operator chose is never
    # overridden. Safe to raise: this is the action's own slice, and the
    # session deadline still clamps it via ``_session_soft_dl`` below, which
    # is why the price is computed once and reused rather than asked twice.
    _rung_cost = _granted_cap_sec(variant_timeout_sec, state)
    declared_total_budget_sec = total_budget_sec
    budget_raised = False
    if total_budget_sec is not None and int(total_budget_sec) == DEFAULT_TOTAL_BUDGET_SEC:  # noqa: SIM102
        if _rung_cost > float(total_budget_sec):
            _raised = int(_rung_cost * _AGENTX_MIN_FUNDED_RUNGS)
            budget_raised = True
            log.warning(
                "conc_sweep: the default total budget %ds cannot fund even one rung at "
                "the granted cap %.0fs, so every rung would be skipped as "
                "insufficient_remaining_for_variant. Raising the budget to %ds (%d rungs) "
                "for this AgentX sweep. Pass --conc-sweep-total-budget-sec to size it "
                "yourself; the session deadline still clamps whatever is set here.",
                total_budget_sec,
                _rung_cost,
                _raised,
                _AGENTX_MIN_FUNDED_RUNGS,
            )
            total_budget_sec = _raised

    has_budget = total_budget_sec is not None
    started_at = time.time()
    deadline = started_at + total_budget_sec if has_budget else None

    # Pre-compute report paths so incremental checkpoints carry them.
    rdir = reports_dir(session_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    json_path = rdir / "conc_sweep_summary.json"
    csv_path = rdir / "conc_sweep_raw.csv"

    # Compute session soft_deadline once (used in both paths).
    _SESSION_CLOSE_RESERVE_SEC = 120.0
    _session_soft_dl: float | None = None
    _session_rem_fn = getattr(state, "remaining_minutes", None)
    if callable(_session_rem_fn):
        _sr = _session_rem_fn()
        if _sr is not None:
            _sr_sec = _sr * 60.0
            _clamped = max(0.0, _sr_sec - _SESSION_CLOSE_RESERVE_SEC)
            _session_soft_dl = min(_rung_cost, _clamped) if _clamped > 0 else None

    if recorder is not None:
        recorder.record_budget(
            declared_total_sec=declared_total_budget_sec,
            granted_total_sec=total_budget_sec,
            rung_cost_sec=_rung_cost,
            raised=budget_raised,
            gate_active=has_budget,
            deadline=deadline,
            session_soft_deadline_sec=_session_soft_dl,
        )

    results: list[VariantResult] = []
    budget_exhausted = False
    budget_skip_reason = ""
    budget_remaining_sec: float | None = None

    # Arm-major single-server path.
    concs_desc = _order_concs_desc(concs)
    log.info(
        "conc_sweep (single-server): arms=optimized,baseline concs=%s isl=%d osl=%d total_budget=%s",
        concs_desc,
        isl,
        osl,
        f"{total_budget_sec}s" if has_budget else "unbounded",
    )
    _budget_state: dict[str, Any] = {
        "budget_exhausted": budget_exhausted,
        "budget_skip_reason": budget_skip_reason,
        "budget_remaining_sec": budget_remaining_sec,
    }
    arms_order = [
        ("optimized", opt_args, dict(opt_envs)),
        ("baseline", "", {}),
    ]
    if recorder is not None:
        recorder.record_plan(
            concs_requested=concs,
            concs_ordered=concs_desc,
            grid_source=grid_source,
            num_prompts_factor=num_prompts_factor,
            variant_timeout_sec=variant_timeout_sec,
            arms_order=[name for name, _args, _envs in arms_order],
        )
    try:
        for arm_name, arm_args, arm_envs in arms_order:
            skip_grid_fn = lambda _an=arm_name, _aa=arm_args, _ae=arm_envs: _build_arm_grid(  # noqa: E731
                _an,
                concs_desc,
                isl=isl,
                osl=osl,
                num_prompts_factor=num_prompts_factor,
                arm_args=_aa,
                arm_envs=_ae,
                overlay_pythonpath=opt_overlay if _an == "optimized" else "",
            )

            # Check overall budget before starting each arm.
            _arm_remaining = (deadline - time.time()) if has_budget and deadline is not None else None
            if has_budget and _arm_remaining is not None and _arm_remaining <= 0:
                _budget_state["budget_exhausted"] = True
                _budget_state["budget_skip_reason"] = "total_budget_exhausted"
                _budget_state["budget_remaining_sec"] = max(0.0, float(_arm_remaining))
                for v in skip_grid_fn():
                    results.append(_budget_skip_result(v))
                if recorder is not None:
                    recorder.record_arm_refused(
                        arm_name,
                        reason="total_budget_exhausted",
                        remaining_sec=max(0.0, float(_arm_remaining)),
                    )
                continue
            if has_budget and _arm_remaining is not None and _arm_remaining < _rung_cost:
                _budget_state["budget_exhausted"] = True
                _budget_state["budget_skip_reason"] = "insufficient_remaining_for_variant"
                _budget_state["budget_remaining_sec"] = max(0.0, float(_arm_remaining))
                for v in skip_grid_fn():
                    results.append(_budget_skip_result(v))
                if recorder is not None:
                    recorder.record_arm_refused(
                        arm_name,
                        reason="insufficient_remaining_for_variant",
                        remaining_sec=max(0.0, float(_arm_remaining)),
                    )
                continue
            if getattr(state, "closing_phase", False) or getattr(state, "stop_reason", ""):
                _budget_state["budget_exhausted"] = True
                _budget_state["budget_skip_reason"] = "session_deadline_reserve"
                _budget_state["budget_remaining_sec"] = 0.0
                for v in skip_grid_fn():
                    results.append(_budget_skip_result(v))
                if recorder is not None:
                    recorder.record_arm_refused(arm_name, reason="session_deadline_reserve", remaining_sec=0.0)
                continue

            if recorder is not None:
                recorder.open_arm(arm_name, extra_server_args=arm_args, extra_envs=arm_envs)
            await _sweep_one_arm_single_server(
                arm_name,
                concs_desc,
                isl=isl,
                osl=osl,
                num_prompts_factor=num_prompts_factor,
                arm_args=arm_args,
                arm_envs=arm_envs,
                base_yaml_path=base_yaml_path,
                workspace=workspace,
                model_path=resolved_model,
                gpu_type=resolved_gpu,
                benchmark_script=benchmark_script,
                variant_timeout_sec=variant_timeout_sec,
                soft_deadline_sec=_session_soft_dl,
                deadline=deadline,
                state=state,
                session_dir=session_dir,
                json_path=json_path,
                csv_path=csv_path,
                started_at=started_at,
                total_budget_sec=total_budget_sec,
                has_budget=has_budget,
                opt_args=opt_args,
                opt_envs=opt_envs,
                _all_results_ref=results,
                _budget_state=_budget_state,
                recorder=recorder,
            )
            # Results are added to `results` in place by _all_results_ref.
    finally:
        # Safety net, independent of each arm's own per-variant teardown: by the time both arms have run (or one
        # raised/was cut short), nothing this conc_sweep started should still be alive -- each arm's own server is
        # only ever kept warm *between* its own CONC-ladder rounds, never past the arm itself.
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                await asyncio.to_thread(_kill_stale_servers)
            except Exception:  # noqa: BLE001 - best-effort safety net
                log.warning(
                    "conc_sweep: post-run _kill_stale_servers failed",
                    exc_info=True,
                )

    budget_exhausted = _budget_state["budget_exhausted"]
    budget_skip_reason = _budget_state["budget_skip_reason"]
    budget_remaining_sec = _budget_state["budget_remaining_sec"]

    elapsed_sec = time.time() - started_at

    # Split by arm via the variant name prefix.
    baseline_points: list[dict[str, Any]] = []
    optimized_points: list[dict[str, Any]] = []
    for vres in results:
        if vres.name.startswith("baseline_"):
            baseline_points.append(_point_from_variant(vres, arm="baseline"))
        elif vres.name.startswith("optimized_"):
            optimized_points.append(_point_from_variant(vres, arm="optimized"))
    baseline_points.sort(key=lambda p: p["conc"])
    optimized_points.sort(key=lambda p: p["conc"])

    comparison, summary = conc_pair_comparison(
        baseline_points,
        optimized_points,
        metric_key=graded_metric_key(benchmark_mode=str(getattr(state, "benchmark_mode", "") or "")),
    )
    budget_limited_no_pair = _budget_limited_without_valid_pair(
        budget_exhausted=budget_exhausted,
        summary=summary,
        baseline_points=baseline_points,
        optimized_points=optimized_points,
    )
    status = "succeeded" if summary["successful_pairs"] else ("skipped" if budget_limited_no_pair else "failed")

    ceiling = _build_roofline_ceiling(
        state,
        concs=concs,
        isl=isl,
        osl=osl,
        baseline_points=baseline_points,
        optimized_points=optimized_points,
    )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "session_id": str(getattr(state, "session_id", "") or session_dir.name),
        "isl": isl,
        "osl": osl,
        "tp": int(getattr(state, "tp", 0) or 0),
        # Names the axis pair the points are drawn on, so a reader never has to infer it from whether
        # e2e_norm_intvty_p90 happens to be null.
        "benchmark_mode": str(getattr(state, "benchmark_mode", "") or ""),
        "concs_requested": concs,
        "baseline": {
            "extra_server_args": "",
            "extra_envs": {},
            "points": baseline_points,
        },
        "optimized": {
            "extra_server_args": opt_args,
            "extra_envs": opt_envs,
            "points": optimized_points,
        },
        "comparison": comparison,
        "summary": summary,
        "workspace": workspace.as_posix(),
        "elapsed_sec": round(elapsed_sec, 2),
        "total_budget_sec": total_budget_sec if has_budget else None,
        "budget_exhausted": budget_exhausted,
    }
    if budget_limited_no_pair:
        payload["was_skipped"] = True
        payload["skip_reason"] = "budget_exhausted_no_successful_pairs"
    if budget_exhausted:
        payload["budget_skip_reason"] = budget_skip_reason
        payload["budget_remaining_sec"] = round(float(budget_remaining_sec or 0.0), 2)
    if ceiling is not None:
        payload["roofline_ceiling"] = ceiling

    if write_reports:
        # Set self-referential paths before the dump so the JSON carries them.
        payload["report_json_path"] = json_path.as_posix()
        payload["report_csv_path"] = csv_path.as_posix()
        _flush_conc_sweep_report(payload, session_dir)

    if recorder is not None:
        recorder.record_progress(comparison=comparison, summary=summary)
        recorder.finish(payload, stop_reason=getattr(state, "stop_reason", ""))

    log.info(
        "conc_sweep: done — successful_pairs=%d failed_pairs=%d best_speedup=%s",
        summary["successful_pairs"],
        summary["failed_pairs"],
        summary["best_speedup"],
    )
    return payload


__all__ = [
    "AGENTX_DEFAULT_CONCS",
    "DEFAULT_CONCS",
    "DEFAULT_NUM_PROMPTS_FACTOR",
    "DEFAULT_TOTAL_BUDGET_SEC",
    "DEFAULT_VARIANT_TIMEOUT_SEC",
    "SCHEMA_VERSION",
    "_build_arm_grid",
    "_flush_conc_sweep_report",
    "_flush_partial_conc_sweep_report",
    "_order_concs_desc",
    "conc_sweep_declined_to_run",
    "default_concs_for_mode",
    "run_conc_sweep",
]
