# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Concurrency sweep over the CONC ladder.

Runs the Magpie grid over CONC values for baseline vs ``current_best``,
producing JSON/CSV curves. Auto-enqueued by the Coordinator as a SWEEP-phase
action on SWEEP entry (opt out via ``--no-enable-conc-sweep``); never
LLM-proposable.
"""

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
from hyperloom.common.timeutil import utc_now_compact
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


log = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"

# Default ladders, one per workload (override via ``--conc-sweep-concs``).
# The synthetic ladder halves from a concurrency the 1024/1024 shape saturates;
# an agentic request carries a measured ISL p50 near 108k tokens, so the same
# card runs out of KV cache two orders of magnitude lower and its rungs are
# spaced across the interactivity range the chart is drawn over.
DEFAULT_CONCS: list[int] = [256, 128, 64, 32, 16, 8, 4, 2]
AGENTX_DEFAULT_CONCS: list[int] = [1, 4, 8, 10, 14, 20, 28]


def default_concs_for_mode(benchmark_mode: Any = "") -> list[int]:
    """The ladder a mode sweeps when the operator names none.

    Args:
        benchmark_mode: ``SharedState.benchmark_mode``; anything that is not
            ``agentx`` reads as the synthetic workload.

    Returns:
        A copy of the ladder, safe for the caller to mutate.
    """
    return list(AGENTX_DEFAULT_CONCS if is_agentx_mode(benchmark_mode) else DEFAULT_CONCS)


# Multiplier applied to each CONC for NUM_PROMPTS.
DEFAULT_NUM_PROMPTS_FACTOR = 5

# Per-variant timeout (seconds); override via ``--conc-sweep-timeout-sec``.
# Synthetic-sized, like every other variant-timeout default here; under AgentX
# ``agentx_variant_timeout_sec`` raises it at the point of use, so this number
# is a floor for the synthetic sweep rather than a bound on an agentic round.
DEFAULT_VARIANT_TIMEOUT_SEC = 1800

# Total wall-clock budget (seconds); override via ``--conc-sweep-total-budget-sec``.
# ``None`` disables the gate; ``<=0`` means no time is left to spend.
DEFAULT_TOTAL_BUDGET_SEC = 9000

# How many rungs the AgentX floor below buys when the default budget cannot fund
# even one. Two, not the full ladder: the point is to make the sweep produce a
# comparison instead of nothing, not to silently authorize twelve hours of GPU.
# A caller who wants the whole ladder passes --conc-sweep-total-budget-sec.
_AGENTX_MIN_FUNDED_RUNGS = 2


def _granted_cap_sec(variant_timeout_sec: int, shared_state: Any = None, conc: int | None = None) -> float:
    """What a variant will actually be granted, for budget arithmetic.

    Every budget gate in this module used to price a variant at the DECLARED
    ``variant_timeout_sec`` -- 1800s, sized for the synthetic 1024/1024 shape.
    ``run_grid`` does not hand the round that number: under AgentX it raises the
    cap to what an agentic round needs before launching. Pricing at 1800s while
    granting 10800s admits a variant the budget cannot pay for, and the round
    then has its cap clamped back down to the remaining time and is killed
    mid-warmup -- the exact failure the cap-raise exists to prevent, just moved
    from the grid runner into the sweep's admission check.

    The same number is also the ceiling on the session soft deadline, for the
    same reason: a soft deadline of 1800s ends an agentic round that has not
    reached its measurement window yet.

    With AgentX off this returns ``variant_timeout_sec`` untouched, so the
    synthetic sweep prices and paces exactly as it did before.

    ``shared_state`` carries the durable AgentX signal. A session resumed into a
    shell that lost HYPERLOOM_AGENTX would otherwise price every rung as
    synthetic here and then have the round granted the raised cap anyway -- the
    two sides disagreeing again, in the direction that admits a rung the budget
    cannot pay for.

    ``conc`` prices the rung about to launch rather than the session; omitted
    where the gate guards a whole arm rather than a single rung.

    Args:
        variant_timeout_sec: The declared per-variant hard timeout, in seconds.
        shared_state: Session state; consulted only when the env var is absent.
        conc: The rung's concurrency, when the gate guards one rung.

    Returns:
        float: The cap the round will actually be granted, in seconds.
    """
    return float(agentx_variant_timeout_sec(variant_timeout_sec, shared_state=shared_state, conc=conc))


def _has_optimization(state: SharedState) -> tuple[bool, str, dict[str, str]]:
    """Return ``(has_opt, args, envs)`` from ``state.current_best`` (either non-empty side counts as optimized).

    Args:
        state: Shared run state whose ``current_best`` is inspected.

    Returns:
        A tuple of ``(has_optimization, extra_server_args, extra_envs)``.
    """
    cb = state.current_best or {}
    args = str(cb.get("extra_server_args") or "").strip()
    envs_raw = cb.get("extra_envs") or {}
    envs = {str(k): str(v) for k, v in envs_raw.items()}
    return (bool(args) or bool(envs)), args, envs


def _budget_skip_result(variant: GridVariant) -> VariantResult:
    """Synthetic VariantResult for a budget-exhausted variant; ``skipped`` status distinguishes "out of time" from "Magpie crashed".

    Args:
        variant: The grid variant that did not get to run.

    Returns:
        A ``VariantResult`` marked ``skipped`` with a budget-exhausted error.
    """
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
    """Flatten a ``VariantResult`` into one row of the curve.

    Args:
        v: The variant result to flatten.
        arm: The arm label (e.g. ``baseline`` or ``optimized``) for the row.

    Returns:
        A dict of the variant's metrics keyed for the curve row.

    ``intvty_p90`` and ``total_token_throughput`` are the pair an agentic run is
    plotted on; they are null on a synthetic run, which is plotted on the
    output-throughput pair instead.
    """
    envs = v.extra_envs or {}
    try:
        conc = int(envs.get("CONC", "0"))
    except (TypeError, ValueError):
        conc = 0
    # aiperf reports the total; the other parsers pass through whatever the
    # framework named, leaving it null on a run that measured both halves. The
    # sum is the same identity ``perf_snapshot_from_mapping`` applies, and a
    # session graded on the total axis fails outright without it.
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
        "intvty_p90": v.intvty_p90,
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
    """One row per (arm, conc) — flat columns for spreadsheet pivots.

    Args:
        csv_path: Destination CSV path (parent dirs are created).
        points: Curve rows to write, one per (arm, conc).
    """
    columns = [
        "arm",
        "conc",
        "status",
        "output_throughput",
        "request_throughput",
        "total_token_throughput",
        "input_throughput",
        "intvty_p90",
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
    """Per-conc decode roofline alongside the measured curves.

    T_cmp is computed once, T_mem per CONC; MBU% = measured/T_peak×100. Returns
    ``None`` when model meta / GPU spec is unavailable.

    Args:
        state: Shared run state providing model path, precision, GPU type, TP.
        concs: Concurrency values to compute ceilings for.
        isl: Input sequence length.
        osl: Output sequence length.
        baseline_points: Measured baseline curve rows (for MBU%).
        optimized_points: Measured optimized curve rows (for MBU%).

    Returns:
        A roofline ceiling payload dict, or ``None`` when model meta or GPU
        spec is unavailable.
    """
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
            """Express a measured throughput as a percent of peak.

            Args:
                measured: Measured throughput value.

            Returns:
                The ratio to ``t_peak`` as a percentage, or ``None`` when
                inputs are non-positive or invalid.
            """
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


def _order_concs_desc(concs: list[int]) -> list[int]:
    """Return a strictly descending, deduplicated copy of the CONC ladder.

    Descending order is required for single-server arm sweeps: the server is
    booted with the highest (most demanding) concurrency first so it can handle
    all lower values without restart.

    Args:
        concs: Requested concurrency ladder (any order, may contain duplicates).

    Returns:
        A deduplicated list sorted in strictly descending order.
    """
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
) -> list[GridVariant]:
    """Build a single-arm grid in descending CONC order.

    Args:
        arm_name: Arm label (e.g. ``baseline`` or ``optimized``).
        concs_desc: Concurrency values in strictly descending order.
        isl: Input sequence length.
        osl: Output sequence length.
        num_prompts_factor: Multiplier applied to each CONC for NUM_PROMPTS.
        arm_args: Extra server args for this arm.
        arm_envs: Extra environment variables for this arm.

    Returns:
        List of grid variants for the arm, one per CONC in descending order.
    """
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
        out.append(
            GridVariant(
                name=f"{arm_name}_conc{conc}",
                extra_server_args=arm_args,
                extra_envs=envs,
                note=f"arm={arm_name} conc={conc} isl={isl} osl={osl}",
            )
        )
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
) -> list[VariantResult]:
    """Sweep one arm across all CONC values reusing a single persistent server.

    Boots the server on the highest CONC (Option A), then reuses it for all
    lower CONCs.  If boot fails, retries with the next lower CONC
    (boot-retry-descend).  Falls back to the legacy per-variant server-restart
    path (Option B) when all boot retries are exhausted.

    The shared ``_all_results_ref`` list is mutated in place so incremental
    checkpoints always see the latest cross-arm view.

    Args:
        arm_name: Label for this arm (``baseline`` or ``optimized``).
        concs_desc: Concurrency ladder in strictly descending order.
        isl: Input sequence length.
        osl: Output sequence length.
        num_prompts_factor: CONC multiplier for NUM_PROMPTS.
        arm_args: Extra server args for this arm.
        arm_envs: Extra environment variables for this arm.
        base_yaml_path: Materialized base Magpie YAML path.
        workspace: Per-sweep workspace root.
        model_path: Resolved model path string.
        gpu_type: Resolved GPU type string.
        variant_timeout_sec: Per-variant hard timeout in seconds.
        soft_deadline_sec: Session-clamped soft deadline in seconds, or None.
        deadline: Absolute wall-clock epoch (``time.time()`` basis) at which the
            total conc-sweep budget expires, or None when unbounded. Distinct
            from ``soft_deadline_sec``, which is a duration.
        state: Shared run state (for incremental checkpoint metadata).
        session_dir: Session directory (for incremental checkpoints).
        json_path: Pre-resolved JSON report path.
        csv_path: Pre-resolved CSV report path.
        started_at: Wall-clock start of the overall sweep (for elapsed_sec).
        total_budget_sec: Configured total budget in seconds (``None`` when
            the budget gate is off).
        has_budget: Whether budget tracking is active.
        opt_args: Optimized server args (for payload metadata).
        opt_envs: Optimized server env vars (for payload metadata).
        _all_results_ref: Shared list of all results collected across arms;
            mutated in place so incremental flushes have the full cross-arm
            picture.
        _budget_state: Mutable dict carrying budget flags (``budget_exhausted``,
            ``budget_skip_reason``, ``budget_remaining_sec``) shared with the
            caller so the main function can inspect the final budget status.

    Returns:
        List of VariantResult for this arm (one per CONC).
    """
    from ..actions.executors._grid_runner import _num_gpus_for_config
    from ..actions.executors._ray_serving import maybe_serving_lease
    from ..actions.executors._server_lifecycle import (
        resolve_lifecycle_params,
        teardown_lifecycle_server,
    )

    arm_results: list[VariantResult] = []
    grid = _build_arm_grid(
        arm_name,
        concs_desc,
        isl=isl,
        osl=osl,
        num_prompts_factor=num_prompts_factor,
        arm_args=arm_args,
        arm_envs=arm_envs,
    )
    if not grid:
        return arm_results

    # Ray-managed GPU execution: one held Ray lease (``num_gpus=TP``)
    # spans this arm's persistent server — boot + every CONC reuse round, or the
    # Option B per-variant restarts — so the shared server's whole lifetime is
    # covered by a single lease and no GPU process outlives it. ``None`` on the
    # local path (multi-node / RAY_EXEC off / tests) keeps the legacy behaviour.
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

    if not lc_eligible:
        # Framework does not support server_lifecycle — fall through to
        # Option B (per-variant server restart via normal run_grid).
        log.info(
            "conc_sweep single-server: arm=%s not lifecycle-eligible (%s); using per-variant server restart (Option B)",
            arm_name,
            lc_reason,
        )
        try:
            return await _sweep_arm_option_b(
                arm_name=arm_name,
                grid=grid,
                base_yaml_path=base_yaml_path,
                workspace=workspace,
                model_path=model_path,
                gpu_type=gpu_type,
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
            )
        finally:
            if arm_lease is not None:
                arm_lease.close()

    # Boot-retry-descend: try each CONC from highest to lowest until boot succeeds.
    # Failed higher-CONC boots are tracked locally; they are only committed to the
    # permanent results when a *lower* CONC eventually boots (they represent genuine
    # capacity failures, e.g. OOM at that CONC). If every CONC fails to boot, the
    # whole grid is retried via Option B instead so nothing is double-counted.
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
        try:
            boot_results = await run_grid(
                base_yaml_path=base_yaml_path,
                base_extra_args="",
                grid=[boot_variant],
                output_root=workspace,
                variant_timeout_sec=variant_timeout_sec,
                model_path=model_path,
                gpu_type=gpu_type,
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

        if boot_failed:
            # Ensure server is torn down before retrying at a lower CONC.
            try:
                teardown_lifecycle_server(pid_dir=pid_dir, framework=framework, port=port)
            except Exception:  # noqa: BLE001
                pass
            failed_boots.append(
                br
                or VariantResult(
                    name=boot_variant.name,
                    extra_server_args=boot_variant.extra_server_args,
                    extra_envs=dict(boot_variant.extra_envs),
                    status="failed",
                    error="single_server_boot_failed",
                    error_class="single_server_boot_failed",
                )
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
        arm_results.append(br)
        _all_results_ref.append(br)
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
        )
        break

    if not boot_succeeded:
        # Every CONC failed to boot the persistent server — retry the full grid
        # via Option B (per-variant restart, no lifecycle) which may succeed where
        # persistent reuse could not. Its results supersede the failed boot attempts.
        log.warning(
            "conc_sweep single-server: arm=%s all boot attempts failed; "
            "falling back to Option B (per-variant restart) for the full ladder",
            arm_name,
        )
        ob_results = await _sweep_arm_option_b(
            arm_name=arm_name,
            grid=grid,
            base_yaml_path=base_yaml_path,
            workspace=workspace,
            model_path=model_path,
            gpu_type=gpu_type,
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
        )
        arm_results.extend(ob_results)
        if arm_lease is not None:
            arm_lease.close()
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
                break

            is_last = r_idx == len(reuse_grid) - 1
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
            for rr in reuse_results:
                arm_results.append(rr)
                _all_results_ref.append(rr)
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
            )
    finally:
        # Safety teardown — idempotent, no-op if already torn down. Reap the
        # server BEFORE releasing the Ray lease so no GPU process outlives it.
        try:
            teardown_lifecycle_server(pid_dir=pid_dir, framework=framework, port=port)
        except Exception:  # noqa: BLE001
            pass
        if arm_lease is not None:
            arm_lease.close()

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
) -> list[VariantResult]:
    """Option B fallback: run each variant with its own server (legacy behaviour).

    Used when ``_sweep_one_arm_single_server`` detects the framework is not
    lifecycle-eligible or all boot retries are exhausted.

    Args:
        arm_name: Label for this arm (``baseline`` or ``optimized``).
        grid: Pre-built grid for this arm.
        base_yaml_path: Materialized base Magpie YAML path.
        workspace: Per-sweep workspace root.
        model_path: Resolved model path string.
        gpu_type: Resolved GPU type string.
        variant_timeout_sec: Per-variant hard timeout in seconds.
        soft_deadline_sec: Session-clamped soft deadline in seconds, or None.
        deadline: Absolute epoch (``time.time()``) at which the total budget
            expires, or None when budget tracking is off (``has_budget`` False).
        state: Shared run state.
        session_dir: Session directory.
        json_path: Pre-resolved JSON report path.
        csv_path: Pre-resolved CSV report path.
        started_at: Wall-clock start of the overall sweep.
        total_budget_sec: Configured total budget in seconds (``None`` when
            the budget gate is off).
        has_budget: Whether budget tracking is active.
        opt_args: Optimized server args.
        opt_envs: Optimized server env vars.
        _all_results_ref: Shared results list (mutated in place).
        _budget_state: Shared budget-status dict (mutated in place).
        serving_lease: Caller-owned Ray serving lease forwarded to ``run_grid``;
            None when the arm runs on the local (non-Ray) path.

    Returns:
        List of VariantResult for the arm (one per CONC).
    """
    arm_results: list[VariantResult] = []
    for variant in grid:
        # Task-level budget checks.
        _ob_rem = (deadline - time.time()) if has_budget and deadline is not None else None
        if has_budget and _ob_rem is not None and _ob_rem <= 0:
            _budget_state["budget_exhausted"] = True
            _budget_state["budget_skip_reason"] = "total_budget_exhausted"
            _budget_state["budget_remaining_sec"] = max(0.0, float(_ob_rem))
            skip_r = _budget_skip_result(variant)
            arm_results.append(skip_r)
            _all_results_ref.append(skip_r)
            continue
        if (
            has_budget
            and _ob_rem is not None
            and _ob_rem < _granted_cap_sec(variant_timeout_sec, state, variant_conc(variant))
        ):
            _budget_state["budget_exhausted"] = True
            _budget_state["budget_skip_reason"] = "insufficient_remaining_for_variant"
            _budget_state["budget_remaining_sec"] = max(0.0, float(_ob_rem))
            skip_r = _budget_skip_result(variant)
            arm_results.append(skip_r)
            _all_results_ref.append(skip_r)
            continue
        if getattr(state, "closing_phase", False) or getattr(state, "stop_reason", ""):
            _budget_state["budget_exhausted"] = True
            _budget_state["budget_skip_reason"] = "session_deadline_reserve"
            _budget_state["budget_remaining_sec"] = 0.0
            skip_r = _budget_skip_result(variant)
            arm_results.append(skip_r)
            _all_results_ref.append(skip_r)
            continue
        try:
            sub = await run_grid(
                base_yaml_path=base_yaml_path,
                base_extra_args="",
                grid=[variant],
                output_root=workspace,
                variant_timeout_sec=variant_timeout_sec,
                model_path=model_path,
                gpu_type=gpu_type,
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
        for r in sub:
            arm_results.append(r)
            _all_results_ref.append(r)
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
) -> None:
    """Build a partial payload from *all_results* and flush it via :func:`_flush_partial_conc_sweep_report`.

    A thin convenience wrapper that avoids repeating the argument list at every
    call site.

    Args:
        state: Shared run state (metadata fields).
        session_dir: Session directory.
        json_path: Pre-resolved JSON report path.
        csv_path: Pre-resolved CSV report path.
        all_results: All results collected so far (cross-arm).
        concs: Full requested concurrency ladder (informational).
        isl: Input sequence length.
        osl: Output sequence length.
        opt_args: Optimized server args.
        opt_envs: Optimized server env vars.
        workspace: Per-sweep workspace root.
        started_at: Wall-clock start of the sweep.
        total_budget_sec: Configured total budget in seconds (``None`` when
            the budget gate is off).
        has_budget: Whether budget tracking is active.
        budget_exhausted: Whether the budget has been exhausted.
        budget_skip_reason: Reason string when budget was exhausted.
        budget_remaining_sec: Remaining budget seconds when exhausted.
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
    )


def _flush_conc_sweep_report(payload: dict[str, Any], session_dir: Path) -> None:
    """Atomically write the conc-sweep summary JSON + CSV to the reports dir.

    Safe to call after each concurrency point: uses an atomic rename so a
    hard kill between write and rename never leaves a partial/corrupt file.
    Internal errors are logged at DEBUG level and swallowed so the sweep loop
    is never interrupted by an IO failure.

    Args:
        payload: The current (possibly partial) sweep payload dict.  Must
            already carry ``report_json_path`` and ``report_csv_path`` keys.
        session_dir: Session directory used to locate the reports sub-dir.
    """
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
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            instrument.record_singleton_section(
                session_dir,
                "conc_sweep_summary",
                payload,
                producer="conc_sweep",
            )
        except Exception:  # noqa: BLE001 — capture must never break the sweep
            log.debug("conc_sweep breakdown capture failed", exc_info=True)
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
) -> None:
    """Build and flush an incremental payload from the results collected so far.

    Extracts partial baseline/optimized points from *results*, builds a minimal
    in-progress payload, sets ``report_json_path`` / ``report_csv_path``, and
    delegates to :func:`_flush_conc_sweep_report`.

    Args:
        results: Variant results collected so far (may be partial).
        state: Shared run state (used for metadata fields).
        session_dir: Session directory for report output.
        json_path: Destination JSON path (already resolved).
        csv_path: Destination CSV path (already resolved).
        concs: Full requested concurrency ladder.
        isl: Input sequence length.
        osl: Output sequence length.
        opt_args: Optimized server args.
        opt_envs: Optimized server env vars.
        workspace: Workspace directory for this sweep run.
        started_at: Wall-clock start time of the sweep.
        total_budget_sec: Total budget in seconds.
        has_budget: Whether budget tracking is active.
        budget_exhausted: Whether the budget has been exhausted.
        budget_skip_reason: Reason string when budget was exhausted.
        budget_remaining_sec: Remaining budget seconds when exhausted.
        partial: When ``True`` the status is set to ``"in_progress"`` rather
            than a terminal status; this makes it easy to distinguish an
            incremental checkpoint from a final write.
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
    """Build a non-fatal skip envelope. Reason is operator-readable.

    Args:
        reason: Operator-readable reason for skipping.
        **extras: Additional key/value fields to merge into the envelope.

    Returns:
        A skip-status payload dict.
    """
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "skipped",
        "skip_reason": reason,
    }
    payload.update(extras)
    return payload


def conc_sweep_declined_to_run(record: Mapping[str, Any] | None) -> bool:
    """Whether a conc-sweep record is one that never started a variant.

    ``was_skipped`` covers two different outcomes: the pre-flight envelope
    from :func:`_skip`, which declines before a server boots, and a sweep that
    ran its whole ladder but exhausted its budget before producing a
    comparable pair (see :func:`_budget_limited_without_valid_pair`). Only the
    second can set ``budget_exhausted``, which separates them without reading
    ``skip_reason``. Any future skip raised after variants start must set it
    too, or it will be misread as a sweep that declined.

    Args:
        record (Mapping[str, Any] | None): A conc-sweep payload or the
            ``last_conc_sweep`` record persisted from one.

    Returns:
        bool: ``True`` when the sweep declined before running anything.
    """
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
) -> dict[str, Any]:
    """Run the full conc-sweep SWEEP-phase action end-to-end (always returns a dict; never raises; no files written when skipped).

    Args:
        state: Shared run state (baseline, current_best, workload shape).
        session_dir: Session directory for workspace and report outputs.
        concs: Concurrency ladder to sweep; ``None`` uses the default ladder.
        variant_timeout_sec: Per-variant timeout in seconds.
        total_budget_sec: Total wall-clock budget in seconds. ``None`` runs the
            ladder unbounded; ``<=0`` means the caller's clamp left no time and
            the sweep skips immediately rather than running unbounded.
        num_prompts_factor: Multiplier applied to each CONC for NUM_PROMPTS.
        write_reports: When ``True``, write the JSON/CSV reports to disk.

    Returns:
        The sweep payload dict (a skip envelope when prerequisites are unmet).
    """
    session_dir = Path(session_dir)
    # ``None`` → default ladder; an explicit empty list short-circuits below.
    concs = list(concs) if concs is not None else default_concs_for_mode(getattr(state, "benchmark_mode", ""))
    isl = int(getattr(state, "isl", 0) or 0)
    osl = int(getattr(state, "osl", 0) or 0)
    baseline_tput = float(getattr(state, "baseline_tput", 0.0) or 0.0)

    has_opt, opt_args, opt_envs = _has_optimization(state)

    if baseline_tput <= 0:
        return _skip("no_baseline_tput")
    if isl <= 0 or osl <= 0:
        return _skip("missing_workload_shape", isl=isl, osl=osl)
    if not has_opt:
        return _skip("no_optimization_to_compare")
    if not concs:
        return _skip("empty_conc_list")
    # A non-positive budget is "no time left", not "budget gate off": running the
    # ladder here would spend wall-clock the caller already accounted as gone.
    # No variant started, so this is a decline (see conc_sweep_declined_to_run)
    # and must not stamp ``budget_exhausted``.
    if total_budget_sec is not None and int(total_budget_sec) <= 0:
        return _skip("no_time_budget_remaining", total_budget_sec=int(total_budget_sec))

    # Prefer the materialized baseline config; fall back to the shipped asset.
    base_yaml_raw = str(getattr(state, "baseline_config_path", "") or "").strip() or str(default_baseline_config())
    base_yaml_path = Path(base_yaml_raw)
    if not base_yaml_path.exists():
        return _skip("baseline_config_missing", config_path=base_yaml_raw)

    task_id = f"conc_sweep_{utc_now_compact()}"
    workspace = runs_root(session_dir) / "conc_sweep" / task_id
    workspace.mkdir(parents=True, exist_ok=True)

    # Re-materialize (idempotent) in case we fell back to the shipped asset.
    resolved_model = resolve_session_model_path(
        state_model_path=str(getattr(state, "model_path", "") or ""),
        for_serving=True,
    )
    # Mirror the main flow (baseline/sweep/...): prefer $GPU_TYPE (cli.py
    # canonicalizes mi325x/mi308x -> mi300x), fall back to state.gpu_type, then
    # canonicalize through _gpu_runner_type so the selected Magpie script is a
    # shipped runner (sglang_mi300x.sh), never the unshipped sglang_mi325x.sh.
    from hyperloom.inference_optimizer.gpu_types import _gpu_runner_type

    resolved_gpu = _gpu_runner_type(
        os.environ.get("GPU_TYPE", "").strip().lower() or str(getattr(state, "gpu_type", "") or "").strip().lower()
    )
    try:
        base_yaml_path = materialize_config_with_envs(
            base_yaml_path,
            workspace,
            model_path=resolved_model or None,
            gpu_type=resolved_gpu or None,
            out_name="conc_sweep_base.with_envs.yaml",
        )
    except FrameworkScriptMismatchError as exc:
        return _skip(
            "framework_script_mismatch",
            error_class="framework_script_mismatch",
            error=str(exc),
            workspace=str(workspace),
        )

    # The module default is synthetic-sized and cannot fund a single AgentX rung.
    # ``_granted_cap_sec`` prices a rung at what ``run_grid`` will actually grant
    # it, which under AgentX is the raised cap (10800s at canonical settings) --
    # larger than DEFAULT_TOTAL_BUDGET_SEC (9000s) on its own. Left alone, the
    # first rung trips "insufficient_remaining_for_variant" and the whole ladder
    # is skipped with zero measurements, which reads like a benchmark failure
    # rather than a budget that was never sized for this workload.
    #
    # The CLI already raises this knob for AgentX; a caller that reaches
    # ``run_conc_sweep`` directly (SDK, tests, any path that does not go through
    # ``_apply_agentx_budget_profile``) got the synthetic default. Give it the
    # same floor here, and only when the caller left the default in place -- a
    # number the operator chose is never overridden. Safe to raise: this is the
    # action's own slice, and the session deadline still clamps it via
    # ``_session_soft_dl`` below.
    if total_budget_sec is not None and int(total_budget_sec) == DEFAULT_TOTAL_BUDGET_SEC:
        _rung_cost = _granted_cap_sec(variant_timeout_sec, state)
        if _rung_cost > float(total_budget_sec):
            _raised = int(_rung_cost * _AGENTX_MIN_FUNDED_RUNGS)
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
            _session_soft_dl = min(_granted_cap_sec(variant_timeout_sec, state), _clamped) if _clamped > 0 else None

    results: list[VariantResult] = []
    budget_exhausted = False
    budget_skip_reason = ""
    budget_remaining_sec: float | None = None

    # Arm-major single-server path.
    # Order: optimized first (more informative for decision-making), then baseline.
    # Within each arm: descending CONC so the server is booted at max capacity.
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
            )

            # Check overall budget before starting each arm.
            _arm_remaining = (deadline - time.time()) if has_budget and deadline is not None else None
            if has_budget and _arm_remaining is not None and _arm_remaining <= 0:
                _budget_state["budget_exhausted"] = True
                _budget_state["budget_skip_reason"] = "total_budget_exhausted"
                _budget_state["budget_remaining_sec"] = max(0.0, float(_arm_remaining))
                for v in skip_grid_fn():
                    results.append(_budget_skip_result(v))
                continue
            if (
                has_budget
                and _arm_remaining is not None
                and _arm_remaining < _granted_cap_sec(variant_timeout_sec, state)
            ):
                _budget_state["budget_exhausted"] = True
                _budget_state["budget_skip_reason"] = "insufficient_remaining_for_variant"
                _budget_state["budget_remaining_sec"] = max(0.0, float(_arm_remaining))
                for v in skip_grid_fn():
                    results.append(_budget_skip_result(v))
                continue
            if getattr(state, "closing_phase", False) or getattr(state, "stop_reason", ""):
                _budget_state["budget_exhausted"] = True
                _budget_state["budget_skip_reason"] = "session_deadline_reserve"
                _budget_state["budget_remaining_sec"] = 0.0
                for v in skip_grid_fn():
                    results.append(_budget_skip_result(v))
                continue

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
            )
            # Results are added to `results` in place by _all_results_ref.
    finally:
        # Safety net, independent of each arm's own per-variant teardown: by
        # the time both arms have run (or one raised/was cut short), nothing
        # this conc_sweep started should still be alive -- each arm's own
        # server is only ever kept warm *between* its own CONC-ladder rounds,
        # never past the arm itself. A round that timed out before its
        # server_lifecycle pidfile was ever written leaves that pidfile-based
        # teardown nothing to find, so this falls back to the broad /proc
        # scan (AMD-AGI/Hyperloom#1354). Skipped under pytest (unsafe there),
        # matching the same guard on the per-launch preclean in
        # _grid_runner.py. Best-effort; never raises.
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
        # Names the axis pair the points are drawn on, so a reader never has to
        # infer it from whether intvty_p90 happens to be null.
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
