# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Post-optimization concurrency sweep.

Runs the Magpie grid over CONC values for baseline vs ``current_best``,
producing JSON/CSV curves. cli.py post-hook (opt out via
``--no-enable-conc-sweep``).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from ..session_paths import reports_dir, runs_root
from .action_executors._grid_runner import (
    GridVariant,
    VariantResult,
    run_grid,
)
from .action_executors._workload_envs import (
    FrameworkScriptMismatchError,
    default_baseline_config,
    materialize_config_with_envs,
)
from .roofline_ceiling import (
    compute_compute_bound_ceiling_tok_per_sec,
    compute_theoretical_peak_output_tok_per_sec,
    load_model_meta,
)
from .shared_state import SharedState


log = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"

# Default ladder (override via ``--conc-sweep-concs``).
DEFAULT_CONCS: list[int] = [1, 2, 4, 8, 16, 32, 64, 128]

# Mirrors ``sweep.py``'s adaptive NUM_PROMPTS heuristic.
DEFAULT_NUM_PROMPTS_FACTOR = 5

# Per-variant timeout (seconds); override via ``--conc-sweep-timeout-sec``.
DEFAULT_VARIANT_TIMEOUT_SEC = 1800

# Total wall-clock budget (seconds); override via ``--conc-sweep-total-budget-sec``, <=0 disables.
DEFAULT_TOTAL_BUDGET_SEC = 9000


def _has_optimization(state: SharedState) -> tuple[bool, str, dict[str, str]]:
    """Return ``(has_opt, args, envs)`` from ``state.current_best`` (either non-empty side counts as optimized)."""
    cb = state.current_best or {}
    args = str(cb.get("extra_server_args") or "").strip()
    envs_raw = cb.get("extra_envs") or {}
    envs = {str(k): str(v) for k, v in envs_raw.items()}
    return (bool(args) or bool(envs)), args, envs


def _build_grid(
    *,
    concs: list[int],
    isl: int,
    osl: int,
    num_prompts_factor: int,
    optimized_args: str,
    optimized_envs: dict[str, str],
) -> list[GridVariant]:
    """Two-arm grid: ``baseline`` × ``optimized`` crossed with every requested CONC."""
    arms: list[tuple[str, str, dict[str, str]]] = [
        ("baseline",  "",              {}),
        ("optimized", optimized_args,  dict(optimized_envs)),
    ]
    out: list[GridVariant] = []
    for arm_name, arm_args, arm_envs in arms:
        for conc in concs:
            num_prompts = max(int(conc) * int(num_prompts_factor), int(conc))
            envs = dict(arm_envs)
            envs.update({
                "CONC":         str(conc),
                "ISL":          str(isl),
                "OSL":          str(osl),
                "NUM_PROMPTS":  str(num_prompts),
            })
            out.append(GridVariant(
                name=f"{arm_name}_conc{conc}",
                extra_server_args=arm_args,
                extra_envs=envs,
                note=f"arm={arm_name} conc={conc} isl={isl} osl={osl}",
            ))
    return out


def _budget_skip_result(variant: GridVariant) -> VariantResult:
    """Synthetic VariantResult for a budget-exhausted variant; ``skipped`` status distinguishes "out of time" from "Magpie crashed"."""
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
    return {
        "arm":                arm,
        "conc":               conc,
        "status":             v.status,
        "output_throughput":  v.output_throughput,
        "request_throughput": v.request_throughput,
        "total_token_throughput": v.total_token_throughput,
        "ttft_mean_ms":       v.ttft_mean_ms,
        "e2el_mean_ms":       v.e2el_mean_ms,
        "duration_seconds":   v.duration_seconds,
        "completed_requests": v.completed_requests,
        "error":              v.error,
        "error_class":        v.error_class,
        "killed_overtime":    v.killed_overtime,
        "workspace":          v.workspace,
        "report_path":        v.report_path,
    }


def _build_comparison(
    baseline_points: list[dict[str, Any]],
    optimized_points: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair points by CONC, compute per-conc speedup, and aggregate.

    Returns ``(per_conc_rows, summary_dict)``.
    """
    by_conc_b = {p["conc"]: p for p in baseline_points}
    by_conc_o = {p["conc"]: p for p in optimized_points}
    concs = sorted(set(by_conc_b) | set(by_conc_o))
    rows: list[dict[str, Any]] = []
    speedups: list[float] = []
    successful_pairs = 0
    failed_pairs = 0
    for c in concs:
        b = by_conc_b.get(c) or {}
        o = by_conc_o.get(c) or {}
        bt = b.get("output_throughput")
        ot = o.get("output_throughput")
        speedup: float | None = None
        delta_pct: float | None = None
        if (
            isinstance(bt, (int, float)) and bt > 0
            and isinstance(ot, (int, float)) and ot > 0
        ):
            speedup = float(ot) / float(bt)
            delta_pct = (speedup - 1.0) * 100.0
            speedups.append(speedup)
            successful_pairs += 1
        else:
            failed_pairs += 1
        rows.append({
            "conc":            c,
            "baseline_tput":   bt,
            "optimized_tput":  ot,
            "speedup":         speedup,
            "delta_pct":       delta_pct,
            "baseline_status":  b.get("status"),
            "optimized_status": o.get("status"),
        })
    summary: dict[str, Any] = {
        "successful_pairs": successful_pairs,
        "failed_pairs":     failed_pairs,
        "best_conc":         None,
        "best_speedup":      None,
        "median_speedup":    None,
        "mean_speedup":      None,
    }
    if speedups:
        # Best = arg-max speedup over per-conc pairs.
        best_idx, best_val = max(
            (
                (i, r["speedup"])
                for i, r in enumerate(rows)
                if isinstance(r.get("speedup"), float)
            ),
            key=lambda x: x[1],
        )
        sorted_sp = sorted(speedups)
        n = len(sorted_sp)
        median = (
            sorted_sp[n // 2] if n % 2 == 1
            else 0.5 * (sorted_sp[n // 2 - 1] + sorted_sp[n // 2])
        )
        summary.update({
            "best_conc":      rows[best_idx]["conc"],
            "best_speedup":   round(best_val, 4),
            "median_speedup": round(median, 4),
            "mean_speedup":   round(sum(speedups) / len(speedups), 4),
        })
    return rows, summary


def _write_csv(csv_path: Path, points: list[dict[str, Any]]) -> None:
    """One row per (arm, conc) — flat columns for spreadsheet pivots."""
    columns = [
        "arm", "conc", "status",
        "output_throughput", "request_throughput", "total_token_throughput",
        "ttft_mean_ms", "e2el_mean_ms", "duration_seconds",
        "completed_requests", "error_class", "error",
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
    """Per-conc decode roofline alongside the measured curves (T_cmp once, T_mem per CONC; MBU% = measured/T_peak×100). ``None`` when model meta / GPU spec is unavailable."""
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
        # Resolve T_peak / binding (mirrors compute_roofline_breakdown_from_state).
        if t_mem <= 0 and t_cmp <= 0:
            bound_kind = "unknown"
            t_peak = 0.0
        elif t_cmp <= 0:
            bound_kind = "memory"
            t_peak = t_mem
        elif t_mem <= 0:
            bound_kind = "compute"
            t_peak = t_cmp
        elif t_cmp < t_mem:
            bound_kind = "compute"
            t_peak = t_cmp
        else:
            bound_kind = "memory"
            t_peak = t_mem

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
            if t_peak <= 0:
                return None
            return round((float(measured) / t_peak) * 100.0, 2)

        bt = (by_conc_b.get(c) or {}).get("output_throughput")
        ot = (by_conc_o.get(c) or {}).get("output_throughput")
        rows.append({
            "conc":              c,
            "t_mem_tok_s":       round(t_mem, 2),
            "t_cmp_tok_s":       round(t_cmp, 2),
            "t_peak_tok_s":      round(t_peak, 2),
            "bound_kind":        bound_kind,
            "mbu_baseline_pct":  _mbu_pct(bt),
            "mbu_optimized_pct": _mbu_pct(ot),
        })

    return {
        "schema_version": 1,
        "source":         "roofline_ceiling.py",
        "gpu_type":       gpu_type,
        "precision":      precision,
        "tp":             num_gpus,
        "isl":            isl,
        "osl":            osl,
        "model_meta": {
            "weight_bytes":        meta.weight_bytes,
            "active_weight_bytes": meta.active_weight_bytes,
            "num_experts":         meta.num_experts,
            "experts_per_tok":     meta.experts_per_tok,
            "expert_weight_bytes": meta.expert_weight_bytes,
            "num_layers":          meta.num_layers,
            "num_kv_heads":        meta.num_kv_heads,
            "head_dim":            meta.head_dim,
            "weight_dtype_bytes":  meta.weight_dtype_bytes,
        },
        "rows": rows,
    }


def _skip(reason: str, **extras: Any) -> dict[str, Any]:
    """Build a non-fatal skip envelope. Reason is operator-readable."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status":         "skipped",
        "skip_reason":    reason,
    }
    payload.update(extras)
    return payload


async def run_conc_sweep(
    state: SharedState,
    session_dir: Path,
    *,
    concs: list[int] | None = None,
    variant_timeout_sec: int = DEFAULT_VARIANT_TIMEOUT_SEC,
    total_budget_sec: int = DEFAULT_TOTAL_BUDGET_SEC,
    num_prompts_factor: int = DEFAULT_NUM_PROMPTS_FACTOR,
    write_reports: bool = True,
) -> dict[str, Any]:
    """Run the full conc-sweep post-hook end-to-end (always returns a dict; never raises; no files written when skipped)."""
    session_dir = Path(session_dir)
    # ``None`` → default ladder; an explicit empty list short-circuits via ``empty_conc_list`` below.
    concs = list(concs) if concs is not None else list(DEFAULT_CONCS)
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

    # Prefer the session's materialized baseline config; fall back to the shipped asset.
    base_yaml_raw = (
        str(getattr(state, "baseline_config_path", "") or "").strip()
        or str(default_baseline_config())
    )
    base_yaml_path = Path(base_yaml_raw)
    if not base_yaml_path.exists():
        return _skip("baseline_config_missing", config_path=base_yaml_raw)

    task_id = time.strftime("conc_sweep_%Y%m%dT%H%M%SZ", time.gmtime())
    workspace = runs_root(session_dir) / "conc_sweep" / task_id
    workspace.mkdir(parents=True, exist_ok=True)

    # Re-materialize (idempotent) in case we fell back to the shipped asset.
    resolved_model = (
        str(getattr(state, "model_path", "") or "").strip()
        or os.environ.get("MODEL_PATH", "").strip()
    )
    resolved_gpu = (
        str(getattr(state, "gpu_type", "") or "").strip().lower()
        or os.environ.get("GPU_TYPE", "").strip().lower()
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

    grid = _build_grid(
        concs=concs,
        isl=isl,
        osl=osl,
        num_prompts_factor=num_prompts_factor,
        optimized_args=opt_args,
        optimized_envs=opt_envs,
    )

    # Independent total wall-clock budget (<=0 disables); variants run one at a time so we can stop cleanly when exhausted.
    has_budget = total_budget_sec > 0
    started_at = time.time()
    deadline = started_at + total_budget_sec if has_budget else None
    log.info(
        "conc_sweep: launching %d variants (concs=%s isl=%d osl=%d "
        "total_budget=%s)",
        len(grid), concs, isl, osl,
        f"{total_budget_sec}s" if has_budget else "unbounded",
    )
    results: list[VariantResult] = []
    budget_exhausted = False
    for idx, variant in enumerate(grid):
        remaining = (
            (deadline - time.time()) if has_budget else None
        )
        if has_budget and remaining is not None and remaining <= 0:
            budget_exhausted = True
            log.warning(
                "conc_sweep: total budget exhausted (%ds); marking "
                "%d remaining variants as skipped",
                total_budget_sec, len(grid) - idx,
            )
            for v in grid[idx:]:
                results.append(_budget_skip_result(v))
            break

        # Per-variant cap = min(timeout, remaining budget) so the last variant doesn't blow the wall-clock.
        effective_timeout = variant_timeout_sec
        if has_budget and remaining is not None:
            effective_timeout = max(1, min(variant_timeout_sec, int(remaining)))

        sub_results = await run_grid(
            base_yaml_path=base_yaml_path,
            base_extra_args="",
            grid=[variant],
            output_root=workspace,
            variant_timeout_sec=effective_timeout,
            model_path=resolved_model,
            gpu_type=resolved_gpu,
        )
        results.extend(sub_results)
    elapsed_sec = time.time() - started_at

    # Split by arm via the variant name prefix we set in ``_build_grid``.
    baseline_points: list[dict[str, Any]] = []
    optimized_points: list[dict[str, Any]] = []
    for v in results:
        if v.name.startswith("baseline_"):
            baseline_points.append(_point_from_variant(v, arm="baseline"))
        elif v.name.startswith("optimized_"):
            optimized_points.append(_point_from_variant(v, arm="optimized"))
    baseline_points.sort(key=lambda p: p["conc"])
    optimized_points.sort(key=lambda p: p["conc"])

    comparison, summary = _build_comparison(baseline_points, optimized_points)

    ceiling = _build_roofline_ceiling(
        state,
        concs=concs,
        isl=isl,
        osl=osl,
        baseline_points=baseline_points,
        optimized_points=optimized_points,
    )

    payload: dict[str, Any] = {
        "schema_version":   SCHEMA_VERSION,
        "status":           "succeeded" if summary["successful_pairs"] else "failed",
        "session_id":       str(getattr(state, "session_id", "") or session_dir.name),
        "isl":              isl,
        "osl":              osl,
        "tp":               int(getattr(state, "tp", 0) or 0),
        "concs_requested":  concs,
        "baseline": {
            "extra_server_args": "",
            "extra_envs":        {},
            "points":            baseline_points,
        },
        "optimized": {
            "extra_server_args": opt_args,
            "extra_envs":        opt_envs,
            "points":            optimized_points,
        },
        "comparison":      comparison,
        "summary":         summary,
        "workspace":       workspace.as_posix(),
        "elapsed_sec":     round(elapsed_sec, 2),
        "total_budget_sec":  total_budget_sec if has_budget else None,
        "budget_exhausted":  budget_exhausted,
    }
    if ceiling is not None:
        payload["roofline_ceiling"] = ceiling

    if write_reports:
        rdir = reports_dir(session_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        json_path = rdir / "conc_sweep_summary.json"
        csv_path = rdir / "conc_sweep_raw.csv"
        # Set self-referential paths BEFORE the dump so the on-disk JSON carries them (consumers read the file, not the in-memory payload).
        payload["report_json_path"] = json_path.as_posix()
        payload["report_csv_path"] = csv_path.as_posix()
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        _write_csv(csv_path, baseline_points + optimized_points)
        # final.json pointer is added by report.py at CLOSE (this action runs before CLOSE).

    log.info(
        "conc_sweep: done — successful_pairs=%d failed_pairs=%d best_speedup=%s",
        summary["successful_pairs"], summary["failed_pairs"], summary["best_speedup"],
    )
    return payload


def format_summary_line(payload: dict[str, Any]) -> str:
    """One-line stdout summary for ``_print_final_summary``."""
    status = payload.get("status", "?")
    if status == "skipped":
        return f"  conc_sweep           : skipped ({payload.get('skip_reason', '?')})"
    s = payload.get("summary", {}) or {}
    succ = s.get("successful_pairs", 0)
    failed = s.get("failed_pairs", 0)
    best_speedup = s.get("best_speedup")
    best_conc = s.get("best_conc")
    median = s.get("median_speedup")
    suffix = ""
    if payload.get("budget_exhausted"):
        budget = payload.get("total_budget_sec")
        suffix = f" [budget_exhausted{f' @{budget}s' if budget else ''}]"
    parts = [
        f"  conc_sweep           : {status} "
        f"(pairs={succ}+{failed}f)"
    ]
    if isinstance(best_speedup, (int, float)) and best_conc is not None:
        parts.append(f"best={best_speedup:.2f}x @ conc={best_conc}")
    if isinstance(median, (int, float)):
        parts.append(f"median={median:.2f}x")
    return " ".join(parts) + suffix


__all__ = [
    "DEFAULT_CONCS",
    "DEFAULT_NUM_PROMPTS_FACTOR",
    "DEFAULT_TOTAL_BUDGET_SEC",
    "DEFAULT_VARIANT_TIMEOUT_SEC",
    "SCHEMA_VERSION",
    "format_summary_line",
    "run_conc_sweep",
]
