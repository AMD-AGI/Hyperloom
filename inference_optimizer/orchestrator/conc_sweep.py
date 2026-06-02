"""Post-optimization concurrency sweep.

Runs the *same* Magpie benchmark machinery used by ``baseline`` /
``sweep`` over a fixed grid of CONC values, once with the baseline
server (no extra args / envs) and once with the final ``current_best``
config. The output is a JSON/CSV report that the frontend pairs with
the roofline ceiling to show "baseline vs optimized" curves across
concurrency.

Trigger
-------
This is a post-hook invoked from ``cli.py`` after the close-sequence
report has been written and right before ``_print_final_summary``. It
is **on by default** (operator opts out via ``--no-enable-conc-sweep``
or ``INFERENCE_OPTIMIZER_ENABLE_CONC_SWEEP=0``); each concurrency point
relaunches sglang and the full grid can take ~30 min on a single
8xMI300 box, so the total wall-clock is bounded by
``--conc-sweep-total-budget-sec`` (default 2.5h) and the remaining
session deadline.

Skip rules (return ``status="skipped"`` without launching anything):

* ``state.baseline_tput <= 0``     — no baseline to compare against.
* ``state.isl <= 0 or state.osl <= 0`` — workload shape unknown.
* ``current_best`` has no extra args **and** no extra envs — nothing
  was actually optimized; running this would just re-measure baseline
  twice.

Outputs
-------
* ``<sd>/runs/conc_sweep/<task_id>/`` — per-variant Magpie workspaces
  (same on-disk layout as ``runs/sweep/...``).
* ``<sd>/reports/conc_sweep_summary.json`` — frontend-friendly summary
  with both curves and a per-conc comparison.
* ``<sd>/reports/conc_sweep_raw.csv``      — one row per benchmark
  point for ad-hoc analysis.
* ``<sd>/reports/final.json`` gains a ``conc_sweep_summary`` pointer
  (see ``action_executors/report.py``).
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

# Frontend-friendly default ladder. Operators can override with
# ``--conc-sweep-concs``. We deliberately span 1 → 128 in powers of 2
# so the curve covers both latency-bound (low CONC) and saturation
# (high CONC) regimes.
DEFAULT_CONCS: list[int] = [1, 2, 4, 8, 16, 32, 64, 128]

# Mirrors ``sweep.py``'s adaptive NUM_PROMPTS heuristic.
DEFAULT_NUM_PROMPTS_FACTOR = 5

# Per-variant timeout (seconds). Baseline + optimized each get one
# subprocess per CONC, so 30 min/point bounds the full sweep at
# 30min × 8 × 2 = 8h in the worst case. Operator override via
# ``--conc-sweep-timeout-sec``.
DEFAULT_VARIANT_TIMEOUT_SEC = 1800

# Total wall-clock budget (seconds) across the whole conc_sweep
# action. Caps how long the action itself may run before stopping
# the remaining variants and writing whatever data was collected.
# Default 9000 (~2.5h); operator override via
# ``--conc-sweep-total-budget-sec``. Set to 0 or negative to disable
# (run unbounded until per-variant timeouts trip). conc_sweep runs
# as a SWEEP-phase action so it is also bounded above by the main
# session wall-clock deadline.
DEFAULT_TOTAL_BUDGET_SEC = 9000


def _has_optimization(state: SharedState) -> tuple[bool, str, dict[str, str]]:
    """Return ``(has_opt, args, envs)`` based on ``state.current_best``.

    Either non-empty side counts as "optimized" — operators sometimes
    accept env-only or arg-only wins.
    """
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
    """Two-arm grid: ``baseline`` (empty overrides) × ``optimized``
    (current_best) crossed with every requested CONC."""
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
    """Synthetic VariantResult for a variant that never ran because the
    total wall-clock budget was already exhausted. Status ``skipped`` is
    new in conc_sweep (vs the grid_runner's ``succeeded`` / ``failed``)
    so the frontend can tell "we ran out of time" apart from "Magpie
    crashed". Speedup aggregation treats null throughput uniformly."""
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
    """Per-conc decode roofline alongside the measured curves.

    Reuses :mod:`orchestrator.roofline_ceiling` (the same module that
    powers ``compute_roofline_breakdown_from_state``) so the ceiling
    here matches the one already surfaced in the rest of the run
    (e.g. ``state.history_snapshots``). T_cmp is batch-independent so
    we compute it once; T_mem and binding are re-derived per CONC.

    MBU% is the headline number: ``measured / T_peak × 100``. Both
    baseline and optimized arms get one when the corresponding point
    succeeded; otherwise ``None`` so renderers can show ``"—"``.

    Safe degrade: returns ``None`` when model meta or GPU spec is
    unavailable so the caller can omit the field, mirroring the rest
    of the conc_sweep payload's no-placeholder policy.
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
        # Resolve T_peak / binding — mirrors the branching used by
        # compute_roofline_breakdown_from_state.
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
    """Run the full conc-sweep post-hook end-to-end.

    Always returns a dict (never raises). When skipped, no files are
    written. When executed, both ``conc_sweep_summary.json`` and
    ``conc_sweep_raw.csv`` are written to ``<sd>/reports/`` and the
    summary dict is returned (and also serialized).
    """
    session_dir = Path(session_dir)
    # ``None`` → fall back to the documented default ladder. An explicit
    # empty list is treated as a caller intent ("nothing to sweep") and
    # short-circuits via ``empty_conc_list`` below.
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

    # Resolve the base YAML. Prefer the session's materialized baseline
    # config (already pinned to TP/MAX_MODEL_LEN/PRECISION/...). Fall
    # back to the shipped baseline asset when missing.
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

    # Workload envs were already baked into ``baseline_config_path`` at
    # baseline time, so we only need to re-materialize if we're falling
    # back to the shipped asset. Idempotent either way.
    resolved_model = (
        str(getattr(state, "model_path", "") or "").strip()
        or os.environ.get("MODEL_PATH", "").strip()
    )
    resolved_gpu = (
        str(getattr(state, "gpu_type", "") or "").strip().lower()
        or os.environ.get("GPU_TYPE", "").strip().lower()
    )
    base_yaml_path = materialize_config_with_envs(
        base_yaml_path,
        workspace,
        model_path=resolved_model or None,
        gpu_type=resolved_gpu or None,
        out_name="conc_sweep_base.with_envs.yaml",
    )

    grid = _build_grid(
        concs=concs,
        isl=isl,
        osl=osl,
        num_prompts_factor=num_prompts_factor,
        optimized_args=opt_args,
        optimized_envs=opt_envs,
    )

    # Independent total wall-clock budget for the post-hook itself.
    # ``<=0`` disables the gate (legacy unbounded behaviour, only
    # per-variant timeouts trip). We run variants one at a time so we
    # can stop cleanly the moment the budget is exhausted — cold-start
    # cost per variant is the same as the underlying sweep executor,
    # which also relaunches Magpie per (CONC, ISL, OSL) combo.
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

        # Per-variant cap = min(per-variant timeout, remaining budget),
        # so the last variant doesn't blow the wall-clock by another
        # full ``variant_timeout_sec``.
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
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        _write_csv(csv_path, baseline_points + optimized_points)
        payload["report_json_path"] = json_path.as_posix()
        payload["report_csv_path"] = csv_path.as_posix()
        # final.json pointer is added by report.py at CLOSE -- the
        # action runs strictly before CLOSE so report.py can read
        # this file freshly. See ``_write_conc_sweep_pointer`` there.

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
