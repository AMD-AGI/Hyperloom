# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The conc-sweep report: one curve row per rung, the roofline ceiling beside the curves, and the JSON/CSV write."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom_common import io as _common_io
from hyperloom.inference_optimizer.roofline_ceiling import (
    ComputeTarget,
    DecodeShape,
    compute_compute_bound_ceiling_tok_per_sec,
    compute_theoretical_peak_output_tok_per_sec,
    select_peak_and_bound,
)
from hyperloom.inference_optimizer.roofline_model_meta import load_model_meta
from hyperloom.inference_optimizer.session.session_paths import reports_dir
from ..actions.executors._grid_runner import VariantResult
from ..actions.stop_attribution import SESSION_TIME_EXHAUSTED_CLASS
from ..state.shared_state import SharedState

log = logging.getLogger(__name__)


def point_from_variant(v: VariantResult, *, arm: str) -> dict[str, Any]:
    """Flatten a ``VariantResult`` into one row of the curve."""
    envs = v.extra_envs or {}
    # The grid sets CONC from an int, and a curve is only readable if every
    # rung names the concurrency it was measured at -- so a value that will not
    # parse is a bug to surface, not a rung to file under zero.
    conc = int(envs.get("CONC", "0"))
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


def budget_limited_without_valid_pair(
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
        if error_class in {"budget_exhausted", SESSION_TIME_EXHAUSTED_CLASS}:
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


@dataclass(frozen=True)
class ArmCurves:
    """The measured points of both arms of a sweep."""

    baseline: list[dict[str, Any]]
    optimized: list[dict[str, Any]]


def build_roofline_ceiling(
    state: SharedState,
    curves: ArmCurves,
    *,
    concs: list[int],
    isl: int,
    osl: int,
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
        ComputeTarget(gpu_type=gpu_type, num_gpus=num_gpus, precision_tag=precision),
        active_weight_bytes=meta.active_weight_bytes,
        weight_bytes=meta.weight_bytes,
        weight_dtype_bytes=meta.weight_dtype_bytes,
    )

    by_conc_b = {p["conc"]: p for p in curves.baseline}
    by_conc_o = {p["conc"]: p for p in curves.optimized}

    rows: list[dict[str, Any]] = []
    for c in concs:
        t_mem = compute_theoretical_peak_output_tok_per_sec(
            meta,
            DecodeShape(isl=isl, osl=osl, concurrency=c),
            gpu_type=gpu_type,
            num_gpus=num_gpus,
            kv_dtype_bytes=meta.weight_dtype_bytes,
        )
        t_peak, bound_kind = select_peak_and_bound(t_mem, t_cmp)
        # Local import avoids a module-level import cycle.
        from hyperloom.inference_optimizer.roofline_snapshot import within_roofline_pct

        def _mbu_pct(measured: Any) -> float | None:
            """Express a measured throughput as a percent of peak."""
            if not isinstance(measured, (int, float)) or measured <= 0:
                return None
            return within_roofline_pct(peak=float(t_peak), achieved=float(measured))

        # Output throughput on both arms regardless of the graded axis, and not a bug to be aligned with it: the
        # peak above is a memory-bandwidth-derived ceiling on output tokens per second, so MBU is only meaningful
        # against the same quantity.
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
        "model_meta": _model_meta_record(meta),
        "rows": rows,
    }


def _model_meta_record(meta: Any) -> dict[str, Any]:
    """The model sizes and shape the roofline ceiling was computed from."""
    return {
        "weight_bytes": meta.weight_bytes,
        "active_weight_bytes": meta.active_weight_bytes,
        "num_experts": meta.num_experts,
        "experts_per_tok": meta.experts_per_tok,
        "expert_weight_bytes": meta.expert_weight_bytes,
        "num_layers": meta.num_layers,
        "num_kv_heads": meta.num_kv_heads,
        "head_dim": meta.head_dim,
        "weight_dtype_bytes": meta.weight_dtype_bytes,
    }


def flush_conc_sweep_report(payload: dict[str, Any], session_dir: Path) -> Exception | None:
    """Atomically write the conc-sweep summary JSON + CSV to the reports dir.

    Returns what stopped the write, so a caller that recorded the paths can say
    they are where the report was meant to go rather than where it is.
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
    except (OSError, ValueError, csv.Error) as exc:
        log.warning("conc_sweep: flush_conc_sweep_report failed", exc_info=True)
        return exc
    return None
