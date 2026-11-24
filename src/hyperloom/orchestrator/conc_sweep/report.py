# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The conc-sweep report: one curve row per rung, the roofline ceiling beside the curves, and the JSON/CSV/PNG write."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom_common import io as _common_io
from hyperloom.inference_optimizer.breakdown.exporter import patch_breakdown_conc_sweep
from hyperloom.inference_optimizer.roofline_ceiling import (
    ComputeTarget,
    DecodeShape,
    compute_compute_bound_ceiling_tok_per_sec,
    compute_theoretical_peak_output_tok_per_sec,
    select_peak_and_bound,
)
from hyperloom.inference_optimizer.roofline_model_meta import load_model_meta
from hyperloom.inference_optimizer.session.session_paths import CONC_SWEEP_SUMMARY_FILENAME, reports_dir

from .plot import ConcCurveOptions, render_conc_sweep_curve

if TYPE_CHECKING:
    from hyperloom.orchestrator.actions.executors.grid._grid_runner import VariantResult
    from hyperloom.orchestrator.state.shared_state import SharedState

log = logging.getLogger(__name__)

SUMMARY_FILE = CONC_SWEEP_SUMMARY_FILENAME
RAW_FILE = "conc_sweep_raw.csv"
CURVE_FILE = "conc_sweep_curve.png"
FINAL_FILE = "final.json"
_ENCODING = "utf-8"
#: The statuses of a sweep that ran its course, whatever it measured.
SETTLED = frozenset({"succeeded", "partial", "skipped"})


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
    with csv_path.open("w", newline="", encoding=_ENCODING) as fp:
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


def read_summary(reports: Path) -> dict[str, Any] | None:
    """The summary an earlier run of the sweep left in ``reports``, or ``None``."""
    path = reports / SUMMARY_FILE
    return json.loads(path.read_text(encoding=_ENCODING)) if path.is_file() else None


def write_report(payload: dict[str, Any], reports: Path, *, breakdown: Path) -> None:
    """Atomically write the summary JSON and the raw CSV into ``reports``, and carry the summary in ``breakdown``.

    ``breakdown`` is the session's ``session_breakdown.json``: CLOSE wrote it before the sweep began, and it is the
    document a session's results are ingested from.
    """
    reports.mkdir(parents=True, exist_ok=True)
    _common_io.atomic_write_text(reports / SUMMARY_FILE, json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    _write_csv(reports / RAW_FILE, [*payload["baseline"]["points"], *payload["optimized"]["points"]])
    patch_breakdown_conc_sweep(breakdown, payload)


def write_curve(payload: dict[str, Any], reports: Path, options: ConcCurveOptions) -> Path | None:
    """Draw the sweep's curve into ``reports``; ``None`` when there is nothing to draw or no matplotlib."""
    return render_conc_sweep_curve(payload, reports / CURVE_FILE, options)


def publish_curve(payload: dict[str, Any], session_dir: Path, options: ConcCurveOptions) -> None:
    """Draw the sweep's curve and point the session's ``final.json`` at the summary and the curve."""
    reports = reports_dir(session_dir)
    curve = write_curve(payload, reports, options)
    final = reports / FINAL_FILE
    if not final.is_file():
        return
    document = json.loads(final.read_text(encoding=_ENCODING))
    document["conc_sweep_summary"] = {
        "report_path": (reports / SUMMARY_FILE).relative_to(session_dir).as_posix(),
        "status": payload["status"],
        "summary": payload["summary"],
    }
    if curve is not None:
        document["conc_sweep_curve_png"] = curve.relative_to(session_dir).as_posix()
    _common_io.atomic_write_text(final, json.dumps(document, indent=2, ensure_ascii=False, default=str))
