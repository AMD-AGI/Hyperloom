#!/usr/bin/env python3
"""Workload-level (end-to-end) roofline for diffusion / scriptable traces.

TraceLens' ``generate_perf_report_pytorch`` produces a *per-kernel* roofline
(each op vs its dtype ceiling). For LLM decode, hyperloom also models a
*workload-level* memory-bandwidth roofline (``roofline_ceiling.py``). Diffusion
is the compute-bound dual of that: the DiT forward runs once per denoise step,
dominated by large batched GEMMs + attention.

This tool aggregates the per-kernel report into a single workload roofline:

    - kernel roofline efficiency = Sigma(ideal kernel time) / Sigma(actual kernel time)
    - gpu busy ratio             = busy_time / wall_time            (from gpu_timeline)
    - end-to-end efficiency      ~ kernel_eff * gpu_busy_ratio
    - per denoise-step timings   = totals / num_denoise_steps       (when provided)

The two efficiency gaps it surfaces are complementary:
  1. scheduling gap  (gpu_busy_ratio < 1): GPU idle / exposed comm / launch gaps.
  2. kernel gap      (kernel_eff  < 1):    kernels below their roofline ceiling.

Input is the ``--output_csvs_dir`` produced by generate_perf_report_pytorch
(``unified_perf_summary.csv`` + optional ``gpu_timeline.csv``), so the workload
roofline stays numerically consistent with the per-kernel roofline.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

UNIFIED_CSV = "unified_perf_summary.csv"
GPU_TIMELINE_CSV = "gpu_timeline.csv"

# Column names as emitted by generate_perf_report_pytorch's unified summary.
COL_KERNEL_TIME_SUM = "Kernel Time (\u00b5s)_sum"
COL_ROOFLINE_TIME = "Roofline Time (\u00b5s)_first"
COL_OP_COUNT = "operation_count"
COL_BOUND = "Roofline Bound"
COL_CATEGORY = "op category"
COL_NAME = "name"


def _to_float(value: Any) -> float:
    """Best-effort float parse; blanks / non-numeric become 0.0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of dict rows (empty list when missing)."""
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def aggregate_unified(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Aggregate the unified per-kernel summary into workload totals.

    Args:
        rows: Parsed ``unified_perf_summary.csv`` rows.

    Returns:
        Dict of workload totals (actual/ideal us, bound split, kernel efficiency).
    """
    sigma_actual_us = 0.0
    sigma_ideal_us = 0.0
    compute_us = 0.0
    memory_us = 0.0
    no_model_us = 0.0
    for r in rows:
        actual = _to_float(r.get(COL_KERNEL_TIME_SUM))
        count = _to_float(r.get(COL_OP_COUNT)) or 1.0
        # Roofline Time is the per-instance ideal; scale by the aggregated count.
        ideal = _to_float(r.get(COL_ROOFLINE_TIME)) * count
        sigma_actual_us += actual
        sigma_ideal_us += ideal
        bound = (r.get(COL_BOUND) or "").upper()
        if "COMPUTE" in bound:
            compute_us += actual
        elif "MEMORY" in bound:
            memory_us += actual
        else:
            no_model_us += actual
    kernel_eff = (sigma_ideal_us / sigma_actual_us) if sigma_actual_us > 0 else 0.0
    return {
        "sigma_actual_kernel_us": sigma_actual_us,
        "sigma_ideal_roofline_us": sigma_ideal_us,
        "kernel_roofline_efficiency": kernel_eff,
        "compute_bound_us": compute_us,
        "memory_bound_us": memory_us,
        "no_perf_model_us": no_model_us,
    }


def parse_gpu_timeline(rows: list[dict[str, str]]) -> dict[str, float]:
    """Extract busy / computation / exposed percentages from gpu_timeline.csv.

    Args:
        rows: Parsed ``gpu_timeline.csv`` rows (``type``, ``time ms``, ``percent``).

    Returns:
        Dict keyed by timeline ``type`` -> percent (empty when the file is absent).
    """
    out: dict[str, float] = {}
    for r in rows:
        kind = (r.get("type") or "").strip()
        if kind:
            out[kind] = _to_float(r.get("percent"))
    return out


def top_kernels(rows: list[dict[str, str]], k: int) -> list[dict[str, Any]]:
    """Return the top-k kernels by actual kernel time for the summary block."""
    ranked = sorted(rows, key=lambda r: _to_float(r.get(COL_KERNEL_TIME_SUM)), reverse=True)
    out: list[dict[str, Any]] = []
    for r in ranked[:k]:
        out.append(
            {
                "name": (r.get(COL_NAME) or "")[:48],
                "category": r.get(COL_CATEGORY) or "",
                "bound": r.get(COL_BOUND) or "",
                "kernel_time_us": _to_float(r.get(COL_KERNEL_TIME_SUM)),
            }
        )
    return out


def dit_analytic_flops(
    hidden_size: int,
    num_layers: int,
    num_tokens: int,
    num_denoise_steps: int,
    ffn_ratio: float = 4.0,
) -> dict[str, float]:
    """A-priori forward FLOPs for a DiT-style transformer denoise run.

    Standard dense-transformer accounting (2 FLOPs per MAC):

      - linear (QKVO + FFN) per token/layer : 2 * (4 + 2*ffn_ratio) * h^2
      - attention (scores + context)        : 2 * (2 * num_tokens * h)

    scaled by ``num_layers * num_tokens * num_denoise_steps``. Positional MLPs,
    adaLN modulation, patch/embed projections and the VAE are ignored, so this
    is a lower bound on the true DiT compute (an optimistic ceiling for the
    reconciliation cross-check, never a hard limit).

    Args:
        hidden_size: Transformer hidden dimension ``h``.
        num_layers: Number of transformer blocks.
        num_tokens: Sequence length (latent patches) per forward.
        num_denoise_steps: Denoise steps in the profiled window.
        ffn_ratio: FFN expansion factor (``intermediate / hidden``).

    Returns:
        Dict with ``linear_flops``, ``attention_flops`` and ``total_flops``.
    """
    h = float(hidden_size)
    per_token_linear = 2.0 * (4.0 + 2.0 * ffn_ratio) * h * h
    per_token_attention = 2.0 * (2.0 * float(num_tokens) * h)
    scale = float(num_layers) * float(num_tokens) * float(num_denoise_steps)
    linear = per_token_linear * scale
    attention = per_token_attention * scale
    return {
        "linear_flops": linear,
        "attention_flops": attention,
        "total_flops": linear + attention,
    }


def dit_analytic_ceiling(
    flops: dict[str, float], achievable_tflops: float
) -> dict[str, Any]:
    """Convert a-priori FLOPs into an achievable-compute time ceiling.

    Args:
        flops: Output of :func:`dit_analytic_flops`.
        achievable_tflops: Sustained matrix TFLOPS for the run dtype (from
            hyperloom's ``HW_SPECS_ACHIEVABLE``); the roofline ceiling divisor.

    Returns:
        Dict with the total FLOPs, the ceiling TFLOPS and the resulting
        ideal (compute-bound) microseconds; empty when inputs are non-positive.
    """
    total = flops.get("total_flops", 0.0)
    if total <= 0 or achievable_tflops <= 0:
        return {}
    ideal_us = total / (achievable_tflops * 1e12) * 1e6
    return {
        "total_flops": total,
        "achievable_tflops": achievable_tflops,
        "ideal_compute_us": ideal_us,
    }


def reconcile(
    totals: dict[str, Any], analytic: dict[str, Any]
) -> dict[str, Any]:
    """Cross-check the a-priori DiT ceiling against the trace-derived roofline.

    Compares the analytic compute-ideal time (from a-priori FLOPs / achievable
    TFLOPS) against TraceLens' summed per-kernel ideal + actual times:

      - ``analytic_vs_trace_ideal_ratio`` ~ 1 => the a-priori model matches the
        kernels TraceLens attributed a perf model to (roofline is trustworthy).
        >> 1 => the trace's per-kernel roofline under-counts DiT compute
        (missing/unmodeled kernels); << 1 => the model omits real work.
      - ``analytic_achieved_efficiency`` = analytic_ideal / trace_actual, the
        end-to-end HW efficiency implied by the a-priori compute lower bound.

    Args:
        totals: Aggregated trace totals from :func:`aggregate_unified`.
        analytic: Output of :func:`dit_analytic_ceiling` (may be empty).

    Returns:
        Dict of reconciliation ratios; empty when the analytic ceiling is absent.
    """
    ideal_us = analytic.get("ideal_compute_us", 0.0)
    if ideal_us <= 0:
        return {}
    trace_ideal = totals.get("sigma_ideal_roofline_us", 0.0)
    trace_actual = totals.get("sigma_actual_kernel_us", 0.0)
    out: dict[str, Any] = {"analytic_ideal_compute_us": ideal_us}
    if trace_ideal > 0:
        out["analytic_vs_trace_ideal_ratio"] = ideal_us / trace_ideal
    if trace_actual > 0:
        out["analytic_achieved_efficiency"] = ideal_us / trace_actual
    return out


def build_report(
    csv_dir: Path,
    num_denoise_steps: int | None,
    top_k: int,
    *,
    dit_geometry: dict[str, Any] | None = None,
    achievable_tflops: float | None = None,
) -> dict[str, Any]:
    """Assemble the workload-level roofline report from a TraceLens CSV dir.

    Args:
        csv_dir: ``--output_csvs_dir`` from generate_perf_report_pytorch.
        num_denoise_steps: Denoise steps in the profiled window (enables per-step).
        top_k: How many hottest kernels to include.

    Returns:
        The full report dict (also what ``--output`` serializes).

    Raises:
        FileNotFoundError: When the unified summary CSV is missing.
    """
    unified_path = csv_dir / UNIFIED_CSV
    unified_rows = _read_csv_rows(unified_path)
    if not unified_rows:
        raise FileNotFoundError(f"missing or empty {unified_path} (run generate_perf_report_pytorch first)")

    totals = aggregate_unified(unified_rows)
    timeline = parse_gpu_timeline(_read_csv_rows(csv_dir / GPU_TIMELINE_CSV))
    report = assemble_report(
        totals,
        timeline,
        num_denoise_steps,
        top_kernels(unified_rows, top_k),
        dit_geometry=dit_geometry,
        achievable_tflops=achievable_tflops,
        source="tracelens_csv",
    )
    report["source_csv_dir"] = str(csv_dir)
    return report


def assemble_report(
    totals: dict[str, Any],
    timeline: dict[str, float],
    num_denoise_steps: int | None,
    top_kernels_list: list[dict[str, Any]],
    *,
    dit_geometry: dict[str, Any] | None = None,
    achievable_tflops: float | None = None,
    source: str = "tracelens_csv",
) -> dict[str, Any]:
    """Assemble the workload roofline report from pre-aggregated inputs.

    Backend-agnostic single source of truth for the report *shape* (totals +
    timeline + end-to-end efficiency + per-denoise-step split + optional analytic
    DiT ceiling). Fed either by the TraceLens perf CSVs (``build_report``) or by
    the bypass analytical candidate set (``build_report_from_bypass``), so both
    routes emit an identically-shaped ``diffusion_roofline.json``.

    Args:
        totals: Workload totals (see ``aggregate_unified`` for the key contract).
        timeline: GPU timeline percentages keyed by type (``busy_time`` etc.).
        num_denoise_steps: Denoise steps in the profiled window (enables per-step).
        top_kernels_list: Pre-ranked hottest-kernel summary entries.
        dit_geometry: Optional DiT geometry enabling the analytic compute ceiling.
        achievable_tflops: Optional achievable peak enabling the analytic ceiling.
        source: Provenance label recorded on the report.

    Returns:
        The full workload-roofline report dict.
    """
    busy_pct = timeline.get("busy_time")
    gpu_busy_ratio = (busy_pct / 100.0) if busy_pct is not None else None
    kernel_eff = totals["kernel_roofline_efficiency"]
    end_to_end_eff = (kernel_eff * gpu_busy_ratio) if gpu_busy_ratio is not None else None

    report: dict[str, Any] = {
        "source": source,
        "totals": totals,
        "gpu_timeline_pct": timeline,
        "gpu_busy_ratio": gpu_busy_ratio,
        "end_to_end_efficiency_estimate": end_to_end_eff,
        "top_kernels": top_kernels_list,
    }

    if num_denoise_steps and num_denoise_steps > 0:
        report["num_denoise_steps"] = num_denoise_steps
        report["per_step"] = {
            "actual_kernel_us": totals["sigma_actual_kernel_us"] / num_denoise_steps,
            "ideal_roofline_us": totals["sigma_ideal_roofline_us"] / num_denoise_steps,
        }

    # Optional a-priori DiT compute ceiling + reconciliation cross-check. Only
    # emitted when the model geometry + achievable TFLOPS are supplied; never
    # fabricated from the trace alone.
    if dit_geometry and achievable_tflops and num_denoise_steps and num_denoise_steps > 0:
        try:
            flops = dit_analytic_flops(
                hidden_size=int(dit_geometry["hidden_size"]),
                num_layers=int(dit_geometry["num_layers"]),
                num_tokens=int(dit_geometry["num_tokens"]),
                num_denoise_steps=int(num_denoise_steps),
                ffn_ratio=float(dit_geometry.get("ffn_ratio", 4.0)),
            )
            ceiling = dit_analytic_ceiling(flops, float(achievable_tflops))
            if ceiling:
                report["analytic_dit_ceiling"] = ceiling
                recon = reconcile(totals, ceiling)
                if recon:
                    report["reconciliation"] = recon
        except (KeyError, TypeError, ValueError):
            pass
    return report


def aggregate_bypass_candidates(hot_kernels: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the bypass analytical candidate set into workload totals.

    The bypass backend has no TraceLens perf CSV; it carries the same numbers on
    each candidate: ``duration_us`` (summed GPU time = the actual side) and
    ``efficiency_percent`` (ideal/actual, from the analytical roofline) with a
    ``bound_type`` for the compute/memory split. Candidates whose roofline is a
    placeholder (no shapes) contribute only to ``no_perf_model_us`` so the kernel
    efficiency reflects modelled kernels only. Mirrors ``aggregate_unified``.

    Args:
        hot_kernels: The bypass candidate dicts (``hot_kernels`` list).

    Returns:
        Workload totals keyed identically to ``aggregate_unified``.
    """
    sigma_actual = 0.0
    sigma_ideal = 0.0
    compute_us = 0.0
    memory_us = 0.0
    no_model_us = 0.0
    for c in hot_kernels:
        actual = _to_float(c.get("duration_us"))
        sigma_actual += actual
        src = str(c.get("roofline_source") or "")
        # Binding-side attainment (cross-route comparable), not the compute-side
        # efficiency_percent which reads ~0 for memory-bound kernels.
        attain = _to_float(c.get("roofline_attainment_pct"))
        if src not in ("", "placeholder") and attain > 0:
            sigma_ideal += actual * (attain / 100.0)
            bound = str(c.get("bound_type") or "").upper()
            if "COMPUTE" in bound:
                compute_us += actual
            elif "MEMORY" in bound:
                memory_us += actual
            else:
                no_model_us += actual
        else:
            no_model_us += actual
    kernel_eff = (sigma_ideal / sigma_actual) if sigma_actual > 0 else 0.0
    return {
        "sigma_actual_kernel_us": sigma_actual,
        "sigma_ideal_roofline_us": sigma_ideal,
        "kernel_roofline_efficiency": kernel_eff,
        "compute_bound_us": compute_us,
        "memory_bound_us": memory_us,
        "no_perf_model_us": no_model_us,
    }


def _top_bypass_kernels(hot_kernels: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Top-k bypass candidates by GPU time for the summary block."""
    ranked = sorted(hot_kernels, key=lambda c: _to_float(c.get("duration_us")), reverse=True)
    return [
        {
            "name": (c.get("name") or "")[:48],
            "category": c.get("kernel_category") or "",
            "bound": c.get("bound_type") or "",
            "kernel_time_us": _to_float(c.get("duration_us")),
        }
        for c in ranked[:k]
    ]


def build_report_from_bypass(
    hot_kernels: list[dict[str, Any]],
    timeline: dict[str, Any],
    num_denoise_steps: int | None,
    top_k: int,
    *,
    dit_geometry: dict[str, Any] | None = None,
    achievable_tflops: float | None = None,
    totals: dict[str, Any] | None = None,
    kernels_aggregated: int | None = None,
) -> dict[str, Any]:
    """Build the workload roofline report from the bypass candidate set.

    Produces an identically-shaped report to ``build_report`` (TraceLens CSV
    path) so downstream consumers read ``diffusion_roofline.json`` the same way
    regardless of route.

    Args:
        hot_kernels: The bypass ``hot_kernels`` candidate dicts (used for the
            top-N summary block).
        timeline: The bypass ``analyze["timeline"]`` (``busy_pct``/``idle_pct``).
        num_denoise_steps: Effective denoise steps (enables the per-step split).
        top_k: How many hottest kernels to include in the summary.
        dit_geometry: Optional DiT geometry for the analytic compute ceiling.
        achievable_tflops: Optional achievable peak for the analytic ceiling.
        totals: Pre-computed WORKLOAD totals over ALL analyzed kernels (from
            ``_bypass_report.build_workload_roofline_totals``). When omitted,
            falls back to aggregating the (top-k capped) ``hot_kernels`` only.
        kernels_aggregated: Count of device kernels the ``totals`` cover (for the
            ``kernels_aggregated`` metadata under full scope). Defaults to
            ``len(hot_kernels)`` when omitted.

    Returns:
        The workload-roofline report dict.
    """
    full_scope = totals is not None
    if totals is None:
        totals = aggregate_bypass_candidates(hot_kernels)
    timeline_pct: dict[str, float] = {}
    if isinstance(timeline, dict):
        if timeline.get("busy_pct") is not None:
            timeline_pct["busy_time"] = _to_float(timeline.get("busy_pct"))
        if timeline.get("idle_pct") is not None:
            timeline_pct["idle_time"] = _to_float(timeline.get("idle_pct"))
    report = assemble_report(
        totals,
        timeline_pct,
        num_denoise_steps,
        _top_bypass_kernels(hot_kernels, top_k),
        dit_geometry=dit_geometry,
        achievable_tflops=achievable_tflops,
        source="bypass_analytical",
    )
    # ``all_device_kernels`` when totals cover every kernel; else the top-k subset.
    # kernels_aggregated must match that scope: the caller's all-kernel count when
    # full scope, else the (top-k) candidate count actually aggregated.
    report["kernel_scope"] = "all_device_kernels" if full_scope else "analyzed_candidates"
    report["kernels_aggregated"] = (
        int(kernels_aggregated) if kernels_aggregated is not None else len(hot_kernels)
    )
    return report


def _fmt_pct(x: float | None) -> str:
    """Render a 0..1 ratio as a percentage string (``n/a`` when None)."""
    return "n/a" if x is None else f"{100 * x:.1f}%"


def print_summary(report: dict[str, Any]) -> None:
    """Print a compact human-readable summary of the workload roofline."""
    t = report["totals"]
    print("=== diffusion workload roofline ===")
    print(f"sigma actual kernel time : {t['sigma_actual_kernel_us'] / 1e6:.3f} s")
    print(f"sigma ideal roofline time: {t['sigma_ideal_roofline_us'] / 1e6:.3f} s")
    print(f"kernel roofline efficiency (ideal/actual): {_fmt_pct(t['kernel_roofline_efficiency'])}")
    print(f"gpu busy ratio           : {_fmt_pct(report.get('gpu_busy_ratio'))}")
    print(f"end-to-end efficiency est: {_fmt_pct(report.get('end_to_end_efficiency_estimate'))}")
    denom = t["sigma_actual_kernel_us"] or 1.0
    print(
        "bound split (of actual):"
        f" compute {100 * t['compute_bound_us'] / denom:.0f}%"
        f" | memory {100 * t['memory_bound_us'] / denom:.0f}%"
        f" | no-perf-model {100 * t['no_perf_model_us'] / denom:.0f}%"
    )
    if "per_step" in report:
        ps = report["per_step"]
        print(
            f"per denoise-step ({report['num_denoise_steps']} steps):"
            f" actual {ps['actual_kernel_us'] / 1e3:.2f} ms"
            f" | ideal {ps['ideal_roofline_us'] / 1e3:.2f} ms"
        )
    if "analytic_dit_ceiling" in report:
        ac = report["analytic_dit_ceiling"]
        print(
            "a-priori DiT ceiling:"
            f" {ac['total_flops'] / 1e12:.1f} TFLOP @ {ac['achievable_tflops']:.0f} TFLOPS"
            f" -> ideal {ac['ideal_compute_us'] / 1e3:.2f} ms"
        )
    if "reconciliation" in report:
        rc = report["reconciliation"]
        if "analytic_vs_trace_ideal_ratio" in rc:
            print(f"reconciliation analytic/trace-ideal ratio: {rc['analytic_vs_trace_ideal_ratio']:.2f}")
        if "analytic_achieved_efficiency" in rc:
            print(f"reconciliation analytic achieved efficiency: {_fmt_pct(rc['analytic_achieved_efficiency'])}")
    print("top kernels by time:")
    for hk in report["top_kernels"]:
        print(f"  {hk['kernel_time_us'] / 1e3:8.2f} ms  [{hk['bound']:12}] {hk['category']:14} {hk['name']}")


def main() -> int:
    """CLI entry point for the diffusion workload roofline aggregator."""
    parser = argparse.ArgumentParser(description="Diffusion workload-level (end-to-end) roofline")
    parser.add_argument(
        "--perf-csv-dir",
        required=True,
        help="Directory produced by generate_perf_report_pytorch (--output_csvs_dir).",
    )
    parser.add_argument(
        "--num-denoise-steps",
        type=int,
        default=0,
        help="Denoise steps in the profiled window; enables per-step timings.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Hottest kernels to list.")
    parser.add_argument("--output", default="", help="Optional path to write the report JSON.")
    # Optional a-priori DiT ceiling cross-check (all four geometry flags + a
    # ceiling source required to activate).
    parser.add_argument("--dit-hidden-size", type=int, default=0, help="DiT hidden dim h (analytic ceiling).")
    parser.add_argument("--dit-num-layers", type=int, default=0, help="DiT transformer block count.")
    parser.add_argument("--dit-num-tokens", type=int, default=0, help="Latent patch/token count per forward.")
    parser.add_argument("--dit-ffn-ratio", type=float, default=4.0, help="FFN expansion ratio.")
    parser.add_argument(
        "--achievable-tflops",
        type=float,
        default=0.0,
        help="Sustained matrix TFLOPS ceiling; overrides --target-platform resolution.",
    )
    parser.add_argument(
        "--target-platform",
        default="",
        help="GPU arch (e.g. MI355X); resolves achievable bf16 TFLOPS from HW_SPECS_ACHIEVABLE.",
    )
    args = parser.parse_args()

    dit_geometry: dict[str, Any] | None = None
    if args.dit_hidden_size > 0 and args.dit_num_layers > 0 and args.dit_num_tokens > 0:
        dit_geometry = {
            "hidden_size": args.dit_hidden_size,
            "num_layers": args.dit_num_layers,
            "num_tokens": args.dit_num_tokens,
            "ffn_ratio": args.dit_ffn_ratio,
        }
    achievable = args.achievable_tflops or None
    if achievable is None and args.target_platform:
        try:
            from inference_optimizer.orchestrator.roofline_ceiling import _resolve_achievable_tflops

            resolved = _resolve_achievable_tflops(args.target_platform, "bf16")
            achievable = resolved if resolved and resolved > 0 else None
        except Exception:
            achievable = None

    report = build_report(
        Path(args.perf_csv_dir).expanduser().resolve(),
        args.num_denoise_steps or None,
        args.top_k,
        dit_geometry=dit_geometry,
        achievable_tflops=achievable,
    )
    print_summary(report)
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
