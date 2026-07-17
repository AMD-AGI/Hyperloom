#!/usr/bin/env python3
###############################################################################
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Independent (TraceLens-free) trace analysis backend for the bypass route.

This tool is the runtime target of ``HYPERLOOM_TRACE_ANALYSIS_ROUTE=bypass``.
It replaces the TraceLens agent / TraceLens deterministic scripts entirely:
it never imports or shells out to TraceLens. It reads the torch-profiler
Kineto trace produced by the ``profile`` step and emits the same downstream
artifact contract the Coordinator / kernel-agent expect:

    - ``<run_dir>/bypass/analysis.md``       (human-readable report)
    - ``<run_dir>/kernel_candidates.json``   (hot kernels + skipped + groups)
    - ``<run_dir>/bypass/summary.json``      (routed vs skipped audit)
    - ``<workspace>/reports/<roofline>.json``(per-kernel roofline sidecar)
    - ``<run_dir>/trace_input_manifest.json``(input record)

and prints a single JSON result object to stdout in the shape
``kernel_request_handlers._shape_tool_result`` consumes.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any

# Sibling modules live next to this tool (invoked by absolute path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bypass_report as _report  # noqa: E402
import _bypass_trace_reader as _reader  # noqa: E402
from _idle_gate import (  # noqa: E402
    build_high_idle_warning,
    resolve_idle_pct_threshold,
)
from _denoise_steps import resolve_perstep_divisor  # noqa: E402
from _io_utils import atomic_write_json, utc_now, write_text  # noqa: E402


AGGREGATION_SCOPE_FULL = "full_trace"
AGGREGATION_SCOPE_STEADY = "steady_state"


def _run_stamp() -> str:
    """Return a compact UTC timestamp used to name the per-run output dir."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _maybe_enrich_rocprof(
    kernel_roofline_path: Path,
    candidates_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Optionally run rocprof-compute enrichment on the roofline sidecar.

    Opt-in via ``HYPERLOOM_ROCPROF_ROOFLINE_ENRICH`` (off by default). Reuses
    ``rocprof_roofline.enrich_kernel_roofline_sidecar``, which degrades
    gracefully (rocprof-compute missing / non-reusable kernel / no benchmark
    files -> row skipped; per-kernel failure -> row failed) and never aborts.

    Args:
        kernel_roofline_path: Path to the written ``kernel_roofline.json``.
        candidates_path: Path to the written ``kernel_candidates.json``.
        run_dir: Per-run output directory used as the profiling workdir.

    Returns:
        The enrich summary dict, ``{"status": "disabled"}`` when the env gate is
        off, or ``{"status": "error: ..."}`` on unexpected failure. Progress is
        logged to stderr so stdout stays a single result-JSON line.
    """
    enrich_value = os.environ.get("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", "0").strip().lower()
    if enrich_value not in {"1", "true", "yes", "on"}:
        return {"status": "disabled"}
    try:
        from rocprof_roofline import enrich_kernel_roofline_sidecar

        timeout_sec = int(os.environ.get("HYPERLOOM_ROCPROF_ROOFLINE_TIMEOUT_SEC", "1800") or 1800)
        enrich_summary = enrich_kernel_roofline_sidecar(
            sidecar_path=str(kernel_roofline_path),
            candidates_path=str(candidates_path),
            workdir=str(run_dir),
            timeout_sec_per_kernel=timeout_sec,
            log_fn=None,
        )
        print(
            "[rocprof_enrich] "
            f"matched={enrich_summary.get('matched', 0)} "
            f"skipped={enrich_summary.get('skipped', 0)} "
            f"failed={enrich_summary.get('failed', 0)} "
            f"rows={enrich_summary.get('rows', 0)}",
            file=sys.stderr,
        )
        return enrich_summary
    except Exception as exc:  # noqa: BLE001 — enrichment must never break bypass
        msg = f"error: {type(exc).__name__}: {exc}"
        print(f"[rocprof_enrich] skipped: {msg}", file=sys.stderr)
        return {"status": msg}


def _emit_quality_warnings(analyze: dict[str, Any], warnings: list[dict[str, Any]]) -> None:
    """Append analysis-quality health signals so weak analyses are never silent.

    Emits (all non-fatal) when:
      * too much GPU time is unclassified (``Others`` share high) -> taxonomy gap;
      * op-attribution coverage is near-zero -> correlation chain broken;
      * steady-state windowing was requested but fell back to the full trace;
      * only CUDA-graph capture shards were found (no main profiler trace).

    Thresholds are env-tunable (``HYPERLOOM_BYPASS_OTHERS_WARN_PCT`` default 40,
    ``HYPERLOOM_BYPASS_CORR_WARN_PCT`` default 10). Only called when the trace
    yielded GPU kernels (otherwise ``bypass_no_gpu_kernels`` already fired).

    Args:
        analyze: Result of :func:`_bypass_trace_reader.analyze_trace`.
        warnings: The ``trace_health_warnings`` list to append to (mutated).
    """
    try:
        others_thr = float(os.environ.get("HYPERLOOM_BYPASS_OTHERS_WARN_PCT", "40") or 40)
    except ValueError:
        others_thr = 40.0
    try:
        corr_thr = float(os.environ.get("HYPERLOOM_BYPASS_CORR_WARN_PCT", "10") or 10)
    except ValueError:
        corr_thr = 10.0

    rollup = _report._category_rollup(analyze)
    others_pct = next((r["gpu_pct"] for r in rollup if r["category"] == "Others"), 0.0)
    if others_pct >= others_thr:
        warnings.append(
            {
                "code": "bypass_high_unclassified_share",
                "severity": "warning",
                "message": (
                    f"{others_pct}% of GPU time is unclassified (category=Others); "
                    "the kernel-name taxonomy likely needs extension for this workload."
                ),
            }
        )

    corr_pct = float((analyze.get("attribution") or {}).get("attributed_pct") or 0.0)
    if corr_pct < corr_thr:
        warnings.append(
            {
                "code": "bypass_low_op_correlation",
                "severity": "info",
                "message": (
                    f"op-attribution coverage is {corr_pct}% (< {corr_thr}%); kernel-name "
                    "classification still applies, but op names/shapes are largely unresolved "
                    "(expected under cudagraph/torch.compile replay)."
                ),
            }
        )

    if analyze.get("steady_window_status"):
        warnings.append(
            {
                "code": "bypass_steady_fallback_full_trace",
                "severity": "info",
                "message": (
                    "steady-state windowing requested but no repeating window found; "
                    "fell back to full-trace share aggregation."
                ),
            }
        )

    if analyze.get("selected_capture_fragment"):
        warnings.append(
            {
                "code": "bypass_only_capture_fragments",
                "severity": "warning",
                "message": (
                    "no main profiler trace found; analysis ran on a sglang CUDA-graph "
                    "capture shard (device-kernel sparse). Ensure the main "
                    "*-TP-*.trace.json.gz (not just capture_traces/bs_*) was captured."
                ),
            }
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser mirroring the flags the handler forwards."""
    p = argparse.ArgumentParser(description="Hyperloom bypass trace analysis (TraceLens-free)")
    p.add_argument("--trace-input", required=True)
    p.add_argument("--session-id", default="")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--workspace-path", default=os.environ.get("USER_DATA_PATH", "/workspace/hyperloom"))
    p.add_argument("--model-name", default="")
    p.add_argument("--framework", default="")
    p.add_argument("--target-platform", default="")
    p.add_argument("--analysis-mode", default="")
    p.add_argument("--split-conc", default="")
    p.add_argument("--split-osl", default="")
    p.add_argument("--split-r", default="")
    p.add_argument("--capture-folder", default="")
    p.add_argument(
        "--steady-state-mode",
        default="",
        help="Steady-state windowing mode. Any non-off value enables windowing, "
        "including the TraceLens splitter chunk types the coordinator forwards "
        "(mixed / decode_only / prefilldecode); off values: '', 0, false, off, none.",
    )
    p.add_argument("--roofline-output-name", default="kernel_roofline.json")
    # Denoise-step count for scriptable/diffusion workloads; 0 = infer.
    p.add_argument("--num-denoise-steps", type=int, default=0)
    # Diffusion analytic-ceiling inputs shared with the TraceLens CLI surface;
    # parsed but unused on this route.
    p.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ""))
    p.add_argument("--precision", default="")
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--width", type=int, default=0)
    p.add_argument("--cfg-batch", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p


#: ``--steady-state-mode`` values that mean "do NOT window" (analyze the full trace).
_STEADY_OFF_VALUES = frozenset({"", "0", "false", "off", "no", "none"})


def _should_enable_steady(*, steady_state_mode: str, framework: str, env_steady: bool) -> bool:
    """Whether to run steady-state windowing for this trace analysis.

    On for: the env opt-in; xDiT (homogeneous denoise steps -> one representative
    step); OR any non-off ``--steady-state-mode``. The last clause covers the
    TraceLens splitter chunk types the coordinator forwards (``mixed`` /
    ``decode_only`` / ``prefilldecode``) plus the legacy opt-in aliases, so a
    text-gen bypass run windows the requested steady chunk instead of silently
    aggregating the full trace (parity with the TraceLens route). bypass's
    repeat-based windowing then anchors a representative step, or degrades to
    full-trace (with the existing warning) when no window is found.
    """
    mode = (steady_state_mode or "").strip().lower()
    return (
        bool(env_steady)
        or (framework or "").lower() == "xdit"
        or mode not in _STEADY_OFF_VALUES
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point: emit the minimal bypass artifact set and a result JSON.

    Returns:
        Process exit code (``0`` on success). The structured result is printed
        to stdout as a single JSON object regardless of exit code.
    """
    args = _build_arg_parser().parse_args(argv)

    workspace = Path(args.workspace_path)
    session_id = args.session_id or workspace.name
    run_dir = workspace / "kernel-agent" / "runs" / session_id / f"{_run_stamp()}_bypass"
    bypass_dir = run_dir / "bypass"
    reports_dir = workspace / "reports"
    run_dir.mkdir(parents=True, exist_ok=True)
    bypass_dir.mkdir(parents=True, exist_ok=True)

    trace_health_warnings: list[dict[str, Any]] = []
    top_k = args.top_k if args.top_k and args.top_k > 0 else 15

    framework_l = (args.framework or "").lower()
    # Steady-state windowing: opt-in via --steady-state-mode / env, always on
    # for xDiT. Falls back to full-trace shares when no repeating window found.
    env_steady = os.environ.get("HYPERLOOM_BYPASS_STEADY_STATE", "").strip().lower() in {"1", "true", "yes", "on"}
    enable_steady = _should_enable_steady(
        steady_state_mode=args.steady_state_mode or "",
        framework=args.framework or "",
        env_steady=env_steady,
    )

    # --- analyze the trace (independent streaming reader) ---
    analyze: dict[str, Any]
    if args.dry_run:
        analyze = {"status": "ok", "timeline": {}, "attribution": {}, "kernels": [], "ops": [], "aggregation_scope": "full_trace"}
    else:
        try:
            # top_k=0 -> keep all device-kernel aggregates; candidate slicing uses top_k.
            analyze = _reader.analyze_trace(
                args.trace_input, top_k=0, steady_state=enable_steady,
                framework=args.framework, emit_launches=True,
            )
        except Exception as exc:  # noqa: BLE001 — never abort the pipeline
            analyze = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    # ``analysis_degraded`` distinguishes an analysis failure (bad/unparsable
    # trace) from a genuine empty result.
    analysis_degraded = False
    if analyze.get("status") != "ok":
        analysis_degraded = True
        trace_health_warnings.append(
            {
                "code": "bypass_trace_parse_failed",
                "severity": "warning",
                "message": f"bypass reader could not analyze trace: {analyze.get('error', 'unknown')}",
            }
        )
        analyze = {"status": "ok", "timeline": {}, "attribution": {}, "kernels": [], "ops": [], "aggregation_scope": "full_trace"}
    elif not analyze.get("kernels"):
        trace_health_warnings.append(
            {
                "code": "bypass_no_gpu_kernels",
                "severity": "warning",
                "message": "bypass reader found no GPU kernel events in the trace",
            }
        )

    # Aggregation scope is driven by the reader (steady_state vs full_trace).
    scope = analyze.get("aggregation_scope", AGGREGATION_SCOPE_FULL)
    steady_window = analyze.get("steady_window")

    # ``estimated`` marks shares not anchored to a real per-step window (steady
    # windowing requested but fell back to the full trace).
    estimated = enable_steady and scope != AGGREGATION_SCOPE_STEADY
    if framework_l == "xdit" and estimated:
        trace_health_warnings.append(
            {
                "code": "bypass_xdit_estimated",
                "severity": "info",
                "message": (
                    "xDiT analysis fell back to full-trace shares (no per-step denoising "
                    "window found; trace lacks step annotations such as ProfilerStep) — "
                    "treat kernel shares / roofline as estimated."
                ),
            }
        )
    elif framework_l == "xdit":
        trace_health_warnings.append(
            {
                "code": "bypass_xdit_steady_anchored",
                "severity": "info",
                "message": (
                    f"xDiT analysis anchored to a real per-step denoising window "
                    f"({(steady_window or {}).get('step_name', 'step')}×"
                    f"{(steady_window or {}).get('step_count', 0)}); per-step kernel "
                    "shares are trace-anchored (not estimated)."
                ),
            }
        )

    # Multi-rank provenance: xDiT TP>1 produces one trace per rank; the reader
    # analyzes one representative rank.
    analyzed_rank = analyze.get("analyzed_rank")
    rank_count = analyze.get("rank_count", 1)
    if isinstance(rank_count, int) and rank_count > 1:
        trace_health_warnings.append(
            {
                "code": "bypass_multi_rank_single_analyzed",
                "severity": "info",
                "message": (
                    f"trace dir has {rank_count} per-rank traces; analyzed rank {analyzed_rank} "
                    "as representative (each rank runs the same kernels on sharded data under "
                    "sequence/tensor parallel)."
                ),
            }
        )

    # Warn when the forwarded --num-denoise-steps differs from the count the
    # reader inferred from ProfilerStep annotations.
    requested_denoise_steps = int(getattr(args, "num_denoise_steps", 0) or 0)
    inferred_denoise_steps = int((steady_window or {}).get("step_count", 0) or 0) or int(
        (analyze.get("attribution") or {}).get("annotation_window_count", 0) or 0
    )
    if requested_denoise_steps > 0 and inferred_denoise_steps > 0 and requested_denoise_steps != inferred_denoise_steps:
        trace_health_warnings.append(
            {
                "code": "bypass_denoise_steps_mismatch",
                "severity": "info",
                "message": (
                    f"requested --num-denoise-steps={requested_denoise_steps} differs from the "
                    f"{inferred_denoise_steps} step(s) inferred from the trace annotations; the "
                    "trace-inferred per-step window is used for kernel shares."
                ),
            }
        )

    # Analysis-quality health signals (observability only; never fatal).
    if analyze.get("kernels"):
        _emit_quality_warnings(analyze, trace_health_warnings)

    # --- build downstream artifacts from classified device kernels ---
    candidates = _report.build_candidates(
        analyze,
        framework=args.framework,
        target_platform=args.target_platform,
        top_k=top_k,
        discover_benchmarks=False,
    )

    analysis_md_path = bypass_dir / "analysis.md"
    candidates_path = run_dir / "kernel_candidates.json"
    summary_path = bypass_dir / "summary.json"
    manifest_path = run_dir / "trace_input_manifest.json"
    roofline_name = args.roofline_output_name or "kernel_roofline.json"
    kernel_roofline_path = reports_dir / roofline_name
    kernel_sequence_path = bypass_dir / "kernel_sequence.json"
    kernel_metrics_csv_path = bypass_dir / "kernel_metrics.csv"
    kernel_summary_csv_path = bypass_dir / "kernel_summary.csv"

    # High-idle gate: when GPU idle exceeds the threshold, per-kernel rewriting
    # cannot move end-to-end latency, so suppress every candidate list and
    # surface a high_gpu_idle_pct warning for the Coordinator.
    idle_pct_value = (analyze.get("timeline") or {}).get("idle_pct")
    idle_pct_threshold = resolve_idle_pct_threshold()
    if isinstance(idle_pct_value, (int, float)) and float(idle_pct_value) > idle_pct_threshold:
        for _cand_key in ("hot_kernels", "routable_kernels", "skipped_kernels", "task_groups"):
            candidates[_cand_key] = []
        trace_health_warnings.append(
            build_high_idle_warning(
                idle_pct=float(idle_pct_value),
                threshold_pct=idle_pct_threshold,
                report_path=analysis_md_path,
            )
        )

    # Stamp the report path onto each candidate (downstream reads it).
    for cand in candidates.get("hot_kernels", []):
        cand["trace_report_path"] = str(analysis_md_path)

    throughput_unit = "img/s" if (args.framework or "").lower() == "xdit" else "tok/s"
    write_text(
        analysis_md_path,
        _report.render_analysis_md(
            candidates,
            analyze,
            model_name=args.model_name,
            framework=args.framework,
            target_platform=args.target_platform,
            throughput_unit=throughput_unit,
            metrics_csv_path=str(kernel_metrics_csv_path),
            summary_csv_path=str(kernel_summary_csv_path),
        ),
    )
    atomic_write_json(candidates_path, candidates, ensure_ascii=False, sort_keys=False, trailing_newline=False)
    # CSV exports: full per-kernel metrics + category summary.
    write_text(kernel_metrics_csv_path, _report.build_metrics_csv(candidates))
    write_text(kernel_summary_csv_path, _report.build_category_summary_csv(candidates))

    summary = _report.build_summary(
        candidates,
        framework=args.framework,
        target_platform=args.target_platform,
        generated_at=utc_now(timespec="seconds"),
        trace_health_warnings=trace_health_warnings,
    )
    summary["estimated"] = estimated
    summary["analysis_degraded"] = analysis_degraded
    # Always present (may be null) so the summary/manifest/result schemas match.
    summary["steady_window"] = steady_window  # always present (may be null)

    atomic_write_json(
        manifest_path,
        {
            "source": "bypass",
            "trace_input": str(args.trace_input),
            "trace_file": analyze.get("trace_file", ""),
            "capture_folder": args.capture_folder or None,
            "aggregation_scope": scope,
            "steady_window": steady_window,
            "estimated": estimated,
            "analysis_degraded": analysis_degraded,
            "analyzed_rank": analyzed_rank,
            "rank_count": rank_count,
            "event_total": analyze.get("event_total", 0),
            "created_at": utc_now(timespec="seconds"),
        },
        ensure_ascii=False,
        sort_keys=False,
        trailing_newline=False,
    )

    kernel_roofline = _report.build_kernel_roofline(
        candidates,
        analysis_md_path=str(analysis_md_path),
        kernel_candidates_path=str(candidates_path),
    )
    atomic_write_json(kernel_roofline_path, kernel_roofline, ensure_ascii=False, sort_keys=False, trailing_newline=False)

    # Optional rocprof-compute enrichment (opt-in; enriches the sidecar in
    # place). Skipped in --dry-run.
    rocprof_enrich: dict[str, Any] = (
        {"status": "disabled"}
        if args.dry_run
        else _maybe_enrich_rocprof(kernel_roofline_path, candidates_path, run_dir)
    )
    summary["rocprof_enrich"] = rocprof_enrich
    atomic_write_json(summary_path, summary, ensure_ascii=False, sort_keys=False, trailing_newline=False)

    # Kernel-fusion opportunities: launch adjacency -> fusable clusters.
    fusion = _report.build_fusion(analyze)
    atomic_write_json(kernel_sequence_path, fusion, ensure_ascii=False, sort_keys=False, trailing_newline=False)

    # Diffusion / scriptable workload-level roofline: aggregate the per-kernel
    # analytical roofline into an end-to-end workload roofline + per-denoise-step
    # split. Best-effort sidecar over all device kernels, independent of the
    # per-kernel high-idle gate, so still emitted in the high-idle regime.
    diffusion_roofline_path: str | None = None
    if (args.framework or "").lower() == "xdit":
        try:
            from diffusion_roofline import build_report_from_bypass  # noqa: E402

            # Per-step divisor is the denoise steps in the analyzed window
            # (trace-inferred), not the requested full schedule.
            _diff_steps = resolve_perstep_divisor(inferred_denoise_steps, requested_denoise_steps)
            # Workload totals cover all analyzed device kernels (not just top-k).
            _workload_totals = _report.build_workload_roofline_totals(
                analyze, target_platform=args.target_platform
            )
            _all_kernels = [k for k in (analyze.get("kernels") or []) if float(k.get("gpu_time_us") or 0.0) > 0]
            _diff_report = build_report_from_bypass(
                candidates.get("hot_kernels", []),
                analyze.get("timeline") or {},
                _diff_steps,
                top_k,
                totals=_workload_totals,
                kernels_aggregated=len(_all_kernels),
            )
            _diff_path = run_dir / "diffusion_roofline.json"
            atomic_write_json(_diff_path, _diff_report, ensure_ascii=False, sort_keys=False, trailing_newline=False)
            diffusion_roofline_path = str(_diff_path)
        except Exception:  # noqa: BLE001 - best-effort sidecar, never blocks the run
            diffusion_roofline_path = None

    hot_kernels = candidates.get("hot_kernels", [])
    result: dict[str, Any] = {
        "status": "ok",
        "route": "bypass",
        "aggregation_scope": scope,
        "steady_window": steady_window,
        "estimated": estimated,
        "analysis_degraded": analysis_degraded,
        "analyzed_rank": analyzed_rank,
        "rank_count": rank_count,
        "num_denoise_steps": requested_denoise_steps or inferred_denoise_steps,
        "framework": args.framework,
        "target_platform": args.target_platform,
        "hot_kernels": hot_kernels,
        "hot_kernels_top15": hot_kernels[:15],
        "routable_kernels": candidates.get("routable_kernels", []),
        "skipped_kernels": candidates.get("skipped_kernels", []),
        "task_groups": candidates.get("task_groups", []),
        "candidates_path": str(candidates_path),
        "trace_report_path": str(analysis_md_path),
        "kernel_roofline_path": str(kernel_roofline_path),
        "tracelens_summary_path": str(summary_path),
        "kernel_sequence_path": str(kernel_sequence_path),
        "kernel_metrics_csv_path": str(kernel_metrics_csv_path),
        "kernel_summary_csv_path": str(kernel_summary_csv_path),
        "fusion": {
            "launch_count": fusion.get("launch_count", 0),
            "fusable_cluster_count": fusion.get("fusable_cluster_count", 0),
            "fusable_time_us": fusion.get("fusable_time_us", 0.0),
        },
        "orchestrator_mode": "bypass",
        "timeline": analyze.get("timeline") or {},
        "attribution": analyze.get("attribution") or {},
        "trace_health_warnings": trace_health_warnings,
        "artifact_paths": {
            "trace_report_path": str(analysis_md_path),
            "kernel_candidates": str(candidates_path),
            "kernel_roofline": str(kernel_roofline_path),
            "tracelens_summary": str(summary_path),
            "kernel_sequence": str(kernel_sequence_path),
            "kernel_metrics_csv": str(kernel_metrics_csv_path),
            "kernel_summary_csv": str(kernel_summary_csv_path),
            "trace_input_manifest": str(manifest_path),
        },
    }
    # Surfaced only when produced (xDiT/scriptable).
    if diffusion_roofline_path:
        result["diffusion_roofline_path"] = diffusion_roofline_path
        result["artifact_paths"]["diffusion_roofline"] = diffusion_roofline_path
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
