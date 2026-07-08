#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

M1 scope: route skeleton — resolve paths, emit *valid* (possibly empty)
artifacts, and a well-formed result JSON so the whole bypass pipeline runs
end-to-end. Real trace parsing / candidate generation / roofline land in the
subsequent milestones (M2+), which fill these same files.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any

# Sibling modules live next to this tool; it is invoked by absolute path, so
# put its own directory on sys.path before importing them.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bypass_report as _report  # noqa: E402
import _bypass_trace_reader as _reader  # noqa: E402
from _idle_gate import (  # noqa: E402
    build_high_idle_warning,
    resolve_idle_pct_threshold,
)
from _denoise_steps import resolve_perstep_divisor  # noqa: E402


AGGREGATION_SCOPE_FULL = "full_trace"
AGGREGATION_SCOPE_STEADY = "steady_state"


def _utc_now_iso() -> str:
    """Return the current UTC time as a second-precision ISO-8601 string."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _run_stamp() -> str:
    """Return a compact UTC timestamp used to name the per-run output dir."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as pretty JSON to ``path`` via a temp-file rename.

    Args:
        path: Destination file path (parent dirs are created).
        payload: JSON-serializable object to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _maybe_enrich_rocprof(
    kernel_roofline_path: Path,
    candidates_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Optionally run rocprof-compute enrichment on the roofline sidecar.

    Opt-in via ``HYPERLOOM_ROCPROF_ROOFLINE_ENRICH`` (off by default), mirroring
    the TraceLens ``write_reports`` enrichment so the bypass sidecar carries the
    same per-kernel ``rocprof_roofline`` audit fields. Reuses
    ``rocprof_roofline.enrich_kernel_roofline_sidecar``, which degrades
    gracefully (rocprof-compute missing / non-reusable kernel / no benchmark
    files -> row skipped; per-kernel failure -> row failed) and never aborts.

    The heavy per-kernel rocprof runs (before GEAK / after kernel-opt) happen
    later in the route-agnostic kernel-opt phase; this stage is only the opt-in
    batch pass.

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
    # Denoise-step count for scriptable/diffusion workloads (forwarded by the
    # coordinator). Consumed for per-step diffusion roofline + step-count
    # validation; 0 means "infer from the trace".
    p.add_argument("--num-denoise-steps", type=int, default=0)
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
    # Steady-state windowing: opt-in via --steady-state-mode / env, and always
    # on for xDiT (homogeneous diffusion steps -> profile one representative
    # step). Falls back to full-trace shares when no repeating window is found.
    env_steady = os.environ.get("HYPERLOOM_BYPASS_STEADY_STATE", "").strip().lower() in {"1", "true", "yes", "on"}
    enable_steady = _should_enable_steady(
        steady_state_mode=args.steady_state_mode or "",
        framework=args.framework or "",
        env_steady=env_steady,
    )
    # ``estimated`` is decided after the scope is known (below): xDiT is estimated
    # only when it could NOT anchor to a real per-step denoising window.

    # --- analyze the trace (independent streaming reader) ---
    analyze: dict[str, Any]
    if args.dry_run:
        analyze = {"status": "ok", "timeline": {}, "attribution": {}, "kernels": [], "ops": [], "aggregation_scope": "full_trace"}
    else:
        try:
            # top_k=0 -> keep all device-kernel aggregates so the category
            # rollup in the report is complete; candidate slicing uses ``top_k``.
            analyze = _reader.analyze_trace(
                args.trace_input, top_k=0, steady_state=enable_steady,
                framework=args.framework, emit_launches=True,
            )
        except Exception as exc:  # noqa: BLE001 — never abort the pipeline
            analyze = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    # ``analysis_degraded`` distinguishes "analysis actually failed" (bad/unparsable
    # trace) from a genuine empty result: the pipeline still degrades gracefully
    # (status stays ``ok``, never aborts) but downstream/record_trace_analyze can
    # tell the LLM the analysis did NOT really succeed instead of trusting empty.
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

    # Aggregation scope is driven by the reader (steady_state vs full_trace) so
    # every artifact reports the same, accurate scope.
    scope = analyze.get("aggregation_scope", AGGREGATION_SCOPE_FULL)
    steady_window = analyze.get("steady_window")

    # ``estimated`` marks shares that are NOT anchored to a real per-step window.
    # It applies to ANY framework that requested steady-state windowing (xDiT
    # auto-on, or text-gen via --steady-state-mode/env) but fell back to full
    # trace: the shares then mix warmup/all-steps and are a mixed estimate. When
    # the window locks on (scope==steady_state), shares are trace-anchored -> not
    # estimated. Text-gen WITHOUT steady requested keeps full-trace as its norm
    # (enable_steady False -> not estimated).
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

    # Multi-rank provenance: xDiT TP>1 produces one trace per rank. The reader
    # deterministically analyzes one representative rank; surface which one and
    # how many existed so a single-rank analysis is never silent.
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

    # Denoise-step-count validation: the coordinator forwards --num-denoise-steps
    # (the profiled/scheduled step count) for scriptable/diffusion workloads. Do
    # NOT silently drop it — compare against the step count the reader inferred
    # from ProfilerStep annotations and warn on a mismatch (it is a key per-step
    # diffusion-roofline input consumed by diffusion_roofline).
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

    # Analysis-quality health signals (observability only; never fatal). Gated
    # on GPU kernels present so shares are meaningful.
    if analyze.get("kernels"):
        _emit_quality_warnings(analyze, trace_health_warnings)

    # --- build downstream artifacts from classified device kernels ---
    # Discover per-kernel benchmark files only when the rocprof roofline
    # enrichment (the sole consumer) is enabled, so default runs skip the grep.
    enrich_enabled = os.environ.get("HYPERLOOM_ROCPROF_ROOFLINE_ENRICH", "0").strip().lower() in {"1", "true", "yes", "on"}
    candidates = _report.build_candidates(
        analyze,
        framework=args.framework,
        target_platform=args.target_platform,
        top_k=top_k,
        discover_benchmarks=enrich_enabled and not args.dry_run,
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

    # High-idle gate (contract parity with the TraceLens route): when the GPU is
    # idle for more than the shared threshold of the trace wall span, per-kernel
    # rewriting cannot move end-to-end latency, so suppress every candidate list
    # and surface a high_gpu_idle_pct warning for the Coordinator to route to
    # parameter optimization. Uses the SAME threshold/metric/warning as TraceLens.
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
    _write_text(
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
    _atomic_write_json(candidates_path, candidates)
    # Structured, code-generated CSV exports (full per-kernel metrics + category
    # summary); machine-readable ground truth that downstream code can load by path.
    _write_text(kernel_metrics_csv_path, _report.build_metrics_csv(candidates))
    _write_text(kernel_summary_csv_path, _report.build_category_summary_csv(candidates))

    summary = _report.build_summary(
        candidates,
        framework=args.framework,
        target_platform=args.target_platform,
        generated_at=_utc_now_iso(),
        trace_health_warnings=trace_health_warnings,
    )
    # summary.json is written below, after the optional rocprof enrichment, so
    # its ``rocprof_enrich`` audit field reflects the enrichment outcome.
    summary["estimated"] = estimated
    summary["analysis_degraded"] = analysis_degraded
    # Always present (may be null) so the summary/manifest/result schemas match.
    summary["steady_window"] = steady_window

    _atomic_write_json(
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
            "created_at": _utc_now_iso(),
        },
    )

    kernel_roofline = _report.build_kernel_roofline(
        candidates,
        analysis_md_path=str(analysis_md_path),
        kernel_candidates_path=str(candidates_path),
    )
    _atomic_write_json(kernel_roofline_path, kernel_roofline)

    # Optional rocprof-compute enrichment (opt-in; enriches the sidecar in
    # place). Skipped in --dry-run; env-gated + graceful degradation otherwise.
    rocprof_enrich: dict[str, Any] = (
        {"status": "disabled"}
        if args.dry_run
        else _maybe_enrich_rocprof(kernel_roofline_path, candidates_path, run_dir)
    )
    summary["rocprof_enrich"] = rocprof_enrich
    _atomic_write_json(summary_path, summary)

    # Kernel-fusion opportunities: time-ordered launch adjacency -> fusable
    # clusters (Elementwise/Norm/Quant chains). A separate artifact carrying the
    # kernel-relationship data the name-aggregated candidates cannot express.
    fusion = _report.build_fusion(analyze)
    _atomic_write_json(kernel_sequence_path, fusion)

    # Diffusion / scriptable workload-level roofline (parity with the TraceLens
    # route's diffusion_roofline.json): aggregate the per-kernel analytical
    # roofline into an end-to-end workload roofline + per-denoise-step split
    # (consuming the effective num_denoise_steps). Best-effort sidecar; never
    # blocks the per-kernel artifacts.
    #
    # NOTE: this is a WORKLOAD-level characterization aggregated from analyze[
    # "kernels"] (all device kernels), intentionally INDEPENDENT of the per-kernel
    # high-idle gate above -- that gate suppresses per-kernel rewrite CANDIDATES,
    # but the workload compute/memory roofline stays valid and useful even in the
    # high-idle regime, so it is still emitted.
    diffusion_roofline_path: str | None = None
    if (args.framework or "").lower() == "xdit":
        try:
            from diffusion_roofline import build_report_from_bypass  # noqa: E402

            # Per-step divisor is the denoise steps IN the analyzed window
            # (trace-inferred, e.g. steady_window.step_count), NOT the requested
            # full schedule -- the mismatch is already surfaced as a warning.
            _diff_steps = resolve_perstep_divisor(inferred_denoise_steps, requested_denoise_steps)
            # Workload totals cover ALL analyzed device kernels (not just the
            # top-k candidate list) so the roofline is not truncated to the
            # hottest few (parity with the TraceLens full-CSV aggregation).
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
            _atomic_write_json(_diff_path, _diff_report)
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
        "rocprof_enrich": rocprof_enrich,
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
    # Surfaced only when produced (xDiT/scriptable), mirroring the TraceLens route.
    if diffusion_roofline_path:
        result["diffusion_roofline_path"] = diffusion_roofline_path
        result["artifact_paths"]["diffusion_roofline"] = diffusion_roofline_path
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
