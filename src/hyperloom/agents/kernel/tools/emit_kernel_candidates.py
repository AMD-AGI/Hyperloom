#!/usr/bin/env python3
###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Emit ``kernel_candidates.json`` from an existing TraceLens ``analysis_output``.

Reads ``analysis.md`` (ranking source of truth) plus TraceLens sidecars already
on disk. Does not run the kernel-agent e2e path (no optimize / Magpie /
Coordinator) and does not re-run the TraceLens orchestrator.

Invoked like the other kernel-agent tools (absolute path, sibling imports):

    python src/hyperloom/agents/kernel/tools/emit_kernel_candidates.py \\
        --analysis-output /path/to/analysis_output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Sibling modules live next to this tool (invoked by absolute path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracelens_analysis import (  # noqa: E402
    _SOURCE_RESOLUTION_NAME,
    _default_top_k,
    _evaluate_high_idle_gate,
    _evaluate_idle_gate_with_graph_guard,
    _evaluate_low_compute_gate,
    _extract_total_time_us_from_gpu_timeline,
    _finalize_candidates,
    _inject_collective_candidates,
    load_roofline_results,
    merge_roofline_into_candidates,
    recover_other_bucket_candidates,
    write_reports,
)
from tracelens_skill_runner import (  # noqa: E402
    _parse_kernel_name_cell,
    extract_compute_pct_from_analysis_md,
    extract_exposed_comm_pct_from_analysis_md,
    extract_idle_pct_from_analysis_md,
    parse_analysis_md,
)

_TRUNCATION_MARK = "..."
_INFERRED_PLACEHOLDERS = ("cannot be inferred", "unknown", "n/a", "-")


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object, or ``{}`` when the file is missing/invalid."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _looks_truncated(value: str) -> bool:
    """True when a TraceLens markdown cell was truncated with ``...``."""
    return _TRUNCATION_MARK in (value or "")


def overlay_metrics_kernel_names(
    candidates: list[dict[str, Any]],
    analysis_output: Path,
) -> None:
    """Replace truncated ``device_kernel_name(s)`` from ``category_data/*_metrics.json``.

    Mutates ``candidates`` in place. Matching is by operation name. Fusion
    sidecars are skipped because they are not per-op ranking rows.
    """
    metrics_dir = analysis_output / "category_data"
    if not metrics_dir.is_dir():
        return
    by_op: dict[str, dict[str, Any]] = {}
    for path in sorted(metrics_dir.glob("*_metrics.json")):
        if path.name == "kernel_fusion_metrics.json":
            continue
        payload = _load_json(path)
        for op in payload.get("operations") or []:
            if not isinstance(op, dict):
                continue
            name = str(op.get("name") or "").strip()
            if name:
                by_op[name] = op
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        op = by_op.get(str(cand.get("name") or "").strip())
        if not op:
            continue
        full_cell = str(op.get("kernel_name") or "")
        parsed = _parse_kernel_name_cell(full_cell.replace("<br>", "\n"))
        current_names = [str(n) for n in (cand.get("device_kernel_names") or []) if n]
        current_one = str(cand.get("device_kernel_name") or "")
        truncated = (
            (not current_names and parsed)
            or _looks_truncated(current_one)
            or any(_looks_truncated(n) for n in current_names)
        )
        if parsed and truncated:
            cand["device_kernel_name"] = parsed[0]
            cand["device_kernel_names"] = parsed
        if not cand.get("shapes"):
            args = str(op.get("args") or "").replace("<br>", "\n").strip()
            shapes = [s.strip() for s in args.split("\n") if s.strip() and s.strip() not in {"-", "—"}]
            if shapes:
                cand["shapes"] = shapes


def _infer_model_name(analysis_output: Path) -> str:
    """Return a usable model name from TraceLens metadata, or empty."""
    info = _load_json(analysis_output / "metadata" / "model_info.json")
    raw = str(info.get("model") or "").strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if lowered in _INFERRED_PLACEHOLDERS or lowered.startswith("cannot be inferred"):
        return ""
    return raw


def _infer_platform(manifest: dict[str, Any], fallback: str) -> str:
    """Prefer the orchestrator platform recorded in ``category_manifest.json``."""
    platform = str(manifest.get("platform") or "").strip()
    return platform or fallback


def _infer_analysis_mode(manifest: dict[str, Any], fallback: str) -> str:
    """Map TraceLens ``comparison_scope`` onto Hyperloom ``analysis_mode``."""
    scope = str(manifest.get("comparison_scope") or "").strip().lower()
    if scope in {"standalone", "comparative"}:
        return scope
    return fallback


def _resolve_trace_input(
    *,
    explicit: str,
    manifest: dict[str, Any],
    analysis_output: Path,
) -> Path:
    """Pick a trace path for the Hyperloom manifest without requiring a probe."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    raw = str(manifest.get("trace_path") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return analysis_output.resolve()


def emit_kernel_candidates_from_analysis_output(
    analysis_output: Path,
    out_dir: Path,
    *,
    model_name: str = "",
    framework: str = "",
    target_platform: str = "MI300X",
    analysis_mode: str = "default",
    source_root: str | None = None,
    top_k: int | None = None,
    trace_input: str = "",
    roofline_json: str = "",
    probe_graph: bool = False,
    runtime_env: str = "local",
) -> dict[str, Any]:
    """Parse existing TraceLens artifacts and write Hyperloom candidate sidecars.

    Args:
        analysis_output: Directory that already contains ``analysis.md``.
        out_dir: Directory to write ``kernel_candidates.json`` and related reports.
        model_name: Optional model identity recorded on candidates.
        framework: Optional framework (``vllm`` / ``sglang`` steer source routing).
        target_platform: GPU target recorded in the candidate report.
        analysis_mode: Hyperloom analysis mode (defaults from the manifest).
        source_root: Optional root for launcher-path AST resolution.
        top_k: Candidate cap; ``None`` uses Hyperloom's default / env override.
        trace_input: Optional raw trace path for the written manifest.
        roofline_json: Optional extra roofline JSON to merge.
        probe_graph: When True and a trace file exists, apply the graph-guard idle gate.
        runtime_env: Runtime label written into the report.

    Returns:
        A dict with ``kernel_candidates_path``, warnings, and the ``write_reports``
        artifact map.

    Raises:
        FileNotFoundError: When ``analysis.md`` is missing.
        RuntimeError: When analysis.md exists but yields no candidates and the
            idle/low-compute gates did not fire (same contract as the
            TraceLens analysis tool).
    """
    analysis_output = analysis_output.expanduser().resolve()
    report_path = analysis_output / "analysis.md"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"analysis.md is required (TraceLens ranking source of truth): {report_path}"
        )

    cap = _default_top_k() if top_k is None else top_k
    manifest = _load_json(analysis_output / "category_data" / "category_manifest.json")
    model_name = model_name or _infer_model_name(analysis_output)
    target_platform = _infer_platform(manifest, target_platform)
    analysis_mode = (
        analysis_mode if analysis_mode and analysis_mode != "default" else _infer_analysis_mode(manifest, analysis_mode)
    )
    resolved_trace = _resolve_trace_input(
        explicit=trace_input,
        manifest=manifest,
        analysis_output=analysis_output,
    )

    idle_pct = extract_idle_pct_from_analysis_md(report_path)
    compute_pct = extract_compute_pct_from_analysis_md(report_path)
    exposed_comm_pct = extract_exposed_comm_pct_from_analysis_md(report_path)

    trace_health_warnings: list[dict[str, Any]] = []
    graph_warning = None
    if probe_graph and resolved_trace.is_file():
        idle_threshold, high_idle_warning, graph_warning = _evaluate_idle_gate_with_graph_guard(
            idle_pct, report_path, resolved_trace
        )
    else:
        idle_threshold, high_idle_warning = _evaluate_high_idle_gate(idle_pct, report_path)
    compute_threshold, low_compute_warning = _evaluate_low_compute_gate(
        compute_pct, exposed_comm_pct, report_path
    )
    if graph_warning is not None:
        high_idle_warning = None
        low_compute_warning = None
        trace_health_warnings.append(graph_warning)

    allow_empty = False
    report_source = "analysis.md"
    candidates: list[dict[str, Any]] = []
    if high_idle_warning is not None or low_compute_warning is not None:
        allow_empty = True
        skipped: list[str] = []
        if high_idle_warning is not None:
            trace_health_warnings.append(high_idle_warning)
            skipped.append("skipped:high_gpu_idle_pct")
        if low_compute_warning is not None:
            trace_health_warnings.append(low_compute_warning)
            skipped.append("skipped:low_gpu_compute_pct")
        report_source = "+".join(skipped)
    else:
        report_cands = parse_analysis_md(report_path, cap)
        overlay_metrics_kernel_names(report_cands, analysis_output)
        fallback_cands = recover_other_bucket_candidates(
            analysis_output,
            report_cands,
            top_k=cap,
            total_window_us=_extract_total_time_us_from_gpu_timeline(analysis_output),
        )
        if fallback_cands:
            overlay_metrics_kernel_names(fallback_cands, analysis_output)
            report_cands = report_cands + fallback_cands
        raw = _inject_collective_candidates(
            analysis_output,
            report_cands,
            health_warnings=trace_health_warnings,
        )
        source_parts = ["analysis.md"]
        if fallback_cands:
            source_parts.append("other_bucket_fallback")
        if len(raw) > len(report_cands):
            source_parts.append("nccl_summary")
        report_source = "+".join(source_parts)
        if not raw:
            allow_empty = True
            candidates = []
        else:
            total_dur = _extract_total_time_us_from_gpu_timeline(analysis_output) or sum(
                float(c.get("duration_us") or 0) for c in raw
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            candidates = _finalize_candidates(
                raw,
                total_dur=total_dur or None,
                perf_report_csv_dir=(analysis_output / "perf_report_csvs"),
                framework=framework or None,
                trace_files=None,
                log_path=out_dir / "emit_kernel_candidates.log",
                source_resolution_out=(out_dir / _SOURCE_RESOLUTION_NAME),
                model_name=model_name,
            )

    if not candidates and not allow_empty:
        raise RuntimeError(
            "No hot-kernel candidates produced from analysis.md (and other-bucket "
            "recovery found nothing). Refusing CSV-as-ranking fallback because "
            "analysis.md is the single source of truth."
        )

    merge_roofline_into_candidates(candidates, load_roofline_results(roofline_json or None))

    args = argparse.Namespace(
        trace_input=str(resolved_trace),
        model_name=model_name,
        framework=framework,
        target_platform=target_platform,
        analysis_mode=analysis_mode,
        runtime_env=runtime_env,
        dry_run=False,
        source_root=source_root,
        roofline_json=roofline_json,
        roofline_output_name="kernel_roofline.json",
        num_denoise_steps=0,
        model_path="",
        precision="",
        height=0,
        width=0,
        cfg_batch=0,
        top_k=cap,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = write_reports(
        out_dir,
        trace_input_type="file" if resolved_trace.is_file() else "analysis_output",
        trace_files=[resolved_trace] if resolved_trace.is_file() else [],
        candidates=candidates,
        args=args,
        existing_report_path=report_path,
        trace_health_warnings=trace_health_warnings,
    )
    kernel_candidates_path = out_dir / "kernel_candidates.json"
    return {
        "tool": "emit_kernel_candidates",
        "analysis_md": str(report_path),
        "kernel_candidates_path": str(kernel_candidates_path),
        "report_source": report_source,
        "idle_pct": idle_pct,
        "compute_pct": compute_pct,
        "idle_pct_threshold": idle_threshold,
        "compute_pct_threshold": compute_threshold,
        "hot_kernel_count": len(candidates),
        "trace_health_warnings": trace_health_warnings,
        "artifact_paths": artifacts,
    }


def build_parser() -> argparse.ArgumentParser:
    """CLI for offline candidate emit from a TraceLens analysis_output."""
    parser = argparse.ArgumentParser(
        description=(
            "Emit kernel_candidates.json from an existing TraceLens "
            "analysis_output (no TraceLens orchestrator rerun)."
        )
    )
    parser.add_argument(
        "--analysis-output",
        required=True,
        help="Directory containing analysis.md and TraceLens sidecars.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Where to write kernel_candidates.json (default: <analysis-output>/hyperloom_kernel_candidates).",
    )
    parser.add_argument("--model-name", default="")
    parser.add_argument("--framework", default="")
    parser.add_argument("--target-platform", default="MI300X")
    parser.add_argument("--analysis-mode", default="default")
    parser.add_argument(
        "--source-root",
        default=os.environ.get("TRACELENS_SOURCE_ROOT", "") or None,
        help="Optional root for resolving TraceLens launcher paths.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Candidate cap (default: _default_top_k / HYPERLOOM_KERNEL_CANDIDATES_TOP_K).",
    )
    parser.add_argument(
        "--trace-input",
        default="",
        help="Optional raw trace path recorded in the Hyperloom manifest.",
    )
    parser.add_argument("--roofline-json", default="")
    parser.add_argument(
        "--probe-graph",
        action="store_true",
        help="If the raw trace exists, apply the graph-under-recording idle guard.",
    )
    parser.add_argument("--runtime-env", default="local")
    return parser


def main() -> int:
    """Emit candidates and print a JSON summary."""
    parser = build_parser()
    ns = parser.parse_args()
    analysis_output = Path(ns.analysis_output)
    out_dir = Path(ns.out_dir) if ns.out_dir else analysis_output / "hyperloom_kernel_candidates"
    result = emit_kernel_candidates_from_analysis_output(
        analysis_output,
        out_dir,
        model_name=ns.model_name,
        framework=ns.framework,
        target_platform=ns.target_platform,
        analysis_mode=ns.analysis_mode,
        source_root=ns.source_root,
        top_k=ns.top_k,
        trace_input=ns.trace_input,
        roofline_json=ns.roofline_json,
        probe_graph=ns.probe_graph,
        runtime_env=ns.runtime_env,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
