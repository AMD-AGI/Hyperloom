#!/usr/bin/env python3
###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Emit ``kernel_candidates.json`` from an existing TraceLens ``analysis_output``.

Thin CLI over :func:`tracelens_analysis.build_kernel_candidates_from_analysis_md`.
Does not run the kernel-agent e2e path (no optimize / Magpie / Coordinator)
and does not re-run the TraceLens orchestrator.

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

import tracelens_analysis as tla
from _io_utils import read_json

_INFERRED_PLACEHOLDERS = frozenset({"cannot be inferred", "unknown", "n/a", "-"})


def _infer_model_name(analysis_output: Path) -> str:
    """Return a usable model name from TraceLens metadata, or empty."""
    info = read_json(analysis_output / "metadata" / "model_info.json", default={}, require_dict=True)
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
    """Read an existing analysis_output and write Hyperloom candidate sidecars.

    Candidate construction and report writing live in ``tracelens_analysis`` so
    this path cannot drift from the pipeline. This function only fills in the
    fields the CLI knows (model, platform, trace path) and prints a summary.

    Raises:
        FileNotFoundError: When ``analysis.md`` is missing.
        RuntimeError: When analysis.md yielded rows that finalizing dropped and
            no idle/low-compute gate explains the empty list.
    """
    analysis_output = analysis_output.expanduser().resolve()
    report_path = analysis_output / "analysis.md"
    if not report_path.is_file():
        raise FileNotFoundError(f"analysis.md is required (TraceLens ranking source of truth): {report_path}")

    manifest = read_json(
        analysis_output / "category_data" / "category_manifest.json",
        default={},
        require_dict=True,
    )
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
    out_dir.mkdir(parents=True, exist_ok=True)
    built = tla.build_kernel_candidates_from_analysis_md(
        report_path,
        analysis_output,
        top_k=top_k,
        framework=framework or None,
        model_name=model_name,
        trace_files=None,
        graph_trace_path=resolved_trace if probe_graph else None,
        log_path=out_dir / "emit_kernel_candidates.log",
        source_resolution_out=out_dir / tla._SOURCE_RESOLUTION_NAME,
        roofline_json=roofline_json,
        probe_graph=probe_graph,
    )
    candidates = built["candidates"]
    if not candidates and not built["allow_empty"]:
        raise RuntimeError(
            "No hot-kernel candidates produced from analysis.md (and other-bucket "
            "recovery found nothing). Refusing CSV-as-ranking fallback because "
            "analysis.md is the single source of truth."
        )
    artifacts = tla.write_kernel_candidate_reports(
        out_dir,
        trace_input=resolved_trace,
        trace_input_type="file" if resolved_trace.is_file() else "analysis_output",
        trace_files=[resolved_trace] if resolved_trace.is_file() else [],
        candidates=candidates,
        model_name=model_name,
        framework=framework,
        target_platform=target_platform,
        analysis_mode=analysis_mode,
        runtime_env=runtime_env,
        source_root=source_root,
        roofline_json=roofline_json,
        existing_report_path=report_path,
        trace_health_warnings=built["trace_health_warnings"],
        top_k=top_k if top_k is not None else tla._default_top_k(),
    )
    return {
        "tool": "emit_kernel_candidates",
        "analysis_md": str(report_path),
        "kernel_candidates_path": str(out_dir / "kernel_candidates.json"),
        "report_source": built["report_source"],
        "idle_pct": built["idle_pct"],
        "compute_pct": built["compute_pct"],
        "idle_pct_threshold": built["idle_pct_threshold"],
        "compute_pct_threshold": built["compute_pct_threshold"],
        "hot_kernel_count": len(candidates),
        "trace_health_warnings": built["trace_health_warnings"],
        "artifact_paths": artifacts,
    }


def build_parser() -> argparse.ArgumentParser:
    """CLI for offline candidate emit from a TraceLens analysis_output."""
    parser = argparse.ArgumentParser(
        description=(
            "Emit kernel_candidates.json from an existing TraceLens analysis_output (no TraceLens orchestrator rerun)."
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
