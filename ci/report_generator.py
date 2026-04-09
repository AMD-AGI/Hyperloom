"""Generate CI summary reports from optimization results."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _parse_metrics_from_report(content: str) -> dict:
    """Fallback: extract baseline/optimized throughput from optimization_report.md.

    Parses the Executive Summary table:
        | Output Throughput (tok/s) | 2313.52 | ~2497 (avg) | **+7.9%** |
        | tok/s/GPU | 289.19 | ~312 (avg) | **+7.9%** |
    Prefers total throughput over per-GPU.
    """
    import re

    result: dict[str, Any] = {}

    gain_m = re.search(r'\*\*\+?([\d.]+)%\*\*', content)
    if gain_m:
        result["gain_pct"] = float(gain_m.group(1))

    tput_row = re.search(
        r'Output\s+Throughput\s*\(tok/s\)\s*\|\s*~?([\d.]+)\s*(?:\([^)]*\))?\s*\|\s*~?([\d.]+)',
        content)
    if tput_row:
        result["baseline_throughput"] = float(tput_row.group(1))
        result["optimized_throughput"] = float(tput_row.group(2))
    else:
        gpu_row = re.search(
            r'tok/s/GPU\s*\|\s*~?([\d.]+)\s*(?:\([^)]*\))?\s*\|\s*~?([\d.]+)',
            content)
        if gpu_row:
            result["baseline_throughput"] = float(gpu_row.group(1))
            result["optimized_throughput"] = float(gpu_row.group(2))

    return result


def extract_optimization_data(result_dir: str) -> dict:
    """Extract key metrics from a Hyperloom optimization result directory."""
    rd = Path(result_dir)
    data: dict[str, Any] = {
        "baseline_throughput": None,
        "optimized_throughput": None,
        "gain_pct": None,
        "actions": [],
        "report_exists": False,
    }

    report_path = rd / "optimization_report.md"
    if report_path.exists():
        data["report_exists"] = True
        data["report_content"] = report_path.read_text()

    # Priority 1: ci_metrics.json (structured, written by agent per prompt)
    ci_metrics_path = rd / "ci_metrics.json"
    if ci_metrics_path.exists():
        try:
            metrics = json.loads(ci_metrics_path.read_text())
            log.info("Loaded ci_metrics.json from %s", result_dir)
            data["baseline_throughput"] = metrics.get("baseline_throughput")
            data["optimized_throughput"] = metrics.get("optimized_throughput")
            data["gain_pct"] = metrics.get("gain_pct")
            data["actions"] = metrics.get("actions_taken", [])
            return data
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("Failed to parse ci_metrics.json in %s: %s", result_dir, e)

    # Priority 2: state.json (legacy)
    state_path = rd / "state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            data["baseline_throughput"] = state.get("baseline_throughput")
            data["optimized_throughput"] = state.get("optimized_throughput")
            if data["baseline_throughput"] and data["optimized_throughput"]:
                data["gain_pct"] = round(
                    (data["optimized_throughput"] - data["baseline_throughput"])
                    / data["baseline_throughput"] * 100, 1)
            data["actions"] = state.get("actions_taken", [])
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("Failed to parse state.json in %s: %s", result_dir, e)

    # Priority 3: parse from optimization_report.md
    if data["gain_pct"] is None and data.get("report_content"):
        parsed = _parse_metrics_from_report(data["report_content"])
        if parsed:
            log.info("Extracted metrics from report markdown (fallback): %s", parsed)
            for k, v in parsed.items():
                if data.get(k) is None:
                    data[k] = v

    return data


def build_model_result(
    model_name: str,
    ifx_key: str,
    image: str,
    precision: str,
    status: str,
    timestamp: str,
    result_dir: str,
    ifx_reference: dict | None = None,
) -> dict:
    """Build a complete result dict for one model."""
    opt_data = extract_optimization_data(result_dir)

    result = {
        "model": model_name,
        "inferenceX_key": ifx_key,
        "image": image,
        "precision": precision,
        "status": status,
        "timestamp": timestamp,
        "baseline_tok_per_gpu": opt_data.get("baseline_throughput"),
        "optimized_tok_per_gpu": opt_data.get("optimized_throughput"),
        "gain_pct": opt_data.get("gain_pct"),
        "actions": opt_data.get("actions", []),
        "report_exists": opt_data.get("report_exists", False),
        "report_content": opt_data.get("report_content"),
    }

    if ifx_reference and result["optimized_tok_per_gpu"]:
        metrics = ifx_reference.get("metrics", {})
        ifx_tput = metrics.get("tput_per_gpu") or metrics.get("output_tput_per_gpu")
        if ifx_tput:
            result["inferenceX_tok_per_gpu"] = ifx_tput
            result["vs_inferenceX_pct"] = round(
                (result["optimized_tok_per_gpu"] - ifx_tput) / ifx_tput * 100, 1)

    return result


def generate_markdown_report(
    results: list[dict],
    trigger: str,
    ifx_commit: str,
    ci_run_id: str,
) -> str:
    """Generate a markdown summary report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Inference Optimization CI Report",
        f"Date: {now}  |  Trigger: {trigger}  |  InferenceX commit: {ifx_commit[:7]}",
        f"CI Run ID: {ci_run_id}",
        "",
        "## Summary",
        "| Model | Precision | Depth | Baseline | Optimized | Gain | vs InferenceX | Status |",
        "|-------|-----------|-------|----------|-----------|------|---------------|--------|",
    ]

    for r in results:
        baseline = f"{r['baseline_tok_per_gpu']:.1f} tok/s/GPU" if r.get("baseline_tok_per_gpu") else "N/A"
        optimized = f"{r['optimized_tok_per_gpu']:.1f} tok/s/GPU" if r.get("optimized_tok_per_gpu") else "N/A"
        gain = f"+{r['gain_pct']:.1f}%" if r.get("gain_pct") is not None else "N/A"
        vs_ifx = f"{r['vs_inferenceX_pct']:+.1f}%" if r.get("vs_inferenceX_pct") is not None else "N/A"
        status_icon = {"completed": "✅", "failed": "❌", "timeout": "⏱️"}.get(r["status"], "❓")

        lines.append(
            f"| {r['model']} | {r['precision']} | full | {baseline} | {optimized} "
            f"| {gain} | {vs_ifx} | {status_icon} |"
        )

    lines.extend(["", "## Image Versions"])
    for r in results:
        lines.append(f"- {r['model']} {r['precision']}: `{r['image']}`")

    lines.extend(["", "## Per-Model Details"])
    for r in results:
        lines.append(f"- **{r['model']}**: status={r['status']}, report={'✅' if r.get('report_exists') else '❌'}")
        if r.get("actions"):
            lines.append(f"  - Actions: {', '.join(str(a) for a in r['actions'])}")

    return "\n".join(lines)


def generate_json_summary(
    results: list[dict],
    trigger: str,
    ifx_commit: str,
    ci_run_id: str,
) -> dict:
    """Generate a machine-readable JSON summary."""
    return {
        "ci_run_id": ci_run_id,
        "trigger": trigger,
        "inferenceX_commit": ifx_commit,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": results,
        "stats": {
            "total": len(results),
            "completed": sum(1 for r in results if r["status"] == "completed"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "timeout": sum(1 for r in results if r["status"] == "timeout"),
            "avg_gain_pct": _avg([r["gain_pct"] for r in results if r.get("gain_pct") is not None]),
        },
    }


def generate_github_summary(results: list[dict], trigger: str, ifx_commit: str) -> str:
    """Generate GitHub Actions Job Summary (written to $GITHUB_STEP_SUMMARY).

    Each model gets its own comparison table.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Hyperloom Inference Optimization Results",
        f"> {now} | Trigger: {trigger} | InferenceX: {ifx_commit[:7]}",
        "",
    ]

    for r in results:
        status_icon = {"completed": "✅", "failed": "❌", "timeout": "⏱️"}.get(r["status"], "❓")
        lines.append(f"## {status_icon} {r['model']} ({r['precision']})")
        lines.append("")
        lines.append(f"Image: `{r['image']}`")
        lines.append("")

        if r["status"] != "completed":
            lines.append(f"**Status: {r['status']}**")
            lines.append("")
            continue

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")

        if r.get("baseline_tok_per_gpu") is not None:
            lines.append(f"| Baseline (output tok/s/GPU) | {r['baseline_tok_per_gpu']:.2f} |")
        if r.get("optimized_tok_per_gpu") is not None:
            lines.append(f"| Optimized (output tok/s/GPU) | {r['optimized_tok_per_gpu']:.2f} |")
        if r.get("gain_pct") is not None:
            lines.append(f"| **Optimization Gain** | **+{r['gain_pct']:.1f}%** |")
        if r.get("inferenceX_tok_per_gpu") is not None:
            lines.append(f"| InferenceX MI355X (output tok/s/GPU) | {r['inferenceX_tok_per_gpu']:.2f} |")
        if r.get("vs_inferenceX_pct") is not None:
            sign = "+" if r["vs_inferenceX_pct"] >= 0 else ""
            lines.append(f"| **vs InferenceX MI355X** | **{sign}{r['vs_inferenceX_pct']:.1f}%** |")
        lines.append("")

        if r.get("actions"):
            lines.append("<details><summary>Optimization Actions</summary>")
            lines.append("")
            for a in r["actions"]:
                lines.append(f"- {a}")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        if r.get("report_exists"):
            lines.append(f"📄 Full report available in artifact: `{r['model']}/optimization_report.md`")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Overall summary table
    completed = [r for r in results if r["status"] == "completed"]
    if len(completed) > 1:
        lines.append("## Overall Summary")
        lines.append("")
        lines.append("| Model | Precision | Baseline | Optimized | Gain | vs InferenceX |")
        lines.append("|-------|-----------|----------|-----------|------|---------------|")
        for r in completed:
            bl = f"{r['baseline_tok_per_gpu']:.1f}" if r.get("baseline_tok_per_gpu") else "N/A"
            opt = f"{r['optimized_tok_per_gpu']:.1f}" if r.get("optimized_tok_per_gpu") else "N/A"
            gain = f"+{r['gain_pct']:.1f}%" if r.get("gain_pct") is not None else "N/A"
            vs = f"{r['vs_inferenceX_pct']:+.1f}%" if r.get("vs_inferenceX_pct") is not None else "N/A"
            lines.append(f"| {r['model']} | {r['precision']} | {bl} | {opt} | {gain} | {vs} |")
        lines.append("")

    return "\n".join(lines)


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 1) if values else None
