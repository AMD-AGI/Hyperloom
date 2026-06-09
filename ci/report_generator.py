# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Generate CI summary reports from optimization results."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


LLM_ENDPOINT = os.environ.get("SAFE_BASE_URL", "") + "/api/v1/llm-proxy/v1/chat/completions"


def _extract_metrics_via_llm(report_content: str) -> dict:
    """Use LLM to extract baseline/optimized throughput from optimization report."""
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        return {}

    try:
        import requests
        prompt = (
            "Extract performance metrics from this optimization report. "
            "Return ONLY a JSON object with these exact fields:\n"
            '{"baseline_throughput": <number>, "optimized_throughput": <number>, '
            '"tok_per_gpu_baseline": <number>, "tok_per_gpu_optimized": <number>, '
            '"gain_pct": <number>}\n'
            "- baseline_throughput and optimized_throughput are total output tok/s (all GPUs)\n"
            "- tok_per_gpu values are per-GPU (divided by TP)\n"
            "- gain_pct is the percentage improvement\n"
            "- If baseline equals optimized, gain_pct should be 0.0\n"
            "Return ONLY the JSON, no markdown fences, no explanation.\n\n"
            f"Report:\n{report_content[:4000]}"
        )

        resp = requests.post(
            LLM_ENDPOINT,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            json={"model": "openai/gpt-4.1-mini", "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0, "max_tokens": 200},
            timeout=30,
            verify=os.environ.get(
                "SSL_CERT_FILE", os.environ.get("REQUESTS_CA_BUNDLE", True)
            ),
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        result = json.loads(content)
        out = {}
        for k in ("baseline_throughput", "optimized_throughput", "tok_per_gpu_baseline",
                   "tok_per_gpu_optimized", "gain_pct"):
            v = result.get(k)
            if isinstance(v, (int, float)):
                out[k] = v
        if "tok_per_gpu_baseline" in out:
            out["baseline_throughput"] = out.get("baseline_throughput") or out["tok_per_gpu_baseline"]
        if "tok_per_gpu_optimized" in out:
            out["optimized_throughput"] = out.get("optimized_throughput") or out["tok_per_gpu_optimized"]
        return out
    except Exception as e:
        log.warning("LLM metrics extraction failed: %s", e)
        return {}


def _first_of(d: dict, *keys: str) -> Any | None:
    """Return the first non-None value found for the given keys."""
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def _parse_metrics_from_report(content: str) -> dict:
    """Fallback: extract baseline/optimized throughput from the optimization_report.md Executive Summary table (prefers total over per-GPU)."""
    import re

    result: dict[str, Any] = {}

    gain_m = re.search(r'\*\*([+-]?[\d.]+)%\*\*', content)
    if gain_m:
        result["gain_pct"] = float(gain_m.group(1))

    gpu_row = re.search(
        r'tok/s/GPU\s*\|\s*~?([\d.]+)\s*(?:\([^)]*\))?\s*\|\s*~?([\d.]+)',
        content)
    if gpu_row:
        result["baseline_throughput"] = float(gpu_row.group(1))
        result["optimized_throughput"] = float(gpu_row.group(2))
    else:
        tput_row = re.search(
            r'Output\s+Throughput\s*\(tok/s\)\s*\|\s*~?([\d.]+)\s*(?:\([^)]*\))?\s*\|\s*~?([\d.]+)',
            content)
        if tput_row:
            result["baseline_throughput"] = float(tput_row.group(1))
            result["optimized_throughput"] = float(tput_row.group(2))

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

            bl = _first_of(metrics,
                           "tok_per_gpu_baseline", "baseline_throughput",
                           "baseline_output_tput_per_gpu", "baseline_output_tput_tok_s",
                           "baseline_tok_per_gpu")
            opt = _first_of(metrics,
                            "tok_per_gpu_optimized", "optimized_throughput",
                            "optimized_output_tput_per_gpu", "optimized_output_tput_tok_s",
                            "optimized_tok_per_gpu")
            gain = _first_of(metrics,
                             "gain_pct", "improvement_pct", "total_improvement_pct")

            # Nested schema fallback
            if bl is None and isinstance(metrics.get("baseline"), dict):
                b = metrics["baseline"]
                bl = _first_of(b, "tok_s_per_gpu", "output_throughput_tok_s",
                               "output_tput_per_gpu", "tput_per_gpu")
            if opt is None and isinstance(metrics.get("optimized"), dict):
                o = metrics["optimized"]
                opt = _first_of(o, "tok_s_per_gpu", "output_throughput_tok_s",
                                "output_tput_per_gpu", "tput_per_gpu")
            if gain is None and isinstance(metrics.get("improvement"), dict):
                imp = metrics["improvement"]
                gain = _first_of(imp, "output_throughput_pct", "tok_s_per_gpu_pct",
                                 "gain_pct", "pct")
            if bl and opt and bl > 0:
                computed_gain = round((opt - bl) / bl * 100, 2)
                if gain is not None and abs(gain - computed_gain) > 1.0:
                    log.warning("gain_pct from agent (%.2f%%) disagrees with computed (%.2f%%), using computed",
                                gain, computed_gain)
                gain = computed_gain

            data["baseline_throughput"] = bl
            data["optimized_throughput"] = opt
            data["gain_pct"] = gain
            raw_actions = metrics.get("actions_taken") or metrics.get("actions", [])
            data["actions"] = list(raw_actions) if isinstance(raw_actions, (list, tuple)) else [str(raw_actions)] if raw_actions else []
            if bl is not None and opt is not None:
                return data
            log.warning("ci_metrics.json loaded but missing key metrics (bl=%s, opt=%s), "
                        "falling back to report markdown", bl, opt)
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("Failed to parse ci_metrics.json in %s: %s", result_dir, e)

    # Priority 2: parse from optimization_report.md (regex)
    if data.get("report_content"):
        parsed = _parse_metrics_from_report(data["report_content"])
        if parsed:
            log.info("Extracted metrics from report markdown (regex fallback): %s", parsed)
            for k, v in parsed.items():
                if data.get(k) is None:
                    data[k] = v

    # Priority 3: LLM extraction from report (if regex missed key fields)
    if (data.get("baseline_throughput") is None or data.get("optimized_throughput") is None) \
            and data.get("report_content"):
        llm_parsed = _extract_metrics_via_llm(data["report_content"])
        if llm_parsed:
            log.info("Extracted metrics via LLM fallback: %s", llm_parsed)
            for k, v in llm_parsed.items():
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
        ifx_tput = metrics.get("output_tput_per_gpu") or metrics.get("tput_per_gpu")
        if ifx_tput and ifx_tput > 0:
            ratio = result["optimized_tok_per_gpu"] / ifx_tput
            if ratio > 3.0:
                tp = ifx_reference.get("decode_tp") or 1
                if tp > 1:
                    log.warning(
                        "optimized (%.1f) is %.0fx InferenceX (%.1f) — likely total "
                        "throughput instead of per-GPU. Dividing by TP=%d.",
                        result["optimized_tok_per_gpu"], ratio, ifx_tput, tp)
                    result["baseline_tok_per_gpu"] = round(result["baseline_tok_per_gpu"] / tp, 2) if result["baseline_tok_per_gpu"] else None
                    result["optimized_tok_per_gpu"] = round(result["optimized_tok_per_gpu"] / tp, 2)
                    if result.get("gain_pct") is not None and result["baseline_tok_per_gpu"]:
                        result["gain_pct"] = round(
                            (result["optimized_tok_per_gpu"] - result["baseline_tok_per_gpu"])
                            / result["baseline_tok_per_gpu"] * 100, 2)
            result["inferenceX_tok_per_gpu"] = ifx_tput
            vs = round((result["optimized_tok_per_gpu"] - ifx_tput) / ifx_tput * 100, 1)
            result["vs_inferenceX_pct"] = 0.0 if vs == -0.0 else vs

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
        gain = _fmt_pct(r.get("gain_pct"))
        vs_ifx = _fmt_pct(r.get("vs_inferenceX_pct"))
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
        actions = r.get("actions")
        if actions:
            if isinstance(actions, (list, tuple)):
                lines.append(f"  - Actions: {', '.join(str(a) for a in actions)}")
            else:
                lines.append(f"  - Actions: {actions}")

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

        if r["status"] not in ("completed", "timeout"):
            lines.append(f"**Status: {r['status']}**")
            lines.append("")
            continue

        if r["status"] == "timeout":
            lines.append(f"**Status: timeout** (sandbox_timeout reached)")
            lines.append("")

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")

        if r.get("baseline_tok_per_gpu") is not None:
            lines.append(f"| Baseline (output tok/s/GPU) | {r['baseline_tok_per_gpu']:.2f} |")
        if r.get("optimized_tok_per_gpu") is not None:
            lines.append(f"| Optimized (output tok/s/GPU) | {r['optimized_tok_per_gpu']:.2f} |")
        if r.get("gain_pct") is not None:
            lines.append(f"| **Optimization Gain** | **{_fmt_pct(r['gain_pct'])}** |")
        if r.get("inferenceX_tok_per_gpu") is not None:
            gpu_label = (r.get("target_gpu") or "MI355X").upper()
            lines.append(f"| InferenceX {gpu_label} (output tok/s/GPU) | {r['inferenceX_tok_per_gpu']:.2f} |")
        if r.get("vs_inferenceX_pct") is not None:
            gpu_label = (r.get("target_gpu") or "MI355X").upper()
            lines.append(f"| **vs InferenceX {gpu_label}** | **{_fmt_pct(r['vs_inferenceX_pct'])}** |")
        lines.append("")

        actions = r.get("actions")
        if actions:
            action_list = list(actions) if isinstance(actions, (list, tuple)) else [str(actions)]
            lines.append("<details><summary>Optimization Actions</summary>")
            lines.append("")
            for a in action_list:
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
            gain = _fmt_pct(r.get("gain_pct"))
            vs = _fmt_pct(r.get("vs_inferenceX_pct"))
            lines.append(f"| {r['model']} | {r['precision']} | {bl} | {opt} | {gain} | {vs} |")
        lines.append("")

    return "\n".join(lines)


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "N/A"
    if abs(v) < 0.05:
        return "--"
    return f"{v:+.1f}%"


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 1) if values else None
