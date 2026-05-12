#!/usr/bin/env python3
"""Aggregate per-task ci_metrics.json into the final summary table.

Inputs:
  --artifacts-dir DIR   Directory containing per-task artifacts (task-artifacts/<task_id>/)
  --manifests-dir DIR   Directory containing submission_manifest.json files (one per matrix job)
  --target-gpu STR      Reference GPU for InferenceX comparison (e.g. b200, h100, mi300x)
  --isl / --osl         Sequence lengths to look up in InferenceX
  --out-dir DIR         Where to write ci_summary.{json,md}

Outputs:
  ci_summary.md          — markdown table for $GITHUB_STEP_SUMMARY
  ci_summary.json        — compact structured rows for downstream tooling
  normalized_results.json / .ndjson
                         — one normalized record per Hyperloom task

Main inputs parsed per task:
  ci_metrics.json, baseline_summary.json, sweep_results.csv,
  kernel_candidates.json, kernel_results.json, run_context.env
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifact_normalizer import collect_normalized_results       # noqa: E402

log = logging.getLogger("build-summary")


# ── InferenceX reference lookup ─────────────────────────────────────────────────

def load_hf_to_ifx_map(yaml_path: Path) -> dict[str, str]:
    """Read inferenceX_models.yaml → {hf_model: api_name}."""
    if not yaml_path.exists():
        return {}
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return {
        m["hf_model"]: m["api_name"]
        for m in cfg.get("models", [])
        if m.get("hf_model") and m.get("api_name")
    }


def fetch_inferenceX_ref(
    repo_id: str,
    hf_to_ifx: dict[str, str],
    target_gpu: str,
    isl: int,
    osl: int,
) -> float | None:
    """Look up InferenceX output_tput_per_gpu for repo_id on target_gpu.

    Returns None if the repo isn't in our HF→InferenceX map, the API has no
    benchmarks, or none match the (hardware, ISL, OSL) we want. None means
    'no comparison available for this row' — the column stays blank.
    """
    api_name = hf_to_ifx.get(repo_id)
    if not api_name:
        log.info("[%s] no InferenceX api_name mapping; skipping ref lookup", repo_id)
        return None
    try:
        from inferenceX_parser import fetch_benchmarks, find_benchmark
        benchmarks = fetch_benchmarks(api_name)
    except Exception as e:
        log.warning("[%s] InferenceX API failed: %s", api_name, e)
        return None
    bench = find_benchmark(benchmarks, target_gpu, isl, osl, precision=None)
    if not bench:
        log.info("[%s] no InferenceX entry for hw=%s isl=%d osl=%d",
                 api_name, target_gpu, isl, osl)
        return None
    m = bench.get("metrics") or {}
    return m.get("output_tput_per_gpu") or m.get("tput_per_gpu")


# ── Row assembly ────────────────────────────────────────────────────────────────

def collect_rows(
    artifacts_dir: Path,
    manifests_dir: Path,
    hf_to_ifx: dict[str, str],
    target_gpu: str,
    isl: int,
    osl: int,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    normalized_results = collect_normalized_results(
        artifacts_dir, manifests_dir, build_run_metadata())

    for item in normalized_results:
        task = item.get("task") or {}
        metrics = item.get("metrics") or {}
        detected = task.get("detected") or {}
        model = task.get("model")
        row = {
            "model": model,
            "task_id": task.get("task_id"),
            "display_name": task.get("display_name"),
            "submit_status": task.get("submit_status"),
            "final_status": task.get("final_status"),
            "framework": metrics.get("framework") or detected.get("framework"),
            "precision": detected.get("precision"),
            "gpu_type": metrics.get("gpu_type"),
            "baseline_tok_per_gpu": (
                metrics.get("tok_per_gpu_baseline")
                or metrics.get("baseline_throughput")
            ),
            "optimized_tok_per_gpu": (
                metrics.get("tok_per_gpu_optimized")
                or metrics.get("optimized_throughput")
            ),
            "gain_pct": metrics.get("gain_pct"),
            "peak_throughput": metrics.get("peak_throughput"),
            "peak_throughput_conc": metrics.get("peak_throughput_conc"),
            "inferenceX_tok_per_gpu": None,
            "vs_inferenceX_pct": None,
            "actions": metrics.get("actions") or [],
            "sweep_points": len(item.get("sweep_points") or []),
            "kernel_candidates": len(item.get("kernel_candidates") or []),
            "kernel_optimizations": len(item.get("kernel_optimizations") or []),
            "artifacts": len(item.get("artifacts") or []),
            "warnings": item.get("warnings") or [],
        }

        ref = fetch_inferenceX_ref(model, hf_to_ifx, target_gpu, isl, osl) if model else None
        row["inferenceX_tok_per_gpu"] = ref
        opt = row["optimized_tok_per_gpu"]
        if ref and opt:
            row["vs_inferenceX_pct"] = (opt - ref) / ref * 100.0

        rows.append(row)

    return rows, normalized_results


def build_run_metadata() -> dict:
    """Capture GitHub Actions context when present; harmless for local runs."""
    return {
        "source": "hyperloom-ci",
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "github_workflow": os.environ.get("GITHUB_WORKFLOW"),
        "github_repository": os.environ.get("GITHUB_REPOSITORY"),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_ref": os.environ.get("GITHUB_REF"),
    }


# ── Markdown rendering ──────────────────────────────────────────────────────────

def fmt_num(v, fmt: str = ".0f") -> str:
    if v is None:
        return "—"
    try:
        return format(v, fmt)
    except Exception:
        return str(v)


def fmt_pct(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{v:+.2f}%"
    except Exception:
        return str(v)


def render_markdown(rows: list[dict], target_gpu: str, isl: int, osl: int) -> str:
    succeeded = sum(1 for r in rows if r.get("final_status") == "Succeeded")
    n = len(rows)
    lines = [
        "# Hyperloom CI Summary",
        f"- Models: {n} (final_status=Succeeded: {succeeded})",
        f"- ISL/OSL: {isl} / {osl}",
        f"- InferenceX reference GPU: `{target_gpu}`",
        "",
        "| Model | Baseline tok/s/GPU | Optimized tok/s/GPU | Gain % | Peak | "
        f"InferenceX ({target_gpu}) tok/s/GPU | vs InferenceX % | Data | Actions |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        actions = ", ".join(r.get("actions") or []) or "—"
        # Markdown-escape pipe chars in actions list
        actions = actions.replace("|", "\\|")[:200]
        peak = fmt_num(r.get("peak_throughput"))
        if r.get("peak_throughput_conc"):
            peak = f"{peak} @ c{r['peak_throughput_conc']}"
        data = (
            f"sweep={r.get('sweep_points', 0)}, "
            f"kernels={r.get('kernel_optimizations', 0)}, "
            f"files={r.get('artifacts', 0)}"
        )
        lines.append(
            f"| `{r['model']}` "
            f"| {fmt_num(r['baseline_tok_per_gpu'])} "
            f"| {fmt_num(r['optimized_tok_per_gpu'])} "
            f"| {fmt_pct(r['gain_pct'])} "
            f"| {peak} "
            f"| {fmt_num(r['inferenceX_tok_per_gpu'])} "
            f"| {fmt_pct(r['vs_inferenceX_pct'])} "
            f"| {data} "
            f"| {actions} |"
        )
    return "\n".join(lines) + "\n"


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts-dir", required=True,
                        help="Root dir of per-task artifacts (task-artifacts/<task_id>/...)")
    parser.add_argument("--manifests-dir", required=True,
                        help="Root dir containing submission_manifest.json file(s)")
    parser.add_argument("--target-gpu", default="b200",
                        help="Reference GPU for InferenceX comparison (default: b200)")
    parser.add_argument("--isl", type=int, default=1024)
    parser.add_argument("--osl", type=int, default=1024)
    parser.add_argument("--ifx-models-yaml", default="",
                        help="Path to inferenceX_models.yaml (default: ./inferenceX_models.yaml)")
    parser.add_argument("--out-dir", default="summary-out")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    yaml_path = Path(args.ifx_models_yaml or
                     Path(__file__).resolve().parent / "inferenceX_models.yaml")
    hf_to_ifx = load_hf_to_ifx_map(yaml_path)
    log.info("HF→InferenceX mappings loaded: %d", len(hf_to_ifx))

    rows, normalized_results = collect_rows(
        Path(args.artifacts_dir),
        Path(args.manifests_dir),
        hf_to_ifx,
        args.target_gpu,
        args.isl,
        args.osl,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = render_markdown(rows, args.target_gpu, args.isl, args.osl)
    (out / "ci_summary.md").write_text(md, encoding="utf-8")
    (out / "ci_summary.json").write_text(
        json.dumps({
            "target_gpu": args.target_gpu,
            "isl": args.isl,
            "osl": args.osl,
            "rows": rows,
        }, indent=2),
        encoding="utf-8",
    )
    (out / "normalized_results.json").write_text(
        json.dumps({
            "target_gpu": args.target_gpu,
            "isl": args.isl,
            "osl": args.osl,
            "results": normalized_results,
        }, indent=2),
        encoding="utf-8",
    )
    (out / "normalized_results.ndjson").write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in normalized_results),
        encoding="utf-8",
    )
    log.info("wrote summary and normalized results under %s", out)

    # Print the table to stdout too, so the GitHub Actions log shows it
    # without having to download the artifact.
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
