#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Normalize Hyperloom optimization artifacts into a stable data shape.

This module is intentionally entrypoint-agnostic: GitHub CI, the Hyperloom web
UI, or a manual local import can all hand it a downloaded artifact directory.
It turns loose result files into one JSON object per optimization task so later
steps can summarize, upload, or publish without parsing markdown.
"""

from __future__ import annotations

import csv
import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("artifact-normalizer")

SCHEMA_VERSION = "hyperloom.ci.normalized.v1"


def _read_json(path: Path | None, warnings: list[str]) -> Any | None:
    """Read and parse a JSON file, recording failures as warnings.

    Args:
        path (Path | None): File to read; ``None`` yields ``None``.
        warnings (list[str]): List appended to with a message on parse failure.

    Returns:
        Any | None: The parsed JSON value, or ``None`` if missing/unparsable.
    """
    if not path:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        warnings.append(f"failed to parse {path.name}: {e}")
        return None


def _to_float(value: Any) -> float | None:
    """Coerce a value to ``float``, tolerating commas and ``SKIPPED``.

    Args:
        value (Any): Number or string to convert.

    Returns:
        float | None: The float value, or ``None`` if it cannot be parsed.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text.upper() == "SKIPPED":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    """Coerce a value to ``int`` via :func:`_to_float`.

    Args:
        value (Any): Number or string to convert.

    Returns:
        int | None: The integer value, or ``None`` if it cannot be parsed.
    """
    number = _to_float(value)
    return int(number) if number is not None else None


def _first_of(data: dict[str, Any], *keys: str) -> Any | None:
    """Return the first non-None value among the given top-level keys.

    Args:
        data (dict[str, Any]): Source mapping.
        *keys (str): Keys to try in order.

    Returns:
        Any | None: The first present non-None value, or ``None``.
    """
    for key in keys:
        if data.get(key) is not None:
            return data[key]
    return None


def _first_nested(data: dict[str, Any], *paths: str) -> Any | None:
    """Return the first non-None value from dotted paths.

    Agents have produced several ci_metrics schemas over time. The original
    schema was flat (baseline_throughput / optimized_throughput / gain_pct);
    newer reports often use nested baseline/best dictionaries with varied key
    names. Dotted lookup keeps the parser compact and backwards compatible.

    Args:
        data (dict[str, Any]): Source mapping to walk.
        *paths (str): Dotted key paths (e.g. ``"baseline.tpot_ms"``) tried in
            order.

    Returns:
        Any | None: The first fully-resolved non-None value, or ``None``.
    """
    for path in paths:
        cur: Any = data
        ok = True
        for part in path.split("."):
            if not isinstance(cur, dict) or cur.get(part) is None:
                ok = False
                break
            cur = cur.get(part)
        if ok:
            return cur
    return None


def _relative(path: Path | None, root: Path) -> str | None:
    """Express ``path`` relative to ``root`` as a POSIX string.

    Args:
        path (Path | None): Path to relativize; ``None`` yields ``None``.
        root (Path): Base directory.

    Returns:
        str | None: The relative POSIX path, or the absolute POSIX path if it
            is not under ``root``; ``None`` when ``path`` is ``None``.
    """
    if path is None:
        return None
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def find_artifact(root: Path, *tails: str) -> Path | None:
    """Find the first file whose relative path ends with one of ``tails``.

    Args:
        root (Path): Directory to search recursively.
        *tails (str): Path suffixes or bare file names to match (case- and
            separator-insensitive).

    Returns:
        Path | None: The first matching file, or ``None`` if none found.
    """
    if not root.exists():
        return None
    normalized = tuple(t.replace("\\", "/").lower().lstrip("/") for t in tails)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix().lower()
        name = path.name.lower()
        if any(rel.endswith(tail) or name == tail for tail in normalized):
            return path
    return None


def parse_env_file(path: Path | None) -> dict[str, str]:
    """Parse a ``KEY=value`` env file, ignoring blanks and comments.

    Args:
        path (Path | None): Env file to read; missing/None yields ``{}``.

    Returns:
        dict[str, str]: Parsed key/value pairs with surrounding quotes
            stripped from values.
    """
    if not path or not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def parse_ci_metrics(data: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a heterogeneous ci_metrics payload into a stable shape.

    Resolves baseline/optimized throughput, gain percentage (including
    multiplier-style ratios), TPOT/TTFT latencies, and run metadata across the
    several historical schemas via :func:`_first_nested`/:func:`_first_of`.
    Computes ``gain_pct`` from throughput when it is not reported directly.

    Args:
        data (dict[str, Any] | None): Raw ci_metrics or session-breakdown
            dict; ``None`` is treated as empty.

    Returns:
        dict[str, Any]: Canonical metrics dict with float/int-coerced fields.
    """
    data = data or {}
    baseline = _first_nested(
        data,
        # Flat canonical / prior CI schema.
        "baseline_throughput",
        "tok_per_gpu_baseline",
        "baseline_output_tput_per_gpu",
        "baseline_output_tput_tok_s",
        "baseline_tok_per_gpu",
        "baseline_throughput_tok_per_s_per_gpu",
        "baseline_output_tput_tok_s_per_gpu",
        "baseline_output_throughput_tok_s_per_gpu",
        # Nested baseline/best schemas emitted by newer agents.
        "baseline.output_throughput_per_gpu",
        "baseline.output_tok_per_s_per_gpu",
        "baseline.output_tok_s_per_gpu",
        "baseline.output_throughput_tok_s_per_gpu",
        "baseline.baseline_throughput_tok_per_s_per_gpu",
        "baseline.throughput_per_gpu",
        "baseline.per_gpu_tok_per_s",
        "baseline.output_tput_tok_s_per_gpu",
        "best.matched_n_baseline_tok_s_per_gpu",
        "best.matched_n_baseline_tok_s_at_192p",
        "best.matched_n_baseline_output_tok_s",
        # Total throughput fallbacks, used only when no per-GPU field exists.
        "baseline.output_throughput_tok_s",
        "baseline.output_tok_per_s",
        "baseline.output_tok_s",
        "baseline.output_tput_tok_s",
        "baseline.output_tokps",
        "baseline.output_throughput_tokps",
    )
    optimized = _first_nested(
        data,
        # Flat canonical / prior CI schema.
        "optimized_throughput",
        "tok_per_gpu_optimized",
        "optimized_output_tput_per_gpu",
        "optimized_output_tput_tok_s",
        "optimized_tok_per_gpu",
        "best_throughput_tok_per_s_per_gpu",
        "best_output_tput_tok_s_per_gpu",
        "best_output_throughput_tok_s_per_gpu",
        # Nested best schemas.
        "best.output_throughput_per_gpu",
        "best.output_tok_per_s_per_gpu",
        "best.output_tok_s_per_gpu",
        "best.output_throughput_tok_s_per_gpu",
        "best.throughput_per_gpu",
        "best.per_gpu_tok_per_s",
        "best.output_tput_tok_s_per_gpu",
        "best.output_throughput_tok_s_at_192p",
        "best.output_tok_s_at_192p",
        "best.output_tput_tok_s_at_192p",
        # Total throughput fallbacks.
        "best.output_throughput_tok_s",
        "best.output_tok_per_s",
        "best.output_tok_s",
        "best.output_tput_tok_s",
        "best.output_tokps",
        "best.output_tokps_mean",
        "best.output_throughput",
        "best.output_throughput_tokps",
    )
    gain = _first_nested(
        data,
        "gain_pct",
        "improvement_pct",
        "total_improvement_pct",
        "speedup_pct",
        "speedup_pct_vs_baseline",
        "best.speedup_pct",
        "best.delta_pct_vs_baseline",
        "best.delta_throughput_pct",
        "best.improvement_pct",
        "best.improvement_pct_vs_matched_n_baseline",
        "improvement.throughput_delta_pct",
        "improvement.output_tok_s_pct",
        "improvement.output_tps_pct",
    )
    # Some agents report speedup as a multiplier (1.10x). Convert to percent
    # only for clear ratio fields.
    ratio_gain = _first_nested(
        data,
        "best.speedup_vs_baseline",
        "best_speedup",
        "speedup_x",
        "improvement.speedup_x",
    )
    if gain is None and ratio_gain is not None:
        rg = _to_float(ratio_gain)
        if rg is not None:
            gain = (rg - 1.0) * 100.0

    baseline_tpot = _first_nested(
        data,
        "baseline_tpot_ms",
        "baseline.tpot_ms",
        "baseline.mean_tpot_ms",
        "baseline.tpot_mean_ms",
        "baseline.tpot_ms_mean",
    )
    optimized_tpot = _first_nested(
        data,
        "best_tpot_ms",
        "optimized_tpot_ms",
        "best.tpot_ms",
        "best.mean_tpot_ms",
        "best.tpot_mean_ms",
        "best.tpot_ms_mean",
    )
    baseline_ttft = _first_nested(
        data,
        "baseline_ttft_ms",
        "baseline.ttft_ms",
        "baseline.mean_ttft_ms",
        "baseline.ttft_mean_ms",
        "baseline.ttft_ms_mean",
    )
    optimized_ttft = _first_nested(
        data,
        "best_ttft_ms",
        "optimized_ttft_ms",
        "best.ttft_ms",
        "best.mean_ttft_ms",
        "best.ttft_mean_ms",
        "best.ttft_ms_mean",
    )

    metrics = {
        "baseline_throughput": _to_float(baseline),
        "optimized_throughput": _to_float(optimized),
        "gain_pct": _to_float(gain),
        "tok_per_gpu_baseline": _to_float(_first_of(data, "tok_per_gpu_baseline", "baseline_tok_per_gpu")),
        "tok_per_gpu_optimized": _to_float(_first_of(data, "tok_per_gpu_optimized", "optimized_tok_per_gpu")),
        "peak_throughput": _to_float(data.get("peak_throughput")),
        "peak_throughput_conc": _to_int(data.get("peak_throughput_conc")),
        "model": data.get("model"),
        "framework": data.get("framework"),
        "framework_version": data.get("framework_version"),
        "gpu_type": data.get("gpu_type"),
        "tp": _to_int(data.get("tp")),
        "conc": _to_int(data.get("conc")),
        "isl": _to_int(data.get("isl")),
        "osl": _to_int(data.get("osl")),
        "baseline_tpot_ms": _to_float(baseline_tpot),
        "optimized_tpot_ms": _to_float(optimized_tpot),
        "baseline_ttft_ms": _to_float(baseline_ttft),
        "optimized_ttft_ms": _to_float(optimized_ttft),
        "actions": data.get("actions_taken") or data.get("actions") or [],
    }
    if metrics["gain_pct"] is None and metrics["baseline_throughput"] and metrics["optimized_throughput"]:
        metrics["gain_pct"] = round(
            (metrics["optimized_throughput"] - metrics["baseline_throughput"])
            / metrics["baseline_throughput"] * 100,
            2,
        )
    return metrics


def parse_baseline_summary(data: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a ``baseline_summary.json`` payload into a stable shape.

    Args:
        data (dict[str, Any] | None): Raw baseline summary; ``None`` is
            treated as empty.

    Returns:
        dict[str, Any]: Baseline throughput, config, torch-compile/aiter
            variants, and run metadata with numeric fields coerced.
    """
    data = data or {}
    return {
        "baseline_tput_per_gpu": _to_float(data.get("baseline_tput_per_gpu")),
        "baseline_config": data.get("baseline_config"),
        "torch_compile_status": data.get("torch_compile_status"),
        "torch_compile_tput": _to_float(data.get("torch_compile_tput")),
        "no_aiter_tput": _to_float(data.get("no_aiter_tput")),
        "model": data.get("model"),
        "gpu_type": data.get("gpu_type"),
        "tp": _to_int(data.get("tp")),
        "conc": _to_int(data.get("conc")),
        "isl": _to_int(data.get("isl")),
        "osl": _to_int(data.get("osl")),
    }


def parse_sweep_results(path: Path | None, warnings: list[str]) -> list[dict[str, Any]]:
    """Parse a ``sweep_results.csv`` into a list of per-point dicts.

    Args:
        path (Path | None): CSV file to read; missing/None yields ``[]``.
        warnings (list[str]): List appended to with a message on failure.

    Returns:
        list[dict[str, Any]]: One dict per sweep point with coerced numeric
            fields and an ``"ok"``/``"skipped"`` status.
    """
    if not path or not path.exists():
        return []
    points: list[dict[str, Any]] = []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                skipped = any(str(v).strip().upper() == "SKIPPED" for v in row.values())
                points.append({
                    "conc": _to_int(row.get("CONC")),
                    "isl": _to_int(row.get("ISL")),
                    "osl": _to_int(row.get("OSL")),
                    "num_prompts": _to_int(row.get("NUM_PROMPTS")),
                    "output_throughput_tok_s": _to_float(row.get("output_throughput_tok_s")),
                    "mean_tpot_ms": _to_float(row.get("mean_tpot_ms")),
                    "mean_ttft_ms": _to_float(row.get("mean_ttft_ms")),
                    "status": "skipped" if skipped else "ok",
                })
    except Exception as e:
        warnings.append(f"failed to parse sweep results: {e}")
    return points


def parse_kernel_candidates(data: Any) -> list[dict[str, Any]]:
    """Normalize a ``kernel_candidates.json`` list into stable dicts.

    Args:
        data (Any): Expected to be a list of candidate dicts; other types
            yield ``[]``.

    Returns:
        list[dict[str, Any]]: One dict per candidate with coerced numeric
            fields.
    """
    if not isinstance(data, list):
        return []
    candidates: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        candidates.append({
            "rank": _to_int(item.get("rank")),
            "name": item.get("name"),
            "tier": item.get("tier"),
            "gpu_pct": _to_float(item.get("gpu_pct")),
            "count": _to_int(item.get("count")),
            "time_ms": _to_float(item.get("time_ms")),
        })
    return candidates


def parse_kernel_results(data: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize a ``kernel_results.json`` payload.

    Args:
        data (dict[str, Any] | None): Raw kernel results; ``None`` is treated
            as empty.

    Returns:
        tuple[list[dict[str, Any]], dict[str, Any]]: ``(kernels, summary)``
            where ``kernels`` is the per-kernel list and ``summary`` is the
            summary dict (empty when absent).
    """
    data = data or {}
    kernels: list[dict[str, Any]] = []
    for item in data.get("kernels") or []:
        if not isinstance(item, dict):
            continue
        kernels.append({
            "name": item.get("name"),
            "micro_speedup": _to_float(item.get("micro_speedup")),
            "correctness": item.get("correctness"),
            "gpu_pct": _to_float(item.get("gpu_pct")),
            "status": item.get("status"),
            "note": item.get("note"),
        })
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    return kernels, summary


def build_artifact_index(task_dir: Path) -> list[dict[str, Any]]:
    """Index every file under a task directory with size and kind.

    Args:
        task_dir (Path): Task artifact directory to scan recursively.

    Returns:
        list[dict[str, Any]]: One entry per file with ``path``,
            ``size_bytes``, and classified ``kind``.
    """
    artifacts: list[dict[str, Any]] = []
    if not task_dir.exists():
        return artifacts
    for path in sorted(task_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(task_dir).as_posix()
        artifacts.append({
            "path": rel,
            "size_bytes": path.stat().st_size,
            "kind": classify_artifact(rel),
        })
    return artifacts


def classify_artifact(path: str) -> str:
    """Classify an artifact file by its name/suffix.

    Args:
        path (str): Relative artifact path.

    Returns:
        str: A kind label such as ``"ci_metrics"``, ``"sweep_results"``,
            ``"log"``, or ``"artifact"`` for anything unrecognized.
    """
    lower = path.lower()
    if lower.endswith("ci_metrics.json"):
        return "ci_metrics"
    if Path(lower).name.startswith("session_breakdown") and lower.endswith(".json"):
        return "session_breakdown"
    if lower.endswith("baseline_summary.json"):
        return "baseline_summary"
    if lower.endswith("sweep_results.csv"):
        return "sweep_results"
    if lower.endswith("kernel_candidates.json"):
        return "kernel_candidates"
    if lower.endswith("kernel_results.json"):
        return "kernel_results"
    if lower.endswith("optimization_report.md"):
        return "optimization_report"
    if lower.endswith("run_context.env"):
        return "run_context"
    if lower.endswith(".log"):
        return "log"
    return "artifact"


def normalize_task_result(
    task_dir: Path,
    manifest_record: dict[str, Any],
    run: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize one task artifact directory.

    ``manifest_record`` is just optional task metadata supplied by the caller.
    The metrics themselves are read from files under ``task_dir``.

    Args:
        task_dir (Path): Directory holding one task's downloaded artifacts.
        manifest_record (dict[str, Any]): Task metadata (ids, status, model,
            overrides) merged into the ``task`` block.
        run (dict[str, Any] | None): Optional run-level metadata.

    Returns:
        dict[str, Any]: A single normalized result object (schema
            ``hyperloom.ci.normalized.v1``) with metrics, baseline, sweep
            points, kernels, artifact index, source-file map, and warnings.
    """
    warnings: list[str] = []

    ci_metrics_path = find_artifact(task_dir, "ci_metrics.json", "hyperloom/ci_metrics.json")
    session_breakdown_path = find_artifact(
        task_dir,
        "session_breakdown.json",
        "session_breakdown_v2.json",
    )
    if session_breakdown_path is None and task_dir.exists():
        session_breakdown_path = next(
            (p for p in sorted(task_dir.rglob("session_breakdown*.json")) if p.is_file()),
            None,
        )
    baseline_summary_path = find_artifact(task_dir, "results/baseline_summary.json", "baseline_summary.json")
    sweep_path = find_artifact(task_dir, "results/sweep_results.csv", "sweep_results.csv")
    kernel_candidates_path = find_artifact(task_dir, "results/kernel_candidates.json", "kernel_candidates.json")
    kernel_results_path = find_artifact(task_dir, "kernel_opt/kernel_results.json", "kernel_results.json")
    run_context_path = find_artifact(task_dir, "results/run_context.env", "run_context.env")
    report_path = find_artifact(task_dir, "optimization_report.md")

    ci_metrics_source_path = ci_metrics_path or session_breakdown_path
    ci_metrics = parse_ci_metrics(_read_json(ci_metrics_source_path, warnings))
    baseline = parse_baseline_summary(_read_json(baseline_summary_path, warnings))
    kernel_optimizations, kernel_summary = parse_kernel_results(_read_json(kernel_results_path, warnings))

    return {
        "schema_version": SCHEMA_VERSION,
        "run": run or {},
        "task": {
            "task_id": manifest_record.get("task_id"),
            # Claw session UUID — same value SaFE / Hyperloom-Web dashboards
            # use to deep-link into the chat transcript. Populated by
            # optimize_submit::wait_and_collect_one from SaFE task.clawSessionId.
            "claw_session_id": manifest_record.get("claw_session_id"),
            "model": manifest_record.get("model"),
            "display_name": manifest_record.get("display_name"),
            "submit_status": manifest_record.get("status"),
            "final_status": manifest_record.get("final_status"),
            "final_phase": manifest_record.get("final_phase"),
            "final_message": manifest_record.get("final_message"),
            "ci_status": manifest_record.get("ci_status"),
            "ci_success": manifest_record.get("ci_success"),
            "delivery_reason": manifest_record.get("delivery_reason"),
            "detected": manifest_record.get("detected"),
            "overrides": manifest_record.get("overrides") or {},
        },
        "metrics": ci_metrics,
        "baseline": baseline,
        "run_context": parse_env_file(run_context_path),
        "sweep_points": parse_sweep_results(sweep_path, warnings),
        "kernel_candidates": parse_kernel_candidates(_read_json(kernel_candidates_path, warnings)),
        "kernel_optimizations": kernel_optimizations,
        "kernel_summary": kernel_summary,
        "artifacts": build_artifact_index(task_dir),
        "source_files": {
            "ci_metrics": _relative(ci_metrics_path, task_dir),
            "session_breakdown": _relative(session_breakdown_path, task_dir),
            "baseline_summary": _relative(baseline_summary_path, task_dir),
            "sweep_results": _relative(sweep_path, task_dir),
            "kernel_candidates": _relative(kernel_candidates_path, task_dir),
            "kernel_results": _relative(kernel_results_path, task_dir),
            "run_context": _relative(run_context_path, task_dir),
            "optimization_report": _relative(report_path, task_dir),
        },
        "warnings": warnings,
    }


def collect_normalized_results(
    artifacts_dir: Path,
    manifests_dir: Path,
    run: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Normalize CI-collected artifacts using submission manifests.

    This is the GitHub Actions adapter around the entrypoint-agnostic
    ``normalize_task_result`` function.

    Args:
        artifacts_dir (Path): Root directory of downloaded task artifacts,
            keyed by task id.
        manifests_dir (Path): Directory searched recursively for
            ``submission_manifest.json`` files.
        run (dict[str, Any] | None): Optional base run metadata; manifest
            fields fill in any missing values.

    Returns:
        list[dict[str, Any]]: One normalized result per manifest record.
    """
    results: list[dict[str, Any]] = []
    manifest_files = sorted(manifests_dir.rglob("submission_manifest.json"))
    if not manifest_files:
        log.warning("no submission_manifest.json under %s", manifests_dir)
        return results

    for manifest_file in manifest_files:
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("skipping malformed manifest %s: %s", manifest_file, e)
            continue
        manifest_run = dict(run or {})
        manifest_run.setdefault("submitted_at", manifest.get("submitted_at"))
        manifest_run.setdefault("api_url", manifest.get("api_url"))
        manifest_run.setdefault("register_workspace", manifest.get("register_workspace"))
        manifest_run.setdefault("submit_workspace", manifest.get("submit_workspace"))
        manifest_run.setdefault("volume", manifest.get("volume"))

        for record in manifest.get("records") or []:
            if not isinstance(record, dict):
                continue
            task_id = record.get("task_id")
            task_dir = artifacts_dir / task_id if task_id else artifacts_dir / "__missing_task_id__"
            results.append(normalize_task_result(task_dir, record, manifest_run))
    return results


def write_single_result(result: dict[str, Any], out_dir: Path) -> None:
    """Write one normalized result in three on-disk formats.

    Emits ``normalized_result.json`` (the bare object),
    ``normalized_results.json`` (wrapped in a ``results`` list), and
    ``normalized_results.ndjson`` (single NDJSON line).

    Args:
        result (dict[str, Any]): The normalized result to persist.
        out_dir (Path): Output directory; created if it does not exist.

    Returns:
        None
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "normalized_result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (out_dir / "normalized_results.json").write_text(
        json.dumps({"results": [result]}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "normalized_results.ndjson").write_text(
        json.dumps(result, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the manual single-directory entrypoint.

    Returns:
        argparse.ArgumentParser: Parser accepting ``--task-dir`` plus optional
            output/metadata flags.
    """
    parser = argparse.ArgumentParser(
        description="Normalize one downloaded Hyperloom artifact directory.",
    )
    parser.add_argument(
        "--task-dir",
        required=True,
        help="Downloaded task artifact directory, e.g. /path/to/hyperloom",
    )
    parser.add_argument("--out-dir", default="summary-out")
    parser.add_argument("--task-id", default="manual-import")
    parser.add_argument("--model", default="")
    parser.add_argument("--display-name", default="")
    parser.add_argument("--final-status", default="Succeeded")
    parser.add_argument("--source", default="manual-web-import")
    return parser


def main() -> int:
    """Normalize a single artifact directory from CLI args and write outputs.

    Returns:
        int: Process exit code — ``0`` on success, ``2`` if the task
            directory does not exist.
    """
    args = _build_parser().parse_args()
    task_dir = Path(args.task_dir)
    if not task_dir.exists():
        print(f"task dir not found: {task_dir}", file=sys.stderr)
        return 2

    result = normalize_task_result(
        task_dir,
        {
            "task_id": args.task_id,
            "model": args.model or None,
            "display_name": args.display_name or task_dir.name,
            "status": "imported",
            "final_status": args.final_status,
        },
        {"source": args.source},
    )
    write_single_result(result, Path(args.out_dir))
    print(json.dumps({
        "out_dir": args.out_dir,
        "baseline": result["metrics"].get("baseline_throughput"),
        "optimized": result["metrics"].get("optimized_throughput"),
        "gain_pct": result["metrics"].get("gain_pct"),
        "sweep_points": len(result.get("sweep_points") or []),
        "kernel_candidates": len(result.get("kernel_candidates") or []),
        "kernel_optimizations": len(result.get("kernel_optimizations") or []),
        "warnings": result.get("warnings") or [],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
