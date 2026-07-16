#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Aggregate per-task ci_metrics.json into the final summary table.

Inputs:
  --artifacts-dir DIR   Directory containing per-task artifacts (task-artifacts/<task_id>/)
  --manifests-dir DIR   Directory containing submission_manifest.json files (one per matrix job)
  --isl / --osl         Benchmark input/output sequence lengths (shown in the summary header)
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
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifact_normalizer import collect_normalized_results  # noqa: E402

log = logging.getLogger("build-summary")


# ── Row assembly ────────────────────────────────────────────────────────────────


def collect_rows(
    artifacts_dir: Path,
    manifests_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """Build one summary row per normalized task result.

    Normalizes per-task artifacts and derives success/status fields.

    Args:
        artifacts_dir (Path): Root dir of per-task artifacts.
        manifests_dir (Path): Root dir of submission_manifest.json file(s).

    Returns:
        tuple[list[dict], list[dict]]: ``(rows, normalized_results)`` where
            ``rows`` are the rendered summary rows and ``normalized_results``
            are the underlying normalized records.
    """
    rows: list[dict] = []
    normalized_results = collect_normalized_results(artifacts_dir, manifests_dir, build_run_metadata())

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
            "ci_status": task.get("ci_status"),
            "ci_success": task.get("ci_success"),
            "delivery_reason": task.get("delivery_reason"),
            "framework": metrics.get("framework") or detected.get("framework"),
            "precision": detected.get("precision"),
            "tp": metrics.get("tp") or detected.get("tp"),
            "params_b": detected.get("params_b"),
            "gpu_type": metrics.get("gpu_type"),
            "baseline_tok_per_gpu": (metrics.get("tok_per_gpu_baseline") or metrics.get("baseline_throughput")),
            "optimized_tok_per_gpu": (metrics.get("tok_per_gpu_optimized") or metrics.get("optimized_throughput")),
            "gain_pct": metrics.get("gain_pct"),
            "peak_throughput": metrics.get("peak_throughput"),
            "peak_throughput_conc": metrics.get("peak_throughput_conc"),
            "actions": metrics.get("actions") or [],
            "sweep_points": len(item.get("sweep_points") or []),
            "kernel_candidates": len(item.get("kernel_candidates") or []),
            "kernel_optimizations": len(item.get("kernel_optimizations") or []),
            "artifacts": len(item.get("artifacts") or []),
            "warnings": item.get("warnings") or [],
        }
        if not row["ci_success"]:
            has_delivered_metrics = (
                row["baseline_tok_per_gpu"] is not None
                or row["optimized_tok_per_gpu"] is not None
                or row["gain_pct"] is not None
                or (item.get("source_files") or {}).get("session_breakdown") is not None
            )
            row["ci_success"] = bool(
                row["final_status"] == "Succeeded" or row["ci_status"] == "Delivered" or has_delivered_metrics
            )
        if not row["ci_status"]:
            row["ci_status"] = "Delivered" if row["ci_success"] else "Missing artifacts"

        rows.append(row)

    return rows, normalized_results


def build_run_metadata() -> dict:
    """Capture GitHub Actions context when present; harmless for local runs.

    Returns:
        dict: Run metadata with the ``source`` tag and ``GITHUB_*`` env values
            (``None`` for any that are unset).
    """
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


def fmt_num(v, fmt: str = ".1f") -> str:
    """Format a numeric value for the table, with a dash placeholder.

    Args:
        v: The value to format, or ``None``.
        fmt (str): A format spec applied to ``v`` (default ``".1f"``).

    Returns:
        str: The formatted number, ``"—"`` if ``v`` is ``None``, or
            ``str(v)`` if formatting fails.
    """
    if v is None:
        return "—"
    try:
        return format(v, fmt)
    except Exception:
        return str(v)


def fmt_pct(v) -> str:
    """Format a percentage value with a sign, with a dash placeholder.

    Args:
        v: The percentage value to format, or ``None``.

    Returns:
        str: ``"+x.xx%"``-style text, ``"—"`` if ``v`` is ``None``, or
            ``str(v)`` if formatting fails.
    """
    if v is None:
        return "—"
    try:
        return f"{v:+.2f}%"
    except Exception:
        return str(v)


def gain_medal(pct: float | None) -> str:
    """Award medals based on gain percentage (sample-table convention).

    Args:
        pct (float | None): The gain percentage.

    Returns:
        str: A medal/emoji string keyed to the gain band, or ``""`` when
            ``pct`` is ``None`` or negative.
    """
    if pct is None:
        return ""
    if pct >= 50:
        return "🥇🥇🥇🥇"
    if pct >= 20:
        return "🥇🥇🥇"
    if pct >= 10:
        return "🥇🥇"
    if pct >= 1:
        return "🥇"
    if pct > 0:
        return "🟢"
    if pct == 0:
        return "➖"
    return ""


def status_icon(row: dict) -> str:
    """Choose a status icon for a summary row.

    Args:
        row (dict): A summary row with ``ci_success``, ``final_status``, and
            baseline/optimized throughput fields.

    Returns:
        str: ``"✅"`` (success), ``"🟡"`` (partial metrics), or ``"❌"``.
    """
    final = row.get("final_status")
    baseline = row.get("baseline_tok_per_gpu")
    optimized = row.get("optimized_tok_per_gpu")
    if row.get("ci_success"):
        return "✅"
    if final and final not in ("Succeeded", "Completed"):
        return "❌"
    if baseline is not None and optimized is not None:
        return "✅"
    if baseline is not None or optimized is not None:
        return "🟡"
    return "❌"


_PARAMS_RX = re.compile(r"(\d+(?:\.\d+)?)\s*[Bb](?:\b|[-_])")


def derive_params(repo_id: str | None) -> str | None:
    """Best-effort: pull '14B', '70B', '1.5B' out of an HF repo_id.

    Args:
        repo_id (str | None): The HF model id to parse.

    Returns:
        str | None: A normalized parameter label (e.g. ``"14B"``, ``"1.5B"``),
            or ``None`` if no count is found.
    """
    if not repo_id:
        return None
    m = _PARAMS_RX.search(repo_id)
    if not m:
        return None
    raw = m.group(1)
    try:
        v = float(raw)
        return f"{v:.1f}B" if "." in raw else f"{int(v)}B"
    except ValueError:
        return None


def short_model_name(repo_id: str | None) -> str:
    """Strip 'owner/' prefix for compact display.

    Args:
        repo_id (str | None): The HF model id.

    Returns:
        str: The portion after the first ``/``, or ``"—"`` if ``repo_id`` is
            falsy.
    """
    if not repo_id:
        return "—"
    return repo_id.split("/", 1)[-1]


def gain_sort_key(row: dict) -> tuple[int, float]:
    """Sort key: rows with a numeric gain first (desc), failures last.

    Args:
        row (dict): A summary row with ``ci_success`` and ``gain_pct``.

    Returns:
        tuple[int, float]: ``(delivered_rank, -gain)`` so delivered rows sort
            before failures and higher gains sort first.
    """
    delivered_rank = 0 if row.get("ci_success") else 1
    pct = row.get("gain_pct")
    if pct is None:
        return (delivered_rank, 1.0)
    try:
        return (delivered_rank, -float(pct))
    except (TypeError, ValueError):
        return (delivered_rank, 1.0)


def render_markdown(rows: list[dict], isl: int, osl: int) -> str:
    """Render the ranked summary in the format used for executive reporting.

    Args:
        rows (list[dict]): Summary rows produced by :func:`collect_rows`.
        isl (int): Input sequence length shown in the header.
        osl (int): Output sequence length shown in the header.

    Returns:
        str: The rendered markdown table (trailing newline included).
    """
    sorted_rows = sorted(rows, key=gain_sort_key)
    n = len(rows)
    delivered = sum(1 for r in rows if r.get("ci_success"))
    safe_succeeded = sum(1 for r in rows if r.get("final_status") == "Succeeded")
    with_gain = sum(
        1
        for r in rows
        if (r.get("gain_pct") or 0) > 0
        and r.get("baseline_tok_per_gpu") is not None
        and r.get("optimized_tok_per_gpu") is not None
    )
    lines = [
        "# Hyperloom CI Summary",
        f"- Models: {n} (Delivered: {delivered}, SaFE Succeeded: {safe_succeeded}, with gain: {with_gain})",
        f"- ISL/OSL: {isl} / {osl}",
        "- Sort: by Gain% (desc); failures last",
        "",
        "| # | Model | Frm | Prec | TP | Params | Baseline tok/s/GPU | " + "Optimized tok/s/GPU | Gain |",
        "|---:|---|---|---|---:|---|---:|---:|---|",
    ]
    for idx, r in enumerate(sorted_rows, start=1):
        name = short_model_name(r.get("model"))
        icon = status_icon(r)
        frm = r.get("framework") or "—"
        prec = r.get("precision") or "—"
        tp = r.get("tp") or "—"
        params = derive_params(r.get("model")) or "—"
        baseline = fmt_num(r.get("baseline_tok_per_gpu"))
        optimized = fmt_num(r.get("optimized_tok_per_gpu"))
        gain = r.get("gain_pct")
        medal = gain_medal(gain)
        if gain is None:
            gain_text = "—"
        elif gain == 0:
            gain_text = f"{medal} 0%".strip()
        else:
            gain_text = f"{medal} {fmt_pct(gain)}".strip()
        lines.append(
            f"| {idx} | `{name}` {icon} | {frm} | {prec} | {tp} | {params} "
            f"| {baseline} | {optimized} | {gain_text} |"
        )
    return "\n".join(lines) + "\n"


# ── Main ────────────────────────────────────────────────────────────────────────


def main() -> int:
    """CLI entry point: build the summary table and write output artifacts.

    Parses arguments, collects rows, and writes ``ci_summary.{md,json}`` plus
    ``normalized_results.{json,ndjson}`` into the output directory.

    Returns:
        int: Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--artifacts-dir", required=True, help="Root dir of per-task artifacts (task-artifacts/<task_id>/...)"
    )
    parser.add_argument("--manifests-dir", required=True, help="Root dir containing submission_manifest.json file(s)")
    parser.add_argument("--isl", type=int, default=1024)
    parser.add_argument("--osl", type=int, default=1024)
    parser.add_argument("--out-dir", default="summary-out")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    rows, normalized_results = collect_rows(
        Path(args.artifacts_dir),
        Path(args.manifests_dir),
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = render_markdown(rows, args.isl, args.osl)
    (out / "ci_summary.md").write_text(md, encoding="utf-8")
    (out / "ci_summary.json").write_text(
        json.dumps(
            {
                "isl": args.isl,
                "osl": args.osl,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "normalized_results.json").write_text(
        json.dumps(
            {
                "isl": args.isl,
                "osl": args.osl,
                "results": normalized_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "normalized_results.ndjson").write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in normalized_results),
        encoding="utf-8",
    )
    log.info("wrote summary and normalized results under %s", out)

    # Echo the table to stdout for the GitHub Actions log.
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
