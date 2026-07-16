# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bypass benchmark analysis layer.

Pure, GPU-free functions that turn raw run artifacts (server.log, stderr,
lm-eval results) into structured, higher-signal analysis. The output is
additive: it lands under ``report["bypass_analysis"]`` and never overrides the
InferenceX-reported measurements, so the downstream contract is unchanged.

Three capabilities:
* steady-state throughput estimate from the engine's periodic throughput logs,
* structured failure attribution (root-cause tag) from logs/stderr,
* normalized eval summary from lm-eval ``results*.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# sglang: "Decode batch. ... gen throughput (token/s): 1234.5"
# vllm:   "Avg generation throughput: 1234.5 tokens/s"
_THROUGHPUT_PATTERNS = (
    re.compile(r"gen throughput \(token/s\):\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"Avg generation throughput:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
)

# Ordered failure signatures: first match wins. Each maps a root-cause tag to
# substrings (case-insensitive) that identify it in logs/stderr.
_FAILURE_SIGNATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("oom", ("out of memory", "outofmemoryerror", "hip out of memory", "cuda out of memory")),
    ("cuda_graph_capture", ("capture cuda graph", "cuda graph capture", "stream capture")),
    ("port_conflict", ("address already in use", "port is already in use", "eaddrinuse")),
    ("detokenizer_stall", ("detokenizer", "decode stalled", "watchdog timeout")),
    ("server_init_dead", (
        "engine core init failed", "engine process failed", "worker died",
        "failed to launch", "server failed to start", "traceback (most recent call last)",
    )),
)


def parse_server_log_throughput(text: str) -> list[float]:
    """Extract every positive periodic generation-throughput sample.

    Args:
        text: The server.log contents.

    Returns:
        A list of positive tokens/sec samples, in log order.
    """
    samples: list[float] = []
    for line in text.splitlines():
        for pattern in _THROUGHPUT_PATTERNS:
            match = pattern.search(line)
            if match:
                try:
                    value = float(match.group(1))
                except ValueError:
                    continue
                if value > 0:
                    samples.append(value)
                break
    return samples


def steady_state_mean(samples: list[float], *, warmup_skip_frac: float = 0.2) -> float | None:
    """Average the steady-state portion of throughput samples.

    Drops the leading ``warmup_skip_frac`` of samples before averaging; falls
    back to the full set when the trim would empty it.

    Args:
        samples: Positive throughput samples in log order.
        warmup_skip_frac: Fraction of leading samples treated as warmup.

    Returns:
        The steady-state mean, or None when there are no samples.
    """
    positive = [s for s in samples if s > 0]
    if not positive:
        return None
    skip = int(len(positive) * max(0.0, min(warmup_skip_frac, 0.9)))
    trimmed = positive[skip:] or positive
    return sum(trimmed) / len(trimmed)


def estimate_steady_state_from_log(server_log: Path, *, warmup_skip_frac: float = 0.2) -> dict[str, Any]:
    """Compute a steady-state throughput block from a server.log file.

    Args:
        server_log: Path to the engine server.log.
        warmup_skip_frac: Warmup fraction passed to :func:`steady_state_mean`.

    Returns:
        A dict with ``sample_count`` and ``steady_state_output_throughput``
        (None when unavailable). Never raises.
    """
    try:
        text = server_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"sample_count": 0, "steady_state_output_throughput": None}
    samples = parse_server_log_throughput(text)
    return {
        "sample_count": len(samples),
        "steady_state_output_throughput": steady_state_mean(
            samples, warmup_skip_frac=warmup_skip_frac
        ),
    }


def classify_failure(*texts: str) -> str | None:
    """Return a structured root-cause tag from logs/stderr, or None.

    The first matching signature (in priority order) wins. ``None`` means no
    known signature matched (caller may treat as ``unknown``).

    Args:
        *texts: Any number of text blobs (server.log, stderr, stdout).

    Returns:
        A root-cause tag string, or None when nothing matched.
    """
    blob = "\n".join(t for t in texts if t).lower()
    if not blob:
        return None
    for tag, needles in _FAILURE_SIGNATURES:
        if any(n in blob for n in needles):
            return tag
    return None


def summarize_eval(workspace: Path) -> dict[str, Any] | None:
    """Normalize an lm-eval ``results*.json`` into a compact eval summary.

    Searches ``workspace`` recursively for lm-eval result files and extracts
    the first recognized accuracy metric.

    Args:
        workspace: Benchmark workspace directory.

    Returns:
        ``{"task", "metric", "accuracy", "source_file"}`` on success, or None
        when no result/metric is found. Never raises.
    """
    result_files = sorted(workspace.rglob("results*.json"))
    if not result_files:
        return None
    latest = result_files[-1]
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, dict):
        return None
    metric_keys = (
        "exact_match,strict-match",
        "exact_match,flexible-extract",
        "exact_match,none",
        "acc,none",
    )
    for task_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for key in metric_keys:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return {
                    "task": task_name,
                    "metric": key,
                    "accuracy": float(value),
                    "source_file": str(latest),
                }
    return None


def build_analysis(
    *,
    workspace: Path,
    server_log: Path,
    success: bool,
    stderr_text: str = "",
    run_eval: bool = False,
) -> dict[str, Any]:
    """Assemble the ``bypass_analysis`` block for a report.

    Args:
        workspace: Benchmark workspace directory.
        server_log: Engine server.log path.
        success: Whether the benchmark succeeded.
        stderr_text: Optional client/server stderr tail for failure attribution.
        run_eval: Whether an eval pass was requested.

    Returns:
        The analysis dict (always safe to embed; values may be None).
    """
    analysis: dict[str, Any] = {
        "throughput": estimate_steady_state_from_log(server_log),
    }
    if not success:
        server_text = ""
        try:
            server_text = server_log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            server_text = ""
        analysis["failure_root_cause"] = classify_failure(server_text, stderr_text) or "unknown"
    if run_eval:
        analysis["eval_summary"] = summarize_eval(workspace)
    return analysis