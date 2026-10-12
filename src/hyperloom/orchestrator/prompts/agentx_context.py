# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX workload and grading blocks both prompts render."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

from hyperloom.common.agentx_workload import BACKEND_ENV, is_mlperf_backend
from hyperloom.common.perf_metric import AGENTX_KEEP_P50_THRESHOLD_PCT
from hyperloom.inference_optimizer.grading import resolved_grading

# Percentiles rendered per axis, in order.
_RENDERED = ("p50", "p90", "p99")


def _shape_row(label: str, dist: Mapping[str, Any] | None) -> str:
    """Render one sequence-length distribution as ``p50 .. p90 .. p99``."""
    dist = dist or {}
    parts = [f"{p} {int(dist[p]):,}" for p in _RENDERED if isinstance(dist.get(p), (int, float))]
    return f"- {label}: " + ("   ".join(parts) if parts else "(not measured)")


def corpus_lines(shape: Mapping[str, Any] | None) -> list[str]:
    """Describe the agentic corpus a session replays, from ``SharedState.agentx_corpus_shape``; ``[]`` when unset."""
    # One renderer for both prompts so the corpus numbers are never two copies of the same literals. The shape is
    # seeded from the canonical corpus and replaced by each aiperf run's measured distribution.
    if not shape:
        return []
    lines = [
        "**Workload: AgentX agentic trace replay.** The corpus fixes the request",
        "shape, so the ISL/OSL carried in state are inert placeholders.",
    ]
    loader = str(shape.get("corpus_loader") or "").strip()
    entries = shape.get("corpus_entries")
    duration = shape.get("duration_s")
    corpus = f"- corpus: {loader or 'agentic traces'}"
    if isinstance(entries, (int, float)) and entries:
        corpus += f", {int(entries)} traces"
    if isinstance(duration, (int, float)) and duration:
        corpus += f", {int(duration)}s window"
    lines.append(corpus)
    lines.append(_shape_row("input/req ", shape.get("isl")))
    lines.append(_shape_row("output/req", shape.get("osl")))
    hit = shape.get("prefix_cache_hit")
    if isinstance(hit, (int, float)) and hit:
        lines.append(f"- prefix cache hit ~{hit:.1%}, so prefill compute is far below the input count")
    return lines


def grading_lines(grading: Mapping[str, Any] | None = None, backend: str = "") -> list[str]:
    """Describe what an AgentX KEEP is decided on, from the session's recorded ``grading`` and ``agentx_backend``.

    A session recorded before the backend field existed falls back to the process environment for it.
    """
    if is_mlperf_backend({BACKEND_ENV: backend} if backend else None):
        return [
            "**Graded on output token throughput** — the MLPerf agentic harness replays a",
            "fixed trajectory set and publishes no per-request interactivity, so a KEEP is",
            "an output-throughput gain that holds the smoke accuracy of the baseline.",
        ]
    on_intvty, _ = resolved_grading(SimpleNamespace(grading=dict(grading or {}), framework="", benchmark_mode="agentx"))
    if not on_intvty:
        return [
            "**Graded on output token throughput** — this session was seeded on that axis,",
            "so a KEEP is an output-throughput gain; interactivity is measured, not graded.",
        ]
    return [
        "**Graded on E2E normalised interactivity P50** — the median of",
        "`OSL_i / E2EL_i` in tok/s/user. InferenceX ranks submissions on the P90 of the",
        "same family; the median is what Hyperloom decides KEEP on, and P90 is held as a guard.",
        f"KEEP needs >=+{AGENTX_KEEP_P50_THRESHOLD_PCT:.0f}% on it while the slow tail (P90) and output",
        "throughput each hold inside the noise band. Anything short of all three is REVERT.",
        "Total token throughput is still measured, and is not the objective.",
    ]
