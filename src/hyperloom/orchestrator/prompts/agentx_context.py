# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX workload and grading blocks both prompts render."""

from __future__ import annotations

from typing import Any, Mapping

from hyperloom.common.perf_metric import AGENTX_KEEP_THRESHOLD_FLOOR_PCT

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


def grading_lines() -> list[str]:
    """Describe what an AgentX KEEP is decided on."""
    return [
        "**Graded on E2E normalised interactivity P90** — the SLOW tail,",
        "`1 / P90(E2EL_i / OSL_i)` in tok/s/user. This is the axis InferenceX ranks a",
        "submission on, so a change that only speeds up the fastest users scores zero.",
        f"KEEP needs >={AGENTX_KEEP_THRESHOLD_FLOOR_PCT:.0f}% on it with per-chip token throughput held",
        "inside the noise band; both axes worse is REVERT; neither dominating is",
        "RECORDED, which is measured and kept in the ledger but not promoted.",
        "Output throughput is still measured, and is not the objective.",
    ]
