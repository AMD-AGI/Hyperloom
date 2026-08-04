# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pure-function builder for the FRAMEWORK candidate-ranker prompt.

Extracted from ``phases/framework.py::_rank_framework_agent_candidates_llm``
so the prompt text is testable and renderable by the audit script.
"""

from __future__ import annotations

_FOOTER = (
    "Prefer PRs from this session's own framework repo, especially those "
    "targeting the serving hot path (MoE/FP8/attention/GEMM/KV-cache/scheduling). "
    "A cross-framework PR is acceptable when it carries transferable high-value "
    "serving tech worth porting. Always choose exactly ONE candidate; reply "
    '{"candidate_id": "<id>", "reason": "<short>"}.'
)

_LOCAL_EXPLORE_NOTE = (
    "One option above is a LOCAL-EXPLORATION arm (its id starts with "
    "'local_explore:'): instead of integrating an upstream PR, a "
    "write-capable specialist authors a patch directly from the live "
    "source + profiling evidence (it may also compare against the "
    "latest upstream code via web search). Prefer it when the "
    "discovered PRs look weak, already-present, or off the current "
    "bottleneck."
)


def build_framework_ranker_prompt(
    *,
    model: str,
    framework: str,
    gpu_type: str,
    precision: str,
    tp: int | str,
    best_throughput: float | str | None,
    candidate_rows: list[str],
    has_local_explore: bool,
    memory_block: str,
) -> str:
    """Build the FRAMEWORK candidate-ranker prompt."""
    lines: list[str] = [
        "You are selecting ONE upstream PR to integrate next, to maximize "
        "LLM serving throughput (tokens/s) for this exact workload:",
        f"- model: {model}",
        f"- framework: {framework}",
        f"- gpu_type: {gpu_type}",
        f"- precision: {precision}",
        f"- tensor_parallel: {tp}",
    ]
    if best_throughput:
        lines.append(f"- current_best_throughput_tok_s: {best_throughput}")
    lines.append("")
    lines.append("Candidates (choose the ONE most likely to raise throughput):")
    lines.extend(candidate_rows)
    if has_local_explore:
        lines.append("")
        lines.append(_LOCAL_EXPLORE_NOTE)
    if memory_block:
        lines.append("")
        lines.append(memory_block)
    lines.append("")
    lines.append(_FOOTER)
    return "\n".join(lines)


__all__ = ["build_framework_ranker_prompt"]
