# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The text of the decode-path discovery prompt, with a named slot for every piece of evidence it cites."""

from __future__ import annotations

from typing import Any, Sequence

from hyperloom_common.coerce import to_float

DISCOVERY_PROMPT_TEMPLATE = """You are analyzing the DECODE path of a {framework} model (`model_type={model_type}`)
on {arch_phrase}, bf16, to find SOURCE-LEVEL KERNEL FUSIONS. Analyze only; do
not edit anything. Return your answer as JSON (schema below).

## The lever (general, not model-specific)
The decode path is launch/dispatch bound: launch_bound_share={diagnosis.launch_bound_share:.2f}
of GPU-busy time is in many tiny fp32 ops ({lb}) rather than the heavy GEMM/attention.
Each tiny op is a separate kernel launch + HBM round-trip. The win is to FUSE a
CONTIGUOUS chain of these tiny ops (as they appear in the decode forward, between two
heavy ops) into ONE kernel — fewer launches, fewer round-trips.

## Measured launch-bound hot kernels (top, this trace)
{hot_lines}
Dominant launch-bound categories: {dom}
Representative decode shapes: {shapes}

## Ordered fusion boundaries (same stream, compute endpoints retained)
{boundary_lines}

## Existing ROCm operator evidence
Retrieved by whole-word overlap between the observed kernel/category names and the
documented operation chain; a higher score means stronger evidence. The score is a
retrieval hint, NOT a correctness claim: you must still confirm from the source and
the card that shape, dtype, and cache layout actually match.
{operator_lines}

{fusion_hints_block}## Your task
{read_verb} up to {max_fusions} CONTIGUOUS op chains in
the DECODE forward that are worth fusing. Include GEMM/attention prologue or epilogue
boundaries when the ordered trace proves adjacency. Treat an existing ROCm operator
that covers a larger boundary as an `integration` candidate and benchmark/wire it
before proposing a new kernel. Judge from the SOURCE and ordered trace what actually
runs back-to-back on the decode path.

{fusion_constraints}

{output_schema_block}

{source_block}
"""


def _boundary_avg_us(boundary: dict[str, Any]) -> float:
    if "avg_chain_us" in boundary:
        return float(boundary["avg_chain_us"])
    return float(boundary.get("total_us", 0.0)) / max(1, int(boundary["count"]))


def _boundary_line(boundary: dict[str, Any]) -> str:
    line = (
        "  - "
        f"{boundary['signature']} "
        f"(repeats={boundary['count']}, "
        f"avg-chain={_boundary_avg_us(boundary):.1f}us, "
        f"kind={boundary['boundary_kind']}, "
        f"removable-launches<={boundary['launches_removed_upper_bound']})"
    )
    # The trailing compute kernel proves adjacency but is not fusable, so the fusable span is spelled out to keep
    # the proposed chain from swallowing the attention (or other compute) kernel it feeds.
    fusable = boundary.get("fusable_categories")
    terminal = str(boundary.get("terminal_compute") or "")
    if fusable:
        line += f"\n    fusable-span={' -> '.join(fusable)}"
    if terminal:
        line += f"\n    NOTE: fuse the prologue only; do NOT include the terminal {terminal} kernel in the fused chain."
    line += f"\n    kernels: {' | '.join(boundary.get('kernels', []))[:500]}"
    return line


def _operator_line(hint: dict[str, Any]) -> str:
    score = hint.get("score")
    score_text = f" score={float(score):.1f}" if score is not None else ""
    return f"  - `{hint['operator']}` ({hint['path']}){score_text}: {hint['evidence']}"


def render_boundary_lines(boundaries: Sequence[dict[str, Any]] | None) -> str:
    """The ordered fusion boundaries of the trace, one entry per same-stream chain."""
    return "\n".join(_boundary_line(boundary) for boundary in boundaries or [])


def render_operator_lines(hints: Sequence[dict[str, Any]] | None) -> str:
    """The existing operators retrieval matched against the trace, with their recall scores."""
    return "\n".join(_operator_line(hint) for hint in hints or [])


def render_fusion_hints(hints: Sequence[dict[str, Any]]) -> str:
    """The advisory profiler-hints section of the discovery prompt; empty when no hints were given."""
    if not hints:
        return ""
    lines = [
        "## External profiler hints (advisory)",
        "Another profiler (e.g. TraceLens) flagged these unfused kernel runs, largest projected saving first. Long",
        "kernel names may be truncated with `...`. They are leads, not answers: propose a chain only where the",
        "trace evidence above and the source confirm it.",
    ]
    for hint in hints:
        pct = to_float(hint.get("projected_e2e_pct"))
        saving = f", projected E2E saving {pct:.1f}%" if pct is not None else ""
        lines.append(f"  - {hint.get('kind') or 'unclassified'}: {hint.get('operation') or ''}{saving}")
        members = [m for m in hint.get("members") or [] if isinstance(m, dict)]
        kernels = " | ".join(" + ".join(map(str, m.get("kernel_names") or [])) for m in members)
        launchers = sorted({str(m["kernel_launcher_path"]) for m in members if m.get("kernel_launcher_path")})
        if kernels:
            lines.append(f"    kernels: {kernels[:500]}")
        if launchers:
            lines.append(f"    launchers: {'; '.join(launchers)}")
    return "\n".join(lines) + "\n\n"
