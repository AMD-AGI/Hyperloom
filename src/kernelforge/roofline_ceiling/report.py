# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a ceiling lands, how it is read back, and how a campaign shows it.

Both files are written by the analyst, not by this module. It names them, reads
one of them, and renders the attainment block a campaign injects into its
planning prompt.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from kernelforge.roofline_ceiling.attainment import measure_attainment
from kernelforge.roofline_ceiling.contract import CeilingReport, load_report

log = logging.getLogger("kernelforge.roofline_ceiling")

#: The answer, machine-readable: ``cases`` and ``mean_ideal_ms``.
REPORT_FILENAME = "performance_ceiling.json"
#: The derivation, for a reader deciding whether to believe the answer. Nothing
#: recomputes the latencies, so this document is all there is to check them by.
DOCUMENT_FILENAME = "performance_ceiling_analysis.md"
#: Kernel trace, driver output, and whatever the analyst's own measurement left
#: behind. Handed to the analyst as a place it may write.
EVIDENCE_DIRNAME = "evidence"

#: Where a ceiling lands when the caller names no output directory. Under the
#: workspace, next to everything else a campaign writes.
WORKSPACE_SUBDIR = "forge_experiments/roofline_ceiling"


def read_report(path: str | Path) -> CeilingReport:
    """Load a published ceiling file. Raises on anything unreadable."""
    return load_report(json.loads(Path(path).read_text(encoding="utf-8")))


def _format_ms(value: float) -> str:
    """Render a latency with enough digits to be useful at microsecond scale."""
    return f"{value:.6g}"


def render_for_prompt(
    report: CeilingReport,
    current_ms: Mapping[str, float] | None = None,
    *,
    target: float = 0.0,
    unscored_cases: Sequence[str] | None = None,
) -> str:
    """Render the roofline block a campaign injects into its planning prompt.

    Names the target, the standing, and which shapes are short of it. The table
    is ordered worst-first, which is the point of showing it: the score is the
    equal-weight mean across cases, so the shape furthest from its ceiling is
    where the next unit of effort buys most.

    A ceiling is derived, not measured, so a case reported near its ceiling is
    also a reason to read the derivation before believing the headroom is gone.
    Under a target the campaign stops on, that is the reader's only defence
    against a work model that understated the job.
    """
    if not report.cases:
        return ""

    standing = measure_attainment(report, current_ms, unscored_cases=unscored_cases)
    lines = [
        "### Roofline attainment",
        "",
        "`attainment = ceiling / measured`: the fraction of the estimated best achievable",
        "latency this kernel is delivering. The ceiling is an estimate, not a measurement,",
        "and it decides no KEEP.",
        "",
    ]

    if standing.usable:
        lines.append(f"Campaign attainment: **{standing.mean * 100:.1f}%** (equal-weight mean across scored cases).")
        if target > 0:
            short = standing.below(target)
            lines.append(
                f"Target: **{target * 100:.0f}%**. "
                + (
                    f"{len(short)} case(s) below it; the campaign ends when the mean reaches the target."
                    if short
                    else "Every case is at or above it."
                )
            )
        lines.append("")

    if standing.cases:
        lines += [
            "| Case | Attainment | Ceiling (ms) | Measured (ms) | Speedup to ceiling |",
            "|:--|--:|--:|--:|--:|",
        ]
        for entry in sorted(standing.cases, key=lambda c: c.attainment):
            lines.append(
                f"| `{entry.case_id}` | {entry.attainment * 100:.1f}% | {_format_ms(entry.t_ideal_ms)} "
                f"| {_format_ms(entry.t_current_ms)} | {entry.remaining_speedup:.2f}x |"
            )
        lines.append("")
    else:
        lines += ["| Case | Ceiling (ms) |", "|:--|--:|"]
        lines += [f"| `{case.case_id}` | {_format_ms(case.t_ideal_ms)} |" for case in report.cases]
        lines.append("")

    if standing.excluded:
        lines.append("Cases with no attainment figure, and why:")
        lines += [f"- `{case_id}`: {reason}" for case_id, reason in sorted(standing.excluded.items())]
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "DOCUMENT_FILENAME",
    "EVIDENCE_DIRNAME",
    "REPORT_FILENAME",
    "WORKSPACE_SUBDIR",
    "read_report",
    "render_for_prompt",
]
