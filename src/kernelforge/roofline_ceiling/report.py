# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publishing, caching and rendering one ceiling report."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from kernelforge.roofline_ceiling.attainment import measure_attainment
from kernelforge.roofline_ceiling.contract import CeilingContractError, CeilingReport, load_report
from kernelforge.roofline_ceiling.specs import peak_source_meaning
from kernelforge.durable_io import atomic_write_text

log = logging.getLogger("kernelforge.roofline_ceiling")

REPORT_FILENAME = "performance_ceiling.json"
DOCUMENT_FILENAME = "performance_ceiling_analysis.md"
EVIDENCE_DIRNAME = "evidence"

#: Where a ceiling lands when the caller names no output directory. Under the
#: workspace, next to everything else a campaign writes.
WORKSPACE_SUBDIR = "forge_experiments/roofline_ceiling"


def publish(report: CeilingReport, output_dir: str | Path) -> Path:
    """Write the report and its human-readable companion; return the JSON path."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / REPORT_FILENAME
    atomic_write_text(json_path, json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    atomic_write_text(destination / DOCUMENT_FILENAME, render_document(report))
    return json_path


def read_report(path: str | Path) -> CeilingReport:
    """Load a published report. Raises on anything unreadable."""
    return load_report(json.loads(Path(path).read_text(encoding="utf-8")))


def _format_ms(value: float) -> str:
    """Render a latency with enough digits to be useful at microsecond scale."""
    return f"{value:.6g}"


def render_document(report: CeilingReport) -> str:
    """Render the operator-facing analysis document.

    A framework-owned header -- the answer, where the roofs came from, and
    anything the validator objected to -- followed verbatim by the analyst's own
    derivation. The published file carries only the latencies and their mean, so
    this document is where a reader deciding whether to believe them looks.
    """
    mean = report.mean_ideal_ms()
    lines: list[str] = [
        "# Performance ceiling",
        "",
        "## Conclusion",
        "",
        "Theoretical achievable latency per scored case. This is an optimistic lower bound under",
        "hardware limits and legal algorithm constraints; it does not claim an implementation",
        "reaching it exists.",
        "",
        "| Case | Ideal latency (ms) |",
        "|:--|--:|",
    ]
    lines += [f"| `{case.case_id}` | {_format_ms(case.t_ideal_ms)} |" for case in report.cases]
    if mean is not None:
        lines.append(f"| **mean (equal weight)** | **{_format_ms(mean)}** |")

    lines += [
        "",
        f"Roofs: `{report.peak_source}` — {peak_source_meaning(report.peak_source)}.",
        "",
    ]

    flagged = [(case.case_id, issue) for case in report.cases for issue in case.issues]
    if flagged:
        lines += ["## Findings", ""]
        lines += [f"- `{case_id}`: {issue}" for case_id, issue in flagged]
        lines.append("")

    lines += [
        "## Derivation",
        "",
        "Written by the analyst. Nothing downstream recomputes these latencies, so this is the",
        "only record of how they were reached.",
        "",
        report.analysis_md.strip(),
    ]

    return "\n".join(lines).rstrip() + "\n"


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
    also a reason to check the derivation before believing the headroom is gone.
    Under a target the campaign stops on, that check is the reader's only
    defence against a work model that understated the job.
    """
    if not report.cases:
        return ""

    standing = measure_attainment(report, current_ms, unscored_cases=unscored_cases)
    lines = [
        "### Roofline attainment",
        "",
        "`attainment = ceiling / measured`: the fraction of the estimated best achievable latency",
        "this kernel is delivering. The ceiling comes from a work model over measured hardware",
        "roofs; it is an estimate, not a measurement, and it decides no KEEP.",
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
    "publish",
    "read_report",
    "render_document",
    "render_for_prompt",
]
