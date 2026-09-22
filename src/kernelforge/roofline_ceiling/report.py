# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publishing, caching and rendering one ceiling report."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from kernelforge.roofline_ceiling.attainment import measure_attainment
from kernelforge.roofline_ceiling.contract import CeilingContractError, CeilingReport, load_report
from kernelforge.roofline_ceiling.specs import peak_source_meaning
from kernelforge.durable_io import atomic_write_text
from kernelforge.resources import default_project_root

log = logging.getLogger("kernelforge.roofline_ceiling")

REPORT_FILENAME = "performance_ceiling.json"
DOCUMENT_FILENAME = "performance_ceiling_analysis.md"
EVIDENCE_DIRNAME = "evidence"

#: Where a ceiling lands when the caller names no output directory. Under the
#: workspace, next to everything else a campaign writes.
WORKSPACE_SUBDIR = "forge_experiments/roofline_ceiling"


def case_set_hash(case_ids: Sequence[str]) -> str:
    """A short, order-independent digest of the scored case set."""
    joined = "\n".join(sorted({str(case_id) for case_id in case_ids}))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def roofs_hash(
    peak_flops: Mapping[str, float] | None,
    bandwidth: Mapping[str, float] | None,
) -> str:
    """A digest of the exact roofs a ceiling was divided by."""
    figures = [f"{name}={float(value):.6g}" for name, value in sorted((peak_flops or {}).items())]
    figures += [f"bw.{name}={float(value):.6g}" for name, value in sorted((bandwidth or {}).items())]
    return hashlib.sha256("\n".join(figures).encode("utf-8")).hexdigest()[:16]


def cache_key(
    *,
    canonical_id: str,
    case_ids: Sequence[str],
    arch: str,
    peak_source: str,
    peak_flops: Mapping[str, float] | None = None,
    bandwidth: Mapping[str, float] | None = None,
) -> str:
    """Identity of one ceiling answer.

    The peak source is part of the key, not metadata on it. A datasheet ceiling
    and an empirical one are different answers to different questions, and
    serving the first from cache when the caller asked for the second is exactly
    the silent degrade this module exists to prevent.

    The roof *values* are in the key for the same reason one step down. Nothing
    recomputes a published ceiling, so a cached one cannot be brought up to date
    after the box is re-measured -- and the source label does not change when it
    is, so the key would not notice. Keying on the figures themselves retires
    the stale answer instead of serving a latency derived from peaks that no
    longer hold.
    """
    material = "|".join(
        [
            canonical_id,
            case_set_hash(case_ids),
            arch,
            peak_source,
            roofs_hash(peak_flops, bandwidth),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def cache_path(key: str, project_root: str | Path | None = None) -> Path:
    """Location of one cached ceiling under the writable state root."""
    root = Path(project_root) if project_root is not None else default_project_root()
    return root / "roofline_ceiling" / f"{key}.json"


def publish(report: CeilingReport, output_dir: str | Path) -> Path:
    """Write the report and its human-readable companion; return the JSON path."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / REPORT_FILENAME
    atomic_write_text(json_path, json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    atomic_write_text(destination / DOCUMENT_FILENAME, render_document(report))
    return json_path


def store_in_cache(report: CeilingReport, key: str, project_root: str | Path | None = None) -> Path:
    """Cache one report so a later campaign on the same kernel does not re-derive it."""
    path = cache_path(key, project_root)
    atomic_write_text(path, json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def read_cached(key: str, project_root: str | Path | None = None) -> CeilingReport | None:
    """Return a cached report, or ``None`` when absent or unreadable."""
    path = cache_path(key, project_root)
    if not path.is_file():
        return None
    try:
        return load_report(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, CeilingContractError) as exc:
        log.warning("ignoring unreadable cached ceiling at %s: %s", path, exc)
        return None


def read_report(path: str | Path) -> CeilingReport:
    """Load a published report. Raises on anything unreadable."""
    return load_report(json.loads(Path(path).read_text(encoding="utf-8")))


def _format_ms(value: float) -> str:
    """Render a latency with enough digits to be useful at microsecond scale."""
    return f"{value:.6g}"


def render_document(report: CeilingReport) -> str:
    """Render the operator-facing analysis document.

    A framework-owned header -- the answer, the figures it was taken against,
    and anything the validator objected to -- followed verbatim by the analyst's
    own derivation. The header is what a reader checks first and the only part
    that cannot drift from the published JSON; the derivation is the only record
    of how the numbers were reached, so it is reproduced rather than summarized.
    """
    hardware = report.hardware
    lines: list[str] = [
        "# Performance ceiling",
        "",
        "## Conclusion",
        "",
        "Theoretical achievable latency per scored case. This is an optimistic lower bound under",
        "hardware limits and legal algorithm constraints; it does not claim an implementation",
        "reaching it exists.",
        "",
        "| Case | Ideal latency (ms) | Bound | Observed under profiler (ms) |",
        "|:--|--:|:--|--:|",
    ]
    for case in report.cases:
        observed = _format_ms(case.profiler_observed_ms) if case.profiler_observed_ms is not None else "n/a"
        lines.append(f"| `{case.case_id}` | {_format_ms(case.t_ideal_ms)} | {case.bound} | {observed} |")

    lines += [
        "",
        f"Confidence: **{report.confidence}**.",
        "",
        "## Hardware the estimate was taken against",
        "",
        f"- Architecture: `{hardware.arch}`",
        f"- Peak source: `{hardware.peak_source}` — {peak_source_meaning(hardware.peak_source)}",
    ]
    for tier, value in sorted(hardware.bandwidth.items()):
        lines.append(f"- Bandwidth ({tier}): {value / 1e12:.4g} TB/s")
    if hardware.dispatch_floor_s > 0:
        lines.append(f"- Dispatch floor: {hardware.dispatch_floor_s * 1e6:.4g} us")
    else:
        lines.append("- Dispatch floor: not measured")
    for path, value in sorted(hardware.peak_flops.items()):
        lines.append(f"- Peak ({path}): {value / 1e12:.4g} TFLOP/s")
    lines.append("")

    flagged = [(case.case_id, issue) for case in report.cases for issue in case.issues]
    if flagged:
        lines += ["## Findings", ""]
        lines += [f"- `{case_id}`: {issue}" for case_id, issue in flagged]
        lines.append("")

    if report.caveats:
        lines += ["## Caveats", ""]
        lines += [f"- {caveat}" for caveat in report.caveats]
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
            "| Case | Attainment | Ceiling (ms) | Measured (ms) | Bound | Speedup to ceiling |",
            "|:--|--:|--:|--:|:--|--:|",
        ]
        for entry in sorted(standing.cases, key=lambda c: c.attainment):
            lines.append(
                f"| `{entry.case_id}` | {entry.attainment * 100:.1f}% | {_format_ms(entry.t_ideal_ms)} "
                f"| {_format_ms(entry.t_current_ms)} | {entry.bound} | {entry.remaining_speedup:.2f}x |"
            )
        lines.append("")
    else:
        lines += [
            "| Case | Ceiling (ms) | Bound |",
            "|:--|--:|:--|",
        ]
        lines += [f"| `{case.case_id}` | {_format_ms(case.t_ideal_ms)} | {case.bound} |" for case in report.cases]
        lines.append("")

    if standing.excluded:
        lines.append("Cases with no attainment figure, and why:")
        lines += [f"- `{case_id}`: {reason}" for case_id, reason in sorted(standing.excluded.items())]
        lines.append("")

    if not report.hardware.is_measured:
        lines.append(
            "Peaks are vendor datasheet figures, not measured on any card, so every ceiling here is "
            "an absolute lower bound that no implementation reaches: attainment reads far lower than "
            "the kernel deserves. Compare cases against each other, not against the target."
        )
    lines.append(f"Ceiling confidence: {report.confidence}.")
    for caveat in report.caveats:
        lines.append(f"- {caveat}")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "DOCUMENT_FILENAME",
    "EVIDENCE_DIRNAME",
    "REPORT_FILENAME",
    "WORKSPACE_SUBDIR",
    "cache_key",
    "cache_path",
    "case_set_hash",
    "publish",
    "read_cached",
    "read_report",
    "render_document",
    "render_for_prompt",
    "roofs_hash",
    "store_in_cache",
]
