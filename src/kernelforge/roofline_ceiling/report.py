# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publishing, caching and rendering one ceiling report."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from kernelforge.roofline_ceiling.contract import CeilingContractError, CeilingReport, load_report
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


def cache_key(*, canonical_id: str, case_ids: Sequence[str], arch: str, peak_source: str) -> str:
    """Identity of one ceiling answer.

    The peak source is part of the key, not metadata on it. A datasheet ceiling
    and an empirical one are different answers to different questions, and
    serving the first from cache when the caller asked for the second is exactly
    the silent degrade this module exists to prevent.
    """
    material = "|".join([canonical_id, case_set_hash(case_ids), arch, peak_source])
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
    """Render the operator-facing analysis document."""
    hardware = report.hardware
    lines: list[str] = [
        "# Performance ceiling analysis",
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
        "## Hardware",
        "",
        f"- Architecture: `{hardware.arch}`",
        f"- Peak source: `{hardware.peak_source}`"
        + ("" if hardware.is_empirical else " — vendor datasheet, an absolute lower bound, not an achievable target"),
        f"- HBM bandwidth: {hardware.hbm_bw_bytes_per_s / 1e12:.4g} TB/s",
    ]
    if hardware.dispatch_floor_s > 0:
        lines.append(f"- Dispatch floor: {hardware.dispatch_floor_s * 1e6:.4g} us")
    else:
        lines.append("- Dispatch floor: not measured")
    lines.append("")

    if report.caveats:
        lines += ["## Caveats", ""]
        lines += [f"- {caveat}" for caveat in report.caveats]
        lines.append("")

    lines += [
        "## Derivation",
        "",
        "`t_stage = dispatch_count * dispatch_floor + extra_latency + max(flops / peak, bytes / bandwidth)`,",
        "summed over serial stages. Compute and memory overlap only within a stage.",
        "",
    ]

    for case in report.cases:
        lines += [f"### `{case.case_id}`", ""]
        if case.bound_note:
            lines += [case.bound_note, ""]
        lines += [
            "| Stage | FLOPs | Bytes | Path | Dispatches | t_compute (us) | t_memory (us) | t_latency (us) |",
            "|:--|--:|--:|:--|--:|--:|--:|--:|",
        ]
        timings = {timing.name: timing for timing in case.timings}
        for stage in case.stages:
            timing = timings.get(stage.name)
            lines.append(
                f"| {stage.name} | {stage.flops:.4g} | {stage.bytes_moved:.4g} | "
                f"`{stage.instruction_path}` | {stage.dispatch_count} | "
                f"{(timing.t_compute_s * 1e6 if timing else 0.0):.4g} | "
                f"{(timing.t_memory_s * 1e6 if timing else 0.0):.4g} | "
                f"{(timing.t_latency_s * 1e6 if timing else 0.0):.4g} |"
            )
        lines.append("")
        for stage in case.stages:
            if stage.formula_flops or stage.formula_bytes:
                lines.append(f"- `{stage.name}` FLOPs: `{stage.formula_flops or 'n/a'}`")
                lines.append(f"- `{stage.name}` bytes: `{stage.formula_bytes or 'n/a'}`")
            for assumption in stage.assumptions:
                lines.append(f"  - {assumption}")
        if case.issues:
            lines += ["", "Issues:"]
            lines += [f"- {issue}" for issue in case.issues]
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_for_prompt(report: CeilingReport, baseline_ms: Mapping[str, float] | None = None) -> str:
    """Render the advisory block a consumer injects into its planning prompt.

    Deliberately framed as advisory. A ceiling is derived, not measured, and a
    pessimistic one told to an implementer as fact is how a case with real
    headroom gets abandoned at "already at 95%".
    """
    if not report.cases:
        return ""

    baseline = dict(baseline_ms or {})
    lines = [
        "### Theoretical ceiling (advisory)",
        "",
        "Derived per case from a work model, not measured. It can be wrong, and a case reported",
        "near its ceiling is a reason to look harder at the model before believing there is no",
        "headroom left. It is not a gate: KEEP is decided by measurement, as always.",
        "",
    ]

    header = "| Case | Ideal (ms) | Bound |"
    divider = "|:--|--:|:--|"
    if baseline:
        header += " Current (ms) | Headroom |"
        divider += "--:|--:|"
    lines += [header, divider]

    for case in report.cases:
        row = f"| `{case.case_id}` | {_format_ms(case.t_ideal_ms)} | {case.bound} |"
        if baseline:
            current = baseline.get(case.case_id)
            if current and current > 0 and case.t_ideal_ms > 0:
                row += f" {_format_ms(current)} | {current / case.t_ideal_ms:.2f}x |"
            else:
                row += " n/a | n/a |"
        lines.append(row)

    lines.append("")
    if not report.hardware.is_empirical:
        lines.append(
            "Peaks are vendor datasheet figures, so these are absolute lower bounds: even an "
            "excellent kernel reads far from them. Compare cases against each other, not against 1.00x."
        )
    lines.append(f"Confidence: {report.confidence}.")
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
    "store_in_cache",
]
