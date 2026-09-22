# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One ceiling estimate, end to end, for every caller that wants one.

Collect the evidence, run the analyst, publish the report. The CLI wraps this
and so does the optimization loop, because two orchestrations of the same three
steps would drift: one would gain the cache lookup the other lacked, or keep
labelling peaks a source the other had already stopped trusting.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernelforge.roofline_ceiling.analyst import run_ceiling_analysis
from kernelforge.roofline_ceiling.contract import CeilingReport
from kernelforge.roofline_ceiling.evidence import collect_evidence
from kernelforge.roofline_ceiling.report import (
    EVIDENCE_DIRNAME,
    WORKSPACE_SUBDIR,
    cache_key,
    publish,
    read_cached,
    store_in_cache,
)

log = logging.getLogger("kernelforge.roofline_ceiling")

SOURCE_CACHE = "cache"
SOURCE_ANALYST = "analyst"


class NoScoredCasesError(RuntimeError):
    """Raised when the driver named no scored case to produce a ceiling for."""


@dataclass(frozen=True)
class CeilingOutcome:
    """A published ceiling and how it was arrived at."""

    report: CeilingReport
    report_path: Path
    source: str
    scored_case_ids: tuple[str, ...]
    notes: tuple[str, ...] = field(default=())


def canonical_id_for(operator: str, arch: str) -> str:
    """The cache identity of one operator's ceiling on one architecture."""
    return f"roofline-ceiling:{operator}:{arch or 'unknown'}"


async def estimate_ceiling(
    backend: Any,
    *,
    workspace: str | Path,
    performance_command: Sequence[str],
    kernel_files: Sequence[str],
    driver_script: str = "",
    case_params: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    op_name: str = "",
    arch: str = "",
    use_cache: bool = True,
    known_case_ids: Sequence[str] | None = None,
    known_case_ms: Mapping[str, float] | None = None,
    agent_model: str = "",
    agent_timeout_sec: int = 3600,
    run_timeout_sec: float = 1800.0,
    project_root: str | Path | None = None,
) -> CeilingOutcome:
    """Estimate, publish and return the per-shape ceiling for one kernel.

    ``known_case_ids`` / ``known_case_ms`` skip the case-discovery run for a
    caller that has already benched the kernel. Raises
    :class:`NoScoredCasesError` when neither the caller nor the driver names a
    scored case, and :class:`~kernelforge.roofline_ceiling.analyst.CeilingAnalysisError`
    when the analyst cannot produce a report that satisfies the contract.
    """
    root = Path(workspace)
    destination = Path(output_dir) if output_dir is not None else root / WORKSPACE_SUBDIR
    artifacts = destination / EVIDENCE_DIRNAME

    evidence, scored_cases = collect_evidence(
        performance_command=performance_command,
        workdir=root,
        artifacts_dir=artifacts,
        arch=arch,
        run_timeout_sec=run_timeout_sec,
        known_case_ids=known_case_ids,
        known_case_ms=dict(known_case_ms or {}),
    )
    if not scored_cases:
        raise NoScoredCasesError(
            "the performance command emitted no scored 'case_ms:' lines, so there are no shapes to "
            "produce a ceiling for; see " + str(artifacts / "performance_run.log")
        )

    operator = op_name.strip() or root.name
    canonical_id = canonical_id_for(operator, evidence.hardware.arch)
    key = cache_key(
        canonical_id=canonical_id,
        case_ids=scored_cases,
        arch=evidence.hardware.arch,
        peak_source=evidence.hardware.peak_source,
        peak_flops=evidence.hardware.peak_flops,
        bandwidth=evidence.hardware.bandwidth,
    )

    if use_cache:
        cached = read_cached(key, project_root)
        if cached is not None:
            return CeilingOutcome(
                report=cached,
                report_path=publish(cached, destination),
                source=SOURCE_CACHE,
                scored_case_ids=tuple(scored_cases),
                notes=evidence.notes,
            )

    report = await run_ceiling_analysis(
        backend,
        canonical_id=canonical_id,
        workdir=str(root),
        kernel_files=kernel_files,
        driver_script=driver_script,
        performance_command=performance_command,
        case_ids=scored_cases,
        case_params=dict(case_params or {}),
        evidence=evidence,
        model=agent_model,
        timeout_sec=agent_timeout_sec,
        project_root=project_root,
    )

    report_path = publish(report, destination)
    if use_cache:
        store_in_cache(report, key, project_root)
    return CeilingOutcome(
        report=report,
        report_path=report_path,
        source=SOURCE_ANALYST,
        scored_case_ids=tuple(scored_cases),
        notes=evidence.notes,
    )


__all__ = [
    "SOURCE_ANALYST",
    "SOURCE_CACHE",
    "CeilingOutcome",
    "NoScoredCasesError",
    "canonical_id_for",
    "estimate_ceiling",
]
