# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One ceiling estimate, end to end, for every caller that wants one.

Settle the evidence, run the analyst, hand back what it wrote. The CLI wraps this
and so does the optimization loop, because two orchestrations of the same three
steps would drift.

Nothing is cached. The analyst measures the machine's roofs during its session,
and a cached report would both freeze figures that are meant to be current and
write them to a file that outlives the run. Re-deriving costs a profiler pass
and one agent session per campaign, which is the price of the roofs being this
box's rather than a record of some earlier box's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernelforge.roofline_ceiling.analyst import run_ceiling_analysis
from kernelforge.roofline_ceiling.contract import CeilingReport
from kernelforge.roofline_ceiling.evidence import collect_evidence
from kernelforge.roofline_ceiling.report import (
    EVIDENCE_DIRNAME,
    REPORT_FILENAME,
    WORKSPACE_SUBDIR,
)

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


async def estimate_ceiling(
    backend: Any,
    *,
    workspace: str | Path,
    performance_command: Sequence[str],
    kernel_files: Sequence[str],
    driver_script: str = "",
    case_params: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    arch: str = "",
    known_case_ids: Sequence[str] | None = None,
    known_case_ms: Mapping[str, float] | None = None,
    agent_model: str = "",
    agent_timeout_sec: int = 3600,
    run_timeout_sec: float = 1800.0,
    project_root: str | Path | None = None,
) -> CeilingOutcome:
    """Estimate and return the per-shape ceiling for one kernel.

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

    report = await run_ceiling_analysis(
        backend,
        workdir=str(root),
        output_dir=destination,
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

    return CeilingOutcome(
        report=report,
        report_path=destination / REPORT_FILENAME,
        source=SOURCE_ANALYST,
        scored_case_ids=tuple(scored_cases),
        notes=evidence.notes,
    )


__all__ = [
    "SOURCE_ANALYST",
    "CeilingOutcome",
    "NoScoredCasesError",
    "estimate_ceiling",
]
