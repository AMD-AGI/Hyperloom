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

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kernelforge.agent_backends.base import AgentProviderError
from kernelforge.roofline_ceiling.analyst import (
    AnalystSession,
    CeilingAnalysisError,
    CeilingTask,
    run_ceiling_analysis,
)
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


#: The failures an estimate can end in on a correctly wired call: no scored
#: case, no readable answer or no derivation beside it, an agent provider that is unavailable or whose
#: session failed (the workspace guard rejecting it included), a session past
#: its watchdog, and the filesystem. A caller that degrades on these must let
#: anything else raise, because anything else is a bug in the call.
ESTIMATE_ERRORS: tuple[type[Exception], ...] = (
    NoScoredCasesError,
    CeilingAnalysisError,
    AgentProviderError,
    asyncio.TimeoutError,
    OSError,
)


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
    task: CeilingTask,
    session: AnalystSession = AnalystSession(),
    output_dir: str | Path | None = None,
) -> CeilingOutcome:
    """Estimate and return the per-shape ceiling for one kernel.

    ``task.known`` skips the case-discovery run for a caller that has already
    benched the kernel. Raises :class:`NoScoredCasesError` when neither the
    caller nor the driver names a scored case, and
    :class:`~kernelforge.roofline_ceiling.analyst.CeilingAnalysisError` when the
    analyst cannot produce a report that satisfies the contract.
    """
    root = Path(task.run.workdir)
    destination = Path(output_dir) if output_dir is not None else root / WORKSPACE_SUBDIR
    artifacts = destination / EVIDENCE_DIRNAME

    evidence = await collect_evidence(task.run, artifacts, task.arch, task.known)
    if not evidence.scored_case_ids:
        raise NoScoredCasesError(
            "the performance command emitted no scored 'case_ms:' lines, so there are no shapes to "
            "produce a ceiling for; see " + str(artifacts / "performance_run.log")
        )

    report = await run_ceiling_analysis(backend, task, evidence, destination, session)

    return CeilingOutcome(
        report=report,
        report_path=destination / REPORT_FILENAME,
        source=SOURCE_ANALYST,
        scored_case_ids=evidence.scored_case_ids,
        notes=evidence.notes,
    )


__all__ = [
    "ESTIMATE_ERRORS",
    "SOURCE_ANALYST",
    "CeilingOutcome",
    "NoScoredCasesError",
    "estimate_ceiling",
]
