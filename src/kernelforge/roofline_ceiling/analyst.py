# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The analyst session: it measures the box, derives the ceiling, writes both files.

Everything about the estimate is the analyst's -- the roofs, the minimum legal
work, how that work composes, and the two files it lands in. This module opens
the session, bounds where it may write, and reads the answer back.

The only thing checked is whether ``performance_ceiling.json`` can be read as
an answer at all. A file that cannot is handed back with the reason; a file
that can is taken as given. Nothing re-derives a latency, because a framework
that could would be asserting a work model this design already found too narrow
for real operators.

Bounding the write is a hook rather than a sandbox flag because the session
needs a shell -- reaching a profiler on an arbitrary image means installing
packages, and that is open-ended work code cannot enumerate. The hook denies
edits outside the output and evidence directories, and the workspace guard
snapshots and restores everything else, so the kernel under optimization comes
out of the session as it went in.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from kernelforge.agent_backends.base import (
    AgentHook,
    AgentHooks,
    AgentRunSpec,
    AgentToolPolicy,
    watchdog_timeout_sec,
)
from kernelforge.roofline_ceiling.contract import CeilingContractError, CeilingReport
from kernelforge.roofline_ceiling.evidence import OBSERVED_CAMPAIGN, EvidenceBundle
from kernelforge.roofline_ceiling.report import (
    DOCUMENT_FILENAME,
    REPORT_FILENAME,
    read_report,
)
from kernelforge.resources import resource_path

log = logging.getLogger("kernelforge.roofline_ceiling")

ROLE_FILENAME = "ceiling_analyst.md"

#: The analyst reads source, runs a profiler and writes two files. Measuring
#: costs turns and so does installing a tool, so this is generous; the session
#: timeout bounds it either way.
DEFAULT_ANALYST_TURNS = 120

#: One repair round. The failure a repair fixes is an unreadable file, and an
#: analyst that cannot write a readable one twice will not write one on the
#: third ask -- it will spend budget agreeing with the error message.
MAX_REPAIR_ROUNDS = 1

#: Tools that put bytes on disk. A shell can too, which is why the workspace
#: guard restores everything outside the writable set rather than this hook
#: being the only line.
_WRITE_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})


class CeilingAnalysisError(RuntimeError):
    """Raised when no readable ceiling file could be obtained."""


def load_role(project_root: str | Path | None = None) -> str:
    """Read the analyst role document shipped with the package."""
    return (resource_path("roofline_ceiling", project_root) / ROLE_FILENAME).read_text(encoding="utf-8")


def build_request(
    *,
    kernel_files: Sequence[str],
    driver_script: str,
    performance_command: Sequence[str],
    case_ids: Sequence[str],
    case_params: Mapping[str, Any],
    evidence: EvidenceBundle,
    output_dir: str,
) -> str:
    """Build the analyst's request payload.

    It states the machine rather than its roofs, because establishing those is
    the analyst's first step, and it states the case set exactly, because that
    is the one thing the answer has to line up with: a ceiling for shapes the
    driver does not score cannot be divided into anything.
    """
    device = evidence.identity
    payload: dict[str, Any] = {
        "task": (
            "Measure this machine's roofs, estimate the theoretical achievable latency of every "
            f"scored case of this kernel against them, and write both {REPORT_FILENAME} and "
            f"{DOCUMENT_FILENAME} into output_dir. Return a one-paragraph summary; the files are "
            "the deliverable."
        ),
        "output_dir": output_dir,
        "output_files": {
            REPORT_FILENAME: (
                "JSON with exactly two keys. 'cases': an object mapping every scored case id to "
                "its ideal latency in milliseconds, a finite positive number. 'mean_ideal_ms': "
                "the equal-weight arithmetic mean of those latencies. Nothing else."
            ),
            DOCUMENT_FILENAME: (
                "Markdown, in the structure your role document prescribes. Nothing recomputes the "
                "latencies, so this is the only record of how each was reached: carry the roofs "
                "you measured and how, the formulas, the per-case arithmetic, what bounds each "
                "shape, and every assumption."
            ),
        },
        "kernel_files": list(kernel_files),
        "driver_script": driver_script,
        "performance_command": list(performance_command),
        "scored_case_ids": list(case_ids),
        "case_parameters": dict(case_params),
        "machine": device.describe(),
        "evidence_dir": str(evidence.artifacts_dir),
        "evidence_files": evidence.artifact_paths(),
        "observed_ms": dict(evidence.observed_ms),
        "observed_ms_meaning": (
            (
                "latency measured by the campaign over repeated runs, with no profiler attached"
                if evidence.observed_origin == OBSERVED_CAMPAIGN
                else "latency seen while profiling, so inflated by the profiler's own overhead"
            )
            + ". A sanity reference only: no ceiling may be back-solved from it, and none may "
            "exceed it."
        ),
    }
    if evidence.notes:
        payload["evidence_notes"] = list(evidence.notes)
    return json.dumps(payload, indent=2, sort_keys=True)


def _writable_only_within(directories: Sequence[str]) -> AgentHooks:
    """Deny edits to anything outside ``directories``.

    The session is given a shell so it can install and run a profiler, which
    means it could in principle touch the kernel it is estimating for. This
    stops the file-editing tools at the boundary and names the boundary in the
    refusal, so the analyst redirects rather than retries blindly.
    """
    roots = [Path(directory).resolve() for directory in directories if str(directory).strip()]

    async def _bound_writes(input_data: dict, tool_use_id: Any, context: Any) -> dict:
        if str(input_data.get("tool_name") or "") not in _WRITE_TOOLS:
            return {}
        target = str((input_data.get("tool_input") or {}).get("file_path") or "").strip()
        if target:
            resolved = Path(target).resolve()
            if any(resolved == root or root in resolved.parents for root in roots):
                return {}
        allowed = ", ".join(str(root) for root in roots)
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"A ceiling run may only write under: {allowed}. The kernel under analysis and "
                    f"everything else in the workspace is read-only. Write {target or 'that file'} "
                    "into the output directory instead."
                ),
            }
        }

    return AgentHooks(pre_tool_use=[AgentHook(matcher="", callback=_bound_writes, timeout_sec=5)])


def _spec(
    *,
    system_prompt: str,
    user_prompt: str,
    workdir: str,
    model: str,
    timeout_sec: int,
    writable_dirs: Sequence[str],
    turns: int,
) -> AgentRunSpec:
    """One analyst session: reads the workspace, writes only where it is told."""
    return AgentRunSpec(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        role="ceiling analyst",
        cwd=workdir,
        model=model,
        writable=True,
        timeout_sec=max(1, int(timeout_sec)),
        tool_policy=AgentToolPolicy(
            read=True,
            search=True,
            write=True,
            shell=True,
            max_turns=turns,
        ),
        # Two lines, because a shell gets past the first: the hook refuses the
        # editing tools outside the output directories, and the workspace guard
        # snapshots every workspace file and restores it after.
        hooks=_writable_only_within(writable_dirs),
        protected_globs=["*"],
        additional_directories=list(writable_dirs),
        # The kernel under analysis is routinely a dirty checkout mid-campaign,
        # and an estimator has no business demanding a clean tree.
        allow_dirty_baseline=True,
    )


async def _ask(backend: Any, spec: AgentRunSpec) -> str:
    """Run one session and return its final text."""
    result = await asyncio.wait_for(
        backend.run(spec),
        timeout=watchdog_timeout_sec(spec.timeout_sec or 0),
    )
    return str(getattr(result, "text", "") or "").strip()


def _repair_prompt(report_path: Path, problem: str) -> str:
    """Ask for the one file to be rewritten, naming what could not be read."""
    return (
        f"{report_path} could not be read as a ceiling: {problem}\n\n"
        f"Rewrite that file. It must be JSON with exactly two keys: 'cases', an object mapping "
        f"every scored case id to its ideal latency in milliseconds as a finite positive number, "
        f"and 'mean_ideal_ms', the equal-weight arithmetic mean of those latencies. Leave "
        f"{DOCUMENT_FILENAME} in place unless the derivation changes too."
    )


async def run_ceiling_analysis(
    backend: Any,
    *,
    workdir: str,
    output_dir: str | Path,
    kernel_files: Sequence[str],
    driver_script: str,
    performance_command: Sequence[str],
    case_ids: Sequence[str],
    case_params: Mapping[str, Any],
    evidence: EvidenceBundle,
    model: str = "",
    timeout_sec: int = 3600,
    turns: int = DEFAULT_ANALYST_TURNS,
    project_root: str | Path | None = None,
) -> CeilingReport:
    """Run the analyst until it leaves a readable ceiling file, or give up."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / REPORT_FILENAME

    system_prompt = load_role(project_root)
    attempt_prompt = build_request(
        kernel_files=kernel_files,
        driver_script=driver_script,
        performance_command=performance_command,
        case_ids=case_ids,
        case_params=case_params,
        evidence=evidence,
        output_dir=str(destination),
    )
    writable_dirs = [str(destination), str(evidence.artifacts_dir)]

    last_problem = ""
    for attempt in range(MAX_REPAIR_ROUNDS + 1):
        await _ask(
            backend,
            _spec(
                system_prompt=system_prompt,
                user_prompt=attempt_prompt,
                workdir=workdir,
                model=model,
                timeout_sec=timeout_sec,
                writable_dirs=writable_dirs,
                turns=turns,
            ),
        )
        try:
            return read_report(report_path)
        except (OSError, ValueError, CeilingContractError) as exc:
            last_problem = str(exc) if not isinstance(exc, OSError) else f"it was not written ({exc})"
            log.warning("ceiling attempt %d left no readable %s: %s", attempt + 1, REPORT_FILENAME, last_problem)
            if attempt >= MAX_REPAIR_ROUNDS:
                break
            attempt_prompt = _repair_prompt(report_path, last_problem)

    raise CeilingAnalysisError(f"no readable {REPORT_FILENAME} after {MAX_REPAIR_ROUNDS + 1} attempts: {last_problem}")


__all__ = [
    "DEFAULT_ANALYST_TURNS",
    "MAX_REPAIR_ROUNDS",
    "ROLE_FILENAME",
    "CeilingAnalysisError",
    "build_request",
    "load_role",
    "run_ceiling_analysis",
]
