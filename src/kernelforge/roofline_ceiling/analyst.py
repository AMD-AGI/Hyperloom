# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The analyst session: it measures the box, derives the ceiling, writes both files.

Everything about the estimate is the analyst's -- the roofs, the minimum legal
work, how that work composes, and the two files it lands in. This module opens
the session, bounds where it may write, and reads the answer back.

Two things are checked: that ``performance_ceiling.json`` can be read as an
answer, and that ``performance_ceiling_analysis.md`` is there and not empty. The
derivation is required because it is the only place a wrong ceiling can be
caught, so a ceiling without one is not published. A file that fails either
check is handed back with the reason; a pair that passes is taken as given.
Nothing re-derives a latency, because a framework that could would be asserting
a work model this design already found too narrow for real operators.

The session needs a shell -- reaching a profiler on an arbitrary image means
installing packages, and that is open-ended work code cannot enumerate -- so
the workspace has to be defended rather than trusted. It is defended twice.

The workspace guard is the real line: granting shell makes it active instead of
skipped. It protects the measurement surface -- the kernel sources and driver by
path, the configuration, harnesses, references and tests by the shared protected
names -- and rolls the tree back and rejects the session on a change to any of
it. A change to any other tracked file it lets through and reports, and this
module fails the estimate on that report, because the analyst may only read.
Build products of running the kernel are left alone: they land in ignored files
or in the toolchain's own named droppings. Any other new file in the workspace
is a violation too, so an analyst writing its answer under the workspace would
have its answer deleted on the way out.

So the analyst writes into a scratch directory outside the workspace, and this
module moves the two deliverables into place afterwards. The hook is the second
line, refusing the editing tools anywhere but that scratch directory, which
turns a wrong path into a message the analyst can act on rather than a rollback
it never sees.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
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
from kernelforge.llm.workspace_policy import TOOL_OWNED_UNTRACKED_GLOBS
from kernelforge.roofline_ceiling.contract import CeilingContractError, CeilingReport
from kernelforge.roofline_ceiling.evidence import OBSERVED_CAMPAIGN, EvidenceBundle
from kernelforge.roofline_ceiling.report import (
    DOCUMENT_FILENAME,
    EVIDENCE_DIRNAME,
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

#: One repair round. The failure a repair fixes is a missing or unreadable
#: file, and an analyst that cannot write a usable one twice will not write one
#: on the third ask -- it will spend budget agreeing with the error message.
MAX_REPAIR_ROUNDS = 1

#: Tools that put bytes on disk. A shell can too, which is why the workspace
#: guard restores everything outside the writable set rather than this hook
#: being the only line.
_WRITE_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})


class CeilingAnalysisError(RuntimeError):
    """Raised when no readable ceiling file with a derivation beside it could be obtained."""


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
                else "latency from one run of the performance command, with no profiler attached; "
                "a single sample per case, so noisier than a median but not inflated"
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
    protected_paths: Sequence[str],
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
        # rejects and rolls back a change to the measurement surface -- the
        # kernel sources and driver exactly, the configuration, harnesses,
        # references and tests by the shared protected names. Not ``["*"]``:
        # the guard snapshots every protected ignored file byte for byte and
        # rejects any new one, and the ignored tree is where running the kernel
        # compiles to, so that glob reads a multi-GiB JIT cache into memory and
        # rejects the session for the build products of its own measurement.
        hooks=_writable_only_within(writable_dirs),
        protected_paths=[path for path in protected_paths if path],
        ignored_untracked_globs=list(TOOL_OWNED_UNTRACKED_GLOBS),
        additional_directories=list(writable_dirs),
        # The kernel under analysis is routinely a dirty checkout mid-campaign,
        # and an estimator has no business demanding a clean tree.
        allow_dirty_baseline=True,
    )


async def _ask(backend: Any, spec: AgentRunSpec) -> None:
    """Run one session, refusing it if it left a workspace file changed.

    The guard rolls back the measurement surface itself. Any other tracked
    file it lets through, and reports as the session's file changes; the
    analyst may only read the workspace, so a change there fails the estimate
    rather than being taken as part of it.
    """
    result = await asyncio.wait_for(
        backend.run(spec),
        timeout=watchdog_timeout_sec(spec.timeout_sec or 0),
    )
    changed = list(getattr(result, "file_changes", None) or [])
    if changed:
        raise CeilingAnalysisError(
            "the analyst session changed workspace files it may only read, and they were left as it changed "
            "them: " + ", ".join(changed)
        )


def _derivation_problem(document_path: Path) -> str:
    """Why the derivation beside the ceiling cannot be published, or ``""`` when it can."""
    try:
        text = document_path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"{document_path} was not written ({exc})"
    except ValueError as exc:
        # A session cut at its deadline mid-write can leave a truncated multi-byte sequence behind.
        return f"{document_path} is not valid UTF-8 text ({exc})"
    return "" if text.strip() else f"{document_path} is empty"


def _repair_prompt(report_path: Path, report_problem: str, document_path: Path, document_problem: str) -> str:
    """Ask for exactly the files that cannot be used, naming what is wrong with each."""
    asks: list[str] = []
    if report_problem:
        asks.append(
            f"{report_path} could not be read as a ceiling: {report_problem}\n\n"
            "Rewrite that file. It must be JSON with exactly two keys: 'cases', an object mapping "
            "every scored case id to its ideal latency in milliseconds as a finite positive number, "
            "and 'mean_ideal_ms', the equal-weight arithmetic mean of those latencies."
        )
    if document_problem:
        asks.append(
            f"{document_problem}. Write the derivation there, in the structure your role document "
            "prescribes: nothing recomputes the latencies, so without it the ceiling cannot be checked "
            "and is not published."
        )
    elif report_problem:
        asks.append(f"Leave {DOCUMENT_FILENAME} in place unless the derivation changes too.")
    return "\n\n".join(asks)


def _move_into_place(scratch: Path, destination: Path) -> None:
    """Put what the analyst wrote where the rest of the system looks for it.

    The two deliverables land in ``destination``; everything else the session
    produced -- profiler output, benchmark scripts, logs -- lands beneath it as
    the record of how the roofs were established.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for name in (REPORT_FILENAME, DOCUMENT_FILENAME):
        shutil.copy2(scratch / name, destination / name)

    workings = destination / EVIDENCE_DIRNAME / "analyst"
    if workings.exists():
        shutil.rmtree(workings, ignore_errors=True)
    shutil.copytree(
        scratch,
        workings,
        ignore=shutil.ignore_patterns(REPORT_FILENAME, DOCUMENT_FILENAME),
        dirs_exist_ok=True,
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
    """Run the analyst until it leaves a readable ceiling and its derivation, or give up."""
    destination = Path(output_dir)
    system_prompt = load_role(project_root)
    scratch = Path(tempfile.mkdtemp(prefix="forge_ceiling_"))
    report_path = scratch / REPORT_FILENAME
    document_path = scratch / DOCUMENT_FILENAME

    try:
        attempt_prompt = build_request(
            kernel_files=kernel_files,
            driver_script=driver_script,
            performance_command=performance_command,
            case_ids=case_ids,
            case_params=case_params,
            evidence=evidence,
            output_dir=str(scratch),
        )

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
                    writable_dirs=[str(scratch)],
                    protected_paths=[*kernel_files, driver_script],
                    turns=turns,
                ),
            )
            report_problem = ""
            try:
                report = read_report(report_path)
            except (OSError, ValueError, CeilingContractError) as exc:
                report_problem = str(exc) if not isinstance(exc, OSError) else f"it was not written ({exc})"
            document_problem = _derivation_problem(document_path)
            if not report_problem and not document_problem:
                _move_into_place(scratch, destination)
                return report

            last_problem = "; ".join(
                problem
                for problem in (
                    f"{REPORT_FILENAME}: {report_problem}" if report_problem else "",
                    document_problem,
                )
                if problem
            )
            log.warning("ceiling attempt %d left no usable answer: %s", attempt + 1, last_problem)
            if attempt >= MAX_REPAIR_ROUNDS:
                break
            attempt_prompt = _repair_prompt(report_path, report_problem, document_path, document_problem)

        raise CeilingAnalysisError(f"no usable ceiling after {MAX_REPAIR_ROUNDS + 1} attempts: {last_problem}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


__all__ = [
    "DEFAULT_ANALYST_TURNS",
    "MAX_REPAIR_ROUNDS",
    "ROLE_FILENAME",
    "CeilingAnalysisError",
    "build_request",
    "load_role",
    "run_ceiling_analysis",
]
