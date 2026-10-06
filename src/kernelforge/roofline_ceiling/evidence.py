# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic inputs to the ceiling: the case set, the machine's identity, the trace.

No agent here. This settles what must not vary between runs -- which shapes are
scored, what the kernel really dispatches, and which machine this is -- before
the analyst session starts.

The roofs are deliberately *not* settled here. The analyst measures them itself
during its session, because getting a profiler onto an arbitrary image is
open-ended work that code cannot enumerate, and because no measured figure is
then written to any artifact this module owns. Nothing checks the figures it
reports: the contract reads the shape of the report and no more, so the
derivation it publishes is the only place a misread profiler column is caught.

Whether the roofs were measured or recalled is recorded by the analyst in the
derivation it publishes. The gap between the two is not a fixed discount -- it
varies per instruction path -- so a reader who needs to know reads the
document rather than a field.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from kernelforge.mcp_server.tools._subprocess import communicate_process_group
from kernelforge.roofline_ceiling.device_profile import DeviceIdentity, describe_device

log = logging.getLogger("kernelforge.roofline_ceiling")

#: ``observed_ms`` came from one run of the performance command, made here
#: with no profiler attached: a single sample per case, not a median.
OBSERVED_SINGLE_RUN = "single_run"
#: ``observed_ms`` was handed in by a caller that had already measured it
#: without a profiler -- a campaign's own per-case medians.
OBSERVED_CAMPAIGN = "campaign_median"


@dataclass(frozen=True)
class EvidenceBundle:
    """Everything the deterministic side settled for one ceiling run."""

    #: Which machine this is, so the analyst measures and names the right one.
    identity: DeviceIdentity
    artifacts_dir: Path
    #: ``case_id -> latency`` the kernel was seen reaching: the analyst's
    #: sanity reference, which no ceiling may exceed. Nothing here enforces it.
    observed_ms: dict[str, float] = field(default_factory=dict)
    #: Which clock produced ``observed_ms``, so the analyst is told what it
    #: holds rather than left to assume.
    observed_origin: str = OBSERVED_SINGLE_RUN
    notes: tuple[str, ...] = ()

    def artifact_paths(self) -> list[str]:
        """Files the analyst may read, relative to the artifacts directory."""
        if not self.artifacts_dir.is_dir():
            return []
        return sorted(
            str(path.relative_to(self.artifacts_dir)) for path in self.artifacts_dir.rglob("*") if path.is_file()
        )


async def _run(
    argv: Sequence[str],
    *,
    cwd: str | Path,
    timeout_sec: float,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> tuple[int, str]:
    """Run one subprocess in its own process group, returning ``(returncode, combined output)``.

    The process started here is never the one doing the work -- it is a
    profiler wrapping the driver, or a shell running it -- so a timeout or a
    cancelled estimate kills the whole group. Killing the direct child alone
    would leave the driver holding the device, and every latency the campaign
    measured afterwards would be taken on a contended one.

    A missing binary, a crash and a timeout are all ordinary outcomes here: the
    caller decides what to do about them, so none of them raise.
    """
    merged = {**os.environ, **(env or {})}
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            env=merged,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        output = f"could not execute {argv[0]!r}: {exc}"
        code = -1
    else:
        try:
            stdout, stderr = await communicate_process_group(proc, timeout=timeout_sec)
        except asyncio.TimeoutError:
            output = f"timed out after {timeout_sec:.0f}s; its process group was killed"
            code = -1
        else:
            output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
            code = proc.returncode if proc.returncode is not None else -1
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
    return code, output


async def capture_kernel_trace(
    *,
    command: Sequence[str],
    workdir: str | Path,
    artifacts_dir: Path,
    timeout_sec: float = 1800.0,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Capture a kernel trace so the analyst can count real dispatches.

    The trace is what tells the analyst which selector branch a shape entered,
    how many kernels one call actually launches, and whether the timed region
    covers the whole semantic operation. Without it the serial stage
    decomposition is guesswork.
    """
    trace_dir = artifacts_dir / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)

    for tool, argv in (
        (
            "rocprofv3",
            ["rocprofv3", "--kernel-trace", "--stats", "--output-format", "csv", "-d", str(trace_dir), "--", *command],
        ),
        ("rocprof", ["rocprof", "--stats", "-o", str(trace_dir / "results.csv"), *command]),
    ):
        if not shutil.which(tool):
            continue
        code, _output = await _run(
            argv,
            cwd=workdir,
            timeout_sec=timeout_sec,
            env=env,
            log_path=artifacts_dir / f"{tool}.log",
        )
        produced = sorted(path for path in trace_dir.rglob("*") if path.is_file())
        if produced:
            return {
                "captured": True,
                "tool": tool,
                "exit_code": code,
                "dir": str(trace_dir),
                "files": [str(path.relative_to(artifacts_dir)) for path in produced],
            }
        log.warning("ceiling kernel trace: %s exited %s and wrote nothing", tool, code)

    return {"captured": False, "detail": "no kernel-trace profiler produced output; tried rocprofv3, rocprof"}


def resolve_identity(arch: str = "") -> DeviceIdentity:
    """Name the machine the ceiling is for, so the analyst measures the right one."""
    return describe_device(arch)


async def discover_scored_cases(
    *,
    command: Sequence[str],
    workdir: str | Path,
    artifacts_dir: Path,
    timeout_sec: float = 1800.0,
    env: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, float], list[str]]:
    """Run the performance command once and read its scored cases off the output.

    Returns ``(scored_case_ids, observed_ms, notes)``. The driver's ``case_ms:``
    lines are the authority on which cases exist -- not the ``shapes:`` list in
    the task configuration, which is a convenience that can fall out of step with
    what the driver actually times. Cases the driver tags ``unscored`` are
    correctness-only and get no ceiling: the objective never counts them.
    """
    from kernelforge.mcp_server.tools.bench import parse_case_timings

    code, output = await _run(
        command,
        cwd=workdir,
        timeout_sec=timeout_sec,
        env=env,
        log_path=artifacts_dir / "performance_run.log",
    )
    case_times, unscored, duplicates = parse_case_timings(output)

    notes: list[str] = []
    if code != 0:
        notes.append(f"performance command exited {code}; its case list may be incomplete")
    if duplicates:
        notes.append("driver emitted duplicate case timings for: " + ", ".join(sorted(duplicates)))

    excluded = set(unscored)
    scored = [case_id for case_id in case_times if case_id not in excluded]
    if excluded:
        notes.append("excluded correctness-only (unscored) case(s): " + ", ".join(sorted(excluded)))
    if not scored:
        notes.append("driver emitted no scored case_ms lines")
    return scored, {case_id: case_times[case_id] for case_id in scored}, notes


async def collect_evidence(
    *,
    performance_command: Sequence[str],
    workdir: str | Path,
    artifacts_dir: str | Path,
    arch: str = "",
    run_timeout_sec: float = 1800.0,
    env: dict[str, str] | None = None,
    known_case_ids: Sequence[str] | None = None,
    known_case_ms: dict[str, float] | None = None,
) -> tuple[EvidenceBundle, list[str]]:
    """Settle every deterministic input one ceiling run needs.

    Returns ``(bundle, scored_case_ids)``. The roofs are not among them: the
    analyst measures those itself and reports them back. What is settled here
    is what must not vary between runs -- the case set, the machine's identity,
    and the trace.

    ``known_case_ids`` and ``known_case_ms`` let a caller that has already
    benched the kernel skip the discovery run. A campaign has: it measured its
    pristine anchor over repeated runs before this is called, which is both a
    better clock than a single run and one fewer driver execution to pay for.
    """
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    identity = resolve_identity(arch)

    if known_case_ids:
        scored = [str(case_id) for case_id in dict.fromkeys(known_case_ids)]
        observed = {case_id: float(value) for case_id, value in (known_case_ms or {}).items() if case_id in set(scored)}
        observed_origin = OBSERVED_CAMPAIGN
        notes = ["scored case set and latencies supplied by the caller; no discovery run was made"]
    else:
        scored, observed, notes = await discover_scored_cases(
            command=performance_command,
            workdir=workdir,
            artifacts_dir=artifacts,
            timeout_sec=run_timeout_sec,
            env=env,
        )
        observed_origin = OBSERVED_SINGLE_RUN

    trace_provenance = await capture_kernel_trace(
        command=performance_command,
        workdir=workdir,
        artifacts_dir=artifacts,
        timeout_sec=run_timeout_sec,
        env=env,
    )
    if not trace_provenance.get("captured"):
        notes.append("kernel trace unavailable: " + str(trace_provenance.get("detail") or "unknown"))
    notes.append(f"machine identified as {identity.slug()}; the analyst measures its roofs itself")

    return (
        EvidenceBundle(
            identity=identity,
            artifacts_dir=artifacts,
            observed_ms=observed,
            observed_origin=observed_origin,
            notes=tuple(notes),
        ),
        scored,
    )


__all__ = [
    "OBSERVED_CAMPAIGN",
    "OBSERVED_SINGLE_RUN",
    "EvidenceBundle",
    "capture_kernel_trace",
    "collect_evidence",
    "discover_scored_cases",
    "resolve_identity",
]
