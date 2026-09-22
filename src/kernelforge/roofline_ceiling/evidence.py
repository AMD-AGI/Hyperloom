# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic inputs to the ceiling: the machine's roofs, the case set, the trace.

No agent, and no shared state with the optimization loop. This settles
everything the analyst cannot answer by reading source -- how fast the machine
is, which shapes are scored, and what the kernel really dispatches -- before
the analyst session starts, so the numbers it reasons over are the numbers the
framework will divide by.

The roofs are looked up, never measured here. They are a property of the
machine, not of the day: a shipped device profile keyed on architecture, device
name and partition mode, and the vendor datasheet behind it when no profile
covers the machine. Measuring at run time bought a figure that moved with the
box's mood and needed a profiler installed; committing it buys one that has
been reviewed and that two campaigns a month apart both divide by. See
``docs/kernelforge/reference/device-profiles.md`` for how a machine gets a
profile.

``peak_source`` is a required field rather than a nicety. A ceiling against a
measured profile and one against a datasheet differ by roughly a factor of two,
so a consumer that cannot tell which it is holding has no number at all.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from kernelforge.roofline_ceiling.contract import Hardware
from kernelforge.roofline_ceiling.device_profile import (
    DeviceIdentity,
    describe_device,
    load_reference,
)
from kernelforge.roofline_ceiling.specs import (
    PEAK_SOURCE_DATASHEET,
    PEAK_SOURCE_REFERENCE,
    arch_spec,
    supported_arches,
)
from kernelforge.fusion.gpu_arch import canon_arch, detect_arch

log = logging.getLogger("kernelforge.roofline_ceiling")

#: ``observed_ms`` was timed by this module, with a profiler attached.
OBSERVED_PROFILED = "profiled"
#: ``observed_ms`` was handed in by a caller that had already measured it
#: without a profiler -- a campaign's own per-case medians.
OBSERVED_CAMPAIGN = "campaign_median"


@dataclass(frozen=True)
class EvidenceBundle:
    """Everything the deterministic side settled for one ceiling run."""

    hardware: Hardware
    artifacts_dir: Path
    #: ``case_id -> latency`` the kernel was seen reaching, a sanity reference
    #: for the analyst and the divisor of the report's overshoot check.
    observed_ms: dict[str, float] = field(default_factory=dict)
    #: Which clock produced ``observed_ms``. A profiled figure is inflated by
    #: the profiler and a campaign median is not, and the analyst is told which
    #: it holds rather than left to assume.
    observed_origin: str = OBSERVED_PROFILED
    notes: tuple[str, ...] = ()

    def artifact_paths(self) -> list[str]:
        """Files the analyst may read, relative to the artifacts directory."""
        if not self.artifacts_dir.is_dir():
            return []
        return sorted(
            str(path.relative_to(self.artifacts_dir)) for path in self.artifacts_dir.rglob("*") if path.is_file()
        )


def _run(
    argv: Sequence[str],
    *,
    cwd: str | Path,
    timeout_sec: float,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> tuple[int, str]:
    """Run one subprocess, returning ``(returncode, combined output)``.

    A missing binary, a crash and a timeout are all ordinary outcomes here: the
    caller decides what to do about them, so none of them raise.
    """
    merged = {**os.environ, **(env or {})}
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd),
            env=merged,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = f"timed out after {timeout_sec:.0f}s: {exc}"
        code = -1
    except OSError as exc:
        output = f"could not execute {argv[0]!r}: {exc}"
        code = -1
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
    return code, output


def capture_kernel_trace(
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
        code, _output = _run(
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


def resolve_hardware(
    *,
    arch: str = "",
    identity: DeviceIdentity | None = None,
    provenance: dict[str, Any] | None = None,
    allow_reference: bool = True,
) -> Hardware:
    """Look up the roofs for this machine, best available source first.

    A committed device profile for this exact configuration, then the vendor
    datasheet. Sources are never blended: in a record half measured and half
    datasheet one term would be roughly twice as generous as the next and no
    field could say which, so a profile that does not cover the machine yields
    to the datasheet whole rather than filling its gaps.
    """
    resolved_arch = canon_arch(arch) or detect_arch() or canon_arch(arch)
    notes = dict(provenance or {})
    notes["arch_resolved_from"] = "caller" if canon_arch(arch) else "rocminfo"

    device = identity or describe_device(resolved_arch)
    if allow_reference:
        reference = load_reference(device)
        if reference is not None:
            notes["device_profile"] = {
                "origin": reference.origin,
                "measurement": reference.measurement,
                "source_by_figure": reference.source_by_figure,
                "notes": list(reference.notes),
                "matched_device": {
                    "arch": device.arch,
                    "device_name": device.device_name,
                    "compute_partition": device.compute_partition,
                    "memory_partition": device.memory_partition,
                },
            }
            return Hardware(
                arch=resolved_arch or device.arch,
                peak_flops=dict(reference.peak_flops),
                bandwidth=dict(reference.bandwidth),
                peak_source=PEAK_SOURCE_REFERENCE,
                dispatch_floor_s=float(reference.dispatch_floor_s),
                provenance=notes,
            )

    spec = arch_spec(resolved_arch)
    if spec is None:
        raise ValueError(
            f"no device profile and no datasheet peaks for arch {resolved_arch or '<undetected>'}; "
            "pass --arch with one of: " + ", ".join(sorted(supported_arches()))
        )
    notes.setdefault("datasheet_source", spec.source)
    notes.setdefault(
        "no_device_profile_for",
        {
            "arch": device.arch,
            "device_name": device.device_name,
            "compute_partition": device.compute_partition,
            "memory_partition": device.memory_partition,
        },
    )
    return Hardware(
        arch=spec.arch,
        peak_flops=dict(spec.peak_flops),
        bandwidth=spec.bandwidth(),
        peak_source=PEAK_SOURCE_DATASHEET,
        # The datasheet quotes no launch cost. Zero leaves every latency term
        # out of the estimate, which the report states as a caveat rather than
        # absorbing silently: a latency-bound shape whose dispatch floor is
        # missing reads as memory-bound and nearly free.
        dispatch_floor_s=0.0,
        provenance=notes,
    )


def discover_scored_cases(
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

    code, output = _run(
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


def collect_evidence(
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

    Returns ``(bundle, scored_case_ids)``. Each step degrades independently and
    records why, so a machine no device profile covers still produces a labelled
    answer instead of no answer or, worse, an unlabelled one.

    ``known_case_ids`` and ``known_case_ms`` let a caller that has already
    benched the kernel skip the discovery run. A campaign has: it measured its
    pristine anchor over repeated runs before this is called, which is both a
    better clock than one profiled pass and one fewer driver execution to pay
    for.
    """
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    identity = describe_device(arch)

    if known_case_ids:
        scored = [str(case_id) for case_id in dict.fromkeys(known_case_ids)]
        observed = {
            case_id: float(value) for case_id, value in (known_case_ms or {}).items() if case_id in set(scored)
        }
        observed_origin = OBSERVED_CAMPAIGN
        notes = ["scored case set and latencies supplied by the caller; no discovery run was made"]
    else:
        scored, observed, notes = discover_scored_cases(
            command=performance_command,
            workdir=workdir,
            artifacts_dir=artifacts,
            timeout_sec=run_timeout_sec,
            env=env,
        )
        observed_origin = OBSERVED_PROFILED

    trace_provenance = capture_kernel_trace(
        command=performance_command,
        workdir=workdir,
        artifacts_dir=artifacts,
        timeout_sec=run_timeout_sec,
        env=env,
    )
    if not trace_provenance.get("captured"):
        notes.append("kernel trace unavailable: " + str(trace_provenance.get("detail") or "unknown"))

    hardware = resolve_hardware(
        arch=arch,
        identity=identity,
        provenance={"kernel_trace": trace_provenance},
    )
    if hardware.peak_source == PEAK_SOURCE_REFERENCE:
        origin = (hardware.provenance.get("device_profile") or {}).get("origin", "a device profile")
        notes.append(f"roofs read from {origin}")
    else:
        notes.append(
            f"no device profile covers this machine ({identity.slug()}), so the roofs are vendor datasheet "
            "figures: every ceiling is an absolute lower bound and attainment reads far below what the "
            "kernel deserves. Add a profile for this configuration to get a usable target."
        )

    return (
        EvidenceBundle(
            hardware=hardware,
            artifacts_dir=artifacts,
            observed_ms=observed,
            observed_origin=observed_origin,
            notes=tuple(notes),
        ),
        scored,
    )


__all__ = [
    "OBSERVED_CAMPAIGN",
    "OBSERVED_PROFILED",
    "EvidenceBundle",
    "capture_kernel_trace",
    "collect_evidence",
    "discover_scored_cases",
    "resolve_hardware",
]
