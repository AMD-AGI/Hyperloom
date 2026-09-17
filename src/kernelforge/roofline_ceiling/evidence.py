# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Measured inputs to the ceiling: empirical roofs, dispatch floor, kernel trace.

Everything here is deterministic and self-contained -- no agent, and no shared
state with the optimization loop. It answers the three questions the analyst
cannot answer by reading source: how fast this box actually is, what one
unavoidable dispatch costs on it, and what the kernel really dispatches.

Every step degrades rather than fails, and every degrade is recorded. A ceiling
computed against a vendor datasheet and one computed against ``--roof-only``
microbenchmarks differ by roughly a factor of two, so a consumer that cannot
tell which it is holding has no number at all. That is why ``peak_source`` is a
required field rather than a nicety, and why nothing in this module falls back
quietly.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from kernelforge.roofline_ceiling.contract import Hardware
from kernelforge.roofline_ceiling.device_profile import (
    DeviceIdentity,
    DeviceProfile,
    describe_device,
    load_local,
    load_reference,
    store_local,
)
from kernelforge.roofline_ceiling.specs import (
    PEAK_SOURCE_DATASHEET,
    PEAK_SOURCE_EMPIRICAL,
    PEAK_SOURCE_REFERENCE,
    arch_spec,
)
from kernelforge.fusion.gpu_arch import canon_arch, detect_arch

log = logging.getLogger("kernelforge.roofline_ceiling")

#: ``roofline.csv`` reports GFLOP/s, GIOP/s and GB/s; the ceiling works in
#: FLOP/s and bytes/s. Verified against rocprofiler-compute's own reader
#: (``utils/roofline_calc.py``), which plots those columns directly against
#: metrics declared in GFLOP/s.
_ROOF_UNIT_SCALE = 1e9

#: Instruction path -> candidate ``roofline.csv`` columns, most specific first.
#:
#: There is more than one candidate because the column set is not stable across
#: rocprofiler-compute releases. The low-precision matrix roofs are the ones that
#: move: some builds report a single merged ``MFMA_FLOPs_F6F4``, matching the
#: ``v_mfma_*_f8f6f4`` opcode family, while others report ``MFMAF4Flops`` and
#: ``MFMAF6Flops`` separately. Pinning either spelling alone silently leaves
#: every FP4/FP6/MXFP kernel without a compute roof on half the toolchains --
#: which is most of the workloads this module exists for.
#:
#: The scaled MXFP paths share their unscaled twin's column because they share
#: the pipeline: gfx950 runs ``v_mfma_scale_*_f8f6f4`` on the same matrix cores,
#: at the rate set by the widest operand.
_ROOF_COLUMNS_BY_PATH: dict[str, tuple[str, ...]] = {
    "bf16_mfma": ("MFMABF16Flops",),
    "fp16_mfma": ("MFMAF16Flops",),
    "fp8_mfma": ("MFMAF8Flops",),
    "mxfp8_scaled_mfma": ("MFMAF8Flops",),
    "fp6_mfma": ("MFMAF6Flops", "MFMA_FLOPs_F6F4"),
    "mxfp6_scaled_mfma": ("MFMAF6Flops", "MFMA_FLOPs_F6F4"),
    "fp4_mfma": ("MFMAF4Flops", "MFMA_FLOPs_F6F4"),
    "mxfp4_scaled_mfma": ("MFMAF4Flops", "MFMA_FLOPs_F6F4"),
    "int8_mfma": ("MFMAI8Ops",),
    "fp32_matrix": ("MFMAF32Flops",),
    "fp64_matrix": ("MFMAF64Flops",),
    "fp16_valu": ("FP16Flops",),
    "bf16_valu": ("BF16Flops",),
    "fp32_valu": ("FP32Flops",),
    "fp64_valu": ("FP64Flops",),
}
#: Memory level -> ``roofline.csv`` column. Every level the tool measured is
#: handed to the analyst, not only HBM: a working set that stays resident in
#: Infinity Cache rides a roof well above HBM, and one bounded by LDS throughput
#: rides one below it. Pinning the ceiling to HBM alone silently mis-bounds both.
_ROOF_BW_COLUMN_BY_TIER: dict[str, str] = {
    "hbm": "HBMBw",
    "mall": "MALLBw",
    "l2": "L2Bw",
    "l1": "L1Bw",
    "lds": "LDSBw",
}
_ROOF_BW_COLUMN = _ROOF_BW_COLUMN_BY_TIER["hbm"]

#: Profilers tried in order for the empirical roofs. ``omniperf`` is the former
#: name of the same tool and still ships on older ROCm images.
_ROOF_TOOLS = ("rocprof-compute", "omniperf")

#: Guide section 6: at least 20 samples, after warmup, taking the stable minimum.
DISPATCH_FLOOR_ROUNDS = 25
DISPATCH_FLOOR_BATCH = 200
DISPATCH_FLOOR_WARMUP = 500

_PROBE_SENTINEL = "__FORGE_DISPATCH_FLOOR__"

# Back-to-back trivial dispatches, timed from inside a captured graph.
#
# The graph is the whole point. Timed eagerly, each iteration also pays the
# framework's per-op host cost -- on an MI355X that was 4.27 us against 1.55 us
# for the same kernel replayed from a graph, so nearly two thirds of the eager
# figure is host submission. Charging a stage's serial dispatches at the eager
# rate overstates the latency term threefold, and it does so exactly on the small
# shapes where that term decides the ceiling.
#
# The graphed figure is also the one that matches how the campaign measures: a
# forge driver is graph-timed, so host submission is already outside its clock.
# Eager remains as a labelled fallback for a runtime without graph capture; a
# caller can tell the two apart by ``mode``.
_DISPATCH_FLOOR_PROBE = f'''
import json
import sys
import time

SENTINEL = "{_PROBE_SENTINEL}"
ROUNDS = {DISPATCH_FLOOR_ROUNDS}
BATCH = {DISPATCH_FLOOR_BATCH}
WARMUP = {DISPATCH_FLOOR_WARMUP}


def main() -> int:
    try:
        import torch
    except ImportError as exc:
        print(SENTINEL + json.dumps({{"ok": False, "detail": "torch is not importable: " + str(exc)}}))
        return 0
    if not torch.cuda.is_available():
        print(SENTINEL + json.dumps({{"ok": False, "detail": "no ROCm/CUDA device is visible to torch"}}))
        return 0

    buffer = torch.zeros(1, device=torch.device("cuda"))
    for _ in range(WARMUP):
        buffer.add_(1.0)
    torch.cuda.synchronize()

    def eager_once():
        """One batch of launches, paying host submission on every one."""
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(BATCH):
            buffer.add_(1.0)
        torch.cuda.synchronize()
        return (time.perf_counter() - started) / BATCH

    mode = "eager"
    detail = ""
    replay = None
    try:
        graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                buffer.add_(1.0)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            for _ in range(BATCH):
                buffer.add_(1.0)
        torch.cuda.synchronize()
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()

        def replay_once():
            """The same batch, replayed, so only device launch is on the clock."""
            torch.cuda.synchronize()
            started = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            return (time.perf_counter() - started) / BATCH

        replay = replay_once
        mode = "graph_replay"
    except Exception as exc:  # noqa: BLE001 - graph capture is the preferred path, not a required one
        detail = "graph capture unavailable, timed eagerly: " + str(exc)[:200]

    measure = replay if replay is not None else eager_once
    samples = [measure() for _ in range(ROUNDS)]
    eager_samples = [eager_once() for _ in range(ROUNDS)] if replay is not None else samples

    payload = {{
        "ok": True,
        "dispatch_floor_s": min(samples),
        "median_s": sorted(samples)[len(samples) // 2],
        "eager_floor_s": min(eager_samples),
        "mode": mode,
        "rounds": ROUNDS,
        "batch": BATCH,
        "device_name": torch.cuda.get_device_name(0),
        "kernel": "trivial in-place elementwise add on a 1-element tensor",
    }}
    if detail:
        payload["detail"] = detail
    print(SENTINEL + json.dumps(payload))
    return 0


sys.exit(main())
'''


@dataclass(frozen=True)
class EvidenceBundle:
    """Everything the deterministic side measured for one ceiling run."""

    hardware: Hardware
    artifacts_dir: Path
    #: ``case_id -> latency`` as seen while profiling. A sanity reference only:
    #: it is taken under a profiler and is not the campaign's baseline.
    observed_ms: dict[str, float] = field(default_factory=dict)
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
    ladder above decides what to do about them, so none of them raise.
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


def measure_dispatch_floor(
    *,
    workdir: str | Path,
    python_exe: str = "",
    timeout_sec: float = 600.0,
    env: dict[str, str] | None = None,
    artifacts_dir: Path | None = None,
) -> tuple[float, dict[str, Any]]:
    """Measure what one unavoidable kernel dispatch costs on this box.

    Returns ``(seconds, provenance)``. Seconds is ``0.0`` when the probe could
    not run, which leaves every latency term at zero and is reported as a caveat
    rather than silently absorbed -- a latency-bound case whose dispatch floor is
    missing reads as memory-bound and nearly free.
    """
    interpreter = python_exe.strip() or "python3"
    handle, script_path = tempfile.mkstemp(prefix="forge_dispatch_floor_", suffix=".py")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(_DISPATCH_FLOOR_PROBE)
        log_path = (artifacts_dir / "dispatch_floor.log") if artifacts_dir else None
        code, output = _run(
            [interpreter, script_path],
            cwd=workdir,
            timeout_sec=timeout_sec,
            env=env,
            log_path=log_path,
        )
    finally:
        Path(script_path).unlink(missing_ok=True)

    marker = output.rfind(_PROBE_SENTINEL)
    if marker < 0:
        detail = f"probe produced no result (exit {code})"
        log.warning("ceiling dispatch-floor probe failed: %s", detail)
        return 0.0, {"measured": False, "detail": detail}

    try:
        payload = json.loads(output[marker + len(_PROBE_SENTINEL) :].splitlines()[0])
    except (ValueError, IndexError) as exc:
        return 0.0, {"measured": False, "detail": f"probe result was unreadable: {exc}"}

    if not payload.get("ok"):
        detail = str(payload.get("detail") or "probe declined")
        log.warning("ceiling dispatch-floor probe declined: %s", detail)
        return 0.0, {"measured": False, "detail": detail}

    seconds = float(payload.get("dispatch_floor_s") or 0.0)
    payload["measured"] = seconds > 0
    return max(seconds, 0.0), payload


def _read_roofline_csv(path: Path, device_id: int = 0) -> dict[str, float]:
    """Read one device's row out of a ``rocprof-compute --roof-only`` result.

    The first column is the device id and is dropped by the tool's own reader
    before the header is taken; this mirrors that so the column names line up.
    """
    with path.open(encoding="utf-8") as stream:
        rows = [row for row in csv.reader(stream) if row]
    if len(rows) < 2:
        return {}
    header = [cell.strip() for cell in rows[0][1:]]
    index = min(device_id, len(rows) - 2)
    values = rows[1 + index][1:]

    peaks: dict[str, float] = {}
    for name, cell in zip(header, values):
        try:
            number = float(str(cell).strip())
        except (TypeError, ValueError):
            continue
        if number > 0:
            peaks[name] = number
    return peaks


def collect_empirical_peaks(
    *,
    command: Sequence[str],
    workdir: str | Path,
    artifacts_dir: Path,
    device_id: int = 0,
    timeout_sec: float = 3600.0,
    env: dict[str, str] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    """Measure this box's roofs with ``rocprof-compute --roof-only``.

    Returns ``(peak_flops_by_instruction_path, bytes_per_s_by_memory_level,
    provenance)``. Both are empty when no profiler could produce a usable
    ``roofline.csv``; the caller then falls back to the datasheet and says so.
    """
    workload = artifacts_dir / "roofline"
    workload.mkdir(parents=True, exist_ok=True)

    tool = next((candidate for candidate in _ROOF_TOOLS if shutil.which(candidate)), "")
    if not tool:
        return (
            {},
            {},
            {
                "measured": False,
                "detail": "no roofline-capable profiler on PATH; tried " + ", ".join(_ROOF_TOOLS),
            },
        )

    argv = [
        tool,
        "profile",
        "--roof-only",
        "--name",
        "ceiling",
        "--path",
        str(workload),
        "--device",
        str(device_id),
        "--",
        *command,
    ]
    code, _output = _run(
        argv,
        cwd=workdir,
        timeout_sec=timeout_sec,
        env=env,
        log_path=artifacts_dir / "roof_only.log",
    )

    # The tool nests its result under ``<path>/<name>/<SoC>``, and that layout
    # has moved between releases, so the file is found rather than constructed.
    found = sorted(workload.rglob("roofline.csv"))
    if not found:
        detail = f"{tool} exited {code} and produced no roofline.csv"
        log.warning("ceiling empirical roofs unavailable: %s", detail)
        return {}, {}, {"measured": False, "tool": tool, "exit_code": code, "detail": detail}

    try:
        columns = _read_roofline_csv(found[0], device_id=device_id)
    except (OSError, ValueError) as exc:
        return {}, {}, {"measured": False, "tool": tool, "detail": f"roofline.csv unreadable: {exc}"}

    if not columns:
        return (
            {},
            {},
            {"measured": False, "tool": tool, "detail": f"{found[0]} carried no positive peaks"},
        )

    peaks: dict[str, float] = {}
    resolved_from: dict[str, str] = {}
    for path, candidates in _ROOF_COLUMNS_BY_PATH.items():
        column = next((name for name in candidates if name in columns), "")
        if column:
            peaks[path] = columns[column] * _ROOF_UNIT_SCALE
            resolved_from[path] = column
    bandwidth = {
        tier: columns[column] * _ROOF_UNIT_SCALE
        for tier, column in _ROOF_BW_COLUMN_BY_TIER.items()
        if column in columns
    }

    provenance: dict[str, Any] = {
        "measured": bool(peaks) and bandwidth.get("hbm", 0.0) > 0,
        "tool": tool,
        "exit_code": code,
        "roofline_csv": str(found[0]),
        "columns_present": sorted(columns),
        # Which spelling this toolchain used, so a peak that looks wrong can be
        # traced back to the column it came from instead of re-derived by hand.
        "column_by_instruction_path": resolved_from,
        "paths_without_a_column": sorted(set(_ROOF_COLUMNS_BY_PATH) - set(peaks)),
    }
    if bandwidth.get("hbm", 0.0) <= 0:
        provenance["detail"] = f"roofline.csv has no positive {_ROOF_BW_COLUMN}"
    return peaks, bandwidth, provenance


def capture_kernel_trace(
    *,
    command: Sequence[str],
    workdir: str | Path,
    artifacts_dir: Path,
    timeout_sec: float = 1800.0,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Capture a kernel trace so the analyst can count real dispatches.

    Guide section 7.1: the trace is what tells the analyst which selector branch
    a shape entered, how many kernels one call actually launches, and whether
    the timed region covers the whole semantic operation. Without it the serial
    stage decomposition is guesswork.
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
    empirical_peaks: dict[str, float] | None = None,
    empirical_bandwidth: dict[str, float] | None = None,
    dispatch_floor_s: float = 0.0,
    provenance: dict[str, Any] | None = None,
    identity: DeviceIdentity | None = None,
    allow_reference: bool = True,
) -> Hardware:
    """Settle the hardware figures, best available source first.

    The order is: measured here, then measured on a reference card of the same
    configuration, then the datasheet. Sources are never blended. In a record
    half measured and half datasheet one term would be roughly twice as generous
    as the next and no field could say which, so a missing HBM figure drops the
    whole record to the next source rather than filling one hole from it.
    """
    resolved_arch = canon_arch(arch) or detect_arch() or canon_arch(arch)
    notes = dict(provenance or {})
    notes["arch_resolved_from"] = "caller" if canon_arch(arch) else "rocminfo"

    peaks = dict(empirical_peaks or {})
    bandwidth = dict(empirical_bandwidth or {})
    if peaks and bandwidth.get("hbm", 0.0) > 0:
        return Hardware(
            arch=resolved_arch,
            peak_flops=peaks,
            bandwidth=bandwidth,
            peak_source=PEAK_SOURCE_EMPIRICAL,
            dispatch_floor_s=float(dispatch_floor_s),
            provenance=notes,
        )

    if allow_reference:
        device = identity or describe_device(resolved_arch)
        reference = load_reference(device)
        if reference is not None:
            notes["reference_profile"] = {
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
                # A launch cost measured elsewhere still beats none, but one
                # measured here wins: this box's probe is cheap and always ran.
                dispatch_floor_s=float(dispatch_floor_s or reference.dispatch_floor_s),
                provenance=notes,
            )

    spec = arch_spec(resolved_arch)
    if spec is None:
        raise ValueError(
            f"no measured roofs and no datasheet peaks for arch {resolved_arch or '<undetected>'}; "
            "pass --arch with one of: " + ", ".join(sorted(_datasheet_arches()))
        )
    notes.setdefault("datasheet_source", spec.source)
    return Hardware(
        arch=spec.arch,
        peak_flops=dict(spec.peak_flops),
        bandwidth=spec.bandwidth(),
        peak_source=PEAK_SOURCE_DATASHEET,
        dispatch_floor_s=float(dispatch_floor_s),
        provenance=notes,
    )


def _datasheet_arches() -> tuple[str, ...]:
    """Import-light accessor kept local so the error message stays specific."""
    from kernelforge.roofline_ceiling.specs import supported_arches

    return supported_arches()


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
    device_id: int = 0,
    roof_only: bool = True,
    remeasure_device: bool = False,
    run_timeout_sec: float = 1800.0,
    roof_timeout_sec: float = 3600.0,
    env: dict[str, str] | None = None,
    project_root: str | Path | None = None,
) -> tuple[EvidenceBundle, list[str]]:
    """Gather every measured input one ceiling run needs.

    Returns ``(bundle, scored_case_ids)``. Each step degrades independently and
    records why, so a host without a profiler still produces a labelled answer
    instead of no answer or, worse, an unlabelled one.

    The roofs are measured at most once per machine. They do not depend on the
    operator, so a second campaign on the same box reads the profile this one
    cached; ``remeasure_device`` forces a fresh measurement when the box has
    changed underneath it.
    """
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    identity = describe_device(arch)

    scored, observed, notes = discover_scored_cases(
        command=performance_command,
        workdir=workdir,
        artifacts_dir=artifacts,
        timeout_sec=run_timeout_sec,
        env=env,
    )

    dispatch_floor_s, dispatch_provenance = measure_dispatch_floor(
        workdir=workdir,
        timeout_sec=run_timeout_sec,
        env=env,
        artifacts_dir=artifacts,
    )
    if not dispatch_provenance.get("measured"):
        notes.append("dispatch floor unmeasured: " + str(dispatch_provenance.get("detail") or "unknown"))

    peaks: dict[str, float] = {}
    bandwidth: dict[str, float] = {}
    roof_provenance: dict[str, Any] = {"measured": False, "detail": "--roof-only was not requested"}

    cached = None if remeasure_device else load_local(identity, project_root)
    if cached is not None:
        peaks, bandwidth = dict(cached.peak_flops), dict(cached.bandwidth)
        roof_provenance = {
            "measured": True,
            "reused": True,
            "origin": cached.origin,
            "measurement": cached.measurement,
            "source_by_figure": cached.source_by_figure,
            "notes": list(cached.notes),
        }
        notes.append(f"reused this box's cached roofs from {cached.origin}; pass --remeasure-device to refresh")
    elif roof_only:
        peaks, bandwidth, roof_provenance = collect_empirical_peaks(
            command=performance_command,
            workdir=workdir,
            artifacts_dir=artifacts,
            device_id=device_id,
            timeout_sec=roof_timeout_sec,
            env=env,
        )
        if roof_provenance.get("measured"):
            # Keyed by the machine, so the next operator on this box reads it
            # instead of paying for the same microbenchmarks again.
            stored = store_local(
                DeviceProfile(
                    peak_flops=peaks,
                    bandwidth=bandwidth,
                    dispatch_floor_s=dispatch_floor_s,
                    source_by_figure={
                        path: f"rocprof-compute --roof-only, column {column}"
                        for path, column in (roof_provenance.get("column_by_instruction_path") or {}).items()
                    },
                    measurement={
                        "measured_at": time.strftime("%Y-%m-%d"),
                        "tool": roof_provenance.get("tool", ""),
                        "roofline_csv": roof_provenance.get("roofline_csv", ""),
                        "dispatch_floor_mode": dispatch_provenance.get("mode", ""),
                    },
                ),
                identity,
                project_root,
            )
            notes.append(f"cached this box's roofs at {stored}")
        else:
            notes.append("empirical roofs unavailable: " + str(roof_provenance.get("detail") or "unknown"))

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
        empirical_peaks=peaks,
        empirical_bandwidth=bandwidth,
        dispatch_floor_s=dispatch_floor_s,
        identity=identity,
        provenance={
            "roofs": roof_provenance,
            "dispatch_floor": dispatch_provenance,
            "kernel_trace": trace_provenance,
        },
    )
    if hardware.peak_source == PEAK_SOURCE_REFERENCE:
        notes.append(
            "roofs came from a shipped reference card of this configuration, not from this box; "
            "install rocprof-compute and rerun to measure locally"
        )

    return (
        EvidenceBundle(
            hardware=hardware,
            artifacts_dir=artifacts,
            observed_ms=observed,
            notes=tuple(notes),
        ),
        scored,
    )


__all__ = [
    "DISPATCH_FLOOR_BATCH",
    "DISPATCH_FLOOR_ROUNDS",
    "EvidenceBundle",
    "capture_kernel_trace",
    "collect_empirical_peaks",
    "collect_evidence",
    "discover_scored_cases",
    "measure_dispatch_floor",
    "resolve_hardware",
]
