# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bench tool — wall-clock GPU kernel benchmarks with proper synchronization."""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from kernelforge.utils.process_group import kill_process_group

# Optional bench-mode companion to the driver contract's --warmup/--iters, asking the driver to time ONE declared case
# instead of its whole suite.
SWEEP_CASE_FLAG = "--bench-case"

# Warmup and timed iterations of one sweep point unless the operator asks for others.
SWEEP_WARMUP_ITERS = 3
SWEEP_BENCH_ITERS = 20

# How case selection actually resolved for one sweep point.
SELECTION_NARROWED = "narrowed"
SELECTION_WHOLE_SUITE = "whole_suite"
SELECTION_REJECTED = "rejected"

# Drivers observed to reject SWEEP_CASE_FLAG outright (argparse exits 2 on an unknown flag), keyed by the invocation
# minus the flag.
_CASE_FLAG_REJECTED: dict[str, bool] = {}

# Dispatch constants a sweep varies reach the driver as environment variables under this prefix by default.
SWEEP_ENV_PREFIX = "FORGE_SWEEP_"

# What a verbatim sweep may not name, by name here and by namespace below.
_RESERVED_ENV_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "TMPDIR",
        "ROCM_PATH",
        "HIP_PATH",
        "HIP_PLATFORM",
        "HIP_CLANG_PATH",
        "PYTORCH_ROCM_ARCH",
        "GPU_TARGET",
        "CC",
        "CXX",
        "CFLAGS",
        "CXXFLAGS",
        "LDFLAGS",
        "CPATH",
        "LIBRARY_PATH",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "TRITON_CACHE_DIR",
        "TRITON_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "PYTORCH_KERNEL_CACHE_PATH",
        "AITER_ROOT_DIR",
        "AITER_JIT_DIR",
    }
)

# Whole namespaces rather than named members: FORGE_ is forge's own and a source reads nothing under it that prefixed
# mode does not already reach, LD_ is the dynamic loader, PYTHON is the interpreter that has to start before the
# kernel exists at all, XDG_ is where every cache honouring the spec lives, and HIPCC is the compiler driver plus the
# *_FLAGS_APPEND variables hipcc reads on every invocation -- naming those families is shorter than tracking their
# members and refuses nothing a kernel would have computed with.
_RESERVED_ENV_PREFIXES = ("FORGE_", "LD_", "PYTHON", "XDG_", "HIPCC")

# What a source that read a swept constant prints, once per knob it consumed: `sweep_const: NAME VALUE`.
SWEEP_ECHO = "sweep_const"

# Marks a measurement as exploratory.
EXPLORATORY_KIND = "exploratory"

_CASE_MS_RE = re.compile(r"case_ms:\s*(\S+)\s+([\d.eE+-]+)[ \t]*(\S*)")
_SWEEP_ECHO_RE = re.compile(rf"{SWEEP_ECHO}:\s*([A-Z][A-Z0-9_]*)\s+(\S+)")
_CONSTANT_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_CONSTANT_VALUE_RE = re.compile(r"^[A-Za-z0-9_.,:=+-]+$")


def _case_flag_memo_key(base_cmd: list[str]) -> str:
    """Key ``_CASE_FLAG_REJECTED`` by the driver invocation, not by the point."""
    return "\x00".join(base_cmd)


class CaseCoverageError(ValueError):
    """Raised when a candidate cannot be scored against baseline cases."""


@dataclass(frozen=True)
class BenchRun:
    """One timed benchmark of a driver: its extra arguments, warmup and timed iterations, and repeats."""

    driver_script: str
    driver_args: list[str] | None = None
    warmup_iters: int = 10
    bench_iters: int = 30
    repeat: int = 1


def bench_command(run: BenchRun) -> list[str]:
    """The argv that runs one timed benchmark of ``run.driver_script``.

    One definition because it is the measurement contract, not a convenience: a
    second caller assembling its own flags eventually times a different region
    than the campaign scores, and a roofline ceiling derived against a different
    timed region is an answer to a question nobody asked.
    """
    args = (run.driver_args or []) + [
        "--warmup",
        str(run.warmup_iters),
        "--iters",
        str(run.bench_iters),
        "--bench-mode",
    ]
    if run.repeat > 1:
        args += ["--repeat", str(run.repeat)]
    return [sys.executable, run.driver_script] + args


def parse_case_timings(text: str) -> tuple[dict[str, float], list[str], set[str]]:
    """Read a driver's ``case_ms:`` lines out of one run's output.

    Returns ``(case_times, unscored_case_ids, duplicate_case_ids)``. This is the
    driver contract every consumer has to agree on: a second reader with its own
    regex would eventually disagree about which cases exist, and a set of cases
    that does not match the scored set is a silently wrong answer rather than a
    loud one.
    """
    case_times: dict[str, float] = {}
    unscored: list[str] = []
    duplicates: set[str] = set()
    for case_id, case_ms, tag in _CASE_MS_RE.findall(text):
        try:
            value = float(case_ms)
        except ValueError:
            continue
        if case_id in case_times:
            duplicates.add(case_id)
        case_times[case_id] = value
        if tag == "unscored":
            unscored.append(case_id)
    return case_times, unscored, duplicates


def _wall_ms_samples(output: str) -> list[float]:
    """The per-iteration ``wall_ms:`` times a driver printed."""
    return [float(m) for m in re.findall(r"wall_ms:\s*([\d.]+)", output)]


def _case_milliseconds(case_times: dict) -> dict[str, float]:
    """Each case's time as a float; raises ValueError naming the first case whose time is not finite and positive."""
    times: dict[str, float] = {}
    for case_id, value in case_times.items():
        times[case_id] = float(value)
        if not math.isfinite(times[case_id]) or times[case_id] <= 0:
            raise ValueError(case_id)
    return times


def _aggregate_refusal(measurements: list, message: str, **extra: Any) -> dict:
    """The aggregate's refusal, which carries every measurement it was given and no timing."""
    return {"success": False, "message": message, **extra, "measurements": measurements}


def aggregate_benchmark_measurements(measurements: list[dict]) -> dict:
    """Aggregate complete independent benchmark runs by per-case median."""
    if not measurements:
        return {"success": False, "message": "NO BENCHMARK MEASUREMENTS"}

    expected_cases: set[str] = set()
    expected_unscored: set[str] = set()
    run_times: list[dict[str, float]] = []
    wall_samples: list[float] = []
    for index, measurement in enumerate(measurements, start=1):
        label = f"MEASUREMENT {index}/{len(measurements)}"
        if not isinstance(measurement, dict):
            return _aggregate_refusal(measurements, f"{label} DID NOT RETURN A RESULT")
        if measurement.get("kind") == EXPLORATORY_KIND:
            return _aggregate_refusal(measurements, f"{label} IS AN EXPLORATORY SWEEP, WHICH CANNOT BE SCORED")
        if not measurement.get("success"):
            failure = f"{label} FAILED: {measurement.get('message', 'benchmark failed')}"
            return _aggregate_refusal(measurements, failure, output=measurement.get("output", ""))

        # Each measurement after the first has to match the one before it, and so the first.
        case_times = dict(measurement.get("case_times") or {})
        case_ids = set(case_times)
        if index > 1 and case_ids != expected_cases:
            mismatch = f"expected={sorted(expected_cases)}, got={sorted(case_ids)}"
            return _aggregate_refusal(measurements, f"MEASUREMENT CASE COVERAGE MISMATCH: {mismatch}")
        expected_cases = case_ids
        if not case_ids:
            return _aggregate_refusal(measurements, "MEASUREMENT REPORTED NO CASE TIMINGS")
        unscored = set(measurement.get("unscored_cases") or [])
        if index > 1 and unscored != expected_unscored:
            mismatch = f"expected={sorted(expected_unscored)}, got={sorted(unscored)}"
            return _aggregate_refusal(measurements, f"MEASUREMENT UNSCORED CASE MISMATCH: {mismatch}")
        expected_unscored = unscored
        try:
            run_times.append(_case_milliseconds(case_times))
        except (TypeError, ValueError) as error:
            return _aggregate_refusal(measurements, f"MEASUREMENT HAS INVALID CASE TIMING: {error}")

        wall = measurement.get("median_ms")
        if isinstance(wall, (int, float)) and math.isfinite(float(wall)):
            wall_samples.append(float(wall))

    case_times = {
        case_id: statistics.median(times[case_id] for times in run_times) for case_id in sorted(expected_cases)
    }

    # Reported, never scored, so the last measurement stands rather than a median that would have to be taken field by
    # field.
    case_bandwidth: dict[str, dict[str, float | int]] = dict(measurements[-1].get("case_bandwidth") or {})
    representative_wall = statistics.median(wall_samples) if len(wall_samples) == len(measurements) else None
    return {
        "success": True,
        "median_ms": representative_wall,
        "case_times": case_times,
        "unscored_cases": sorted(expected_unscored),
        "case_bandwidth": case_bandwidth,
        "measurement_count": len(measurements),
        "measurements": measurements,
        "message": (f"BENCH: {len(measurements)} independent measurements, per-case median"),
    }


async def measure_wallclock(run: BenchRun, *, measurements: int, timeout_sec: int = 300) -> dict:
    """Run independent benchmarks and aggregate their per-case medians."""
    results = []
    for _ in range(measurements):
        result = await bench_wallclock(run, timeout_sec=timeout_sec)
        results.append(result)
        if not result.get("success"):
            break
    return aggregate_benchmark_measurements(results)


def calculate_mean_case_speedup(
    case_times: dict[str, float] | None,
    baseline_case_times: dict[str, float] | None,
    unscored_cases: set[str] | list[str] | None = None,
) -> float | None:
    """Return the equal-weight arithmetic mean of per-case speedups."""
    if not baseline_case_times:
        return None
    if not case_times:
        raise CaseCoverageError("candidate emitted no per-case timings")
    baseline_ids = set(baseline_case_times)
    candidate_ids = set(case_times)
    if candidate_ids != baseline_ids:
        missing = sorted(baseline_ids - candidate_ids)
        unexpected = sorted(candidate_ids - baseline_ids)
        raise CaseCoverageError(
            f"candidate case coverage differs from baseline: missing={missing}, unexpected={unexpected}"
        )
    excluded = {str(c) for c in (unscored_cases or ())}
    speedups: list[float] = []
    for case_id, baseline_ms in baseline_case_times.items():
        if case_id in excluded:
            continue
        candidate_ms = case_times.get(case_id)
        if (
            not isinstance(baseline_ms, (int, float))
            or not math.isfinite(float(baseline_ms))
            or float(baseline_ms) <= 0.0
        ):
            raise CaseCoverageError(f"baseline case {case_id!r} has invalid timing")
        if (
            not isinstance(candidate_ms, (int, float))
            or not math.isfinite(float(candidate_ms))
            or float(candidate_ms) <= 0.0
        ):
            raise CaseCoverageError(f"candidate missing valid timing for baseline case {case_id!r}")
        speedups.append(float(baseline_ms) / float(candidate_ms))
    if not speedups:
        # Every case excluded means there is nothing to score against.
        raise CaseCoverageError("no scored cases remain after exclusions")
    mean_case_speedup = sum(speedups) / len(speedups)
    return mean_case_speedup if mean_case_speedup > 0.0 else None


def calculate_measurement_case_speedups(
    benchmark: dict | None,
    baseline_case_times: dict[str, float] | None,
    *,
    expected_measurements: int,
) -> list[float]:
    """Score every independent benchmark run against one fixed pristine baseline."""
    if not isinstance(benchmark, dict) or not benchmark.get("success"):
        raise CaseCoverageError("benchmark measurements are unavailable")
    measurements = benchmark.get("measurements")
    if not isinstance(measurements, list) or len(measurements) != expected_measurements:
        raise CaseCoverageError(f"exactly {expected_measurements} benchmark measurements are required")
    scores: list[float] = []
    for index, measurement in enumerate(measurements, start=1):
        if isinstance(measurement, dict) and measurement.get("kind") == EXPLORATORY_KIND:
            raise CaseCoverageError(f"measurement {index} is an exploratory sweep, which cannot be scored")
        if not isinstance(measurement, dict) or not measurement.get("success"):
            raise CaseCoverageError(f"measurement {index} failed")
        score = calculate_mean_case_speedup(
            measurement.get("case_times"),
            baseline_case_times,
            measurement.get("unscored_cases"),
        )
        if score is None:
            raise CaseCoverageError(f"measurement {index} has no mean case speedup")
        scores.append(float(score))
    return scores


async def bench_wallclock(
    run: BenchRun,
    *,
    timeout_sec: int = 300,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> dict:
    """Run a wall-clock benchmark with GPU synchronization."""
    cmd = bench_command(run)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        await kill_process_group(proc)
        return {"success": False, "message": f"TIMEOUT after {timeout_sec}s"}
    except asyncio.CancelledError:
        await kill_process_group(proc)
        raise

    stdout_text = stdout.decode(errors="replace")
    stderr_text = stderr.decode(errors="replace")
    full_output = stdout_text + "\n" + stderr_text

    if proc.returncode != 0:
        return {
            "success": False,
            "message": f"BENCH CRASHED (exit {proc.returncode})",
            "output": full_output[-2000:],
        }

    times = _wall_ms_samples(full_output)

    # Or parse a single pre-aggregated summary line.
    agg_match = re.search(r"(median_ms|mean_ms):\s*([\d.]+)", full_output)

    # Per-case timings for equal-weight suite scoring.
    case_times, unscored_cases, duplicate_case_ids = parse_case_timings(full_output)

    if duplicate_case_ids:
        return {
            "success": False,
            "message": ("DUPLICATE CASE TIMINGS: " + ", ".join(sorted(duplicate_case_ids))),
            "output": full_output[-1500:],
        }

    # Optional per-case bandwidth, for kernels whose wall time alone cannot say what got faster.
    case_bandwidth: dict[str, dict[str, float | int]] = {}
    for cid, nbytes, algbw, busbw in re.findall(
        r"case_bw:\s*(\S+)\s+bytes=(\d+)\s+algbw=([\d.eE+-]+)GB/s\s+busbw=([\d.eE+-]+)GB/s",
        full_output,
    ):
        try:
            case_bandwidth[cid] = {
                "bytes": int(nbytes),
                "algbw_gbs": float(algbw),
                "busbw_gbs": float(busbw),
            }
        except ValueError:
            continue

    if times:
        times_sorted = sorted(times)
        median = times_sorted[len(times_sorted) // 2]
        result = {
            "success": True,
            "median_ms": round(median, 4),
            "min_ms": round(min(times), 4),
            "max_ms": round(max(times), 4),
            "n_samples": len(times),
            "all_times_ms": [round(t, 4) for t in times],
            "case_times": case_times,
            "unscored_cases": unscored_cases,
            "case_bandwidth": case_bandwidth,
            "message": (f"BENCH: median={median:.4f} ms (min={min(times):.4f}, max={max(times):.4f}, n={len(times)})"),
        }
        if on_result:
            on_result(result)
        return result
    if agg_match:
        stat_label = agg_match.group(1)  # "median_ms" or "mean_ms"
        agg_value = float(agg_match.group(2))
        stat_name = "mean" if stat_label == "mean_ms" else "median"
        result = {
            # ``median_ms`` is this tool's stable representative-time field; the value is the driver's aggregate as-is
            # (labeled honestly above).
            "success": True,
            "median_ms": round(agg_value, 4),
            "stat": stat_name,
            "case_times": case_times,
            "unscored_cases": unscored_cases,
            "case_bandwidth": case_bandwidth,
            "message": f"BENCH: {stat_name}={agg_value:.4f} ms",
        }
        if on_result:
            on_result(result)
        return result
    return {
        "success": False,
        "message": (
            "NO TIMING DATA in output. Driver must print 'wall_ms: X.XX' "
            "per iteration or a single 'median_ms: X.XX' / 'mean_ms: X.XX' summary."
        ),
        "output": full_output[-1500:],
    }


def _sweep_failure(message: str, **extra: Any) -> dict:
    """Build a sweep result that carries no timing at all."""
    return {
        "success": False,
        "kind": EXPLORATORY_KIND,
        "message": f"SWEEP: {message}",
        **extra,
    }


def _sweep_environment(
    constants: dict[str, Any] | None,
    *,
    prefix_constants: bool = True,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return the child environment for one sweep point, and what it exports."""
    env = dict(os.environ)
    exported: dict[str, str] = {}
    for name, value in (constants or {}).items():
        key = str(name)
        if not _CONSTANT_NAME_RE.match(key):
            raise ValueError(f"constant name {name!r} is not an upper-case identifier")
        text = str(int(value)) if isinstance(value, bool) else str(value)
        if not _CONSTANT_VALUE_RE.match(text):
            raise ValueError(f"constant {key} has an unusable value {value!r}")
        if not prefix_constants and (key in _RESERVED_ENV_NAMES or key.startswith(_RESERVED_ENV_PREFIXES)):
            raise ValueError(
                f"constant {key} would overwrite a variable this measurement "
                f"runs on, not a knob of the kernel; a verbatim sweep may only "
                f"name a constant the source itself reads"
            )
        env[(SWEEP_ENV_PREFIX + key) if prefix_constants else key] = text
        exported[key] = text
    return env, dict(sorted(exported.items()))


async def _sweep_run(cmd: list[str], env: dict[str, str], timeout_sec: int) -> tuple[int | None, str]:
    """Run one driver invocation in a scratch directory that is then removed."""
    scratch = tempfile.mkdtemp(prefix="forge-sweep-")
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=scratch,
            env=env,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            await kill_process_group(proc)
            return None, ""
        except asyncio.CancelledError:
            await kill_process_group(proc)
            raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    output = stdout.decode(errors="replace") + "\n" + stderr.decode(errors="replace")
    return proc.returncode, output


class _PointRefused(Exception):
    """The driver's output does not measure the point; ``args`` are the refusal and the failure fields it adds."""


def _override_consumption(
    exported: dict[str, str], output: str, prefix_constants: bool, described: str
) -> dict[str, str]:
    """Which exported constants the driver echoed back; raises _PointRefused when the output does not measure them."""
    echoed = dict(_SWEEP_ECHO_RE.findall(output))
    consumption = {name: ("consumed" if name in echoed else "unread") for name in exported}
    unread = [name for name in exported if name not in echoed]
    if unread and prefix_constants:
        raise _PointRefused(
            f"{described}: NOTHING READ {', '.join(unread)} -- no "
            f"'{SWEEP_ECHO}: NAME VALUE' line came back, so this point ran the "
            f"default configuration and is not a measurement of the constant",
            {"override_consumption": consumption},
        )
    diverged = sorted(
        f"{name}: asked {exported[name]}, read {echoed[name]}"
        for name in exported
        if name in echoed and echoed[name] != exported[name]
    )
    if diverged:
        message = f"{described}: DRIVER READ A DIFFERENT CONFIGURATION ({'; '.join(diverged)})"
        raise _PointRefused(message, {"override_consumption": consumption})
    return consumption


def _requested_case_ms(
    case: str, seen: dict[str, float], duplicates: set[str], flag_exit: int | None, described: str
) -> float:
    """The one usable time the driver reported for the requested case; raises _PointRefused when there is none."""
    if case not in seen:
        # The retry ran the driver's WHOLE suite, so what came back is the set of cases this driver knows.
        misnamed = (
            f" -- {SWEEP_CASE_FLAG} was rejected (exit {flag_exit}) for a case "
            "this driver does not declare, which is the case id and not the flag"
            if flag_exit is not None
            else ""
        )
        reported = f"(reported: {sorted(seen) or 'nothing'}){misnamed}"
        raise _PointRefused(f"{described}: DRIVER REPORTED NO TIMING FOR CASE {case!r} {reported}", {})
    if case in duplicates:
        raise _PointRefused(f"{described}: DRIVER REPORTED CASE {case!r} MORE THAN ONCE", {})
    case_ms = seen[case]
    if not math.isfinite(case_ms) or case_ms <= 0.0:
        raise _PointRefused(f"{described}: DRIVER REPORTED AN UNUSABLE TIME FOR CASE {case!r}: {case_ms}", {})
    return case_ms


def _whole_suite_note(flag_exit: int | None, rejected_before: bool, seen: dict[str, float]) -> str:
    """Why the driver timed every case it knows instead of the one it was asked for."""
    how = (
        f"rejected {SWEEP_CASE_FLAG} (exit {flag_exit}), so it was re-run without it and"
        if flag_exit is not None
        else (
            f"is known to reject {SWEEP_CASE_FLAG}, so it was not asked again and"
            if rejected_before
            else f"ignored {SWEEP_CASE_FLAG} and"
        )
    )
    return f"driver {how} ran its whole suite: {', '.join(sorted(seen))}"


def _sweep_caveats(whole_suite: str, consumption: dict[str, str], samples: list[float], spread: bool) -> str:
    """The bracketed caveats on what an exploratory sweep number does not show; an empty note does not apply."""
    unread = [name for name, state in consumption.items() if state == "unread"]
    unconfirmed = (
        f"UNCONFIRMED: no '{SWEEP_ECHO}: NAME VALUE' line came back for "
        f"{', '.join(unread)}, so nothing proves the source read the "
        "override; this number means something only against a no-override "
        "reference measured in the same round"
    )
    # Lines came back, but they timed every case the driver ran, so their spread is not this case's and "none came
    # back" would be a lie about what the driver printed.
    unspread = (
        "the wall_ms lines time the driver's whole suite rather than this "
        "case, so this point has no measured spread of its own and a small "
        "difference against it means nothing"
        if samples
        else "no per-iteration wall_ms lines came back: this point has no "
        "measured spread, so a small difference against it means nothing"
    )
    notes = (whole_suite, unconfirmed if unread else "", "" if spread else unspread)
    return "".join(f" [{note}]" for note in notes if note)


async def sweep_case(
    *,
    bench: BenchRun,
    case_id: str,
    constants: dict[str, Any] | None = None,
    timeout_sec: int = 120,
    prefix_constants: bool = True,
) -> dict:
    """Time ONE declared case at ONE point in the dispatch-constant space."""
    case = str(case_id).strip()
    if len(case.split()) != 1:
        return _sweep_failure(f"INVALID CASE ID {case_id!r}")
    try:
        env, exported = _sweep_environment(constants, prefix_constants=prefix_constants)
    except ValueError as error:
        return _sweep_failure(str(error).upper())
    described = ", ".join(f"{k}={v}" for k, v in exported.items()) or "no overrides"
    point_failure = partial(_sweep_failure, case_id=case, constants=exported)

    base_cmd = bench_command(bench)
    memo_key = _case_flag_memo_key(base_cmd)
    rejected_before = _CASE_FLAG_REJECTED.get(memo_key, False)
    cmd = list(base_cmd) if rejected_before else base_cmd + [SWEEP_CASE_FLAG, case]
    returncode, full_output = await _sweep_run(cmd, env, timeout_sec)
    # Set only when the flagged invocation is what failed, which is what tells the message and the memo apart from a
    # configuration that will not run.
    flag_exit: int | None = None
    if returncode not in (0, None) and not rejected_before:
        flag_exit = returncode
        returncode, full_output = await _sweep_run(base_cmd, env, timeout_sec)
    if returncode is None:
        return point_failure(f"{described}: TIMEOUT after {timeout_sec}s")
    tail = full_output[-1500:]
    if returncode != 0:
        retried = f" (also exit {flag_exit} with {SWEEP_CASE_FLAG})" if flag_exit is not None else ""
        message = f"{described}: CONFIGURATION DID NOT RUN (exit {returncode}){retried}"
        return point_failure(message, case_selection=SELECTION_REJECTED, output=tail)
    try:
        consumption = _override_consumption(exported, full_output, prefix_constants, described)
        seen, _unscored, duplicates = parse_case_timings(full_output)
        # The requested case exists and the flagged invocation still failed, which leaves the flag itself as what the
        # driver would not take.
        if flag_exit is not None and case in seen:
            _CASE_FLAG_REJECTED[memo_key] = True
        case_ms = _requested_case_ms(case, seen, duplicates, flag_exit, described)
    except _PointRefused as refused:
        message, fields = refused.args
        return point_failure(message, **fields, output=tail)

    # Two different facts, and only the first is safe to infer from the output: ``narrowed`` says nothing but this
    # case came back, which is what makes the wall_ms lines this case's spread.
    narrowed = set(seen) == {case}
    selected = narrowed and not rejected_before and flag_exit is None
    whole_suite = "" if selected else _whole_suite_note(flag_exit, rejected_before, seen)
    samples = _wall_ms_samples(full_output)
    spread = narrowed and bool(samples)
    return {
        "success": True,
        "kind": EXPLORATORY_KIND,
        "case_id": case,
        "case_ms": round(case_ms, 6),
        "constants": exported,
        "narrowed": narrowed,
        "case_selection": SELECTION_NARROWED if selected else SELECTION_WHOLE_SUITE,
        "override_consumption": consumption,
        "warmup_iters": bench.warmup_iters,
        "bench_iters": bench.bench_iters,
        **(
            {"n_samples": len(samples), "wall_min_ms": round(min(samples), 6), "wall_max_ms": round(max(samples), 6)}
            if spread
            else {}
        ),
        "message": f"SWEEP (EXPLORATORY, NOT AN ACCEPTANCE RESULT): {case} = {case_ms:.6f} ms at {described}"
        + _sweep_caveats(whole_suite, consumption, samples, spread),
    }


def _sweep_cli(argv: list[str] | None = None) -> int:
    """One command, one data point -- the shell face of ``sweep_case``."""
    parser = argparse.ArgumentParser(
        prog="python3 -m kernelforge.evaluation.benchmark.runner",
        description=(
            "Time one declared case at one point in the dispatch-constant space. "
            "Exploratory only: the acceptance gate refuses these measurements."
        ),
    )
    parser.add_argument("--driver", required=True, help="the same driver (or lock wrapper) the gate runs")
    parser.add_argument("--case", required=True, help="one CASE_ID from the task's scored cases")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            f"dispatch constant, exported as {SWEEP_ENV_PREFIX}NAME; "
            f"the source must echo '{SWEEP_ECHO}: NAME VALUE' when "
            "it reads one"
        ),
    )
    parser.add_argument(
        "--verbatim-names",
        action="store_true",
        help=(
            f"export each --set name as-is instead of under "
            f"{SWEEP_ENV_PREFIX}, to reach a knob the source "
            "already defines; the echo then becomes a report "
            "rather than a requirement, and a name the "
            "measurement itself runs on (PATH, "
            "HIP_VISIBLE_DEVICES, ...) is refused"
        ),
    )
    parser.add_argument("--warmup", type=int, default=SWEEP_WARMUP_ITERS)
    parser.add_argument("--iters", type=int, default=SWEEP_BENCH_ITERS)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)

    constants: dict[str, str] = {}
    for assignment in args.set:
        name, separator, value = assignment.partition("=")
        if not separator:
            print(f"SWEEP: --set NEEDS NAME=VALUE, GOT {assignment!r}", flush=True)
            return 2
        constants[name.strip()] = value.strip()

    result = asyncio.run(
        sweep_case(
            bench=BenchRun(args.driver, warmup_iters=args.warmup, bench_iters=args.iters),
            case_id=args.case,
            constants=constants,
            timeout_sec=args.timeout,
            prefix_constants=not args.verbatim_names,
        )
    )
    print(result["message"], flush=True)
    if not result["success"] and result.get("output"):
        print(result["output"], flush=True)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(_sweep_cli())
