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
from typing import Any, Callable

from ._subprocess import kill_process_group

# Optional bench-mode companion to the driver contract's --warmup/--iters, asking the driver to time ONE declared case
# instead of its whole suite.
SWEEP_CASE_FLAG = "--bench-case"

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


def aggregate_benchmark_measurements(measurements: list[dict]) -> dict:
    """Aggregate complete independent benchmark runs by per-case median."""
    if not measurements:
        return {"success": False, "message": "NO BENCHMARK MEASUREMENTS"}

    expected_cases: set[str] | None = None
    expected_unscored: set[str] | None = None
    case_samples: dict[str, list[float]] = {}
    wall_samples: list[float] = []

    for index, measurement in enumerate(measurements, start=1):
        if not isinstance(measurement, dict):
            return {
                "success": False,
                "message": (f"MEASUREMENT {index}/{len(measurements)} DID NOT RETURN A RESULT"),
                "measurements": measurements,
            }
        if measurement.get("kind") == EXPLORATORY_KIND:
            return {
                "success": False,
                "message": (f"MEASUREMENT {index}/{len(measurements)} IS AN EXPLORATORY SWEEP, WHICH CANNOT BE SCORED"),
                "measurements": measurements,
            }
        if not measurement.get("success"):
            return {
                "success": False,
                "message": (
                    f"MEASUREMENT {index}/{len(measurements)} FAILED: {measurement.get('message', 'benchmark failed')}"
                ),
                "output": measurement.get("output", ""),
                "measurements": measurements,
            }

        case_times = dict(measurement.get("case_times") or {})
        case_ids = set(case_times)
        if expected_cases is None:
            expected_cases = case_ids
            case_samples = {case_id: [] for case_id in sorted(case_ids)}
        elif case_ids != expected_cases:
            return {
                "success": False,
                "message": (
                    f"MEASUREMENT CASE COVERAGE MISMATCH: expected={sorted(expected_cases)}, got={sorted(case_ids)}"
                ),
                "measurements": measurements,
            }
        if not case_ids:
            return {
                "success": False,
                "message": "MEASUREMENT REPORTED NO CASE TIMINGS",
                "measurements": measurements,
            }

        unscored = {str(case_id) for case_id in measurement.get("unscored_cases") or []}
        if expected_unscored is None:
            expected_unscored = unscored
        elif unscored != expected_unscored:
            return {
                "success": False,
                "message": (
                    f"MEASUREMENT UNSCORED CASE MISMATCH: expected={sorted(expected_unscored)}, got={sorted(unscored)}"
                ),
                "measurements": measurements,
            }

        try:
            for case_id, value in case_times.items():
                numeric = float(value)
                if not math.isfinite(numeric) or numeric <= 0:
                    raise ValueError(case_id)
                case_samples[case_id].append(numeric)
        except (TypeError, ValueError) as error:
            return {
                "success": False,
                "message": f"MEASUREMENT HAS INVALID CASE TIMING: {error}",
                "measurements": measurements,
            }

        wall = measurement.get("median_ms")
        if isinstance(wall, (int, float)) and math.isfinite(float(wall)):
            wall_samples.append(float(wall))

    case_times = {case_id: statistics.median(values) for case_id, values in case_samples.items()}
    # Reported, never scored, so the last measurement stands rather than a median that would have to be taken field by
    # field.
    case_bandwidth: dict[str, dict[str, float | int]] = dict(measurements[-1].get("case_bandwidth") or {})
    representative_wall = statistics.median(wall_samples) if len(wall_samples) == len(measurements) else None
    return {
        "success": True,
        "median_ms": representative_wall,
        "case_times": case_times,
        "unscored_cases": sorted(expected_unscored or set()),
        "case_bandwidth": case_bandwidth,
        "measurement_count": len(measurements),
        "measurements": measurements,
        "message": (f"BENCH: {len(measurements)} independent measurements, per-case median"),
    }


async def measure_wallclock(
    *,
    driver_script: str,
    driver_args: list[str] | None = None,
    measurements: int,
    warmup_iters: int = 10,
    bench_iters: int = 30,
    timeout_sec: int = 300,
    repeat: int = 1,
) -> dict:
    """Run independent benchmarks and aggregate their per-case medians."""
    results = []
    for _ in range(measurements):
        result = await bench_wallclock(
            driver_script=driver_script,
            driver_args=driver_args,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            timeout_sec=timeout_sec,
            repeat=repeat,
        )
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
    driver_script: str,
    driver_args: list[str] | None = None,
    warmup_iters: int = 10,
    bench_iters: int = 30,
    timeout_sec: int = 300,
    on_result: Callable[[dict[str, Any]], None] | None = None,
    *,
    repeat: int = 1,
) -> dict:
    """Run a wall-clock benchmark with GPU synchronization."""
    args = (driver_args or []) + [
        "--warmup",
        str(warmup_iters),
        "--iters",
        str(bench_iters),
        "--bench-mode",
    ]
    if repeat > 1:
        args += ["--repeat", str(repeat)]
    cmd = [sys.executable, driver_script] + args

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

    # Parse individual wall_ms values
    times = [float(m) for m in re.findall(r"wall_ms:\s*([\d.]+)", full_output)]

    # Or parse a single pre-aggregated summary line.
    agg_match = re.search(r"(median_ms|mean_ms):\s*([\d.]+)", full_output)

    # Per-case timings for equal-weight suite scoring.
    case_times: dict[str, float] = {}
    duplicate_case_ids: set[str] = set()
    unscored_cases: list[str] = []
    for cid, cms, tag in _CASE_MS_RE.findall(full_output):
        try:
            if cid in case_times:
                duplicate_case_ids.add(cid)
            case_times[cid] = float(cms)
        except ValueError:
            continue
        if tag == "unscored":
            unscored_cases.append(cid)

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
    elif agg_match:
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
    else:
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


async def sweep_case(
    *,
    driver_script: str,
    case_id: str,
    constants: dict[str, Any] | None = None,
    warmup_iters: int = 3,
    bench_iters: int = 20,
    timeout_sec: int = 120,
    prefix_constants: bool = True,
) -> dict:
    """Time ONE declared case at ONE point in the dispatch-constant space."""
    case = str(case_id).strip()
    if not case or len(case.split()) != 1:
        return _sweep_failure(f"INVALID CASE ID {case_id!r}")
    try:
        env, exported = _sweep_environment(constants, prefix_constants=prefix_constants)
    except ValueError as error:
        return _sweep_failure(str(error).upper())
    described = ", ".join(f"{k}={v}" for k, v in exported.items()) or "no overrides"

    base_cmd = [
        sys.executable,
        driver_script,
        "--warmup",
        str(warmup_iters),
        "--iters",
        str(bench_iters),
        "--bench-mode",
    ]
    memo_key = _case_flag_memo_key(base_cmd)
    rejected_before = _CASE_FLAG_REJECTED.get(memo_key, False)
    carried_flag = not rejected_before
    cmd = (base_cmd + [SWEEP_CASE_FLAG, case]) if carried_flag else list(base_cmd)

    returncode, full_output = await _sweep_run(cmd, env, timeout_sec)
    # Set only when the flagged invocation is what failed, which is what tells the message and the memo apart from a
    # configuration that will not run.
    flag_exit: int | None = None
    if returncode not in (0, None) and carried_flag:
        flag_exit = returncode
        returncode, full_output = await _sweep_run(base_cmd, env, timeout_sec)
    if returncode is None:
        return _sweep_failure(
            f"{described}: TIMEOUT after {timeout_sec}s",
            case_id=case,
            constants=exported,
        )
    if returncode != 0:
        retried = f" (also exit {flag_exit} with {SWEEP_CASE_FLAG})" if flag_exit is not None else ""
        return _sweep_failure(
            f"{described}: CONFIGURATION DID NOT RUN (exit {returncode}){retried}",
            case_id=case,
            constants=exported,
            case_selection=SELECTION_REJECTED,
            output=full_output[-1500:],
        )

    echoed = dict(_SWEEP_ECHO_RE.findall(full_output))
    unread = sorted(name for name in exported if name not in echoed)
    consumption = {name: ("consumed" if name in echoed else "unread") for name in exported}
    if unread and prefix_constants:
        return _sweep_failure(
            f"{described}: NOTHING READ {', '.join(unread)} -- no "
            f"'{SWEEP_ECHO}: NAME VALUE' line came back, so this point ran the "
            f"default configuration and is not a measurement of the constant",
            case_id=case,
            constants=exported,
            override_consumption=consumption,
            output=full_output[-1500:],
        )
    diverged = sorted(
        f"{name}: asked {exported[name]}, read {echoed[name]}"
        for name in exported
        if name in echoed and echoed[name] != exported[name]
    )
    if diverged:
        return _sweep_failure(
            f"{described}: DRIVER READ A DIFFERENT CONFIGURATION ({'; '.join(diverged)})",
            case_id=case,
            constants=exported,
            override_consumption=consumption,
            output=full_output[-1500:],
        )

    seen: dict[str, float] = {}
    duplicated = False
    for cid, cms, _tag in _CASE_MS_RE.findall(full_output):
        try:
            value = float(cms)
        except ValueError:
            continue
        duplicated = duplicated or (cid == case and cid in seen)
        seen[cid] = value

    if case not in seen:
        # The retry ran the driver's WHOLE suite, so what came back is the set of cases this driver knows.
        misnamed = (
            f" -- {SWEEP_CASE_FLAG} was rejected (exit {flag_exit}) for a case "
            "this driver does not declare, which is the case id and not the flag"
            if flag_exit is not None
            else ""
        )
        return _sweep_failure(
            f"{described}: DRIVER REPORTED NO TIMING FOR CASE {case!r} "
            f"(reported: {sorted(seen) or 'nothing'}){misnamed}",
            case_id=case,
            constants=exported,
            output=full_output[-1500:],
        )
    # The requested case exists and the flagged invocation still failed, which leaves the flag itself as what the
    # driver would not take.
    if flag_exit is not None:
        _CASE_FLAG_REJECTED[memo_key] = True
    if duplicated:
        return _sweep_failure(
            f"{described}: DRIVER REPORTED CASE {case!r} MORE THAN ONCE",
            case_id=case,
            constants=exported,
            output=full_output[-1500:],
        )
    case_ms = seen[case]
    if not math.isfinite(case_ms) or case_ms <= 0.0:
        return _sweep_failure(
            f"{described}: DRIVER REPORTED AN UNUSABLE TIME FOR CASE {case!r}: {case_ms}",
            case_id=case,
            constants=exported,
            output=full_output[-1500:],
        )

    # Two different facts, and only the first is safe to infer from the output: ``narrowed`` says nothing but this
    # case came back, which is what makes the wall_ms lines this case's spread.
    selected_by_flag = carried_flag and flag_exit is None
    narrowed = set(seen) == {case}
    selection = SELECTION_NARROWED if (selected_by_flag and narrowed) else SELECTION_WHOLE_SUITE
    result = {
        "success": True,
        "kind": EXPLORATORY_KIND,
        "case_id": case,
        "case_ms": round(case_ms, 6),
        "constants": exported,
        "narrowed": narrowed,
        "case_selection": selection,
        "override_consumption": consumption,
        "warmup_iters": warmup_iters,
        "bench_iters": bench_iters,
    }
    samples = [float(m) for m in re.findall(r"wall_ms:\s*([\d.]+)", full_output)]
    if narrowed and samples:
        result["n_samples"] = len(samples)
        result["wall_min_ms"] = round(min(samples), 6)
        result["wall_max_ms"] = round(max(samples), 6)
    notes = []
    if selection == SELECTION_WHOLE_SUITE:
        how = (
            f"rejected {SWEEP_CASE_FLAG} (exit {flag_exit}), so it was re-run without it and"
            if flag_exit is not None
            else (
                f"is known to reject {SWEEP_CASE_FLAG}, so it was not asked again and"
                if rejected_before
                else f"ignored {SWEEP_CASE_FLAG} and"
            )
        )
        notes.append(f"driver {how} ran its whole suite: {', '.join(sorted(seen))}")
    if unread:
        notes.append(
            f"UNCONFIRMED: no '{SWEEP_ECHO}: NAME VALUE' line came back for "
            f"{', '.join(unread)}, so nothing proves the source read the "
            "override; this number means something only against a no-override "
            "reference measured in the same round"
        )
    if "wall_min_ms" not in result:
        notes.append(
            # Lines came back, but they timed every case the driver ran, so their spread is not this case's and "none
            # came back" would be a lie about what the driver printed.
            "the wall_ms lines time the driver's whole suite rather than this "
            "case, so this point has no measured spread of its own and a small "
            "difference against it means nothing"
            if samples
            else "no per-iteration wall_ms lines came back: this point has no "
            "measured spread, so a small difference against it means nothing"
        )
    suffix = "".join(f" [{note}]" for note in notes)
    result["message"] = (
        f"SWEEP (EXPLORATORY, NOT AN ACCEPTANCE RESULT): {case} = {case_ms:.6f} ms at {described}{suffix}"
    )
    return result


def _sweep_cli(argv: list[str] | None = None) -> int:
    """One command, one data point -- the shell face of ``sweep_case``."""
    parser = argparse.ArgumentParser(
        prog="python3 -m kernelforge.mcp_server.tools.bench",
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
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
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
            driver_script=args.driver,
            case_id=args.case,
            constants=constants,
            warmup_iters=args.warmup,
            bench_iters=args.iters,
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
