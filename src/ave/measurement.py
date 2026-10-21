# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-owned correctness and repeated timing of a prepared Ave driver."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import math
from importlib.util import cache_from_source
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from ave.task import OptimizationRun

KEEP_MEASUREMENT_COUNT = 3
DRIVER_TIMEOUT_SEC = 300
ALLCLOSE_PASS = "allclose: True"


class MeasurementError(RuntimeError):
    """The prepared driver failed correctness or produced invalid timings."""


@dataclass(frozen=True)
class Measurement:
    """Independent timings of an identical set of cases."""

    samples: tuple[dict[str, float], ...]

    @property
    def case_ms(self) -> dict[str, float]:
        return {case: statistics.fmean(sample[case] for sample in self.samples) for case in self.samples[0]}

    def scores(self, baseline: dict[str, float]) -> list[float]:
        if any(sample.keys() != baseline.keys() for sample in self.samples):
            raise MeasurementError("benchmark cases differ from the baseline")
        return [statistics.fmean(baseline[case] / sample[case] for case in baseline) for sample in self.samples]


def driver_environment(workspace: Path) -> dict[str, str]:
    """Put the measured source tree first without replacing its toolchain."""
    env = dict(os.environ)
    paths = [str(workspace)]
    paths.extend(str(workspace / name) for name in ("python", "src") if (workspace / name).is_dir())
    paths.extend(filter(None, env.get("PYTHONPATH", "").split(os.pathsep)))
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run_driver(run: OptimizationRun, arguments: tuple[str, ...], log: Path) -> str:
    remaining = run.deadline_unix - time.time()
    if remaining <= 0:
        raise TimeoutError("Ave deadline reached before measurement")
    command = [sys.executable, str(run.driver), *arguments]
    with subprocess.Popen(
        command,
        cwd=run.workspace,
        env=driver_environment(run.workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            output, _ = process.communicate(timeout=min(DRIVER_TIMEOUT_SEC, remaining))
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            log.write_text(output, encoding="utf-8")
            raise TimeoutError(f"driver timed out; see {log}") from error
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
    log.write_text(output, encoding="utf-8")
    if process.returncode != 0:
        raise MeasurementError(f"driver exited {process.returncode}: {output[-3000:]}")
    return output


def parse_case_times(output: str) -> dict[str, float]:
    """Read positive, finite timings, refusing duplicates and empty output."""
    cases: dict[str, float] = {}
    for line in output.splitlines():
        if not line.startswith("case_ms:"):
            continue
        fields = line.split()
        if len(fields) != 3:
            raise MeasurementError(f"invalid timing: {line}")
        try:
            duration = float(fields[2])
        except ValueError as error:
            raise MeasurementError(f"invalid timing: {line}") from error
        if fields[1] in cases or not math.isfinite(duration) or duration <= 0:
            raise MeasurementError(f"duplicate or invalid timing: {line}")
        cases[fields[1]] = duration
    if not cases:
        raise MeasurementError("driver emitted no case_ms timings")
    return cases


def measure(run: OptimizationRun, directory: Path) -> Measurement:
    """Require the full correctness suite, then independently repeat graph timings."""
    directory.mkdir(parents=True, exist_ok=True)
    for source in run.sources:
        if source.endswith(".py"):
            Path(cache_from_source(str(run.workspace / source))).unlink(missing_ok=True)
    output = _run_driver(run, (), directory / "correctness.log")
    signals = [line.strip() for line in output.splitlines() if line.strip().startswith("allclose:")]
    if not signals or any(line != ALLCLOSE_PASS for line in signals):
        raise MeasurementError(f"driver did not pass its correctness suite: {output[-3000:]}")
    arguments = ("--bench-mode", "--warmup", "3", "--iters", "20")
    samples = tuple(
        parse_case_times(_run_driver(run, arguments, directory / f"benchmark-{index}.log"))
        for index in range(KEEP_MEASUREMENT_COUNT)
    )
    result = Measurement(samples)
    result.scores(samples[0])
    return result
