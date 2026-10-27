# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A compile past its time is given up with every process it started: the compiler's own jobs do not outlive it."""

from __future__ import annotations

import os
import signal
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from hyperloom.measure.jit.native.store import BuildInfraError
from hyperloom.measure.runner.native import isolated
from hyperloom.measure.runner.native.builder import plan
from hyperloom.measure.runner.native.isolated import IsolatedBuilds, Isolation
from hyperloom.measure.runner.native.planning import Planned
from hyperloom.measure.runner.native.tests._native import COMPONENT, KERNEL, Bench

BUDGET_SEC = 2
#: How long a killed process may take to be gone.
_GONE_WITHIN_SEC = 10.0
#: How long past its budget a build may take to be given up (the stand-ins sleep 300 s).
_GIVEN_UP_WITHIN_SEC = 60.0
#: A process that starts a sleeper, says the sleeper's pid in the file ``argv[1]`` names, and waits for it.
_FORKS_A_SLEEPER = (
    "import pathlib, subprocess, sys, time\n"
    "sleeper = subprocess.Popen(['sleep', '300'])\n"
    "pathlib.Path(sys.argv[1]).write_text(str(sleeper.pid))\n"
    "time.sleep(300)\n"
)


def _gone(pid: int) -> bool:
    """Whether ``pid`` has ended: there is no such process, or it ended and only waits to be reaped."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return True
    return stat.rsplit(")", 1)[1].split()[0] in ("Z", "X")


def _waited_out(pid: int) -> bool:
    deadline = time.monotonic() + _GONE_WITHIN_SEC
    while not _gone(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return _gone(pid)


@pytest.fixture
def sleeper(tmp_path: Path) -> Iterator[Path]:
    """Where the stand-in says its sleeper's pid; a sleeper still there after the test is ended (its pid was recorded)."""
    pid_file = tmp_path / "sleeper.pid"
    yield pid_file
    if pid_file.is_file() and not _gone(pid := int(pid_file.read_text())):
        os.kill(pid, signal.SIGKILL)


def test_a_compile_past_its_budget_is_infra_and_its_compilers_children_are_killed(
    tmp_path: Path, sleeper: Path
) -> None:
    bench = Bench(tmp_path, budget_sec=BUDGET_SEC)
    patch = bench.patch({KERNEL: f"#fork {sleeper}\n"})

    started = time.monotonic()
    with pytest.raises(BuildInfraError, match="timed out or was killed"):
        bench.build(patch)

    assert time.monotonic() - started < BUDGET_SEC + _GIVEN_UP_WITHIN_SEC, "given up when the budget ran out"
    assert sleeper.is_file(), "the stand-in compiler started its sleeper before the budget ran out"
    assert _waited_out(int(sleeper.read_text())), "the compiler's own child is killed with it"


class _ForksASleeper:
    """The build child, standing in: it starts a sleeper and never ends."""

    def __init__(self, pid_file: Path) -> None:
        self._pid_file = pid_file

    def command(self, work: Path) -> list[str]:
        return [sys.executable, "-c", _FORKS_A_SLEEPER, str(self._pid_file)]


class _Planner:
    def __init__(self, planned: Planned) -> None:
        self._planned = planned

    def planned(self, inputs: object) -> Planned:
        return self._planned


def test_a_build_child_past_its_time_is_infra_and_every_process_it_started_is_killed(
    tmp_path: Path, sleeper: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(isolated, "_MARGIN_SEC", 0)
    bench = Bench(tmp_path, budget_sec=BUDGET_SEC)
    request = bench.request(bench.patch({KERNEL: "int kernel(void) { return 2; }\n"}))
    inputs, root = plan(request, bench.blobs)
    runner = IsolatedBuilds(
        _Planner(Planned(request, root)),  # type: ignore[arg-type]
        bench.blobs,
        Isolation((), None),
        {COMPONENT: "unused:factory"},
        _ForksASleeper(sleeper),  # type: ignore[arg-type]
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    started = time.monotonic()
    with pytest.raises(BuildInfraError, match=f"the build runner of {COMPONENT} did not end in time"):
        runner(inputs)(scratch)

    assert time.monotonic() - started < BUDGET_SEC + _GIVEN_UP_WITHIN_SEC, "given up when its time ran out"
    assert sleeper.is_file(), "the stand-in child started its sleeper before its time ran out"
    assert _waited_out(int(sleeper.read_text())), "the child's own child is killed with it"
