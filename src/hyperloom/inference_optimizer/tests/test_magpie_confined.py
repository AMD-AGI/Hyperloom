# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Magpie's leftover-server sweep, run through ``magpie_confined``, signals only the caller's sandbox."""

from __future__ import annotations

import signal
import types
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import magpie_confined
from hyperloom.orchestrator.actions.executors.benchmark_backend import MagpieBackend

OURS, THEIRS, UNTAGGED = 101, 202, 303
ENVIRONS = {
    OURS: [b"PATH=/bin", b"E2B_SANDBOX_ID=lsb-a"],
    THEIRS: [b"PATH=/bin", b"E2B_SANDBOX_ID=lsb-b"],
    UNTAGGED: [b"PATH=/bin"],
}

_FAKE_BENCHMARKER = """
import os
import signal

class BenchmarkMode:
    @staticmethod
    def _cleanup_server_processes(framework):
        for pid in (101, 202, 303):
            os.kill(pid, signal.SIGKILL)
"""


def _benchmarker() -> types.ModuleType:
    module = types.ModuleType("benchmarker")
    exec(_FAKE_BENCHMARKER, module.__dict__)
    return module


@pytest.fixture
def signalled(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(magpie_confined.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    return sent


def test_unconfined_the_sweep_kills_every_sandbox_s_server(signalled: list[tuple[int, int]]) -> None:
    _benchmarker().BenchmarkMode._cleanup_server_processes("sglang")

    assert [pid for pid, _ in signalled] == [OURS, THEIRS, UNTAGGED]


def test_confined_the_sweep_kills_only_its_own_sandbox_s_server(signalled: list[tuple[int, int]]) -> None:
    module = _benchmarker()
    real_os = module.os
    magpie_confined.confine_sweep(module, b"E2B_SANDBOX_ID=lsb-a", environ=lambda pid: ENVIRONS[pid])

    module.BenchmarkMode()._cleanup_server_processes("sglang")

    assert signalled == [(OURS, signal.SIGKILL)]
    assert module.os is real_os


def test_a_magpie_without_the_sweep_runs_unconfined_and_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    module = types.ModuleType("benchmarker")
    module.BenchmarkMode = type("BenchmarkMode", (), {})

    magpie_confined.confine_sweep(module, b"E2B_SANDBOX_ID=lsb-a")

    assert "BenchmarkMode._cleanup_server_processes not found" in capsys.readouterr().err


def test_the_magpie_backend_runs_magpie_through_the_confinement(tmp_path: Path) -> None:
    argv = MagpieBackend().build_command(python_exe="py", config_path=tmp_path / "c.yaml", output_dir=tmp_path)

    assert argv[:2] == ["py", str(Path(magpie_confined.__file__))]
    assert argv[2:4] == ["-v", "benchmark"]
    assert argv[4:] == [
        "--benchmark-config",
        str(tmp_path / "c.yaml"),
        "--output-dir",
        str(tmp_path),
        "--run-mode",
        "local",
    ]
