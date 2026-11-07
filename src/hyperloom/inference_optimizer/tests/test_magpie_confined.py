# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Magpie's leftover-server sweep, run through ``magpie_confined``, signals only the caller's sandbox or host."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import types
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import magpie_confined
from hyperloom.orchestrator.actions.executors.benchmark_backend import build_benchmark_command

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


def test_confined_the_sweep_kills_only_its_own_sandbox_s_server(
    monkeypatch: pytest.MonkeyPatch, signalled: list[tuple[int, int]]
) -> None:
    module = _benchmarker()
    real_os = module.os
    monkeypatch.setattr(magpie_confined, "process_environ", lambda pid: ENVIRONS[pid])
    magpie_confined.confine_sweep(module, b"lsb-a")

    module.BenchmarkMode()._cleanup_server_processes("sglang")

    assert signalled == [(OURS, signal.SIGKILL)]
    assert module.os is real_os


def test_a_magpie_without_the_sweep_runs_unconfined_and_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    module = types.ModuleType("benchmarker")
    module.BenchmarkMode = type("BenchmarkMode", (), {})

    magpie_confined.confine_sweep(module, b"lsb-a")

    assert "BenchmarkMode._cleanup_server_processes not found" in capsys.readouterr().err


def test_a_magpie_round_runs_magpie_through_the_confinement(tmp_path: Path) -> None:
    argv = build_benchmark_command(python_exe="py", config_path=tmp_path / "c.yaml", output_dir=tmp_path)

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


def test_outside_a_sandbox_magpie_s_sweep_spares_every_sandbox_s_server(
    monkeypatch: pytest.MonkeyPatch, signalled: list[tuple[int, int]]
) -> None:
    module = _benchmarker()
    for name in ("Magpie", "Magpie.modes", "Magpie.modes.benchmark"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setattr(sys.modules["Magpie.modes.benchmark"], "benchmarker", module, raising=False)
    monkeypatch.setitem(sys.modules, "Magpie.modes.benchmark.benchmarker", module)
    monkeypatch.delenv("E2B_SANDBOX_ID", raising=False)
    monkeypatch.setattr(magpie_confined, "process_environ", lambda pid: ENVIRONS[pid])
    monkeypatch.setattr(
        magpie_confined.runpy, "run_module", lambda *a, **k: module.BenchmarkMode._cleanup_server_processes("sglang")
    )
    monkeypatch.setattr(sys, "argv", ["magpie_confined.py"])

    magpie_confined.main()

    assert signalled == [(UNTAGGED, signal.SIGKILL)]


def test_the_sweep_spares_a_process_whose_environment_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch, signalled: list[tuple[int, int]]
) -> None:
    module = _benchmarker()
    monkeypatch.setattr(magpie_confined, "process_environ", lambda pid: None if pid == UNTAGGED else ENVIRONS[pid])
    magpie_confined.confine_sweep(module, b"")

    module.BenchmarkMode._cleanup_server_processes("sglang")

    assert signalled == []


def test_magpie_run_by_the_backend_imports_the_standard_library_s_profile(tmp_path: Path) -> None:
    magpie = tmp_path / "Magpie"
    (magpie / "modes" / "benchmark").mkdir(parents=True)
    for package in (magpie, magpie / "modes", magpie / "modes" / "benchmark"):
        (package / "__init__.py").write_text("")
    (magpie / "modes" / "benchmark" / "benchmarker.py").write_text("")
    (magpie / "__main__.py").write_text("import cProfile\nimport profile\nprint(profile.__file__)\n")
    argv = build_benchmark_command(python_exe=sys.executable, config_path=tmp_path / "c.yaml", output_dir=tmp_path)

    done = subprocess.run(argv, env={**os.environ, "PYTHONPATH": str(tmp_path)}, capture_output=True, text=True)

    assert done.returncode == 0, done.stderr
    assert Path(done.stdout.strip()).parent != Path(magpie_confined.__file__).parent
