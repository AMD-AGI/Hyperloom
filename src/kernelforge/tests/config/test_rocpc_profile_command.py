# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The packaged profiling script profiles the command its caller hands it, and nothing it derives itself."""

from __future__ import annotations

import importlib.util
import signal
from types import ModuleType, SimpleNamespace

import pytest

from kernelforge.config.resources import resource_path

SCRIPT = resource_path("local_knowledge") / "common_methodology" / "profiling" / "rocpc_profile.py"


@pytest.fixture
def rocpc_profile(monkeypatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_rocpc_profile_command_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # A standalone run installs termination handlers; the test process keeps its own.
    handlers = SimpleNamespace(SIGTERM=signal.SIGTERM, SIGINT=signal.SIGINT, signal=lambda *_args: None)
    monkeypatch.setattr(module, "signal", handlers)
    return module


def _record_profiler_runs(monkeypatch, module: ModuleType) -> list[dict]:
    """Stand in for an installed rocprof-compute, recording each invocation instead of running it."""
    runs: list[dict] = []
    monkeypatch.setattr(module, "_resolve_libexec", lambda: "/libexec")
    monkeypatch.setattr(module, "_detect_rocpc_python", lambda _libexec: "/usr/bin/python3")
    monkeypatch.setattr(module, "_torch_import_under_rocprofv3", lambda _python: (True, ""))

    def record(_python, _libexec, native, cwd=None, timeout=0):
        runs.append({"native": native, "cwd": cwd})
        return 1, "stopped by the test"

    monkeypatch.setattr(module, "_run", record)
    return runs


def test_the_command_after_the_separator_is_profiled_verbatim(rocpc_profile, tmp_path, monkeypatch) -> None:
    runs = _record_profiler_runs(monkeypatch, rocpc_profile)
    command = ["python3", "/scratch/adapter.py", "--case", "n1024", "--launches", "3"]

    rocpc_profile.main(["--out", str(tmp_path), "--", *command])

    (profile,) = runs
    assert profile["native"] == ["profile", "-n", "run", "--no-roof", "-b", "2", "--", *command]
    # The command runs from the output directory, which is why the script asks for absolute paths.
    assert profile["cwd"] == str(tmp_path)


def test_roofline_collection_profiles_the_same_command(rocpc_profile, tmp_path, monkeypatch) -> None:
    runs = _record_profiler_runs(monkeypatch, rocpc_profile)

    rocpc_profile.main(["--roofline", "--out", str(tmp_path), "--", "/scratch/run_kernel.sh"])

    assert runs[0]["native"] == ["profile", "-n", "run", "--", "/scratch/run_kernel.sh"]


@pytest.mark.parametrize(
    "argv",
    [[], ["--roofline"], ["--driver", "driver.py", "--", "python3", "/scratch/adapter.py"]],
    ids=["no arguments", "no command", "an option the script does not define"],
)
def test_a_run_that_does_not_state_its_command_is_refused(rocpc_profile, monkeypatch, argv) -> None:
    runs = _record_profiler_runs(monkeypatch, rocpc_profile)

    with pytest.raises(SystemExit) as refused:
        rocpc_profile.main(argv)

    assert refused.value.code == 2
    assert runs == []
