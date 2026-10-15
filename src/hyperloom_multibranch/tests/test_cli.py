# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the ``hyperloom-multibranch`` command line: options, defaults, validation and exit codes."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom_multibranch import cli
from hyperloom_multibranch.notebook.rsi.archive import ARCHIVE_ENV
from hyperloom_multibranch.scheduling.budget import MB_DEFAULT_MAX_HOURS
from hyperloom_multibranch.scheduling.local_peers import MB_FAKE_MISSING_SESSION_FACTORY
from hyperloom_multibranch.scheduling.roster import RosterCaps
from hyperloom_multibranch.scheduling.settings import KernelOptimizer, RunSettings, Workload

WORKLOAD_ARGS = [
    "--model", "/models/Llama-3-8B", "--framework", "vllm", "--tp", "2", "--isl", "1024", "--osl", "512",
    "--conc", "16", "--precision", "fp8", "--gpu-type", "mi300x",
]  # fmt: skip


#: Main-flow runtime packages the scheduler must not load; it shares no process state with the main flow.
MAIN_FLOW_RUNTIME = (
    "hyperloom.orchestrator.loop",
    "hyperloom.orchestrator.phases",
    "hyperloom.orchestrator.state",
    "hyperloom.orchestrator.bus",
    "hyperloom.orchestrator.actions.executors",
    "hyperloom.orchestrator.specialists.subprocess_",
    "hyperloom.orchestrator.specialists.runner",
    "hyperloom.orchestrator.kernel.patch_conflict_merge",
    "hyperloom.inference_optimizer.cli",
)
IMPORT_PROBE = (
    "import sys, hyperloom_multibranch.cli, hyperloom_multibranch.scheduling.lifecycle, "
    "hyperloom_multibranch.scheduling.exports; "
    f"print([m for m in sys.modules if m.startswith({MAIN_FLOW_RUNTIME!r})])"
)


def _settings(*extra: str) -> RunSettings:
    return cli.settings_from(cli.build_parser().parse_args(["run", *WORKLOAD_ARGS, *extra]))


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert cli.PROG in capsys.readouterr().out


def test_run_defaults() -> None:
    assert _settings() == RunSettings(
        Workload("/models/Llama-3-8B", "vllm", 2, 1024, 512, 16, "fp8", "mi300x", KernelOptimizer.FORGE),
        max_hours=MB_DEFAULT_MAX_HOURS,
        caps=RosterCaps(max_branches=4, max_k_per_branch=2, max_parallel_tests=4),
        judge_interval_seconds=600.0,
    )


def test_run_options_override_the_defaults() -> None:
    settings = _settings(
        "--gpu-type", "mi355x", "--max-hours", "12", "--max-branches", "6", "--max-parallel-tests", "3",
        "--max-k-per-branch", "3", "--judge-interval-sec", "120",
    )  # fmt: skip
    assert settings.workload.gpu_type == "mi355x"
    assert settings.max_hours == 12.0
    assert settings.caps == RosterCaps(6, 3, 3)
    assert settings.judge_interval_seconds == 120.0


def test_kernel_optimizer_can_be_geak() -> None:
    assert _settings("--kernel-optimizer", "geak").workload.kernel_optimizer is KernelOptimizer.GEAK


def test_parallel_tests_default_fills_the_machine() -> None:
    settings = cli.settings_from(cli.build_parser().parse_args(["run", *WORKLOAD_ARGS[:5], "8", *WORKLOAD_ARGS[6:]]))
    assert settings.caps.max_parallel_tests == 1


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["run"],
        ["run", *WORKLOAD_ARGS, "--tp", "0"],
        ["run", *WORKLOAD_ARGS, "--tp", "16"],
        ["run", *WORKLOAD_ARGS, "--max-parallel-tests", "5"],
        ["run", *WORKLOAD_ARGS, "--max-hours", "-1"],
        ["run", *WORKLOAD_ARGS, "--max-hours", "0.01"],
        ["run", *WORKLOAD_ARGS, "--max-branches", "0"],
        ["run", *WORKLOAD_ARGS, "--max-parallel-tests", "0"],
        ["run", *WORKLOAD_ARGS, "--judge-interval-sec", "0"],
        ["run", *WORKLOAD_ARGS, "--conc", "many"],
        ["run", *WORKLOAD_ARGS[:-2]],
        ["run", *WORKLOAD_ARGS, "--gpu-type", "h100"],
        ["run", *WORKLOAD_ARGS, "--kernel-optimizer", "triton"],
    ],
)
def test_usage_errors_exit_two(argv: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == cli.EXIT_USAGE
    assert not any(tmp_path.iterdir())


@pytest.fixture
def experience_kb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYPERLOOM_KB_URL", "http://127.0.0.1:9/kb")
    monkeypatch.setenv("HYPERLOOM_KB_TOKEN", "dummy-token")


@pytest.mark.usefixtures("experience_kb")
def test_run_without_peers_fails_loudly_before_creating_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))

    assert cli.main(["run", *WORKLOAD_ARGS]) == cli.EXIT_FAILED

    captured = capsys.readouterr()
    assert MB_FAKE_MISSING_SESSION_FACTORY in captured.err
    assert captured.out == ""
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("unset", "named"),
    [
        (("HYPERLOOM_KB_URL", "HYPERLOOM_KB_TOKEN"), "HYPERLOOM_KB_URL"),
        (("HYPERLOOM_KB_URL",), "HYPERLOOM_KB_URL"),
        (("HYPERLOOM_KB_TOKEN",), "HYPERLOOM_KB_TOKEN"),
    ],
)
@pytest.mark.usefixtures("experience_kb")
def test_run_without_the_experience_kb_fails_loudly_before_creating_anything(
    unset: tuple[str, ...],
    named: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    for name in unset:
        monkeypatch.delenv(name)

    assert cli.main(["run", *WORKLOAD_ARGS]) == cli.EXIT_FAILED

    captured = capsys.readouterr()
    assert "Experience KB" in captured.err and named in captured.err
    assert MB_FAKE_MISSING_SESSION_FACTORY not in captured.err
    assert captured.out == ""
    assert not any(tmp_path.iterdir())


@pytest.mark.usefixtures("experience_kb")
def test_the_rsi_archive_defaults_to_the_shared_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv(ARCHIVE_ENV, "")
    monkeypatch.delenv(ARCHIVE_ENV)

    cli.main(["run", *WORKLOAD_ARGS])

    assert os.environ[ARCHIVE_ENV] == cli.RSI_ARCHIVE_DEFAULT == "/shared_nfs/hyperloom/rsi-archive/"


@pytest.mark.usefixtures("experience_kb")
def test_a_named_rsi_archive_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv(ARCHIVE_ENV, str(tmp_path / "archive"))

    cli.main(["run", *WORKLOAD_ARGS])

    assert os.environ[ARCHIVE_ENV] == str(tmp_path / "archive")


def test_importing_the_command_and_the_session_loads_no_main_flow_runtime() -> None:
    probe = subprocess.run([sys.executable, "-c", IMPORT_PROBE], capture_output=True, text=True, check=True)
    assert probe.stdout.strip() == "[]"
