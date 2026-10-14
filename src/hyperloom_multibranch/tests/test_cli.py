# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the ``hyperloom-multibranch`` command line: options, defaults, validation and exit codes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom_multibranch import cli
from hyperloom_multibranch.scheduling.budget import MB_DEFAULT_MAX_HOURS
from hyperloom_multibranch.scheduling.peers import MB_FAKE_MISSING_PEER_MODULES
from hyperloom_multibranch.scheduling.roster import RosterCaps
from hyperloom_multibranch.scheduling.settings import RunSettings, Workload

WORKLOAD_ARGS = [
    "--model", "/models/Llama-3-8B", "--framework", "vllm", "--tp", "2", "--isl", "1024", "--osl", "512",
    "--conc", "16", "--precision", "fp8",
]  # fmt: skip


#: Main-flow runtime packages the scheduler must not load; it shares no process state with the main flow.
MAIN_FLOW_RUNTIME = (
    "hyperloom.orchestrator.loop",
    "hyperloom.orchestrator.phases",
    "hyperloom.orchestrator.state",
    "hyperloom.orchestrator.bus",
    "hyperloom.orchestrator.actions.executors",
    "hyperloom.orchestrator.specialists",
    "hyperloom.inference_optimizer.cli",
)
IMPORT_PROBE = (
    "import sys, hyperloom_multibranch.cli, hyperloom_multibranch.scheduling.lifecycle; "
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
        Workload("/models/Llama-3-8B", "vllm", 2, 1024, 512, 16, "fp8"),
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


def test_parallel_tests_default_is_at_least_one_for_a_wide_tp() -> None:
    settings = cli.settings_from(cli.build_parser().parse_args(["run", *WORKLOAD_ARGS[:5], "16", *WORKLOAD_ARGS[6:]]))
    assert settings.caps.max_parallel_tests == 1


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["run"],
        ["run", *WORKLOAD_ARGS, "--tp", "0"],
        ["run", *WORKLOAD_ARGS, "--max-hours", "-1"],
        ["run", *WORKLOAD_ARGS, "--max-hours", "0.01"],
        ["run", *WORKLOAD_ARGS, "--max-branches", "0"],
        ["run", *WORKLOAD_ARGS, "--max-parallel-tests", "0"],
        ["run", *WORKLOAD_ARGS, "--judge-interval-sec", "0"],
        ["run", *WORKLOAD_ARGS, "--conc", "many"],
    ],
)
def test_usage_errors_exit_two(argv: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == cli.EXIT_USAGE
    assert not any(tmp_path.iterdir())


def test_run_without_peers_fails_loudly_before_creating_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))

    assert cli.main(["run", *WORKLOAD_ARGS]) == cli.EXIT_FAILED

    captured = capsys.readouterr()
    assert all(module in captured.err for module in MB_FAKE_MISSING_PEER_MODULES)
    assert captured.out == ""
    assert not any(tmp_path.iterdir())


def test_importing_the_command_and_the_session_loads_no_main_flow_runtime() -> None:
    probe = subprocess.run([sys.executable, "-c", IMPORT_PROBE], capture_output=True, text=True, check=True)
    assert probe.stdout.strip() == "[]"
