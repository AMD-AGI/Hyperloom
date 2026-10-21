# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the ``hyperloom-multibranch`` command line: options, defaults, validation and exit codes.

``final-only`` is the emergency recovery of a session that ended before FINAL, one killed with its machine included:
it takes that session's directory and nothing else, refuses before building anything when the directory cannot
support FINAL -- a session another build ran, and one still running, included -- and holds the directory against a
second recovery while it works.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV
from hyperloom.common.llm_config import (
    AGENT_BACKEND_CLAUDE,
    AGENT_BACKEND_CODEX,
    ANTHROPIC_CREDENTIAL_ENV_ORDER,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CODEX_MODEL,
)
from hyperloom.common.owner_scope import local_owner_scope
from hyperloom.common.perf_metric import GRADED_INTVTY_P50
from hyperloom.measure import compile as compile_module
from hyperloom.measure.compile import PackageBase
from hyperloom_multibranch import cli
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.notebook.rsi.archive import ARCHIVE_ENV
from hyperloom_multibranch.scheduling.budget import MB_DEFAULT_MAX_HOURS
from hyperloom_multibranch.scheduling.layout import SessionLayout
from hyperloom_multibranch.scheduling.lifecycle import PersistedRun, code_version, persisted_run
from hyperloom_multibranch.scheduling import local_peers
from hyperloom_multibranch.scheduling.local_peers import AGENTX_ENV
from hyperloom_multibranch.scheduling.peers import PeerFactory
from hyperloom_multibranch.scheduling.processes import SCHEDULER_BRANCH, ProcessRecord
from hyperloom_multibranch.scheduling.roster import RosterCaps
from hyperloom_multibranch.scheduling.settings import (
    FrontierPolicyMode,
    InferenceXShape,
    KernelOptimizer,
    RunSettings,
    Workload,
)
from hyperloom_multibranch.scheduling.stops import SessionStopReason

#: What the session's launcher sets to name the image the session runs on.
IMAGE_VARS = ("HYPERLOOM_ACTIVE_IMAGE_REF", "HYPERLOOM_ACTIVE_IMAGE_DIGEST")
IMAGE = ("registry.example/vllm:v1", "sha256:" + "2" * 64)
WORKLOAD_ARGS = [
    "--model", "/models/Llama-3-8B", "--framework", "vllm", "--tp", "2", "--isl", "1024", "--osl", "512",
    "--conc", "16", "--precision", "fp8", "--workload", "inferencex", "--gpu-type", "mi300x",
]  # fmt: skip
#: An ownership scope of a pod that is gone: another boot, so none of its pids is a process of this environment.
ANOTHER_POD = "0" * 32 + ":1:1"
#: The same workload as an AgentX trace replay: no sequence lengths.
AGENTX_ARGS = [
    "--model", "/models/Llama-3-8B", "--framework", "vllm", "--tp", "2", "--conc", "16", "--precision", "fp8",
    "--workload", "agentx", "--gpu-type", "mi300x",
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


def _settings(*extra: str, workload: list[str] = WORKLOAD_ARGS) -> RunSettings:
    return cli.settings_from(cli.build_parser().parse_args(["run", *workload, *extra]))


def _without(args: list[str], option: str) -> list[str]:
    at = args.index(option)
    return [*args[:at], *args[at + 2 :]]


def test_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert cli.PROG in capsys.readouterr().out


def test_run_defaults() -> None:
    assert _settings() == RunSettings(
        Workload(
            "/models/Llama-3-8B", "vllm", 2, 16, "fp8", "mi300x", KernelOptimizer.FORGE, InferenceXShape(1024, 512)
        ),
        max_hours=MB_DEFAULT_MAX_HOURS,
        max_gpu_hours=0.0,
        max_llm_usd=0.0,
        caps=RosterCaps(max_branches=4, max_k_per_branch=2, max_parallel_tests=4),
        judge_interval_seconds=600.0,
    )
    assert _settings().frontier_policy is FrontierPolicyMode.AUTO


def test_frontier_policy_can_keep_the_defaults() -> None:
    assert _settings("--frontier-policy", "default").frontier_policy is FrontierPolicyMode.DEFAULT


def test_run_options_override_the_defaults() -> None:
    settings = _settings(
        "--gpu-type", "mi355x", "--max-hours", "12", "--max-gpu-hours", "40", "--max-llm-usd", "25.5",
        "--max-branches", "6", "--max-parallel-tests", "3", "--max-k-per-branch", "3", "--judge-interval-sec", "120",
    )  # fmt: skip
    assert settings.workload.gpu_type == "mi355x"
    assert (settings.max_hours, settings.max_gpu_hours, settings.max_llm_usd) == (12.0, 40.0, 25.5)
    assert settings.caps == RosterCaps(6, 3, 3)
    assert settings.judge_interval_seconds == 120.0


def test_an_agentx_run_has_no_sequence_lengths_and_is_graded_on_p50_interactivity_at_its_conc_alone() -> None:
    workload = _settings(workload=AGENTX_ARGS).workload

    assert workload == Workload("/models/Llama-3-8B", "vllm", 2, 16, "fp8", "mi300x", KernelOptimizer.FORGE, None)
    assert (workload.kind, workload.metric, workload.final_concs) == (WorkloadKind.AGENTX, GRADED_INTVTY_P50, (16,))


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
        ["run", *WORKLOAD_ARGS, "--max-gpu-hours", "-1"],
        ["run", *WORKLOAD_ARGS, "--max-gpu-hours", "nan"],
        ["run", *WORKLOAD_ARGS, "--max-gpu-hours", "inf"],
        ["run", *WORKLOAD_ARGS, "--max-llm-usd", "-1"],
        ["run", *WORKLOAD_ARGS, "--max-llm-usd", "nan"],
        ["run", *WORKLOAD_ARGS, "--max-llm-usd", "inf"],
        ["run", *WORKLOAD_ARGS, "--max-branches", "0"],
        ["run", *WORKLOAD_ARGS, "--max-parallel-tests", "0"],
        ["run", *WORKLOAD_ARGS, "--judge-interval-sec", "0"],
        ["run", *WORKLOAD_ARGS, "--conc", "many"],
        ["run", *WORKLOAD_ARGS[:-2]],
        ["run", *WORKLOAD_ARGS, "--gpu-type", "h100"],
        ["run", *WORKLOAD_ARGS, "--gpu-type", "mi325x"],
        ["run", *WORKLOAD_ARGS, "--gpu-type", "mi308x"],
        ["run", *WORKLOAD_ARGS, "--kernel-optimizer", "triton"],
        ["run", *WORKLOAD_ARGS, "--frontier-policy", "bandit"],
        ["run", *_without(WORKLOAD_ARGS, "--workload")],
        ["run", *WORKLOAD_ARGS, "--workload", "sharegpt"],
        ["run", *_without(WORKLOAD_ARGS, "--isl")],
        ["run", *_without(WORKLOAD_ARGS, "--osl")],
        ["run", *WORKLOAD_ARGS, "--isl", "0"],
        ["run", *AGENTX_ARGS, "--isl", "1024"],
        ["run", *AGENTX_ARGS, "--osl", "512"],
    ],
)
def test_usage_errors_exit_two(argv: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == cli.EXIT_USAGE
    assert not any(tmp_path.iterdir())


def test_each_specialist_prompt_file_is_one_user_direction_in_order(tmp_path: Path) -> None:
    first, second = tmp_path / "first.md", tmp_path / "second.md"
    first.write_text("  Fuse the MoE gate into the router kernel.\n", encoding="utf-8")
    second.write_text("Tune the attention block size.", encoding="utf-8")

    settings = _settings("--specialist-prompt", str(first), "--specialist-prompt", str(second))

    assert settings.user_directions == ("Fuse the MoE gate into the router kernel.", "Tune the attention block size.")
    assert _settings().user_directions == ()


def _prompt_file(tmp_path: Path, name: str, text: str | bytes | None) -> str:
    """A prompt file holding ``text``; none is written for ``None``."""

    path = tmp_path / name
    if isinstance(text, bytes):
        path.write_bytes(text)
    elif text is not None:
        path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.mark.parametrize(
    ("files", "extra", "error"),
    [
        ([("missing.md", None)], [], "cannot be read"),
        ([("empty.md", " \n")], [], "is empty"),
        ([("binary.md", b"\xff\xfe")], [], "cannot be read"),
        ([("a.md", "a"), ("b.md", "b")], ["--max-branches", "1"], "given 2 times, more than --max-branches 1"),
    ],
)
def test_a_bad_specialist_prompt_exits_two_before_creating_anything(
    files: list[tuple[str, str | bytes | None]],
    extra: list[str],
    error: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("USER_DATA_PATH", str(workspace))
    argv = [arg for name, text in files for arg in ("--specialist-prompt", _prompt_file(prompts, name, text))]

    with pytest.raises(SystemExit) as exc:
        cli.main(["run", *WORKLOAD_ARGS, *argv, *extra])

    assert exc.value.code == cli.EXIT_USAGE
    assert error in capsys.readouterr().err
    assert not any(workspace.iterdir())


def test_reusing_a_prelude_without_the_image_digest_exits_two_naming_the_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("USER_DATA_PATH", str(workspace))
    monkeypatch.delenv("HYPERLOOM_ACTIVE_IMAGE_DIGEST", raising=False)

    with pytest.raises(SystemExit) as exc:
        cli.main(["run", *WORKLOAD_ARGS, "--reuse-prelude", str(earlier)])

    assert exc.value.code == cli.EXIT_USAGE
    assert "HYPERLOOM_ACTIVE_IMAGE_DIGEST" in capsys.readouterr().err
    assert not workspace.exists()


@pytest.fixture(autouse=True)
def eight_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    """This machine shows eight cards, and the operator names no AgentX switch or benchmark YAML."""

    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.delenv(AGENTX_ENV, raising=False)
    monkeypatch.delenv(BENCHMARK_CONFIG_ENV, raising=False)


@pytest.mark.parametrize(
    ("workload", "switch"),
    [(WORKLOAD_ARGS, "1"), (WORKLOAD_ARGS, "true"), (AGENTX_ARGS, "0"), (AGENTX_ARGS, ""), (AGENTX_ARGS, "maybe")],
)
def test_an_agentx_switch_that_contradicts_the_workload_exits_two(
    workload: list[str],
    switch: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv(AGENTX_ENV, switch)

    with pytest.raises(SystemExit) as exc:
        cli.main(["run", *workload])

    assert exc.value.code == cli.EXIT_USAGE
    assert AGENTX_ENV in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("workload", [WORKLOAD_ARGS, AGENTX_ARGS])
def test_a_benchmark_yaml_the_operator_names_exits_two(
    workload: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv(BENCHMARK_CONFIG_ENV, "/operator/benchmark.yaml")

    with pytest.raises(SystemExit) as exc:
        cli.main(["run", *workload])

    assert exc.value.code == cli.EXIT_USAGE
    assert BENCHMARK_CONFIG_ENV in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def _magpie(monkeypatch: pytest.MonkeyPatch, **pin: str) -> None:
    """The installed Magpie's InferenceX module: ``pin`` its ``INFERENCEX_DEFAULT_REF`` (absent when empty), its
    project at the checkout's root."""

    magpie = SimpleNamespace(**pin, resolve_inferencex_root=str)
    monkeypatch.setattr(local_peers, "_magpie_inferencex", lambda: magpie)


@pytest.fixture
def ready_environment(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """No Experience KB, no Recipe KB, the launcher's image named, the shipped tool pins, an InferenceX checkout that is
    also the one a Magpie pinning an AgentX commit runs on, a model the price sheet lists, a writable RSI archive
    outside the test's workspace, and the framework's package installed in the image."""

    for name in ("HYPERLOOM_KB_URL", "HYPERLOOM_KB_TOKEN", "MAGPIE_REF", "INFERENCEX_REF"):
        monkeypatch.delenv(name, raising=False)
    for name, value in zip(IMAGE_VARS, IMAGE):
        monkeypatch.setenv(name, value)
    inferencex = tmp_path_factory.mktemp("InferenceX")
    (inferencex / "benchmarks").mkdir()
    (inferencex / "benchmarks" / "benchmark_lib.sh").write_text("")
    monkeypatch.setenv("INFERENCEX_PATH", str(inferencex))
    monkeypatch.setenv("MAGPIE_INFERENCEX_PATH", str(inferencex))
    _magpie(monkeypatch, INFERENCEX_DEFAULT_REF="4" * 40)
    monkeypatch.delenv("KB_STORE_URL", raising=False)
    monkeypatch.delenv("KB_STORE_TOKEN", raising=False)
    monkeypatch.setenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
    monkeypatch.setenv(ARCHIVE_ENV, str(tmp_path_factory.mktemp("rsi-archive")))
    installed = PackageBase("vllm", tmp_path_factory.mktemp("site-packages") / "vllm", "vllm", "0.0.0")
    monkeypatch.setattr(compile_module, "locate_installed_package", lambda layout: installed)


@pytest.fixture
def agent_cli(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> list[RunSettings]:
    """Stand-ins for both agent CLIs first on ``PATH``; returns the settings of every session the command would run,
    which runs none."""

    bin_dir = tmp_path_factory.mktemp("bin")
    for backend in (AGENT_BACKEND_CLAUDE, AGENT_BACKEND_CODEX):
        stand_in = bin_dir / backend
        stand_in.write_text("#!/bin/sh\nexit 0\n")
        stand_in.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    runs: list[RunSettings] = []

    def run_session(settings: RunSettings, peer_factory: PeerFactory, services: object) -> int:
        runs.append(settings)
        return 0

    def final_only(persisted: PersistedRun, peer_factory: PeerFactory, services: object) -> int:
        runs.append(persisted.settings)
        return 0

    monkeypatch.setattr(cli, "run_session", run_session)
    monkeypatch.setattr(cli, "final_only", final_only)
    return runs


@pytest.mark.usefixtures("ready_environment")
def test_a_ready_environment_without_an_experience_kb_runs_the_session_on_the_model_backed_specialists(
    agent_cli: list[RunSettings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))

    assert cli.main(["run", *WORKLOAD_ARGS]) == 0

    assert agent_cli == [_settings()]


@pytest.mark.parametrize(("workload", "switch"), [(AGENTX_ARGS, None), (AGENTX_ARGS, "1"), (WORKLOAD_ARGS, "0")])
@pytest.mark.usefixtures("ready_environment")
def test_an_agentx_switch_that_agrees_with_the_workload_runs_the_session(
    workload: list[str],
    switch: str | None,
    agent_cli: list[RunSettings],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    if switch is not None:
        monkeypatch.setenv(AGENTX_ENV, switch)

    assert cli.main(["run", *workload]) == 0

    assert agent_cli == [_settings(workload=workload)]


@pytest.mark.usefixtures("ready_environment", "agent_cli")
def test_run_without_the_agent_cli_fails_loudly_before_creating_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))

    assert cli.main(["run", *WORKLOAD_ARGS]) == cli.EXIT_FAILED

    captured = capsys.readouterr()
    assert "CLI the specialists run is not on PATH" in captured.err
    assert captured.out == ""
    assert not any(tmp_path.iterdir())


@pytest.mark.usefixtures("ready_environment", "agent_cli")
def test_tool_pins_that_cannot_be_read_fail_loudly_before_creating_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("MAGPIE_REF", "main")

    assert cli.main(["run", *WORKLOAD_ARGS]) == cli.EXIT_FAILED

    captured = capsys.readouterr()
    assert "measurement pins" in captured.err and "MAGPIE_REF" in captured.err
    assert captured.out == ""
    assert not any(tmp_path.iterdir())


@pytest.mark.usefixtures("ready_environment", "agent_cli")
def test_an_agentx_run_on_a_magpie_that_pins_no_inferencex_commit_fails_loudly_before_creating_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    _magpie(monkeypatch)

    assert cli.main(["run", *AGENTX_ARGS]) == cli.EXIT_FAILED

    captured = capsys.readouterr()
    assert "INFERENCEX_DEFAULT_REF" in captured.err and "MAGPIE_REF" in captured.err
    assert captured.out == ""
    assert not any(tmp_path.iterdir())


@pytest.mark.usefixtures("ready_environment", "agent_cli")
def test_an_unnamed_image_fails_loudly_before_creating_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    for name in IMAGE_VARS:
        monkeypatch.delenv(name)

    assert cli.main(["run", *WORKLOAD_ARGS]) == cli.EXIT_FAILED

    captured = capsys.readouterr()
    assert "measurement pins" in captured.err and all(name in captured.err for name in IMAGE_VARS)
    assert captured.out == ""
    assert not any(tmp_path.iterdir())


def _codex_specialists(monkeypatch: pytest.MonkeyPatch, model: str) -> None:
    """Credential only the OpenAI side, so the specialists run the codex CLI on ``model`` while the judge keeps
    ``CLAUDE_MODEL``."""

    for name in (*ANTHROPIC_CREDENTIAL_ENV_ORDER, "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("CODEX_MODEL", model)


@pytest.mark.parametrize(
    ("judge", "specialists", "warned"),
    [
        ("no-such-model", DEFAULT_CODEX_MODEL, "judge runs on no-such-model"),
        (DEFAULT_CLAUDE_MODEL, "no-such-model", "the specialists run on no-such-model"),
    ],
)
@pytest.mark.usefixtures("ready_environment")
def test_a_model_without_a_price_runs_the_session_on_its_tokens_and_says_so(
    judge: str,
    specialists: str,
    warned: str,
    agent_cli: list[RunSettings],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    _codex_specialists(monkeypatch, specialists)
    monkeypatch.setenv("CLAUDE_MODEL", judge)

    assert cli.main(["run", *WORKLOAD_ARGS]) == 0

    assert len(agent_cli) == 1
    assert f"{warned}, which the price sheet does not list" in caplog.text


@pytest.mark.parametrize(
    ("judge", "specialists", "named"),
    [
        ("no-such-model", DEFAULT_CODEX_MODEL, "the judge's model no-such-model"),
        (DEFAULT_CLAUDE_MODEL, "no-such-model", "the specialists' model no-such-model"),
    ],
)
@pytest.mark.usefixtures("ready_environment")
def test_a_dollar_cap_over_a_model_without_a_price_exits_two_before_creating_anything(
    judge: str,
    specialists: str,
    named: str,
    agent_cli: list[RunSettings],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    _codex_specialists(monkeypatch, specialists)
    monkeypatch.setenv("CLAUDE_MODEL", judge)

    with pytest.raises(SystemExit) as exc:
        cli.main(["run", *WORKLOAD_ARGS, "--max-llm-usd", "5"])

    assert exc.value.code == cli.EXIT_USAGE
    captured = capsys.readouterr()
    assert f"--max-llm-usd cannot count the dollars of {named}, which the price sheet does not list" in captured.err
    assert captured.out == ""
    assert agent_cli == []
    assert not any(tmp_path.iterdir())


@pytest.mark.usefixtures("ready_environment")
def test_a_specialist_prompt_runs_the_session_on_its_direction(
    agent_cli: list[RunSettings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("USER_DATA_PATH", str(workspace))
    prompt = _prompt_file(tmp_path, "direction.md", "Fuse the MoE gate into the router kernel.")

    assert cli.main(["run", *WORKLOAD_ARGS, "--specialist-prompt", prompt]) == 0

    assert [settings.user_directions for settings in agent_cli] == [("Fuse the MoE gate into the router kernel.",)]


@pytest.mark.usefixtures("ready_environment")
def test_a_dollar_cap_over_priced_models_runs_the_session(
    agent_cli: list[RunSettings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    _codex_specialists(monkeypatch, DEFAULT_CODEX_MODEL)

    assert cli.main(["run", *WORKLOAD_ARGS, "--max-llm-usd", "5"]) == 0

    assert [run.max_llm_usd for run in agent_cli] == [5.0]


@pytest.mark.usefixtures("ready_environment", "agent_cli")
def test_a_framework_whose_source_cannot_be_laid_out_fails_loudly_before_creating_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    framework_at = WORKLOAD_ARGS.index("--framework") + 1
    argv = ["run", *WORKLOAD_ARGS[:framework_at], "tensorrt-llm", *WORKLOAD_ARGS[framework_at + 1 :]]

    assert cli.main(argv) == cli.EXIT_FAILED

    captured = capsys.readouterr()
    assert "tensorrt-llm has no package layout" in captured.err
    assert captured.out == ""
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("unset", ["KB_STORE_URL", "KB_STORE_TOKEN"])
@pytest.mark.usefixtures("ready_environment", "agent_cli")
def test_a_half_configured_recipe_kb_fails_loudly_before_creating_anything(
    unset: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("KB_STORE_URL", "http://127.0.0.1:9/kb-store")
    monkeypatch.setenv("KB_STORE_TOKEN", "dummy-token")
    monkeypatch.delenv(unset)

    assert cli.main(["run", *WORKLOAD_ARGS]) == cli.EXIT_FAILED

    captured = capsys.readouterr()
    assert "Recipe KB" in captured.err and unset in captured.err
    assert captured.out == ""
    assert not any(tmp_path.iterdir())


@pytest.mark.usefixtures("ready_environment", "agent_cli")
def test_the_rsi_archive_defaults_to_the_shared_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.delenv(ARCHIVE_ENV)

    cli.main(["run", *WORKLOAD_ARGS])

    assert os.environ[ARCHIVE_ENV] == cli.RSI_ARCHIVE_DEFAULT == "/shared_nfs/hyperloom/rsi-archive/"


@pytest.mark.usefixtures("ready_environment", "agent_cli")
def test_a_named_rsi_archive_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "archive"
    archive.mkdir()
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))
    monkeypatch.setenv(ARCHIVE_ENV, str(archive))

    cli.main(["run", *WORKLOAD_ARGS])

    assert os.environ[ARCHIVE_ENV] == str(archive)


@pytest.mark.usefixtures("ready_environment", "agent_cli")
def test_an_rsi_archive_that_is_not_a_writable_directory_fails_loudly_before_creating_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv(ARCHIVE_ENV, str(tmp_path.parent / "no-such-archive"))

    assert cli.main(["run", *WORKLOAD_ARGS]) == cli.EXIT_FAILED

    captured = capsys.readouterr()
    assert ARCHIVE_ENV in captured.err and "no-such-archive" in captured.err
    assert not any(tmp_path.iterdir())


def test_importing_the_command_and_the_session_loads_no_main_flow_runtime() -> None:
    probe = subprocess.run([sys.executable, "-c", IMPORT_PROBE], capture_output=True, text=True, check=True)
    assert probe.stdout.strip() == "[]"


def _ended_before_final(root: Path) -> SessionLayout:
    """A session directory as a run that ended before FINAL leaves it: the manifest of :func:`_settings`, the root
    recipe PRELUDE committed, and the report CLOSE wrote with no final choice in it."""

    layout = SessionLayout(root)
    layout.reports.mkdir(parents=True)
    layout.prelude.mkdir(parents=True)
    _manifest(layout)
    _write(layout.root_recipe, {"components": []})
    _report(layout)
    return layout


def _manifest(layout: SessionLayout, **fields: object) -> None:
    _write(
        layout.manifest,
        {
            "code_version": encode(code_version()),
            "session_id": layout.session_id,
            "settings": encode(_settings()),
            "policy_versions": {"frontier": "default"},
            **fields,
        },
    )


def _write(path: Path, document: dict[str, object]) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def _report(layout: SessionLayout, **fields: object) -> None:
    _write(layout.report, {"stop_reason": "time_exhausted", "final": None, **fields})


def _tree(root: Path) -> list[Path]:
    return sorted(root.rglob("*"))


def _killed(layout: SessionLayout, scope: str) -> None:
    """The session killed with the machine it ran on: no report, and this live process recorded under ``scope``."""

    layout.report.unlink()
    record = ProcessRecord(os.getpid(), os.getpgid(0), time.time(), SCHEDULER_BRANCH, ("mb",), scope)
    layout.pids.write_text(f"{json.dumps(asdict(record))}\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("damage", "error"),
    [
        (lambda layout: layout.manifest.unlink(), "is no multi-branch session that ran"),
        (lambda layout: _manifest(layout, code_version={"package": "0.0", "git_sha": "beef"}), "ran on CodeVersion"),
        (lambda layout: layout.root_recipe.unlink(), "committed no root"),
        (lambda layout: _killed(layout, local_owner_scope()), "is still running"),
        (lambda layout: _report(layout, final={"chosen": {"commit": "c9"}}), "already chose c9"),
        (lambda layout: _report(layout, stop_reason="prelude_baseline_failed"), "PRELUDE failed"),
    ],
    ids=["no manifest", "another build", "no root", "still running", "final ran", "prelude failed"],
)
@pytest.mark.usefixtures("ready_environment")
def test_a_session_final_cannot_run_over_exits_two_and_changes_nothing(
    damage: Callable[[SessionLayout], None],
    error: str,
    agent_cli: list[RunSettings],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _ended_before_final(tmp_path / "MB-20261020T141241Z-64f775fe")
    damage(layout)
    before = _tree(layout.root)

    with pytest.raises(SystemExit) as exc:
        cli.main(["final-only", str(layout.root)])

    assert exc.value.code == cli.EXIT_USAGE
    captured = capsys.readouterr()
    assert error in captured.err and captured.out == ""
    assert agent_cli == [] and _tree(layout.root) == before


@pytest.mark.usefixtures("ready_environment", "agent_cli")
def test_final_only_finishes_a_session_that_was_killed_before_it_could_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _ended_before_final(tmp_path / "MB-20261020T141241Z-64f775fe")
    _killed(layout, ANOTHER_POD)
    finished: list[PersistedRun] = []

    def final_only(persisted: PersistedRun, peer_factory: PeerFactory, services: object) -> int:
        finished.append(persisted)
        return 1

    monkeypatch.setattr(cli, "final_only", final_only)

    assert cli.main(["final-only", str(layout.root)]) == 1

    [persisted] = finished
    assert (persisted.ended, persisted.settings) == (SessionStopReason.KILLED, _settings())


@pytest.mark.usefixtures("ready_environment")
def test_final_only_over_a_path_that_is_no_directory_exits_two(
    agent_cli: list[RunSettings], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "no-such-session"

    with pytest.raises(SystemExit) as exc:
        cli.main(["final-only", str(missing)])

    assert exc.value.code == cli.EXIT_USAGE
    assert f"{missing} is not a directory" in capsys.readouterr().err
    assert agent_cli == [] and not missing.exists()


def test_a_final_only_holds_the_session_against_a_second_one_and_releases_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    layout = _ended_before_final(tmp_path / "MB-20261020T141241Z-64f775fe")
    persisted = persisted_run(layout.root)

    async def while_it_runs(run: PersistedRun, peer_factory: object, services: object) -> int:
        assert layout.final_only_lock.read_text(encoding="utf-8") == f"{os.getpid()}\n"
        assert cli.final_only(run, peer_factory, services) == cli.EXIT_USAGE
        return 0

    monkeypatch.setattr(cli, "_final_only", while_it_runs)

    assert cli.final_only(persisted, None, None) == 0

    assert str(layout.final_only_lock) in capsys.readouterr().err
    assert not layout.final_only_lock.exists()


@pytest.mark.usefixtures("ready_environment")
def test_final_only_finishes_the_session_on_the_settings_its_manifest_holds(
    agent_cli: list[RunSettings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("USER_DATA_PATH", str(workspace))
    layout = _ended_before_final(tmp_path / "MB-20261020T141241Z-64f775fe")

    assert cli.main(["final-only", str(layout.root)]) == 0

    assert agent_cli == [_settings()]
    assert not any(workspace.iterdir())
