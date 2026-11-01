# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the ``hyperloom-multibranch`` command line: options, defaults, validation and exit codes.

``final-only`` is the emergency recovery of a session that ended before FINAL, one killed with its machine included:
it takes that session's directory and nothing else, refuses before building anything when the directory cannot
support FINAL -- a session another build ran, and one still running, included -- and holds the directory against a
second recovery while it works.

``hosted run`` maps the campaign spec a host injected onto the options of ``run`` and builds the session's settings as
``run`` does, measuring on the grant's cards and ending a CLOSE grace before the grant does; it tells the host in
``exit.json`` that the run is over once CLOSE ran or when nothing a restart changes keeps a session from running, and
leaves the host to start a crashed run again, which then finishes the session the first start began.

``run --measurement hosted`` measures in the sandboxes of a grant handed to it in the environment, without a campaign
spec, a run id or a state directory, refuses a missing grant variable by name and what would run on this machine, and
stops before the grant ends as ``hosted run`` does.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
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
from hyperloom.common.outside_facts.image_check import (
    MODE_ENV,
    FactResult,
    ImageCheckFailed,
    ImageCheckMode,
    ImageCheckReport,
)
from hyperloom.common.outside_facts.image_check_eval import Status
from hyperloom.common.owner_scope import local_owner_scope
from hyperloom.common.perf_metric import GRADED_INTVTY_P50
from hyperloom.common.outside_facts.rule_pack import rule_pack_identity
from hyperloom.measure import compile as compile_module
from hyperloom.measure import points
from hyperloom.measure.compile import PackageBase
from hyperloom.measure.points import PointError
from hyperloom.measure.runner.cache_build import CacheBuildError
from hyperloom_multibranch import cli
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.blackboard.codec import encode
from hyperloom_multibranch.blackboard.knowledge.reference_leads import INFERENCEX_REFERENCE
from hyperloom_multibranch.blackboard.rsi.archive import ARCHIVE_ENV
from hyperloom_multibranch.scheduling.budget import MB_CLOSE_GRACE_CAP_SECONDS, MB_DEFAULT_MAX_HOURS
from hyperloom_multibranch.scheduling.hosted_run import GRANT_MARGIN_SECONDS
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling import review
from hyperloom_multibranch.scheduling.layout import REVIEW_DIR, SessionLayout, create_session
from hyperloom_multibranch.scheduling.lifecycle import PersistedRun, persisted_run
from hyperloom_multibranch.scheduling.manifest import SessionManifest, code_version
from hyperloom_multibranch.scheduling import local_peers
from hyperloom_multibranch.scheduling.local_peers import AGENTX_ENV
from hyperloom_multibranch.scheduling.peers import PeerFactory
from hyperloom_multibranch.scheduling.processes import SCHEDULER_BRANCH, ProcessRecord
from hyperloom_multibranch.scheduling.reference import ReferenceSource
from hyperloom_multibranch.scheduling.tests.checkouts import knowledge_checkout
from hyperloom_multibranch.scheduling.roster import RosterCaps
from hyperloom_multibranch.scheduling.settings import (
    AGENT_PORT,
    DOSSIER_MINUTES,
    InferenceXShape,
    KernelOptimizer,
    PlaybookMode,
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


def test_reference_help_says_which_name_briefs_inferencex_reference_launches(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        cli.main(["run", "--help"])

    words = " ".join(capsys.readouterr().out.split())
    assert "leads from InferenceX's reference launches for the workload only when an InferenceX checkout" in words
    assert f"is given as --reference {INFERENCEX_REFERENCE}=DIR;" in words


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


def test_a_session_plays_the_champion_playbook_unless_it_is_a_challenger_session() -> None:
    assert _settings().playbook is PlaybookMode.CHAMPION
    assert _settings("--playbook", "challenger").playbook is PlaybookMode.CHALLENGER
    with pytest.raises(SystemExit):
        _settings("--playbook", "favourite")


def test_run_options_override_the_defaults() -> None:
    settings = _settings(
        "--gpu-type", "mi355x", "--max-hours", "12", "--max-gpu-hours", "40", "--max-llm-usd", "25.5",
        "--max-branches", "6", "--max-parallel-tests", "3", "--max-k-per-branch", "3", "--judge-interval-sec", "120",
    )  # fmt: skip
    assert settings.workload.gpu_type == "mi355x"
    assert (settings.max_hours, settings.max_gpu_hours, settings.max_llm_usd) == (12.0, 40.0, 25.5)
    assert settings.caps == RosterCaps(6, 3, 3)
    assert settings.judge_interval_seconds == 120.0


def test_a_gpu_type_names_the_same_board_in_any_case() -> None:
    assert _settings("--gpu-type", "MI355X").workload.gpu_type == "mi355x"


def test_an_unknown_gpu_type_is_refused_naming_the_types_a_session_runs_on(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", *WORKLOAD_ARGS, "--gpu-type", "MI325X"])

    assert exc.value.code == cli.EXIT_USAGE
    assert "invalid choice: 'mi325x' (choose from 'mi300x', 'mi350x', 'mi355x')" in capsys.readouterr().err


def test_an_agentx_run_has_no_sequence_lengths_and_is_graded_on_p50_interactivity_at_its_conc_alone() -> None:
    workload = _settings(workload=AGENTX_ARGS).workload

    assert workload == Workload("/models/Llama-3-8B", "vllm", 2, 16, "fp8", "mi300x", KernelOptimizer.FORGE, None)
    assert (workload.kind, workload.metric, workload.final_concs) == (WorkloadKind.AGENTX, GRADED_INTVTY_P50, (16,))


def test_an_agentx_run_is_served_at_the_ep_it_names_and_at_ep_1_by_default() -> None:
    assert _settings(workload=AGENTX_ARGS).workload.ep == 1
    assert _settings("--ep", "8", workload=AGENTX_ARGS).workload.ep == 8


def test_run_help_names_the_ep_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["run", "--help"])

    assert "--ep EP" in capsys.readouterr().out


def test_an_agentx_model_directory_is_recorded_by_its_resolved_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "models" / "Llama-3-8B").mkdir(parents=True)
    (tmp_path / "llama").symlink_to(tmp_path / "models" / "Llama-3-8B")
    monkeypatch.chdir(tmp_path)
    agentx = ["--model", "llama", *AGENTX_ARGS[2:]]
    inferencex = ["--model", "llama", *WORKLOAD_ARGS[2:]]

    assert _settings(workload=agentx).workload.model == str(tmp_path.resolve() / "models" / "Llama-3-8B")
    assert _settings(workload=inferencex).workload.model == "llama"


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
        ["run", *_without(WORKLOAD_ARGS, "--workload")],
        ["run", *WORKLOAD_ARGS, "--workload", "sharegpt"],
        ["run", *_without(WORKLOAD_ARGS, "--isl")],
        ["run", *_without(WORKLOAD_ARGS, "--osl")],
        ["run", *WORKLOAD_ARGS, "--isl", "0"],
        ["run", *AGENTX_ARGS, "--isl", "1024"],
        ["run", *AGENTX_ARGS, "--osl", "512"],
        ["run", *AGENTX_ARGS, "--ep", "0"],
        ["run", *WORKLOAD_ARGS, "--ep", "2"],
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


def test_each_autonomous_agent_is_one_agent_branch_in_order() -> None:
    settings = _settings(
        "--autonomous-agent", "https://github.com/acme/agent@v1", "--autonomous-agent", "git@github.com:acme/b.git"
    )

    assert settings.agents == ("https://github.com/acme/agent@v1", "git@github.com:acme/b.git")
    assert _settings().agents == ()


@pytest.mark.parametrize(
    "extra",
    [
        ["--autonomous-agent", "https://user:ghp_secret@github.com/acme/agent"],
        ["--autonomous-agent", "https://ghp_secret@github.com/acme/agent"],
        ["--autonomous-agent", "git+https://ghp_secret@github.com/acme/agent"],
        ["--max-branches", "1", "--autonomous-agent", "/a", "--autonomous-agent", "/b"],
    ],
)
def test_an_agent_source_with_credentials_or_one_too_many_is_a_usage_error(extra: list[str]) -> None:
    with pytest.raises(ValueError, match="--autonomous-agent"):
        _settings(*extra)


@pytest.mark.parametrize("source", ["ssh://git@github.com/acme/agent.git", "git@github.com:acme/agent.git@v1"])
def test_an_agent_source_whose_user_is_an_ssh_login_is_accepted(source: str) -> None:
    assert _settings("--autonomous-agent", source).agents == (source,)


def test_an_ssh_source_with_a_password_carries_credentials() -> None:
    with pytest.raises(ValueError, match="credentials"):
        _settings("--autonomous-agent", "ssh://git:secret@github.com/acme/agent.git")


def test_the_agent_directions_map_is_kept_by_its_absolute_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "map.json").write_text(json.dumps({"directions": [{"id": "m-attn", "title": "attention"}]}))

    settings = _settings("--autonomous-agent", "/agents/arbor", "--agent-directions", "map.json")

    assert settings.agent_directions == str(tmp_path / "map.json")
    assert _settings("--autonomous-agent", "/agents/arbor").agent_directions == ""


def test_the_dossier_pass_has_ninety_minutes_unless_the_run_gives_it_others_and_none_below_zero() -> None:
    assert _settings("--autonomous-agent", "/agents/arbor").agent_dossier_minutes == DOSSIER_MINUTES == 90.0
    assert _settings("--agent-dossier-minutes", "0").agent_dossier_minutes == 0.0
    with pytest.raises(ValueError, match="--agent-dossier-minutes -5 is negative"):
        _settings("--agent-dossier-minutes", "-5")


def test_the_agent_pack_is_kept_by_name_and_empty_without_one() -> None:
    assert _settings("--autonomous-agent", "/agents/arbor", "--agent-pack", "agentx-qwen38-1").agent_pack == (
        "agentx-qwen38-1"
    )
    assert _settings("--autonomous-agent", "/agents/arbor").agent_pack == ""


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--agent-pack", "agentx-qwen38-1"], "needs an --autonomous-agent"),
        (["--autonomous-agent", "/agents/arbor", "--agent-pack", "packs/agentx"], "not the name of one pack"),
        (["--autonomous-agent", "/agents/arbor", "--agent-pack", ".."], "not the name of one pack"),
        (["--autonomous-agent", "/agents/arbor", "--agent-pack", ""], "not the name of one pack"),
    ],
)
def test_an_agent_pack_without_an_agent_or_not_one_directory_is_a_usage_error(extra: list[str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _settings(*extra)


def test_the_agent_launch_file_is_kept_by_its_absolute_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "launch.json").write_text(json.dumps({"server_flags": "--tp 1", "server_env": {}}))

    settings = _settings("--autonomous-agent", "/agents/arbor", "--agent-launch", "launch.json")

    assert settings.agent_launch == str(tmp_path / "launch.json")
    assert _settings("--autonomous-agent", "/agents/arbor").agent_launch == ""


@pytest.mark.parametrize(
    ("document", "agent", "match"),
    [
        ({"server_flags": "--tp 1", "server_env": {}}, False, "needs an --autonomous-agent"),
        ({"server_env": {}}, True, "no server_flags"),
        (None, True, "cannot be read"),
    ],
    ids=["no-agent", "no-flags", "no-file"],
)
def test_an_agent_launch_without_an_agent_or_unusable_is_a_usage_error(
    tmp_path: Path, document: object, agent: bool, match: str
) -> None:
    path = tmp_path / "launch.json"
    if document is not None:
        path.write_text(json.dumps(document))
    extra = ["--autonomous-agent", "/agents/arbor"] if agent else []

    with pytest.raises(ValueError, match=match):
        _settings(*extra, "--agent-launch", str(path))


def test_the_agent_port_is_the_run_s_own_or_the_shared_default() -> None:
    assert _settings("--autonomous-agent", "/agents/arbor", "--agent-port", "32100").agent_port == 32100
    assert _settings("--autonomous-agent", "/agents/arbor").agent_port == AGENT_PORT


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--agent-port", "32100"], "needs an --autonomous-agent"),
        (["--autonomous-agent", "/agents/arbor", "--agent-port", "0"], "is not a port"),
        (["--autonomous-agent", "/agents/arbor", "--agent-port", "65536"], "is not a port"),
    ],
    ids=["no-agent", "zero", "past-the-last-port"],
)
def test_an_agent_port_without_an_agent_or_that_is_no_port_is_a_usage_error(extra: list[str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _settings(*extra)


def test_a_run_stops_after_its_map_only_when_asked() -> None:
    assert _settings("--autonomous-agent", "/agents/arbor", "--stop-after-map").stop_after_map
    assert not _settings("--autonomous-agent", "/agents/arbor").stop_after_map


def test_the_review_gate_is_on_unless_the_run_turns_it_off() -> None:
    assert _settings("--autonomous-agent", "/agents/arbor").review_gate
    assert not _settings("--autonomous-agent", "/agents/arbor", "--no-review-gate").review_gate


def test_a_person_shows_and_approves_a_session_s_review_and_a_second_decision_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    session, map_file = tmp_path / "MB-session", tmp_path / "directions.json"
    map_file.write_text(json.dumps({"directions": [{"id": "m-attn", "title": "attention"}]}))
    review.request(session / REVIEW_DIR, session, map_file, None, "dossier pass", 1793500000.0)

    assert cli.main(["review", "--session", str(session), "show"]) == 0
    shown = capsys.readouterr().out
    assert "`m-attn`" in shown and "Status: pending" in shown
    assert cli.main(["review", "--session", str(session), "approve", "--by", "alice", "--note", "ok"]) == 0
    assert "approved by alice" in capsys.readouterr().out
    assert cli.main(["review", "--session", str(session), "reject"]) == review.EXIT_REFUSED
    assert "the review is already approved" in capsys.readouterr().err


def test_a_review_command_for_a_session_with_no_review_is_refused(tmp_path: Path) -> None:
    assert cli.main(["review", "--session", str(tmp_path), "show"]) == review.EXIT_REFUSED
    assert cli.main(["review", "--session", str(tmp_path), "reject", "--note", "no"]) == review.EXIT_REFUSED


def test_stopping_after_the_map_without_an_agent_or_with_a_map_given_is_a_usage_error(tmp_path: Path) -> None:
    given = tmp_path / "map.json"
    given.write_text(json.dumps({"directions": [{"id": "m-attn"}]}))

    with pytest.raises(ValueError, match="needs an --autonomous-agent"):
        _settings("--stop-after-map")
    with pytest.raises(ValueError, match="already gives them"):
        _settings("--autonomous-agent", "/agents/arbor", "--agent-directions", str(given), "--stop-after-map")


def _reviewed(tmp_path: Path) -> tuple[Path, Path]:
    """A reviewed map and the dossier it was made with, as a ``--stop-after-map`` session keeps them."""

    given, dossier = tmp_path / "directions.json", tmp_path / "dossier" / "qwen3-8b"
    given.write_text(json.dumps({"directions": [{"id": "m-attn"}]}))
    dossier.mkdir(parents=True)
    (dossier / "README.md").write_text("# dossier\n")
    return given, dossier


def test_the_reviewed_dossier_is_kept_by_its_absolute_path_beside_its_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    given, _ = _reviewed(tmp_path)
    agent = ["--autonomous-agent", "/agents/arbor", "--agent-directions", str(given)]

    assert _settings(*agent, "--agent-dossier", "dossier/qwen3-8b").agent_dossier == str(
        tmp_path / "dossier" / "qwen3-8b"
    )
    assert _settings(*agent).agent_dossier == ""


def test_a_dossier_without_its_map_or_that_is_no_dossier_is_a_usage_error(tmp_path: Path) -> None:
    given, dossier = _reviewed(tmp_path)

    with pytest.raises(ValueError, match="with the --agent-directions map"):
        _settings("--autonomous-agent", "/agents/arbor", "--agent-dossier", str(dossier))
    with pytest.raises(ValueError, match="is no dossier"):
        _settings(
            "--autonomous-agent", "/agents/arbor", "--agent-directions", str(given), "--agent-dossier", str(tmp_path)
        )


@pytest.mark.parametrize(
    ("document", "agent", "match"),
    [
        ({"directions": [{"id": "m-attn"}]}, False, "needs an --autonomous-agent"),
        ({"directions": []}, True, "holds no directions"),
        ({"map": []}, True, "holds no directions"),
        (None, True, "cannot be read"),
    ],
)
def test_an_agent_directions_map_without_an_agent_or_directions_is_a_usage_error(
    tmp_path: Path, document: object, agent: bool, match: str
) -> None:
    path = tmp_path / "map.json"
    path.write_text("{not json" if document is None else json.dumps(document))
    agents = ["--autonomous-agent", "/agents/arbor"] if agent else []

    with pytest.raises(ValueError, match=match):
        _settings(*agents, "--agent-directions", str(path))


def test_dev_leases_lend_specialists_dev_sandboxes_and_default_to_none() -> None:
    assert _settings().dev_leases == 0
    assert _settings("--dev-leases", "2").dev_leases == 2
    with pytest.raises(ValueError, match="--dev-leases"):
        _settings("--dev-leases", "-1")


def test_each_reference_is_a_named_directory_pinned_at_the_commit_it_was_taken_from(tmp_path: Path) -> None:
    commit = knowledge_checkout(tmp_path / "GEAK", "gemm.md")

    settings = _settings("--reference", f"geak-tuning={tmp_path / 'GEAK'}")

    assert settings.references == (ReferenceSource("geak-tuning", str(tmp_path / "GEAK"), commit),)
    assert _settings().references == ()


def test_the_same_reference_name_twice_exits_two_before_creating_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    knowledge_checkout(tmp_path / "GEAK", "gemm.md")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("USER_DATA_PATH", str(workspace))
    spec = f"geak-tuning={tmp_path / 'GEAK'}"

    with pytest.raises(SystemExit) as exc:
        cli.main(["run", *WORKLOAD_ARGS, "--reference", spec, "--reference", spec])

    assert exc.value.code == cli.EXIT_USAGE
    assert "--reference is given one NAME twice" in capsys.readouterr().err
    assert not any(workspace.iterdir())


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
def agent_binaries(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Stand-ins for both agent CLIs first on ``PATH``."""

    bin_dir = tmp_path_factory.mktemp("bin")
    for backend in (AGENT_BACKEND_CLAUDE, AGENT_BACKEND_CODEX):
        stand_in = bin_dir / backend
        stand_in.write_text("#!/bin/sh\nexit 0\n")
        stand_in.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")


@pytest.fixture
def agent_cli(agent_binaries: None, monkeypatch: pytest.MonkeyPatch) -> list[RunSettings]:
    """The :func:`agent_binaries`; returns the settings of every session the command would run, which runs none."""

    runs: list[RunSettings] = []

    def run_session(settings: RunSettings, peer_factory: PeerFactory, services: object, *report_root: object) -> int:
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


@pytest.mark.usefixtures("ready_environment", "agent_binaries")
def test_a_canonical_measurement_that_cannot_be_taken_fails_like_the_other_start_up_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    for name in (*local_peers.workload_environment(_settings().workload), BENCHMARK_CONFIG_ENV):
        monkeypatch.setenv(name, "")
    refusal = "the canonical measurement runs sglang_mi355x.sh from the InferenceX checkout, which has no such file"

    def no_launcher(*args: object, **kwargs: object) -> object:
        raise PointError(refusal)

    monkeypatch.setattr(points, "default_canonical", no_launcher)

    assert cli.main(["run", *WORKLOAD_ARGS]) == cli.EXIT_FAILED

    captured = capsys.readouterr()
    assert captured.err == f"{cli.PROG}: cannot run a session, the canonical measurement cannot be taken: {refusal}\n"
    session_dir = Path(captured.out.strip())
    assert session_dir.is_dir() and tmp_path in session_dir.parents


@pytest.mark.usefixtures("ready_environment", "agent_cli")
def test_a_strict_image_check_that_fails_is_refused_like_the_other_start_up_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    failing = FactResult("sglang.renamed-flag", "sglang", Status.FAIL, (), "the flag is gone")
    report = ImageCheckReport(ImageCheckMode.STRICT, rule_pack_identity(), "", {}, (failing,))

    def session_refused(settings: RunSettings, peer_factory: PeerFactory, services: object) -> int:
        raise ImageCheckFailed(report)

    monkeypatch.setattr(cli, "run_session", session_refused)

    assert cli.main(["run", *WORKLOAD_ARGS]) == cli.EXIT_FAILED

    expected = f"image check failed under {MODE_ENV}=strict: sglang.renamed-flag"
    assert capsys.readouterr().err == f"{cli.PROG}: cannot run a session, {expected}\n"


@pytest.mark.usefixtures("ready_environment", "agent_cli")
def test_an_image_check_mode_that_is_no_mode_exits_two_before_creating_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv(MODE_ENV, "loud")

    with pytest.raises(SystemExit) as exited:
        cli.main(["run", *WORKLOAD_ARGS])

    assert exited.value.code == cli.EXIT_USAGE
    assert f"{MODE_ENV}='loud'" in capsys.readouterr().err
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
def test_an_unset_rsi_archive_is_refused_naming_the_variable_before_anything_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.delenv(ARCHIVE_ENV)

    assert cli.main(["run", *WORKLOAD_ARGS]) == cli.EXIT_FAILED

    assert f"{ARCHIVE_ENV} is not set" in capsys.readouterr().err
    assert ARCHIVE_ENV not in os.environ, "no archive is assumed on the deployment's behalf"
    assert not any(tmp_path.iterdir())


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
    """The manifest a run of :func:`_settings` wrote, with ``fields`` written over it to damage one of them."""

    settings = _settings()
    document = encode(
        SessionManifest(
            code_version=code_version(),
            rule_pack=dict(rule_pack_identity()),
            session_id=layout.session_id,
            started_at=0.0,
            deadline=3600.0,
            settings=settings,
            metric=settings.workload.metric,
            policy_versions={"frontier": "default"},
        )
    )
    _write(layout.manifest, {**document, **fields})


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


#: A grant of four GPUs that has a day left.
GRANT = {"max_sandboxes": 4, "max_gpus": 4, "gpu_hours": 100, "max_runtime_sec": 86400}
#: The campaign spec of :data:`WORKLOAD_ARGS` as a multibranch run, as a host injects it.
SPEC: dict[str, object] = {
    "schema": "hyperloom.campaign-spec/v1",
    "flow": "multibranch",
    "model": {"path": "/models/Llama-3-8B"},
    "framework": {"name": "vllm", "image": IMAGE[0]},
    "hardware": {"gpu_type": "mi300x", "tp": 2},
    "workload": {"conc": 16, "isl": 1024, "osl": 512, "precision": "fp8"},
    "objective": {"target_gain_pct": 10},
    "budget": {"max_hours": 2, "max_gpu_hours": 12, "max_llm_usd": 40},
    "placement": {"profile": "hosted-gpu", "max_parallel_tests": 1},
    "multibranch": {"workload": "inferencex", "max_branches": 3},
}
#: The ``run`` options the same session is spelled with on the command line.
SPEC_RUN_ARGS = (
    "--max-hours", "2", "--max-gpu-hours", "12", "--max-llm-usd", "40", "--max-branches", "3",
    "--max-parallel-tests", "1",
)  # fmt: skip


@dataclass
class Host:
    """A host's inputs to ``hosted run``: its state directory, the workspace sessions are created in, and the delay of
    every grant stop armed."""

    state_dir: Path
    workspace: Path
    spec_file: Path
    stops: list[float]

    def exit_document(self) -> dict[str, object]:
        return json.loads((self.state_dir / "exit.json").read_text(encoding="utf-8"))

    def started(self, seconds_ago: float) -> None:
        stamp = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
        os.environ["HYPERLOOM_RUN_CREATED_AT"] = stamp.isoformat()

    def spec(self, **sections: object) -> None:
        self.spec_file.write_text(json.dumps({**SPEC, **sections}), encoding="utf-8")


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Host:
    """The inputs a host injects into a run created ten minutes ago, the grant stop recorded rather than armed."""

    state_dir, workspace = tmp_path / "state", tmp_path / "workspace"
    state_dir.mkdir()
    workspace.mkdir()
    given = Host(state_dir, workspace, tmp_path / "spec.json", [])
    given.spec()
    for name, value in {
        "HYPERLOOM_INPUT_FILE": str(given.spec_file),
        "HYPERLOOM_RUN_ID": "run-1",
        "HYPERLOOM_STATE_DIR": str(state_dir),
        "HYPERLOOM_GRANT_LIMITS": json.dumps(GRANT),
        "E2B_API_URL": "http://service.invalid",
        "E2B_SANDBOX_URL": "http://sandboxes.invalid",
        "HYPERLOOM_SANDBOX_KEY_MEASUREMENT": "m-key-000000000000",
        "HYPERLOOM_SANDBOX_TEMPLATES": json.dumps({"measurement": {"2": "ar-01ja-measurement"}}),
        "HYPERLOOM_SOURCE_REPO": "AMD-AGI/Hyperloom",
        "HYPERLOOM_SOURCE_SHA": "c" * 40,
        "USER_DATA_PATH": str(workspace),
        "HYPERLOOM_RUNNER_CACHE": str(tmp_path / "runner-cache"),
    }.items():
        monkeypatch.setenv(name, value)
    for name in ("HYPERLOOM_MCP_PORT", "HYPERLOOM_MCP_TOKEN", "HYPERLOOM_EVENTS_DIR"):
        monkeypatch.delenv(name, raising=False)
    given.started(seconds_ago=600)
    monkeypatch.setattr(cli, "arm_grant_stop", given.stops.append)
    return given


def _session_ending(stop_reason: str | None, returncode: int) -> Callable[..., int]:
    """A session that creates its directory in the workspace, reports it, and ends with ``returncode`` after a CLOSE
    that reported ``stop_reason``, or without a CLOSE when it is ``None``."""

    def run_session(settings: RunSettings, peer_factory: object, services: object, report_root: Callable) -> int:
        layout = create_session(Path(os.environ["USER_DATA_PATH"]), settings.workload.model_basename, Clock())
        report_root(layout.root)
        if stop_reason is not None:
            _report(layout, stop_reason=stop_reason)
            return returncode
        raise CacheBuildError(f"{'c' * 40} is not a commit of the source")

    return run_session


@pytest.mark.usefixtures("ready_environment")
def test_a_hosted_spec_runs_the_session_the_same_options_of_run_would(
    host: Host, agent_cli: list[RunSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    on_the_command_line = _settings(*SPEC_RUN_ARGS)
    monkeypatch.setattr(local_peers, "machine_cards", _no_local_cards)

    assert cli.main(["hosted", "run"]) == 0

    assert agent_cli == [on_the_command_line]
    assert "HYPERLOOM_SANDBOX_KEY_MEASUREMENT" not in os.environ, "no child process inherits a host credential"


@pytest.mark.usefixtures("ready_environment")
def test_a_hosted_spec_names_its_gpu_type_in_any_case(
    host: Host, agent_cli: list[RunSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    host.spec(hardware={"gpu_type": "MI355X", "tp": 2})
    on_the_command_line = _settings(*SPEC_RUN_ARGS, "--gpu-type", "mi355x")
    monkeypatch.setattr(local_peers, "machine_cards", _no_local_cards)

    assert cli.main(["hosted", "run"]) == 0

    assert agent_cli == [on_the_command_line]


@pytest.mark.parametrize(
    ("name", "value", "named"),
    [
        ("HYPERLOOM_RUNNER_CACHE", None, "HYPERLOOM_RUNNER_CACHE is not set to an absolute path"),
        ("HYPERLOOM_RUNNER_CACHE", "", "HYPERLOOM_RUNNER_CACHE is not set to an absolute path"),
        ("HYPERLOOM_RUNNER_CACHE", "runner-cache", "HYPERLOOM_RUNNER_CACHE is not set to an absolute path"),
        (ARCHIVE_ENV, None, f"{ARCHIVE_ENV} is not set"),
    ],
)
@pytest.mark.usefixtures("ready_environment")
def test_a_hosted_run_without_its_shared_storage_ends_for_good_naming_the_variable(
    name: str, value: str | None, named: str, host: Host, agent_cli: list[RunSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)

    assert cli.main(["hosted", "run"]) == cli.EXIT_FAILED

    document = host.exit_document()
    assert (document["final"], document["returncode"]) == (True, cli.EXIT_FAILED)
    assert named in str(document["error"])
    assert agent_cli == [] and not any(host.workspace.iterdir())


@pytest.mark.usefixtures("ready_environment")
def test_a_hosted_spec_lends_dev_sandboxes_and_runs_agents_with_the_grants_dev_key(
    host: Host, agent_cli: list[RunSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    host.spec(multibranch={**SPEC["multibranch"], "dev_leases": 2, "autonomous_agents": ["/agents/a", "/agents/b"]})
    monkeypatch.setenv("HYPERLOOM_SANDBOX_KEY_DEV", "dev-key-0123456789")
    agents = ("--autonomous-agent", "/agents/a", "--autonomous-agent", "/agents/b")
    on_the_command_line = _settings(*SPEC_RUN_ARGS, "--dev-leases", "2", *agents)
    monkeypatch.setattr(local_peers, "machine_cards", _no_local_cards)

    assert cli.main(["hosted", "run"]) == 0

    assert agent_cli == [on_the_command_line]


@pytest.mark.parametrize(
    ("agents", "named"),
    [(["/agents/a"], "need HYPERLOOM_SANDBOX_KEY_DEV"), ("/agents/a", "must be a list of agent sources")],
    ids=["no-dev-key", "not-a-list"],
)
@pytest.mark.usefixtures("ready_environment")
def test_a_hosted_spec_whose_agents_cannot_run_ends_for_good_naming_why(
    agents: object, named: str, host: Host, agent_cli: list[RunSettings]
) -> None:
    host.spec(multibranch={**SPEC["multibranch"], "autonomous_agents": agents})

    assert cli.main(["hosted", "run"]) == cli.EXIT_USAGE

    document = host.exit_document()
    assert named in str(document["error"]) and agent_cli == []


def test_the_hosted_runner_cache_is_the_directory_the_variable_names() -> None:
    assert cli.hosted_runner_cache({"HYPERLOOM_RUNNER_CACHE": " /data/hl/runner-cache "}) == Path(
        "/data/hl/runner-cache"
    )


def _no_local_cards() -> tuple[int, ...]:
    raise AssertionError("a hosted run measures on the grant's cards, not this container's")


@pytest.mark.parametrize(("max_sandboxes", "tests"), [(8, 4), (3, 3)])
@pytest.mark.usefixtures("ready_environment")
def test_a_hosted_session_runs_as_many_tests_at_once_as_the_grant_has_cards_and_sandboxes_for(
    max_sandboxes: int, tests: int, host: Host, agent_cli: list[RunSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local_peers, "machine_cards", _no_local_cards)
    monkeypatch.setenv("HYPERLOOM_GRANT_LIMITS", json.dumps({**GRANT, "max_gpus": 8, "max_sandboxes": max_sandboxes}))
    host.spec(placement={"profile": "hosted-gpu"})

    assert cli.main(["hosted", "run"]) == 0

    [settings] = agent_cli
    assert settings.caps.max_parallel_tests == tests, "TP=2 on 8 GPUs, at most one test per sandbox"


@pytest.mark.usefixtures("ready_environment")
def test_a_grant_that_states_only_some_totals_ends_the_run_for_good_naming_them(
    host: Host, agent_cli: list[RunSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HYPERLOOM_GRANT_LIMITS", json.dumps({"max_gpus": 4, "max_runtime_sec": 86400}))

    assert cli.main(["hosted", "run"]) == cli.EXIT_USAGE

    document = host.exit_document()
    assert (document["final"], document["returncode"]) == (True, cli.EXIT_USAGE)
    assert "states no max_sandboxes, gpu_hours" in str(document["error"])
    assert agent_cli == [] and not any(host.workspace.iterdir())


@pytest.mark.usefixtures("ready_environment")
def test_a_hosted_session_ends_a_close_grace_and_a_margin_before_the_grant_and_is_stopped_a_grace_before_it(
    host: Host, agent_cli: list[RunSettings]
) -> None:
    host.started(seconds_ago=GRANT["max_runtime_sec"] - 3600)

    assert cli.main(["hosted", "run"]) == 0

    [settings] = agent_cli
    grace = MB_CLOSE_GRACE_CAP_SECONDS
    assert settings.max_hours == pytest.approx((3600 - grace - GRANT_MARGIN_SECONDS) / 3600, abs=0.01)
    assert host.stops == [pytest.approx(3600 - grace, abs=30)]


@pytest.mark.parametrize(
    ("sections", "field"),
    [
        ({"multibranch": None}, "multibranch.workload"),
        ({"multibranch": {"max_branches": 3}}, "multibranch.workload"),
        ({"workload": {"isl": 1024, "osl": 512, "precision": "fp8"}}, "workload.conc"),
        ({"workload": {"conc": 16, "isl": 1024, "osl": 512}}, "workload.precision"),
        ({"multibranch": {"workload": "agentx"}}, "--isl is not taken by an agentx workload"),
    ],
)
@pytest.mark.usefixtures("ready_environment")
def test_a_spec_a_session_cannot_run_on_ends_the_run_for_good_naming_the_field_and_builds_nothing(
    sections: dict[str, object], field: str, host: Host, agent_cli: list[RunSettings]
) -> None:
    host.spec(**sections)

    assert cli.main(["hosted", "run"]) == cli.EXIT_USAGE

    document = host.exit_document()
    assert (document["final"], document["returncode"]) == (True, cli.EXIT_USAGE)
    assert field in str(document["error"])
    assert agent_cli == [] and host.stops == []
    assert not any(host.workspace.iterdir())


@pytest.mark.usefixtures("ready_environment", "agent_cli")
def test_a_hosted_session_that_ran_close_ends_the_run_for_good_with_its_stop_reason(
    host: Host, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "run_session", _session_ending("time_exhausted", 0))

    assert cli.main(["hosted", "run"]) == 0

    assert host.exit_document() == {"final": True, "returncode": 0, "stop_reason": "time_exhausted"}
    session = json.loads((host.state_dir / "session.json").read_text(encoding="utf-8"))["session_dir"]
    assert Path(session).parent.parent == host.workspace


@pytest.mark.usefixtures("ready_environment", "agent_cli")
def test_a_runner_cache_that_cannot_be_built_ends_the_run_for_good(host: Host, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "run_session", _session_ending(None, 0))

    assert cli.main(["hosted", "run"]) == cli.EXIT_FAILED

    document = host.exit_document()
    assert (document["final"], document["returncode"]) == (True, cli.EXIT_FAILED)
    assert "is not a commit of the source" in str(document["error"])


@pytest.mark.usefixtures("ready_environment")
def test_a_missing_package_of_the_hosted_extra_ends_the_run_for_good_naming_it(
    host: Host, agent_cli: list[RunSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, "hyperloom_multibranch.toolbox.hosted_toolbox", raising=False)
    monkeypatch.delitem(sys.modules, "hyperloom.sandbox.e2b", raising=False)
    monkeypatch.setitem(sys.modules, "e2b", None)

    assert cli.main(["hosted", "run"]) == cli.EXIT_FAILED

    document = host.exit_document()
    assert (document["final"], document["returncode"]) == (True, cli.EXIT_FAILED)
    assert "e2b" in str(document["error"])
    assert agent_cli == [] and not any(host.workspace.iterdir())


@pytest.mark.parametrize(
    "crash", [RuntimeError("the scheduler crashed"), ModuleNotFoundError("No module named 'e2b'", name="e2b")]
)
@pytest.mark.usefixtures("ready_environment", "agent_cli")
def test_a_hosted_session_that_crashes_leaves_the_host_to_start_the_run_again(
    crash: Exception, host: Host, monkeypatch: pytest.MonkeyPatch
) -> None:
    def crashes(*args: object) -> int:
        raise crash

    monkeypatch.setattr(cli, "run_session", crashes)

    with pytest.raises(type(crash)):
        cli.main(["hosted", "run"])

    assert not (host.state_dir / "exit.json").exists()


@pytest.mark.usefixtures("ready_environment")
def test_a_restart_finishes_the_session_an_earlier_start_began_and_ends_the_run_for_good(
    host: Host, agent_cli: list[RunSettings], tmp_path: Path
) -> None:
    layout = _ended_before_final(tmp_path / "MB-20261020T141241Z-64f775fe")
    (host.state_dir / "session.json").write_text(json.dumps({"session_dir": str(layout.root)}), encoding="utf-8")
    host.spec(multibranch=None)

    assert cli.main(["hosted", "run"]) == 0

    assert agent_cli == [_settings()], "the session is finished on the settings its manifest holds, not the spec's"
    assert host.exit_document() == {"final": True, "returncode": 0, "stop_reason": "time_exhausted"}
    assert len(host.stops) == 1


@pytest.mark.usefixtures("ready_environment")
def test_a_restart_over_a_session_that_already_chose_ends_the_run_for_good(
    host: Host, agent_cli: list[RunSettings], tmp_path: Path
) -> None:
    layout = _ended_before_final(tmp_path / "MB-20261020T141241Z-64f775fe")
    _report(layout, final={"chosen": {"commit": "c9"}})
    (host.state_dir / "session.json").write_text(json.dumps({"session_dir": str(layout.root)}), encoding="utf-8")

    assert cli.main(["hosted", "run"]) == cli.EXIT_USAGE

    document = host.exit_document()
    assert (document["final"], document["returncode"]) == (True, cli.EXIT_USAGE)
    assert "already chose c9" in str(document["error"])
    assert agent_cli == []


# -- run --measurement hosted -----------------------------------------------------------------------------------------
#: A grant handed to a session started by hand: the sandbox group, the measurement key, the limits and the source.
HANDED = {
    "E2B_API_URL": "http://service.invalid",
    "E2B_SANDBOX_URL": "http://sandboxes.invalid",
    "HYPERLOOM_SANDBOX_KEY_MEASUREMENT": "m-key-000000000000",
    "HYPERLOOM_SANDBOX_TEMPLATES": json.dumps({"measurement": {"2": "ar-01ja-measurement"}}),
    "HYPERLOOM_GRANT_LIMITS": json.dumps({**GRANT, "max_gpus": 8, "max_sandboxes": 3}),
    "HYPERLOOM_SOURCE_REPO": "AMD-AGI/Hyperloom",
    "HYPERLOOM_SOURCE_SHA": "c" * 40,
}
#: What only a Claw host injects, which a session started by hand does not have.
CLAW_ONLY = (
    "HYPERLOOM_INPUT_FILE",
    "HYPERLOOM_RUN_ID",
    "HYPERLOOM_STATE_DIR",
    "HYPERLOOM_MCP_PORT",
    "HYPERLOOM_MCP_TOKEN",
)


@dataclass
class Started:
    """The settings and peers of every session the command started, and the delay of every grant stop armed."""

    sessions: list[tuple[RunSettings, PeerFactory]]
    stops: list[float]


@pytest.fixture
def handed(agent_binaries: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Started:
    """A grant handed over ten minutes after it was created, no Claw input set, sessions recorded rather than run."""

    for name, value in HANDED.items():
        monkeypatch.setenv(name, value)
    for name in CLAW_ONLY:
        monkeypatch.delenv(name, raising=False)
    stamp = datetime.now(timezone.utc) - timedelta(seconds=600)
    monkeypatch.setenv("HYPERLOOM_RUN_CREATED_AT", stamp.isoformat())
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "workspace"))
    monkeypatch.setenv("HYPERLOOM_RUNNER_CACHE", str(tmp_path / "runner-cache"))
    started = Started([], [])

    def run_session(settings: RunSettings, peer_factory: PeerFactory, services: object) -> int:
        started.sessions.append((settings, peer_factory))
        return 0

    monkeypatch.setattr(cli, "run_session", run_session)
    monkeypatch.setattr(cli, "arm_grant_stop", started.stops.append)
    monkeypatch.setattr(local_peers, "machine_cards", _no_local_cards)
    return started


def test_run_measures_on_local_cards_unless_told_hosted() -> None:
    parser = cli.build_parser()

    assert parser.parse_args(["run", *WORKLOAD_ARGS]).measurement == "local"
    assert parser.parse_args(["run", *WORKLOAD_ARGS, "--measurement", "hosted"]).measurement == "hosted"
    with pytest.raises(SystemExit):
        parser.parse_args(["run", *WORKLOAD_ARGS, "--measurement", "cloud"])


@pytest.mark.usefixtures("ready_environment")
def test_a_hosted_run_measures_in_the_sandboxes_of_the_grant_handed_to_it(handed: Started) -> None:
    assert cli.main(["run", *WORKLOAD_ARGS, "--measurement", "hosted"]) == 0

    [(settings, peer_factory)] = handed.sessions
    backend = peer_factory.args[0].backend
    assert type(backend).__name__ == "HostedSandboxes" and not backend.on_this_machine
    host = backend.inputs.host
    assert (host.e2b_api_url, host.run_id, host.input_file, host.state_dir) == (
        "http://service.invalid",
        None,
        None,
        None,
    )
    assert host.sandbox_keys.measurement.reveal() == HANDED["HYPERLOOM_SANDBOX_KEY_MEASUREMENT"]
    assert settings.caps.max_parallel_tests == 3, "TP=2 on 8 GPUs, at most one test per sandbox"
    on_three_cards = cli.settings_from(cli.build_parser().parse_args(["run", *WORKLOAD_ARGS]), lambda tp: 3)
    assert settings == replace(on_three_cards, max_hours=settings.max_hours), "the flags give the workload"
    assert "HYPERLOOM_SANDBOX_KEY_MEASUREMENT" not in os.environ, "no child process inherits a grant key"


@pytest.mark.usefixtures("ready_environment")
def test_a_hosted_run_ends_before_the_grant_and_is_stopped_a_grace_before_it(
    handed: Started, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp = datetime.now(timezone.utc) - timedelta(seconds=GRANT["max_runtime_sec"] - 3600)
    monkeypatch.setenv("HYPERLOOM_RUN_CREATED_AT", stamp.isoformat())

    assert cli.main(["run", *WORKLOAD_ARGS, "--measurement", "hosted", "--max-hours", "0"]) == 0

    [(settings, _)] = handed.sessions
    grace = MB_CLOSE_GRACE_CAP_SECONDS
    assert settings.max_hours == pytest.approx((3600 - grace - GRANT_MARGIN_SECONDS) / 3600, abs=0.01)
    assert handed.stops == [pytest.approx(3600 - grace, abs=30)]


@pytest.mark.parametrize(
    "variable",
    [
        "HYPERLOOM_SANDBOX_KEY_MEASUREMENT",
        "HYPERLOOM_SANDBOX_TEMPLATES",
        "HYPERLOOM_GRANT_LIMITS",
        "HYPERLOOM_RUN_CREATED_AT",
        "E2B_API_URL",
        "HYPERLOOM_SOURCE_SHA",
    ],
)
@pytest.mark.usefixtures("ready_environment")
def test_a_grant_missing_a_variable_is_a_usage_error_naming_it_and_no_secret(
    variable: str, handed: Started, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(variable)

    with pytest.raises(SystemExit) as exc:
        cli.main(["run", *WORKLOAD_ARGS, "--measurement", "hosted"])

    assert exc.value.code == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert variable in err and HANDED["HYPERLOOM_SANDBOX_KEY_MEASUREMENT"] not in err
    assert handed.sessions == [] and handed.stops == []


BRANCHES_IN_DEV_SANDBOXES = [["--dev-leases", "1"], ["--autonomous-agent", "/agents/a"]]


@pytest.mark.parametrize("extra", BRANCHES_IN_DEV_SANDBOXES, ids=["dev-leases", "agent"])
@pytest.mark.usefixtures("ready_environment")
def test_a_hosted_run_lends_dev_sandboxes_and_runs_agents_in_the_grants_dev_sandboxes(
    extra: list[str], handed: Started, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HYPERLOOM_SANDBOX_KEY_DEV", "dev-key")

    assert cli.main(["run", *WORKLOAD_ARGS, "--measurement", "hosted", *extra]) == 0

    [(settings, peer_factory)] = handed.sessions
    assert (settings.dev_leases, settings.agents) in {(1, ()), (0, ("/agents/a",))}
    assert peer_factory.args[0].backend.inputs.host.sandbox_keys.dev.reveal() == "dev-key"
    assert "HYPERLOOM_SANDBOX_KEY_DEV" not in os.environ, "no child process inherits a grant key"


@pytest.mark.parametrize("extra", BRANCHES_IN_DEV_SANDBOXES, ids=["dev-leases", "agent"])
@pytest.mark.parametrize(
    ("given", "missing"),
    [({"HYPERLOOM_SANDBOX_KEY_DEV": None}, "HYPERLOOM_SANDBOX_KEY_DEV"), ({"USER_DATA_PATH": None}, "USER_DATA_PATH")],
    ids=["no-dev-key", "no-user-data"],
)
@pytest.mark.usefixtures("ready_environment")
def test_a_hosted_run_without_what_a_dev_sandbox_needs_is_a_usage_error_naming_it(
    extra: list[str],
    given: dict[str, str | None],
    missing: str,
    handed: Started,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HYPERLOOM_SANDBOX_KEY_DEV", "dev-key")
    for name in given:
        monkeypatch.delenv(name)

    with pytest.raises(SystemExit) as exc:
        cli.main(["run", *WORKLOAD_ARGS, "--measurement", "hosted", *extra])

    assert exc.value.code == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert f"need {missing}" in err and "dev-key" not in err
    assert handed.sessions == []


@pytest.mark.usefixtures("ready_environment")
def test_a_local_run_ignores_a_grant_in_the_environment_and_measures_on_this_machines_cards(
    handed: Started, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(local_peers, "machine_cards", lambda: (0, 1))

    assert cli.main(["run", *WORKLOAD_ARGS]) == 0

    [(settings, peer_factory)] = handed.sessions
    assert peer_factory.args[0].backend == local_peers.LocalCards((0, 1))
    assert settings == _settings() and handed.stops == []
    assert os.environ["HYPERLOOM_SANDBOX_KEY_MEASUREMENT"] == HANDED["HYPERLOOM_SANDBOX_KEY_MEASUREMENT"]
