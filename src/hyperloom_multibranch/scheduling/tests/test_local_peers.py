# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real peers of a session: built inside the session directory on model-backed specialist sessions and on this
machine's measurement host, and refused when this machine has no agent CLI for them, its pins cannot be read or the
image lacks the framework's package."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import urllib.request
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from functools import partial
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV, benchmark_config_path, is_magpie_agentx
from hyperloom.contracts.measurement import AttachmentRef, Purpose, TestKind, TestRequest
from hyperloom.inference_optimizer import inferencex_checkout
from hyperloom.inference_optimizer.cli.benchmark_config import configure_benchmark
from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.contracts.recipe import Recipe, RecipeDelta
from hyperloom.common.env import forge_explicitly_enabled
from hyperloom.common.perf_metric import agentx_enabled
from hyperloom.common.visible_devices import COUNTING_VISIBLE_DEVICE_VARS
from hyperloom.measure import compile as compile_module
from hyperloom.measure.compile import ComponentLayout, LocalBlobStore, PackageBase, UnsupportedComponentError
from hyperloom.measure.hosted_spec import SpecContext
from hyperloom.measure.pins import ActiveImage, pinned_tool_commits
from hyperloom_kb import ReadStatus
from hyperloom_multibranch.contracts.gate import RegressionOutcome
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.blackboard.records import LineageEvent
from hyperloom_multibranch.contracts.blackboard.vocab import Conclusion, InboxKind, LearningKind, LeaseKind, LineageKind
from hyperloom_multibranch.blackboard import STALL_TESTS, LocalBlackboard
from hyperloom_multibranch.blackboard.rsi.archive import ARCHIVE_ENV, TreeArchive
from hyperloom_multibranch.blackboard.rsi.lesson_recall import LessonRecall
from hyperloom_multibranch.blackboard.rsi.lessons import LessonPool, LessonRevision, lesson_id_for, revision_id_for
from hyperloom_multibranch.blackboard.rsi.similarity import task_conditions
from hyperloom_multibranch.blackboard.tests.factories import START, delta, intent, judgement, profile, request, result
from hyperloom_multibranch.contracts.toolbox import SCOPE_SPECIALIST, SpecialistGrant
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.rulegate.critic import NoCritic
from hyperloom_multibranch.rulegate.fix_case import FIX_CASE_ID, FixCase, read_case
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.regression import RegressionSet
from hyperloom_multibranch.rulegate.rules import AgentXRule, InferenceXRule
from hyperloom_multibranch.rulegate.service import BASELINE_TESTS, GateService
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.forks import ForkDesk
from hyperloom_multibranch.scheduling.hosted_peers import HostedSandboxes
from hyperloom_multibranch.scheduling.layout import SESSION_PREFIX, SessionLayout, create_session
from hyperloom_multibranch.scheduling import local_peers as local_peers_module
from hyperloom_multibranch.scheduling.local_peers import (
    AGENTX_ENV,
    AGENTX_TRACE_SEC,
    InferenceXCheckout,
    LocalCards,
    MeasurementHost,
    SessionWiring,
    branch_sandboxes,
    local_peers,
    machine_cards,
    machine_parallel_tests,
    measurement_host,
    model_backed_sessions,
    native_trees,
    require_framework_package,
    specialist_command,
    specialist_recall,
    workload_environment,
    workload_rule,
    _agent_spec,
    _specialist_role,
)
from hyperloom_multibranch.scheduling.peers import Peers, PeerSetup, PeersUnavailable
from hyperloom_multibranch.scheduling.prelude import prelude_argv
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.roster import Roster, RosterCaps
from hyperloom_multibranch.scheduling.settings import AGENT_PORT, InferenceXShape, KernelOptimizer, Workload
from hyperloom_multibranch.specialist import agent_sessions as agent_sessions_module
from hyperloom_multibranch.specialist import model_sessions as model_sessions_module
from hyperloom_multibranch.specialist import worktree as worktree_module
from hyperloom_multibranch.specialist.agent_session import AgentCommand, AgentSetup
from hyperloom_multibranch.specialist.agent_sessions import LocalAgentSandboxes, RemoteAgentSandboxes
from hyperloom_multibranch.specialist.dev_sandbox import DevLeases
from hyperloom_multibranch.specialist.remote_sandbox import RemoteAgentSandbox
from hyperloom_multibranch.toolbox.hosted_leases import HostedBranchLease
from hyperloom_multibranch.specialist.gate_client import GateClient
from hyperloom_multibranch.specialist.model_sessions import ModelSessions
from hyperloom_multibranch.specialist.prompts import role_prompt
from hyperloom_multibranch.specialist.runtime import SessionStart, SpecialistRuntime
from hyperloom_multibranch.specialist.worktree import BranchWorktree
from hyperloom_multibranch.toolbox import local_hosted
from hyperloom.sandbox.e2b import E2BSandboxProvider
from hyperloom.sandbox.socket import Role as SandboxRole, SandboxInfo
from hyperloom_multibranch.toolbox import hosted_toolbox
from hyperloom_multibranch.toolbox.hosted_executor import DEFAULT_TEST_TIMEOUT_SEC
from hyperloom_multibranch.toolbox.hosted_toolbox import HostedToolbox, HostedToolboxInputs
from hyperloom_multibranch.toolbox.local_hosted import LocalHostedToolbox
from hyperloom_multibranch.toolbox.mcp_tools import McpSpecialistTokens
from hyperloom_multibranch.toolbox.queue import QueuePolicy

WORKLOAD = Workload(
    "/models/Llama-3-8B", "sglang", 2, 16, "bf16", "mi355x", KernelOptimizer.FORGE, InferenceXShape(1024, 512)
)
AGENTX = replace(WORKLOAD, inferencex=None)
IMAGE = ActiveImage("registry.example/sglang:v1", "sha256:" + "2" * 64)
LOCAL = LocalCards((0, 1, 2, 3))
HOST = MeasurementHost(LOCAL, IMAGE, "a" * 40, InferenceXCheckout("b" * 40, "/opt/InferenceX"))
AGENTX_CHECKOUT = InferenceXCheckout("c" * 40, "/opt/InferenceX-agentx/inferencex-e2e")
#: What the single-lineage child took to measure the GLM-5.3 TP4 AgentX baseline that a session's toolbox killed at an
#: hour, the limit of every test then.
SINGLE_LINEAGE_AGENTX_BASELINE_SEC = 88 * 60
IMAGE_VARS = ("HYPERLOOM_ACTIVE_IMAGE_REF", "HYPERLOOM_ACTIVE_IMAGE_DIGEST")
DEV_SANDBOX_TOOLS = {f"dev_sandbox_{verb}" for verb in ("lease", "run", "start", "read", "release")}


def _call_tool(grant: SpecialistGrant, name: str, arguments: dict[str, str] | None = None) -> dict[str, object]:
    message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    }
    request = urllib.request.Request(
        grant.url,
        data=json.dumps(message).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {grant.token}"},
    )
    with urllib.request.urlopen(request, timeout=10) as reply:
        return json.loads(reply.read())


def _list_tools(grant: SpecialistGrant) -> set[str]:
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    request = urllib.request.Request(
        grant.url,
        data=json.dumps(message).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {grant.token}"},
    )
    with urllib.request.urlopen(request, timeout=10) as reply:
        return {tool["name"] for tool in json.loads(reply.read())["result"]["tools"]}


def test_the_workload_reaches_the_renderer_as_hyperloom_optimize_exports_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HL_MODEL_BASE", raising=False)
    assert workload_environment(WORKLOAD) == {
        "FRAMEWORK": "sglang",
        "MODEL_PATH": "/models/Llama-3-8B",
        "TP": "2",
        "ISL": "1024",
        "OSL": "512",
        "PRECISION": "bf16",
        "GPU_TYPE": "mi355x",
        "TARGET_GPU_TYPE": "mi355x",
        "KERNEL_OPT_BACKEND_ORDER": "forge",
    }


def test_an_mi350x_is_measured_on_mi355x_scripts_and_named_as_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HL_MODEL_BASE", raising=False)
    environment = workload_environment(replace(WORKLOAD, gpu_type="mi350x"))

    assert (environment["GPU_TYPE"], environment["TARGET_GPU_TYPE"]) == ("mi355x", "mi350x")


def test_an_agentx_workload_reaches_the_renderer_and_the_prelude_child_as_agentx_without_sequence_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HL_MODEL_BASE", raising=False)
    environment = workload_environment(AGENTX)

    assert not {"ISL", "OSL"} & set(environment)
    assert environment[AGENTX_ENV] == "1" and agentx_enabled(environment)
    assert AGENTX_ENV not in workload_environment(WORKLOAD)


def test_the_rule_follows_the_workload(tmp_path: Path) -> None:
    blackboard = LocalBlackboard(tmp_path, Writer(Role.GATE, "rule-gate"))

    assert workload_rule(WORKLOAD, blackboard) == InferenceXRule(blackboard, WORKLOAD.conc)
    assert isinstance(workload_rule(AGENTX, blackboard), AgentXRule)


def test_the_specialists_are_told_an_agentx_workload_and_no_sequence_lengths() -> None:
    assert _specialist_role(WORKLOAD) == "Llama-3-8B on sglang, bf16, TP 2, ISL 1024, OSL 512, concurrency 16, mi355x"
    assert _specialist_role(AGENTX) == "Llama-3-8B on sglang, bf16, TP 2, AgentX trace replay, concurrency 16, mi355x"


def test_the_agents_spec_names_the_pack_the_session_gives_its_arbors_and_none_by_default() -> None:
    assert _agent_spec(WORKLOAD, "agentx-qwen38-1").pack == "agentx-qwen38-1"
    assert _agent_spec(WORKLOAD).pack is None


def test_the_agents_spec_names_the_launch_file_the_session_gives_its_arbors_and_none_by_default() -> None:
    assert _agent_spec(WORKLOAD, "", "/runs/launch.json").launch == "/runs/launch.json"
    assert _agent_spec(WORKLOAD).launch is None


def test_the_agents_spec_starts_its_port_blocks_where_the_session_says() -> None:
    assert _agent_spec(WORKLOAD, "", "", 32100).port == 32100
    assert _agent_spec(WORKLOAD).port == AGENT_PORT


@pytest.mark.parametrize("optimizer", list(KernelOptimizer))
def test_the_kernel_optimizer_flag_picks_the_backend_kernel_tuning_runs(
    optimizer: KernelOptimizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in workload_environment(replace(WORKLOAD, kernel_optimizer=optimizer)).items():
        monkeypatch.setenv(name, value)
    assert forge_explicitly_enabled() is (optimizer is KernelOptimizer.FORGE)


def test_the_model_is_exported_as_the_directory_a_server_loads_it_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "Llama-3-8B").mkdir()
    monkeypatch.setenv("HL_MODEL_BASE", str(tmp_path))
    hub_id = Workload(
        "meta-llama/Llama-3-8B", "vllm", 1, 1, "fp8", "mi355x", KernelOptimizer.FORGE, InferenceXShape(1, 1)
    )
    assert workload_environment(hub_id)["MODEL_PATH"] == str(tmp_path / "Llama-3-8B")


def test_the_specialists_run_the_agent_cli_found_on_path(tmp_path: Path) -> None:
    cli = tmp_path / "claude"
    cli.write_text("#!/bin/sh\nexit 0\n")
    cli.chmod(0o755)

    command = specialist_command(
        {"PATH": str(tmp_path), "ANTHROPIC_API_KEY": "dummy", "CLAUDE_MODEL": "claude-sonnet-5-5"}
    )

    assert (command.backend, command.executable, command.model) == ("claude", str(cli), "claude-sonnet-5-5")


def test_without_the_agent_cli_the_peers_are_refused(tmp_path: Path) -> None:
    with pytest.raises(PeersUnavailable, match="claude CLI the specialists run is not on PATH"):
        specialist_command({"PATH": str(tmp_path), "ANTHROPIC_API_KEY": "dummy"})


def test_a_specialist_model_the_price_sheet_lacks_runs_on_its_tokens_and_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cli = tmp_path / "claude"
    cli.write_text("#!/bin/sh\nexit 0\n")
    cli.chmod(0o755)

    command = specialist_command({"PATH": str(tmp_path), "ANTHROPIC_API_KEY": "dummy", "CLAUDE_MODEL": "glm-5-3"})

    assert command.model == "glm-5-3"
    assert "the specialists run on glm-5-3, which the price sheet does not list" in caplog.text


@pytest.fixture
def four_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    """This machine shows four cards, the upper half of an eight-card node."""

    for name in COUNTING_VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")


def _inferencex_checkout(root: Path) -> Path:
    (root / "benchmarks").mkdir(parents=True)
    (root / "benchmarks" / "benchmark_lib.sh").write_text("")
    return root


@pytest.mark.usefixtures("four_cards")
def test_the_host_is_this_machines_cards_on_the_launchers_image_and_the_pinned_tool_commits(tmp_path: Path) -> None:
    checkout = _inferencex_checkout(tmp_path / "InferenceX")
    env = {IMAGE_VARS[0]: IMAGE.ref, IMAGE_VARS[1]: IMAGE.digest, "INFERENCEX_PATH": str(checkout)}

    host = measurement_host(env, WORKLOAD, LocalCards(machine_cards()))

    tools = pinned_tool_commits({})
    assert host == MeasurementHost(
        LocalCards((4, 5, 6, 7)), IMAGE, tools.magpie, InferenceXCheckout(tools.inferencex, str(checkout))
    )
    assert machine_parallel_tests(2) == 2


def test_a_machine_without_an_inferencex_checkout_refuses_the_peers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(inferencex_checkout, "detect", lambda: "")
    env = {IMAGE_VARS[0]: IMAGE.ref, IMAGE_VARS[1]: IMAGE.digest}

    with pytest.raises(PeersUnavailable, match="no InferenceX checkout"):
        measurement_host(env, WORKLOAD, LOCAL)
    with pytest.raises(PeersUnavailable, match=f"no InferenceX checkout .*{tmp_path.name}"):
        measurement_host({**env, "INFERENCEX_PATH": str(tmp_path)}, WORKLOAD, LOCAL)


def _git_checkout(root: Path) -> str:
    """A Magpie-layout InferenceX repository at ``root``, its project in ``inferencex-e2e``; its HEAD commit."""

    _inferencex_checkout(root / "inferencex-e2e")
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "c"], check=True)
    return subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _magpie(monkeypatch: pytest.MonkeyPatch, default_dir: Path, **pin: str) -> None:
    """The installed Magpie, as its InferenceX module: ``pin`` its ``INFERENCEX_DEFAULT_REF`` (absent when empty),
    ``default_dir`` its default clone location, and its root resolver taking the ``inferencex-e2e`` project."""

    def resolve_inferencex_root(path: str) -> str:
        project = Path(path) / "inferencex-e2e"
        if not (project / "benchmarks" / "benchmark_lib.sh").is_file():
            raise RuntimeError(f"Expected one supported InferenceX project at {path}")
        return str(project)

    module = SimpleNamespace(
        **pin,
        InferenceXManager=lambda: SimpleNamespace(default_dir=str(default_dir)),
        resolve_inferencex_root=resolve_inferencex_root,
    )
    monkeypatch.setattr(local_peers_module, "_magpie_inferencex", lambda: module)


def test_an_agentx_host_measures_on_the_checkout_at_the_commit_magpie_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default, named = tmp_path / "default", tmp_path / "named"
    default_commit, named_commit = _git_checkout(default), _git_checkout(named)
    env = {IMAGE_VARS[0]: IMAGE.ref, IMAGE_VARS[1]: IMAGE.digest, "INFERENCEX_PATH": str(tmp_path / "synthetic")}
    tools = pinned_tool_commits({})

    _magpie(monkeypatch, default, INFERENCEX_DEFAULT_REF=default_commit)
    assert measurement_host(env, AGENTX, LOCAL) == MeasurementHost(
        LOCAL, IMAGE, tools.magpie, InferenceXCheckout(default_commit, str(default / "inferencex-e2e"))
    )
    _magpie(monkeypatch, default, INFERENCEX_DEFAULT_REF=named_commit)
    host = measurement_host({**env, "MAGPIE_INFERENCEX_PATH": str(named)}, AGENTX, LOCAL)
    assert host.inferencex == InferenceXCheckout(named_commit, str(named / "inferencex-e2e"))


def test_an_agentx_host_refuses_a_checkout_off_the_magpie_pin_and_a_magpie_without_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git_checkout(tmp_path / "InferenceX")
    env = {IMAGE_VARS[0]: IMAGE.ref, IMAGE_VARS[1]: IMAGE.digest}
    pinned = "4" * 40

    _magpie(monkeypatch, tmp_path / "InferenceX", INFERENCEX_DEFAULT_REF=pinned)
    with pytest.raises(PeersUnavailable, match=f"no InferenceX checkout at {pinned}.* at {tmp_path}/InferenceX"):
        measurement_host(env, AGENTX, LOCAL)
    with pytest.raises(PeersUnavailable, match=f"no InferenceX checkout at {pinned}.* at {tmp_path}/missing"):
        measurement_host({**env, "MAGPIE_INFERENCEX_PATH": str(tmp_path / "missing")}, AGENTX, LOCAL)
    _magpie(monkeypatch, tmp_path / "InferenceX")
    with pytest.raises(PeersUnavailable, match=f"INFERENCEX_DEFAULT_REF.*MAGPIE_REF {pinned_tool_commits({}).magpie}"):
        measurement_host(env, AGENTX, LOCAL)


def test_an_agentx_host_refuses_a_checkout_whose_tracked_files_differ_from_its_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "InferenceX"
    commit = _git_checkout(root)
    env = {IMAGE_VARS[0]: IMAGE.ref, IMAGE_VARS[1]: IMAGE.digest}
    _magpie(monkeypatch, root, INFERENCEX_DEFAULT_REF=commit)
    (root / "inferencex-e2e" / "untracked-output.txt").write_text("results\n")
    assert measurement_host(env, AGENTX, LOCAL).inferencex == InferenceXCheckout(commit, str(root / "inferencex-e2e"))

    tracked = next(
        path for path in (root / "inferencex-e2e").rglob("*") if path.is_file() and path.name != "untracked-output.txt"
    )
    tracked.write_text(tracked.read_text() + "# a local recipe\n")
    with pytest.raises(PeersUnavailable, match=f"modified.*{tracked.relative_to(root)}"):
        measurement_host(env, AGENTX, LOCAL)


@pytest.mark.parametrize("mask", ["ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"])
def test_the_cards_are_the_ids_the_visible_device_mask_names(mask: str, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in COUNTING_VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(mask, "6,2")

    assert machine_cards() == (6, 2)


@pytest.mark.parametrize("framework", ["tensorrt-llm", "sglang"])
def test_a_framework_whose_source_cannot_be_laid_out_refuses_the_peers(
    framework: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def not_installed(layout: ComponentLayout) -> PackageBase:
        raise UnsupportedComponentError(f"package {layout.import_name} is not installed in this image")

    monkeypatch.setattr(compile_module, "locate_installed_package", not_installed)

    with pytest.raises(PeersUnavailable, match=f"{framework} .*(no package layout|not installed in this image)"):
        require_framework_package(framework)


def test_a_session_nobody_names_the_image_of_refuses_the_peers() -> None:
    with pytest.raises(PeersUnavailable, match=f"{IMAGE_VARS[0]}, {IMAGE_VARS[1]} not set"):
        measurement_host({}, WORKLOAD, LOCAL)


@pytest.mark.parametrize(
    ("env", "named"),
    [
        ({IMAGE_VARS[0]: IMAGE.ref}, IMAGE_VARS[1]),
        ({IMAGE_VARS[0]: IMAGE.ref, IMAGE_VARS[1]: "sha256:" + "0" * 64}, "not a real sha256"),
        ({IMAGE_VARS[0]: IMAGE.ref, IMAGE_VARS[1]: IMAGE.digest, "MAGPIE_REF": "main"}, "MAGPIE_REF"),
    ],
    ids=["half-named-image", "zero-digest", "moving-tool-ref"],
)
def test_pins_that_cannot_be_read_refuse_the_peers(env: dict[str, str], named: str) -> None:
    with pytest.raises(PeersUnavailable, match=named):
        measurement_host(env, WORKLOAD, LOCAL)


def test_a_machine_that_shows_no_card_fits_no_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="0 cards"):
        machine_parallel_tests(1)


@dataclass(frozen=True)
class _Built:
    layout: SessionLayout
    setup: PeerSetup
    peers: Peers
    wiring: SessionWiring


@asynccontextmanager
async def _peers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workload: Workload,
    host: MeasurementHost = HOST,
    dev_leases: int = 0,
) -> AsyncIterator[_Built]:
    """The real peers of a session of ``workload`` on ``host`` in ``tmp_path`` lending ``dev_leases`` dev sandboxes,
    with the wiring their specialist sessions were built on; a branch's worktree, holding no component, is served
    only when the session lends dev sandboxes. Every variable the build exports is restored afterwards."""

    for name in (*workload_environment(workload), BENCHMARK_CONFIG_ENV):
        monkeypatch.setenv(name, "")
    monkeypatch.delenv("GPU_TYPE", raising=False)
    monkeypatch.delenv("HYPERLOOM_KB_URL", raising=False)
    clock = Clock()
    layout = create_session(tmp_path, workload.model_basename, clock)
    desk = ForkDesk(Roster(RosterCaps(1, 1, 1), layout.roster, clock), DecisionLog(layout.decisions, clock), clock)
    blobs = LocalBlobStore(layout.blobs)
    setup = PeerSetup(
        layout,
        workload,
        desk,
        ProcessRegistry(layout.pids, clock),
        blobs,
        NoCritic(),
        {"geak": layout.reference / "geak"},
        dev_leases,
        (),
    )
    wired: list[SessionWiring] = []

    def sessions(wiring: SessionWiring) -> ModelSessions:
        wired.append(wiring)

        def factory(start: SessionStart) -> object:
            raise AssertionError("no specialist runs here")

        def worktrees(branch: str) -> BranchWorktree:
            if not dev_leases:
                raise AssertionError("no worktree is used here")
            return SimpleNamespace(path=tmp_path / branch, holds=lambda component: False)

        return ModelSessions(factory, worktrees)

    peers = local_peers(host, sessions, setup)
    [wiring] = wired
    try:
        yield _Built(layout, setup, peers, wiring)
    finally:
        peers.mcp.stop()
        await peers.toolbox.close()


@pytest.fixture
async def built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_Built]:
    """The real peers of an InferenceX session in ``tmp_path``."""

    monkeypatch.delenv("HL_MODEL_BASE", raising=False)
    async with _peers(tmp_path, monkeypatch, WORKLOAD) as peers:
        yield peers


async def test_an_agentx_session_measures_the_agentx_benchmark_its_prelude_child_is_bound_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "models" / "Llama-3-8B").mkdir(parents=True)
    monkeypatch.setenv("HL_MODEL_BASE", str(tmp_path / "models"))
    workload = replace(AGENTX, model="meta-llama/Llama-3-8B")
    host = replace(HOST, inferencex=AGENTX_CHECKOUT)
    specs: list[SpecContext] = []
    build = local_hosted.build_local_toolbox

    def signed(*args: object, **options: object) -> LocalHostedToolbox:
        specs.append(args[-1])
        return build(*args, **options)

    monkeypatch.setattr(local_hosted, "build_local_toolbox", signed)

    async with _peers(tmp_path / "workspace", monkeypatch, workload, host) as built:
        toolbox = built.layout.toolbox
        assert benchmark_config_path() == (toolbox / "benchmark.yaml").resolve()
        canonical = yaml.safe_load((toolbox / "canonical" / "canonical.yaml").read_text())["benchmark"]
        assert is_magpie_agentx(canonical) and canonical["agentx"]["mode"] == "canonical"
        assert canonical["inferencex_path"] == host.inferencex.path
        assert [(spec.magpie_commit, spec.inferencex_commit) for spec in specs] == [
            (host.magpie_commit, host.inferencex.commit)
        ]
        assert (canonical["model"], canonical["envs"]["MODEL_PATH"]) == (
            workload.model,
            str(tmp_path / "models" / "Llama-3-8B"),
        )
        assert (str(canonical["envs"]["TP"]), str(canonical["envs"]["CONC"])) == (str(workload.tp), str(workload.conc))
        argv = prelude_argv(workload, 1.0, built.layout.prelude_launch)
        args = _build_parser().parse_args(list(argv[argv.index("optimize") :]))
        configure_benchmark(args)
        assert (args.model, args.framework, args.tp, args.ep, args.conc, args.precision) == (
            workload.model,
            workload.framework,
            workload.tp,
            1,
            workload.conc,
            workload.precision,
        )


async def test_an_agentx_session_on_a_model_directory_serves_it_at_its_ep_in_the_sessions_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "models" / "DeepSeek-V3"
    model.mkdir(parents=True)
    workload = replace(AGENTX, model=str(model), ep=8)
    host = replace(HOST, inferencex=AGENTX_CHECKOUT)

    async with _peers(tmp_path / "workspace", monkeypatch, workload, host) as built:
        benchmark = yaml.safe_load((built.layout.toolbox / "benchmark.yaml").read_text())["benchmark"]
        assert (benchmark["docker_image"], benchmark["envs"]["EP_SIZE"]) == (IMAGE.ref, 8)
        assert benchmark["envs"]["MODEL_PATH"] == str(model)
        argv = prelude_argv(workload, 1.0, built.layout.prelude_launch)
        args = _build_parser().parse_args(list(argv[argv.index("optimize") :]))
        configure_benchmark(args)
        assert (str(args.model), args.ep, args.docker_image) == (str(model), 8, IMAGE.ref)


def _options_given(monkeypatch: pytest.MonkeyPatch, module: object, builder: str) -> list[object]:
    """The options every toolbox ``builder`` of ``module`` builds from here on is given, its default when none is; it
    still builds it."""

    given: list[object] = []
    build = getattr(module, builder)
    default = inspect.signature(build).parameters["options"].default

    def recorded(*args: object, options: object = default) -> object:
        given.append(options)
        return build(*args, options=options)

    monkeypatch.setattr(module, builder, recorded)
    return given


async def test_an_agentx_test_may_outlast_its_trace_window_and_an_inferencex_test_keeps_the_default_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "models" / "GLM-5.3"
    model.mkdir(parents=True)
    monkeypatch.delenv("HL_MODEL_BASE", raising=False)
    given = _options_given(monkeypatch, local_hosted, "build_local_toolbox")

    async with _peers(
        tmp_path / "agentx", monkeypatch, replace(AGENTX, model=str(model)), replace(HOST, inferencex=AGENTX_CHECKOUT)
    ):
        pass
    async with _peers(tmp_path / "inferencex", monkeypatch, WORKLOAD):
        pass

    [agentx, inferencex] = given
    assert agentx.timeout_sec > max(AGENTX_TRACE_SEC, SINGLE_LINEAGE_AGENTX_BASELINE_SEC)
    assert inferencex.timeout_sec == DEFAULT_TEST_TIMEOUT_SEC


def test_the_specialists_recall_no_past_learnings_whatever_experience_kb_the_environment_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HYPERLOOM_KB_URL", "http://127.0.0.1:9/kb")
    monkeypatch.delenv(ARCHIVE_ENV, raising=False)

    past = specialist_recall(tmp_path).past(profile(), "kv cache", limit=5)

    assert (past.status, past.items) == (ReadStatus.DISABLED.value, ())


def test_the_specialists_recall_past_sessions_lessons_from_the_pool_beside_the_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ARCHIVE_ENV, str(tmp_path / "archive"))
    lesson = LessonRevision(
        revision_id_for("past"),
        lesson_id_for("past"),
        LearningKind.PITFALL,
        "the bare --cuda-graph-max-bs is ambiguous; use the -decode form",
        task_conditions(profile()),
        "specialist:past-1/b05",
        START,
        session_id="past-1",
    )
    LessonPool.beside(TreeArchive(tmp_path / "archive")).write(lesson)

    recall = specialist_recall(tmp_path / "reports")

    assert recall.knows(lesson.lesson_id) and recall.past_revision() != ""


async def test_the_real_peers_live_inside_the_session_directory(built: _Built) -> None:
    layout, setup, peers, wiring = built.layout, built.setup, built.peers, built.wiring
    assert (wiring.setup, wiring.repo, wiring.briefs) == (setup, peers.repo, peers.briefs)
    assert isinstance(peers.lifecycle, SpecialistRuntime) and peers.briefs.warm == ()
    assert peers.briefs.repo is peers.repo and isinstance(peers.briefs.learnings, LessonRecall)
    assert isinstance(peers.toolbox, LocalHostedToolbox) and isinstance(peers.repo, GitExperimentRepo)
    canonical = yaml.safe_load((layout.toolbox / "canonical" / "canonical.yaml").read_text())["benchmark"]
    assert (canonical["model"], canonical["runner_type"], canonical["inferencex_path"]) == (
        WORKLOAD.model,
        WORKLOAD.gpu_type,
        HOST.inferencex.path,
    )
    assert not (layout.toolbox / "measurement-config.json").exists()
    assert benchmark_config_path() is None and not (layout.toolbox / "benchmark.yaml").exists()
    assert peers.base_image_digest == IMAGE.digest
    assert peers.mcp.url.startswith("http://127.0.0.1:")
    assert {name: os.environ[name] for name in workload_environment(WORKLOAD)} == workload_environment(WORKLOAD)
    root = await peers.repo.init_root(Recipe(IMAGE.digest))
    assert await peers.repo.head(INTEGRATION_BRANCH) == root
    patch = await peers.repo.store_patch("sglang", "--- a/x\n+++ b/x\n")
    assert LocalBlobStore(layout.blobs).get(patch.sha256) == b"--- a/x\n+++ b/x\n"
    assert isinstance(peers.gate, GateService) and await peers.gate.baseline_test_count() == BASELINE_TESTS
    await peers.toolbox.start()
    await peers.toolbox.set_policy(QueuePolicy(per_branch_k=1, global_concurrency=2))
    state = await peers.toolbox.queue_state()
    assert (state.queued, state.running) == (0, 0)
    assert isinstance(peers.reserved, RuleGate) and not await peers.reserved.land_split("b09")
    assert isinstance(peers.regression, RegressionSet)
    scheduler = LocalBlackboard(layout.blackboard, Writer(Role.SCHEDULER, "test-scheduler"))
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id=root))
    assert await peers.regression.check(root, RecipeDelta(), sources=()) == RegressionOutcome(True)
    state = await peers.toolbox.queue_state()
    assert (state.queued, state.running) == (0, 0)
    assert isinstance(wiring.tokens, McpSpecialistTokens)
    grant = wiring.tokens.issue("b01", frozenset({SCOPE_SPECIALIST}))
    assert grant.url == peers.mcp.url
    reply = await asyncio.to_thread(_call_tool, grant, "queue_state")
    assert reply["result"]["isError"] is False
    tools = await asyncio.to_thread(_list_tools, grant)
    served = {
        "worktree_sync",
        "worktree_check",
        "worktree_patch",
        "blackboard_brief",
        "request_repair",
        "decline_repair",
        "fix_case",
    }
    assert served <= tools
    assert all(entry.name.startswith(SESSION_PREFIX) for entry in layout.root.iterdir())


async def test_a_plain_explore_runs_no_accuracy_eval_while_confirm_and_integration_tests_still_do(
    built: _Built,
) -> None:
    peers = built.peers
    root = await peers.repo.init_root(Recipe(IMAGE.digest))
    blackboard = LocalBlackboard(built.layout.blackboard, Writer(Role.SCHEDULER, "test-scheduler"))

    async def submitted(purpose: Purpose, retest_of: str | None = None) -> TestRequest:
        test = TestRequest(TestKind.TEST, purpose, "b01", root, delta(), retest_of=retest_of)
        return (await blackboard.record(await peers.toolbox.submit_test(test))).request

    explore = await submitted(Purpose.EXPLORE)
    confirm = await submitted(Purpose.CONFIRM, explore.request_id)
    integration = await submitted(Purpose.INTEGRATION)

    assert (explore.accuracy, confirm.accuracy, integration.accuracy) == (False, True, True)


async def test_a_session_that_lends_no_dev_sandboxes_serves_no_dev_sandbox_tools(built: _Built) -> None:
    grant = built.wiring.tokens.issue("b01", frozenset({SCOPE_SPECIALIST}))

    tools = await asyncio.to_thread(_list_tools, grant)

    assert built.wiring.dev_leases is None
    assert not {name for name in tools if name.startswith("dev_sandbox_")}


async def test_a_session_lending_dev_sandboxes_leases_the_toolbox_cards_to_a_specialist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HL_MODEL_BASE", raising=False)
    async with _peers(tmp_path, monkeypatch, WORKLOAD, dev_leases=2) as built:
        leases = built.wiring.dev_leases
        assert isinstance(leases, DevLeases) and leases.limits.cap == 2
        await built.peers.toolbox.start()
        scheduler = LocalBlackboard(built.layout.blackboard, Writer(Role.SCHEDULER, "test-scheduler"))
        await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
        await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "b01", from_commit="c0"))
        grant = built.wiring.tokens.issue("b01", frozenset({SCOPE_SPECIALIST}))
        assert DEV_SANDBOX_TOOLS <= await asyncio.to_thread(_list_tools, grant)

        leased = await asyncio.to_thread(_call_tool, grant, "dev_sandbox_lease", {"cards": 1, "minutes": 5})
        assert leased["result"]["isError"] is False, leased
        assert json.loads(leased["result"]["content"][0]["text"])["cards"] == 1
        [hold] = await scheduler.card_holds()
        assert (hold.start.kind, hold.start.branch_id) == (LeaseKind.DEV, "b01")
        await leases.reclaim("b01", "its branch stopped")
        released = await asyncio.to_thread(_call_tool, grant, "dev_sandbox_release")

    assert "no_lease" in released["result"]["content"][0]["text"]


def test_a_hosted_session_lends_the_dev_sandboxes_of_its_grant_on_its_tp_and_runs_its_agents_there(
    tmp_path: Path,
) -> None:
    layout = SimpleNamespace(branch_dir=lambda branch: tmp_path / "MB-branches" / branch)
    setup = SimpleNamespace(layout=layout, workload=replace(WORKLOAD, tp=4), processes=None)
    worktree = SimpleNamespace(path=tmp_path / "worktree", holds=lambda component: False)
    lease = HostedBranchLease("lease-1", "b01", (0, 1, 2, 3), Path("/app/work"), {}, 4, "dev-4", 3600, object())

    hosted = branch_sandboxes(HostedSandboxes(lambda layout, tp: None), setup, lambda branch: worktree)
    local = branch_sandboxes(LocalCards((0, 1)), setup, lambda branch: worktree)

    assert (hosted.dev_cards, local.dev_cards) == (4, 1)
    assert isinstance(hosted.agents, RemoteAgentSandboxes) and isinstance(local.agents, LocalAgentSandboxes)
    sandbox = hosted.dev("b01", lease)
    assert (
        isinstance(sandbox, RemoteAgentSandbox) and sandbox.root == tmp_path / "MB-branches" / "b01" / "dev" / "lease-1"
    )
    assert (sandbox.cards, sandbox.lifetime_s) == (4, 3600)


class _IdleService(E2BSandboxProvider):
    """The host's sandbox service with no sandbox running: what a session closes before it measured anything."""

    def find(self, metadata: Mapping[str, str], *, role: SandboxRole) -> list[SandboxInfo]:
        return []


def _service_sandboxes(monkeypatch: pytest.MonkeyPatch, asked: list[tuple[Path, int]]) -> HostedSandboxes:
    """The host's idle sandbox service as a session's backend, which records in ``asked`` the toolbox directory and TP
    it is asked for inputs of and refuses to read this machine's cards."""

    def inputs(layout: SessionLayout, tp: int) -> HostedToolboxInputs:
        asked.append((layout.toolbox, tp))
        return HostedToolboxInputs(
            run_id="run-1",
            session_dir=layout.toolbox,
            api_url="http://service.invalid",
            sandbox_url="http://sandboxes.invalid",
            measurement_key="m-key",
            sandbox_templates={"measurement": {str(tp): "ar-01ja-measurement"}},
            source_sha="c" * 40,
            tp=tp,
        )

    def no_cards() -> tuple[int, ...]:
        raise AssertionError("a hosted session reads none of this machine's cards")

    monkeypatch.setattr(local_peers_module, "machine_cards", no_cards)
    monkeypatch.setattr(hosted_toolbox, "E2BSandboxProvider", _IdleService)
    return HostedSandboxes(inputs)


async def test_a_hosted_session_measures_in_the_services_sandboxes_under_its_toolbox_directory_on_no_local_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[tuple[Path, int]] = []
    monkeypatch.delenv("HL_MODEL_BASE", raising=False)
    host = replace(HOST, backend=_service_sandboxes(monkeypatch, asked))
    async with _peers(tmp_path, monkeypatch, WORKLOAD, host) as built:
        toolbox, layout = built.peers.toolbox, built.layout
        assert isinstance(toolbox, HostedToolbox)
        assert asked == [(layout.toolbox, WORKLOAD.tp)]
        assert toolbox.trace_root == layout.toolbox and toolbox.blobs is built.setup.blobs
        assert built.peers.base_image_digest == IMAGE.digest
        assert (layout.toolbox / "canonical" / "canonical.yaml").is_file()


async def test_an_agentx_test_in_the_services_sandboxes_may_outlast_its_trace_window_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "models" / "GLM-5.3"
    model.mkdir(parents=True)
    given = _options_given(monkeypatch, hosted_toolbox, "build_hosted_toolbox")
    host = MeasurementHost(_service_sandboxes(monkeypatch, []), IMAGE, "a" * 40, AGENTX_CHECKOUT)

    async with _peers(tmp_path / "workspace", monkeypatch, replace(AGENTX, model=str(model)), host):
        pass

    [agentx] = given
    assert agentx.timeout_sec > max(AGENTX_TRACE_SEC, SINGLE_LINEAGE_AGENTX_BASELINE_SEC)


def _agent_clients(monkeypatch: pytest.MonkeyPatch) -> list[GateClient | None]:
    """The gate's client every session's agents are built with from here on; no specialist session is built."""

    clients: list[GateClient | None] = []

    def agents(*args: object, client: GateClient | None = None) -> object:
        clients.append(client)
        return SimpleNamespace()

    def sessions(*args: object, **kwargs: object) -> ModelSessions:
        return ModelSessions(lambda start: None, lambda branch: None)

    monkeypatch.setattr(model_sessions_module, "model_sessions", sessions)
    monkeypatch.setattr(agent_sessions_module, "AgentSessions", agents)
    return clients


def _with_agent(wiring: SessionWiring) -> SessionWiring:
    return replace(wiring, setup=replace(wiring.setup, agents=("https://agents.example/tuner.git",)))


async def test_an_agent_of_an_inferencex_session_measures_with_the_client_of_the_rendered_canonical(
    built: _Built, monkeypatch: pytest.MonkeyPatch
) -> None:
    clients = _agent_clients(monkeypatch)

    model_backed_sessions(AgentCommand("claude", "claude"))(_with_agent(built.wiring))

    [client] = clients
    assert client is not None and (client.conc, client.isl, client.osl) == (WORKLOAD.conc, 1024, 512)
    assert client.inferencex_root == HOST.inferencex.path


async def test_an_agent_of_an_agentx_session_gets_no_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "models" / "Llama-3-8B").mkdir(parents=True)
    monkeypatch.setenv("HL_MODEL_BASE", str(tmp_path / "models"))
    workload = replace(AGENTX, model="meta-llama/Llama-3-8B")
    host = replace(HOST, inferencex=AGENTX_CHECKOUT)
    clients = _agent_clients(monkeypatch)

    async with _peers(tmp_path / "workspace", monkeypatch, workload, host) as built:
        model_backed_sessions(AgentCommand("claude", "claude"))(_with_agent(built.wiring))

    assert clients == [None]


async def test_a_session_without_agents_reads_no_canonical(built: _Built, monkeypatch: pytest.MonkeyPatch) -> None:
    clients = _agent_clients(monkeypatch)
    unread = replace(built.wiring, canonical=built.layout.toolbox / "no-such-canonical.yaml")

    model_backed_sessions(AgentCommand("claude", "claude"))(unread)

    assert clients == [None]


async def test_the_specialists_are_told_of_and_give_back_the_dev_sandboxes_the_session_lends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[AgentSetup] = []

    def sessions(setup: AgentSetup, **_: object) -> ModelSessions:
        built.append(setup)
        return ModelSessions(lambda start: None, lambda branch: None)

    monkeypatch.setattr(model_sessions_module, "model_sessions", sessions)
    leases = SimpleNamespace(limits=SimpleNamespace(cap=1))
    layout = SimpleNamespace(logs=tmp_path, source=tmp_path, branch_dir=lambda branch: tmp_path)
    for held in (None, leases):
        wiring = SimpleNamespace(
            setup=SimpleNamespace(
                layout=layout,
                workload=WORKLOAD,
                references={},
                processes=None,
                agents=(),
                agent_pack="",
                agent_launch="",
                agent_port=AGENT_PORT,
            ),
            tokens=None,
            repo=None,
            briefs=SimpleNamespace(text=None),
            toolbox=None,
            blackboards=None,
            dev_leases=held,
            native={},
            agent_sandboxes=None,
        )
        model_backed_sessions(AgentCommand("claude", "claude"))(wiring)

    without, lending = built
    assert (without.dev_leases, lending.dev_leases) == (None, leases)
    assert without.role == role_prompt(_specialist_role(WORKLOAD), references={})
    assert lending.role == role_prompt(_specialist_role(WORKLOAD), references={}, dev_leases=1) != without.role


def _package(root: Path, name: str, dist: str, version: str, *files: str) -> PackageBase:
    """A package ``name`` installed at ``root / name``, with ``files`` (relative to ``root``) beside it."""

    for relative in (f"{name}/__init__.py", *files):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("x = 1\n")
    return PackageBase(name, root / name, dist, version)


def _develop_aiter(checkout: Path) -> PackageBase:
    """aiter installed in develop mode from a git checkout whose commit its version names, csrc beside the package."""

    _package(checkout, "aiter", "amd-aiter", "", "csrc/kernels/gemm.cu")
    commit = _git_checkout_of(checkout)
    return PackageBase("aiter", checkout / "aiter", "amd-aiter", f"0.1.23.dev183+g{commit[:9]}.d20261008")


def _git_checkout_of(root: Path) -> str:
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run([*git, "remote", "add", "origin", "https://github.com/ROCm/aiter"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "c"], check=True)
    return subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


@pytest.mark.parametrize(
    ("aiter", "laid_out"),
    [
        ("develop", ["aiter/__init__.py", "csrc/kernels/gemm.cu", "python/sglang/__init__.py"]),
        ("wheel", ["python/sglang/__init__.py"]),
        ("absent", ["python/sglang/__init__.py"]),
    ],
)
def test_the_specialists_source_base_holds_aiter_beside_the_framework_only_from_a_develop_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, aiter: str, laid_out: list[str]
) -> None:
    installed = {"sglang": _package(tmp_path / "site", "sglang", "sglang", "0.5.0")}
    if aiter == "develop":
        installed["aiter"] = _develop_aiter(tmp_path / "aiter-checkout")
    elif aiter == "wheel":
        installed["aiter"] = _package(tmp_path / "site", "aiter", "amd-aiter", "0.1.23.dev183+ge7d2453f2.d20261008")

    def locate(layout: ComponentLayout) -> PackageBase:
        if layout.import_name not in installed:
            raise UnsupportedComponentError(f"package {layout.import_name} is not installed in this image")
        return installed[layout.import_name]

    monkeypatch.setattr(worktree_module, "SourceBase", partial(worktree_module.SourceBase, locator=locate))
    layout = SimpleNamespace(
        logs=tmp_path / "logs",
        source=tmp_path / "MB-source",
        branch_dir=lambda branch: tmp_path / branch,
    )
    wiring = SimpleNamespace(
        setup=SimpleNamespace(
            layout=layout,
            workload=WORKLOAD,
            references={},
            processes=None,
            agents=(),
            agent_pack="",
            agent_launch="",
            agent_port=AGENT_PORT,
        ),
        tokens=None,
        repo=None,
        briefs=SimpleNamespace(text=None),
        toolbox=None,
        blackboards=None,
        dev_leases=None,
        native={},
        agent_sandboxes=None,
    )

    model_backed_sessions(AgentCommand("claude", "claude"))(wiring)

    tracked = subprocess.run(["git", "ls-files"], cwd=layout.source, check=True, capture_output=True, text=True)
    assert sorted(tracked.stdout.split()) == laid_out
    left_out = [record.getMessage() for record in caplog.records if "leaves aiter out" in record.getMessage()]
    assert len(left_out) == (0 if aiter == "develop" else 1)
    assert all(("no git checkout" if aiter == "wheel" else "not installed") in message for message in left_out)


async def test_a_fix_case_a_specialist_declares_is_stored_in_the_session_blob_store(built: _Built) -> None:
    grant = built.wiring.tokens.issue("b01", frozenset({SCOPE_SPECIALIST}))
    case = FixCase("#!/bin/bash\nexit 0\n", "fp8 scales overflow at 8k context; clamp them before the cast")

    reply = await asyncio.to_thread(
        _call_tool, grant, "fix_case", {"repro_sh": case.repro_sh, "diagnosis": case.diagnosis}
    )

    assert reply["result"]["isError"] is False
    ref = AttachmentRef.from_document(json.loads(reply["result"]["content"][0]["text"]))
    assert (ref.id, ref.kind) == (FIX_CASE_ID, "dir")
    assert read_case(built.setup.blobs.get(ref.sha256)) == case


async def test_a_branchs_specialist_handle_is_one_writer_for_the_whole_session(built: _Built) -> None:
    """Every MCP call used to build a handle, hence a writer; an inbox cursor holds a sequence number for each writer, and
    the 1,258 of a 12 h session made one first ``poll_inbox`` answer 94,039 characters for three items."""
    blackboards = built.wiring.blackboards
    assert blackboards("b01") is blackboards("b01") and blackboards("b01") is not blackboards("b02")


async def test_the_specialist_blackboard_handle_briefs_a_branch_after_stall_tests_judged_tests_without_a_commit(
    built: _Built,
) -> None:
    root = built.layout.blackboard
    scheduler = LocalBlackboard(root, Writer(Role.SCHEDULER, "test-scheduler"))
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, "b01", from_commit="c0"))
    toolbox = LocalBlackboard(root, Writer(Role.TOOLBOX, "test-toolbox"))
    gate = LocalBlackboard(root, Writer(Role.GATE, "test-gate"))
    handle = built.wiring.blackboards("b01")
    assert isinstance(handle, LocalBlackboard)

    async def judged(count: int) -> list[str]:
        for n in range(count):
            request_id = f"r{n}-{count}"
            await toolbox.write_request(request(request_id, intent("b01", "c0", target=request_id)))
            await toolbox.write_result(result(request_id))
            await gate.write_judgement(request_id, judgement(0.1, (request_id,)), Conclusion.RECORDED_ONLY, None)
        return [item.summary for item in (await handle.poll("b01")).items if item.kind is InboxKind.BRIEF]

    assert await judged(STALL_TESTS - 1) == []
    assert [summary.split(":")[0] for summary in await judged(1)] == ["stall"]


def test_a_session_on_a_host_that_builds_native_code_lays_out_the_native_tree_and_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hyperloom_multibranch.toolbox import build_preflight
    from hyperloom_multibranch.toolbox.build_preflight import ImageUpstream

    installed = {
        "sglang": _package(tmp_path / "site", "sglang", "sglang", "0.5.0"),
        "sgl_kernel": _package(tmp_path / "site", "sgl_kernel", "sglang-kernel", "0.4.10"),
    }
    checkout = tmp_path / "build" / "sources" / "sgl_kernel"
    _package(
        checkout / "pending",
        "python/sglang/kernels/aot/python/sgl_kernel",
        "",
        "",
        "python/sglang/kernels/aot/csrc/gemm.cu",
    )
    ref = _git_checkout_of(checkout / "pending")
    (checkout / "pending").rename(checkout / ref)
    monkeypatch.setattr(build_preflight, "image_upstreams", lambda root: [ImageUpstream("sgl_kernel", ref, None)])
    environ = {
        "HYPERLOOM_BUILD_STORE": str(tmp_path / "build" / "store"),
        "HYPERLOOM_BUILD_SOURCES": str(tmp_path / "build" / "sources"),
        "HYPERLOOM_BUILD_IMAGE_ROOT": str(tmp_path / "site"),
    }
    roles: list[str] = []

    def locate(layout: ComponentLayout) -> PackageBase:
        if layout.import_name not in installed:
            raise UnsupportedComponentError(f"package {layout.import_name} is not installed in this image")
        return installed[layout.import_name]

    monkeypatch.setattr(worktree_module, "SourceBase", partial(worktree_module.SourceBase, locator=locate))
    real = model_sessions_module.model_sessions

    def sessions(setup: AgentSetup, **kwargs: object) -> ModelSessions:
        roles.append(setup.role)
        return real(setup, **kwargs)

    monkeypatch.setattr(model_sessions_module, "model_sessions", sessions)
    for native, source in ((native_trees(environ), "with"), (native_trees({}), "without")):
        layout = SimpleNamespace(
            logs=tmp_path / "logs", source=tmp_path / source, branch_dir=lambda branch: tmp_path / branch
        )
        wiring = SimpleNamespace(
            setup=SimpleNamespace(
                layout=layout,
                workload=WORKLOAD,
                references={},
                processes=None,
                agents=(),
                agent_pack="",
                agent_launch="",
                agent_port=AGENT_PORT,
            ),
            tokens=None,
            repo=None,
            briefs=SimpleNamespace(text=None),
            toolbox=None,
            blackboards=None,
            dev_leases=None,
            native=native,
            agent_sandboxes=None,
        )
        model_backed_sessions(AgentCommand("claude", "claude"))(wiring)

    laid = {
        source: subprocess.run(
            ["git", "ls-files"], cwd=tmp_path / source, check=True, capture_output=True, text=True
        ).stdout.split()
        for source in ("with", "without")
    }
    assert "python/sglang/kernels/aot/csrc/gemm.cu" in laid["with"]
    assert "python/sglang/kernels/aot/python/sgl_kernel/__init__.py" in laid["with"]
    assert laid["without"] == ["python/sglang/__init__.py"]
    with_native, without = roles[0], roles[-1]
    assert "python/sglang/kernels/aot" in with_native and "pending_build" in with_native
    assert without == role_prompt(_specialist_role(WORKLOAD), references={})
