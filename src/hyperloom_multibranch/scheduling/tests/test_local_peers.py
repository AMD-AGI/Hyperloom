# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real peers of a session: built inside the session directory on model-backed specialist sessions and on this
machine's measurement host, and refused when this machine has no agent CLI for them, its pins cannot be read or the
image lacks the framework's package."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import subprocess
import urllib.request
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from inference_suite import INFERENCEX_CLIENT_COMMIT
from inference_suite.staging import INFERENCEX_REQUIRED
from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV, benchmark_config_path, is_agentx
from hyperloom.common import agent_isolation
from hyperloom.common.agent_isolation import BWRAP, NONE
from hyperloom.common.env_safety import SPECIALIST_USER_ENV
from hyperloom.contracts.measurement import AttachmentRef, Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import Recipe, RecipeDelta
from hyperloom_common import bwrap as bwrap_module
from hyperloom_common.bwrap import BwrapProbe
from hyperloom_common.env import forge_explicitly_enabled
from hyperloom.common.perf_metric import agentx_enabled
from hyperloom.common import visible_devices
from hyperloom.common.visible_devices import COUNTING_VISIBLE_DEVICE_VARS
from hyperloom.measure import compile as compile_module
from hyperloom.measure.compile import ComponentLayout, LocalBlobStore, PackageBase, UnsupportedComponentError
from hyperloom.measure.hosted_spec import SpecContext
from hyperloom.measure.pins import ActiveImage
from hyperloom.orchestrator.actions.executors import TOOL_CACHE_ENV
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
from hyperloom_multibranch.rulegate.fix_case import FIX_CASE_ID, FixCase, read_case
from hyperloom_multibranch.rulegate.gate import RuleGate
from hyperloom_multibranch.rulegate.regression import RegressionSet
from hyperloom_multibranch.rulegate.agentx import AgentXObjective
from hyperloom_multibranch.rulegate.rules import AgentXRule, InferenceXRule
from hyperloom_multibranch.rulegate.inferencex import BASELINE_TESTS
from hyperloom_multibranch.rulegate.service import GateService
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.forks import ForkDesk
from hyperloom_multibranch.scheduling.hosted_peers import HostedSandboxes
from hyperloom_multibranch.scheduling.layout import SESSION_PREFIX, SessionLayout, create_session
from hyperloom_multibranch.scheduling import launch_isolation as launch_isolation_module
from hyperloom_multibranch.scheduling import local_peers as local_peers_module
from hyperloom_multibranch.scheduling.launch_isolation import HOSTED, launch_isolation
from hyperloom_multibranch.scheduling.local_peers import (
    AGENTX_ENV,
    AGENTX_OUTSIDE_TRACE_SEC,
    AGENTX_TRACE_SEC,
    InferenceXCheckout,
    LocalCards,
    MeasurementHost,
    SessionWiring,
    attempt_timeout_sec,
    branch_sandboxes,
    load_peers,
    local_peers,
    machine_cards,
    machine_parallel_tests,
    measurement_host,
    model_backed_sessions,
    native_trees,
    render_nodes,
    require_framework_package,
    specialist_command,
    specialist_recall,
    workload_environment,
    workload_rule,
    _agent_spec,
    _lease_area,
    _specialist_role,
    specialist_readable,
)
from hyperloom_multibranch.scheduling.peers import Peers, PeerSetup, PeersUnavailable
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.roster import Roster, RosterCaps
from hyperloom_multibranch.scheduling.settings import (
    AGENT_PORT,
    SHARE_MINUTES,
    InferenceXShape,
    KernelOptimizer,
    Workload,
)
from hyperloom_multibranch.specialist import agent_sessions as agent_sessions_module
from hyperloom_multibranch.specialist import model_sessions as model_sessions_module
from hyperloom_multibranch.specialist import worktree as worktree_module
from hyperloom_multibranch.specialist.agent_session import AgentCommand, AgentSetup
from hyperloom_multibranch.specialist.agent_sessions import LocalAgentSandboxes, RemoteAgentSandboxes
from hyperloom_multibranch.specialist.dev_sandbox import DEV_DIR, DevLeases
from hyperloom_multibranch.specialist.remote_sandbox import RemoteAgentSandbox
from hyperloom_multibranch.tests.fakes import RefusingCritic
from hyperloom_multibranch.tests.staging_rig import verified_tool_cache
from hyperloom.inference_optimizer.cli import base_launch as cli_base_launch
from hyperloom.measure.runner import suite_publish
from hyperloom.measure.runner.suite_publish import SuitePublishError
from hyperloom.measure.tests.bundle_factory import base_launch
from hyperloom.orchestrator.state.base_launch import ServerBase, freeze_base, frozen_base, server_base
from inference_suite import SourceSnapshot
from inference_suite.recipe.resolve import UnreadSource
from inference_suite.contracts import BaseLaunch
from hyperloom_multibranch.toolbox.hosted_leases import HostedBranchLease
from hyperloom_multibranch.specialist.gate_client import AgentxGateClient, GateClient
from hyperloom_multibranch.specialist.model_sessions import ModelSessions
from hyperloom_multibranch.specialist.prompts import RoleWorkload, role_prompt
from hyperloom_multibranch.specialist.runtime import SessionStart, SpecialistRuntime
from hyperloom_multibranch.specialist.worktree import BranchWorktree
from hyperloom_multibranch.toolbox import local_hosted
from hyperloom.sandbox.e2b import E2BSandboxProvider
from hyperloom.sandbox.socket import Role as SandboxRole, SandboxInfo
from hyperloom_multibranch.toolbox import hosted_toolbox
from hyperloom_multibranch.toolbox.hosted_executor import DEFAULT_TEST_TIMEOUT_SEC
from hyperloom_multibranch.toolbox.hosted_toolbox import HostedToolbox, HostedToolboxInputs
from hyperloom_multibranch.toolbox.branch_leases import BranchLeases
from hyperloom_multibranch.toolbox.local_hosted import LocalHostedToolbox
from hyperloom_multibranch.toolbox.mcp_tools import McpSpecialistTokens
from hyperloom_multibranch.toolbox.queue import QueuePolicy

WORKLOAD = Workload(
    "/models/Llama-3-8B", "sglang", 2, 16, "bf16", "mi355x", KernelOptimizer.FORGE, InferenceXShape(1024, 512)
)
AGENTX = replace(WORKLOAD, inferencex=None)
TOTAL_AGENTX = replace(AGENTX, agentx_objective=AgentXObjective.TOTAL_TOKEN_THROUGHPUT)
BASE = ServerBase(args="--tp-size 2 --enable-hierarchical-cache --hicache-size 180", env={"SGLANG_X": "true"})
IMAGE = ActiveImage("registry.example/sglang:v1", "sha256:" + "2" * 64)
LOCAL = LocalCards((0, 1, 2, 3))
HOST = MeasurementHost(LOCAL, IMAGE, InferenceXCheckout(INFERENCEX_CLIENT_COMMIT, "/opt/tools/inferencex/client"))
#: An AgentX baseline test that runs past an hour, the limit every test once had.
AGENTX_TEST_PAST_AN_HOUR_SEC = 90 * 60
IMAGE_VARS = ("HYPERLOOM_ACTIVE_IMAGE_REF", "HYPERLOOM_ACTIVE_IMAGE_DIGEST")
DEV_SANDBOX_TOOLS = {f"dev_sandbox_{verb}" for verb in ("lease", "run", "start", "read", "release")}


@pytest.fixture(autouse=True)
def _own_jit_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A local toolbox's compiled-code cache is the host's, below its temporary directory, unless one is named: each
    test names one of its own."""
    monkeypatch.setenv(local_hosted.JIT_STORE_ENV, str(tmp_path / "jit-store"))


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


def test_an_mi350x_is_exported_as_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HL_MODEL_BASE", raising=False)
    environment = workload_environment(replace(WORKLOAD, gpu_type="mi350x"))

    assert (environment["GPU_TYPE"], environment["TARGET_GPU_TYPE"]) == ("mi350x", "mi350x")


def test_an_agentx_workload_reaches_the_renderer_as_agentx_without_sequence_lengths(
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
    assert workload_rule(AGENTX, blackboard) == AgentXRule(AgentXObjective.P50_INTERACTIVITY)
    assert workload_rule(TOTAL_AGENTX, blackboard) == AgentXRule(AgentXObjective.TOTAL_TOKEN_THROUGHPUT)


def test_the_agents_are_asked_for_the_metric_the_session_is_graded_on() -> None:
    assert (_agent_spec(AGENTX).metric, _agent_spec(TOTAL_AGENTX).metric) == (
        "e2e_norm_intvty_p50",
        "total_token_throughput",
    )


def test_the_specialists_are_told_an_agentx_workload_and_no_sequence_lengths() -> None:
    assert _specialist_role(WORKLOAD) == "Llama-3-8B on sglang, bf16, TP 2, ISL 1024, OSL 512, concurrency 16, mi355x"
    assert _specialist_role(AGENTX) == (
        "Llama-3-8B on sglang, bf16, TP 2, AgentX trace replay graded on P50 interactivity (e2e_norm_intvty_p50), "
        "concurrency 16, mi355x"
    )
    assert "graded on total token throughput (total_token_throughput)," in _specialist_role(TOTAL_AGENTX)


def test_the_agents_spec_names_the_pack_the_session_gives_its_arbors_and_none_by_default() -> None:
    assert _agent_spec(WORKLOAD, "agentx-qwen38-1").pack == "agentx-qwen38-1"
    assert _agent_spec(WORKLOAD).pack is None


def test_the_agents_spec_names_the_launch_file_the_session_gives_its_arbors_and_none_by_default() -> None:
    assert _agent_spec(WORKLOAD, "", "/runs/launch.json").launch == "/runs/launch.json"
    assert _agent_spec(WORKLOAD).launch is None


def test_the_agents_spec_starts_its_port_blocks_where_the_session_says() -> None:
    assert _agent_spec(WORKLOAD, "", "", 32100).port == 32100
    assert _agent_spec(WORKLOAD).port == AGENT_PORT


def test_the_agents_spec_hands_their_review_gate_over_as_the_session_says() -> None:
    assert _agent_spec(WORKLOAD, "", "", AGENT_PORT, "host").review == "host"
    assert _agent_spec(WORKLOAD).review == ""


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


def test_a_model_that_resolves_to_no_local_directory_refuses_the_local_peers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HL_MODEL_BASE", str(tmp_path / "models"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    hub_id = replace(AGENTX, model="amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4")

    with pytest.raises(PeersUnavailable, match=r"amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4 .*HL_MODEL_BASE="):
        load_peers(hub_id)


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


@pytest.mark.usefixtures("four_cards")
@pytest.mark.parametrize("workload", [WORKLOAD, AGENTX], ids=["inferencex", "agentx"])
def test_the_host_is_this_machines_cards_on_the_launchers_image_and_the_staged_inferencex_client(
    workload: Workload, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = verified_tool_cache(tmp_path / "tools", monkeypatch)
    env = {IMAGE_VARS[0]: IMAGE.ref, IMAGE_VARS[1]: IMAGE.digest}

    host = measurement_host(env, LocalCards(machine_cards()))

    assert host == MeasurementHost(
        LocalCards((4, 5, 6, 7)), IMAGE, InferenceXCheckout(INFERENCEX_CLIENT_COMMIT, str(client))
    )
    assert machine_parallel_tests(2) == 2


def test_a_tool_cache_whose_staging_does_not_verify_refuses_the_peers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = tmp_path / "tools"
    env = {IMAGE_VARS[0]: IMAGE.ref, IMAGE_VARS[1]: IMAGE.digest}
    monkeypatch.setenv(TOOL_CACHE_ENV, str(tools))

    with pytest.raises(PeersUnavailable, match=f"no inference_suite staging .*{tools}"):
        measurement_host(env, LOCAL)
    client = verified_tool_cache(tools, monkeypatch)
    (client / INFERENCEX_REQUIRED[0]).write_text("edited\n")
    with pytest.raises(PeersUnavailable, match=f"no inference_suite staging .*{tools}"):
        measurement_host(env, LOCAL)


@pytest.mark.parametrize("mask", ["ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"])
def test_the_cards_are_the_ids_the_visible_device_mask_names(mask: str, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in COUNTING_VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(mask, "6,2")

    assert machine_cards() == (6, 2)


def _shown_and_held(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, shown: int, held: int) -> None:
    """No visible-device mask; ``rocm-smi`` shows ``shown`` cards and the machine holds ``held`` render nodes, beside a
    node that is not one."""

    for name in COUNTING_VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(visible_devices, "rocm_smi_gpu_count", lambda: shown)
    nodes = tmp_path / "dri"
    nodes.mkdir()
    (nodes / "card1").touch()
    for number in range(held):
        (nodes / f"renderD{128 + number}").touch()
    monkeypatch.setattr(local_peers_module, "RENDER_NODE_DIR", str(nodes))


@pytest.mark.parametrize(("shown", "held"), [(8, 8), (2, 3), (0, 0)])
def test_without_a_mask_the_cards_are_as_many_as_rocm_smi_shows_when_the_machine_can_open_them(
    shown: int, held: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _shown_and_held(monkeypatch, tmp_path, shown, held)

    assert machine_cards() == tuple(range(shown))


def test_a_container_given_fewer_cards_than_rocm_smi_shows_names_both_counts_and_the_mask_to_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _shown_and_held(monkeypatch, tmp_path, 8, 2)

    with pytest.raises(PeersUnavailable) as refused:
        machine_cards()

    assert str(refused.value) == (
        f"rocm-smi shows 8 cards, more than the 2 this machine can open (its render nodes, {tmp_path / 'dri'}/renderD*):"
        " name the cards it was given in ROCR_VISIBLE_DEVICES"
    )
    with pytest.raises(ValueError, match="rocm-smi shows 8 cards, more than the 2 this machine can open"):
        machine_parallel_tests(1)


def test_a_mask_names_the_cards_whatever_render_nodes_the_machine_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _shown_and_held(monkeypatch, tmp_path, 8, 0)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5")

    assert machine_cards() == (4, 5)


def test_a_machine_without_render_nodes_holds_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local_peers_module, "RENDER_NODE_DIR", str(tmp_path / "missing"))

    assert render_nodes() == 0


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
        measurement_host({}, LOCAL)


@pytest.mark.parametrize(
    ("env", "named"),
    [
        ({IMAGE_VARS[0]: IMAGE.ref}, IMAGE_VARS[1]),
        ({IMAGE_VARS[0]: IMAGE.ref, IMAGE_VARS[1]: "sha256:" + "0" * 64}, "not a real sha256"),
    ],
    ids=["half-named-image", "zero-digest"],
)
def test_an_image_that_cannot_be_pinned_refuses_the_peers(env: dict[str, str], named: str) -> None:
    with pytest.raises(PeersUnavailable, match=named):
        measurement_host(env, LOCAL)


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
    freeze: bool = True,
    inferencex_commit: str = "",
    reference_script: str = "",
    base: BaseLaunch | None = None,
    agent_launch: str = "/launches/arbor.json",
) -> AsyncIterator[_Built]:
    """The real peers of a session of ``workload`` on ``host`` in ``tmp_path`` lending ``dev_leases`` dev sandboxes,
    with the wiring their specialist sessions were built on; a branch's worktree, holding no component, is served
    only when the session lends dev sandboxes. The session froze ``base`` (a fixed one by default) unless ``freeze``
    is false, from the operator's ``reference_script`` when one is given. Every variable the build exports is restored afterwards."""

    for name in (*workload_environment(workload), BENCHMARK_CONFIG_ENV):
        monkeypatch.setenv(name, "")
    monkeypatch.delenv("GPU_TYPE", raising=False)
    monkeypatch.delenv("HYPERLOOM_KB_URL", raising=False)
    clock = Clock()
    layout = create_session(tmp_path, workload.model_basename, clock)
    if freeze:
        freeze_base(layout.toolbox, base or base_launch())
    desk = ForkDesk(Roster(RosterCaps(1, 1, 1), layout.roster, clock), DecisionLog(layout.decisions, clock), clock)
    blobs = LocalBlobStore(layout.blobs)
    setup = PeerSetup(
        layout,
        workload,
        desk,
        ProcessRegistry(layout.pids, clock),
        blobs,
        RefusingCritic(),
        {"geak": layout.reference / "geak"},
        dev_leases,
        (),
        reference_script,
        agent_launch=agent_launch,
        inferencex_commit=inferencex_commit,
    )
    wired: list[SessionWiring] = []

    def sessions(wiring: SessionWiring) -> ModelSessions:
        wired.append(wiring)

        def factory(start: SessionStart) -> object:
            raise AssertionError("no specialist runs here")

        def worktrees(branch: str) -> BranchWorktree:
            if not dev_leases:
                raise AssertionError("no worktree is used here")
            return SimpleNamespace(path=tmp_path / branch, holds=lambda component: False, linked=())

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


async def test_an_agentx_session_measures_the_agentx_benchmark_it_writes_for_its_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "models" / "Llama-3-8B").mkdir(parents=True)
    monkeypatch.setenv("HL_MODEL_BASE", str(tmp_path / "models"))
    workload = replace(AGENTX, model="meta-llama/Llama-3-8B")
    host = HOST
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
        assert is_agentx(canonical) and canonical["agentx"]["mode"] == "canonical"
        assert canonical["inferencex_path"] == host.inferencex.path
        assert [(spec.image_digest, spec.gpu_type) for spec in specs] == [(host.image.digest, workload.gpu_type)]
        assert (canonical["model"], canonical["envs"]["MODEL_PATH"]) == (
            workload.model,
            str(tmp_path / "models" / "Llama-3-8B"),
        )
        assert (str(canonical["envs"]["TP"]), str(canonical["envs"]["CONC"])) == (str(workload.tp), str(workload.conc))
        launch, frozen = canonical["agentx"]["launch_overrides"], frozen_base(toolbox)
        assert frozen is not None and frozen.launch.args
        assert launch["replace_args"] is True
        assert launch["append_args"][: len(frozen.launch.args)] == list(frozen.launch.args)


async def test_an_agentx_session_on_a_model_directory_serves_it_at_its_ep_in_the_sessions_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "models" / "DeepSeek-V3"
    model.mkdir(parents=True)
    workload = replace(AGENTX, model=str(model), ep=8)
    host = HOST

    async with _peers(tmp_path / "workspace", monkeypatch, workload, host) as built:
        benchmark = yaml.safe_load((built.layout.toolbox / "benchmark.yaml").read_text())["benchmark"]
        assert (benchmark["docker_image"], benchmark["envs"]["EP_SIZE"]) == (IMAGE.ref, 8)
        assert benchmark["envs"]["MODEL_PATH"] == str(model)
        assert "DURATION" not in benchmark["envs"]


async def test_an_agentx_session_replays_the_trace_window_it_names_and_its_tests_are_limited_by_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "models" / "GLM-5.3"
    model.mkdir(parents=True)
    workload = replace(AGENTX, model=str(model), agentx_duration_sec=600)
    given = _options_given(monkeypatch, local_hosted, "build_local_toolbox")

    async with _peers(tmp_path / "workspace", monkeypatch, workload, HOST) as built:
        canonical = yaml.safe_load((built.layout.toolbox / "canonical" / "canonical.yaml").read_text())["benchmark"]
        assert str(canonical["envs"]["DURATION"]) == "600"

    [options] = given
    assert options.timeout_sec == 600 + AGENTX_OUTSIDE_TRACE_SEC


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

    async with _peers(tmp_path / "agentx", monkeypatch, replace(AGENTX, model=str(model)), HOST):
        pass
    async with _peers(tmp_path / "inferencex", monkeypatch, WORKLOAD):
        pass

    [agentx, inferencex] = given
    assert agentx.timeout_sec > max(AGENTX_TRACE_SEC, AGENTX_TEST_PAST_AN_HOUR_SEC)
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


@pytest.fixture(autouse=True)
def publishes_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test here fetches the staging a local session publishes for its runners."""

    monkeypatch.setattr(suite_publish, "publish_own_inference_suite", _Publishes())


class _Publishes:
    """Publishes the runners' staging by name only, recording the runner cache of each publish."""

    def __init__(self, refusal: str = "") -> None:
        self.caches: list[Path] = []
        self.refusal = refusal

    def __call__(self, cache_root: Path) -> Path:
        if self.refusal:
            raise SuitePublishError(self.refusal)
        self.caches.append(cache_root)
        return cache_root / "inference-suite" / "key"


async def test_a_local_session_publishes_the_staging_its_runners_read_into_their_runner_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HYPERLOOM_RUNNER_CACHE", str(tmp_path / "runner-cache"))
    publishes = _Publishes()
    monkeypatch.setattr(suite_publish, "publish_own_inference_suite", publishes)

    async with _peers(tmp_path / "workspace", monkeypatch, WORKLOAD):
        assert publishes.caches == [(tmp_path / "runner-cache").resolve()]


@pytest.mark.parametrize("tests_end_by", [None, 1_800_000_000.0])
def test_the_local_cards_hand_their_toolbox_the_time_every_test_must_end_by(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tests_end_by: float | None
) -> None:
    monkeypatch.setenv("HYPERLOOM_RUNNER_CACHE", str(tmp_path / "runner-cache"))
    monkeypatch.setattr(suite_publish, "publish_own_inference_suite", _Publishes())
    built: list[object] = []
    monkeypatch.setattr(local_hosted, "build_local_toolbox", lambda *args, options: built.append(options) or options)
    layout = SimpleNamespace(toolbox=tmp_path / "MB-toolbox", session_id="MB-1")
    setup = SimpleNamespace(layout=layout, workload=WORKLOAD)

    LocalCards((0, 1), tests_end_by).toolbox(setup, None, None, None)  # type: ignore[arg-type]

    [options] = built
    assert options.tests_end_by == tests_end_by  # type: ignore[attr-defined]


async def test_a_local_session_whose_staging_cannot_be_published_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(suite_publish, "publish_own_inference_suite", _Publishes("exited 2: staging_missing: offline"))

    with pytest.raises(PeersUnavailable, match="cannot be published: exited 2: staging_missing: offline"):
        async with _peers(tmp_path, monkeypatch, WORKLOAD):
            pass


async def test_a_fresh_session_freezes_its_base_launch_in_the_toolbox_directory_before_any_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HL_MODEL_BASE", raising=False)
    queries: list[object] = []

    def offline(query: object, *, user_file: object = None, inferencex_commit: str = "") -> SourceSnapshot:
        queries.append(query)
        return SourceSnapshot(None, 1_793_000_000, user_file=user_file, unread=UnreadSource("InferenceX", "offline"))

    monkeypatch.setattr(cli_base_launch, "fetch_sources", offline)

    async with _peers(tmp_path, monkeypatch, WORKLOAD, freeze=False) as built:
        frozen = frozen_base(built.layout.toolbox)
        assert frozen is not None and [frozen.launch.query] == queries
        assert frozen.launch.topology.tp == WORKLOAD.tp


async def test_a_fresh_session_reads_inferencex_at_the_commit_its_setup_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HL_MODEL_BASE", raising=False)
    commits: list[str] = []

    def offline(query: object, *, user_file: object = None, inferencex_commit: str = "") -> SourceSnapshot:
        commits.append(inferencex_commit)
        return SourceSnapshot(None, 1_793_000_000, user_file=user_file, unread=UnreadSource("InferenceX", "offline"))

    monkeypatch.setattr(cli_base_launch, "fetch_sources", offline)

    async with _peers(tmp_path, monkeypatch, WORKLOAD, freeze=False, inferencex_commit="8a76a78e41"):
        assert commits == ["8a76a78e41"]


async def test_a_reference_script_is_frozen_as_the_base_and_served_to_the_arbors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HL_MODEL_BASE", raising=False)
    script = tmp_path / "launch.sh"
    script.write_text(
        "python3 -m sglang.launch_server --model-path /models/Llama-3-8B --tp 2 --chunked-prefill-size 8\n"
    )

    async with _peers(
        tmp_path / "run", monkeypatch, WORKLOAD, freeze=False, reference_script=str(script), agent_launch=""
    ) as built:
        frozen = frozen_base(built.layout.toolbox)
        assert frozen is not None and frozen.launch.source.path == str(script)
        assert built.wiring.setup.agent_launch == str(frozen.path)


async def test_without_a_launch_file_the_arbors_are_served_the_session_s_frozen_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _peers(tmp_path, monkeypatch, WORKLOAD, agent_launch="") as built:
        frozen = frozen_base(built.layout.toolbox)
        assert frozen is not None and built.wiring.setup.agent_launch == str(frozen.path)


async def test_an_arbor_launch_file_is_served_in_place_of_the_frozen_base(built: _Built) -> None:
    assert built.wiring.setup.agent_launch == "/launches/arbor.json"


async def test_the_specialists_are_wired_the_frozen_base_launch_s_server_half(built: _Built) -> None:
    frozen = frozen_base(built.layout.toolbox)

    assert frozen is not None and built.wiring.base == server_base(frozen.launch)
    assert built.wiring.base.args


async def test_the_real_peers_live_inside_the_session_directory(built: _Built) -> None:
    layout, setup, peers, wiring = built.layout, built.setup, built.peers, built.wiring
    assert (wiring.setup, wiring.repo, wiring.briefs) == (setup, peers.repo, peers.briefs)
    assert isinstance(peers.lifecycle, SpecialistRuntime) and peers.briefs.warm == ()
    assert peers.briefs.repo is peers.repo and isinstance(peers.briefs.learnings, LessonRecall)
    assert isinstance(peers.toolbox, LocalHostedToolbox) and isinstance(peers.repo, GitExperimentRepo)
    canonical = yaml.safe_load((layout.toolbox / "canonical" / "canonical.yaml").read_text())["benchmark"]
    assert (canonical["model"], canonical["inferencex_path"]) == (WORKLOAD.model, HOST.inferencex.path)
    assert "runner_type" not in canonical
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
    setup = SimpleNamespace(
        layout=layout, workload=replace(WORKLOAD, tp=4), processes=None, agents=("https://agents.example/tuner.git",)
    )
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


@pytest.mark.parametrize(
    ("problems", "launched", "level", "said_first"),
    [
        ((), BWRAP, logging.INFO, "agent isolation: specialists bwrap, sandboxes bwrap, tests bwrap"),
        (
            ("bwrap: not found",),
            NONE,
            logging.WARNING,
            "specialists and agent and dev sandboxes and tests run unconfined on this host (",
        ),
    ],
    ids=["bubblewrap", "no-bubblewrap"],
)
async def test_peers_on_this_machine_record_and_say_once_how_their_sandboxes_and_tests_run(
    problems: tuple[str, ...],
    launched: str,
    level: int,
    said_first: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("HL_MODEL_BASE", raising=False)
    monkeypatch.delenv(SPECIALIST_USER_ENV, raising=False)
    monkeypatch.setattr(agent_isolation, "_STATE", agent_isolation._Process())
    monkeypatch.setattr(bwrap_module, "_PROBES", {})
    monkeypatch.setattr(
        bwrap_module, "_probe", lambda path, *_: BwrapProbe(path or "/usr/bin/bwrap", (0, 13, 1), problems)
    )

    with caplog.at_level(logging.INFO, logger=launch_isolation_module.__name__):
        async with _peers(tmp_path, monkeypatch, WORKLOAD) as built:
            isolation = built.peers.isolation

    assert isolation == launch_isolation(local=True)
    assert (isolation.specialists, isolation.sandboxes, isolation.tests) == (launched, launched, launched)
    said = [record for record in caplog.records if record.name == launch_isolation_module.__name__]
    assert [record.levelno for record in said] == [level]
    assert said[0].getMessage().startswith(said_first)


async def test_peers_measuring_in_the_services_sandboxes_record_their_sandboxes_and_tests_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HL_MODEL_BASE", raising=False)
    host = replace(HOST, backend=_service_sandboxes(monkeypatch, []))

    async with _peers(tmp_path, monkeypatch, WORKLOAD, host) as built:
        isolation = built.peers.isolation

    assert (isolation.sandboxes, isolation.tests) == (HOSTED, HOSTED), "the grant's sandboxes run them, not this host"
    here = (agent_isolation.SANDBOX_ACCOUNT, agent_isolation.TEST_ACCOUNT)
    assert not any(why in reason for reason in isolation.reasons for why in here), "nor why they are not isolated here"


async def test_an_agentx_test_in_the_services_sandboxes_may_outlast_its_trace_window_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "models" / "GLM-5.3"
    model.mkdir(parents=True)
    given = _options_given(monkeypatch, hosted_toolbox, "build_hosted_toolbox")
    host = replace(HOST, backend=_service_sandboxes(monkeypatch, []))

    async with _peers(tmp_path / "workspace", monkeypatch, replace(AGENTX, model=str(model)), host):
        pass

    [agentx] = given
    assert agentx.timeout_sec > max(AGENTX_TRACE_SEC, AGENTX_TEST_PAST_AN_HOUR_SEC)


def _agent_clients(monkeypatch: pytest.MonkeyPatch) -> list[GateClient | None]:
    """The gate's client every session's agents are built with from here on; no specialist session is built."""

    clients: list[GateClient | None] = []

    def agents(*args: object, client: GateClient | None = None, dossier: str | None = None, **_: object) -> object:
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
    assert client is not None
    assert (client.workload.concurrency, client.workload.isl, client.workload.osl) == (WORKLOAD.conc, 1024, 512)


@pytest.mark.parametrize(
    ("given", "dossier"), [("/reviewed/dossier/qwen3-8b", "/reviewed/dossier/qwen3-8b"), ("", None)]
)
async def test_the_agents_are_given_the_reviewed_dossier_the_run_names_and_none_without(
    built: _Built, monkeypatch: pytest.MonkeyPatch, given: str, dossier: str | None
) -> None:
    dossiers: list[str | None] = []

    def agents(*args: object, client: GateClient | None = None, dossier: str | None = None, **_: object) -> object:
        dossiers.append(dossier)
        return SimpleNamespace()

    _agent_clients(monkeypatch)
    monkeypatch.setattr(agent_sessions_module, "AgentSessions", agents)
    wiring = _with_agent(built.wiring)

    model_backed_sessions(AgentCommand("claude", "claude"))(
        replace(wiring, setup=replace(wiring.setup, agent_dossier=given))
    )

    assert dossiers == [dossier]


def _agentx_base() -> BaseLaunch:
    """An AgentX base frozen at the workload's topology, its client env naming the corpus the replay reads."""

    from hyperloom.orchestrator.actions.executors.tests.test_suite_accuracy_round import _agentx_base as agentx_base

    base = agentx_base()
    return replace(
        base,
        topology=replace(base.topology, tp=AGENTX.tp),
        client=replace(base.client, env={"WEKA_LOADER_OVERRIDE": "semianalysis_cc_traces_weka_062126"}),
    )


async def test_an_agent_of_an_agentx_session_measures_with_the_gate_s_replay_bounded_as_one_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "models" / "Llama-3-8B").mkdir(parents=True)
    monkeypatch.setenv("HL_MODEL_BASE", str(tmp_path / "models"))
    workload = replace(AGENTX, model="meta-llama/Llama-3-8B")
    clients = _agent_clients(monkeypatch)

    async with _peers(tmp_path / "workspace", monkeypatch, workload, HOST, base=_agentx_base()) as built:
        verified_tool_cache(tmp_path / "tools", monkeypatch)
        model_backed_sessions(AgentCommand("claude", "claude"))(_with_agent(built.wiring))

    [client] = clients
    assert isinstance(client, AgentxGateClient)
    assert client.seconds == attempt_timeout_sec(workload)
    assert client.config.agentx is not None and client.config.agentx.concurrency == workload.conc
    assert client.config.server.base.path == str(built.layout.toolbox / "base_launch.json")


async def test_an_agentx_session_whose_base_cannot_replay_refuses_its_agents_the_peers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "models" / "Llama-3-8B").mkdir(parents=True)
    monkeypatch.setenv("HL_MODEL_BASE", str(tmp_path / "models"))
    workload = replace(AGENTX, model="meta-llama/Llama-3-8B")
    _agent_clients(monkeypatch)
    base = replace(_agentx_base(), client=replace(_agentx_base().client, env={}))

    async with _peers(tmp_path / "workspace", monkeypatch, workload, HOST, base=base) as built:
        verified_tool_cache(tmp_path / "tools", monkeypatch)
        with pytest.raises(PeersUnavailable, match="WEKA_LOADER_OVERRIDE"):
            model_backed_sessions(AgentCommand("claude", "claude"))(_with_agent(built.wiring))


async def test_a_session_without_agents_reads_no_canonical(built: _Built, monkeypatch: pytest.MonkeyPatch) -> None:
    clients = _agent_clients(monkeypatch)
    unread = replace(built.wiring, canonical=built.layout.toolbox / "no-such-canonical.yaml")

    model_backed_sessions(AgentCommand("claude", "claude"))(unread)

    assert clients == [None]


async def test_an_agentx_session_s_specialists_are_told_its_objective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[AgentSetup] = []

    def sessions(setup: AgentSetup, **_: object) -> ModelSessions:
        built.append(setup)
        return ModelSessions(lambda start: None, lambda branch: None)

    monkeypatch.setattr(model_sessions_module, "model_sessions", sessions)
    layout = SimpleNamespace(logs=tmp_path, source=tmp_path, branch_dir=lambda branch: tmp_path)
    setup = SimpleNamespace(
        layout=layout,
        workload=TOTAL_AGENTX,
        references={},
        processes=None,
        agents=(),
        agent_pack="",
        agent_launch="",
        agent_port=AGENT_PORT,
        agent_dossier="",
        agent_review="",
        agent_maps=None,
        agent_share_minutes=SHARE_MINUTES,
    )
    wiring = SimpleNamespace(
        setup=setup,
        tokens=None,
        repo=None,
        briefs=SimpleNamespace(text=None),
        toolbox=None,
        blackboards=None,
        dev_leases=None,
        native={},
        agent_sandboxes=None,
        base=BASE,
    )
    model_backed_sessions(AgentCommand("claude", "claude"))(wiring)

    objective = AgentXObjective.TOTAL_TOKEN_THROUGHPUT
    assert built[0].role == role_prompt(
        RoleWorkload(_specialist_role(TOTAL_AGENTX), objective), references={}, base=BASE
    )
    assert objective.description in " ".join(built[0].role.split())


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
                agent_dossier="",
                agent_review="",
                agent_maps=None,
                agent_share_minutes=SHARE_MINUTES,
            ),
            tokens=None,
            repo=None,
            briefs=SimpleNamespace(text=None),
            toolbox=None,
            blackboards=None,
            dev_leases=held,
            native={},
            agent_sandboxes=None,
            base=BASE,
        )
        model_backed_sessions(AgentCommand("claude", "claude"))(wiring)

    without, lending = built
    assert (without.dev_leases, lending.dev_leases) == (None, leases)
    assert without.role == role_prompt(RoleWorkload(_specialist_role(WORKLOAD)), references={}, base=BASE)
    assert "Server arguments: --tp-size 2 --enable-hierarchical-cache --hicache-size 180." in without.role
    assert (
        lending.role
        == role_prompt(RoleWorkload(_specialist_role(WORKLOAD)), references={}, base=BASE, dev_leases=1)
        != without.role
    )
    # A hosted lease's sandbox roots in the branch's directory: the branch's specialist is shown that area alone.
    assert without.lease_area is None and lending.lease_area is not None
    assert lending.lease_area("s1") == tmp_path / DEV_DIR and (tmp_path / DEV_DIR).is_dir()


def test_the_specialists_read_the_session_s_references_and_the_model_directory_it_serves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "models" / "Llama-3-8B").mkdir(parents=True)
    monkeypatch.setenv("HL_MODEL_BASE", str(tmp_path / "models"))
    references = {"geak": tmp_path / "MB-reference" / "geak"}
    served = replace(WORKLOAD, model="meta-llama/Llama-3-8B")
    unknown = replace(WORKLOAD, model="meta-llama/Not-Here")

    assert specialist_readable(references, served) == (references["geak"], tmp_path / "models" / "Llama-3-8B")
    assert specialist_readable({}, unknown) == ()


def test_a_local_toolbox_s_specialist_reads_its_own_holder_area_of_the_leases(tmp_path: Path) -> None:
    toolbox = object.__new__(LocalHostedToolbox)
    object.__setattr__(toolbox, "leases", BranchLeases(object(), tmp_path / "branch-sandboxes", 1, {}))
    wiring = SimpleNamespace(toolbox=toolbox, dev_leases=object(), setup=None)

    area = _lease_area(wiring)

    assert area is not None and area("s1") == tmp_path / "branch-sandboxes" / "s1" and area("s1").is_dir()
    assert _lease_area(SimpleNamespace(toolbox=toolbox, dev_leases=None, setup=None)) is None


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

    monkeypatch.setattr(worktree_module, "locate_installed_package", locate)
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
            agent_dossier="",
            agent_review="",
            agent_maps=None,
            agent_share_minutes=SHARE_MINUTES,
        ),
        tokens=None,
        repo=None,
        briefs=SimpleNamespace(text=None),
        toolbox=None,
        blackboards=None,
        dev_leases=None,
        native={},
        agent_sandboxes=None,
        base=BASE,
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

    monkeypatch.setattr(worktree_module, "locate_installed_package", locate)
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
                agent_dossier="",
                agent_review="",
                agent_maps=None,
                agent_share_minutes=SHARE_MINUTES,
            ),
            tokens=None,
            repo=None,
            briefs=SimpleNamespace(text=None),
            toolbox=None,
            blackboards=None,
            dev_leases=None,
            native=native,
            agent_sandboxes=None,
            base=BASE,
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
    assert without == role_prompt(RoleWorkload(_specialist_role(WORKLOAD)), references={}, base=BASE)
