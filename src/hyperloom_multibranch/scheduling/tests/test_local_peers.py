# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real peers of a session: built inside the session directory on model-backed specialist sessions and on this
machine's measurement host, and refused when this machine has no agent CLI for them, its pins cannot be read or the
image lacks the framework's package."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import Recipe
from hyperloom.common.visible_devices import COUNTING_VISIBLE_DEVICE_VARS
from hyperloom.measure import compile as compile_module
from hyperloom.measure.compile import ComponentLayout, LocalBlobStore, PackageBase, UnsupportedComponentError
from hyperloom.measure.pins import ActiveImage, ToolCommits, pinned_tool_commits
from hyperloom.orchestrator.kernel.request_handlers import resolve_gemm_tuning_backend
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.toolbox import SCOPE_SPECIALIST, SpecialistGrant
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.rulegate.service import BASELINE_TESTS, GateService
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.forks import ForkDesk
from hyperloom_multibranch.scheduling.layout import SESSION_PREFIX, create_session
from hyperloom_multibranch.scheduling.local_peers import (
    MB_FAKE_IMAGE_REF,
    MeasurementHost,
    SessionWiring,
    local_peers,
    machine_cards,
    machine_parallel_tests,
    measurement_host,
    require_framework_package,
    specialist_command,
    workload_environment,
)
from hyperloom_multibranch.scheduling.mb_fakes import MB_FAKE_BASE_IMAGE_DIGEST
from hyperloom_multibranch.scheduling.peers import PeerSetup, PeersUnavailable
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.roster import Roster, RosterCaps
from hyperloom_multibranch.scheduling.settings import KernelOptimizer, Workload
from hyperloom_multibranch.specialist.model_sessions import ModelSessions
from hyperloom_multibranch.specialist.runtime import SessionStart, SpecialistRuntime
from hyperloom_multibranch.specialist.worktree import BranchWorktree
from hyperloom_multibranch.toolbox.local_hosted import LocalHostedToolbox
from hyperloom_multibranch.toolbox.mcp_tools import McpSpecialistTokens
from hyperloom_multibranch.toolbox.queue import QueuePolicy

WORKLOAD = Workload("/models/Llama-3-8B", "sglang", 2, 1024, 512, 16, "bf16", "mi355x", KernelOptimizer.FORGE)
IMAGE = ActiveImage("registry.example/sglang:v1", "sha256:" + "2" * 64)
HOST = MeasurementHost((0, 1, 2, 3), IMAGE, ToolCommits("a" * 40, "b" * 40))
IMAGE_VARS = ("HYPERLOOM_ACTIVE_IMAGE_REF", "HYPERLOOM_ACTIVE_IMAGE_DIGEST")


def _call_tool(grant: SpecialistGrant, name: str) -> dict[str, object]:
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": {}}}
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


@pytest.mark.parametrize("optimizer", list(KernelOptimizer))
def test_the_kernel_optimizer_flag_picks_the_backend_kernel_tuning_runs(
    optimizer: KernelOptimizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in workload_environment(replace(WORKLOAD, kernel_optimizer=optimizer)).items():
        monkeypatch.setenv(name, value)
    assert resolve_gemm_tuning_backend({}) == optimizer.value


def test_the_model_is_exported_as_the_directory_a_server_loads_it_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "Llama-3-8B").mkdir()
    monkeypatch.setenv("HL_MODEL_BASE", str(tmp_path))
    hub_id = Workload("meta-llama/Llama-3-8B", "vllm", 1, 1, 1, 1, "fp8", "mi355x", KernelOptimizer.FORGE)
    assert workload_environment(hub_id)["MODEL_PATH"] == str(tmp_path / "Llama-3-8B")


def test_the_specialists_run_the_agent_cli_found_on_path(tmp_path: Path) -> None:
    cli = tmp_path / "claude"
    cli.write_text("#!/bin/sh\nexit 0\n")
    cli.chmod(0o755)

    command = specialist_command({"PATH": str(tmp_path), "ANTHROPIC_API_KEY": "dummy", "CLAUDE_MODEL": "m-1"})

    assert (command.backend, command.executable, command.model) == ("claude", str(cli), "m-1")


def test_without_the_agent_cli_the_peers_are_refused(tmp_path: Path) -> None:
    with pytest.raises(PeersUnavailable, match="claude CLI the specialists run is not on PATH"):
        specialist_command({"PATH": str(tmp_path), "ANTHROPIC_API_KEY": "dummy"})


@pytest.fixture
def four_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    """This machine shows four cards, the upper half of an eight-card node."""

    for name in COUNTING_VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,5,6,7")


@pytest.mark.usefixtures("four_cards")
def test_the_host_is_this_machines_cards_on_the_launchers_image_and_the_pinned_tool_commits() -> None:
    env = {IMAGE_VARS[0]: IMAGE.ref, IMAGE_VARS[1]: IMAGE.digest}

    host = measurement_host(env)

    assert host == MeasurementHost((4, 5, 6, 7), IMAGE, pinned_tool_commits({}))
    assert machine_parallel_tests(2) == 2


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


@pytest.mark.usefixtures("four_cards")
def test_a_session_nobody_names_the_image_of_runs_on_the_placeholder_image() -> None:
    assert measurement_host({}).image == ActiveImage(MB_FAKE_IMAGE_REF, MB_FAKE_BASE_IMAGE_DIGEST)


@pytest.mark.parametrize(
    ("env", "named"),
    [
        ({IMAGE_VARS[0]: IMAGE.ref}, IMAGE_VARS[1]),
        ({IMAGE_VARS[0]: IMAGE.ref, IMAGE_VARS[1]: MB_FAKE_BASE_IMAGE_DIGEST}, "not a real sha256"),
        ({"MAGPIE_REF": "main"}, "MAGPIE_REF"),
    ],
    ids=["half-named-image", "zero-digest", "moving-tool-ref"],
)
@pytest.mark.usefixtures("four_cards")
def test_pins_that_cannot_be_read_refuse_the_peers(env: dict[str, str], named: str) -> None:
    with pytest.raises(PeersUnavailable, match=named):
        measurement_host(env)


def test_a_machine_that_shows_no_card_fits_no_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="0 cards"):
        machine_parallel_tests(1)


async def test_the_real_peers_live_inside_the_session_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HL_MODEL_BASE", raising=False)
    for name in workload_environment(WORKLOAD):
        monkeypatch.setenv(name, "")
    monkeypatch.delenv("GPU_TYPE", raising=False)
    monkeypatch.delenv("HYPERLOOM_KB_URL", raising=False)
    clock = Clock()
    layout = create_session(tmp_path, WORKLOAD.model_basename, clock)
    desk = ForkDesk(Roster(RosterCaps(1, 1, 1), layout.roster, clock), DecisionLog(layout.decisions, clock), clock)
    setup = PeerSetup(layout, WORKLOAD, desk, ProcessRegistry(layout.pids, clock), LocalBlobStore(layout.blobs))
    wired: list[SessionWiring] = []

    def sessions(wiring: SessionWiring) -> ModelSessions:
        wired.append(wiring)

        def factory(start: SessionStart) -> object:
            raise AssertionError("no specialist runs here")

        def worktrees(branch: str) -> BranchWorktree:
            raise AssertionError("no worktree is used here")

        return ModelSessions(factory, worktrees)

    peers = local_peers(HOST, sessions, setup)
    try:
        [wiring] = wired
        assert (wiring.setup, wiring.repo, wiring.briefs) == (setup, peers.repo, peers.briefs)
        assert isinstance(peers.lifecycle, SpecialistRuntime) and peers.briefs.warm == ()
        assert isinstance(peers.toolbox, LocalHostedToolbox) and isinstance(peers.repo, GitExperimentRepo)
        assert (layout.toolbox / "canonical" / "canonical.yaml").is_file()
        assert not (layout.toolbox / "measurement-config.json").exists()
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
            "notebook_brief",
            "request_repair",
            "decline_repair",
        }
        assert served <= tools
    finally:
        peers.mcp.stop()
        await peers.toolbox.close()
    assert all(entry.name.startswith(SESSION_PREFIX) for entry in layout.root.iterdir())
