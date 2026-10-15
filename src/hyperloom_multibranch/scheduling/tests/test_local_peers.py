# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real peers of a session: built inside the session directory, and refused while the specialist runtime has no
model-backed session factory."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import Recipe
from hyperloom.measure.compile import LocalBlobStore
from hyperloom.orchestrator.kernel.request_handlers import resolve_gemm_tuning_backend
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.contracts.toolbox import SCOPE_SPECIALIST, SpecialistGrant, SpecialistTokens
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.rulegate.service import BASELINE_TESTS, GateService
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.decisions import DecisionLog
from hyperloom_multibranch.scheduling.forks import ForkDesk
from hyperloom_multibranch.scheduling.layout import SESSION_PREFIX, create_session
from hyperloom_multibranch.scheduling.local_peers import (
    MB_FAKE_MISSING_SESSION_FACTORY,
    MeasurementHost,
    load_peers,
    local_peers,
    mb_fake_measurement_host,
    workload_environment,
)
from hyperloom_multibranch.scheduling.mb_fakes import MB_FAKE_BASE_IMAGE_DIGEST
from hyperloom_multibranch.scheduling.peers import PeerSetup, PeersUnavailable
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.roster import Roster, RosterCaps
from hyperloom_multibranch.scheduling.settings import KernelOptimizer, Workload
from hyperloom_multibranch.scheduling.tests.peer_fakes import RecordedSessions
from hyperloom_multibranch.specialist.runtime import SpecialistRuntime
from hyperloom_multibranch.toolbox.local_hosted import LocalHostedToolbox
from hyperloom_multibranch.toolbox.mcp_tools import McpSpecialistTokens
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.toolbox import RealToolbox

WORKLOAD = Workload("/models/Llama-3-8B", "sglang", 2, 1024, 512, 16, "bf16", "mi355x", KernelOptimizer.FORGE)
HOST = MeasurementHost((0, 1, 2, 3), "local", MB_FAKE_BASE_IMAGE_DIGEST, "a" * 40, "b" * 40)


def _call_tool(grant: SpecialistGrant, name: str) -> dict[str, object]:
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": {}}}
    request = urllib.request.Request(
        grant.url,
        data=json.dumps(message).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {grant.token}"},
    )
    with urllib.request.urlopen(request, timeout=10) as reply:
        return json.loads(reply.read())


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


def test_loading_the_peers_names_the_missing_session_factory() -> None:
    with pytest.raises(PeersUnavailable) as refused:
        load_peers()
    assert refused.value.missing == MB_FAKE_MISSING_SESSION_FACTORY
    assert "SpecialistRuntime" in str(refused.value)


def test_the_placeholder_host_is_one_machine_on_the_root_recipes_image() -> None:
    host = mb_fake_measurement_host()
    assert host.cards == tuple(range(8)) and host.image_digest == MB_FAKE_BASE_IMAGE_DIGEST


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
    desk = ForkDesk(Roster(RosterCaps(1, 1, 1), layout.roster, clock), DecisionLog(layout.decisions, clock))
    setup = PeerSetup(layout, WORKLOAD, desk, ProcessRegistry(layout.pids, clock), LocalBlobStore(layout.blobs))
    built: list[tuple[PeerSetup, ExperimentRepo, RealToolbox]] = []
    granted: list[SpecialistTokens] = []

    def sessions(
        given: PeerSetup, repo: ExperimentRepo, toolbox: RealToolbox, tokens: SpecialistTokens
    ) -> RecordedSessions:
        built.append((given, repo, toolbox))
        granted.append(tokens)
        return RecordedSessions(toolbox, repo, LocalNotebook(layout.notebook, Writer(Role.OBSERVER, "observer")))

    peers = local_peers(HOST, sessions, setup)
    try:
        assert built == [(setup, peers.repo, peers.toolbox)] and isinstance(peers.lifecycle, SpecialistRuntime)
        assert isinstance(peers.toolbox, LocalHostedToolbox) and isinstance(peers.repo, GitExperimentRepo)
        assert (layout.toolbox / "canonical" / "canonical.yaml").is_file()
        config = json.loads(layout.measurement_config.read_text(encoding="utf-8"))
        assert config["base_image_digest"] == HOST.image_digest
        assert config["config"]["workload"]["kernel_optimizer"] == "forge"
        assert config["config"]["primary_concurrency"] == WORKLOAD.conc
        assert (config["config"]["magpie_commit"], config["config"]["cards"]) == ("a" * 40, [0, 1, 2, 3])
        assert peers.mcp.url.startswith("http://127.0.0.1:")
        assert {name: os.environ[name] for name in workload_environment(WORKLOAD)} == workload_environment(WORKLOAD)
        root = await peers.repo.init_root(Recipe(MB_FAKE_BASE_IMAGE_DIGEST))
        assert await peers.repo.head(INTEGRATION_BRANCH) == root
        patch = await peers.repo.store_patch("sglang", "--- a/x\n+++ b/x\n")
        assert LocalBlobStore(layout.blobs).get(patch.sha256) == b"--- a/x\n+++ b/x\n"
        assert isinstance(peers.gate, GateService) and await peers.gate.baseline_test_count() == BASELINE_TESTS
        await peers.toolbox.start()
        await peers.toolbox.set_policy(QueuePolicy(per_branch_k=1, global_concurrency=2))
        state = await peers.toolbox.queue_state()
        assert (state.queued, state.running) == (0, 0)
        [tokens] = granted
        assert isinstance(tokens, McpSpecialistTokens)
        grant = tokens.issue("b01", frozenset({SCOPE_SPECIALIST}))
        assert grant.url == peers.mcp.url
        reply = await asyncio.to_thread(_call_tool, grant, "queue_state")
        assert reply["result"]["isError"] is False
    finally:
        peers.mcp.stop()
        await peers.toolbox.close()
    assert all(entry.name.startswith(SESSION_PREFIX) for entry in layout.root.iterdir())
