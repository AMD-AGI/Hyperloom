# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The real peers of a session: built inside the session directory, and refused while the specialist runtime is
missing."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import Recipe
from hyperloom.measure.compile import LocalBlobStore
from hyperloom_multibranch.contracts.fork import Admission, AdmissionStatus
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.repo import ExperimentRepo
from hyperloom_multibranch.repo.git_repo import GitExperimentRepo
from hyperloom_multibranch.scheduling.clock import Clock
from hyperloom_multibranch.scheduling.layout import SESSION_PREFIX, create_session
from hyperloom_multibranch.scheduling.local_peers import (
    MB_FAKE_MISSING_PEER_MODULES,
    MeasurementHost,
    load_peers,
    local_peers,
    mb_fake_measurement_host,
    workload_environment,
)
from hyperloom_multibranch.scheduling.mb_fakes import MB_FAKE_BASE_IMAGE_DIGEST
from hyperloom_multibranch.scheduling.peers import PeerSetup, PeersUnavailable
from hyperloom_multibranch.scheduling.processes import ProcessRegistry
from hyperloom_multibranch.scheduling.settings import Workload
from hyperloom_multibranch.tests.fakes import RecordingLifecycle, RecordingScheduler
from hyperloom_multibranch.toolbox.local_hosted import LocalHostedToolbox
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.toolbox import RealToolbox

WORKLOAD = Workload("/models/Llama-3-8B", "sglang", 2, 1024, 512, 16, "bf16", "mi355x")
HOST = MeasurementHost((0, 1, 2, 3), "local", MB_FAKE_BASE_IMAGE_DIGEST, "a" * 40, "b" * 40)


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
    }


def test_the_model_is_exported_as_the_directory_a_server_loads_it_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "Llama-3-8B").mkdir()
    monkeypatch.setenv("HL_MODEL_BASE", str(tmp_path))
    hub_id = Workload("meta-llama/Llama-3-8B", "vllm", 1, 1, 1, 1, "fp8", "mi355x")
    assert workload_environment(hub_id)["MODEL_PATH"] == str(tmp_path / "Llama-3-8B")


def test_loading_the_peers_names_the_missing_specialist_runtime() -> None:
    with pytest.raises(PeersUnavailable) as refused:
        load_peers()
    assert refused.value.modules == MB_FAKE_MISSING_PEER_MODULES == ("hyperloom_multibranch.specialist",)


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
    clock = Clock()
    layout = create_session(tmp_path, WORKLOAD.model_basename, clock)
    desk = RecordingScheduler(Admission(AdmissionStatus.REJECTED, "test"))
    setup = PeerSetup(layout, WORKLOAD, desk, ProcessRegistry(layout.pids, clock), LocalBlobStore(layout.blobs))
    lifecycle = RecordingLifecycle()
    built: list[tuple[PeerSetup, ExperimentRepo, RealToolbox]] = []

    def lifecycles(given: PeerSetup, repo: ExperimentRepo, toolbox: RealToolbox) -> RecordingLifecycle:
        built.append((given, repo, toolbox))
        return lifecycle

    peers = local_peers(HOST, lifecycles, setup)
    try:
        assert built == [(setup, peers.repo, peers.toolbox)] and peers.lifecycle is lifecycle
        assert isinstance(peers.toolbox, LocalHostedToolbox) and isinstance(peers.repo, GitExperimentRepo)
        assert (layout.toolbox / "canonical" / "canonical.yaml").is_file()
        assert {name: os.environ[name] for name in workload_environment(WORKLOAD)} == workload_environment(WORKLOAD)
        root = await peers.write_root(Recipe(MB_FAKE_BASE_IMAGE_DIGEST))
        assert await peers.repo.head(INTEGRATION_BRANCH) == root
        await peers.toolbox.start()
        await peers.toolbox.set_policy(QueuePolicy(per_branch_k=1, global_concurrency=2))
        state = await peers.toolbox.queue_state()
        assert (state.queued, state.running) == (0, 0)
    finally:
        await peers.toolbox.close()
    assert all(entry.name.startswith(SESSION_PREFIX) for entry in layout.root.iterdir())
