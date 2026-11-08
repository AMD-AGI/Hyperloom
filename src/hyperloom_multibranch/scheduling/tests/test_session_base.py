# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session freezes the base launch its tests run on once, resolved as ``hyperloom optimize`` resolves it, and a
resumed session keeps the one it froze; an operator's base launch record is the base, frozen without reading any
recipe source, and the Arbor agents are served the frozen base."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest

from hyperloom.inference_optimizer.cli import base_launch as cli_base_launch
from hyperloom.measure.tests.bundle_factory import base_launch
from hyperloom.inference_optimizer.session.base_launch import write_base_launch
from hyperloom.orchestrator.state.base_launch import FrozenBase, freeze_base, frozen_base
from hyperloom_multibranch.scheduling.peers import PeersUnavailable
from hyperloom_multibranch.scheduling.session_base import freeze_session_base, served_launch
from hyperloom_multibranch.scheduling.settings import InferenceXShape, KernelOptimizer, Workload
from hyperloom_multibranch.specialist.arbor_adapter import read_launch
from hyperloom_multibranch.tests.base_records import MODEL_ID, base_record, write_record
from inference_suite import SourceSnapshot
from inference_suite.contracts import Workload as LaunchWorkload
from inference_suite.recipe import sources
from inference_suite.recipe.resolve import UnreadSource

WORKLOAD = Workload(
    "/models/Llama-3.1-8B-Instruct", "sglang", 2, 16, "fp8", "mi355x", KernelOptimizer.FORGE, InferenceXShape(1024, 512)
)
AGENTX = replace(WORKLOAD, inferencex=None, ep=2)
#: The AgentX workload the record of :func:`base_record` serves.
RECORDED = Workload(MODEL_ID, "vllm", 2, 8, "fp8", "mi355x", KernelOptimizer.FORGE, None)


@pytest.fixture
def asked(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, object]]:
    """Every ``(query, user_file)`` the sources were read for; they answer offline, so only the default table and the
    operator's file serve."""

    calls: list[tuple[object, object]] = []

    def fetch(query: object, *, user_file: object = None) -> SourceSnapshot:
        calls.append((query, user_file))
        return SourceSnapshot(None, 1_793_000_000, user_file=user_file, unread=UnreadSource("InferenceX", "offline"))

    monkeypatch.setattr(cli_base_launch, "fetch_sources", fetch)
    return calls


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """InferenceX cannot be read, as on a host without egress."""

    def unread(*_: object) -> NoReturn:
        raise AssertionError("a recipe source was read")

    monkeypatch.setattr(sources, "_inferencex", unread)


def test_a_fresh_session_freezes_the_base_resolved_for_its_workload_in_the_toolbox_directory(
    tmp_path: Path, asked: list
) -> None:
    freeze_session_base(tmp_path, WORKLOAD, "")

    [(query, user_file)] = asked
    assert user_file is None
    assert (query.model_id, query.framework, query.board, query.workload) == (
        "Llama-3.1-8B-Instruct",
        "sglang",
        "mi355x",
        LaunchWorkload.FIXED,
    )
    assert (query.tp, query.ep, query.isl, query.osl, query.concurrencies) == (2, 1, 1024, 512, (8, 16, 32))
    frozen = frozen_base(tmp_path)
    assert frozen is not None and frozen.launch.query == query
    assert frozen.launch.model.path == WORKLOAD.model and frozen.launch.topology.tp == WORKLOAD.tp


def test_a_hub_id_the_model_base_resolves_is_asked_for_by_that_id_and_served_from_the_directory(
    tmp_path: Path, asked: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "models" / "Llama-3.1-8B-Instruct").mkdir(parents=True)
    monkeypatch.setenv("HL_MODEL_BASE", str(tmp_path / "models"))

    freeze_session_base(tmp_path / "toolbox", replace(WORKLOAD, model="meta-llama/Llama-3.1-8B-Instruct"), "")

    assert asked[0][0].model_id == "meta-llama/Llama-3.1-8B-Instruct"
    frozen = frozen_base(tmp_path / "toolbox")
    assert frozen is not None and frozen.launch.model.path == str(tmp_path / "models" / "Llama-3.1-8B-Instruct")


def test_a_resumed_session_keeps_the_base_it_froze_and_reads_no_source(tmp_path: Path, asked: list) -> None:
    kept = freeze_base(tmp_path, base_launch())

    freeze_session_base(tmp_path, replace(WORKLOAD, tp=8), "")

    assert asked == [] and frozen_base(tmp_path) == kept


def test_a_session_whose_frozen_base_was_changed_after_it_froze_is_refused(tmp_path: Path, asked: list) -> None:
    frozen = freeze_base(tmp_path / "replaced", base_launch())
    write_base_launch(tmp_path / "replaced", dataclasses.replace(base_launch(), args=("--edited",)))
    frozen = freeze_base(tmp_path / "edited", base_launch())
    frozen.path.write_text(json.dumps({**base_launch().to_document(), "args": ["--edited"]}))

    with pytest.raises(PeersUnavailable, match="frozen base launch cannot be read: .*changed after it was frozen"):
        freeze_session_base(tmp_path / "replaced", WORKLOAD, "")
    with pytest.raises(PeersUnavailable, match="frozen base launch cannot be read: base_digest_mismatch"):
        freeze_session_base(tmp_path / "edited", WORKLOAD, "")
    assert asked == []


def test_an_agentx_session_no_source_serves_is_refused_and_freezes_nothing(tmp_path: Path, asked: list) -> None:
    with pytest.raises(PeersUnavailable, match="no source serves this session's base launch"):
        freeze_session_base(tmp_path, AGENTX, "")

    [(query, _)] = asked
    assert (query.workload, query.tp, query.ep, query.isl, query.concurrencies) == (
        LaunchWorkload.AGENTX,
        2,
        2,
        None,
        (16,),
    )
    assert frozen_base(tmp_path) is None


def test_an_agent_launch_that_is_a_base_launch_record_is_the_base_and_an_arbor_launch_file_is_not(
    tmp_path: Path, asked: list
) -> None:
    record = tmp_path / "record.json"
    record.write_text(json.dumps(base_launch().to_document()))
    arbor = tmp_path / "launch.json"
    arbor.write_text(json.dumps({"server_flags": "--tp 1", "server_env": {}}))
    workload = replace(WORKLOAD, tp=1)

    freeze_session_base(tmp_path / "with-record", workload, str(record))
    freeze_session_base(tmp_path / "with-arbor-file", workload, str(arbor))

    [(_, user_file), (_, none)] = asked
    assert user_file.location == str(record) and none is None
    frozen = frozen_base(tmp_path / "with-record")
    assert frozen is not None
    assert dataclasses.replace(frozen.launch, query=base_launch().query) == dataclasses.replace(
        base_launch(), model=dataclasses.replace(base_launch().model, path=workload.model)
    )


def test_a_base_on_another_topology_than_the_workloads_is_refused(tmp_path: Path, asked: list) -> None:
    record = tmp_path / "record.json"
    record.write_text(json.dumps(base_launch().to_document()))

    with pytest.raises(PeersUnavailable, match="the query asks for tp 2, the launch runs 1"):
        freeze_session_base(tmp_path / "toolbox", WORKLOAD, str(record))
    assert frozen_base(tmp_path / "toolbox") is None


def test_a_record_is_frozen_as_the_base_without_a_source_and_arbor_is_served_the_frozen_base(
    tmp_path: Path, offline: None
) -> None:
    record = write_record(tmp_path / "base_launch.json")

    frozen = freeze_session_base(tmp_path / "toolbox", RECORDED, str(record))

    assert (frozen.launch.args, frozen.launch.golden_al) == (base_record().args, base_record().golden_al)
    assert served_launch(frozen, str(record)) == str(frozen.path)
    assert read_launch(frozen.path).digest == frozen.launch.digest()


def test_a_resumed_session_keeps_its_base_and_serves_it_for_a_record(tmp_path: Path, offline: None) -> None:
    toolbox = tmp_path / "toolbox"
    first = freeze_session_base(toolbox, RECORDED, str(write_record(tmp_path / "first.json")))
    other = write_record(tmp_path / "other.json", base_record(env={"VLLM_ROCM_USE_AITER": "0"}))

    resumed = freeze_session_base(toolbox, RECORDED, str(other))

    assert resumed.launch == first.launch
    assert served_launch(resumed, str(other)) == str(first.path)


def test_an_arbor_launch_file_is_served_as_it_is(tmp_path: Path) -> None:
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"server_flags": "--max-num-seqs 64", "server_env": {}}), encoding="utf-8")
    frozen = FrozenBase(tmp_path / "toolbox" / "base_launch.json", base_record())

    assert served_launch(frozen, str(launch)) == str(launch)
    assert served_launch(frozen, "") == ""
