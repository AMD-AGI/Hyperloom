# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session's base launch: an operator's base launch record is the base, frozen without reading any recipe source,
and the Arbor agents are served the frozen base."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest

from hyperloom.orchestrator.state.base_launch import FrozenBase
from hyperloom_multibranch.scheduling.peers import PeersUnavailable
from hyperloom_multibranch.scheduling.session_base import freeze_session_base, served_launch
from hyperloom_multibranch.scheduling.settings import KernelOptimizer, Workload
from hyperloom_multibranch.specialist.arbor_adapter import read_launch
from hyperloom_multibranch.tests.base_records import MODEL_ID, base_record, write_record
from inference_suite.recipe import sources

#: The AgentX workload the record of :func:`base_record` serves.
WORKLOAD = Workload(MODEL_ID, "vllm", 2, 8, "fp8", "mi355x", KernelOptimizer.FORGE, None)


@pytest.fixture(autouse=True)
def _no_source(monkeypatch: pytest.MonkeyPatch) -> None:
    def unread(*_: object) -> NoReturn:
        raise AssertionError("a recipe source was read")

    monkeypatch.setattr(sources, "_inferencex", unread)


def test_a_record_is_frozen_as_the_base_without_a_source_and_arbor_is_served_the_frozen_base(tmp_path: Path) -> None:
    record = write_record(tmp_path / "base_launch.json")

    frozen = freeze_session_base(tmp_path / "toolbox", WORKLOAD, str(record))

    assert (frozen.launch.args, frozen.launch.golden_al) == (base_record().args, base_record().golden_al)
    assert served_launch(frozen, str(record)) == str(frozen.path)
    assert read_launch(frozen.path).digest == frozen.launch.digest()


def test_a_resumed_session_keeps_its_base_and_serves_it_for_a_record(tmp_path: Path) -> None:
    toolbox = tmp_path / "toolbox"
    first = freeze_session_base(toolbox, WORKLOAD, str(write_record(tmp_path / "first.json")))
    other = write_record(tmp_path / "other.json", base_record(env={"VLLM_ROCM_USE_AITER": "0"}))

    resumed = freeze_session_base(toolbox, WORKLOAD, str(other))

    assert resumed.launch == first.launch
    assert served_launch(resumed, str(other)) == str(first.path)


def test_an_arbor_launch_file_is_served_as_it_is(tmp_path: Path) -> None:
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"server_flags": "--max-num-seqs 64", "server_env": {}}), encoding="utf-8")
    frozen = FrozenBase(tmp_path / "toolbox" / "base_launch.json", base_record())

    assert served_launch(frozen, str(launch)) == str(launch)
    assert served_launch(frozen, "") == ""


def test_a_record_on_another_topology_is_refused(tmp_path: Path) -> None:
    record = write_record(tmp_path / "base_launch.json")

    with pytest.raises(PeersUnavailable, match="the query asks for tp 4"):
        freeze_session_base(tmp_path / "toolbox", replace(WORKLOAD, tp=4), str(record))
