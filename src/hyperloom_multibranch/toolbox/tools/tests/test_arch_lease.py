# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The arch microbenchmark of a toolbox analysis runs on a card leased for it, never on a card a test holds."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.sandbox.socket import SandboxUnavailable
from hyperloom_multibranch.contracts.blackboard.vocab import LeaseKind
from hyperloom_multibranch.contracts.toolbox import BranchLease, LeaseCharge, RequestRejected
from hyperloom_multibranch.toolbox.tools import roofline as roofline_module
from hyperloom_multibranch.toolbox.tools.base import ToolRequest
from hyperloom_multibranch.toolbox.tools.roofline import ARCH_GPUS_PARAM, roofline_tool, tracelens_tool

#: The card the measurement test holds, and the one the pool leases out beside it.
_TEST_CARD = 0
_FREE_CARD = 3


class PoolBesideATest:
    """A pool whose card 0 runs a measurement test: a pausing lease gets card 3, and records who held it and when it
    went back. It has no other way to lease, so a tool that asked for an ordinary lease fails."""

    def __init__(self, *, refusal: Exception | None = None) -> None:
        self.refusal = refusal
        self.leases: list[tuple[str, int, LeaseCharge]] = []
        self.released: list[str] = []

    async def lease_pausing(self, holder: str, cards: int, root: Path, *, charge: LeaseCharge) -> BranchLease:
        if self.refusal is not None:
            raise self.refusal
        self.leases.append((holder, cards, charge))
        return BranchLease("lease-1", charge.branch, (_FREE_CARD,), root, {}, 1)

    def release(self, lease_id: str) -> None:
        self.released.append(lease_id)


class RecordingHandler:
    """The analysis: records the payload it got and whether the lease was still held while it ran."""

    def __init__(self, pool: PoolBesideATest) -> None:
        self.pool = pool
        self.payloads: list[dict] = []
        self.held_while_running: list[bool] = []

    async def __call__(self, payload: dict, *, session_dir: Path) -> dict:
        del session_dir
        self.payloads.append(payload)
        self.held_while_running.append(bool(self.pool.leases) and not self.pool.released)
        return {"status": "completed", "hot_kernels": ["gemm"]}


def _without_arch_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(roofline_module, "resolve_arch_json_path", lambda _platform: None)


async def test_the_roofline_measures_the_board_on_a_leased_card_not_the_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _without_arch_spec(monkeypatch)
    pool = PoolBesideATest()
    tool = roofline_tool(pool)
    handler = RecordingHandler(pool)
    tool._handler = handler
    workdir = tmp_path / "job-7"
    workdir.mkdir()

    await tool.run(ToolRequest("roofline", "b1", {"trace_input": "/t", "target_platform": "MI355X"}), workdir)

    assert handler.payloads[0][ARCH_GPUS_PARAM] == [_FREE_CARD]
    assert _TEST_CARD not in handler.payloads[0][ARCH_GPUS_PARAM]
    assert pool.leases == [("tool:job-7", 1, LeaseCharge(LeaseKind.TOOL_JOB, "b1", "roofline"))]
    assert handler.held_while_running == [True]
    assert pool.released == ["lease-1"]


async def test_a_board_with_an_arch_spec_takes_no_card(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(roofline_module, "resolve_arch_json_path", lambda _platform: tmp_path / "MI355X.json")
    pool = PoolBesideATest()
    tool = roofline_tool(pool)
    handler = RecordingHandler(pool)
    tool._handler = handler

    await tool.run(ToolRequest("roofline", "b1", {"trace_input": "/t"}), tmp_path)

    assert pool.leases == []
    assert ARCH_GPUS_PARAM not in handler.payloads[0]


@pytest.mark.parametrize(
    "pool",
    [
        None,
        PoolBesideATest(refusal=SandboxUnavailable("the toolbox is closing")),
        PoolBesideATest(refusal=RequestRejected("the blackboard did not record the grant")),
    ],
    ids=["no pool", "toolbox closing", "lease refused"],
)
async def test_without_a_leased_card_the_analysis_is_given_none(
    pool: PoolBesideATest | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _without_arch_spec(monkeypatch)
    tool = roofline_tool(pool)
    handler = RecordingHandler(pool or PoolBesideATest())
    tool._handler = handler

    await tool.run(ToolRequest("roofline", "b1", {"trace_input": "/t"}), tmp_path)

    assert handler.payloads[0][ARCH_GPUS_PARAM] == []


async def test_the_caller_cannot_name_the_cards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _without_arch_spec(monkeypatch)
    pool = PoolBesideATest()
    tool = roofline_tool(pool)
    handler = RecordingHandler(pool)
    tool._handler = handler

    await tool.run(ToolRequest("roofline", "b1", {"trace_input": "/t", ARCH_GPUS_PARAM: [_TEST_CARD]}), tmp_path)

    assert handler.payloads[0][ARCH_GPUS_PARAM] == [_FREE_CARD]


async def test_the_tracelens_report_route_takes_no_card(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _without_arch_spec(monkeypatch)
    pool = PoolBesideATest()
    tool = tracelens_tool(pool)
    handler = RecordingHandler(pool)
    tool._handler = handler

    await tool.run(ToolRequest("tracelens", "b1", {"trace_input": "/t", "analysis_route": "report"}), tmp_path)

    assert pool.leases == []
    assert ARCH_GPUS_PARAM not in handler.payloads[0]


async def test_the_tracelens_orchestrator_route_measures_on_a_leased_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _without_arch_spec(monkeypatch)
    pool = PoolBesideATest()
    tool = tracelens_tool(pool)
    handler = RecordingHandler(pool)
    tool._handler = handler

    await tool.run(ToolRequest("tracelens", "b1", {"trace_input": "/t", "analysis_route": "non-llm"}), tmp_path)

    assert handler.payloads[0][ARCH_GPUS_PARAM] == [_FREE_CARD]
    assert pool.released == ["lease-1"]
