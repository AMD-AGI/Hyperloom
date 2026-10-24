# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A process that is not the test's on a card: the test is retried on another card, a lease is not given a held card, and
the signed result says what was checked of the cards (board #126: an agent's commands ran on cards a test measured on).

The pool is the real gated pool over a local provider that owns two cards; the kernel's list is a stand-in that the test
writes, and the fake runner fails as the real runner does when it finds a process on its card.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from hyperloom.contracts.measurement import FailureClass, TestResult, Validity
from hyperloom.measure.tests.hosted_fixtures import Rig, make_rig, point_yaml
from hyperloom.sandbox.card_occupancy import CardGuard, Holder
from hyperloom.sandbox.gated_pool import CardsHeld, GatedPool
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox import local_hosted
from hyperloom_multibranch.toolbox.local_hosted import LocalOptions, LocalSession
from hyperloom_multibranch.toolbox.hosted_executor import HostedExecutor
from hyperloom_multibranch.toolbox.tests.card_list import INTRUDER, CardList
from hyperloom_multibranch.toolbox.tests.helpers import Rig as ToolboxRig
from hyperloom_multibranch.toolbox.tests.helpers import build, req
from hyperloom_multibranch.toolbox.tests.test_branch_leases import local  # noqa: F401 - fixture
from hyperloom_multibranch.toolbox.tests.test_card_placement import FOREIGN, POLICY, _PointPerBranch, _settled, _until

Make = Callable[..., Awaitable[tuple[Rig, ToolboxRig, "CardList"]]]
Local = Callable[..., Awaitable[Any]]


@pytest_asyncio.fixture
async def make(tmp_path: Path) -> AsyncIterator[Make]:
    made: list[tuple[Rig, ToolboxRig]] = []

    async def build_rig(**points: dict[str, object]) -> tuple[Rig, ToolboxRig, CardList]:
        rig = make_rig(tmp_path, gpus=2, max_parallel_tests=2, runner_start_timeout_sec=2)
        holders = CardList()
        rig.provider.pool = GatedPool(range(2), CardGuard.ENFORCE, read=holders, poll_sec=0.01)
        files = {branch: point_yaml(tmp_path, f"{branch}.yaml", **fake) for branch, fake in points.items()}
        toolbox = build(
            tmp_path,
            executor=HostedExecutor(rig.backend, tmp_path / "session"),  # type: ignore[arg-type]
            planner=_PointPerBranch(files),
            policy=POLICY,
            workload_kind=WorkloadKind.AGENTX,
        )
        made.append((rig, toolbox))
        await toolbox.toolbox.start()
        return rig, toolbox, holders

    yield build_rig
    for rig, toolbox in made:
        await toolbox.toolbox.close()
        await asyncio.to_thread(rig.backend.close)
        await asyncio.to_thread(rig.provider.close)


def _settled_result(box: Any, request_id: str) -> TestResult | None:
    entry = box._log.get(request_id)
    if entry is None or entry.outcome is None or "result" not in entry.outcome:
        return None
    return TestResult.from_document(entry.outcome["result"])


def _result(tb: ToolboxRig, request_id: str) -> TestResult:
    entry = tb.toolbox._log.get(request_id)
    assert entry is not None and entry.outcome is not None and "result" in entry.outcome, entry and entry.outcome
    return TestResult.from_document(entry.outcome["result"])


async def test_a_test_refused_for_a_process_on_its_card_is_retried_on_the_other_card_and_that_result_counts(
    local: Local,
) -> None:
    holders = CardList()
    holders.arrives_after_first_hand_out = (
        True  # card 0 is clean when the first sandbox gets it, then the intruder comes
    )
    box = await local(a=FOREIGN, card_holders=holders, card_guard=CardGuard.ENFORCE)

    request_id = await box.submit_test(req("a"))
    await _until(lambda: _settled_result(box, request_id) is not None, "the test to settle", seconds=60)

    result = _settled_result(box, request_id)
    assert result is not None and result.validity is Validity.SUCCEEDED and result.failure_class is None
    assert result.sandbox is not None and result.sandbox.gpu_ids == (1,), "the retry measured on the other card"
    first, second = result.attempts
    assert first.failure_class is FailureClass.INFRA
    assert first.failure_reason is not None and first.failure_reason.startswith("foreign_process_on_card")
    assert second.failure_class is None
    assert box.provider.pool.free_cards() == (0, 1), "both cards went back to the pool"


async def test_a_test_whose_every_card_has_an_intruder_is_never_a_measurement(make: Make) -> None:
    """Both cards held for good: the pool never hands one out, so no sandbox starts and nothing is measured."""
    rig, tb, holders = await make(a={"fake_delay": 0.1})
    holders.holders = (INTRUDER, Holder(card=1, pid=4243, command="python3 other.py"))

    request_id = await tb.toolbox.submit_test(req("a"))
    await asyncio.wait_for(_settled(tb, request_id), 120)

    result = _result(tb, request_id)
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
    assert result.rounds == () and rig.provider.find({}, role="measurement") == []


async def test_the_signed_result_says_the_cards_were_clean_or_that_nobody_could_say(make: Make) -> None:
    rig, tb, _ = await make(
        clean={"fake_card_occupancy": json.dumps({"state": "clean", "reason": None, "checks": 6})},
        blind={
            "fake_card_occupancy": json.dumps(
                {"state": "unknown", "reason": "no AMD GPU driver is loaded", "checks": 0}
            )
        },
        silent={},
    )
    ids = {name: await tb.toolbox.submit_test(req(name)) for name in ("clean", "blind", "silent")}
    for request_id in ids.values():
        await asyncio.wait_for(_settled(tb, request_id), 60)

    facts = {name: _result(tb, request_id).sandbox for name, request_id in ids.items()}

    assert facts["clean"] is not None and (facts["clean"].card_occupancy, facts["clean"].card_occupancy_reason) == (
        "clean",
        "",
    )
    assert facts["blind"] is not None
    assert (facts["blind"].card_occupancy, facts["blind"].card_occupancy_reason) == (
        "unknown",
        "no AMD GPU driver is loaded",
    )
    assert facts["silent"] is not None and facts["silent"].card_occupancy == "", (
        "a runner that said nothing says nothing"
    )


async def test_a_result_kept_beside_a_process_says_so_in_its_facts_and_carries_a_warning(make: Make) -> None:
    """Only a runner whose guard observes signs this state: one that enforces refuses the result instead."""
    reason = "1 held the test's cards: pid 900 on card 0, first seen 2026-10-24T04:05:06Z during the test: python3 x.py"
    rig, tb, _ = await make(
        kept={"fake_card_occupancy": json.dumps({"state": "foreign", "reason": reason, "checks": 5})},
        clean={"fake_card_occupancy": json.dumps({"state": "clean", "reason": None, "checks": 5})},
    )
    ids = {name: await tb.toolbox.submit_test(req(name)) for name in ("kept", "clean")}
    for request_id in ids.values():
        await asyncio.wait_for(_settled(tb, request_id), 60)

    kept, clean = (_result(tb, ids[name]) for name in ("kept", "clean"))

    assert kept.validity is Validity.SUCCEEDED and kept.sandbox is not None
    assert (kept.sandbox.card_occupancy, kept.sandbox.card_occupancy_reason) == ("foreign", reason)
    assert kept.nonfatal_warnings == (f"foreign_process_on_card, kept because the card guard only observes: {reason}",)
    assert clean.nonfatal_warnings == ()


# --- the guard reaches the runners and the pool of the local toolbox -----------------------------------------------------


def _runner_envs(tmp_path: Path, options: LocalOptions) -> dict[str, str]:
    session = LocalSession(tmp_path, "run-guard", range(2))
    provider = local_hosted._provider(session, (0, 1), local_hosted._runner_template(session, options), options, None)
    try:
        return dict(provider._templates["runner-1"].envs)
    finally:
        provider.close()


@pytest.mark.parametrize(
    ("environ", "option", "expected"),
    [
        ({}, None, "observe"),
        ({"HYPERLOOM_CARD_GUARD": "enforce"}, None, "enforce"),
        ({"HYPERLOOM_CARD_GUARD": "observe"}, None, "observe"),
        ({"HYPERLOOM_CARD_GUARD": "enforce"}, CardGuard.OBSERVE, "observe"),
        ({}, CardGuard.ENFORCE, "enforce"),
    ],
)
def test_the_runners_are_given_the_guard_the_option_or_the_host_environment_names_and_observe_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, environ: dict[str, str], option: CardGuard | None, expected: str
) -> None:
    monkeypatch.delenv("HYPERLOOM_CARD_GUARD", raising=False)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)

    assert _runner_envs(tmp_path, LocalOptions(card_guard=option))["HYPERLOOM_CARD_GUARD"] == expected


def test_a_host_environment_that_names_no_guard_is_refused_when_the_toolbox_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HYPERLOOM_CARD_GUARD", "enfroce")

    with pytest.raises(ValueError, match="HYPERLOOM_CARD_GUARD is 'enfroce', not one of"):
        local_hosted.card_guard_of(LocalOptions())


@pytest.mark.parametrize(("guard", "refused"), [(CardGuard.ENFORCE, True), (CardGuard.OBSERVE, False)])
def test_the_pool_of_the_local_provider_follows_the_guard(tmp_path: Path, guard: CardGuard, refused: bool) -> None:
    holders = CardList()
    holders.holders = (INTRUDER,)
    options = LocalOptions(card_guard=guard, card_holders=holders)
    session = LocalSession(tmp_path, "run-pool", range(1))
    provider = local_hosted._provider(session, (0,), local_hosted._runner_template(session, options), options, None)
    try:
        if refused:
            with pytest.raises(CardsHeld, match="card 0 by pid 4242"):
                provider.pool.acquire(1, 0.0)
        else:
            assert provider.pool.acquire(1, 0.0) == (0,)
    finally:
        provider.close()
