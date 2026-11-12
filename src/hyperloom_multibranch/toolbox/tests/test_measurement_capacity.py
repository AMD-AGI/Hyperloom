# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``measurement_capacity``: the cards a toolbox can still measure on, and the cards the provider lost for good."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from hyperloom.contracts.measurement import TestResult
from hyperloom.measure.tests.hosted_fixtures import spec_context
from hyperloom.sandbox.card_occupancy import CardGuard, Holder
from hyperloom_multibranch.contracts.blackboard import Role, Writer
from hyperloom_multibranch.contracts.blackboard.vocab import QuarantineAction
from hyperloom_multibranch.contracts.toolbox import MeasurementCapacity, WorkloadKind
from hyperloom_multibranch.blackboard import LocalBlackboard
from hyperloom_multibranch.toolbox.tests.fake import FakeToolbox
from hyperloom_multibranch.toolbox.tests.fake_model import Fate, FakeToolboxConfig
from hyperloom.sandbox.gpu_pool import GpuPool
from hyperloom_multibranch.toolbox.local_hosted import (
    FREED_CARDS_REASON,
    LEAKED_CARDS_REASON,
    LocalHostedToolbox,
    LocalOptions,
    LocalSession,
    build_local_toolbox,
)
from hyperloom_multibranch.toolbox.quarantine import Quarantine
from hyperloom_multibranch.toolbox.tests.card_list import CardList
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, build, req
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import PointPlanner
from hyperloom.measure.tests.test_hosted_real_runner import _point
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup

NODE = "node-under-test"


def test_capacity_rejects_more_usable_cards_than_the_pool() -> None:
    with pytest.raises(ValueError, match="usable_cards 5 must be between 0 and total_cards 4"):
        MeasurementCapacity(4, 5)
    assert MeasurementCapacity(4, 4).known is True


def test_leased_cards_are_usable_cards_no_test_runs_on() -> None:
    capacity = MeasurementCapacity(8, 7, leased_cards=2)
    assert (capacity.usable_cards, capacity.measurement_cards) == (7, 5)
    assert MeasurementCapacity(8, 7).leased_cards == 0
    with pytest.raises(ValueError, match="leased_cards 8 must be between 0 and usable_cards 7"):
        MeasurementCapacity(8, 7, leased_cards=8)


def test_unknown_capacity_says_so() -> None:
    unknown = MeasurementCapacity.unknown()
    assert (unknown.known, unknown.total_cards, unknown.usable_cards) == (False, 0, 0)


async def test_a_real_toolbox_without_a_card_pool_reports_unknown(tmp_path: Path) -> None:
    assert await build(tmp_path).toolbox.measurement_capacity() == MeasurementCapacity.unknown()


def _local_toolbox(
    tmp_path: Path, blackboard: LocalBlackboard, cards: range = range(8), options: LocalOptions = LocalOptions()
) -> LocalHostedToolbox:
    session = tmp_path / "session"
    return build_local_toolbox(
        LocalSession(session, "run-1", cards, hostname=NODE),
        blackboard,
        AcceptanceSetup(FakeRepo(), PointPlanner(_point(tmp_path)), IMAGE, WorkloadKind.AGENTX),
        spec_context(tmp_path, blob_root=str(session / "hosted" / "blobs")),
        options=options,
    )


async def _gpu_events(blackboard: LocalBlackboard) -> list[Any]:
    return [event for event in await blackboard.quarantined() if event.gpu_ids]


async def test_without_leaks_every_card_is_usable_and_nothing_is_written(tmp_path: Path) -> None:
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, blackboard)
    try:
        await toolbox.start()
        await toolbox.tick()
        assert await toolbox.measurement_capacity() == MeasurementCapacity(8, 8)
        assert await blackboard.quarantined() == ()
    finally:
        await toolbox.close()


async def test_leaked_cards_shrink_usable_and_are_written_once_per_new_set(tmp_path: Path) -> None:
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, blackboard)
    try:
        await toolbox.start()
        toolbox.provider.quarantined.update({5, 3})
        assert await toolbox.measurement_capacity() == MeasurementCapacity(8, 6)
        await toolbox.tick()
        await toolbox.tick()  # nothing new: no second event
        (event,) = await _gpu_events(blackboard)
        assert (event.node, event.action, event.reason) == (NODE, QuarantineAction.QUARANTINE, LEAKED_CARDS_REASON)
        assert event.gpu_ids == (3, 5)
        toolbox.provider.quarantined.add(1)
        await toolbox.tick()
        events = await _gpu_events(blackboard)
        assert sorted(e.gpu_ids for e in events) == [(1,), (3, 5)]
        assert await toolbox.measurement_capacity() == MeasurementCapacity(8, 5)
    finally:
        await toolbox.close()


async def test_a_leaked_card_does_not_quarantine_the_node(tmp_path: Path) -> None:
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, blackboard)
    try:
        await toolbox.start()
        toolbox.provider.quarantined.add(2)
        await toolbox.tick()
        assert toolbox.quarantined == frozenset()
        reloaded = Quarantine(blackboard)
        await reloaded.load()
        assert reloaded.nodes == frozenset(), "a card event must not read back as a whole-node quarantine"
    finally:
        await toolbox.close()


async def test_a_local_toolbox_is_one_node_and_never_quarantines_it(tmp_path: Path) -> None:
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, blackboard)
    try:
        await toolbox.start()
        assert toolbox.single_node
        await toolbox._node_failed(NODE, "infra failure", reproduced=True)
        await toolbox._node_failed(NODE, "unclear failure", reproduced=False)
        await toolbox._node_failed(NODE, "unclear failure", reproduced=False)
        assert toolbox.quarantined == frozenset() and await blackboard.quarantined() == ()
        assert toolbox.node_failures == {NODE: 3}
    finally:
        await toolbox.close()


async def test_node_quarantine_leaves_usable_cards_unchanged(tmp_path: Path) -> None:
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, blackboard)
    try:
        await toolbox.start()
        assert await toolbox._quarantine.add(NODE, "infra failure") is True
        assert toolbox.quarantined == {NODE}
        assert await toolbox.measurement_capacity() == MeasurementCapacity(8, 8)
    finally:
        await toolbox.close()


async def test_a_refused_write_is_retried_and_the_cards_are_not_marked_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, blackboard)
    try:
        await toolbox.start()
        toolbox.provider.quarantined.add(4)
        real = toolbox._quarantine.add_cards

        async def refuse(*args: Any, **kwargs: Any) -> None:
            raise OSError("blackboard down")

        monkeypatch.setattr(toolbox._quarantine, "add_cards", refuse)
        await toolbox.tick()
        assert await _gpu_events(blackboard) == []
        monkeypatch.setattr(toolbox._quarantine, "add_cards", real)
        await toolbox.tick()
        assert [e.gpu_ids for e in await _gpu_events(blackboard)] == [(4,)]
    finally:
        await toolbox.close()


async def test_a_failure_to_record_leaked_cards_that_is_not_the_blackboards_is_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, blackboard)
    try:
        await toolbox.start()
        toolbox.provider.quarantined.add(4)

        async def broken(*args: Any, **kwargs: Any) -> None:
            raise TypeError("add_cards() takes 3 positional arguments")

        monkeypatch.setattr(toolbox._quarantine, "add_cards", broken)
        with pytest.raises(TypeError, match="positional arguments"):
            await toolbox.tick()
        monkeypatch.undo()
    finally:
        await toolbox.close()


async def test_the_renewal_loop_logs_an_unexpected_failure_to_record_leaked_cards_and_records_them_next_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The renewal loop's round also records leaked cards: an error there must not end the loop that renews claims."""
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, blackboard)
    monkeypatch.setattr(toolbox, "_ttl", 0.03)  # the loop runs a round every third of the claim TTL: every 10 ms
    real, calls = toolbox._quarantine.add_cards, []

    async def broken_once(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        if len(calls) == 1:
            raise TypeError("add_cards() takes 3 positional arguments")
        return await real(*args, **kwargs)

    monkeypatch.setattr(toolbox._quarantine, "add_cards", broken_once)
    try:
        await toolbox.start()
        toolbox.provider.quarantined.add(4)
        for _ in range(500):
            if await _gpu_events(blackboard):
                break
            await asyncio.sleep(0.01)
        assert [e.gpu_ids for e in await _gpu_events(blackboard)] == [(4,)]
    finally:
        await toolbox.close()

    (logged,) = [record for record in caplog.records if record.exc_info and record.exc_info[0] is TypeError]
    assert logged.levelname == "ERROR" and "positional arguments" in str(logged.exc_info[1])  # type: ignore[index]


async def test_cards_leaked_during_a_test_are_reported_when_it_settles(tmp_path: Path) -> None:
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, blackboard)
    try:
        await toolbox.start()

        async def leaking(job: Any) -> TestResult:
            toolbox.provider.quarantined.add(7)
            raise RuntimeError("the attempt ended")

        toolbox._executor.run = leaking  # type: ignore[method-assign]
        await toolbox.submit_test(req())
        await toolbox.drain()
        assert [e.gpu_ids for e in await _gpu_events(blackboard)] == [(7,)]
    finally:
        await toolbox.close()


# --- the fake ------------------------------------------------------------------------------------------------------


async def test_the_fake_fleet_capacity_is_its_cards() -> None:
    box = FakeToolbox(FakeToolboxConfig(nodes=("a", "b", "c"), gpu_count=2))
    assert await box.measurement_capacity() == MeasurementCapacity(6, 6)


async def test_the_fake_loses_the_cards_of_a_quarantined_node_and_every_test_then_fails() -> None:
    box = FakeToolbox(FakeToolboxConfig(nodes=("a", "b"), gpu_count=4))
    box.script(lambda r: True, [Fate.INFRA] * 10)
    request_id = await box.submit_test(req())
    await box.drain()
    assert box.quarantined == {"a", "b"}
    assert await box.measurement_capacity() == MeasurementCapacity(8, 0)
    assert box.results[request_id].validity.value == "measurement_failed"


async def test_cards_an_earlier_run_lost_stay_lost_after_a_restart(tmp_path: Path) -> None:
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
    first = _local_toolbox(tmp_path / "one", blackboard)
    try:
        await first.start()
        first.provider.quarantined.update({0, 6})
        await first.tick()
    finally:
        await first.close()
    second = _local_toolbox(tmp_path / "two", blackboard)
    try:
        await second.start()
        assert await second.measurement_capacity() == MeasurementCapacity(8, 6)
        assert second.provider.pool.free_cards() == (1, 2, 3, 4, 5, 7), "a lost card is never leased again"
        await second.tick()
        assert [e.gpu_ids for e in await _gpu_events(blackboard)] == [(0, 6)], "restoring writes nothing new"
    finally:
        await second.close()


async def test_restoring_lost_cards_works_while_a_process_holds_a_card_that_is_not_lost(tmp_path: Path) -> None:
    """A leak is kept whoever holds the cards: a held card must not stop the toolbox from starting, and stays in the pool."""
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
    first = _local_toolbox(tmp_path / "one", blackboard)
    try:
        await first.start()
        first.provider.quarantined.update({0, 6})
        await first.tick()
    finally:
        await first.close()
    holders = CardList()
    holders.holders = (Holder(card=3, pid=4242, command="python3 intruder.py"), Holder(card=0, pid=4243, command="x"))
    second = _local_toolbox(
        tmp_path / "two", blackboard, options=LocalOptions(card_holders=holders, card_guard=CardGuard.ENFORCE)
    )
    try:
        await second.start()
        assert await second.measurement_capacity() == MeasurementCapacity(8, 6)
        assert second.provider.pool.free_cards() == (1, 2, 3, 4, 5, 7), "the held card is still the pool's"
        assert await asyncio.to_thread(second.provider.pool.acquire, 5, 0.0) == (1, 2, 4, 5, 7), "but is not handed out"
    finally:
        await second.close()


async def test_lifting_a_node_keeps_the_cards_lost_for_good(tmp_path: Path) -> None:
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, blackboard)
    try:
        await toolbox.start()
        toolbox.provider.quarantined.add(2)
        await toolbox.tick()
        await toolbox._quarantine.add(NODE, "infra failure")
        assert await toolbox._quarantine.lift(NODE, "operator") is True
        assert [e.gpu_ids for e in await _gpu_events(blackboard)] == [(2,)]
        assert toolbox.quarantined == frozenset()
    finally:
        await toolbox.close()


async def test_cards_leaked_while_shutting_down_are_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, blackboard)
    await toolbox.start()
    real_close = toolbox.provider.close

    def close_and_leak() -> None:
        toolbox.provider.quarantined.add(1)
        real_close()

    monkeypatch.setattr(toolbox.provider, "close", close_and_leak)
    await toolbox.close()
    assert [e.gpu_ids for e in await _gpu_events(blackboard)] == [(1,)]


def _frees_once(provider: Any, card: int) -> Any:
    """A ``release_freed_cards`` that finds ``card``'s sandbox gone at its first call and nothing after."""
    freed: list[int] = []

    def _release() -> tuple[int, ...]:
        if freed:
            return ()
        freed.append(card)
        provider.quarantined.discard(card)
        provider.pool.release([card])
        return (card,)

    return _release


async def test_freed_cards_come_back_and_only_they_are_lifted_in_the_blackboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
    first = _local_toolbox(tmp_path / "one", blackboard)
    try:
        await first.start()
        provider = first.provider
        provider.quarantined.update(GpuPool.acquire(provider.pool, 2, 0.0))
        await first.tick()
        assert [e.gpu_ids for e in await _gpu_events(blackboard)] == [(0, 1)]
        monkeypatch.setattr(provider, "release_freed_cards", _frees_once(provider, 1))
        await first.tick()
        assert [e.gpu_ids for e in await _gpu_events(blackboard)] == [(0,)], "card 0 is still lost"
        assert await first.measurement_capacity() == MeasurementCapacity(8, 7)
        assert provider.pool.free_cards() == (1, 2, 3, 4, 5, 6, 7)
    finally:
        await first.close()
    second = _local_toolbox(tmp_path / "two", blackboard)
    try:
        await second.start()
        assert second.provider.quarantined == {0}, "a restart takes back only the card still lost"
    finally:
        await second.close()


async def test_a_refused_lift_frees_the_card_and_leaves_it_recorded_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, blackboard)
    try:
        await toolbox.start()
        provider = toolbox.provider
        provider.quarantined.update(GpuPool.acquire(provider.pool, 1, 0.0))
        await toolbox.tick()

        async def refuse(*args: Any, **kwargs: Any) -> None:
            raise OSError("blackboard down")

        monkeypatch.setattr(toolbox._quarantine, "return_cards", refuse)
        monkeypatch.setattr(provider, "release_freed_cards", _frees_once(provider, 0))
        await toolbox.tick()
        assert await toolbox.measurement_capacity() == MeasurementCapacity(8, 8)
        assert [e.gpu_ids for e in await _gpu_events(blackboard)] == [(0,)], "a restart would still take it back"
    finally:
        await toolbox.close()


async def test_a_freed_card_lost_again_is_recorded_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, blackboard)
    try:
        await toolbox.start()
        provider = toolbox.provider
        provider.quarantined.update(GpuPool.acquire(provider.pool, 2, 0.0))
        await toolbox.tick()
        monkeypatch.setattr(provider, "release_freed_cards", _frees_once(provider, 1))
        await toolbox.tick()
        provider.quarantined.add(GpuPool.acquire(provider.pool, 1, 0.0)[0])
        await toolbox.tick()
        assert sorted(e.gpu_ids for e in await _gpu_events(blackboard)) == [(0,), (1,)]
    finally:
        await toolbox.close()


async def test_returning_cards_lifts_each_recorded_set_and_keeps_the_rest_lost(tmp_path: Path) -> None:
    blackboard = LocalBlackboard(tmp_path / "blackboard", Writer(Role.TOOLBOX, "toolbox"))
    quarantine = Quarantine(blackboard)
    await quarantine.add_cards(NODE, (0, 1), LEAKED_CARDS_REASON)
    await quarantine.add_cards(NODE, (4,), LEAKED_CARDS_REASON)
    await quarantine.add_cards("another-node", (1,), LEAKED_CARDS_REASON)
    await quarantine.return_cards(NODE, (1, 4), FREED_CARDS_REASON)
    assert sorted((e.node, e.gpu_ids) for e in await _gpu_events(blackboard)) == [("another-node", (1,)), (NODE, (0,))]
