# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``measurement_capacity``: the cards a toolbox can still measure on, and the cards the provider lost for good."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.contracts.measurement import TestResult
from hyperloom.measure.tests.hosted_fixtures import spec_context
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.vocab import QuarantineAction
from hyperloom_multibranch.contracts.toolbox import MeasurementCapacity, WorkloadKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox.fake import FakeToolbox
from hyperloom_multibranch.toolbox.fake_model import Fate, FakeToolboxConfig
from hyperloom_multibranch.toolbox.local_hosted import (
    LEAKED_CARDS_REASON,
    LocalHostedToolbox,
    LocalSession,
    build_local_toolbox,
)
from hyperloom_multibranch.toolbox.quarantine import Quarantine
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, build, req
from hyperloom_multibranch.toolbox.tests.test_hosted_executor import PointPlanner
from hyperloom.measure.tests.test_hosted_real_runner import _point
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup

NODE = "node-under-test"


def test_capacity_rejects_more_usable_cards_than_the_pool() -> None:
    with pytest.raises(ValueError, match="usable_cards 5 must be between 0 and total_cards 4"):
        MeasurementCapacity(4, 5)
    assert MeasurementCapacity(4, 4).known is True


def test_unknown_capacity_says_so() -> None:
    unknown = MeasurementCapacity.unknown()
    assert (unknown.known, unknown.total_cards, unknown.usable_cards) == (False, 0, 0)


async def test_a_real_toolbox_without_a_card_pool_reports_unknown(tmp_path: Path) -> None:
    assert await build(tmp_path).toolbox.measurement_capacity() == MeasurementCapacity.unknown()


def _local_toolbox(tmp_path: Path, notebook: LocalNotebook, cards: range = range(8)) -> LocalHostedToolbox:
    session = tmp_path / "session"
    return build_local_toolbox(
        LocalSession(session, "run-1", cards, hostname=NODE),
        notebook,
        AcceptanceSetup(FakeRepo(), PointPlanner(_point(tmp_path)), IMAGE, WorkloadKind.AGENTX),
        spec_context(tmp_path, benchmark_backend="bypass", blob_root=str(session / "hosted" / "blobs")),
    )


async def _gpu_events(notebook: LocalNotebook) -> list[Any]:
    return [event for event in await notebook.quarantined() if event.gpu_ids]


async def test_without_leaks_every_card_is_usable_and_nothing_is_written(tmp_path: Path) -> None:
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, notebook)
    try:
        await toolbox.start()
        await toolbox.tick()
        assert await toolbox.measurement_capacity() == MeasurementCapacity(8, 8)
        assert await notebook.quarantined() == ()
    finally:
        await toolbox.close()


async def test_leaked_cards_shrink_usable_and_are_written_once_per_new_set(tmp_path: Path) -> None:
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, notebook)
    try:
        await toolbox.start()
        toolbox.provider.quarantined.update({5, 3})
        assert await toolbox.measurement_capacity() == MeasurementCapacity(8, 6)
        await toolbox.tick()
        await toolbox.tick()  # nothing new: no second event
        (event,) = await _gpu_events(notebook)
        assert (event.node, event.action, event.reason) == (NODE, QuarantineAction.QUARANTINE, LEAKED_CARDS_REASON)
        assert event.gpu_ids == (3, 5)
        toolbox.provider.quarantined.add(1)
        await toolbox.tick()
        events = await _gpu_events(notebook)
        assert sorted(e.gpu_ids for e in events) == [(1,), (3, 5)]
        assert await toolbox.measurement_capacity() == MeasurementCapacity(8, 5)
    finally:
        await toolbox.close()


async def test_a_leaked_card_does_not_quarantine_the_node(tmp_path: Path) -> None:
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, notebook)
    try:
        await toolbox.start()
        toolbox.provider.quarantined.add(2)
        await toolbox.tick()
        assert toolbox.quarantined == frozenset()
        reloaded = Quarantine(notebook)
        await reloaded.load()
        assert reloaded.nodes == frozenset(), "a card event must not read back as a whole-node quarantine"
    finally:
        await toolbox.close()


async def test_node_quarantine_leaves_usable_cards_unchanged(tmp_path: Path) -> None:
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, notebook)
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
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, notebook)
    try:
        await toolbox.start()
        toolbox.provider.quarantined.add(4)
        real = toolbox._quarantine.add_cards

        async def refuse(*args: Any, **kwargs: Any) -> None:
            raise OSError("notebook down")

        monkeypatch.setattr(toolbox._quarantine, "add_cards", refuse)
        await toolbox.tick()
        assert await _gpu_events(notebook) == []
        monkeypatch.setattr(toolbox._quarantine, "add_cards", real)
        await toolbox.tick()
        assert [e.gpu_ids for e in await _gpu_events(notebook)] == [(4,)]
    finally:
        await toolbox.close()


async def test_cards_leaked_during_a_test_are_reported_when_it_settles(tmp_path: Path) -> None:
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, notebook)
    try:
        await toolbox.start()

        async def leaking(job: Any) -> TestResult:
            toolbox.provider.quarantined.add(7)
            raise RuntimeError("the attempt ended")

        toolbox._executor.run = leaking  # type: ignore[method-assign]
        await toolbox.submit_test(req())
        await toolbox.drain()
        assert [e.gpu_ids for e in await _gpu_events(notebook)] == [(7,)]
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
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
    first = _local_toolbox(tmp_path / "one", notebook)
    try:
        await first.start()
        first.provider.quarantined.update({0, 6})
        await first.tick()
    finally:
        await first.close()
    second = _local_toolbox(tmp_path / "two", notebook)
    try:
        await second.start()
        assert await second.measurement_capacity() == MeasurementCapacity(8, 6)
        assert second.provider.pool.free_cards() == (1, 2, 3, 4, 5, 7), "a lost card is never leased again"
        await second.tick()
        assert [e.gpu_ids for e in await _gpu_events(notebook)] == [(0, 6)], "restoring writes nothing new"
    finally:
        await second.close()


async def test_lifting_a_node_keeps_the_cards_lost_for_good(tmp_path: Path) -> None:
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, notebook)
    try:
        await toolbox.start()
        toolbox.provider.quarantined.add(2)
        await toolbox.tick()
        await toolbox._quarantine.add(NODE, "infra failure")
        assert await toolbox._quarantine.lift(NODE, "operator") is True
        assert [e.gpu_ids for e in await _gpu_events(notebook)] == [(2,)]
        assert toolbox.quarantined == frozenset()
    finally:
        await toolbox.close()


async def test_cards_leaked_while_shutting_down_are_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    toolbox = _local_toolbox(tmp_path, notebook)
    await toolbox.start()
    real_close = toolbox.provider.close

    def close_and_leak() -> None:
        toolbox.provider.quarantined.add(1)
        real_close()

    monkeypatch.setattr(toolbox.provider, "close", close_and_leak)
    await toolbox.close()
    assert [e.gpu_ids for e in await _gpu_events(notebook)] == [(1,)]
