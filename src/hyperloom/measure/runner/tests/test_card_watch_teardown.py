# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The test's own server stopping, replayed through the runner's watch in a pod: its memory stays listed after its
descriptors close, and that is not a foreign holder however long it stays; a holder that was never the test's still is.

:data:`TEARDOWNS` are the 88 results of two long sessions on an 8-card pod whose signed statement said ``foreign``: every
one was the test's own server's memory on its own cards, first seen this many seconds before the result was signed. A
row is the test's cards, the cards the statement named, the GiB on each and that lead. What the sessions did not record
is how long the kernel kept listing the memory, so each row is replayed with the memory freed after a short and after a
long time.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import CARD_CLEAN, CARD_FOREIGN
from hyperloom.measure.runner.card_watch import AFTER_TEST, DURING_TEST, CardStatement, CardWatch
from hyperloom.measure.runner.tests.conftest import SettleClock
from hyperloom.sandbox import card_occupancy
from hyperloom.sandbox.card_occupancy import CardGuard, OwnProcesses
from hyperloom.sandbox.card_occupancy import read_holders as read_the_machine
from hyperloom.sandbox.tests.kernel_tree import KernelTree

_GIB = 1 << 30
_HOST_PID = 1_800_000
_POD_PID = 6_000
_FOREIGN_PID = 1_900_000
_CENSUS_ONCE = card_occupancy._census_once
#: How long the kernel kept the memory listed: under the census's settle time, as a GPU pod measured for a server that
#: exited, and longer than the rest of the test.
SHORT_LINGER_SEC = 3.0
LONG_LINGER_SEC = 60.0

TEARDOWNS: tuple[tuple[tuple[int, ...], tuple[int, ...], float, int], ...] = (
    ((0, 1), (1,), 236.5, 34),
    ((2, 3), (2,), 238.4, 35),
    ((2, 3), (2, 3), 11.8, 33),
    ((2, 3), (2, 3), 236.4, 17),
    ((2, 3), (2, 3), 238.3, 34),
    ((2, 3), (2, 3), 238.3, 34),
    ((2, 3), (2, 3), 238.4, 34),
    ((2, 3), (2, 3), 238.6, 34),
    ((2, 3), (2, 3), 238.7, 34),
    ((2, 3), (2, 3), 238.7, 35),
    ((2, 3), (2, 3), 239.0, 35),
    ((2, 3), (2, 3), 239.3, 33),
    ((2, 3), (2, 3), 239.3, 34),
    ((2, 3), (2, 3), 239.3, 34),
    ((2, 3), (2, 3), 239.3, 34),
    ((2, 3), (2, 3), 257.1, 34),
    ((2, 3), (3,), 237.8, 34),
    ((2, 3), (3,), 238.3, 34),
    ((2, 5), (2, 5), 238.0, 33),
    ((2, 5), (2, 5), 238.0, 34),
    ((2, 5), (2, 5), 238.0, 34),
    ((2, 5), (2, 5), 238.0, 36),
    ((2, 5), (5,), 238.3, 34),
    ((2, 7), (2,), 236.9, 36),
    ((2, 7), (2,), 237.0, 20),
    ((2, 7), (2, 7), 232.2, 37),
    ((2, 7), (2, 7), 232.2, 37),
    ((2, 7), (2, 7), 236.9, 34),
    ((2, 7), (2, 7), 236.9, 34),
    ((2, 7), (2, 7), 237.1, 34),
    ((2, 7), (2, 7), 237.3, 34),
    ((3, 4), (3,), 238.4, 33),
    ((3, 4), (3, 4), 236.5, 34),
    ((3, 4), (3, 4), 236.9, 33),
    ((3, 4), (3, 4), 237.1, 35),
    ((3, 4), (3, 4), 237.3, 34),
    ((3, 4), (3, 4), 238.3, 34),
    ((3, 4), (3, 4), 238.3, 35),
    ((3, 4), (3, 4), 238.4, 35),
    ((3, 4), (3, 4), 238.9, 36),
    ((3, 6), (3, 6), 232.2, 35),
    ((3, 6), (3, 6), 232.2, 38),
    ((3, 6), (3, 6), 234.9, 37),
    ((3, 6), (3, 6), 237.9, 34),
    ((3, 6), (3, 6), 238.0, 34),
    ((4, 5), (4,), 239.3, 35),
    ((4, 5), (4, 5), 232.2, 37),
    ((4, 5), (4, 5), 232.8, 37),
    ((4, 5), (4, 5), 236.5, 34),
    ((4, 5), (4, 5), 236.5, 34),
    ((4, 5), (4, 5), 236.5, 36),
    ((4, 5), (4, 5), 236.9, 34),
    ((4, 5), (4, 5), 237.9, 32),
    ((4, 5), (4, 5), 237.9, 35),
    ((4, 5), (4, 5), 237.9, 35),
    ((4, 5), (4, 5), 238.2, 35),
    ((4, 5), (4, 5), 238.3, 32),
    ((4, 5), (4, 5), 238.6, 34),
    ((4, 5), (4, 5), 238.9, 34),
    ((4, 5), (4, 5), 238.9, 35),
    ((4, 5), (4, 5), 239.0, 35),
    ((4, 5), (4, 5), 239.3, 34),
    ((4, 5), (4, 5), 239.3, 34),
    ((4, 5), (4, 5), 239.3, 36),
    ((4, 5), (4, 5), 256.9, 34),
    ((5, 6), (5, 6), 236.5, 32),
    ((5, 6), (5, 6), 236.5, 34),
    ((5, 6), (5, 6), 236.8, 35),
    ((5, 6), (5, 6), 237.1, 34),
    ((5, 6), (5, 6), 237.1, 35),
    ((5, 6), (5, 6), 237.1, 35),
    ((5, 6), (5, 6), 237.3, 35),
    ((5, 6), (5, 6), 238.3, 34),
    ((5, 6), (5, 6), 238.4, 35),
    ((5, 6), (5, 6), 238.7, 9),
    ((6, 7), (6,), 238.3, 34),
    ((6, 7), (6, 7), 236.8, 33),
    ((6, 7), (6, 7), 238.1, 21),
    ((6, 7), (6, 7), 238.1, 35),
    ((6, 7), (6, 7), 238.3, 34),
    ((6, 7), (6, 7), 238.3, 35),
    ((6, 7), (6, 7), 238.4, 35),
    ((6, 7), (6, 7), 238.7, 33),
    ((6, 7), (6, 7), 238.8, 35),
    ((6, 7), (6, 7), 238.9, 33),
    ((6, 7), (6, 7), 238.9, 35),
    ((6, 7), (6, 7), 238.9, 35),
    ((6, 7), (6, 7), 239.4, 34),
)


@dataclass
class _Machine:
    """The pod's kernel list and ``/proc``; memory of an ended process is dropped from the list at ``freed_at``."""

    tree: KernelTree
    clock: SettleClock
    lingering: list[int] = field(default_factory=list)
    freed_at: float = float("inf")

    def serve(self, cards: tuple[int, ...], gib: float) -> None:
        for card in cards:
            vram = int(gib * _GIB)
            self.tree.holds(_HOST_PID + card, card, vram=vram, queues=6)
            self.tree.process(_POD_PID + card)
            self.tree.renders(_POD_PID + card, card, kib=vram >> 10)

    def stop(self, cards: tuple[int, ...], lingering: tuple[int, ...], linger_sec: float) -> None:
        """The server's processes end: gone from ``/proc``, their queues gone, their memory still listed on
        ``lingering`` cards until the driver frees it."""
        for card in cards:
            shutil.rmtree(self.tree.procfs / str(_POD_PID + card))  # with its descriptors
            entry = self.tree.kfd / "proc" / str(_HOST_PID + card)
            if card in lingering:
                shutil.rmtree(entry / "queues")
                (entry / "queues").mkdir()
                self.lingering.append(_HOST_PID + card)
            else:
                shutil.rmtree(entry)
        self.freed_at = self.clock.now + linger_sec

    def free_if_due(self) -> None:
        if self.clock.now >= self.freed_at:
            for pid in self.lingering:
                shutil.rmtree(self.tree.kfd / "proc" / str(pid), ignore_errors=True)

    def read(self, cards):
        self.free_if_due()
        return read_the_machine(cards, kfd=self.tree.kfd, procfs=self.tree.procfs)


def _machine(root: Path, clock: SettleClock, monkeypatch: pytest.MonkeyPatch) -> _Machine:
    root.mkdir()
    machine = _Machine(KernelTree(root, pid_namespace="pid:[4026552279]"), clock)
    monkeypatch.setattr(card_occupancy, "read_holders", machine.read)

    def census_once(kfd, procfs, selected):  # the settle's re-reads see the memory freed too
        machine.free_if_due()
        return _CENSUS_ONCE(kfd, procfs, selected)

    monkeypatch.setattr(card_occupancy, "_census_once", census_once)
    return machine


def _replay(machine: _Machine, row, linger_sec: float, *, intruder_gib: float = 0.0) -> CardStatement:
    """A test whose server runs, then stops ``lead`` seconds before the result is signed, looked at every second; with
    ``intruder_gib``, a process outside the pod takes memory on the first card just after the server stopped."""
    cards, lingering, gib, lead = row
    clock = machine.clock
    watch = CardWatch(
        cards, OwnProcesses(frozenset({100}), procfs=machine.tree.procfs), CardGuard.ENFORCE, clock=lambda: clock.now
    )
    machine.serve(cards, gib)
    for _ in range(3):
        watch.check(DURING_TEST)
        clock.now += 1.0
    machine.stop(cards, lingering, linger_sec)
    if intruder_gib:
        machine.tree.holds(_FOREIGN_PID, cards[0], vram=int(intruder_gib * _GIB), queues=2)
    end = clock.now + lead
    while clock.now < end:
        watch.check(DURING_TEST)
        clock.now += 1.0
    watch.check(AFTER_TEST)
    statement = watch.statement()
    assert statement is not None
    return statement


@pytest.mark.parametrize("linger_sec", [SHORT_LINGER_SEC, LONG_LINGER_SEC], ids=["freed-in-3s", "freed-after-60s"])
def test_the_88_teardowns_two_sessions_called_foreign_are_clean_however_long_the_memory_stays_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settle_clock: SettleClock, linger_sec: float
) -> None:
    states = []
    for number, row in enumerate(TEARDOWNS):
        statement = _replay(_machine(tmp_path / str(number), settle_clock, monkeypatch), row, linger_sec)
        states.append((statement.state, statement.reason))

    foreign = [reason for state, reason in states if state == CARD_FOREIGN]
    assert not foreign, f"{len(foreign)} of {len(TEARDOWNS)} replayed teardowns read foreign; first: {foreign[0]}"
    assert [state for state, _ in states] == [CARD_CLEAN] * len(TEARDOWNS)


def test_a_process_outside_the_pod_that_takes_memory_on_the_card_as_the_server_stops_is_still_foreign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settle_clock: SettleClock
) -> None:
    statement = _replay(
        _machine(tmp_path / "m", settle_clock, monkeypatch), TEARDOWNS[0], LONG_LINGER_SEC, intruder_gib=2.6
    )

    assert statement.state == CARD_FOREIGN
    assert (
        statement.reason is not None
        and "on card 0 that no process this container sees accounts for" in statement.reason
    )


def test_a_second_process_with_memory_beside_the_running_server_is_still_foreign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settle_clock: SettleClock
) -> None:
    machine = _machine(tmp_path / "m", settle_clock, monkeypatch)
    watch = CardWatch((2, 3), OwnProcesses(frozenset({100}), procfs=machine.tree.procfs), CardGuard.ENFORCE)
    machine.serve((2, 3), 238.0)
    watch.check(DURING_TEST)
    machine.tree.holds(_FOREIGN_PID, 3, vram=3 * _GIB, queues=2)
    watch.check(DURING_TEST)

    statement = watch.statement()
    assert statement is not None and statement.state == CARD_FOREIGN
    assert statement.reason is not None and "2 processes with memory on card 3 at once" in statement.reason


def test_memory_left_by_a_process_the_watch_never_saw_accounted_for_is_foreign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settle_clock: SettleClock
) -> None:
    """The remembered holders are only those a look found accounted for: memory of an ended process the watch never saw
    as the test's (here a host process that held the card alone) still counts after the settle time."""
    machine = _machine(tmp_path / "m", settle_clock, monkeypatch)
    machine.tree.holds(_FOREIGN_PID, 4, vram=200 * _GIB, queues=0)
    watch = CardWatch((4,), OwnProcesses(frozenset({100}), procfs=machine.tree.procfs), CardGuard.ENFORCE)

    watch.check(DURING_TEST)

    statement = watch.statement()
    assert statement is not None and statement.state == CARD_FOREIGN
