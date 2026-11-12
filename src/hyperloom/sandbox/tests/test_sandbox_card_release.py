# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An ended sandbox's cards go back to the pool once the kernel lists nothing on them, and a provider's close reaps its
sandboxes all at once.

The ranks of a torn-down server stay listed with their memory for tens of seconds after the reap, so a card handed out at
once would be found held by the next test's card guard. The wait is bounded: a holder that never goes is handed out with
the card, and the guard refuses it as it always did.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence

import pytest

from hyperloom.sandbox import card_occupancy, local, local_sandbox, reaper
from hyperloom.sandbox.card_occupancy import METHOD_CENSUS, CardGuard, CardReading, Holder
from hyperloom.sandbox.gated_pool import CardsHeld, GatedPool
from hyperloom.sandbox.local import LocalSandboxProvider, LocalTemplate
from hyperloom.sandbox.local_sandbox import LocalSandbox
from hyperloom.sandbox.socket import CreateOptions, SandboxUnavailable
from hyperloom.sandbox.tests.kernel_tree import KernelTree

_SERVER_RANK = Holder(0, 4242, "", int(106.5 * (1 << 30)), 0)


class _Listed:
    """What the kernel lists on card 0: a torn-down server's rank from :meth:`hold` until ``seconds`` later, else
    nothing."""

    def __init__(self) -> None:
        self.free_at = 0.0

    def hold(self, seconds: float) -> None:
        self.free_at = time.monotonic() + seconds

    def __call__(self, cards: Sequence[int]) -> CardReading:
        if 0 in cards and time.monotonic() < self.free_at:
            return CardReading(holders=(_SERVER_RANK,), method=METHOD_CENSUS)
        return CardReading(method=METHOD_CENSUS)


@pytest.fixture
def make():
    providers: list[LocalSandboxProvider] = []

    def _make(card_memory: Callable[[Sequence[int]], CardReading]) -> LocalSandboxProvider:
        provider = LocalSandboxProvider([LocalTemplate("dev", "dev", gpus=1)], hostname="node-a", gpus=[0])
        provider.card_memory = card_memory
        providers.append(provider)
        return provider

    yield _make
    for provider in providers:
        provider.close()


@pytest.fixture(autouse=True)
def quick_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(local, "_RELEASE_STEP_SEC", 0.05)


def _create(provider: LocalSandboxProvider, wait_sec: float = 30.0) -> LocalSandbox:
    sandbox = provider.create("dev", role="dev", timeout_sec=120, metadata={}, options=CreateOptions(wait_sec=wait_sec))
    assert isinstance(sandbox, LocalSandbox)
    return sandbox


def _wait(predicate, seconds: float = 20.0) -> None:
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.05)


def test_the_next_test_on_a_card_a_torn_down_server_still_holds_waits_for_it_and_runs(make) -> None:
    listed = _Listed()
    provider = make(listed)
    first = _create(provider)
    listed.hold(1.0)
    first.kill()
    assert provider.pool.free_cards() == (), "card 0 is listed held: it is not handed out yet"

    second = _create(provider)

    assert time.monotonic() >= listed.free_at, "the card was handed out only once nothing was listed on it"
    assert second._base_env["ROCR_VISIBLE_DEVICES"] == "0"


def test_a_holder_that_never_goes_is_handed_out_after_the_wait_and_the_guard_still_refuses_it(
    make, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(local, "CARD_RELEASE_WAIT_SEC", 0.5)
    listed = _Listed()
    provider = make(listed)
    provider.pool = GatedPool([0], CardGuard.ENFORCE, read=listed, poll_sec=0.05)
    first = _create(provider)
    listed.hold(math.inf)
    ended = time.monotonic()
    with caplog.at_level(logging.WARNING, logger=local.__name__):
        first.kill()
        assert provider.pool.free_cards() == ()
        _wait(lambda: provider.pool.free_cards() == (0,))
    assert time.monotonic() - ended >= local.CARD_RELEASE_WAIT_SEC, "the wait is bounded, and not shorter"
    assert "still hold memory" in caplog.text and "kernel id 4242 with 106.5 GiB" in caplog.text

    with pytest.raises(CardsHeld, match="card 0 by pid 4242"):
        _create(provider, wait_sec=0.3)


def test_cards_go_back_at_once_where_nothing_is_listed_or_the_list_cannot_be_read(make) -> None:
    for read in (_Listed(), lambda cards: CardReading(unavailable="no AMD GPU driver is loaded")):
        provider = make(read)
        _create(provider).kill()
        assert provider.pool.free_cards() == (0,)


def test_a_provider_whose_caller_sets_no_reading_never_reads_the_hosts_cards(tmp_path, monkeypatch) -> None:
    """The kernel's list says card 0 is held, as on a GPU host where another job runs: a provider left as built does not
    read it, and its card goes back at once."""
    tree = KernelTree(tmp_path)
    tree.holds(4242, 0, vram=100 << 30)
    monkeypatch.setattr(card_occupancy, "KFD_PATH", tree.kfd)
    assert card_occupancy.read_listed((0,)).held_cards() == {0}, "the host's list as the real read sees it"
    with LocalSandboxProvider([LocalTemplate("dev", "dev", gpus=1)], hostname="node-a", gpus=[0]) as provider:
        _create(provider).kill()
        assert provider.pool.free_cards() == (0,)


def test_withheld_cards_freed_later_wait_for_the_card_too(make, monkeypatch) -> None:
    listed = _Listed()
    provider = make(listed)
    sandbox = _create(provider)
    exited = subprocess.Popen([sys.executable, "-c", "pass"])
    exited.wait()
    monkeypatch.setattr(local_sandbox, "reap_sandbox_processes", lambda *a, **k: reaper.Leftovers(((exited.pid, 1),)))
    with pytest.raises(SandboxUnavailable, match="its GPUs stay leased"):
        sandbox.kill()
    listed.hold(1.0)

    assert provider.release_freed_cards() == (0,), "nothing of the sandbox runs"
    assert provider.quarantined == set() and provider.pool.free_cards() == ()
    _wait(lambda: provider.pool.free_cards() == (0,))
    assert time.monotonic() >= listed.free_at


def test_a_close_ends_the_wait_for_cards(make, monkeypatch) -> None:
    listed = _Listed()
    provider = make(listed)
    first = _create(provider)
    listed.hold(math.inf)
    first.kill()
    name = f"hl-card-release-{first.sandbox_id}"
    assert any(thread.name == name for thread in threading.enumerate())
    provider.close()
    _wait(lambda: all(thread.name != name for thread in threading.enumerate()), seconds=5.0)
    assert provider.pool.free_cards() == ()


def test_a_close_reaps_its_sandboxes_all_at_once(monkeypatch) -> None:
    provider = LocalSandboxProvider([LocalTemplate("dev", "dev")], hostname="node-a")
    sandboxes = [_create(provider) for _ in range(3)]
    # Each reap ends only once every other one has begun: one after another, the first would wait for the rest in vain.
    together = threading.Barrier(len(sandboxes), timeout=10.0)
    kill = LocalSandbox.kill
    met: list[str] = []

    def _kill(self: LocalSandbox) -> None:
        together.wait()
        met.append(self.sandbox_id)
        kill(self)

    monkeypatch.setattr(LocalSandbox, "kill", _kill)
    provider.close()
    assert sorted(met) == sorted(s.sandbox_id for s in sandboxes)
    assert all(s.ended for s in sandboxes)
    assert not any(os.path.exists(s.directory) for s in sandboxes)
