# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rig the start-up script and the cache builder are tested against, built once for the session."""

from __future__ import annotations

import pytest

from hyperloom.measure.runner import hardware
from hyperloom.measure.runner.tests.bootstrap_rig import Rig, make_rig


@pytest.fixture(scope="session")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Rig:
    """A stand-in for the framework image, a miniature repository and the runner cache built from it."""
    return make_rig(tmp_path_factory.mktemp("rig") / "rig")


class SettleClock:
    """Stands in for ``time`` where the card probe reads a holder again: a sleep moves the clock on at once, and the sleeps
    are kept."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def settle_clock(monkeypatch: pytest.MonkeyPatch) -> SettleClock:
    """Reading a card again to let an exiting process go, or its memory be released, takes no wall time in these tests."""
    fake = SettleClock()
    monkeypatch.setattr(hardware, "time", fake)
    return fake
