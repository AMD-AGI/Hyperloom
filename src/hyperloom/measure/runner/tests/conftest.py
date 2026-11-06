# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rig the start-up script and the cache builder are tested against, built once for the session."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.tests.test_aiter_serving_once import (
    CARD_KERNEL,
    MODULE,
    Prewarm,
    module_of,
    serving_image,
)
from hyperloom.measure.runner import card_watch, hardware
from hyperloom.measure.runner import run as run_module
from hyperloom.measure.runner.tests.bootstrap_rig import Rig, make_rig
from hyperloom.orchestrator.actions.executors._aiter_serving_once import ServingPreflight
from hyperloom.sandbox import card_occupancy
from hyperloom.sandbox.card_occupancy import CardReading


@pytest.fixture(scope="session")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Rig:
    """A stand-in for the framework image, a miniature repository and the runner cache built from it."""
    return make_rig(tmp_path_factory.mktemp("rig") / "rig")


@pytest.fixture(autouse=True)
def _nobody_holds_the_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner tests do not read the kernel's list of the machine they run on: a test that needs one says what it holds."""
    monkeypatch.setattr(card_occupancy, "read_holders", lambda cards: CardReading())


@pytest.fixture(autouse=True)
def _no_serving_preflight_of_this_machine(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner tests never run the launch preflight on an aiter of the machine they run on, which it may move modules
    of aside and build into: a test that asks for :func:`serving_aiter` runs it on that stand-in."""
    if "serving_aiter" in request.fixturenames:
        return
    monkeypatch.setattr(
        run_module,
        "build_image_own_once",
        lambda build: ServingPreflight({"action": "noop", "reason": "the runner tests run none"}),
    )


@pytest.fixture
def serving_aiter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Prewarm]:
    """The image's aiter the runner's preflight finds (``test_aiter_serving_once``), whose ``module_gemm_a8w8`` covers
    the image's own table for the card, and the prewarm's build in place of the CPU one, which notes what it builds."""
    package = serving_image(tmp_path / "serving", monkeypatch)
    (package / "jit" / f"{MODULE}.so").write_bytes(module_of(CARD_KERNEL))
    build = Prewarm()
    monkeypatch.setattr(run_module, "preflight_build", build)
    return package, build


class SettleClock:
    """Stands in for ``time`` where the card probe and the card watch read a holder again: a sleep moves the clock on at
    once, and the sleeps are kept."""

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
    monkeypatch.setattr(card_watch, "time", fake)
    monkeypatch.setattr(hardware, "time", fake)
    monkeypatch.setattr(card_occupancy, "time", fake)
    return fake
