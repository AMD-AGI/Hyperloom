# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The rig the start-up script and the cache builder are tested against, built once for the session."""

from __future__ import annotations

import pytest

from hyperloom.measure.runner.tests.bootstrap_rig import Rig, make_rig
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
