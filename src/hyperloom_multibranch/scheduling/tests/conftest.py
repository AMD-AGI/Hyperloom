# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every test here runs the same on a GPU host and on a CPU host, busy or idle.

The local peers build the real local toolbox, whose card pool asks the kernel's list who holds a card before it lends
one: a test that read the host's own list would find no free card wherever other work is on the cards. Every test
here reads an idle machine of its own and no GPU tool of the host (:mod:`hyperloom.measure.tests.gpu_free_host`).
The peers describe the card their tests run on, so that machine's cards are described as MI355Xs, and a hosted
session's runners by the recording of an MI325X node the GPU card tests read (:func:`described_cards`).
"""

from pathlib import Path

import pytest

from hyperloom.measure.tests.gpu_free_host import (  # noqa: F401  (autouse fixtures)
    idle_card_list,
    machine_without_readable_gpus,
)
from hyperloom.orchestrator.tests._fixtures import MI355X_CARD
from hyperloom_common import gpu_card
from hyperloom_common.gpu_card import SESSION_CARD_ENV
from hyperloom_multibranch.scheduling.hosted_peers import RUNNER_CARD_ENV

#: ``amd-smi static --json`` of a node of four MI325Xs.
RECORDED_MI325X = Path(gpu_card.__file__).parent / "tests" / "fixtures" / "amd_smi_static_mi325x.json"
#: The cards ``amd-smi`` describes on the test's machine.
MACHINE_CARDS = [MI355X_CARD] * 8


@pytest.fixture(autouse=True)
def described_cards(monkeypatch: pytest.MonkeyPatch) -> None:
    """``amd-smi`` describes :data:`MACHINE_CARDS`, the app host is given :data:`RECORDED_MI325X` for its runners, and
    the card a session publishes is taken back after the test."""
    monkeypatch.setattr(gpu_card, "read_cards", lambda: list(MACHINE_CARDS))
    monkeypatch.setenv(RUNNER_CARD_ENV, str(RECORDED_MI325X))
    monkeypatch.setenv(SESSION_CARD_ENV, "")
