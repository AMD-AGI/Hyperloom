# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A test whose cards an enforcing pool will not hand out waits for them; it is never failed as the candidate's.

The pool of the provider is the one the toolbox builds (``GatedPool``) with a probe the test sets: a process holds the
only card. The test waits for the runner as long as a runner may take to start, and runs once the holder lets go; when it
never does, the test ends as the infrastructure failure a test with no runner is, which is retried, not as the candidate's.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from hyperloom.measure.returncodes import HOSTED_RUNNER_UNAVAILABLE_RC
from hyperloom.measure.tests.hosted_fixtures import Rig, make_call, make_rig, point_yaml
from hyperloom.sandbox.card_occupancy import CardGuard, CardReading, Holder
from hyperloom.sandbox.gated_pool import GatedPool

_HOLDER = Holder(card=0, pid=4242, command="python3 -c import torch", vram=2 << 30)


class _Probe:
    """Who holds the cards: the test sets it; ``asked`` counts the looks."""

    def __init__(self) -> None:
        self.holders: tuple[Holder, ...] = (_HOLDER,)
        self.asked = 0

    def __call__(self, cards: Sequence[int]) -> CardReading:
        self.asked += 1
        return CardReading(holders=tuple(h for h in self.holders if h.card in cards))


@pytest.fixture
def held(tmp_path: Path) -> Iterator[tuple[Rig, _Probe]]:
    def build(start_timeout_sec: float) -> tuple[Rig, _Probe]:
        built = make_rig(tmp_path, gpus=1, runner_start_timeout_sec=start_timeout_sec)
        probe = _Probe()
        built.provider.pool = GatedPool(range(1), CardGuard.ENFORCE, read=probe, poll_sec=0.02)
        made.append(built)
        return built, probe

    made: list[Rig] = []
    yield build  # type: ignore[misc]
    for built in made:
        built.backend.close()
        built.provider.close()


def test_a_test_whose_only_card_is_held_runs_once_the_holder_lets_go(held, tmp_path) -> None:
    rig, probe = held(5.0)
    timer = threading.Timer(0.5, lambda: setattr(probe, "holders", ()))
    timer.start()
    try:
        outcome = rig.backend.run_test(make_call(tmp_path, point=point_yaml(tmp_path)))
    finally:
        timer.cancel()

    assert (outcome.returncode, outcome.failure_class, outcome.failure_reason) == (0, None, None)
    assert probe.asked > 2, "the pool looked again while the test waited"


def test_a_test_whose_card_stays_held_is_an_infrastructure_failure_not_the_candidates(held, tmp_path) -> None:
    rig, _ = held(0.3)

    outcome = rig.backend.run_test(make_call(tmp_path, point=point_yaml(tmp_path)))

    assert outcome.returncode == HOSTED_RUNNER_UNAVAILABLE_RC
    assert outcome.failure_class is not None
    assert (outcome.failure_class.value, outcome.failure_reason) == ("infra", "runner_start_timeout")
    assert "no 1 free GPUs that no process holds: held card 0 by pid 4242" in outcome.stderr, "refused for the holder"
    assert rig.provider.pool.free_cards() == (0,), "the held card stays in the pool"
