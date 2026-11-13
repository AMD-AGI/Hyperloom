# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A launch reads nothing an earlier launch left: interleaved graph replays of three widths, launches
issued back to back, and a run of launches across the epoch's wrap, all on TP4."""

import pytest
from ranks import BACK_TO_BACK, REPLAY_ROUNDS, REPLAY_WIDTHS, TP, WRAP_LAUNCHES, launch_ranks


@pytest.fixture(scope="module")
def verdicts() -> list[dict]:
    runs = launch_ranks("replay", timeout=1800.0)
    for run in runs:
        assert run.returncode == 0 and run.finished, run.tail()
    return [v for run in runs for v in run.verdicts()]


def _check(verdicts: list[dict], check: str, per_rank: int) -> None:
    found = [v for v in verdicts if v["check"] == check]
    assert len(found) == TP * per_rank, f"{check}: {len(found)} verdicts, expected {TP * per_rank}"
    failed = [v for v in found if not v["passed"]]
    assert not failed, failed[:8]


def test_interleaved_graph_replays_of_three_widths(verdicts):
    _check(verdicts, "graph", REPLAY_ROUNDS * len(REPLAY_WIDTHS))


def test_back_to_back_launches_each_move_the_epoch_on(verdicts):
    _check(verdicts, "back_to_back", BACK_TO_BACK)
    _check(verdicts, "epoch", 1)


def test_launches_across_the_epoch_wrap(verdicts):
    _check(verdicts, "wrap", WRAP_LAUNCHES)
    _check(verdicts, "wrap_epoch", 1)
