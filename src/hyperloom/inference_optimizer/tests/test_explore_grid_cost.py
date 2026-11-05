# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An explore is charged for every variant its grid carries, not for one round."""

from __future__ import annotations

from hyperloom.orchestrator.loop.time_budget import explore_admission_minutes


def _grid(n: int) -> dict:
    return {"grid": [{"name": f"v{i}", "extra_args": f"--flag-{i}"} for i in range(n)]}


def test_a_four_variant_grid_costs_four_rounds():
    """Measured: a four-variant explore ran 39.9 minutes against the ~12 it was admitted at.

    Every variant is its own server launch and benchmark on the one serving
    lane, so a queue priced per task reads as a third of what it will take.
    """
    assert explore_admission_minutes(10.0, _grid(4)) == 40.0


def test_a_single_variant_grid_is_one_round():
    assert explore_admission_minutes(10.0, _grid(1)) == 10.0


def test_a_grid_the_executor_will_seed_is_priced_as_one_round():
    """No grid supplied means the executor builds one; one round is the only honest floor here."""
    assert explore_admission_minutes(10.0, {}) == 10.0
    assert explore_admission_minutes(10.0, None) == 10.0
    assert explore_admission_minutes(10.0, {"grid": []}) == 10.0


def test_the_phase_that_overran_would_now_refuse_its_queue():
    """23 rounds queued in 52 minutes, none drained 95 minutes later, all cancelled at the exit.

    Eleven explores of four variants charge 440 minutes against a 274-minute
    phase, where per-task pricing charged 132 and called it affordable.
    """
    per_task = sum(explore_admission_minutes(10.0, _grid(4)) for _ in range(11))

    assert per_task == 440.0
    assert per_task > 274.0
