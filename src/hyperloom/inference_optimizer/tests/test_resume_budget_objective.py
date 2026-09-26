# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A bare ``--resume-from`` keeps the session's budget and stop target."""

from __future__ import annotations

import argparse
from typing import Any

from hyperloom.inference_optimizer.cli import _restore_budget_and_objective
from hyperloom.inference_optimizer.cli.parser import DEFAULT_MAX_HOURS


class _State:
    """Just the one field the restore reads."""

    def __init__(self, max_minutes: float) -> None:
        self.max_minutes = max_minutes


def _args(**overrides: Any) -> argparse.Namespace:
    # ``max_hours=None`` is what an omitted ``--max-hours`` parses to; a float
    # here means the operator named one.
    ns = argparse.Namespace(
        max_hours=None,
        target_gain=None,
        target_tput=None,
        target_baseline_dir=None,
        target_roofline=None,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def test_bare_resume_restores_budget_and_gain_target() -> None:
    """The Robustness Monitor's auto-resume passes no flags; it must not shorten the run."""
    args = _args()
    _restore_budget_and_objective(args, _State(1440), {"objective": {"kind": "gain_pct", "value": 300.0}})
    assert args.max_hours == 24.0
    assert args.target_gain == 300.0


def test_an_explicit_flag_on_the_resume_wins() -> None:
    """Re-passing a flag changes it; the archive must not overwrite the operator."""
    args = _args(max_hours=8.0, target_gain=50.0)
    _restore_budget_and_objective(args, _State(1440), {"objective": {"kind": "gain_pct", "value": 300.0}})
    assert args.max_hours == 8.0
    assert args.target_gain == 50.0


def test_an_explicit_budget_equal_to_the_default_still_wins() -> None:
    """A leg asking for the default hour count must tighten, not inherit.

    ``--max-hours`` carries no argparse default precisely so this value is
    distinguishable from absence; ``_start_run`` runs the leg against the
    smaller of the two.
    """
    args = _args(max_hours=DEFAULT_MAX_HOURS)
    _restore_budget_and_objective(args, _State(480), {"objective": {"kind": "time_only", "value": None}})
    assert args.max_hours == DEFAULT_MAX_HOURS


def test_one_explicit_target_replaces_the_persisted_objective_outright() -> None:
    """build_objective refuses two targets, so a new one must not join the old."""
    args = _args(target_tput=900.0)
    _restore_budget_and_objective(args, _State(1440), {"objective": {"kind": "gain_pct", "value": 300.0}})
    assert args.target_tput == 900.0
    assert args.target_gain is None


def test_a_compound_objective_restores_every_target_it_named() -> None:
    """A gain target paired with a roofline target is one legal objective."""
    args = _args()
    recorded = {
        "kind": "gain_pct",
        "value": 300.0,
        "objectives": [{"kind": "gain_pct", "value": 300.0}, {"kind": "roofline_pct", "value": 80.0}],
    }
    _restore_budget_and_objective(args, _State(1440), {"objective": recorded})
    assert args.target_gain == 300.0
    assert args.target_roofline == 80.0


def test_a_time_only_session_restores_the_budget_and_no_target() -> None:
    """Nothing to restore is not the same as restoring nothing."""
    args = _args()
    _restore_budget_and_objective(args, _State(600), {"objective": {"kind": "time_only", "value": None}})
    assert args.max_hours == 10.0
    assert args.target_gain is None


def test_an_unusable_archive_restores_nothing() -> None:
    """A manifest without an objective, and a state without a budget, both degrade quietly.

    ``max_hours`` stays unset so the caller can still supply
    :data:`DEFAULT_MAX_HOURS`; restoring it here would claim the archive named one.
    """
    args = _args()
    assert _restore_budget_and_objective(args, _State(0), {}) == []
    assert args.max_hours is None
    assert args.target_gain is None
