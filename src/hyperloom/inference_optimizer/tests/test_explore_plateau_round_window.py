# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""EXPLORE plateau: the gain window, and what counts as an unproductive round.

The plateau is the only exit EXPLORE has that is not a clock. These cover the
two ways it could never be reached, so a regression that re-disables it shows up
here rather than as a phase that always spends its whole budget.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.phases.machine_state import (
    DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK,
    compute_plateau_explore,
)

GATE = "HYPERLOOM_EXPLORE_PLATEAU_ROUND_WINDOW"


def _state(winners, rounds):
    return SimpleNamespace(
        explore_search={"winners_history": list(winners)},
        specialist_rounds=list(rounds),
    )


def _barren_rounds(n, *, start=0, proposals=12):
    """Rounds that proposed a full grid and kept none of it."""
    return [
        {"round_id": f"r{i}", "proposals_total": proposals, "proposals_kept": 0}
        for i in range(start, start + n)
    ]


def _productive_round(idx, *, kept=1, proposals=12):
    return {"round_id": f"r{idx}", "proposals_total": proposals, "proposals_kept": kept}


# --- the gain window ---------------------------------------------------------


def test_single_keep_no_longer_disables_the_gain_arm(monkeypatch):
    """One early win must not hold the plateau off for the rest of the cycle.

    A winner is recorded only because it cleared the KEEP threshold, so a window
    over the last N winners always sums above the plateau floor. Scoped to
    recent rounds instead, the sum falls away once those rounds stop keeping.
    """
    monkeypatch.setenv(GATE, "1")
    winners = [{"round_id": "r0", "gain_pct": 1.5}]
    rounds = [_productive_round(0)] + _barren_rounds(
        DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK, start=1
    )
    triggered, ev = compute_plateau_explore(_state(winners, rounds))
    assert triggered is True
    assert ev["gain_window"] == "recent_rounds"
    assert ev["recent_keep_gain_pct"] == 0.0


def test_legacy_window_is_unsatisfiable_after_one_keep(monkeypatch):
    """The behaviour the gate restores, pinned so the contrast is explicit."""
    monkeypatch.setenv(GATE, "0")
    winners = [{"round_id": "r0", "gain_pct": 1.5}]
    rounds = [_productive_round(0)] + _barren_rounds(
        DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK, start=1
    )
    triggered, ev = compute_plateau_explore(_state(winners, rounds))
    assert triggered is False
    assert ev["gain_window"] == "recent_winners"
    assert ev["recent_keep_gain_pct"] == pytest.approx(1.5)


def test_a_win_inside_the_window_still_holds_the_plateau_off(monkeypatch):
    """The arm must still say 'no' while recent rounds are producing gain."""
    monkeypatch.setenv(GATE, "1")
    rounds = _barren_rounds(4) + [_productive_round(4)]
    winners = [{"round_id": "r4", "gain_pct": 2.0}]
    triggered, ev = compute_plateau_explore(_state(winners, rounds))
    assert triggered is False
    assert ev["recent_keep_gain_pct"] == pytest.approx(2.0)


def test_gain_sums_only_winners_from_the_recent_rounds(monkeypatch):
    monkeypatch.setenv(GATE, "1")
    rounds = [_productive_round(i) for i in range(8)]
    winners = [{"round_id": f"r{i}", "gain_pct": 1.0} for i in range(8)]
    _, ev = compute_plateau_explore(_state(winners, rounds), lookback=3)
    assert ev["recent_keep_gain_pct"] == pytest.approx(3.0)
    assert ev["winners_seen"] == 3


# --- what counts as unproductive --------------------------------------------


def test_rounds_that_keep_nothing_count_toward_the_streak(monkeypatch):
    """Proposing a grid and keeping none of it is not a productive round."""
    monkeypatch.setenv(GATE, "1")
    rounds = [_productive_round(0)] + _barren_rounds(6, start=1)
    _, ev = compute_plateau_explore(_state([], rounds))
    assert ev["empty_streak"] == 6
    assert ev["empty_streak_basis"] == "kept_nothing"


def test_legacy_streak_ignores_rounds_that_merely_kept_nothing(monkeypatch):
    monkeypatch.setenv(GATE, "0")
    _, ev = compute_plateau_explore(_state([], _barren_rounds(6)))
    assert ev["empty_streak"] == 0
    assert ev["empty_streak_basis"] == "no_proposals"


def test_a_kept_round_breaks_the_streak(monkeypatch):
    monkeypatch.setenv(GATE, "1")
    rounds = _barren_rounds(4) + [_productive_round(4)] + _barren_rounds(2, start=5)
    _, ev = compute_plateau_explore(_state([], rounds))
    assert ev["empty_streak"] == 2


def test_rounds_with_no_proposals_still_count(monkeypatch):
    """The original signal is a subset of the new one, not a casualty of it."""
    monkeypatch.setenv(GATE, "1")
    rounds = [
        {"round_id": f"r{i}", "proposals_total": 0, "proposals_kept": 0} for i in range(5)
    ]
    _, ev = compute_plateau_explore(_state([], rounds))
    assert ev["empty_streak"] == 5


def test_malformed_round_is_filtered_before_the_streak_walk(monkeypatch):
    """Non-dict rows are dropped by the cycle filter, so they neither count nor break.

    Worth pinning: the streak predicates guard against non-dicts, which reads as
    though a malformed row would break the streak. It cannot reach them.
    """
    monkeypatch.setenv(GATE, "1")
    rounds = _barren_rounds(3) + ["not-a-dict"] + _barren_rounds(2, start=4)
    _, ev = compute_plateau_explore(_state([], rounds))
    assert ev["specialist_rounds_seen"] == 5
    assert ev["empty_streak"] == 5


def test_legacy_kept_count_fallback_key(monkeypatch):
    """Older round summaries used ``kept_count``."""
    monkeypatch.setenv(GATE, "1")
    rounds = [{"round_id": f"r{i}", "proposal_count": 4, "kept_count": 0} for i in range(5)]
    _, ev = compute_plateau_explore(_state([], rounds))
    assert ev["empty_streak"] == 5


# --- degrading safely --------------------------------------------------------


def test_winners_without_round_attribution_fall_back(monkeypatch):
    """Missing attribution must not read as an empty window and exit the phase."""
    monkeypatch.setenv(GATE, "1")
    winners = [{"gain_pct": 1.5}, {"gain_pct": 2.0}]
    rounds = [_productive_round(0)] + _barren_rounds(6, start=1)
    triggered, ev = compute_plateau_explore(_state(winners, rounds))
    assert ev["gain_window"] == "recent_winners"
    assert triggered is False


def test_rounds_without_ids_fall_back(monkeypatch):
    monkeypatch.setenv(GATE, "1")
    winners = [{"round_id": "r0", "gain_pct": 1.5}]
    rounds = [{"proposals_total": 12, "proposals_kept": 0} for _ in range(6)]
    _, ev = compute_plateau_explore(_state(winners, rounds))
    assert ev["gain_window"] == "recent_winners"


def test_no_winners_at_all_uses_the_round_window(monkeypatch):
    monkeypatch.setenv(GATE, "1")
    triggered, ev = compute_plateau_explore(_state([], _barren_rounds(6)))
    assert ev["gain_window"] == "recent_rounds"
    assert triggered is True


def test_empty_state_is_inert(monkeypatch):
    monkeypatch.setenv(GATE, "1")
    triggered, ev = compute_plateau_explore(SimpleNamespace())
    assert triggered is False
    assert ev["empty_streak"] == 0
    assert ev["recent_keep_gain_pct"] == 0.0


def test_lookback_disabled_short_circuits(monkeypatch):
    monkeypatch.setenv(GATE, "1")
    triggered, ev = compute_plateau_explore(_state([], _barren_rounds(9)), lookback=0)
    assert triggered is False
    assert ev["reason"] == "lookback_disabled"


def test_streak_below_threshold_does_not_trigger(monkeypatch):
    """Both arms are required; a short streak is not a plateau."""
    monkeypatch.setenv(GATE, "1")
    rounds = [_productive_round(0)] + _barren_rounds(2, start=1)
    triggered, ev = compute_plateau_explore(_state([], rounds))
    assert ev["recent_keep_gain_pct"] == 0.0
    assert triggered is False


def test_unset_gate_defaults_to_the_round_window(monkeypatch):
    monkeypatch.delenv(GATE, raising=False)
    _, ev = compute_plateau_explore(_state([], _barren_rounds(6)))
    assert ev["gain_window"] == "recent_rounds"
    assert ev["empty_streak_basis"] == "kept_nothing"


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "OFF"])
def test_gate_accepts_falsey_spellings(monkeypatch, raw):
    monkeypatch.setenv(GATE, raw)
    _, ev = compute_plateau_explore(_state([], _barren_rounds(6)))
    assert ev["empty_streak_basis"] == "no_proposals"


def test_unparseable_gain_is_skipped_not_fatal(monkeypatch):
    monkeypatch.setenv(GATE, "1")
    rounds = [_productive_round(0)]
    winners = [{"round_id": "r0", "gain_pct": "nonsense"}]
    _, ev = compute_plateau_explore(_state(winners, rounds))
    assert ev["recent_keep_gain_pct"] == 0.0
