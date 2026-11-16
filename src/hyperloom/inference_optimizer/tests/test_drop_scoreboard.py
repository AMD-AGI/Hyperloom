# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Drop scoreboard tests."""

from __future__ import annotations

import json

from hyperloom.orchestrator.state.shared_state import SharedState


def _legacy_state_payload() -> dict:
    """A legacy state.json snapshot with action_scores + legacy fields."""
    return {
        "session_id": "legacy-sid",
        "baseline_tput": 1234.0,
        "cumulative_gain": 2.5,
        "cumulative_gain_validated": 2.0,
        "action_scores": {
            "backends": {"base_score": 5.0, "score_mult": 0.8},
            "params": {"base_score": 4.0, "score_mult": 1.0},
            "kernel_opt": {"base_score": 7.0, "score_mult": 0.6},
        },
        "cooldown_until_tick": {"backends": 42},
        "score_violation": {"params": 3},
        "locked_reason": {"backends": "policy_loop:foo"},
        "score_mult": {"backends": 0.7},
        "effective_score": {"backends": 4.2},
    }


def test_from_dict_drops_action_scores_silently():
    """Default ``drop`` mode: legacy fields are stripped, no warning."""
    raw = _legacy_state_payload()
    loaded = SharedState.from_dict(raw)
    assert not hasattr(loaded, "action_scores")
    assert loaded.baseline_tput == 1234.0
    assert loaded.cumulative_gain_validated == 2.0


def test_from_dict_drops_the_unvalidated_gain():
    """The raw gain was a second copy of the validated one; a resume must not revive it."""
    loaded = SharedState.from_dict(_legacy_state_payload())
    assert not hasattr(loaded, "cumulative_gain")
    assert loaded.cumulative_gain_validated == 2.0


def test_load_or_init_roundtrips_through_drop(tmp_path, monkeypatch):
    """Write a legacy state.json with action_scores, load + save, confirm field gone."""
    sd = tmp_path / "session"
    sd.mkdir()
    (sd / "state.json").write_text(json.dumps(_legacy_state_payload()))
    loaded = SharedState.load_or_init(sd)
    assert not hasattr(loaded, "action_scores")
    loaded.save(sd)
    written = json.loads((sd / "state.json").read_text())
    assert "action_scores" not in written
    assert "cumulative_gain" not in written
