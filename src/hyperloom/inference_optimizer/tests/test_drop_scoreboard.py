# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Drop scoreboard tests."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from hyperloom.orchestrator.state.shared_state import SharedState


def test_shared_state_has_no_action_scores_field():
    """``action_scores`` is dropped from the dataclass."""
    s = SharedState()
    assert not hasattr(s, "action_scores"), (
        "v0.8 §3.9 retired action_scores; field must be removed from SharedState (KB_design §3.9 Inv-9.1)."
    )


def test_shared_state_has_no_scoring_helpers():
    """Scoring helpers were retired with the scoreboard."""
    s = SharedState()
    for name in (
        "get_action_score",
        "put_action_score",
        "all_action_scores",
        "to_action_scores_summary",
    ):
        assert not hasattr(s, name), f"{name!r} should be removed (KB_design §3.9 §4.2)"


def test_shared_state_keeps_params_no_promote_streak_as_fact():
    """The streak field stays because it is a *fact*, not a priority."""
    s = SharedState()
    assert s.params_no_promote_streak == 0
    assert "params_no_promote_streak" in s.to_prompt_summary()


def test_shared_state_keeps_tick_and_target_gap_pct():
    """``tick`` (counter) and ``target_gap_pct`` (fact) both stay."""
    s = SharedState()
    s.increment_tick()
    s.increment_tick()
    assert s.tick == 2
    assert s.target_gap_pct == 0.0


def test_shared_state_all_top_actions_policy_locked_removed():
    """The "everything's locked" stub is deleted from SharedState."""
    s = SharedState()
    assert not hasattr(s, "all_top_actions_policy_locked")


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


def test_scoring_module_was_retired():
    """The retired ``orchestrator/scoring.py`` scoreboard module never comes back
    (distinct from the ``hyperloom.orchestrator.scoring`` subpackage holding the
    always-advisory ``proposal_scorer.py``).
    """
    scoring_pkg = importlib.import_module("hyperloom.orchestrator.scoring")
    assert not hasattr(scoring_pkg, "get_action_score")
    assert not hasattr(scoring_pkg, "put_action_score")
    assert not hasattr(scoring_pkg, "all_action_scores")


def test_coordinator_has_no_scoring_methods():
    """Every scoreboard hook on Coordinator is removed."""
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    for name in (
        "_score_action_keep",
        "_score_action_discard",
        "_score_action_failure",
        "_score_action_no_promote",
        "_score_action_lock",
        "_apply_action_score_update",
        "_ensure_action_scores_seeded",
    ):
        assert not hasattr(Coordinator, name), f"{name!r} must be deleted (KB_gaps/Dead-B §4.1-§4.3)"


def test_coordinator_source_has_no_scoreboard_callers():
    """Defense in depth: no call sites remain in the coordinator body."""
    from hyperloom.orchestrator.loop import coordinator as _c

    src = Path(_c.__file__).read_text(encoding="utf-8")
    for needle in (
        "_score_action_",
        "_apply_action_score_update(",
        "_ensure_action_scores_seeded(",
        "to_action_scores_summary(",
    ):
        assert needle not in src, f"coordinator still references retired symbol {needle!r}"


def test_pruned_family_advisory_observation_has_no_scoreboard_vocab():
    """The pruned-family advisory string must not mention "Action scores"."""
    from hyperloom.orchestrator.loop import intent_router as _c

    src = Path(_c.__file__).read_text(encoding="utf-8")
    advisory_idx = src.find('"delegate_pruned_advisory"')
    assert advisory_idx >= 0
    window = src[advisory_idx : advisory_idx + 800]
    assert "Action scores" not in window
    assert "phase-allowed action" in window


def test_orchestration_prompt_has_no_scoreboard_block():
    """DECISION FRAMEWORK must not steer toward live scoring vocab."""
    from hyperloom.orchestrator.prompts.prompt_builder import (
        FULL_ENABLED_ACTIONS,
        build_orchestration_prompt,
    )
    from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE

    reg = ACTION_CATALOGUE
    prompt = build_orchestration_prompt(
        action_registry=reg,
        enabled_actions=FULL_ENABLED_ACTIONS,
        framework="sglang",
        kernel_enabled=True,
        objective_kind="gain_pct",
        objective_value=10.0,
        max_minutes=120,
    )
    forbidden = (
        "eff_score=",
        "score_mult *=",
        "score_mult=",
        "cooldown_until_tick",
        "[locked:",
        "[cooldown",
        "ucb_bonus",
        "aging_bonus",
        "effective_score",
    )
    for needle in forbidden:
        assert needle not in prompt, f"prompt still references retired scoring token {needle!r}"
    assert "no system-side" in prompt.lower() or "no system-side" in prompt
    assert "Phase-aware action selection" in prompt


def test_orchestration_md_has_no_score_view():
    """The ``orchestration.md`` fragment should be free of score-view directives."""
    from hyperloom.inference_optimizer.session.paths import asset_system_prompts_dir

    fragment = (asset_system_prompts_dir() / "orchestration.md").read_text(encoding="utf-8")
    forbidden = (
        "Action scores top-12",
        "score_violation",
        "cooldown N",
        "[cooldown ",
        "[locked: ",
        "effective_score",
    )
    for needle in forbidden:
        assert needle not in fragment, f"orchestration.md still references retired token {needle!r}"
    assert "decision priority" in fragment.lower()
