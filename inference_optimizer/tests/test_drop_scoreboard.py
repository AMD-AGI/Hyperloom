"""v0.8 §3.9 — drop scoreboard tests.

Covers KB_design/3.9_drop_scoreboard/README.md:

* SharedState no longer carries ``action_scores`` (Inv-9.1).
* Legacy ``action_scores`` / ``cooldown_until_tick`` / ``score_violation``
  / ``locked_reason`` payloads in a resumed state.json are silently
  dropped (default ``--legacy-action-scores=drop``) or logged
  (``--legacy-action-scores=warn``).
* ``params_no_promote_streak`` is preserved as a *fact* (LLM reads
  it; the system no longer derives a *priority* from it).
* ``orchestrator.scoring`` module is gone.
* ``Coordinator`` no longer carries any ``_score_action_*`` /
  ``_apply_action_score_update`` / ``_ensure_action_scores_seeded`` stubs
  (KB_gaps/Dead-B): both the methods and their call sites are deleted.
* The Orchestration prompt no longer carries ``Action scores`` block.
* PolicyGate ``family_pruned`` denial hint never mentions "Action scores".
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.shared_state import SharedState


# ===========================================================================
# 1. SharedState dataclass surface
# ===========================================================================
def test_shared_state_has_no_action_scores_field():
    """KB_design §3.9 §4.1: ``action_scores`` dropped from the dataclass."""
    s = SharedState()
    assert not hasattr(s, "action_scores"), (
        "v0.8 §3.9 retired action_scores; field must be removed from "
        "SharedState (KB_design §3.9 Inv-9.1)."
    )


def test_shared_state_has_no_scoring_helpers():
    """``get_action_score`` / ``put_action_score`` / ``all_action_scores``
    / ``to_action_scores_summary`` were retired with the scoreboard."""
    s = SharedState()
    for name in (
        "get_action_score", "put_action_score", "all_action_scores",
        "to_action_scores_summary",
    ):
        assert not hasattr(s, name), (
            f"{name!r} should be removed (KB_design §3.9 §4.2)"
        )


def test_shared_state_keeps_params_no_promote_streak_as_fact():
    """v0.8 §3.9 keeps the streak field — it's a *fact*, not a priority.

    Inv-9.1 only forbids system-side *priority values*. KEEP/REVERT
    counts and streak counters are allowed because the LLM reads
    them as evidence, not as ordering.
    """
    s = SharedState()
    assert s.params_no_promote_streak == 0
    assert "params_no_promote_streak" in s.to_prompt_summary()


def test_shared_state_keeps_tick_and_target_gap_pct():
    """``tick`` is a monotonic counter (plateau / phase math). The
    ``target_gap_pct`` is a *fact* (how much gain is still needed)
    not a multiplier. Both stay."""
    s = SharedState()
    s.increment_tick()
    s.increment_tick()
    assert s.tick == 2
    assert s.target_gap_pct == 0.0


def test_shared_state_all_top_actions_policy_locked_removed():
    """KB_gaps/Dead-B — the scoreboard-based "everything's locked" stub
    is deleted (plateau judges took over). The attribute must not exist
    on ``SharedState`` at all."""
    s = SharedState()
    assert not hasattr(s, "all_top_actions_policy_locked")


# ===========================================================================
# 2. Legacy migration — drop / warn modes
# ===========================================================================
def _legacy_state_payload() -> dict:
    """A v0.6-shaped state.json snapshot loaded with action_scores +
    a couple of related legacy fields."""
    return {
        "session_id": "legacy-sid",
        "baseline_tput": 1234.0,
        "cumulative_gain": 2.5,
        "action_scores": {
            "backends": {"base_score": 5.0, "score_mult": 0.8},
            "params":   {"base_score": 4.0, "score_mult": 1.0},
            "kernel_opt": {"base_score": 7.0, "score_mult": 0.6},
        },
        "cooldown_until_tick": {"backends": 42},
        "score_violation":     {"params": 3},
        "locked_reason":       {"backends": "policy_loop:foo"},
        "score_mult":          {"backends": 0.7},
        "effective_score":     {"backends": 4.2},
    }


def test_from_dict_drops_action_scores_silently(monkeypatch):
    """Default ``drop`` mode: legacy fields are stripped, no warning."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES", raising=False)
    raw = _legacy_state_payload()
    loaded = SharedState.from_dict(raw)
    assert not hasattr(loaded, "action_scores")
    assert loaded.baseline_tput == 1234.0
    assert loaded.cumulative_gain == 2.5


def test_from_dict_drop_mode_logs_at_info_level(monkeypatch, caplog):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES", raising=False)
    raw = _legacy_state_payload()
    with caplog.at_level(logging.INFO,
                          logger="inference_optimizer.orchestrator.shared_state"):
        SharedState.from_dict(raw)
    matched = [r for r in caplog.records if "v0.8 §3.9" in r.getMessage()]
    assert matched, "drop mode should log at INFO level"
    assert all(r.levelno == logging.INFO for r in matched)


def test_from_dict_warn_mode_emits_warning(monkeypatch, caplog):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES", "warn")
    raw = _legacy_state_payload()
    with caplog.at_level(logging.WARNING,
                          logger="inference_optimizer.orchestrator.shared_state"):
        SharedState.from_dict(raw)
    matched = [r for r in caplog.records if "v0.8 §3.9" in r.getMessage()]
    assert matched, "warn mode should log a WARNING"
    assert any(r.levelno == logging.WARNING for r in matched)


def test_from_dict_no_legacy_fields_means_no_log(monkeypatch, caplog):
    """A clean v0.8 state.json doesn't produce any §3.9 log line."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES", "warn")
    raw = {"session_id": "fresh", "baseline_tput": 999.0}
    with caplog.at_level(logging.WARNING,
                          logger="inference_optimizer.orchestrator.shared_state"):
        SharedState.from_dict(raw)
    matched = [r for r in caplog.records if "§3.9" in r.getMessage()]
    assert matched == []


def test_load_or_init_roundtrips_through_drop(tmp_path, monkeypatch):
    """Full filesystem path: write a v0.6 state.json with action_scores,
    load it, save it back, confirm the field is gone."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_LEGACY_ACTION_SCORES", raising=False)
    sd = tmp_path / "session"
    sd.mkdir()
    (sd / "state.json").write_text(json.dumps(_legacy_state_payload()))
    loaded = SharedState.load_or_init(sd)
    assert not hasattr(loaded, "action_scores")
    loaded.save(sd)
    written = json.loads((sd / "state.json").read_text())
    assert "action_scores" not in written


# ===========================================================================
# 3. orchestrator.scoring module is gone
# ===========================================================================
def test_scoring_module_was_retired():
    with pytest.raises(ImportError):
        importlib.import_module("inference_optimizer.orchestrator.scoring")


# ===========================================================================
# 4. Coordinator scoring surface fully removed (KB_gaps/Dead-B)
# ===========================================================================
def test_coordinator_has_no_scoring_methods():
    """KB_gaps/Dead-B — every v0.6 scoreboard hook on Coordinator is
    physically removed (methods + their callers)."""
    from inference_optimizer.orchestrator.coordinator import Coordinator

    for name in (
        "_score_action_keep",
        "_score_action_discard",
        "_score_action_failure",
        "_score_action_no_promote",
        "_score_action_lock",
        "_apply_action_score_update",
        "_ensure_action_scores_seeded",
    ):
        assert not hasattr(Coordinator, name), (
            f"{name!r} must be deleted (KB_gaps/Dead-B §4.1-§4.3)"
        )


def test_coordinator_source_has_no_scoreboard_callers():
    """Defense in depth: no call sites remain in the coordinator body."""
    from inference_optimizer.orchestrator import coordinator as _c

    src = Path(_c.__file__).read_text(encoding="utf-8")
    for needle in (
        "_score_action_",
        "_apply_action_score_update(",
        "_ensure_action_scores_seeded(",
        "to_action_scores_summary(",
    ):
        assert needle not in src, (
            f"coordinator still references retired symbol {needle!r}"
        )


def test_pruned_family_advisory_observation_has_no_scoreboard_vocab():
    """KB_gaps/Dead-B §B.4 — the pruned-family advisory observation
    string must not mention "Action scores" any more.

    Loosen P3_19 demoted the prune dispatch from a hard PolicyDenied to
    an advisory observation so the LLM may still pick the family if it
    judges the prune speculative; the scoreboard-vocab guard simply
    moved to the advisory hint string.
    """
    from inference_optimizer.orchestrator import coordinator as _c

    src = Path(_c.__file__).read_text(encoding="utf-8")
    advisory_idx = src.find('"delegate_pruned_advisory"')
    assert advisory_idx >= 0
    window = src[advisory_idx : advisory_idx + 800]
    assert "Action scores" not in window
    assert "phase-allowed action" in window


# ===========================================================================
# 5. Orchestration prompt has no Action scores top-12 block
# ===========================================================================
def test_orchestration_prompt_has_no_scoreboard_block():
    """KB_design §3.9 §8 — DECISION FRAMEWORK section must not steer
    the LLM toward ``eff_score`` / ``cooldown`` / ``score_mult``.

    The only allowed mentions of "Action scores" are historical
    callouts that explain ``v0.8 retired ...`` so the LLM knows the
    surface is gone.
    """
    from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
        FULL_ENABLED_ACTIONS,
        build_orchestration_prompt,
    )
    from inference_optimizer.orchestrator.action_registry import ActionRegistry
    reg = ActionRegistry().load()
    prompt = build_orchestration_prompt(
        action_registry=reg,
        enabled_actions=FULL_ENABLED_ACTIONS,
        framework="sglang", kernel_enabled=True,
        objective_kind="gain_pct", objective_value=10.0, max_minutes=120,
    )
    # No live scoring vocabulary.
    forbidden = (
        "eff_score=", "score_mult *=", "score_mult=",
        "cooldown_until_tick", "[locked:", "[cooldown",
        "ucb_bonus", "aging_bonus", "effective_score",
    )
    for needle in forbidden:
        assert needle not in prompt, (
            f"prompt still references retired scoring token {needle!r}"
        )
    # KB_design §3.9 Inv-9.1 mention is present (the LLM is told why
    # there's no scoreboard).
    assert "Inv-9.1" in prompt
    # New phase-aware action selection block landed.
    assert "Phase-aware action selection" in prompt


def test_kernel_opt_body_has_no_scoreboard_vocab():
    """KB_gaps/Dead-D — the kernel-pipeline body injected into the
    Orchestration prompt whenever KERNEL is enabled must NOT carry any
    v0.6 scoreboard vocabulary."""
    from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
        _KERNEL_OPT_PIPELINE_BODY,
    )

    haystack = _KERNEL_OPT_PIPELINE_BODY.lower()
    # ``Action scores`` may appear as a historical callout (the body
    # explicitly tells the LLM the surface was retired); the *live*
    # scoring vocab below must not.
    forbidden = (
        "scoreboard",
        "score_mult",
        "marathon_priors",
        "effective_score",
        "eff_score",
        "cooldown_until_tick",
        "scoreboard surfaces",
        "scoreboard decides",
        "action scores top-12",
    )
    for needle in forbidden:
        assert needle not in haystack, (
            f"_KERNEL_OPT_PIPELINE_BODY still references retired token "
            f"{needle!r} (KB_gaps/Dead-D §5.1)"
        )


def test_kernel_opt_body_references_v08_decision_signals():
    """KB_gaps/Dead-D §5.1 — the body must surface the v0.8 decision
    facts (gaps[] / last_action_failures / last_kernel_opt / PARTIAL
    cap / KERNEL plateau advisory) so KERNEL-phase action selection has
    a concrete fact list instead of an implicit scoreboard."""
    from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
        _KERNEL_OPT_PIPELINE_BODY,
    )

    body = _KERNEL_OPT_PIPELINE_BODY
    for signal in (
        "state.gaps[]",
        "last_action_failures",
        "last_kernel_opt",
        "KERNEL plateau",
        "rejected_kernel_ids",
        "_DEFAULT_KERNEL_OPT_MAX_PARTIAL",
    ):
        assert signal in body, (
            f"_KERNEL_OPT_PIPELINE_BODY missing v0.8 decision signal "
            f"{signal!r} (KB_gaps/Dead-D §5.1)"
        )


def test_orchestration_md_has_no_score_view():
    """The rules fragment (``orchestration.md``) should also be free of
    score-view directives (KB_design §3.9 §8)."""
    from inference_optimizer.paths import asset_system_prompts_dir

    fragment = (asset_system_prompts_dir() / "orchestration.md").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "Action scores top-12",
        "score_violation",
        "cooldown N",
        "[cooldown ",
        "[locked: ",
        "effective_score",
    )
    for needle in forbidden:
        assert needle not in fragment, (
            f"orchestration.md still references retired token {needle!r}"
        )
    # The v0.8 §3.9 decision rule shows up.
    assert "§3.9" in fragment


# ===========================================================================
# 6. CLI flag presence
# ===========================================================================
def test_cli_exposes_legacy_action_scores_flag():
    """``--legacy-action-scores`` must be wired (drop / warn)."""
    from inference_optimizer.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args([
        "optimize",
        "--model", "/tmp/dummy-model",
        "--legacy-action-scores", "warn",
    ])
    assert args.legacy_action_scores == "warn"
    args2 = parser.parse_args([
        "optimize",
        "--model", "/tmp/dummy-model",
    ])
    # Default is "drop" (or the env override).
    assert args2.legacy_action_scores in ("drop", "warn")


def test_cli_rejects_unknown_legacy_action_scores_value():
    from inference_optimizer.cli import _build_parser
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "optimize",
            "--model", "/tmp/dummy-model",
            "--legacy-action-scores", "keep",
        ])
