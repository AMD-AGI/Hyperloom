"""N32 — Actionability tags + cheap_exhausted note in Action scores prompt.

Background: N30 boosted deep-family eff_score when N19c gate fired,
but the LLM still selected familiar shallow rows (params / backends)
instead of the boosted-to-top deep rows. Root cause uncovered with
the user (May 2026): N30's ranking surfaced kernel-owned actions
(deep_kernel_analysis / operator_tuning) above kernel_opt, and the
LLM couldn't tell at a glance whether the top rows were
``propose_action``-able or required a ``request`` emit. When the
top eff_score rows were kernel-owned-but-unmarked, the LLM tended
to skip past unfamiliar names and land on shallow rows it recognised.

N32 keeps the scoring intact (no MUST/SHALL prescriptions in the
prompt) and instead surfaces the propose path next to each row:

  ``[REQUEST: kernel-owned, kind=run_optimization]`` -- emit via
  ``request{target_agent='kernel', kind=...}``
  ``[propose_action]`` -- emit via ``delegate/propose_action``

Plus the highest-eff_score unlocked row gets ``← top actionable``
so the LLM doesn't have to scan two columns.

A cheap_exhausted header note explains the sudden eff_score jump
on deep rows when N30 fires.

Tests pinned here:

* Header note appears only when ``_is_cheap_exhausted`` returns True
* Each row has the correct actionability tag
* Top eff_score unlocked row carries the ``← top actionable`` hint
* Locked / cooldown'd rows still get their existing tags (no
  collision with the new actionability ones)
* Pre-cheap-exhausted prompt stays clean (no header note,
  actionability tags still surface)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.action_registry import ActionRegistry
from inference_optimizer.orchestrator.scoring import (
    ActionScore,
    seed_action_scores,
)


@pytest.fixture
def registry() -> ActionRegistry:
    return ActionRegistry()


def _seeded_state(registry: ActionRegistry, *, cheap_exhausted: bool) -> SharedState:
    s = SharedState()
    s.baseline_tput = 100.0
    s.tick = 10
    # Seed all registry actions with their default priors. Use
    # ``moe_mla`` to match the empirical SOLAR-10.7B-like case where
    # deep_kernel_analysis (base 8) outranks kernel_opt (base 6) at
    # the top of the cheap-exhausted ranking after N30 boost. dense
    # would flip those (kernel_opt 8 vs deep_kernel_analysis 2)
    # which we cover in a separate test below.
    enabled = [m.name for m in registry.all()]
    s.action_scores = seed_action_scores(
        registry, model_class="moe_mla", enabled=enabled,
    )
    if cheap_exhausted:
        # Make _is_cheap_exhausted True: snapshot_id=1, at least one
        # cheap attempt, delta < EPS.
        s.last_trace_analyze = {
            "roofline_snapshot_id": 1,
            "analysis_md_path": "/tmp/analysis.md",
        }
        s.params_attempts = [{"task_id": "p1"}]
        s.last_cheap_delta_gain = -0.05  # below EPS 0.3
    return s


def test_cheap_exhausted_header_appears(registry):
    s = _seeded_state(registry, cheap_exhausted=True)
    out = s.to_action_scores_summary(registry=registry, top_k=12)
    assert "cheap_exploration_exhausted=True" in out
    assert "N30 boost applied" in out


def test_cheap_not_exhausted_header_absent(registry):
    s = _seeded_state(registry, cheap_exhausted=False)
    out = s.to_action_scores_summary(registry=registry, top_k=12)
    assert "cheap_exploration_exhausted" not in out


def test_kernel_owned_actions_tagged_request(registry):
    """deep_kernel_analysis / operator_tuning / kernel_opt /
    integrate / vendor_kernel_config all carry an explicit REQUEST tag
    with the correct ``kind=`` so the LLM doesn't try ``propose_action``
    on them. integrate often ranks below top_k=12 with the moe_mla
    priors (base 0); we just verify the tag format is correct for the
    ones that DO appear in the visible top-K."""
    s = _seeded_state(registry, cheap_exhausted=True)
    out = s.to_action_scores_summary(registry=registry, top_k=20)
    assert "[REQUEST: kernel-owned, kind=deep_kernel_analysis]" in out
    assert "[REQUEST: kernel-owned, kind=operator_tuning]" in out
    assert "[REQUEST: kernel-owned, kind=run_optimization]" in out
    assert "[REQUEST: kernel-owned, kind=vendor_kernel_config]" in out
    # integrate may or may not be in top-20 depending on aging /
    # UCB; verify the tag format if present.
    if "integrate" in out:
        assert "[REQUEST: kernel-owned, kind=integrate]" in out


def test_propose_actions_tagged_propose(registry):
    """params / backends / sweep / roofline / report / target_analysis
    are all ``propose_action``-able and carry that tag."""
    s = _seeded_state(registry, cheap_exhausted=True)
    out = s.to_action_scores_summary(registry=registry, top_k=12)
    # The non-kernel-owned families share the same [propose_action] tag.
    # We assert at least the 4 commonly-shown rows.
    for non_kernel in ("params", "backends", "sweep", "roofline"):
        # Match within a per-row context to avoid false positives from
        # other rows' tag text.
        rows = [l for l in out.splitlines() if f"   {non_kernel}" in l]
        assert rows, f"{non_kernel!r} row not present in output"
        assert "[propose_action]" in rows[0], (
            f"{non_kernel!r} row missing [propose_action] tag: {rows[0]!r}"
        )


def test_top_actionable_hint_on_highest_unlocked(registry):
    """The highest-eff_score unlocked row gets ``← top actionable``.
    Under moe_mla + cheap_exhausted + N30 boost: deep_kernel_analysis
    is top (base 8 * 2 = 16 + UCB ~1.7 + aging ~0.6 ≈ 18.5)."""
    s = _seeded_state(registry, cheap_exhausted=True)
    out = s.to_action_scores_summary(registry=registry, top_k=12)
    # The hint must appear exactly once.
    assert out.count("← top actionable") == 1
    hint_line = [l for l in out.splitlines() if "← top actionable" in l][0]
    assert "deep_kernel_analysis" in hint_line


def test_no_top_hint_when_all_locked(registry):
    """When every row is locked (e.g. policy_loop), no row gets the
    actionable hint -- nothing is actionable."""
    s = _seeded_state(registry, cheap_exhausted=False)
    for name, raw in (s.action_scores or {}).items():
        if isinstance(raw, dict):
            a = ActionScore.from_dict(raw)
            a.locked_reason = "policy_loop:test"
            s.action_scores[name] = a.to_dict()
    out = s.to_action_scores_summary(registry=registry, top_k=12)
    assert "← top actionable" not in out


def test_locked_row_keeps_locked_tag_not_actionability(registry):
    """Locked rows show ``[locked: ...]`` (not [propose_action] /
    [REQUEST]) -- actionability tag is only for unlocked rows so the
    LLM doesn't mis-read a locked deep row as REQUEST-ready."""
    s = _seeded_state(registry, cheap_exhausted=True)
    # Lock kernel_opt specifically.
    raw = s.action_scores.get("kernel_opt") or {}
    a = ActionScore.from_dict(raw)
    a.locked_reason = "grid_exhausted"
    s.action_scores["kernel_opt"] = a.to_dict()
    # Use top_k=99 so locked rows (sorted below positive scores at
    # _LOCKED_SCORE=-1) are rendered too -- the registry has ~22
    # actions and locked rows sort to the very bottom.
    out = s.to_action_scores_summary(registry=registry, top_k=99)
    ko_rows = [l for l in out.splitlines() if "   kernel_opt" in l]
    assert ko_rows
    assert "[locked: grid_exhausted]" in ko_rows[0]
    assert "[REQUEST: kernel-owned" not in ko_rows[0]


def test_dense_model_kernel_opt_is_top(registry):
    """dense model_class priors flip the deep ordering: kernel_opt
    base=8.0 outranks deep_kernel_analysis base=2.0 -> after N30
    boost (16 vs 4), kernel_opt is top actionable."""
    s = SharedState()
    s.baseline_tput = 100.0
    s.tick = 10
    enabled = [m.name for m in registry.all()]
    s.action_scores = seed_action_scores(
        registry, model_class="dense", enabled=enabled,
    )
    s.last_trace_analyze = {"roofline_snapshot_id": 1}
    s.params_attempts = [{"task_id": "p1"}]
    s.last_cheap_delta_gain = -0.05
    out = s.to_action_scores_summary(registry=registry, top_k=12)
    hint_line = [l for l in out.splitlines() if "← top actionable" in l][0]
    assert "kernel_opt" in hint_line


def test_cooldown_row_keeps_cooldown_tag(registry):
    """Same precedence: cooldown'd rows get [cooldown N] not
    actionability tag."""
    s = _seeded_state(registry, cheap_exhausted=True)
    raw = s.action_scores.get("params") or {}
    a = ActionScore.from_dict(raw)
    a.cooldown_until_tick = s.tick + 5  # cooldown 5 ticks
    s.action_scores["params"] = a.to_dict()
    out = s.to_action_scores_summary(registry=registry, top_k=12)
    params_rows = [l for l in out.splitlines() if "   params" in l]
    assert params_rows
    assert "[cooldown 5]" in params_rows[0]
    assert "[propose_action]" not in params_rows[0]
