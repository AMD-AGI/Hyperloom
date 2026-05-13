"""Tests for the per-action scoring mechanism.

Covers the plan ``action-scoring-in-shared-state``:

* Group 1 — pure-function unit tests for :mod:`scoring`.
* Group 2 — anti-loop: cooldown + diminishing returns after consecutive KEEPs.
* Group 3 — anti-starvation: aging + UCB lift a low-base action over time.
* Group 4 — locked rows render and score as -1.0.
* Group 5 — :class:`SharedState` JSON round-trip preserves the new fields,
  and old state.json files (missing the new fields) load with defaults.
* Group 6 — :class:`Coordinator` integration: KEEP-tier then DISCARD-tier
  delegate result updates ``action_scores`` deterministically.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator import scoring
from inference_optimizer.orchestrator.action_registry import (
    ActionMetadata,
    ActionRegistry,
)
from inference_optimizer.orchestrator.backends import (
    MockBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.scoring import (
    ActionScore,
    DISCARD_MULT,
    KEEP_DECAY_FLOOR,
    STREAK_PENALTY_FLOOR,
    STREAK_PENALTY_MULT,
    STREAK_THRESHOLD,
    apply_discard,
    apply_failure,
    apply_keep,
    apply_lock,
    apply_no_promote,
    apply_unlock,
    compute_initial_priors_from_metadata,
    effective_score,
    rank_top_k,
    seed_action_scores,
    target_gap_multiplier,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.paths import make_session_dir


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def registry() -> ActionRegistry:
    return ActionRegistry().load()


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_SESSION_DIR", str(tmp_path))
    return make_session_dir()


def _heartbeat() -> Intent:
    return Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "ok"},
    )


def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(silent, name="o"),
        "kernel":        MockBackend(silent, name="k"),
        "critic":        MockBackend(silent, name="c"),
        "robustness":    MockBackend(silent, name="r"),
    }


# ===========================================================================
# Group 1 — pure functions
# ===========================================================================
def _meta(
    name: str,
    *,
    lo: float = 1.0,
    hi: float = 5.0,
    cost: float = 5.0,
    acc_risk: float = 0.0,
    crash_risk: float = 0.0,
) -> ActionMetadata:
    """Hand-rolled ActionMetadata for unit tests so they don't depend on
    the live registry yaml."""
    return ActionMetadata(
        name=name,
        family="explore",
        cost_minutes_p50=cost,
        cost_minutes_p75=cost * 2.0,
        expected_gain_pct=(lo, hi),
        accuracy_risk=acc_risk,
        crash_risk=crash_risk,
        prerequisites=[],
        requires_lanes=[],
        allowed_tools=[],
        side_effects=[],
        preferred_backend="claude",
        preferred_model="claude-opus-4-7",
        max_turns=10,
        lease_ttl_sec=600,
        description="",
        pipeline_phase="explore",
        typical_runtime_min=cost,
        applicable_when=[],
    )


def test_compute_initial_priors_from_metadata_basic():
    m = _meta("backends", lo=2.0, hi=8.0, cost=5.0, acc_risk=0.0, crash_risk=0.05)
    # ((2+8)/2)/5 * (1-0) * (1-0.05) = 5/5 * 0.95 = 0.95
    assert compute_initial_priors_from_metadata(m) == pytest.approx(0.95)


def test_compute_initial_priors_zero_for_measurement_actions():
    # profile yaml carries expected_gain_pct = [0.0, 0.0] — base = 0.0
    m = _meta("profile", lo=0.0, hi=0.0, cost=3.0)
    assert compute_initial_priors_from_metadata(m) == pytest.approx(0.0)


def test_seed_action_scores_uses_marathon_prior_when_available(registry):
    seeded = seed_action_scores(
        registry,
        model_class="moe_mla",
        enabled=["backends", "operator_tuning", "deep_kernel_analysis", "params"],
    )
    # marathon moe_mla: operator_tuning=7.0, deep_kernel_analysis=8.0
    assert seeded["operator_tuning"]["base_score"] == pytest.approx(7.0)
    assert seeded["deep_kernel_analysis"]["base_score"] == pytest.approx(8.0)
    # backends has no marathon entry — falls back to auto
    auto_backends = compute_initial_priors_from_metadata(registry.get("backends"))
    assert seeded["backends"]["base_score"] == pytest.approx(auto_backends)


def test_seed_action_scores_auto_for_unknown_model_class(registry):
    seeded = seed_action_scores(
        registry,
        model_class="unknown_class",
        enabled=["operator_tuning"],
    )
    auto = compute_initial_priors_from_metadata(registry.get("operator_tuning"))
    assert seeded["operator_tuning"]["base_score"] == pytest.approx(auto)


def test_seed_action_scores_normalises_model_class(registry):
    a = seed_action_scores(registry, model_class="moe-mla", enabled=["operator_tuning"])
    b = seed_action_scores(registry, model_class="MoE+MLA", enabled=["operator_tuning"])
    assert a["operator_tuning"]["base_score"] == pytest.approx(7.0)
    assert b["operator_tuning"]["base_score"] == pytest.approx(7.0)


def test_effective_score_locked_returns_sentinel():
    a = ActionScore(base_score=5.0, locked_reason="grid_exhausted")
    m = _meta("params")
    assert effective_score(a, meta=m, tick=10, total_runs=5) == -1.0


def test_effective_score_ignores_cooldown_until_tick():
    """Mandatory cooldowns were retired; ``cooldown_until_tick`` survives as
    a backward-compat field in state.json but must NOT lock the row.

    The streak-based penalty in :func:`apply_failure` /
    :func:`apply_no_promote` plus the linear aging bonus already
    discourage spamming an action; an additional cooldown produced a
    degenerate idempotency-loop signature (see
    ``orch_idempotency_and_scoring_fix`` plan)."""
    a = ActionScore(base_score=5.0, cooldown_until_tick=20)
    m = _meta("backends")
    # Even with cooldown_until_tick > tick, the row scores normally now.
    eff_during_cooldown = effective_score(a, meta=m, tick=10, total_runs=5)
    eff_after_cooldown = effective_score(a, meta=m, tick=25, total_runs=5)
    assert eff_during_cooldown > 0.0
    # Aging keeps growing past the (now-cosmetic) cooldown boundary.
    assert eff_after_cooldown >= eff_during_cooldown


def test_effective_score_includes_aging_bonus():
    a = ActionScore(base_score=2.0, last_run_tick=0)
    m = _meta("params", acc_risk=0.0, crash_risk=0.0)
    eff_t1 = effective_score(a, meta=m, tick=1, total_runs=1)
    eff_t40 = effective_score(a, meta=m, tick=40, total_runs=1)
    assert eff_t40 > eff_t1  # aging bonus monotonic


def test_apply_keep_decays_score_mult_with_floor():
    a = ActionScore(base_score=3.0, score_mult=1.0)
    # Big gain — would push mult below the floor without clamping.
    apply_keep(a, gain_pct=100.0, tick=5, action_name="backends")
    assert a.score_mult == pytest.approx(KEEP_DECAY_FLOOR)
    assert a.runs == 1
    assert a.keeps == 1
    # apply_keep no longer sets cooldown_until_tick (cooldowns retired).
    assert a.cooldown_until_tick == 0


def test_apply_keep_small_gain_minor_decay():
    a = ActionScore(base_score=3.0, score_mult=1.0)
    apply_keep(a, gain_pct=2.0, tick=10, action_name="backends")
    # 1 - 0.1*2 = 0.8
    assert a.score_mult == pytest.approx(0.8)


def test_apply_discard_aliases_apply_failure():
    """`apply_discard` is retained as a backward-compat alias that funnels
    into the failure streak. A single call must NOT immediately knock
    `score_mult` (only the 3-strike threshold does); but it must bump
    `consecutive_failures` and the `discards` KPI counter."""
    a = ActionScore(base_score=3.0, score_mult=1.0)
    apply_discard(a, tick=5, action_name="backends")
    # No immediate score_mult drop — DISCARD_MULT is kept as a legacy
    # constant but the active code path is now streak-based.
    assert a.score_mult == pytest.approx(1.0)
    assert a.discards == 1
    assert a.consecutive_failures == 1
    assert a.consecutive_no_promote == 0
    assert a.last_gain_pct == 0.0
    # Cooldown stays untouched.
    assert a.cooldown_until_tick == 0
    # DISCARD_MULT is still exported as a documented legacy constant.
    assert 0.0 < DISCARD_MULT < 1.0


def test_apply_lock_unlock():
    a = ActionScore(base_score=3.0)
    apply_lock(a, "grid_exhausted")
    assert a.locked_reason == "grid_exhausted"
    # Lock is sticky — second call doesn't change the reason
    apply_lock(a, "other_reason")
    assert a.locked_reason == "grid_exhausted"
    apply_unlock(a)
    assert a.locked_reason == ""


def test_target_gap_multiplier_bands():
    assert target_gap_multiplier(target_gap_pct=0.0, cumulative_gain=0.0) == 1.0
    # remaining = 35 - 5 = 30 (boundary lands in 1.6 band per design)
    assert target_gap_multiplier(target_gap_pct=35.0, cumulative_gain=5.0) == 1.6
    # remaining = 35 - 30 = 5 → 1.0 band (< 15)
    assert target_gap_multiplier(target_gap_pct=35.0, cumulative_gain=30.0) == 1.0
    # remaining 0 → 0.1
    assert target_gap_multiplier(target_gap_pct=10.0, cumulative_gain=20.0) == 0.1


def test_rank_top_k_orders_by_effective_score(registry):
    seeded = seed_action_scores(
        registry, model_class="moe_mla",
        enabled=["backends", "operator_tuning", "params", "kernel_opt"],
    )
    rows = rank_top_k(seeded, registry, tick=1, k=4)
    names = [r[0] for r in rows]
    # operator_tuning prior=7.0 vs kernel_opt prior=6.0 — operator wins
    assert names[0] == "operator_tuning"
    assert names[1] == "kernel_opt"


# ===========================================================================
# Group 2 — anti-loop: three consecutive backends KEEPs decay the mult
# (cooldown gating removed in favour of streak-based dampening)
# ===========================================================================
def test_consecutive_keeps_decay_score_mult_only(registry):
    """Three back-to-back KEEPs at +2% should compound the diminishing-
    returns decay to (0.8)^3 = 0.512, but the row must remain selectable
    at every intermediate tick — mandatory cooldowns were retired."""
    seeded = seed_action_scores(
        registry, model_class="moe_mla",
        enabled=["backends", "params"],
    )

    # Run 1
    b = ActionScore.from_dict(seeded["backends"])
    apply_keep(b, gain_pct=2.0, tick=1, action_name="backends")
    seeded["backends"] = b.to_dict()
    assert b.cooldown_until_tick == 0  # cooldown no longer written
    rows = rank_top_k(seeded, registry, tick=2, k=2)
    name_to_eff = {r[0]: r[1] for r in rows}
    # Both rows must remain scorable (no -1.0 sentinel from cooldown).
    assert name_to_eff["backends"] > 0.0
    assert name_to_eff["params"] > 0.0

    # Run 2 — score_mult = 0.8 * 0.8 = 0.64
    b = ActionScore.from_dict(seeded["backends"])
    apply_keep(b, gain_pct=2.0, tick=2, action_name="backends")
    seeded["backends"] = b.to_dict()
    assert b.score_mult == pytest.approx(0.64)
    assert b.score_mult >= KEEP_DECAY_FLOOR

    # Run 3 — score_mult = 0.64 * 0.8 = 0.512, still above floor
    b = ActionScore.from_dict(seeded["backends"])
    apply_keep(b, gain_pct=2.0, tick=3, action_name="backends")
    seeded["backends"] = b.to_dict()
    assert b.score_mult == pytest.approx(0.512)
    assert b.score_mult >= KEEP_DECAY_FLOOR


# ---------------------------------------------------------------------------
# Group 2b — streak-based penalty (replaces mandatory cooldown)
# ---------------------------------------------------------------------------
def test_apply_failure_bumps_streak_and_penalizes_at_threshold():
    """Three consecutive failures should shave ``score_mult`` by exactly
    one :data:`STREAK_PENALTY_MULT` factor and reset the streak."""
    a = ActionScore(base_score=4.0, score_mult=1.0)
    apply_failure(a, tick=1, action_name="backends")
    assert a.consecutive_failures == 1
    assert a.score_mult == pytest.approx(1.0)  # no penalty yet

    apply_failure(a, tick=2, action_name="backends")
    assert a.consecutive_failures == 2
    assert a.score_mult == pytest.approx(1.0)  # still no penalty

    apply_failure(a, tick=3, action_name="backends")
    # Threshold crossed — exactly one STREAK_PENALTY_MULT applied, streak reset.
    assert a.score_mult == pytest.approx(STREAK_PENALTY_MULT)
    assert a.consecutive_failures == 0
    assert a.runs == 3
    assert a.discards == 3
    # No-promote streak untouched.
    assert a.consecutive_no_promote == 0
    # Cooldown remains 0 — explicit goal of the streak-only design.
    assert a.cooldown_until_tick == 0


def test_apply_no_promote_bumps_streak_and_penalizes_at_threshold():
    """Same shape as failure but on a separate streak counter so the two
    pathologies surface independently in diagnostics."""
    a = ActionScore(base_score=4.0, score_mult=1.0)
    for tick in range(STREAK_THRESHOLD - 1):
        apply_no_promote(a, tick=tick, action_name="backends")
        assert a.score_mult == pytest.approx(1.0)
    apply_no_promote(a, tick=STREAK_THRESHOLD, action_name="backends")
    assert a.score_mult == pytest.approx(STREAK_PENALTY_MULT)
    assert a.consecutive_no_promote == 0
    # Failure streak untouched throughout.
    assert a.consecutive_failures == 0


def test_keep_resets_both_streaks():
    """A successful promote must clear BOTH penalty streaks so we don't
    carry an old streak into a new productive phase.

    apply_failure / apply_no_promote zero the *other* streak on each
    call (they're mutually exclusive paths), so to exercise the
    keep-resets-both semantic we seed both counters directly and verify
    apply_keep zeroes them in one shot.
    """
    a = ActionScore(base_score=4.0, score_mult=1.0)
    a.consecutive_failures = 2
    a.consecutive_no_promote = 2
    apply_keep(a, gain_pct=1.0, tick=3, action_name="backends")
    assert a.consecutive_failures == 0
    assert a.consecutive_no_promote == 0


def test_streak_penalty_respects_floor():
    """Repeatedly tripping the streak penalty must never push
    ``score_mult`` below :data:`STREAK_PENALTY_FLOOR`."""
    a = ActionScore(base_score=4.0, score_mult=1.0)
    # Trip the failure streak many times — each cycle of 3 failures
    # multiplies by STREAK_PENALTY_MULT = 0.85, so without the floor we'd
    # converge to 0.0.
    for cycle in range(40):  # 40 * 3 = 120 failures
        for k in range(STREAK_THRESHOLD):
            apply_failure(a, tick=cycle * STREAK_THRESHOLD + k, action_name="backends")
    assert a.score_mult >= STREAK_PENALTY_FLOOR
    # Sanity: the floor must actually be the binding constraint after this
    # many cycles, not an artefact of arithmetic.
    assert a.score_mult == pytest.approx(STREAK_PENALTY_FLOOR)


# ===========================================================================
# Group 3 — anti-starvation: aging + UCB
# ===========================================================================
def test_low_base_action_surfaces_via_aging(registry):
    # Seed two siblings with the same low base. One ("running") gets a run
    # every few ticks; the other ("starved") never runs. After enough ticks
    # the starved row should overtake the regularly-running row because of
    # the aging bonus.
    seeded = seed_action_scores(
        registry, model_class="moe_mla",
        enabled=["backends", "comm_optimization"],
    )
    # Force comm_optimization base lower than backends to make sure the
    # ordering at tick=0 is unambiguous.
    comm = ActionScore.from_dict(seeded["comm_optimization"])
    comm.base_score = 2.0
    seeded["comm_optimization"] = comm.to_dict()

    bk = ActionScore.from_dict(seeded["backends"])
    bk.base_score = 2.0
    seeded["backends"] = bk.to_dict()

    # At tick 0 both score similarly. Run backends every 5 ticks for 50 ticks
    # and never run comm_optimization.
    for t in range(0, 50, 5):
        bk = ActionScore.from_dict(seeded["backends"])
        apply_keep(bk, gain_pct=0.5, tick=t, action_name="backends")
        seeded["backends"] = bk.to_dict()

    rows = rank_top_k(seeded, registry, tick=50, k=2)
    name_to_eff = {r[0]: r[1] for r in rows}
    # Comm_optimization should rank strictly higher than backends thanks to
    # aging (50 ticks × 0.05 = +2.5) and UCB bonus (runs=0 vs runs=10).
    assert name_to_eff["comm_optimization"] > name_to_eff["backends"]
    assert rows[0][0] == "comm_optimization"


def test_aging_bonus_monotonic(registry):
    # Verify that the same action's effective score keeps climbing with
    # idle time, even with a small base.
    seeded = seed_action_scores(
        registry, model_class="moe_mla", enabled=["comm_optimization"],
    )
    a = ActionScore.from_dict(seeded["comm_optimization"])
    a.base_score = 2.0
    a.last_run_tick = 0
    seeded["comm_optimization"] = a.to_dict()
    eff_5 = rank_top_k(seeded, registry, tick=5, k=1)[0][1]
    eff_50 = rank_top_k(seeded, registry, tick=50, k=1)[0][1]
    assert eff_50 > eff_5


# ===========================================================================
# Group 4 — locked rows
# ===========================================================================
def test_locked_row_renders_with_locked_tag(registry):
    s = SharedState()
    s.action_scores = seed_action_scores(
        registry, model_class="moe_mla",
        enabled=["backends", "params"],
    )
    # Lock params: grid_exhausted
    raw = s.action_scores["params"]
    a = ActionScore.from_dict(raw)
    apply_lock(a, "grid_exhausted")
    s.action_scores["params"] = a.to_dict()
    s.tick = 5
    summary = s.to_action_scores_summary(registry=registry, top_k=12)
    assert "[locked: grid_exhausted]" in summary
    assert "locked: params(grid_exhausted)" in summary
    # effective_score should be -1 for locked
    eff = effective_score(
        ActionScore.from_dict(s.action_scores["params"]),
        meta=registry.get("params"),
        tick=5, total_runs=0,
    )
    assert eff == -1.0


# ===========================================================================
# Group 5 — serialization round-trip
# ===========================================================================
def test_shared_state_action_scores_round_trip(tmp_path, registry):
    s = SharedState()
    s.action_scores = seed_action_scores(
        registry, model_class="moe_mla",
        enabled=["backends", "params", "operator_tuning"],
    )
    # Mutate one row so we have non-default fields to verify.
    a = ActionScore.from_dict(s.action_scores["backends"])
    apply_keep(a, gain_pct=2.5, tick=3, action_name="backends")
    s.action_scores["backends"] = a.to_dict()
    s.tick = 17
    s.target_gap_pct = 12.5
    s.save(tmp_path)

    s2 = SharedState.load_or_init(tmp_path)
    assert s2.tick == 17
    assert s2.target_gap_pct == pytest.approx(12.5)
    assert "backends" in s2.action_scores
    b = ActionScore.from_dict(s2.action_scores["backends"])
    assert b.runs == 1
    assert b.keeps == 1
    assert b.last_gain_pct == pytest.approx(2.5)
    assert b.score_mult == pytest.approx(0.75)  # 1 - 0.1*2.5
    # Mandatory cooldowns retired; apply_keep no longer touches the field.
    assert b.cooldown_until_tick == 0
    # New streak fields round-trip through JSON.
    assert b.consecutive_failures == 0
    assert b.consecutive_no_promote == 0


def test_old_state_json_loads_with_defaults(tmp_path):
    # Pretend state.json was written by a pre-scoring build (no new fields).
    legacy = {
        "session_id": "legacy",
        "baseline_tput": 800.0,
        "cumulative_gain": 5.0,
    }
    (tmp_path / "state.json").write_text(json.dumps(legacy), encoding="utf-8")
    s = SharedState.load_or_init(tmp_path)
    assert s.session_id == "legacy"
    assert s.baseline_tput == 800.0
    assert s.action_scores == {}
    assert s.tick == 0
    assert s.target_gap_pct == 0.0


def test_corrupted_action_scores_entry_dropped(tmp_path):
    # Defensive: non-dict entry inside action_scores is filtered out instead
    # of raising during load.
    raw = {
        "action_scores": {
            "backends": {"base_score": 1.0},
            "params": "not-a-dict",
        },
    }
    (tmp_path / "state.json").write_text(json.dumps(raw), encoding="utf-8")
    s = SharedState.load_or_init(tmp_path)
    assert "backends" in s.action_scores
    assert "params" not in s.action_scores


# ===========================================================================
# Group 6 — Coordinator integration
# ===========================================================================
@pytest.mark.asyncio
async def test_coordinator_scores_keep_then_no_promote(session_dir):
    """Drive Coordinator._promote_to_shared_state with one KEEP-tier and
    one NO-PROMOTE backends result; verify the action_scores row evolves
    deterministically (runs=2, keeps=1, discards=1) and that the
    no-promote streak is being tracked. Cooldown is intentionally NOT
    asserted any more — mandatory cooldowns were retired."""
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        c.shared_state.baseline_tput = 800.0
        c.shared_state.current_best = {"action": "baseline", "tput": 800.0}
        c.shared_state.cumulative_gain = 0.0
        c.shared_state.save(session_dir)

        # KEEP-tier: +10% over current_best — promotes.
        keep_result = {
            "status": "succeeded",
            "output_throughput": 880.0,
            "best_variant": {
                "name": "attn_triton",
                "extra_sglang_args": "--attention-backend triton",
            },
        }
        await c.tick(1)
        await c._promote_to_shared_state("backends", keep_result)

        backends_after_keep = ActionScore.from_dict(
            c.shared_state.action_scores["backends"],
        )
        assert backends_after_keep.runs == 1
        assert backends_after_keep.keeps == 1
        assert backends_after_keep.discards == 0
        # Both streaks reset by the successful promote.
        assert backends_after_keep.consecutive_failures == 0
        assert backends_after_keep.consecutive_no_promote == 0
        # Cooldowns disabled — field stays at 0.
        assert backends_after_keep.cooldown_until_tick == 0

        # NO-PROMOTE: +0.05% — below threshold, succeeded but no promote.
        no_promote_result = {
            "status": "succeeded",
            "output_throughput": 880.5,
            "best_variant": {
                "name": "attn_aiter",
            },
        }
        await c.tick(1)
        await c._promote_to_shared_state("backends", no_promote_result)

        b = ActionScore.from_dict(c.shared_state.action_scores["backends"])
        assert b.runs == 2
        assert b.keeps == 1
        assert b.discards == 1
        # First no-promote — streak counter incremented, no penalty yet.
        assert b.consecutive_no_promote == 1
        assert b.consecutive_failures == 0
        assert b.cooldown_until_tick == 0

        # tick advanced twice via Coordinator.tick(1) calls.
        assert c.shared_state.tick >= 2
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_seeds_action_scores_on_construction(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        assert c.shared_state.action_scores
        # The marathon prior for moe_mla operator_tuning is 7.0; with the
        # default model_class fallback we expect 7.0 here.
        assert (
            c.shared_state.action_scores["operator_tuning"]["base_score"]
            == pytest.approx(7.0)
        )
        # Persisted in state.json
        reloaded = SharedState.load_or_init(session_dir)
        assert "operator_tuning" in reloaded.action_scores
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_coordinator_renders_scoreboard_in_orchestration_prompt(session_dir):
    c = Coordinator(session_dir, backends=_silent_backends())
    try:
        await c.tick(1)
        prompt = await c._compose_prompt("orchestration")
        assert "=== Action scores" in prompt
        assert "operator_tuning" in prompt
        # Kernel prompt should NOT include the scoreboard.
        kernel_prompt = await c._compose_prompt("kernel")
        assert "=== Action scores" not in kernel_prompt
    finally:
        await c.stop()
