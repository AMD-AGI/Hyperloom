"""Tests for ``orchestrator.scheduler`` — IMPL-CHECKLIST §3.11‒3.30."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from inference_optimizer.paths import asset_actions_dir
from inference_optimizer.orchestrator.action_registry import (
    ActionMetadata,
    ActionRegistry,
)
from inference_optimizer.orchestrator.execution_mode import ExecutionMode
from inference_optimizer.orchestrator.objective import build_objective
from inference_optimizer.orchestrator.scheduler import (
    ActionScore,
    BudgetAwareScheduler,
)
from inference_optimizer.orchestrator.score_priors import ModelClass
from inference_optimizer.orchestrator.shared_state import SharedState


PACKAGE_ACTIONS_DIR = asset_actions_dir()


@pytest.fixture(scope="module")
def registry() -> ActionRegistry:
    return ActionRegistry(PACKAGE_ACTIONS_DIR).load()


@pytest.fixture
def state() -> SharedState:
    return SharedState(
        session_id="t",
        model_path="/srv/models/llama-3-8b",
        model_name="llama-3-8b",
        model_class="dense",
        max_minutes=120.0,
        elapsed_minutes=10.0,
        execution_mode=ExecutionMode.GUIDED_KERNEL_OPT,
    )


def _scheduler(
    registry: ActionRegistry,
    *,
    mode: ExecutionMode = ExecutionMode.GUIDED_KERNEL_OPT,
    env: dict[str, str] | None = None,
) -> BudgetAwareScheduler:
    env = env or {"MODEL_PATH": "/srv/models/llama-3-8b", "MAX_HOURS": "2"}
    obj = build_objective(env)
    return BudgetAwareScheduler(
        objective=obj,
        mode=mode,
        env=env,
        action_registry=registry,
    )


# ---------------------------------------------------------------------------
# pressure
# ---------------------------------------------------------------------------
def test_pressure_grows_as_time_runs_out(registry, state: SharedState):
    sch = _scheduler(registry)
    state.elapsed_minutes = 5.0
    p1 = sch.pressure(state)
    state.elapsed_minutes = 110.0
    p2 = sch.pressure(state)
    assert p2 > p1
    assert 0.0 <= p1 <= 1.0
    assert 0.0 <= p2 <= 1.0


def test_pressure_floors_at_zero_when_no_max(registry, state: SharedState):
    sch = _scheduler(registry)
    state.max_minutes = 0.0
    assert 0.0 <= sch.pressure(state) <= 1.0


# ---------------------------------------------------------------------------
# mode_gate / depth_gate / diminishing / lane_available
# ---------------------------------------------------------------------------
def test_mode_gate_zeroes_disallowed_action(registry, state: SharedState):
    """Plan A: kernel_opt removed; deep_kernel_analysis stands in as a
    representative action that's not allowed in quick mode."""
    sch = _scheduler(registry, mode=ExecutionMode.QUICK_PARAM_SWEEP)
    a = registry.get("deep_kernel_analysis")
    assert a is not None
    s = sch.score(a, state)
    assert s.breakdown["mode_gate"] == 0.0
    assert s.score == 0.0


def test_mode_gate_allows_quick_action(registry, state: SharedState):
    sch = _scheduler(registry, mode=ExecutionMode.QUICK_PARAM_SWEEP)
    a = registry.get("bench_runner")
    s = sch.score(a, state)
    assert s.breakdown["mode_gate"] == 1.0


def test_depth_gate_blocks_too_long_action(registry, state: SharedState):
    """``framework_rebuild`` (~75 min p75) should be killed when only 30 min remain."""
    sch = _scheduler(registry, mode=ExecutionMode.MARATHON_MULTI_AGENT)
    state.max_minutes = 60.0
    state.elapsed_minutes = 30.0
    a = registry.get("framework_rebuild")
    s = sch.score(a, state)
    assert s.breakdown["depth_gate"] == 0.0


def test_depth_gate_allows_short_action(registry, state: SharedState):
    sch = _scheduler(registry, mode=ExecutionMode.GUIDED_KERNEL_OPT)
    a = registry.get("bench_runner")
    s = sch.score(a, state)
    assert s.breakdown["depth_gate"] == 1.0


def test_diminishing_factor_reduces_after_repeats(registry, state: SharedState):
    sch = _scheduler(registry)
    a = registry.get("bench_runner")
    history = [
        {"family": "prep"} for _ in range(3)
    ]
    s = sch.score(a, state, history=history)
    assert s.breakdown["diminishing"] == pytest.approx(0.7 ** 3)


def test_lane_available_zero_when_lane_held(registry, state: SharedState):
    sch = _scheduler(registry)
    a = registry.get("bench_runner")
    summary = {"benchmark_lane": {"holder": "agent-x", "ttl": 100}}
    s = sch.score(a, state, lock_summary=summary)
    assert s.breakdown["lane_available"] == 0.0


def test_lane_available_one_when_unrelated_lane_held(registry, state: SharedState):
    sch = _scheduler(registry)
    a = registry.get("bench_runner")
    summary = {"workspace_mutation": {"holder": "agent-x", "ttl": 100}}
    s = sch.score(a, state, lock_summary=summary)
    assert s.breakdown["lane_available"] == 1.0


# ---------------------------------------------------------------------------
# pick_next
# ---------------------------------------------------------------------------
def test_pick_next_returns_action_with_max_score(registry, state: SharedState):
    sch = _scheduler(registry)
    pick = sch.pick_next(state)
    assert pick is not None
    # dense + guided → 'kernel_opt' (prior=8, gain 3‑25%, allowed)
    # bench_runner has prior=1 (not in DENSE table) and 0 gain → score=0.
    assert pick.name in {a.name for a in registry.allowed_for_mode(sch.mode)}


def test_pick_next_returns_none_when_all_modes_blocked(
    registry, state: SharedState, monkeypatch
):
    sch = _scheduler(registry, mode=ExecutionMode.QUICK_PARAM_SWEEP)
    state.max_minutes = 0.001  # everything fails depth_gate
    state.elapsed_minutes = 0.0
    pick = sch.pick_next(state)
    assert pick is None


def test_pick_next_prefers_followup_queue(registry, state: SharedState):
    sch = _scheduler(registry)
    sch._enqueue_followup("report")
    pick = sch.pick_next(state)
    assert pick is not None and pick.name == "report"


def test_pick_next_skips_unknown_followup(registry, state: SharedState):
    sch = _scheduler(registry)
    sch._enqueue_followup("does-not-exist")
    sch._enqueue_followup("bench_runner")
    pick = sch.pick_next(state)
    assert pick is not None and pick.name == "bench_runner"


# ---------------------------------------------------------------------------
# update_after_action — §9.3 rules 1-7
# ---------------------------------------------------------------------------
def test_rule1_succeeded_boosts_family(registry, state: SharedState):
    sch = _scheduler(registry)
    bench = registry.get("bench_runner")  # prep family
    sch.update_after_action(bench, gain_pct=2.0, status="succeeded")
    # Other prep-family actions (setup / classify / baseline / ...) should
    # all see a positive adjustment.
    boosted = [v for v in sch.adjustments.values() if v > 1.0]
    assert boosted, "expected at least one prep-family adjustment boost"


def test_rule2_failed_halves_family(registry, state: SharedState):
    """Plan A: kernel_opt removed; deep_kernel_analysis stands in as the
    trigger action whose family-mates should be halved by Rule 2."""
    sch = _scheduler(registry)
    a = registry.get("deep_kernel_analysis")
    assert a is not None
    sch.update_after_action(a, gain_pct=0.0, status="failed")
    halved = [v for v in sch.adjustments.values() if v == 0.5]
    assert halved, "expected at least one deep_kernel-family halving"


def test_rule3_combined_backends_pushed(registry, state: SharedState):
    sch = _scheduler(registry)
    a = registry.get("backends")
    sch.update_after_action(a, gain_pct=5.0, status="succeeded")
    assert "combined_backends_test" in sch.followups


def test_rule5_kernel_keep_pushes_profile(
    registry, state: SharedState
):
    """Plan A: Rule 5 enqueues 'profile' only — the kernel_opt followup
    was removed because executor cannot delegate(kernel_opt); the kernel
    agent path is driven by the request_kernel_optimization subskill."""
    sch = _scheduler(registry)
    a = registry.get("deep_kernel_analysis")  # stand-in deep_kernel action
    assert a is not None
    sch.update_after_action(a, gain_pct=10.0, status="succeeded")
    assert "profile" in sch.followups
    # kernel_opt is no longer enqueued (no such registered action).
    assert "kernel_opt" not in sch.followups


def test_rule6_kernel_discard_reduces_remaining(
    registry, state: SharedState
):
    sch = _scheduler(registry)
    a = registry.get("deep_kernel_analysis")
    assert a is not None
    sch.update_after_action(a, gain_pct=0.0, status="reverted")
    # any other deep_kernel-family entry should now have a < 1.0 adjustment
    any_dim = [
        v for k, v in sch.adjustments.items()
        if v < 1.0 and k != "deep_kernel_analysis"
    ]
    assert any_dim


def test_rule7_low_scores_push_sweep_then_report(
    registry, state: SharedState
):
    sch = _scheduler(registry)
    # Force every score < 1.0 by making gains zero across all actions.
    fake_scores = [
        ActionScore(name="x", score=0.5),
        ActionScore(name="y", score=0.2),
    ]
    sch._maybe_apply_rule_7(fake_scores)
    assert sch.followups == ["sweep", "report"]


# ---------------------------------------------------------------------------
# breakdown integrity
# ---------------------------------------------------------------------------
def test_score_breakdown_keys_present(registry, state: SharedState):
    sch = _scheduler(registry)
    a = registry.get("bench_runner")
    s = sch.score(a, state)
    expected_keys = {
        "base", "pressure", "mode_gate", "depth_gate",
        "diminishing", "lane_available", "prior", "adjustment",
    }
    assert expected_keys <= set(s.breakdown)
