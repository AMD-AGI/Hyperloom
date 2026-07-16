# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Role realignment and phase-aware prompts tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.registry import ActionRegistry
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.prompts.prompt_builder import (
    build_orchestration_prompt,
    default_enabled_actions,
)
from hyperloom.inference_optimizer.session.paths import make_session_dir


# fixtures
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


@pytest.fixture
def registry() -> ActionRegistry:
    return ActionRegistry().load()


# Static system prompts carry phase semantics
def test_orchestration_prompt_includes_phase_contract(registry):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=False),
        framework="sglang",
        objective_kind="gain_pct",
        objective_value=10.0,
        max_minutes=120,
    )
    assert "PHASE CONTRACT" in text
    for phase in ("PRELUDE", "FRAMEWORK_AGENT", "EXPLORE", "KERNEL_AGENT", "SWEEP", "CLOSE"):
        assert phase in text, f"missing phase {phase} from orchestration prompt"
    assert "phase-allowed actions" in text.lower()
    assert "policy_denied" in text.lower()


def test_orchestration_prompt_no_kernel_marks_kernel_skipped(registry):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=True),
        framework="sglang",
        max_minutes=120,
    )
    assert "(DISABLED: --no-kernel — phase skipped)" in text


def test_orchestration_prompt_no_explore_trims_catalogue_and_marks_skipped(registry):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=False, no_explore=True),
        framework="sglang",
        kernel_enabled=True,
        explore_enabled=False,
        max_minutes=120,
    )
    assert "(DISABLED: --no-explore — phase skipped)" in text
    assert "explore_enabled  : false" in text
    # The `explore` action bullet must be gone from the catalogue.
    assert "- **explore** —" not in text
    # specialist/integrate_patch stay visible (KERNEL still uses them).
    assert "- **specialist** —" in text
    assert "- **integrate_patch** —" in text


def test_orchestration_prompt_no_framework_agent_marks_skipped_and_context(registry):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=False),
        framework="sglang",
        kernel_enabled=True,
        framework_agent_phase_enabled=False,
        max_minutes=120,
    )
    assert "(DISABLED: --no-framework-agent — phase skipped)" in text
    assert "framework_agent_phase_enabled : false" in text


def test_orchestration_prompt_all_enabled_session_context_true(registry):
    text = build_orchestration_prompt(
        action_registry=registry,
        enabled_actions=default_enabled_actions(no_kernel=False),
        framework="sglang",
        kernel_enabled=True,
        explore_enabled=True,
        framework_agent_phase_enabled=True,
        max_minutes=120,
    )
    assert "explore_enabled  : true" in text
    assert "framework_agent_phase_enabled : true" in text
    assert "(DISABLED:" not in text


def test_role_md_files_carry_phase_awareness():
    """Static rules fragments + Robustness markdown carry phase awareness."""
    from hyperloom.inference_optimizer.session.paths import asset_system_prompts_dir

    root = asset_system_prompts_dir()
    for name in ("orchestration", "kernel_agent", "critic", "robustness"):
        body = (root / f"{name}.md").read_text(encoding="utf-8")
        if name == "robustness":
            assert "Phase & specialist awareness" in body
        elif name == "critic":
            assert "Phase-specific rules" in body
        else:
            assert "Phase awareness" in body, f"{name}.md missing phase awareness"
        assert "PRELUDE" in body or "PHASE_PRELUDE" in body
        assert "EXPLORE" in body
        assert "KERNEL_AGENT" in body


# SharedState renderers
def test_shared_state_phase_status_summary_renders_compact_block():
    s = SharedState(max_minutes=60)
    s.record_phase_transition(
        to_phase="EXPLORE",
        reason="prelude_done",
        evidence={"baseline_tput": 100},
        ts="2026-05-19T00:00:00+00:00",
        ts_unix=1_000_000.0,
    )
    out = s.to_phase_status_summary(
        budget_pct={"EXPLORE": 0.5},
        now_unix=1_000_120.0,
    )
    assert "phase     : EXPLORE" in out
    assert "entered" in out
    assert "elapsed_sec=120" in out
    # 60 min × 60s × 0.5 = 1800s cap; elapsed 120s → 1680s remaining.
    assert "remaining_sec=1680" in out
    # EXPLORE allowlist carries explore + specialist + recover only.
    assert "explore" in out and "specialist" in out


def test_shared_state_phase_status_summary_no_max_minutes_marks_unlimited():
    s = SharedState(max_minutes=0)
    s.record_phase_transition(
        to_phase="EXPLORE",
        reason="prelude_done",
        evidence={},
        ts="2026-05-19T00:00:00+00:00",
        ts_unix=1.0,
    )
    out = s.to_phase_status_summary(now_unix=10.0)
    assert "unlimited run" in out.lower()


def test_shared_state_phase_budget_telemetry_reports_per_phase_elapsed():
    s = SharedState(max_minutes=60)
    s.record_phase_transition(
        to_phase="PRELUDE",
        reason="phase_entered",
        evidence={},
        ts="2026-05-19T00:00:00+00:00",
        ts_unix=1_000_000.0,
    )
    s.record_phase_transition(
        to_phase="EXPLORE",
        reason="prelude_done",
        evidence={},
        ts="2026-05-19T00:01:00+00:00",
        ts_unix=1_000_060.0,
    )
    out = s.to_phase_budget_telemetry(now_unix=1_000_300.0)
    # PRELUDE: 60s elapsed, cap 180s (5% of 3600s), used 33%.
    assert "PRELUDE: elapsed=60s" in out
    # EXPLORE: 240s elapsed (300-60), cap 2160s (60% of 3600s), used 11%.
    assert "EXPLORE: elapsed=240s" in out
    # Both lines present.
    assert out.count("elapsed=") == 2


def test_shared_state_phase_budget_telemetry_includes_framework():
    s = SharedState(max_minutes=60)
    s.record_phase_transition(
        to_phase="PRELUDE",
        reason="phase_entered",
        evidence={},
        ts="2026-05-19T00:00:00+00:00",
        ts_unix=1_000_000.0,
    )
    s.record_phase_transition(
        to_phase="FRAMEWORK_AGENT",
        reason="prelude_done",
        evidence={},
        ts="2026-05-19T00:01:00+00:00",
        ts_unix=1_000_060.0,
    )
    s.record_phase_transition(
        to_phase="EXPLORE",
        reason="framework_agent_phase_done",
        evidence={},
        ts="2026-05-19T00:03:00+00:00",
        ts_unix=1_000_180.0,
    )
    out = s.to_phase_budget_telemetry(now_unix=1_000_300.0)
    assert "PRELUDE: elapsed=60s" in out
    assert "FRAMEWORK_AGENT: elapsed=120s" in out
    assert "EXPLORE: elapsed=120s" in out
    assert out.count("elapsed=") == 3


def test_shared_state_warm_start_summary_empty_when_no_recipe():
    assert SharedState().to_warm_start_summary() == ""


def test_shared_state_warm_start_summary_renders_recipe_and_pitfalls():
    s = SharedState()
    s.warm_start_recipe = {
        "workload": "deepseek-r1",
        "hw": "mi300x",
        "raw": "recipe_id=42 stack=sglang/0.4.10\nbest_config={'foo':'bar'}\nwhat_worked=[A, B]",
    }
    s.warm_start_pitfalls = [
        {"raw": "OOM on fp8 expert_dtype — switch to fp4"},
        {"raw": "TP=8 + ISL>=8k causes nccl hang"},
    ]
    out = s.to_warm_start_summary()
    assert "workload=deepseek-r1" in out
    assert "hw=mi300x" in out
    assert "recipe_id=42" in out
    assert "pitfalls (2):" in out
    assert "OOM on fp8" in out


# Coordinator per-tick prompt assembly
def _silent_intent() -> Intent:
    return Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "ok"},
    )


@pytest.fixture
def coordinator_with_mocks(session_dir):
    from hyperloom.orchestrator.roles import (
        MockBackend,
        MockCriticBackend,
        MockRobustnessBackend,
        ScriptedPlan,
    )
    from hyperloom.orchestrator.loop.coordinator import Coordinator

    silent = ScriptedPlan(turns=[], default_intent=_silent_intent())
    backends = {
        "orchestration": MockBackend(silent, name="orch"),
        "kernel_agent": MockBackend(silent, name="kernel_agent"),
        "critic": MockCriticBackend(),
        "robustness": MockRobustnessBackend(),
    }
    return Coordinator(session_dir, backends=backends)


@pytest.mark.asyncio
async def test_compose_prompt_emits_phase_block_for_every_role(
    coordinator_with_mocks,
):
    c = coordinator_with_mocks
    try:
        for role in ("orchestration", "kernel_agent", "critic", "robustness"):
            prompt = await c._compose_prompt(role)
            assert "=== Phase ===" in prompt, f"{role}: phase block missing"
            assert "phase     : PRELUDE" in prompt, f"{role}: phase value missing"
            assert "allowed" in prompt, f"{role}: allowed-actions line missing"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_compose_prompt_orchestration_renders_warm_start_when_set(
    coordinator_with_mocks,
    session_dir,
):
    c = coordinator_with_mocks
    try:
        c.shared_state.warm_start_recipe = {
            "workload": "qwen3-8b",
            "hw": "mi325x",
            "raw": "recipe_id=99 best_throughput=2100",
        }
        c.shared_state.save(session_dir)
        prompt = await c._compose_prompt("orchestration")
        assert "=== Warm start (Cortex T0) ===" in prompt
        assert "workload=qwen3-8b" in prompt
        assert "recipe_id=99" in prompt
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_compose_prompt_orchestration_omits_warm_start_when_empty(
    coordinator_with_mocks,
):
    c = coordinator_with_mocks
    try:
        prompt = await c._compose_prompt("orchestration")
        assert "=== Warm start" not in prompt
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_compose_prompt_robustness_renders_specialist_health(
    coordinator_with_mocks,
):
    c = coordinator_with_mocks
    try:
        prompt = await c._compose_prompt("robustness")
        assert "=== Specialist health ===" in prompt
        assert "running=0 stale=0" in prompt
        assert "stale_threshold_sec=600" in prompt
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_compose_prompt_robustness_includes_budget_telemetry(
    coordinator_with_mocks,
    session_dir,
):
    c = coordinator_with_mocks
    try:
        # Skip FRAMEWORK so this exercises PRELUDE → EXPLORE; force a transition for telemetry.
        c.shared_state.framework_agent_phase_enabled = False
        c.shared_state.baseline_tput = 1500.0
        c.shared_state.save(session_dir)
        await c.tick(1)
        prompt = await c._compose_prompt("robustness")
        assert "=== Phase budget telemetry ===" in prompt
        assert "PRELUDE: elapsed=" in prompt
        assert "EXPLORE: elapsed=" in prompt
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_scan_stale_specialists_returns_empty_when_no_specialists(
    coordinator_with_mocks,
):
    c = coordinator_with_mocks
    try:
        stale = await c._scan_stale_specialists()
        assert stale == []
    finally:
        await c.stop()


# Phase budget telemetry math (independent of coordinator)
def test_to_phase_budget_telemetry_handles_empty_history():
    s = SharedState()
    out = s.to_phase_budget_telemetry()
    assert out == "(no phase history yet)"
