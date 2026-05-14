"""Phase 2 — SharedState extensions + Coordinator mission/time/validate-stack.

These tests pin the new public surface introduced when Orchestration
gained an explicit Mission progress section, time-budget awareness, and
a ``validate_stack`` requirement after each KEEP'd entry on
``optimization_stack``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.backends import (
    MockBackend,
    MockCriticBackend,
    MockKernelBackend,
    MockRobustnessBackend,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.orchestrator.policy import (
    CORE_STATE_FIELDS,
    PolicyDenied,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.paths import make_session_dir


# ===========================================================================
# fixtures
# ===========================================================================
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _silent_intent() -> Intent:
    return Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "ok"},
    )


def _backends_full() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_silent_intent())
    return {
        "orchestration": MockBackend(silent, name="orch"),
        "kernel":        MockKernelBackend(),
        "critic":        MockCriticBackend(),
        "robustness":    MockRobustnessBackend(),
    }


def _backends_no_kernel() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_silent_intent())
    return {
        "orchestration": MockBackend(silent, name="orch"),
        "critic":        MockCriticBackend(),
        "robustness":    MockRobustnessBackend(),
    }


def _no_kernel_role_registry():
    """Mirror of cli.py's no-kernel role registry — drop the 'kernel' role."""
    from inference_optimizer.orchestrator.agent_role import default_role_registry
    return {k: v for k, v in default_role_registry().items() if k != "kernel"}


# ===========================================================================
# SharedState — Phase 2 fields + helpers
# ===========================================================================
def test_shared_state_new_fields_default_values():
    s = SharedState()
    assert s.cumulative_gain_validated == 0.0
    assert s.cumulative_gain_validated_ts == ""
    assert s.cumulative_gain_validated_stack_len == 0


def test_elapsed_minutes_handles_unset_or_bad_start_ts():
    s = SharedState(start_ts="")
    assert s.elapsed_minutes() == 0.0
    s.start_ts = "not-an-isoformat-string"
    assert s.elapsed_minutes() == 0.0


def test_elapsed_minutes_uses_injected_now():
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    s = SharedState(start_ts=base.isoformat(timespec="microseconds"))
    later = base + timedelta(minutes=42, seconds=30)
    assert s.elapsed_minutes(now=later) == pytest.approx(42.5, abs=1e-6)


def test_remaining_minutes_clamps_to_zero_and_handles_unbounded():
    base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    s = SharedState(
        start_ts=base.isoformat(timespec="microseconds"),
        max_minutes=60,
    )
    assert s.remaining_minutes(now=base + timedelta(minutes=20)) == pytest.approx(40.0)
    # Past the budget — clamp.
    assert s.remaining_minutes(now=base + timedelta(minutes=120)) == pytest.approx(0.0)
    # Unbounded run.
    s.max_minutes = 0
    assert s.remaining_minutes(now=base + timedelta(minutes=10)) is None


def test_optimization_stack_has_unvalidated_keeps():
    s = SharedState()
    assert not s.optimization_stack_has_unvalidated_keeps()
    s.optimization_stack = [{"action": "backends", "variant_name": "v1"}]
    assert s.optimization_stack_has_unvalidated_keeps()
    s.cumulative_gain_validated_stack_len = 1
    assert not s.optimization_stack_has_unvalidated_keeps()
    s.optimization_stack.append({"action": "params", "variant_name": "v2"})
    assert s.optimization_stack_has_unvalidated_keeps()


def test_to_mission_summary_shape_and_unvalidated_flag():
    base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    s = SharedState(
        baseline_tput=100.0,
        cumulative_gain=5.0,
        cumulative_gain_validated=3.5,
        cumulative_gain_validated_ts="2026-01-01T00:00:00+00:00",
        cumulative_gain_validated_stack_len=1,
        optimization_stack=[
            {"action": "backends", "variant_name": "aiter"},
            {"action": "params", "variant_name": "p1"},
        ],
        start_ts=base.isoformat(timespec="microseconds"),
        max_minutes=120,
        current_best={"action": "params", "tput": 110.0, "variant_name": "p1"},
    )
    text = s.to_mission_summary(now=base + timedelta(minutes=30))
    assert "baseline" in text and "100.0" in text
    assert "current" in text and "params" in text
    assert "per-round-sum=5.00%" in text
    assert "validated=3.50%" in text
    assert "elapsed=30.0min" in text
    assert "remaining=90.0min" in text
    assert "budget=120min" in text
    # Stack has 2 entries, validated at len=1 → unvalidated flag fires
    assert "stack changed" in text


def test_to_mission_summary_unbounded_budget():
    s = SharedState(start_ts=datetime.now(timezone.utc).isoformat(timespec="microseconds"))
    text = s.to_mission_summary()
    assert "budget=unlimited" in text
    assert "remaining=" not in text


def test_to_prompt_summary_includes_validated_gain():
    s = SharedState(
        cumulative_gain_validated=4.2,
        cumulative_gain_validated_ts="2026-05-11T12:00:00+00:00",
        cumulative_gain_validated_stack_len=3,
    )
    text = s.to_prompt_summary()
    assert "cumulative_gain_validated=4.2%" in text
    assert "stack_len_at_validation=3" in text
    assert "ts=2026-05-11T12:00:00+00:00" in text


def test_save_load_roundtrips_phase2_fields(tmp_path):
    s = SharedState(
        cumulative_gain_validated=7.5,
        cumulative_gain_validated_ts="2026-05-11T12:00:00+00:00",
        cumulative_gain_validated_stack_len=4,
    )
    s.save(tmp_path)
    loaded = SharedState.load_or_init(tmp_path)
    assert loaded.cumulative_gain_validated == 7.5
    assert loaded.cumulative_gain_validated_ts == "2026-05-11T12:00:00+00:00"
    assert loaded.cumulative_gain_validated_stack_len == 4


# ===========================================================================
# PolicyGate — new core fields
# ===========================================================================
def test_validated_gain_fields_are_core_state_fields():
    """LLM agents must not be able to UPDATE_STATE the validated gain."""
    for f in (
        "cumulative_gain_validated",
        "cumulative_gain_validated_ts",
        "cumulative_gain_validated_stack_len",
    ):
        assert f in CORE_STATE_FIELDS


# ===========================================================================
# Coordinator._compose_prompt — Mission progress section
# ===========================================================================
@pytest.mark.asyncio
async def test_compose_prompt_includes_mission_section_for_orchestration(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    coord.shared_state.baseline_tput = 100.0
    coord.shared_state.cumulative_gain = 2.5
    coord.shared_state.optimization_stack = [
        {"action": "backends", "variant_name": "aiter"},
    ]
    text = await coord._compose_prompt("orchestration")
    assert "=== Mission progress ===" in text
    assert "baseline  : 100.0" in text
    assert "per-round-sum=2.50%" in text
    # The unvalidated-stack warning must surface in mission so the
    # decision framework sees it before the SharedState dump.
    assert "stack changed" in text
    # Mission appears BEFORE the regular Shared session state dump.
    assert text.index("=== Mission progress ===") < text.index("=== Shared session state ===")


@pytest.mark.asyncio
async def test_compose_prompt_omits_mission_section_for_other_agents(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    text = await coord._compose_prompt("kernel")
    assert "=== Mission progress ===" not in text
    # Critic / kernel still get the regular shared session state.
    assert "=== Shared session state ===" in text


# ===========================================================================
# Coordinator._required_next_step — validate_stack TODO
# ===========================================================================
def test_required_next_step_validate_stack_after_unvalidated_keep(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_select_kernels = {
        "trace_input": "/tmp/profile.tar.gz", "candidates_path": "/tmp/x.json",
    }
    # No KEEPs yet — no TODO
    assert coord._required_next_step() == ""
    # New KEEP appears, validated_at_len still 0 → TODO fires
    s.optimization_stack = [{"action": "backends", "variant_name": "aiter"}]
    todo = coord._required_next_step()
    assert "validate_stack required" in todo
    assert "stack_len=1" in todo
    assert "validated_at_len=0" in todo
    # Validate caught up → TODO clears
    s.cumulative_gain_validated_stack_len = 1
    assert coord._required_next_step() == ""


def test_required_next_step_no_kernel_skips_profile_select(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    sd = make_session_dir()
    coord = Coordinator(
        sd,
        backends=_backends_no_kernel(),
        role_registry=_no_kernel_role_registry(),
    )
    s = coord.shared_state
    s.baseline_tput = 100.0
    # No-kernel mode: profile/select_kernels should not be required.
    assert coord._required_next_step() == ""
    # validate_stack TODO still fires once a KEEP lands.
    s.optimization_stack = [{"action": "backends", "variant_name": "aiter"}]
    assert "validate_stack required" in coord._required_next_step()


def test_required_next_step_baseline_still_first(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    s = coord.shared_state
    s.optimization_stack = [{"action": "backends", "variant_name": "aiter"}]
    # baseline_tput == 0 — baseline TODO must take precedence
    todo = coord._required_next_step()
    assert "baseline is required now" in todo
    assert "validate_stack required" not in todo


# ===========================================================================
# Coordinator._sequence_denial_for_action — validate_stack guard
# ===========================================================================
def test_sequence_denial_blocks_explore_after_unvalidated_keep(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_select_kernels = {
        "trace_input": "/tmp/profile.tar.gz", "candidates_path": "/tmp/x.json",
    }
    s.optimization_stack = [{"action": "backends", "variant_name": "aiter"}]
    # backends/params/sweep/report are denied
    for action in ("backends", "params", "sweep", "report"):
        denied = coord._sequence_denial_for_action(action)
        assert isinstance(denied, PolicyDenied), (
            f"{action!r} should be denied while validate_stack is required"
        )
        assert denied.rule == "validate_stack_required"
    # validate_stack itself is allowed
    assert coord._sequence_denial_for_action("validate_stack") is None
    # baseline still allowed (rare ad-hoc rebench)
    assert coord._sequence_denial_for_action("baseline") is None


def test_sequence_denial_clears_after_validation(session_dir):
    coord = Coordinator(session_dir, backends=_backends_full())
    s = coord.shared_state
    s.baseline_tput = 100.0
    s.last_profile_trace = "/tmp/profile.tar.gz"
    s.last_select_kernels = {
        "trace_input": "/tmp/profile.tar.gz", "candidates_path": "/tmp/x.json",
    }
    s.optimization_stack = [{"action": "backends", "variant_name": "aiter"}]
    s.cumulative_gain_validated_stack_len = 1
    # Now backends should be allowed again
    assert coord._sequence_denial_for_action("backends") is None
    assert coord._sequence_denial_for_action("params") is None
