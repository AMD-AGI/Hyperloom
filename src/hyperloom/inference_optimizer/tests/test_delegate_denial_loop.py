# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Delegate idempotency + policy-denial ladder tests."""

from __future__ import annotations

import re

import pytest

from hyperloom.orchestrator.roles import MockBackend, ScriptedPlan
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.paths import make_session_dir


def _silent_coordinator(session_dir) -> Coordinator:
    silent = ScriptedPlan(turns=[])
    return Coordinator(
        session_dir,
        backends={
            "orchestration": MockBackend(silent, name="o"),
            "critic": MockBackend(silent, name="c"),
            "robustness": MockBackend(silent, name="r"),
        },
    )


def _delegate(
    *,
    action: str = "long_running",
    params: dict | None = None,
    key: str | None = "dup-key-1",
) -> Intent:
    payload: dict = {"action_name": action, "params": params or {"x": 1}}
    if key is not None:
        payload["idempotency_key"] = key
    return Intent(type=IntentType.DELEGATE, payload=payload)


@pytest.fixture
def session_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


@pytest.mark.asyncio
async def test_delegate_terminal_collision_appends_retry_suffix(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        first, _ = await c.tasks.create_or_return_existing(
            kind="long_running",
            params={"x": 1},
            idempotency_key="dup-key-1",
        )
        await c.tasks.transition(first.task_id, "running", evidence={})
        await c.tasks.transition(first.task_id, "succeeded", evidence={})

        await c._handle_delegate("orchestration", _delegate(key="dup-key-1"))
        queued = await c.tasks.by_state("queued")
        keys = {t.idempotency_key for t in queued}
        assert "dup-key-1-retry1" in keys
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_delegate_running_collision_denies_without_new_task(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        await c.tasks.create_or_return_existing(
            kind="long_running",
            params={"x": 1},
            idempotency_key="dup-key-run",
        )
        before = len(await c.tasks.by_state("queued"))
        await c._handle_delegate("orchestration", _delegate(key="dup-key-run"))
        after = len(await c.tasks.by_state("queued"))
        assert after == before
        obs = await c.bus.tail(topic="observation")
        denied = [
            m
            for m in obs
            if m.payload.get("kind") == "policy_denied" and m.payload.get("rule") == "duplicate_idempotency_key_running"
        ]
        assert denied
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_delegate_fallback_key_uses_tick_and_content_fingerprint(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        c.shared_state.tick = 42
        await c._handle_delegate(
            "orchestration",
            _delegate(key=None, params={"grid": [{"name": "a"}]}),
        )
        queued = await c.tasks.by_state("queued")
        assert len(queued) == 1
        key = queued[0].idempotency_key
        assert re.match(
            r"^orchestration:long_running:t42:[0-9a-f]{10}$",
            key,
        )
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_policy_denial_streak_records_streak_at_two(session_dir):
    """The denial streak is tracked via ``SharedState.policy_denial_streak`` as a count, not a priority lock."""
    c = _silent_coordinator(session_dir)
    try:
        from hyperloom.orchestrator.policy.gate import PolicyDenied

        intent = _delegate(action="backends", key="k1")
        pd = PolicyDenied("denied", rule="duplicate_idempotency_key", hint="wait")
        await c._record_policy_denied(
            "orchestration",
            intent,
            pd,
            action_name="backends",
        )
        await c._record_policy_denied(
            "orchestration",
            intent,
            pd,
            action_name="backends",
        )
        streak = c.shared_state.policy_denial_streak.get(
            "backends:duplicate_idempotency_key",
            0,
        )
        assert streak >= 2
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_policy_denial_streak_no_longer_prunes_family_at_five(session_dir):
    """Long-run continuity: the streak >= 5 auto-prune_family reaction was removed; the family is NOT pruned."""
    c = _silent_coordinator(session_dir)
    try:
        from hyperloom.orchestrator.policy.gate import PolicyDenied

        intent = _delegate(action="params", key="k1")
        pd = PolicyDenied("denied", rule="duplicate_idempotency_key", hint="wait")
        for _ in range(5):
            await c._record_policy_denied(
                "orchestration",
                intent,
                pd,
                action_name="params",
            )
        assert "params" not in c.shared_state.pruned_families
        assert (
            c.shared_state.policy_denial_streak.get(
                "params:duplicate_idempotency_key",
                0,
            )
            == 5
        )
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_policy_denial_streak_no_longer_stops_run_at_ten(session_dir):
    """Long-run continuity: the streak >= 10 ``policy_loop`` stop was removed; stop_reason stays unset."""
    c = _silent_coordinator(session_dir)
    try:
        from hyperloom.orchestrator.policy.gate import PolicyDenied

        intent = _delegate(action="backends", key="k1")
        pd = PolicyDenied("denied", rule="duplicate_idempotency_key", hint="wait")
        for _ in range(10):
            await c._record_policy_denied(
                "orchestration",
                intent,
                pd,
                action_name="backends",
            )
        assert not (c.shared_state.stop_reason or "").strip()
        assert (
            c.shared_state.policy_denial_streak.get(
                "backends:duplicate_idempotency_key",
                0,
            )
            == 10
        )
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_successful_delegate_resets_policy_denial_streak(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        from hyperloom.orchestrator.policy.gate import PolicyDenied

        intent = _delegate(key="k-reset")
        pd = PolicyDenied("denied", rule="duplicate_idempotency_key", hint="wait")
        await c._record_policy_denied(
            "orchestration",
            intent,
            pd,
            action_name="long_running",
        )
        assert c.shared_state.policy_denial_streak.get("long_running:duplicate_idempotency_key") == 1
        await c._handle_delegate("orchestration", _delegate(key="fresh-key"))
        assert not any(k.startswith("long_running:") for k in c.shared_state.policy_denial_streak)
    finally:
        await c.stop()


# PolicyGate denial paths (analysis actions are never LLM-proposable)

import pytest

from hyperloom.orchestrator.roles.agent_role import default_role_registry
from hyperloom.inference_optimizer.protocol.intent import (
    Intent,
    IntentType,
)
from hyperloom.orchestrator.phases.machine_state import (
    PHASE_ALLOWED_ACTIONS,
    PHASE_FRAMEWORK_AGENT,
)
from hyperloom.orchestrator.policy.gate import (
    PolicyDenied,
    PolicyGate,
)
from hyperloom.orchestrator.prompts.prompt_builder import (
    FULL_ENABLED_ACTIONS,
)


@pytest.fixture
def gate() -> PolicyGate:
    """Plain gate without an action catalogue / shared_state."""
    return PolicyGate(role_registry=default_role_registry())


# ``roofline`` and ``profile`` are Coordinator-enqueued; PolicyGate denies any
# propose/delegate/request that names either action.
_INTERNAL_ANALYSIS_ACTIONS = ("roofline", "profile")


@pytest.mark.parametrize("action_name", _INTERNAL_ANALYSIS_ACTIONS)
def test_delegate_with_analysis_action_is_denied(gate, action_name):
    intent = Intent(
        type=IntentType.DELEGATE,
        payload={
            "action_name": action_name,
            "predicted_gain_pct": 1.0,
            "params": {},
        },
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "phase_incompatible"
    assert action_name in str(exc.value)
    assert "Coordinator-managed" in str(exc.value)


@pytest.mark.parametrize("action_name", _INTERNAL_ANALYSIS_ACTIONS)
def test_propose_action_with_analysis_action_is_denied(gate, action_name):
    intent = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": action_name,
            "predicted_gain_pct": 1.0,
        },
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "phase_incompatible"
    assert action_name in str(exc.value)


@pytest.mark.parametrize("action_name", _INTERNAL_ANALYSIS_ACTIONS)
def test_request_with_analysis_kind_is_denied(gate, action_name):
    """A REQUEST whose ``kind`` names roofline/profile is denied by R1 phase_incompatible."""
    intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel_agent",
            "kind": action_name,
        },
    )
    with pytest.raises(PolicyDenied) as exc:
        gate.validate_intent("orchestration", intent)
    assert exc.value.rule == "phase_incompatible"


# Supporting infrastructure parity
def test_phase_explore_allowlist_drops_legacy_actions():
    """The EXPLORE allowlist contains only the canonical action set."""
    assert PHASE_ALLOWED_ACTIONS[PHASE_FRAMEWORK_AGENT] == frozenset(
        {
            "explore",
            "specialist",
            "integrate_patch",
            "roofline",
            "profile",
            "recover",
        }
    )


def test_full_enabled_actions_still_contains_explore():
    """Sanity: ``explore`` / ``baseline`` stay enabled; ``recover`` is intentionally NOT enabled.

    ``sweep`` is absent by design: the concurrency ladder is Coordinator-internal
    and the workload grid it used to fan out over is gone.
    """
    assert "explore" in FULL_ENABLED_ACTIONS
    assert "sweep" not in FULL_ENABLED_ACTIONS
    assert "recover" not in FULL_ENABLED_ACTIONS
    assert "baseline" in FULL_ENABLED_ACTIONS


def test_cli_real_executors_still_contains_explore():
    """Sanity: the canonical EXPLORE-phase executors stay registered."""
    from hyperloom.inference_optimizer.cli.executors import _REAL_EXECUTORS_FULL

    assert "explore" in _REAL_EXECUTORS_FULL
    assert "conc_sweep" in _REAL_EXECUTORS_FULL
    assert "sweep" not in _REAL_EXECUTORS_FULL
    assert "baseline" in _REAL_EXECUTORS_FULL


# Mission-summary + robustness prompt point at explore, not retired names
def test_dead_c_mission_summary_tag_points_at_explore():
    """The ``stack changed`` warning points at ``explore``, not the retired ``validate_stack``."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    s = SharedState(
        baseline_tput=100.0,
        optimization_stack=[{"action": "integrate", "kernel_id": "k1"}],
    )
    text = s.to_mission_summary()
    assert "stack changed" in text
    assert "RUN `explore`" in text
    assert "validate_stack" not in text


def test_mission_summary_surfaces_resume_pending_revalidation():
    from hyperloom.orchestrator.state.shared_state import SharedState

    s = SharedState(
        baseline_tput=100.0,
        optimization_stack=[{"action": "integrate_patch", "variant_name": "p1"}],
        cumulative_gain_validated_stack_len=1,
        resume_pending_revalidation=True,
    )
    text = s.to_mission_summary()
    assert "resume_pending_revalidation=true" in text
    assert "recheck current stack" in text


# The Robustness prune_branch family list used to live in robustness.md, which was
# loaded every tick and discarded by the backend. The role is prompt-driven no
# longer, so the file is gone and test_agent_roles_and_policy asserts it stays gone.
