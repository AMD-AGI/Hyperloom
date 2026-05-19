"""Delegate idempotency + policy-denial ladder tests."""

from __future__ import annotations

import re

import pytest

from inference_optimizer.orchestrator.backends import MockBackend, ScriptedPlan
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.paths import make_session_dir


def _silent_coordinator(session_dir) -> Coordinator:
    silent = ScriptedPlan(turns=[])
    return Coordinator(
        session_dir,
        backends={
            "orchestration": MockBackend(silent, name="o"),
            "kernel": MockBackend(silent, name="k"),
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
            m for m in obs
            if m.payload.get("kind") == "policy_denied"
            and m.payload.get("rule") == "duplicate_idempotency_key_running"
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
    """v0.8 §3.9 — the v0.6 scoreboard's ``locked_reason`` was retired.
    The denial streak is still tracked via
    :attr:`SharedState.policy_denial_streak` (a pure fact); the LLM
    sees it as a count, not a priority lock."""
    c = _silent_coordinator(session_dir)
    try:
        from inference_optimizer.orchestrator.policy import PolicyDenied

        intent = _delegate(action="backends", key="k1")
        pd = PolicyDenied("denied", rule="duplicate_idempotency_key", hint="wait")
        await c._record_policy_denied(
            "orchestration", intent, pd, action_name="backends",
        )
        await c._record_policy_denied(
            "orchestration", intent, pd, action_name="backends",
        )
        streak = c.shared_state.policy_denial_streak.get(
            "backends:duplicate_idempotency_key", 0,
        )
        assert streak >= 2
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_policy_denial_streak_prunes_family_at_five(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        from inference_optimizer.orchestrator.policy import PolicyDenied
        intent = _delegate(action="params", key="k1")
        pd = PolicyDenied("denied", rule="duplicate_idempotency_key", hint="wait")
        for _ in range(5):
            await c._record_policy_denied(
                "orchestration", intent, pd, action_name="params",
            )
        assert "params" in c.shared_state.pruned_families
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_policy_denial_streak_sets_stop_reason_at_ten(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        from inference_optimizer.orchestrator.policy import PolicyDenied
        intent = _delegate(action="backends", key="k1")
        pd = PolicyDenied("denied", rule="duplicate_idempotency_key", hint="wait")
        for _ in range(10):
            await c._record_policy_denied(
                "orchestration", intent, pd, action_name="backends",
            )
        assert c.shared_state.stop_reason == "policy_loop"
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_successful_delegate_resets_policy_denial_streak(session_dir):
    c = _silent_coordinator(session_dir)
    try:
        from inference_optimizer.orchestrator.policy import PolicyDenied
        intent = _delegate(key="k-reset")
        pd = PolicyDenied("denied", rule="duplicate_idempotency_key", hint="wait")
        await c._record_policy_denied(
            "orchestration", intent, pd, action_name="long_running",
        )
        assert c.shared_state.policy_denial_streak.get(
            "long_running:duplicate_idempotency_key"
        ) == 1
        await c._handle_delegate("orchestration", _delegate(key="fresh-key"))
        assert not any(
            k.startswith("long_running:")
            for k in c.shared_state.policy_denial_streak
        )
    finally:
        await c.stop()
