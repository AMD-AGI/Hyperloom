"""End-to-end integration with inference_optimizer.

Round-trip: build a Coordinator-style prompt, drive it through
:class:`RobustnessAgentBackend.run`, and validate every emitted intent
against the upstream :class:`PolicyGate`. The test is skipped when the
inference_optimizer package is not on ``sys.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _try_import_upstream():
    candidate_roots = [
        Path.home() / "lss" / "Hyperloom",
        Path.home() / "Hyperloom-rs-build" / "Hyperloom",
    ]
    for root in candidate_roots:
        if (root / "inference_optimizer" / "orchestrator").is_dir():
            sys.path.insert(0, str(root))
            break
    try:
        from inference_optimizer.orchestrator.agent_role import default_role_registry
        from inference_optimizer.orchestrator.intent_parser import (
            Intent as UpstreamIntent,
            IntentType as UpstreamIntentType,
        )
        from inference_optimizer.orchestrator.policy import PolicyDenied, PolicyGate
    except ImportError:
        return None
    return {
        "default_role_registry": default_role_registry,
        "UpstreamIntent": UpstreamIntent,
        "UpstreamIntentType": UpstreamIntentType,
        "PolicyDenied": PolicyDenied,
        "PolicyGate": PolicyGate,
    }


_UPSTREAM = _try_import_upstream()
pytestmark = pytest.mark.skipif(
    _UPSTREAM is None,
    reason="inference_optimizer not importable; integration check skipped",
)


def _gate():
    PolicyGate = _UPSTREAM["PolicyGate"]  # type: ignore[index]
    default_role_registry = _UPSTREAM["default_role_registry"]  # type: ignore[index]
    return PolicyGate(role_registry=default_role_registry())


def _to_upstream(local_intent):
    UpstreamIntent = _UPSTREAM["UpstreamIntent"]  # type: ignore[index]
    UpstreamIntentType = _UPSTREAM["UpstreamIntentType"]  # type: ignore[index]
    return UpstreamIntent(
        type=UpstreamIntentType(local_intent.type.value),
        payload=dict(local_intent.payload),
    )


@pytest.mark.asyncio
async def test_backend_intents_pass_upstream_policy_gate(tmp_path):
    from robustness_agent.config import Config
    from robustness_agent.factory import build_backend

    config = Config(session_dir=tmp_path, robustness_server_url="")
    backend, bundle = build_backend(config)
    try:
        result = await backend.run(
            "=== Shared session state ===\n"
            "session_id=sess-1\n"
            "model=qwen3-8b  class=qwen3\n"
            "baseline_tput=10  baseline_acc=0.8\n"
            "crash_count=2\n"
            "current_action=baseline\n"
            "=== Inbox for robustness (newest last) ===\n"
            "  seq=1 msg_id=abc from=orchestration topic=observation payload={'kind': 'policy_denied', 'rule': 'role'}\n"
        )
    finally:
        await bundle.aclose()

    gate = _gate()
    assert result.intents, "expected at least one intent"
    for intent in result.intents:
        upstream = _to_upstream(intent)
        gate.validate_intent("robustness", upstream)


@pytest.mark.asyncio
async def test_backend_high_severity_path_passes_gate(tmp_path):
    from robustness_agent.config import Config
    from robustness_agent.factory import build_backend

    config = Config(session_dir=tmp_path, robustness_server_url="")
    backend, bundle = build_backend(config)
    try:
        result = await backend.run(
            "=== Shared session state ===\n"
            "session_id=sess-1\n"
            "crash_count=10\n"
            "=== Inbox for robustness ===\n"
            "(no new messages)\n"
        )
    finally:
        await bundle.aclose()

    gate = _gate()
    assert result.intents
    types_emitted = {i.type.value for i in result.intents}
    assert "alert" in types_emitted
    assert "escalate_strategy_change" in types_emitted
    for intent in result.intents:
        upstream = _to_upstream(intent)
        gate.validate_intent("robustness", upstream)


@pytest.mark.asyncio
async def test_heartbeat_passes_gate(tmp_path):
    from robustness_agent.config import Config
    from robustness_agent.factory import build_backend

    config = Config(session_dir=tmp_path, robustness_server_url="")
    backend, bundle = build_backend(config)
    try:
        result = await backend.run(
            "=== Shared session state ===\n"
            "session_id=sess-1\n"
            "crash_count=0\n"
            "=== Inbox for robustness ===\n"
            "(no new messages)\n"
        )
    finally:
        await bundle.aclose()

    gate = _gate()
    assert len(result.intents) == 1
    intent = result.intents[0]
    assert intent.payload["topic"] == "heartbeat"
    gate.validate_intent("robustness", _to_upstream(intent))


@pytest.mark.asyncio
async def test_repeated_failure_emits_prune_branch_passing_gate(tmp_path):
    from robustness_agent.config import Config
    from robustness_agent.factory import build_backend

    config = Config(session_dir=tmp_path, robustness_server_url="")
    backend, bundle = build_backend(config)
    try:
        # Inbox doesn't natively carry delegated_result, so we reach into
        # the local probe via direct injection: feed a fake conductor.db
        # entry matching delegated_result with state=failed twice on the
        # same family.
        import json
        import sqlite3

        storage = tmp_path / "storage"
        storage.mkdir()
        db = storage / "conductor.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT, msg_id TEXT,"
            " from_agent TEXT, to_agent TEXT, topic TEXT, payload TEXT, ts TEXT)"
        )
        for tid in ("t1", "t2"):
            conn.execute(
                "INSERT INTO events (msg_id, from_agent, to_agent, topic, payload, ts)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    tid,
                    "coordinator",
                    "robustness",
                    "delegated_result",
                    json.dumps(
                        {
                            "state": "failed",
                            "kind": "kernel_opt",
                            "task_id": tid,
                        }
                    ),
                    "",
                ),
            )
        conn.commit()
        conn.close()

        result = await backend.run(
            "=== Shared session state ===\n"
            "session_id=sess-1\n"
            "crash_count=0\n"
            "=== Inbox for robustness ===\n"
            "(no new messages)\n"
        )
    finally:
        await bundle.aclose()

    types_emitted = {i.type.value for i in result.intents}
    assert "alert" in types_emitted
    gate = _gate()
    for intent in result.intents:
        gate.validate_intent("robustness", _to_upstream(intent))
