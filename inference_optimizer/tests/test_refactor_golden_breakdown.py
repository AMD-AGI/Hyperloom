# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Refactor safety net: golden behavioral baseline for the Coordinator.

This test drives a *fixed, deterministic* sequence of intents through a real
``Coordinator`` (with mock backends) and snapshots the resulting observable
behavior — the persisted ``SharedState`` and the decision/result message
topics on the bus. The snapshot is captured once as a golden fixture and
asserted byte-for-byte (after normalizing non-deterministic fields) on every
subsequent run.

Why this exists
---------------
The structural refactor (see the "break four God-objects" plan) extracts the
Coordinator's ``_handle_*`` intent routing and ``record_*`` state mutations into
collaborator objects (``IntentRouter``, ``ResultRecorder``) and moves behavior
off ``SharedState``. Those moves must be *behavior-preserving*. This test is the
gate: if an extraction changes what the Coordinator does — different state,
different bus decisions — the golden mismatch fails the build before the change
can land.

It deliberately exercises the exact methods the refactor touches:
``_handle_intent`` → ``_handle_propose_action`` / ``_handle_review_verdict`` /
``_handle_delegate``, plus the ``SharedState`` mutations they trigger.

Regenerating the golden
-----------------------
Set ``REFRESH_GOLDEN=1`` to rewrite the golden file from current behavior. Only
do this when a behavior change is *intended* and reviewed — never to silence an
unexpected diff during a refactor.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.backends import (
    Backend,
    MockBackend,
    MockTurn,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.protocol.intent import Intent, IntentType
from inference_optimizer.paths import make_session_dir


GOLDEN_PATH = Path(__file__).parent / "golden" / "coordinator_behavior_golden.json"


# --------------------------------------------------------------------------- #
# Backend scaffolding (mirrors tests/test_coordinator_runtime.py)
# --------------------------------------------------------------------------- #
def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE,
                  payload={"topic": "heartbeat", "body_md": "ok"})


def _silent_plan() -> ScriptedPlan:
    return ScriptedPlan(turns=[], default_intent=_heartbeat())


def _build_backends(scripts: dict[str, ScriptedPlan]) -> dict[str, Backend]:
    backends: dict[str, Backend] = {}
    for name in ("orchestration", "kernel", "critic", "robustness"):
        backends[name] = MockBackend(scripts.get(name, _silent_plan()), name=name)
    return backends


async def _async_return(value: Any):
    return value


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    sd = make_session_dir()
    from .conftest import seed_target_analysis_marker
    seed_target_analysis_marker(sd)
    return sd


# --------------------------------------------------------------------------- #
# Normalization: strip fields that legitimately vary run-to-run
# --------------------------------------------------------------------------- #
# Any key whose name matches one of these is replaced with a stable sentinel
# wherever it appears (at any depth). These are timestamps, ids, paths, and
# host/pid facts — none of which encode Coordinator *behavior*.
_VOLATILE_KEY_SUFFIXES = (
    "_ts", "_at", "_at_utc", "_unix", "_msg_id", "_id",
)
_VOLATILE_KEY_EXACT = frozenset({
    "ts", "created_at", "start_ts", "session_id", "claw_session_id",
    "session_dir", "msg_id", "in_reply_to", "seq", "pid", "host",
    "workspace", "baseline_config_path", "model_path", "code_revision",
    "target_proposal_msg_id", "idempotency_key", "proposal_id",
    "task_id", "proposal_msg_id",
})


def _is_volatile_key(key: str) -> bool:
    if key in _VOLATILE_KEY_EXACT:
        return True
    return any(key.endswith(suf) for suf in _VOLATILE_KEY_SUFFIXES)


def _normalize(obj: Any) -> Any:
    """Recursively replace volatile values with a stable sentinel.

    Preserves *presence* and *type-shape* (so a refactor that drops a field
    still fails) while ignoring run-to-run noise in the value itself.
    """
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in sorted(obj.items()):
            if _is_volatile_key(k):
                out[k] = "<volatile>" if v is not None else None
            else:
                out[k] = _normalize(v)
        return out
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    if isinstance(obj, str):
        # Absolute paths under the tmp session dir leak into many fields.
        if obj.startswith("/") and ("/runs/" in obj or "/sessions/" in obj):
            return "<path>"
        return obj
    return obj


def _message_digest(messages: list[Any]) -> list[dict[str, Any]]:
    """Reduce bus messages to the behavior-bearing fields, normalized."""
    rows = []
    for m in messages:
        rows.append(_normalize({
            "from_agent": m.from_agent,
            "to_agent": m.to_agent,
            "topic": m.topic,
            "payload": m.payload,
        }))
    # Stable order independent of seq/ts: sort by serialized content.
    rows.sort(key=lambda r: json.dumps(r, sort_keys=True))
    return rows


# --------------------------------------------------------------------------- #
# The deterministic scenario the golden locks down
# --------------------------------------------------------------------------- #
async def _drive_scenario(session_dir: Path) -> dict[str, Any]:
    """Run a fixed propose -> approve -> delegate flow; return observable state.

    Exercises _handle_intent dispatch into _handle_propose_action,
    _handle_review_verdict, and _handle_delegate, plus the SharedState
    mutations and bus decisions they produce.
    """
    propose = Intent(type=IntentType.PROPOSE_ACTION, payload={
        "action_name": "baseline", "predicted_gain_pct": 0.0,
    })
    plans = {"orchestration": ScriptedPlan(turns=[MockTurn(intents=[propose])])}
    c = Coordinator(session_dir, backends=_build_backends(plans))
    c.sub.register_executor("baseline", lambda ctx: _async_return({
        "status": "succeeded", "tput": 1840.0,
    }))
    try:
        # Tick 1: orchestration proposes baseline -> pending proposal created.
        await c.tick(1)
        proposal_id = next(iter(c.state.pending_proposals.keys()))

        # Critic approves -> coordinator materializes + emits decision.
        await c._handle_intent("critic", Intent(
            type=IntentType.REVIEW_VERDICT, payload={
                "target_proposal_msg_id": proposal_id,
                "verdict": "approve",
                "reasoning": "golden-baseline",
            },
        ))

        # Explicit delegate -> dispatcher runs the registered executor.
        await c._handle_intent("orchestration", Intent(
            type=IntentType.DELEGATE, payload={
                "action_name": "baseline", "params": {"runs": 1},
                "idempotency_key": "golden-deleg-1",
            },
        ))
        # Let the dispatcher drain the delegated task.
        await c.tick(1)

        decisions = await c.bus.tail(n=200, topic="decision")
        results = await c.bus.tail(n=200, topic="delegated_result")
        verdicts = await c.bus.tail(n=200, topic="review_verdict")

        return {
            "shared_state": _normalize(c.shared_state.to_dict()),
            "bus_decisions": _message_digest(decisions),
            "bus_delegated_result": _message_digest(results),
            "bus_review_verdict": _message_digest(verdicts),
        }
    finally:
        await c.stop()


# --------------------------------------------------------------------------- #
# The test
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_coordinator_behavior_matches_golden(session_dir: Path) -> None:
    observed = await _drive_scenario(session_dir)

    if os.environ.get("REFRESH_GOLDEN") == "1" or not GOLDEN_PATH.exists():
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(
            json.dumps(observed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if os.environ.get("REFRESH_GOLDEN") == "1":
            pytest.skip("Golden refreshed (REFRESH_GOLDEN=1)")
        # First-ever run: golden seeded, nothing to compare against yet.
        pytest.skip(f"Golden baseline created at {GOLDEN_PATH}")

    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    # Compare section by section for a readable failure.
    for section in sorted(set(expected) | set(observed)):
        assert section in observed, f"refactor dropped golden section: {section}"
        assert section in expected, f"refactor added new section: {section}"
        assert observed[section] == expected[section], (
            f"behavior drift in '{section}' — a refactor changed Coordinator "
            f"behavior. If intended, regenerate with REFRESH_GOLDEN=1 and "
            f"review the diff."
        )
