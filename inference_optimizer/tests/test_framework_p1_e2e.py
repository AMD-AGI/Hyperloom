"""P1 mock end-to-end: orchestration REQUEST framework_optimize ->
handler -> orchestration RESPONSE -> re-propose framework_integrate ->
handler -> KEEP verdict.

Hermetic. Drives the Coordinator with scripted backends so no LLM /
subprocess / GPU is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.backends import (
    FrameworkMockBackend,
    MockBackend,
    MockTurn,
    ScriptedPlan,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.intent_parser import Intent, IntentType
from inference_optimizer.paths import make_session_dir


def _heartbeat() -> Intent:
    return Intent(
        type=IntentType.SEND_MESSAGE,
        payload={"topic": "heartbeat", "body_md": "ok"},
    )


def _request(kind: str, msg_id: str = "test-req") -> Intent:
    return Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "framework",
            "kind": kind,
            "idempotency_key": msg_id,
        },
    )


@pytest.fixture
def session_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


# ---------------------------------------------------------------------------
# Coordinator + framework backend wiring
# ---------------------------------------------------------------------------
def _build_5role_backends(session_dir: Path) -> dict[str, object]:
    """4 silent + 1 framework_mock backend. Framework role is included
    in role_registry because backend dict contains 'framework'."""
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(silent, name="o"),
        "kernel":        MockBackend(silent, name="k"),
        "critic":        MockBackend(silent, name="c"),
        "robustness":    MockBackend(silent, name="r"),
        "framework":     FrameworkMockBackend(session_dir=session_dir),
    }


def test_coordinator_with_framework_backend_includes_framework_role(session_dir):
    backends = _build_5role_backends(session_dir)
    c = Coordinator(session_dir, backends=backends)
    try:
        assert "framework" in c.role_registry
        # Tick order honours canonical 5-tuple.
        assert c._tick_roles == (
            "orchestration", "kernel", "framework", "critic", "robustness",
        )
    finally:
        # No async tasks were started -- direct stop is fine. Use a
        # fresh loop because pytest-asyncio may have closed the default
        # one in earlier test sessions.
        import asyncio
        asyncio.new_event_loop().run_until_complete(c.stop())


def test_coordinator_without_framework_backend_drops_framework_role(
    session_dir,
):
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    backends_no_fw = {
        "orchestration": MockBackend(silent, name="o"),
        "kernel":        MockBackend(silent, name="k"),
        "critic":        MockBackend(silent, name="c"),
        "robustness":    MockBackend(silent, name="r"),
    }
    c = Coordinator(session_dir, backends=backends_no_fw)
    try:
        assert "framework" not in c.role_registry
        assert "framework" not in c._tick_roles
    finally:
        import asyncio
        asyncio.new_event_loop().run_until_complete(c.stop())


# ---------------------------------------------------------------------------
# Framework REQUEST handler path -- the actual P1 e2e
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_framework_optimize_request_produces_response(session_dir):
    """When orchestration emits a REQUEST{target_agent='framework',
    kind='framework_optimize'} intent, the Coordinator's _handle_request
    runs the programmatic handler and a RESPONSE lands on the bus
    addressed back to orchestration."""
    silent_plan = ScriptedPlan(turns=[], default_intent=_heartbeat())
    # Orchestration scripts one REQUEST intent on its first turn then
    # falls back to heartbeats. The other roles stay silent.
    orch_plan = ScriptedPlan(
        turns=[MockTurn(intents=[_request("framework_optimize")])],
        default_intent=_heartbeat(),
    )
    backends = {
        "orchestration": MockBackend(orch_plan, name="o"),
        "kernel":        MockBackend(silent_plan, name="k"),
        "critic":        MockBackend(silent_plan, name="c"),
        "robustness":    MockBackend(silent_plan, name="r"),
        "framework":     FrameworkMockBackend(session_dir=session_dir),
    }
    c = Coordinator(session_dir, backends=backends)
    try:
        await c.tick(3)

        # Look for a RESPONSE on the bus with kind=framework_optimize_done.
        msgs = await c.bus.tail(n=50)
        responses = [
            m for m in msgs
            if (m.topic == "response"
                and m.from_agent == "framework"
                and m.to_agent == "orchestration")
        ]
        assert len(responses) >= 1, (
            f"expected at least one framework->orchestration RESPONSE, "
            f"got {[(m.from_agent, m.to_agent, m.topic) for m in msgs]}"
        )
        rsp = responses[-1]
        result = (rsp.payload or {}).get("result") or {}
        assert (rsp.payload or {}).get("kind") == "framework_optimize_done"
        assert result.get("payload_kind") == "OptimizeSuccess"
        assert result.get("predicted_gain_pct", 0.0) >= 3.0
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_framework_integrate_request_produces_keep_response(
    session_dir,
):
    """Same routing flow for framework_integrate -- handler returns KEEP."""
    silent_plan = ScriptedPlan(turns=[], default_intent=_heartbeat())
    integrate_intent = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "framework",
            "kind": "framework_integrate",
            "patch_id": "fw-e2e-001",
            "idempotency_key": "test-fw-int-1",
        },
    )
    orch_plan = ScriptedPlan(
        turns=[MockTurn(intents=[integrate_intent])],
        default_intent=_heartbeat(),
    )
    backends = {
        "orchestration": MockBackend(orch_plan, name="o"),
        "kernel":        MockBackend(silent_plan, name="k"),
        "critic":        MockBackend(silent_plan, name="c"),
        "robustness":    MockBackend(silent_plan, name="r"),
        "framework":     FrameworkMockBackend(session_dir=session_dir),
    }
    c = Coordinator(session_dir, backends=backends)
    try:
        await c.tick(3)
        msgs = await c.bus.tail(n=50)
        responses = [
            m for m in msgs
            if (m.topic == "response"
                and m.from_agent == "framework"
                and m.to_agent == "orchestration")
        ]
        assert len(responses) >= 1
        rsp = responses[-1]
        result = (rsp.payload or {}).get("result") or {}
        assert result.get("payload_kind") == "IntegrateSuccess"
        assert result.get("verdict") == "KEEP"
        assert result.get("patch_id") == "fw-e2e-001"
    finally:
        await c.stop()
