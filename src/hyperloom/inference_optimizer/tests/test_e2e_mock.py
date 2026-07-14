# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Mock end-to-end Coordinator loop tests."""

from __future__ import annotations

import re

import pytest
from hyperloom.orchestrator.roles import (
    BackendTurnResult,
    MockBackend,
    MockCriticBackend,
    MockRobustnessBackend,
    MockTurn,
    ScriptedPlan,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


_REQUEST_RE = re.compile(r"msg_id=([a-f0-9]+)\s+from=(\w+)\s+topic=request\s+payload=(.*)$", re.MULTILINE)
_KIND_RE = re.compile(r"['\"]kind['\"]\s*:\s*['\"]([\w-]+)['\"]")


class _KernelResponderBackend:
    """Minimal test responder for Coordinator request routing."""

    name = "kernel-test-responder"

    def __init__(self):
        self.calls: list[dict[str, object]] = []
        self._answered_msg_ids: set[str] = set()

    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        max_turns: int = 1,
    ) -> BackendTurnResult:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt, "tools": tools, "max_turns": max_turns})
        intents: list[Intent] = []
        for match in _REQUEST_RE.finditer(prompt):
            msg_id, _from_agent, raw_payload = match.groups()
            if msg_id in self._answered_msg_ids:
                continue
            kind_match = _KIND_RE.search(raw_payload)
            kind = kind_match.group(1) if kind_match else "unknown"
            self._answered_msg_ids.add(msg_id)
            intents.append(
                Intent(
                    type=IntentType.RESPONSE,
                    payload={
                        "in_reply_to": msg_id,
                        "kind": f"{kind}_done",
                        "status": "ok",
                        "result": {"source": "test", "chosen": ["mock_kernel_1"]},
                    },
                )
            )
        return BackendTurnResult(intents=intents or [_heartbeat()], raw_text="(kernel test responder)")


# Full 4-agent main-loop e2e
@pytest.mark.asyncio
async def test_e2e_propose_approve_dispatch_with_mock_executor(session_dir):
    """Orchestration → Critic (mock) → dispatcher → succeeded."""
    propose = Intent(
        type=IntentType.PROPOSE_ACTION,
        payload={
            "action_name": "baseline",
            "predicted_gain_pct": 0.0,
        },
    )
    backends = {
        "orchestration": MockBackend(
            ScriptedPlan(turns=[MockTurn(intents=[propose])], default_intent=_heartbeat()),
            name="orchestration",
        ),
        "kernel_agent": MockBackend(ScriptedPlan(turns=[], default_intent=_heartbeat()), name="kernel_agent"),
        "critic": MockCriticBackend(),
        "robustness": MockRobustnessBackend(),
    }
    c = Coordinator(session_dir, backends=backends)
    c.sub.register_executor("baseline", lambda ctx: _async_value({"tput": 1840}))
    try:
        # tick 1 propose, tick 2 approve+materialize, tick 3 dispatch.
        await c.tick(3)

        proposals = await c.bus.tail(topic="proposal")
        verdicts = await c.bus.tail(topic="review_verdict")
        decisions = await c.bus.tail(topic="decision")
        results = await c.bus.tail(topic="delegated_result")

        assert len(proposals) >= 1
        assert any(v.payload.get("verdict") == "approve" for v in verdicts)
        assert any(d.payload.get("kind") == "approved_proposal" for d in decisions)
        assert any(
            r.payload.get("state") == "succeeded" and r.payload.get("result", {}).get("tput") == 1840 for r in results
        )
    finally:
        await c.stop()


async def _async_value(v):
    return v


@pytest.mark.asyncio
async def test_e2e_request_response_round_trip(session_dir):
    """Plan A: orchestration REQUEST → kernel RESPONSE → routed back."""
    req = Intent(
        type=IntentType.REQUEST,
        payload={
            "target_agent": "kernel_agent",
            "kind": "explore_options",
            "params": {"top_k": 5},
        },
    )
    backends = {
        "orchestration": MockBackend(
            ScriptedPlan(turns=[MockTurn(intents=[req])], default_intent=_heartbeat()),
            name="orchestration",
        ),
        "kernel_agent": _KernelResponderBackend(),
        "critic": MockCriticBackend(),
        "robustness": MockRobustnessBackend(),
    }
    c = Coordinator(session_dir, backends=backends)
    try:
        # tick 1 mirror REQUEST to kernel, tick 2 RESPONSE routed back.
        await c.tick(2)

        kernel_inbox = await c.bus.tail(to_agent="kernel_agent", topic="request")
        assert kernel_inbox

        responses = await c.bus.tail(to_agent="orchestration", topic="response")
        assert responses
        r = responses[0]
        assert r.from_agent == "kernel_agent"
        assert r.payload["status"] == "ok"
        assert r.payload["kind"] == "explore_options_done"
        assert r.payload["result"]["chosen"] == ["mock_kernel_1"]
    finally:
        await c.stop()


@pytest.mark.asyncio
async def test_e2e_robustness_heartbeats_throughout(session_dir):
    backends = {
        "orchestration": MockBackend(ScriptedPlan(turns=[], default_intent=_heartbeat()), name="orchestration"),
        "kernel_agent": MockBackend(ScriptedPlan(turns=[], default_intent=_heartbeat()), name="kernel_agent"),
        "critic": MockCriticBackend(),
        "robustness": MockRobustnessBackend(),
    }
    c = Coordinator(session_dir, backends=backends)
    try:
        await c.tick(4)
        beats = await c.bus.tail(topic="heartbeat", n=100)
        robustness_beats = [b for b in beats if b.from_agent == "robustness"]
        assert len(robustness_beats) == 4
    finally:
        await c.stop()
