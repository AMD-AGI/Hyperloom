"""P0 end-to-end main-loop demo.

Wires up the Coordinator with all four mock agents and walks through the
single canonical flow that P0 must support:

    1. Orchestration proposes ``baseline``
    2. Critic (mock) auto-approves
    3. Coordinator materializes the approved proposal as a ``baseline`` task
    4. Dispatcher runs the task via the registered ``baseline`` runner
       (a one-line Python lambda for this demo)
    5. Orchestration emits REQUEST{target=kernel, kind=trace_analyze}
    6. Kernel (mock) auto-responds with RESPONSE{kind=trace_analyze_done}
    7. Coordinator routes the response back to Orchestration's inbox
    8. Orchestration acknowledges via send_message

Robustness ticks heartbeats throughout. No real LLMs, no real GPU work,
no network. Run:

    python -m inference_optimizer.examples.p0_main_loop

Output: a sequence of bullet-pointed events ending with the final state.
"""

from __future__ import annotations

import asyncio

from ..orchestrator.backends import (
    MockBackend,
    MockCriticBackend,
    MockKernelBackend,
    MockRobustnessBackend,
    MockTurn,
    ScriptedPlan,
)
from ..orchestrator.coordinator import Coordinator
from ..orchestrator.intent_parser import Intent, IntentType
from ..paths import make_session_dir


def _orchestration_plan() -> ScriptedPlan:
    """Three scripted turns: propose → request kernel → done observation."""
    return ScriptedPlan(turns=[
        # Turn 1 — propose baseline
        MockTurn(intents=[
            Intent(type=IntentType.PROPOSE_ACTION, payload={
                "action_name": "baseline",
                "predicted_gain_pct": 0.0,
                "reason": "establish current tput",
            }),
        ]),
        # Turn 2 — silent (waiting for Critic verdict + dispatch result)
        MockTurn(intents=[Intent(type=IntentType.SEND_MESSAGE,
                                  payload={"topic": "heartbeat", "body_md": "waiting for verdict"})]),
        # Turn 3 — REQUEST kernel
        MockTurn(intents=[
            Intent(type=IntentType.REQUEST, payload={
                "target_agent": "kernel",
                "kind": "trace_analyze",
                "params": {"top_k": 5},
                "reason": "kick off Plan A kernel optimization",
            }),
        ]),
        # Turn 4+ — silent heartbeats
    ], default_intent=Intent(type=IntentType.SEND_MESSAGE,
                              payload={"topic": "heartbeat", "body_md": "ok"}))


async def _baseline_executor(ctx) -> dict:
    """Pretend to run a baseline benchmark."""
    return {"tput_tok_per_s_per_gpu": 1840.0, "p99_latency_ms": 152, "source": "mock"}


async def _run_demo(ticks: int = 6) -> dict:
    session_dir = make_session_dir()

    backends = {
        "orchestration": MockBackend(_orchestration_plan(), name="orchestration"),
        "kernel":        MockKernelBackend(),
        "critic":        MockCriticBackend(),
        "robustness":    MockRobustnessBackend(),
    }

    coordinator = Coordinator(session_dir, backends=backends)
    coordinator.sub.register_executor("baseline", _baseline_executor)

    print(f"Session dir: {session_dir}")
    print(f"DB:          {coordinator.db.db_path}")
    print(f"Running {ticks} ticks across 4 agents...")
    print()

    try:
        for tick_n in range(ticks):
            await coordinator.tick(1)
            counts = await _summarize_bus(coordinator)
            print(f"  [tick {tick_n + 1}] events: {counts}")

        # Pull a few highlight events
        print()
        print("---- highlights ----")
        proposals = await coordinator.bus.tail(topic="proposal")
        for p in proposals:
            print(f"  proposal from={p.from_agent} action={p.payload.get('action_name')}")
        verdicts = await coordinator.bus.tail(topic="review_verdict")
        for v in verdicts:
            print(f"  verdict source={v.from_agent} -> {v.payload.get('verdict')}")
        decisions = await coordinator.bus.tail(topic="decision")
        for d in decisions:
            print(f"  decision kind={d.payload.get('kind')} action={d.payload.get('action_name')}")
        results = await coordinator.bus.tail(topic="delegated_result")
        for r in results:
            print(f"  delegated_result task={r.payload.get('task_id')[:8]} state={r.payload.get('state')} result={r.payload.get('result')}")
        responses = await coordinator.bus.tail(topic="response")
        for r in responses:
            print(f"  response from={r.from_agent} kind={r.payload.get('kind')} status={r.payload.get('status')}")

        return {
            "session_dir": str(session_dir),
            "proposals_seen": len(proposals),
            "verdicts_seen": len(verdicts),
            "decisions_seen": len(decisions),
            "delegated_results_seen": len(results),
            "responses_seen": len(responses),
        }
    finally:
        await coordinator.stop()


async def _summarize_bus(coordinator: Coordinator) -> dict[str, int]:
    out: dict[str, int] = {}
    for topic in ("proposal", "review_verdict", "decision", "delegated_result",
                   "request", "response", "heartbeat", "alert"):
        msgs = await coordinator.bus.tail(topic=topic, n=100)
        if msgs:
            out[topic] = len(msgs)
    return out


def main() -> None:
    summary = asyncio.run(_run_demo())
    print()
    print("---- summary ----")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
