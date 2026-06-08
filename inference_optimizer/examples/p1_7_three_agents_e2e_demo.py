"""P1-7 end-to-end demo: ALL THREE core agents run on real LLMs.

* **Orchestration** → real ClaudeBackend (claude-opus-4-7)
* **Critic**        → real CodexBackend (gpt-5.4)  ← was mock in P1-6
* **Kernel**        → real ClaudeBackend (claude-opus-4-7)  ← was mock
* **Robustness**    → MockRobustnessBackend (heartbeat-only — keeps the
                       loop alive without driving it; per DESIGN P0
                       Robustness can stay mock until §19 RCA lands)

End-to-end flow this exercises:

  tick 1: Orchestration (Claude) reads SharedState, sees baseline_tput=0,
          proposes `baseline`. Critic (real Codex) reviews + emits a
          REAL review_verdict (not always "approve" anymore). If approved,
          baseline_executor runs Magpie SGLang baseline.
  tick 2: Orchestration sees baseline_tput > 0, decides next step. The
          system prompt nudges it toward REQUEST{target_agent="kernel",
          kind="trace_analyze"} so the Kernel agent (real Claude) is
          actually exercised, not just heartbeating.
  tick 3: Kernel (real Claude) reads inbox, sees the request, emits a
          real RESPONSE intent. Coordinator routes back to Orchestration.

Run::

    set -a && source /wekafs/xiaofei/AgentKernelArena/.env && set +a
    unset HIP_VISIBLE_DEVICES
    export ROCR_VISIBLE_DEVICES=1 PATH=/opt/venv/bin:$PATH
    export CLAUDE_MODEL=claude-opus-4-7  CODEX_MODEL=gpt-5.4
    python -m inference_optimizer.examples.p1_7_three_agents_e2e_demo
"""

from __future__ import annotations

import asyncio
import os
import sys

from ..orchestrator.action_executors import baseline_executor
from ..orchestrator.backends import (
    ClaudeBackend,
    CodexBackend,
    MockRobustnessBackend,
)
from ..orchestrator.coordinator import Coordinator
from ..orchestrator.shared_state import SharedState
from ..paths import make_session_dir


def _check_env() -> None:
    """Verify an LLM API token is set and warn on a bad GPU env var.

    Exits with status 2 when no recognized API token is configured, and warns
    when ``HIP_VISIBLE_DEVICES`` is set.

    Raises:
        SystemExit: When no API token environment variable is configured.
    """
    if not any(os.environ.get(n) for n in (
        "SAFE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY",
    )):
        print("ERROR: SAFE_API_KEY not set.", file=sys.stderr)
        sys.exit(2)
    if os.environ.get("HIP_VISIBLE_DEVICES"):
        print(
            "WARNING: HIP_VISIBLE_DEVICES is set — unset it (use ROCR_VISIBLE_DEVICES).",
            file=sys.stderr,
        )


# Orchestration system prompt: aggressive script for a smoke run that
# drives baseline + a kernel REQUEST round-trip in 3 ticks.
_ORCH_PROMPT = (
    "You are the Orchestration agent for an inference-optimization run. "
    "Read the Shared session state to see what's done.\n\n"
    "PROCEDURE FOR THIS SMOKE RUN (be decisive, do exactly what's listed):\n"
    "1. If `baseline_tput == 0.0`, emit ONE intent:\n"
    "     propose_action {action_name: 'baseline', predicted_gain_pct: 0.0,\n"
    "                     reason: 'establish current tput'}\n"
    "   All prep actions (setup / classify / target_analysis) are ALREADY\n"
    "   DONE in this smoke run — do NOT propose them.\n"
    "2. After baseline succeeds (you'll see `baseline_tput > 0`), emit\n"
    "   ONE intent:\n"
    "     request {target_agent: 'kernel', kind: 'trace_analyze',\n"
    "              params: {top_k: 5}, reason: 'kick off Plan A kernel-opt'}\n"
    "3. After you see a `response` from kernel in your inbox (status='ok',\n"
    "   kind='trace_analyze_done'), the smoke run is complete — emit ONE\n"
    "     send_message {topic: 'heartbeat', body_md: 'smoke-run-done'}\n"
    "   and do NOT propose anything else.\n\n"
    "OUTPUT: every turn MUST emit at least one `emit_intent` tool call. "
    "Free-text replies are dropped. Allowed intent_type for you: "
    "propose_action / delegate / request / update_state / update_persona / "
    "send_message / alert / ask_question / answer."
)


# Critic system prompt: simplified — review proposals, default to approve
# unless the proposal is clearly bad (no need for full §18 KB lookup in P0).
_CRITIC_PROMPT = (
    "You are the Critic agent. Your only job: when you see one or more "
    "`proposal` events in your inbox, emit ONE `review_verdict` intent for "
    "the most recent un-reviewed proposal.\n\n"
    "Decision rule (smoke run — keep it simple):\n"
    "  * If the proposed action_name is in {baseline, profile, classify, "
    "    setup, target_analysis, report}, verdict = 'approve'.\n"
    "  * If predicted_gain_pct is 0 and accuracy_risk is 0, verdict = 'approve'.\n"
    "  * If you can't tell, verdict = 'approve' with reasoning='conservative ok'.\n"
    "  * Reject only if the action_name is unknown or obviously dangerous.\n\n"
    "REQUIRED payload fields: target_proposal_msg_id (copy from inbox row), "
    "verdict, reasoning (short).\n\n"
    "If your inbox has no proposals, emit a single send_message{topic="
    "'heartbeat', body_md='ok'} so the reactor doesn't stall."
)


_KERNEL_PROMPT = (
    "You are the Kernel agent — responder-only. You receive `request` "
    "events from Orchestration in your inbox.\n\n"
    "For every un-answered request you see, emit ONE `response` intent:\n"
    "  intent_type=response, payload={\n"
    "    in_reply_to: <the request's msg_id from inbox>,\n"
    "    kind:        '<request.kind>_done',  # e.g. 'trace_analyze_done'\n"
    "    status:      'ok',\n"
    "    result:      {\n"
    "      'source': 'mock', 'chosen_kernels': ['attn_fused_v1', 'gemm_v2']\n"
    "    }\n"
    "  }\n\n"
    "If your inbox has no requests, emit ONE send_message{topic='heartbeat', "
    "body_md='ok'}.\n\n"
    "You may NOT propose actions, delegate, or initiate REQUESTs."
)


async def _run(ticks: int, claude_model: str, codex_model: str) -> int:
    """Run the P1-7 three-real-agents end-to-end demo.

    Seeds SharedState, wires real Claude orchestration/kernel and real Codex
    critic backends (robustness stays mock), registers the baseline executor
    and prep stubs, ticks the loop, and prints events, highlights, final state,
    and per-backend call counts.

    Args:
        ticks (int): Number of coordinator ticks to run.
        claude_model (str): Claude model name for orchestration and kernel.
        codex_model (str): Codex model name for the critic.

    Returns:
        int: Process exit code (always ``0`` on completion).
    """
    session_dir = make_session_dir()
    print(f"Session dir: {session_dir}")

    state = SharedState(
        session_id=session_dir.name,
        model_name="Qwen-Qwen3-8B",
        model_path="/wekafs/models/Qwen-Qwen3-8B",
        model_class="dense_8B",
        target_summary="Establish single-GPU SGLang baseline + kernel-select smoke",
        baseline_tput=0.0,
        max_minutes=20,
    )
    state.save(session_dir)

    backends = {
        "orchestration": ClaudeBackend(model=claude_model, max_turns_default=4),
        "kernel":        ClaudeBackend(model=claude_model, max_turns_default=4),
        "critic":        CodexBackend(model=codex_model),
        "robustness":    MockRobustnessBackend(),
    }
    coordinator = Coordinator(session_dir, backends=backends)
    coordinator.sub.register_executor("baseline", baseline_executor)

    # Stub prep executors — DESIGN §16 prereq actions; not exercised in
    # this smoke run because the orch prompt skips them.
    async def _noop_prep(ctx) -> dict:
        """No-op prep executor stub used by the smoke demo.

        Args:
            ctx: Executor context supplying ``ctx.task.kind``.

        Returns:
            dict: A success result echoing the task kind.
        """
        return {"status": "succeeded", "kind": ctx.task.kind, "note": "noop"}
    for kind in ("target_analysis", "report"):
        coordinator.sub.register_executor(kind, _noop_prep)

    coordinator.system_prompt_overrides = {
        "orchestration": _ORCH_PROMPT,
        "critic":        _CRITIC_PROMPT,
        "kernel":        _KERNEL_PROMPT,
    }

    print("3 real LLM agents wired: "
          "orchestration=Claude, kernel=Claude, critic=Codex")
    print(f"Running {ticks} ticks")
    print()

    try:
        for n in range(ticks):
            try:
                await coordinator.tick(1)
            except Exception as exc:  # noqa: BLE001
                print(f"  [tick {n+1}] ERROR: {type(exc).__name__}: {exc}")
                continue
            counts: dict[str, int] = {}
            for topic in ("proposal", "review_verdict", "decision",
                           "delegated_result", "request", "response",
                           "heartbeat", "alert", "observation"):
                msgs = await coordinator.bus.tail(topic=topic, n=200)
                if msgs:
                    counts[topic] = len(msgs)
            tput = coordinator.shared_state.baseline_tput
            print(f"  [tick {n+1}] events={counts} baseline_tput={tput:.1f}")

        print()
        print("---- highlights ----")
        for topic in ("proposal", "review_verdict", "decision",
                       "delegated_result", "request", "response", "alert"):
            msgs = await coordinator.bus.tail(topic=topic, n=20)
            for m in msgs:
                summary = {k: v for k, v in m.payload.items()
                            if k in ("action_name", "verdict", "reasoning",
                                     "kind", "state", "result", "status",
                                     "from_agent", "severity", "summary",
                                     "task_id", "in_reply_to", "target_agent",
                                     "output_throughput")}
                for k, v in list(summary.items()):
                    if isinstance(v, str) and len(v) > 80:
                        summary[k] = v[:77] + "..."
                print(f"  {topic:18s} from={m.from_agent:13s} {summary}")

        print()
        print("---- final SharedState ----")
        print(coordinator.shared_state.to_prompt_summary())

        print()
        print("---- backend call counts ----")
        for name, b in backends.items():
            calls = getattr(b, "calls", [])
            print(f"  {name:13s}: {len(calls)} calls")
    finally:
        await coordinator.stop()
    return 0


def main() -> None:
    """Check the environment then run the demo, exiting with its code.

    Raises:
        SystemExit: Carrying the exit code returned by :func:`_run`.
    """
    _check_env()
    ticks = int(os.environ.get("COORDINATOR_TICKS", "4"))
    claude_model = os.environ.get("CLAUDE_MODEL", "claude-opus-4-7")
    codex_model = os.environ.get("CODEX_MODEL", "gpt-5.4")
    sys.exit(asyncio.run(_run(ticks=ticks, claude_model=claude_model,
                                codex_model=codex_model)))


if __name__ == "__main__":
    main()
