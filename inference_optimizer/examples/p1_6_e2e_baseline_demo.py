"""P1-6 end-to-end demo: real 4-agent loop runs a real SGLang baseline.

Wire-up:

* **Orchestration**  — real ClaudeBackend (claude-opus-4-7 via AMD proxy)
* **Critic**         — MockCriticBackend (auto-approve every proposal)
* **Kernel**         — MockKernelBackend (auto-respond to REQUEST)
* **Robustness**     — MockRobustnessBackend (heartbeat-only)

* `baseline` runner — :func:`baseline_executor` runs the Magpie SGLang
  CLI subprocess against ``baseline_sglang.yaml``, parses
  ``benchmark_report.json``, returns real throughput / latency numbers.

* SharedState seeded with ``model_name`` + ``model_path`` + ``baseline_tput=0``
  so Orchestration knows what to do on its very first tick.

Run::

    set -a && source /wekafs/xiaofei/AgentKernelArena/.env && set +a
    export CLAUDE_MODEL=claude-opus-4-7
    unset HIP_VISIBLE_DEVICES
    export ROCR_VISIBLE_DEVICES=1
    python -m inference_optimizer.examples.p1_6_e2e_baseline_demo

Expect:

  tick 1: Orchestration (Claude) reads SharedState, sees baseline_tput=0,
          proposes `baseline`. Critic mock approves. Coordinator
          materializes a task. Dispatcher runs baseline_executor — this
          launches Magpie + sglang server (~2 min for Qwen3-8B), parses
          throughput, writes delegated_result to the bus.
  tick 2: Orchestration sees baseline_tput > 0 in its inbox + state
          summary, proposes the next action (or wraps up).
"""

from __future__ import annotations

import asyncio
import os
import sys

from ..orchestrator.action_executors import baseline_executor
from ..orchestrator.backends import (
    ClaudeBackend,
    MockCriticBackend,
    MockKernelBackend,
    MockRobustnessBackend,
)
from ..orchestrator.coordinator import Coordinator
from ..orchestrator.shared_state import SharedState
from ..paths import make_session_dir


def _check_env() -> None:
    """Verify an LLM API token is set and warn on a bad GPU env var.

    Exits with status 2 when no recognized API token is configured, and warns
    when ``HIP_VISIBLE_DEVICES`` is set (which breaks GPU detection on this
    stack).

    Raises:
        SystemExit: When no API token environment variable is configured.
    """
    if not any(os.environ.get(n) for n in (
        "SAFE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY",
    )):
        print(
            "ERROR: SAFE_API_KEY / legacy ANTHROPIC key not set.\n"
            "Run: set -a && source /wekafs/xiaofei/AgentKernelArena/.env && set +a",
            file=sys.stderr,
        )
        sys.exit(2)
    if os.environ.get("HIP_VISIBLE_DEVICES"):
        print(
            "WARNING: HIP_VISIBLE_DEVICES is set — on this stack it makes "
            "torch.cuda.is_available() return False. Unset it; we rely on "
            "ROCR_VISIBLE_DEVICES from the YAML.",
            file=sys.stderr,
        )


async def _run(ticks: int, model: str | None) -> int:
    """Run the P1-6 end-to-end baseline demo.

    Seeds SharedState with a Qwen3-8B goal, builds a coordinator with a real
    Claude orchestration backend plus mocks, registers the real baseline
    executor and no-op prep stubs, ticks the loop, and prints per-tick events,
    highlights, and the final SharedState summary.

    Args:
        ticks (int): Number of coordinator ticks to run.
        model (str | None): Claude model name; ``None`` uses the SDK default.

    Returns:
        int: Process exit code (always ``0`` on completion).
    """
    session_dir = make_session_dir()
    print(f"Session dir: {session_dir}")

    # Seed SharedState so Orchestration has a goal on first tick.
    state = SharedState(
        session_id=session_dir.name,
        model_name="Qwen-Qwen3-8B",
        model_path="/wekafs/models/Qwen-Qwen3-8B",
        model_class="dense_8B",
        target_summary="Establish single-GPU SGLang baseline on Qwen3-8B (TP=1, CONC=8)",
        baseline_tput=0.0,
        cumulative_gain=0.0,
        max_minutes=15,
    )
    state.save(session_dir)

    backends = {
        "orchestration": ClaudeBackend(model=model, max_turns_default=4),
        "kernel":        MockKernelBackend(),
        "critic":        MockCriticBackend(),
        "robustness":    MockRobustnessBackend(),
    }
    coordinator = Coordinator(session_dir, backends=backends)
    coordinator.sub.register_executor("baseline", baseline_executor)
    # Stub prep executors — DESIGN §16 puts setup / classify / target_analysis
    # before baseline, but for this smoke run we don't need real prep work
    # (model_class etc. are already in SharedState). Returning empty success
    # lets the orchestration loop progress to baseline quickly.
    async def _noop_prep(ctx) -> dict:
        """No-op prep executor stub used by the smoke demo.

        Args:
            ctx: Executor context supplying ``ctx.task.kind``.

        Returns:
            dict: A success result echoing the task kind.
        """
        return {"status": "succeeded", "kind": ctx.task.kind, "note": "no-op stub for smoke demo"}
    for kind in ("target_analysis", "report"):
        coordinator.sub.register_executor(kind, _noop_prep)

    # Override orchestration system prompt for the smoke demo: model_class +
    # session_id are already populated, so prereqs (setup / classify /
    # target_analysis) are effectively done. Tell the agent to propose
    # `baseline` directly — otherwise Claude burns ticks exploring prereqs.
    coordinator.system_prompt_overrides = {
        "orchestration": (
            "You are the Orchestration agent for an inference-optimization run. "
            "Read the Shared session state below to see what's already done.\n\n"
            "PROCEDURE FOR THIS SMOKE RUN:\n"
            "1. If `baseline_tput == 0.0`, propose action `baseline` immediately. "
            "All prep actions (setup / classify / target_analysis) are ALREADY "
            "DONE — `model_class` is set, `target_summary` is set. DO NOT propose "
            "setup / classify / target_analysis. Skip straight to baseline.\n"
            "2. After baseline succeeds (you'll see `baseline_tput > 0` in the "
            "Shared state), the smoke run is complete — emit a single "
            "`send_message{topic='heartbeat', body_md='baseline-done'}`. Do not "
            "propose more actions.\n\n"
            "OUTPUT: every turn MUST emit exactly one or more `emit_intent` tool "
            "calls. Free-text replies are ignored.\n\n"
            "Allowed intent_type for you: `propose_action`, `delegate`, "
            "`request`, `update_state`, `update_persona`, `send_message`, "
            "`alert`, `ask_question`, `answer`. Use `propose_action{action_name, "
            "predicted_gain_pct, reason}` to propose."
        ),
    }

    print("Registered baseline runner + 4 prep stub executors. SharedState seeded.")
    print("Overrode orchestration system prompt to skip prep + go straight to baseline.")
    print(f"Running {ticks} ticks across 4 agents (orchestration=Claude real)")
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
                       "delegated_result", "alert"):
            msgs = await coordinator.bus.tail(topic=topic, n=20)
            for m in msgs:
                summary = {k: v for k, v in m.payload.items()
                            if k in ("action_name", "verdict", "kind",
                                     "state", "result", "status",
                                     "from_agent", "severity", "summary",
                                     "task_id", "in_reply_to",
                                     "output_throughput",
                                     "request_throughput",
                                     "completed_requests")}
                # Trim long values
                for k, v in list(summary.items()):
                    if isinstance(v, str) and len(v) > 60:
                        summary[k] = v[:60] + "..."
                print(f"  {topic:18s} from={m.from_agent:13s} {summary}")

        print()
        print("---- final SharedState ----")
        print(coordinator.shared_state.to_prompt_summary())
    finally:
        await coordinator.stop()

    return 0


def main() -> None:
    """Check the environment then run the demo, exiting with its code.

    Raises:
        SystemExit: Carrying the exit code returned by :func:`_run`.
    """
    _check_env()
    ticks = int(os.environ.get("COORDINATOR_TICKS", "3"))
    model = os.environ.get("CLAUDE_MODEL", "claude-opus-4-7")
    sys.exit(asyncio.run(_run(ticks=ticks, model=model)))


if __name__ == "__main__":
    main()
