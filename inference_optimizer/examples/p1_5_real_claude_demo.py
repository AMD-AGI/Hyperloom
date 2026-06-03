"""P1-5 real Claude demo.

Same shape as p0_main_loop, but the **Orchestration** reactor is wired to
the real :class:`ClaudeBackend` instead of a scripted MockBackend. Critic
/ Kernel / Robustness keep their reactive mocks so the loop stays
self-driving end-to-end without burning Codex tokens.

Usage::

    set -a && source /wekafs/xiaofei/AgentKernelArena/.env && set +a
    python -m inference_optimizer.examples.p1_5_real_claude_demo

The script:

* Verifies that ``SAFE_API_KEY`` (or legacy ``ANTHROPIC_AUTH_TOKEN`` / ``ANTHROPIC_API_KEY``) is set
* Builds a Coordinator with all 4 agents wired
* Registers a one-line ``baseline`` runner (same as p0_main_loop)
* Runs ``COORDINATOR_TICKS`` ticks (default 4) so Orchestration has room
  to propose → see verdict → request Kernel → see response
* Prints highlight events + a per-tick event count so we can watch the
  4-agent main path actually flow with a real LLM driving Orchestration
"""

from __future__ import annotations

import asyncio
import os
import sys

from ..orchestrator.backends import (
    ClaudeBackend,
    MockCriticBackend,
    MockKernelBackend,
    MockRobustnessBackend,
)
from ..orchestrator.coordinator import Coordinator
from ..paths import make_session_dir


ENV_TOKEN_NAMES = ("SAFE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")


def _check_env() -> None:
    """Verify an LLM API token is present and report the base URL.

    Exits the process with status 2 when none of the recognized token
    environment variables are set.

    Raises:
        SystemExit: When no API token environment variable is configured.
    """
    if not any(os.environ.get(n) for n in ENV_TOKEN_NAMES):
        print(
            f"ERROR: none of {ENV_TOKEN_NAMES} are set in this shell.\n"
            f"Run `set -a && source /wekafs/xiaofei/AgentKernelArena/.env "
            f"&& set +a` first.",
            file=sys.stderr,
        )
        sys.exit(2)
    base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL")
    if base:
        print(f"Using LLM base URL={base}")


async def _baseline_executor(ctx) -> dict:
    """Pretend to run a baseline benchmark for the demo.

    Args:
        ctx: Executor context supplying ``ctx.task.task_id``.

    Returns:
        dict: Mock baseline metrics including the originating task id.
    """
    return {
        "tput_tok_per_s_per_gpu": 1840.0,
        "p99_latency_ms": 152,
        "source": "mock",
        "task_id": ctx.task.task_id,
    }


async def _run(ticks: int, model: str | None) -> int:
    """Run the real-Claude main-loop demo for a number of ticks.

    Builds a coordinator with a real Claude orchestration backend and mock
    critic/kernel/robustness agents, ticks the loop, and prints per-tick
    event counts, highlights, and orchestration backend stats.

    Args:
        ticks (int): Number of coordinator ticks to run.
        model (str | None): Claude model name; ``None`` uses the SDK default.

    Returns:
        int: Process exit code (always ``0`` on completion).
    """
    session_dir = make_session_dir()
    print(f"Session dir: {session_dir}")

    backends = {
        # Real Claude — Orchestration drives the loop
        "orchestration": ClaudeBackend(model=model, max_turns_default=2),
        # Mocks for the rest (still self-driving)
        "kernel":     MockKernelBackend(),
        "critic":     MockCriticBackend(),
        "robustness": MockRobustnessBackend(),
    }

    coordinator = Coordinator(session_dir, backends=backends)
    coordinator.sub.register_executor("baseline", _baseline_executor)

    print(f"Running {ticks} ticks across 4 agents (orchestration=Claude real)")
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
            print(f"  [tick {n+1}] events: {counts}")

        print()
        print("---- highlights ----")
        for topic in ("proposal", "review_verdict", "decision",
                       "delegated_result", "request", "response", "alert"):
            msgs = await coordinator.bus.tail(topic=topic, n=20)
            for m in msgs:
                summary = {k: v for k, v in m.payload.items()
                            if k in ("action_name", "verdict", "kind",
                                     "state", "result", "status",
                                     "from_agent", "severity", "summary",
                                     "to_agent", "in_reply_to")}
                print(f"  {topic:18s} from={m.from_agent:13s} {summary}")

        print()
        print("---- orchestration backend stats ----")
        ob = backends["orchestration"]
        for entry in ob.calls[-5:]:
            print(f"  {entry}")
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
    model = os.environ.get("CLAUDE_MODEL")  # None → SDK default
    sys.exit(asyncio.run(_run(ticks=ticks, model=model)))


if __name__ == "__main__":
    main()
