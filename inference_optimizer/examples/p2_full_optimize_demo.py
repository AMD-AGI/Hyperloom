"""P2-4 end-to-end demo: full skill flow exercised with bounded ticks.

This is the runnable companion to ``inference_optimizer optimize`` for
people who want a Python script form (and a small smoke ticks count by
default so it doesn't take 2h to validate).

What it exercises:

  Orchestration (real Claude opus-4-7) is told to walk this DFS plan:

     baseline → profile → backends → params → integrate → report

  For each action, Critic (real Codex gpt-5.4) reviews + approves.
  For the kernel-owned actions, Orchestration emits a REQUEST →
  Coordinator's `kernel_request_handlers` runs the Hyperloom kernel-agent
  shell tools (or stub for integrate) and emits the RESPONSE.

  After ``COORDINATOR_TICKS`` (default 6), the report runner writes the
  final.md / final.json under ``$SESSION_DIR/report/``.

Usage::

    set -a && source /wekafs/xiaofei/AgentKernelArena/.env && set +a
    unset HIP_VISIBLE_DEVICES
    export ROCR_VISIBLE_DEVICES=1 PATH=/opt/venv/bin:$PATH
    export CLAUDE_MODEL=claude-opus-4-7  CODEX_MODEL=gpt-5.4
    export COORDINATOR_TICKS=6        # ~5-10 min total
    python -m inference_optimizer.examples.p2_full_optimize_demo

For a true 2h end-to-end run with a hard target, prefer the CLI::

    inference_optimizer optimize \\
        --model /wekafs/models/Qwen-Qwen3-8B \\
        --target-gain 10 --max-hours 2 -v
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from ..orchestrator.action_executors import (
    baseline_executor,
    explore_executor,
    report_executor,
    sweep_executor,
)
from ..orchestrator.backends import (
    ClaudeBackend,
    CodexBackend,
    MockRobustnessBackend,
)
from ..orchestrator.coordinator import Coordinator
from ..orchestrator.objective import TargetGainObjective
from ..orchestrator.shared_state import SharedState
from ..paths import make_session_dir


_ORCH_PROMPT = (
    "You are the Orchestration agent. Read the Shared session state to see "
    "progress.\n\n"
    "Walk this DFS-style plan in order:\n"
    "  1. baseline_tput == 0          → propose `baseline`\n"
    "  2. baseline done, no trace     → propose `profile`\n"
    "  3. profile done                → propose `backends`\n"
    "  4. backends done               → propose `params`\n"
    "  5. params done                 → emit REQUEST{target='kernel',\n"
    "                                       kind='trace_analyze',\n"
    "                                       params={trace_input: <main_trace_path from prior delegated_result>}}\n"
    "  6. trace_analyze response in  → emit REQUEST{target='kernel',\n"
    "                                       kind='integrate',\n"
    "                                       params={base_tput: <current best>,\n"
    "                                                kernel_id: <first hot kernel>,\n"
    "                                                config_path: <baseline config>}}\n"
    "  7. integrate response in       → propose `report` (regular delegate)\n"
    "  8. report done OR target_gain  → emit send_message{topic='heartbeat',\n"
    "                                       body_md='done'} and stop\n\n"
    "OUTPUT: every turn MUST emit at least one `emit_intent` tool call.\n"
)


_CRITIC_PROMPT = (
    "You are the Critic. Approve every well-formed proposal whose\n"
    "action_name is one of: baseline, profile, backends, params, sweep,\n"
    "integrate, report. Reject anything else with a brief reason.\n"
    "REQUIRED payload: target_proposal_msg_id, verdict, reasoning."
)


_KERNEL_PROMPT = (
    "You are the Kernel agent. For requests not handled by Coordinator's\n"
    "programmatic dispatch, emit RESPONSE{in_reply_to, kind='<kind>_done',\n"
    "status='ok', result={...}}. If your inbox has nothing, heartbeat."
)


async def _noop_prep(ctx) -> dict:
    return {"status": "succeeded", "kind": ctx.task.kind, "note": "noop-stub"}


async def _run(ticks: int, target_gain: float) -> int:
    session_dir = make_session_dir()
    print(f"Session dir: {session_dir}")

    state = SharedState(
        session_id=session_dir.name,
        model_name="Qwen-Qwen3-8B",
        model_path="/wekafs/models/Qwen-Qwen3-8B",
        model_class="dense_8B",
        target_summary=f"baseline+profile+backends+params+kernel-opt; target {target_gain}% gain",
        baseline_tput=0.0,
        max_minutes=int(ticks * 5),  # rough budget hint
    )
    state.save(session_dir)

    backends = {
        "orchestration": ClaudeBackend(model="claude-opus-4-7", max_turns_default=4),
        "kernel":        CodexBackend(model="gpt-5.4"),
        "critic":        CodexBackend(model="gpt-5.4"),
        "robustness":    MockRobustnessBackend(),
    }
    coordinator = Coordinator(session_dir, backends=backends)
    coordinator.sub.register_executor("baseline", baseline_executor)
    coordinator.sub.register_executor("explore",  explore_executor)
    coordinator.sub.register_executor("sweep",    sweep_executor)
    coordinator.sub.register_executor("report",   report_executor)
    for kind in ("target_analysis", "recover"):
        coordinator.sub.register_executor(kind, _noop_prep)

    coordinator.system_prompt_overrides = {
        "orchestration": _ORCH_PROMPT,
        "critic":        _CRITIC_PROMPT,
        "kernel":        _KERNEL_PROMPT,
    }

    print(f"Driving {ticks} ticks with target_gain={target_gain}%")
    print()

    objective = TargetGainObjective(target_gain_pct=target_gain)
    try:
        stop_reason = await coordinator.run(
            objective=objective,
            max_ticks=ticks,
            tick_interval_sec=0.0,
            install_signal_handlers=False,
        )
    finally:
        await coordinator.stop()

    print()
    print("================ Final summary ================")
    print(f"  stop_reason     : {stop_reason}")
    print(f"  baseline_tput   : {coordinator.shared_state.baseline_tput:.1f} tok/s/GPU")
    print(f"  cumulative_gain : {coordinator.shared_state.cumulative_gain:.2f}%")
    print(f"  current_best    : {coordinator.shared_state.current_best}")
    report_md = session_dir / "report" / "final.md"
    if report_md.exists():
        print(f"  report          : {report_md}")
    print("===============================================")
    return 0


def main() -> None:
    ticks = int(os.environ.get("COORDINATOR_TICKS", "6"))
    target_gain = float(os.environ.get("TARGET_GAIN", "10"))
    sys.exit(asyncio.run(_run(ticks=ticks, target_gain=target_gain)))


if __name__ == "__main__":
    main()
