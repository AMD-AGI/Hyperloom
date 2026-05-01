"""CLI entry — DESIGN v0.6 §22.

Usage::

    inference-optimizer optimize \\
        --model /wekafs/models/Qwen-Qwen3-8B \\
        --target-gain 10 \\
        --max-hours 2

Single subcommand for now (``optimize``). Wires Claude+Codex backends,
registers all available action_executors, builds the requested objective,
and starts ``Conductor.run()`` until target / time / SIGTERM.

Env vars consumed (besides the standard backend creds):

  ANTHROPIC_AUTH_TOKEN  /  ANTHROPIC_BASE_URL  — required for Claude
  ROCR_VISIBLE_DEVICES                         — pin the GPU
  CLAUDE_MODEL                                 — default claude-opus-4-7
  CODEX_MODEL                                  — default gpt-5.4
  INFERENCE_OPTIMIZER_SESSION_ROOT             — overrides default session root
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from .orchestrator.action_executors import (
    backends_executor,
    baseline_executor,
    params_executor,
    profile_executor,
    report_executor,
    sweep_executor,
)
from .orchestrator.backends import (
    ClaudeBackend,
    CodexBackend,
    MockRobustnessBackend,
)
from .orchestrator.conductor import Conductor
from .orchestrator.objective import build_objective
from .orchestrator.shared_state import SharedState
from .paths import make_session_dir, session_root


log = logging.getLogger("inference_optimizer.cli")


_DEFAULT_ORCH_PROMPT = (
    "You are the Orchestration agent for an inference-optimization run.\n"
    "Read the Shared session state below to see progress against the goal.\n\n"
    "Decision rules:\n"
    "1. If `baseline_tput == 0.0`, propose action `baseline` immediately.\n"
    "   Prep actions (setup / classify / target_analysis) are stubbed for\n"
    "   the smoke run — do NOT propose them.\n"
    "2. After baseline succeeds, follow this DFS-style plan (per\n"
    "   DESIGN §16): backends → params → kernel-opt → integrate. Each\n"
    "   round, pick the action with the highest expected value given the\n"
    "   current Shared state.\n"
    "3. To dispatch a kernel-owned action (kernel_opt / integrate /\n"
    "   deep_kernel_analysis / operator_tuning / vendor_kernel_config),\n"
    "   you MUST emit `request{target_agent='kernel', kind=...}`. PolicyGate\n"
    "   rejects direct delegate of these actions.\n"
    "4. After every successful action, observe the new `baseline_tput` /\n"
    "   `cumulative_gain` and decide whether to continue.\n"
    "5. If the goal is reached (Shared state shows `stop_reason='target_reached'`),\n"
    "   emit a single send_message{topic='heartbeat', body_md='goal-reached'}\n"
    "   and stop.\n\n"
    "OUTPUT: every turn MUST emit at least one `emit_intent` tool call. "
    "Free-text replies are dropped.\n"
)

_DEFAULT_CRITIC_PROMPT = (
    "You are the Critic agent. Your only job: review proposals from\n"
    "Orchestration and emit one `review_verdict` per un-reviewed proposal.\n\n"
    "Decision rule (smoke-grade — keep it simple):\n"
    "  * baseline / profile / classify / setup / target_analysis / report /\n"
    "    backends / params / sweep / dream  → approve\n"
    "  * kernel_opt / integrate / operator_tuning / vendor_kernel_config /\n"
    "    deep_kernel_analysis  → approve (Orchestration sends them via\n"
    "    REQUEST anyway, you just OK the proposal flow)\n"
    "  * Reject only if action_name is unknown or accuracy_risk > 0.3\n"
    "    without obvious justification.\n\n"
    "Required payload: target_proposal_msg_id, verdict, reasoning."
)

_DEFAULT_KERNEL_PROMPT = (
    "You are the Kernel agent — responder-only. You receive `request`\n"
    "events from Orchestration in your inbox.\n\n"
    "For every un-answered request, emit ONE `response` intent in reply.\n"
    "Schema:\n"
    "  intent_type: response\n"
    "  payload: {\n"
    "    in_reply_to: <request msg_id>,\n"
    "    kind:        '<request.kind>_done',\n"
    "    status:      'ok' | 'failed' | 'needs_review',\n"
    "    result:      { /* whatever the request asked for */ }\n"
    "  }\n\n"
    "If your inbox has no requests, emit one send_message{topic='heartbeat',\n"
    "body_md='ok'}. You may NOT propose, delegate, or initiate REQUESTs."
)


async def _noop_prep(ctx) -> dict:
    return {"status": "succeeded", "kind": ctx.task.kind, "note": "noop-stub"}


def _build_backends(
    *, claude_model: str, codex_model: str, kernel_codex: bool
) -> dict[str, Any]:
    backends: dict[str, Any] = {
        "orchestration": ClaudeBackend(model=claude_model, max_turns_default=4),
        "critic":        CodexBackend(model=codex_model),
        "robustness":    MockRobustnessBackend(),
    }
    if kernel_codex:
        # Per user request — Codex Kernel agent is faster than Claude for
        # short responder-only turns.
        backends["kernel"] = CodexBackend(model=codex_model)
    else:
        backends["kernel"] = ClaudeBackend(model=claude_model, max_turns_default=4)
    return backends


def _seed_shared_state(session_dir: Path, args: argparse.Namespace) -> SharedState:
    state = SharedState(
        session_id=session_dir.name,
        model_name=Path(args.model).name,
        model_path=str(args.model),
        model_class=args.model_class or "",
        target_summary=args.target_summary or _default_target_summary(args),
        baseline_tput=0.0,
        cumulative_gain=0.0,
        max_minutes=int((args.max_hours or 0) * 60),
    )
    state.save(session_dir)
    return state


def _default_target_summary(args: argparse.Namespace) -> str:
    if args.target_gain:
        return (
            f"Establish baseline on {Path(args.model).name} then drive "
            f"cumulative_gain to >= {args.target_gain}% within "
            f"{args.max_hours}h."
        )
    if args.target_tput:
        return (
            f"Establish baseline on {Path(args.model).name} then reach "
            f"{args.target_tput} tok/s/GPU within {args.max_hours}h."
        )
    return f"Optimize {Path(args.model).name} for up to {args.max_hours}h (no target)."


def _register_executors(conductor: Conductor) -> None:
    """Wire all currently-available action executors.

    P2-1 ships only `baseline` (real Magpie). Stubs for prep + kernel-owned
    actions keep the Orchestration loop from stalling while later phases
    fill in the real ones.
    """
    conductor.sub.register_executor("baseline", baseline_executor)
    conductor.sub.register_executor("profile",  profile_executor)
    conductor.sub.register_executor("backends", backends_executor)
    conductor.sub.register_executor("params",   params_executor)
    conductor.sub.register_executor("sweep",    sweep_executor)
    conductor.sub.register_executor("report",   report_executor)
    for kind in ("setup", "classify", "target_analysis",
                  "kernel_opt", "integrate", "deep_kernel_analysis",
                  "operator_tuning", "vendor_kernel_config",
                  "dream", "re_explore", "recover",
                  "comm_optimization", "compiler_tuning"):
        conductor.sub.register_executor(kind, _noop_prep)


def _print_final_summary(state: SharedState, stop_reason: str) -> None:
    print()
    print("================ Final summary ================")
    print(f"  stop_reason     : {stop_reason}")
    print(f"  session_id      : {state.session_id}")
    print(f"  model           : {state.model_name}")
    print(f"  baseline_tput   : {state.baseline_tput:.1f} tok/s/GPU")
    print(f"  cumulative_gain : {state.cumulative_gain:.2f}%")
    print(f"  current_best    : {state.current_best}")
    print(f"  pruned_families : {state.pruned_families}")
    print(f"  crash_count     : {state.crash_count}")
    print("===============================================")


async def _run_optimize(args: argparse.Namespace) -> int:
    if args.resume:
        # Resume mode: skip the SharedState seed so Conductor.__init__
        # picks up the existing state.json + SQLite event log unchanged.
        session_dir = session_root() / args.resume
        if not session_dir.exists():
            print(f"ERROR: --resume session not found: {session_dir}",
                  file=sys.stderr)
            sys.exit(2)
        state = SharedState.load_or_init(session_dir)
        print(f"Resuming session: {session_dir}")
        print(f"  prior baseline_tput   : {state.baseline_tput:.1f}")
        print(f"  prior cumul_gain      : {state.cumulative_gain:.2f}%")
        print(f"  prior current_best    : "
              f"{(state.current_best or {}).get('action')}/"
              f"{(state.current_best or {}).get('tput')}")
        print(f"  prior stop_reason     : {state.stop_reason or '(none)'}")
    else:
        if not args.model:
            print("ERROR: --model is required for new runs (or use --resume "
                  "<session_id>)", file=sys.stderr)
            sys.exit(2)
        session_dir = make_session_dir(args.session_name) if args.session_name \
            else make_session_dir()
        print(f"Session dir: {session_dir}")
        state = _seed_shared_state(session_dir, args)

    objective = build_objective({
        "MAX_HOURS": str(args.max_hours),
        "TARGET_GAIN_PCT": str(args.target_gain) if args.target_gain else "",
        "TARGET_TPUT_PER_GPU": str(args.target_tput) if args.target_tput else "",
        "TARGET_DIR": args.target_baseline_dir or "",
    })
    print(f"Objective       : kind={objective.kind()} {objective.describe()}")
    backends = _build_backends(
        claude_model=args.claude_model,
        codex_model=args.codex_model,
        kernel_codex=args.kernel_codex,
    )
    conductor = Conductor(session_dir, backends=backends)
    conductor.system_prompt_overrides = {
        "orchestration": args.orch_prompt or _DEFAULT_ORCH_PROMPT,
        "critic":        args.critic_prompt or _DEFAULT_CRITIC_PROMPT,
        "kernel":        args.kernel_prompt or _DEFAULT_KERNEL_PROMPT,
    }
    _register_executors(conductor)

    print(f"Backends        : "
          f"orchestration=Claude({args.claude_model}), "
          f"kernel={'Codex' if args.kernel_codex else 'Claude'}, "
          f"critic=Codex({args.codex_model}), robustness=mock")
    print(f"Max ticks       : {args.max_ticks or 'unlimited'} "
          f"(budget = {args.max_hours}h)")
    print(f"Tick interval   : {args.tick_interval_sec}s")
    print()

    try:
        stop_reason = await conductor.run(
            objective=objective,
            max_minutes=args.max_hours * 60.0,
            tick_interval_sec=args.tick_interval_sec,
            max_ticks=args.max_ticks,
            install_signal_handlers=True,
        )
    finally:
        await conductor.stop()

    _print_final_summary(conductor.shared_state, stop_reason)
    return 0 if stop_reason in ("target_reached", "time_exhausted", "max_ticks") else 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="inference-optimizer",
        description="Inference Optimizer v0.6 — multi-agent SGLang/vLLM optimization",
    )
    p.add_argument("--verbose", "-v", action="count", default=0,
                    help="Verbose logging (-v INFO, -vv DEBUG)")
    sub = p.add_subparsers(dest="command", required=True)

    opt = sub.add_parser("optimize",
                          help="Drive a multi-agent optimization run on a model")
    opt.add_argument("--model", "-m", type=Path, default=None,
                      help="Model path (required for new runs; ignored when "
                           "--resume is set — model is read from state.json)")
    opt.add_argument("--max-hours", type=float, default=2.0,
                      help="Wall-clock budget in hours (default 2.0)")
    grp = opt.add_mutually_exclusive_group()
    grp.add_argument("--target-gain", type=float, default=None,
                      help="Stop when cumulative_gain >= N%% over baseline")
    grp.add_argument("--target-tput", type=float, default=None,
                      help="Stop when current best tok/s/GPU >= N")
    grp.add_argument("--target-baseline-dir", type=str, default=None,
                      help="Stop when current best matches the baseline in DIR")
    opt.add_argument("--session-name", type=str, default=None,
                      help="Override auto-generated session id (for new runs)")
    opt.add_argument("--resume", type=str, default=None,
                      help="Resume from an existing session id. Skips the "
                           "SharedState seed and lets the Conductor replay "
                           "the prior event log + state.json. Mutually "
                           "exclusive with --session-name in practice.")
    opt.add_argument("--model-class", type=str, default=None,
                      help="Optional model class hint (dense_8B / moe_mla / ...)")
    opt.add_argument("--target-summary", type=str, default=None,
                      help="Free-text goal summary surfaced in prompts")
    opt.add_argument("--max-ticks", type=int, default=None,
                      help="Hard tick cap (None = unlimited; mostly for tests)")
    opt.add_argument("--tick-interval-sec", type=float, default=0.0,
                      help="Sleep between ticks (0 = no sleep)")
    opt.add_argument("--claude-model", type=str,
                      default=os.environ.get("CLAUDE_MODEL", "claude-opus-4-7"))
    opt.add_argument("--codex-model", type=str,
                      default=os.environ.get("CODEX_MODEL", "gpt-5.4"))
    opt.add_argument("--kernel-codex", action="store_true", default=True,
                      help="Use Codex backend for Kernel agent (default — faster). "
                           "Pass --kernel-claude to switch.")
    opt.add_argument("--kernel-claude", action="store_false", dest="kernel_codex",
                      help="Use Claude backend for Kernel agent")
    opt.add_argument("--orch-prompt", type=str, default=None,
                      help="Override Orchestration system prompt (file path or inline)")
    opt.add_argument("--critic-prompt", type=str, default=None,
                      help="Override Critic system prompt")
    opt.add_argument("--kernel-prompt", type=str, default=None,
                      help="Override Kernel system prompt")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    if args.command == "optimize":
        # Resolve any --*-prompt that point at a file.
        for attr in ("orch_prompt", "critic_prompt", "kernel_prompt"):
            v = getattr(args, attr)
            if v and Path(v).exists():
                setattr(args, attr, Path(v).read_text(encoding="utf-8"))
        return asyncio.run(_run_optimize(args))
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
