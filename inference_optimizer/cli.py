"""CLI entry — DESIGN v0.6 §22.

Usage::

    inference_optimizer optimize \\
        --model /wekafs/models/<your-model> \\
        --target-gain 10 \\
        --max-hours 2

    # or via env (matches the rest of the pipeline / Dockerfile convention):
    export MODEL_PATH=/wekafs/models/<your-model>
    inference_optimizer optimize --target-gain 10 --max-hours 2

Single subcommand for now (``optimize``). Wires Claude+Codex backends,
registers all available action_executors, builds the requested objective,
and starts ``Coordinator.run()`` until target / time / SIGTERM.

Env vars consumed (besides the standard backend creds):

  MODEL_PATH                                   — required if --model not passed;
                                                 also exported back to subprocess
                                                 env so Magpie YAMLs get the
                                                 correct model path injected
                                                 instead of the YAML's hardcoded
                                                 fallback.
  OPENAI_BASE_URL + SAFE_API_KEY — canonical LiteLLM endpoint; compatibility aliases are exported for Claude/OOB/GEAK
  ROCR_VISIBLE_DEVICES                         — pin the GPU
  CLAUDE_MODEL                                 — default claude-opus-4-7
  CODEX_MODEL                                  — default gpt-5.4
  INFERENCE_OPTIMIZER_SESSION_ROOT             — overrides default session root
  INFERENCE_OPTIMIZER_KB_ROOT                  — marathon KB dir (kb_query.py +
                                                 entries.jsonl); default:
                                                 Hyperloom/marathon/skills/kb
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
    MockCriticBackend,
    MockRobustnessBackend,
)
from .orchestrator.coordinator import Coordinator
from .orchestrator.objective import build_objective
from .orchestrator.shared_state import SharedState
from .paths import make_session_dir, session_root


log = logging.getLogger("inference_optimizer.cli")


_DEFAULT_ORCH_PROMPT = (
    "You are the Orchestration agent for an inference-optimization run.\n"
    "Read the Shared session state below to see progress against the goal.\n\n"
    "===== DECISION FRAMEWORK (follow EVERY tick) =====\n"
    "Before proposing anything, evaluate Shared state to pick the next\n"
    "highest-value action:\n"
    "  if `cumulative_gain >= target_gain_pct`:\n"
    "      → propose `report` (one shot). Then emit a single\n"
    "        send_message{topic='heartbeat', body_md='goal-reached'}\n"
    "        every following tick.\n"
    "  if `stop_reason` is set:\n"
    "      → emit one heartbeat 'goal-reached' and stop emitting actions.\n"
    "  if `baseline_tput == 0`:\n"
    "      → propose `baseline`.\n"
    "  if `last_profile_trace` is empty:\n"
    "      → propose `profile` (writes torch_trace; SharedState then has\n"
    "        last_profile_trace = real path you'll use for select_kernels).\n"
    "  if `last_profile_trace` is set AND `last_profile_args` already\n"
    "  matches the active server config (current_best.extra_sglang_args,\n"
    "  or empty when current_best.tput == baseline_tput):\n"
    "      → DO NOT propose `profile` again. Profile is deterministic for\n"
    "        the same server config + workload, so re-running it cannot\n"
    "        change the hot-kernel list.\n"
    "  if `last_select_kernels.reusable_native_kernel_ids` is empty AND\n"
    "  `last_profile_trace` is set:\n"
    "      → kernel-opt has no eligible target. DO NOT propose `profile`,\n"
    "        DO NOT emit select_kernels/run_optimization. Fall back to\n"
    "        params/sweep/heartbeat instead.\n"
    "  if `params_no_promote_streak >= 5`:\n"
    "      → params has plateaued (5+ rounds didn't promote). Switch to\n"
    "        kernel-opt path (REQUEST select_kernels → run_optimization →\n"
    "        integrate). Do NOT re-propose params/backends/sweep until\n"
    "        kernel-opt produces a result.\n"
    "  if backends/params haven't been tried this session (count proposals\n"
    "  in your inbox):\n"
    "      → propose backends first, then params. One round each.\n"
    "  otherwise:\n"
    "      → kernel-opt path (see Pipeline below). It's the most expensive\n"
    "        but also the highest-ceiling lever once params plateaued.\n\n"
    "===== KERNEL-OPT PIPELINE (sequential, no backtracking) =====\n"
    "step K1 (skip when cached): emit\n"
    "  request{target_agent: 'kernel', kind: 'select_kernels',\n"
    "          params: {trace_input: <verbatim last_profile_trace value>,\n"
    "                   top_k: 10}}\n"
    "  STRICT: if `last_select_kernels.trace_input` already equals\n"
    "  `last_profile_trace`, the candidate list is cached and you MUST\n"
    "  skip K1. Go directly to K2 using `last_select_kernels.candidates_path`\n"
    "  and the kernel_id list under `last_select_kernels.top5`. Re-emit\n"
    "  `select_kernels` only when `last_profile_trace` changes (i.e. after\n"
    "  a fresh `profile`).\n\n"
    "step K2: pick the next reusable native kernel from\n"
    "  `last_select_kernels.reusable_native_kernel_ids` in order, skipping\n"
    "  any whose kernel_id already appears in last_kernel_opt.kernel_id.\n"
    "  HARD RULES:\n"
    "    - kernel_id MUST appear in `reusable_native_kernel_ids`. Do NOT\n"
    "      pick from raw `hot_kernels_top15` if the entry is not in that\n"
    "      list — top hot kernels are often Tensile/CK/vendor binaries\n"
    "      and will be rejected with `non_reusable_kernel`.\n"
    "    - If `reusable_native_kernel_ids` is empty, do NOT keep emitting\n"
    "      run_optimization. Heartbeat instead and consider re-profiling.\n"
    "  Then emit\n"
    "  request{target_agent: 'kernel', kind: 'run_optimization',\n"
    "          params: {kernel_id: <picked kernel_id>,\n"
    "                   source_file: <from hot_kernels[i].source_file>,\n"
    "                   candidates_path: <select_kernels_done.candidates_path>,\n"
    "                   backends: 'claude',\n"
    "                   budget_minutes: 60}}\n\n"
    "step K3: when `run_optimization_done` arrives, look at\n"
    "  result.proposal.decision and result.verification:\n"
    "    KEEP        → emit request{kind: 'integrate', params:\n"
    "                               {kernel_id: <result.kernel_id>,\n"
    "                                patch_path: <result.best_artifact_path OR result.verification.best_artifact_path>,\n"
    "                                target_file: <result.source_file>,\n"
    "                                base_tput: <current_best.tput>,\n"
    "                                extra_sglang_args: <current_best.extra_sglang_args>,\n"
    "                                config_path: <baseline yaml absolute path>}}\n"
    "    PARTIAL/REVERT → don't integrate; pick the NEXT hot kernel\n"
    "                     (skip kernels with kernel_id == last_kernel_opt.kernel_id)\n"
    "                     and re-issue step K2 with that one.\n\n"
    "===== KERNEL TARGETING (native vs torch.compile) =====\n"
    "First decide the final serving mode as a framework/params choice:\n"
    "SGLang may run with or without `--enable-torch-compile`; vLLM commonly\n"
    "runs with compile/CUDAGraph optimizations by default unless eager/-O0 is\n"
    "explicitly requested. `select_kernels` should profile that final serving\n"
    "mode, BUT kernel-opt may only rewrite reusable native sources that still\n"
    "appear in that trace. Never optimize `/tmp/torchinductor*`, Inductor cache,\n"
    "or `triton_poi_*`/`triton_red_*` runtime-generated kernels — they are tied\n"
    "to one compile graph/cache and the patch is not reusable. If compile-on\n"
    "leaves no high-share reusable native kernels, stop kernel-opt and continue\n"
    "with framework/params/compile configuration tuning instead.\n\n"
    "===== HARD RULES =====\n"
    "* `kind` MUST be EXACTLY one of: 'select_kernels' / 'run_optimization' /\n"
    "  'integrate' / 'apply_patch' (these have programmatic handlers).\n"
    "  `kernel_opt` is NOT a recognised kind — never use it.\n"
    "* Never invent a trace_input path. ONLY use SharedState.last_profile_trace.\n"
    "* If your last action was a propose_action, do NOT re-propose the same\n"
    "  action in the next 3 ticks (give the dispatcher time to run it).\n"
    "* Every turn MUST emit at least one `emit_intent` tool call.\n"
    "  Free-text replies are dropped.\n"
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
    "Native-only rule: run_optimization must refuse runtime-generated\n"
    "torch.compile/Inductor/Triton cache kernels. Only reusable framework\n"
    "sources under stable repos (aiter/sglang/vllm source trees) are valid\n"
    "kernel-opt targets; otherwise return status='failed' with a clear reason.\n\n"
    "If your inbox has no requests, emit one send_message{topic='heartbeat',\n"
    "body_md='ok'}. You may NOT propose, delegate, or initiate REQUESTs."
)


_GFX_TO_RUNNER: dict[str, str] = {
    # Mirror Magpie/modes/benchmark/image_selector.py:138-140. Listed here so
    # we can log the resolved value at session start instead of waiting for
    # Magpie subprocess output deep in the run.
    "gfx942":  "mi300x",
    "gfx950":  "mi355x",
    "gfx1100": "mi325x",
}


def _autodetect_gpu_type() -> str | None:
    """Return mi300x|mi325x|mi355x or None if undetectable.

    Tries `rocm-smi --showproductname` first (most reliable), then falls
    back to torch.cuda.get_device_properties(0).gcnArchName parsing. Both
    are best-effort — on CPU-only or non-ROCm boxes we silently return
    None so the caller can defer to Magpie's own detection layer.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=5,
        ).stdout.upper()
        for tag in ("MI355X", "MI325X", "MI300X"):
            if tag in out:
                return tag.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
        pass
    try:
        import torch
        arch = torch.cuda.get_device_properties(0).gcnArchName
        gfx = arch.split(":", 1)[0].lower()
        return _GFX_TO_RUNNER.get(gfx)
    except Exception:  # noqa: BLE001
        return None


async def _noop_prep(ctx) -> dict:
    return {"status": "succeeded", "kind": ctx.task.kind, "note": "noop-stub"}


def _build_backends(
    *, claude_model: str, codex_model: str, kernel_codex: bool,
    critic_mock: bool = False,
) -> dict[str, Any]:
    backends: dict[str, Any] = {
        "orchestration": ClaudeBackend(model=claude_model, max_turns_default=4),
        "critic":        MockCriticBackend() if critic_mock else CodexBackend(model=codex_model),
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


def _register_executors(coordinator: Coordinator) -> None:
    """Wire all currently-available action executors.

    P2-1 ships only `baseline` (real Magpie). Stubs for prep + kernel-owned
    actions keep the Orchestration loop from stalling while later phases
    fill in the real ones.
    """
    coordinator.sub.register_executor("baseline", baseline_executor)
    coordinator.sub.register_executor("profile",  profile_executor)
    coordinator.sub.register_executor("backends", backends_executor)
    coordinator.sub.register_executor("params",   params_executor)
    coordinator.sub.register_executor("sweep",    sweep_executor)
    coordinator.sub.register_executor("report",   report_executor)
    for kind in ("setup", "classify", "target_analysis",
                  "kernel_opt", "integrate", "deep_kernel_analysis",
                  "operator_tuning", "vendor_kernel_config",
                  "dream", "re_explore", "recover",
                  "comm_optimization", "compiler_tuning"):
        coordinator.sub.register_executor(kind, _noop_prep)


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
        # Resume mode: skip the SharedState seed so Coordinator.__init__
        # picks up the existing state.json + SQLite event log unchanged.
        session_dir = session_root() / args.resume
        if not session_dir.exists():
            print(f"ERROR: --resume session not found: {session_dir}",
                  file=sys.stderr)
            sys.exit(2)
        state = SharedState.load_or_init(session_dir)
        prior_stop = state.stop_reason
        print(f"Resuming session: {session_dir}")
        print(f"  prior baseline_tput   : {state.baseline_tput:.1f}")
        print(f"  prior cumul_gain      : {state.cumulative_gain:.2f}%")
        print(f"  prior current_best    : "
              f"{(state.current_best or {}).get('action')}/"
              f"{(state.current_best or {}).get('tput')}")
        print(f"  prior stop_reason     : {prior_stop or '(none)'}")
        # CRITICAL: a leftover stop_reason from the prior run (most often
        # "time_exhausted") fools Orchestration into thinking the work is
        # already done — it just heartbeats forever. Clear it so the new
        # run has a clean signal. The Coordinator's run() always re-sets
        # stop_reason at exit anyway.
        prior_crash = state.crash_count
        if prior_stop or prior_crash >= 3:
            state.stop_reason = ""
            # Reset persisted crash_count so a fresh resume isn't immediately
            # tripped into "emergency" by accumulated failures from prior runs
            # (e.g. authentication errors before .env was loaded).
            state.crash_count = 0
            state.save(session_dir)
            print(
                f"  → cleared stop_reason and reset crash_count "
                f"(was {prior_crash}) for fresh resume"
            )
    else:
        # Resolve model path from --model first, then $MODEL_PATH env. Without
        # either, fail fast: silently falling back to the YAML's hardcoded
        # `/wekafs/models/Qwen-Qwen3-8B` was the cause of "the optimizer ran
        # the wrong model" reports — explicit > implicit.
        if not args.model:
            args.model = os.environ.get("MODEL_PATH") or ""
        if not args.model:
            print(
                "ERROR: model is required. Pass --model <path> or set "
                "MODEL_PATH env (or use --resume <session_id>).",
                file=sys.stderr,
            )
            sys.exit(2)
        # Re-export the resolved value so downstream subprocess executors
        # (baseline / profile / sweep / backends / params) inject it into
        # the Magpie YAML instead of trusting the YAML's hardcoded `model:`.
        os.environ["MODEL_PATH"] = str(args.model)

        # Resolve GPU runner type: --gpu-type > $GPU_TYPE > rocm-smi probe.
        # Result is the canonical Magpie label (mi300x / mi355x). MI325X has
        # the same architecture as MI300X but Magpie does not yet ship
        # sglang_mi325x.sh / vllm_mi325x.sh, so we map mi325x -> mi300x with
        # a warning so the run actually succeeds.
        gpu_type = (args.gpu_type or os.environ.get("GPU_TYPE", "")).strip().lower()
        if not gpu_type:
            gpu_type = _autodetect_gpu_type() or ""
            if gpu_type:
                print(f"GPU type        : {gpu_type} (auto-detected)")
        if gpu_type == "mi325x":
            print(
                "WARN: mi325x maps to mi300x (same arch; Magpie has no "
                "sglang_mi325x.sh / vllm_mi325x.sh yet)",
                file=sys.stderr,
            )
            gpu_type = "mi300x"
        if gpu_type:
            os.environ["GPU_TYPE"] = gpu_type
            print(f"GPU type        : {gpu_type} (will inject runner_type into Magpie YAML)")
        else:
            os.environ.pop("GPU_TYPE", None)
            print("GPU type        : <unset> (Magpie will auto-detect)")
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
        critic_mock=args.critic_mock,
    )
    # Bug A fix: expose the active session_dir to in-process executors
    # (e.g. ReportExecutor) that don't get session_dir threaded through
    # task.params. This is read in report.py::_resolve_session_dir.
    os.environ["INFERENCE_OPTIMIZER_SESSION_DIR"] = str(session_dir)

    coordinator = Coordinator(session_dir, backends=backends)
    coordinator.system_prompt_overrides = {
        "orchestration": args.orch_prompt or _DEFAULT_ORCH_PROMPT,
        "critic":        args.critic_prompt or _DEFAULT_CRITIC_PROMPT,
        "kernel":        args.kernel_prompt or _DEFAULT_KERNEL_PROMPT,
    }
    _register_executors(coordinator)

    print(f"Backends        : "
          f"orchestration=Claude({args.claude_model}), "
          f"kernel={'Codex' if args.kernel_codex else 'Claude'}, "
          f"critic={'mock' if args.critic_mock else f'Codex({args.codex_model})'}, "
          f"robustness=mock")
    print(f"Max ticks       : {args.max_ticks or 'unlimited'} "
          f"(budget = {args.max_hours}h)")
    print(f"Tick interval   : {args.tick_interval_sec}s")
    print()

    try:
        stop_reason = await coordinator.run(
            objective=objective,
            max_minutes=args.max_hours * 60.0,
            tick_interval_sec=args.tick_interval_sec,
            max_ticks=args.max_ticks,
            install_signal_handlers=True,
        )
    finally:
        await coordinator.stop()

    _print_final_summary(coordinator.shared_state, stop_reason)
    return 0 if stop_reason in (
        "target_reached",
        "no_more_leverage",
        "time_exhausted",
        "max_ticks",
    ) else 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="inference_optimizer",
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
    opt.add_argument(
        "--gpu-type", choices=["mi300x", "mi325x", "mi355x"], default=None,
        help="Override GPU runner type passed to Magpie (sets benchmark."
             "runner_type). When omitted, the optimizer auto-detects via "
             "rocm-smi; falls back to Magpie's own auto-detection if rocm-smi "
             "is unavailable. mi325x is treated as mi300x (same architecture; "
             "Magpie does not yet ship sglang_mi325x.sh / vllm_mi325x.sh).",
    )
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
                           "SharedState seed and lets the Coordinator replay "
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
    opt.add_argument("--critic-mock", action="store_true", default=False,
                      help="Use mock Critic auto-approval when Codex credentials "
                           "are unavailable. Intended for validation runs.")
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
