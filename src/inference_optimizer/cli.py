"""CLI entry point for the Inference Optimizer skill.

Usage::

    python -m inference_optimizer \
        --model /path/to/model \
        --max-hours 0.05 \
        --target-gain-pct 5 \
        [--backend mock] \
        [--session-id resume-abc123]

Run modes (DESIGN §3.4):
    quick     <2h     param sweep + minimal kernel
    guided    2-6h    + critic + sage + watchdog
    marathon  >6h     + persona distill + KB synthesis

Status (v0.7):
    ``--backend mock``   end-to-end dry-run (no API key, no network).
    ``--backend claude`` real Claude via ``claude-agent-sdk``.
                         Requires Node.js >=18 and the ``claude`` CLI;
                         add ``--auto-install`` to fetch them into
                         ``~/.cache/inference-optimizer/`` automatically.
    ``--backend codex``  real Codex / GPT via the ``openai`` SDK
                         (no-tools, ``validated_json_output``).
                         Reads ``OPENAI_API_KEY`` from env. Use this for
                         Critic / Sage roles in guided / marathon.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import uuid
from pathlib import Path

from .bootstrap import (
    BootstrapError,
    InstallReport,
    MissingDependency,
    ensure_claude_cli,
)
from .orchestrator.action_registry import ActionRegistry, ActionRegistryError
from .orchestrator.backends import Backend, MockBackend
from .orchestrator.conductor import Conductor
from .orchestrator.env_probe import fill_default_env, probe_environment
from .paths import (
    SkillRootNotFound,
    db_path_for,
    make_session_dir,
    session_root,
    skill_actions_dir,
)


# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m inference_optimizer",
        description="Inference Optimizer — unified multi-agent skill",
    )

    # required
    p.add_argument("--model", required=True,
                   help="path or HF id of the target model (MODEL_PATH)")
    p.add_argument("--max-hours", type=float, required=True,
                   help="time budget; selects mode "
                        "(quick<2 / guided 2-6 / marathon>6)")

    # objective (at most one)
    p.add_argument("--target-gain-pct", type=float, default=None,
                   help="absolute pp gain over baseline (TARGET_GAIN_PCT)")
    p.add_argument("--target-tput-per-gpu", type=float, default=None,
                   help="absolute tok/s/GPU target (TARGET_TPUT_PER_GPU)")
    p.add_argument("--target-dir", type=str, default=None,
                   help="baseline session dir to match (TARGET_DIR)")

    # session lifecycle
    p.add_argument("--session-id", type=str, default=None,
                   help="resume an existing session by id (default: new uuid)")
    p.add_argument("--session-root", type=str, default=None,
                   help="override INFERENCE_OPTIMIZER_SESSION_ROOT for this run")

    # backend
    p.add_argument("--backend", choices=("mock", "claude", "codex"),
                   default="mock",
                   help="agent backend (mock | claude | codex)")
    p.add_argument("--mock-default-topic", type=str, default="heartbeat",
                   help="topic the MockBackend uses for default emits")
    p.add_argument("--claude-model", type=str, default=None,
                   help="claude model id, e.g. 'claude-opus-4-7' "
                        "(default: SDK default / ANTHROPIC_MODEL)")
    p.add_argument("--codex-model", type=str, default=None,
                   help="codex/openai model id, e.g. 'gpt-5.4' "
                        "(default: OPENAI_MODEL env / SDK default)")
    p.add_argument("--codex-base-url", type=str, default=None,
                   help="optional OpenAI-compatible endpoint "
                        "(Azure / proxy / Foundry)")

    # bootstrap
    p.add_argument("--auto-install", action="store_true",
                   help="if Node.js or the claude CLI is missing, install "
                        "them into ~/.cache/inference-optimizer (no sudo)")
    p.add_argument("--no-auto-install", dest="auto_install",
                   action="store_false",
                   help="explicitly disable bootstrap auto-install")
    p.set_defaults(auto_install=None)
    p.add_argument("--bootstrap-cache-dir", type=str, default=None,
                   help="override ~/.cache/inference-optimizer for "
                        "auto-installed Node + claude CLI")

    # cadence (mostly for tests)
    p.add_argument("--reactor-tick-s", type=float, default=None,
                   help="reactor wake interval (default 2.0s)")
    p.add_argument("--clock-tick-s", type=float, default=None,
                   help="clock + heartbeat interval (default 5.0s)")

    # action registry / executor wiring
    p.add_argument("--no-action-registry", action="store_true",
                   help="skip loading the action catalogue (no dispatcher, "
                        "no executors). Useful for the very first dry-run.")
    p.add_argument("--actions-dir", type=str, default=None,
                   help="override path to the action metadata directory "
                        "(default: <skill>/actions/)")

    # GPU / framework env (passed through to ActionExecutor scripts)
    p.add_argument("--tp", type=int, default=None,
                   help="tensor-parallel size (default: GPU_COUNT auto-probe)")
    p.add_argument("--conc", type=int, default=None,
                   help="benchmark concurrency (default: derived from TP)")
    p.add_argument("--isl", type=int, default=None,
                   help="input sequence length (default: 1024)")
    p.add_argument("--osl", type=int, default=None,
                   help="output sequence length (default: 256)")
    p.add_argument("--port", type=int, default=None,
                   help="server port for sglang/vllm (default: 8888)")
    p.add_argument("--framework", choices=("sglang", "vllm"), default=None,
                   help="serving framework (default: auto — prefer sglang)")
    p.add_argument("--inferencex-path", type=str, default=None,
                   help="path to the InferenceX checkout (must contain "
                        "benchmarks/benchmark_lib.sh and utils/bench_serving/)")
    p.add_argument("--no-env-probe", action="store_true",
                   help="skip the GPU / framework auto-probe at startup")

    # observability
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))

    return p


def _build_env(args: argparse.Namespace) -> dict[str, str]:
    """Translate CLI args into the env-block contract of DESIGN §16.

    We populate two layered variable names:

    * ``MODEL_PATH`` / ``MAX_HOURS`` — the conductor's contract
      (objective / mode selection).
    * ``MODEL`` / ``TP`` / ``CONC`` / ``ISL`` / ``OSL`` /
      ``INFERENCEX_PATH`` / ``PORT`` / ``FRAMEWORK`` — the
      shell-script contract used by ``run_baseline.sh``,
      ``run_profile.sh``, ``run_sweep.sh`` (and resolved by every
      :class:`ActionExecutor` via ``ctx.require_env``).

    Both layers see the same value: ``MODEL`` mirrors ``MODEL_PATH``
    so the operator only needs to set one knob.
    """
    env: dict[str, str] = {
        "MODEL_PATH": args.model,
        "MODEL": args.model,
        "MAX_HOURS": str(args.max_hours),
    }
    targets = [
        ("TARGET_GAIN_PCT", args.target_gain_pct),
        ("TARGET_TPUT_PER_GPU", args.target_tput_per_gpu),
        ("TARGET_DIR", args.target_dir),
    ]
    set_targets = [(k, v) for k, v in targets if v not in (None, "")]
    if len(set_targets) > 1:
        raise SystemExit(
            f"At most one target may be set; got {[k for k, _ in set_targets]!r}"
        )
    for k, v in set_targets:
        env[k] = str(v)

    # Pass-through GPU / framework knobs (only set when the operator
    # supplied an explicit value — auto-probe fills in the rest).
    pairs: list[tuple[str, object]] = [
        ("TP", args.tp),
        ("CONC", args.conc),
        ("ISL", args.isl),
        ("OSL", args.osl),
        ("PORT", args.port),
        ("FRAMEWORK", args.framework),
        ("INFERENCEX_PATH", args.inferencex_path),
    ]
    for k, v in pairs:
        if v not in (None, ""):
            env[k] = str(v)
    return env


def _build_backend(args: argparse.Namespace) -> Backend:
    if args.backend == "mock":
        return MockBackend(default_topic=args.mock_default_topic)
    if args.backend == "claude":
        # Imported lazily so the rest of the CLI works even if the SDK
        # is not installed (e.g. running --backend mock).
        from .orchestrator.backends.claude import ClaudeBackend

        return ClaudeBackend(model=args.claude_model)
    if args.backend == "codex":
        from .orchestrator.backends.codex import CodexBackend

        model = args.codex_model or os.environ.get("OPENAI_MODEL")
        return CodexBackend(model=model, base_url=args.codex_base_url)
    raise SystemExit(
        f"backend={args.backend!r} is not wired yet "
        "(see IMPLEMENTATION-CHECKLIST Phase 6)"
    )


def _resolve_auto_install(args: argparse.Namespace) -> bool:
    """Resolve --auto-install / --no-auto-install / env var precedence.

    Order:
        1. CLI flag, if explicitly given.
        2. ``INFERENCE_OPTIMIZER_AUTO_INSTALL`` env (``1`` / ``true`` / ``yes``).
        3. Default ``False``.
    """
    if args.auto_install is not None:
        return bool(args.auto_install)
    raw = os.environ.get("INFERENCE_OPTIMIZER_AUTO_INSTALL", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _bootstrap_for_backend(args: argparse.Namespace, log: logging.Logger) -> InstallReport | None:
    """If the backend needs the claude CLI, run the bootstrap probe/install."""
    if args.backend != "claude":
        return None
    cache = Path(args.bootstrap_cache_dir).expanduser() if args.bootstrap_cache_dir else None
    auto = _resolve_auto_install(args)
    log.info(
        "bootstrap: backend=claude auto_install=%s cache_dir=%s",
        auto,
        cache,
    )
    try:
        report = ensure_claude_cli(auto_install=auto, cache_dir=cache)
    except MissingDependency as exc:
        sys.stderr.write(str(exc) + "\n\n")
        sys.stderr.write(
            "Re-run with --auto-install to download Node.js + the Claude CLI "
            "into a per-user cache directory (no sudo).\n"
        )
        raise SystemExit(2) from exc
    except BootstrapError as exc:
        sys.stderr.write(f"bootstrap failed: {exc}\n")
        raise SystemExit(2) from exc

    log.info("bootstrap report:\n%s", report.summary())
    return report


def _build_action_registry(
    args: argparse.Namespace, log: logging.Logger
) -> ActionRegistry | None:
    """Load the action catalogue once at startup so dispatcher_loop runs
    and ``ActionExecutor`` can be looked up by name. Returns ``None`` when
    the user opts out (``--no-action-registry``) or when the catalogue
    cannot be located (logged at INFO level — the conductor still works
    without it, just without sub-agent dispatch)."""
    if args.no_action_registry:
        log.info("action registry: disabled via --no-action-registry")
        return None
    try:
        actions_dir = (
            Path(args.actions_dir).expanduser()
            if args.actions_dir
            else skill_actions_dir()
        )
    except SkillRootNotFound as exc:
        log.warning("action registry: cannot locate skill root: %s", exc)
        return None
    if not actions_dir.is_dir():
        log.warning("action registry: actions dir not found: %s", actions_dir)
        return None
    try:
        reg = ActionRegistry(actions_dir).load()
    except ActionRegistryError as exc:
        log.warning("action registry: load failed (%s); continuing without it", exc)
        return None
    log.info(
        "action registry: loaded %d actions from %s",
        len(reg), actions_dir,
    )
    return reg


def _resolve_session_dir(args: argparse.Namespace) -> Path:
    if args.session_root:
        os.environ["INFERENCE_OPTIMIZER_SESSION_ROOT"] = args.session_root
    sid = args.session_id or uuid.uuid4().hex[:12]
    if args.session_id:
        # honour a pre-existing dir if it matches; otherwise create.
        existing = session_root() / sid
        if existing.exists():
            return existing
    return make_session_dir(sid)


# ---------------------------------------------------------------------------
async def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    log = logging.getLogger("inference_optimizer.cli")

    session_dir = _resolve_session_dir(args)
    env = _build_env(args)

    bootstrap_report = _bootstrap_for_backend(args, log)
    backend = _build_backend(args)
    action_registry = _build_action_registry(args, log)

    # GPU / framework auto-probe — only fills defaults the operator
    # didn't already pin via CLI flags.
    if not args.no_env_probe:
        probe = probe_environment(env=os.environ)
        log.info(
            "env probe: gpu_count=%s gpu_type=%s framework=%s "
            "rocm_smi=%s amd_smi=%s nvidia_smi=%s",
            probe.gpu_count, probe.gpu_type, probe.framework,
            probe.rocm_smi, probe.amd_smi, probe.nvidia_smi,
        )
        env = fill_default_env(env, probe)

    print(
        "[inference-optimizer] starting\n"
        f"  session_dir : {session_dir}\n"
        f"  db          : {db_path_for(session_dir)}\n"
        f"  model       : {env['MODEL_PATH']}\n"
        f"  max_hours   : {env['MAX_HOURS']}\n"
        f"  backend     : {args.backend}\n"
        f"  actions     : "
        + (f"{len(action_registry)} loaded\n" if action_registry else "(none — dispatcher disabled)\n")
        + f"  gpu         : count={env.get('GPU_COUNT', '?')} "
        + f"type={env.get('GPU_TYPE', '?')}\n"
        + f"  framework   : {env.get('FRAMEWORK', '?')} "
        + f"(version={env.get('FRAMEWORK_VERSION', '?')})\n"
        + f"  config      : tp={env.get('TP', '?')} "
        + f"conc={env.get('CONC', '?')} "
        + f"isl={env.get('ISL', '?')} osl={env.get('OSL', '?')} "
        + f"port={env.get('PORT', '?')}\n"
        + f"  inferencex  : {env.get('INFERENCEX_PATH', '(unset — executors will fall back to LLM)')}\n"
        + f"  log_level   : {args.log_level}\n"
        + (
            f"  bootstrap   : node={bootstrap_report.probe_after.node_path} "
            f"claude={bootstrap_report.probe_after.claude_path}\n"
            if bootstrap_report
            else ""
        ),
        file=sys.stderr,
        flush=True,
    )

    conductor = Conductor(
        session_dir,
        backend=backend,
        env=env,
        action_registry=action_registry,
        reactor_tick_s=args.reactor_tick_s,
        clock_tick_s=args.clock_tick_s,
    )

    loop = asyncio.get_running_loop()
    stop_signum: list[int] = []

    def _on_signal(signum: int) -> None:
        log.warning("signal %s received → graceful stop", signum)
        stop_signum.append(signum)
        if conductor.ctx is not None:
            conductor.ctx.state.set_stopping("emergency")

    if hasattr(signal, "SIGINT"):
        try:
            loop.add_signal_handler(signal.SIGINT, _on_signal, signal.SIGINT)
        except NotImplementedError:
            # Windows Proactor loop doesn't support add_signal_handler.
            signal.signal(signal.SIGINT,
                          lambda s, _f: _on_signal(s))  # noqa: ARG005

    ctx = await conductor.run()

    print(
        "[inference-optimizer] stopped\n"
        f"  reason         : {ctx.state.stop_reason}\n"
        f"  elapsed_min    : {ctx.state.elapsed_minutes:.2f}\n"
        f"  cumulative_gain: {ctx.state.cumulative_gain:.2f}%\n"
        f"  session_dir    : {session_dir}\n",
        file=sys.stderr,
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
