"""Marathon harness — CLI entry point, 4 async tasks, graceful shutdown."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .state import MarathonState
from .llm import LLMClient
from .ipc import init_ipc_files
from .oob_backends import OOBBackends
from .orchestrator import Orchestrator
from .kernel_manager import KernelManager
from .watchdog import Watchdog
from .dashboard import Dashboard

log = logging.getLogger(__name__)


def _has_launch_script(scripts_dir: Path) -> bool:
    """Check if a scripts/ dir contains any server launch script (by content, not name)."""
    from .workload import _find_script_by_content
    return _find_script_by_content(scripts_dir, "launch") is not None


def _detect_warm_start_mode(base_dir: str | None) -> str:
    """Detect warm-start mode from existing data.

    Sprint output can be either:
      1. A dir with handoff/config.json (structured handoff)
      2. A standalone repo with scripts/ containing serve/launch .sh files
    Both are treated as "sprint" mode.
    """
    if not base_dir:
        return "cold"
    bd = Path(base_dir)
    if (bd / "handoff" / "config.json").exists():
        return "sprint"
    if _has_launch_script(bd / "scripts"):
        return "sprint_repo"
    if bd.exists() and any(bd.iterdir()):
        return "baseline"
    return "cold"


def _make_session_dir(base_dir: str) -> str:
    """Create timestamped session dir as a sibling of base_dir.

    Sessions are stored OUTSIDE the Sprint repo to keep it immutable:
      /path/to/glm5-optimized/            ← Sprint repo (read-only)
      /path/to/glm5-optimized-sessions/   ← Marathon sessions
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    bd = Path(base_dir).resolve()
    sessions_root = bd.parent / f"{bd.name}-sessions"
    d = sessions_root / ts
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _snapshot_session_environment(session_dir: str, base_dir: str, args: Any) -> None:
    """Capture full environment snapshot so every session is reproducible.

    Writes to session_dir/environment/:
      - env_vars.json        — all environment variables at launch
      - launch_args.json     — CLI args used to start the marathon
      - serve_script.sh      — copy of the launch script
      - bench_script.sh      — copy of the benchmark script
      - pip_freeze.txt       — installed Python packages
      - rocm_info.txt        — GPU hardware/driver info
      - git_state.json       — git branch, commit, dirty files for base_dir
      - system_packages.json — key system package versions (aiter, vllm, torch)
    """
    env_dir = Path(session_dir) / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    import subprocess

    # 1. All env vars
    (env_dir / "env_vars.json").write_text(
        json.dumps(dict(os.environ), indent=2, sort_keys=True))

    # 2. CLI args
    try:
        (env_dir / "launch_args.json").write_text(
            json.dumps(vars(args), indent=2, default=str))
    except Exception:
        pass

    # 3. Copy sprint scripts
    scripts_dir = Path(base_dir) / "scripts"
    if scripts_dir.is_dir():
        for script in scripts_dir.glob("*.sh"):
            shutil.copy2(script, env_dir / script.name)

    # 4. Copy optimization CSVs / config files
    opt_dir = Path(base_dir) / "optimizations"
    if opt_dir.is_dir():
        opt_snap = env_dir / "optimizations"
        opt_snap.mkdir(exist_ok=True)
        for f in opt_dir.iterdir():
            if f.is_file() and f.stat().st_size < 1_000_000:
                shutil.copy2(f, opt_snap / f.name)

    # 5. pip freeze
    try:
        result = subprocess.run(
            ["pip", "freeze"], capture_output=True, text=True, timeout=30)
        (env_dir / "pip_freeze.txt").write_text(result.stdout)
    except Exception:
        pass

    # 6. ROCm / GPU info
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showdriverversion"],
            capture_output=True, text=True, timeout=15)
        (env_dir / "rocm_info.txt").write_text(result.stdout + result.stderr)
    except Exception:
        pass

    # 7. Git state of base_dir
    try:
        git_info = {}
        for key, cmd in [
            ("branch", ["git", "-C", base_dir, "rev-parse", "--abbrev-ref", "HEAD"]),
            ("commit", ["git", "-C", base_dir, "rev-parse", "HEAD"]),
            ("diff_stat", ["git", "-C", base_dir, "diff", "--stat"]),
        ]:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            git_info[key] = r.stdout.strip()
        r = subprocess.run(
            ["git", "-C", base_dir, "status", "--porcelain"],
            capture_output=True, text=True, timeout=10)
        git_info["dirty_files"] = r.stdout.strip().splitlines()
        (env_dir / "git_state.json").write_text(json.dumps(git_info, indent=2))
    except Exception:
        pass

    # 8. Key system package versions
    pkg_versions = {}
    for pkg in ["vllm", "torch", "aiter", "triton", "transformers"]:
        try:
            mod = __import__(pkg)
            pkg_versions[pkg] = getattr(mod, "__version__", "installed (no __version__)")
        except ImportError:
            pkg_versions[pkg] = "NOT INSTALLED"
    (env_dir / "system_packages.json").write_text(json.dumps(pkg_versions, indent=2))

    # 9. Snapshot key system files that optimizations may modify
    sys_files = {}
    for path_str in [
        "/usr/local/lib/python3.12/dist-packages/aiter/fused_moe.py",
        "/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/fused_moe.py",
    ]:
        p = Path(path_str)
        if p.exists():
            try:
                import hashlib
                content = p.read_bytes()
                sys_files[path_str] = {
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size": len(content),
                    "mtime": p.stat().st_mtime,
                }
            except Exception:
                pass
    if sys_files:
        (env_dir / "system_files.json").write_text(json.dumps(sys_files, indent=2))

    log.info("Session environment snapshot saved to %s (%d files)",
             env_dir, len(list(env_dir.rglob("*"))))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="marathon",
        description="Marathon Inference Optimization Harness — 24h autonomous GPU performance optimizer.",
    )
    p.add_argument("model_name", help="Model identifier (e.g. DeepSeek-R1-0528)")
    p.add_argument("base_dir", help="Base result directory (with handoff/, baseline trace, etc.)")
    p.add_argument("--model-class", default="moe_mla",
                   choices=["dense", "moe_mla", "moe_swa", "moe_mla_nsa"],
                   help="Model architecture class for scoring priors")
    p.add_argument("--gpu-type", default="MI355X", help="GPU type")
    p.add_argument("--gpu-count", type=int, default=8, help="GPU count")
    p.add_argument("--tp", type=int, default=8, help="Tensor parallelism degree")
    p.add_argument("--framework", default="sglang",
                   help="Inference framework (e.g. sglang, vllm, atom, tensorrt_llm, lmdeploy)")
    p.add_argument("--session-dir", default="",
                   help="Resume from existing session dir (skip creation)")
    p.add_argument("--llm-model", default="claude-sonnet-4-20250514",
                   help="LLM model for scoped calls")
    p.add_argument(
        "--env-file", default="",
        help="Path to .env (API keys). If omitted, uses TBO/.env or ./.env when that file exists.",
    )
    p.add_argument("--inferencex-path", default="",
                   help="Path to InferenceX benchmarking repo (auto-detected from TBO if not set)")
    p.add_argument("--max-cost-usd", type=float, default=0,
                   help="Graceful shutdown when LLM cost exceeds this (0 = unlimited)")
    p.add_argument("--max-hours", type=float, default=24,
                   help="Wall-clock time limit in hours (default: 24)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint in session dir")
    p.add_argument(
        "--claw-url", default="",
        help="Primus-Claw backend URL (e.g. http://localhost:8000). "
             "When set, LLM calls route through Claw sessions instead of "
             "calling claude_code_sdk directly — no Anthropic API key needed.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print config and exit")
    return p


def _load_env(env_file: str) -> dict[str, str]:
    """Load env vars from .env file, merge with os.environ."""
    env = dict(os.environ)
    if env_file and Path(env_file).exists():
        for line in Path(env_file).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                key, val = k.strip(), v.strip().strip("'\"")
                if val:
                    env[key] = val
    return env


def _resolve_env_file(explicit: str) -> str:
    """Use explicit --env-file if set; else TBO/.env or cwd/.env when present."""
    if explicit and Path(explicit).exists():
        return explicit
    tbo_root = Path(__file__).resolve().parents[2]
    for candidate in (tbo_root / ".env", Path.cwd() / ".env"):
        if candidate.is_file():
            return str(candidate)
    return explicit


def _apply_env_to_process(env: dict[str, str]) -> None:
    """Copy merged env into os.environ so subprocesses see API keys."""
    os.environ.update(env)


def _resolve_inferencex_path(explicit: str) -> str:
    """Resolve InferenceX path: explicit arg > sibling dir > hardcoded fallback."""
    if explicit and Path(explicit).is_dir():
        return str(Path(explicit).resolve())
    pkg_root = Path(__file__).resolve().parent
    candidates = [
        pkg_root.parent / "InferenceX",
        Path("/shared_nfs/nehaprakriya/TBO/inference_optimization/InferenceX"),
    ]
    for c in candidates:
        if (c / "benchmarks").is_dir():
            return str(c)
    return ""


def _resolve_claw_url(explicit: str, env: dict[str, str]) -> str:
    """Resolve Claw URL: explicit --claw-url > CLAW_URL env var."""
    if explicit:
        return explicit.rstrip("/")
    from_env = env.get("CLAW_URL", "")
    return from_env.rstrip("/") if from_env else ""


def _is_marathon_process(pid: int) -> bool:
    """Check /proc/{pid}/cmdline to verify this is actually a marathon process."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
        return "marathon_harness" in cmdline or "marathon" in cmdline
    except (OSError, PermissionError):
        return False


def _acquire_session_lock(session_dir: str) -> Path:
    """Write a PID lockfile; kill any stale holder first.

    Prevents multiple marathon processes from writing to the same
    session directory concurrently (the root cause of state.json
    flip-flopping).

    Uses /proc/{pid}/cmdline verification to avoid killing unrelated
    processes that recycled the PID.
    """
    lock_path = Path(session_dir) / ".marathon.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if lock_path.exists():
        try:
            old_pid = int(lock_path.read_text().strip())
            if old_pid != os.getpid():
                os.kill(old_pid, 0)  # raises OSError if dead
                if not _is_marathon_process(old_pid):
                    log.warning(
                        "PID %d is alive but is NOT a marathon process (recycled PID) — "
                        "removing stale lock",
                        old_pid,
                    )
                else:
                    log.warning(
                        "Stale marathon (PID %d) still running on this session — sending SIGTERM",
                        old_pid,
                    )
                    os.kill(old_pid, signal.SIGTERM)
                    for _ in range(30):
                        time.sleep(0.5)
                        try:
                            os.kill(old_pid, 0)
                        except OSError:
                            break
                    else:
                        if _is_marathon_process(old_pid):
                            log.warning("PID %d did not exit after 15s — sending SIGKILL", old_pid)
                            os.kill(old_pid, signal.SIGKILL)
                            time.sleep(1)
                        else:
                            log.warning("PID %d is no longer a marathon process — skipping SIGKILL", old_pid)
        except (ValueError, OSError):
            pass

    lock_path.write_text(str(os.getpid()))
    return lock_path


async def async_main(args: argparse.Namespace) -> None:
    env = _load_env(args.env_file)
    _apply_env_to_process(env)

    claw_url = _resolve_claw_url(args.claw_url, env)

    if claw_url:
        log.info("LLM backend: Primus-Claw @ %s", claw_url)
    elif env.get("ANTHROPIC_API_KEY"):
        log.info("LLM backend: claude_code_sdk (ANTHROPIC_API_KEY set, len=%d)",
                 len(env["ANTHROPIC_API_KEY"]))
    else:
        log.warning("No LLM backend configured — set --claw-url or ANTHROPIC_API_KEY")

    inferencex_path = _resolve_inferencex_path(args.inferencex_path)
    if inferencex_path:
        log.info("InferenceX path: %s", inferencex_path)
    else:
        log.warning("InferenceX path not found — benchmarking may fail")

    session_dir = args.session_dir or _make_session_dir(args.base_dir)
    log.info("Session dir: %s", session_dir)

    lock_path = _acquire_session_lock(session_dir)
    log.info("Session lock acquired (PID %d)", os.getpid())

    init_ipc_files(session_dir)

    if args.resume and session_dir:
        ckpt_latest = Path(session_dir) / "checkpoints" / "latest"
        if ckpt_latest.exists():
            ckpt_target = ckpt_latest.resolve()
            log.info("Resuming from checkpoint: %s", ckpt_target)
            state = MarathonState.load(ckpt_target)
            state.session_dir = session_dir
        else:
            log.warning("--resume specified but no checkpoint found, creating fresh state")
            state = MarathonState.load_or_create(
                session_dir, model_name=args.model_name, model_class=args.model_class,
                base_dir=args.base_dir, gpu_type=args.gpu_type,
                gpu_count=args.gpu_count, tp=args.tp, framework=args.framework,
            )
    else:
        state = MarathonState.load_or_create(
            session_dir, model_name=args.model_name, model_class=args.model_class,
            base_dir=args.base_dir, gpu_type=args.gpu_type,
            gpu_count=args.gpu_count, tp=args.tp, framework=args.framework,
        )
    state.session_id = Path(session_dir).name
    state.save()

    _snapshot_session_environment(session_dir, args.base_dir, args)

    mode = _detect_warm_start_mode(args.base_dir)
    log.info("Model: %s | Class: %s | Warm-start: %s", args.model_name, args.model_class, mode)

    from .prompts import configure as configure_prompts
    system_prompt = configure_prompts(
        inferencex_path=inferencex_path or "(not set)",
        base_dir=args.base_dir,
        framework=args.framework,
    )

    # Each component gets its own LLM client tagged with its role so
    # Claw sessions are identifiable and isolated.
    llm_orch = LLMClient(
        model=args.llm_model, env=env, system_prompt=system_prompt,
        inferencex_path=inferencex_path, base_dir=args.base_dir,
        claw_url=claw_url, role="orchestrator",
    )
    llm_km = LLMClient(
        model=args.llm_model, env=env, system_prompt=system_prompt,
        inferencex_path=inferencex_path, base_dir=args.base_dir,
        claw_url=claw_url, role="kernel-manager",
    )
    llm_wd = LLMClient(
        model=args.llm_model, env=env, system_prompt=system_prompt,
        inferencex_path=inferencex_path, base_dir=args.base_dir,
        claw_url=claw_url, role="watchdog",
    )

    oob = OOBBackends(env=env, claw_url=claw_url)

    dashboard = Dashboard(state, session_dir)

    shutdown = asyncio.Event()

    def _handle_signal(sig: int, _: Any) -> None:
        log.info("Received signal %d, shutting down gracefully …", sig)
        shutdown.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    from . import server
    from .gpu_lock import GpuLock
    from .workload import kill_rogue_servers
    gpu_lock = GpuLock()
    gpu_lock.start_watchdog()

    # Kill any rogue inference servers left from previous runs or Claw sessions
    rogues = await kill_rogue_servers()
    if rogues:
        log.warning("Cleaned up %d rogue server(s) at marathon startup", rogues)

    orch = Orchestrator(
        state, llm_orch, session_dir, server, oob, dashboard, shutdown,
        max_cost_usd=args.max_cost_usd,
        max_wall_hours=args.max_hours,
        inferencex_path=inferencex_path,
        gpu_lock=gpu_lock,
    )
    km = KernelManager(state, llm_km, session_dir, oob, dashboard, shutdown,
                       gpu_lock=gpu_lock)
    wd = Watchdog(state, llm_wd, session_dir, env, dashboard, shutdown)

    if args.dry_run:
        print(json.dumps({
            "session_dir": session_dir,
            "model": args.model_name,
            "class": args.model_class,
            "mode": mode,
            "gpu": f"{args.gpu_count}x {args.gpu_type}",
            "llm_backend": "claw" if claw_url else "claude_code_sdk",
            "claw_url": claw_url or "(not set)",
            "state_file": str(Path(session_dir) / "state.json"),
        }, indent=2))
        return

    log.info("Launching: orchestrator + kernel_manager + watchdog + dashboard")

    _RESTART_DELAY_S = 3
    _MAX_RESTARTS = 5

    task_factories = {
        "orchestrator": lambda: orch.run(),
        "kernel_manager": lambda: km.run(),
        "watchdog": lambda: wd.run(),
        "dashboard": lambda: dashboard.run(shutdown),
    }
    restartable = {"kernel_manager", "watchdog", "dashboard"}
    restart_counts: dict[str, int] = {name: 0 for name in restartable}

    tasks = {
        name: asyncio.create_task(factory(), name=name)
        for name, factory in task_factories.items()
    }

    try:
        while not shutdown.is_set():
            done, _ = await asyncio.wait(
                tasks.values(), return_when=asyncio.FIRST_COMPLETED,
            )
            should_exit = False
            for d in done:
                name = d.get_name()
                exc = d.exception() if not d.cancelled() else None

                if name == "orchestrator":
                    if exc:
                        log.error("Orchestrator failed: %s", exc)
                    else:
                        log.info("Orchestrator finished normally")
                    should_exit = True
                elif name in restartable:
                    if exc:
                        log.error("Task %s crashed: %s", name, exc)
                    else:
                        log.warning("Task %s exited unexpectedly", name)
                    restart_counts[name] = restart_counts.get(name, 0) + 1
                    if restart_counts[name] <= _MAX_RESTARTS:
                        log.info("Restarting %s (attempt %d/%d) in %ds …",
                                 name, restart_counts[name], _MAX_RESTARTS, _RESTART_DELAY_S)
                        await asyncio.sleep(_RESTART_DELAY_S)
                        tasks[name] = asyncio.create_task(
                            task_factories[name](), name=name,
                        )
                    else:
                        log.error("Task %s exceeded max restarts (%d) — giving up on it",
                                  name, _MAX_RESTARTS)
                else:
                    should_exit = True

            if should_exit:
                break
    except asyncio.CancelledError:
        log.warning("Main wait cancelled — initiating shutdown")

    shutdown.set()
    pending = [t for t in tasks.values() if not t.done()]

    if pending:
        GRACEFUL_SHUTDOWN_S = 30
        log.info("Waiting up to %ds for %d agent(s) to finish gracefully…",
                 GRACEFUL_SHUTDOWN_S, len(pending))
        _, still_pending = await asyncio.wait(pending, timeout=GRACEFUL_SHUTDOWN_S)
        for p in still_pending:
            log.warning("Force-cancelling %s", p.get_name())
            p.cancel()
        await asyncio.gather(*still_pending, return_exceptions=True)

    state.checkpoint("final")
    state.save()

    lock_path.unlink(missing_ok=True)
    log.info("Session lock released")

    log.info(
        "Marathon complete. Gain: %.1f%% | Cost: $%.2f | LLM calls: %d",
        state.cumulative_gain_pct, state.total_llm_cost_usd, state.total_llm_calls,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args()
    args.env_file = _resolve_env_file(args.env_file)
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
