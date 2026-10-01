# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU validation: run the A/B arms one after another from the same framework snapshot.

Arms share the host's framework installs (sessions patch vLLM/aiter in place) and Magpie stops
every vLLM server on the host after a benchmark, so arms never overlap. Before each arm the
snapshot (when configured) is restored and the configured compile caches are cleared, so no arm
inherits another's work. An arm is not launched while a Ray cluster is reachable from its
environment: Hyperloom would reuse it, and its workers keep the environment, and so the code
tree, of whoever started it. When an arm's optimizer has exited, the driver stops the processes
whose environment carries that arm's USER_DATA_PATH (its Ray daemons and servers). It stops
nothing else: other leftover servers make the step wait, then fail, and a Ray cluster the round
did not start is stopped only under ``ab.stop_ray``.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

from meta_rsi.rsi.config import Arm
from meta_rsi.rsi.pipeline import RoundContext, StepFailed, read_env_file
from meta_rsi.rsi.state import utc_now

BUSY_MARKERS = ("vllm serve", "VLLM::", "hyperloom.inference_optimizer.cli")
OPTIMIZER_MARKER = "hyperloom.inference_optimizer.cli"
DRIVER_ENV_PREFIXES = (
    "HYPERLOOM_",
    "PULSE_",
    "ANTHROPIC_",
    "OPENAI_",
    "CLAUDE_",
    "KERNEL_AGENT",
    "MAGPIE_",
    "GEAK_",
)
DRIVER_ENV_NAMES = ("USER_DATA_PATH", "INFERENCEX_PATH", "PYTHONPATH", "VIRTUAL_ENV")
MAX_ARM_ATTEMPTS = 2
IDLE_WAIT_SEC = 300.0
IDLE_POLL_SEC = 20.0
VERSION_PACKAGES = ("vllm", "torch", "amd-aiter", "ray", "claude-agent-sdk")
VERSION_PROBE = """
import importlib.metadata as m, json, sys
def version(name):
    try:
        return m.version(name)
    except m.PackageNotFoundError:
        return None
print(json.dumps({name: version(name) for name in sys.argv[1:]}))
"""
RAY_PROBE = """
import subprocess, sys
from hyperloom.agents.kernel.tools.backends import ray_runtime
if sys.argv[1:] == ["stop"]:
    subprocess.run(["ray", "stop", "--force"], timeout=ray_runtime.DEFAULT_RAY_STOP_TIMEOUT_SEC, check=False)
print("reachable" if ray_runtime.ray_status_ok() else "unreachable")
"""
LEFTOVER_POLL_SEC = 1.0


def arm_dir(ctx: RoundContext, name: str) -> Path:
    return ctx.config.ab.sessions_dir / name


def _live(proc: Path) -> bool:
    """Whether a /proc entry is a running process rather than a zombie."""
    return (proc / "stat").read_text().rsplit(")", 1)[-1].split()[0] != "Z"


def _cmdline(proc: Path) -> str:
    return (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore").strip()


def busy_processes() -> list[str]:
    """Command lines of running vLLM servers or optimizers on this host (zombies excluded)."""
    found = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if not _live(proc):
                continue
            cmdline = _cmdline(proc)
        except OSError:
            continue
        if any(marker in cmdline for marker in BUSY_MARKERS):
            found.append(f"{proc.name}: {cmdline[:160]}")
    return found


def optimizer_alive(pid: int, udp: Path) -> bool:
    """Whether ``pid`` is still this arm's optimizer: not a zombie, and not a pid reused by anything else."""
    proc = Path("/proc") / str(pid)
    try:
        if not _live(proc):
            return False
        cmdline = _cmdline(proc)
    except OSError:
        return False
    return OPTIMIZER_MARKER in cmdline and str(udp / "optimizer_runs") in cmdline


def arm_processes(udp: Path) -> list[int]:
    """Live processes whose environment carries this arm's USER_DATA_PATH, i.e. everything the arm started."""
    entry = f"USER_DATA_PATH={udp}".encode()
    found = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if _live(proc) and entry in (proc / "environ").read_bytes().split(b"\0"):
                found.append(int(proc.name))
        except OSError:
            continue
    return found


def stop_arm_processes(ctx: RoundContext, udp: Path) -> None:
    """Stop what an arm left running once its optimizer is gone: TERM, then KILL after the grace period."""
    from hyperloom.agents.kernel.tools.backends.ray_runtime import DEFAULT_RAY_STOP_TIMEOUT_SEC

    pids = arm_processes(udp)
    if pids:
        ctx.log(f"stopping {len(pids)} processes left by the arm in {udp}")
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                continue
        waited = 0.0
        while pids and waited < DEFAULT_RAY_STOP_TIMEOUT_SEC:
            ctx.sleep(LEFTOVER_POLL_SEC)
            waited += LEFTOVER_POLL_SEC
            pids = arm_processes(udp)
        if not pids:
            return
    raise StepFailed(f"processes left by the arm in {udp} survived SIGKILL: {pids}")


def ray_reachable(ctx: RoundContext, env: dict[str, str], tree: Path, *mode: str) -> bool:
    """Hyperloom's own reachability test (``ray_status_ok``), run in the arm's environment and tree."""
    probe = ctx.runner.run(
        [str(ctx.config.ab.python), "-c", RAY_PROBE, *mode], cwd=tree, env=env, log=ctx.path("logs", "ab.log")
    )
    return probe.stdout.strip().splitlines()[-1:] == ["reachable"]


def ensure_no_ray(ctx: RoundContext, env: dict[str, str], tree: Path) -> None:
    """Refuse to launch an arm that would reuse a running Ray cluster, or stop it under ab.stop_ray."""
    if not ray_reachable(ctx, env, tree):
        return
    if not ctx.config.ab.stop_ray:
        raise StepFailed(
            "a Ray cluster is reachable from the arm's environment; the arm would reuse it, and its workers "
            "keep the environment (and code tree) of whoever started it. Stop it, or set ab.stop_ray: true "
            "on a node this round has to itself"
        )
    ctx.log("ab.stop_ray: running `ray stop --force`, which stops every Ray cluster on this node")
    if ray_reachable(ctx, env, tree, "stop"):
        raise StepFailed("a Ray cluster is still reachable after `ray stop --force`; stop it before the next arm")


def arm_env(ctx: RoundContext, arm: Arm, udp: Path, tree: Path) -> dict[str, str]:
    """A clean environment for one arm: the host's, minus driver and Hyperloom settings, plus the arm's."""
    ab, agent = ctx.config.ab, ctx.config.agent
    env = {k: v for k, v in os.environ.items() if not k.startswith(DRIVER_ENV_PREFIXES) and k not in DRIVER_ENV_NAMES}
    if agent.env_file:
        env.update(read_env_file(agent.env_file))
    claude = arm.models.get("claude", agent.model)
    kernel_root = str(tree / "src/hyperloom/agents/kernel")
    env.update(
        PATH=f"{ab.python.parent}:{env.get('PATH', '/usr/local/bin:/usr/bin:/bin')}",
        USER_DATA_PATH=str(udp),
        PYTHONPATH=str(tree / "src"),
        HYPERLOOM_KERNEL_AGENT_ROOT=kernel_root,
        KERNEL_AGENT_ROOT=kernel_root,
        CLAUDE_MODEL=claude,
        ANTHROPIC_MODEL=claude,
        GEAK_CLAUDE_MODEL=arm.models.get("geak", claude),
    )
    if ab.kernel_agent_env:
        env["KERNEL_AGENT_ENV"] = str(udp / "runtime/kernel-agent.env.sh")
    env.update(ab.env)
    return env


def kernel_agent_env_text(template: str, env: dict[str, str], python: Path) -> str:
    """The installer's kernel-agent env file with the arm's locations swapped in, by variable name."""
    udp = env["USER_DATA_PATH"]
    overrides = {
        "USER_DATA_PATH": udp,
        "HYPERLOOM_RUNTIME_DIR": f"{udp}/runtime",
        "KERNEL_AGENT_ENV": f"{udp}/runtime/kernel-agent.env.sh",
        "HYPERLOOM_ROOT": f"{udp}/runtime/source-mirrors",
        "HYPERLOOM_KERNEL_AGENT_ROOT": env["HYPERLOOM_KERNEL_AGENT_ROOT"],
        "KERNEL_AGENT_ROOT": env["KERNEL_AGENT_ROOT"],
        "PYTHONPATH": env["PYTHONPATH"],
        "MAGPIE_PYTHON": str(python),
        "GEAK_CLAUDE_MODEL": env["GEAK_CLAUDE_MODEL"],
    }
    lines, seen = [], set()
    for line in template.splitlines():
        name = line.removeprefix("export ").split("=", 1)[0].strip()
        if line.startswith("export ") and name in overrides:
            seen.add(name)
            line = f"export {name}='{overrides[name]}'"
        lines.append(line)
    lines += [f"export {k}='{v}'" for k, v in overrides.items() if k not in seen]
    return "\n".join(lines) + "\n"


def arm_args(ctx: RoundContext, arm: Arm, udp: Path, scenario: list[str]) -> list[str]:
    ab, agent = ctx.config.ab, ctx.config.agent
    args = list(scenario)
    if "--max-hours" in args:
        i = args.index("--max-hours")
        del args[i : i + 2]
    claude = arm.models.get("claude", agent.model)
    args += ["--max-hours", f"{ab.hours:g}", *ab.extra_args, "--claude-model", claude]
    specialist = arm.models.get("specialist", claude)
    if specialist != claude:
        args += ["--specialist-model", specialist]
    return [*args, "--launch-info-file", str(udp / "optimizer_runs/launch.json")]


def launch(ctx: RoundContext, arm: Arm, scenario: list[str]) -> subprocess.Popen:
    """Start one arm's optimizer in its own session; its record files go under optimizer_runs/."""
    ab, udp, tree = ctx.config.ab, arm_dir(ctx, arm.name), ctx.worktree(arm.tree)
    if udp.exists():
        raise StepFailed(f"{udp} already exists; move it away to run arm {arm.name} again")
    runs = udp / "optimizer_runs"
    runs.mkdir(parents=True)
    env, args = arm_env(ctx, arm, udp, tree), arm_args(ctx, arm, udp, scenario)
    if ab.kernel_agent_env:
        ka = udp / "runtime/kernel-agent.env.sh"
        ka.parent.mkdir(parents=True)
        ka.write_text(kernel_agent_env_text(ab.kernel_agent_env.read_text(), env, ab.python))
        ka.chmod(0o600)
    (runs / "cli_args.txt").write_text(" ".join(args) + "\n")
    (runs / "tree_revision.txt").write_text(ctx.git("log", "--oneline", "-1", cwd=tree) + "\n")
    (runs / "env_names.txt").write_text("\n".join(sorted(env)) + "\n")
    cmd = [str(ab.python), "-m", "hyperloom.inference_optimizer.cli", "--verbose", "optimize", *args]
    with open(runs / "run.log", "w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=tree,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    (runs / "run.pid").write_text(f"{proc.pid}\n")
    return proc


def take_snapshot(snapshot_dir: Path, paths: tuple[Path, ...]) -> bool:
    """Copy the framework installs once; returns False when a finished snapshot already exists."""
    if (snapshot_dir / "SNAPSHOT_DONE").exists():
        return False
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, src in enumerate(paths):
        copy = snapshot_dir / f"{i:02d}_{src.name}"
        if copy.exists():
            shutil.rmtree(copy)
        shutil.copytree(src, copy, symlinks=True)
        manifest.append({"source": str(src), "copy": copy.name})
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (snapshot_dir / "SNAPSHOT_DONE").write_text(utc_now() + "\n")
    return True


def swap_in(src: Path, dst: Path) -> None:
    """Replace ``dst`` with a copy of ``src``, never leaving ``dst`` half-written."""
    new, old = dst.with_name(f".{dst.name}.rsi-new"), dst.with_name(f".{dst.name}.rsi-old")
    for leftover in (new, old):
        if leftover.exists():
            shutil.rmtree(leftover)
    shutil.copytree(src, new, symlinks=True)
    if dst.exists():
        dst.rename(old)
    new.rename(dst)
    if old.exists():
        shutil.rmtree(old)


def restore_snapshot(snapshot_dir: Path) -> None:
    for entry in json.loads((snapshot_dir / "manifest.json").read_text()):
        swap_in(snapshot_dir / entry["copy"], Path(entry["source"]))


def clear_caches(caches: tuple[Path, ...]) -> None:
    """Empty each configured cache directory, keeping it in place; a symlinked one through its target."""
    for cache in caches:
        root = cache.resolve()
        if not root.exists():
            continue
        if not root.is_dir():
            raise StepFailed(f"ab.clear_caches entry {cache} is not a directory")
        try:
            for entry in root.iterdir():
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
        except OSError as exc:
            raise StepFailed(f"cannot clear {cache} before the arm: {exc}") from exc


def preflight(ctx: RoundContext) -> list[str]:
    """What would make an arm fail or not compare; also records the stack versions."""
    ab, problems = ctx.config.ab, []
    for arm in ab.arms:
        tree = ctx.worktree(arm.tree)
        env = arm_env(ctx, arm, arm_dir(ctx, arm.name), tree)
        for tool in ("claude", "vllm"):
            if not shutil.which(tool, path=env["PATH"]):
                problems.append(f"`{tool}` is not on arm {arm.name}'s PATH (specialists and Magpie call it by name)")
        if (tree / ".env").exists():
            problems.append(f"{tree}/.env would override the arm's settings; remove it")
        probe = ctx.runner.run(
            [str(ab.python), "-c", "import hyperloom.inference_optimizer.cli"], cwd=tree, env=env, check=False
        )
        if probe.returncode != 0:
            problems.append(f"arm {arm.name} cannot import the optimizer: {probe.stderr.strip().splitlines()[-1:]}")
    ab.sessions_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(ab.sessions_dir).free / 1e9
    if free_gb < ab.min_free_gb:
        problems.append(f"{ab.sessions_dir} has {free_gb:.0f} GB free; ab.min_free_gb is {ab.min_free_gb:g}")
    problems += [f"snapshot source {p} does not exist" for p in ab.snapshot_paths if not p.exists()]
    versions = ctx.runner.run([str(ab.python), "-c", VERSION_PROBE, *VERSION_PACKAGES], check=False)
    ctx.path("ab", "environment.json").write_text(versions.stdout or json.dumps({"error": versions.stderr[-500:]}))
    return problems


def wait_idle(ctx: RoundContext) -> None:
    waited = 0.0
    while busy := busy_processes():
        if waited >= IDLE_WAIT_SEC:
            raise StepFailed(
                "vLLM servers or optimizers are still running; stop them before the next arm:\n" + "\n".join(busy)
            )
        ctx.sleep(IDLE_POLL_SEC)
        waited += IDLE_POLL_SEC


def arm_outcome(udp: Path) -> str:
    """``finished`` when the arm's session stopped on its own, else ``interrupted``."""
    states = sorted(udp.glob("*/*/state.json"), key=lambda p: p.stat().st_mtime)
    if not states:
        return "interrupted"
    state = json.loads(states[-1].read_text())
    return "finished" if state.get("stop_reason") or state.get("phase") == "CLOSE" else "interrupted"


def run_arm(ctx: RoundContext, arm: Arm, rec: dict, scenario: list[str]) -> None:
    """Launch (or keep watching) one arm until its optimizer exits, then record the outcome."""
    ab, udp = ctx.config.ab, arm_dir(ctx, arm.name)
    proc = None
    if rec["status"] == "running" and not optimizer_alive(int(rec["pid"]), udp):
        stop_arm_processes(ctx, udp)
        rec["status"] = arm_outcome(udp)
        return
    if rec["status"] != "running":
        wait_idle(ctx)
        tree = ctx.worktree(arm.tree)
        ensure_no_ray(ctx, arm_env(ctx, arm, udp, tree), tree)
        if ab.snapshot_dir:
            restore_snapshot(ab.snapshot_dir)
        clear_caches(ab.clear_caches)
        proc = launch(ctx, arm, scenario)
        rec.update(status="running", pid=proc.pid, attempt=rec["attempt"] + 1, started=utc_now())
        ctx.state.save()
        ctx.log(f"arm {arm.name}: started (pid {proc.pid})")
    while (proc.poll() is None) if proc else optimizer_alive(int(rec["pid"]), udp):
        ctx.sleep(ab.poll_sec)
    stop_arm_processes(ctx, udp)
    rec.update(status=arm_outcome(udp), ended=utc_now())


def settle(ctx: RoundContext, arm: Arm, rec: dict) -> None:
    """After an interrupted arm: keep its partial result, or move it aside for a rerun."""
    if rec["status"] != "interrupted":
        return
    if ctx.config.ab.on_interrupt == "keep" or rec["attempt"] >= MAX_ARM_ATTEMPTS:
        rec.update(status="finished", partial=True)
        return
    udp = arm_dir(ctx, arm.name)
    udp.rename(udp.with_name(f"{udp.name}.interrupted-{rec['attempt']}"))
    rec["status"] = "pending"


def ab(ctx: RoundContext) -> dict:
    """Run every configured arm to completion, resuming running arms after a driver restart."""
    cfg = ctx.config.ab
    scenario = json.loads((ctx.round_dir / "scenario.json").read_text())["args"]
    problems = preflight(ctx)
    if problems:
        raise StepFailed("A/B preflight failed:\n- " + "\n- ".join(problems))
    if cfg.snapshot_dir and take_snapshot(cfg.snapshot_dir, cfg.snapshot_paths):
        ctx.log(f"framework snapshot taken in {cfg.snapshot_dir}")
    arms = ctx.state.data.setdefault("ab", {})
    for arm in cfg.arms:
        rec = arms.setdefault(arm.name, {"status": "pending", "attempt": 0})
        while rec["status"] != "finished":
            run_arm(ctx, arm, rec, scenario)
            settle(ctx, arm, rec)
            ctx.state.save()
    return {"arms": arms}
