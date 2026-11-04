"""Preflight checks before a campaign starts."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from arbor import frameworks
from arbor.execute.watch import gpu_vram_used


def stage_weights(
    src: str | Path,
    dst: str | Path = "/tmp",
) -> Path:
    """Copy model weights from NFS to local disk for faster loading.

    Returns the local path. Skips if already staged and readable.
    Clears stale root-owned directories from prior docker cp runs.
    """
    import os
    src = Path(src)
    if not src.name:
        raise ValueError(f"src path has no filename component: {src!r}")
    local = Path(dst) / src.name
    # Guard: local must be a direct child of dst, never dst itself
    if local == Path(dst):
        raise ValueError(f"refusing rm -rf: local path {local} equals dst root")

    # Check if already staged and writable by us
    if local.exists() and (local / "config.json").exists():
        if os.access(local / "config.json", os.R_OK):
            print(f"[preflight] weights already staged at {local}")
            return local
        # Exists but not readable — likely from a root docker cp, must clear
        print(f"[preflight] stale weights at {local} (permission issue) — clearing with sudo")
        subprocess.run(["sudo", "rm", "-rf", str(local)], check=True)

    print(f"[preflight] staging weights: {src} → {local}  (~5 min)")
    shutil.copytree(src, local, dirs_exist_ok=True)
    print(f"[preflight] staged at {local}")
    return local


def stop_server(container: str, gpu_ids: list[int] | None = None,
                graceful_wait: int = 15, vram_check_timeout: int = 60) -> None:
    """Stop the serving server gracefully (SIGTERM, wait, verify VRAM released).

    NEVER use pkill -9 / SIGKILL — SGLang holds KFD GPU memory until the
    Python process exits cleanly. SIGKILL skips cleanup, leaving 250GB of
    zombie VRAM that blocks the next launch with a memory-imbalance error.

    Waits until GPU VRAM drops below 10GB before returning so the next
    launch doesn't fail with 'memory capacity unbalanced'.

    vLLM too: this signalled only `sglang.launch_server`, so on a vLLM
    campaign nothing stopped. Model M B on Cluster B (2026-10-07) kept a server and
    its EngineCore holding ~265 GB per GPU through every "stop"; the next
    launch failed or the harness reused the stale server. `vllm serve` is the
    API server, which takes its engine down on SIGTERM; an EngineCore or
    worker it leaves behind renames itself `VLLM::...`, so those get SIGTERM
    too once the grace period has passed.

    The patterns are every framework's (arbor/frameworks/), SGLang's first: a
    sweep that stops whichever server is up, whatever the config names.
    """
    import time
    for pattern in frameworks.union("stop_patterns"):
        subprocess.run(["docker", "exec", container, "pkill", "-TERM", "-f", pattern],
            capture_output=True)
    time.sleep(graceful_wait)
    for pattern in frameworks.union("post_grace_patterns"):
        subprocess.run(["docker", "exec", container, "pkill", "-TERM", "-f", pattern],
            capture_output=True)
    if gpu_ids is None:
        return
    deadline = time.time() + vram_check_timeout
    while time.time() < deadline:
        vram = gpu_vram_used(gpu_ids)
        if all(gb < 10 for gb in vram.values()):
            return
        time.sleep(5)
    vram = gpu_vram_used(gpu_ids)
    held = {k: v for k, v in vram.items() if v >= 10}
    if held:
        print(f"[stop_server] WARNING: GPUs still holding memory after {vram_check_timeout}s: {held}")


# ── Component mode: our own server, by its PID file ──────────────────────────

def _proc_stat(pid: int) -> tuple[str, int, float] | None:
    """(state, process group, start in clock ticks since boot) from /proc/<pid>/stat."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        rest = raw[raw.rindex(")") + 2:].split()   # the name may hold spaces and ")"
        return rest[0], int(rest[2]), float(rest[19])
    except (OSError, ValueError, IndexError):
        return None


def _started_at(pid: int) -> float | None:
    """When ``pid`` started, in epoch seconds (to about a second); None off Linux."""
    st = _proc_stat(pid)
    try:
        btime = next(int(l.split()[1]) for l in Path("/proc/stat").read_text().splitlines()
                     if l.startswith("btime "))
        return btime + st[2] / os.sysconf("SC_CLK_TCK") if st else None
    except (OSError, ValueError, StopIteration):
        return None


def group_alive(pgid: int) -> bool:
    """Whether process group ``pgid`` has a member that is not a zombie.

    Read from /proc where there is one: a zombie still answers ``kill(…, 0)``, and a
    leader the harness itself started stays one until something reaps it.
    """
    if Path("/proc/self/stat").exists():
        for d in Path("/proc").iterdir():
            if d.name.isdigit():
                st = _proc_stat(int(d.name))
                if st and st[1] == pgid and st[0] not in ("Z", "X"):
                    return True
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_of(pid_file: Path) -> int | None:
    try:
        pid = int(Path(pid_file).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return pid if pid > 1 else None


def signal_own_server(pid_file: str | Path, sig: int = signal.SIGTERM) -> int | None:
    """Send ``sig`` to the process group the PID file names; its id, or None when there
    is nothing of ours to signal (no file, a group that is gone, or a PID reused).

    The launch (arbor/frameworks/base.py) runs the server under ``setsid``, and the
    server's shell writes its own PID, so that PID is its group's id and the group is
    the server and its children (vLLM's EngineCore, SGLang's schedulers). A live
    process with that PID that started after the file was written, or that leads no
    group, is not ours: the PID was reused, and it is left alone. Never our own group.
    """
    pid_file = Path(pid_file)
    pgid = _pid_of(pid_file)
    if pgid is None or pgid in (os.getpid(), os.getpgrp()):
        return None
    if not group_alive(pgid):
        pid_file.unlink(missing_ok=True)
        return None
    st = _proc_stat(pgid)
    if st is not None:                                  # the leader is alive
        started = _started_at(pgid)
        written = pid_file.stat().st_mtime
        if st[1] != pgid or (started is not None and started > written + 2):
            print(f"[stop_server] {pid_file} names PID {pgid}, which is not the server it "
                  f"recorded (reused) -- leaving it alone")
            pid_file.unlink(missing_ok=True)
            return None
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        return None
    return pgid


def wait_group_gone(pgid: int, timeout: float, poll: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while group_alive(pgid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll)
    return True


def stop_own_server(pid_file: str | Path, timeout: float = 120) -> dict:
    """Stop the server Arbor launched in component mode, and only it (docs/component-mode.md).

    SIGTERM to the process group in ``pid_file``, a wait of up to ``timeout`` seconds
    for every member to exit, then the PID file is removed. Never SIGKILL (SGLang
    leaks GPU memory on it: stop_server above), never a pattern, and no VRAM wait or
    kill: the host owns the cards and whatever else runs on them. A group still
    running after the wait keeps its PID file, so the next stop tries again.

    ``status``: "none" (nothing of ours to stop), "stopped", or "running".
    """
    pid_file = Path(pid_file)
    pgid = signal_own_server(pid_file)
    if pgid is None:
        return {"status": "none"}
    print(f"[stop_server] SIGTERM to our server's process group {pgid} ({pid_file})")
    if wait_group_gone(pgid, timeout):
        pid_file.unlink(missing_ok=True)
        return {"status": "stopped", "pgid": pgid}
    print(f"[stop_server] WARNING: process group {pgid} still running after {timeout:.0f}s "
          f"(not sending SIGKILL)")
    return {"status": "running", "pgid": pgid}


def check_server_health(port: int, timeout: int = 600) -> bool:
    """Wait for the serving endpoint to be healthy."""
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = subprocess.run(
                ["curl", "-sf", f"http://localhost:{port}/health"],
                capture_output=True, timeout=5,
            )
            if r.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, OSError):
            pass
        time.sleep(5)
    return False
