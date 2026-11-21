# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pod-side in-place restart driver for atomesh and sglang-router pods, and the bash payloads that run it."""

from __future__ import annotations

import shlex
import uuid
from pathlib import Path

from . import process_image

REMOTE_DRIVER = r"""
import json

# Output goes to the launch log and, when writable, to the console (the container's stdout).
_SPAWN_SH = (
    'log="$1.log"; '
    'if [ -w "$2" ]; then exec > >(tee -a "$log" >> "$2") 2>&1; else exec >> "$log" 2>&1; fi; '
    'exec /bin/bash "$1"'
)


def _spawn_detached(launch, env):
    console = os.environ.get("HL_CONSOLE", "/proc/1/fd/1")
    pid = os.fork()
    if pid == 0:
        os.setsid()
        log_fd = os.open(launch + ".log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        null_fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null_fd, 0)
        environ = dict(item.split("=", 1) for item in env if "=" in item)
        os.execve("/bin/bash", ["/bin/bash", "-c", _SPAWN_SH, "hl-spawn", launch, console], environ)
    return pid


def _spawn_env(pid, pidfile):
    # The server's own environment, else the container's (PID 1): never the ssh session's.
    # A server that rewrites its process title can leave an empty environ behind.
    env = baseline_environ(pid if pid > 0 else 1, pidfile + ".environ")
    return env or read_environ(1)


def _held_by_launcher(pid):
    # The process PID 1 waits on: ending it ends the container, so it is only ever exec'd in place.
    try:
        return _ppid(pid) == 1 and os.getsid(pid) == os.getsid(1)
    except OSError:
        return False


def _cmdline(pid):
    '''Command line of ``pid`` with NULs turned into spaces.'''
    return open("/proc/%d/cmdline" % pid, "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")


def _runs_server(pid, needle):
    '''Whether ``pid`` runs the server or its launch script.'''
    cmd = _cmdline(pid)
    launch = os.environ.get("HL_LAUNCH_PATH", "")
    return bool(needle and needle in cmd) or bool(launch and launch in cmd)


def _find_server(pidfile, needle):
    '''Server pid: the pidfile's if it still runs the server, else a ``needle`` match, preferring a PID-1 child.'''
    self_pid = os.getpid()
    if os.path.exists(pidfile):
        try:
            cand = int(open(pidfile, encoding="utf-8").read().strip())
            os.kill(cand, 0)
            if cand != self_pid and _runs_server(cand, needle):
                return cand
        except (OSError, ValueError):
            pass
    fallback = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        cand = int(entry)
        if cand in (1, self_pid):
            continue
        try:
            cmd = _cmdline(cand)
            parent = _ppid(cand)
        except OSError:
            continue
        if needle not in cmd:
            continue
        if parent == 1:
            return cand
        if fallback == 0:
            fallback = cand
    return fallback


def restart_remote_main():
    '''Exec the launch script in place of the launcher-held server, else respawn or spawn it detached.'''
    launch = os.environ.get("HL_LAUNCH_PATH", "")
    needle = os.environ.get("HL_NEEDLE", "")
    pidfile = os.environ["HL_PIDFILE"]
    pid = _find_server(pidfile, needle)
    if pid <= 0:
        new_pid = _spawn_detached(launch, _spawn_env(pid, pidfile))
        open(pidfile, "w", encoding="utf-8").write(str(new_pid))
        print(json.dumps({"status": "ok", "action": "spawn", "pid": new_pid}))
        return
    if _held_by_launcher(pid):
        replace_process_image(pid, ["/bin/bash", launch], env_snapshot=pidfile + ".environ")
        open(pidfile, "w", encoding="utf-8").write(str(pid))
        print(json.dumps({"status": "ok", "action": "exec", "pid": pid}))
        return
    env = _spawn_env(pid, pidfile)
    _kill_descendants(pid)
    try:
        os.kill(pid, 9)
    except OSError:
        pass
    new_pid = _spawn_detached(launch, env)
    open(pidfile, "w", encoding="utf-8").write(str(new_pid))
    print(json.dumps({"status": "ok", "action": "respawn", "pid": new_pid}))


restart_remote_main()
"""


def _remote_program() -> str:
    """process_image helpers plus the pod-side restart driver."""
    return Path(process_image.__file__).read_text(encoding="utf-8") + "\n" + REMOTE_DRIVER


def pidfile_path(role: str) -> str:
    """Pod-local file holding the pid of the server this restart launched."""
    return f"/tmp/hyperloom-mn-{role}.pid"


def launch_path(role: str) -> str:
    """Pod-local launch script the restart writes and execs for ``role``."""
    return f"/tmp/hyperloom-mn-{role}.sh"


def bash_payload(role: str, script_body: str, needle: str) -> str:
    """Bash that writes the launch script and runs the in-place restart helper."""
    marker = uuid.uuid4().hex
    launch_end = f"HL_LAUNCH_{marker}"
    py_end = f"HL_PY_{marker}"
    return "\n".join(
        [
            f"launch={launch_path(role)}",
            f"cat > \"$launch\" << '{launch_end}'",
            script_body.rstrip("\n"),
            launch_end,
            'chmod +x "$launch"',
            'export HL_LAUNCH_PATH="$launch"',
            f"export HL_NEEDLE={shlex.quote(needle)}",
            f"export HL_PIDFILE={pidfile_path(role)}",
            f"python3 - << '{py_end}'",
            _remote_program().rstrip("\n"),
            py_end,
        ]
    )
