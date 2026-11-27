# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pod-side in-place restart driver for atomesh and sglang-router pods, and the bash payloads that run it."""

from __future__ import annotations

import shlex
import uuid
from pathlib import Path

from . import process_image
from .pd_cluster import _close_inherited_handle_lines

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
    # A server that rewrites its process title (sglang::router) can leave an empty environ behind.
    env = baseline_environ(pid if pid > 0 else 1, pidfile + ".environ")
    return env or read_environ(1)


def _ppid(pid):
    stat = open("/proc/%d/stat" % pid, encoding="utf-8").read()
    return int(stat[stat.rfind(")") + 2 :].split()[1])


def _sid(pid):
    return os.getsid(pid)


def _is_parked(pid, parent, cmd):
    # The launcher's own child exec'd into `sleep infinity` (by this driver's stop or another tool).
    argv = cmd.split()
    if parent != 1 or len(argv) != 2 or os.path.basename(argv[0]) != "sleep" or argv[1] != "infinity":
        return False
    try:
        return _sid(pid) == _sid(1)
    except OSError:
        return False


def _held_by_launcher(pid):
    # The process PID 1 waits on: ending it ends the container, so it is only ever exec'd in place.
    try:
        return _ppid(pid) == 1 and _sid(pid) == _sid(1)
    except OSError:
        return False


def _runs_server(pid, needle):
    '''Whether ``pid`` runs the server, its launch script, or the parked launcher child.'''
    cmd = open("/proc/%d/cmdline" % pid, "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
    launch = os.environ.get("HL_LAUNCH_PATH", "")
    if any(part and part in cmd for part in needle.split("|")) or (launch and launch in cmd):
        return True
    return _is_parked(pid, _ppid(pid), cmd)


def _find_server(pidfile, needle):
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
    parked = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        cand = int(entry)
        if cand in (1, self_pid):
            continue
        try:
            cmd = open("/proc/%d/cmdline" % cand, "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
            parent = _ppid(cand)
        except OSError:
            continue
        if not any(part in cmd for part in needle.split("|")):
            if parked == 0 and _is_parked(cand, parent, cmd):
                parked = cand
            continue
        if parent == 1:
            return cand
        if fallback == 0:
            fallback = cand
    return parked or fallback


def restart_remote_main():
    launch = os.environ.get("HL_LAUNCH_PATH", "")
    needle = os.environ.get("HL_NEEDLE", "")
    pidfile = os.environ["HL_PIDFILE"]
    action = os.environ.get("HL_ACTION", "restart")
    pid = _find_server(pidfile, needle)
    if action == "ensure" and pid > 0:
        print(json.dumps({"status": "ok", "action": "running", "pid": pid}))
        return
    if action == "stop":
        if pid <= 0:
            print(json.dumps({"status": "ok", "action": "stop", "pid": 0}))
            return
        if _held_by_launcher(pid):
            park = os.environ.get("HL_PARK_SH") or "exec /bin/sleep infinity"
            replace_process_image(pid, ["/bin/bash", "-c", park], env_snapshot=pidfile + ".environ")
            open(pidfile, "w", encoding="utf-8").write(str(pid))
            print(json.dumps({"status": "ok", "action": "stop-exec", "pid": pid}))
            return
        _spawn_env(pid, pidfile)
        for victim in [pid] + descendant_pids(pid):
            try:
                os.kill(victim, 9)
            except OSError:
                pass
        try:
            os.remove(pidfile)
        except OSError:
            pass
        print(json.dumps({"status": "ok", "action": "stop-kill", "pid": pid}))
        return
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
    for victim in [pid] + descendant_pids(pid):
        try:
            os.kill(victim, 9)
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


def park_script() -> str:
    """Bash a stopped launcher-held pid execs: close inherited GPU handles and sockets, then sleep."""
    return "\n".join([*_close_inherited_handle_lines(), "exec /bin/sleep infinity"])


def bash_payload(role: str, script_body: str, needle: str, action: str) -> str:
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
            f"export HL_ACTION={shlex.quote(action)}",
            f"export HL_PARK_SH={shlex.quote(park_script())}",
            f"python3 - << '{py_end}'",
            _remote_program().rstrip("\n"),
            py_end,
        ]
    )


def ensure_payload(role: str, command: str, needle: str) -> str:
    """Bash that starts ``command`` detached on the pod unless a process matching ``needle`` already runs."""
    return bash_payload(role, f"exec {command}\n", needle, "ensure")
