# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The host-side driver of a PD cluster launch, sent over SSH and run by ``python3 -`` on each host.

It starts, restarts or stops the server of one role on its host. A server the container launcher waits on (its
parent is PID 1, in PID 1's session) is never killed: ending it would end the container. ``execve`` is injected into
it instead (``replace_process_image``), keeping its pid. Any other server is killed with its descendants and the
launch script spawned detached. The driver reads its inputs from ``HL_*`` environment variables and prints one JSON
line. It imports nothing outside the standard library.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import json
import os
import struct
from collections.abc import Iterator

_PTRACE_POKEDATA = 5
_PTRACE_CONT = 7
_PTRACE_GETREGS = 12
_PTRACE_SETREGS = 13
_PTRACE_ATTACH = 16
_PTRACE_DETACH = 17
_SYS_EXECVE = 59
_SIGKILL = 9
_WORD = 8
_STACK_ALIGN = 2 * _WORD
_HEX = 16
_UTF8 = "utf-8"
_RAW_BYTES = "surrogateescape"
_BASH = "/bin/bash"
#: Output goes to the launch log and, when writable, to the container's stdout.
_SPAWN_SH = (
    'log="$1.log"; '
    'if [ -w "$2" ]; then exec > >(tee -a "$log" >> "$2") 2>&1; else exec >> "$log" 2>&1; fi; '
    'exec /bin/bash "$1"'
)


class _Regs(ctypes.Structure):
    """x86_64 ``user_regs_struct``."""

    _fields_ = [
        (name, ctypes.c_ulonglong)
        for name in (
            "r15 r14 r13 r12 rbp rbx r11 r10 r9 r8 rax rcx rdx rsi rdi orig_rax rip cs eflags rsp ss fs_base gs_base "
            "ds es fs gs"
        ).split()
    ]


def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.ptrace.restype = ctypes.c_long
    return libc


def _check(rc: int) -> None:
    """Raise ``OSError`` from ``errno`` when the ptrace call that returned ``rc`` failed."""
    if rc < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _ptrace(request: int, pid: int, addr: int = 0, data: int = 0) -> None:
    """One ptrace request; ``OSError`` when it fails."""
    _check(_libc().ptrace(ctypes.c_int(request), ctypes.c_int(pid), ctypes.c_void_p(addr), ctypes.c_ulong(data)))


def _registers(request: int, pid: int, regs: _Regs) -> None:
    """Read (``GETREGS``) or write (``SETREGS``) the tracee's registers."""
    _check(_libc().ptrace(ctypes.c_int(request), ctypes.c_int(pid), None, ctypes.byref(regs)))


def _align(value: int, size: int) -> int:
    """``value`` rounded up to a multiple of ``size``, a power of two."""
    return (value + size - 1) & ~(size - 1)


def _poke(pid: int, addr: int, blob: bytes) -> None:
    """Write ``blob`` into the tracee one word at a time."""
    padded = blob.ljust(_align(len(blob), _WORD), b"\0")
    for offset in range(0, len(padded), _WORD):
        _ptrace(_PTRACE_POKEDATA, pid, addr + offset, struct.unpack_from("<Q", padded, offset)[0])


def _mappings(pid: int) -> list[tuple[int, int, str]]:
    """``(start, end, permissions)`` of each mapping of ``pid``."""
    found = []
    with open(f"/proc/{pid}/maps", encoding=_UTF8) as maps:
        for line in maps:
            fields = line.split()
            start, end = (int(part, _HEX) for part in fields[0].split("-"))
            found.append((start, end, fields[1]))
    return found


def _syscall_gadget(pid: int) -> int:
    """The address of a ``syscall`` opcode in an executable mapping; the stack is not executable."""
    with open(f"/proc/{pid}/mem", "rb", buffering=0) as mem:
        for start, end, permissions in _mappings(pid):
            if "x" not in permissions:
                continue
            try:
                mem.seek(start)
                index = mem.read(min(end - start, 8_000_000)).find(b"\x0f\x05")
            except OSError:
                continue
            if index >= 0:
                return start + index
    raise OSError(f"no syscall gadget in pid {pid}")


def _stack_slot(pid: int, rsp: int, size: int) -> int:
    """A writable address below ``rsp`` that holds ``size`` bytes."""
    for start, end, _permissions in _mappings(pid):
        if start <= rsp < end:
            addr = (rsp - size - 64) & ~(_STACK_ALIGN - 1)
            if addr < start:
                raise OSError(f"pid {pid} stack cannot hold {size} bytes")
            return addr
    raise OSError(f"pid {pid} has no mapping for rsp {rsp:#x}")


def _write_strings(pid: int, addr: int, items: list[str]) -> tuple[int, int]:
    """NUL-terminated ``items`` and their NULL-ended pointer array at ``addr``: the array and the next free byte."""
    cursor = addr
    pointers = []
    for item in items:
        raw = item.encode(_UTF8, _RAW_BYTES) + b"\0"
        _poke(pid, cursor, raw)
        pointers.append(cursor)
        cursor += _align(len(raw), _WORD)
    packed = b"".join(struct.pack("<Q", pointer) for pointer in pointers) + struct.pack("<Q", 0)
    _poke(pid, cursor, packed)
    return cursor, cursor + len(packed)


def _split_environ(raw: bytes) -> list[str]:
    return [part.decode(_UTF8, _RAW_BYTES) for part in raw.split(b"\0") if part]


def read_environ(pid: int) -> list[str]:
    """``KEY=VALUE`` entries of ``/proc/<pid>/environ``; empty when unreadable."""
    try:
        with open(f"/proc/{pid}/environ", "rb") as environ:
            return _split_environ(environ.read())
    except OSError:
        return []


def baseline_environ(pid: int, snapshot: str) -> list[str]:
    """The environment ``pid`` had before its first replacement, kept at ``snapshot`` on first use."""
    with contextlib.suppress(OSError), open(snapshot, "rb") as kept:
        return _split_environ(kept.read())
    items = read_environ(pid)
    with open(snapshot + ".tmp", "wb") as staged:
        staged.write(b"".join(item.encode(_UTF8, _RAW_BYTES) + b"\0" for item in items))
    os.replace(snapshot + ".tmp", snapshot)
    return items


def _parent(pid: int) -> int:
    """The parent pid of ``pid`` from ``/proc/<pid>/stat``."""
    with open(f"/proc/{pid}/stat", encoding=_UTF8) as stat:
        text = stat.read()
    return int(text[text.rfind(")") + 2 :].split()[1])


def _procs() -> Iterator[tuple[int, int]]:
    """``(pid, parent pid)`` of every process whose stat is readable."""
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            with contextlib.suppress(OSError, IndexError, ValueError):
                yield int(entry), _parent(int(entry))


def descendant_pids(pid: int) -> list[int]:
    """Every process below ``pid``."""
    children: dict[int, list[int]] = {}
    for child, parent in _procs():
        children.setdefault(parent, []).append(child)
    found: list[int] = []
    stack = list(children.get(pid, []))
    while stack:
        child = stack.pop()
        found.append(child)
        stack.extend(children.get(child, []))
    return found


def _inject_execve(pid: int, argv: list[str], environ: list[str]) -> None:
    """Point the stopped tracee at ``execve(argv, environ)`` and let it run; its descendants are killed once the
    exec is staged, so a failure before that leaves them running."""
    regs = _Regs()
    _registers(_PTRACE_GETREGS, pid, regs)
    gadget = _syscall_gadget(pid)
    slot = _stack_slot(pid, regs.rsp, 256 + sum(len(item) + 2 * _WORD for item in argv + environ))
    argv_array, cursor = _write_strings(pid, slot, argv)
    environ_array, _end = _write_strings(pid, _align(cursor, _STACK_ALIGN), environ)
    regs.rip, regs.rax, regs.rdi, regs.rsi, regs.rdx = gadget, _SYS_EXECVE, slot, argv_array, environ_array
    _registers(_PTRACE_SETREGS, pid, regs)
    _kill(descendant_pids(pid))
    _ptrace(_PTRACE_CONT, pid)
    _waited, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status) or os.WIFSIGNALED(status):
        raise OSError(f"process {pid} died during exec (status={status})")


def replace_process_image(pid: int, argv: list[str], snapshot: str) -> None:
    """``execve`` ``argv`` inside ``pid``, keeping the pid, with the environment ``pid`` had before its first
    replacement. Its descendants are killed while it is stopped, so the launcher keeps waiting on the same pid. An
    ``argv[0]`` that is not an executable file is refused before ``pid`` is touched."""
    if not argv or any("\0" in part for part in argv):
        raise ValueError("argv must be non-empty and contain no NULs")
    if not os.access(argv[0], os.X_OK):
        raise OSError(f"{argv[0]} is not an executable file; pid {pid} left untouched")
    _ptrace(_PTRACE_ATTACH, pid)
    try:
        os.waitpid(pid, 0)
        _inject_execve(pid, argv, baseline_environ(pid, snapshot))
    except Exception:
        with contextlib.suppress(OSError):
            _ptrace(_PTRACE_DETACH, pid)
        raise
    _ptrace(_PTRACE_DETACH, pid)


def _kill(pids: list[int]) -> None:
    """``SIGKILL`` each of ``pids``, skipping those already gone."""
    for pid in pids:
        with contextlib.suppress(OSError):
            os.kill(pid, _SIGKILL)


def _kill_tree(pid: int) -> None:
    """``SIGKILL`` ``pid`` and every process below it."""
    _kill([pid, *descendant_pids(pid)])


def _cmdline(pid: int) -> str:
    with open(f"/proc/{pid}/cmdline", "rb") as cmdline:
        return cmdline.read().replace(b"\0", b" ").decode(_UTF8, "replace")


def _held_by_launcher(pid: int) -> bool:
    """Whether PID 1 waits on ``pid``: ending it ends the container."""
    try:
        return _parent(pid) == 1 and os.getsid(pid) == os.getsid(1)
    except OSError:
        return False


def _is_parked(pid: int, command: str) -> bool:
    """Whether ``pid`` is the launcher's child that a stop turned into ``sleep infinity``."""
    argv = command.split()
    return len(argv) == 2 and os.path.basename(argv[0]) == "sleep" and argv[1] == "infinity" and _held_by_launcher(pid)


def _holds_needle(command: str, needle: str) -> bool:
    """Whether ``command`` holds one of the ``|``-separated parts of ``needle``."""
    return any(part in command for part in needle.split("|"))


def _runs_server(pid: int, needle: str) -> bool:
    """Whether ``pid`` runs the role's server: its cmdline holds a ``needle`` part or the launch script, or it is the
    launcher's parked child."""
    try:
        command = _cmdline(pid)
    except OSError:
        return False
    launch = os.environ.get("HL_LAUNCH_PATH", "")
    return _holds_needle(command, needle) or bool(launch and launch in command) or _is_parked(pid, command)


def _pidfile_server(pidfile: str, needle: str) -> int:
    """The live pid recorded in ``pidfile`` when it still runs the role's server, else 0."""
    try:
        with open(pidfile, encoding=_UTF8) as recorded:
            pid = int(recorded.read().strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return 0
    return pid if _runs_server(pid, needle) else 0


def find_server(pidfile: str, needle: str) -> int:
    """The server of the role: the pidfile's live pid while it runs the server, else a process whose cmdline holds a ``needle`` part (one
    PID 1 waits on first), else the launcher's parked child; 0 when there is none."""
    pid = _pidfile_server(pidfile, needle)
    if pid:
        return pid
    fallback = parked = 0
    for candidate, parent in _procs():
        if candidate in (1, os.getpid()):
            continue
        try:
            command = _cmdline(candidate)
        except OSError:
            continue
        if _holds_needle(command, needle):
            if parent == 1:
                return candidate
            fallback = fallback or candidate
        elif not parked and _is_parked(candidate, command):
            parked = candidate
    return parked or fallback


def _snapshot(pidfile: str) -> str:
    """Where the server's environment before its first replacement is kept."""
    return pidfile + ".environ"


def _spawn_environ(pid: int, pidfile: str) -> dict[str, str]:
    """The environment a spawned server starts with: the session's on a bare host, else the server's own before its
    first replacement, else the container's (PID 1)."""
    if os.environ.get("HL_ENV_SOURCE") == "session":
        return {key: value for key, value in os.environ.items() if not key.startswith("HL_")}
    items = baseline_environ(pid if pid > 0 else 1, _snapshot(pidfile)) or read_environ(1)
    return dict(item.split("=", 1) for item in items if "=" in item)


def _spawn(launch: str, environ: dict[str, str]) -> int:
    """``launch`` started in its own session, its output in ``<launch>.log``."""
    console = os.environ.get("HL_CONSOLE", "/proc/1/fd/1")
    pid = os.fork()
    if pid == 0:
        os.setsid()
        log = os.open(launch + ".log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.dup2(log, 1)
        os.dup2(log, 2)
        if log > 2:
            os.close(log)
        null = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null, 0)
        if null > 0:
            os.close(null)
        os.execve(_BASH, [_BASH, "-c", _SPAWN_SH, "hl-spawn", launch, console], environ)
    return pid


def _record(pidfile: str, pid: int) -> None:
    """Write ``pid`` to ``pidfile``."""
    with open(pidfile, "w", encoding=_UTF8) as recorded:
        recorded.write(str(pid))


def _exec_held(pid: int, pidfile: str, argv: list[str]) -> None:
    """``execve`` ``argv`` into the launcher-held server ``pid`` and record it."""
    replace_process_image(pid, argv, _snapshot(pidfile))
    _record(pidfile, pid)


def _stop(pid: int, pidfile: str) -> str:
    """Stop the server: a launcher-held one becomes ``sleep infinity``, any other is killed."""
    if pid <= 0:
        return "none"
    if _held_by_launcher(pid):
        _exec_held(pid, pidfile, ["/bin/sleep", "infinity"])
        return "stop-exec"
    _spawn_environ(pid, pidfile)
    _kill_tree(pid)
    with contextlib.suppress(OSError):
        os.remove(pidfile)
    return "stop-kill"


def _restart(pid: int, pidfile: str, launch: str) -> tuple[str, int]:
    """Run ``launch`` in place of the server: exec'd into a launcher-held one, else spawned after a kill."""
    if pid > 0 and _held_by_launcher(pid):
        _exec_held(pid, pidfile, [_BASH, launch])
        return "exec", pid
    environ = _spawn_environ(pid, pidfile)
    if pid > 0:
        _kill_tree(pid)
    spawned = _spawn(launch, environ)
    _record(pidfile, spawned)
    return ("respawn" if pid > 0 else "spawn"), spawned


def main() -> None:
    """Apply ``HL_ACTION`` (``restart`` or ``stop``) to the server of the role and print what was done."""
    pidfile = os.environ["HL_PIDFILE"]
    pid = find_server(pidfile, os.environ.get("HL_NEEDLE", ""))
    if os.environ.get("HL_ACTION") == "stop":
        action = _stop(pid, pidfile)
    else:
        action, pid = _restart(pid, pidfile, os.environ["HL_LAUNCH_PATH"])
    print(json.dumps({"status": "ok", "action": action, "pid": pid}))


if __name__ == "__main__":
    main()
