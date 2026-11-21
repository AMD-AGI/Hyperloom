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

import ctypes
import ctypes.util
import json
import os
import struct

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


def _ptrace(request: int, pid: int, addr: int = 0, data: int = 0) -> int:
    """One ptrace request; ``OSError`` when it fails."""
    rc = _libc().ptrace(ctypes.c_int(request), ctypes.c_int(pid), ctypes.c_void_p(addr), ctypes.c_ulong(data))
    if rc < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(rc)


def _registers(request: int, pid: int, regs: _Regs) -> None:
    """Read (``GETREGS``) or write (``SETREGS``) the tracee's registers."""
    if _libc().ptrace(ctypes.c_int(request), ctypes.c_int(pid), None, ctypes.byref(regs)) < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


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
            if len(fields) >= 2:
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
    try:
        with open(snapshot, "rb") as kept:
            return _split_environ(kept.read())
    except OSError:
        pass
    items = read_environ(pid)
    with open(snapshot + ".tmp", "wb") as staged:
        staged.write(b"".join(item.encode(_UTF8, _RAW_BYTES) + b"\0" for item in items))
    os.replace(snapshot + ".tmp", snapshot)
    return items


def _parent(pid: int) -> int:
    with open(f"/proc/{pid}/stat", encoding=_UTF8) as stat:
        text = stat.read()
    return int(text[text.rfind(")") + 2 :].split()[1])


def descendant_pids(pid: int) -> list[int]:
    """Every process below ``pid``."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                children.setdefault(_parent(int(entry)), []).append(int(entry))
            except (OSError, IndexError, ValueError):
                continue
    found: list[int] = []
    stack = list(children.get(pid, []))
    while stack:
        child = stack.pop()
        found.append(child)
        stack.extend(children.get(child, []))
    return found


def _inject_execve(pid: int, argv: list[str], environ: list[str]) -> None:
    """Point the stopped tracee at ``execve(argv, environ)`` and let it run."""
    regs = _Regs()
    _registers(_PTRACE_GETREGS, pid, regs)
    gadget = _syscall_gadget(pid)
    slot = _stack_slot(pid, regs.rsp, 256 + sum(len(item) + 2 * _WORD for item in argv + environ))
    argv_array, cursor = _write_strings(pid, slot, argv)
    environ_array, _end = _write_strings(pid, _align(cursor, _STACK_ALIGN), environ)
    regs.rip, regs.rax, regs.rdi, regs.rsi, regs.rdx = gadget, _SYS_EXECVE, slot, argv_array, environ_array
    _registers(_PTRACE_SETREGS, pid, regs)
    _ptrace(_PTRACE_CONT, pid)
    _waited, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status) or os.WIFSIGNALED(status):
        raise OSError(f"process {pid} died during exec (status={status})")


def replace_process_image(pid: int, argv: list[str], snapshot: str) -> None:
    """``execve`` ``argv`` inside ``pid``, keeping the pid, with the environment ``pid`` had before its first
    replacement. Its descendants are killed while it is stopped, so the launcher keeps waiting on the same pid."""
    if not argv or any("\0" in part for part in argv):
        raise ValueError("argv must be non-empty and contain no NULs")
    _ptrace(_PTRACE_ATTACH, pid)
    try:
        os.waitpid(pid, 0)
        _kill(descendant_pids(pid))
        _inject_execve(pid, argv, baseline_environ(pid, snapshot))
    except Exception:
        try:
            _ptrace(_PTRACE_DETACH, pid)
        except OSError:
            pass
        raise
    _ptrace(_PTRACE_DETACH, pid)


def _kill(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, _SIGKILL)
        except OSError:
            pass


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


def _pidfile_server(pidfile: str) -> int:
    try:
        with open(pidfile, encoding=_UTF8) as recorded:
            pid = int(recorded.read().strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return 0
    return pid


def find_server(pidfile: str, needle: str) -> int:
    """The server of the role: the pidfile's live pid, else a process whose cmdline holds a ``needle`` part (one
    PID 1 waits on first), else the launcher's parked child; 0 when there is none."""
    pid = _pidfile_server(pidfile)
    if pid:
        return pid
    fallback = parked = 0
    for entry in os.listdir("/proc"):
        candidate = int(entry) if entry.isdigit() else 0
        if candidate in (0, 1, os.getpid()):
            continue
        try:
            command, parent = _cmdline(candidate), _parent(candidate)
        except (OSError, IndexError, ValueError):
            continue
        if any(part in command for part in needle.split("|")):
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
        log = os.open(launch + ".log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(log, 1)
        os.dup2(log, 2)
        os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
        os.execve(_BASH, [_BASH, "-c", _SPAWN_SH, "hl-spawn", launch, console], environ)
    return pid


def _record(pidfile: str, pid: int) -> None:
    with open(pidfile, "w", encoding=_UTF8) as recorded:
        recorded.write(str(pid))


def _stop(pid: int, pidfile: str) -> str:
    """Stop the server: a launcher-held one becomes ``sleep infinity``, any other is killed."""
    if pid <= 0:
        return "none"
    if _held_by_launcher(pid):
        replace_process_image(pid, ["/bin/sleep", "infinity"], _snapshot(pidfile))
        _record(pidfile, pid)
        return "stop-exec"
    _spawn_environ(pid, pidfile)
    _kill([pid, *descendant_pids(pid)])
    try:
        os.remove(pidfile)
    except OSError:
        pass
    return "stop-kill"


def _restart(pid: int, pidfile: str, launch: str) -> tuple[str, int]:
    """Run ``launch`` in place of the server: exec'd into a launcher-held one, else spawned after a kill."""
    if pid > 0 and _held_by_launcher(pid):
        replace_process_image(pid, [_BASH, launch], _snapshot(pidfile))
        _record(pidfile, pid)
        return "exec", pid
    environ = _spawn_environ(pid, pidfile)
    if pid > 0:
        _kill([pid, *descendant_pids(pid)])
    spawned = _spawn(launch, environ)
    _record(pidfile, spawned)
    return ("respawn" if pid > 0 else "spawn"), spawned


def main() -> None:
    """Apply ``HL_ACTION`` (``restart`` or ``stop``) to the server of the role and print what was done."""
    pidfile = os.environ["HL_PIDFILE"]
    pid = find_server(pidfile, os.environ.get("HL_NEEDLE", ""))
    if os.environ.get("HL_ACTION") == "stop":
        print(json.dumps({"status": "ok", "action": _stop(pid, pidfile), "pid": pid}))
        return
    action, pid = _restart(pid, pidfile, os.environ["HL_LAUNCH_PATH"])
    print(json.dumps({"status": "ok", "action": action, "pid": pid}))


if __name__ == "__main__":
    main()
