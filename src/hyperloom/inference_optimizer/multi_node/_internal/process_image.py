# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Replace a running process image in place, keeping its pid.

The platform launcher waits on the server pid. Killing that pid makes the
launcher exit and the kubelet recreates the pod with the original command.
``replace_process_image`` runs ``execve`` inside the existing pid instead.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import signal
import struct

_PTRACE_POKEDATA = 5
_PTRACE_SINGLESTEP = 9
_PTRACE_GETREGS = 12
_PTRACE_SETREGS = 13
_PTRACE_ATTACH = 16
_PTRACE_DETACH = 17
_SYS_EXECVE = 59
_WORD = 8
_HEX = 16
_ALIGN_MASK = ~0xF
_UTF8 = "utf-8"
_RAW_BYTES = "surrogateescape"

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.ptrace.restype = ctypes.c_long


class _Regs(ctypes.Structure):
    """x86_64 ``user_regs_struct`` layout."""

    _fields_ = [
        ("r15", ctypes.c_ulonglong),
        ("r14", ctypes.c_ulonglong),
        ("r13", ctypes.c_ulonglong),
        ("r12", ctypes.c_ulonglong),
        ("rbp", ctypes.c_ulonglong),
        ("rbx", ctypes.c_ulonglong),
        ("r11", ctypes.c_ulonglong),
        ("r10", ctypes.c_ulonglong),
        ("r9", ctypes.c_ulonglong),
        ("r8", ctypes.c_ulonglong),
        ("rax", ctypes.c_ulonglong),
        ("rcx", ctypes.c_ulonglong),
        ("rdx", ctypes.c_ulonglong),
        ("rsi", ctypes.c_ulonglong),
        ("rdi", ctypes.c_ulonglong),
        ("orig_rax", ctypes.c_ulonglong),
        ("rip", ctypes.c_ulonglong),
        ("cs", ctypes.c_ulonglong),
        ("eflags", ctypes.c_ulonglong),
        ("rsp", ctypes.c_ulonglong),
        ("ss", ctypes.c_ulonglong),
        ("fs_base", ctypes.c_ulonglong),
        ("gs_base", ctypes.c_ulonglong),
        ("ds", ctypes.c_ulonglong),
        ("es", ctypes.c_ulonglong),
        ("fs", ctypes.c_ulonglong),
        ("gs", ctypes.c_ulonglong),
    ]


def _ptrace(request: int, pid: int, addr: int = 0, data: int = 0) -> int:
    """Issue one ptrace request and raise ``OSError`` on failure."""
    rc = _libc.ptrace(ctypes.c_int(request), ctypes.c_int(pid), ctypes.c_void_p(addr), ctypes.c_ulong(data))
    if rc < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return int(rc)


def _regs_call(request: int, pid: int, regs: _Regs) -> None:
    """Issue a ``GETREGS`` or ``SETREGS`` ptrace request on ``regs`` and raise ``OSError`` on failure."""
    if _libc.ptrace(request, ctypes.c_int(pid), None, ctypes.byref(regs)) < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def _poke(pid: int, addr: int, blob: bytes) -> None:
    """Write ``blob`` into the tracee, one native word at a time."""
    padded = blob + b"\0" * ((_WORD - len(blob) % _WORD) % _WORD)
    for offset in range(0, len(padded), _WORD):
        word = struct.unpack_from("<Q", padded, offset)[0]
        _ptrace(_PTRACE_POKEDATA, pid, addr + offset, word)


def _syscall_gadget(pid: int) -> int:
    """Return the address of an existing ``syscall`` opcode in an executable mapping.

    The stack is not executable, so the injected ``execve`` has to reuse a
    ``syscall`` instruction the process already mapped.
    """
    maps_path = f"/proc/{pid}/maps"
    mem_path = f"/proc/{pid}/mem"
    with open(maps_path, encoding=_UTF8) as maps, open(mem_path, "rb", buffering=0) as mem:
        for line in maps:
            fields = line.split()
            if len(fields) < 2 or "x" not in fields[1]:
                continue
            start_s, end_s = fields[0].split("-")
            start, end = int(start_s, _HEX), int(end_s, _HEX)
            try:
                mem.seek(start)
                blob = mem.read(min(end - start, 8_000_000))
            except OSError:
                continue
            idx = blob.find(b"\x0f\x05")
            if idx >= 0:
                return start + idx
    raise OSError(f"no syscall gadget in pid {pid}")


def _stack_slot(pid: int, rsp: int, nbytes: int) -> int:
    """Return a writable address below ``rsp`` that fits ``nbytes`` bytes."""
    with open(f"/proc/{pid}/maps", encoding=_UTF8) as maps:
        for line in maps:
            fields = line.split()
            if len(fields) < 2:
                continue
            start_s, end_s = fields[0].split("-")
            start, end = int(start_s, _HEX), int(end_s, _HEX)
            if not start <= rsp < end:
                continue
            # Stay clear of the 128-byte red zone below rsp: a failed exec resumes the original image.
            addr = (rsp - nbytes - 256) & _ALIGN_MASK
            if addr < start:
                raise OSError(f"pid {pid} stack cannot hold {nbytes} bytes")
            return addr
    raise OSError(f"pid {pid} has no mapping for rsp {rsp:#x}")


def _write_cstr_table(pid: int, addr: int, items: list[str]) -> tuple[int, int, int]:
    """Write NUL-terminated strings and a NULL-ended pointer array at ``addr``.

    Returns ``(pointer_array, next_free, first_string)``. ``first_string`` is 0
    when ``items`` is empty.
    """
    cursor = addr
    ptrs: list[int] = []
    for item in items:
        raw = item.encode(_UTF8, _RAW_BYTES) + b"\0"
        _poke(pid, cursor, raw)
        ptrs.append(cursor)
        cursor += (len(raw) + 7) & ~7
    array = cursor
    packed = b"".join(struct.pack("<Q", ptr) for ptr in ptrs) + struct.pack("<Q", 0)
    _poke(pid, array, packed)
    return array, array + len(packed), ptrs[0] if ptrs else 0


def read_environ(pid: int) -> list[str]:
    """Return ``KEY=VALUE`` strings from ``/proc/<pid>/environ``."""
    try:
        raw = open(f"/proc/{pid}/environ", "rb").read()
    except OSError:
        return []
    return [part.decode(_UTF8, _RAW_BYTES) for part in raw.split(b"\0") if part]


def baseline_environ(pid: int, snapshot_path: str) -> list[str]:
    """Environment ``pid`` had before its first in-place replacement, recorded at ``snapshot_path`` on first use."""
    try:
        raw = open(snapshot_path, "rb").read()
        return [part.decode(_UTF8, _RAW_BYTES) for part in raw.split(b"\0") if part]
    except OSError:
        pass
    items = read_environ(pid)
    tmp_path = snapshot_path + ".tmp"
    with open(tmp_path, "wb") as fh:
        fh.write(b"".join(item.encode(_UTF8, _RAW_BYTES) + b"\0" for item in items))
    os.replace(tmp_path, snapshot_path)
    return items


def _ppid(pid: int) -> int:
    """Parent pid of ``pid`` from ``/proc/<pid>/stat``; ``OSError`` when it cannot be read."""
    stat = open(f"/proc/{pid}/stat", encoding=_UTF8).read()
    return int(stat[stat.rfind(")") + 2 :].split()[1])


def descendant_pids(pid: int) -> list[int]:
    """Return every process whose ancestor is ``pid``, not including ``pid``."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            parent = _ppid(int(entry))
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(parent, []).append(int(entry))
    found: list[int] = []
    stack = list(children.get(pid, []))
    while stack:
        child = stack.pop()
        if child == pid:
            continue
        found.append(child)
        stack.extend(children.get(child, []))
    return found


def _kill_descendants(pid: int) -> None:
    """SIGKILL every descendant of ``pid``."""
    for child in descendant_pids(pid):
        try:
            os.kill(child, signal.SIGKILL)
        except OSError:
            pass


def _check_exec(pid: int, status: int, gadget: int, saved: _Regs) -> None:
    """Raise ``OSError`` unless the single-stepped ``execve`` replaced the image; a failed one restores ``saved``."""
    if not os.WIFSTOPPED(status) or os.WSTOPSIG(status) != signal.SIGTRAP:
        raise OSError(f"process {pid} stopped with status {status} during exec")
    after = _Regs()
    _regs_call(_PTRACE_GETREGS, pid, after)
    ret = ctypes.c_longlong(after.rax).value
    if after.rip != gadget + 2 or not -4096 < ret < 0:
        return
    _regs_call(_PTRACE_SETREGS, pid, saved)
    raise OSError(-ret, f"execve in pid {pid} failed: {os.strerror(-ret)}")


def _exec_staged(pid: int, regs: _Regs, gadget: int, saved: _Regs) -> None:
    """Load the staged ``execve`` registers into the stopped ``pid`` and single-step it; ``OSError`` when it fails."""
    _regs_call(_PTRACE_SETREGS, pid, regs)
    _ptrace(_PTRACE_SINGLESTEP, pid)
    _pid, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status) or os.WIFSIGNALED(status):
        raise OSError(f"process {pid} died during exec (status={status})")
    _check_exec(pid, status, gadget, saved)


def replace_process_image(
    pid: int,
    argv: list[str],
    *,
    env_snapshot: str = "",
) -> None:
    """``execve`` ``argv`` inside ``pid``, preserving the pid and its environment.

    With ``env_snapshot`` the new image starts from the environment ``pid`` had
    before its first replacement, not from the one the current image inherited.
    An ``argv[0]`` that is not an executable file is refused before ``pid`` is touched.
    Descendants are killed while the target is stopped, once the exec is staged, so a
    waiting parent (the container launcher) keeps waiting on the same pid; a failure
    before that leaves the process and its descendants running. The injected ``syscall``
    is single-stepped. Once the descendants are gone, a failed exec restores the original
    registers, detaches and SIGKILLs ``pid`` so the platform recreates the pod, and raises
    ``OSError`` carrying the errno.
    """
    if not argv or any("\0" in part for part in argv):
        raise ValueError("argv must be non-empty and contain no NULs")
    if not os.access(argv[0], os.X_OK):
        raise OSError(f"{argv[0]} is not an executable file; pid {pid} left untouched")
    _ptrace(_PTRACE_ATTACH, pid)
    try:
        os.waitpid(pid, 0)
        regs = _Regs()
        _regs_call(_PTRACE_GETREGS, pid, regs)
        saved = _Regs.from_buffer_copy(regs)
        gadget = _syscall_gadget(pid)
        env_items = baseline_environ(pid, env_snapshot) if env_snapshot else read_environ(pid)
        # Strings, two pointer arrays, and alignment slack.
        nbytes = 256 + sum(len(item.encode(_UTF8, _RAW_BYTES)) + 2 * _WORD for item in argv + env_items)
        slot = _stack_slot(pid, regs.rsp, nbytes)
        argv_array, cursor, filename = _write_cstr_table(pid, slot, argv)
        env_array, _next, _env0 = _write_cstr_table(pid, (cursor + 15) & _ALIGN_MASK, env_items)
        regs.rip = gadget
        regs.rax = _SYS_EXECVE
        regs.rdi = filename
        regs.rsi = argv_array
        regs.rdx = env_array
        _kill_descendants(pid)
        try:
            _exec_staged(pid, regs, gadget, saved)
        except OSError as exc:
            try:
                _ptrace(_PTRACE_DETACH, pid)
            except OSError:
                pass
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            message = f"{exc.strerror or exc}; killed pid {pid} so the platform recreates it"
            raise (OSError(exc.errno, message) if exc.errno else OSError(message)) from exc
    except Exception:
        try:
            _ptrace(_PTRACE_DETACH, pid)
        except OSError:
            pass
        raise
    _ptrace(_PTRACE_DETACH, pid)
