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
import struct

_PTRACE_POKEDATA = 5
_PTRACE_CONT = 7
_PTRACE_GETREGS = 12
_PTRACE_SETREGS = 13
_PTRACE_ATTACH = 16
_PTRACE_DETACH = 17
_SYS_EXECVE = 59

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


def _poke(pid: int, addr: int, blob: bytes) -> None:
    """Write ``blob`` into the tracee, one native word at a time."""
    padded = blob + b"\0" * ((8 - len(blob) % 8) % 8)
    for offset in range(0, len(padded), 8):
        word = struct.unpack_from("<Q", padded, offset)[0]
        _ptrace(_PTRACE_POKEDATA, pid, addr + offset, word)


def _syscall_gadget(pid: int) -> int:
    """Return the address of an existing ``syscall`` opcode in an executable mapping.

    The stack is not executable, so the injected ``execve`` has to reuse a
    ``syscall`` instruction the process already mapped.
    """
    maps_path = f"/proc/{pid}/maps"
    mem_path = f"/proc/{pid}/mem"
    with open(maps_path, encoding="utf-8") as maps, open(mem_path, "rb", buffering=0) as mem:
        for line in maps:
            fields = line.split()
            if len(fields) < 2 or "x" not in fields[1]:
                continue
            start_s, end_s = fields[0].split("-")
            start, end = int(start_s, 16), int(end_s, 16)
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
    with open(f"/proc/{pid}/maps", encoding="utf-8") as maps:
        for line in maps:
            fields = line.split()
            if len(fields) < 2:
                continue
            start_s, end_s = fields[0].split("-")
            start, end = int(start_s, 16), int(end_s, 16)
            if not start <= rsp < end:
                continue
            addr = (rsp - nbytes - 64) & ~0xF
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
        raw = item.encode("utf-8") + b"\0"
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
    return [part.decode("utf-8", "surrogateescape") for part in raw.split(b"\0") if part]


def merged_environ(pid: int, overrides: dict[str, str] | None = None) -> list[str]:
    """Process environment with ``overrides`` replacing same-named keys."""
    overrides = overrides or {}
    merged: list[str] = []
    for item in read_environ(pid):
        key = item.split("=", 1)[0]
        if key in overrides:
            continue
        merged.append(item)
    for key, value in overrides.items():
        merged.append(f"{key}={value}")
    return merged


def descendant_pids(pid: int) -> list[int]:
    """Return every process whose ancestor is ``pid``, not including ``pid``."""
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = open(f"/proc/{entry}/stat", encoding="utf-8").read()
        except OSError:
            continue
        rparen = stat.rfind(")")
        if rparen < 0:
            continue
        fields = stat[rparen + 2 :].split()
        if len(fields) < 2:
            continue
        children.setdefault(int(fields[1]), []).append(int(entry))
    found: list[int] = []
    stack = list(children.get(pid, []))
    while stack:
        child = stack.pop()
        if child == pid:
            continue
        found.append(child)
        stack.extend(children.get(child, []))
    return found


def replace_process_image(pid: int, argv: list[str], *, env_overrides: dict[str, str] | None = None) -> None:
    """``execve`` ``argv`` inside ``pid``, preserving the pid and its environment.

    Descendants are killed while the target is stopped so a waiting parent
    (the container launcher) keeps waiting on the same pid.
    """
    if not argv or any("\0" in part for part in argv):
        raise ValueError("argv must be non-empty and contain no NULs")
    _ptrace(_PTRACE_ATTACH, pid)
    try:
        os.waitpid(pid, 0)
        for child in descendant_pids(pid):
            try:
                os.kill(child, 9)
            except OSError:
                pass
        regs = _Regs()
        if _libc.ptrace(_PTRACE_GETREGS, ctypes.c_int(pid), None, ctypes.byref(regs)) < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        gadget = _syscall_gadget(pid)
        env_items = merged_environ(pid, env_overrides)
        # Strings, two pointer arrays, and alignment slack.
        nbytes = 256 + sum(len(item) + 16 for item in argv + env_items)
        slot = _stack_slot(pid, regs.rsp, nbytes)
        argv_array, cursor, filename = _write_cstr_table(pid, slot, argv)
        env_array, _next, _env0 = _write_cstr_table(pid, (cursor + 15) & ~0xF, env_items)
        regs.rip = gadget
        regs.rax = _SYS_EXECVE
        regs.rdi = filename
        regs.rsi = argv_array
        regs.rdx = env_array
        if _libc.ptrace(_PTRACE_SETREGS, ctypes.c_int(pid), None, ctypes.byref(regs)) < 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        _ptrace(_PTRACE_CONT, pid)
        _pid, status = os.waitpid(pid, 0)
        if os.WIFEXITED(status) or os.WIFSIGNALED(status):
            raise OSError(f"process {pid} died during exec (status={status})")
    except Exception:
        try:
            _ptrace(_PTRACE_DETACH, pid)
        except OSError:
            pass
        raise
    _ptrace(_PTRACE_DETACH, pid)
