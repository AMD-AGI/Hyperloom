# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An agent's bubblewrap sandbox: the policy, the argv it becomes, and the probe that runs it once against a canary.

The sandbox starts from an empty root. A path exists inside only when the policy binds it, so what is not bound (the
app's state, its keys, other tasks' trees) is absent rather than forbidden; read-only binds refuse writes and metadata
changes (``EROFS``); a mask hides a credential inside a bound tree (``EACCES``). The agent gets its own user, pid and
IPC namespaces, a fresh ``/proc``, no capabilities and ``no_new_privs``; it shares the network unless the policy asks
for its own.

bubblewrap prints the namespaces' ``child-pid`` on ``--json-status-fd`` as soon as it has cloned, before it mounts
anything, and an ``exit-code`` only when the command ran: :class:`BwrapStatus` reads that difference.

Stdlib only, so every package can launch through it, and a host without Hyperloom installed can run it as a script
(``python3 -I bwrap.py probe``).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import errno
import json
import logging
import os
import re
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

__all__ = [
    "BASE_DEV_NODES",
    "BWRAP_FLOOR",
    "NESTED_SAFE",
    "SYSTEM_ROOTS",
    "BwrapPolicy",
    "BwrapProbe",
    "BwrapSetupError",
    "BwrapStatus",
    "Holder",
    "Scratch",
    "bwrap_probe",
    "interpreter_roots",
    "main",
    "probe_bwrap",
]

log = logging.getLogger(__name__)

Version = tuple[int, int, int]

#: The oldest bubblewrap the adapter runs: ``--perms`` (the mode of a masked directory) arrived in 0.5.0.
BWRAP_FLOOR: Version = (0, 5, 0)
#: Before this version the mount points bubblewrap creates follow symlinks an agent left in a writable tree, out of
#: the sandbox (GHSA-pxhw-h44j-8pfx): a mount there may land elsewhere and create a file or directory on the host, so
#: older ones refuse it (:meth:`BwrapPolicy.problems`).
NESTED_SAFE: Version = (0, 12, 0)
#: The system trees a sandbox that runs programs binds read-only; a root that is a symlink (``/bin -> usr/bin``)
#: becomes the same symlink, and one the host lacks is skipped.
SYSTEM_ROOTS = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32", "/etc", "/opt")
#: The device nodes every sandbox gets, each bound only where the host has it (a GPU container's ``/dev`` may lack
#: ``/dev/full``, which is why ``--dev`` is not used).
BASE_DEV_NODES = ("/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom", "/dev/tty")

_BWRAP = "bwrap"
_DEV = "/dev"
_PROC = "/proc"
_SYS = "/sys"
_TMP = "/tmp"
_SHM = "/dev/shm"
_BIND = "--bind"
_RO_BIND_TRY = "--ro-bind-try"
_SYMLINK = "--symlink"
_TMPFS = "--tmpfs"
_DEV_LINKS = (
    ("/proc/self/fd", "/dev/fd"),
    ("/proc/self/fd/0", "/dev/stdin"),
    ("/proc/self/fd/1", "/dev/stdout"),
    ("/proc/self/fd/2", "/dev/stderr"),
)
#: Mounted by the sandbox itself, so never a bind or a mask.
_OWN_MOUNTS = frozenset(("/", _PROC, _DEV, _SYS, _TMP, _SHM))
_SCRATCH_PREFIX = "hl-agent-"
_TIMEOUT = 10.0
_USERNS_SYSCTLS = (
    "/proc/sys/user/max_user_namespaces",
    "/proc/sys/kernel/unprivileged_userns_clone",
    "/proc/sys/kernel/apparmor_restrict_unprivileged_userns",
)
_PYTHON_FLAGS = ("-I", "-S", "-c")
_VERSION = re.compile(r"bubblewrap (\d+)\.(\d+)\.(\d+)")
#: Under ``unshare --mount`` bubblewrap starts from a private copy of the mount tree, which a root mount marked
#: unbindable (as some container runtimes leave it) otherwise refuses to every bind with ``EINVAL``.
_PRIVATE_MOUNTS = ("--mount", "--")
_HOLD = "import signal; print('ready', flush=True); signal.pause()"


class BwrapSetupError(RuntimeError):
    """bubblewrap could not set the sandbox up, so the command never started; the message is bubblewrap's line."""


def _shown(version: Version) -> str:
    return ".".join(map(str, version))


def _depth(path: str) -> int:
    return len(PurePosixPath(path).parts)


def _below(path: str, parent: str) -> bool:
    return path != parent and path.startswith(parent.rstrip("/") + "/")


def _at_or_below(path: str, parent: str) -> bool:
    return path == parent or _below(path, parent)


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _normalized(paths: Sequence[str]) -> tuple[str, ...]:
    return tuple(os.path.normpath(path) if os.path.isabs(path) else path for path in paths)


@dataclass(frozen=True)
class _Mount:
    destination: str
    args: tuple[str, ...]
    mask: bool = False


def _scratch_mount(source: str | None, destination: str) -> _Mount:
    return _Mount(destination, (_BIND, source, destination) if source else (_TMPFS, destination))


def _mask_mount(path: str) -> _Mount | None:
    """A mask over what ``path`` is now; ``None`` when nothing is there: a mount point would create it."""
    if os.path.isdir(path):
        return _Mount(path, ("--perms", "0000", _TMPFS, path, "--remount-ro", path), mask=True)
    if os.path.lexists(path):
        return _Mount(path, ("--ro-bind", "/dev/null", path), mask=True)
    return None


@dataclass(frozen=True)
class BwrapPolicy:
    """What one sandbox may reach. Every path is absolute and bound at the same path inside.

    ``read_only`` and ``writable`` are bound (a missing read-only path is skipped); ``masked`` hides a path inside a
    bound tree: a directory becomes an empty mode-0000 tmpfs, anything else an unreadable ``/dev/null``. ``devices``
    are bound with device access (``/dev/kfd``, ``/dev/dri``). ``tmp`` and ``shm`` are bound at ``/tmp`` and
    ``/dev/shm`` (a :class:`Scratch`; a sandbox whose commands share files there passes the same pair to each), else
    each is a fresh tmpfs. ``sys_paths`` are bound read-only (a compile narrows ``/sys`` to the CPU tree).
    ``own_network`` gives the sandbox a network of its own (loopback only). ``die_with_parent`` kills the sandbox when
    the *thread* that started it ends (``PR_SET_PDEATHSIG``), so it is only for a launch made on a thread that
    outlives it; without it, killing bubblewrap's own process leaves the sandbox running (kill its pid 1, the
    status's ``child_pid``, or the whole process tree). ``join`` holds the user and pid namespace descriptors of a
    :class:`Holder` to join instead of making new ones. ``marks`` are set in the command's environment only, so
    bubblewrap's own processes do not carry them (they show in bubblewrap's command line: never a secret).
    ``app_only`` paths are never bound, nor anything above them.
    """

    read_only: tuple[str, ...]
    writable: tuple[str, ...] = ()
    masked: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    tmp: str | None = None
    shm: str | None = None
    sys_paths: tuple[str, ...] = (_SYS,)
    own_network: bool = False
    die_with_parent: bool = True
    join: tuple[int, int] | None = None
    marks: tuple[tuple[str, str], ...] = ()
    chdir: str | None = None
    app_only: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("read_only", "writable", "masked", "devices", "sys_paths", "app_only"):
            object.__setattr__(self, name, _normalized(getattr(self, name)))
        for name in ("tmp", "shm", "chdir"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _normalized((value,))[0])

    def problems(self, version: Version) -> list[str]:
        """Why this policy must not be launched with bubblewrap ``version``; empty when it may.

        A relative path; a bind or mask of a path the sandbox mounts itself (``/``, ``/proc`` and below, ``/dev``
        and below, ``/sys``, ``/tmp``); a device outside ``/dev``; a bind at or above an ``app_only`` path; and,
        before :data:`NESTED_SAFE`, a mount below what an agent could already have written: a read-only bind or mask
        below a writable bind, and, for a launch that joins a :class:`Holder`, any bind or mask below the ``tmp`` or
        ``shm`` that earlier commands shared. A writable path below another writable one is not a problem: it is
        redundant and left out of the argv. Below a launch's own fresh ``tmp`` and ``shm`` nothing is checked.
        """
        named = (*self.read_only, *self.writable, *self.masked, *self.devices, *self.sys_paths, *self.app_only)
        relative = [
            f"{path!r} is not an absolute path"
            for path in (*named, self.tmp, self.shm, self.chdir)
            if path is not None and not os.path.isabs(path)
        ]
        if relative:
            return relative
        found = self._placement_problems() + self._exposure_problems()
        if version < NESTED_SAFE:
            found += self._nesting_problems(version)
        return found

    def _placement_problems(self) -> list[str]:
        found = [
            f"{path} is mounted by the sandbox itself, so it cannot be bound or masked"
            for path in (*self.read_only, *self.writable, *self.masked)
            if path in _OWN_MOUNTS or _below(path, _PROC) or _below(path, _DEV)
        ]
        found += [f"device {path} is not under {_DEV}" for path in self.devices if not _below(path, _DEV)]
        found += [f"{path} is not under {_SYS}" for path in self.sys_paths if not _at_or_below(path, _SYS)]
        return found

    def _exposure_problems(self) -> list[str]:
        apps = [os.path.realpath(path) for path in self.app_only]
        sources = (*self.read_only, *self.writable, *(path for path in (self.tmp, self.shm) if path))
        return [
            f"{source} would expose {app}, which only the app may reach"
            for source in sources
            for app in apps
            if _at_or_below(app, os.path.realpath(source))
        ]

    def _nesting_problems(self, version: Version) -> list[str]:
        writable = self._kept_writable()
        shared = [mount for source, mount in ((self.tmp, _TMP), (self.shm, _SHM)) if source and self.join is not None]
        nested = [(path, parent) for path in (*self.read_only, *self.masked) for parent in writable]
        nested += [(path, parent) for path in (*self.read_only, *self.masked, *writable) for parent in shared]
        return [
            f"{path} would be mounted below the agent-writable {parent}, which needs bubblewrap "
            f"{_shown(NESTED_SAFE)} (this is {_shown(version)})"
            for path, parent in nested
            if _below(path, parent)
        ]

    def _kept_writable(self) -> tuple[str, ...]:
        """``writable`` without the paths a writable bind above them already shows (no read-only bind or mask
        between)."""
        writable = tuple(dict.fromkeys(self.writable))
        bound = (*writable, *self.read_only, *self.masked)

        def shown_by_parent(path: str) -> bool:
            above = [parent for parent in bound if _below(path, parent)]
            return bool(above) and max(above, key=_depth) in writable

        return tuple(path for path in writable if not shown_by_parent(path))

    def argv(self, launcher: Sequence[str], command: Sequence[str], *, status_fd: int | None = None) -> list[str]:
        """The command line that runs ``command`` in this sandbox: ``launcher`` (:attr:`BwrapProbe.launcher`), the
        namespace flags, every mount sorted by destination depth with the masks last (so a bind below ``/tmp`` or a
        mask below a bind is not hidden by the mount above it), then ``--chdir``, the marks and
        ``--json-status-fd``. The masks and symlinked roots are read from the filesystem now.
        """
        tail = ["--chdir", self.chdir] if self.chdir else []
        for name, value in self.marks:
            tail += ["--setenv", name, value]
        if status_fd is not None:
            tail += ["--json-status-fd", str(status_fd)]
        mounts = [arg for mount in self._mounts() for arg in mount.args]
        return [*launcher, *self._namespace_flags(), *mounts, *tail, "--", *command]

    def _namespace_flags(self) -> list[str]:
        if self.join is None:
            flags = ["--unshare-user", "--unshare-pid"]
        else:
            flags = ["--userns", str(self.join[0]), "--pidns", str(self.join[1])]
        flags += ["--unshare-ipc", "--new-session", "--cap-drop", "ALL"]
        if self.die_with_parent:
            flags.append("--die-with-parent")
        if self.own_network:
            flags.append("--unshare-net")
        return flags

    def _mounts(self) -> list[_Mount]:
        writable = self._kept_writable()
        bound = (*self.read_only, *writable)
        mounts = [self._read_only_mount(path, bound) for path in self.read_only]
        mounts += [_Mount(path, (_BIND, path, path)) for path in writable]
        mounts.append(_Mount(_DEV, (_TMPFS, _DEV)))
        mounts += [_Mount(node, ("--dev-bind-try", node, node)) for node in (*BASE_DEV_NODES, *self.devices)]
        mounts += [_Mount(link, (_SYMLINK, target, link)) for target, link in _DEV_LINKS]
        mounts.append(_Mount(_PROC, ("--proc", _PROC)))
        mounts += [_Mount(path, (_RO_BIND_TRY, path, path)) for path in self.sys_paths]
        mounts += [_scratch_mount(self.tmp, _TMP), _scratch_mount(self.shm, _SHM)]
        mounts += [mount for mount in map(_mask_mount, self.masked) if mount is not None]
        return sorted(mounts, key=lambda mount: (mount.mask, _depth(mount.destination)))

    @staticmethod
    def _read_only_mount(path: str, bound: Sequence[str]) -> _Mount:
        """A symlink whose target another bind shows stays that symlink (``/bin -> /usr/bin``); else a bind."""
        if os.path.islink(path):
            target = os.path.realpath(path)
            if any(_at_or_below(target, other) for other in bound if other != path):
                return _Mount(path, (_SYMLINK, target, path))
        return _Mount(path, (_RO_BIND_TRY, path, path))


def _remove_tree(path: str) -> None:
    """Remove ``path`` even where the agent left a directory it cannot list (mode 0000)."""
    for parent, names, _files in os.walk(path):
        for name in names:
            child = os.path.join(parent, name)
            if not os.path.islink(child):
                os.chmod(child, stat.S_IRWXU)
    shutil.rmtree(path)


@dataclass(frozen=True)
class Scratch:
    """A sandbox's private ``/tmp`` and ``/dev/shm``: two directories of mode 0700 with unpredictable names, so no
    other process of the same uid finds them by name. The ``/tmp`` one is on the host's ``/tmp`` (disk, not the
    memory cgroup); the ``/dev/shm`` one is full size."""

    tmp: str
    shm: str

    @classmethod
    def create(cls) -> Scratch:
        """Make the two directories; :meth:`remove` deletes them."""
        tmp = tempfile.mkdtemp(prefix=_SCRATCH_PREFIX)
        try:
            shm = tempfile.mkdtemp(prefix=_SCRATCH_PREFIX, dir=_SHM if os.access(_SHM, os.W_OK | os.X_OK) else None)
        except OSError:
            _remove_tree(tmp)
            raise
        return cls(tmp, shm)

    def remove(self) -> None:
        """Delete both directories and what the sandbox left in them."""
        _remove_tree(self.tmp)
        _remove_tree(self.shm)


def interpreter_roots() -> tuple[str, ...]:
    """Where this interpreter lives outside :data:`SYSTEM_ROOTS` (its prefixes and the tree its executable resolves
    into): with them bound read-only, a sandbox can run ``sys.executable -I -S``."""
    home = os.path.dirname(os.path.dirname(os.path.realpath(sys.executable)))
    found = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix, home}
    return tuple(sorted(path for path in found if not any(_at_or_below(path, root) for root in SYSTEM_ROOTS)))


@dataclass(frozen=True)
class BwrapStatus:
    """What bubblewrap wrote on ``--json-status-fd``. ``exit_code`` is set only when the command ran and ended (a
    death by signal ``n`` is ``128 + n``); a launch whose setup failed (a bind, the exec) has a ``child_pid`` but no
    ``exit_code``, and so does a bubblewrap that was itself killed."""

    child_pid: int | None
    exit_code: int | None

    @classmethod
    def parse(cls, data: str | bytes) -> BwrapStatus:
        """Read the JSON objects bubblewrap wrote, one per line."""
        events = [json.loads(line) for line in data.splitlines() if line.strip()]
        child = next((event["child-pid"] for event in events if "child-pid" in event), None)
        code = next((event["exit-code"] for event in events if "exit-code" in event), None)
        return cls(child, code)


@dataclass(frozen=True)
class BwrapProbe:
    """Whether bubblewrap runs a sandbox here, and how: ``problems`` is empty when it does, else each names the step
    and bubblewrap's own line. ``prefix`` goes before every launch (``unshare --mount --`` where the root mount
    cannot be bound) and ``notes`` say why. A ``transient`` probe got no answer in time: never cached, never a
    final refusal."""

    path: str | None
    version: Version | None
    problems: tuple[str, ...]
    prefix: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    transient: bool = False

    @property
    def ok(self) -> bool:
        """Whether a sandbox can be launched here."""
        return not self.problems and self.path is not None and self.version is not None

    @property
    def launcher(self) -> tuple[str, ...]:
        """What a launch's argv starts with (:meth:`BwrapPolicy.argv`)."""
        if not self.ok:
            raise RuntimeError("bubblewrap cannot run a sandbox here: " + "; ".join(self.problems))
        return (*self.prefix, str(self.path))


_CANARY = r"""
import errno, json, os, sys
app_pidns, ro, rw, secret_dir, secret_file = sys.argv[1:]
def outcome(action):
    try:
        action()
    except OSError as exc:
        return errno.errorcode.get(exc.errno, str(exc.errno))
    return "ok"
def umount(path):
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.umount2(path.encode(), 0) != 0:
        raise OSError(ctypes.get_errno(), "umount2")
with open("/proc/self/status") as handle:
    status = dict(line.split(":", 1) for line in handle.read().splitlines() if ":" in line)
print(json.dumps({
    "mask_dir": outcome(lambda: os.listdir(secret_dir)),
    "mask_file": outcome(lambda: open(secret_file, "rb").close()),
    "ro_read": outcome(lambda: open(os.path.join(ro, "visible"), "rb").close()),
    "ro_create": outcome(lambda: open(os.path.join(ro, "created"), "wb").close()),
    "ro_chmod": outcome(lambda: os.chmod(os.path.join(ro, "visible"), 0o600)),
    "rw_write": outcome(lambda: open(os.path.join(rw, "created"), "wb").close()),
    "umount_mask": outcome(lambda: umount(secret_dir)),
    "pid_namespace": "own" if os.readlink("/proc/self/ns/pid") != app_pidns else "the app's",
    "NoNewPrivs": status.get("NoNewPrivs", "").strip(),
    "CapEff": status.get("CapEff", "").strip(),
}))
"""
_CANARY_EXPECTED = {
    "mask_dir": "EACCES",
    "mask_file": "EACCES",
    "ro_read": "ok",
    "ro_create": "EROFS",
    "ro_chmod": "EROFS",
    "rw_write": "ok",
    "umount_mask": "EPERM",
    "pid_namespace": "own",
    "NoNewPrivs": "1",
    "CapEff": "0000000000000000",
}


@dataclass(frozen=True)
class _Canary:
    """A throwaway tree the probe's sandbox reads, writes and must not see."""

    root: str
    scratch: Scratch

    @property
    def read_only(self) -> str:
        return os.path.join(self.root, "ro")

    def policy(self) -> BwrapPolicy:
        """The shape of an agent's policy: system roots, a read-only tree with a masked directory and file in it, a
        writable tree, private ``/tmp`` and ``/dev/shm`` (all below the host's ``/tmp``, so the depth order counts)."""
        return BwrapPolicy(
            read_only=(*SYSTEM_ROOTS, *interpreter_roots(), self.read_only),
            writable=(os.path.join(self.root, "rw"),),
            masked=(os.path.join(self.read_only, "secret"), os.path.join(self.read_only, "secretfile")),
            tmp=self.scratch.tmp,
            shm=self.scratch.shm,
            die_with_parent=False,
        )

    def run(self, launcher: Sequence[str], timeout: float) -> tuple[str, ...]:
        """The problems the canary found through ``launcher``; ``subprocess.TimeoutExpired`` when it did not end."""
        policy = self.policy()
        args = (os.readlink("/proc/self/ns/pid"), self.read_only, *policy.writable, *policy.masked)
        status, write = os.pipe()
        try:
            process = subprocess.Popen(
                policy.argv(launcher, [sys.executable, *_PYTHON_FLAGS, _CANARY, *args], status_fd=write),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={"PATH": os.environ.get("PATH", os.defpath)},
                pass_fds=(write,),
            )
        finally:
            os.close(write)
        with process, contextlib.closing(os.fdopen(status, "rb", buffering=0)) as status_file:
            try:
                out, err = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_launch(process, status_file.fileno())
                raise
        if process.returncode != 0:
            return (_last_line(err) or f"bubblewrap exited {process.returncode}",)
        try:
            seen = json.loads(_last_line(out))
        except ValueError:
            return (f"the sandbox printed no result: {_last_line(out)!r}",)
        return tuple(
            f"the sandbox showed {name} {seen.get(name)}"
            for name, expected in _CANARY_EXPECTED.items()
            if seen.get(name) != expected
        )


def _kill_launch(process: subprocess.Popen[str], status: int) -> None:
    """SIGKILL a launch that did not end: the sandbox's pid 1 first (named on the status descriptor ``status`` once
    bubblewrap has cloned it), which outlives bubblewrap's own process, then bubblewrap."""
    if select.select([status], [], [], 0)[0]:
        written = os.read(status, 65536).rpartition(b"\n")[0]
        child = BwrapStatus.parse(written).child_pid
        if child is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(child, signal.SIGKILL)
    process.kill()
    process.wait()


@contextlib.contextmanager
def _canary() -> Iterator[_Canary]:
    root = tempfile.mkdtemp(prefix="hl-bwrap-probe-")
    scratch = None
    try:
        for name in ("ro/secret", "rw"):
            os.makedirs(os.path.join(root, name))
        for name, text in (("ro/visible", "visible\n"), ("ro/secretfile", "secret\n"), ("ro/secret/inner", "secret\n")):
            with open(os.path.join(root, name), "w", encoding="ascii") as handle:
                handle.write(text)
        scratch = Scratch.create()
        yield _Canary(root, scratch)
    finally:
        if scratch is not None:
            scratch.remove()
        _remove_tree(root)


def _unbindable(problems: tuple[str, ...]) -> bool:
    return bool(problems) and "Can't bind mount" in problems[0] and "Invalid argument" in problems[0]


def _with_private_mounts(
    canary: _Canary, path: str, version: Version, first: tuple[str, ...], timeout: float
) -> BwrapProbe:
    """The canary again under ``unshare --mount``, after binds failed with ``EINVAL`` (an unbindable root)."""
    unshare = shutil.which("unshare")
    if unshare is None:
        return BwrapProbe(
            path, version, (*first, "unshare (util-linux) is not on PATH to give a launch its own mounts")
        )
    prefix = (unshare, *_PRIVATE_MOUNTS)
    again = canary.run((*prefix, path), timeout)
    if again:
        return BwrapProbe(path, version, (*first, *(f"under unshare --mount: {problem}" for problem in again)))
    note = (
        f"binds from the root mount fail here ({first[0]}): every launch runs under unshare --mount, "
        "which gives it a private copy of the mount tree"
    )
    return BwrapProbe(path, version, (), prefix=prefix, notes=(note,))


def _transient(path: str, version: Version | None, timeout: float) -> BwrapProbe:
    return BwrapProbe(path, version, (f"bubblewrap at {path}: no answer in {timeout:g} s",), transient=True)


def _canary_probe(path: str, version: Version, timeout: float) -> BwrapProbe:
    try:
        with _canary() as canary:
            problems = canary.run((path,), timeout)
            if _unbindable(problems):
                return _with_private_mounts(canary, path, version, problems, timeout)
            return BwrapProbe(path, version, problems)
    except subprocess.TimeoutExpired:
        return _transient(path, version, timeout)


def probe_bwrap(bwrap: str | None = None, *, timeout: float = _TIMEOUT) -> BwrapProbe:
    """Whether ``bwrap`` (default: the one on ``PATH``) runs a sandbox here: not setuid, at least
    :data:`BWRAP_FLOOR`, and a canary sandbox of an agent's shape holds (masks unreadable, read-only binds
    ``EROFS``, the writable tree writable, its own pid namespace, no capabilities, ``no_new_privs``). Not cached:
    :func:`bwrap_probe` is."""
    path = bwrap or shutil.which(_BWRAP)
    if path is None:
        return BwrapProbe(None, None, ("bubblewrap: not found on PATH",))
    try:
        if os.stat(path).st_mode & stat.S_ISUID:
            return BwrapProbe(
                path, None, (f"bubblewrap at {path} is setuid; only the unprivileged build is supported",)
            )
        shown = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return _transient(path, None, timeout)
    except OSError as exc:
        return BwrapProbe(path, None, (f"bubblewrap at {path}: {exc.strerror}",))
    match = _VERSION.search(shown.stdout)
    if match is None:
        return BwrapProbe(path, None, (f"bubblewrap at {path} did not say its version: {_last_line(shown.stderr)}",))
    version = (int(match[1]), int(match[2]), int(match[3]))
    if version < BWRAP_FLOOR:
        problem = f"bubblewrap {_shown(version)} is older than {_shown(BWRAP_FLOOR)} (needs --perms)"
        return BwrapProbe(path, version, (problem,))
    return _canary_probe(path, version, timeout)


_PROBES: dict[tuple[object, ...], BwrapProbe] = {}
_PROBES_LOCK = threading.Lock()


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="ascii") as handle:
            return handle.read().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _identity(path: str | None) -> tuple[int, ...] | None:
    try:
        info = os.stat(path) if path else None
    except OSError:
        return None
    return None if info is None else (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size)


def _namespace(name: str) -> int | None:
    """This process's namespace ``name`` (its nsfs inode number; the inode's times change when it is re-instantiated)."""
    try:
        return os.stat(f"/proc/self/ns/{name}").st_ino
    except OSError:
        return None


def _probe_key(path: str | None) -> tuple[object, ...]:
    """What the probe's answer depends on: the binary, the interpreter, the ids, the namespaces, the sysctls."""
    namespaces = tuple(_namespace(name) for name in ("user", "mnt", "pid"))
    sysctls = tuple(_read(sysctl) for sysctl in _USERNS_SYSCTLS)
    return (path, _identity(path), sys.executable, os.geteuid(), os.getegid(), namespaces, sysctls)


def bwrap_probe() -> BwrapProbe:
    """:func:`probe_bwrap` for the ``bwrap`` on ``PATH``, run once per process for each :func:`_probe_key`: a
    concurrent caller waits for the one run, and a transient answer is not kept."""
    path = shutil.which(_BWRAP)
    key = _probe_key(path)
    with _PROBES_LOCK:
        found = _PROBES.get(key)
        if found is None:
            found = probe_bwrap(path)
            if not found.transient:
                _PROBES[key] = found
                for note in found.notes:
                    log.info("bubblewrap: %s", note)
    return found


def _first_line(fd: int, deadline: float) -> bytes:
    """The first line read from ``fd``; what came before the end when it closes first."""
    data = b""
    while b"\n" not in data:
        left = deadline - time.monotonic()
        if left <= 0 or not select.select([fd], [], [], left)[0]:
            raise TimeoutError("bubblewrap did not answer in time")
        chunk = os.read(fd, 4096)
        if not chunk:
            return data
        data += chunk
    return data.split(b"\n", 1)[0]


@dataclass
class Holder:
    """A long-lived sandbox whose user and pid namespaces a sandbox's commands join (:meth:`joined`), so a server one
    command starts outlives it and a later command can see and signal it; each command still mounts its own tree
    and has its own IPC namespace.

    The holder has no ``--die-with-parent`` (the thread that starts it may end first) and no marks. Its pid 1 does
    not end with bubblewrap's own process, so the holder is held by a pidfd of that pid 1 (``pidfd``), and
    :meth:`stop` kills through it, which ends the namespace and every command in it.
    """

    process: subprocess.Popen[bytes]
    pid: int
    pidfd: int

    @classmethod
    def start(cls, launcher: Sequence[str], policy: BwrapPolicy, *, timeout: float = _TIMEOUT) -> Holder:
        """Start the holder in ``policy`` (which must bind ``sys.executable``'s :func:`interpreter_roots`).

        Raises:
            BwrapSetupError: bubblewrap could not set the sandbox up.
            TimeoutError: it neither started nor failed within ``timeout`` (the attempt is killed).
        """
        policy = dataclasses.replace(policy, die_with_parent=False, join=None, marks=())
        status, write = os.pipe()
        try:
            process = subprocess.Popen(
                policy.argv(launcher, [sys.executable, *_PYTHON_FLAGS, _HOLD], status_fd=write),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(write,),
                start_new_session=True,
            )
        finally:
            os.close(write)
        holder = cls(process, 0, -1)
        try:
            deadline = time.monotonic() + timeout
            cloned = BwrapStatus.parse(_first_line(status, deadline))
            with contextlib.suppress(ProcessLookupError):
                if cloned.child_pid is not None:
                    holder = cls(process, cloned.child_pid, os.pidfd_open(cloned.child_pid))
            ready = _first_line(process.stdout.fileno(), deadline) if process.stdout else b""
        except TimeoutError:
            holder.stop()
            raise
        finally:
            os.close(status)
        if ready != b"ready" or holder.pidfd < 0:
            printed = holder.stop()
            raise BwrapSetupError(_last_line(printed) or f"bubblewrap exited {process.returncode}")
        return holder

    def _ended(self) -> bool:
        return self.pidfd < 0 or select.select([self.pidfd], [], [], 0)[0] != []

    def _gone(self) -> ProcessLookupError:
        return ProcessLookupError(errno.ESRCH, "the sandbox holder has ended", str(self.pid))

    @contextlib.contextmanager
    def _namespace(self, name: str) -> Iterator[int]:
        """A descriptor of the holder's namespace ``name``, closed when the block ends."""
        try:
            fd = os.open(f"/proc/{self.pid}/ns/{name}", os.O_RDONLY | os.O_CLOEXEC)
        except FileNotFoundError:
            raise self._gone() from None
        try:
            yield fd
        finally:
            os.close(fd)

    @contextlib.contextmanager
    def joined(self, policy: BwrapPolicy) -> Iterator[BwrapPolicy]:
        """``policy`` joining this holder's namespaces, for launches started inside the ``with``: give the Popen
        ``pass_fds=joined.join``. The descriptors close when the block ends.

        Raises:
            ProcessLookupError: the holder has ended.
        """
        with self._namespace("user") as user, self._namespace("pid") as pid:
            # Checked after opening: the pidfd still pins the holder's pid 1, so the pid was not reused by another
            # process whose namespaces were opened instead.
            if self._ended():
                raise self._gone()
            yield dataclasses.replace(policy, join=(user, pid))

    def stop(self) -> str:
        """End the sandbox and every command in it (SIGKILL to its pid 1). Returns what bubblewrap printed on
        stderr."""
        if self.pidfd >= 0:
            with contextlib.suppress(ProcessLookupError):
                signal.pidfd_send_signal(self.pidfd, signal.SIGKILL)
            os.close(self.pidfd)
            self.pidfd = -1
        if self.process.poll() is None:
            self.process.kill()
        _out, err = self.process.communicate()
        return err.decode(errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    """``python -I -m hyperloom_common.bwrap probe``: ``ok bubblewrap X.Y.Z PATH`` and any note, exit 0; or one line
    per problem, exit 1."""
    parser = argparse.ArgumentParser(
        prog="python -I -m hyperloom_common.bwrap", description="Check that bubblewrap can sandbox an agent here."
    )
    parser.add_argument("command", choices=("probe",))
    parser.parse_args(argv)
    found = probe_bwrap()
    if found.ok and found.version is not None:
        print(f"ok bubblewrap {_shown(found.version)} {found.path}")
    for line in (*found.notes, *found.problems):
        print(line)
    return 0 if found.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
