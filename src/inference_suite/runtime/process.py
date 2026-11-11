# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which processes a round owns, and ending them.

Every child starts in a session of its own with a fresh run mark in its env. What a reap reaches is the child's tree
and every process still carrying its mark, however it detached; it never scans the host by port or command line, and
never signals the caller's own process group. ``server.pid`` names a server across rounds: the incarnation it was
(pid and start time), its mark, and the launch it runs.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from hyperloom_common.contract_fields import ContractFields, field_names
from hyperloom_common.proctree import (
    REAP_MARK_ENV,
    collect_tree,
    group_members,
    kill_tree,
    marked_pids,
    proc_identity,
    running,
)

from inference_suite.frameworks import Framework

log = logging.getLogger(__name__)

#: How long a reaped child is given to be collected once nothing of its tree is left running.
_COLLECT_SEC = 1.0
#: How long SIGKILL is given to take effect: a server rank frees its device memory and unpins its host cache as it
#: exits, which took 5-9 s for each rank of a TP4 GLM-5.3 server with HiCache on MI355X.
_KILL_CONFIRM_SEC = 60.0
_PROC = Path("/proc")
#: ``/proc/net`` tables of the TCP sockets a reuse check looks for the server's listener in.
_TCP_TABLES = ("tcp", "tcp6")
#: The socket state ``/proc/net/tcp`` writes for a listener.
_LISTEN_STATE = "0A"
_SOCKET_LINK = "socket:["
_ZOMBIE = "Z"


class PidfileError(ValueError):
    """``server.pid`` does not hold the document this module writes."""


_FIELDS = ContractFields(PidfileError)


@dataclass(frozen=True)
class Reaped:
    """What one reap reached: every pid it signalled, and those still running after SIGKILL."""

    reaped: tuple[int, ...]
    survivors: tuple[int, ...]


def _alive(pid: int, start_time: int) -> bool:
    identity = proc_identity(pid)
    return identity is not None and identity[1] == start_time and running(pid)


def reap(roots: Iterable[int], mark: str) -> Reaped:
    """SIGTERM, then SIGKILL, the trees under ``roots`` and every process whose env carries ``mark``."""

    tree = collect_tree([*roots, *marked_pids(REAP_MARK_ENV, mark)])
    if not tree.members and not tree.groups:
        return Reaped((), ())
    kill_tree(tree, confirm_sec=_KILL_CONFIRM_SEC)
    survivors = tuple(sorted({pid for pid, start_time in tree.members if _alive(pid, start_time)}))
    return Reaped(tuple(sorted({pid for pid, _ in tree.members})), survivors)


@dataclass(frozen=True)
class Spawned:
    """A child this round started, and the mark everything it starts inherits."""

    proc: subprocess.Popen[bytes]
    mark: str

    def reap(self) -> Reaped:
        # Once collected, the pid may name an unrelated process; until then even a zombie holds it.
        roots = () if self.proc.poll() is not None else (self.proc.pid,)
        outcome = reap(roots, self.mark)
        try:
            self.proc.wait(timeout=_COLLECT_SEC)
        except subprocess.TimeoutExpired:
            log.warning("pid %d outlived its reap; leaving it to init to collect", self.proc.pid)
        return outcome


def spawn(
    argv: Sequence[str], env: Mapping[str, str], stdout: Path, stderr: Path | None = None, cwd: Path | None = None
) -> Spawned:
    """Start ``argv`` in its own session with exactly ``env`` and a fresh mark, appending its output to ``stdout``.

    ``stderr`` defaults to the same file. Appending keeps what an earlier server wrote to a shared log. ``cwd`` is where
    the child starts, this process's own directory when ``None``.
    """

    mark = uuid.uuid4().hex
    with ExitStack() as stack:
        files = []
        for path in (stdout, stderr):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                files.append(stack.enter_context(path.open("ab")))
        proc = subprocess.Popen(
            list(argv),
            env={**env, REAP_MARK_ENV: mark},
            stdin=subprocess.DEVNULL,
            stdout=files[0],
            stderr=files[1] if len(files) > 1 else subprocess.STDOUT,
            start_new_session=True,
            cwd=cwd,
        )
    return Spawned(proc, mark)


def listening_inodes(port: int) -> frozenset[int]:
    """The inodes of the TCP sockets listening on ``port`` in this network namespace."""

    inodes: set[int] = set()
    for table in _TCP_TABLES:
        try:
            rows = (_PROC / "net" / table).read_text(encoding="ascii").splitlines()[1:]
        except FileNotFoundError:
            continue
        for row in rows:
            fields = row.split()
            if fields[3] == _LISTEN_STATE and int(fields[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(int(fields[9]))
    return frozenset(inodes)


def socket_inodes(pid: int) -> frozenset[int]:
    """The inodes of the sockets ``pid`` holds open; empty for a process that is gone or not ours to read."""

    fd_dir = _PROC / str(pid) / "fd"
    try:
        fds = os.listdir(fd_dir)
    except OSError:
        return frozenset()
    inodes: set[int] = set()
    for fd in fds:
        try:
            target = os.readlink(fd_dir / fd)
        except OSError:
            continue
        if target.startswith(_SOCKET_LINK):
            inodes.add(int(target[len(_SOCKET_LINK) : -1]))
    return frozenset(inodes)


def group_listens(pgid: int, port: int) -> bool:
    """Whether a live member of process group ``pgid`` holds the socket listening on ``port``."""

    listening = listening_inodes(port)
    return bool(listening) and any(
        listening & socket_inodes(pid) for pid, _start, state in group_members(pgid) if state != _ZOMBIE
    )


@dataclass(frozen=True)
class ServerPidfile:
    """``server.pid``: the server incarnation a round started, how to find what it left, and the launch it runs."""

    pid: int
    pgid: int
    start_time: int
    run_mark: str
    port: int
    framework: Framework
    launch_digest: str

    def __post_init__(self) -> None:
        for name in ("pid", "pgid", "port"):
            _FIELDS.integer(getattr(self, name), f"the pidfile's {name}", minimum=1)
        _FIELDS.integer(self.start_time, "the pidfile's start_time", minimum=0)
        _FIELDS.text(self.run_mark, "the pidfile's run_mark")
        _FIELDS.text(self.launch_digest, "the pidfile's launch_digest")
        object.__setattr__(self, "framework", _FIELDS.enum(Framework, self.framework, "the pidfile's framework"))

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {name: getattr(self, name) for name in field_names(type(self))}
        document["framework"] = self.framework.value
        return document

    @classmethod
    def from_document(cls, document: object) -> ServerPidfile:
        fields = _FIELDS.known(document, field_names(cls), "the pidfile")
        return cls(**{name: fields.get(name) for name in field_names(cls)})

    @property
    def leader_alive(self) -> bool:
        """Whether the recorded pid still names the incarnation that was started, not a later reuse of the pid."""
        return _alive(self.pid, self.start_time)

    def write(self, path: Path) -> None:
        """Replace ``path`` whole, so a concurrent ``stop`` never reads half of it."""

        staging = path.with_name(f".{path.name}.{os.getpid()}")
        staging.write_text(json.dumps(self.to_document(), sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, path)

    @classmethod
    def read(cls, path: Path) -> ServerPidfile | None:
        """The pidfile at ``path``, ``None`` when there is none; one that does not parse raises :class:`PidfileError`."""

        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            document = json.loads(text)
        except ValueError as exc:
            raise PidfileError(f"{path} is not JSON: {exc}") from exc
        return cls.from_document(document)


__all__ = [
    "PidfileError",
    "Reaped",
    "ServerPidfile",
    "Spawned",
    "group_listens",
    "listening_inodes",
    "reap",
    "socket_inodes",
    "spawn",
]
