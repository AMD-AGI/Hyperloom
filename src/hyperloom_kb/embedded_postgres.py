# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One PostgreSQL server per data home, run from the binaries the pgembed wheel ships and reachable only by socket.

PostgreSQL refuses to run as root, so a root caller runs it as a dedicated system user. The user follows the data
directory: an existing directory keeps the uid that owns it, so a recreated container that mounts the same home still
starts it. No directory's permissions are ever widened; a home the database user cannot reach is refused instead, as is
a home on a file system that cannot keep the data directory private to that user. It runs on Linux and macOS, on the
Python 3.12 or newer the pgembed wheel installs on; elsewhere a service is given a PostgreSQL server instead.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shlex
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

if os.name == "posix":
    import grp
    import pwd

PGDATA_DIR = "postgres"
DB_USER = "hyperloom-kb-db"
DATABASE = "postgres"
START_TIMEOUT_SECONDS = 60
# ``sun_path`` holds 108 bytes on Linux and 104 on macOS, terminator included, and the server names its socket
# ``.s.PGSQL.<port>``.
_SOCKET_PATH_LIMIT = 103 if sys.platform == "darwin" else 107
_SOCKET_NAME = ".s.PGSQL.5432"
# PostgreSQL's own default socket directory: short on every POSIX system, unlike a TMPDIR.
_SHORT_SOCKET_ROOT = Path("/tmp")
_LOG = "server.log"
# PostgreSQL starts only on a data directory that no one but its owner, and at most its group, may read.
_PGDATA_FORBIDDEN_MODE = 0o027


class EmbeddedPostgresError(RuntimeError):
    """Raised when the embedded PostgreSQL server of a home cannot be brought to serving."""


@dataclass(frozen=True)
class _DatabaseUser:
    """Who runs the server: the caller itself, or for a root caller a dedicated system user."""

    name: str
    uid: int | None = None
    gid: int | None = None

    def owns(self, path: Path) -> bool:
        return self.uid is None or path.stat().st_uid == self.uid


def unavailable_reason() -> str:
    """Why this system cannot run an embedded server, or ``""`` when it can."""

    if os.name != "posix":
        return "the embedded database runs on Linux and macOS"
    if importlib.util.find_spec("pgembed") is None:
        return "the embedded database needs the pgembed package, which installs on Python 3.12 or newer"
    return ""


def _binaries() -> Path:
    reason = unavailable_reason()
    if reason:
        raise EmbeddedPostgresError(reason)
    spec = importlib.util.find_spec("pgembed")
    if spec is None or not spec.submodule_search_locations:
        raise EmbeddedPostgresError("the pgembed package that provides the embedded PostgreSQL is not installed")
    binaries = Path(next(iter(spec.submodule_search_locations))) / "pginstall" / "bin"
    if not (binaries / "pg_ctl").is_file():
        raise EmbeddedPostgresError(f"pgembed ships no PostgreSQL binaries at {binaries}")
    return binaries


def _account(*command: str) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise EmbeddedPostgresError(f"cannot create the database account ({' '.join(command)}): {detail}") from exc


def _system_user(*arguments: str) -> None:
    _account("useradd", "--system", "--no-create-home", "--shell", "/usr/sbin/nologin", *arguments)


def _owner_of(pgdata: Path) -> _DatabaseUser:
    """The user owning an initialized ``pgdata``, given an account first when this system has none for its uid."""

    owner = pgdata.stat()
    try:
        return _DatabaseUser(pwd.getpwuid(owner.st_uid).pw_name, owner.st_uid, owner.st_gid)
    except KeyError:
        pass
    try:
        grp.getgrgid(owner.st_gid)
    except KeyError:
        _account("groupadd", "--non-unique", "--gid", str(owner.st_gid), f"{DB_USER}-{owner.st_gid}")
    name = f"{DB_USER}-{owner.st_uid}" if _user_exists(DB_USER) else DB_USER
    _system_user("--non-unique", "--uid", str(owner.st_uid), "--gid", str(owner.st_gid), name)
    return _DatabaseUser(name, owner.st_uid, owner.st_gid)


def _database_user(pgdata: Path) -> _DatabaseUser:
    if os.geteuid() != 0:
        return _DatabaseUser(pwd.getpwuid(os.geteuid()).pw_name)
    if (pgdata / "PG_VERSION").is_file() and pgdata.stat().st_uid != 0:
        return _owner_of(pgdata)
    if not _user_exists(DB_USER):
        _system_user("--user-group", DB_USER)
    entry = pwd.getpwnam(DB_USER)
    return _DatabaseUser(DB_USER, entry.pw_uid, entry.pw_gid)


def _user_exists(name: str) -> bool:
    try:
        pwd.getpwnam(name)
    except KeyError:
        return False
    return True


def _blocking_directory(path: Path, uid: int, groups: set[int]) -> tuple[Path, str] | None:
    """The first existing directory on the way to ``path`` that ``uid`` in ``groups`` may not traverse, and its mode."""

    for directory in (path, *path.parents):
        if not directory.exists():
            continue
        status = directory.stat()
        if status.st_uid == uid:
            allowed = status.st_mode & stat.S_IXUSR
        elif status.st_gid in groups:
            allowed = status.st_mode & stat.S_IXGRP
        else:
            allowed = status.st_mode & stat.S_IXOTH
        if not allowed:
            return directory, stat.filemode(status.st_mode)
    return None


def _require_reachable(path: Path, user: _DatabaseUser, what: str) -> None:
    """Refuse a ``path`` the database user cannot reach rather than widen any directory on the way to it."""

    if user.uid is None or user.gid is None:
        return
    blocking = _blocking_directory(path, user.uid, set(os.getgrouplist(user.name, user.gid)))
    if blocking is not None:
        directory, mode = blocking
        raise EmbeddedPostgresError(
            f"the database user {user.name} (uid {user.uid}) cannot reach {what} through {directory} ({mode}); "
            "place it under directories every user may traverse"
        )


def root_run_problem(home: Path) -> str:
    """Why a root caller could not run ``home``'s database, or ``""`` when it could.

    The server would run as a dedicated user that owns none of the directories on the way to the home or to the
    binaries, so each of them must let other users traverse it.
    """

    for path, what in ((home.expanduser().absolute(), "the KB home"), (_binaries(), "the PostgreSQL binaries")):
        blocking = _blocking_directory(path, -1, set())
        if blocking is not None:
            directory, mode = blocking
            return f"{what} {path} is under {directory} ({mode}), which other users may not traverse"
    return ""


def _give(path: Path, user: _DatabaseUser) -> None:
    if user.uid is None or user.gid is None:
        return
    try:
        os.chown(path, user.uid, user.gid)
    except PermissionError as exc:
        raise EmbeddedPostgresError(
            f"cannot give {path} to the database user {user.name}: {exc.strerror}; its file system refuses ownership "
            "changes, as NFS exported with root_squash does, so place the KB home on a local file system"
        ) from exc


def _require_private(pgdata: Path) -> None:
    mode = pgdata.stat().st_mode
    if stat.S_IMODE(mode) & _PGDATA_FORBIDDEN_MODE:
        raise EmbeddedPostgresError(
            f"{pgdata} is {stat.filemode(mode)}, but PostgreSQL needs its data directory private to its owner; its file "
            "system does not keep that, as a Windows drive mounted into WSL does not, so place the KB home on a local "
            "Linux or macOS file system"
        )


def _run(user: _DatabaseUser, command: list[str], *, cwd: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    # The server inherits whatever this process hands it, so a pipe would keep ``pg_ctl start`` waiting on output
    # that never closes; its own output goes to a file instead.
    with tempfile.TemporaryFile("w+") as output:
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                user=user.uid,
                group=user.gid,
                extra_groups=[] if user.uid is not None else None,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise EmbeddedPostgresError(f"{Path(command[0]).name} did not finish: {exc}") from exc
        output.seek(0)
        return subprocess.CompletedProcess(command, result.returncode, output.read(), "")


def _socket_directory(pgdata: Path, user: _DatabaseUser) -> Path:
    if len(str(pgdata / _SOCKET_NAME)) <= _SOCKET_PATH_LIMIT:
        return pgdata
    digest = hashlib.sha256(str(pgdata.resolve()).encode()).hexdigest()[:12]
    directory = _SHORT_SOCKET_ROOT / f"hyperloom-kb-{digest}"
    directory.mkdir(mode=0o700, exist_ok=True)
    _give(directory, user)
    return directory


def _conninfo(socket_directory: Path) -> str:
    return f"host={socket_directory} dbname={DATABASE} user=postgres"


@dataclass(frozen=True)
class EmbeddedPostgres:
    """A running embedded server; ``started`` says whether this process started it and so owns stopping it."""

    pgdata: Path
    conninfo: str
    started: bool
    _user: _DatabaseUser
    _binaries: Path

    def stop(self) -> None:
        command = [str(self._binaries / "pg_ctl"), "-D", str(self.pgdata), "-m", "fast", "-w", "stop"]
        result = _run(self._user, command, cwd=self.pgdata.parent, timeout=START_TIMEOUT_SECONDS)
        if result.returncode != 0:
            raise EmbeddedPostgresError(f"pg_ctl stop failed: {result.stdout.strip()[-500:]}")


def _running_socket_directory(pgdata: Path) -> Path | None:
    """The socket directory a running server of ``pgdata`` listens in, read from its ``postmaster.pid``."""

    try:
        lines = (pgdata / "postmaster.pid").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    return Path(lines[4]) if len(lines) > 4 and lines[4] else None


def start_embedded_postgres(home: Path) -> EmbeddedPostgres:
    """Serve ``home``'s database, initializing it on first use and reusing a server that already serves it."""

    binaries = _binaries()
    home = home.expanduser().resolve()
    pgdata = home / PGDATA_DIR
    home.mkdir(parents=True, exist_ok=True)
    user = _database_user(pgdata)
    _require_reachable(pgdata.parent, user, f"the database directory {pgdata}")
    _require_reachable(binaries, user, f"the PostgreSQL binaries at {binaries}")
    if not (pgdata / "PG_VERSION").is_file():
        pgdata.mkdir(exist_ok=True)
        pgdata.chmod(0o700)
        _give(pgdata, user)
        _require_private(pgdata)
        initdb = [
            str(binaries / "initdb"),
            "-D",
            str(pgdata),
            "--auth=trust",
            "--encoding=UTF8",
            "--username=postgres",
            "--no-instructions",
        ]
        result = _run(user, initdb, cwd=home, timeout=START_TIMEOUT_SECONDS)
        if result.returncode != 0:
            raise EmbeddedPostgresError(f"initdb of {pgdata} failed: {result.stdout.strip()[-1000:]}")
    if not user.owns(pgdata):
        raise EmbeddedPostgresError(
            f"{pgdata} belongs to uid {pgdata.stat().st_uid}, not the database user {user.name}"
        )
    pg_ctl = str(binaries / "pg_ctl")
    status = _run(user, [pg_ctl, "-D", str(pgdata), "status"], cwd=home, timeout=START_TIMEOUT_SECONDS)
    running = _running_socket_directory(pgdata)
    if status.returncode == 0 and running is not None:
        return EmbeddedPostgres(pgdata, _conninfo(running), False, user, binaries)
    socket_directory = _socket_directory(pgdata, user)
    options = f"-c listen_addresses='' -c unix_socket_permissions=0700 -k {shlex.quote(str(socket_directory))}"
    command = [pg_ctl, "-D", str(pgdata), "-l", str(pgdata / _LOG), "-w", "-t", str(START_TIMEOUT_SECONDS)]
    result = _run(user, [*command, "-o", options, "start"], cwd=home, timeout=START_TIMEOUT_SECONDS + 10)
    if result.returncode != 0:
        try:
            log_tail = (pgdata / _LOG).read_text(encoding="utf-8", errors="replace").strip()[-1000:]
        except OSError:
            log_tail = ""
        raise EmbeddedPostgresError(f"PostgreSQL did not start on {pgdata}: {result.stdout.strip()[-500:]} {log_tail}")
    return EmbeddedPostgres(pgdata, _conninfo(socket_directory), True, user, binaries)


__all__ = [
    "DB_USER",
    "PGDATA_DIR",
    "EmbeddedPostgres",
    "EmbeddedPostgresError",
    "root_run_problem",
    "start_embedded_postgres",
    "unavailable_reason",
]
