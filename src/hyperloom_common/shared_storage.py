# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Directories and files that several accounts of one group write in turn.

On shared storage each container of a run may be another host account. The accounts share one group, and the key
directories are made setgid in it once (``2775``), so what is created below them takes that group.

Permissions are widened only where the storage is set up to be shared: when the nearest existing directory above what
is created is setgid, group-writable and not writable by others (:func:`parent_is_shared`), judged on the directory the
walk to it holds open, never by a second look up the path. There what this module creates has an explicit mode, set by
``fchmod`` after the create: directories ``2775`` (each one of a chain it makes) and files ``664``, whatever the umask.
Anywhere else (a ``0755`` home, a ``1777`` ``/tmp``, a setgid directory the group cannot write, one reached through a
link at its own name) the umask alone decides, exactly as for a plain ``mkdir`` or ``open``, so what is made there stays
as private as the account keeps its files; a mode asked for is masked by the umask, never widened. A mode narrower than
the shared one (``0600`` for a file that holds a credential) is kept as asked, shared or not. Nothing is chowned, and an existing entry is never chmodded (another account's would fail ``EPERM``): a file is
replaced by rename instead, and never opened in place, since a member of the group may have put a link there. Nor is
a link followed on the way to it once the way has passed a directory another account can write or owns
(:func:`open_directory`): a member of the group, or that owner, could have put one there for any directory of the path. The umask of
this process is never changed, since its threads share it and it would also open up the files only this account may
read.

Nothing the application executes or sources may live in a tree the group can write.
"""

from __future__ import annotations

import errno
import logging
import os
import secrets
import shutil
import stat
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SHARED_DIR_MODE = 0o2775
SHARED_FILE_MODE = 0o664

_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
#: How each directory of a path is opened on the way (:func:`open_directory`): only to name what is below it, which
#: needs no read permission (a home directory may be ``0711``), and never through a link.
_WAY = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
#: How a link on the way is opened to check whose it is and read it (:meth:`_Walk.follow`).
_LINK = os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC
_NEW_FILE = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
_APPEND = os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC
_PROBE_PREFIX = ".hl-writable-"
#: What a new file is made with, before ``fchmod`` gives it its mode: the umask never decides it.
_OWNER_ONLY = 0o600
#: The suffix of the new file a writer makes beside the one it replaces.
_TEMPORARY_SUFFIX = ".tmp"
#: How many names a new temporary file is tried at before the writer gives up.
_TEMPORARY_TRIES = 100
#: How many links one path may go through, as the kernel allows (``MAXSYMLINKS``).
_MAX_LINKS = 40
#: A directory with one of these bits lets accounts other than its owner put a link in it.
_OPEN_TO_OTHERS = stat.S_IWGRP | stat.S_IWOTH
#: When a link cannot be made, a new file is made at its final name instead (:func:`create_new_shared_file`).
_NO_LINKS = frozenset({errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS})
#: What the kernel writes its reports under /proc in.
_PROC_ENCODING = "ascii"


def open_directory(path: Path, *, make: int | None = None) -> int:
    """``path`` opened as a directory: an ``O_PATH`` descriptor (the caller closes it) that what is written below it
    goes through, as ``dir_fd``; :func:`read_directory` opens it for listing or a flush.

    Each directory of the path is opened below the one before it, never following another account's link once the way
    has passed a directory that accounts other than its owner can write (:data:`_OPEN_TO_OTHERS`, the shared tree
    itself, ``/tmp``) or that another account than this one and root owns (a member's home or project directory):
    any member of the group, or that owner, could have put one there, to any directory the account that follows it
    can write. Links above that point (``/var/run``, a mount reached through a link of the host's), and this account's
    and root's own anywhere, are followed. In a user namespace host root, and every account it does not map, own their
    directories as the kernel's overflow id; whether one namespace maps a single id (a container's root) or several
    ranges (rootless subuids, a Kubernetes pod without host users, where the overflow id is itself mapped), a directory
    of that id is trusted as root's and only its mode decides (:func:`_overflow`). The tradeoff: a directory owned by a
    real account that is the overflow id (a host ``nobody``) is trusted as root's too. A link of that id is still not
    followed past a directory others can write or own. With ``make``, each missing directory is made with that mode
    (:func:`make_shared_dirs`).

    Raises:
        OSError: ``ELOOP`` for a link not followed or too many links, ``ENOENT`` for a missing directory without
            ``make``, ``FileExistsError`` (with ``make``) or ``ENOTDIR`` for something else where a directory belongs;
            each names the part of the path it is about.
    """
    return _walk_to(path, make=make).descriptor


def parent_is_shared(path: Path) -> bool:
    """Whether what is created at ``path`` lands in storage set up to be shared: the nearest existing directory above
    ``path`` is setgid, group-writable and not writable by others, and was not reached through a link standing at its
    own name.

    It is judged on that directory as the walk of :func:`open_directory` holds it open, the one every writer of this
    module judges and writes through. A directory that cannot be reached counts as not shared: the umask then decides,
    which can only keep what is made more private.
    """
    try:
        walk = _walk_to(Path(os.path.abspath(path)).parent, make=None, stop_at_missing=True)
    except OSError:
        return False
    try:
        return walk.set_up_to_share()
    finally:
        os.close(walk.descriptor)


def _walk_to(path: Path, *, make: int | None, stop_at_missing: bool = False) -> _Walk:
    """The walk of :func:`open_directory` to ``path``, done: its :attr:`_Walk.descriptor` is the caller's to close.
    With ``stop_at_missing`` it ends at the nearest directory of ``path`` that exists."""
    parts = list(Path(os.path.abspath(path)).parts[1:])
    walk = _Walk(os.open("/", _WAY))
    try:
        while parts:
            walk.arrive(len(parts))
            name = parts.pop(0)
            if name in ("", "."):
                continue
            if stop_at_missing and not _exists(walk.descriptor, name):
                walk.linked = walk.linked or bool(walk.expanding)  # a link to nothing stands there: not shared
                break
            info = walk.look_up(name, make)
            if info is None:
                parts.insert(0, name)  # made meanwhile by another process: looked at again
            elif stat.S_ISLNK(info.st_mode):
                parts[:0] = walk.follow(name, len(parts))
            else:
                walk.enter(name, info, make)
        else:
            walk.arrive(0)
    except BaseException:
        os.close(walk.descriptor)
        raise
    return walk


def _exists(directory: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _set_up_to_share(mode: int) -> bool:
    """Whether a directory of ``mode`` is storage set up to be shared: setgid, group-writable, not writable by others."""
    return stat.S_ISDIR(mode) and bool(mode & stat.S_ISGID) and bool(mode & stat.S_IWGRP) and not mode & stat.S_IWOTH


def _trusted(uid: int) -> bool:
    """Whether what ``uid`` owns is this account's or root's: no other account can make or change it."""
    return uid in (0, os.geteuid())


def _owner_trusted(uid: int) -> bool:
    """Whether a directory ``uid`` owns leaves its links to be followed, as far as its owner goes: one of
    :func:`_trusted`'s, or the overflow id, which may be root (:func:`_overflow`)."""
    return _trusted(uid) or _overflow(uid)


def _overflow(uid: int) -> bool:
    """Whether ``uid`` is the kernel's overflow id, the one a user namespace shows for an id it does not map: there
    host root, whose directories (``/``, ``/var``, a host bind mount) every path passes, cannot be told from it, in a
    namespace of one id or of several ranges (where the overflow id may itself be mapped). ``False`` when the kernel
    does not say."""
    try:
        with open("/proc/sys/kernel/overflowuid", encoding=_PROC_ENCODING) as overflow:
            return uid == int(overflow.read())
    except (OSError, ValueError):
        return False


class _Walk:
    """Where :func:`open_directory` is on its way: the directory open as :attr:`descriptor` (``O_PATH``) is
    :attr:`here`, and :attr:`shared` once a directory passed can be written by accounts other than its owner, or is
    owned by an account other than this one and root (which can put any link there). The overflow id may be root as a
    user namespace shows it (:func:`_overflow`), whose directories (``/``, ``/var``) every path passes, so only the
    mode decides for its directories."""

    def __init__(self, descriptor: int) -> None:
        self.descriptor = descriptor
        self.here = Path("/")
        self.shared = False
        self.links = 0
        #: How many parts of the path were left after each link being walked: its target is walked until as many are.
        self.expanding: list[int] = []
        #: Whether :attr:`here` is the end of a link's target: a link stood at its name.
        self.linked = False
        #: Whether the directories made on the way are widened (:meth:`set_up_to_share`), judged once, at the first.
        self.widen: bool | None = None

    def arrive(self, left: int) -> None:
        """Note that ``left`` parts of the path are still to walk: a link whose target is walked to its end leaves
        :attr:`linked` set, until the walk moves on."""
        ended = [mark for mark in self.expanding if mark >= left]
        self.linked = bool(ended)
        self.expanding = [mark for mark in self.expanding if mark < left]

    def set_up_to_share(self) -> bool:
        """Whether :attr:`here` is storage set up to be shared (:func:`parent_is_shared`)."""
        return not self.linked and _set_up_to_share(os.fstat(self.descriptor).st_mode)

    def look_up(self, name: str, make: int | None) -> os.stat_result | None:
        """``lstat`` of ``name`` here, made first with ``make`` when it is missing; ``None`` when another process
        made it meanwhile."""
        here = os.fstat(self.descriptor)
        self.shared = self.shared or bool(here.st_mode & _OPEN_TO_OTHERS) or not _owner_trusted(here.st_uid)
        try:
            return os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            if make is None:
                raise FileNotFoundError(errno.ENOENT, "no such directory", str(self.here / name)) from None
        if self.expanding:
            # as mkdir(2) does: nothing is made at the missing target of a link
            raise FileExistsError(errno.EEXIST, "a link to a directory that does not exist", str(self.here / name))
        if self.widen is None:
            self.widen = self.set_up_to_share()
        if not _make_directory(self.descriptor, name, make, self.here / name, widen=self.widen):
            return None
        return os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)

    def follow(self, name: str, left: int) -> tuple[str, ...]:
        """The parts of the link ``name`` here to walk instead of it, ``left`` parts of the path being left after it;
        an absolute one starts again from ``/``.

        Past a directory others can write or another account owns, only a link of this account's or root's is
        followed, which no other account can make (one of the overflow id is not): it is
        opened itself (``O_PATH``), so the link whose owner is checked is the one read.

        Raises:
            OSError: ``ELOOP`` for another account's link past such a directory, or after too many links.
        """
        link = os.open(name, _LINK, dir_fd=self.descriptor)
        try:
            owner = os.fstat(link).st_uid
            if self.shared and not _trusted(owner):
                raise OSError(
                    errno.ELOOP,
                    f"a link of uid {owner} is not followed past a directory another account can write or owns",
                    str(self.here / name),
                )
            target = Path(os.readlink("", dir_fd=link))
        finally:
            os.close(link)
        self.links += 1
        if self.links > _MAX_LINKS:
            raise OSError(errno.ELOOP, os.strerror(errno.ELOOP), str(self.here / name))
        self.expanding.append(left)
        if not target.is_absolute():
            return target.parts
        self._move("/", Path("/"))
        self.shared = False  # the target's own way decides from here
        return target.parts[1:]

    def enter(self, name: str, info: os.stat_result, make: int | None) -> None:
        """Go into the directory ``name`` here, whose ``lstat`` is ``info``."""
        if not stat.S_ISDIR(info.st_mode):
            if make is not None:
                raise FileExistsError(errno.EEXIST, "not a directory", str(self.here / name))
            raise NotADirectoryError(errno.ENOTDIR, "not a directory", str(self.here / name))
        self._move(name, self.here.parent if name == ".." else self.here / name, self.descriptor)  # ELOOP if changed

    def _move(self, name: str, here: Path, directory: int | None = None) -> None:
        """Hold the directory ``name`` of ``directory`` (``here``) instead, opened as the way is (:data:`_WAY`): it is
        the walk's from the open on, whatever closing the last one does."""
        left = self.descriptor
        self.descriptor = os.open(name, _WAY, dir_fd=directory)
        self.here = here
        os.close(left)


def read_directory(descriptor: int) -> int:
    """The directory open as ``descriptor`` (:func:`open_directory`) opened again for reading: to list it or flush it.
    The caller closes it."""
    return os.open(".", _DIRECTORY, dir_fd=descriptor)


def _make_directory(parent: int, name: str, mode: int, made: Path, *, widen: bool) -> bool:
    """Make the directory ``name`` (``made``) of ``parent``: with ``mode`` set by ``fchmod`` when ``widen``, else with
    ``mode`` as the umask masks it; ``False`` when it was there already."""
    try:
        os.mkdir(name, mode, dir_fd=parent)
    except FileExistsError:
        return False
    except OSError as exc:
        raise type(exc)(exc.errno, exc.strerror, str(made)) from None  # the whole path, not the name below dir_fd
    if widen:
        _share(parent, name, mode, made)
    return True


def _share(parent: int, name: str, mode: int, directory: Path) -> None:
    """Give the directory ``name`` of ``parent`` (``directory``), opened without following a link, its ``mode``;
    logged when it does not keep setgid."""
    descriptor = os.open(name, _DIRECTORY, dir_fd=parent)
    try:
        os.fchmod(descriptor, mode)
        setgid_kept = os.fstat(descriptor).st_mode & stat.S_ISGID
    finally:
        os.close(descriptor)
    if mode & stat.S_ISGID and not setgid_kept:
        log.warning(
            "%s did not keep the setgid bit: this account is not in its group, so what is made below it takes this "
            "account's own group",
            directory,
        )


def make_shared_dirs(path: Path, *, mode: int = SHARED_DIR_MODE) -> Path:
    """Create each missing directory of ``path`` with ``mode`` (default ``2775``) and return ``path``.

    The way is walked as :func:`open_directory` walks it, so a link a member of the group put there is never followed,
    and nothing is made at the missing target of a link (``FileExistsError``, as ``mkdir`` refuses it). Below shared
    storage (:func:`parent_is_shared`, judged once, at the nearest directory that exists) the mode is set by ``fchmod``
    on each new directory, so the umask never decides it; the group comes from a setgid parent. Anywhere else ``mode``
    is only asked of ``mkdir``, which masks it with the umask. An existing directory is left as it is. A new shared
    directory that did not keep the setgid bit (this account is not in its group, which a setgid parent gave it) is
    logged: what is made below it takes this account's group, which the next account may not share.
    """
    os.close(open_directory(path, make=mode))
    return path


def share_new_directory(directory: Path, *, mode: int = SHARED_DIR_MODE) -> None:
    """Give a directory this process has just made (``mkdir``, ``mkdtemp``) its shared ``mode``, by ``fchmod`` on the
    directory opened without following a link, in its parent reached as :func:`open_directory` reaches it, when that
    parent is shared storage (:func:`parent_is_shared`); logged as in :func:`make_shared_dirs` when it does not keep
    setgid. Anywhere else it is left as it was made; a link at its name is refused either way."""
    with shared_parent(directory) as (parent, name, widen):
        if widen:
            _share(parent, name, mode, directory)
        else:
            os.close(os.open(name, _DIRECTORY, dir_fd=parent))  # a link at its name is refused (ELOOP)


@contextmanager
def shared_parent(path: Path) -> Iterator[tuple[int, str, bool]]:
    """The directory of ``path`` opened by :func:`open_directory` (``O_PATH``, closed after the block), the name of
    ``path`` in it, and whether that directory is shared storage (:func:`parent_is_shared`): for a writer of its own
    that opens ``path`` as ``dir_fd`` and name, so that it goes the way the helpers here go and is judged where it
    writes."""
    walk = _walk_to(Path(path).parent, make=None)
    try:
        yield walk.descriptor, Path(path).name, walk.set_up_to_share()
    finally:
        os.close(walk.descriptor)


def _final_mode(mode: int, widen: bool) -> int:
    """``mode`` below shared storage; anywhere else ``mode`` as a plain create under this process's umask leaves it."""
    return mode if widen else mode & ~current_umask()


def _new_temporary(directory: int, name: str, mode: int, flags: int = 0) -> tuple[int, str]:
    """A new file beside ``name`` in ``directory`` under a name of its own (``O_EXCL``, never through a link), with
    ``mode`` set by ``fchmod``: its descriptor, open for writing, and that name."""
    for _ in range(_TEMPORARY_TRIES):
        temporary = f".{name}.{secrets.token_hex(6)}{_TEMPORARY_SUFFIX}"
        try:
            descriptor = os.open(
                temporary, _NEW_FILE | flags, _OWNER_ONLY, dir_fd=directory
            )  # the umask never decides it
        except FileExistsError:
            continue
        try:
            os.fchmod(descriptor, mode)
        except BaseException:
            os.close(descriptor)
            with suppress(OSError):
                os.unlink(temporary, dir_fd=directory)
            raise
        return descriptor, temporary
    raise FileExistsError(errno.EEXIST, "no free name for a new file beside it", name)


def create_shared_file(path: Path) -> None:
    """Create ``path`` empty with mode ``664`` below shared storage (:func:`parent_is_shared`), else as the umask leaves
    ``664``; an existing file is replaced by rename, never truncated or chmodded."""
    with shared_parent(path) as (directory, name, widen):
        try:
            descriptor = os.open(
                name, _NEW_FILE, _OWNER_ONLY, dir_fd=directory
            )  # the umask never decides it: fchmod below
        except FileExistsError:
            descriptor = None
        if descriptor is not None:
            try:
                os.fchmod(descriptor, _final_mode(SHARED_FILE_MODE, widen))
            finally:
                os.close(descriptor)
            return
    write_shared_bytes(path, b"")


def open_shared_append(path: Path) -> int:
    """``path`` opened for appending, never through a link: made ``664`` by ``fchmod`` when this call creates it, so the
    next account of the group appends to it too, and opened as it is when it exists. One another account made without
    group write fails ``EACCES``: a file that is appended to cannot be replaced by rename."""
    try:
        descriptor = os.open(path, _APPEND | os.O_CREAT | os.O_EXCL, 0o600)  # the umask never decides it: fchmod below
    except FileExistsError:
        return os.open(path, _APPEND)
    try:
        os.fchmod(descriptor, SHARED_FILE_MODE)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def open_shared(path: Path, access: int, mode: int = SHARED_FILE_MODE) -> int:
    """Open ``path`` for ``access`` (``os.O_RDWR``, ``os.O_WRONLY | os.O_APPEND``) and return the descriptor, creating it
    when it is not there: with ``mode`` below shared storage (:func:`parent_is_shared`), else as the umask leaves it
    (a narrower ``mode`` is kept as asked, shared or not).

    It is reached through its parent as :func:`open_directory` reaches it, and never through a link. A file that exists
    is opened as it is, never truncated, chmodded or replaced by rename: a lock or a log that another process holds open
    would be left on an unlinked inode. Another account's ``0600`` file therefore raises ``PermissionError``.
    """
    with shared_parent(path) as (directory, name, widen):
        try:
            descriptor = os.open(
                name, access | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, _OWNER_ONLY, dir_fd=directory
            )  # the umask never decides it: fchmod below
        except FileExistsError:
            return os.open(name, access | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
        try:
            os.fchmod(descriptor, _final_mode(mode, widen))
        except BaseException:
            os.close(descriptor)
            raise
    return descriptor


@dataclass(frozen=True)
class Flush:
    """What :func:`write_shared_bytes` flushes to disk: the new ``file`` before the rename, its ``directory`` after."""

    file: bool = False
    directory: bool = False


def write_shared_bytes(
    path: Path,
    data: bytes,
    *,
    mode: int = SHARED_FILE_MODE,
    private_mode: int | None = None,
    flush: Flush = Flush(),
) -> None:
    """Write ``data`` to a new file beside ``path`` and rename it over ``path``: the file is then this account's,
    whoever owned the one it replaced. Below shared storage (:func:`parent_is_shared`) the file has ``mode`` whatever
    the umask; anywhere else ``private_mode`` when given, else ``mode`` masked by the umask. ``flush`` says what is
    flushed to disk; a flush that fails raises its ``OSError``."""
    with shared_parent(path) as (directory, name, widen):
        if widen or private_mode is None:
            private_mode = _final_mode(mode, widen)
        descriptor, temporary = _new_temporary(directory, name, private_mode)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                if flush.file:
                    handle.flush()
                    os.fsync(handle.fileno())
            os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        except BaseException:
            with suppress(OSError):
                os.unlink(temporary, dir_fd=directory)
            raise
        if flush.directory:
            flushed = read_directory(directory)
            try:
                os.fsync(flushed)
            finally:
                os.close(flushed)


def open_shared_replacement(path: Path, *, mode: int = SHARED_FILE_MODE) -> int:
    """A new, empty file with ``mode``, already renamed over ``path``: a descriptor open for writing it.

    For a file a process keeps writing while it runs (a log): the file is made beside ``path`` under a new name of its
    own (``O_EXCL``, so never through a link there) and put in place before the first byte, so ``path`` names this
    process's file from the start. What was at ``path`` (another leg's file, a link a member of the group put there)
    is replaced, never opened. ``mode`` is masked by the umask outside shared storage (:func:`parent_is_shared`). The
    caller closes the descriptor.
    """
    with shared_parent(path) as (directory, name, widen):
        descriptor, temporary = _new_temporary(directory, name, _final_mode(mode, widen))
        try:
            os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        except BaseException:
            os.close(descriptor)
            with suppress(OSError):
                os.unlink(temporary, dir_fd=directory)
            raise
    return descriptor


def create_new_shared_file(path: Path, *, mode: int = SHARED_FILE_MODE) -> int:
    """Create ``path`` new with ``mode``: a descriptor open for appending to it.

    The file is made under a name of its own and given its mode first, then linked to ``path`` (which fails for any
    name there already, a link included), so ``path`` never names it with another mode, even when the process is
    killed meanwhile. Where the file system makes no links it is made at ``path`` itself (``O_EXCL``, never through a
    link). ``mode`` is masked by the umask outside shared storage (:func:`parent_is_shared`).

    Raises:
        FileExistsError: When anything, a link included, has that name already; it is left as it is.
    """
    with shared_parent(path) as (directory, name, widen):
        mode = _final_mode(mode, widen)
        descriptor, temporary = _new_temporary(directory, name, mode, os.O_APPEND)
        try:
            os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
        except OSError as exc:
            os.close(descriptor)
            with suppress(OSError):
                os.unlink(temporary, dir_fd=directory)
            if exc.errno not in _NO_LINKS:
                raise
            return _create_in_place(directory, name, mode)
        except BaseException:
            os.close(descriptor)
            with suppress(OSError):
                os.unlink(temporary, dir_fd=directory)
            raise
        with suppress(OSError):
            os.unlink(temporary, dir_fd=directory)
    return descriptor


def _create_in_place(directory: int, name: str, mode: int) -> int:
    descriptor = os.open(name, _NEW_FILE | os.O_APPEND, _OWNER_ONLY, dir_fd=directory)
    try:
        os.fchmod(descriptor, mode)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def copy_shared_file(source: Path, path: Path, *, mode: int = SHARED_FILE_MODE) -> None:
    """Copy ``source`` to a new file beside ``path`` with ``mode`` (masked by the umask outside shared storage) and
    rename it over ``path``, as :func:`write_shared_bytes` does, without holding the content in memory."""
    with shared_parent(path) as (directory, name, widen):
        descriptor, temporary = _new_temporary(directory, name, _final_mode(mode, widen))
        try:
            with os.fdopen(descriptor, "wb") as handle, open(source, "rb") as content:
                shutil.copyfileobj(content, handle)
            os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        except BaseException:
            with suppress(OSError):
                os.unlink(temporary, dir_fd=directory)
            raise


def write_shared_text(path: Path, text: str, *, mode: int = SHARED_FILE_MODE) -> None:
    """:func:`write_shared_bytes` of ``text`` as UTF-8."""
    write_shared_bytes(path, text.encode("utf-8"), mode=mode)


def current_umask() -> int:
    """This process's umask, read from ``/proc/self/status``: setting it to learn it would race the process's threads.

    Raises:
        OSError: When the kernel does not report it (``ENOTSUP``).
    """
    with open("/proc/self/status", encoding=_PROC_ENCODING) as status:
        for line in status:
            if line.startswith("Umask:"):
                return int(line.split()[1], 8)
    raise OSError(errno.ENOTSUP, "the kernel reports no umask in /proc/self/status")


def writable_problems(dirs: Mapping[str, Path]) -> list[str]:
    """One line for each named directory this account cannot create and remove a file in:
    ``"<name> <path>: <errno name> (owner <uid>:<gid> <mode>)"``.

    A real write, not ``os.access``: on NFS and in a user namespace neither ``access()`` nor a comparison of ids says
    what a write will do.
    """
    problems = []
    for name, directory in dirs.items():
        try:
            descriptor, probe = tempfile.mkstemp(prefix=_PROBE_PREFIX, dir=directory)
            os.close(descriptor)
            os.unlink(probe)
        except OSError as exc:
            problems.append(f"{name} {directory}: {_errno_name(exc)}{ownership(directory)}")
    return problems


def setgid_warnings(dirs: Mapping[str, Path]) -> list[str]:
    """One line for each named directory without the setgid bit: what an account creates there takes that account's own
    group, and the helpers of this module keep it as private as the umask leaves it, so the next account of the run may
    not be able to write it."""
    warnings = []
    for name, directory in dirs.items():
        try:
            mode = os.stat(directory).st_mode
        except OSError:
            continue  # :func:`writable_problems` names a directory that cannot be read
        if not mode & stat.S_ISGID:
            warnings.append(
                f"{name} {directory} is not setgid: what is made in it takes the group of the account that makes it and is "
                f"not opened to the group, so the next account of the run may not be able to write it "
                f"(chmod g+s {directory})"
            )
    return warnings


def _errno_name(exc: OSError) -> str:
    return errno.errorcode.get(exc.errno, str(exc.errno)) if exc.errno is not None else type(exc).__name__


def ownership(path: Path) -> str:
    """``" (owner <uid>:<gid> <mode>)"`` of ``path`` for a message, or ``""`` when it cannot be read."""
    try:
        info = os.stat(path)
    except OSError:
        return ""
    return f" (owner {info.st_uid}:{info.st_gid} {stat.S_IMODE(info.st_mode):04o})"


__all__ = [
    "SHARED_DIR_MODE",
    "SHARED_FILE_MODE",
    "Flush",
    "copy_shared_file",
    "create_new_shared_file",
    "create_shared_file",
    "current_umask",
    "make_shared_dirs",
    "open_directory",
    "open_shared",
    "open_shared_append",
    "read_directory",
    "open_shared_replacement",
    "ownership",
    "parent_is_shared",
    "share_new_directory",
    "shared_parent",
    "setgid_warnings",
    "writable_problems",
    "write_shared_bytes",
    "write_shared_text",
]
