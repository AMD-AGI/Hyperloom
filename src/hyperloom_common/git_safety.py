# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``safe.directory`` handling for git subprocesses, and the app's git over a tree it does not trust.

``app_git`` starts no program a repository's config names, nor one a file that config includes names. The config
keys git runs as a program were derived from git's own key list (``git help --config``) and documentation
(``Documentation/config``, git 2.53 and 2.56), and each is handled one of three ways:

* pinned on the command line, which outranks every config file (``NO_PROGRAMS``);
* blanked by name, for a key that carries a free-form driver or hook name (``_DRIVER_PROGRAMS``);
* unreachable, because the environment allows no transport (``NO_TRANSPORT``): ssh and ``core.sshCommand``, git://
  and ``core.gitProxy``, ``ext::``, remote helpers, ``remote.<name>.uploadpack``/``receivepack``, and the
  credential helpers and askpass an http fetch would start.

The rest run only in commands the app does not run (an alias, which only a subcommand git lacks reaches; difftool,
mergetool, send-email, imap-send, instaweb, help, ``archive`` with ``tar.<format>.command``, ``submodule update``
with an ``update`` command, ``add -p`` with ``interactive.diffFilter``, an explicit ``gc``/``repack``/``prune`` with
``gc.recentObjectsHook``), or are read only from the system and global files the app skips
(``uploadpack.packObjectsHook``).

A repository nested in the tree is another config, which no pin here reaches: git checks one for local changes by
running git in it. So a ``diff`` or ``status`` passes ``--ignore-submodules=dirty`` (a ``.gitmodules`` in the tree
outranks the pin), and an ``add`` names only untracked paths, since ``add`` checks every nested repository the index
already tracks and no option turns that off; ``app_git_diff`` does both.
"""

from __future__ import annotations

import errno
import os
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from hyperloom_common.fs_utils import read_text_nofollow
from hyperloom_common.shared_storage import SHARED_CHILD_UMASK, SHARED_DIR_MODE, current_umask, directory_is_shared


_GITFILE = ".git"
_DEFAULT_TIMEOUT_SEC = 120.0
#: Options that keep git from starting a program a repository's config names: command-line scope outranks every
#: config file, and git hands these on to the git it runs in a submodule.
NO_PROGRAMS = (
    "--no-pager",  # core.pager, pager.<cmd>
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.hooksPath=/dev/null",  # every hook, those in the hooks directory too
    "-c",
    "core.editor=:",  # git starts no editor named ':'
    "-c",
    "sequence.editor=:",
    "-c",
    "diff.external=",  # cannot start: a diff without --no-ext-diff fails rather than run one
    "-c",
    "core.attributesFile=/dev/null",
    "-c",
    "core.alternateRefsCommand=false",
    # Signing and verification. gpg.program and gpg.openpgp.program set the same program, so one pin holds both.
    "-c",
    "gpg.program=false",
    "-c",
    "gpg.ssh.program=false",
    "-c",
    "gpg.x509.program=false",
    "-c",
    "gpg.ssh.defaultKeyCommand=false",
    "-c",
    "log.showSignature=false",
    "-c",
    "commit.gpgSign=false",
    "-c",
    "tag.gpgSign=false",
    "-c",
    "tag.forceSignAnnotated=false",
    "-c",
    "merge.verifySignatures=false",
    "-c",
    "maintenance.auto=false",  # the gc after a commit, merge, rebase or am runs gc.recentObjectsHook
    "-c",
    "help.autoCorrect=0",  # a mistyped subcommand never becomes the alias it resembles
    # A submodule is another repository with its own drivers: git runs git in it, under its config, not this one.
    "-c",
    "submodule.recurse=false",
    "-c",
    "diff.submodule=short",
    "-c",
    "diff.ignoreSubmodules=dirty",
)
#: The environment that allows git no transport, so a fetch (a promisor remote's lazy one too) starts nothing.
NO_TRANSPORT = MappingProxyType({"GIT_ALLOW_PROTOCOL": ""})
# A driver has a free-form name that any in-tree .gitattributes can apply to any path, and a hook defined in config
# (git 2.54 on) a free-form name that core.hooksPath does not reach, so no fixed pin covers either: each such key the
# repository's config defines is blanked by name. Git runs no filter whose program is empty; for an empty diff, merge
# or trailer program it starts nothing and that step fails instead; a hook is disabled.
_DRIVER_PROGRAMS = (
    r"^(filter\..+\.(clean|smudge|process)|diff\..+\.(command|textconv)|merge\..+\.driver|trailer\..+\.(command|cmd)"
    r"|hook\..+\.command)$"
)
_NO_SUCH_KEY = 1  # git config's exit status when no key matches
_ADD_BATCH = 256  # paths per add: at PATH_MAX each, 1 MiB, inside the kernel's 2 MiB for arguments


#: What the next account of the group writes when it commits in a repository another account made: the objects and
#: their fan-out directories, the refs (``reftable`` too, should git make one), the reflogs, which git appends to in
#: place, and the linked worktrees' admin directories. Everything else in the git directory is its maker's alone.
_WORKTREES = "worktrees"  # a repository's linked worktrees' admin directories
_SHARED_STATE = ("objects", "refs", "reftable", "logs", _WORKTREES)
#: What git reads from a git directory that could start or select a program: the config, the hooks and ``info``
#: (``attributes``, ``exclude``). In a shared repository only the account that made the repository writes them.
_CONFIG = "config"
_PROGRAM_STATE = ("hooks", "info")
#: Names git reads from a git directory that the app's repositories never hold there: a ``commondir`` would hand git
#: another repository's config, and a ``config.worktree`` another config.
_NEVER_IN_COMMON = ("commondir", "config.worktree")
_OPEN_TO_OTHERS = stat.S_IWGRP | stat.S_IWOTH
_DIR_NOFOLLOW = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def git_dir_umask() -> int:
    """The umask a git directory the app makes is made under: this process's with group and other write masked too,
    so nothing git makes there is ever writable by another account, and nothing is wider than the umask allows.
    :func:`share_objects_and_refs` then opens what the group writes."""
    return current_umask() | _OPEN_TO_OTHERS


def shared_group(common: Path) -> int | None:
    """The group whose accounts take turns writing the repository whose git directory is ``common``: its ``objects``
    and ``refs`` are storage set up to be shared (setgid, group-writable, not writable by others,
    :func:`~hyperloom_common.shared_storage.directory_is_shared`), as :func:`share_objects_and_refs` lays them out.
    ``None`` for a repository of one account. Every directory is opened without following a link at its name; a link
    at ``common`` is refused (``ENOTDIR``).

    Whether ``common`` itself is group-writable decides nothing here: in a shared repository it may be only with the
    sticky bit, which :func:`check_shared_repository` enforces."""
    top = os.open(common, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        groups = set()
        for name in ("objects", "refs"):
            try:
                held = os.open(name, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=top)
            except (FileNotFoundError, NotADirectoryError):
                return None
            try:
                if not directory_is_shared(held):
                    return None
                groups.add(os.fstat(held).st_gid)
            finally:
                os.close(held)
        return groups.pop() if len(groups) == 1 else None
    finally:
        os.close(top)


def share_objects_and_refs(git_dir: Path) -> None:
    """Lay out the repository whose git directory is ``git_dir``, just made by this account below shared storage, for
    the run's next leg, which may be another account of the group: only what that account's commit writes is the
    group's (:data:`_SHARED_STATE`: directories ``2775``, reflogs ``664``), and everything else, the config, ``HEAD``,
    ``info`` and ``hooks`` among it, is writable by this account alone. ``git_dir`` itself becomes ``3775``: git makes
    ``HEAD.lock`` when it moves the branch ``HEAD`` names and ``packed-refs.lock`` when it deletes a ref, both in the
    git directory, and the sticky bit keeps the group from renaming or removing what this account made there. An
    empty ``info`` and ``hooks`` are made where there are none, so no other account makes one. Walks without
    following a link.

    Called once, by the account that made the repository, after it is made and before another account uses it; the
    config git made with ``--shared=group`` has git keep what it makes later the group's.

    Raises:
        OSError: ``git_dir`` is a link or is not this account's.
    """
    top = os.open(git_dir, _DIR_NOFOLLOW)
    try:
        if os.fstat(top).st_uid != os.geteuid():
            raise OSError(errno.EPERM, "only the account that made a repository lays it out", str(git_dir))
        os.fchmod(top, 0o3775)
        for name in _PROGRAM_STATE:
            if not _present(top, name):
                os.mkdir(name, 0o755, dir_fd=top)
        for name in os.listdir(top):
            info = os.stat(name, dir_fd=top, follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                continue
            if name in _SHARED_STATE and stat.S_ISDIR(info.st_mode):
                _walk_modes(top, name, directories=_shared_dir, files=_group_appends if name == "logs" else None)
            else:
                _walk_modes(top, name, directories=_closed, files=_closed)
    finally:
        os.close(top)


def _shared_dir(_mode: int) -> int:
    return SHARED_DIR_MODE


def _group_appends(mode: int) -> int:
    return (mode | stat.S_IWGRP) & ~stat.S_IWOTH


def _closed(mode: int) -> int:
    return mode & ~_OPEN_TO_OTHERS


def _present(directory: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _walk_modes(
    directory: int,
    name: str,
    *,
    directories: Callable[[int], int],
    files: Callable[[int], int] | None,
) -> None:
    """Give the entry ``name`` of ``directory`` and, for a directory, everything below it the mode ``directories``
    or ``files`` makes of its own (``None``: files keep theirs); links are left as they are."""
    info = os.stat(name, dir_fd=directory, follow_symlinks=False)
    if stat.S_ISREG(info.st_mode) and files is not None:
        _set_mode(directory, name, files(stat.S_IMODE(info.st_mode)), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISDIR(info.st_mode):
        return
    held = os.open(name, _DIR_NOFOLLOW, dir_fd=directory)
    try:
        wanted = directories(stat.S_IMODE(info.st_mode))
        if stat.S_IMODE(os.fstat(held).st_mode) != wanted:
            os.fchmod(held, wanted)
        for entry in os.listdir(held):
            _walk_modes(held, entry, directories=directories, files=files)
    finally:
        os.close(held)


def _set_mode(directory: int, name: str, mode: int, flags: int) -> None:
    held = os.open(name, flags | os.O_CLOEXEC, dir_fd=directory)
    try:
        if stat.S_IMODE(os.fstat(held).st_mode) != mode:
            os.fchmod(held, mode)
    finally:
        os.close(held)


def check_shared_repository(common: Path) -> int | None:
    """Refuse, before git runs, a shared repository (:func:`shared_group`) whose git directory ``common`` holds state
    another account could have written; the group, or ``None`` for a repository of one account, which is not checked.

    The account that made the repository is the owner of its top directory (``common`` for a bare repository, the
    directory holding ``.git`` otherwise), which no account of the group can change. Refused:

    * ``common`` or the top directory that is a link, writable by others, or writable by the group without the sticky
      bit, or that the maker does not own;
    * the directory holding the top directory (the session directory) when the group can rename entries in it (it is
      group- or other-writable without the sticky bit) and the maker does not own it: another account could have put a
      repository of its own in place of the maker's, and every entry of that one would carry the group;
    * a ``config`` that is no regular file, is a link, is writable by the group or others, or is not the maker's;
    * ``hooks`` or ``info``, or an entry of either, that is a link, is writable by the group or others, or is not the
      maker's;
    * a ``commondir`` or ``config.worktree`` in ``common``, which the app's repositories never hold there.

    The config stays readable to the group: the next account's git has to read it.

    Raises:
        OSError: ``EPERM`` naming what was refused.
    """
    group = shared_group(common)
    if group is None:
        return None
    top = common.parent if common.name == _GITFILE else common
    maker = _refuse_unless_kept(top.lstat(), top, owner=None)
    if top != common:
        _refuse_unless_kept(common.lstat(), common, owner=maker)
    held = top.parent.lstat()
    if held.st_mode & _OPEN_TO_OTHERS and not held.st_mode & stat.S_ISVTX and held.st_uid != maker:
        raise _refused(top.parent, "the group can put another repository in place of this one, which it did not make")
    config = (common / _CONFIG).lstat()
    if not stat.S_ISREG(config.st_mode):
        raise _refused(common / _CONFIG, "the config is no file")
    _refuse_unless_kept(config, common / _CONFIG, owner=maker, sticky=False)
    for name in _PROGRAM_STATE:
        for path in _below(common / name):
            _refuse_unless_kept(path.lstat(), path, owner=maker, sticky=False)
    for name in _NEVER_IN_COMMON:
        if (common / name).is_symlink() or (common / name).exists():
            raise _refused(common / name, "the app's repositories never hold this in their git directory")
    return group


def _below(path: Path) -> list[Path]:
    """``path`` and everything below it, without following a link; ``[]`` when it is not there."""
    if not path.is_symlink() and not path.exists():
        return []
    found = [path]
    if path.is_symlink() or not path.is_dir():
        return found
    for here, dirs, files in os.walk(path, followlinks=False):
        found += [Path(here) / name for name in (*dirs, *files)]
    return found


def _refuse_unless_kept(info: os.stat_result, path: Path, *, owner: int | None, sticky: bool = True) -> int:
    """Refuse ``path`` (as ``info`` shows it) when it is a link, writable by others, or writable by the group without
    the sticky bit (never, with ``sticky=False``), or not ``owner``'s when one is given; its owner."""
    mode = info.st_mode
    if stat.S_ISLNK(mode):
        raise _refused(path, "a link")
    if mode & stat.S_IWOTH or (mode & stat.S_IWGRP and not (sticky and mode & stat.S_ISVTX)):
        raise _refused(path, f"writable by others than its owner (mode {stat.S_IMODE(mode):04o})")
    if owner is not None and info.st_uid != owner:
        raise _refused(path, f"owned by uid {info.st_uid}, not by uid {owner}, who made the repository")
    return info.st_uid


def _refused(path: Path, why: str) -> OSError:
    return OSError(errno.EPERM, f"git's own state of a shared repository is refused: {why}", str(path))


def _names_its_repository(admin: Path, common: Path) -> None:
    """Refuse the admin directory ``admin`` unless its ``commondir`` names ``common``: git takes the config and the
    objects from ``GIT_COMMON_DIR``, but its ref store still follows the file, which the group writes in a shared
    repository."""
    named = read_text_nofollow(admin / "commondir").strip()
    if os.path.realpath(admin / named) != os.path.realpath(common):
        raise OSError(errno.EPERM, f"the admin dir's commondir names another repository than {common}", str(admin))


def _app_owned(info: os.stat_result, group: int | None) -> bool:
    """Whether the entry ``info`` is the app's: its uid's, or, in a shared repository, its accounts' ``group``'s."""
    return info.st_uid == os.geteuid() or (group is not None and info.st_gid == group)


def _app_owned_dir(path: Path, group: int | None = None) -> None:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or not _app_owned(info, group):
        raise OSError(errno.EPERM, "not a directory owned by the app", str(path))


def trusted_admin_dir(worktree: Path, common: Path | None = None) -> Path:
    """The admin directory of the linked worktree ``worktree``, proven to be the app's.

    The path the gitfile names is resolved before anything is checked, and the resolved path is what is returned and
    what git is given, so no symlink is followed after the check. Ownership alone does not prove it when whoever writes
    the worktree writes files the app's uid owns (a sandbox whose user namespace maps its user to the app's): it can
    lay out an admin directory of its own beside a gitfile naming it. One in the worktree is refused; for one it lays
    out anywhere else it writes, a caller that knows the repository it made the worktree from passes its git directory
    as ``common``, and the admin directory must then be one of that repository's.

    The app is its uid, and, when ``common`` is the git directory of a shared repository (:func:`shared_group`: its
    objects and refs are set up to be shared), the group they are in too: the run's legs, accounts of that group, add
    the worktrees in turn, and ``worktrees`` is the group's to write. No one else writes there, so the group owning an
    entry under it proves the same as the uid does. What git reads besides from such a repository, its config, hooks
    and ``info``, is its maker's alone, which :func:`app_git` checks (:func:`check_shared_repository`).

    Raises:
        OSError: ``.git`` is not a gitfile, or the directory it names is not an
            app-owned admin directory of a repository whose ``worktrees`` entry
            points back at this gitfile, or of ``common`` when given.
    """
    gitfile = worktree / _GITFILE
    first = read_text_nofollow(gitfile).strip()
    if not first.startswith("gitdir:"):
        raise OSError(errno.EINVAL, "not a gitdir pointer", str(gitfile))
    named = Path(first.removeprefix("gitdir:").strip())
    # Resolved once, here, and only the resolved path is checked and handed to git: a symlink on the way (one the
    # worktree's writer owns) could be pointed elsewhere between this check and each git run that resolves it again.
    admin = Path(os.path.realpath(named if named.is_absolute() else worktree / named))
    if admin.is_relative_to(os.path.realpath(worktree)):
        raise OSError(errno.EPERM, "the admin dir lies in the worktree, which its writer writes", str(admin))
    group = None if common is None else shared_group(common)
    _app_owned_dir(admin, group)
    _app_owned_dir(admin.parent, group)
    _app_owned_dir(admin.parent.parent, group)
    back = admin / "gitdir"
    if admin.parent.name != _WORKTREES or not _app_owned(back.lstat(), group):
        raise OSError(errno.EPERM, "not an admin dir of a linked worktree", str(admin))
    if os.path.realpath(read_text_nofollow(back).strip()) != os.path.realpath(gitfile):
        raise OSError(errno.EPERM, "the admin dir does not belong to this worktree", str(admin))
    if common is not None and os.path.realpath(admin.parent.parent) != os.path.realpath(common):
        raise OSError(errno.EPERM, f"the admin dir is not one of the repository {common}", str(admin))
    return admin


def recorded_admin_dir(repo: Path, worktree: Path, *, timeout: float = _DEFAULT_TIMEOUT_SEC) -> Path:
    """The admin directory of the linked worktree ``worktree``, as the repository ``repo`` it was added from records it.

    Only ``repo``'s git directory is read, never what the worktree holds: its ``.git`` file is the agent's to rewrite,
    and an agent that runs as the app's own uid (in a bubblewrap sandbox) makes an admin directory no ownership check
    tells from the app's (:func:`trusted_admin_dir`). The one entry of the repository's ``worktrees`` whose ``gitdir``
    names the worktree's ``.git`` is it.

    Raises:
        OSError: ``repo`` has no git directory, or not exactly one of its linked worktrees is ``worktree``.
        subprocess.SubprocessError: git did not answer within ``timeout``.
    """
    done = subprocess.run(
        ["git", *NO_PROGRAMS, "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if done.returncode != 0:
        raise OSError(errno.ENOENT, f"no git directory: {done.stderr.strip()}", str(repo))
    entries = Path(done.stdout.strip()) / _WORKTREES
    gitfile = os.path.realpath(worktree / _GITFILE)
    found = [entry for entry in _linked_entries(entries) if _names(entry, gitfile)]
    if len(found) != 1:
        raise OSError(errno.ENOENT, f"{len(found)} linked worktrees of {repo} are this one", str(worktree))
    return found[0]


def _linked_entries(entries: Path) -> list[Path]:
    """The admin directories under a repository's ``worktrees`` directory."""
    try:
        return sorted(entry for entry in entries.iterdir() if entry.is_dir() and not entry.is_symlink())
    except FileNotFoundError:
        return []


def _names(entry: Path, gitfile: str) -> bool:
    """Whether the admin directory ``entry`` belongs to the worktree whose resolved ``.git`` is ``gitfile``."""
    try:
        named = read_text_nofollow(entry / "gitdir").strip()
    except (OSError, UnicodeDecodeError):
        return False
    return os.path.realpath(os.path.join(entry, named)) == gitfile


def worktree_head(worktree: Path | None, *, admin: Path | None = None) -> str:
    """The commit a linked worktree sits at, read through ``admin`` (:func:`app_git`) or else its trusted admin dir;
    ``""`` when there is none to trust."""
    if worktree is None or (admin is None and not (worktree / _GITFILE).exists()):
        return ""
    try:
        done = app_git(worktree, "rev-parse", "HEAD", admin=admin)
    except OSError:
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def blank_driver_programs(git: list[str], worktree: Path, env: dict[str, str], timeout: float) -> list[str]:
    """The ``-c <key>=`` pairs that blank every driver program the config ``git`` reads defines, includes followed.

    Raises:
        OSError: The config could not be read, or names a driver ``-c`` cannot address (its name holds ``=``).
    """
    done = subprocess.run(
        [*git, "config", "--null", "--name-only", "--get-regexp", _DRIVER_PROGRAMS],
        cwd=worktree,
        env=env,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if done.returncode not in (0, _NO_SUCH_KEY):
        raise OSError(errno.EIO, f"git config: {os.fsdecode(done.stderr).strip()}", str(worktree))
    keys = sorted({os.fsdecode(key) for key in done.stdout.split(b"\0") if key})
    if any("=" in key for key in keys):
        raise OSError(errno.EINVAL, "a driver name holds '=', which git -c cannot override", str(worktree))
    return [arg for key in keys for arg in ("-c", _blanked(key))]


def _blanked(key: str) -> str:
    """The ``-c`` setting that stops the program ``key`` names: a hook is disabled, any other program emptied."""
    return f"{key.rpartition('.')[0]}.enabled=false" if key.startswith("hook.") else f"{key}="


def app_git(
    worktree: Path,
    *args: str,
    timeout: float = _DEFAULT_TIMEOUT_SEC,
    env: Mapping[str, str] | None = None,
    admin: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``git <args>`` in ``worktree`` against its trusted admin dir, never against what the worktree holds.

    The admin dir is ``admin``, as the caller found it from the repository's side when given: recorded there
    (:func:`recorded_admin_dir`), and the worktree's ``.git`` then not read at all, or the one ``.git`` names proven one
    of the repository's (:func:`trusted_admin_dir` with its ``common``). Otherwise it is the one that ``.git`` names,
    believed only when the app owns it (:func:`trusted_admin_dir`), which proves nothing against an agent that runs as
    the app's uid.

    No program the repository's config names runs: the ``NO_PROGRAMS`` pins, every configured driver program blanked,
    and no transport. The driver programs are read from the config just before the command, which reads it again; the
    config lies in the repository's git directory, ``admin``'s grandparent, which git is pinned to
    (``GIT_COMMON_DIR``), so a ``commondir`` the admin directory holds names no other. That config is the app's uid's,
    or, in a repository several accounts of one group take turns writing (:func:`shared_group`), the account's that
    made the repository, and writable by no one else: :func:`check_shared_repository` refuses one that is not, before
    git runs, whichever way ``admin`` was found. So no one else writes a driver in between. A repository nested in the
    tree is the caller's to keep out (see the module docstring). ``env`` adds variables of the caller's (an index
    file, an identity) under the ones that keep this so. In a shared repository git runs under the shared child umask,
    so what it makes -- objects, refs, the worktree's files -- stays the group's to write.

    Raises:
        OSError: The worktree's git metadata is not the app's, its shared repository holds state another account could
            have written, or its config cannot be neutralized (nothing was run).
    """
    admin = trusted_admin_dir(worktree) if admin is None else admin
    common = admin.parent.parent
    group = check_shared_repository(common)
    _names_its_repository(admin, common)
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("GIT_")}, **(env or {})}
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_ATTR_NOSYSTEM="1",
        GIT_TERMINAL_PROMPT="0",
        GIT_COMMON_DIR=str(common),
        **NO_TRANSPORT,
    )
    git = [
        "git",
        "--git-dir",
        str(admin),
        "--work-tree",
        str(worktree),
        "-c",
        f"safe.directory={worktree}",
        "-c",
        f"safe.directory={admin}",
        *NO_PROGRAMS,
    ]
    argv = [*git, *blank_driver_programs(git, worktree, env, timeout), *args]
    # Decoded as file names are, so a path git prints goes back to git unchanged.
    return subprocess.run(
        argv,
        cwd=worktree,
        env=env,
        umask=-1 if group is None else SHARED_CHILD_UMASK,
        capture_output=True,
        encoding="utf-8",
        errors="surrogateescape",
        timeout=timeout,
        check=False,
    )


def app_git_diff(
    worktree: Path, base: str, pathspec: Sequence[str], *, admin: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """``git diff <base>`` over ``pathspec`` in ``worktree`` by ``app_git`` (through ``admin`` when given), untracked
    paths shown as created.

    The untracked paths are marked intent-to-add by name, never by ``add -A``: an ``add`` over a path the index
    tracks checks a repository nested there for local changes by running git in it, under that repository's own
    config, and no option turns that off. Marking is best effort, as it always was. The diff runs no external diff
    or textconv, and checks no nested repository: ``--ignore-submodules=dirty`` outranks a ``.gitmodules`` in the tree.

    Raises:
        OSError: As ``app_git``.
        subprocess.SubprocessError: As ``app_git``.
    """
    listed = app_git(worktree, "ls-files", "--others", "--exclude-standard", "-z", "--", *pathspec, admin=admin)
    untracked = [path for path in listed.stdout.split("\0") if path] if listed.returncode == 0 else []
    for start in range(0, len(untracked), _ADD_BATCH):
        app_git(worktree, "--literal-pathspecs", "add", "-N", "--", *untracked[start : start + _ADD_BATCH], admin=admin)
    return app_git(
        worktree,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--ignore-submodules=dirty",
        base,
        "--",
        *pathspec,
        admin=admin,
    )


__all__ = [
    "NO_PROGRAMS",
    "NO_TRANSPORT",
    "app_git",
    "app_git_diff",
    "blank_driver_programs",
    "check_shared_repository",
    "git_dir_umask",
    "recorded_admin_dir",
    "repo_root",
    "safe_directory_args",
    "share_objects_and_refs",
    "shared_group",
    "trusted_admin_dir",
    "worktree_head",
]


def repo_root(target: str | Path) -> str | None:
    """Nearest ancestor of ``target`` holding a ``.git`` entry, else None."""
    try:
        current = Path(target).expanduser().resolve()
    except OSError:
        return None
    for candidate in (current, *current.parents):
        try:
            if (candidate / ".git").exists():
                return str(candidate)
        except OSError:
            continue
    return None


def safe_directory_args(args: list[str], *, cwd: str | Path | None = None) -> list[str]:
    """Prepend a ``safe.directory`` exception for the repo ``args`` targets."""
    target: str | Path | None = cwd
    try:
        target = args[args.index("-C") + 1]
    except (ValueError, IndexError):
        pass
    if target is None:
        return args
    root = repo_root(target)
    if root is None:
        return args
    return ["-c", f"safe.directory={root}", *args]
