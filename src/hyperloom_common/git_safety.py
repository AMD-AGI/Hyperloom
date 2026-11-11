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
import logging
import os
import secrets
import stat
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from hyperloom_common.fs_utils import read_text_nofollow
from hyperloom_common.shared_storage import child_umask, directory_is_shared, parent_is_shared

logger = logging.getLogger(__name__)


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


_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_GROUP_OR_OTHER_WRITE = stat.S_IWGRP | stat.S_IWOTH
#: What git reads from a repository's own git directory besides objects and refs: its config, and the hooks and the
#: ``info`` directory (``attributes``, ``exclude``). Below shared storage only the app's uid writes them.
_CONFIG = "config"
_COMMON_STATE = ("hooks", "info")
#: What git reads from a linked worktree's admin directory to find and configure its repository.
_GITDIR = "gitdir"  # an admin directory's pointer back at its worktree's gitfile
_ADMIN_FILES = (_GITDIR, "commondir", "config.worktree")
_OWNER_FILE, _OWNER_DIR = 0o600, 0o700


def below_shared_storage(git_dir: Path) -> bool:
    """Whether the git directory ``git_dir`` is storage set up to be shared (setgid, group-writable, closed to others;
    :func:`~hyperloom_common.shared_storage.directory_is_shared`) or lies in such a directory
    (:func:`~hyperloom_common.shared_storage.parent_is_shared`): there each leg of a run may be another account of the
    group, and git's own state is kept the app's uid's alone. A link at ``git_dir``'s own name is refused
    (``ENOTDIR``)."""
    descriptor = os.open(git_dir, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        if directory_is_shared(descriptor):
            return True
    finally:
        os.close(descriptor)
    return parent_is_shared(git_dir)


def _app_owned_dir(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise OSError(errno.EPERM, "not a directory owned by the app", str(path))


def _owner_only(info: os.stat_result, path: Path) -> None:
    """Refuse ``path`` (as ``info`` shows it) unless the app's uid owns it and no one else can write it."""
    if info.st_uid != os.geteuid() or info.st_mode & _GROUP_OR_OTHER_WRITE or stat.S_ISLNK(info.st_mode):
        raise OSError(
            errno.EPERM,
            f"below shared storage only the app's uid writes git's own state (owner {info.st_uid},"
            f" mode {stat.S_IMODE(info.st_mode):04o})",
            str(path),
        )


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _check_admin_state(admin: Path) -> None:
    """Below shared storage, refuse a linked worktree's admin directory ``admin`` unless git's own state it and its
    repository hold is the app's uid's alone: the admin directory and its ``gitdir``, ``commondir`` and
    ``config.worktree``, the repository's ``config``, and its ``hooks`` and ``info`` with what they hold. Another
    account of the group is trusted for none of them, only for objects and refs. Elsewhere nothing is checked here.

    Raises:
        OSError: One of them is another account's, or writable by the group or others.
    """
    common = admin.parent.parent
    if not below_shared_storage(common):
        return
    _owner_only(admin.lstat(), admin)
    for name in _ADMIN_FILES:
        found = _lstat_or_none(admin / name)
        if found is not None:
            _owner_only(found, admin / name)
    config = common / _CONFIG
    found = config.lstat()
    if not stat.S_ISREG(found.st_mode):
        raise OSError(errno.EPERM, "the repository's config is no file", str(config))
    _owner_only(found, config)
    for name in _COMMON_STATE:
        top = common / name
        if _lstat_or_none(top) is None:
            continue
        for path in (top, *top.rglob("*")):
            _owner_only(path.lstat(), path)


def own_git_admin(git_dir: Path, *, bare: bool) -> None:
    """Below shared storage (:func:`below_shared_storage`), make git's own state in the git directory ``git_dir`` the
    app's uid's alone, whatever the umask: ``config`` ``0600``, ``hooks`` and ``info`` ``0700`` with their files
    ``0600``. Objects and refs keep the modes git gave them. Elsewhere nothing changes.

    A config the app's uid does not own, or one the group or others can write, is never read: an earlier leg of
    another account of the group made it, or anyone of the group could have rewritten it. A fresh one replaces it,
    holding only the repository format git's own ``init`` records (``sharedRepository`` when ``git_dir`` itself is
    shared storage, as ``git init --shared=group`` records it), so this account runs git on the shared objects and
    refs without trusting what the other wrote. The rename needs the group's write on ``git_dir``, which shared
    storage gives. ``hooks`` and ``info`` of another account are refused, not adopted: git never runs a hook of
    the app's (``core.hooksPath`` is pinned), and an ``info`` this account cannot read git only warns about.

    Raises:
        OSError: ``git_dir`` is a link or no directory, its ``hooks`` or ``info`` holds an entry of another account
            or a link, or the config cannot be replaced.
    """
    try:
        if not below_shared_storage(git_dir):
            return
    except FileNotFoundError:
        return  # no repository here yet: nothing to own, and none is made
    descriptor = os.open(git_dir, _DIR_FLAGS)
    try:
        _own_config(descriptor, git_dir, bare=bare)
        for name in _COMMON_STATE:
            _own_tree(descriptor, name, git_dir / name)
    finally:
        os.close(descriptor)


def own_worktree_admin(admin: Path) -> None:
    """Below shared storage, make the linked worktree admin directory ``admin`` (the one the repository records,
    :func:`recorded_admin_dir`) the app's uid's alone: the directory ``0700``, its ``gitdir``, ``commondir`` and
    ``config.worktree`` ``0600``. Elsewhere nothing changes.

    Raises:
        OSError: ``admin`` or one of those files is another account's, or a link.
    """
    if not below_shared_storage(admin.parent.parent):
        return
    parent = os.open(admin.parent, _DIR_FLAGS)
    try:
        _own_tree(parent, admin.name, admin, names=_ADMIN_FILES)
    finally:
        os.close(parent)


def _own_config(directory: int, git_dir: Path, *, bare: bool) -> None:
    try:
        held = os.open(_CONFIG, _FILE_FLAGS, dir_fd=directory)
    except FileNotFoundError:
        return  # no repository here yet: nothing to own, and none is made
    except OSError as error:
        if error.errno not in (errno.ELOOP, errno.EACCES, errno.ENXIO):
            raise
        found = f"one this account cannot open ({error.strerror})"
    else:
        try:
            info = os.fstat(held)
            found = f"owner {info.st_uid}, mode {stat.S_IMODE(info.st_mode):04o}"
            if stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and not info.st_mode & _GROUP_OR_OTHER_WRITE:
                if stat.S_IMODE(info.st_mode) != _OWNER_FILE:
                    os.fchmod(held, _OWNER_FILE)  # no one else could have written it: kept, closed to them
                return
        finally:
            os.close(held)
    logger.warning("%s: a config this account does not hold alone (%s) is replaced, unread", git_dir / _CONFIG, found)
    shared = directory_is_shared(directory)
    lines = ["[core]", "\trepositoryformatversion = 0", "\tfilemode = true", f"\tbare = {str(bare).lower()}"]
    lines += [] if bare else ["\tlogallrefupdates = true"]
    lines += ["\tsharedrepository = 1"] if shared else []
    temporary = f".{_CONFIG}.{secrets.token_hex(8)}"
    made = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, _OWNER_FILE, dir_fd=directory
    )
    try:
        try:
            os.fchmod(made, _OWNER_FILE)
            os.write(made, ("\n".join(lines) + "\n").encode("ascii"))
        finally:
            os.close(made)
        os.replace(temporary, _CONFIG, src_dir_fd=directory, dst_dir_fd=directory)
    except BaseException:
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass
        raise


def _own_tree(directory: int, name: str, path: Path, names: Sequence[str] | None = None) -> None:
    """Make the directory ``name`` under ``directory`` (at ``path``) ``0700`` and every file it holds ``0600`` (only
    ``names`` when given), walking without following a link; nothing when it is not there.

    Raises:
        OSError: An entry is another account's, or a link.
    """
    try:
        held = os.open(name, _DIR_FLAGS, dir_fd=directory)
    except FileNotFoundError:
        return
    try:
        _owner(os.fstat(held), path)
        os.fchmod(held, _OWNER_DIR)
        for entry in names if names is not None else os.listdir(held):
            try:
                info = os.lstat(entry, dir_fd=held)
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(info.st_mode):
                _own_tree(held, entry, path / entry)
                continue
            _owner(info, path / entry)
            if not stat.S_ISREG(info.st_mode):
                raise OSError(errno.EPERM, "neither a file nor a directory in git's own state", str(path / entry))
            _chmod_file(held, entry)
    finally:
        os.close(held)


def _chmod_file(directory: int, name: str) -> None:
    held = os.open(name, _FILE_FLAGS, dir_fd=directory)
    try:
        os.fchmod(held, _OWNER_FILE)
    finally:
        os.close(held)


def _owner(info: os.stat_result, path: Path) -> None:
    if info.st_uid != os.geteuid() or stat.S_ISLNK(info.st_mode):
        raise OSError(errno.EPERM, f"git's own state here is another account's (owner {info.st_uid})", str(path))


def trusted_admin_dir(worktree: Path, common: Path | None = None) -> Path:
    """The admin directory of the linked worktree ``worktree``, proven to be the app's.

    The path the gitfile names is resolved before anything is checked, and the resolved path is what is returned and
    what git is given, so no symlink is followed after the check. Ownership alone does not prove it when whoever writes
    the worktree writes files the app's uid owns (a sandbox whose user namespace maps its user to the app's): it can
    lay out an admin directory of its own beside a gitfile naming it. One in the worktree is refused; for one it lays
    out anywhere else it writes, a caller that knows the repository it made the worktree from passes its git directory
    as ``common``, and the admin directory must then be one of that repository's.

    The app is its uid alone, on shared storage too: an admin directory another account of the group made is
    refused, whatever group owns it. Below shared storage (:func:`below_shared_storage`) the admin directory, its
    ``gitdir``, ``commondir`` and ``config.worktree``, and the repository's ``config``, ``hooks`` and ``info`` must
    also be writable by no one else (:func:`own_git_admin` and :func:`own_worktree_admin` make them so).

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
    _app_owned_dir(admin)
    _app_owned_dir(admin.parent)
    _app_owned_dir(admin.parent.parent)
    back = admin / _GITDIR
    if admin.parent.name != "worktrees" or back.lstat().st_uid != os.geteuid():
        raise OSError(errno.EPERM, "not an admin dir of a linked worktree", str(admin))
    if os.path.realpath(read_text_nofollow(back).strip()) != os.path.realpath(gitfile):
        raise OSError(errno.EPERM, "the admin dir does not belong to this worktree", str(admin))
    if common is not None and os.path.realpath(admin.parent.parent) != os.path.realpath(common):
        raise OSError(errno.EPERM, f"the admin dir is not one of the repository {common}", str(admin))
    _check_admin_state(admin)
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
    entries = Path(done.stdout.strip()) / "worktrees"
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
        named = read_text_nofollow(entry / _GITDIR).strip()
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

    The admin dir is ``admin``, as the caller found it from the repository's side (:func:`recorded_admin_dir`), when
    given: the worktree's ``.git`` is then not read at all. Otherwise it is the one that ``.git`` names, believed only
    when the app owns it (:func:`trusted_admin_dir`), which proves nothing against an agent that runs as the app's uid.

    No program the repository's config names runs: the ``NO_PROGRAMS`` pins, every configured driver program blanked,
    and no transport. The driver programs are read from the config just before the command, which reads it again; the
    config lies in the admin and common directories, the app's uid's, and below shared storage git's own state there
    must be writable by no one else (:func:`_check_admin_state`, checked whichever way ``admin`` was found). The git
    directory there stays the group's to write for objects and refs, so another account could still rename a config
    in between: the pins and the blanking are what hold then. A repository nested in the tree is the caller's to keep out (see the module docstring).
    ``env`` adds variables of the caller's (an index file, an identity) under the ones that keep this so. A caller that
    holds the repository passes ``admin=trusted_admin_dir(worktree, common)`` to have the admin dir checked against it.
    In a shared repository (the admin dir's git directory set up to be shared) git runs under the shared child umask, so what it
    makes -- objects, refs, the worktree's files -- stays the group's to write.

    Raises:
        OSError: The worktree's git metadata is not the app's, or its config cannot be neutralized (nothing was run).
    """
    if admin is None:
        admin = trusted_admin_dir(worktree)
    else:
        _check_admin_state(admin)
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("GIT_")}, **(env or {})}
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_ATTR_NOSYSTEM="1",
        GIT_TERMINAL_PROMPT="0",
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
        umask=child_umask(admin.parent.parent),
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
    "below_shared_storage",
    "blank_driver_programs",
    "own_git_admin",
    "own_worktree_admin",
    "recorded_admin_dir",
    "repo_root",
    "safe_directory_args",
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
