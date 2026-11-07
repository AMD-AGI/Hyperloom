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
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from hyperloom_common.fs_utils import read_text_nofollow
from hyperloom_common.shared_storage import SHARED_CHILD_UMASK


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


def _shared_group(common: Path | None) -> int | None:
    """The group whose accounts write the repository at ``common`` in turn, when its git directory is a shared one
    (setgid, as ``hyperloom_common.shared_storage`` makes it): each leg of a run may be another account of that group,
    and what the earlier one made is that group's. ``None`` when ``common`` is not given or not shared."""
    if common is None:
        return None
    info = os.lstat(common)
    return info.st_gid if stat.S_ISDIR(info.st_mode) and info.st_mode & stat.S_ISGID else None


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

    The app is its uid, and, when ``common`` is a shared git directory (setgid), the group that directory is in too:
    the run's legs, accounts of that group, make the repository and its worktrees in turn. No one else writes there,
    so the group owning an entry under it proves the same as the uid does.

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
    group = _shared_group(common)
    _app_owned_dir(admin, group)
    _app_owned_dir(admin.parent, group)
    _app_owned_dir(admin.parent.parent, group)
    back = admin / "gitdir"
    if admin.parent.name != "worktrees" or not _app_owned(back.lstat(), group):
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
    common: Path | None = None,
    admin: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``git <args>`` in ``worktree`` against its trusted admin dir, never against what the worktree holds.

    The admin dir is ``admin``, as the caller found it from the repository's side (:func:`recorded_admin_dir`), when
    given: the worktree's ``.git`` is then not read at all. Otherwise it is the one that ``.git`` names, believed only
    when the app owns it (:func:`trusted_admin_dir`), which proves nothing against an agent that runs as the app's uid.

    No program the repository's config names runs: the ``NO_PROGRAMS`` pins, every configured driver program blanked,
    and no transport. The driver programs are read from the config just before the command, which reads it again; the
    config lies in the admin and common directories, the app's (on shared storage, its accounts'), so no one else
    writes a driver in between. A repository nested in the tree is the caller's to keep out (see the module docstring).
    ``env`` adds variables of the caller's (an index file, an identity) under the ones that keep this so; ``common``
    is as :func:`trusted_admin_dir`'s. In a shared repository (a setgid ``common``) git runs under the shared child
    umask, so what it makes -- objects, refs, the worktree's files -- stays the group's to write.

    Raises:
        OSError: The worktree's git metadata is not the app's, or its config cannot be neutralized (nothing was run).
    """
    admin = trusted_admin_dir(worktree, common) if admin is None else admin
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
        umask=SHARED_CHILD_UMASK if _shared_group(common) is not None else -1,
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
