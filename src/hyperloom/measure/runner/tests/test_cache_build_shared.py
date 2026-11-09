# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner cache on shared storage, where its builders may be different accounts of one group, and so may anything
else that group runs: a lock every one of them can open for writing, a build that runs nothing and writes through
nothing another account of the group can change, and an entry in place that only its builder can change."""

from __future__ import annotations

import errno
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from hyperloom_common.tests.shared_accounts import Accounts, other_accounts, theirs
from hyperloom.measure.runner import cache_build
from hyperloom.measure.runner.cache_build import (
    BuildOptions,
    CacheBuildError,
    CacheConflict,
    CacheEntry,
    build_lock,
    ensure_runner_cache,
)
from hyperloom.measure.runner.cache_layout import (
    BOOTSTRAP_NAME,
    MANIFEST_NAME,
    SOURCE_ARCHIVE,
    WHEELS_DIR,
    LockPin,
    check_entry,
)
from hyperloom.measure.runner.tests.bootstrap_rig import Rig, copy_cache

_SRC = str(Path(cache_build.__file__).parents[3])
_SHARED_WRITE = stat.S_IWGRP | stat.S_IWOTH
_ENTRY_READ = stat.S_IRGRP | stat.S_IXGRP


def _modes(tree: Path) -> dict[str, int]:
    """Each directory and regular file below ``tree`` (``tree`` itself as ``.``) and its permission bits."""
    found = {".": stat.S_IMODE(os.lstat(tree).st_mode)}
    for path in tree.rglob("*"):
        info = os.lstat(path)
        if stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode):
            found[path.relative_to(tree).as_posix()] = stat.S_IMODE(info.st_mode)
    return found


@pytest.mark.parametrize(
    ("umask", "lock", "cache"),
    [(0o002, 0o664, 0o2775), (0o022, 0o664, 0o2775), (0o027, 0o660, 0o2770), (0o077, 0o660, 0o2770)],
    ids=["umask-002", "umask-022", "umask-027", "umask-077"],
)
def test_the_lock_and_the_cache_let_the_group_write_and_the_entry_no_one_but_its_builder(
    rig: Rig, tmp_path: Path, umask: int, lock: int, cache: int
) -> None:
    # Storage set up to be shared: setgid, group-writable, not writable by others.
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o2775)
    root = shared / "cache"
    command = [sys.executable, "-m", "hyperloom.measure.runner.cache_build", "--sha", rig.cache.sha]
    command += ["--repo", str(rig.repo), "--out", str(root), "--no-index", "--find-links", str(rig.wheelhouse)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=300, umask=umask, check=False)
    assert done.returncode == 0, done.stderr
    assert stat.S_IMODE(build_lock(root, rig.cache.sha).stat().st_mode) == lock
    assert stat.S_IMODE(root.stat().st_mode) == cache
    # The entry is readable by others as the builder's umask lets them, by the group always (the other accounts'
    # builders check and use it), and writable by its builder alone, even under a umask that lets the group write what
    # the builder makes; so its own directory is never 0700 like a builder's unfinished copy.
    entry = _modes(root / rig.cache.sha)
    assert (entry["."], entry[BOOTSTRAP_NAME]) == (0o755 & ~umask | 0o050, 0o755)
    assert {name for name, mode in entry.items() if mode & _SHARED_WRITE} == set()
    assert {name for name, mode in entry.items() if mode & stat.S_IRGRP == 0} == set()
    tree = root / rig.cache.sha
    assert {name for name in entry if (tree / name).is_dir() and entry[name] & stat.S_IXGRP == 0} == set()


@pytest.mark.parametrize("parent", [0o755, 0o2755, 0o2777, 0o1777], ids=["0755", "2755", "2777", "1777"])
@pytest.mark.parametrize("umask", [0o022, 0o077], ids=["umask-022", "umask-077"])
def test_the_lock_and_the_cache_stay_private_where_the_storage_is_not_set_up_to_be_shared(
    rig: Rig, tmp_path: Path, parent: int, umask: int
) -> None:
    private = tmp_path / "private"
    private.mkdir()
    private.chmod(parent)
    root = private / "cache"
    command = [sys.executable, "-m", "hyperloom.measure.runner.cache_build", "--sha", rig.cache.sha]
    command += ["--repo", str(rig.repo), "--out", str(root), "--no-index", "--find-links", str(rig.wheelhouse)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=300, umask=umask, check=False)
    assert done.returncode == 0, done.stderr
    # A directory made in a setgid one inherits the bit from the kernel, not from the builder.
    inherited = parent & stat.S_ISGID
    assert stat.S_IMODE(build_lock(root, rig.cache.sha).stat().st_mode) == 0o666 & ~umask
    assert stat.S_IMODE(root.stat().st_mode) == inherited | 0o777 & ~umask


@pytest.mark.parametrize("lock", ["link", "read-only"])
def test_a_lock_that_cannot_be_opened_for_writing_is_named(rig: Rig, tmp_path: Path, lock: str) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    path = build_lock(cache, rig.cache.sha)
    if lock == "link":
        (tmp_path / "elsewhere").write_text("")
        path.symlink_to(tmp_path / "elsewhere")
    else:
        if os.geteuid() == 0:
            pytest.skip("root opens a read-only file for writing")
        path.write_text("")
        path.chmod(0o444)
    with pytest.raises(CacheBuildError, match=f"cannot open the build lock {path} for writing: "):
        ensure_runner_cache(rig.repo, rig.cache.sha, cache, rig.build_options())
    assert not (cache / rig.cache.sha).exists()


def test_the_lock_is_not_opened_through_another_accounts_link_on_its_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache directory swapped for a link after it was made: the lock goes the walk's way, so the link of another
    account below ``/tmp`` (which every account can write) is not followed to where the lock would be made."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    cache = tmp_path / "cache"
    cache.symlink_to(elsewhere)
    theirs(cache, monkeypatch)
    path = build_lock(cache, "0" * 40)
    with pytest.raises(CacheBuildError, match=f"cannot open the build lock {path} for writing: "):
        os.close(cache_build._open_lock(path))
    assert list(elsewhere.iterdir()) == []


def test_the_entry_is_built_where_only_its_builder_can_reach(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    download = cache_build._download_wheels
    building: list[tuple[Path, int]] = []

    def watching_download(lock: Path, wheels: Path, pins: list[LockPin], options: BuildOptions) -> dict[str, str]:
        found = download(lock, wheels, pins, options)
        # The temporary directory the entry is made in, which holds the build environment's interpreter.
        scratch = wheels.parent.parent
        building.append((scratch, stat.S_IMODE(scratch.stat().st_mode)))
        return found

    monkeypatch.setattr(cache_build, "_download_wheels", watching_download)
    ensure_runner_cache(rig.repo, rig.cache.sha, cache, rig.build_options())

    ((scratch, mode),) = building
    assert not scratch.is_relative_to(cache) and mode & 0o077 == 0, (scratch, oct(mode))
    assert not scratch.exists()


_KILLED_AFTER_THE_BUILD_TOOLS = """
import os, signal, sys
sys.path.insert(0, sys.argv[1])
from hyperloom.measure.runner import cache_build
tools_environment = cache_build._tools_environment
def killed_after(*args):
    tools_environment(*args)
    os.kill(os.getpid(), signal.SIGKILL)
cache_build._tools_environment = killed_after
cache_build.main(sys.argv[2:])
"""

_TAKE_LOCKS = """
import errno, fcntl, json, os, sys
seen = {}
for name in sys.argv[1:]:
    try:
        fcntl.flock(os.open(name, os.O_RDWR), fcntl.LOCK_EX)
        seen[os.path.basename(name)] = "ok"
    except OSError as exc:
        seen[os.path.basename(name)] = errno.errorcode[exc.errno]
print(json.dumps(seen))
"""


def test_a_killed_build_leaves_the_cache_only_a_lock_another_accounts_builder_takes(rig: Rig, tmp_path: Path) -> None:
    with other_accounts() as accounts:
        cache = accounts.shared_dir("cache")
        sha = rig.cache.sha
        arguments = ["--sha", sha, "--repo", str(rig.repo), "--out", str(cache)]
        arguments += ["--no-index", "--find-links", str(rig.wheelhouse)]
        temporary = tmp_path / "tmp"
        temporary.mkdir()
        killed = subprocess.run(
            [sys.executable, "-I", "-c", _KILLED_AFTER_THE_BUILD_TOOLS, _SRC, *arguments],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            env={**os.environ, "TMPDIR": str(temporary)},
        )
        assert killed.returncode == -signal.SIGKILL, killed.stderr
        # What the killed builder left is in its own temporary directory, not in the cache.
        assert list(temporary.glob(f"hyperloom-runner-build-*/{sha}/.build-env-*/env/bin"))
        assert {path.name for path in cache.iterdir()} == {build_lock(cache, sha).name}
        old_style = cache / f".{sha}.lock"
        old_style.write_text("")
        old_style.chmod(0o644)

        names = [str(build_lock(cache, sha)), str(old_style)]
        taken = json.loads(accounts.run(accounts.second, f"import sys\nsys.argv[1:] = {names!r}\n{_TAKE_LOCKS}"))

        assert taken == {build_lock(cache, sha).name: "ok", old_style.name: "EACCES"}


_KILLED_BEFORE_THE_RENAME = """
import os, signal, sys
sys.path.insert(0, sys.argv[1])
from hyperloom.measure.runner import cache_build
copy_tree = cache_build._copy_tree
def killed_after(source, directory, last="", **flags):
    copy_tree(source, directory, last, **flags)
    if last:  # the whole entry is copied, the manifest last: the builder dies before the rename
        os.kill(os.getpid(), signal.SIGKILL)
cache_build._copy_tree = killed_after
cache_build.main(sys.argv[2:])
"""


def _build_command(rig: Rig, cache: Path, code: str | None = None) -> list[str]:
    arguments = ["--sha", rig.cache.sha, "--repo", str(rig.repo), "--out", str(cache)]
    arguments += ["--no-index", "--find-links", str(rig.wheelhouse)]
    if code is None:
        return [sys.executable, "-m", "hyperloom.measure.runner.cache_build", *arguments]
    return [sys.executable, "-I", "-c", code, _SRC, *arguments]


@pytest.mark.parametrize("umask", [0o022, 0o077])
def test_a_builder_killed_with_the_copy_whole_but_not_yet_in_place_leaves_what_the_next_build_sweeps(
    rig: Rig, tmp_path: Path, umask: int
) -> None:
    cache = tmp_path / "cache"
    sha = rig.cache.sha
    killed = subprocess.run(
        _build_command(rig, cache, _KILLED_BEFORE_THE_RENAME),
        capture_output=True,
        text=True,
        timeout=300,
        umask=umask,
        check=False,
    )
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    (leftover,) = cache.glob(f".{sha}.tmp-*")
    assert (leftover / MANIFEST_NAME).is_file(), "killed with the copy whole"
    assert stat.S_IMODE(leftover.stat().st_mode) & 0o777 == 0o700, "still the builder's alone"

    done = subprocess.run(
        _build_command(rig, cache), capture_output=True, text=True, timeout=300, umask=umask, check=False
    )
    assert done.returncode == 0, done.stderr
    assert {path.name for path in cache.iterdir()} == {sha, build_lock(cache, sha).name}


def test_a_finished_entry_renamed_to_a_leftovers_name_is_never_swept_whatever_the_umask_it_was_built_under(
    rig: Rig, tmp_path: Path
) -> None:
    """Another account of the group can rename an entry in the cache to ``.<sha>.tmp-*``; a builder of the entry's
    account that then sweeps must not take it for its own unfinished copy, even when it was built under umask 077."""
    cache = tmp_path / "cache"
    sha = rig.cache.sha
    built = subprocess.run(
        _build_command(rig, cache), capture_output=True, text=True, timeout=300, umask=0o077, check=False
    )
    assert built.returncode == 0, built.stderr
    renamed = cache / f".{sha}.tmp-1-renamed"
    (cache / sha).rename(renamed)

    again = subprocess.run(
        _build_command(rig, cache), capture_output=True, text=True, timeout=300, umask=0o077, check=False
    )
    assert again.returncode == 0, again.stderr
    assert (renamed / MANIFEST_NAME).is_file(), "a finished entry is never swept"
    assert {path.name for path in cache.iterdir()} == {sha, build_lock(cache, sha).name, renamed.name}


#: The builder dies just after its copy is renamed into place, before the ``fchmod`` that gives the group its read.
_KILLED_AFTER_THE_RENAME = """
import os, signal, sys
sys.path.insert(0, sys.argv[1])
from hyperloom.measure.runner import cache_build
real_rename = os.rename
def rename_then_die(source, target, **kwargs):
    real_rename(source, target, **kwargs)
    if kwargs.get("src_dir_fd") is not None and ".tmp-" in source:  # the copy's rename into place
        os.kill(os.getpid(), signal.SIGKILL)
cache_build.os.rename = rename_then_die
cache_build.main(sys.argv[2:])
"""


@pytest.mark.parametrize("umask", [0o022, 0o077])
def test_a_builder_killed_between_the_rename_and_its_fchmod_leaves_an_entry_the_next_build_sets_aside_and_sweeps(
    rig: Rig, tmp_path: Path, umask: int
) -> None:
    cache = tmp_path / "cache"
    sha = rig.cache.sha
    killed = subprocess.run(
        _build_command(rig, cache, _KILLED_AFTER_THE_RENAME),
        capture_output=True,
        text=True,
        timeout=300,
        umask=umask,
        check=False,
    )
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    assert stat.S_IMODE((cache / sha).stat().st_mode) & 0o777 == 0o700, "in place, but no other account can read it"

    done = subprocess.run(
        _build_command(rig, cache), capture_output=True, text=True, timeout=300, umask=umask, check=False
    )
    assert done.returncode == 0, done.stderr
    assert f"set aside {cache / sha}" in done.stderr
    assert {path.name for path in cache.iterdir()} == {sha, build_lock(cache, sha).name}, "set aside, then swept"
    assert stat.S_IMODE((cache / sha).stat().st_mode) & _ENTRY_READ == _ENTRY_READ


def test_a_builders_leftover_renamed_into_place_is_set_aside_and_the_entry_built_again(
    rig: Rig, tmp_path: Path
) -> None:
    """An account of the group can rename a builder's unfinished copy to the entry's name; the next builder must not
    take it for a conflicting entry that stays for ever."""
    cache = tmp_path / "cache"
    sha = rig.cache.sha
    cache.mkdir()
    _own_dir(cache / sha, 0o700, "half")

    done = subprocess.run(
        _build_command(rig, cache), capture_output=True, text=True, timeout=300, umask=0o022, check=False
    )
    assert done.returncode == 0, done.stderr
    assert (cache / sha / MANIFEST_NAME).is_file()
    assert {path.name for path in cache.iterdir()} == {sha, build_lock(cache, sha).name}


_PLANT_PTH = """
import errno, glob, sys
targets = glob.glob(sys.argv[1] + "/env/lib/python*/site-packages") or [sys.argv[1]]
try:
    with open(targets[0] + "/planted.pth", "w") as out:
        out.write("import os\\n")
    print("written")
except OSError as exc:
    print(errno.errorcode[exc.errno])
"""

#: Rename the directory the builder made in the cache away and put something of this account's in its place: a
#: directory of its own, empty (``dir``) or with links where the builder's files go to a file of the builder's that this
#: account cannot even read (``links``), or a link to a directory of the builder's (``link``).
_SWAP = """
import os, sys
cache, pending, kind, target, *names = sys.argv[1:]
pending = os.path.join(cache, pending)
os.rename(pending, os.path.join(cache, ".moved"))
if kind == "link":
    os.symlink(target, pending)
else:
    os.mkdir(pending, 0o700)
    for name in names:
        os.symlink(target, os.path.join(pending, name))
print("swapped")
"""


#: The builder's own files that :data:`_SWAP` plants links at.
_PLANTED = (BOOTSTRAP_NAME, MANIFEST_NAME, SOURCE_ARCHIVE)


def _as(accounts: Accounts, code: str, *arguments: str) -> str:
    return accounts.run(accounts.second, f"import sys\nsys.argv[1:] = {list(arguments)!r}\n{code}").strip()


@pytest.mark.parametrize("swap", ["dir-before-it-is-opened", "link-before-it-is-opened", "after-it-is-opened"])
def test_another_account_of_the_group_neither_changes_what_a_build_runs_nor_makes_it_write_through_a_link(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, swap: str
) -> None:
    victim = tmp_path / "builders-own"
    victim.write_text("the builder's own\n")
    victim.chmod(0o600)
    # A directory of the builder's, empty and 0700 as the one it has just made: only that the name is a link to it, not
    # what it is, tells it from that one.
    builders = tmp_path / "builders-dir"
    builders.mkdir(0o700)
    tools_environment, open_new_directory = cache_build._tools_environment, cache_build._open_new_directory
    seen: dict[str, str] = {}
    pending: list[str] = []
    with other_accounts() as accounts:
        cache = accounts.shared_dir("cache")

        def planting(pinned: Path, scratch: Path) -> Path:
            python = tools_environment(pinned, scratch)
            seen["pth"] = _as(accounts, _PLANT_PTH, str(scratch))
            return python

        def swapping(parent: int, name: str) -> int:
            if "swap" in seen:
                return open_new_directory(parent, name)
            pending.append(name)
            if swap == "after-it-is-opened":
                descriptor = open_new_directory(parent, name)
                seen["swap"] = _as(accounts, _SWAP, str(cache), name, "links", str(victim), *_PLANTED)
                return descriptor
            seen["swap"] = _as(accounts, _SWAP, str(cache), name, swap.split("-")[0], str(builders))
            return open_new_directory(parent, name)

        monkeypatch.setattr(cache_build, "_tools_environment", planting)
        monkeypatch.setattr(cache_build, "_open_new_directory", swapping)
        built = _build_or_error(lambda: ensure_runner_cache(rig.repo, rig.cache.sha, cache, rig.build_options()))

        # The build environment is out of the account's reach, and nothing the builder wrote went through its links.
        assert seen == {"pth": "EACCES", "swap": "swapped"}
        assert (victim.read_text(), stat.S_IMODE(victim.stat().st_mode)) == ("the builder's own\n", 0o600)
        final, moved, (name,) = cache / rig.cache.sha, cache / ".moved", pending
        if swap == "after-it-is-opened":
            # The builder filled the directory it opened, wherever the account moved it; what the account put in place
            # is its own, and the pin the build returns refuses it.
            assert isinstance(built, CacheEntry), built
            assert check_entry(moved, rig.cache.sha, built.manifest_sha256, verify_hash=True) == []
            assert all((final / planted).is_symlink() for planted in _PLANTED)
            assert check_entry(final, rig.cache.sha, built.manifest_sha256) != []
            return
        # The builder refused what the name opened before it wrote anything, and left it as the account put it.
        assert isinstance(built, CacheBuildError), built
        assert not final.exists() and list(moved.iterdir()) == [] and list(builders.iterdir()) == []
        if swap == "dir-before-it-is-opened":
            assert f"so nothing is written to it: it is owned by uid {accounts.second}," in str(built)
            assert (cache / name).lstat().st_uid == accounts.second and list((cache / name).iterdir()) == []
        else:
            assert f"cannot open {name}, the directory the builder has just made: Not a directory" in str(built)
            assert os.readlink(cache / name) == str(builders)


def _build_or_error(build: Callable[[], CacheEntry]) -> CacheEntry | OSError | CacheBuildError:
    try:
        return build()
    except (OSError, CacheBuildError) as exc:
        return exc


def _link_to_a_copy(path: Path) -> None:
    copy = path.with_name(f".{path.name}.copy")
    shutil.copyfile(path, copy)
    path.unlink()
    path.symlink_to(copy.name)


@pytest.mark.parametrize("where", ["the-entry", BOOTSTRAP_NAME, MANIFEST_NAME, "a-wheel"])
def test_an_entry_with_a_link_where_a_builder_makes_a_file_is_refused_and_not_read(
    rig: Rig, tmp_path: Path, where: str
) -> None:
    cache = copy_cache(rig, tmp_path / "cache")
    entry = cache.root / cache.sha
    if where == "the-entry":
        entry.rename(cache.root / "elsewhere")
        entry.symlink_to("elsewhere")
    elif where == "a-wheel":
        _link_to_a_copy(next((entry / WHEELS_DIR).iterdir()))
    else:
        _link_to_a_copy(entry / where)
    # Each link reaches the very bytes the builder made, so only the link itself is the difference.
    assert check_entry(entry, cache.sha, verify_hash=True) == []
    with pytest.raises(CacheConflict, match=r"is left alone: .* is a link"):
        ensure_runner_cache(rig.repo, cache.sha, cache.root, rig.build_options())
    assert entry.is_symlink() if where == "the-entry" else (entry / MANIFEST_NAME).exists()


# -- what a builder removes of what it finds in the cache, and what it copies into -------------------------------------

_SHA = "a" * 40
_LEFTOVER = f".{_SHA}.tmp-"


def _own_dir(path: Path, mode: int, *files: str) -> Path:
    """A new directory of this account's at ``path`` with ``mode``, holding a file of each name in ``files``."""
    path.mkdir()
    for name in files:
        (path / name).write_text(f"{name}\n")
    path.chmod(mode)
    return path


def _snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    """``root`` (as ``.``) and each path below it: its owner, permission bits and content or link target."""
    found: dict[str, tuple[int, int, str]] = {}
    for path in [root, *sorted(root.rglob("*"))]:
        info = os.lstat(path)
        content = os.readlink(path) if path.is_symlink() else path.read_text() if path.is_file() else ""
        found[path.relative_to(root).as_posix()] = (info.st_uid, stat.S_IMODE(info.st_mode), content)
    return found


def test_a_builder_removes_of_the_leftovers_only_a_directory_its_account_made_for_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    outside = _own_dir(tmp_path / "outside", 0o700, "kept")
    # What a build of this account killed while copying left: removed, and nothing through its link.
    own = _own_dir(cache / f"{_LEFTOVER}1-own", 0o700, "half")
    _own_dir(own / "source", 0o755, "file")
    (own / "outside").symlink_to(outside)
    # What another account of the group may rename to such a name, none of it a directory the builder made for itself:
    # an entry of the builder's, a file of its own and a link to a directory of its own.
    _own_dir(cache / f"{_LEFTOVER}2-entry", 0o755, MANIFEST_NAME)
    (cache / f"{_LEFTOVER}3-file").write_text("file\n")
    (cache / f"{_LEFTOVER}4-link").symlink_to(outside)
    # And a directory of the builder's own when it is listed, put away by the account and replaced by an entry or a
    # link before the builder opens it, or by an empty directory of the builder's once the builder has emptied it.
    for name in ("5-to-an-entry", "6-to-a-link", "7-after-emptying"):
        _own_dir(cache / f"{_LEFTOVER}{name}", 0o700, "half")
    other_entry, other_empty = _own_dir(cache / "entry", 0o755, MANIFEST_NAME), _own_dir(cache / "empty", 0o755)
    late, emptied = os.lstat(cache / f"{_LEFTOVER}7-after-emptying"), os.lstat(other_empty)
    remove, empty = cache_build._remove_directory, cache_build._empty

    def swapping_before_the_open(parent: int, name: str, ours: Callable[[os.stat_result], bool]) -> None:
        if name.endswith("-to-an-entry"):
            (cache / name).rename(tmp_path / "moved-5")
            other_entry.rename(cache / name)
        elif name.endswith("-to-a-link"):
            (cache / name).rename(tmp_path / "moved-6")
            (cache / name).symlink_to(outside)
        remove(parent, name, ours)

    def swapping_once_emptied(directory: int) -> None:
        empty(directory)
        if os.path.samestat(os.fstat(directory), late):
            (cache / f"{_LEFTOVER}7-after-emptying").rename(tmp_path / "moved-7")
            other_empty.rename(cache / f"{_LEFTOVER}7-after-emptying")

    monkeypatch.setattr(cache_build, "_remove_directory", swapping_before_the_open)
    monkeypatch.setattr(cache_build, "_empty", swapping_once_emptied)
    kept = {name: _snapshot(cache / f"{_LEFTOVER}{name}") for name in ("2-entry", "3-file", "4-link")}
    outside_before = _snapshot(outside)

    cache_build._sweep_leftovers(cache, _SHA)

    assert {path.name for path in cache.iterdir()} == {
        f"{_LEFTOVER}{name}" for name in (*kept, "5-to-an-entry", "6-to-a-link", "7-after-emptying")
    }
    assert {name: _snapshot(cache / f"{_LEFTOVER}{name}") for name in kept} == kept
    assert sorted(path.name for path in (cache / f"{_LEFTOVER}5-to-an-entry").iterdir()) == [MANIFEST_NAME]
    assert os.readlink(cache / f"{_LEFTOVER}6-to-a-link") == str(outside) and _snapshot(outside) == outside_before
    assert os.path.samestat(os.lstat(cache / f"{_LEFTOVER}7-after-emptying"), emptied)


_THEIR_LEFTOVER = """
import os, sys
os.mkdir(sys.argv[1], 0o700)
with open(os.path.join(sys.argv[1], "half"), "w") as out:
    out.write("half\\n")
print("made")
"""


def test_a_builder_leaves_what_a_build_of_another_account_left() -> None:
    with other_accounts() as accounts:
        cache = accounts.shared_dir("cache")
        theirs = cache / f"{_LEFTOVER}1-theirs"
        assert _as(accounts, _THEIR_LEFTOVER, str(theirs)) == "made"
        before = _snapshot(theirs)
        cache_build._sweep_leftovers(cache, _SHA)
        assert _snapshot(theirs) == before and before["."][0] == accounts.second


def _made(tmp_path: Path) -> Path:
    """An entry as the builder makes it in its own directory, to copy into the cache."""
    made = _own_dir(tmp_path / "made", 0o755, MANIFEST_NAME)
    _own_dir(made / "source", 0o755, "file")
    return made


@pytest.mark.parametrize(
    ("put", "refusal"),
    [
        ("a-shared-directory", "so nothing is written to it: its mode is 0775, not 0700"),
        ("a-copy-in-progress", "so nothing is written to it: it is not empty"),
        ("a-link", "the directory the builder has just made: Not a directory"),
    ],
)
def test_a_copy_refuses_what_was_put_in_place_of_its_new_directory_and_leaves_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, put: str, refusal: str
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    made = _made(tmp_path)
    # Directories of the builder's, which another account of the group can rename where it has just made its own: one
    # the group may write, another copy in progress, and (through a link) one outside the cache like the one it made.
    target = {
        "a-shared-directory": lambda: _own_dir(cache / "shared", 0o775),
        "a-copy-in-progress": lambda: _own_dir(cache / ".other.tmp-1-x", 0o700, "half"),
        "a-link": lambda: _own_dir(tmp_path / "elsewhere", 0o700),
    }[put]()
    before = _snapshot(target)
    open_new_directory = cache_build._open_new_directory
    pending: list[str] = []

    def swapping(parent: int, name: str) -> int:
        if not pending:
            pending.append(name)
            (cache / name).rename(tmp_path / "moved")
            if put == "a-link":
                (cache / name).symlink_to(target)
            else:
                target.rename(cache / name)
        return open_new_directory(parent, name)

    monkeypatch.setattr(cache_build, "_open_new_directory", swapping)
    with pytest.raises(CacheBuildError, match=refusal):
        cache_build._publish(made, cache / _SHA)
    planted = cache / pending[0]
    assert not (cache / _SHA).exists() and list((tmp_path / "moved").iterdir()) == []
    assert _snapshot(target if put == "a-link" else planted) == before
    assert planted.is_symlink() == (put == "a-link")


@pytest.mark.parametrize("swapped", [False, True], ids=["in-place", "swapped-after-it-is-opened"])
def test_a_copy_that_fails_removes_the_directory_it_opened_and_nothing_put_in_its_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, swapped: bool
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    made = _made(tmp_path)
    in_progress = _own_dir(cache / ".other.tmp-1-x", 0o700, "half")
    before = _snapshot(in_progress)
    open_new_directory, copy_file = cache_build._open_new_directory, cache_build._copy_file
    pending: list[str] = []

    def opening(parent: int, name: str) -> int:
        descriptor = open_new_directory(parent, name)
        if not pending:
            pending.append(name)
            if swapped:
                (cache / name).rename(tmp_path / "moved")
                in_progress.rename(cache / name)
        return descriptor

    def failing_on_the_manifest(source: Path, directory: int, name: str, mode: int) -> None:
        if name == MANIFEST_NAME:
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        copy_file(source, directory, name, mode)

    monkeypatch.setattr(cache_build, "_open_new_directory", opening)
    monkeypatch.setattr(cache_build, "_copy_file", failing_on_the_manifest)
    with pytest.raises(OSError, match=os.strerror(errno.ENOSPC)):
        cache_build._publish(made, cache / _SHA)
    if swapped:
        # The directory the copy filled is wherever the account moved it, the builder's own to remove.
        assert {path.name for path in cache.iterdir()} == {pending[0]} and _snapshot(cache / pending[0]) == before
        assert (tmp_path / "moved" / "source" / "file").is_file()
    else:
        assert {path.name for path in cache.iterdir()} == {in_progress.name} and _snapshot(in_progress) == before
