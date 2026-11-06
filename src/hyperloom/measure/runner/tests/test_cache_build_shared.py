# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner cache on shared storage, where its builders may be different accounts of one group, and so may anything
else that group runs: a lock every one of them can open for writing, a build that runs nothing and writes through
nothing another account of the group can change, and an entry in place that only its builder can change."""

from __future__ import annotations

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

from hyperloom_common.tests.shared_accounts import Accounts, other_accounts
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
    root = tmp_path / "cache"
    command = [sys.executable, "-m", "hyperloom.measure.runner.cache_build", "--sha", rig.cache.sha]
    command += ["--repo", str(rig.repo), "--out", str(root), "--no-index", "--find-links", str(rig.wheelhouse)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=300, umask=umask, check=False)
    assert done.returncode == 0, done.stderr
    assert stat.S_IMODE(build_lock(root, rig.cache.sha).stat().st_mode) == lock
    assert stat.S_IMODE(root.stat().st_mode) == cache
    # The entry is readable as the builder's umask lets others read, and writable by its builder alone, even under a
    # umask that lets the group write what the builder makes.
    entry = _modes(root / rig.cache.sha)
    assert (entry["."], entry[BOOTSTRAP_NAME]) == (0o755 & ~umask, 0o755)
    assert {name for name, mode in entry.items() if mode & _SHARED_WRITE} == set()


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

#: Rename the directory the builder made in the cache away and put one of this account's own in its place, with links
#: where the builder's files go to a file of the builder's that this account cannot even read.
_SWAP = """
import os, sys
cache, pending, victim, *names = sys.argv[1:]
os.rename(os.path.join(cache, pending), os.path.join(cache, ".moved"))
os.mkdir(os.path.join(cache, pending))
for name in names:
    os.symlink(victim, os.path.join(cache, pending, name))
print("swapped")
"""


#: The builder's own files that :data:`_SWAP` plants links at.
_PLANTED = (BOOTSTRAP_NAME, MANIFEST_NAME, SOURCE_ARCHIVE)


def _as(accounts: Accounts, code: str, *arguments: str) -> str:
    return accounts.run(accounts.second, f"import sys\nsys.argv[1:] = {list(arguments)!r}\n{code}").strip()


@pytest.mark.parametrize("swap", ["before-it-is-opened", "after-it-is-opened"])
def test_another_account_of_the_group_neither_changes_what_a_build_runs_nor_makes_it_write_through_a_link(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, swap: str
) -> None:
    victim = tmp_path / "builders-own"
    victim.write_text("the builder's own\n")
    victim.chmod(0o600)
    tools_environment, open_new_directory = cache_build._tools_environment, cache_build._open_new_directory
    seen: dict[str, str] = {}
    with other_accounts() as accounts:
        cache = accounts.shared_dir("cache")

        def planting(pinned: Path, scratch: Path) -> Path:
            python = tools_environment(pinned, scratch)
            seen["pth"] = _as(accounts, _PLANT_PTH, str(scratch))
            return python

        def swapping(parent: int, name: str) -> int:
            if "swap" in seen:
                return open_new_directory(parent, name)
            if swap == "after-it-is-opened":
                descriptor = open_new_directory(parent, name)
                seen["swap"] = _as(accounts, _SWAP, str(cache), name, str(victim), *_PLANTED)
                return descriptor
            seen["swap"] = _as(accounts, _SWAP, str(cache), name, str(victim), *_PLANTED)
            return open_new_directory(parent, name)

        monkeypatch.setattr(cache_build, "_tools_environment", planting)
        monkeypatch.setattr(cache_build, "_open_new_directory", swapping)
        built = _build_or_error(lambda: ensure_runner_cache(rig.repo, rig.cache.sha, cache, rig.build_options()))

        # The build environment is out of the account's reach, and nothing the builder wrote went through its links.
        assert seen == {"pth": "EACCES", "swap": "swapped"}
        assert (victim.read_text(), stat.S_IMODE(victim.stat().st_mode)) == ("the builder's own\n", 0o600)
        final, moved = cache / rig.cache.sha, cache / ".moved"
        if swap == "before-it-is-opened":
            # The builder opened the account's directory, and found its links where its own files go.
            assert isinstance(built, FileExistsError), built
            assert not final.exists() and list(moved.iterdir()) == []
        else:
            # The builder filled the directory it opened, wherever the account moved it; what the account put in place
            # is its own, and the pin the build returns refuses it.
            assert isinstance(built, CacheEntry), built
            assert check_entry(moved, rig.cache.sha, built.manifest_sha256, verify_hash=True) == []
            assert all((final / planted).is_symlink() for planted in _PLANTED)
            assert check_entry(final, rig.cache.sha, built.manifest_sha256) != []


def _build_or_error(build: Callable[[], CacheEntry]) -> CacheEntry | OSError:
    try:
        return build()
    except OSError as exc:
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
