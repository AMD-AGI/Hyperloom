# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner cache on shared storage, where its builders may be different accounts of one group: a lock every one of
them can open for writing, a build tree another account's builder can remove when the builder that made it was killed,
and an entry in place that only its builder can change."""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom_common.tests.shared_accounts import other_accounts
from hyperloom.measure.runner import cache_build
from hyperloom.measure.runner.cache_build import BuildOptions, CacheBuildError, build_lock, ensure_runner_cache
from hyperloom.measure.runner.cache_layout import LockPin
from hyperloom.measure.runner.tests.bootstrap_rig import Rig

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
    ("umask", "lock", "cache"), [(0o022, 0o664, 0o2775), (0o027, 0o660, 0o2770), (0o077, 0o660, 0o2770)]
)
def test_the_lock_and_the_cache_let_the_group_write_and_others_what_the_builders_umask_does(
    rig: Rig, tmp_path: Path, umask: int, lock: int, cache: int
) -> None:
    root = tmp_path / "cache"
    command = [sys.executable, "-m", "hyperloom.measure.runner.cache_build", "--sha", rig.cache.sha]
    command += ["--repo", str(rig.repo), "--out", str(root), "--no-index", "--find-links", str(rig.wheelhouse)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=300, umask=umask, check=False)
    assert done.returncode == 0, done.stderr
    assert stat.S_IMODE(build_lock(root, rig.cache.sha).stat().st_mode) == lock
    assert stat.S_IMODE(root.stat().st_mode) == cache


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


def test_the_tree_is_the_groups_while_it_is_built_and_its_builders_alone_once_in_place(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    download = cache_build._download_wheels
    building: list[dict[str, int]] = []

    def watching_download(lock: Path, wheels: Path, pins: list[LockPin], options: BuildOptions) -> dict[str, str]:
        found = download(lock, wheels, pins, options)
        building.append(_modes(wheels.parent))
        return found

    monkeypatch.setattr(cache_build, "_download_wheels", watching_download)
    entry = ensure_runner_cache(rig.repo, rig.cache.sha, cache, rig.build_options())

    (tree,) = building
    directories = {name: mode for name, mode in tree.items() if (entry.path / name).is_dir()}
    assert {"source", "wheels"} <= set(directories)
    assert {name for name, mode in directories.items() if mode & 0o2070 != 0o2070} == set(), directories
    assert {name for name, mode in _modes(entry.path).items() if mode & _SHARED_WRITE} == set()


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

_SWEEP = """
import errno, fcntl, json, os, shutil, sys
seen = {}
for name in sys.argv[1:]:
    try:
        if name.endswith("lock"):
            fcntl.flock(os.open(name, os.O_RDWR), fcntl.LOCK_EX)
        else:
            shutil.rmtree(name)
        seen[os.path.basename(name)] = "ok"
    except OSError as exc:
        seen[os.path.basename(name)] = errno.errorcode[exc.errno]
print(json.dumps(seen))
"""


def test_another_accounts_builder_can_take_the_lock_and_remove_what_a_killed_build_left(rig: Rig) -> None:
    with other_accounts() as accounts:
        cache = accounts.shared_dir("cache")
        sha = rig.cache.sha
        arguments = ["--sha", sha, "--repo", str(rig.repo), "--out", str(cache)]
        arguments += ["--no-index", "--find-links", str(rig.wheelhouse)]
        killed = subprocess.run(
            [sys.executable, "-I", "-c", _KILLED_AFTER_THE_BUILD_TOOLS, _SRC, *arguments],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert killed.returncode == -signal.SIGKILL, killed.stderr
        (left,) = cache.glob(f".{sha}.tmp-*")
        assert (left / "source").is_dir() and list(left.glob(".build-env-*/env/bin"))
        old_style = cache / f".{sha}.lock"
        old_style.write_text("")
        old_style.chmod(0o644)

        names = [str(build_lock(cache, sha)), str(old_style), str(left)]
        swept = json.loads(accounts.run(accounts.second, f"import sys\nsys.argv[1:] = {names!r}\n{_SWEEP}"))

        assert swept == {build_lock(cache, sha).name: "ok", old_style.name: "EACCES", left.name: "ok"}
        assert not left.exists()
