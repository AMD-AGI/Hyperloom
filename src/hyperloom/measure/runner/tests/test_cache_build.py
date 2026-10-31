# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner cache builder: one commit builds once and atomically, a second call reuses the entry, an entry that
differs is never overwritten, builders of one commit exclude each other, and an unfinished entry is never used.

The wheels come from a local wheelhouse (``--no-index --find-links``), so nothing here needs a package index.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.measure.runner import cache_build
from hyperloom.measure.runner.cache_build import (
    BuildOptions,
    CacheBuildError,
    CacheConflict,
    WheelSource,
    WheelTarget,
    ensure_runner_cache,
)
from hyperloom.measure.runner.cache_layout import (
    CACHE_ENV,
    MANIFEST_NAME,
    CacheError,
    LockPin,
    check_cache,
    check_entry,
    parse_lock,
    requirement_line,
    tree_digest,
)
from hyperloom.measure.runner.tests.bootstrap_rig import (
    UNFETCHABLE,
    Rig,
    build_sdist,
    commit_edit,
    copy_cache,
    lock_text,
)

_NO_WHEELS = BuildOptions(source=WheelSource(no_index=True))
_LOCK_WHEEL = "lock_pkg-1.0-py3-none-any.whl"
_BUILD = "hyperloom.measure.runner.cache_build"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command(rig: Rig, cache: Path, *extra: str) -> list[str]:
    return [sys.executable, "-m", _BUILD, "--sha", rig.cache.sha, "--repo", str(rig.repo), "--out", str(cache), *extra]


def _wheelhouse_args(rig: Rig) -> list[str]:
    return ["--no-index", "--find-links", str(rig.wheelhouse)]


def _names(cache: Path) -> set[str]:
    return {path.name for path in cache.iterdir()}


def test_a_build_exports_the_commit_with_its_wheels_and_a_manifest_that_checks_out(rig: Rig, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    entry = ensure_runner_cache(rig.repo, rig.cache.sha, cache, rig.build_options())
    assert (entry.path, entry.reused) == (cache / rig.cache.sha, False)
    assert entry.manifest_sha256 == _digest(entry.path / MANIFEST_NAME)
    files = {
        path.relative_to(entry.path / "source").as_posix()
        for path in (entry.path / "source").rglob("*")
        if path.is_file()
    }
    assert files == set(_git(rig.repo, "ls-tree", "-r", "--name-only", rig.cache.sha).splitlines())
    assert sorted(path.name for path in (entry.path / "wheels").iterdir()) == [
        "already_pkg-1.0-py3-none-any.whl",
        _LOCK_WHEEL,
        "sdist_pkg-1.0-py3-none-any.whl",
        "setuptools-79.0.1-py3-none-any.whl",
    ]
    assert check_entry(entry.path, rig.cache.sha, verify_hash=True) == []
    assert check_cache(cache, rig.cache.sha) == []
    manifest = json.loads((entry.path / MANIFEST_NAME).read_text())
    assert manifest["git_tree"] == _git(rig.repo, "rev-parse", f"{rig.cache.sha}^{{tree}}")
    assert manifest["source"]["sha256"] == tree_digest(entry.path / "source").sha256
    assert _names(cache) == {rig.cache.sha, f".{rig.cache.sha}.lock"}


def test_a_pin_the_index_serves_only_as_a_source_distribution_is_built_and_the_sdist_recorded(
    rig: Rig, tmp_path: Path
) -> None:
    entry = ensure_runner_cache(rig.repo, rig.cache.sha, tmp_path / "cache", rig.build_options())
    wheels = {wheel["name"]: wheel for wheel in json.loads((entry.path / MANIFEST_NAME).read_text())["wheels"]}
    assert wheels["sdist-pkg"]["sdist_sha256"] == rig.sdist_pkg_sha256
    assert wheels["sdist-pkg"]["sha256"] != rig.sdist_pkg_sha256
    assert wheels["lock-pkg"]["sdist_sha256"] == "" and wheels["lock-pkg"]["sha256"] == rig.lock_pkg_sha256
    assert [path.name for path in (entry.path / "wheels").iterdir() if not path.name.endswith(".whl")] == []
    assert [path.name for path in entry.path.iterdir() if path.name.startswith(".")] == []
    # The source distribution declares a build requirement that no wheelhouse or index has: it was built with the
    # lock's own tools, and nothing the sdist asks for was fetched.
    with tarfile.open(rig.wheelhouse / "sdist_pkg-1.0.tar.gz") as sdist:
        declared = sdist.extractfile("sdist_pkg-1.0/pyproject.toml")
        assert declared is not None and UNFETCHABLE in declared.read().decode()


def _platform_sdist_house(rig: Rig, tmp_path: Path) -> tuple[Path, str]:
    """A copy of the wheelhouse whose ``sdist_pkg`` builds a wheel for one platform, and that source distribution's hash."""
    house = tmp_path / "house"
    shutil.copytree(rig.wheelhouse, house)
    (house / "sdist_pkg-1.0.tar.gz").unlink()
    return house, build_sdist(house, "sdist_pkg", "1.0", "ORIGIN = 'sdist'\n", "cp310-cp310-linux_x86_64")[1]


def test_a_source_distribution_that_builds_a_platform_wheel_is_refused(rig: Rig, tmp_path: Path) -> None:
    house, sdist_sha = _platform_sdist_house(rig, tmp_path)
    work = tmp_path / "work"
    work.mkdir()

    def lock_of_the_platform_sdist(repo: Path) -> None:
        (repo / "requirements" / "runner.lock").write_text(
            lock_text(
                lock_pkg=("1.0", rig.lock_pkg_sha256),
                sdist_pkg=("1.0", sdist_sha),
                setuptools=("79.0.1", rig.tools_sha256),
            )
        )

    commit = commit_edit(rig, work, lock_of_the_platform_sdist)
    cache = tmp_path / "cache"
    options = BuildOptions(source=WheelSource(no_index=True, find_links=(str(house),)))
    with pytest.raises(
        CacheBuildError, match="sdist_pkg-1.0.tar.gz did not build exactly one pure-Python wheel"
    ) as refused:
        ensure_runner_cache(commit.repo, commit.sha, cache, options)
    assert "sdist_pkg-1.0-cp310-cp310-linux_x86_64.whl" in str(refused.value)
    _assert_nothing_left(cache, commit.sha)


def test_a_second_build_of_the_commit_uses_the_entry_and_downloads_nothing(rig: Rig, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    first = ensure_runner_cache(rig.repo, rig.cache.sha, cache, rig.build_options())
    written = (first.path / MANIFEST_NAME).stat().st_mtime_ns
    # A build would fail with these options (no index and no wheels), so only the entry itself can answer.
    again = ensure_runner_cache(rig.repo, rig.cache.sha, cache, _NO_WHEELS)
    assert (again.path, again.manifest_sha256, again.reused) == (first.path, first.manifest_sha256, True)
    assert (first.path / MANIFEST_NAME).stat().st_mtime_ns == written


def _edit_source(entry: Path) -> None:
    main = entry / "source" / "src/hyperloom/measure/runner/__main__.py"
    main.write_text(main.read_text() + "\n# not what was built\n")


def _damage_wheel_keeping_its_size(entry: Path) -> None:
    wheel = entry / "wheels" / _LOCK_WHEEL
    damaged = bytearray(wheel.read_bytes())
    damaged[len(damaged) // 2] ^= 0xFF
    wheel.write_bytes(bytes(damaged))


def _name_another_tree(entry: Path) -> None:
    manifest = json.loads((entry / MANIFEST_NAME).read_text())
    manifest["git_tree"] = "f" * 40
    (entry / MANIFEST_NAME).write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    ("damage", "difference"),
    [
        (_edit_source, "is not what the manifest recorded"),
        (_damage_wheel_keeping_its_size, f"wheel {_LOCK_WHEEL} for lock-pkg==1.0 does not match the hash"),
        (_name_another_tree, "its manifest records tree " + "f" * 40),
    ],
)
def test_an_entry_that_differs_from_what_a_build_writes_is_refused_and_left_as_it_was(
    rig: Rig, tmp_path: Path, damage: Callable[[Path], None], difference: str
) -> None:
    cache = copy_cache(rig, tmp_path / "cache")
    entry = cache.root / cache.sha
    damage(entry)
    before = (tree_digest(entry), _digest(entry / MANIFEST_NAME))
    with pytest.raises(CacheConflict, match="is left alone") as refused:
        ensure_runner_cache(rig.repo, rig.cache.sha, cache.root, rig.build_options())
    assert difference in str(refused.value)
    assert (tree_digest(entry), _digest(entry / MANIFEST_NAME)) == before
    assert _names(cache.root) == {rig.cache.sha, f".{rig.cache.sha}.lock"}


def test_an_entry_built_for_another_target_is_refused_not_overwritten(rig: Rig, tmp_path: Path) -> None:
    cache = copy_cache(rig, tmp_path / "cache")
    other = replace(
        rig.build_options(), target=WheelTarget(python_version="3.10", platforms=("manylinux_2_28_x86_64",))
    )
    with pytest.raises(CacheConflict, match="its wheels are for") as refused:
        ensure_runner_cache(rig.repo, rig.cache.sha, cache.root, other)
    assert '"python_version": null' in str(refused.value)
    assert check_cache(cache.root, rig.cache.sha) == []


def test_an_unfinished_entry_is_refused_by_the_check_and_replaced_by_a_build(rig: Rig, tmp_path: Path) -> None:
    cache = copy_cache(rig, tmp_path / "cache")
    entry = cache.root / cache.sha
    (entry / MANIFEST_NAME).unlink()
    stale = cache.root / f".{cache.sha}.tmp-1-dead"
    stale.mkdir()
    (stale / "half").write_text("x")
    assert "has no manifest: an unfinished build" in check_cache(cache.root, cache.sha)[0]
    built = ensure_runner_cache(rig.repo, cache.sha, cache.root, rig.build_options())
    assert built.reused is False and check_entry(entry, cache.sha, verify_hash=True) == []
    assert _names(cache.root) == {cache.sha, f".{cache.sha}.lock"}


def _assert_nothing_left(cache: Path, sha: str) -> None:
    assert not (cache / sha).exists()
    assert [name for name in _names(cache) if ".tmp-" in name] == []


def test_a_commit_the_repository_does_not_have_or_that_is_abbreviated_builds_nothing(rig: Rig, tmp_path: Path) -> None:
    with pytest.raises(CacheBuildError, match="looking for " + "1" * 40):
        ensure_runner_cache(rig.repo, "1" * 40, tmp_path / "cache", rig.build_options())
    with pytest.raises(CacheBuildError, match="is not a full lowercase commit hash"):
        ensure_runner_cache(rig.repo, rig.cache.sha[:12], tmp_path / "cache", rig.build_options())
    with pytest.raises(CacheBuildError, match="is not a full lowercase commit hash"):
        ensure_runner_cache(rig.repo, rig.cache.sha.upper(), tmp_path / "cache", rig.build_options())
    assert not (tmp_path / "cache").exists()


def _tools_pin(repo: Path) -> str:
    pins = parse_lock((repo / "requirements" / "runner.lock").read_text())
    return "".join(
        requirement_line(pin.name, pin.version, sorted(pin.hashes)) for pin in pins if pin.name == "setuptools"
    )


def _lock_without_hashes(repo: Path) -> None:
    (repo / "requirements" / "runner.lock").write_text("lock_pkg==1.0\nalready_pkg==1.0\n" + _tools_pin(repo))


def _lock_with_a_pin_the_wheelhouse_lacks(repo: Path) -> None:
    lock = repo / "requirements" / "runner.lock"
    lock.write_text(lock.read_text() + lock_text(ghost_pkg=("1.0", "0" * 64)))


def _lock_that_is_not_a_lock(repo: Path) -> None:
    (repo / "requirements" / "runner.lock").write_text("lock_pkg>=1\n")


def _no_lock(repo: Path) -> None:
    (repo / "requirements" / "runner.lock").unlink()


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_lock_without_hashes, "Hashes are required"),
        (_lock_with_a_pin_the_wheelhouse_lacks, "No matching distribution found for ghost_pkg==1.0"),
        (_lock_that_is_not_a_lock, "runner.lock line 1 is not a pin or a hash"),
        (_no_lock, "has no requirements/runner.lock"),
    ],
)
def test_a_lock_that_cannot_be_downloaded_fails_the_build_and_leaves_no_entry(
    rig: Rig, tmp_path: Path, edit: Callable[[Path], None], message: str
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    commit = commit_edit(rig, work, edit)
    cache = tmp_path / "cache"
    with pytest.raises(CacheError, match=message):
        ensure_runner_cache(commit.repo, commit.sha, cache, rig.build_options())
    _assert_nothing_left(cache, commit.sha)


def test_a_lock_that_does_not_pin_setuptools_cannot_have_a_source_distribution_built(rig: Rig, tmp_path: Path) -> None:
    def lock_without_the_build_tool(repo: Path) -> None:
        (repo / "requirements" / "runner.lock").write_text(
            lock_text(
                lock_pkg=("1.0", rig.lock_pkg_sha256),
                already_pkg=("1.0", rig.already_pkg_sha256),
                sdist_pkg=("1.0", rig.sdist_pkg_sha256),
            )
        )

    work = tmp_path / "work"
    work.mkdir()
    commit = commit_edit(rig, work, lock_without_the_build_tool)
    with pytest.raises(CacheBuildError, match="the lock pins no setuptools"):
        ensure_runner_cache(commit.repo, commit.sha, tmp_path / "cache", rig.build_options())
    _assert_nothing_left(tmp_path / "cache", commit.sha)


def _fake_download(files: dict[str, bytes]) -> Callable[..., dict[str, str]]:
    def download(lock: Path, wheels: Path, pins: list[LockPin], options: BuildOptions) -> dict[str, str]:
        wheels.mkdir()
        for name, content in files.items():
            (wheels / name).write_bytes(content)
        return {}

    return download


@pytest.mark.parametrize(
    ("served", "message"),
    [
        (
            {"stranger-1.0-py3-none-any.whl": b"not in the lock"},
            "stranger-1.0-py3-none-any.whl is not a file the lock allows",
        ),
        ({}, "no wheel was downloaded for lock-pkg==1.0, already-pkg==1.0"),
        ({"twice.whl": b"lock", "again.whl": b"lock"}, "two wheels for lock-pkg==1.0"),
    ],
)
def test_wheels_that_do_not_match_the_lock_pin_for_pin_fail_the_build(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: dict[str, bytes], message: str
) -> None:
    work = tmp_path / "work"
    work.mkdir()
    sha256 = hashlib.sha256(b"lock").hexdigest()

    def lock_pinned_to_the_served_bytes(repo: Path) -> None:
        (repo / "requirements" / "runner.lock").write_text(
            lock_text(lock_pkg=("1.0", sha256), already_pkg=("1.0", "e" * 64))
        )

    commit = commit_edit(rig, work, lock_pinned_to_the_served_bytes)
    monkeypatch.setattr(f"{_BUILD}._download_wheels", _fake_download(served))
    cache = tmp_path / "cache"
    with pytest.raises(CacheBuildError, match=message):
        ensure_runner_cache(commit.repo, commit.sha, cache, rig.build_options())
    _assert_nothing_left(cache, commit.sha)


def test_the_entry_is_built_beside_its_final_place_and_appears_whole_in_one_step(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    final = cache / rig.cache.sha
    download = cache_build._download_wheels
    seen: list[tuple[bool, list[str]]] = []

    def watching_download(lock: Path, wheels: Path, pins: list[LockPin], options: BuildOptions) -> dict[str, str]:
        seen.append((final.exists(), [path.name for path in cache.glob(f".{rig.cache.sha}.tmp-*")]))
        assert not (wheels.parent / MANIFEST_NAME).exists()
        return download(lock, wheels, pins, options)

    monkeypatch.setattr(cache_build, "_download_wheels", watching_download)
    ensure_runner_cache(rig.repo, rig.cache.sha, cache, rig.build_options())
    # While it is built the final directory is absent, the work is in a directory of its own with no manifest yet.
    assert [(exists, len(tmp)) for exists, tmp in seen] == [(False, 1)]
    assert (final / MANIFEST_NAME).is_file()


# -- builders of one commit exclude each other ------------------------------------------------------------------------


def _hold(lock: Path) -> int:
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor


def test_a_builder_that_cannot_get_the_lock_in_time_fails_and_builds_nothing(rig: Rig, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    descriptor = _hold(cache / f".{rig.cache.sha}.lock")
    try:
        with pytest.raises(CacheBuildError, match="another build still holds .*after 0 s"):
            ensure_runner_cache(rig.repo, rig.cache.sha, cache, replace(rig.build_options(), lock_timeout_sec=0.3))
    finally:
        os.close(descriptor)
    assert not (cache / rig.cache.sha).exists()
    assert ensure_runner_cache(rig.repo, rig.cache.sha, cache, rig.build_options()).reused is False


def test_a_builder_that_waits_for_another_uses_the_entry_that_one_finished(rig: Rig, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    descriptor = _hold(cache / f".{rig.cache.sha}.lock")
    # This builder has no wheels to build from: it can only succeed by using what the first one finishes.
    waiting = subprocess.Popen(
        _command(rig, cache, "--no-index"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        time.sleep(0.6)
        assert waiting.poll() is None, "the builder did not wait for the lock"
        pending = cache / f".{rig.cache.sha}.tmp-other"
        shutil.copytree(rig.cache.root / rig.cache.sha, pending)
        pending.rename(cache / rig.cache.sha)
    finally:
        os.close(descriptor)
    out, err = waiting.communicate(timeout=60)
    assert waiting.returncode == 0, err
    answer = json.loads(out)
    assert answer["reused"] is True and answer["path"] == str(cache / rig.cache.sha)


def test_two_builders_started_together_build_once_and_both_get_a_whole_entry(rig: Rig, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    command = _command(rig, cache, *_wheelhouse_args(rig))
    runs = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    results = [run.communicate(timeout=120) for run in runs]
    assert [run.returncode for run in runs] == [0, 0], [err for _, err in results]
    answers = [json.loads(out) for out, _ in results]
    assert sorted(answer["reused"] for answer in answers) == [False, True]
    assert {answer["manifest_sha256"] for answer in answers} == {_digest(cache / rig.cache.sha / MANIFEST_NAME)}
    assert check_entry(cache / rig.cache.sha, rig.cache.sha, verify_hash=True) == []
    assert _names(cache) == {rig.cache.sha, f".{rig.cache.sha}.lock"}


# -- the command line --------------------------------------------------------------------------------------------------


def test_the_command_prints_one_json_line_and_reads_the_cache_from_the_environment(rig: Rig, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    command = [sys.executable, "-m", _BUILD, "--sha", rig.cache.sha, "--repo", str(rig.repo), *_wheelhouse_args(rig)]
    done = subprocess.run(
        command, env={**os.environ, CACHE_ENV: str(cache)}, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    answer = json.loads(done.stdout)
    assert answer == {"path": str(cache / rig.cache.sha), "manifest_sha256": answer["manifest_sha256"], "reused": False}


def test_the_command_needs_a_cache_directory_and_says_why_a_build_failed(rig: Rig, tmp_path: Path) -> None:
    environ = {name: value for name, value in os.environ.items() if name != CACHE_ENV}
    lacking = subprocess.run(
        [sys.executable, "-m", _BUILD, "--sha", rig.cache.sha, "--repo", str(rig.repo)],
        env=environ,
        capture_output=True,
        text=True,
        check=False,
    )
    assert lacking.returncode == 2 and f"--out or ${CACHE_ENV} is required" in lacking.stderr
    failed = subprocess.run(
        _command(rig, tmp_path / "cache", "--no-index"), capture_output=True, text=True, check=False
    )
    assert failed.returncode == 1 and failed.stdout == ""
    assert "runner cache not built: pip download of the build tools the lock pins failed" in failed.stderr
