# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A wheel the builder built from a source distribution is reused only when building that source distribution again
makes it: the entry keeps the source distribution, and on reuse the builder checks it against the commit's lock, builds
it again offline in a directory of its own and compares the wheel it makes with the cached one, member by member.

A lock lists a source distribution's hash the way it lists a wheel's, so a cached wheel recorded as built from a file
the lock lists is bound to the commit only by that rebuild. The planted entries of the first test pass the entry's own
check and are staged by a sandbox handed their manifest's hash: only the builder's reuse check keeps them from being
pinned.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from hyperloom.measure.runner import cache_build
from hyperloom.measure.runner.cache_build import BuildOptions, CacheConflict, WheelSource, ensure_runner_cache
from hyperloom.measure.runner.cache_layout import (
    LOCK_FILE,
    MANIFEST_NAME,
    SDISTS_DIR,
    SOURCE_DIR,
    WHEELS_DIR,
    check_entry,
    manifest_from_json,
    manifest_to_json,
    stage_entry,
    tree_digest,
)
from hyperloom.measure.runner.tests.bootstrap_rig import (
    Rig,
    build_sdist,
    build_wheel,
    commit_edit,
    copy_cache,
    lock_text,
    tripwire,
)

#: Options a build cannot succeed with (no index and no wheels): only a reuse of the entry can answer.
_NO_WHEELS = BuildOptions(source=WheelSource(no_index=True))
_LOCK_WHEEL = "lock_pkg-1.0-py3-none-any.whl"
_SDIST = "sdist_pkg-1.0.tar.gz"
_SDIST_WHEEL = "sdist_pkg-1.0-py3-none-any.whl"
_RENAMED = "sdist_pkg-1.0-py2.py3-none-any.whl"
#: The real ``lock_pkg`` wheel's names where the entry keeps a source distribution: a source distribution's two.
_TAR_GZ, _ZIP = "lock_pkg-1.0.tar.gz", "lock_pkg-1.0.zip"
#: Why a reuse refuses a source distribution recorded with a hash the commit's lock does not list.
_NOT_IN_THE_LOCK = (
    f"{SDISTS_DIR}/{_SDIST} is not the source distribution of {_SDIST_WHEEL} for sdist-pkg==1.0 that the manifest "
    "records and the lock lists"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Planted:
    """What an account that can write the cache can make itself.

    Attributes:
        sdist: Another ``sdist_pkg`` 1.0 source distribution.
        sdist_wheel: The wheel the builder builds from it (in an entry of a commit of its own whose lock pins it).
        lock_wheel: A ``lock_pkg`` 1.0 wheel of other code, with the real one's file name.
    """

    sdist: Path
    sdist_wheel: Path
    lock_wheel: Path


@pytest.fixture(scope="module")
def planted(rig: Rig, tmp_path_factory: pytest.TempPathFactory) -> Planted:
    root = tmp_path_factory.mktemp("planted")
    house = root / "house"
    shutil.copytree(rig.wheelhouse, house)
    (house / _SDIST).unlink()
    _, sdist_sha = build_sdist(house, "sdist_pkg", "1.0", "ORIGIN = 'planted'\n")

    def lock_of_the_planted_sdist(repo: Path) -> None:
        (repo / "requirements" / "runner.lock").write_text(
            lock_text(
                lock_pkg=("1.0", rig.lock_pkg_sha256),
                already_pkg=("1.0", rig.already_pkg_sha256),
                sdist_pkg=("1.0", sdist_sha),
                setuptools=("79.0.1", rig.tools_sha256),
            )
        )

    (root / "work").mkdir()
    commit = commit_edit(rig, root / "work", lock_of_the_planted_sdist)
    options = BuildOptions(source=WheelSource(no_index=True, find_links=(str(house),)))
    entry = ensure_runner_cache(commit.repo, commit.sha, root / "cache", options).path
    lock_wheel, _ = build_wheel(root, "lock_pkg", "1.0", "ORIGIN = 'planted'\n", requires="unlisted_pkg>=1")
    return Planted(entry / SDISTS_DIR / _SDIST, entry / WHEELS_DIR / _SDIST_WHEEL, lock_wheel)


@pytest.fixture
def rebuilt_from(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """The source distributions the builder builds a wheel from, in the order it does."""
    pure_wheel = cache_build._pure_wheel
    built: list[Path] = []

    def recording(python: Path, sdist: Path, out: Path) -> Path:
        built.append(sdist)
        return pure_wheel(python, sdist, out)

    monkeypatch.setattr(cache_build, "_pure_wheel", recording)
    return built


def _record_wheel(entry: Path, name: str, held: str, **fields: Any) -> None:
    """Record the size and hash of the entry's wheel file ``held``, the one of pin ``name``, as it now is, and
    ``fields``, in the manifest, in the very text the builder writes."""
    wheel = entry / WHEELS_DIR / held
    manifest = manifest_from_json((entry / MANIFEST_NAME).read_bytes())
    wheels = tuple(
        replace(listed, size=wheel.stat().st_size, sha256=_digest(wheel), **fields) if listed.name == name else listed
        for listed in manifest.wheels
    )
    (entry / MANIFEST_NAME).write_text(manifest_to_json(replace(manifest, wheels=wheels)))


def _lock_wheel_claiming_a_sdist_it_does_not_hold(entry: Path, planted: Planted) -> None:
    """Other ``lock_pkg`` code, recorded as built from a source distribution whose hash is the real wheel's, which the
    lock lists."""
    real = _digest(entry / WHEELS_DIR / _LOCK_WHEEL)
    shutil.copyfile(planted.lock_wheel, entry / WHEELS_DIR / _LOCK_WHEEL)
    _record_wheel(entry, "lock-pkg", _LOCK_WHEEL, sdist_sha256=real)


def _lock_wheel_recorded_as_built_from_a_copy_of_itself(entry: Path, held: str) -> None:
    """Keep a copy of the entry's real ``lock_pkg`` wheel as ``sdists/<held>``, a file whose hash the lock lists, and
    record the wheel as built from it."""
    shutil.copyfile(entry / WHEELS_DIR / _LOCK_WHEEL, entry / SDISTS_DIR / held)
    _record_wheel(entry, "lock-pkg", _LOCK_WHEEL, sdist_sha256=_digest(entry / SDISTS_DIR / held), sdist_file=held)


def _lock_wheel_with_the_real_one_kept_as_its_sdist(held: str) -> Callable[[Path, Planted], None]:
    """The same, with the real ``lock_pkg`` wheel kept as that source distribution under the file name ``held``."""

    def plant(entry: Path, planted: Planted) -> None:
        _lock_wheel_recorded_as_built_from_a_copy_of_itself(entry, held)
        shutil.copyfile(planted.lock_wheel, entry / WHEELS_DIR / _LOCK_WHEEL)
        _record_wheel(entry, "lock-pkg", _LOCK_WHEEL)

    return plant


def _does_not_build(held: str) -> str:
    return f"{SDISTS_DIR}/{held} does not build {_LOCK_WHEEL} again: building a wheel from {held} failed"


def _sdist_replaced_with_the_wheel_it_builds(entry: Path, planted: Planted) -> None:
    """Another source distribution where the entry keeps ``sdist_pkg``'s, and the wheel it builds in place of the cached
    one; the manifest keeps the lock's hash of the real source distribution, as a sandbox requires."""
    shutil.copyfile(planted.sdist, entry / SDISTS_DIR / _SDIST)
    shutil.copyfile(planted.sdist_wheel, entry / WHEELS_DIR / _SDIST_WHEEL)
    _record_wheel(entry, "sdist-pkg", _SDIST_WHEEL)


def _sdist_damaged(entry: Path, planted: Planted) -> None:
    sdist = entry / SDISTS_DIR / _SDIST
    damaged = bytearray(sdist.read_bytes())
    damaged[len(damaged) // 2] ^= 0xFF
    sdist.write_bytes(bytes(damaged))


def _built_wheel_replaced(entry: Path, planted: Planted) -> None:
    """The wheel another source distribution builds, in place of the cached one; the real source distribution stays."""
    shutil.copyfile(planted.sdist_wheel, entry / WHEELS_DIR / _SDIST_WHEEL)
    _record_wheel(entry, "sdist-pkg", _SDIST_WHEEL)


def _built_wheel_renamed(entry: Path, planted: Planted) -> None:
    """The built wheel's own bytes under another file name, which tells pip other tags."""
    (entry / WHEELS_DIR / _SDIST_WHEEL).rename(entry / WHEELS_DIR / _RENAMED)
    _record_wheel(entry, "sdist-pkg", _RENAMED, file=_RENAMED)


def _built_wheel_not_a_zip(entry: Path, planted: Planted) -> None:
    (entry / WHEELS_DIR / _SDIST_WHEEL).write_bytes(b"not a zip archive")
    _record_wheel(entry, "sdist-pkg", _SDIST_WHEEL)


@pytest.mark.parametrize(
    ("plant", "difference"),
    [
        (
            _lock_wheel_claiming_a_sdist_it_does_not_hold,
            f"{_LOCK_WHEEL} for lock-pkg==1.0 is recorded as built from a source distribution that the entry does "
            f"not hold in {SDISTS_DIR}",
        ),
        # Under a source distribution's name pip finds no project in it.
        (_lock_wheel_with_the_real_one_kept_as_its_sdist(_TAR_GZ), _does_not_build(_TAR_GZ)),
        (_lock_wheel_with_the_real_one_kept_as_its_sdist(_ZIP), _does_not_build(_ZIP)),
        # Under its own name pip copies it, and its members are not the planted wheel's.
        (
            _lock_wheel_with_the_real_one_kept_as_its_sdist(_LOCK_WHEEL),
            f"{_LOCK_WHEEL} is not the wheel its source distribution builds; the members that differ: "
            "lock_pkg-1.0.dist-info/RECORD, lock_pkg.py",
        ),
        (_sdist_replaced_with_the_wheel_it_builds, _NOT_IN_THE_LOCK),
        (_sdist_damaged, f"{SDISTS_DIR}/{_SDIST} is not the source distribution of {_SDIST_WHEEL}"),
        (
            _built_wheel_replaced,
            f"{_SDIST_WHEEL} is not the wheel its source distribution builds; the members that differ: sdist_pkg.py",
        ),
        (_built_wheel_renamed, f"{SDISTS_DIR}/{_SDIST} builds {_SDIST_WHEEL}, not {_RENAMED}"),
        (_built_wheel_not_a_zip, f"{_SDIST_WHEEL} cannot be read as a wheel: BadZipFile"),
    ],
    ids=[
        "no-sdist-held",
        "wheel-as-tar-gz",
        "wheel-as-zip",
        "wheel-as-whl",
        "sdist-replaced",
        "sdist-damaged",
        "wheel-replaced",
        "wheel-renamed",
        "wheel-not-a-zip",
    ],
)
def test_a_wheel_the_source_distribution_the_lock_lists_does_not_build_again_is_refused_on_reuse(
    rig: Rig, planted: Planted, tmp_path: Path, plant: Callable[[Path, Planted], None], difference: str
) -> None:
    cache = copy_cache(rig, tmp_path / "cache")
    entry = cache.root / cache.sha
    plant(entry, planted)
    assert check_entry(entry, cache.sha, verify_hash=True) == []
    # A sandbox handed this manifest's hash stages it, the planted wheel included.
    pin = _digest(entry / MANIFEST_NAME)
    assert stage_entry(cache.root, cache.sha, tmp_path / "scratch", pin) == []
    before = (tree_digest(entry), pin)
    with pytest.raises(CacheConflict, match="is left alone") as refused:
        ensure_runner_cache(rig.repo, cache.sha, cache.root, _NO_WHEELS)
    assert difference in str(refused.value)
    assert (tree_digest(entry), _digest(entry / MANIFEST_NAME)) == before
    assert {path.name for path in cache.root.iterdir()} == {cache.sha, f".{cache.sha}.lock"}


def test_an_entry_with_a_wheel_built_from_a_source_distribution_is_reused_after_building_it_again_offline(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rebuilt_from: list[Path]
) -> None:
    cache = copy_cache(rig, tmp_path / "cache")
    entry = cache.root / cache.sha
    built = {wheel.name: wheel for wheel in manifest_from_json((entry / MANIFEST_NAME).read_bytes()).wheels}
    assert (built["sdist-pkg"].sdist_file, built["sdist-pkg"].sdist_sha256) == (_SDIST, rig.sdist_pkg_sha256)
    assert _digest(entry / SDISTS_DIR / _SDIST) == rig.sdist_pkg_sha256
    assert (built["lock-pkg"].sdist_file, built["lock-pkg"].sdist_sha256) == ("", "")
    before = tree_digest(entry)
    with tripwire() as wire:
        for name, value in wire.env.items():
            monkeypatch.setenv(name, value)
        reused = ensure_runner_cache(rig.repo, cache.sha, cache.root, _NO_WHEELS)
    assert (reused.reused, reused.manifest_sha256) == (True, _digest(entry / MANIFEST_NAME))
    assert wire.connections == 0
    # The source distribution was built again from a copy outside the cache, and the entry is as it was.
    assert [path.name for path in rebuilt_from] == [_SDIST]
    assert not rebuilt_from[0].is_relative_to(cache.root)
    assert tree_digest(entry) == before
    assert {path.name for path in cache.root.iterdir()} == {cache.sha, f".{cache.sha}.lock"}


def test_a_wheel_the_lock_lists_recorded_as_built_from_a_whl_copy_of_itself_is_reused_as_that_wheel(
    rig: Rig, tmp_path: Path, rebuilt_from: list[Path]
) -> None:
    """pip copies a ``.whl`` it is asked to build, so the comparison is what decides; the wheel it passes installs what
    the wheel the lock lists installs."""
    cache = copy_cache(rig, tmp_path / "cache")
    entry = cache.root / cache.sha
    _lock_wheel_recorded_as_built_from_a_copy_of_itself(entry, _LOCK_WHEEL)
    pin = _digest(entry / MANIFEST_NAME)
    reused = ensure_runner_cache(rig.repo, cache.sha, cache.root, _NO_WHEELS)
    assert (reused.reused, reused.manifest_sha256) == (True, pin)
    assert sorted(path.name for path in rebuilt_from) == [_LOCK_WHEEL, _SDIST]
    assert _digest(entry / WHEELS_DIR / _LOCK_WHEEL) == rig.lock_pkg_sha256


def _sdist_recorded_with_its_own_hash(entry: Path, planted: Planted) -> None:
    """Another source distribution and the wheel it builds, recorded with that source distribution's own hash, which the
    commit's lock does not list: the entry check passes it only while it reads another lock in ``source/``."""
    shutil.copyfile(planted.sdist, entry / SDISTS_DIR / _SDIST)
    shutil.copyfile(planted.sdist_wheel, entry / WHEELS_DIR / _SDIST_WHEEL)
    _record_wheel(entry, "sdist-pkg", _SDIST_WHEEL, sdist_sha256=_digest(planted.sdist))


def _sdist_listed_by_the_lock_in_the_entrys_source(entry: Path, planted: Planted) -> None:
    """The same, with the lock in the entry's ``source/`` listing that hash in place of the real source distribution's:
    only the lock of the archive the builder makes again does not list it."""
    lock, real = entry / SOURCE_DIR / LOCK_FILE, _digest(entry / SDISTS_DIR / _SDIST)
    assert real in lock.read_text()
    swapped = lock.read_text().replace(real, _digest(planted.sdist))
    _sdist_recorded_with_its_own_hash(entry, planted)
    lock.write_text(swapped)


def _wheel_recorded_while_another_is_held(entry: Path, planted: Planted) -> None:
    """The wheel another source distribution builds, recorded in the manifest, while the cache holds the built wheel
    itself: the entry check passes it only while it reads the planted wheel."""
    held = entry / WHEELS_DIR / _SDIST_WHEEL
    built = held.read_bytes()
    shutil.copyfile(planted.sdist_wheel, held)
    _record_wheel(entry, "sdist-pkg", _SDIST_WHEEL)
    held.write_bytes(built)


@pytest.mark.parametrize(
    ("plant", "difference"),
    [
        (_sdist_recorded_with_its_own_hash, _NOT_IN_THE_LOCK),
        (_sdist_listed_by_the_lock_in_the_entrys_source, _NOT_IN_THE_LOCK),
        (_wheel_recorded_while_another_is_held, f"wheel {_SDIST_WHEEL} for sdist-pkg==1.0 does not match the hash"),
    ],
    ids=["sdist-not-in-the-commits-lock", "sdist-in-the-entrys-lock", "wheel-not-the-recorded-one"],
)
def test_a_reuse_builds_only_what_the_commits_lock_lists_and_compares_only_the_wheel_the_manifest_records(
    rig: Rig,
    planted: Planted,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rebuilt_from: list[Path],
    plant: Callable[[Path, Planted], None],
    difference: str,
) -> None:
    """The entry check reads the lock in ``source/`` and the wheels as they lie in the cache, which can change while it
    reads them; here it passes, as a check fooled that way would. The source distribution is checked against the lock
    of the commit the builder made again, not the one in ``source/``, before any of its code runs, and the wheel
    compared is the builder's own copy, checked against the manifest's hash."""
    cache = copy_cache(rig, tmp_path / "cache")
    entry = cache.root / cache.sha
    plant(entry, planted)
    assert check_entry(entry, cache.sha, verify_hash=True) != []
    monkeypatch.setattr(cache_build, "check_entry", lambda *args, **kwargs: [])
    with pytest.raises(CacheConflict, match="is left alone") as refused:
        ensure_runner_cache(rig.repo, cache.sha, cache.root, _NO_WHEELS)
    assert difference in str(refused.value)
    assert rebuilt_from == []
