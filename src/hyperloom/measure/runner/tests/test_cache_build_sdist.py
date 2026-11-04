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
import stat
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from hyperloom.measure.runner import cache_build
from hyperloom.measure.runner.cache_build import (
    BuildOptions,
    CacheConflict,
    WheelSource,
    WheelTarget,
    ensure_runner_cache,
)
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
    rewrite_wheel,
    tripwire,
    unicode_path_name,
)

#: Options a build cannot succeed with (no index and no wheels): only a reuse of the entry can answer.
_NO_WHEELS = BuildOptions(source=WheelSource(no_index=True))
_LOCK_WHEEL = "lock_pkg-1.0-py3-none-any.whl"
_SDIST = "sdist_pkg-1.0.tar.gz"
_SDIST_WHEEL = "sdist_pkg-1.0-py3-none-any.whl"
_RENAMED = "sdist_pkg-1.0-py2.py3-none-any.whl"
#: The real ``lock_pkg`` wheel's names where the entry keeps a source distribution: a source distribution's two.
_TAR_GZ, _ZIP = "lock_pkg-1.0.tar.gz", "lock_pkg-1.0.zip"
#: A Python that has a lock of its own in a commit of the last test, and that lock.
_CP314, _CP314_LOCK = WheelTarget(python_version="3.14"), "requirements/runner-cp314.lock"
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


def _rewrite_built_wheel(entry: Path, edit: Callable[[zipfile.ZipInfo], zipfile.ZipInfo] | None = None) -> None:
    """The built wheel's members written again (:func:`rewrite_wheel`) in place of the cached wheel, and recorded."""
    cached = entry / WHEELS_DIR / _SDIST_WHEEL
    rewrite_wheel(Path(shutil.copyfile(cached, cached.with_suffix(".orig"))), cached, edit)
    cached.with_suffix(".orig").unlink()
    _record_wheel(entry, "sdist-pkg", _SDIST_WHEEL)


def _built_module_stored_as_sitecustomize(entry: Path, planted: Planted) -> None:
    """The built wheel's ``sdist_pkg.py`` stored under the raw name ``sitecustomize.py``, with a Unicode Path extra field
    naming it ``sdist_pkg.py``: pip installs ``sitecustomize.py`` where its zipfile reads the raw name (before Python
    3.12), while a builder whose zipfile reads the field (3.12 and newer) sees the members of the rebuilt wheel."""
    _rewrite_built_wheel(entry, unicode_path_name("sdist_pkg.py", "sitecustomize.py"))


def _built_module_made_executable(entry: Path, planted: Planted) -> None:
    def executable(member: zipfile.ZipInfo) -> zipfile.ZipInfo:
        if member.filename == "sdist_pkg.py":
            member.external_attr = (stat.S_IFREG | 0o755) << 16
        return member

    _rewrite_built_wheel(entry, executable)


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
        (
            _built_module_stored_as_sitecustomize,
            f"{_SDIST_WHEEL} cannot be read as a wheel: BadZipFile: its member 'sitecustomize.py' has a Unicode Path "
            "extra field",
        ),
        (
            _built_module_made_executable,
            f"{_SDIST_WHEEL} is not the wheel its source distribution builds; the members that differ: sdist_pkg.py",
        ),
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
        "member-under-a-unicode-path-name",
        "member-made-executable",
    ],
)
def test_a_wheel_the_source_distribution_the_lock_lists_does_not_build_again_is_refused_on_reuse(
    rig: Rig,
    planted: Planted,
    tmp_path: Path,
    rebuilt_from: list[Path],
    plant: Callable[[Path, Planted], None],
    difference: str,
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
    if "cannot be read as a wheel" in difference:
        # A cached wheel that is not read as pip reads it is refused before its source distribution is built.
        assert rebuilt_from == []


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


def test_a_built_wheel_written_again_with_other_timestamps_and_member_order_is_reused(
    rig: Rig, tmp_path: Path, rebuilt_from: list[Path]
) -> None:
    """The comparison reads each member's raw name, content and executable bit, not how the zip is laid out."""
    cache = copy_cache(rig, tmp_path / "cache")
    entry = cache.root / cache.sha
    before = (entry / WHEELS_DIR / _SDIST_WHEEL).read_bytes()
    _rewrite_built_wheel(entry)
    assert (entry / WHEELS_DIR / _SDIST_WHEEL).read_bytes() != before
    reused = ensure_runner_cache(rig.repo, cache.sha, cache.root, _NO_WHEELS)
    assert (reused.reused, reused.manifest_sha256) == (True, _digest(entry / MANIFEST_NAME))
    assert [path.name for path in rebuilt_from] == [_SDIST]


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


def test_a_reuse_checks_the_built_wheels_against_the_lock_of_the_entrys_python_not_the_shared_one(
    rig: Rig,
    planted: Planted,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rebuilt_from: list[Path],
) -> None:
    """A commit whose shared lock lists another ``sdist_pkg`` source distribution than the lock of Python 3.14: an
    honest entry for 3.14 is reused, and one recorded with the source distribution only the shared lock lists is
    refused, with the entry check made to pass."""

    def two_locks(repo: Path) -> None:
        shared = repo / LOCK_FILE
        own = shared.read_text()
        assert rig.sdist_pkg_sha256 in own
        (repo / _CP314_LOCK).write_text(own)
        shared.write_text(own.replace(rig.sdist_pkg_sha256, _digest(planted.sdist)))

    (tmp_path / "work").mkdir()
    commit = commit_edit(rig, tmp_path / "work", two_locks)
    cache, offline = tmp_path / "cache", replace(_NO_WHEELS, target=_CP314)
    built = ensure_runner_cache(commit.repo, commit.sha, cache, replace(rig.build_options(), target=_CP314))
    reused = ensure_runner_cache(commit.repo, commit.sha, cache, offline)
    assert (reused.reused, reused.manifest_sha256) == (True, built.manifest_sha256)

    _sdist_recorded_with_its_own_hash(built.path, planted)
    monkeypatch.setattr(cache_build, "check_entry", lambda *args, **kwargs: [])
    with pytest.raises(CacheConflict, match="is left alone") as refused:
        ensure_runner_cache(commit.repo, commit.sha, cache, offline)
    assert _NOT_IN_THE_LOCK in str(refused.value)
    # Built once by the build and once by the honest reuse; the planted one never.
    assert [path.name for path in rebuilt_from] == [_SDIST, _SDIST]


def test_a_source_distribution_is_built_again_only_when_it_is_the_one_the_manifest_records(
    rig: Rig, planted: Planted, tmp_path: Path, rebuilt_from: list[Path]
) -> None:
    """A commit whose lock lists two source distributions of ``sdist_pkg`` 1.0, the real one and another: an entry that
    records the real one but holds the other, with the wheel the other builds, passes the entry check, and the builder
    refuses it before building the held file, although the lock lists that file too."""

    def two_sdists(repo: Path) -> None:
        lock, real = repo / LOCK_FILE, f"--hash=sha256:{rig.sdist_pkg_sha256}\n"
        assert lock.read_text().count(real) == 1
        lock.write_text(lock.read_text().replace(real, f"{real[:-1]} \\\n    --hash=sha256:{_digest(planted.sdist)}\n"))

    (tmp_path / "work").mkdir()
    commit = commit_edit(rig, tmp_path / "work", two_sdists)
    entry = ensure_runner_cache(commit.repo, commit.sha, tmp_path / "cache", rig.build_options()).path
    _sdist_replaced_with_the_wheel_it_builds(entry, planted)
    assert check_entry(entry, commit.sha, verify_hash=True) == []
    before = (tree_digest(entry), _digest(entry / MANIFEST_NAME))
    with pytest.raises(CacheConflict, match="is left alone") as refused:
        ensure_runner_cache(commit.repo, commit.sha, tmp_path / "cache", _NO_WHEELS)
    assert f"{_NOT_IN_THE_LOCK}: its SHA-256 is {_digest(planted.sdist)}" in str(refused.value)
    assert (tree_digest(entry), _digest(entry / MANIFEST_NAME)) == before
    # Built once, by the build, from the source distribution it downloaded; the held one never.
    assert [path.parent.name for path in rebuilt_from] == [WHEELS_DIR]


#: A ``lock_pkg`` 1.0 wheel with a build tag, which pip prefers to the real wheel's file when both are offered.
_BUILD_TAGGED = "lock_pkg-1.0-1-py3-none-any.whl"


def _second_lock_pkg_entry(behind: bool) -> Callable[[Path, Planted], None]:
    """Other ``lock_pkg`` code as ``wheels/<_BUILD_TAGGED>``, and a second ``lock-pkg==1.0`` entry recording it put in
    front of the real one or behind it. Behind it, the entry also claims to be built from the real wheel, whose hash
    the lock lists, as it must to pass a check that reads only the last entry of a pin."""

    def plant(entry: Path, planted: Planted) -> None:
        held = Path(shutil.copyfile(planted.lock_wheel, entry / WHEELS_DIR / _BUILD_TAGGED))
        manifest = manifest_from_json((entry / MANIFEST_NAME).read_bytes())
        wheels = list(manifest.wheels)
        real = next(index for index, wheel in enumerate(wheels) if wheel.name == "lock-pkg")
        claim = {"sdist_sha256": wheels[real].sha256} if behind else {}
        second = replace(wheels[real], file=_BUILD_TAGGED, size=held.stat().st_size, sha256=_digest(held), **claim)
        wheels.insert(real + behind, second)
        (entry / MANIFEST_NAME).write_text(manifest_to_json(replace(manifest, wheels=tuple(wheels))))

    return plant


@pytest.mark.parametrize("behind", [False, True], ids=["in-front-of-the-real-entry", "behind-the-real-entry"])
def test_a_manifest_that_lists_a_pin_twice_is_refused_by_the_sandbox_and_the_builder(
    rig: Rig, planted: Planted, tmp_path: Path, behind: bool
) -> None:
    """The sandbox stages every wheel the manifest lists and pip, given two hashes for one pin, takes either file; so
    one pin listed twice is refused by the check the builder and the sandbox share, and only for that."""
    cache = copy_cache(rig, tmp_path / "cache")
    entry, scratch = cache.root / cache.sha, tmp_path / "scratch"
    _second_lock_pkg_entry(behind)(entry, planted)
    twice = ["the manifest lists lock-pkg==1.0 2 times, and a pin has one wheel"]
    assert check_entry(entry, cache.sha, verify_hash=True) == twice
    pin = _digest(entry / MANIFEST_NAME)
    assert stage_entry(cache.root, cache.sha, scratch, pin) == twice
    assert not scratch.exists() or list(scratch.iterdir()) == []
    before = (tree_digest(entry), pin)
    with pytest.raises(CacheConflict, match="is left alone") as refused:
        ensure_runner_cache(rig.repo, cache.sha, cache.root, _NO_WHEELS)
    assert twice[0] in str(refused.value)
    assert (tree_digest(entry), _digest(entry / MANIFEST_NAME)) == before
