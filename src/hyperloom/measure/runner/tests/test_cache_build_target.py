# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner cache is built for the runner's interpreter, named by the host, not for the builder's: the wheels of
that interpreter are downloaded whatever the builder runs, nothing defaults to the builder, and an entry whose wheels
the runner could not install is refused before it is renamed into place.

The binary wheels are crafted into a local wheelhouse (``--no-index --find-links``); nothing needs a package index.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.measure.runner import cache_build
from hyperloom.measure.runner.cache_build import (
    RUNNER_TARGET_ENV,
    BuildOptions,
    CacheBuildError,
    WheelSource,
    ensure_runner_cache,
    runner_wheel_target,
)
from hyperloom.measure.runner.cache_layout import (
    LOCK_FILE,
    MANIFEST_NAME,
    LockPin,
    manifest_from_json,
    requirement_line,
)
from hyperloom.measure.runner.tests.bootstrap_rig import Commit, Rig, build_wheel, commit_edit

#: The interpreters the wheelhouse has a binary wheel of ``bin_pkg`` for; none of them is ``any``.
_INTERPRETERS = ("cp310", "cp311", "cp312", "cp313")
_PLATFORM = "manylinux_2_17_x86_64"
_BIN = "bin_pkg"


def _binary_wheelhouse(rig: Rig, root: Path, interpreters: tuple[str, ...]) -> tuple[Path, dict[str, str]]:
    """A copy of the rig's wheelhouse with one ``bin_pkg`` 1.0 wheel per interpreter; each wheel's hash by interpreter."""
    wheelhouse = root / "wheelhouse"
    shutil.copytree(rig.wheelhouse, wheelhouse)
    digests = {}
    for interpreter in interpreters:
        tag = f"{interpreter}-{interpreter}-{_PLATFORM}"
        _, digests[interpreter] = build_wheel(wheelhouse, _BIN, "1.0", f"ORIGIN = {interpreter!r}\n", tag=tag)
    return wheelhouse, digests


def _commit_pinning(rig: Rig, root: Path, digests: list[str]) -> Commit:
    """A commit whose lock also pins ``bin_pkg==1.0`` with the hashes ``digests``."""

    def pin_binary(repo: Path) -> None:
        with (repo / LOCK_FILE).open("a", encoding="utf-8") as lock:
            lock.write(requirement_line(_BIN, "1.0", sorted(digests)))

    work = root / "work"
    work.mkdir()
    return commit_edit(rig, work, pin_binary)


def _options(tag: str, wheelhouse: Path) -> BuildOptions:
    return BuildOptions(
        runner_wheel_target({RUNNER_TARGET_ENV: tag}), WheelSource(no_index=True, find_links=(str(wheelhouse),))
    )


@pytest.mark.parametrize("interpreter", ["cp310", "cp313"])
def test_the_wheels_are_those_of_the_named_interpreter_whatever_the_builder_runs(
    rig: Rig, tmp_path: Path, interpreter: str
) -> None:
    wheelhouse, digests = _binary_wheelhouse(rig, tmp_path, _INTERPRETERS)
    commit = _commit_pinning(rig, tmp_path, list(digests.values()))

    entry = ensure_runner_cache(
        commit.repo, commit.sha, tmp_path / "cache", _options(f"{interpreter}-manylinux_2_28_x86_64", wheelhouse)
    )

    manifest = manifest_from_json((entry.path / MANIFEST_NAME).read_bytes())
    binary = [wheel for wheel in manifest.wheels if wheel.name == "bin-pkg"]
    assert [wheel.file for wheel in binary] == [f"bin_pkg-1.0-{interpreter}-{interpreter}-{_PLATFORM}.whl"]
    assert binary[0].sha256 == digests[interpreter]
    assert manifest.target["python_version"] == f"3.{interpreter[3:]}"
    assert f"cp{interpreter[2:]}" in manifest.target["abis"]


def test_an_entry_whose_wheels_the_runner_cannot_install_is_refused_before_it_is_in_place(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wheelhouse, digests = _binary_wheelhouse(rig, tmp_path, (_BUILDER,))
    commit = _commit_pinning(rig, tmp_path, list(digests.values()))
    monkeypatch.setattr(cache_build, "_download_wheels", _downloaded_for(f"{_BUILDER}-manylinux_2_28_x86_64"))
    cache = tmp_path / "cache"
    runner = "cp310" if _BUILDER != "cp310" else "cp311"

    with pytest.raises(CacheBuildError, match=rf"checking that the runner \(Python 3\.{runner[3:]}\) can") as refused:
        ensure_runner_cache(commit.repo, commit.sha, cache, _options(f"{runner}-manylinux_2_28_x86_64", wheelhouse))

    assert "bin-pkg" in str(refused.value).lower().replace("_", "-")
    assert sorted(path.name for path in cache.iterdir()) == [f".{commit.sha}.lock"]


#: The interpreter running the test: its wheels are what a builder that ignored the target would have fetched, and
#: what pip would accept were the check made for the builder instead of the runner.
_BUILDER = f"cp3{sys.version_info.minor}"
_REAL_DOWNLOAD = cache_build._download_wheels


def _downloaded_for(tag: str) -> Callable[..., dict[str, str]]:
    """The real download, but of the wheels of ``tag``: what a builder that used its own interpreter fetched."""

    def download(lock: Path, wheels: Path, pins: list[LockPin], options: BuildOptions) -> dict[str, str]:
        return _REAL_DOWNLOAD(
            lock, wheels, pins, replace(options, target=runner_wheel_target({RUNNER_TARGET_ENV: tag}))
        )

    return download


@pytest.mark.parametrize("value", ["", "3.10", "cp310", "py3-none-any", "cp310-manylinux2014_x86_64"])
def test_a_host_that_does_not_name_the_runners_interpreter_builds_nothing(value: str) -> None:
    environ = {RUNNER_TARGET_ENV: value} if value else {}
    with pytest.raises(CacheBuildError, match=f"\\${RUNNER_TARGET_ENV} must name the runner's interpreter"):
        runner_wheel_target(environ)


def test_the_target_names_every_platform_tag_the_runners_glibc_accepts() -> None:
    target = runner_wheel_target({RUNNER_TARGET_ENV: "cp310-manylinux_2_28_x86_64"})

    assert (target.python_version, target.implementation, target.abis) == ("3.10", "cp", ("cp310", "abi3", "none"))
    assert target.platforms[0] == "manylinux_2_28_x86_64"
    for accepted in ("manylinux_2_17_x86_64", "manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64"):
        assert accepted in target.platforms
    assert "manylinux_2_29_x86_64" not in target.platforms
    assert target.platforms[-2:] == ("linux_x86_64", "any")
