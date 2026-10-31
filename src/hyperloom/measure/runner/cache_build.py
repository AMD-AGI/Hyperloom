# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Builds the runner cache of one commit (layout and meaning: :mod:`hyperloom.measure.runner.cache_layout`).

It runs on a machine with egress, in the application that starts a campaign, before the first measurement request::

    python -m hyperloom.measure.runner.cache_build --sha <commit> --repo <checkout> [--out <cache>] \
        [--wheel-target cp314-manylinux_2_28_x86_64]

``--out`` defaults to ``$HYPERLOOM_RUNNER_CACHE``; ``--wheel-target`` takes the tag ``$HYPERLOOM_RUNNER_WHEEL_TARGET``
does (:func:`runner_wheel_target`) and stands for the separate options below. A build that fails fails the campaign: nothing falls back to
installing over the network inside a sandbox. The wheels are downloaded for the interpreter and platform of the
*framework image*, which is not the builder's own unless it is told: ``--python-version 3.10 --implementation cp
--abi cp310 --abi abi3 --abi none`` and one ``--platform`` for every platform tag the image accepts
(``manylinux_2_28_x86_64``, ``manylinux_2_17_x86_64``, ``manylinux2014_x86_64`` and so on down to ``manylinux1_x86_64``)
for a CPython 3.10 image on glibc 2.28, and ``--index-url`` (or ``--find-links`` with ``--no-index``) for where they come
from. pip does not widen a ``--platform`` to the older tags it covers, so naming only the newest one turns every pin
published as ``manylinux2014`` into a source build. The lock's hashes are those of the files the index serves. A pin that the index
serves only as a source distribution is built here into a wheel (the sdist's hash must be one the lock lists, and the
manifest records it); that wheel must be pure Python, because a platform wheel built on the builder is not a wheel for
the image, and the build is refused otherwise. Reading or building a source distribution runs its code on the builder, so
that happens only in a throwaway virtual environment, without build isolation, whose only build tools are the
``setuptools`` (and ``wheel``) that the lock pins, installed from wheels downloaded by their lock hashes: what a source
distribution's own build requirements name is never fetched. The lock must therefore pin ``setuptools``
(``pip-compile --allow-unsafe`` does). The builder's pip must be 22.3 or newer (``pip --python``).

- A commit is built once. The entry is made in a directory of its own beside the final one, its manifest is written last
  and the directory is renamed into place in one step, so ``<cache>/<sha>`` is either absent or complete.
- Builders of one commit exclude each other with an exclusive ``flock`` on ``<cache>/.<sha>.lock``; the one that waits
  finds the finished entry when it gets the lock and uses it, with no download. The lock is also what keeps two hosts that
  share the cache directory over NFS apart (the cache's file system must support ``flock``, as NFSv4 does).
- An entry that is already there is used when it is complete and still what a build would write (same commit tree, same
  wheel target, source and wheels intact); one that differs is never overwritten, the build refuses and names the
  difference. An entry with no manifest is the remains of a build that did not finish; it is removed and built again.
- ``<entry>/bootstrap.sh`` is this builder's start-up script (:data:`~hyperloom.measure.runner.bootstrap.BOOTSTRAP_SCRIPT`,
  mode 0755), not the commit's, and the manifest records its SHA-256. An entry that is otherwise what a build would write
  but whose manifest records another script (a builder of another version made it) or none (it predates the script)
  gets this builder's script and a manifest that records it, each replaced in one rename, under the lock; the result's
  ``manifest_sha256`` is that new manifest's. A ``bootstrap.sh`` that does not match the hash its manifest records is
  a difference like any other, and the build refuses.

The cache must stay readable by the account the sandbox runs as: run the builder with a ``umask`` that leaves new files
readable by others (``022``).

Before the entry is renamed into place, the builder asks pip whether the requirements the sandbox installs
(``runner-wheels.lock``) resolve for the wheel target from the entry's wheels alone (``pip install --dry-run --no-index
--only-binary=:all:``, the sandbox's own flags), so a cache the runner could not install is refused here instead of
failing in the sandbox. A hosted session names the runner's interpreter in ``$HYPERLOOM_RUNNER_WHEEL_TARGET``
(:func:`runner_wheel_target`); it never builds for its own.

The pins come from the lock of the target's Python (:func:`~hyperloom.measure.runner.cache_layout.lock_file`):
``requirements/runner-cp314.lock`` for ``--python-version 3.14``, ``requirements/runner.lock`` for a version without a
lock of its own or when no ``--python-version`` is given. A build for a Python that has its own lock must therefore name
it (``--python-version``, or the ``cpXY`` of ``$HYPERLOOM_RUNNER_WHEEL_TARGET``).
"""

from __future__ import annotations

import argparse
import re
import fcntl
import hashlib
import importlib.metadata
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from hyperloom.measure.runner.bootstrap import BOOTSTRAP_SCRIPT
from hyperloom.measure.runner.cache_layout import (
    BOOTSTRAP_NAME,
    CACHE_ENV,
    MANIFEST_NAME,
    SOURCE_DIR,
    WHEELS_DIR,
    CacheError,
    LockPin,
    Manifest,
    WheelEntry,
    check_entry,
    file_sha256,
    is_full_commit,
    lock_file,
    manifest_from_json,
    manifest_to_json,
    parse_lock,
    requirement_line,
    requirements_text,
    tree_digest,
)

__all__ = [
    "BuildOptions",
    "RUNNER_TARGET_ENV",
    "CacheBuildError",
    "CacheConflict",
    "CacheEntry",
    "WheelSource",
    "WheelTarget",
    "ensure_runner_cache",
    "exclusive",
    "main",
    "runner_wheel_target",
]

DEFAULT_LOCK_TIMEOUT_SEC = 1800.0
_LOCK_POLL_SEC = 0.1
_TAIL = 2000
_UTF8 = "utf-8"
#: ``bootstrap.sh`` is run with ``sh`` and may be run directly: readable and executable by the sandbox's account.
_SCRIPT_MODE = 0o755
_BOOTSTRAP_SHA256 = hashlib.sha256(BOOTSTRAP_SCRIPT.encode(_UTF8)).hexdigest()
_BUILD_TOOLS = ("setuptools", "wheel")
_NO_INDEX, _FIND_LINKS, _NO_DEPS = "--no-index", "--find-links", "--no-deps"
_ISOLATED, _ONLY_BINARY, _REQUIRE_HASHES = "--isolated", "--only-binary=:all:", "--require-hashes"
#: pip never asks a question and never checks for a newer pip.
_PIP_QUIET = ("--disable-pip-version-check", "--no-input")
#: The runner's interpreter and newest platform tag as one wheel tag, e.g. ``cp310-manylinux_2_28_x86_64``.
RUNNER_TARGET_ENV = "HYPERLOOM_RUNNER_WHEEL_TARGET"
_TARGET_TAG = re.compile(r"cp(?P<major>3)(?P<minor>[0-9]+)-manylinux_2_(?P<glibc>[0-9]+)_(?P<arch>x86_64|aarch64)")
#: The legacy manylinux names of a glibc 2 minor version; pip accepts both spellings and downloads only what it is told.
_LEGACY_MANYLINUX = {17: "manylinux2014", 12: "manylinux2010", 5: "manylinux1"}
_OLDEST_GLIBC_MINOR = 5
#: The first pip whose ``install --python-version`` checks ``Requires-Python`` against that version; older ones check it
#: against the interpreter pip runs on, the builder's, which says nothing about the runner's.
_PIP_CHECKS_TARGET_PYTHON = (24, 2)


class CacheBuildError(CacheError):
    """The cache cannot be built; the message says why. A campaign that gets it does not start."""


class CacheConflict(CacheBuildError):
    """An entry of the commit is there and is not what a build would write; it is left as it was."""


@dataclass(frozen=True)
class WheelTarget:
    """The interpreter and platform the wheels are for (``pip download``'s own; none given is the builder's)."""

    python_version: str | None = None
    implementation: str | None = None
    abis: tuple[str, ...] = ()
    platforms: tuple[str, ...] = ()

    def manifest_form(self) -> dict[str, Any]:
        """What the manifest records of it."""
        return {
            "python_version": self.python_version,
            "implementation": self.implementation,
            "abis": list(self.abis),
            "platforms": list(self.platforms),
        }

    def pip_args(self) -> list[str]:
        args = ["--python-version", self.python_version] if self.python_version else []
        args += ["--implementation", self.implementation] if self.implementation else []
        for flag, values in (("--abi", self.abis), ("--platform", self.platforms)):
            args += [part for value in values for part in (flag, value)]
        return args


def runner_wheel_target(environ: Mapping[str, str]) -> WheelTarget:
    """The wheel target ``$HYPERLOOM_RUNNER_WHEEL_TARGET`` names: CPython ``X.Y`` (ABIs ``cpXY``, ``abi3``, ``none``) on
    glibc ``2.N`` of one architecture, with every platform tag that glibc accepts, newest first.

    Raises:
        CacheBuildError: When the variable is unset or not of the form ``cp310-manylinux_2_28_x86_64``: the builder's
            own interpreter is not the runner's, so nothing defaults to it.
    """
    value = environ.get(RUNNER_TARGET_ENV, "").strip()
    match = _TARGET_TAG.fullmatch(value)
    if match is None:
        raise CacheBuildError(
            f"${RUNNER_TARGET_ENV} must name the runner's interpreter as cpXY-manylinux_2_N_ARCH, "
            f"e.g. cp310-manylinux_2_28_x86_64 (got {value[:80]!r})"
        )
    major, minor, arch = match["major"], match["minor"], match["arch"]
    platforms: list[str] = []
    for glibc in range(int(match["glibc"]), _OLDEST_GLIBC_MINOR - 1, -1):
        platforms.append(f"manylinux_2_{glibc}_{arch}")
        if glibc in _LEGACY_MANYLINUX:
            platforms.append(f"{_LEGACY_MANYLINUX[glibc]}_{arch}")
    platforms += [f"linux_{arch}", "any"]
    return WheelTarget(f"{major}.{minor}", "cp", (f"cp{major}{minor}", "abi3", "none"), tuple(platforms))


@dataclass(frozen=True)
class WheelSource:
    """Where ``pip download`` looks: an index, local directories (with ``no_index``), or what pip is configured with."""

    index_url: str | None = None
    find_links: tuple[str, ...] = ()
    no_index: bool = False

    def pip_args(self) -> list[str]:
        args = [_NO_INDEX] if self.no_index else []
        args += ["--index-url", self.index_url] if self.index_url else []
        return args + [part for link in self.find_links for part in (_FIND_LINKS, link)]


@dataclass(frozen=True)
class BuildOptions:
    """How to build: what the wheels are for, where they come from and how long to wait for another builder."""

    target: WheelTarget = WheelTarget()
    source: WheelSource = WheelSource()
    lock_timeout_sec: float = DEFAULT_LOCK_TIMEOUT_SEC


@dataclass(frozen=True)
class CacheEntry:
    """A finished entry: its directory, the SHA-256 of its manifest file (what a host may pin) and whether this call
    built it (``reused`` is true when it was there already)."""

    path: Path
    manifest_sha256: str
    reused: bool


def _run(command: Sequence[str], what: str) -> str:
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise CacheBuildError(f"{what} failed (exit {done.returncode}): {(done.stdout + done.stderr).strip()[-_TAIL:]}")
    return done.stdout.strip()


def _git(repo: Path, *args: str) -> list[str]:
    return ["git", "-C", str(repo), *args]


def _commit_tree(repo: Path, sha: str) -> str:
    """The tree object of commit ``sha`` in ``repo``, which also proves the repository has that very commit."""
    if _run(_git(repo, "cat-file", "-t", sha), f"looking for {sha} in {repo}") != "commit":
        raise CacheBuildError(f"{sha} is not a commit of {repo}")
    return _run(_git(repo, "rev-parse", f"{sha}^{{tree}}"), f"reading the tree of {sha}")


def _export_source(repo: Path, sha: str, destination: Path) -> None:
    destination.mkdir()
    tarball = destination.parent / "source.tar"
    _run(_git(repo, "archive", "--format=tar", "-o", str(tarball), sha), f"exporting {sha} from {repo}")
    _run(["tar", "-x", "--no-same-owner", "-f", str(tarball), "-C", str(destination)], f"unpacking {sha}")
    tarball.unlink()


def _pip(*args: str, python: Path | None = None) -> list[str]:
    """The builder's own pip, acting on the environment of ``python`` when given (``pip --python``, pip 22.3 or newer)."""
    return [sys.executable, "-m", "pip", *(["--python", str(python)] if python else []), *args]


def _pip_download(requirements: Path, dest: Path, extra: Sequence[str], what: str, python: Path | None = None) -> None:
    dest.mkdir()
    command = _pip("download", _NO_DEPS, _REQUIRE_HASHES, *_PIP_QUIET, python=python)
    _run([*command, "--dest", str(dest), "-r", str(requirements), *extra], what)


def _build_environment(pins: list[LockPin], options: BuildOptions, scratch: Path) -> Path:
    """The interpreter of a virtual environment in ``scratch`` that holds the build tools the lock pins and nothing else
    that was fetched: the wheels of the tools are downloaded binary-only (no source code runs) and installed by the
    hashes the lock lists."""
    tools = [pin for pin in pins if pin.name in _BUILD_TOOLS]
    if all(pin.name != "setuptools" for pin in tools):
        raise CacheBuildError("the lock pins no setuptools, so a source distribution in it cannot be built")
    pinned = scratch / "build-tools.lock"
    pinned.write_text("".join(requirement_line(pin.name, pin.version, sorted(pin.hashes)) for pin in tools), _UTF8)
    downloads = [*options.source.pip_args(), _ONLY_BINARY]
    _pip_download(pinned, scratch / "tools", downloads, "pip download of the build tools the lock pins")
    _run([sys.executable, "-m", "venv", "--without-pip", str(scratch / "env")], "creating the build environment")
    python = scratch / "env" / "bin" / "python"
    install = [_ISOLATED, "install", _NO_INDEX, _FIND_LINKS, str(scratch / "tools"), _ONLY_BINARY]
    install += [_REQUIRE_HASHES, _NO_DEPS, *_PIP_QUIET, "-r", str(pinned)]
    _run(_pip(*install, python=python), "installing the build tools the lock pins")
    return python


def _wheel_from_sdist(python: Path, sdist: Path, wheels: Path) -> str:
    """Build the pure-Python wheel of ``sdist`` into ``wheels`` with ``python``'s tools, remove the source distribution
    and return the wheel's file name."""
    with tempfile.TemporaryDirectory(dir=wheels.parent, prefix=".wheel-build-") as scratch:
        build = [_ISOLATED, "wheel", "--no-build-isolation", _NO_INDEX, _NO_DEPS, "--no-cache-dir", *_PIP_QUIET]
        build += ["-w", scratch, str(sdist)]
        _run(_pip(*build, python=python), f"building a wheel from {sdist.name}")
        made = sorted(path.name for path in Path(scratch).iterdir())
        if len(made) != 1 or not made[0].endswith("-none-any.whl"):
            raise CacheBuildError(
                f"{sdist.name} did not build exactly one pure-Python wheel (it built {made}): "
                "a wheel built here is not one for the image"
            )
        shutil.move(Path(scratch) / made[0], wheels / made[0])
    sdist.unlink()
    return made[0]


def _build_sdists(python: Path, wheels: Path, pins: list[LockPin]) -> dict[str, str]:
    """Replace each source distribution in ``wheels`` by the wheel built from it; the built wheels' file names, each
    mapped to the SHA-256 of its source distribution, which must be a file the lock lists."""
    allowed = {digest for pin in pins for digest in pin.hashes}
    built: dict[str, str] = {}
    for sdist in (path for path in sorted(wheels.iterdir()) if not path.name.endswith(".whl")):
        digest = file_sha256(sdist)
        if digest not in allowed:
            raise CacheBuildError(f"{sdist.name} is not a file the lock allows")
        built[_wheel_from_sdist(python, sdist, wheels)] = digest
    return built


def _download_wheels(lock: Path, wheels: Path, pins: list[LockPin], options: BuildOptions) -> dict[str, str]:
    """Download what the lock pins into ``wheels`` and build the source distributions among it, in an environment of
    the lock's own build tools; see :func:`_wheel_entries` for the result."""
    with tempfile.TemporaryDirectory(dir=wheels.parent, prefix=".build-env-") as scratch:
        python = _build_environment(pins, options, Path(scratch))
        extra = ["--no-build-isolation", *options.target.pip_args(), *options.source.pip_args()]
        _pip_download(lock, wheels, extra, "pip download of the wheels the lock pins", python)
        return _build_sdists(python, wheels, pins)


def _wheel_entries(wheels: Path, pins: list[LockPin], built: Mapping[str, str]) -> tuple[WheelEntry, ...]:
    """One entry per pin, each file matched to its pin by the hash the lock lists for it (for a built wheel, the hash of
    the source distribution it was built from, which ``built`` records)."""
    owner = {digest: pin for pin in pins for digest in pin.hashes}
    entries: dict[tuple[str, str], WheelEntry] = {}
    for path in sorted(wheels.iterdir()):
        digest, sdist = file_sha256(path), built.get(path.name, "")
        pin = owner.get(sdist or digest)
        if pin is None:
            raise CacheBuildError(f"{path.name} is not a file the lock allows")
        if (pin.name, pin.version) in entries:
            raise CacheBuildError(f"two wheels for {pin.name}=={pin.version}")
        entries[(pin.name, pin.version)] = WheelEntry(
            pin.name, pin.version, path.name, path.stat().st_size, digest, sdist
        )
    missing = [f"{pin.name}=={pin.version}" for pin in pins if (pin.name, pin.version) not in entries]
    if missing:
        raise CacheBuildError(f"no wheel was downloaded for {', '.join(missing)}")
    return tuple(entries.values())


def _pip_version() -> tuple[int, ...]:
    """The builder's pip as numbers (``(24, 2)``); ``()`` when it cannot be read."""
    try:
        return tuple(int(part) for part in re.findall(r"[0-9]+", importlib.metadata.version("pip"))[:2])
    except importlib.metadata.PackageNotFoundError:
        return ()


def _requires_python_flags(target: WheelTarget) -> list[str]:
    """``--ignore-requires-python`` when the builder's pip would check ``Requires-Python`` against its own interpreter
    instead of the target's (pip before 24.2): a numpy that needs Python 3.12 is no reason to refuse a 3.14 entry built
    on 3.10. The wheel tags are checked either way, and the sandbox's pip checks ``Requires-Python`` for itself."""
    if target.python_version and _pip_version() < _PIP_CHECKS_TARGET_PYTHON:
        return ["--ignore-requires-python"]
    return []


def _check_installable(entry: Path, manifest: Manifest, target: WheelTarget) -> None:
    """Refuse an entry whose wheels alone do not satisfy what the sandbox installs, for the target's interpreter."""
    with tempfile.TemporaryDirectory(dir=entry, prefix=".resolve-") as scratch:
        requirements = Path(scratch) / "runner-wheels.lock"
        requirements.write_text(requirements_text(manifest), encoding=_UTF8)
        install = [_ISOLATED, "install", "--dry-run", "--target", str(Path(scratch) / "site"), _NO_INDEX]
        install += [_FIND_LINKS, str(entry / WHEELS_DIR), _ONLY_BINARY, _REQUIRE_HASHES, _NO_DEPS]
        install += [*_PIP_QUIET, *target.pip_args(), *_requires_python_flags(target), "-r", str(requirements)]
        interpreter = f"Python {target.python_version}" if target.python_version else "the builder's interpreter"
        _run(_pip(*install), f"checking that the runner ({interpreter}) can install the entry's wheels")


def _replace_file(path: Path, text: str, mode: int | None = None) -> None:
    """Write ``text`` to ``path`` in one rename, so a reader sees the old file or the new one."""
    pending = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    try:
        pending.write_text(text, encoding=_UTF8)
        if mode is not None:
            pending.chmod(mode)
        os.replace(pending, path)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise


def _write_bootstrap(entry: Path) -> str:
    """Write this builder's start-up script as ``entry/bootstrap.sh`` and return its SHA-256."""
    _replace_file(entry / BOOTSTRAP_NAME, BOOTSTRAP_SCRIPT, _SCRIPT_MODE)
    return file_sha256(entry / BOOTSTRAP_NAME)


def _with_this_bootstrap(entry: Path, manifest: Manifest) -> None:
    """Give a finished entry whose manifest records another script, or none, this builder's script and a manifest
    that records it; the script first, so the manifest never pins a script that is not there."""
    if manifest.bootstrap_sha256 == _BOOTSTRAP_SHA256:
        return
    refreshed = replace(manifest, bootstrap_sha256=_write_bootstrap(entry))
    _replace_file(entry / MANIFEST_NAME, manifest_to_json(refreshed))


def _build(repo: Path, sha: str, tree: str, final: Path, options: BuildOptions) -> CacheEntry:
    """Make the entry beside ``final`` and rename it into place; the manifest is the last file written."""
    pending = final.parent / f".{sha}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
    pending.mkdir()
    try:
        source = pending / SOURCE_DIR
        _export_source(repo, sha, source)
        name = lock_file(source, options.target.python_version)
        lock = source / name
        if not lock.is_file():
            raise CacheBuildError(f"commit {sha} has no {name}")
        wheels = pending / WHEELS_DIR
        pins = parse_lock(lock.read_text(encoding=_UTF8), name)
        entries = _wheel_entries(wheels, pins, _download_wheels(lock, wheels, pins, options))
        bootstrap = _write_bootstrap(pending)
        manifest = Manifest(sha, tree, tree_digest(source), entries, options.target.manifest_form(), bootstrap)
        _check_installable(pending, manifest, options.target)
        (pending / MANIFEST_NAME).write_text(manifest_to_json(manifest), encoding=_UTF8)
        problems = check_entry(pending, sha, verify_hash=True)
        if problems:
            raise CacheBuildError(f"the entry just built does not verify: {'; '.join(problems[:3])}")
        os.rename(pending, final)
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    return CacheEntry(final, file_sha256(final / MANIFEST_NAME), reused=False)


def _differences(final: Path, sha: str, tree: str, options: BuildOptions) -> list[str]:
    """How the finished entry ``final`` differs from what a build would write, one line each; empty when it does not."""
    problems = check_entry(final, sha, verify_hash=True)
    if problems:
        return problems
    manifest = manifest_from_json((final / MANIFEST_NAME).read_text(encoding=_UTF8))
    if manifest.git_tree != tree:
        problems.append(f"its manifest records tree {manifest.git_tree}, but the tree of {sha} is {tree}")
    if manifest.target != options.target.manifest_form():
        problems.append(f"its wheels are for {json.dumps(manifest.target)}, not {options.target.manifest_form()}")
    return problems


def _use_or_clear(final: Path, sha: str, tree: str, options: BuildOptions) -> CacheEntry | None:
    """The finished entry at ``final`` if it is what a build would write, ``None`` when there is nothing to use.

    An entry that has no manifest is an unfinished build's remains and is removed.

    Raises:
        CacheConflict: When the entry is finished and differs.
    """
    if not os.path.lexists(final):
        return None
    if final.is_dir() and not (final / MANIFEST_NAME).is_file():
        shutil.rmtree(final)
        return None
    problems = _differences(final, sha, tree, options)
    if problems:
        raise CacheConflict(
            f"{final} is not what a build of {sha} writes, so it is left alone: {'; '.join(problems[:3])}"
        )
    _with_this_bootstrap(final, manifest_from_json((final / MANIFEST_NAME).read_bytes()))
    return CacheEntry(final, file_sha256(final / MANIFEST_NAME), reused=True)


@contextmanager
def _exclusive(path: Path, timeout_sec: float) -> Iterator[None]:
    """Hold an exclusive ``flock`` on ``path`` for the block; wait for it for at most ``timeout_sec``."""
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        deadline = time.monotonic() + timeout_sec
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise CacheBuildError(f"another build still holds {path} after {timeout_sec:.0f} s") from None
                time.sleep(_LOCK_POLL_SEC)
        yield
    finally:
        os.close(descriptor)


def ensure_runner_cache(repo: Path, sha: str, cache_root: Path, options: BuildOptions = BuildOptions()) -> CacheEntry:
    """The finished cache entry of commit ``sha`` below ``cache_root``, built from ``repo`` when it is not there.

    Safe to call from several processes at once for the same commit: one builds, the others wait and use its entry.

    Raises:
        CacheConflict: When an entry of the commit is there and differs from what a build would write.
        CacheBuildError: When the commit, the lock or the wheels cannot be had, or another build never finishes.
    """
    if not is_full_commit(sha):
        raise CacheBuildError(f"{sha!r} is not a full lowercase commit hash")
    tree = _commit_tree(repo, sha)
    cache_root.mkdir(parents=True, exist_ok=True)
    final = cache_root / sha
    with _exclusive(cache_root / f".{sha}.lock", options.lock_timeout_sec):
        for leftover in cache_root.glob(f".{sha}.tmp-*"):
            shutil.rmtree(leftover, ignore_errors=True)
        return _use_or_clear(final, sha, tree, options) or _build(repo, sha, tree, final, options)


def _repeatable(parser: argparse.ArgumentParser, flag: str, help_text: str) -> None:
    parser.add_argument(flag, action="append", default=[], help=help_text)


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="hyperloom.measure.runner.cache_build", description=__doc__.split("\n")[0])
    parser.add_argument("--sha", required=True, help="the full commit hash")
    parser.add_argument("--repo", type=Path, required=True, help="a clone that has the commit")
    parser.add_argument("--out", type=Path, default=os.environ.get(CACHE_ENV), help=f"the cache (default ${CACHE_ENV})")
    parser.add_argument(
        "--wheel-target",
        help=f"the image's interpreter and platform as one tag, e.g. cp314-manylinux_2_28_x86_64, as ${RUNNER_TARGET_ENV} "
        "names it; instead of --python-version, --implementation, --abi and --platform",
    )
    parser.add_argument("--python-version", help="the image's Python, e.g. 3.10")
    parser.add_argument("--implementation", help="the image's implementation, e.g. cp")
    _repeatable(parser, "--abi", "an ABI tag of the image, e.g. cp310")
    _repeatable(parser, "--platform", "a platform tag, e.g. manylinux_2_28_x86_64")
    parser.add_argument("--index-url", help="the package index to download from")
    _repeatable(parser, _FIND_LINKS, "a directory or URL of wheels")
    parser.add_argument(_NO_INDEX, action="store_true", help=f"use only {_FIND_LINKS}")
    parser.add_argument("--lock-timeout-sec", type=float, default=DEFAULT_LOCK_TIMEOUT_SEC)
    args = parser.parse_args(argv)
    if args.out is None:
        parser.error(f"--out or ${CACHE_ENV} is required")
    if args.wheel_target and (args.python_version or args.implementation or args.abi or args.platform):
        parser.error("--wheel-target names the whole target: give it alone, or the separate options")
    return args


def _target(args: argparse.Namespace) -> WheelTarget:
    if args.wheel_target:
        return runner_wheel_target({RUNNER_TARGET_ENV: args.wheel_target})
    return WheelTarget(args.python_version, args.implementation, tuple(args.abi), tuple(args.platform))


def main(argv: Sequence[str] | None = None) -> int:
    """The command line: print one JSON line about the entry and exit 0, or say why not on stderr and exit 1."""
    args = _arguments(argv)
    try:
        options = BuildOptions(
            _target(args), WheelSource(args.index_url, tuple(args.find_links), args.no_index), args.lock_timeout_sec
        )
        entry = ensure_runner_cache(args.repo, args.sha, args.out, options)
    except (CacheError, OSError) as exc:
        print(f"runner cache not built: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"path": str(entry.path), "manifest_sha256": entry.manifest_sha256, "reused": entry.reused}))
    return 0


#: Public name of the per-key ``flock`` the native build service (``hyperloom.measure.runner.native``) coalesces with.
exclusive = _exclusive


if __name__ == "__main__":
    sys.exit(main())
