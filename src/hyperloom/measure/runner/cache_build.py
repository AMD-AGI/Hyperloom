# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Builds the runner cache of one commit (layout and meaning: :mod:`hyperloom.measure.runner.cache_layout`).

It runs on a machine with egress, in the application that starts a campaign, before the first measurement request::

    python -m hyperloom.measure.runner.cache_build --sha <commit> --repo <checkout> [--out <cache>]

``--out`` defaults to ``$HYPERLOOM_RUNNER_CACHE``. A build that fails fails the campaign: nothing falls back to
installing over the network inside a sandbox. The wheels are downloaded for the interpreter and platform of the
*framework image*, which is not the builder's own unless it is told: ``--python-version 3.10 --implementation cp
--abi cp310 --abi abi3 --abi none`` and one ``--platform`` for every platform tag the image accepts
(``manylinux_2_28_x86_64``, ``manylinux_2_17_x86_64``, ``manylinux2014_x86_64`` and so on down to ``manylinux1_x86_64``)
for a CPython 3.10 image on glibc 2.28, and ``--index-url`` (or ``--find-links`` with ``--no-index``) for where they come
from. pip does not widen a ``--platform`` to the older tags it covers, so naming only the newest one turns every pin
published as ``manylinux2014`` into a source build. The lock's hashes are those of the files the index serves. A pin that the index
serves only as a source distribution is built here into a wheel (the sdist's hash must be one the lock lists; the entry
keeps the sdist in ``sdists/`` and the manifest records it); that wheel must be pure Python, because a platform wheel
built on the builder is not a wheel for the image, and the build is refused otherwise. Reading or building a source
distribution runs its code on the builder, so that happens only in a throwaway virtual environment, without build isolation, whose only build tools are the
``setuptools`` (and ``wheel``) that the lock pins, installed from wheels downloaded by their lock hashes: what a source
distribution's own build requirements name is never fetched. The lock must therefore pin ``setuptools``
(``pip-compile --allow-unsafe`` does). The builder's pip must be 22.3 or newer (``pip --python``).

- A commit is built once. The entry is made in a directory of its own beside the final one, its manifest is written last
  and the directory is renamed into place in one step, so ``<cache>/<sha>`` is either absent or complete.
- Builders of one commit exclude each other with an exclusive ``flock`` on ``<cache>/.<sha>.lock``; the one that waits
  finds the finished entry when it gets the lock and uses it, with no download. The lock is also what keeps two hosts that
  share the cache directory over NFS apart (the cache's file system must support ``flock``, as NFSv4 does).
- An entry that is already there is used when it is complete and still what a build would write; one that differs is
  never modified: the build refuses and names the difference (:class:`CacheConflict`). Its manifest file is read once,
  the entry is checked against those bytes and the pin returned is their SHA-256, as a build returns the SHA-256 of the
  text it wrote: never a fresh read of the cache. What the manifest records is checked against the commit, not only
  against the entry: the same commit tree and wheel target, this builder's script, the commit's archive and source
  made again from the clone in a directory of the builder's own (as a build makes them), whose size and hash must be
  what the manifest records, and every wheel built from a source distribution built again (below). An entry with no
  manifest is the remains of a build that did not finish; it is removed and built again.
- ``<entry>/source.tar`` is the commit's ``git archive`` made with every setting that changes its bytes fixed
  (``git -c tar.umask=0022 -c core.autocrlf=false -c core.eol=lf -c core.attributesFile=/dev/null archive --format=tar
  <sha>``), in a throwaway repository that borrows only the objects of the builder's clone and with git's system and
  global configuration and attributes off (:func:`_archive_source`). Only the commit and its own ``.gitattributes``
  decide what is in it; nothing the builder's clone, configuration or environment says does, so builders whose git
  writes ``tar`` the same way make the same bytes, and anyone with the commit can make it again and compare it with the
  size and SHA-256 the manifest records. ``<entry>/source`` is that archive unpacked with the builder's umask
  (``--no-same-permissions``); the builder reads the lock from it, the sandbox runs from its own copy of the archive.
- ``<entry>/bootstrap.sh`` is this builder's start-up script (:data:`~hyperloom.measure.runner.bootstrap.BOOTSTRAP_SCRIPT`,
  mode 0755), not the commit's, and the manifest records its SHA-256. The script embeds this builder's own
  ``bootstrap.py`` and ``cache_layout.py``, so a builder of another Hyperloom version writes another one. An entry whose
  manifest records another script, or none, is a difference like any other: the build refuses, names both hashes and
  leaves the entry byte for byte as it was, because a campaign that started on it has pinned its manifest
  (``HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256``). Builders of different versions that build one commit use separate
  ``HYPERLOOM_RUNNER_CACHE`` directories.

What the pin vouches for. A sandbox accepts only the manifest whose SHA-256 its template pins, unpacks only the
``source.tar`` bytes of the size and hash that manifest records, reads the lock from that copy, and installs only the
wheels the manifest lists, each matched to a pin of that lock and hashed against the manifest as it is copied
(:func:`~hyperloom.measure.runner.cache_layout.stage_entry`). The builder pins a manifest only when that archive is the
commit's own, so the runner's code and its lock are the commit's, whoever can write the cache, and so is every wheel
downloaded for a pin, whose hash is one the commit's lock lists. A wheel built here from a source distribution is
matched to its pin by the source distribution's hash, the one the lock lists, and a lock cannot tell that hash from a
wheel's. So the builder pins such a wheel only when it has built it itself: on a build, from the file it downloaded; on
reuse, from the entry's copy in ``sdists/``, which must have the hash the manifest records and be a file the commit's
lock lists for that pin, built again offline in a directory of the builder's own with the lock's build tools (installed
from the entry's own wheels of them, by the lock's hashes) into a wheel with the cached wheel's name, files and
contents (:func:`_built_wheel_differences`). A wheel passed off as a source distribution does not build, and an entry
that claims a source distribution it does not hold is refused. So the pin vouches for such a wheel as what the commit's
build tools make of a file the commit's lock lists, whoever can write the cache. A source distribution whose build does
not write the same files every time (one that stamps a time or a path into a file) makes every reuse of its entry a
:class:`CacheConflict`; build that commit into another cache directory.

The cache must stay readable by the account the sandbox runs as: run the builder with a ``umask`` that leaves new files
readable by others (``022``).

Before the entry is renamed into place, the builder asks pip whether the requirements the sandbox installs
(``runner-wheels.lock``) resolve for the wheel target from the entry's wheels alone (``pip install --dry-run --no-index
--only-binary=:all:``, the sandbox's own flags), so a cache the runner could not install is refused here instead of
failing in the sandbox. A hosted session names the runner's interpreter in ``$HYPERLOOM_RUNNER_WHEEL_TARGET``
(:func:`runner_wheel_target`); it never builds for its own.
"""

from __future__ import annotations

import argparse
import re
import fcntl
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.measure.runner.bootstrap import BOOTSTRAP_SCRIPT
from hyperloom.measure.runner.cache_layout import (
    BOOTSTRAP_NAME,
    CACHE_ENV,
    LOCK_FILE,
    MANIFEST_NAME,
    SDISTS_DIR,
    SOURCE_ARCHIVE,
    SOURCE_DIR,
    WHEELS_DIR,
    CacheError,
    FileDigest,
    LockPin,
    Manifest,
    WheelEntry,
    check_entry,
    file_sha256,
    is_full_commit,
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
#: Every setting of ``git archive`` that changes its bytes, fixed: entry modes 0644 and 0755, no line-ending conversion
#: and no attributes file of the builder's (:func:`_archive_source` says what else it takes away).
_ARCHIVE_CONFIG = ("tar.umask=0022", "core.autocrlf=false", "core.eol=lf", f"core.attributesFile={os.devnull}")
_ARCHIVE_SETTINGS = tuple(part for setting in _ARCHIVE_CONFIG for part in ("-c", setting))
#: What ``git`` reads besides the repository, switched off for the archive: system and global configuration, and the
#: system's attributes (``/etc/gitattributes``).
_GIT_NO_SYSTEM_OR_GLOBAL = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_ATTR_NOSYSTEM": "1"}
_GIT = "git"
_OBJECTS = "objects"
#: The file that names the object directories a repository borrows, relative to the repository.
_ALTERNATES = (_OBJECTS, "info", "alternates")
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
#: What :mod:`zipfile` raises for an archive it cannot read: damaged or cut short, of a compression it lacks, encrypted.
_UNREADABLE_ZIP = (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError, RuntimeError)
#: The builder's copy of a cached wheel that a reuse compares with the one it builds again.
_CACHED_WHEEL = "cached.whl"
#: Below the scratch directory of a build environment: the wheels of the build tools, and the virtual environment.
_TOOLS_DIR, _ENV_DIR = "tools", "env"


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


def _run(command: Sequence[str], what: str, env: Mapping[str, str] | None = None) -> str:
    done = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    if done.returncode != 0:
        raise CacheBuildError(f"{what} failed (exit {done.returncode}): {(done.stdout + done.stderr).strip()[-_TAIL:]}")
    return done.stdout.strip()


def _git(repo: Path, *args: str) -> list[str]:
    return [_GIT, "-C", str(repo), *args]


def _commit_tree(repo: Path, sha: str) -> str:
    """The tree object of commit ``sha`` in ``repo``, which also proves the repository has that very commit."""
    if _run(_git(repo, "cat-file", "-t", sha), f"looking for {sha} in {repo}") != "commit":
        raise CacheBuildError(f"{sha} is not a commit of {repo}")
    return _run(_git(repo, "rev-parse", f"{sha}^{{tree}}"), f"reading the tree of {sha}")


def _archive_source(repo: Path, sha: str, archive: Path) -> None:
    """Write the ``git archive`` of commit ``sha`` of ``repo`` to ``archive``, bytes that only the commit decides.

    git applies the attributes of the ``.gitattributes`` files in the commit, which are the same for everyone, but also
    the builder's own: ``$GIT_DIR/info/attributes``, ``core.attributesFile`` (``~/.config/git/attributes`` unless set)
    and the system's ``gitattributes``, any of which can leave files out (``export-ignore``) or rewrite them (``eol``,
    ``filter``). So the archive is made in a new, empty repository beside ``archive`` that borrows only ``repo``'s
    objects (``objects/info/alternates``): ``repo``'s configuration, ``info/attributes``, replace refs and grafts do not
    apply. git runs with no ``GIT_`` variable of the builder's environment, an empty ``HOME`` and no system or global
    configuration or attributes (:data:`_GIT_NO_SYSTEM_OR_GLOBAL`), and :data:`_ARCHIVE_SETTINGS` fixes the rest.
    """
    objects = Path(repo, _run(_git(repo, "rev-parse", "--git-path", _OBJECTS), f"finding the objects of {repo}"))
    with tempfile.TemporaryDirectory(dir=archive.parent, prefix=".archive-") as scratch:
        home, borrower = Path(scratch, "home"), Path(scratch, "repo")
        home.mkdir()
        env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
        env.update(HOME=str(home), XDG_CONFIG_HOME=str(home), **_GIT_NO_SYSTEM_OR_GLOBAL)
        object_format = ["--object-format=sha256"] if len(sha) == 64 else []
        _run([_GIT, "init", "-q", "--bare", "--template=", *object_format, str(borrower)], "making a repository", env)
        alternates = borrower.joinpath(*_ALTERNATES)
        alternates.parent.mkdir(parents=True, exist_ok=True)
        alternates.write_text(f"{objects.resolve()}\n", encoding=_UTF8)
        export = [_GIT, f"--git-dir={borrower}", *_ARCHIVE_SETTINGS, "archive", "--format=tar"]
        _run([*export, "-o", str(archive.absolute()), sha], f"exporting {sha} from {repo}", env)


def _export_source(repo: Path, sha: str, entry: Path) -> FileDigest:
    """Write the commit's archive as ``entry/source.tar`` (:func:`_archive_source`) and unpack it into ``entry/source``
    with the builder's umask; the archive's size and SHA-256."""
    archive, source = entry / SOURCE_ARCHIVE, entry / SOURCE_DIR
    _archive_source(repo, sha, archive)
    source.mkdir()
    unpack = ["tar", "-x", "--no-same-owner", "--no-same-permissions", "-f", str(archive), "-C", str(source)]
    _run(unpack, f"unpacking {sha}")
    if not (source / LOCK_FILE).is_file():
        raise CacheBuildError(f"commit {sha} has no {LOCK_FILE}")
    return FileDigest(archive.stat().st_size, file_sha256(archive))


def _pip(*args: str, python: Path | None = None) -> list[str]:
    """The builder's own pip, acting on the environment of ``python`` when given (``pip --python``, pip 22.3 or newer)."""
    return [sys.executable, "-m", "pip", *(["--python", str(python)] if python else []), *args]


def _pip_download(requirements: Path, dest: Path, extra: Sequence[str], what: str, python: Path | None = None) -> None:
    dest.mkdir()
    command = _pip("download", _NO_DEPS, _REQUIRE_HASHES, *_PIP_QUIET, python=python)
    _run([*command, "--dest", str(dest), "-r", str(requirements), *extra], what)


def _tools_lock(pins: list[LockPin], scratch: Path) -> Path:
    """``scratch/build-tools.lock``: the build tools the lock pins (``setuptools``, and ``wheel`` when it pins one), with
    the lock's hashes.

    Raises:
        CacheBuildError: When the lock pins no ``setuptools``.
    """
    tools = [pin for pin in pins if pin.name in _BUILD_TOOLS]
    if all(pin.name != "setuptools" for pin in tools):
        raise CacheBuildError("the lock pins no setuptools, so a source distribution in it cannot be built")
    pinned = scratch / "build-tools.lock"
    pinned.write_text("".join(requirement_line(pin.name, pin.version, sorted(pin.hashes)) for pin in tools), _UTF8)
    return pinned


def _tools_environment(pinned: Path, scratch: Path) -> Path:
    """The interpreter of a new virtual environment in ``scratch`` that holds the build tools ``pinned`` names and
    nothing else, installed binary-only (no source code runs) from the wheels in ``scratch/tools``, each checked by pip
    against the hashes the lock lists."""
    env = scratch / _ENV_DIR
    _run([sys.executable, "-m", "venv", "--without-pip", str(env)], "creating the build environment")
    python = env / "bin" / "python"
    install = [_ISOLATED, "install", _NO_INDEX, _FIND_LINKS, str(scratch / _TOOLS_DIR), _ONLY_BINARY]
    install += [_REQUIRE_HASHES, _NO_DEPS, *_PIP_QUIET, "-r", str(pinned)]
    _run(_pip(*install, python=python), "installing the build tools the lock pins")
    return python


def _build_environment(pins: list[LockPin], options: BuildOptions, scratch: Path) -> Path:
    """The build environment of a build (:func:`_tools_environment`), whose tools' wheels are downloaded by the lock's
    hashes."""
    pinned = _tools_lock(pins, scratch)
    downloads = [*options.source.pip_args(), _ONLY_BINARY]
    _pip_download(pinned, scratch / _TOOLS_DIR, downloads, "pip download of the build tools the lock pins")
    return _tools_environment(pinned, scratch)


def _pure_wheel(python: Path, sdist: Path, out: Path) -> Path:
    """Build the wheel of ``sdist`` into the new directory ``out`` with ``python``'s tools, offline and without build
    isolation, and return it.

    Raises:
        CacheBuildError: When the build fails, or makes anything but one pure-Python wheel: a platform wheel built on
            the builder is not one for the image.
    """
    out.mkdir()
    build = [_ISOLATED, "wheel", "--no-build-isolation", _NO_INDEX, _NO_DEPS, "--no-cache-dir", *_PIP_QUIET]
    _run(_pip(*build, "-w", str(out), str(sdist), python=python), f"building a wheel from {sdist.name}")
    made = sorted(path.name for path in out.iterdir())
    if len(made) != 1 or not made[0].endswith("-none-any.whl"):
        raise CacheBuildError(
            f"{sdist.name} did not build exactly one pure-Python wheel (it built {made}): "
            "a wheel built here is not one for the image"
        )
    return out / made[0]


def _build_sdists(python: Path, wheels: Path, pins: list[LockPin]) -> dict[str, str]:
    """Build each source distribution in ``wheels`` into the wheel that takes its place there, and move the source
    distribution to the entry's ``sdists``; the built wheels' file names, each mapped to the file name of its source
    distribution, whose hash must be one the lock lists."""
    allowed = {digest for pin in pins for digest in pin.hashes}
    sdists = wheels.parent / SDISTS_DIR
    built: dict[str, str] = {}
    for sdist in (path for path in sorted(wheels.iterdir()) if not path.name.endswith(".whl")):
        if file_sha256(sdist) not in allowed:
            raise CacheBuildError(f"{sdist.name} is not a file the lock allows")
        with tempfile.TemporaryDirectory(dir=wheels.parent, prefix=".wheel-build-") as scratch:
            wheel = _pure_wheel(python, sdist, Path(scratch) / "out")
            shutil.move(wheel, wheels / wheel.name)
        sdists.mkdir(exist_ok=True)
        shutil.move(sdist, sdists / sdist.name)
        built[wheel.name] = sdist.name
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
    the source distribution it was built from, whose file in the entry's ``sdists`` ``built`` names)."""
    owner = {digest: pin for pin in pins for digest in pin.hashes}
    entries: dict[tuple[str, str], WheelEntry] = {}
    for path in sorted(wheels.iterdir()):
        sdist = built.get(path.name, "")
        digest, sdist_digest = file_sha256(path), file_sha256(wheels.parent / SDISTS_DIR / sdist) if sdist else ""
        pin = owner.get(sdist_digest or digest)
        if pin is None:
            raise CacheBuildError(f"{path.name} is not a file the lock allows")
        if (pin.name, pin.version) in entries:
            raise CacheBuildError(f"two wheels for {pin.name}=={pin.version}")
        entries[(pin.name, pin.version)] = WheelEntry(
            pin.name, pin.version, path.name, path.stat().st_size, digest, sdist_digest, sdist
        )
    missing = [f"{pin.name}=={pin.version}" for pin in pins if (pin.name, pin.version) not in entries]
    if missing:
        raise CacheBuildError(f"no wheel was downloaded for {', '.join(missing)}")
    return tuple(entries.values())


def _check_installable(entry: Path, manifest: Manifest, target: WheelTarget) -> None:
    """Refuse an entry whose wheels alone do not satisfy what the sandbox installs, for the target's interpreter."""
    with tempfile.TemporaryDirectory(dir=entry, prefix=".resolve-") as scratch:
        requirements = Path(scratch) / "runner-wheels.lock"
        requirements.write_text(requirements_text(manifest), encoding=_UTF8)
        install = [_ISOLATED, "install", "--dry-run", "--target", str(Path(scratch) / "site"), _NO_INDEX]
        install += [_FIND_LINKS, str(entry / WHEELS_DIR), _ONLY_BINARY, _REQUIRE_HASHES, _NO_DEPS]
        install += [*_PIP_QUIET, *target.pip_args(), "-r", str(requirements)]
        interpreter = f"Python {target.python_version}" if target.python_version else "the builder's interpreter"
        _run(_pip(*install), f"checking that the runner ({interpreter}) can install the entry's wheels")


def _write_bootstrap(entry: Path) -> str:
    """Write this builder's start-up script as ``entry/bootstrap.sh`` and return its SHA-256."""
    script = entry / BOOTSTRAP_NAME
    script.write_text(BOOTSTRAP_SCRIPT, encoding=_UTF8)
    script.chmod(_SCRIPT_MODE)
    return file_sha256(script)


def _builder_problems(raw: bytes, manifest: Manifest) -> list[str]:
    """How a finished entry's manifest, whose file holds ``raw``, differs from the one this builder writes for it."""
    if manifest.bootstrap_sha256 != _BOOTSTRAP_SHA256:
        recorded = f"SHA-256 {manifest.bootstrap_sha256}" if manifest.bootstrap_sha256 else "none"
        return [
            f"its manifest records {BOOTSTRAP_NAME} {recorded}, this builder writes {BOOTSTRAP_NAME} SHA-256 "
            f"{_BOOTSTRAP_SHA256}: a builder of another Hyperloom version made it, and a campaign may have pinned its "
            f"manifest; build into a separate {CACHE_ENV} directory"
        ]
    if raw != manifest_to_json(manifest).encode(_UTF8):
        return ["its manifest is not the text this builder writes for what it records"]
    return []


def _build(repo: Path, sha: str, tree: str, final: Path, options: BuildOptions) -> CacheEntry:
    """Make the entry beside ``final`` and rename it into place; the manifest is the last file written, and the pin
    returned is the SHA-256 of the text written, which the check of the new entry is pinned to."""
    pending = final.parent / f".{sha}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
    pending.mkdir()
    try:
        archive = _export_source(repo, sha, pending)
        source = pending / SOURCE_DIR
        lock = source / LOCK_FILE
        wheels = pending / WHEELS_DIR
        pins = parse_lock(lock.read_text(encoding=_UTF8))
        entries = _wheel_entries(wheels, pins, _download_wheels(lock, wheels, pins, options))
        bootstrap = _write_bootstrap(pending)
        target = options.target.manifest_form()
        manifest = Manifest(sha, tree, tree_digest(source), entries, target, bootstrap, archive)
        _check_installable(pending, manifest, options.target)
        text = manifest_to_json(manifest).encode(_UTF8)
        (pending / MANIFEST_NAME).write_bytes(text)
        pin = hashlib.sha256(text).hexdigest()
        problems = check_entry(pending, sha, pin, verify_hash=True)
        if problems:
            raise CacheBuildError(f"the entry just built does not verify: {'; '.join(problems[:3])}")
        os.rename(pending, final)
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    return CacheEntry(final, pin, reused=False)


def _source_differences(archive: FileDigest, made: Path, manifest: Manifest) -> list[str]:
    """How the source ``manifest`` records differs from the commit's, whose archive (of digest ``archive``) and unpacked
    tree :func:`_export_source` made again in the builder's own directory ``made``: compared by size and hash."""
    source = tree_digest(made / SOURCE_DIR)
    problems: list[str] = []
    if manifest.source_archive != archive:
        problems.append(
            f"its {SOURCE_ARCHIVE} is not the archive of {manifest.sha}: the manifest records "
            f"{manifest.source_archive}, the commit's is {archive}"
        )
    if manifest.source != source:
        problems.append(
            f"its {SOURCE_DIR} is not the tree of {manifest.sha}: the manifest records {manifest.source}, not {source}"
        )
    return problems


def _wheel_files(wheel: Path) -> list[tuple[str, str]]:
    """What a wheel installs, as its ``RECORD`` would list it but read from the bytes: every member's name and the
    SHA-256 of its content, in name order. A zip's timestamps, member order and compression are not part of it, so two
    builds of one source distribution compare equal when they wrote the same files."""
    with zipfile.ZipFile(wheel) as archive:
        return sorted((info.filename, hashlib.sha256(archive.read(info)).hexdigest()) for info in archive.infolist())


def _held_sdist_problems(entry: Path, wheel: WheelEntry, pin: LockPin | None, work: Path) -> list[str]:
    """Copy the source distribution the entry holds for ``wheel`` to ``work/sdist/<its file name>`` and the wheel to
    ``work/cached.whl``, in the new directory ``work``; why the copies are not the files the manifest records, or the
    source distribution is not a file the commit's lock lists for ``pin``."""
    label = f"{wheel.file} for {wheel.name}=={wheel.version}"
    held = entry / SDISTS_DIR / wheel.sdist_file
    if Path(wheel.sdist_file).name != wheel.sdist_file or not held.is_file():
        return [f"{label} is recorded as built from a source distribution that the entry does not hold in {SDISTS_DIR}"]
    sdist = work / "sdist" / wheel.sdist_file
    sdist.parent.mkdir(parents=True)
    shutil.copyfile(held, sdist)
    shutil.copyfile(entry / WHEELS_DIR / wheel.file, work / _CACHED_WHEEL)
    digest = file_sha256(sdist)
    if digest != wheel.sdist_sha256 or pin is None or digest not in pin.hashes:
        return [
            f"{SDISTS_DIR}/{wheel.sdist_file} is not the source distribution of {label} that the manifest records "
            f"and the lock lists: its SHA-256 is {digest}"
        ]
    if file_sha256(work / _CACHED_WHEEL) != wheel.sha256:
        return [f"wheel {label} does not match the hash in the manifest"]
    return []


def _rebuilt_wheel_problems(wheel: WheelEntry, python: Path, work: Path) -> list[str]:
    """Build the source distribution :func:`_held_sdist_problems` copied into ``work`` again; why the wheel it makes is
    not the cached wheel's copy beside it (another name, or other files or contents)."""
    try:
        rebuilt = _pure_wheel(python, work / "sdist" / wheel.sdist_file, work / "rebuilt")
        cached, made = _wheel_files(work / _CACHED_WHEEL), _wheel_files(rebuilt)
    except CacheBuildError as exc:
        return [f"{SDISTS_DIR}/{wheel.sdist_file} does not build {wheel.file} again: {exc}"]
    except _UNREADABLE_ZIP as exc:
        return [f"{wheel.file} cannot be read as a wheel: {type(exc).__name__}: {exc}"]
    if rebuilt.name != wheel.file:
        return [f"{SDISTS_DIR}/{wheel.sdist_file} builds {rebuilt.name}, not {wheel.file}"]
    if cached != made:
        differ = sorted({name for name, _ in set(cached) ^ set(made)})
        return [
            f"{wheel.file} is not the wheel its source distribution builds; the members that differ: "
            f"{', '.join(differ[:3]) or 'one it lists twice'}"
        ]
    return []


def _entry_tools(entry: Path, manifest: Manifest, pins: list[LockPin], scratch: Path) -> Path:
    """The build environment of a reuse (:func:`_tools_environment`), with nothing downloaded: the tools are installed
    from copies of the entry's own wheels of them, which pip checks against the lock's hashes."""
    pinned = _tools_lock(pins, scratch)
    tools = scratch / _TOOLS_DIR
    tools.mkdir()
    for wheel in manifest.wheels:
        if wheel.name in _BUILD_TOOLS:
            shutil.copyfile(entry / WHEELS_DIR / wheel.file, tools / Path(wheel.file).name)
    return _tools_environment(pinned, scratch)


def _built_wheel_differences(entry: Path, manifest: Manifest, pins: list[LockPin]) -> list[str]:
    """How the wheels ``manifest`` records as built from a source distribution differ from what the builder makes of
    the copy the entry keeps, each checked in a directory of the builder's own: (1) the copy must have the hash the
    manifest records and be a file the commit's lock (``pins``) lists for that pin, before any of its code runs; (2) it
    is built again offline with the lock's build tools, installed from the entry's wheels of them by the lock's hashes
    (:func:`_entry_tools`); (3) the wheel that makes must have the cached wheel's name, files and contents
    (:func:`_wheel_files`)."""
    built = [wheel for wheel in manifest.wheels if wheel.sdist_sha256]
    if not built:
        return []
    pinned = {(pin.name, pin.version): pin for pin in pins}
    with tempfile.TemporaryDirectory(prefix="hyperloom-runner-rebuild-") as scratch:
        work = Path(scratch)
        held = [
            _held_sdist_problems(entry, wheel, pinned.get((wheel.name, wheel.version)), work / str(index))
            for index, wheel in enumerate(built)
        ]
        if any(held):
            return [problem for problems in held for problem in problems]
        try:
            python = _entry_tools(entry, manifest, pins, work)
        except CacheBuildError as exc:
            return [f"the build tools of its source distributions cannot be installed from its wheels: {exc}"]
        return [
            problem
            for index, wheel in enumerate(built)
            for problem in _rebuilt_wheel_problems(wheel, python, work / str(index))
        ]


def _differences(repo: Path, tree: str, entry: Path, raw: bytes, options: BuildOptions) -> list[str]:
    """How the manifest text ``raw`` of ``entry``, which agrees with it (:func:`check_entry`, so it is of the entry's
    commit), differs from what a build of that commit would write, one line each; empty when it does not. What it
    records is checked against the commit, not against the entry: the commit's tree, its archive and source made again
    from ``repo`` (:func:`_source_differences`), and the wheels built from a source distribution built again against
    that remade source's lock (:func:`_built_wheel_differences`)."""
    manifest = manifest_from_json(raw)
    problems: list[str] = []
    if manifest.git_tree != tree:
        problems.append(f"its manifest records tree {manifest.git_tree}, but the tree of {manifest.sha} is {tree}")
    if manifest.target != options.target.manifest_form():
        problems.append(f"its wheels are for {json.dumps(manifest.target)}, not {options.target.manifest_form()}")
    with tempfile.TemporaryDirectory(prefix="hyperloom-runner-source-") as scratch:
        made = Path(scratch)
        problems += _source_differences(_export_source(repo, manifest.sha, made), made, manifest)
        problems += _builder_problems(raw, manifest)
        if problems:
            return problems
        pins = parse_lock((made / SOURCE_DIR / LOCK_FILE).read_text(encoding=_UTF8))
    return _built_wheel_differences(entry, manifest, pins)


def _use_or_clear(repo: Path, sha: str, tree: str, final: Path, options: BuildOptions) -> CacheEntry | None:
    """The finished entry at ``final`` if it is what a build would write, ``None`` when there is nothing to use.

    An entry that has no manifest is an unfinished build's remains and is removed. A finished one is never written to.
    Its manifest file is read once: the entry is checked against those bytes (:func:`check_entry` pinned to their
    hash, so a manifest written over meanwhile is refused), those bytes against the commit, its wheels built from a
    source distribution included (:func:`_differences`), and the pin returned is their SHA-256.

    Raises:
        CacheConflict: When the entry is finished and differs.
    """
    if not os.path.lexists(final):
        return None
    if final.is_dir() and not (final / MANIFEST_NAME).is_file():
        shutil.rmtree(final)
        return None
    raw = (final / MANIFEST_NAME).read_bytes()
    pin = hashlib.sha256(raw).hexdigest()
    problems = check_entry(final, sha, pin, verify_hash=True) or _differences(repo, tree, final, raw, options)
    if problems:
        raise CacheConflict(
            f"{final} is not what a build of {sha} writes, so it is left alone: {'; '.join(problems[:3])}"
        )
    return CacheEntry(final, pin, reused=True)


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
        return _use_or_clear(repo, sha, tree, final, options) or _build(repo, sha, tree, final, options)


def _repeatable(parser: argparse.ArgumentParser, flag: str, help_text: str) -> None:
    parser.add_argument(flag, action="append", default=[], help=help_text)


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="hyperloom.measure.runner.cache_build", description=__doc__.split("\n")[0])
    parser.add_argument("--sha", required=True, help="the full commit hash")
    parser.add_argument("--repo", type=Path, required=True, help="a clone that has the commit")
    parser.add_argument("--out", type=Path, default=os.environ.get(CACHE_ENV), help=f"the cache (default ${CACHE_ENV})")
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
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """The command line: print one JSON line about the entry and exit 0, or say why not on stderr and exit 1."""
    args = _arguments(argv)
    options = BuildOptions(
        WheelTarget(args.python_version, args.implementation, tuple(args.abi), tuple(args.platform)),
        WheelSource(args.index_url, tuple(args.find_links), args.no_index),
        args.lock_timeout_sec,
    )
    try:
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
