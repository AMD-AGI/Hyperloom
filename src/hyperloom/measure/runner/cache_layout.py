# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner cache: what the sealed runner installs from, and the one check of it that the builder and the runner share.

A measurement sandbox has no network. Its source and its Python wheels come from a cache directory on shared storage
that the sandbox mounts read-only; :mod:`hyperloom.measure.runner.cache_build` fills it on a machine that has egress and
the start-up script (:mod:`hyperloom.measure.runner.bootstrap`) installs from it with no index. One commit's entry is::

    <cache>/<sha>/source/      the repository tree at that commit (``git archive``), its runner locks included
    <cache>/<sha>/wheels/      one wheel per pin of the lock of the entry's Python (:func:`lock_file`)
    <cache>/<sha>/bootstrap.sh the start-up script (``BOOTSTRAP_SCRIPT``) of the builder that made or last refreshed
                               the entry, mode 0755; a template step runs it with ``sh``
    <cache>/<sha>/manifest     JSON, written last: the commit and its tree, the digest of ``source``, the SHA-256 of
                               ``bootstrap.sh`` and, for each wheel, its pin, file, size and SHA-256 (see :class:`WheelEntry`)

The sandbox installs exactly the wheels the manifest lists, each pinned to the hash the manifest records
(:func:`requirements_text`); the builder has checked every one of them against the lock. An entry without a manifest is
an unfinished build, and nothing uses it. ``<cache>`` is the directory
``HYPERLOOM_RUNNER_CACHE`` names, on the builder's machine as well as inside the sandbox (the mount point may differ).

One lock per Python. A pin has wheels for the Python versions its release supports, so no one set of pins installs on
every interpreter a framework image ships (``numpy==2.2.6`` has no CPython 3.14 wheel, and the release that has one has
none for 3.10). A Python version with a lock of its own has it as ``requirements/runner-cpXY.lock`` (``cp314`` for 3.14);
every other version uses ``requirements/runner.lock``. The entry's lock is the one of the Python its manifest's
``target`` names, read from the entry's own source tree, which the manifest pins, so the builder and the sandbox always
pick the same file. The sandbox also refuses an entry whose wheels are for another Python than its own interpreter.

This module is the script's checker as well: the start-up script embeds this file's text and runs it with the sandbox's
own interpreter before it does anything else, so it uses nothing but the standard library and runs on every Python a
framework image has (3.10 to 3.14).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "BOOTSTRAP_NAME",
    "CACHE_ENV",
    "DEFAULT_CACHE_DIR",
    "LOCK_FILE",
    "MANIFEST_NAME",
    "MANIFEST_PIN_ENV",
    "REQUIREMENTS_FILE",
    "SOURCE_DIR",
    "WHEELS_DIR",
    "CacheError",
    "LockPin",
    "Manifest",
    "TreeDigest",
    "WheelEntry",
    "canonical_name",
    "check_cache",
    "check_entry",
    "file_sha256",
    "is_full_commit",
    "lock_file",
    "manifest_from_json",
    "manifest_to_json",
    "parse_lock",
    "requirement_line",
    "requirements_text",
    "tree_digest",
]

#: The directory of the cache, on the builder's machine and in the sandbox.
CACHE_ENV = "HYPERLOOM_RUNNER_CACHE"
#: Where the sandbox expects the cache when the template does not set :data:`CACHE_ENV`.
DEFAULT_CACHE_DIR = "/runner-cache"
#: Optional: the SHA-256 of the manifest file the host got from the builder. When set, the runner accepts only that one.
MANIFEST_PIN_ENV = "HYPERLOOM_RUNNER_CACHE_MANIFEST_SHA256"
MANIFEST_NAME = "manifest"
#: The start-up script in an entry, which the manifest pins by its SHA-256.
BOOTSTRAP_NAME = "bootstrap.sh"
#: The hash-pinned requirements the sandbox installs from, written below its scratch directory.
REQUIREMENTS_FILE = "runner-wheels.lock"
SOURCE_DIR = "source"
WHEELS_DIR = "wheels"
#: The lock of every Python version that has none of its own (see :func:`lock_file`).
LOCK_FILE = "requirements/runner.lock"
_VERSION_LOCK = "requirements/runner-cp{digits}.lock"
_SCHEMA = 1
_UTF8 = "utf-8"
_SHOWN = 5
_CHUNK = 1 << 20
_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_PIN = re.compile(r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;\\]+)(?: \\)?")
_HASH = re.compile(r"\s+--hash=sha256:(?P<digest>[0-9a-f]{64})(?: \\)?")


class CacheError(Exception):
    """A cache, a manifest or a lock that cannot be read as what it must be; the message names what is wrong."""


@dataclass(frozen=True)
class LockPin:
    """One ``name==version`` entry of a hash lock: the canonical name and the SHA-256 of every file the lock allows."""

    name: str
    version: str
    hashes: frozenset[str]


@dataclass(frozen=True)
class TreeDigest:
    """What a directory tree holds: the count and bytes of its files and one hash over their paths and contents."""

    files: int
    size: int
    sha256: str


@dataclass(frozen=True)
class WheelEntry:
    """One wheel of the cache: the pin it satisfies and the file that does.

    A pin that the index serves only as a source distribution is built into a wheel by the builder; ``sdist_sha256`` is
    the hash of that source distribution, which the lock lists, and ``sha256`` is the built wheel's. Empty for a wheel
    that was downloaded, whose own hash the lock lists.
    """

    name: str
    version: str
    file: str
    size: int
    sha256: str
    sdist_sha256: str = ""


@dataclass(frozen=True)
class Manifest:
    """What a finished build records. ``git_tree`` is the commit's tree object, ``target`` what the wheels are for and
    ``bootstrap_sha256`` the hash of the entry's ``bootstrap.sh`` (empty in a manifest written before entries had one)."""

    sha: str
    git_tree: str
    source: TreeDigest
    wheels: tuple[WheelEntry, ...]
    target: dict[str, Any]
    bootstrap_sha256: str = ""


def canonical_name(name: str) -> str:
    """The PEP 503 form of a distribution name, the one a lock's pins and the manifest compare by."""
    return re.sub(r"[-_.]+", "-", name).lower()


def is_full_commit(sha: str) -> bool:
    """Whether ``sha`` is a full lowercase commit hash (SHA-1 or SHA-256 object format)."""
    return _COMMIT.fullmatch(sha) is not None


def file_sha256(path: Path) -> str:
    """The SHA-256 of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def lock_file(source: Path, python_version: str | None) -> str:
    """The lock, as a path in the source tree ``source``, whose pins are the wheels of Python ``python_version``
    (``"3.14"``, as a wheel target records it): ``requirements/runner-cpXY.lock`` when the tree has one for that
    version, :data:`LOCK_FILE` otherwise and when no version is named."""
    digits = (python_version or "").replace(".", "")
    own = _VERSION_LOCK.format(digits=digits)
    return own if digits.isdigit() and (source / own).is_file() else LOCK_FILE


def _python_problems(manifest: Manifest, python_version: str) -> list[str]:
    """Why the wheels of ``manifest`` are not for the interpreter ``python_version`` (empty when either is unnamed)."""
    wanted = manifest.target.get("python_version")
    if not python_version or not wanted or str(wanted).replace(".", "") == python_version.replace(".", ""):
        return []
    return [
        f"the cache entry's wheels are for Python {wanted}, but this interpreter is Python {python_version}: "
        "build the cache for this interpreter (HYPERLOOM_RUNNER_WHEEL_TARGET)"
    ]


def parse_lock(text: str, name: str = LOCK_FILE) -> list[LockPin]:
    """The pins of a ``pip-compile --generate-hashes`` file; ``name`` is what an error calls it.

    Raises:
        CacheError: For a line that is neither a pin, one of its hashes, a comment nor blank.
    """
    entries: list[tuple[str, str, set[str]]] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        pin, digest = _PIN.fullmatch(line), _HASH.fullmatch(line)
        if pin:
            entries.append((canonical_name(pin["name"]), pin["version"], set()))
        elif digest and entries:
            entries[-1][2].add(digest["digest"])
        else:
            raise CacheError(f"{name} line {number} is not a pin or a hash")
    return [LockPin(name, version, frozenset(hashes)) for name, version, hashes in entries]


def _entries(directory: Path) -> Iterator[os.DirEntry[str]]:
    """Every file and symbolic link below ``directory``, each directory's entries in name order."""
    for entry in sorted(os.scandir(directory), key=lambda item: item.name):
        if entry.is_dir(follow_symlinks=False):
            yield from _entries(Path(entry.path))
        else:
            yield entry


def _row(root: Path, entry: os.DirEntry[str]) -> tuple[str, int]:
    """The line one entry adds to a tree's hash, and its size."""
    path = Path(entry.path)
    info = entry.stat(follow_symlinks=False)
    if entry.is_symlink():
        target = os.readlink(path)
        kind, content, size = "l", hashlib.sha256(target.encode()).hexdigest(), len(target)
    else:
        kind, content, size = "x" if info.st_mode & 0o111 else "f", file_sha256(path), info.st_size
    return f"{kind}\t{size}\t{content}\t{path.relative_to(root).as_posix()}\n", size


def tree_digest(root: Path) -> TreeDigest:
    """The count, bytes and one SHA-256 of every file and symbolic link below ``root``.

    The hash covers each entry's path, its size, whether it is a link or executable and its content (a link's target),
    so a file that is missing, added, changed or made executable changes it.
    """
    outer = hashlib.sha256()
    files = size = 0
    for entry in _entries(root):
        row, length = _row(root, entry)
        outer.update(row.encode())
        files, size = files + 1, size + length
    return TreeDigest(files, size, outer.hexdigest())


def manifest_to_json(manifest: Manifest) -> str:
    """The text of the manifest file."""
    return json.dumps({"schema": _SCHEMA, **asdict(manifest)}, indent=2, sort_keys=True) + "\n"


def manifest_from_json(text: str | bytes) -> Manifest:
    """Read the manifest file's content.

    Raises:
        CacheError: When it is not JSON, has another schema or lacks a field.
    """
    try:
        data = json.loads(text)
        if data["schema"] != _SCHEMA:
            raise CacheError(f"the manifest has schema {data['schema']!r}, this reader knows {_SCHEMA}")
        return Manifest(
            data["sha"],
            data["git_tree"],
            TreeDigest(**data["source"]),
            tuple(WheelEntry(**wheel) for wheel in data["wheels"]),
            dict(data["target"]),
            str(data.get("bootstrap_sha256", "")),
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise CacheError(f"the manifest cannot be read: {type(exc).__name__}: {exc}") from exc


def _source_problems(source: Path, manifest: Manifest) -> list[str]:
    found = tree_digest(source)
    if found == manifest.source:
        return []
    return [
        f"the source in {source} is not what the manifest recorded: {found.files} files and {found.size} bytes, "
        f"the manifest has {manifest.source.files} and {manifest.source.size}"
    ]


def _wheel_problems(wheels: Path, wheel: WheelEntry, pin: LockPin, verify_hash: bool) -> list[str]:
    label = f"{pin.name}=={pin.version}"
    path = wheels / wheel.file
    if not path.is_file():
        return [f"missing wheel {wheel.file} for {label} in {wheels}"]
    if (wheel.sdist_sha256 or wheel.sha256) not in pin.hashes:
        return [f"{wheel.file} for {label} is not, or was not built from, a file the lock lists for it"]
    if path.stat().st_size != wheel.size:
        return [f"wheel {wheel.file} for {label} is {path.stat().st_size} bytes, the manifest says {wheel.size}"]
    if verify_hash and file_sha256(path) != wheel.sha256:
        return [f"wheel {wheel.file} for {label} does not match the hash in the manifest"]
    return []


def _wheels_problems(wheels: Path, manifest: Manifest, pins: list[LockPin], verify_hash: bool) -> list[str]:
    listed = {(wheel.name, wheel.version): wheel for wheel in manifest.wheels}
    problems: list[str] = []
    for pin in pins:
        wheel = listed.pop((pin.name, pin.version), None)
        if wheel is None:
            problems.append(f"no wheel for {pin.name}=={pin.version} in the manifest")
        else:
            problems += _wheel_problems(wheels, wheel, pin, verify_hash)
    return problems + [f"the manifest lists {name}=={version}, which the lock does not pin" for name, version in listed]


def _bootstrap_problems(entry: Path, manifest: Manifest) -> list[str]:
    script = entry / BOOTSTRAP_NAME
    if not manifest.bootstrap_sha256:
        return []
    if not script.is_file() or file_sha256(script) != manifest.bootstrap_sha256:
        return [f"{script} is missing or does not match the hash in the manifest"]
    return []


def _entry_problems(entry: Path, sha: str, manifest: Manifest, verify_hash: bool, python_version: str) -> list[str]:
    if manifest.sha != sha:
        return [f"the manifest of {entry} is for commit {manifest.sha}, not {sha}"]
    source, wheels = entry / SOURCE_DIR, entry / WHEELS_DIR
    if not source.is_dir() or not wheels.is_dir():
        return [f"the cache entry {entry} lacks its {SOURCE_DIR} or {WHEELS_DIR} directory"]
    wrong_python = _python_problems(manifest, python_version)
    if wrong_python:
        return wrong_python
    name = lock_file(source, manifest.target.get("python_version"))
    lock = source / name
    if not lock.is_file():
        return [f"the source in {source} has no {name}"]
    pins = parse_lock(lock.read_text(encoding=_UTF8), name)
    problems = _source_problems(source, manifest) + _wheels_problems(wheels, manifest, pins, verify_hash)
    return problems + (_bootstrap_problems(entry, manifest) if verify_hash else [])


def check_entry(
    entry: Path, sha: str, manifest_pin: str = "", verify_hash: bool = False, python_version: str = ""
) -> list[str]:
    """Every reason the cache entry directory ``entry`` cannot install ``sha`` offline; empty when it can.

    Every wheel the lock pins must be there with its recorded size, the source must be the tree the manifest recorded and
    the lock (the one of the manifest's Python, :func:`lock_file`) the one the wheels were chosen from. With
    ``python_version`` (``"3.14"``, the interpreter that is to install them), wheels the manifest records for another
    Python are refused. The wheels' content is not read unless ``verify_hash``: ``pip`` checks
    the hash of each wheel it installs against the lock, and the builder reads them all. ``bootstrap.sh`` too is read
    only with ``verify_hash``: the script that runs this check cannot vouch for itself (see
    :mod:`hyperloom.measure.runner.bootstrap`).
    """
    manifest_file = entry / MANIFEST_NAME
    if not manifest_file.is_file():
        return [f"the cache entry {entry} has no {MANIFEST_NAME}: an unfinished build"]
    raw = manifest_file.read_bytes()
    if manifest_pin and hashlib.sha256(raw).hexdigest() != manifest_pin:
        return [f"the manifest of {entry} is not the one pinned by {MANIFEST_PIN_ENV}"]
    try:
        return _entry_problems(entry, sha, manifest_from_json(raw), verify_hash, python_version)
    except (CacheError, ValueError) as exc:
        return [str(exc)]


def requirement_line(name: str, version: str, digests: Iterable[str]) -> str:
    """One pin of a requirements file that checks hashes: ``name==version`` and a ``--hash`` for each of ``digests``."""
    return f"{name}=={version}" + "".join(f" \\\n    --hash=sha256:{digest}" for digest in digests) + "\n"


def requirements_text(manifest: Manifest) -> str:
    """The requirements file the sandbox installs from: each listed wheel pinned to the hash the manifest records."""
    return "".join(requirement_line(wheel.name, wheel.version, [wheel.sha256]) for wheel in manifest.wheels)


def check_cache(root: Path, sha: str, manifest_pin: str = "", python_version: str = "") -> list[str]:
    """Every reason the cache below ``root`` cannot install the commit ``sha`` offline, for the interpreter
    ``python_version`` when it is named; empty when it can."""
    if not is_full_commit(sha):
        return [f"the source commit {sha!r} is not a full lowercase commit hash"]
    if not root.is_dir():
        return [f"the runner cache {root} does not exist: {CACHE_ENV} names the directory the template mounts"]
    if not (root / sha).is_dir():
        return [f"the runner cache {root} has no entry for commit {sha}"]
    return check_entry(root / sha, sha, manifest_pin, python_version=python_version)


def _checked(root: str, sha: str, requirements: str) -> list[str]:
    """What is wrong with the cache; when nothing is, write the requirements file to ``requirements`` (if given)."""
    running = f"{sys.version_info[0]}.{sys.version_info[1]}"
    problems = check_cache(Path(root), sha, os.environ.get(MANIFEST_PIN_ENV, ""), running)
    if not problems and requirements:
        manifest = manifest_from_json((Path(root) / sha / MANIFEST_NAME).read_bytes())
        Path(requirements).parent.mkdir(parents=True, exist_ok=True)
        Path(requirements).write_text(requirements_text(manifest), encoding=_UTF8)
    return problems


def main(argv: Sequence[str]) -> int:
    """``python - <cache> <sha> [<requirements>]``: print what is wrong with the cache (at most a few lines) and exit 1;
    when nothing is, write the requirements file the sandbox installs from, if a path is given, and exit 0."""
    root, sha, *requirements = argv
    try:
        problems = _checked(root, sha, "".join(requirements))
    except OSError as exc:
        problems = [f"the runner cache could not be read: {exc}"]
    for problem in problems[:_SHOWN]:
        print(problem)
    if len(problems) > _SHOWN:
        print(f"... and {len(problems) - _SHOWN} more")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
