# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner cache: what the sealed runner installs from, and the one check of it that the builder and the runner share.

A measurement sandbox has no network. Its source and its Python wheels come from a cache directory on shared storage
that the sandbox mounts; :mod:`hyperloom.measure.runner.cache_build` fills it on a machine that has egress and the
start-up script (:mod:`hyperloom.measure.runner.bootstrap`) installs from it with no index. One commit's entry is::

    <cache>/<sha>/source.tar   the repository tree at that commit, the ``git archive`` the builder makes with fixed
                               settings and none of its own attributes (the commit decides the bytes), runner.lock
                               included
    <cache>/<sha>/source/      that archive unpacked; the builder reads it, the sandbox never does
    <cache>/<sha>/wheels/      one wheel per pin of ``requirements/runner.lock``
    <cache>/<sha>/bootstrap.sh the start-up script (``BOOTSTRAP_SCRIPT``) of the builder that made the entry, mode 0755;
                               a template step runs it with ``sh``
    <cache>/<sha>/manifest     JSON, written last: the commit and its tree, the size and SHA-256 of ``source.tar``, the
                               digest of ``source``, the SHA-256 of ``bootstrap.sh`` and, for each wheel, its pin, file,
                               size and SHA-256 (see :class:`WheelEntry`)

The sandbox runs nothing from the cache itself (:func:`stage_entry`). It reads the manifest once; it reads
``source.tar`` once, hashing the bytes as it reads them, and unpacks those same bytes below its own scratch directory
only when they are the archive the manifest records; it reads the lock from that copy, copies every wheel the manifest
lists next to it, hashing each as it copies, and writes the requirements pip installs from the manifest it read. A file
of the cache that changes after that check never reaches the runner or pip, whoever can write the cache. An entry whose
manifest records no ``source.tar`` (an older builder made it) is refused.

The sandbox installs exactly the wheels the manifest lists, each pinned to the hash the manifest records
(:func:`requirements_text`); the builder has checked every one of them against the lock. An entry without a manifest is
an unfinished build, and nothing uses it. ``<cache>`` is the directory
``HYPERLOOM_RUNNER_CACHE`` names, on the builder's machine as well as inside the sandbox (the mount point may differ).

This module is the script's checker as well: the start-up script embeds this file's text and runs it with the sandbox's
own interpreter before it does anything else, so it uses nothing but the standard library and the Python of the
framework image (3.10). That interpreter must have tarfile's extraction filters (Python 3.10.12, 3.11.4, 3.12 or newer):
on an older one the check refuses every entry and says so (:func:`_filter_problems`).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

__all__ = [
    "BOOTSTRAP_NAME",
    "CACHE_ENV",
    "DEFAULT_CACHE_DIR",
    "LOCAL_SOURCE_DIR",
    "LOCAL_WHEELS_DIR",
    "LOCK_FILE",
    "MANIFEST_NAME",
    "MANIFEST_PIN_ENV",
    "REQUIREMENTS_FILE",
    "SOURCE_ARCHIVE",
    "SOURCE_DIR",
    "WHEELS_DIR",
    "CacheError",
    "FileDigest",
    "LockPin",
    "Manifest",
    "TreeDigest",
    "WheelEntry",
    "canonical_name",
    "check_cache",
    "check_entry",
    "file_sha256",
    "is_full_commit",
    "manifest_from_json",
    "manifest_to_json",
    "parse_lock",
    "requirement_line",
    "requirements_text",
    "stage_entry",
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
SOURCE_ARCHIVE = "source.tar"
SOURCE_DIR = "source"
#: Below the sandbox's scratch directory: the source unpacked from the checked archive, and the checked wheels.
LOCAL_SOURCE_DIR = "hl-source"
LOCAL_WHEELS_DIR = "hl-wheels"
WHEELS_DIR = "wheels"
LOCK_FILE = "requirements/runner.lock"
_SCHEMA = 1
_UTF8 = "utf-8"
_SHOWN = 5
_CHUNK = 1 << 20
#: The Pythons whose tarfile has the ``data`` extraction filter that :func:`_unpack` needs.
_FILTER_PYTHONS = "3.10.12, 3.11.4, 3.12 or newer"
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
class FileDigest:
    """The size and SHA-256 of one file's bytes."""

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
    """What a finished build records. ``git_tree`` is the commit's tree object, ``target`` what the wheels are for,
    ``bootstrap_sha256`` the hash of the entry's ``bootstrap.sh`` (empty in a manifest written before entries had one)
    and ``source_archive`` the size and hash of its ``source.tar`` (``None`` in one written before entries had that)."""

    sha: str
    git_tree: str
    source: TreeDigest
    wheels: tuple[WheelEntry, ...]
    target: dict[str, Any]
    bootstrap_sha256: str = ""
    source_archive: FileDigest | None = None


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


def _copy_hashed(path: Path, sink: BinaryIO) -> FileDigest:
    """Copy the file ``path`` into ``sink`` in chunks; the size and SHA-256 of the bytes copied, which are read once."""
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
            sink.write(chunk)
            size += len(chunk)
    return FileDigest(size, digest.hexdigest())


def parse_lock(text: str) -> list[LockPin]:
    """The pins of a ``pip-compile --generate-hashes`` file.

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
            raise CacheError(f"{LOCK_FILE} line {number} is not a pin or a hash")
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
        archive = data.get("source_archive")
        return Manifest(
            data["sha"],
            data["git_tree"],
            TreeDigest(**data["source"]),
            tuple(WheelEntry(**wheel) for wheel in data["wheels"]),
            dict(data["target"]),
            str(data.get("bootstrap_sha256", "")),
            None if archive is None else FileDigest(**archive),
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


def _older_builder(entry: Path) -> str:
    return (
        f"the manifest of {entry} records no {SOURCE_ARCHIVE}: an older Hyperloom built this entry, and the runner now "
        f"runs only from a checked archive; build the commit again with this Hyperloom, into a {CACHE_ENV} directory of "
        "its own"
    )


def _archive_problems(entry: Path, manifest: Manifest, verify_hash: bool) -> list[str]:
    archive, recorded = entry / SOURCE_ARCHIVE, manifest.source_archive
    if recorded is None:
        return [_older_builder(entry)]
    if not archive.is_file() or archive.stat().st_size != recorded.size:
        return [f"{archive} is missing or is not the size the manifest records"]
    if verify_hash and file_sha256(archive) != recorded.sha256:
        return [f"{archive} does not match the hash in the manifest"]
    return []


def _entry_pins(source: Path) -> list[LockPin]:
    """The pins of the entry's lock in the source tree ``source``.

    Raises:
        CacheError: When the tree has no lock, or it is not a lock.
    """
    if not (source / LOCK_FILE).is_file():
        raise CacheError(f"the source in {source} has no {LOCK_FILE}")
    return parse_lock((source / LOCK_FILE).read_text(encoding=_UTF8))


def _entry_problems(entry: Path, manifest: Manifest, verify_hash: bool) -> list[str]:
    source, wheels = entry / SOURCE_DIR, entry / WHEELS_DIR
    if not source.is_dir() or not wheels.is_dir():
        return [f"the cache entry {entry} lacks its {SOURCE_DIR} or {WHEELS_DIR} directory"]
    pins = _entry_pins(source)
    problems = _archive_problems(entry, manifest, verify_hash) + _source_problems(source, manifest)
    problems += _wheels_problems(wheels, manifest, pins, verify_hash)
    return problems + (_bootstrap_problems(entry, manifest) if verify_hash else [])


def check_entry(entry: Path, sha: str, manifest_pin: str = "", verify_hash: bool = False) -> list[str]:
    """Every reason the cache entry directory ``entry`` is not a whole build of ``sha``, as it lies in the cache; empty
    when it is. This is the builder's check; the sandbox's is :func:`stage_entry`.

    Every wheel the lock pins must be there with its recorded size, ``source.tar`` must be there with its recorded size,
    the source must be the tree the manifest recorded and the lock the one the wheels were chosen from. The content of
    the wheels and of ``source.tar`` is not read unless ``verify_hash``, and ``bootstrap.sh`` is read only then: the
    script that runs the sandbox's check cannot vouch for itself (see :mod:`hyperloom.measure.runner.bootstrap`).
    """
    problems, manifest = _pinned_manifest(entry, sha, manifest_pin)
    if manifest is None:
        return problems
    try:
        return _entry_problems(entry, manifest, verify_hash)
    except (CacheError, ValueError) as exc:
        return [str(exc)]


def _pinned_manifest(entry: Path, sha: str, manifest_pin: str) -> tuple[list[str], Manifest | None]:
    """The entry's manifest, read once, when it is there, is the one ``manifest_pin`` names (if it names one) and is of
    commit ``sha``; otherwise the reasons it is not, and ``None``. What a caller checks and uses is this object, never
    the file as it is later."""
    manifest_file = entry / MANIFEST_NAME
    if not manifest_file.is_file():
        return [f"the cache entry {entry} has no {MANIFEST_NAME}: an unfinished build"], None
    raw = manifest_file.read_bytes()
    if manifest_pin and hashlib.sha256(raw).hexdigest() != manifest_pin:
        return [f"the manifest of {entry} is not the one pinned by {MANIFEST_PIN_ENV}"], None
    try:
        manifest = manifest_from_json(raw)
    except CacheError as exc:
        return [str(exc)], None
    if manifest.sha != sha:
        return [f"the manifest of {entry} is for commit {manifest.sha}, not {sha}"], None
    return [], manifest


def requirement_line(name: str, version: str, digests: Iterable[str]) -> str:
    """One pin of a requirements file that checks hashes: ``name==version`` and a ``--hash`` for each of ``digests``."""
    return f"{name}=={version}" + "".join(f" \\\n    --hash=sha256:{digest}" for digest in digests) + "\n"


def requirements_text(manifest: Manifest) -> str:
    """The requirements file the sandbox installs from: each listed wheel pinned to the hash the manifest records."""
    return "".join(requirement_line(wheel.name, wheel.version, [wheel.sha256]) for wheel in manifest.wheels)


def _entry_dir_problems(root: Path, sha: str) -> list[str]:
    if not is_full_commit(sha):
        return [f"the source commit {sha!r} is not a full lowercase commit hash"]
    if not root.is_dir():
        return [f"the runner cache {root} does not exist: {CACHE_ENV} names the directory the template mounts"]
    if not (root / sha).is_dir():
        return [f"the runner cache {root} has no entry for commit {sha}"]
    return []


def check_cache(root: Path, sha: str, manifest_pin: str = "") -> list[str]:
    """Every reason the cache below ``root`` has no whole entry of the commit ``sha``; empty when it has
    (:func:`check_entry`)."""
    return _entry_dir_problems(root, sha) or check_entry(root / sha, sha, manifest_pin)


def _filter_problems() -> list[str]:
    """Why this interpreter cannot unpack ``source.tar`` safely: its tarfile has no ``data`` filter (Python before
    3.10.12, 3.11.4 or 3.12), where ``extractall(filter=...)`` would fail with a ``TypeError``; empty when it has one."""
    if hasattr(tarfile, "data_filter"):
        return []
    version = ".".join(map(str, sys.version_info[:3]))
    return [
        f"this Python is {version}, whose tarfile has no data filter, and {SOURCE_ARCHIVE} is unpacked only with it: "
        f"the runner needs Python {_FILTER_PYTHONS} (HYPERLOOM_BASE_PYTHON)"
    ]


def _unpack(archive: io.BytesIO, destination: Path) -> None:
    """Unpack the archive's bytes into the new directory ``destination`` with tarfile's ``data`` filter, which refuses a
    member or a link that would land outside it and strips the leading ``/`` of an absolute member name (the filter is
    in Python 3.10.12, 3.11.4, 3.12 and newer; :func:`_staged_source` refuses before this on an older one)."""
    destination.mkdir(parents=True)
    archive.seek(0)
    with tarfile.open(fileobj=archive, mode="r:") as tar:
        tar.extractall(destination, filter="data")


def _staged_source(entry: Path, manifest: Manifest, destination: Path) -> list[str]:
    """Unpack the entry's ``source.tar`` into ``destination`` when its bytes, read once, are the archive the manifest
    records; the reasons they are not."""
    archive, recorded = entry / SOURCE_ARCHIVE, manifest.source_archive
    if recorded is None:
        return [_older_builder(entry)]
    if not archive.is_file():
        return [f"the cache entry {entry} has no {SOURCE_ARCHIVE}"]
    unsafe = _filter_problems()
    if unsafe:
        return unsafe
    copy = io.BytesIO()
    found = _copy_hashed(archive, copy)
    if found != recorded:
        return [
            f"{archive} is not the archive the manifest records: {found.size} bytes with SHA-256 {found.sha256}, "
            f"the manifest has {recorded.size} bytes with {recorded.sha256}"
        ]
    _unpack(copy, destination)
    return []


def _staged_wheels(wheels: Path, manifest: Manifest, destination: Path) -> list[str]:
    """Copy every wheel the manifest lists from ``wheels`` into ``destination``, each hashed as it is copied; the ones
    whose bytes are not what the manifest records."""
    destination.mkdir(parents=True)
    problems: list[str] = []
    for wheel in manifest.wheels:
        with (destination / Path(wheel.file).name).open("wb") as sink:
            found = _copy_hashed(wheels / wheel.file, sink)
        if found != FileDigest(wheel.size, wheel.sha256):
            problems.append(
                f"wheel {wheel.file} for {wheel.name}=={wheel.version} does not match the hash in the manifest"
            )
    return problems


def _stage(entry: Path, manifest: Manifest, scratch: Path) -> list[str]:
    wheels, source = entry / WHEELS_DIR, scratch / LOCAL_SOURCE_DIR
    if not wheels.is_dir():
        return [f"the cache entry {entry} lacks its {WHEELS_DIR} directory"]
    problems = _staged_source(entry, manifest, source)
    if problems:
        return problems
    pins = _entry_pins(source)
    problems = _wheels_problems(wheels, manifest, pins, verify_hash=False)
    problems = problems or _staged_wheels(wheels, manifest, scratch / LOCAL_WHEELS_DIR)
    if not problems:
        (scratch / REQUIREMENTS_FILE).write_text(requirements_text(manifest), encoding=_UTF8)
    return problems


def _remove(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def stage_entry(root: Path, sha: str, scratch: Path, manifest_pin: str = "") -> list[str]:
    """Check the cache entry of ``sha`` below ``root`` on the sandbox's own disk, and stage what the runner runs from it
    below ``scratch``; every reason it cannot, empty when it did.

    The manifest is read once (:func:`_pinned_manifest`); ``source.tar`` is read once and unpacked into
    :data:`LOCAL_SOURCE_DIR` only when those bytes are the archive the manifest records; the lock is read from that copy;
    every wheel the manifest lists is copied into :data:`LOCAL_WHEELS_DIR`, hashed as it is copied; and
    :data:`REQUIREMENTS_FILE` pins those wheels to the hashes of the manifest that was read. Nothing of the cache is read
    after that, so what the runner imports and pip installs is what was checked. When anything is wrong, none of the
    three is left below ``scratch``.
    """
    made = [scratch / LOCAL_SOURCE_DIR, scratch / LOCAL_WHEELS_DIR, scratch / REQUIREMENTS_FILE]
    _remove(made)
    staged = False
    try:
        problems = _entry_dir_problems(root, sha)
        if not problems:
            problems, manifest = _pinned_manifest(root / sha, sha, manifest_pin)
            if manifest is not None:
                problems = _stage(root / sha, manifest, scratch)
        staged = not problems
        return problems
    except (CacheError, ValueError, tarfile.TarError) as exc:
        return [str(exc)]
    finally:
        if not staged:
            _remove(made)


def _checked(root: str, sha: str, scratch: str) -> list[str]:
    """What is wrong with the cache: the sandbox's check, which also stages what the runner runs (:func:`stage_entry`),
    when ``scratch`` is given, else the check of the entry as it lies in the cache (:func:`check_cache`)."""
    pin = os.environ.get(MANIFEST_PIN_ENV, "")
    if scratch:
        return stage_entry(Path(root), sha, Path(scratch), pin)
    return check_cache(Path(root), sha, pin)


def main(argv: Sequence[str]) -> int:
    """``python - <cache> <sha> [<scratch>]``: print what is wrong with the cache (at most a few lines) and exit 1; when
    nothing is, stage what the runner runs below ``scratch`` if it is given (:func:`stage_entry`) and exit 0."""
    root, sha, *scratch = argv
    try:
        problems = _checked(root, sha, "".join(scratch))
    except OSError as exc:
        problems = [f"the runner cache could not be read: {exc}"]
    for problem in problems[:_SHOWN]:
        print(problem)
    if len(problems) > _SHOWN:
        print(f"... and {len(problems) - _SHOWN} more")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
