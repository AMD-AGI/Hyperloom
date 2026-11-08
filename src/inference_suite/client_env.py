# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX client environment: a pinned relocatable CPython with exactly the wheels ``agentx-client.lock`` pins.

The harness and its whole dependency tree are pinned in ``agentx-client.lock``, shipped beside this module, one wheel
and its sha256 per distribution. :func:`fetch_inputs` downloads the interpreter build
(:data:`~inference_suite.pins.AGENTX_PYTHON_BUILD`, refused unless its sha256 is the pin's) and those wheels by their
hashes; :func:`install` unpacks the build into ``<root>/agentx-python`` and installs the wheels with ``--no-deps
--require-hashes`` from those inputs alone, so two hosts that install from one lock hold the same distributions, then
patches the harness's aiperf to export ``error_osl`` (:mod:`inference_suite.aiperf_patch`). The
build is relocatable and the client runs as ``agentx-python/bin/python3 -m ...``, never through a script's shebang,
so the environment can be built in one directory and renamed into another.

:func:`client_identity` is the environment's normalised identity: each distribution installed, by name and version,
with the sha256 the lock pins for its wheel, and the interpreter by version and build sha256. The distributions the
build ships itself (pip) are part of the build and named by its sha256.

``python -m inference_suite.client_env lock --requirements <file>`` writes a lock for the requirements InferenceX pins;
the lock's header says how it was made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from inference_suite import aiperf_patch
from inference_suite.contracts.normalised import NormalisedManifest, PackageIdentity, canonical_name
from inference_suite.pins import AGENTX_PYTHON_BUILD, INFERENCEX_CLIENT_COMMIT, PinnedArchive

#: The lock shipped beside this module, and the name it is staged under.
AGENTX_LOCK = Path(__file__).with_name("agentx-client.lock")
#: The client environment below a staged root: the unpacked build, the interpreter the load runs, what was installed.
CLIENT_DIR = "agentx-python"
CLIENT_PYTHON = f"{CLIENT_DIR}/bin/python3"
BUILD_FILE = f"{CLIENT_DIR}/BUILD.json"
STAGED_LOCK = f"{CLIENT_DIR}/{AGENTX_LOCK.name}"
_SITE_PACKAGES = f"lib/python{'.'.join(AGENTX_PYTHON_BUILD.version.split('.')[:2])}/site-packages"
#: The client's files staging patches, below a staged root.
PATCHED_FILES = tuple(f"{CLIENT_DIR}/{_SITE_PACKAGES}/{relative}" for relative in aiperf_patch.PATCHED_FILES)
#: Inside an inputs directory: the interpreter build and the lock's wheels.
INPUT_ARCHIVE = "python-build.tar.gz"
INPUT_WHEELS = "wheels"
#: The platforms the lock's wheels are chosen for: glibc 2.17 to 2.28 on x86_64, what the measurement images run.
LOCK_PLATFORMS = (*(f"manylinux_2_{minor}_x86_64" for minor in range(28, 16, -1)), "manylinux2014_x86_64")
#: The directory the build's archive unpacks to.
_ARCHIVE_ROOT = "python"
_LOCK_LINE = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>\S+) --hash=sha256:(?P<sha256>[0-9a-f]{64})"
)
_DIST_INFO = re.compile(r"(?P<name>.+)-(?P<version>[^-]+)\.dist-info")
_PIP = ("-I", "-m", "pip")
_PIP_QUIET = ("--disable-pip-version-check", "--no-input", "--isolated", "--quiet")
_BINARY_ONLY = "--only-binary=:all:"
#: Exactly the lock's wheels: no dependency resolved beyond it, each file checked against its pinned sha256.
_LOCKED = ("--no-deps", "--require-hashes", _BINARY_ONLY)
_CHUNK = 1 << 20
_TAIL = 2000
_LOCK_HEADER = """\
# The AgentX client's full dependency tree: one wheel per distribution, by sha256. Staging installs it with --no-deps.
# Made for CPython {version} (build sha256 {build}), glibc 2.28 x86_64, from {requirements} of InferenceX {pin}.
# To regenerate, with network, after either pin moves:
#   python -m inference_suite.client_env lock --requirements <InferenceX>/inferencex-e2e/{requirements} \\
#       > src/inference_suite/{lock}
"""


class ClientEnvError(ValueError):
    """The client environment cannot be fetched, installed or identified; the message says why."""


@dataclass(frozen=True)
class LockPin:
    """One distribution the lock pins: its name, version and the sha256 of its wheel."""

    name: str
    version: str
    sha256: str


def read_lock(text: str) -> tuple[LockPin, ...]:
    """The pins of a lock's ``text``, one ``name==version --hash=sha256:<hex>`` line each; ``#`` lines are comments.

    Raises:
        ClientEnvError: A line is not a pin, or a distribution is pinned twice.
    """
    pins: dict[str, LockPin] = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        match = _LOCK_LINE.fullmatch(line.strip())
        if match is None:
            raise ClientEnvError(f"the lock line {line!r} is not name==version --hash=sha256:<hex>")
        name = canonical_name(match["name"])
        if name in pins:
            raise ClientEnvError(f"the lock pins {name} twice")
        pins[name] = LockPin(name, match["version"], match["sha256"])
    return tuple(pins.values())


def lock_text(report: Mapping[str, object], requirements: str) -> str:
    """The lock for what a ``pip install --dry-run --report`` resolved: each distribution with its wheel's sha256."""
    items = report.get("install")
    if not isinstance(items, list):
        raise ClientEnvError("the pip report lists nothing to install")
    lines = []
    for item in items:
        metadata, archive = item["metadata"], item["download_info"]["archive_info"]
        lines.append(
            f"{canonical_name(metadata['name'])}=={metadata['version']} --hash=sha256:{archive['hashes']['sha256']}"
        )
    header = _LOCK_HEADER.format(
        version=AGENTX_PYTHON_BUILD.version,
        build=AGENTX_PYTHON_BUILD.sha256,
        requirements=requirements,
        pin=INFERENCEX_CLIENT_COMMIT,
        lock=AGENTX_LOCK.name,
    )
    return header + "".join(f"{line}\n" for line in sorted(lines))


def _run(argv: Sequence[str], what: str) -> str:
    try:
        done = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ClientEnvError(f"{what} failed: {exc}") from exc
    if done.returncode != 0:
        raise ClientEnvError(f"{what} exited with code {done.returncode}: {done.stderr.strip()[-_TAIL:]}")
    return done.stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as sink:
        shutil.copyfileobj(response, sink, _CHUNK)


def _checked_archive(archive: Path, build: PinnedArchive) -> None:
    actual = _sha256(archive)
    if actual != build.sha256:
        raise ClientEnvError(f"{archive} has sha256 {actual}; the interpreter build is pinned at {build.sha256}")


def _unpack(archive: Path, destination: Path) -> Path:
    """Unpack the build into the new directory ``destination``; its interpreter, checked to be the pin's version."""
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".unpack-") as scratch:
        _run(["tar", "-x", "--no-same-owner", "-z", "-f", str(archive), "-C", scratch], f"unpacking {archive}")
        os.rename(Path(scratch, _ARCHIVE_ROOT), destination)
    python = destination / "bin" / "python3"
    version = _run([str(python), "-I", "-c", "import platform; print(platform.python_version())"], "the interpreter")
    if version.strip() != AGENTX_PYTHON_BUILD.version:
        raise ClientEnvError(f"the interpreter build is Python {version.strip()}, not {AGENTX_PYTHON_BUILD.version}")
    return python


def _platform_args() -> list[str]:
    major_minor = ".".join(AGENTX_PYTHON_BUILD.version.split(".")[:2])
    abi = "cp" + major_minor.replace(".", "")
    args = ["--python-version", major_minor, "--implementation", "cp"]
    args += [part for tag in (abi, "abi3", "none") for part in ("--abi", tag)]
    return args + [part for platform in LOCK_PLATFORMS for part in ("--platform", platform)]


def fetch_inputs(inputs: Path) -> None:
    """Download the pinned interpreter build and the lock's wheels into the empty directory ``inputs``.

    Raises:
        ClientEnvError: The build is not the pinned one, or a wheel cannot be had at the hash the lock pins.
    """
    archive = inputs / INPUT_ARCHIVE
    _download(AGENTX_PYTHON_BUILD.url, archive)
    _checked_archive(archive, AGENTX_PYTHON_BUILD)
    with tempfile.TemporaryDirectory(dir=inputs, prefix=".python-") as scratch:
        python = _unpack(archive, Path(scratch, _ARCHIVE_ROOT))
        download = [str(python), *_PIP, "download", *_PIP_QUIET, *_LOCKED]
        download += [*_platform_args(), "--dest", str(inputs / INPUT_WHEELS), "-r", str(AGENTX_LOCK)]
        _run(download, "downloading the wheels the AgentX client lock pins")


def _installed(site: Path) -> dict[str, str]:
    """Each distribution installed in ``site``, by canonical name, with its version."""
    installed = {}
    for entry in site.glob("*.dist-info"):
        match = _DIST_INFO.fullmatch(entry.name)
        if match is not None:
            installed[canonical_name(match["name"])] = match["version"]
    return installed


def _site_packages(environment: Path) -> Path:
    return environment / _SITE_PACKAGES


def install(inputs: Path, root: Path) -> None:
    """Build the client environment in ``root/agentx-python`` from ``inputs`` alone, fetching nothing, and patch its
    aiperf to export ``error_osl``.

    Raises:
        ClientEnvError: The build in ``inputs`` is not the pinned one, a wheel the lock pins is absent from it or
            not at the lock's hash, or the installed aiperf is not the text the patch anchors on.
    """
    archive = inputs / INPUT_ARCHIVE
    _checked_archive(archive, AGENTX_PYTHON_BUILD)
    environment = root / CLIENT_DIR
    python = _unpack(archive, environment)
    bundled = sorted(_installed(_site_packages(environment)))
    shutil.copyfile(AGENTX_LOCK, root / STAGED_LOCK)
    install_args = [
        str(python),
        *_PIP,
        "install",
        *_PIP_QUIET,
        "--no-index",
        "--find-links",
        str(inputs / INPUT_WHEELS),
    ]
    install_args += [*_LOCKED, "-r", str(root / STAGED_LOCK)]
    _run(install_args, "installing the AgentX client from its lock")
    try:
        aiperf_patch.patch_aiperf(_site_packages(environment))
    except aiperf_patch.AiperfPatchError as exc:
        raise ClientEnvError(f"the installed AgentX client cannot be patched to export error_osl: {exc}") from exc
    build = {"version": AGENTX_PYTHON_BUILD.version, "sha256": AGENTX_PYTHON_BUILD.sha256, "bundled": bundled}
    (root / BUILD_FILE).write_text(json.dumps(build, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def client_identity(root: Path, inferencex: str) -> NormalisedManifest:
    """The normalised identity of the client environment staged under ``root`` at the InferenceX pin ``inferencex``.

    Raises:
        ClientEnvError: The environment holds a distribution its lock does not pin, lacks one it does, or holds one at
            another version, or its build record or lock does not read.
    """
    try:
        build = json.loads((root / BUILD_FILE).read_text(encoding="utf-8"))
        pins = read_lock((root / STAGED_LOCK).read_text(encoding="utf-8"))
        version, sha256, bundled = build["version"], build["sha256"], set(build["bundled"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ClientEnvError(f"the AgentX client staged in {root} cannot be read: {exc}") from exc
    installed = _installed(_site_packages(root / CLIENT_DIR))
    pinned = {pin.name: pin.version for pin in pins}
    found = {name: version for name, version in installed.items() if name not in bundled}
    if found != pinned:
        differ = sorted(name for name in found.keys() | pinned.keys() if found.get(name) != pinned.get(name))
        raise ClientEnvError(f"the AgentX client in {root} is not what its lock pins: {', '.join(differ)}")
    packages = tuple(PackageIdentity(pin.name, pin.version, pin.sha256) for pin in pins)
    return NormalisedManifest(inferencex, version, sha256, packages)


def staged_files(root: Path) -> Iterator[Path]:
    """The files of the client environment a staging manifest names: the interpreter, the build record, the lock,
    every installed distribution's ``RECORD`` and the files staging patched."""
    yield from (root / CLIENT_PYTHON, root / BUILD_FILE, root / STAGED_LOCK, *(root / f for f in PATCHED_FILES))
    yield from sorted(_site_packages(root / CLIENT_DIR).glob("*.dist-info/RECORD"))


def _generate_lock(requirements: Path, label: str) -> str:
    with tempfile.TemporaryDirectory(prefix="agentx-lock-") as scratch:
        work = Path(scratch)
        archive = work / INPUT_ARCHIVE
        _download(AGENTX_PYTHON_BUILD.url, archive)
        _checked_archive(archive, AGENTX_PYTHON_BUILD)
        python = _unpack(archive, work / _ARCHIVE_ROOT)
        report = work / "report.json"
        resolve = [str(python), *_PIP, "install", *_PIP_QUIET, "--dry-run", "--ignore-installed", _BINARY_ONLY]
        resolve += [
            "--target",
            str(work / "target"),
            *_platform_args(),
            "--report",
            str(report),
            "-r",
            str(requirements),
        ]
        _run(resolve, f"resolving {requirements}")
        return lock_text(json.loads(report.read_text(encoding="utf-8")), label)


def main(argv: Sequence[str] | None = None) -> int:
    """``lock --requirements <file>`` prints the lock that file resolves to under the pinned interpreter."""
    parser = argparse.ArgumentParser(prog="python -m inference_suite.client_env")
    commands = parser.add_subparsers(dest="command", required=True)
    lock_parser = commands.add_parser("lock", help="resolve the AgentX client's requirements into a lock")
    lock_parser.add_argument("--requirements", type=Path, required=True, help="the requirements InferenceX pins")
    args = parser.parse_args(argv)
    label = "/".join(args.requirements.parts[-4:])
    try:
        sys.stdout.write(_generate_lock(args.requirements, label))
    except ClientEnvError as exc:
        print(f"no lock: {exc}", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "AGENTX_LOCK",
    "BUILD_FILE",
    "CLIENT_DIR",
    "CLIENT_PYTHON",
    "INPUT_ARCHIVE",
    "INPUT_WHEELS",
    "LOCK_PLATFORMS",
    "PATCHED_FILES",
    "STAGED_LOCK",
    "ClientEnvError",
    "LockPin",
    "client_identity",
    "fetch_inputs",
    "install",
    "lock_text",
    "main",
    "read_lock",
    "staged_files",
]


if __name__ == "__main__":
    sys.exit(main())
