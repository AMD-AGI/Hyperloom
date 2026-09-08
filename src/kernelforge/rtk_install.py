# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Put the ``rtk`` output filter on PATH, so that :mod:`kernelforge.rtk` finds it.

:mod:`kernelforge.rtk` has always been written to no-op when ``rtk`` is absent,
which is the right behaviour and also the reason nobody noticed it was absent
everywhere: across 43 recorded end-to-end campaigns the binary appears in not
one of them. The filter was wired in and then never installed, so every one of
those runs paid full price for tool output that the code reads as trimmed.

Installing it cannot live only in Hyperloom's ``install.sh``. Forge is also run
standalone -- ``pip install -e ".[forge]"`` per its quickstart, which is how the
kernel arena reaches it -- and that path never executes a shell installer. It
cannot be a dependency of the ``forge`` extra either: ``rtk`` is a Rust binary,
and the PyPI project of that name is an unrelated Raspberry Pi GPIO library, so
``pip install rtk`` installs the wrong software. Hence this module and the
``kernelforge install-rtk`` command in front of it: one implementation both
install paths call.

The release is pinned by tag *and* by digest. The digest is what makes the
download tamper-evident -- fetching ``checksums.txt`` from the same release
would only prove the archive matches whatever that release currently serves.
Upgrading is therefore a deliberate edit of the table below, with the new
digests read off the upstream release.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import stat
import subprocess
import sysconfig
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

#: Upstream release this installer pins. Bumping it means replacing every digest
#: in :data:`_ASSETS` from that release's ``checksums.txt``.
RTK_VERSION = "v0.48.0"

_RELEASE_URL = "https://github.com/rtk-ai/rtk/releases/download/{version}/{asset}"

#: How long a single download may take. An installer that hangs is worse than
#: one that fails: ``install.sh`` treats a failure as a warning and moves on.
_DOWNLOAD_TIMEOUT_SEC = 180


@dataclass(frozen=True)
class _Asset:
    """One release archive: its file name and the digest it must hash to."""

    name: str
    sha256: str


#: ``(system, machine) -> asset``. Linux takes the musl build on x86_64 because
#: it is statically linked and so does not care which libc the ROCm image ships.
_ASSETS: dict[tuple[str, str], _Asset] = {
    ("linux", "x86_64"): _Asset(
        "rtk-x86_64-unknown-linux-musl.tar.gz",
        "e4e650fa1677c0de2f6839a6040d7b17f312d32f163c402b75af70e9e5af1a91",
    ),
    ("linux", "aarch64"): _Asset(
        "rtk-aarch64-unknown-linux-gnu.tar.gz",
        "5ed65486a96077bd6bba7c87fdc9d0e4a1918d19619be3c87380888389a30c7c",
    ),
    ("darwin", "x86_64"): _Asset(
        "rtk-x86_64-apple-darwin.tar.gz",
        "a95f2c23e08572dcc84ddff5fbe432e41e7f94369622eb086cca49ae0b6f61e8",
    ),
    ("darwin", "arm64"): _Asset(
        "rtk-aarch64-apple-darwin.tar.gz",
        "4fa025cc93a744b6963f4e53a008e5ba3f74b6a38061f4a47c639e1c3023e0db",
    ),
}

#: Machine names that mean the same architecture as the key they map to.
_MACHINE_ALIASES = {"amd64": "x86_64", "arm64": "arm64", "aarch64": "aarch64", "x86_64": "x86_64"}


class RtkInstallError(RuntimeError):
    """Raised when the binary could not be placed on disk, with the reason."""


def _current_asset() -> _Asset:
    """The release archive for the running machine.

    Raises:
        RtkInstallError: When upstream publishes nothing for this platform.
    """
    system = platform.system().lower()
    machine = _MACHINE_ALIASES.get(platform.machine().lower(), platform.machine().lower())
    # macOS reports ``arm64`` and Linux ``aarch64`` for the same silicon; the
    # table is keyed the way each platform spells it rather than normalising,
    # because the asset names differ too.
    asset = _ASSETS.get((system, machine))
    if asset is None:
        raise RtkInstallError(
            f"rtk {RTK_VERSION} publishes no build for {system}/{machine}; "
            "install it from https://github.com/rtk-ai/rtk and put it on PATH"
        )
    return asset


def default_destination() -> Path:
    """Where to write the binary so the caller's own shell will find it.

    The scripts directory of the running interpreter -- the one that already
    holds the ``kernelforge`` entry point. Whoever can run ``kernelforge`` has
    that directory on PATH by construction, which a guessed ``/usr/local/bin``
    does not guarantee and cannot write to unprivileged.
    """
    return Path(sysconfig.get_path("scripts"))


def installed_version(binary: str = "rtk") -> str:
    """The version ``binary`` reports, or ``""`` when it is absent or mute.

    Any failure reads as "not installed": this only ever decides whether to
    download, and a binary that cannot answer ``--version`` is one to replace.
    """
    path = shutil.which(binary)
    if path is None:
        return ""
    try:
        result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    # ``rtk --version`` prints "rtk 0.48.0"; the tag carries a leading "v".
    reported = result.stdout.strip().split()
    return f"v{reported[-1]}" if reported else ""


def _download(url: str, target: Path) -> None:
    """Fetch ``url`` into ``target``, failing with the URL in the message."""
    try:
        with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SEC) as response:  # noqa: S310 - pinned https URL
            target.write_bytes(response.read())
    except Exception as error:  # noqa: BLE001 - network boundary, reported not classified
        raise RtkInstallError(f"could not download {url}: {error}") from error


def _verify(archive: Path, expected: str) -> None:
    """Refuse an archive whose digest is not the pinned one."""
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != expected:
        raise RtkInstallError(f"{archive.name} hashes to {actual}, expected {expected}; refusing to install it")


def _extract_binary(archive: Path, workdir: Path) -> Path:
    """Pull the single ``rtk`` entry out of ``archive``.

    Members are matched by base name and extracted one at a time rather than
    with ``extractall``, so a path in the archive cannot decide where anything
    lands on this filesystem.
    """
    with tarfile.open(archive, "r:gz") as tar:
        member = next((entry for entry in tar.getmembers() if entry.isfile() and Path(entry.name).name == "rtk"), None)
        if member is None:
            raise RtkInstallError(f"{archive.name} contains no 'rtk' binary")
        extracted = workdir / "rtk"
        source = tar.extractfile(member)
        if source is None:
            raise RtkInstallError(f"{archive.name} entry {member.name!r} could not be read")
        with source, extracted.open("wb") as handle:
            shutil.copyfileobj(source, handle)
    return extracted


def install_rtk(destination: Path | None = None, *, force: bool = False) -> tuple[Path, str]:
    """Install the pinned ``rtk`` into ``destination``.

    Args:
        destination (Path | None): Directory to write the binary into. Defaults
            to :func:`default_destination`.
        force (bool): Download even when the pinned version is already on PATH.

    Returns:
        tuple[Path, str]: The binary's path and one of ``"installed"`` or
        ``"present"`` -- the latter when the pinned version was already there
        and nothing was downloaded.

    Raises:
        RtkInstallError: On an unsupported platform, a failed download, a digest
            mismatch, or a destination that cannot be written.
    """
    if not force:
        current = installed_version()
        if current == RTK_VERSION:
            existing = shutil.which("rtk")
            # ``which`` found it a line ago; the guard is for the race, not the logic.
            if existing is not None:
                return Path(existing), "present"

    asset = _current_asset()
    target_dir = Path(destination) if destination is not None else default_destination()
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RtkInstallError(f"cannot create {target_dir}: {error}") from error
    if not os.access(target_dir, os.W_OK):
        raise RtkInstallError(f"{target_dir} is not writable; pass a directory on PATH that is")

    with tempfile.TemporaryDirectory(prefix="rtk-install-") as tmp:
        workdir = Path(tmp)
        archive = workdir / asset.name
        _download(_RELEASE_URL.format(version=RTK_VERSION, asset=asset.name), archive)
        _verify(archive, asset.sha256)
        binary = _extract_binary(archive, workdir)
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        final = target_dir / "rtk"
        try:
            # Replace rather than write in place: an ``rtk`` mid-execution keeps
            # the inode it started with instead of reading a half-written file.
            shutil.move(str(binary), str(final))
        except OSError as error:
            raise RtkInstallError(f"cannot write {final}: {error}") from error
    return final, "installed"
