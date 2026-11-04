# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Take the declared outputs of a finished compile out of its tree.

An output is a file the recipe's globs match below the package directory. Each must be a regular ELF file no larger than
the recipe's cap, reached without a symlink anywhere on its path; the compile ran candidate-controlled code, so a
symlink, a special file or a file that is not ELF is that candidate's failure, never followed or stored. An output equal to
the image's own file (bytes and mode) is left out: the record lists only what differs from the image.
"""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from pathlib import Path

from hyperloom.measure import fsafe
from hyperloom.measure.jit.native.record import EXECUTABLE, PLAIN, OutputFile
from hyperloom.measure.jit.store import BLOBS_DIR
from hyperloom.measure.runner.cache_layout import file_sha256

__all__ = ["ELF_MAGIC", "HarvestError", "harvest"]

ELF_MAGIC = b"\x7fELF"
_BLOB_MODE = 0o600


class HarvestError(Exception):
    """The compile's outputs are not what the recipe declares: the candidate's failure."""


def harvest(
    package: Path, globs: tuple[str, ...], image: Path | None, max_bytes: int, staging: Path
) -> tuple[OutputFile, ...]:
    """The outputs below ``package`` that differ from ``image`` (every one when ``image`` is ``None``), each copied to
    ``staging/blobs/<sha256>``."""
    package = Path(package)
    found = sorted({path.relative_to(package).as_posix() for pattern in globs for path in package.glob(pattern)})
    if not found:
        raise HarvestError(f"the build made none of its declared outputs ({', '.join(globs)})")
    (Path(staging) / BLOBS_DIR).mkdir(parents=True, exist_ok=True)
    package_fd = fsafe.open_root(package)
    try:
        outputs = [_take(package_fd, relative, max_bytes, Path(staging) / BLOBS_DIR) for relative in found]
    finally:
        os.close(package_fd)
    return tuple(output for output in outputs if image is None or not _same_as_image(Path(image) / output.path, output))


def _take(package_fd: int, relative: str, max_bytes: int, blobs: Path) -> OutputFile:
    try:
        source = fsafe.open_beneath(package_fd, relative)
    except fsafe.UnsafeEntry as exc:
        raise HarvestError(
            f"{relative}: an output must be a regular file reached without a symlink ({exc.kind})"
        ) from None
    except fsafe.MissingEntry:
        raise HarvestError(f"{relative}: the output went away while it was read") from None
    try:
        info = os.fstat(source)
        if info.st_size > max_bytes:
            raise HarvestError(f"{relative}: {info.st_size} bytes is more than the {max_bytes} an output may be")
        if os.pread(source, len(ELF_MAGIC), 0) != ELF_MAGIC:
            raise HarvestError(f"{relative}: an output must be an ELF file")
        sha256, size = _copy(source, blobs, max_bytes)
    finally:
        os.close(source)
    return OutputFile(relative, sha256, size, EXECUTABLE if info.st_mode & stat.S_IXUSR else PLAIN)


def _copy(source: int, blobs: Path, max_bytes: int) -> tuple[str, int]:
    """Copy ``source`` to ``blobs/<sha256>`` while hashing it; the size is checked again on the bytes actually read."""
    pending = blobs / f".pending-{uuid.uuid4().hex}"
    digest, total = hashlib.sha256(), 0
    try:
        with open(pending, "xb") as out:
            for chunk in fsafe.stream(source):
                total += len(chunk)
                if total > max_bytes:
                    raise HarvestError(f"the output grew past {max_bytes} bytes while it was read")
                digest.update(chunk)
                out.write(chunk)
        os.chmod(pending, _BLOB_MODE)
        os.replace(pending, blobs / digest.hexdigest())
    finally:
        pending.unlink(missing_ok=True)
    return digest.hexdigest(), total


def _same_as_image(image_file: Path, output: OutputFile) -> bool:
    if image_file.is_symlink() or not image_file.is_file():
        return False
    mode = EXECUTABLE if image_file.stat().st_mode & stat.S_IXUSR else PLAIN
    if mode != output.mode or image_file.stat().st_size != output.size:
        return False
    return file_sha256(image_file) == output.sha256
