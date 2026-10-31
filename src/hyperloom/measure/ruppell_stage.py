# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage a ruppell tree into the runner cache, where a profile test's sandbox finds it without network.

Run it where the runner cache is written, on a machine with egress, as ``cache_build`` and the prompt sets are::

    python -m hyperloom.measure.ruppell_stage --commit <sha> --archive <tar.gz path or URL> [--token-file F] \\
        [--cache $HYPERLOOM_RUNNER_CACHE]

The archive is ``git archive``'s tarball of the commit (the board serves it at
``<board>/api/v1/repos/boards/ruppell/archive/<sha>.tar.gz``; ``--token-file`` holds the token a URL needs, sent as
``Authorization: token``). The entry (:func:`~hyperloom.measure.ruppell.entry_dir`) is the tree with ``.commit`` added,
every file mode 0644 or, when executable, 0755, then its manifest, written last, the whole made in a directory of its own
that becomes the entry in one rename. An entry already there is checked, not rewritten.

The tree of the commit a profile test pins (:data:`~hyperloom.measure.ruppell.RUPPELL_COMMIT`) must have the pinned
digest; another commit is staged with the digest it has, which is printed, for a change that moves the pin.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import tarfile
import urllib.request
import uuid
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from hyperloom.measure.ruppell import (
    COMMIT_FILE,
    MANIFEST_NAME,
    RUPPELL_COMMIT,
    RUPPELL_TREE_SHA256,
    SOURCE_DIR,
    entry_dir,
    entry_manifest,
    entry_problem,
)
from hyperloom.measure.runner.cache_layout import CACHE_ENV, DEFAULT_CACHE_DIR, is_full_commit, tree_digest

__all__ = ["StageError", "main", "stage"]

_FETCH_TIMEOUT_SEC = 300
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
_UTF8 = "utf-8"


class StageError(Exception):
    """The archive is not the commit's tree, or the entry in place is another one."""


def _archive_bytes(archive: str, token_file: str | None) -> bytes:
    if "://" not in archive:
        return Path(archive).read_bytes()
    request = urllib.request.Request(archive)
    if token_file:
        token = Path(token_file).read_text(encoding=_UTF8).strip()
        request.add_header("Authorization", f"token {token}")
    with urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_SEC) as response:
        raw = response.read(_MAX_ARCHIVE_BYTES + 1)
    if len(raw) > _MAX_ARCHIVE_BYTES:
        raise StageError(f"{archive} is larger than {_MAX_ARCHIVE_BYTES} bytes")
    return raw


def _member_path(name: str) -> PurePosixPath | None:
    """The path of a member below the archive's one top directory, ``None`` for that directory itself.

    Raises:
        StageError: The path is absolute or climbs out.
    """
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise StageError(f"the archive holds an unsafe path: {name}")
    return PurePosixPath(*path.parts[1:]) if len(path.parts) > 1 else None


def _extract(raw: bytes, commit: str, source: Path) -> None:
    """Write the archive's files below ``source``: regular files and directories only, the top directory stripped."""
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        recorded = archive.pax_headers.get("comment")
        if recorded and recorded != commit:
            raise StageError(f"the archive is of {recorded}, not {commit}")
        members = archive.getmembers()
        if len({PurePosixPath(m.name).parts[0] for m in members if m.name}) != 1:
            raise StageError("the archive does not hold one top directory")
        for member in members:
            relative = _member_path(member.name)
            if relative is None:
                continue
            target = source / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True, mode=0o755)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                handle = archive.extractfile(member)
                if handle is None:
                    raise StageError(f"{member.name} cannot be read")
                target.write_bytes(handle.read())
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
            else:
                raise StageError(f"the archive holds {member.name}, which is not a file or a directory")


def stage(cache: Path, commit: str, raw: bytes) -> tuple[Path, str]:
    """Stage the tree of ``commit`` from the tarball ``raw`` below ``cache``; return the entry and its tree's digest.

    Raises:
        StageError: The archive is not the commit's, the pinned commit's tree is not the pinned one, or an entry in place
            differs from this one.
    """
    if not is_full_commit(commit):
        raise StageError(f"{commit!r} is not a full commit hash")
    final = entry_dir(cache, commit)
    incoming = final.parent / f".incoming-{uuid.uuid4().hex}"
    try:
        source = incoming / SOURCE_DIR
        source.mkdir(parents=True, mode=0o755)
        _extract(raw, commit, source)
        (source / COMMIT_FILE).write_text(commit + "\n", encoding=_UTF8)
        (source / COMMIT_FILE).chmod(0o644)
        digest = tree_digest(source)
        if commit == RUPPELL_COMMIT and digest.sha256 != RUPPELL_TREE_SHA256:
            raise StageError(f"the tree of {commit} has digest {digest.sha256}, not the pinned {RUPPELL_TREE_SHA256}")
        manifest = entry_manifest(commit, digest.files, digest.size, digest.sha256)
        if final.exists():
            problem = entry_problem(final, commit, digest.sha256)
            if problem or (final / MANIFEST_NAME).read_bytes() != manifest:
                raise StageError(f"the entry {final} exists and is not this one: {problem or 'another manifest'}")
            return final, digest.sha256
        (incoming / MANIFEST_NAME).write_bytes(manifest)
        try:
            os.rename(incoming, final)
        except OSError:
            if not final.exists():
                raise
    finally:
        shutil.rmtree(incoming, ignore_errors=True)
    return final, digest.sha256


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hyperloom.measure.ruppell_stage", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--commit", default=RUPPELL_COMMIT, help="the full commit hash (default: the pinned one)")
    parser.add_argument("--archive", required=True, help="the commit's tar.gz, a path or a URL")
    parser.add_argument("--token-file", help="a file holding the token a URL needs")
    parser.add_argument("--cache", default=os.environ.get(CACHE_ENV) or DEFAULT_CACHE_DIR)
    args = parser.parse_args(argv)
    try:
        entry, digest = stage(Path(args.cache), args.commit, _archive_bytes(args.archive, args.token_file))
    except (StageError, OSError, tarfile.TarError) as exc:
        print(f"ruppell_stage: {exc}", file=sys.stderr)
        return 1
    print(f"{entry} {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
