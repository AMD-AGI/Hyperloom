# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Clean checkouts of Magpie and InferenceX at the commits the instruction sheet names (design 4.8.1 step 2).

The runner never uses the session's checkouts: they are on storage the candidate's author can write. Each commit gets
its own directory in the runner's cache, fetched by hash and checked out detached, so a directory that exists at the
right commit is the one that was fetched.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

__all__ = ["ToolError", "ensure_inferencex", "ensure_magpie"]

INFERENCEX_REPO = "https://github.com/SemiAnalysisAI/InferenceX.git"
MAGPIE_REPO = "https://github.com/AMD-AGI/Magpie.git"
_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_FETCH_TIMEOUT_SEC = 900
_INFERENCEX_MARKER = Path("benchmarks") / "benchmark_lib.sh"
_MAGPIE_MARKER = Path("Magpie") / "__init__.py"


class ToolError(Exception):
    """A checkout could not be prepared at the commit that was asked for."""


def _git(*args: str, cwd: Path | None = None, timeout: int = 120) -> str:
    try:
        done = subprocess.run(["git", *args], capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ToolError(f"git {args[0]} failed: {exc}") from exc
    if done.returncode != 0:
        raise ToolError(f"git {args[0]} failed: {done.stderr.strip()[-300:]}")
    return done.stdout.strip()


def _head(path: Path) -> str | None:
    try:
        return _git("rev-parse", "HEAD", cwd=path)
    except ToolError:
        return None


def _checkout(name: str, repo: str, commit: str, cache_dir: Path, marker: Path) -> Path:
    if not _COMMIT.fullmatch(commit):
        raise ToolError(f"{name}: {commit!r} is not a full commit hash")
    destination = cache_dir / f"{name}-{commit}"
    if destination.is_dir() and _head(destination) == commit and (destination / marker).is_file():
        return destination
    scratch = cache_dir / f".{name}-{commit}.{os.getpid()}.tmp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        scratch.mkdir()
        _git("init", "-q", str(scratch))
        _git("fetch", "-q", "--depth", "1", repo, commit, cwd=scratch, timeout=_FETCH_TIMEOUT_SEC)
        _git("checkout", "-q", "--detach", "FETCH_HEAD", cwd=scratch)
        if _head(scratch) != commit or not (scratch / marker).is_file():
            raise ToolError(f"{name} at {commit} is not the expected tree (no {marker})")
        if destination.exists():
            _remove(destination)
        scratch.rename(destination)
    except BaseException:
        _remove(scratch)
        raise
    return destination


def _remove(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def ensure_inferencex(commit: str, cache_dir: Path) -> Path:
    """A clean InferenceX checkout at ``commit``; ``INFERENCEX_REPO`` overrides where it is fetched from."""
    return _checkout(
        "InferenceX", os.environ.get("INFERENCEX_REPO") or INFERENCEX_REPO, commit, cache_dir, _INFERENCEX_MARKER
    )


def ensure_magpie(commit: str, cache_dir: Path) -> Path:
    """A clean Magpie checkout at ``commit``; ``MAGPIE_REPO`` overrides where it is fetched from."""
    return _checkout("Magpie", os.environ.get("MAGPIE_REPO") or MAGPIE_REPO, commit, cache_dir, _MAGPIE_MARKER)
