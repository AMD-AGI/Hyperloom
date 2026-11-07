# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tools the instruction sheet names: clean checkouts of InferenceX (and, for a v1 sheet, Magpie) at the commits it
names, and the inference_suite staging it names by its manifest's sha256.

The runner never uses the session's checkouts: they are on storage the candidate's author can write. Each commit gets
its own directory in the runner's cache, fetched by hash and checked out detached, so a directory that exists at the
right commit is the one that was fetched. A staging gets its own directory in the runner cache, named by the manifest's
sha256, so a directory whose manifest has that sha256 is the staging the sheet names.
"""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from inference_suite import CLIENT_FILE, manifest_sha256

__all__ = ["INFERENCE_SUITE_DIR", "ToolError", "ensure_inference_suite", "ensure_inferencex", "ensure_magpie"]

INFERENCEX_REPO = "https://github.com/SemiAnalysisAI/InferenceX.git"
MAGPIE_REPO = "https://github.com/AMD-AGI/Magpie.git"
_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_FETCH_TIMEOUT_SEC = 900
_SHA256 = re.compile(r"[0-9a-f]{64}")
#: Where the stagings live below the runner cache, one directory per manifest sha256.
INFERENCE_SUITE_DIR = "inference-suite"
#: Staging builds the AgentX harness venv with pip, from the wheels the runner cache holds.
_STAGE_TIMEOUT_SEC = 1800
#: What marks an InferenceX tree: the benchmark library a v1 sheet's Magpie runs, at the root or below
#: ``inferencex-e2e/``, or the client inference_suite stages a v2 sheet's rounds from (the pinned commit has no library).
_INFERENCEX_MARKERS = (
    Path("benchmarks") / "benchmark_lib.sh",
    Path("inferencex-e2e") / "benchmarks" / "benchmark_lib.sh",
    Path("inferencex-e2e") / CLIENT_FILE,
)
_MAGPIE_MARKERS = (Path("Magpie") / "__init__.py",)


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


def _has_marker(tree: Path, markers: tuple[Path, ...]) -> bool:
    return any((tree / marker).is_file() for marker in markers)


def _checkout(name: str, repo: str, commit: str, cache_dir: Path, markers: tuple[Path, ...]) -> Path:
    if not _COMMIT.fullmatch(commit):
        raise ToolError(f"{name}: {commit!r} is not a full commit hash")
    destination = cache_dir / f"{name}-{commit}"
    if destination.is_dir() and _head(destination) == commit and _has_marker(destination, markers):
        return destination
    scratch = cache_dir / f".{name}-{commit}.{os.getpid()}.tmp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        scratch.mkdir()
        _git("init", "-q", str(scratch))
        _git("fetch", "-q", "--depth", "1", repo, commit, cwd=scratch, timeout=_FETCH_TIMEOUT_SEC)
        _git("checkout", "-q", "--detach", "FETCH_HEAD", cwd=scratch)
        if _head(scratch) != commit or not _has_marker(scratch, markers):
            expected = " or ".join(str(marker) for marker in markers)
            raise ToolError(f"{name} at {commit} is not the expected tree (no {expected})")
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
        "InferenceX", os.environ.get("INFERENCEX_REPO") or INFERENCEX_REPO, commit, cache_dir, _INFERENCEX_MARKERS
    )


def ensure_magpie(commit: str, cache_dir: Path) -> Path:
    """A clean Magpie checkout at ``commit``; ``MAGPIE_REPO`` overrides where it is fetched from."""
    return _checkout("Magpie", os.environ.get("MAGPIE_REPO") or MAGPIE_REPO, commit, cache_dir, _MAGPIE_MARKERS)


def ensure_inference_suite(staging: str, inferencex: Path, runner_cache: Path) -> Path:
    """The inference_suite tool cache whose staging manifest has the sha256 ``staging``.

    It is ``<runner_cache>/inference-suite/<staging>/``, staged there from the InferenceX checkout ``inferencex`` by
    ``python -m inference_suite stage`` when no manifest with that sha256 is there yet. Staging builds the AgentX
    venv in place, so it runs in its final directory under a lock that runners sharing the cache take, and stage
    writes the manifest last, so a matching manifest is a whole tree. A staging whose manifest is not the one the
    sheet names is removed and refused: the runner's pins are not the application's.
    """
    if not _SHA256.fullmatch(staging):
        raise ToolError(f"inference_suite: {staging!r} is not a staging manifest sha256")
    stagings = runner_cache / INFERENCE_SUITE_DIR
    destination = stagings / staging
    if manifest_sha256(destination) == staging:
        return destination
    try:
        stagings.mkdir(parents=True, exist_ok=True)
        with open(stagings / f".{staging}.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if manifest_sha256(destination) != staging:
                _stage(staging, inferencex, destination)
    except OSError as exc:
        raise ToolError(f"inference_suite: the runner cache {stagings} cannot be staged into: {exc}") from exc
    return destination


def _stage(staging: str, inferencex: Path, destination: Path) -> None:
    _remove(destination)
    destination.mkdir(parents=True)
    argv = [sys.executable, "-m", "inference_suite", "stage", "--source", str(inferencex)]
    try:
        done = subprocess.run(
            [*argv, "--tool-cache", str(destination)], capture_output=True, text=True, timeout=_STAGE_TIMEOUT_SEC
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _remove(destination)
        raise ToolError(f"inference_suite stage failed: {exc}") from exc
    staged = manifest_sha256(destination)
    if done.returncode != 0 or staged != staging:
        _remove(destination)
        raise ToolError(
            f"inference_suite stage into {destination} exited {done.returncode} with manifest {staged}, not the "
            f"sheet's {staging}: {done.stderr.strip()[-300:]}"
        )
