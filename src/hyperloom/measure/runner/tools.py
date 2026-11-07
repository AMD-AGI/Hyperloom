# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tools a test runs: clean checkouts of InferenceX (and, for a v1 sheet, Magpie) at the commits the instruction
sheet names, and for a v2 sheet the inference_suite staging of this runner's own pins.

The runner never uses the session's checkouts: they are on storage the candidate's author can write. Each commit gets
its own directory in the runner's cache, fetched by hash and checked out detached, so a directory that exists at the
right commit is the one that was fetched. The inference_suite staging is never built here: ``cache_build`` publishes it
as ``<runner cache>/inference-suite/<staging key>/`` (:func:`inference_suite.staging_key`) on a host with network, and
the runner only verifies it and compares its normalised manifest with the one the sheet carries.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from inference_suite import (
    CLIENT_FILE,
    NormalisedManifest,
    StagingError,
    normalised_manifest,
    staging_key,
    verified_manifest_sha256,
)

__all__ = [
    "INFERENCE_SUITE_DIR",
    "StagedSuite",
    "StagingMismatch",
    "ToolError",
    "ensure_inference_suite",
    "ensure_inferencex",
    "ensure_magpie",
]

INFERENCEX_REPO = "https://github.com/SemiAnalysisAI/InferenceX.git"
MAGPIE_REPO = "https://github.com/AMD-AGI/Magpie.git"
_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_FETCH_TIMEOUT_SEC = 900
#: Where the published stagings live below the runner cache, one directory per staging key.
INFERENCE_SUITE_DIR = "inference-suite"
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


class StagingMismatch(ToolError):
    """The runner's published staging is not the software the sheet's normalised manifest names."""


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


@dataclass(frozen=True)
class StagedSuite:
    """A published inference_suite staging in the runner cache: its directory (a tool cache) and its normalised
    manifest, the one the test ran."""

    tool_cache: Path
    normalised: NormalisedManifest


def ensure_inference_suite(runner_cache: Path, expected: NormalisedManifest) -> StagedSuite:
    """The published staging of this runner's pins, ``<runner_cache>/inference-suite/<staging key>/``, once it verifies
    and its normalised manifest, with this interpreter's lm-eval, is ``expected``, the sheet's.

    Raises:
        StagingMismatch: The staging is other software than the sheet's; the message names each package that differs.
        ToolError: No staging of these pins is published, or the published one does not verify.
    """
    tool_cache = runner_cache / INFERENCE_SUITE_DIR / staging_key()
    if verified_manifest_sha256(tool_cache) is None:
        raise ToolError(
            f"inference_suite: no published staging at {tool_cache} that verifies; cache_build publishes it"
        )
    try:
        mine = normalised_manifest(tool_cache)
    except StagingError as exc:
        raise ToolError(f"inference_suite: the staging at {tool_cache} cannot be identified: {exc}") from exc
    differences = mine.differences(expected)
    if differences:
        raise StagingMismatch(
            f"inference_suite: the staging at {tool_cache} (here) is not the application's (there): "
            + "; ".join(differences)
        )
    return StagedSuite(tool_cache, mine)
