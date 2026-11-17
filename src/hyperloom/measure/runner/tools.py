# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tools a test runs: a clean checkout of InferenceX at the commit the instruction sheet names, and the
inference_suite staging of this runner's own pins.

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
]

INFERENCEX_REPO = "https://github.com/SemiAnalysisAI/InferenceX.git"
_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_FETCH_TIMEOUT_SEC = 900
_FILE_URL = "file://"
#: The global git configuration a fetch from a repository on this machine runs with, in the new repository's ``.git``
#: while it lasts (:func:`_trusting`).
_TRUST_CONFIG = "fetch-trust.gitconfig"
#: Where the published stagings live below the runner cache, one directory per staging key.
INFERENCE_SUITE_DIR = "inference-suite"
#: What marks an InferenceX tree: its benchmark library, at the root or below ``inferencex-e2e/``, or the client
#: inference_suite stages a sheet's rounds from (the pinned commit has no library).
_INFERENCEX_MARKERS = (
    Path("benchmarks") / "benchmark_lib.sh",
    Path("inferencex-e2e") / "benchmarks" / "benchmark_lib.sh",
    Path("inferencex-e2e") / CLIENT_FILE,
)


class ToolError(Exception):
    """A checkout could not be prepared at the commit that was asked for."""


class StagingMismatch(ToolError):
    """The runner's published staging is not the software the sheet's normalised manifest names."""


def _git(*args: str, cwd: Path | None = None, timeout: int = 120, env: dict[str, str] | None = None) -> str:
    try:
        done = subprocess.run(["git", *args], capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env)
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
        _fetch(repo, commit, scratch)
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


def _fetch(repo: str, commit: str, scratch: Path) -> None:
    """Fetch ``commit`` from ``repo`` into the new repository ``scratch``; one on this machine trusted for this fetch
    alone (:func:`_trusting`)."""
    local = _local_repository(repo)
    if local is None:
        _git("fetch", "-q", "--depth", "1", repo, commit, cwd=scratch, timeout=_FETCH_TIMEOUT_SEC)
        return
    config = scratch / ".git" / _TRUST_CONFIG
    try:
        env = _trusting(local, config, scratch)
        _git("fetch", "-q", "--depth", "1", repo, commit, cwd=scratch, timeout=_FETCH_TIMEOUT_SEC, env=env)
    finally:
        config.unlink(missing_ok=True)


def _local_repository(repo: str) -> str | None:
    """The directory ``repo`` names on this machine (an absolute path, or a ``file://`` URL); ``None`` for a remote."""
    path = repo.removeprefix(_FILE_URL)
    return path if os.path.isabs(path) else None


def _trusting(repository: str, config: Path, cwd: Path) -> dict[str, str]:
    """This process's environment with ``config`` written as the global git configuration, trusting ``repository``
    alone (``safe.directory``): the repository, and its ``.git``, the directory git judges in a work tree.

    A runner's home is its sandbox's and the host's git configuration does not reach it, so git refuses a repository
    another account owns (a mirror on shared storage) as of dubious ownership. Git before 2.38 reads ``safe.directory``
    from the system and global files only, never from ``-c`` or ``GIT_CONFIG_COUNT``, so it is a file, named for
    this fetch alone by ``GIT_CONFIG_GLOBAL``: nothing else the runner or a test runs trusts it.
    """
    named = os.path.normpath(repository)
    for directory in (named, os.path.join(named, ".git")):
        _git("config", "--file", str(config), "--add", "safe.directory", directory, cwd=cwd)
    return {**os.environ, "GIT_CONFIG_GLOBAL": str(config)}


def _remove(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def ensure_inferencex(commit: str, cache_dir: Path) -> Path:
    """A clean InferenceX checkout at ``commit``; ``INFERENCEX_REPO`` overrides where it is fetched from."""
    return _checkout(
        "InferenceX", os.environ.get("INFERENCEX_REPO") or INFERENCEX_REPO, commit, cache_dir, _INFERENCEX_MARKERS
    )


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
