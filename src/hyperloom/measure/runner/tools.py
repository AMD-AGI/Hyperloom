# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tools a test runs: a clean checkout of InferenceX at the commit the instruction sheet names, and the
inference_suite staging of this runner's own pins.

The runner never uses the session's checkouts: they are on storage the candidate's author can write. Each test
checks the commit out into a directory of its own, fetched by hash and checked out detached. The commits a test of the
host fetched are kept in a git object store below the runner cache (:data:`INFERENCEX_STORE`), which every later test
fetches from instead of the network. The server under test can write that store, so a test never runs from it: git
hashes every object it receives, and an object that is not the one the commit names fails the fetch, which then goes to
the network with a warning. The inference_suite staging is never built here: ``cache_build`` publishes it
as ``<runner cache>/inference-suite/<staging key>/`` (:func:`inference_suite.staging_key`) on a host with network, and
the runner only verifies it and compares its normalised manifest with the one the sheet carries.
"""

from __future__ import annotations

import logging
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
    "INFERENCEX_STORE",
    "INFERENCE_SUITE_DIR",
    "TOOL_CHECKOUTS_DIR",
    "StagedSuite",
    "StagingMismatch",
    "ToolError",
    "ensure_inference_suite",
    "ensure_inferencex",
]

log = logging.getLogger(__name__)

INFERENCEX_REPO = "https://github.com/SemiAnalysisAI/InferenceX.git"
_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_FETCH_TIMEOUT_SEC = 900
_SHALLOW_FETCH = ("fetch", "-q", "--depth", "1")
_FILE_URL = "file://"
#: The global git configuration a fetch from a repository on this machine runs with, in the new repository's ``.git``
#: while it lasts (:func:`_trusting`).
_TRUST_CONFIG = "fetch-trust.gitconfig"
#: Where the published stagings live below the runner cache, one directory per staging key.
INFERENCE_SUITE_DIR = "inference-suite"
#: Where the tool object stores live below the runner cache.
TOOL_CHECKOUTS_DIR = "tools"
#: The bare repository below :data:`TOOL_CHECKOUTS_DIR` that keeps every InferenceX commit a test of the host fetched.
INFERENCEX_STORE = "InferenceX.git"
#: What marks an InferenceX tree: its benchmark library, at the root or below ``inferencex-e2e/``, or the client
#: inference_suite stages a sheet's rounds from (the pinned commit has no library).
_INFERENCEX_MARKERS = (
    Path("benchmarks") / "benchmark_lib.sh",
    Path("inferencex-e2e") / "benchmarks" / "benchmark_lib.sh",
    Path("inferencex-e2e") / CLIENT_FILE,
)


@dataclass(frozen=True)
class _Tool:
    """A repository a test checks out: its name and the files that mark its tree."""

    name: str
    markers: tuple[Path, ...]


_INFERENCEX = _Tool("InferenceX", _INFERENCEX_MARKERS)


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


def _checkout(tool: _Tool, repo: str, commit: str, cache_dir: Path, store: Path) -> Path:
    if not _COMMIT.fullmatch(commit):
        raise ToolError(f"{tool.name}: {commit!r} is not a full commit hash")
    destination = cache_dir / f"{tool.name}-{commit}"
    if destination.is_dir() and _head(destination) == commit and _has_marker(destination, tool.markers):
        return destination
    scratch = cache_dir / f".{tool.name}-{commit}.{os.getpid()}.tmp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        from_store = _fetch_from_store(scratch, store, commit)
        if not from_store:
            _remove(scratch)
            _new_repository(scratch)
            _fetch(repo, commit, scratch)
        _git("checkout", "-q", "--detach", "FETCH_HEAD", cwd=scratch)
        if _head(scratch) != commit or not _has_marker(scratch, tool.markers):
            expected = " or ".join(str(marker) for marker in tool.markers)
            raise ToolError(f"{tool.name} at {commit} is not the expected tree (no {expected})")
        if not from_store:
            _keep(store, scratch, commit)
        if destination.exists():
            _remove(destination)
        scratch.rename(destination)
    except BaseException:
        _remove(scratch)
        raise
    return destination


def _new_repository(scratch: Path) -> None:
    scratch.mkdir()
    _git("init", "-q", str(scratch))


def _fetch(repo: str, commit: str, scratch: Path) -> None:
    """Fetch ``commit`` from ``repo`` into the new repository ``scratch``; a directory on this machine is fetched from
    by its resolved path, trusted for this fetch alone (:func:`_trusting`)."""
    resolved = _local_directory(repo)
    if resolved is None:
        _git(*_SHALLOW_FETCH, repo, commit, cwd=scratch, timeout=_FETCH_TIMEOUT_SEC)
        return
    config = scratch / ".git" / _TRUST_CONFIG
    try:
        env = _trusting(resolved, config, scratch)
        _git(*_SHALLOW_FETCH, resolved, commit, cwd=scratch, timeout=_FETCH_TIMEOUT_SEC, env=env)
    finally:
        config.unlink(missing_ok=True)


def _local_directory(repo: str) -> str | None:
    """Where the directory ``repo`` names on this machine (an absolute path, or a ``file://`` URL) resolves to; ``None``
    for a remote, or for what is no directory here, which the fetch refuses as before."""
    path = repo.removeprefix(_FILE_URL)
    if not os.path.isabs(path) or not os.path.isdir(path):
        return None
    return os.path.realpath(path)


def _trusting(directory: str, config: Path, cwd: Path) -> dict[str, str]:
    """This process's environment with ``config`` written as the global git configuration, trusting the repository
    at ``directory`` alone (``safe.directory``), by exact path: it and its ``.git``, which git judges in a work tree.

    A runner's home is its sandbox's and the host's git configuration does not reach it, so git refuses a repository
    another account owns (a mirror on shared storage) as of dubious ownership. Only the system and global files can
    trust it there: neither ``-c`` nor ``GIT_CONFIG_COUNT`` does for the ``git-upload-pack`` the local transport starts
    (git 2.34, as Ubuntu 22.04 ships it). So it is a file, named for this fetch alone by ``GIT_CONFIG_GLOBAL``, as
    ``hosted/app_bootstrap.sh`` fetches the Hyperloom source: nothing else the runner or a test runs trusts it.
    """
    for trusted in (directory, os.path.join(directory, ".git")):
        _git("config", "--file", str(config), "--add", "safe.directory", trusted, cwd=cwd)
    return {**os.environ, "GIT_CONFIG_GLOBAL": str(config)}


def _fetch_from_store(scratch: Path, store: Path, commit: str) -> bool:
    """Whether ``commit`` came from ``store`` into a fresh repository at ``scratch``.

    The file:// transport makes git hash every object it receives instead of linking the store's files, so an object
    the server under test rewrote in the store fails the fetch rather than reaching the checkout. A store that fails
    the fetch is warned about with git's reason, which tells one that lacks the commit from one that was changed.
    """
    if not _is_repository(store):
        return False
    _new_repository(scratch)
    try:
        _git(*_SHALLOW_FETCH, store.resolve().as_uri(), commit, cwd=scratch)
    except ToolError as exc:
        log.warning("%s was not fetched from %s, so it is fetched from the network: %s", commit, store, exc)
        return False
    return True


def _keep(store: Path, scratch: Path, commit: str) -> None:
    """Add ``commit`` from the checked ``scratch`` to ``store`` for later tests of the host.

    A store this runner cannot write (a read-only runner cache) or another test updating it at once only means a later
    test fetches the commit from the network again, which the log says. Hooks are off: the server under test can
    write the store's config.
    """
    try:
        if not _is_repository(store):
            store.mkdir(parents=True, exist_ok=True)
            _git("init", "-q", "--bare", str(store))
        _git(
            "-c",
            "core.hooksPath=/dev/null",
            f"--git-dir={store}",
            *_SHALLOW_FETCH,
            scratch.resolve().as_uri(),
            f"{commit}:refs/tools/{commit}",
        )
    except (OSError, ToolError) as exc:
        log.warning("%s was not kept in %s, so a later test fetches it from the network again: %s", commit, store, exc)


def _is_repository(store: Path) -> bool:
    return (store / "HEAD").is_file()


def _remove(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def ensure_inferencex(commit: str, cache_dir: Path, store: Path) -> Path:
    """A clean InferenceX checkout at ``commit`` below ``cache_dir``, this test's own; fetched from ``store`` (the
    host's :data:`INFERENCEX_STORE`) when it holds the commit intact, else from ``INFERENCEX_REPO`` or upstream."""
    return _checkout(_INFERENCEX, os.environ.get("INFERENCEX_REPO") or INFERENCEX_REPO, commit, cache_dir, store)


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
