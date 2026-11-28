# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tools a test runs: a clean checkout of InferenceX at the commit the instruction sheet names, and the
inference_suite staging of this runner's own pins.

The runner never uses the session's checkouts: they are on storage the candidate's author can write. Each test
checks the commit out into a directory of its own below its scratch, fetched by hash and checked out detached. The
runner cache may be mounted read-only, and a test never writes it. The host fills the git object store below it
(:data:`INFERENCEX_STORE`): ``cache_build``, on a host with network, fetches the pinned InferenceX commit into it
(:func:`fill_inferencex_store`). A test fetches its commit from there, reading only, when the store holds it; otherwise
it fetches from ``INFERENCEX_REPO`` or upstream into its own scratch. A test never runs from the store: git hashes every
object it receives, and an object that is not the one the commit names fails the fetch, which then goes to
``INFERENCEX_REPO`` or upstream with a warning. No alternates are set up either, since git does not hash an object it
reads through them. The inference_suite staging is never built here: ``cache_build`` publishes it
as ``<runner cache>/inference-suite/<staging key>/`` (:func:`inference_suite.staging_key`) on a host with network, and
the runner only verifies it and compares its normalised manifest with the one the sheet carries.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from hyperloom_common.shared_storage import make_shared_dirs
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
    "fill_inferencex_store",
]

log = logging.getLogger(__name__)

INFERENCEX_REPO = "https://github.com/SemiAnalysisAI/InferenceX.git"
_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_FETCH_TIMEOUT_SEC = 900
#: How long a test waits for the store before it goes to ``INFERENCEX_REPO`` or upstream: the store is on the runner
#: cache's mount, and a hung mount must not cost a test the network fetch's whole timeout first.
_STORE_FETCH_TIMEOUT_SEC = 120
#: The mode of a new store: the builders' group writes it, the sandbox's account reads it.
_STORE_MODE = 0o2775
_SHALLOW_FETCH = ("fetch", "-q", "--depth", "1")
#: The lock file a shallow fetch into the store holds while it runs, so that one fill at a time fetches into it.
_SHALLOW_LOCK = "shallow.lock"
#: What git's message says when a lock file it takes is there already (another fill holds it, or a killed one left
#: it), in the wordings of the git releases in use; the error keeps only the end of the message, so each is matched.
#: The last is a shallow fetch's when another fill committed the shallow file after this one read it.
_LOCK_HELD = (
    ".lock': File exists",
    "Another git process seems to be running in this repository",
    "remove the file manually to continue",
    "shallow file has changed since we read it",
)
#: How long a git command stopped at its timeout has, after ``SIGTERM``, to remove its lock files and exit.
_STOP_GRACE_SEC = 10
#: How old a lock file of the store is before a fill takes it for one a killed fill left and removes it: a live fill's
#: fetch holds it at most :data:`_FETCH_TIMEOUT_SEC`, and is stopped then, removing it itself.
_STALE_LOCK_SEC = _FETCH_TIMEOUT_SEC + 300
#: How long a fill waits, in all, for the other fills of the store before it gives up.
_LOCK_WAIT_SEC = 2 * _STALE_LOCK_SEC
#: The longest pause between two attempts of a fill that waits for another.
_LOCK_PAUSE_SEC = 15.0
_FILE_URL = "file://"
#: The global git configuration a fetch from a repository on this machine runs with, in the new repository's ``.git``
#: while it lasts (:func:`_trusting`).
_TRUST_CONFIG = "fetch-trust.gitconfig"
#: Where the published stagings live below the runner cache, one directory per staging key.
INFERENCE_SUITE_DIR = "inference-suite"
#: Where the tool object stores live below the runner cache.
TOOL_CHECKOUTS_DIR = "tools"
#: The bare repository below :data:`TOOL_CHECKOUTS_DIR` that holds the pinned InferenceX commit: the host fills it
#: (:func:`fill_inferencex_store`), and a test fetches from it, read-only, when it holds the commit.
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


def _git(*args: str, cwd: Path | None = None, timeout: float = 120, env: dict[str, str] | None = None) -> str:
    """Run git; one still running at ``timeout`` is stopped (:func:`_stop`), so it removes the lock files it holds."""
    try:
        git = subprocess.Popen(
            ["git", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        raise ToolError(f"git {args[0]} failed: {exc}") from exc
    try:
        out, err = git.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _stop(git)
        raise ToolError(f"git {args[0]} failed: it was still running after {timeout} s and was stopped") from None
    except BaseException:
        _stop(git)
        raise
    if git.returncode != 0:
        raise ToolError(f"git {args[0]} failed: {err.strip()[-300:]}")
    return out.strip()


def _stop(git: subprocess.Popen[str]) -> None:
    """Stop ``git`` and the processes it started (its own session): ``SIGTERM`` first, on which git removes the lock
    files and temporary files it holds, and ``SIGKILL`` after :data:`_STOP_GRACE_SEC` if it is still running then; one
    that outlives that too (blocked on a hung mount) is left to exit when it can."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(git.pid, sig)
        except ProcessLookupError:
            pass
        try:
            git.communicate(timeout=_STOP_GRACE_SEC)
            return
        except subprocess.TimeoutExpired:
            continue


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
        from_store = _fetch_from_store(scratch, store, commit, repo)
        if not from_store:
            _remove(scratch)
            _new_repository(scratch)
            _fetch(repo, commit, scratch)
        _git("checkout", "-q", "--detach", "FETCH_HEAD", cwd=scratch)
        if _head(scratch) != commit or not _has_marker(scratch, tool.markers):
            expected = " or ".join(str(marker) for marker in tool.markers)
            raise ToolError(f"{tool.name} at {commit} is not the expected tree (no {expected})")
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


def _fetch(repo: str, commit: str, scratch: Path, timeout: int = _FETCH_TIMEOUT_SEC) -> None:
    """Fetch ``commit`` from ``repo`` into the new repository ``scratch``; a directory on this machine is fetched from
    by its resolved path, trusted for this fetch alone (:func:`_trusting`)."""
    resolved = _local_directory(repo)
    if resolved is None:
        _git(*_SHALLOW_FETCH, repo, commit, cwd=scratch, timeout=timeout)
        return
    config = scratch / ".git" / _TRUST_CONFIG
    try:
        env = _trusting([resolved], config, scratch)
        _git(*_SHALLOW_FETCH, resolved, commit, cwd=scratch, timeout=timeout, env=env)
    finally:
        config.unlink(missing_ok=True)


def _local_directory(repo: str) -> str | None:
    """Where the directory ``repo`` names on this machine (an absolute path, or a ``file://`` URL) resolves to; ``None``
    for a remote, or for what is no directory here, which the fetch refuses as before."""
    path = urllib.parse.unquote(repo[len(_FILE_URL) :]) if repo.startswith(_FILE_URL) else repo
    if not os.path.isabs(path) or not os.path.isdir(path):
        return None
    return os.path.realpath(path)


def _trusting(directories: list[str], config: Path, cwd: Path) -> dict[str, str]:
    """This process's environment with ``config`` written as the global git configuration, trusting the repositories
    at ``directories`` alone (``safe.directory``), by exact path: each and its ``.git``, which git judges in a work
    tree.

    A runner's home is its sandbox's and the host's git configuration does not reach it, so git refuses a repository
    another account owns (a mirror on shared storage) as of dubious ownership. Only the system and global files can
    trust it there: neither ``-c`` nor ``GIT_CONFIG_COUNT`` does for the ``git-upload-pack`` the local transport starts
    (git 2.34, as Ubuntu 22.04 ships it). So it is a file, named for this fetch alone by ``GIT_CONFIG_GLOBAL``, as
    ``hosted/app_bootstrap.sh`` fetches the Hyperloom source: nothing else the runner or a test runs trusts it.
    """
    for directory in directories:
        for trusted in (directory, os.path.join(directory, ".git")):
            _git("config", "--file", str(config), "--add", "safe.directory", trusted, cwd=cwd)
    return {**os.environ, "GIT_CONFIG_GLOBAL": str(config)}


def _fetch_from_store(scratch: Path, store: Path, commit: str, repo: str) -> bool:
    """Whether ``commit`` came from ``store`` into a fresh repository at ``scratch``.

    A fetch (unlike a local clone) receives a pack and makes git hash every object in it instead of linking the store's
    files, so an object rewritten in the store fails the fetch rather than reaching the checkout. A store that fails
    the fetch is warned about with git's reason, which tells one that lacks the commit from one that was changed.
    The fetch runs in ``scratch`` and only reads ``store`` (``git-upload-pack`` writes nothing), so a store on a
    read-only mount, or one another account owns (trusted for this fetch alone, :func:`_fetch`), serves as well; it has
    :data:`_STORE_FETCH_TIMEOUT_SEC` before ``repo`` is fetched from instead.
    """
    if not _is_repository(store):
        return False
    _new_repository(scratch)
    try:
        _fetch(store.resolve().as_uri(), commit, scratch, _STORE_FETCH_TIMEOUT_SEC)
    except ToolError as exc:
        log.warning("%s was not fetched from %s, so it is fetched from %s: %s", commit, store, repo, exc)
        return False
    return True


def _is_repository(store: Path) -> bool:
    return (store / "HEAD").is_file()


def _store_git(store: Path) -> list[str]:
    """The options of a git command the host runs in the store: another builder of the group may have made it, so no
    hook and no fsmonitor command it names runs."""
    return ["-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", f"--git-dir={store}"]


def _host_config(config: Path, trusted: list[str]) -> dict[str, str]:
    """:func:`_trusting` ``trusted`` on the host, whose own global git configuration (a proxy, say) is kept by an
    include."""
    own = os.environ.get("GIT_CONFIG_GLOBAL") or os.path.expanduser("~/.gitconfig")
    if os.path.isfile(own):
        _git("config", "--file", str(config), "--add", "include.path", own, cwd=config.parent)
    return _trusting(trusted, config, config.parent)


def _holds(store: Path, commit: str, env: dict[str, str]) -> bool:
    try:
        _git(*_store_git(store), "cat-file", "-e", f"{commit}^{{commit}}", env=env)
    except ToolError:
        return False
    return True


def _bare_store(store: Path) -> None:
    """Make the bare repository ``store`` unless it is there: initialised in a new directory beside it and renamed into
    place, so a reader never finds a half-made repository, and a concurrent maker's repository is kept. Like the rest
    of the cache it is the builders' group's to write (each builder may fetch a new pin into it) and others' to read
    (the sandbox's account); :data:`TOOL_CHECKOUTS_DIR` is made as ``inference-suite/`` is, ``2775`` below a setgid
    runner cache (:func:`~hyperloom_common.shared_storage.make_shared_dirs`), so a builder of another account of the
    group makes its temporary directory there as well."""
    make_shared_dirs(store.parent)
    made = Path(tempfile.mkdtemp(prefix=f".{store.name}.", suffix=".tmp", dir=store.parent))
    try:
        _git("init", "-q", "--bare", "--shared=group", str(made))
        made.chmod(_STORE_MODE)
        try:
            made.rename(store)
        except OSError:
            if not _is_repository(store):
                raise
    finally:
        _remove(made)


def fill_inferencex_store(runner_cache: Path, commit: str) -> Path:
    """Make the runner cache's :data:`INFERENCEX_STORE` hold ``commit``, fetched from ``INFERENCEX_REPO`` or upstream;
    the host runs it where the runner cache is built (``cache_build``), with network, and a sandbox never does. A store
    that already holds the commit is left as it is, so it fetches nothing then. Only that commit is fetched, shallow,
    and kept as ``refs/tools/<commit>``; git writes the objects before the ref, so a reader that finds the commit finds
    all of it. The store may be another builder's of the group: the host's git commands in it trust it for that command
    alone and run no hook or fsmonitor it names. Returns the store.

    Builders fill the store at the same time without a lock of their own (:func:`_fetch_into_store`): a shallow fetch
    holds the store's ``shallow.lock``, so a fill that finds it held waits and tries again until the other fill is done,
    and a lock older than :data:`_STALE_LOCK_SEC`, which only a killed fill leaves, is removed.

    Raises:
        ToolError: ``commit`` is no full commit hash, or it cannot be fetched.
    """
    if not _COMMIT.fullmatch(commit):
        raise ToolError(f"InferenceX: {commit!r} is not a full commit hash")
    # Absolute: the host's git commands run in a directory of their own, naming the store by --git-dir.
    store = Path(os.path.abspath(runner_cache / TOOL_CHECKOUTS_DIR / INFERENCEX_STORE))
    repo = os.environ.get("INFERENCEX_REPO") or INFERENCEX_REPO
    resolved = _local_directory(repo)
    try:
        if not _is_repository(store):
            _bare_store(store)
        with tempfile.TemporaryDirectory(prefix="inferencex-fill.") as work:
            trusted = [os.path.realpath(store), *([resolved] if resolved is not None else [])]
            env = _host_config(Path(work) / _TRUST_CONFIG, trusted)
            if not _holds(store, commit, env):
                fetch = [*_store_git(store), *_SHALLOW_FETCH, resolved or repo, f"{commit}:refs/tools/{commit}"]
                _fetch_into_store(store, commit, fetch, Path(work), env)
    except (OSError, ToolError) as exc:
        raise ToolError(f"InferenceX {commit} was not fetched into {store}: {exc}") from exc
    return store


def _store_locks(store: Path, commit: str) -> tuple[Path, ...]:
    """The lock files a fill of ``commit`` takes in ``store``: the shallow file's and its ref's."""
    return store / _SHALLOW_LOCK, store / "refs" / "tools" / f"{commit}.lock"


def _fetch_into_store(store: Path, commit: str, fetch: list[str], work: Path, env: dict[str, str]) -> None:
    """Run ``fetch`` until ``store`` holds ``commit``. A fetch that fails on a lock file another fill holds is tried
    again after a pause, which doubles up to :data:`_LOCK_PAUSE_SEC`; one that fails because a concurrent fill fetched
    the commit first is done. Before each attempt a lock file older than :data:`_STALE_LOCK_SEC` is removed
    (:func:`_remove_stale_lock`). Any other failure, or waiting longer than :data:`_LOCK_WAIT_SEC` in all, raises.

    Whether another fill holds a lock is git's message alone: a fetch of this fill's own still running at
    :data:`_FETCH_TIMEOUT_SEC` is stopped with ``SIGTERM`` (:func:`_git`), removes the locks it took, and its failure
    raises at once; a lock file that is merely there is never read as another fill's, so a fill never waits on one its
    own fetch left. A fetch that does not exit on ``SIGTERM`` may leave its lock behind; the next fill removes it once
    stale, never sooner, since a concurrent fill may have taken it while this fill's fetch had not.
    """
    deadline = time.monotonic() + _LOCK_WAIT_SEC
    pause = 0.5
    while True:
        for lock in _store_locks(store, commit):
            _remove_stale_lock(lock)
        try:
            _git(*fetch, cwd=work, timeout=_FETCH_TIMEOUT_SEC, env=env)
            return
        except ToolError as exc:
            if _holds(store, commit, env):
                return  # a concurrent fill fetched it
            if not any(words in str(exc) for words in _LOCK_HELD):
                raise
            if time.monotonic() >= deadline:
                raise ToolError(f"another fill held a lock of {store} for {_LOCK_WAIT_SEC} s: {exc}") from exc
        log.info("a lock of %s is held by another fill; %s is fetched again in %.1f s", store, commit, pause)
        time.sleep(pause)
        pause = min(pause * 2, _LOCK_PAUSE_SEC)


def _remove_stale_lock(lock: Path) -> None:
    """Remove ``lock`` when it is older than :data:`_STALE_LOCK_SEC`, left by a fill killed while it held it. It is
    first renamed aside, a name of this process's own, and judged there, so that of two fills that find it stale only
    one removes it; a lock that is fresh once aside, which another fill took in the meantime, is linked back."""
    try:
        if time.time() - lock.stat().st_mtime < _STALE_LOCK_SEC:
            return
        aside = lock.with_name(f"{lock.name}.{os.getpid()}.stale")
        os.rename(lock, aside)
    except FileNotFoundError:
        return
    try:
        if time.time() - aside.stat().st_mtime < _STALE_LOCK_SEC:
            try:
                os.link(aside, lock)
            except FileExistsError:
                pass  # a third fill took the lock meanwhile; it is that fill's now
        else:
            log.warning("removed %s, which a killed fill left", lock)
    finally:
        aside.unlink(missing_ok=True)


def _remove(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def ensure_inferencex(commit: str, cache_dir: Path, store: Path) -> Path:
    """A clean InferenceX checkout at ``commit`` below ``cache_dir``, this test's own; fetched from ``store`` (the
    host's :data:`INFERENCEX_STORE`, read only) when it holds the commit intact, else from ``INFERENCEX_REPO`` or
    upstream. Nothing is written outside ``cache_dir``."""
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
