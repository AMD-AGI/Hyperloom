"""The tools a test runs: clean checkouts of InferenceX (and, for a v1 sheet, Magpie) at the commits the instruction
sheet names, and for a v2 sheet the inference_suite staging of this runner's own pins.

The runner never uses the session's checkouts: they are on storage the candidate's author can write. Each commit gets
its own directory in the runner's cache, fetched by hash and checked out detached, so a directory that exists at the
right commit is the one that was fetched. The inference_suite staging gets a directory in the runner cache per
InferenceX pin; the runner stages it there itself, offline, and names the manifest it ran.
"""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from inference_suite import CLIENT_FILE, INFERENCEX_CLIENT_COMMIT, verified_manifest_sha256

__all__ = [
    "INFERENCE_SUITE_DIR",
    "StagedSuite",
    "ToolError",
    "ensure_inference_suite",
    "ensure_inferencex",
    "ensure_magpie",
]

INFERENCEX_REPO = "https://github.com/SemiAnalysisAI/InferenceX.git"
MAGPIE_REPO = "https://github.com/AMD-AGI/Magpie.git"
_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_FETCH_TIMEOUT_SEC = 900
#: Where the stagings live below the runner cache, one directory per InferenceX pin.
INFERENCE_SUITE_DIR = "inference-suite"
#: Staging builds the AgentX harness venv with pip, offline, from the sandbox's wheel directory.
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


@dataclass(frozen=True)
class StagedSuite:
    """An inference_suite tool cache in the runner cache, and the sha256 of the staging manifest it holds."""

    tool_cache: Path
    staging: str


def ensure_inference_suite(inferencex: Path, runner_cache: Path, wheels: Path) -> StagedSuite:
    """The inference_suite tool cache this runner's pins stage, ``<runner_cache>/inference-suite/<InferenceX pin>/``.

    One that verifies is used as it is. Otherwise ``python -m inference_suite stage --wheels`` stages it offline from
    the InferenceX checkout ``inferencex``, installing the AgentX harness from the wheel directory ``wheels`` alone;
    stage refuses a checkout off the pin and leaves a damaged cache as it found it. The venv it builds holds its own
    absolute path, so it is staged in its final directory, under a lock that runners sharing the cache take.

    The staging is this runner's own: its manifest names the harness the sandbox's interpreter and wheels resolved,
    which is never byte for byte the application's, so the result manifest records it rather than the sheet's.
    """
    stagings = runner_cache / INFERENCE_SUITE_DIR
    destination = stagings / INFERENCEX_CLIENT_COMMIT
    staged = _intact(destination)
    if staged is not None:
        return staged
    try:
        stagings.mkdir(parents=True, exist_ok=True)
        with open(stagings / f".{INFERENCEX_CLIENT_COMMIT}.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return _intact(destination) or _stage(inferencex, destination, wheels)
    except OSError as exc:
        raise ToolError(f"inference_suite: the runner cache {stagings} cannot be staged into: {exc}") from exc


def _intact(tool_cache: Path) -> StagedSuite | None:
    digest = verified_manifest_sha256(tool_cache)
    return None if digest is None else StagedSuite(tool_cache, digest)


def _stage(inferencex: Path, destination: Path, wheels: Path) -> StagedSuite:
    argv = [sys.executable, "-m", "inference_suite", "stage", "--source", str(inferencex)]
    argv += ["--tool-cache", str(destination), "--wheels", str(wheels)]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=_STAGE_TIMEOUT_SEC)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ToolError(f"inference_suite stage failed: {exc}") from exc
    staged = _intact(destination)
    if done.returncode != 0 or staged is None:
        raise ToolError(
            f"inference_suite stage into {destination} exited {done.returncode} without a staging that verifies: "
            f"{done.stderr.strip()[-300:]}"
        )
    return staged
