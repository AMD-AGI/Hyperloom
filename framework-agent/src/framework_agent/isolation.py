# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Per-candidate isolation primitives — git worktree + venv lifecycle.

Split out of ``explorer.py`` per merged-design §2.2 so the explorer
keeps only the orchestration loop and the isolation logic can be unit
tested in isolation. Public surface:

* :func:`prepare_repo_cache`       — mirror-clone (or fetch) the
  upstream repo into ``work_dir/_repos/<slug>``.
* :func:`prepare_candidate_workspace` — create per-candidate dir +
  detached worktree + venv. Returns the resolved paths so callers can
  thread them into ``render_template`` variable bags.
* :func:`cleanup_workspace`        — remove a candidate's worktree /
  venv on completion. Respects ``keep_winner_only``.
* :func:`disk_preflight`           — refuse to start an N-candidate run
  when the work_dir mount has < ``min_free_gb`` (default 20 GB,
  overridable via ``FRAMEWORK_EXPLORER_DISK_MIN_GB``).

Subprocess wrappers (``_run_subprocess`` / ``_run_git``) are also
re-exported here so explorer.py can import them from one place.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .logging_setup import get_logger
from .models import Candidate, ExploreRequest

log = get_logger(__name__)


# Threshold env var per merged-design §14.3.
_DISK_MIN_GB_ENV = "FRAMEWORK_EXPLORER_DISK_MIN_GB"
_DEFAULT_DISK_MIN_GB = 20.0
# Per-candidate disk budget (worktree + venv + audit material). Tuned
# from observed sglang/vllm worktrees (~700MB) + venv with --system-
# site-packages (~50MB) + headroom for build artefacts.
PER_CANDIDATE_GB = 1.5


class DiskPreflightError(RuntimeError):
    """Raised when the work_dir mount lacks the required free GB."""


@dataclass
class WorkspacePaths:
    """Resolved per-candidate workspace layout returned by prepare_candidate_workspace."""

    candidate_dir: Path
    worktree_dir: Path
    venv_dir: Path


# ---------------------------------------------------------------------------
# Subprocess helpers (verbatim from explorer.py — single source of truth here)
# ---------------------------------------------------------------------------
def _run_subprocess(
    args: list[str], *, cwd: Path | None = None, timeout_sec: int = 1800
) -> None:
    """Run a subprocess with a timeout; raise CalledProcessError on non-zero."""
    log.debug("subprocess %s cwd=%s timeout=%ds", " ".join(args[:4]), cwd, timeout_sec)
    subprocess.run(args, cwd=str(cwd) if cwd else None, check=True, timeout=timeout_sec)


def _run_git(args: list[str], *, cwd: Path | None = None, timeout_sec: int = 1800) -> None:
    """Run a git command with a timeout; thin wrapper over :func:`_run_subprocess`."""
    _run_subprocess(args, cwd=cwd, timeout_sec=timeout_sec)


# ---------------------------------------------------------------------------
# Disk preflight (merged-design §4.5)
# ---------------------------------------------------------------------------
def _resolve_min_free_gb(explicit: float | None) -> float:
    """Pick the threshold (explicit > env > default 20 GB)."""
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get(_DISK_MIN_GB_ENV)
    if raw:
        try:
            return float(raw)
        except ValueError:
            log.warning(
                "%s=%r is not a number; falling back to default %.1f GB",
                _DISK_MIN_GB_ENV, raw, _DEFAULT_DISK_MIN_GB,
            )
    return _DEFAULT_DISK_MIN_GB


def disk_preflight(
    work_dir: Path,
    n_candidates: int,
    *,
    min_free_gb: float | None = None,
    per_candidate_gb: float = PER_CANDIDATE_GB,
) -> None:
    """Refuse to start if the work_dir mount lacks enough free space.

    Required = ``max(min_free_gb, n_candidates * per_candidate_gb)`` so a
    large N candidate run never sneaks under the floor.

    The work_dir is created when missing so :func:`shutil.disk_usage` does
    not fail when the parent dir exists but the target one doesn't yet.
    Raises :class:`DiskPreflightError` on failure with a human-readable
    message that includes the actual / required GB.
    """
    floor_gb = _resolve_min_free_gb(min_free_gb)
    required_gb = max(floor_gb, float(n_candidates) * per_candidate_gb)
    work_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(str(work_dir))
    free_gb = usage.free / (1024 ** 3)
    log.info(
        "disk_preflight: work_dir=%s free=%.1fGB required=%.1fGB (n=%d, "
        "floor=%.1fGB, per_cand=%.1fGB)",
        work_dir, free_gb, required_gb, n_candidates, floor_gb, per_candidate_gb,
    )
    if free_gb < required_gb:
        raise DiskPreflightError(
            f"insufficient disk on {work_dir}: free={free_gb:.1f}GB, "
            f"required={required_gb:.1f}GB "
            f"(n_candidates={n_candidates}, per_cand={per_candidate_gb}GB, "
            f"floor={floor_gb}GB). "
            f"Free space or lower max_search_candidates / set "
            f"{_DISK_MIN_GB_ENV} to a smaller value."
        )


# ---------------------------------------------------------------------------
# Repo cache (mirror clone)
# ---------------------------------------------------------------------------
def _repo_cache_dir(req: ExploreRequest) -> Path:
    """Stable per-repo cache directory under work_dir/_repos."""
    safe = "".join(ch if ch.isalnum() else "-" for ch in req.repo_url.lower()).strip("-")
    return req.work_dir / "_repos" / (safe or "repo")


def prepare_repo_cache(req: ExploreRequest) -> Path:
    """Mirror-clone the repo into the cache dir; fetch when already present."""
    repo_dir = _repo_cache_dir(req)
    if repo_dir.exists():
        log.debug("prepare_repo_cache: fetching existing mirror at %s", repo_dir)
        _run_git(["git", "fetch", "--all", "--tags", "--prune"], cwd=repo_dir)
        return repo_dir
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    log.info("prepare_repo_cache: cloning --mirror %s -> %s", req.repo_url, repo_dir)
    _run_git(["git", "clone", "--mirror", req.repo_url, str(repo_dir)])
    return repo_dir


def _worktree_ref(candidate: Candidate) -> str:
    """Choose the ref to materialise in a detached worktree."""
    if candidate.head_sha:
        return candidate.head_sha
    if candidate.ref.startswith("PR:"):
        number = candidate.ref.split(":", 1)[1]
        return f"refs/pull/{number}/head"
    return candidate.ref


def fetch_candidate_ref(repo_dir: Path, candidate: Candidate) -> None:
    """Pre-fetch the candidate's ref into the cache mirror."""
    if candidate.head_sha:
        _run_git(["git", "fetch", "origin", candidate.head_sha], cwd=repo_dir)
        return
    if not candidate.ref.startswith("PR:"):
        return
    number = candidate.ref.split(":", 1)[1]
    _run_git(
        [
            "git",
            "fetch",
            "origin",
            f"refs/pull/{number}/head:refs/pull/{number}/head",
        ],
        cwd=repo_dir,
    )


# ---------------------------------------------------------------------------
# Per-candidate workspace lifecycle
# ---------------------------------------------------------------------------
def prepare_candidate_workspace(
    req: ExploreRequest,
    candidate: Candidate,
    *,
    index: int,
    execute: bool,
) -> WorkspacePaths:
    """Materialise ``candidate_dir`` + (when execute) worktree + venv.

    Returns :class:`WorkspacePaths` instead of an ad-hoc tuple so callers
    don't have to remember the field order. ``execute=False`` /
    ``prepare_candidate_env=False`` short-circuits before the git
    worktree and venv steps so plan mode stays cheap.
    """
    candidate_dir = req.work_dir / "candidates" / f"{index:02d}_{candidate.slug}"
    worktree_dir = candidate_dir / "worktree"
    venv_dir = candidate_dir / "venv"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    if not execute or not req.prepare_candidate_env:
        log.debug(
            "prepare_candidate_workspace[%02d] %s: plan mode (no worktree/venv)",
            index, candidate.ref,
        )
        return WorkspacePaths(candidate_dir, worktree_dir, venv_dir)

    repo_dir = prepare_repo_cache(req)
    fetch_candidate_ref(repo_dir, candidate)
    if worktree_dir.exists():
        shutil.rmtree(worktree_dir)
    log.info(
        "prepare_candidate_workspace[%02d] %s: worktree -> %s",
        index, candidate.ref, worktree_dir,
    )
    _run_git(
        [
            "git",
            "--git-dir",
            str(repo_dir),
            "worktree",
            "add",
            "--detach",
            str(worktree_dir),
            _worktree_ref(candidate),
        ]
    )
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    log.info(
        "prepare_candidate_workspace[%02d] %s: venv -> %s",
        index, candidate.ref, venv_dir,
    )
    _run_subprocess(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
        timeout_sec=600,
    )
    return WorkspacePaths(candidate_dir, worktree_dir, venv_dir)


def cleanup_workspace(
    workspace: WorkspacePaths,
    *,
    is_winner: bool,
    keep_winner_only: bool,
    repo_dir: Path | None = None,
) -> None:
    """Drop worktree + venv from disk when policy says so.

    Policy matrix:

    * ``keep_winner_only=False`` (default) — keep everything (legacy behaviour
      to preserve audit material on disk).
    * ``keep_winner_only=True`` and ``is_winner=True``  — keep this candidate.
    * ``keep_winner_only=True`` and ``is_winner=False`` — remove worktree +
      venv so an N-candidate run doesn't waste N×1.5GB of disk on losers.
      The ``candidate_dir`` itself (and ``pr.patches`` / ``pr_files.json``
      audit artefacts inside it) are kept so reviewers can still diff.

    Best-effort: any cleanup error is logged but never re-raised so a
    cleanup failure does not mask the underlying explore result.
    """
    if not keep_winner_only or is_winner:
        return
    if repo_dir is not None:
        # Detach the worktree from the mirror first so `git worktree list`
        # in the repo cache does not point at a now-missing directory.
        try:
            _run_git(
                ["git", "worktree", "remove", "--force", str(workspace.worktree_dir)],
                cwd=repo_dir,
                timeout_sec=60,
            )
        except Exception:  # noqa: BLE001 — fall back to plain rmtree
            log.debug(
                "cleanup_workspace: git worktree remove failed; falling back to rmtree",
                exc_info=True,
            )
    for path in (workspace.worktree_dir, workspace.venv_dir):
        try:
            if path.exists():
                shutil.rmtree(path)
                log.info("cleanup_workspace: removed %s", path)
        except OSError as exc:
            log.warning("cleanup_workspace: failed to remove %s: %s", path, exc)


__all__ = [
    "DiskPreflightError",
    "PER_CANDIDATE_GB",
    "WorkspacePaths",
    "cleanup_workspace",
    "disk_preflight",
    "fetch_candidate_ref",
    "prepare_candidate_workspace",
    "prepare_repo_cache",
]
