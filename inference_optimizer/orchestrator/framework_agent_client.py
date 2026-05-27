"""Coordinator-side thin client for the framework-agent FRAMEWORK_PR
phase subcommands (``fa phase-discover`` / ``fa phase-fetch`` /
``fa phase-emit-proposal``).

Used by :meth:`Coordinator._run_framework_pr_phase` to drive the
per-candidate pump that lives between PRELUDE and EXPLORE. Each
subcommand is invoked via ``asyncio.to_thread`` so the Coordinator
reactor loop never blocks on the CLI; failures degrade to empty /
``status=failed`` results instead of raising.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)


# Repo URL table — keyed by framework name. Mirrors
# ``specialist_domains.SpecialistDomain.pr_repos`` for serving_specialist
# but flattens it to the single repo URL each ``fa phase-*`` request
# carries (the request schema accepts one ``repo_url`` per call).
_FRAMEWORK_TO_REPO_URL: dict[str, str] = {
    "sglang": "https://github.com/sgl-project/sglang.git",
    "vllm":   "https://github.com/ROCm/vllm.git",
}


def repo_url_for_framework(framework: str) -> str:
    """Return the canonical GitHub repo URL for ``framework``.

    Returns an empty string for unknown frameworks; the caller is
    expected to bail out / log when this happens.
    """
    return _FRAMEWORK_TO_REPO_URL.get((framework or "").strip().lower(), "")


def _resolve_fa_binary() -> str | None:
    """Return the absolute path to the ``fa`` binary, or ``None`` if
    it cannot be located.

    Resolution order:

    1. ``$FA_BIN`` env var (operator pin / unit-test injection).
    2. ``shutil.which('fa')`` — the production path after
       ``framework-agent/scripts/install.sh`` ran.
    3. ``$FRAMEWORK_AGENT_ROOT/scripts/fa`` — repo-local fallback for
       sessions that source the repo's runtime env but don't have the
       global ``fa`` symlink on $PATH yet.
    """
    explicit = (os.environ.get("FA_BIN") or "").strip()
    if explicit and Path(explicit).exists():
        return explicit
    via_path = shutil.which("fa")
    if via_path:
        return via_path
    fa_root = (os.environ.get("FRAMEWORK_AGENT_ROOT") or "").strip()
    if fa_root:
        candidate = Path(fa_root) / "scripts" / "fa"
        if candidate.exists():
            return str(candidate)
    return None


DEFAULT_FA_PHASE_TIMEOUT_SEC: float = 60.0


def _run_fa_subcommand_sync(
    fa_bin: str,
    subcommand: str,
    request_path: Path,
    timeout_sec: float,
) -> "tuple[int, str, str]":
    """Sync helper: run ``fa <subcommand> --request <path> --out -``.

    Mirrors :func:`_run_fa_candidates_sync` for the FRAMEWORK_PR phase
    subcommands (``phase-discover`` / ``phase-fetch`` /
    ``phase-emit-proposal``). Never raises.
    """
    cmd = [fa_bin, subcommand, "--request", str(request_path), "--out", "-"]
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError as exc:
        return 127, "", f"fa binary not found: {exc!r}"
    except subprocess.TimeoutExpired as exc:
        return 124, "", f"fa {subcommand} timed out after {timeout_sec}s: {exc!r}"
    return cp.returncode, cp.stdout, cp.stderr


async def _invoke_fa_phase(
    *,
    subcommand: str,
    request: dict[str, Any],
    session_dir: Path,
    timeout_sec: float = DEFAULT_FA_PHASE_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Generic async runner for ``fa phase-*`` subcommands.

    Writes ``request`` as a temp JSON, runs the subcommand, returns the
    parsed JSON output. Raises :class:`RuntimeError` on binary missing
    / non-zero exit / JSON parse failure so callers can degrade.
    """
    fa_bin = _resolve_fa_binary()
    if not fa_bin:
        raise RuntimeError(
            f"fa binary not found (subcommand={subcommand!r}); "
            "checked $FA_BIN, $PATH, $FRAMEWORK_AGENT_ROOT/scripts/fa"
        )
    tmp_dir = session_dir / ".fa-tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    request_path = tmp_dir / f"phase-{subcommand}-{uuid.uuid4().hex[:12]}.json"
    request_path.write_text(
        json.dumps(request, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    try:
        rc, stdout, stderr = await asyncio.to_thread(
            _run_fa_subcommand_sync,
            fa_bin, subcommand, request_path, timeout_sec,
        )
    finally:
        with contextlib.suppress(OSError):
            request_path.unlink()
    if rc != 0:
        raise RuntimeError(
            f"fa {subcommand} exited rc={rc}; stderr={(stderr or '')[-512:]!r}"
        )
    try:
        return json.loads(stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(
            f"fa {subcommand} produced invalid JSON: {exc!r}; "
            f"first 200 chars={ (stdout or '')[:200]!r}"
        )


async def phase_discover(
    *,
    model: str,
    framework: str,
    gpu_type: str,
    gaps: list[dict[str, str]],
    session_dir: Path,
    repo_url: str = "",
    max_candidates: int = 5,
    batch_id: str = "",
    timeout_sec: float = DEFAULT_FA_PHASE_TIMEOUT_SEC,
) -> dict[str, Any]:
    """FRAMEWORK_PR-phase batch discovery shim.

    Returns the parsed payload from ``fa phase-discover``:
    ``{batch_id, framework, repo_url, candidates: [...]}``.
    """
    resolved_repo_url = (repo_url or repo_url_for_framework(framework)).strip()
    request = {
        "model":     model,
        "framework": (framework or "sglang").strip().lower(),
        "gpu_type":  gpu_type,
        "gaps":      gaps,
        "repo_url":  resolved_repo_url,
        "work_dir":  str(session_dir / ".fa-tmp" / "phase-discover"),
        "max_search_candidates": int(max_candidates),
        "batch_id":  batch_id,
    }
    return await _invoke_fa_phase(
        subcommand="phase-discover",
        request=request,
        session_dir=session_dir,
        timeout_sec=timeout_sec,
    )


async def phase_fetch(
    *,
    pr_url: str,
    repo: str,
    ref: str,
    framework: str,
    worktree_dir: Path,
    session_dir: Path,
    repo_url: str = "",
    title: str = "",
    timeout_sec: float = DEFAULT_FA_PHASE_TIMEOUT_SEC,
) -> dict[str, Any]:
    """FRAMEWORK_PR-phase candidate-fetch shim.

    Returns ``{status, worktree_path, applied_files, message}``.
    """
    request = {
        "pr_url":       pr_url,
        "repo":         repo,
        "ref":          ref,
        "framework":    (framework or "sglang").strip().lower(),
        "repo_url":     repo_url or repo_url_for_framework(framework),
        "worktree_dir": str(worktree_dir),
        "title":        title,
    }
    return await _invoke_fa_phase(
        subcommand="phase-fetch",
        request=request,
        session_dir=session_dir,
        timeout_sec=timeout_sec,
    )


async def phase_emit_proposal(
    *,
    task_id: str,
    pr_url: str,
    worktree_path: str,
    gap_canonical_id: str,
    framework: str,
    session_dir: Path,
    patches_written: list[str] | None = None,
    rationale: str = "",
    timeout_sec: float = DEFAULT_FA_PHASE_TIMEOUT_SEC,
) -> dict[str, Any]:
    """FRAMEWORK_PR-phase ``specialist_done`` envelope emitter.

    Returns the envelope dict produced by ``fa phase-emit-proposal``.
    """
    request = {
        "task_id":          task_id,
        "pr_url":           pr_url,
        "worktree_path":    worktree_path,
        "gap_canonical_id": gap_canonical_id,
        "framework":        (framework or "sglang").strip().lower(),
        "patches_written":  list(patches_written or []),
        "rationale":        rationale,
    }
    return await _invoke_fa_phase(
        subcommand="phase-emit-proposal",
        request=request,
        session_dir=session_dir,
        timeout_sec=timeout_sec,
    )


__all__ = [
    "DEFAULT_FA_PHASE_TIMEOUT_SEC",
    "phase_discover",
    "phase_fetch",
    "phase_emit_proposal",
    "repo_url_for_framework",
]
