"""Coordinator-side thin client for the framework-agent FRAMEWORK_PR
phase subcommands.

Only ``fa phase-discover`` is actually wired into the Coordinator
pump: the pump calls it to obtain a batch of candidate PRs, then the
Coordinator's own Critic gate + ``FrameworkPrExecutor`` (which curls
``candidate.diff_url`` directly and runs ``git apply``) handles the
rest. ``fa phase-fetch`` and ``fa phase-emit-proposal`` remain
available on the standalone ``fa`` CLI but are NOT wrapped here —
adding shims for unused subcommands invited the "dead API misleads
readers" problem flagged in the PR-327 review.

``phase_discover`` is invoked via ``asyncio.to_thread`` so the
Coordinator reactor loop never blocks on the CLI; failures degrade to
empty / raised ``RuntimeError`` results that the pump's retry counter
absorbs.
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
#
# The canonical mapping lives in ``framework_agent.repo_map`` so the
# standalone ``fa`` CLI can resolve repo URLs without reverse-importing
# inference_optimizer. We delegate to that module when available
# (the normal install path puts framework-agent on the same venv) and
# fall back to an inline copy for IO-only test environments.
try:
    from framework_agent.repo_map import (  # type: ignore[import-not-found]
        repo_url_for_framework,
    )
except ImportError:  # pragma: no cover — exercised only in IO-only test envs

    _FRAMEWORK_TO_REPO_URL: dict[str, str] = {
        "sglang": "https://github.com/sgl-project/sglang.git",
        "vllm":   "https://github.com/ROCm/vllm.git",
    }

    def repo_url_for_framework(framework: str) -> str:
        """Return the canonical GitHub repo URL for ``framework``.

        Returns an empty string for unknown frameworks; the caller is
        expected to bail out / log when this happens.
        """
        return _FRAMEWORK_TO_REPO_URL.get(
            (framework or "").strip().lower(), "",
        )


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


DEFAULT_FA_PHASE_TIMEOUT_SEC: float = 180.0
# Number of consecutive ``fa phase-discover`` failures the Coordinator
# tolerates before marking ``framework_pr_phase_done = True`` and
# advancing to EXPLORE. Bumped from the implicit-1 of the original
# silent-collapse behaviour so a transient timeout or a slow PR scan
# doesn't kill the whole phase.
DISCOVER_FAILURE_RETRY_LIMIT: int = 3


def _run_fa_subcommand_sync(
    fa_bin: str,
    subcommand: str,
    request_path: Path,
    timeout_sec: float,
) -> "tuple[int, str, str]":
    """Sync helper: run ``fa <subcommand> --request <path> --out -``.

    Only ``phase-discover`` calls into here today; the function stays
    subcommand-agnostic so adding a second wired subcommand later
    (should we ever need to bring ``phase-emit-proposal`` back into
    the Coordinator loop) does not require touching it. Never raises.
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


# NOTE: ``phase_fetch`` / ``phase_emit_proposal`` shims used to live
# here but had zero callers in inference_optimizer (the Coordinator
# pump only calls ``phase_discover``; ``FrameworkPrExecutor`` curls
# ``candidate.diff_url`` directly via ``git apply``). The PR-327 review
# flagged them as dead-API-that-misleads-readers, so they were removed.
# The standalone ``fa`` CLI still ships both subcommands for ad-hoc /
# external use — re-add wrappers here only when the Coordinator
# actually wires a new caller.


__all__ = [
    "DEFAULT_FA_PHASE_TIMEOUT_SEC",
    "DISCOVER_FAILURE_RETRY_LIMIT",
    "phase_discover",
    "repo_url_for_framework",
]
