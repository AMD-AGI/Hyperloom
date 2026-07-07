# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coordinator-side thin client for the framework-agent
``fa phase-discover`` subcommand.

The only framework-agent entry point wired into the Coordinator pump:
returns a batch of candidate PRs; the Critic gate + ``FrameworkAgentExecutor``
handle the rest. Invoked via ``asyncio.to_thread`` so the reactor never
blocks; failures degrade to empty / ``RuntimeError`` that the pump absorbs.
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

# tree-reform.MD P2.5: framework-agent was promoted into
# hyperloom.agents.framework (always importable alongside orchestrator now
# that both live under the same installed ``hyperloom`` distribution), so the
# former IO-only ``except ImportError`` fallback copy of
# ``_FRAMEWORK_TO_REPO_URL`` is no longer needed.
from hyperloom.agents.framework.repo_map import repo_url_for_framework


log = logging.getLogger(__name__)


def _resolve_fa_binary() -> str | None:
    """Return the absolute path to the ``fa`` binary, or ``None``.

    Resolution order: ``$FA_BIN``; ``shutil.which('fa')``;
    ``$FRAMEWORK_AGENT_ROOT/scripts/fa``.

    Returns:
        The absolute path to the ``fa`` binary, or ``None`` when it cannot be
        located.
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
# Consecutive ``fa phase-discover`` failures tolerated before advancing to EXPLORE.
DISCOVER_FAILURE_RETRY_LIMIT: int = 3


def _run_fa_subcommand_sync(
    fa_bin: str,
    subcommand: str,
    request_path: Path,
    timeout_sec: float,
) -> "tuple[int, str, str]":
    """Sync helper: run ``fa <subcommand> --request <path> --out -``. Never raises.

    Args:
        fa_bin: Path to the ``fa`` binary.
        subcommand: The ``fa`` subcommand to run.
        request_path: Path to the request JSON file.
        timeout_sec: Subprocess wall-clock timeout in seconds.

    Returns:
        A ``(returncode, stdout, stderr)`` tuple; failures map to ``127``
        (missing binary) or ``124`` (timeout).
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

    Writes ``request`` as temp JSON, runs the subcommand, returns parsed
    JSON. Raises :class:`RuntimeError` on missing binary / non-zero exit /
    parse failure.

    Args:
        subcommand: The ``fa phase-*`` subcommand to run.
        request: The request payload serialized to temp JSON.
        session_dir: The session directory under which the temp request lives.
        timeout_sec: Subprocess wall-clock timeout in seconds.

    Returns:
        The parsed JSON payload returned by the subcommand.

    Raises:
        RuntimeError: If the ``fa`` binary is missing, the subcommand exits
            non-zero, or its output is not valid JSON.
    """
    fa_bin = _resolve_fa_binary()
    if not fa_bin:
        raise RuntimeError(
            f"fa binary not found (subcommand={subcommand!r}); checked $FA_BIN, $PATH, $FRAMEWORK_AGENT_ROOT/scripts/fa"
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
            fa_bin,
            subcommand,
            request_path,
            timeout_sec,
        )
    finally:
        with contextlib.suppress(OSError):
            request_path.unlink()
    if rc != 0:
        raise RuntimeError(f"fa {subcommand} exited rc={rc}; stderr={(stderr or '')[-512:]!r}")
    try:
        return json.loads(stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(f"fa {subcommand} produced invalid JSON: {exc!r}; first 200 chars={(stdout or '')[:200]!r}")


async def phase_discover(
    *,
    model: str,
    framework: str,
    gpu_type: str,
    gaps: list[dict[str, str]],
    session_dir: Path,
    repo_url: str = "",
    keywords: list[str] | None = None,
    max_candidates: int = 5,
    batch_id: str = "",
    pr_states: list[str] | None = None,
    excluded_candidate_ids: list[str] | None = None,
    failed_candidate_context: list[dict[str, Any]] | None = None,
    timeout_sec: float = DEFAULT_FA_PHASE_TIMEOUT_SEC,
) -> dict[str, Any]:
    """FRAMEWORK-phase batch discovery shim.

    Non-empty ``keywords`` is used verbatim for the primus-cortex AND-search
    (fa skips its own ``extract_keywords``); empty/``None`` keeps the legacy
    behaviour. Returns the ``fa phase-discover`` payload
    ``{batch_id, framework, repo_url, candidates: [...]}``.

    Args:
        model: The model identifier.
        framework: The target framework (defaults to ``sglang`` when empty).
        gpu_type: The target GPU type.
        gaps: The performance gaps driving discovery.
        session_dir: The session directory for temp request staging.
        repo_url: Optional explicit repo URL; resolved from ``framework`` when
            empty.
        keywords: Optional verbatim AND-search keywords; ``None``/empty keeps
            the legacy keyword extraction.
        max_candidates: Maximum number of candidate PRs to request.
        batch_id: Optional batch identifier.
        pr_states: Optional PR-state filter (open / merged / closed).
        excluded_candidate_ids: Candidate ids (pr_url / ref / ``PR:<n>``) the
            session has already discovered or reached a terminal verdict on;
            fa hard-filters these out of the batch (Step B hard-dedup).
        failed_candidate_context: Compact ``{ref, status, gain_pct, why}`` rows
            of candidates that already failed this session; fa uses them to
            de-prioritise same-PR / equivalent candidates. Truncated to the
            most recent 10 to keep the request bounded.
        timeout_sec: Subprocess wall-clock timeout in seconds.

    Returns:
        The ``fa phase-discover`` payload dict.
    """
    resolved_repo_url = (repo_url or repo_url_for_framework(framework)).strip()
    request = {
        "model": model,
        "framework": (framework or "sglang").strip().lower(),
        "gpu_type": gpu_type,
        "gaps": gaps,
        "repo_url": resolved_repo_url,
        "work_dir": str(session_dir / ".fa-tmp" / "phase-discover"),
        "max_search_candidates": int(max_candidates),
        "batch_id": batch_id,
    }
    excluded = [str(x).strip() for x in (excluded_candidate_ids or []) if str(x).strip()]
    if excluded:
        # Dedup preserving order for a deterministic request.
        seen_ex: set[str] = set()
        request["excluded_candidate_ids"] = [x for x in excluded if not (x in seen_ex or seen_ex.add(x))]
    failed_ctx = [f for f in (failed_candidate_context or []) if isinstance(f, dict)]
    if failed_ctx:
        request["failed_candidate_context"] = failed_ctx[-10:]
    # Plumb the primus_cortex (PR-Monitor) base URL so the fa subprocess can
    # query internal primus PRs; without it phase-discover falls back to
    # GitHub-only. The PR-Monitor REST service IS the primus_cortex backend.
    primus_url = (os.environ.get("PRIMUS_CORTEX_PR_API") or os.environ.get("PR_MONITOR_URL") or "").strip()
    if primus_url:
        request["primus_cortex_url"] = primus_url
    ps = [str(s).strip().lower() for s in (pr_states or []) if str(s).strip()]
    if ps:
        request["pr_states"] = ps
    kw = [str(k).strip().lower() for k in (keywords or []) if str(k).strip()]
    if kw:
        # Dedup preserving order for a deterministic request.
        seen: set[str] = set()
        request["keywords"] = [k for k in kw if not (k in seen or seen.add(k))]
    return await _invoke_fa_phase(
        subcommand="phase-discover",
        request=request,
        session_dir=session_dir,
        timeout_sec=timeout_sec,
    )


async def phase_audit(
    *,
    candidate: dict[str, Any],
    framework: str,
    framework_source_roots: list[str],
    session_dir: Path,
    repo_url: str = "",
    diff_url: str = "",
    diff_text: str = "",
    primus_cortex_url: str = "",
    use_llm: bool = False,
    model: str = "",
    timeout_sec: float = DEFAULT_FA_PHASE_TIMEOUT_SEC,
) -> dict[str, Any]:
    """FRAMEWORK-phase semantic-audit shim (``fa phase-audit``).

    Builds the request and runs the subcommand; returns the
    ``semantic_audit`` verdict. The caller treats a missing binary / non-zero
    exit / parse failure (raised as :class:`RuntimeError`) as ``unknown`` and
    preserves legacy routing.

    Args:
        candidate: The discovered candidate row (carries repo / pr_number /
            diff_url / pr_url / ref).
        framework: The target framework.
        framework_source_roots: Live source roots the audit reads to judge.
        session_dir: Session dir for temp request staging + audit work_dir.
        repo_url: Optional explicit repo URL (else from candidate / framework).
        diff_url: Optional explicit unified-diff URL.
        diff_text: Optional inline unified diff (skips any fetch).
        primus_cortex_url: Optional Primus Cortex base URL for patch fetch.
        use_llm: Opt-in single chat-completion refine (default off).
        model: Optional LLM model slug for the refine layer.
        timeout_sec: Subprocess wall-clock timeout in seconds.

    Returns:
        The ``fa phase-audit`` verdict dict.
    """
    request: dict[str, Any] = {
        "candidate": candidate,
        "framework": (framework or "sglang").strip().lower(),
        "framework_source_roots": list(framework_source_roots or []),
        "repo_url": (repo_url or str(candidate.get("repo") or "")).strip(),
        "work_dir": str(session_dir / ".fa-tmp" / "phase-audit"),
        "use_llm": bool(use_llm),
    }
    if diff_url or candidate.get("diff_url"):
        request["diff_url"] = (diff_url or str(candidate.get("diff_url") or "")).strip()
    if diff_text:
        request["diff_text"] = diff_text
    if primus_cortex_url:
        request["primus_cortex_url"] = primus_cortex_url
    if model:
        request["model"] = model
    return await _invoke_fa_phase(
        subcommand="phase-audit",
        request=request,
        session_dir=session_dir,
        timeout_sec=timeout_sec,
    )


# NOTE: ``phase_fetch`` / ``phase_emit_proposal`` shims were removed as dead
# API (zero callers); re-add only when the Coordinator wires a new caller.


__all__ = [
    "DEFAULT_FA_PHASE_TIMEOUT_SEC",
    "DISCOVER_FAILURE_RETRY_LIMIT",
    "phase_audit",
    "phase_discover",
    "repo_url_for_framework",
]
