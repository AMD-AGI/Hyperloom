# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Upstream PR reference lookup for one forge-loop run."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from hyperloom.common.env import env_bool
from kernelforge.tools.git import git

if TYPE_CHECKING:
    # Imported lazily at runtime to keep CLI startup off the knowledge stack.
    from kernelforge.knowledge.pr_knowledge.context import PrReferenceQuery
    from kernelforge.knowledge.pr_knowledge.references import PRRefsResult
    from kernelforge.loop.bootstrap.run_context import _ForgeRun
    from kernelforge.loop.iteration.iteration_types import IterationConfig


# End-to-end wall clock for every PR reference lookup of one run: preflight, repository listing, path probing,
# discovery, and detail enrichment.
PR_KB_BUDGET_SEC = 45.0


def _pr_kb_enabled(flag: bool | None) -> bool:
    """Resolve the PR KB switch: CLI flag wins, env is the fallback, default off."""
    if flag is not None:
        return bool(flag)
    return env_bool("PR_KB_ENABLE")


def _git_remote_url(workspace: Path) -> str:
    """Return the origin URL, or ``""`` when it cannot be read."""
    try:
        result = git(
            "remote",
            "get-url",
            "origin",
            cwd=workspace,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(f"Warning: failed to read git origin: {error}", file=sys.stderr)
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _pr_refs_event_fields(reason: str, stats: dict) -> dict:
    """Build the PR refresh event appended after campaign initialization."""
    return {
        "position": "A",
        "reason": reason or "ok",
        "degraded_reason": stats.get("degraded_reason"),
        "candidates": stats.get("candidates"),
        "surfaced": stats.get("surfaced"),
        "injected_entries": stats.get("injected_entries"),
        "injected_bytes": stats.get("injected_bytes"),
        "http_calls": stats.get("http_calls"),
        "skipped_cached_empty": stats.get("skipped_cached_empty"),
        "from_snapshot": stats.get("from_snapshot"),
        "distill_absent": stats.get("distill_absent"),
        "distill_dropped": stats.get("distill_dropped"),
        "fallback_used": stats.get("fallback_used"),
        "relevance_dropped": stats.get("relevance_dropped"),
    }


def _collect_pr_references(workspace_dir: str, query: PrReferenceQuery, budget_sec: float) -> PRRefsResult | None:
    """Run position A, absorbing every recoverable failure into a warning."""
    from kernelforge.knowledge.pr_knowledge.references import (
        PR_KB_RECOVERABLE,
        collect_references,
    )

    try:
        return collect_references(
            workspace_dir=workspace_dir,
            query=query,
            budget_sec=budget_sec,
            # The campaign freshness guard runs inside the loop; until it passes this invocation may not modify the
            # workspace it was pointed at.
            persist=False,
        )
    except PR_KB_RECOVERABLE as error:
        print(f"  [pr-kb] unavailable ({type(error).__name__}: {error})")
        return None


def _write_pr_provenance(
    *,
    workspace_dir: str,
    surfaced: tuple[str, ...],
    winning_iteration: int,
    experiment_id: str = "",
) -> None:
    """Write exposure data for the references injected into this run."""
    if not surfaced:
        return

    from kernelforge.knowledge.pr_knowledge.references import (
        PR_KB_RECOVERABLE,
        write_provenance,
    )

    try:
        write_provenance(
            workspace_dir,
            {
                "experiment_id": experiment_id,
                "winning_iteration": winning_iteration,
                "surfaced": list(surfaced),
            },
        )
    except PR_KB_RECOVERABLE as error:
        print(f"Warning: failed to write PR provenance: {error}", file=sys.stderr)


def _collect_pr_kb_context(run: _ForgeRun) -> tuple[str, str]:
    """Inject PR references into the loop's prompt context, and say which repo they came from.

    Independent of the experience-KB lifecycle, and skipped outright when
    too little of the deadline is left to collect them.
    """
    if not _pr_kb_enabled(run.request.knowledge.pr_kb):
        return "", ""
    from kernelforge.knowledge.pr_knowledge.context import (
        REASON_LOCAL_FAILURE,
        REASON_SKIPPED_DEADLINE,
        PrReferenceQuery,
    )

    iter_config, campaign, deadline = run.iter_config, run.campaign, run.deadline
    if deadline.remaining(reserve=deadline.finalize_reserve_sec) < 20.0:
        print("  [pr-kb] skipped (deadline)")
        iter_config.pr_kb_event = _pr_refs_event_fields(REASON_SKIPPED_DEADLINE, {})
        return "", ""
    pr_result = _collect_pr_references(
        iter_config.workspace_dir,
        PrReferenceQuery(
            kernel_backend=campaign.kernel_backend or "",
            git_remote=_git_remote_url(Path(iter_config.workspace_dir).resolve()),
            source_files=campaign.source_files,
            operator_name=campaign.operator_name or "",
            target_functions=iter_config.target_functions,
        ),
        min(PR_KB_BUDGET_SEC, deadline.remaining(reserve=deadline.finalize_reserve_sec)),
    )
    if pr_result is None:
        iter_config.pr_kb_event = _pr_refs_event_fields(REASON_LOCAL_FAILURE, {"degraded_reason": REASON_LOCAL_FAILURE})
        return "", ""
    _record_pr_references(iter_config, pr_result)
    return pr_result.prompt_context, pr_result.repo


def _record_pr_references(iter_config: IterationConfig, pr_result: PRRefsResult) -> None:
    """Carry the references a lookup returned into the loop's prompt context, its events and its provenance."""
    iter_config.pr_reference_context = pr_result.prompt_context
    iter_config.pr_reference_labels = tuple(
        f"{reference.repo}#{reference.number}" for reference in pr_result.references
    )
    iter_config.pr_kb_event = _pr_refs_event_fields(pr_result.reason, pr_result.stats)
    iter_config.pr_kb_snapshot = pr_result.pending_snapshot
    if pr_result.injected:
        print(
            f"  [pr-kb] injected {pr_result.stats.get('injected_entries', 0)} "
            f"references ({pr_result.stats.get('injected_bytes', 0)} B)"
        )
    else:
        print(f"  [pr-kb] no references ({pr_result.reason or 'empty'})")
