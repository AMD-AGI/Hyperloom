# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Upstream PR reference lookup for one forge-loop run."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import click

from hyperloom.common.env import env_bool
from kernelforge.llm.git import git

if TYPE_CHECKING:
    # Imported lazily at runtime to keep CLI startup off the knowledge stack.
    from kernelforge.knowledge.pr_knowledge.references import PRRefsResult


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
        click.echo(f"Warning: failed to read git origin: {error}", err=True)
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


def _collect_pr_references(
    *,
    workspace_dir: str,
    kernel_backend: str,
    git_remote: str,
    source_files: Iterable[str],
    operator_name: str,
    target_functions: Iterable[str],
    budget_sec: float,
) -> PRRefsResult | None:
    """Run position A, absorbing every recoverable failure into a warning."""
    from kernelforge.knowledge.pr_knowledge.references import (
        PR_KB_RECOVERABLE,
        collect_references,
    )

    try:
        return collect_references(
            workspace_dir=workspace_dir,
            kernel_backend=kernel_backend,
            git_remote=git_remote,
            source_files=source_files,
            operator_name=operator_name,
            target_functions=target_functions,
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
        click.echo(f"Warning: failed to write PR provenance: {error}", err=True)
