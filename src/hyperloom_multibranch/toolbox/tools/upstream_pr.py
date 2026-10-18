# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``upstream_pr_search`` tool: find upstream pull requests that look relevant, with the framework agent's search."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from hyperloom.agents.framework.models import Candidate, CandidateSearchRequest, PRMonitorConfig
from hyperloom.agents.framework.sources import SourceConfigError, enumerate_candidates
from hyperloom.common.github_urls import repo_slug
from hyperloom_multibranch.toolbox.tools.base import InvalidRequest, ToolFailure, ToolOutput, ToolRequest

UPSTREAM_PR_TOOL = "upstream_pr_search"
MAX_LIMIT = 20
DEFAULT_LIMIT = 5
_STATES = frozenset({"open", "merged", "closed", "all"})

Enumerate = Callable[[CandidateSearchRequest], list[Candidate]]


def _strings(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return tuple(part for part in value.replace(",", " ").split() if part)
    return tuple(str(v).strip() for v in value if str(v).strip())


def build_search(params: Mapping[str, Any]) -> CandidateSearchRequest:
    """Turn tool parameters into the framework agent's search request; raises :class:`InvalidRequest`."""
    repo_url = str(params.get("repo_url") or "").strip()
    try:
        repo_slug(repo_url)
    except ValueError as exc:
        raise InvalidRequest(f"repo_url {repo_url!r} is not a GitHub repository URL") from exc
    try:
        limit = int(DEFAULT_LIMIT if params.get("limit") is None else params["limit"])
    except (TypeError, ValueError) as exc:
        raise InvalidRequest("limit must be an integer") from exc
    if not 1 <= limit <= MAX_LIMIT:
        raise InvalidRequest(f"limit must be between 1 and {MAX_LIMIT}")
    states = _strings(params.get("states")) or ("open",)
    if not set(states) <= _STATES:
        raise InvalidRequest(f"states must come from {sorted(_STATES)}")
    monitor_url = str(params.get("pr_monitor_url") or "").strip()
    return CandidateSearchRequest(
        repo_url=repo_url,
        max_search_candidates=limit,
        pr_monitor=PRMonitorConfig(monitor_url) if monitor_url else None,
        keywords=_strings(params.get("keywords")),
        search_modes=("pr_monitor", "github") if monitor_url else ("github",),
        pr_states=states,
    )


class UpstreamPrSearch:
    """Params: ``repo_url`` (required), ``keywords``, ``limit`` (default 5, at most 20), ``states``, ``pr_monitor_url``."""

    name = UPSTREAM_PR_TOOL

    def __init__(self, enumerate_fn: Enumerate = enumerate_candidates) -> None:
        self._enumerate = enumerate_fn

    def validate(self, request: ToolRequest) -> None:
        build_search(request.params)

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        search = build_search(request.params)
        try:
            found = await asyncio.to_thread(self._enumerate, search)
        except SourceConfigError as exc:
            raise ToolFailure(f"upstream PR search is misconfigured: {exc}") from exc
        except Exception as exc:
            raise ToolFailure(f"upstream PR search failed: {type(exc).__name__}: {exc}") from exc
        prs = [{"ref": c.ref, "title": c.title, "url": c.html_url, "score": c.score, "source": c.source} for c in found]
        top = f"; top: {prs[0]['ref']} {prs[0]['title']}" if prs else ""
        return ToolOutput(
            result={"repo_url": search.repo_url, "keywords": list(search.keywords), "candidates": prs},
            summary=f"{len(prs)} upstream PR candidate(s) in {search.repo_url}{top}",
        )
