# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PR candidate source dispatcher."""

from __future__ import annotations

import logging
from typing import Callable, Iterable

from ..keywords import score_title_with_anti_signal
from ..models import Candidate, CandidateSearchRequest
from ._shared import GitHubPr
from . import github as github_backend
from .pr_monitor import (
    PRMonitorError,
    list_perf_prs,
    search_perf_prs_via_pr_monitor_search,
)


class SourceConfigError(RuntimeError):
    """Raised when a requested search_mode is missing its configuration."""


def _dedupe(items: Iterable[Candidate]) -> list[Candidate]:
    """Stable-deduplicate candidates by ref, preserving first-seen order."""
    seen: set[str] = set()
    out: list[Candidate] = []
    for item in items:
        if item.ref in seen:
            continue
        seen.add(item.ref)
        out.append(item)
    return out


def _pr_to_candidate(pr: GitHubPr) -> Candidate:
    """Convert a GitHubPr (any backend) into a downstream Candidate."""
    return Candidate(ref=pr.ref, title=pr.title, html_url=pr.html_url)


_log = logging.getLogger(__name__)


def enumerate_candidates(request: CandidateSearchRequest) -> list[Candidate]:
    """Enumerate candidates per ``request.search_modes`` and union the results."""
    out: list[Candidate] = []
    for mode in request.search_modes:
        runner = _SEARCH_BACKENDS.get(mode)
        if runner is None:
            raise SourceConfigError(f"unknown search_mode: {mode!r}")
        found = runner(request)
        _log.info(
            "enumerate_candidates: %s returned %d candidate(s)",
            mode,
            len(found),
        )
        out.extend(found)

    deduped = _dedupe(out)
    _log.info("enumerate_candidates: total=%d after dedup (searched=%d)", len(deduped), len(out))
    return deduped


def _run_github(request: CandidateSearchRequest) -> list[Candidate]:
    """Query anonymous GitHub Search; network failures yield an empty list."""
    # A request that names the failure searches for it; perf discovery names
    # nothing and takes the perf terms.
    terms = tuple(_resolve_keywords(request)) or github_backend.PERF_TERMS
    prs = github_backend.search_perf_prs(
        request.repo_url,
        limit=request.max_search_candidates,
        terms=terms,
    )
    return [_pr_to_candidate(pr) for pr in prs]


def _resolve_keywords(request: CandidateSearchRequest) -> list[str]:
    """Resolve the keyword list for pr_monitor search + client rerank."""
    return [k.lower() for k in request.keywords if k.strip()]


def _rank_by_keyword_overlap(prs: list[GitHubPr], keywords: list[str]) -> list[GitHubPr]:
    """Stable-rerank PRs by anti-aware keyword score."""
    return sorted(
        prs,
        key=lambda pr: score_title_with_anti_signal(pr.title or "", keywords),
        reverse=True,
    )


def _run_pr_monitor(request: CandidateSearchRequest) -> list[Candidate]:
    """Query pr_monitor, reranking by the request keywords."""
    cfg = request.pr_monitor
    if cfg is None:
        raise SourceConfigError("search_modes contains 'pr_monitor' but the request carries no pr_monitor config")
    requested = max(1, request.max_search_candidates)

    keywords = _resolve_keywords(request)

    if not keywords:
        prs = list_perf_prs(request.repo_url, monitor=cfg, limit=requested)
        return [_pr_to_candidate(pr) for pr in prs]

    over_fetch = requested * 3
    query = " ".join(keywords)
    try:
        prs = search_perf_prs_via_pr_monitor_search(
            request.repo_url,
            monitor=cfg,
            query=query,
            limit=over_fetch,
        )
    except PRMonitorError:
        # Service may not implement /v1/search/prs; fall back to the plain listing.
        prs = list_perf_prs(request.repo_url, monitor=cfg, limit=over_fetch)

    # /v1/search/prs uses word-AND matching; a long query can filter the pool to zero, so fall back to the plain
    # listing + client rerank.
    if not prs:
        prs = list_perf_prs(request.repo_url, monitor=cfg, limit=over_fetch)

    ranked = _rank_by_keyword_overlap(prs, keywords)[:requested]
    return [_pr_to_candidate(pr) for pr in ranked]


# Registry of PR-source runners keyed by ``search_mode``.
_SEARCH_BACKENDS: dict[str, Callable[[CandidateSearchRequest], list[Candidate]]] = {
    "pr_monitor": _run_pr_monitor,
    "github": _run_github,
}


__all__ = [
    "SourceConfigError",
    "enumerate_candidates",
]
