# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PR candidate source dispatcher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from ..logging_setup import get_logger
from ..keywords import (
    extract_keywords,
    score_title_with_anti_signal,
)
from ..models import Candidate, ExploreRequest
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


def _pr_to_candidate(
    pr: GitHubPr,
    repo_url: str,
    source: str,
    *,
    score: float = 0.0,
) -> Candidate:
    """Convert a GitHubPr (any backend) into a downstream Candidate."""
    return Candidate(
        ref=pr.ref,
        repo=repo_url,
        source=source,
        title=pr.title,
        html_url=pr.html_url,
        score=float(score),
    )


_log = get_logger(__name__)


# Error policies for a search backend.
_HARD_FAIL = "hard_fail"
_BEST_EFFORT = "best_effort"


@dataclass(frozen=True)
class BackendSpec:
    """A PR-source backend: its mode name, runner, and error policy."""

    name: str
    run: Callable[[ExploreRequest], list[Candidate]]
    error_policy: str

    def invoke(self, request: ExploreRequest) -> list[Candidate]:
        """Run the backend under its error policy."""
        if self.error_policy == _BEST_EFFORT:
            try:
                return self.run(request)
            except Exception:  # noqa: BLE001 — best-effort source degrades to []
                return []
        return self.run(request)


# Registry of PR-source backends keyed by ``search_mode``.
_SEARCH_BACKENDS: dict[str, BackendSpec] = {
    "gbrain_pr_kb": BackendSpec("gbrain_pr_kb", lambda req: _run_pr_kb(req), _BEST_EFFORT),
    "pr_monitor": BackendSpec("pr_monitor", lambda req: _run_pr_monitor(req), _HARD_FAIL),
    "github": BackendSpec("github", lambda req: _run_github(req), _BEST_EFFORT),
}


def enumerate_candidates(request: ExploreRequest) -> list[Candidate]:
    """Enumerate candidates per ``request.search_modes`` and union the results."""
    out: list[Candidate] = []

    for ref in request.candidate_refs:
        out.append(Candidate(ref=ref, repo=request.repo_url, source="explicit"))

    if not request.search_perf_prs:
        _log.info(
            "enumerate_candidates: search_perf_prs=False; explicit_refs=%d",
            len(request.candidate_refs),
        )
        return _dedupe(out)

    for mode in request.search_modes:
        spec = _SEARCH_BACKENDS.get(mode)
        if spec is None:
            raise SourceConfigError(f"unknown search_mode: {mode!r}")
        found = spec.invoke(request)
        _log.info(
            "enumerate_candidates: %s returned %d candidate(s)",
            spec.name,
            len(found),
        )
        out.extend(found)

    deduped = _dedupe(out)
    _log.info(
        "enumerate_candidates: total=%d after dedup (explicit=%d, searched=%d)",
        len(deduped),
        len(request.candidate_refs),
        len(out) - len(request.candidate_refs),
    )
    return deduped


def _run_pr_kb(request: ExploreRequest) -> list[Candidate]:
    """Query the gbrain PR KB; best-effort - empty list on any failure."""
    from .pr_kb import enumerate_pr_kb

    return enumerate_pr_kb(request)


def _run_github(request: ExploreRequest) -> list[Candidate]:
    """Query anonymous GitHub Search; best-effort - empty list on failure."""
    prs = github_backend.search_perf_prs(
        request.repo_url,
        gap_description=request.gap_description,
        limit=request.max_search_candidates,
        states=request.pr_states,
    )
    return [_pr_to_candidate(pr, request.repo_url, "github") for pr in prs]


def _resolve_keywords(request: ExploreRequest) -> list[str]:
    """Resolve the keyword list for pr_monitor search + client rerank."""
    if request.keywords:
        return [k.lower() for k in request.keywords if k.strip()]
    if (request.gap_description or "").strip():
        return extract_keywords(request.gap_description)
    return []


def _rank_by_keyword_overlap(prs: list[GitHubPr], keywords: list[str]) -> list[GitHubPr]:
    """Stable-rerank PRs by anti-aware keyword score."""
    if not keywords:
        return list(prs)
    return sorted(
        prs,
        key=lambda pr: score_title_with_anti_signal(pr.title or "", keywords),
        reverse=True,
    )


def _run_pr_monitor(request: ExploreRequest) -> list[Candidate]:
    """Query pr_monitor with gap-aware ranking."""
    cfg = request.pr_monitor
    if cfg is None:
        raise SourceConfigError(
            "search_modes contains 'pr_monitor', but PR Monitor is unavailable; "
            "provide pr_monitor.base_url (remote mode requires KB_STORE_URL) or "
            "remove 'pr_monitor' from search_modes"
        )
    label = cfg.default_label
    requested = max(1, request.max_search_candidates)

    # Step 4: optionally broaden PR-state coverage. merged/closed PRs are the backport-relevant ones that may already
    # be in the local dev build; semantic audit downstream judges + dedups them.
    states = request.pr_states
    broad = any(s in ("merged", "closed", "all") for s in states)
    search_state = "all" if broad else "open"
    # Only forward ``state`` to the label-only list endpoint when broadening.
    list_state_kwargs: dict[str, str] = {"state": search_state} if broad else {}

    keywords = _resolve_keywords(request)

    if not keywords:
        prs = list_perf_prs(
            request.repo_url,
            base_url=cfg.base_url,
            limit=requested,
            label=label,
            timeout_sec=cfg.timeout_sec,
            **list_state_kwargs,
        )
        return [_pr_to_candidate(pr, request.repo_url, "pr_monitor") for pr in prs]

    over_fetch = max(requested * 3, requested)
    query = " ".join(keywords)
    try:
        prs = search_perf_prs_via_pr_monitor_search(
            request.repo_url,
            base_url=cfg.base_url,
            query=query,
            limit=over_fetch,
            state=search_state,
            timeout_sec=cfg.timeout_sec,
        )
    except PRMonitorError:
        # Service may not implement /v1/search/prs; fall back to label-only listing.
        prs = list_perf_prs(
            request.repo_url,
            base_url=cfg.base_url,
            limit=over_fetch,
            label=label,
            timeout_sec=cfg.timeout_sec,
            **list_state_kwargs,
        )

    # /v1/search/prs uses word-AND matching; a long query can filter the pool to zero, so fall back to label-only
    # listing + client rerank.
    if not prs:
        prs = list_perf_prs(
            request.repo_url,
            base_url=cfg.base_url,
            limit=over_fetch,
            label=label,
            timeout_sec=cfg.timeout_sec,
            **list_state_kwargs,
        )

    # Rank then trim; scores are carried on Candidate.score.
    ranked = _rank_by_keyword_overlap(prs, keywords)[:requested]
    return [
        _pr_to_candidate(
            pr,
            request.repo_url,
            "pr_monitor",
            score=score_title_with_anti_signal(pr.title or "", keywords),
        )
        for pr in ranked
    ]


__all__ = [
    "SourceConfigError",
    "enumerate_candidates",
    "_rank_by_keyword_overlap",
    "_resolve_keywords",
]
