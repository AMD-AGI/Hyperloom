# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PR candidate source dispatcher.

Routes :class:`ExploreRequest` to one or more backends per
``request.search_modes`` and merges into a deduplicated :class:`Candidate`
list. Backends: ``primus_cortex`` (internal REST, hard-fail on errors) and
``github`` (anonymous Search, best-effort, ``[]`` on failure).

Contract: empty ``search_modes`` -> ``[]``; a mode requested without its
config -> :class:`SourceConfigError`; per-mode errors propagate per the
backend's policy.
"""

from __future__ import annotations

from typing import Iterable

from ..logging_setup import get_logger
from ..keywords import (
    extract_keywords,
    score_title_with_anti_signal,
)
from ..models import Candidate, ExploreRequest
from ._shared import GitHubPr
from . import github as github_backend
from .primus_cortex import (
    PrimusCortexError,
    list_perf_prs,
    search_perf_prs_via_primus_search,
)


class SourceConfigError(RuntimeError):
    """Raised when a requested search_mode is missing its configuration."""


def _dedupe(items: Iterable[Candidate]) -> list[Candidate]:
    """Stable-deduplicate candidates by ref, preserving first-seen order.

    Args:
        items (Iterable[Candidate]): Candidates to deduplicate.

    Returns:
        list[Candidate]: Candidates with duplicate refs removed, first-seen
            order preserved.
    """
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
    """Convert a GitHubPr (any backend) into a downstream Candidate.

    ``score`` is the gap-relevance value from :func:`_rank_by_keyword_overlap`;
    0.0 for paths that skip ranking (explicit refs, label-only listing).
    """
    return Candidate(
        ref=pr.ref,
        repo=repo_url,
        source=source,
        title=pr.title,
        html_url=pr.html_url,
        score=float(score),
    )


_log = get_logger(__name__)


def enumerate_candidates(request: ExploreRequest) -> list[Candidate]:
    """Enumerate candidates per ``request.search_modes`` and union the results.

    Order:
      1. Explicit ``candidate_refs`` (always first; source='explicit').
      2. For each enabled mode in ``request.search_modes``: query the
         backend, map to Candidate(source=mode), append.
      3. Deduplicate by ref, preserving the first occurrence.

    Hard-fails when ``primus_cortex`` is requested without configuration,
    or when the primus-cortex transport fails.

    Args:
        request (ExploreRequest): Request carrying explicit refs, search modes,
            repo URL, and search configuration.

    Returns:
        list[Candidate]: Deduplicated candidates unioned across explicit refs
            and every enabled search mode.

    Raises:
        SourceConfigError: If an unknown search mode is requested, or
            ``primus_cortex`` is requested without configuration.
        PrimusCortexError: If a primus_cortex query fails.
    """
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
        if mode == "primus_cortex":
            found = _run_primus_cortex(request)
            _log.info(
                "enumerate_candidates: primus_cortex returned %d candidate(s)",
                len(found),
            )
            out.extend(found)
        elif mode == "github":
            found = _run_github(request)
            _log.info(
                "enumerate_candidates: github returned %d candidate(s)",
                len(found),
            )
            out.extend(found)
        else:
            raise SourceConfigError(f"unknown search_mode: {mode!r}")

    deduped = _dedupe(out)
    _log.info(
        "enumerate_candidates: total=%d after dedup (explicit=%d, searched=%d)",
        len(deduped), len(request.candidate_refs), len(out) - len(request.candidate_refs),
    )
    return deduped


def _run_github(request: ExploreRequest) -> list[Candidate]:
    """Query anonymous GitHub Search; best-effort - empty list on failure.

    Args:
        request (ExploreRequest): Request supplying repo URL, gap description,
            and candidate cap.

    Returns:
        list[Candidate]: Candidates from GitHub Search, or an empty list on any
            failure.
    """
    prs = github_backend.search_perf_prs(
        request.repo_url,
        gap_description=request.gap_description,
        limit=request.max_search_candidates,
    )
    return [_pr_to_candidate(pr, request.repo_url, "github") for pr in prs]


def _resolve_keywords(request: ExploreRequest) -> list[str]:
    """Resolve the keyword list for primus_cortex search + client rerank.

    Priority: (1) ``request.keywords`` non-empty -> used verbatim (lowercased,
    bypasses extract_keywords); (2) ``gap_description`` -> auto-extract via
    :func:`extract_keywords`; (3) else ``[]`` (label-only path).
    """
    if request.keywords:
        return [k.lower() for k in request.keywords if k.strip()]
    if (request.gap_description or "").strip():
        return extract_keywords(request.gap_description)
    return []


def _rank_by_keyword_overlap(
    prs: list[GitHubPr], keywords: list[str]
) -> list[GitHubPr]:
    """Stable-rerank PRs by anti-aware keyword score.

    Uses :func:`score_title_with_anti_signal` so wrong-axis PRs (e.g.
    ``MegaMoE`` when gap calls for ``dense``) are demoted below correct-axis
    PRs. Higher score first; ties preserve upstream order. Zero-score PRs drop
    to the tail but are not filtered out. Anti-signal is gated per-gap-keyword.
    """
    if not keywords:
        return list(prs)
    return sorted(
        prs,
        key=lambda pr: score_title_with_anti_signal(pr.title or "", keywords),
        reverse=True,
    )


def _run_primus_cortex(request: ExploreRequest) -> list[Candidate]:
    """Query primus-cortex with gap-aware ranking; hard-fail on transport errors.

    With non-empty keywords, prefer the free-text ``/v1/search/prs`` endpoint
    (over-fetch, then client-rerank by title overlap, trim to
    ``max_search_candidates``); fall back to ``list_perf_prs`` if search is
    unimplemented. With no keywords, use the cheap label-only path.
    """
    cfg = request.primus_cortex
    if cfg is None:
        raise SourceConfigError(
            "search_modes contains 'primus_cortex' but no primus_cortex "
            "block was provided (nor PRIMUS_CORTEX_PR_API env var)"
        )
    label = cfg.default_label
    requested = max(1, request.max_search_candidates)

    keywords = _resolve_keywords(request)

    if not keywords:
        prs = list_perf_prs(
            request.repo_url,
            base_url=cfg.base_url,
            limit=requested,
            label=label,
            timeout_sec=cfg.timeout_sec,
        )
        return [_pr_to_candidate(pr, request.repo_url, "primus_cortex") for pr in prs]

    over_fetch = max(requested * 3, requested)
    query = " ".join(keywords)
    try:
        prs = search_perf_prs_via_primus_search(
            request.repo_url,
            base_url=cfg.base_url,
            query=query,
            limit=over_fetch,
            state="open",
            timeout_sec=cfg.timeout_sec,
        )
    except PrimusCortexError:
        # Service may not implement /v1/search/prs; fall back to label-only
        # listing and rerank the larger pool client-side.
        prs = list_perf_prs(
            request.repo_url,
            base_url=cfg.base_url,
            limit=over_fetch,
            label=label,
            timeout_sec=cfg.timeout_sec,
        )

    # /v1/search/prs uses word-AND matching, so a long multi-keyword query can
    # filter the pool to zero even when relevant PRs exist. Fall back to
    # label-only listing + client rerank so IO's --framework-pr-discover
    # doesn't abort with "no candidates".
    if not prs:
        prs = list_perf_prs(
            request.repo_url,
            base_url=cfg.base_url,
            limit=over_fetch,
            label=label,
            timeout_sec=cfg.timeout_sec,
        )

    # Rank then trim; scores are carried on Candidate.score for IO's
    # framework_pr arm to log.
    ranked = _rank_by_keyword_overlap(prs, keywords)[:requested]
    return [
        _pr_to_candidate(
            pr,
            request.repo_url,
            "primus_cortex",
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
