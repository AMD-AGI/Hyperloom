# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PR candidate source dispatcher.

Routes :class:`ExploreRequest` to one or more backends per
``request.search_modes`` and merges into a deduplicated :class:`Candidate`
list. Backends: ``gbrain_pr_kb`` (gbrain PR KB pages, best-effort, ``[]`` on
failure), ``primus_cortex`` (Primus Cortex REST, hard-fail on errors) and
``github`` (Search, best-effort, ``[]`` on failure).

Contract: empty ``search_modes`` -> ``[]``; a mode requested without its
config -> :class:`SourceConfigError`; per-mode errors propagate per the
backend's policy.
"""

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

    Args:
        pr: The source PR record.
        repo_url: Repository URL the PR belongs to.
        source: Source label recorded on the candidate (e.g. ``"github"``).
        score: Gap-relevance value from :func:`_rank_by_keyword_overlap`;
            ``0.0`` for paths that skip ranking.

    Returns:
        The downstream :class:`Candidate`.
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


# Error policies for a search backend. ``_HARD_FAIL`` lets the backend's
# exceptions propagate (a misconfigured / unreachable *required* source aborts
# the whole enumeration); ``_BEST_EFFORT`` degrades any failure to ``[]`` so an
# optional source never fails the run.
_HARD_FAIL = "hard_fail"
_BEST_EFFORT = "best_effort"


@dataclass(frozen=True)
class BackendSpec:
    """A PR-source backend: its mode name, runner, and error policy.

    Attributes:
        name (str): The ``search_mode`` id this backend serves (also the
            ``Candidate.source`` label its runner stamps).
        run (Callable): Maps an :class:`ExploreRequest` to that backend's
            candidates.
        error_policy (str): :data:`_HARD_FAIL` or :data:`_BEST_EFFORT`.
    """

    name: str
    run: Callable[[ExploreRequest], list[Candidate]]
    error_policy: str

    def invoke(self, request: ExploreRequest) -> list[Candidate]:
        """Run the backend under its error policy.

        Best-effort backends swallow any exception and return ``[]``; hard-fail
        backends let their exceptions propagate (``SourceConfigError`` for
        missing config, ``PrimusCortexError`` for transport).

        Args:
            request (ExploreRequest): The request to dispatch to this backend.

        Returns:
            list[Candidate]: The backend's candidates, or ``[]`` when a
                best-effort backend failed.
        """
        if self.error_policy == _BEST_EFFORT:
            try:
                return self.run(request)
            except Exception:  # noqa: BLE001 — best-effort source degrades to []
                return []
        return self.run(request)


# Registry of PR-source backends keyed by ``search_mode``. Runners are resolved
# by name at call time (via the module-level ``_run_*`` functions) so a test can
# monkeypatch an individual backend and have the dispatch pick it up.
_SEARCH_BACKENDS: dict[str, BackendSpec] = {
    "gbrain_pr_kb": BackendSpec("gbrain_pr_kb", lambda req: _run_pr_kb(req), _BEST_EFFORT),
    "primus_cortex": BackendSpec("primus_cortex", lambda req: _run_primus_cortex(req), _HARD_FAIL),
    "github": BackendSpec("github", lambda req: _run_github(req), _BEST_EFFORT),
}


def enumerate_candidates(request: ExploreRequest) -> list[Candidate]:
    """Enumerate candidates per ``request.search_modes`` and union the results.

    Order:
      1. Explicit ``candidate_refs`` (always first; source='explicit').
      2. For each enabled mode in ``request.search_modes``: query the
         backend, map to Candidate(source=mode), append.
      3. Deduplicate by ref, preserving the first occurrence.

    Hard-fails when ``primus_cortex`` is requested without configuration,
    or when the primus_cortex transport fails.

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
    """Query the gbrain PR KB; best-effort - empty list on any failure.

    Delegates to :func:`hyperloom.agents.framework.sources.pr_kb.enumerate_pr_kb`,
    which is disabled/degrades to ``[]`` when PR KB is off or gbrain is
    unreachable.

    Args:
        request (ExploreRequest): Request supplying repo URL + gap description.

    Returns:
        list[Candidate]: gbrain_pr_kb candidates, or ``[]`` on any failure.
    """
    from .pr_kb import enumerate_pr_kb

    return enumerate_pr_kb(request)


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
        states=request.pr_states,
    )
    return [_pr_to_candidate(pr, request.repo_url, "github") for pr in prs]


def _resolve_keywords(request: ExploreRequest) -> list[str]:
    """Resolve the keyword list for primus_cortex search + client rerank.

    Priority: (1) ``request.keywords`` non-empty -> used verbatim (lowercased,
    bypasses extract_keywords); (2) ``gap_description`` -> auto-extract via
    :func:`extract_keywords`; (3) else ``[]`` (label-only path).

    Args:
        request: The explore request carrying keywords / gap description.

    Returns:
        The resolved keyword list (possibly empty).
    """
    if request.keywords:
        return [k.lower() for k in request.keywords if k.strip()]
    if (request.gap_description or "").strip():
        return extract_keywords(request.gap_description)
    return []


def _rank_by_keyword_overlap(prs: list[GitHubPr], keywords: list[str]) -> list[GitHubPr]:
    """Stable-rerank PRs by anti-aware keyword score.

    Uses :func:`score_title_with_anti_signal` so wrong-axis PRs (e.g.
    ``MegaMoE`` when gap calls for ``dense``) are demoted below correct-axis
    PRs. Zero-score PRs drop to the tail but are not filtered out.

    Args:
        prs: PRs to rerank.
        keywords: Active keywords driving the score.

    Returns:
        PRs sorted by descending score; ties preserve upstream order.
    """
    if not keywords:
        return list(prs)
    return sorted(
        prs,
        key=lambda pr: score_title_with_anti_signal(pr.title or "", keywords),
        reverse=True,
    )


def _run_primus_cortex(request: ExploreRequest) -> list[Candidate]:
    """Query primus_cortex with gap-aware ranking.

    With non-empty keywords, prefer the free-text ``/v1/search/prs`` endpoint
    (over-fetch, then client-rerank by title overlap, trim to
    ``max_search_candidates``); fall back to ``list_perf_prs`` if search is
    unimplemented. With no keywords, use the cheap label-only path.

    Args:
        request: The explore request carrying the Primus config.

    Returns:
        Candidates from Primus Cortex, ranked when keywords are present.

    Raises:
        SourceConfigError: If ``primus_cortex`` is requested without config.
        PrimusCortexError: On Primus transport errors.
    """
    cfg = request.primus_cortex
    if cfg is None:
        raise SourceConfigError(
            "search_modes contains 'primus_cortex' but no primus_cortex block was provided (nor KB_STORE_URL env var)"
        )
    label = cfg.default_label
    requested = max(1, request.max_search_candidates)

    # Step 4: optionally broaden PR-state coverage. merged/closed PRs are the
    # backport-relevant ones that may already be in the local dev build;
    # semantic audit downstream judges + dedups them. Default remains open-only
    # for perf discovery; enablement explicitly requests "all".
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
        return [_pr_to_candidate(pr, request.repo_url, "primus_cortex") for pr in prs]

    over_fetch = max(requested * 3, requested)
    query = " ".join(keywords)
    try:
        prs = search_perf_prs_via_primus_search(
            request.repo_url,
            base_url=cfg.base_url,
            query=query,
            limit=over_fetch,
            state=search_state,
            timeout_sec=cfg.timeout_sec,
        )
    except PrimusCortexError:
        # Service may not implement /v1/search/prs; fall back to label-only listing.
        prs = list_perf_prs(
            request.repo_url,
            base_url=cfg.base_url,
            limit=over_fetch,
            label=label,
            timeout_sec=cfg.timeout_sec,
            **list_state_kwargs,
        )

    # /v1/search/prs uses word-AND matching; a long query can filter the pool to
    # zero, so fall back to label-only listing + client rerank.
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
