# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PR candidate source dispatcher.

Routes :class:`ExploreRequest` to one or more backends based on
``request.search_modes`` and merges results into a single deduplicated
list of :class:`Candidate` records.

Backends:

* ``primus_cortex`` - internal REST service (hard-fail on errors).
* ``github``        - anonymous GitHub Search fallback (best-effort,
  may be rate-limited; returns empty list on failure).

Dispatcher contract:

* If ``search_modes`` is empty -> return ``[]`` (caller must rely on
  ``candidate_refs``).
* If a mode is configured but its config is missing (e.g. ``primus_cortex``
  requested without ``primus_cortex`` block / env var) -> raise
  :class:`SourceConfigError`.
* Per-mode errors propagate per the backend's policy (primus_cortex
  hard-fails on network/parse errors; github is best-effort and
  returns an empty list when the GitHub Search API is unavailable
  or rate-limited).
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

    ``score`` carries the gap-relevance value produced by
    :func:`_rank_by_keyword_overlap`; defaults to 0.0 for code paths that
    skip ranking (explicit refs, label-only listing with no keywords).

    Args:
        pr (GitHubPr): Source PR record from a discovery backend.
        repo_url (str): Repo URL to record on the candidate.
        source (str): Origin tag (e.g. ``"primus_cortex"`` or ``"github"``).
        score (float): Gap-relevance score to attach. Defaults to 0.0.

    Returns:
        Candidate: A candidate carrying the PR ref, repo, title, URL, and score.
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
    """Resolve the keyword list for the primus_cortex search + client rerank.

    Priority (matches the C-extension UX contract):

    1. ``request.keywords`` non-empty -> use it verbatim (explicit override
       from the IO ``--framework-keywords`` flag). Lowercased; whitelist
       and CamelCase logic are bypassed.
    2. Else ``gap_description`` non-empty -> auto-extract via
       :func:`extract_keywords` (whitelist filter + CamelCase rescue).
    3. Else -> empty list. Caller drops into the cheapest label-only
       :func:`list_perf_prs` path.

    Args:
        request (ExploreRequest): Request carrying explicit ``keywords`` and/or
            ``gap_description``.

    Returns:
        list[str]: Lowercased keyword list, possibly empty.
    """
    if request.keywords:
        return [k.lower() for k in request.keywords if k.strip()]
    if (request.gap_description or "").strip():
        return extract_keywords(request.gap_description)
    return []


def _rank_by_keyword_overlap(
    prs: list[GitHubPr], keywords: list[str]
) -> list[GitHubPr]:
    """Stable-rerank PRs by anti-aware keyword score (B3 anti-correlation fix).

    Uses :func:`score_title_with_anti_signal` so PRs whose titles contain
    tokens *opposite* to the gap (e.g. ``MegaMoE`` when gap calls for
    ``dense``, NVIDIA Hopper signals when gap targets ``mi300x``,
    ``fp8`` quant when gap targets ``bf16``) are demoted below relevant
    PRs that score positive on the correct axis.

    Higher score first; ties preserve upstream order (Python's sort is
    stable). PRs that score zero (no positive overlap, or positive
    fully erased by anti penalty) drop to the tail but are NOT
    filtered out — callers that want to drop them can post-filter on
    score themselves.

    Anti-signal activation is gated per-gap-keyword (see
    :data:`framework_agent.keywords._ANTI_KEYWORDS`), so when the gap
    has no orthogonal-axis trigger the behaviour is identical to the
    prior positive-overlap-only scoring.

    Args:
        prs (list[GitHubPr]): PRs to rerank.
        keywords (list[str]): Active gap keywords; empty leaves order untouched.

    Returns:
        list[GitHubPr]: PRs sorted by descending anti-aware score, ties stable.
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

    Strategy (B2 fix):

    * If ``gap_description`` yields non-empty keywords, prefer the
      free-text ``/v1/search/prs`` endpoint with an over-fetch factor
      so the service returns PRs relevant to the gap; then rerank
      client-side by keyword overlap with the PR title and trim to
      ``max_search_candidates``.
    * If the service does not implement free-text search (404 / other
      :class:`PrimusCortexError` from the search call), fall back to
      ``list_perf_prs`` with the same over-fetch + client-rerank
      pipeline so we still get the best of the available pool.
    * If ``gap_description`` is empty or the extractor returns no
      keywords, preserve the old label-only behaviour (cheapest path).

    Args:
        request (ExploreRequest): Request carrying repo URL, primus_cortex
            config, keywords/gap description, and candidate cap.

    Returns:
        list[Candidate]: Gap-ranked candidates from primus_cortex, trimmed to
            ``max_search_candidates``.

    Raises:
        SourceConfigError: If no primus_cortex configuration is present.
        PrimusCortexError: On primus_cortex transport / parse errors from the
            listing endpoint.
    """
    cfg = request.primus_cortex
    if cfg is None:
        raise SourceConfigError(
            "search_modes contains 'primus_cortex' but no primus_cortex "
            "block was provided (nor PRIMUS_CORTEX_PR_API env var)"
        )
    label = cfg.default_label
    requested = max(1, request.max_search_candidates)

    # C: explicit request.keywords (from IO --framework-keywords) wins over
    # the gap_description auto-extract path. See ``_resolve_keywords``.
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
        # The service may not implement /v1/search/prs; fall back to
        # the label-only listing endpoint and apply the same client-side
        # rerank to the larger pool.
        prs = list_perf_prs(
            request.repo_url,
            base_url=cfg.base_url,
            limit=over_fetch,
            label=label,
            timeout_sec=cfg.timeout_sec,
        )

    # B2 v2: primus' /v1/search/prs uses word-AND matching, so a long
    # multi-keyword query (e.g. "fp8 moe sglang") can easily filter the
    # whole candidate pool to zero even when relevant PRs exist. When
    # that happens, fall back to the label-only listing endpoint and let
    # the client-side rerank do its job on the broader pool. This keeps
    # IO's --framework-pr-discover from aborting with
    # FrameworkPRError("no candidates") whenever the gap text happens to
    # land on a query the service-side index can't AND-match.
    if not prs:
        prs = list_perf_prs(
            request.repo_url,
            base_url=cfg.base_url,
            limit=over_fetch,
            label=label,
            timeout_sec=cfg.timeout_sec,
        )

    # Rank then trim. Compute scores alongside the sort so the dispatcher
    # can transport the per-candidate relevance value over to IO via the
    # Candidate.score field (downstream framework_pr arm logs them in
    # state.arms[framework_pr].history).
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
