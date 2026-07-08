# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PR KB (gbrain) candidate discovery backend (design P2, D6).

Enumerates candidate PRs from the gbrain PR KB for ``request.repo_url``:
index page (structured) ∪ semantic ``search`` (scoped to this repo's meta
prefix), with a ``list_pages`` fallback when both yield nothing (the index
page may be absent and ``search`` does not surface pr-kb-meta pages). Merged
+ deduped by PR number. Best-effort: any gbrain failure yields ``[]`` so the
dispatcher falls back to Cortex / GitHub.
"""

from __future__ import annotations

import os

from ..gbrain_page_client import GbrainPageError, build_gbrain_page_client_from_env
from ..logging_setup import get_logger
from ..models import Candidate, ExploreRequest
from ..pr_kb import index_slug, parse_index_prs
from ..pr_kb_slug import files_slug, normalise_repo, repo_slug, slug_prefix

_log = get_logger(__name__)


def _min_relevance() -> float:
    """Return the query relevance floor (``PR_KB_MIN_RELEVANCE``, default 0.2)."""
    raw = os.environ.get("PR_KB_MIN_RELEVANCE", "")
    try:
        return float(raw) if raw else 0.2
    except ValueError:
        return 0.2


def _candidate(repo_url: str, repo_n: str, number: int, *, title: str = "", labels: tuple[str, ...] = ()) -> Candidate:
    """Build a gbrain_pr_kb Candidate from a repo + PR number."""
    return Candidate(
        ref=f"PR:{number}",
        repo=repo_url,
        source="gbrain_pr_kb",
        title=title,
        labels=labels,
        html_url=f"https://github.com/{repo_n}/pull/{number}",
        pr_kb_files_slug=files_slug(repo_n, number),
    )


def enumerate_pr_kb(request: ExploreRequest) -> list[Candidate]:
    """Enumerate PR KB candidates for ``request.repo_url`` (index ∪ query).

    Args:
        request: The explore request (repo_url + gap_description drive it).

    Returns:
        Deduped gbrain_pr_kb candidates; ``[]`` when disabled/unconfigured/unreachable.
    """
    if (os.environ.get("PR_KB_ENABLE", "1") or "1").strip() == "0":
        return []
    client = build_gbrain_page_client_from_env()
    if client is None:
        return []
    if not client.health():
        _log.info("pr_kb: gbrain unreachable; skipping source")
        return []

    repo_n = normalise_repo(request.repo_url)
    if not repo_n:
        return []
    limit = max(1, int(request.max_search_candidates or 5))
    by_number: dict[int, Candidate] = {}

    # (1) index page — structured enumeration.
    try:
        page = client.get_page(index_slug(repo_n))
        for entry in parse_index_prs(page or {}):
            num = entry.get("number")
            if not isinstance(num, int):
                continue
            labels = tuple(str(x) for x in (entry.get("labels") or []) if str(x).strip())
            by_number[num] = _candidate(
                request.repo_url, repo_n, num, title=str(entry.get("title") or ""), labels=labels
            )
    except GbrainPageError as exc:
        _log.warning("pr_kb: index page fetch failed: %r", exc)

    # (2) semantic query — scoped to this repo's meta prefix.
    meta_prefix = f"{slug_prefix()}-meta/{repo_slug(repo_n)}/pr/"
    floor = _min_relevance()
    try:
        hits = client.query(request.gap_description or " ".join(request.keywords), limit=limit * 3)
    except GbrainPageError as exc:
        _log.warning("pr_kb: query failed: %r", exc)
        hits = []
    for hit in hits:
        slug = str(hit.get("slug") or hit.get("path") or "")
        if not slug.startswith(meta_prefix):
            continue
        rel = hit.get("score", hit.get("relevance"))
        if isinstance(rel, (int, float)) and float(rel) < floor:
            continue
        tail = slug.rsplit("/", 1)[-1]
        if not tail.isdigit():
            continue
        num = int(tail)
        if num not in by_number:
            by_number[num] = _candidate(
                request.repo_url, repo_n, num, title=str(hit.get("title") or "")
            )

    # (3) list_pages fallback — the index page may be absent and the semantic
    # ``search`` tool does not surface pr-kb-meta pages, so when the first two
    # sources yield nothing, enumerate meta pages directly and client-side
    # filter by this repo's meta prefix (gbrain list_pages has no server-side
    # prefix filter and caps results, so this is best-effort; design D6).
    if not by_number:
        list_cap = 500
        try:
            pages = client.list_pages(limit=list_cap)
        except GbrainPageError as exc:
            _log.warning("pr_kb: list_pages fallback failed: %r", exc)
            pages = []
        for entry in pages:
            slug = str(entry.get("slug") or entry.get("path") or "")
            if not slug.startswith(meta_prefix):
                continue
            tail = slug.rsplit("/", 1)[-1]
            if not tail.isdigit():
                continue
            num = int(tail)
            if num not in by_number:
                by_number[num] = _candidate(
                    request.repo_url, repo_n, num, title=str(entry.get("title") or "")
                )
        # list_pages has no server-side prefix filter and caps its result set.
        # When it returns a full page (>= cap) and none matched this repo, the
        # repo's meta pages may simply be beyond the cap — surface that as a
        # WARNING so "this repo suddenly finds no candidates" is diagnosable
        # rather than a silent empty return.
        if not by_number and len(pages) >= list_cap:
            _log.warning(
                "pr_kb: list_pages hit the %d-page cap with no match for %s "
                "(meta prefix %r) — its meta pages may be beyond the cap; "
                "candidates under-reported",
                list_cap,
                repo_n,
                meta_prefix,
            )

    out = sorted(by_number.values(), key=lambda c: c.pr_number or 0, reverse=True)[:limit]
    _log.info("pr_kb: enumerated %d candidate(s) for %s", len(out), repo_n)
    return out


__all__ = ["enumerate_pr_kb"]
