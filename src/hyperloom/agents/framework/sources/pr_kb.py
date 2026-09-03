# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PR KB (gbrain) candidate discovery backend.

Enumerates candidate PRs from the gbrain PR KB for ``request.repo_url``:
index page (structured) ∪ semantic ``search`` (scoped to this repo's meta
prefix), with a ``list_pages`` fallback when both yield nothing. Merged +
deduped by PR number. Best-effort: any gbrain failure yields ``[]`` so the
dispatcher falls back to PR Monitor / GitHub.
"""

from __future__ import annotations

import os

from ..gbrain_page_client import GbrainPageError, build_gbrain_page_client_from_env
from ..logging_setup import get_logger
from ..models import Candidate, ExploreRequest
from ..pr_kb import index_slug, parse_index_prs
from ..pr_kb_slug import files_slug, normalise_repo, repo_slug, slug_prefix

_log = get_logger(__name__)


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

    # index page — structured enumeration.
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

    # semantic query — scoped to this repo's meta prefix.
    meta_prefix = f"{slug_prefix()}-meta/{repo_slug(repo_n)}/pr/"
    try:
        hits = client.query(request.gap_description or " ".join(request.keywords), limit=limit * 3)
    except GbrainPageError as exc:
        _log.warning("pr_kb: query failed: %r", exc)
        hits = []
    for hit in hits:
        slug = str(hit.get("slug") or hit.get("path") or "")
        if not slug.startswith(meta_prefix):
            continue
        tail = slug.rsplit("/", 1)[-1]
        if not tail.isdigit():
            continue
        num = int(tail)
        if num not in by_number:
            by_number[num] = _candidate(request.repo_url, repo_n, num, title=str(hit.get("title") or ""))

    # list_pages fallback — enumerate meta pages directly and client-side
    # filter by this repo's meta prefix (best-effort; no server-side filter).
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
                by_number[num] = _candidate(request.repo_url, repo_n, num, title=str(entry.get("title") or ""))
        # Warn when the cap was hit with no match: matches may be beyond the cap.
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
