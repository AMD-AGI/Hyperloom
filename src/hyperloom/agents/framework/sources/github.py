# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Anonymous GitHub Search backend for perf PR candidate discovery.

Best-effort, zero-deps fallback when ``primus_cortex`` is unavailable.
No hard-fail: rate-limits, transport errors, or a non-GitHub remote return
``[]``. Queries are keyword-driven from ``gap_description`` via
:func:`framework_agent.keywords.extract_keywords`, falling back to PERF_TERMS.
Anonymous (no token), so subject to GitHub's 60 req/h IP limit.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

from ..keywords import extract_keywords
from ._shared import GitHubPr, _repo_slug


PERF_TERMS = (
    "perf",
    "performance",
    "throughput",
    "rocm",
    "aiter",
    "flash",
    "decode",
)


def _state_qualifier(states: tuple[str, ...]) -> str:
    """Map pr_states to a GitHub search state qualifier.

    Open-only (default) keeps ``is:open``; any merged/closed/all broadens to
    all PR states (omit the state qualifier) so merged PRs are searchable.

    Args:
        states: The requested PR states.

    Returns:
        ``"is:open"`` for open-only, else ``""`` (all states).
    """
    broad = any(s in ("merged", "closed", "all") for s in (states or ("open",)))
    return "" if broad else "is:open"


def _build_query(repo: str, gap_description: str, states: tuple[str, ...] = ("open",)) -> str:
    """Compose a GitHub Search query string from gap_description + repo scope.

    Keywords extracted from ``gap_description`` drive the OR-term clause;
    when no keywords are found the curated :data:`PERF_TERMS` list is used.

    Args:
        repo (str): Repository slug in ``owner/name`` form to scope the search.
        gap_description (str): Free-text gap description used to derive search
            keywords.
        states (tuple[str, ...]): PR states to include (``("open",)`` default;
            merged/closed/all broadens beyond open).

    Returns:
        str: A GitHub Search query scoped to ``repo`` and the requested states.
    """
    keywords = extract_keywords(gap_description) if gap_description else []
    if not keywords:
        terms = PERF_TERMS
    else:
        terms = tuple(keywords)
    parts = [f"repo:{repo}", "is:pr"]
    state_q = _state_qualifier(states)
    if state_q:
        parts.append(state_q)
    parts.append("(" + " OR ".join(terms) + ")")
    return " ".join(parts)


def search_perf_prs(
    repo_url: str,
    *,
    gap_description: str = "",
    limit: int = 5,
    states: tuple[str, ...] = ("open",),
    timeout_sec: float = 10.0,
) -> list[GitHubPr]:
    """Return open perf-ish PRs via anonymous GitHub Search API.

    Best-effort: rate-limits or non-GitHub remotes return an empty list
    rather than raising. Callers that need hard-fail behaviour should
    use the ``primus_cortex`` backend instead.

    Args:
        repo_url (str): Git URL of the target repo; parsed to an ``owner/name``
            slug.
        gap_description (str): Free-text gap description used to derive search
            keywords. Defaults to empty.
        limit (int): Maximum number of PRs to return. Defaults to 5.
        timeout_sec (float): Per-request HTTP timeout in seconds. Defaults to 10.

    Returns:
        list[GitHubPr]: Matching open PRs (at most ``limit``), or an empty list
            on any failure or non-GitHub remote.
    """
    try:
        repo = _repo_slug(repo_url)
    except ValueError:
        return []
    query = _build_query(repo, gap_description, states)
    url = "https://api.github.com/search/issues?" + urllib.parse.urlencode(
        {"q": query, "sort": "updated", "order": "desc", "per_page": str(limit)}
    )
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "framework-agent/0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # nosec B310 - fixed GitHub HTTPS API URL.
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - best-effort policy
        return []
    out: list[GitHubPr] = []
    for item in data.get("items") or []:
        if not isinstance(item, dict):
            continue
        number = item.get("number")
        if not isinstance(number, int):
            continue
        out.append(
            GitHubPr(
                number=number,
                title=str(item.get("title") or ""),
                html_url=str(item.get("html_url") or ""),
            )
        )
    return out[:limit]


__all__ = ["search_perf_prs", "PERF_TERMS"]
