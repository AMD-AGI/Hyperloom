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
    "perf", "performance", "throughput", "rocm", "aiter", "flash", "decode",
)


def _build_query(repo: str, gap_description: str) -> str:
    """Compose a GitHub Search query string from gap_description + repo scope."""
    keywords = extract_keywords(gap_description) if gap_description else []
    if not keywords:
        terms = PERF_TERMS
    else:
        terms = tuple(keywords)
    return " ".join(
        [
            f"repo:{repo}",
            "is:pr",
            "is:open",
            "(" + " OR ".join(terms) + ")",
        ]
    )


def search_perf_prs(
    repo_url: str,
    *,
    gap_description: str = "",
    limit: int = 5,
    timeout_sec: float = 10.0,
) -> list[GitHubPr]:
    """Return open perf-ish PRs via anonymous GitHub Search API.

    Best-effort: rate-limits or non-GitHub remotes return an empty list
    rather than raising. Callers that need hard-fail behaviour should
    use the ``primus_cortex`` backend instead.
    """
    try:
        repo = _repo_slug(repo_url)
    except ValueError:
        return []
    query = _build_query(repo, gap_description)
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
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
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
