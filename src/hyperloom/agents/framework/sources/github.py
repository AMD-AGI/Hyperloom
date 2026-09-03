# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GitHub Search backend for perf PR candidate discovery.

Best-effort, zero-deps fallback when ``pr_monitor`` is unavailable.
No hard-fail: rate-limits, transport errors, or a non-GitHub remote return
``[]``. Queries are keyword-driven from ``gap_description`` via
:func:`hyperloom.agents.framework.keywords.extract_keywords`, falling back to
PERF_TERMS. Anonymous by default (GitHub's 60 req/h IP limit); a bearer token
is attached when ``GITHUB_TOKEN`` / ``GH_TOKEN`` is set.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

from ..keywords import extract_keywords
from ._shared import GitHubPr, _repo_slug


def _auth_headers(accept: str) -> dict[str, str]:
    """Build request headers, adding a bearer token when one is configured.

    Stays anonymous (GitHub's 60 req/h IP limit) when neither ``GITHUB_TOKEN``
    nor ``GH_TOKEN`` is set.
    """
    headers = {"Accept": accept, "User-Agent": "framework-agent/0.1"}
    token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


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
    """Return perf-ish PRs via the GitHub Search API (open-only by default).

    Best-effort: rate-limits or non-GitHub remotes return an empty list
    rather than raising. Callers that need hard-fail behaviour should
    use the ``pr_monitor`` backend instead.

    Args:
        repo_url (str): Git URL of the target repo; parsed to an ``owner/name``
            slug.
        gap_description (str): Free-text gap description used to derive search
            keywords. Defaults to empty.
        limit (int): Maximum number of PRs to return. Defaults to 5.
        states (tuple[str, ...]): PR states to include (``("open",)`` default;
            merged/closed/all broadens beyond open).
        timeout_sec (float): Per-request HTTP timeout in seconds. Defaults to 10.

    Returns:
        list[GitHubPr]: Matching PRs (at most ``limit``), or an empty list
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
    req = urllib.request.Request(url, headers=_auth_headers("application/vnd.github+json"))
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


def pr_patches(repo_slug: str, number: int, *, timeout_sec: float = 30.0) -> str:
    """Return a merged PR's unified diff (``git apply``-ready), or ``""``.

    Requests the diff media type from the GitHub REST API; on any failure
    falls back to the public ``https://github.com/{slug}/pull/{n}.diff`` view.
    Best-effort: transport / rate-limit errors return ``""``.

    Args:
        repo_slug: Repo in ``owner/name`` form.
        number: PR number.
        timeout_sec: Per-request HTTP timeout.

    Returns:
        str: The PR's unified diff text, or ``""`` on failure.
    """
    slug = str(repo_slug or "").strip().strip("/")
    if not slug or number <= 0:
        return ""
    api_url = f"https://api.github.com/repos/{slug}/pulls/{int(number)}"
    for url, accept in (
        (api_url, "application/vnd.github.diff"),
        (f"https://github.com/{slug}/pull/{int(number)}.diff", "text/plain"),
    ):
        req = urllib.request.Request(url, headers=_auth_headers(accept))
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # nosec B310 - fixed GitHub HTTPS URL.
                text = resp.read().decode("utf-8", "replace")
            if text.strip():
                return text
        except Exception:  # noqa: BLE001 - best-effort; try the fallback URL
            continue
    return ""


def fetch_raw_file(repo_slug: str, ref: str, path: str, *, timeout_sec: float = 30.0) -> str:
    """Return a single file's raw contents at ``ref``, or ``""`` on failure.

    Args:
        repo_slug: Repo in ``owner/name`` form.
        ref: Branch / tag / commit SHA.
        path: Repo-relative file path.
        timeout_sec: Per-request HTTP timeout.

    Returns:
        str: The file contents, or ``""`` on any failure.
    """
    slug = str(repo_slug or "").strip().strip("/")
    ref_s = str(ref or "").strip().strip("/")
    path_s = str(path or "").strip().lstrip("/")
    if not slug or not ref_s or not path_s:
        return ""
    url = f"https://raw.githubusercontent.com/{slug}/{ref_s}/{urllib.parse.quote(path_s)}"
    req = urllib.request.Request(url, headers=_auth_headers("text/plain"))
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # nosec B310 - fixed GitHub HTTPS URL.
            return resp.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - best-effort policy
        return ""


__all__ = ["search_perf_prs", "PERF_TERMS", "pr_patches", "fetch_raw_file"]
