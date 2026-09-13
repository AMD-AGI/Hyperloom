# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GitHub Search backend for perf PR candidate discovery."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

from ..keywords import extract_keywords
from ._shared import GitHubPr, _repo_slug


def _auth_headers(accept: str) -> dict[str, str]:
    """Build request headers, adding a bearer token when one is configured."""
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
    """Map pr_states to a GitHub search state qualifier."""
    broad = any(s in ("merged", "closed", "all") for s in (states or ("open",)))
    return "" if broad else "is:open"


def _build_query(repo: str, gap_description: str, states: tuple[str, ...] = ("open",)) -> str:
    """Compose a GitHub Search query string from gap_description + repo scope."""
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
    """Return perf-ish PRs via the GitHub Search API (open-only by default)."""
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
    """Return a merged PR's unified diff (``git apply``-ready), or ``\"\"``."""
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
    """Return a single file's raw contents at ``ref``, or ``\"\"`` on failure."""
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
