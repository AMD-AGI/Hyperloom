# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GitHub Search backend for perf PR candidate discovery, and the strict reads of a repository's pull requests.

``search_perf_prs`` and ``pr_patches`` are best-effort: a GitHub that cannot be asked reads as "nothing found". The
strict reads (``search_issues``, ``pull_request``, ``pr_diff``) are for a caller that must tell the two apart: they
raise :class:`GitHubUnavailable`, which says why, and an empty answer means GitHub answered with none.
"""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from hyperloom.common.github_urls import repo_slug as parse_repo_slug

from ._shared import GitHubPr

_ENCODING = "utf-8"
_PR_TIMEOUT_SEC = 30.0


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

#: GitHub Search answers HTTP 422 past five boolean operators.
MAX_OR_TERMS = 6


def _build_query(repo: str, *, terms: tuple[str, ...]) -> str:
    """Compose a GitHub Search query scoped to ``repo`` that ORs the first ``MAX_OR_TERMS`` of ``terms``."""
    return f"repo:{repo} is:pr (" + " OR ".join(terms[:MAX_OR_TERMS]) + ")"


def search_perf_prs(
    repo_url: str,
    *,
    limit: int = 5,
    terms: tuple[str, ...],
    timeout_sec: float = 10.0,
) -> list[GitHubPr]:
    """Return PRs in every state matching ``terms`` via the GitHub Search API."""
    try:
        repo = parse_repo_slug(repo_url)
    except ValueError:
        return []
    query = _build_query(repo, terms=terms)
    url = "https://api.github.com/search/issues?" + urllib.parse.urlencode({"q": query, "per_page": str(limit)})
    req = urllib.request.Request(url, headers=_auth_headers("application/vnd.github+json"))
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:  # nosec B310 - fixed GitHub HTTPS API URL.
            data = json.loads(resp.read().decode(_ENCODING))
    except (OSError, http.client.HTTPException, ValueError):  # best-effort policy
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


def pr_patches(repo_slug: str, number: int, *, timeout_sec: float = _PR_TIMEOUT_SEC) -> str:
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
                text = resp.read().decode(_ENCODING, "replace")
            if text.strip():
                return text
        except (OSError, http.client.HTTPException, ValueError):  # best-effort; try the fallback URL
            continue
    return ""


_API = "https://api.github.com"
_JSON = "application/vnd.github+json"
_DIFF = "application/vnd.github.diff"
_HTTP_UNAUTHORIZED = 401
_HTTP_FORBIDDEN, _HTTP_TOO_MANY = 403, 429
#: How long GitHub's documentation says to wait after a rate limit whose answer names no wait.
_UNNAMED_WAIT_SEC = 60
_MINUTE = 60
#: What GitHub answers the diff of a pull request too large for it to render with.
_HTTP_DIFF_REFUSED = (406, 422)


class GitHubUnavailable(RuntimeError):
    """GitHub cannot be used from this host; the message says why. ``status`` is the HTTP status GitHub refused with,
    ``None`` when it was not reached or answered something other than what was asked."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class DiffRefused(GitHubUnavailable):
    """GitHub knows the pull request and serves no diff of it."""


@dataclass(frozen=True)
class PullRequest:
    """A pull request as GitHub serves it."""

    number: int
    title: str
    html_url: str


def token_configured() -> bool:
    """Whether this process's environment holds a GitHub token, so that a request carries ``Authorization``."""
    return "Authorization" in _auth_headers(_JSON)


def _token_hint() -> str:
    """What to do about a refusal of a request that was made without a token."""
    if token_configured():
        return ""
    return "; the host has no GITHUB_TOKEN or GH_TOKEN, and an anonymous search is allowed very few requests"


def _wait_sec(refused: urllib.error.HTTPError) -> int | None:
    """The seconds GitHub asks this host to wait before it asks again, ``None`` when ``refused`` is no rate limit.

    GitHub names the wait of a secondary limit in ``Retry-After`` and the end of a spent quota in
    ``x-ratelimit-reset``. A secondary limit may name neither: only its message then tells it from a request the
    token is not allowed, and GitHub's documentation asks for :data:`_UNNAMED_WAIT_SEC`.
    """
    headers = refused.headers
    retry_after = headers.get("Retry-After") or ""
    if retry_after.isdecimal():
        return int(retry_after)
    reset = headers.get("x-ratelimit-reset") or ""
    if headers.get("x-ratelimit-remaining") == "0" and reset.isdecimal():
        return max(int(reset) - int(time.time()), 1)
    if refused.code == _HTTP_TOO_MANY or b"rate limit" in refused.read().lower():
        return _UNNAMED_WAIT_SEC
    return None


def _refusal(refused: urllib.error.HTTPError, asked: str) -> str:
    """Why GitHub refused, and whether asking again later can help."""
    code = refused.code
    if code == _HTTP_UNAUTHORIZED:
        return "GitHub rejected the token of this host (GITHUB_TOKEN or GH_TOKEN)"
    if code not in (_HTTP_FORBIDDEN, _HTTP_TOO_MANY):
        return f"GitHub answered {asked} with HTTP {code}"
    hint = _token_hint()
    wait = _wait_sec(refused)
    if wait is None:
        who = "this host" if hint else "this host's token"
        return f"GitHub does not allow {who} {asked} (HTTP {code}); asking again will not change that{hint}"
    after = f"{wait} s" if wait < 2 * _MINUTE else f"{-(-wait // _MINUTE)} min"
    return f"GitHub's rate limit for this host is spent (HTTP {code}): ask again in {after}{hint}"


def _unreachable(exc: Exception) -> GitHubUnavailable:
    return GitHubUnavailable(
        f"GitHub cannot be reached from this host ({type(exc).__name__}); check its network and proxy"
    )


def _get(path: str, *, accept: str, asked: str, timeout_sec: float) -> bytes:
    """What GitHub's API answers ``path`` with; ``asked`` names the request in a refusal.

    Raises:
        GitHubUnavailable: GitHub could not be reached, or refused, or the token of this host cannot be sent.
    """
    request = urllib.request.Request(f"{_API}{path}", headers=_auth_headers(accept))
    try:
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # nosec B310 - fixed GitHub HTTPS API host.
                return response.read()
        except urllib.error.HTTPError as exc:
            raise GitHubUnavailable(_refusal(exc, asked), exc.code) from None
    except ValueError:  # http.client's, for a header it cannot send, whose value its text quotes; only the token varies
        raise GitHubUnavailable(
            "the GitHub token of this host (GITHUB_TOKEN or GH_TOKEN) holds a line break or another character no "
            "header can carry"
        ) from None
    except (OSError, http.client.HTTPException) as exc:
        raise _unreachable(exc) from None


def _get_json(path: str, *, asked: str, timeout_sec: float) -> Any:
    body = _get(path, accept=_JSON, asked=asked, timeout_sec=timeout_sec)
    try:
        return json.loads(body.decode(_ENCODING))
    except ValueError as exc:  # a proxy or a captive portal answered in GitHub's place
        raise _unreachable(exc) from None


def _repo_path(repo_url: str) -> str:
    """``/repos/<owner>/<name>`` of the GitHub repository at ``repo_url``; raises ``ValueError`` for any other URL."""
    return f"/repos/{urllib.parse.quote(parse_repo_slug(repo_url))}"


@dataclass(frozen=True)
class SearchAnswer:
    """The first ``items`` GitHub's search found, best match first, and ``total_count``: all that matched."""

    items: list[Any]
    total_count: int


def search_issues(query: str, *, per_page: int, timeout_sec: float = 10.0) -> SearchAnswer:
    """What GitHub's issue and pull-request search answers ``query`` with.

    Raises:
        GitHubUnavailable: GitHub could not be asked, refused, or did not answer with a search result.
    """
    path = "/search/issues?" + urllib.parse.urlencode({"q": query, "per_page": str(per_page)})
    answer = _get_json(path, asked="the search", timeout_sec=timeout_sec)
    if (
        not isinstance(answer, dict)
        or not isinstance(answer.get("items"), list)
        or not isinstance(answer.get("total_count"), int)
    ):
        raise GitHubUnavailable("GitHub answered the search with something that is not a search result")
    return SearchAnswer(answer["items"], answer["total_count"])


def pull_request(repo_url: str, number: int, *, timeout_sec: float = _PR_TIMEOUT_SEC) -> PullRequest:
    """The pull request ``number`` of the repository at ``repo_url``.

    Raises:
        ValueError: ``repo_url`` is not a GitHub repository URL.
        GitHubUnavailable: GitHub could not be asked, refused, or did not answer with a pull request.
    """
    asked = f"pull request {number}"
    item = _get_json(f"{_repo_path(repo_url)}/pulls/{number}", asked=asked, timeout_sec=timeout_sec)
    if not isinstance(item, dict) or not isinstance(item.get("number"), int):
        raise GitHubUnavailable(f"GitHub answered {asked} with something that is not a pull request")
    return PullRequest(
        number=item["number"], title=str(item.get("title") or ""), html_url=str(item.get("html_url") or "")
    )


def pr_diff(repo_url: str, number: int, *, timeout_sec: float = _PR_TIMEOUT_SEC) -> str:
    """The unified diff of the pull request ``number`` of the repository at ``repo_url``.

    Raises:
        ValueError: ``repo_url`` is not a GitHub repository URL.
        DiffRefused: GitHub serves no diff of this pull request.
        GitHubUnavailable: GitHub could not be asked, or refused for another reason.
    """
    path = f"{_repo_path(repo_url)}/pulls/{number}"
    try:
        body = _get(path, accept=_DIFF, asked=f"the diff of pull request {number}", timeout_sec=timeout_sec)
    except GitHubUnavailable as exc:
        if exc.status in _HTTP_DIFF_REFUSED:
            raise DiffRefused(
                f"GitHub serves no diff of pull request {number} (HTTP {exc.status}): it is too large", exc.status
            ) from None
        raise
    return body.decode(_ENCODING, "replace")


#: The public name of ``_build_query``, the search query of a repository, for a caller of :func:`search_issues`.
build_query = _build_query

__all__ = [
    "MAX_OR_TERMS",
    "PERF_TERMS",
    "DiffRefused",
    "GitHubUnavailable",
    "PullRequest",
    "SearchAnswer",
    "build_query",
    "pr_diff",
    "pr_patches",
    "pull_request",
    "search_issues",
    "search_perf_prs",
    "token_configured",
]
