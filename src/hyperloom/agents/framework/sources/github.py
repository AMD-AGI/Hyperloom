# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The strict search of GitHub's pull requests.

``search_issues`` is for a caller that must tell a GitHub that cannot be asked from one that found nothing: it raises
:class:`GitHubUnavailable`, which says why, and an empty answer means GitHub answered with none.
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

_ENCODING = "utf-8"


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


_API = "https://api.github.com"
_JSON = "application/vnd.github+json"
_HTTP_UNAUTHORIZED = 401
_HTTP_FORBIDDEN, _HTTP_TOO_MANY = 403, 429
#: How long GitHub's documentation says to wait after a rate limit whose answer names no wait.
_UNNAMED_WAIT_SEC = 60
_MINUTE = 60


class GitHubUnavailable(RuntimeError):
    """GitHub's search cannot be used from this host; the message says why."""


def token_configured() -> bool:
    """Whether this process's environment holds a GitHub token, so that a request carries ``Authorization``."""
    return "Authorization" in _auth_headers(_JSON)


def _token_hint() -> str:
    """What to do about a refusal of a request that was made without a token."""
    if token_configured():
        return ""
    return "; the host has no GITHUB_TOKEN or GH_TOKEN, and an anonymous host is allowed very few requests"


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


def _refusal(refused: urllib.error.HTTPError) -> str:
    """Why GitHub refused the search, and whether asking again later can help."""
    code = refused.code
    if code == _HTTP_UNAUTHORIZED:
        return "GitHub rejected the token of this host (GITHUB_TOKEN or GH_TOKEN)"
    if code not in (_HTTP_FORBIDDEN, _HTTP_TOO_MANY):
        return f"GitHub answered the search with HTTP {code}"
    hint = _token_hint()
    wait = _wait_sec(refused)
    if wait is None:
        who = "this host" if hint else "this host's token"
        return f"GitHub does not allow {who} the search (HTTP {code}); asking again will not change that{hint}"
    after = f"{wait} s" if wait < 2 * _MINUTE else f"{-(-wait // _MINUTE)} min"
    return f"GitHub's rate limit for this host is spent (HTTP {code}): ask again in {after}{hint}"


def _unreachable(exc: Exception) -> GitHubUnavailable:
    return GitHubUnavailable(
        f"GitHub cannot be reached from this host ({type(exc).__name__}); check its network and proxy"
    )


def _get(path: str, timeout_sec: float) -> bytes:
    """What GitHub's API answers the search at ``path`` with.

    Raises:
        GitHubUnavailable: GitHub could not be reached, or refused, or the token of this host cannot be sent.
    """
    request = urllib.request.Request(f"{_API}{path}", headers=_auth_headers(_JSON))
    try:
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # nosec B310 - fixed GitHub HTTPS API host.
                return response.read()
        except urllib.error.HTTPError as exc:
            raise GitHubUnavailable(_refusal(exc)) from None
    except ValueError:  # http.client's, for a header it cannot send, whose value its text quotes; only the token varies
        raise GitHubUnavailable(
            "the GitHub token of this host (GITHUB_TOKEN or GH_TOKEN) holds a line break or another character no "
            "header can carry"
        ) from None
    except (OSError, http.client.HTTPException) as exc:
        raise _unreachable(exc) from None


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
    body = _get(path, timeout_sec)
    try:
        answer = json.loads(body.decode(_ENCODING))
    except ValueError as exc:  # a proxy or a captive portal answered in GitHub's place
        raise _unreachable(exc) from None
    if (
        not isinstance(answer, dict)
        or not isinstance(answer.get("items"), list)
        or not isinstance(answer.get("total_count"), int)
    ):
        raise GitHubUnavailable("GitHub answered the search with something that is not a search result")
    return SearchAnswer(answer["items"], answer["total_count"])


#: The public name of ``_build_query``, the search query of a repository, for a caller of :func:`search_issues`.
build_query = _build_query

__all__ = [
    "MAX_OR_TERMS",
    "PERF_TERMS",
    "GitHubUnavailable",
    "SearchAnswer",
    "build_query",
    "search_issues",
    "token_configured",
]
