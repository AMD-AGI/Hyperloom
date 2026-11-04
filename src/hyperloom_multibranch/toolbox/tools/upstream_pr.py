# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``upstream_pr_search`` tool: find upstream pull requests that look relevant, with the framework agent's search.

The search runs in the process that hosts the toolbox, never in a measurement sandbox, and answers only for the repositories
the rule pack names (``UPSTREAM_PR_REPOS``). The framework agent's own GitHub search is best-effort: an unreachable host, a
refused token and an exhausted quota all come back as an empty list, which would read as "no pull request matches". The
search made here asks GitHub the same question with the same query and fails instead, saying why
(:class:`GitHubUnavailable`); an empty list means GitHub answered and found none. A search that also names a monitor service
is the framework agent's own (host configuration only: a specialist cannot name one).
"""

from __future__ import annotations

import asyncio
import http.client
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.agents.framework.models import Candidate, CandidateSearchRequest, PRMonitorConfig
from hyperloom.agents.framework.sources import SourceConfigError, enumerate_candidates
from hyperloom.agents.framework.sources.github import PERF_TERMS, auth_headers, build_query
from hyperloom.common.github_urls import repo_slug
from hyperloom.common.outside_facts import UPSTREAM_PR_REPOS
from hyperloom_multibranch.toolbox.tools.base import InvalidRequest, ToolFailure, ToolOutput, ToolRequest

UPSTREAM_PR_TOOL = "upstream_pr_search"
MAX_LIMIT = 20
DEFAULT_LIMIT = 5
_STATES = frozenset({"open", "merged", "closed", "all"})
#: The names the framework agent gives its two sources of candidates (its ``search_modes``).
_MONITOR, _GITHUB = "pr_monitor", "github"
_SEARCHED = frozenset(repo.lower() for repo in UPSTREAM_PR_REPOS)
#: GitHub's search endpoint for issues and pull requests.
_SEARCH_URL = "https://api.github.com/search/issues"
_TIMEOUT_SEC = 10.0
_HTTP_UNAUTHORIZED = 401
_HTTP_RATE_LIMITED = (403, 429)


@dataclass(frozen=True)
class UpstreamSearch:
    """The framework agent's search ``request``, and the pull-request ``states`` the caller asked for."""

    request: CandidateSearchRequest
    states: tuple[str, ...] = ("open",)

    def state_qualifier(self) -> str:
        """``is:open`` when the caller asked for open pull requests only; any other state searches them all."""
        return "is:open" if set(self.states) == {"open"} else ""


Enumerate = Callable[[UpstreamSearch], list[Candidate]]


class GitHubUnavailable(RuntimeError):
    """GitHub's search cannot be used from this host; the message says why."""


def _token_hint() -> str:
    """What to do about it: a request that carries no ``Authorization`` header was made without a token."""
    held = "Authorization" in auth_headers("application/json")
    return (
        ""
        if held
        else "; the host has no GITHUB_TOKEN or GH_TOKEN, and an anonymous search is allowed very few requests"
    )


def _refusal(code: int) -> str:
    if code == _HTTP_UNAUTHORIZED:
        return "GitHub rejected the token of this host (GITHUB_TOKEN or GH_TOKEN)"
    if code in _HTTP_RATE_LIMITED:
        return f"GitHub refused the search (HTTP {code}), most likely a rate limit{_token_hint()}"
    return f"GitHub answered the search with HTTP {code}"


def _github_answer(search: UpstreamSearch) -> dict[str, Any]:
    """GitHub's answer to ``search``, the query the framework agent's own search sends, held to the caller's states.

    Raises:
        GitHubUnavailable: GitHub could not be asked, refused, or did not answer with a search result.
    """
    request = search.request
    terms = tuple(word.lower() for word in request.keywords if word.strip()) or PERF_TERMS
    query = " ".join(filter(None, (build_query(repo_slug(request.repo_url), terms=terms), search.state_qualifier())))
    url = f"{_SEARCH_URL}?" + urllib.parse.urlencode({"q": query, "per_page": str(request.max_search_candidates)})
    request = urllib.request.Request(url, headers=auth_headers("application/vnd.github+json"))
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SEC) as response:  # nosec B310 - fixed GitHub HTTPS API URL.
            answer = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise GitHubUnavailable(_refusal(exc.code)) from None
    except (OSError, http.client.HTTPException, ValueError) as exc:
        raise GitHubUnavailable(
            f"GitHub cannot be reached from this host ({type(exc).__name__}); check its network and proxy"
        ) from None
    if not isinstance(answer, dict) or not isinstance(answer.get("items"), list):
        raise GitHubUnavailable("GitHub answered the search with something that is not a search result")
    return answer


def search_github(search: UpstreamSearch) -> list[Candidate]:
    """The pull requests GitHub's search finds for ``search``, best first.

    Raises:
        GitHubUnavailable: GitHub could not be asked or refused (never an empty list for that).
    """
    if _MONITOR in search.request.search_modes:
        return enumerate_candidates(search.request)
    found = []
    for item in _github_answer(search)["items"]:
        number = item.get("number") if isinstance(item, dict) else None
        if isinstance(number, int):
            found.append(
                Candidate(
                    ref=f"PR:{number}", title=str(item.get("title") or ""), html_url=str(item.get("html_url") or "")
                )
            )
    return found[: search.request.max_search_candidates]


def _strings(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return tuple(part for part in value.replace(",", " ").split() if part)
    return tuple(str(v).strip() for v in value if str(v).strip())


def build_search(params: Mapping[str, Any]) -> UpstreamSearch:
    """Turn tool parameters into the framework agent's search request; raises :class:`InvalidRequest`."""
    repo_url = str(params.get("repo_url") or "").strip()
    try:
        slug = repo_slug(repo_url)
    except ValueError as exc:
        raise InvalidRequest(f"repo_url {repo_url!r} is not a GitHub repository URL") from exc
    if slug.lower() not in _SEARCHED:
        raise InvalidRequest(f"{slug} is not an upstream repository this tool searches: {', '.join(UPSTREAM_PR_REPOS)}")
    try:
        limit = int(DEFAULT_LIMIT if params.get("limit") is None else params["limit"])
    except (TypeError, ValueError) as exc:
        raise InvalidRequest("limit must be an integer") from exc
    if not 1 <= limit <= MAX_LIMIT:
        raise InvalidRequest(f"limit must be between 1 and {MAX_LIMIT}")
    states = _strings(params.get("states")) or ("open",)
    if not set(states) <= _STATES:
        raise InvalidRequest(f"states must come from {sorted(_STATES)}")
    monitor_url = str(params.get("pr_monitor_url") or "").strip()
    request = CandidateSearchRequest(
        repo_url=repo_url,
        max_search_candidates=limit,
        pr_monitor=PRMonitorConfig(monitor_url) if monitor_url else None,
        keywords=_strings(params.get("keywords")),
        search_modes=(_MONITOR, _GITHUB) if monitor_url else (_GITHUB,),
    )
    return UpstreamSearch(request, states)


class UpstreamPrSearch:
    """Params: ``repo_url`` (required), ``keywords``, ``limit`` (default 5, at most 20), ``states``, ``pr_monitor_url``."""

    name = UPSTREAM_PR_TOOL

    def __init__(self, enumerate_fn: Enumerate = search_github) -> None:
        self._enumerate = enumerate_fn

    def validate(self, request: ToolRequest) -> None:
        build_search(request.params)

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        search = build_search(request.params)
        try:
            found = await asyncio.to_thread(self._enumerate, search)
        except SourceConfigError as exc:
            raise ToolFailure(f"upstream PR search is misconfigured: {exc}") from exc
        except GitHubUnavailable as exc:
            raise ToolFailure(f"upstream PR search cannot use GitHub: {exc}") from exc
        except Exception as exc:
            raise ToolFailure(f"upstream PR search failed: {type(exc).__name__}: {exc}") from exc
        prs = [{"ref": c.ref, "title": c.title, "url": c.html_url} for c in found]
        top = f"; top: {prs[0]['ref']} {prs[0]['title']}" if prs else ""
        repo_url = search.request.repo_url
        return ToolOutput(
            result={"repo_url": repo_url, "keywords": list(search.request.keywords), "candidates": prs},
            summary=f"{len(prs)} upstream PR candidate(s) in {repo_url}{top}",
        )
