# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``upstream_pr_search`` tool: find upstream pull requests that look relevant.

The tool runs in the process that hosts the toolbox, never in a measurement sandbox, and answers only for the repositories
the rule pack names (``UPSTREAM_PR_REPOS``). The framework agent's own GitHub search is best-effort: an unreachable
host, a refused token and an exhausted quota all come back as an empty list, which would read as "no pull request
matches". The search made here asks GitHub the same question, each keyword as a quoted phrase, and fails instead,
saying why (:class:`GitHubUnavailable`); an empty list means GitHub answered and found none.

Pull-request text is data: the summary carries the ref and the title of the first candidate only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom.agents.framework.models import Candidate, CandidateSearchRequest
from hyperloom.agents.framework.sources.github import PERF_TERMS, GitHubUnavailable, build_query, search_issues
from hyperloom.common.github_urls import repo_slug
from hyperloom.common.outside_facts import UPSTREAM_PR_REPOS
from hyperloom_multibranch.toolbox.tools.base import InvalidRequest, ToolFailure, ToolOutput, ToolRequest

UPSTREAM_PR_TOOL = "upstream_pr_search"
MAX_LIMIT = 20
DEFAULT_LIMIT = 10
_OPEN = "open"
_STATES = frozenset({_OPEN, "merged", "closed", "all"})
#: What is searched when the caller names no state: open pull requests only.
_OPEN_ONLY = (_OPEN,)
#: The framework agent's name for its GitHub source of candidates (one of its ``search_modes``).
_GITHUB = "github"
_SEARCHED = frozenset(repo.lower() for repo in UPSTREAM_PR_REPOS)
_TIMEOUT_SEC = 10.0


@dataclass(frozen=True)
class UpstreamSearch:
    """The framework agent's search ``request``, and the pull-request ``states`` the caller asked for."""

    request: CandidateSearchRequest
    states: tuple[str, ...] = _OPEN_ONLY

    def state_qualifier(self) -> str:
        """``is:open`` when the caller asked for open pull requests only; any other state searches them all."""
        return "is:open" if set(self.states) == {_OPEN} else ""


@dataclass(frozen=True)
class Found:
    """The candidates of a search, best first, and ``matched``: how many pull requests matched in all."""

    candidates: list[Candidate]
    matched: int


Enumerate = Callable[[UpstreamSearch], Found]


#: What a keyword may not carry into its quotes: a quote ends the phrase, and GitHub reads a backslash before the
#: closing quote as a quote of the phrase, which then runs on into the next keyword. Each becomes a space, not
#: nothing: the search finds a backslash between two words as it finds a space there.
_BREAKS_A_PHRASE = str.maketrans({'"': " ", "\\": " "})


def _words(keyword: str) -> str:
    """What GitHub is asked to find for ``keyword``: its words, in lower case, one space apart."""
    return " ".join(keyword.lower().translate(_BREAKS_A_PHRASE).split())


def _phrases(keywords: tuple[str, ...]) -> tuple[str, ...]:
    """Each of ``keywords`` as a quoted phrase of a GitHub search. Unquoted, GitHub requires the leading words of a
    keyword of several words of every result, whatever the other keywords are, and reads a qualifier in a keyword."""
    return tuple(f'"{_words(keyword)}"' for keyword in keywords)


def search_github(search: UpstreamSearch) -> Found:
    """The pull requests GitHub's search finds for ``search``: those that hold any one of its first
    :data:`~hyperloom.agents.framework.sources.github.MAX_OR_TERMS` keywords, held to the caller's states.

    Raises:
        GitHubUnavailable: GitHub could not be asked or refused (never an empty list for that).
    """
    request = search.request
    terms = _phrases(request.keywords) or PERF_TERMS
    query = " ".join(filter(None, (build_query(repo_slug(request.repo_url), terms=terms), search.state_qualifier())))
    answer = search_issues(query, per_page=request.max_search_candidates, timeout_sec=_TIMEOUT_SEC)
    candidates = []
    for item in answer.items:
        number = item.get("number") if isinstance(item, dict) else None
        if isinstance(number, int):
            candidates.append(
                Candidate(
                    ref=f"PR:{number}", title=str(item.get("title") or ""), html_url=str(item.get("html_url") or "")
                )
            )
    return Found(candidates[: request.max_search_candidates], answer.total_count)


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
    states = _strings(params.get("states")) or _OPEN_ONLY
    if not set(states) <= _STATES:
        raise InvalidRequest(f"states must come from {sorted(_STATES)}")
    keywords = _strings(params.get("keywords"))
    wordless = [keyword for keyword in keywords if not _words(keyword)]
    if wordless:
        raise InvalidRequest(f"keyword {wordless[0]!r} holds no word to search for")
    request = CandidateSearchRequest(
        repo_url=repo_url,
        max_search_candidates=limit,
        keywords=keywords,
        search_modes=(_GITHUB,),
    )
    return UpstreamSearch(request, states)


def _candidates(found: Found, search: UpstreamSearch) -> ToolOutput:
    """What ``search`` ``found``, with how many pull requests the limit left out."""
    prs = [{"ref": c.ref, "title": c.title, "url": c.html_url} for c in found.candidates]
    not_shown = found.matched - len(prs)
    shown = f"{len(prs)} of {found.matched}" if not_shown else str(len(prs))
    top = f"; top: {prs[0]['ref']} {prs[0]['title']}" if prs else ""
    repo_url = search.request.repo_url
    return ToolOutput(
        result={
            "repo_url": repo_url,
            "keywords": list(search.request.keywords),
            "candidates": prs,
            "matched_prs": found.matched,
            "not_shown": not_shown,
        },
        summary=f"{shown} upstream PR candidate(s) in {repo_url}{top}",
    )


class UpstreamPrSearch:
    """Params: ``repo_url`` (required), ``keywords``, ``limit`` (default 10, at most 20) and ``states``."""

    name = UPSTREAM_PR_TOOL

    def __init__(self, enumerate_fn: Enumerate = search_github) -> None:
        self._enumerate = enumerate_fn

    def validate(self, request: ToolRequest) -> None:
        build_search(request.params)

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        search = build_search(request.params)
        try:
            return _candidates(await asyncio.to_thread(self._enumerate, search), search)
        except GitHubUnavailable as exc:
            raise ToolFailure(f"upstream PR search cannot use GitHub: {exc}") from exc
