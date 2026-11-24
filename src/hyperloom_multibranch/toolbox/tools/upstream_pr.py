# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``upstream_pr_search`` tool: find upstream pull requests that look relevant, and read the diff of one.

The tool runs in the process that hosts the toolbox, never in a measurement sandbox, and answers only for the repositories
the rule pack names (``UPSTREAM_PR_REPOS``). The framework agent's own GitHub search is best-effort: an unreachable
host, a refused token and an exhausted quota all come back as an empty list, which would read as "no pull request
matches". The search made here asks GitHub the same question, each keyword as a quoted phrase, and fails instead,
saying why (:class:`GitHubUnavailable`); an empty list means GitHub answered and found none. A search that also names
a monitor service is the framework agent's own (host configuration only: a specialist cannot name one).

A diff is read a page at a time (:data:`DIFF_CHARS`): a result says how long the diff is, how long the part of each
path is and where the next page starts, and a request names the offset to go on from, or one path to read alone.
The pages of a diff are one download of it: a pull request may move between two requests, so the read of offset 0
downloads the diff and the tool holds it for the later pages and paths the same branch reads.

Pull-request text is data: titles, paths and diffs are in the result, and the summary carries a ref and a title only.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom_common.unified_diff import touched_paths
from hyperloom.agents.framework.models import Candidate, CandidateSearchRequest, PRMonitorConfig
from hyperloom.agents.framework.sources import SourceConfigError, enumerate_candidates
from hyperloom.agents.framework.sources.github import (
    PERF_TERMS,
    GitHubUnavailable,
    build_query,
    pr_diff,
    pull_request,
    search_issues,
)
from hyperloom.common.github_urls import repo_slug
from hyperloom.common.outside_facts import UPSTREAM_PR_REPOS
from hyperloom_multibranch.toolbox.tools.base import InvalidRequest, ToolFailure, ToolOutput, ToolRequest

UPSTREAM_PR_TOOL = "upstream_pr_search"
MAX_LIMIT = 20
DEFAULT_LIMIT = 10
#: The characters of a diff one result carries: a page of it.
DIFF_CHARS = 60_000
_FILE_SECTION = re.compile(r"(?m)^(?=diff --git )")
_OPEN = "open"
_STATES = frozenset({_OPEN, "merged", "closed", "all"})
#: What is searched when the caller names no state: open pull requests only.
_OPEN_ONLY = (_OPEN,)
#: The names the framework agent gives its two sources of candidates (its ``search_modes``).
_MONITOR, _GITHUB = "pr_monitor", "github"
_SEARCHED = frozenset(repo.lower() for repo in UPSTREAM_PR_REPOS)
_TIMEOUT_SEC = 10.0


@dataclass(frozen=True)
class UpstreamSearch:
    """The framework agent's search ``request``, the pull-request ``states`` the caller asked for, and ``number``: the
    one pull request the caller asked to read instead of searching (``None`` for a search). Its diff is read from the
    character ``offset`` of it, or of the part of it that changes ``file`` when the caller names one."""

    request: CandidateSearchRequest
    states: tuple[str, ...] = _OPEN_ONLY
    number: int | None = None
    offset: int = 0
    file: str = ""

    def state_qualifier(self) -> str:
        """``is:open`` when the caller asked for open pull requests only; any other state searches them all."""
        return "is:open" if set(self.states) == {_OPEN} else ""


@dataclass(frozen=True)
class Found:
    """The candidates of a search, best first, and ``matched``: how many pull requests matched in all, ``None`` from a
    source that does not count them (a monitor service)."""

    candidates: list[Candidate]
    matched: int | None = None


Enumerate = Callable[[UpstreamSearch], Found]


@dataclass(frozen=True)
class PullRequestDiff:
    """One pull request and its diff."""

    ref: str
    title: str
    url: str
    files: tuple[str, ...]
    diff: str


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
    if _MONITOR in request.search_modes:
        return Found(enumerate_candidates(request))
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


def _read(search: UpstreamSearch) -> PullRequestDiff:
    """The pull request ``search.number`` and its diff, as GitHub serves them.

    Raises:
        GitHubUnavailable: GitHub could not be asked or refused.
    """
    assert search.number is not None
    repo_url = search.request.repo_url
    listed = pull_request(repo_url, search.number)
    diff = pr_diff(repo_url, search.number)
    return PullRequestDiff(f"PR:{search.number}", listed.title, listed.html_url, tuple(touched_paths(diff)), diff)


def _strings(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return tuple(part for part in value.replace(",", " ").split() if part)
    return tuple(str(v).strip() for v in value if str(v).strip())


def _number(params: Mapping[str, Any]) -> int | None:
    """The pull request ``params`` ask to read, ``None`` when they ask for a search; raises :class:`InvalidRequest`."""
    if params.get("number") is None:
        return None
    try:
        number = int(params["number"])
    except (TypeError, ValueError) as exc:
        raise InvalidRequest("number must be an integer") from exc
    if number < 1:
        raise InvalidRequest("number must be the number of a pull request")
    if _strings(params.get("keywords")):
        raise InvalidRequest("number reads one pull request and keywords search for some: give one of the two")
    return number


def _offset(params: Mapping[str, Any]) -> int:
    """The character ``params`` ask a diff to be read from; raises :class:`InvalidRequest`."""
    try:
        offset = int(params.get("offset") or 0)
    except (TypeError, ValueError) as exc:
        raise InvalidRequest("offset must be an integer") from exc
    if offset < 0:
        raise InvalidRequest("offset must be 0 or the next_offset of an earlier page")
    return offset


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
    monitor_url = str(params.get("pr_monitor_url") or "").strip()
    request = CandidateSearchRequest(
        repo_url=repo_url,
        max_search_candidates=limit,
        pr_monitor=PRMonitorConfig(monitor_url) if monitor_url else None,
        keywords=keywords,
        search_modes=(_MONITOR, _GITHUB) if monitor_url else (_GITHUB,),
    )
    number, offset, file = _number(params), _offset(params), str(params.get("file") or "").strip()
    if number is None and (offset or file):
        raise InvalidRequest("offset and file read the diff of one pull request: name it in number")
    return UpstreamSearch(request, states, number, offset, file)


def _file_parts(diff: str) -> dict[str, str]:
    """The part of ``diff`` that changes each path, by path in the diff's order: its sections that name the path, each
    from a ``diff --git`` line to the next."""
    parts: dict[str, str] = {}
    for section in _FILE_SECTION.split(diff):
        for path in touched_paths(section):
            parts[path] = parts.get(path, "") + section
    return parts


def _paged_text(pr: PullRequestDiff, parts: Mapping[str, str], search: UpstreamSearch) -> str:
    """What ``search`` reads a page of: the diff of ``pr``, or the one of its ``parts`` that changes ``search.file``.

    Raises:
        ToolFailure: The diff has no part for that file, or ends before ``search.offset``.
    """
    if search.file and search.file not in parts:
        raise ToolFailure(
            f"the diff of {pr.ref} has no part that changes {search.file!r}: file_chars of a read without file has "
            "the paths it holds"
        )
    text = parts[search.file] if search.file else pr.diff
    if search.offset and search.offset >= len(text):
        read = (
            f"the part of the diff of {pr.ref} that changes {search.file!r}" if search.file else f"the diff of {pr.ref}"
        )
        raise ToolFailure(f"offset {search.offset} is past the end of the {len(text)} characters of {read}")
    return text


def _page_end(text: str, offset: int) -> int:
    """Where the page of ``text`` that starts at ``offset`` ends: after its last whole line within :data:`DIFF_CHARS`
    characters, or at that many characters when one line is longer."""
    end = offset + DIFF_CHARS
    if end >= len(text):
        return len(text)
    return text.rfind("\n", offset, end) + 1 or end


def _page(pr: PullRequestDiff, search: UpstreamSearch) -> ToolOutput:
    """The page ``search`` reads of the diff of ``pr``, with what is left to read of it."""
    parts = _file_parts(pr.diff)
    text = _paged_text(pr, parts, search)
    end = _page_end(text, search.offset)
    next_offset = end if end < len(text) else None
    read = "its diff"
    if search.offset or next_offset or search.file:
        read = (
            f"characters {search.offset}-{end} of the {len(text)} of {'one file of ' if search.file else ''}its diff"
            + (f", next_offset {next_offset}" if next_offset else "")
        )
    return ToolOutput(
        result={
            "repo_url": search.request.repo_url,
            "ref": pr.ref,
            "title": pr.title,
            "url": pr.url,
            "files": list(pr.files),
            "diff": text[search.offset : end],
            "offset": search.offset,
            "next_offset": next_offset,
            "file": search.file or None,
            "diff_chars": len(pr.diff),
            "file_chars": {path: len(part) for path, part in parts.items()},
        },
        summary=f"{pr.ref} {pr.title}: {read}",
    )


def _candidates(found: Found, search: UpstreamSearch) -> ToolOutput:
    """What ``search`` ``found``, with how many pull requests the limit left out."""
    prs = [{"ref": c.ref, "title": c.title, "url": c.html_url} for c in found.candidates]
    not_shown = None if found.matched is None else found.matched - len(prs)
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
    """Params: ``repo_url`` (required), and either ``number`` (the pull request whose diff is returned, a page of it
    from ``offset``, of the whole diff or of the part that changes ``file``) or a search: ``keywords``, ``limit``
    (default 10, at most 20), ``states``, ``pr_monitor_url``. ``enumerate_fn`` runs the search."""

    name = UPSTREAM_PR_TOOL

    def __init__(self, enumerate_fn: Enumerate = search_github) -> None:
        self._enumerate = enumerate_fn
        #: The diff each branch last downloaded of a pull request, by branch, repository and number.
        self._held: dict[tuple[str, str, int | None], PullRequestDiff] = {}

    def validate(self, request: ToolRequest) -> None:
        build_search(request.params)

    def _diff(self, branch: str, search: UpstreamSearch) -> PullRequestDiff:
        """The diff ``search`` reads a page of: downloaded by a read of offset 0 (of one path only when ``branch``
        holds no diff of the pull request), else the one that download left.

        Raises:
            GitHubUnavailable: GitHub could not be asked or refused.
            ToolFailure: The offset goes on from a diff the tool does not hold.
        """
        key = (branch, repo_slug(search.request.repo_url).lower(), search.number)
        if not search.offset and (not search.file or key not in self._held):
            self._held[key] = _read(search)
        elif key not in self._held:
            raise ToolFailure(
                f"offset {search.offset} goes on from a diff of PR:{search.number} this toolbox does not hold: read "
                "it again from offset 0"
            )
        return self._held[key]

    def _answer(self, branch: str, search: UpstreamSearch) -> ToolOutput:
        """A page of the diff of the pull request ``search`` names, or the candidates its search finds."""
        if search.number is not None:
            return _page(self._diff(branch, search), search)
        return _candidates(self._enumerate(search), search)

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        search = build_search(request.params)
        try:
            return await asyncio.to_thread(self._answer, request.branch, search)
        except SourceConfigError as exc:
            raise ToolFailure(f"upstream PR search is misconfigured: {exc}") from exc
        except GitHubUnavailable as exc:
            raise ToolFailure(f"upstream PR search cannot use GitHub: {exc}") from exc
