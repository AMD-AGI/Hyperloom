#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Report upstream PRs that credit Hyperloom, were merged, and are missing from the results page.

``docs/reference/upstream-results.md`` is where the repo claims what Hyperloom has landed upstream,
and the README quotes its strongest rows. The measured effect in a row has to be read out of the
PR's own end-to-end table by a human, so the page cannot be generated -- but falling behind is
silent, and a merged optimization that never reaches the page is evidence thrown away. This names
the gap, leaving writing the row as the only manual step.

Attribution has used more than one wording (``Found by Hyperloom``, ``Powered by Hyperloom``,
``Found and validated with ROCm Hyperloom``), so the search is any upstream PR body that names
Hyperloom at all, and every merged hit not already on the page is reported for a human to accept or
reject. It also prints the attributed / merged / open / closed-unmerged counts the page's "As of"
line quotes. Deliberately not a CI gate: an upstream merge would turn an unrelated PR's checks red.

Usage:
    python scripts/report_upstream_results.py [REPO_ROOT]

Reads ``GITHUB_TOKEN`` when set, which raises the search rate limit. Exit code 1 when a merged PR is
missing from the page, so a scheduled run can page someone.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import NamedTuple

_THIS_REPO = Path(__file__).resolve().parents[1]
_RESULTS_PAGE = Path("docs/reference/upstream-results.md")
_UPSTREAM_REPOS = ("vllm-project/vllm", "sgl-project/sglang")
_PULL_URL_RE = re.compile(r"github\.com/([\w.-]+/[\w.-]+)/pull/(\d+)")
_ENCODING = "utf-8"
_TIMEOUT_S = 30
_PAGE_SIZE = 100
# GitHub's search API returns no result past the 1000th, whatever the page.
_SEARCH_CAP = 1000


class MergedPull(NamedTuple):
    """A merged upstream PR whose body names Hyperloom."""

    repo: str
    number: int
    merged_on: str
    title: str

    def __str__(self) -> str:
        return f"{self.repo}#{self.number}  merged {self.merged_on}  {self.title}"


class Report(NamedTuple):
    """The merged PRs the page omits, and the attribution counts its "As of" line quotes."""

    missing: list[MergedPull]
    attributed: int
    merged: int
    open: int

    @property
    def closed_unmerged(self) -> int:
        return self.attributed - self.merged - self.open

    def counts(self) -> str:
        return (
            f"{self.merged} of {self.attributed} attributed PRs merged, "
            f"{self.open} still open, {self.closed_unmerged} closed unmerged"
        )


def listed(page: Path) -> set[tuple[str, int]]:
    """Every ``(repo, number)`` the results page already links to."""
    if not page.is_file():
        raise RuntimeError(f"{page} does not exist; it is the page this reports against")
    text = page.read_text(encoding=_ENCODING)
    return {(match.group(1), int(match.group(2))) for match in _PULL_URL_RE.finditer(text)}


def attributed(repo: str, token: str) -> list[dict]:
    """Every PR in ``repo`` whose body names Hyperloom, read page by page."""
    query = urllib.parse.quote(f"repo:{repo} Hyperloom in:body type:pr")
    items: list[dict] = []
    page = 1
    while True:
        payload = _get_json(f"{_api()}/search/issues?q={query}&per_page={_PAGE_SIZE}&page={page}", token)
        batch, total = payload.get("items"), payload.get("total_count")
        if not isinstance(batch, list) or not isinstance(total, int):
            raise RuntimeError(f"unexpected reply searching {repo}")
        if total > _SEARCH_CAP:
            raise RuntimeError(f"{total} PRs in {repo} name Hyperloom; the search API returns at most {_SEARCH_CAP}")
        items.extend(batch)
        if not batch or len(items) >= total:
            return items
        page += 1


def merged_pulls(repo: str, items: list[dict]) -> list[MergedPull]:
    """The merged PRs among ``items``, newest merge first."""
    found = []
    for item in items:
        merged_at = (item.get("pull_request") or {}).get("merged_at")
        if merged_at:
            found.append(MergedPull(repo, int(item["number"]), merged_at[:10], item.get("title") or ""))
    return sorted(found, key=lambda pull: (pull.merged_on, pull.number), reverse=True)


def check(root: Path, token: str = "") -> Report:
    """The merged attributed PRs missing from the results page under ``root``, with the attribution counts."""
    already = listed(root / _RESULTS_PAGE)
    missing: list[MergedPull] = []
    total = merged = still_open = 0
    for repo in _UPSTREAM_REPOS:
        items = attributed(repo, token)
        pulls = merged_pulls(repo, items)
        total += len(items)
        merged += len(pulls)
        still_open += sum(1 for item in items if item.get("state") == "open")
        missing.extend(pull for pull in pulls if (pull.repo, pull.number) not in already)
    return Report(missing, total, merged, still_open)


def _api() -> str:
    return os.environ.get("GITHUB_API_URL", "https://api.github.com")


def _get_json(url: str, token: str) -> dict:
    if not url.startswith("https://"):
        raise RuntimeError(f"refusing to read {url}: the API URL must be https")
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:  # nosec B310 -- https checked above
            return json.loads(response.read().decode(_ENCODING))
    except OSError as exc:
        raise RuntimeError(f"could not read {url}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0]).resolve() if args else _THIS_REPO
    report = check(root, os.environ.get("GITHUB_TOKEN", ""))
    for pull in report.missing:
        print(pull)
    if report.missing:
        print(f"\n{len(report.missing)} merged PR(s) absent from {_RESULTS_PAGE.as_posix()}.")
        print("Read each PR's end-to-end table for the row; quote no micro-benchmark as an end-to-end gain.")
    else:
        print(f"{_RESULTS_PAGE.as_posix()} lists every merged PR that credits Hyperloom.")
    print(f'For the page\'s "As of" line: {report.counts()}.')
    return 1 if report.missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
