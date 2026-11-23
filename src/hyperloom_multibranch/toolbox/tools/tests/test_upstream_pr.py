# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from hyperloom.agents.framework.models import Candidate
from hyperloom.agents.framework.sources import SourceConfigError
from hyperloom_multibranch.toolbox.tools import InvalidRequest, JobStatus, ToolJobs, ToolRequest, UpstreamPrSearch
from hyperloom_multibranch.toolbox.tools.tests.sinks import RecordingSink
from hyperloom_multibranch.toolbox.tools.upstream_pr import (
    DIFF_CHARS,
    Found,
    UpstreamSearch,
    build_search,
    search_github,
)

REPO = "https://github.com/sgl-project/sglang"


def req(**params: object) -> ToolRequest:
    return ToolRequest("upstream_pr_search", "s1", {"repo_url": REPO, **params})


def test_the_search_is_github_only_until_a_pr_monitor_is_named() -> None:
    search = build_search({"repo_url": REPO, "keywords": "moe, gemm", "limit": 3, "states": ["merged"]})
    assert search.request.search_modes == ("github",)
    assert search.request.keywords == ("moe", "gemm")
    assert (search.request.max_search_candidates, search.states, search.request.pr_monitor) == (3, ("merged",), None)
    monitored = build_search({"repo_url": REPO, "pr_monitor_url": "http://monitor.example/"})
    assert monitored.request.search_modes == ("pr_monitor", "github")
    assert monitored.request.pr_monitor is not None


def test_only_a_search_for_open_pull_requests_is_held_to_them() -> None:
    assert build_search({"repo_url": REPO}).state_qualifier() == "is:open"
    assert build_search({"repo_url": REPO, "states": ["open", "merged"]}).state_qualifier() == ""


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"repo_url": ""}, "not a GitHub repository URL"),
        ({"repo_url": "ftp://x/y"}, "not a GitHub repository URL"),
        ({"repo_url": REPO, "limit": 0}, "between 1 and 20"),
        ({"repo_url": REPO, "limit": 99}, "between 1 and 20"),
        ({"repo_url": REPO, "limit": "many"}, "must be an integer"),
        ({"repo_url": REPO, "states": ["draft"]}, "states must come from"),
    ],
)
def test_a_bad_search_is_refused_with_its_reason(params: dict, match: str) -> None:
    with pytest.raises(InvalidRequest, match=match):
        build_search(params)


@pytest.mark.parametrize(
    ("keywords", "terms"),
    [
        (
            ["gated delta net", "HIP", "ROCm", "RDNA3", "mmq", "mmvq"],
            '("gated delta net" OR "hip" OR "rocm" OR "rdna3" OR "mmq" OR "mmvq")',
        ),
        ("few-row MMQ", '("few-row" OR "mmq")'),
        (
            ['mmq") repo:someone/private-repo ("x', "is:closed"],
            '("mmq ) repo:someone/private-repo ( x" OR "is:closed")',
        ),
        (["rdna3\\", "\\", " author:someone \\", "ggml-cuda\\mmq"], '("rdna3" OR "author:someone" OR "ggml-cuda mmq")'),
    ],
    ids=["a-phrase-among-words", "a-string-of-words", "a-qualifier-in-a-keyword", "a-backslash-in-a-keyword"],
)
def test_each_keyword_is_one_phrase_of_the_query(monkeypatch: pytest.MonkeyPatch, keywords: object, terms: str) -> None:
    queries: list[str] = []

    def urlopen(request: urllib.request.Request, timeout: float | None = None) -> _Response:
        queries.extend(urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query)["q"])
        return _Response(b'{"total_count": 0, "items": []}')

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    search_github(build_search({"repo_url": REPO, "keywords": keywords}))
    assert queries == [f"repo:sgl-project/sglang is:pr {terms} is:open"]


async def run(tmp_path: Path, enumerate_fn, request: ToolRequest) -> tuple[JobStatus, RecordingSink]:
    sink = RecordingSink()
    runner = ToolJobs([UpstreamPrSearch(enumerate_fn)], sink, tmp_path, push_delay_seconds=0)
    return await runner.wait(runner.submit(request)), sink


async def test_candidates_come_back_through_the_inbox(tmp_path: Path) -> None:
    seen: list[UpstreamSearch] = []

    def enumerate_fn(search: UpstreamSearch) -> Found:
        seen.append(search)
        return Found([Candidate("PR:42", "Faster MoE decode", f"{REPO}/pull/42")], 1)

    status, sink = await run(tmp_path, enumerate_fn, req(keywords=["moe"]))
    assert status is JobStatus.SUCCEEDED
    assert seen[0].request.keywords == ("moe",)
    assert sink.pointers[0].summary == f"succeeded: 1 upstream PR candidate(s) in {REPO}; top: PR:42 Faster MoE decode"
    record = json.loads(Path(sink.pointers[0].location).read_text())
    assert record["result"] == {
        "repo_url": REPO,
        "keywords": ["moe"],
        "candidates": [{"ref": "PR:42", "title": "Faster MoE decode", "url": f"{REPO}/pull/42"}],
        "matched_prs": 1,
        "not_shown": 0,
    }


async def test_no_match_is_a_success_with_an_empty_list(tmp_path: Path) -> None:
    status, sink = await run(tmp_path, lambda search: Found([], 0), req())
    assert status is JobStatus.SUCCEEDED
    assert json.loads(Path(sink.pointers[0].location).read_text())["result"]["candidates"] == []


async def test_a_misconfigured_source_fails_with_that_reason(tmp_path: Path) -> None:
    def broken(search: UpstreamSearch) -> Found:
        raise SourceConfigError("no pr_monitor config")

    status, sink = await run(tmp_path, broken, req())
    assert status is JobStatus.FAILED
    assert sink.pointers[0].summary == "failed: upstream PR search is misconfigured: no pr_monitor config"


async def test_an_unreachable_backend_fails_instead_of_crashing(tmp_path: Path) -> None:
    def down(search: UpstreamSearch) -> Found:
        raise TimeoutError("pr_monitor unreachable")

    status, sink = await run(tmp_path, down, req())
    assert status is JobStatus.FAILED
    assert "TimeoutError: pr_monitor unreachable" in sink.pointers[0].summary


async def test_a_bad_repo_is_refused_at_submit(tmp_path: Path) -> None:
    runner = ToolJobs([UpstreamPrSearch(lambda s: Found([]))], RecordingSink(), tmp_path)
    with pytest.raises(InvalidRequest, match="not a GitHub repository URL"):
        runner.submit(ToolRequest("upstream_pr_search", "s1", {"repo_url": "nope"}))


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"number": 20, "keywords": ["draft"]}, "number reads one pull request"),
        ({"number": "twenty"}, "number must be an integer"),
        ({"number": 0}, "number must be the number of a pull request"),
        ({"number": 20, "offset": "far"}, "offset must be an integer"),
        ({"number": 20, "offset": -1}, "offset must be 0 or the next_offset of an earlier page"),
        ({"keywords": ["draft"], "offset": 60000}, "offset and file read the diff of one pull request"),
        ({"keywords": ["draft"], "file": "a.cpp"}, "offset and file read the diff of one pull request"),
    ],
)
def test_a_bad_number_is_refused_with_its_reason(params: dict, match: str) -> None:
    with pytest.raises(InvalidRequest, match=match):
        build_search({"repo_url": REPO, **params})


# --- one pull request, read from GitHub -----------------------------------------------------------------------------


def diff_of(*paths: str) -> str:
    """A git diff that changes one line of each of ``paths``."""
    return "".join(f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-a\n+b\n" for path in paths)


def changed(path: str, lines: int) -> str:
    """The part of a git diff that adds ``lines`` lines to ``path``."""
    added = "".join(f"+{path} line {n}\n" for n in range(lines))
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1,{lines} @@\n-a\n{added}"


LONG_PARTS = {
    "src/alpha.cpp": changed("src/alpha.cpp", 3000),
    "src/beta.cpp": changed("src/beta.cpp", 2000),
    "docs/gamma.md": changed("docs/gamma.md", 1000),
}
LONG_DIFF = "".join(LONG_PARTS.values())


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def _github(monkeypatch: pytest.MonkeyPatch, diff: str) -> list[str]:
    """A GitHub that serves pull request 42 of sglang and ``diff`` as its diff; returns the URLs asked."""
    asked: list[str] = []

    def urlopen(request: urllib.request.Request, timeout: float | None = None) -> _Response:
        asked.append(request.full_url)
        assert request.full_url == "https://api.github.com/repos/sgl-project/sglang/pulls/42"
        if request.get_header("Accept") != "application/vnd.github.diff":
            listed = {"number": 42, "title": "Faster MoE decode", "html_url": f"{REPO}/pull/42"}
            return _Response(json.dumps(listed).encode())
        return _Response(diff.encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return asked


async def read_pages(tmp_path: Path, offset: int | None = 0, **params: object) -> list[tuple[JobStatus, dict]]:
    """The status and the result record of each page of pull request 42 read with ``params``, from ``offset`` to its
    last page or to the one that fails."""
    sink = RecordingSink()
    runner = ToolJobs([UpstreamPrSearch()], sink, tmp_path, push_delay_seconds=0)
    pages: list[tuple[JobStatus, dict]] = []
    while offset is not None:
        status = await runner.wait(runner.submit(req(number=42, offset=offset, **params)))
        pages.append((status, json.loads(Path(sink.pointers[-1].location).read_text())))
        offset = pages[-1][1]["result"]["next_offset"] if status is JobStatus.SUCCEEDED else None
    return pages


async def test_a_number_returns_that_pull_requests_diff_and_changed_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = _github(monkeypatch, diff_of("python/sglang/moe.py", "README.md"))
    ((status, record),) = await read_pages(tmp_path)
    assert status is JobStatus.SUCCEEDED
    assert record["result"] == {
        "repo_url": REPO,
        "ref": "PR:42",
        "title": "Faster MoE decode",
        "url": f"{REPO}/pull/42",
        "files": ["python/sglang/moe.py", "README.md"],
        "diff": diff_of("python/sglang/moe.py", "README.md"),
        "offset": 0,
        "next_offset": None,
        "file": None,
        "diff_chars": len(diff_of("python/sglang/moe.py", "README.md")),
        "file_chars": {
            "python/sglang/moe.py": len(diff_of("python/sglang/moe.py")),
            "README.md": len(diff_of("README.md")),
        },
    }
    assert record["summary"] == "PR:42 Faster MoE decode: its diff"
    assert len(asked) == 2


async def test_a_diff_longer_than_a_page_is_read_whole_over_its_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert len(LONG_DIFF) > 2 * DIFF_CHARS
    _github(monkeypatch, LONG_DIFF)
    records = [record for _, record in await read_pages(tmp_path)]
    pages = [record["result"] for record in records]
    assert len(pages) == 3 and "".join(page["diff"] for page in pages) == LONG_DIFF
    assert all(len(page["diff"]) <= DIFF_CHARS and page["diff"].endswith("\n") for page in pages)
    assert [page["offset"] for page in pages[1:]] == [page["next_offset"] for page in pages[:-1]]
    for page in pages:
        assert page["diff_chars"] == len(LONG_DIFF) and page["file"] is None
        assert page["file_chars"] == {path: len(part) for path, part in LONG_PARTS.items()}
    first, last = len(pages[0]["diff"]), len(LONG_DIFF)
    read = "PR:42 Faster MoE decode: characters"
    assert records[0]["summary"] == f"{read} 0-{first} of the {last} of its diff, next_offset {first}"
    assert records[2]["summary"] == f"{read} {pages[2]['offset']}-{last} of the {last} of its diff"


async def test_a_line_longer_than_a_page_is_cut_inside_and_still_read_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diff = diff_of("a.cpp") + "+" + "x" * (2 * DIFF_CHARS) + "\n"
    _github(monkeypatch, diff)
    pages = [record["result"]["diff"] for _, record in await read_pages(tmp_path)]
    assert "".join(pages) == diff
    assert all(len(page) <= DIFF_CHARS for page in pages)


async def test_a_file_returns_only_the_part_of_the_diff_that_changes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _github(monkeypatch, LONG_DIFF)
    ((_, record),) = await read_pages(tmp_path, file="src/beta.cpp")
    result = record["result"]
    assert result["diff"] == LONG_PARTS["src/beta.cpp"] and result["file"] == "src/beta.cpp"
    assert (result["offset"], result["next_offset"], result["diff_chars"]) == (0, None, len(LONG_DIFF))
    size = len(LONG_PARTS["src/beta.cpp"])
    assert record["summary"] == f"PR:42 Faster MoE decode: characters 0-{size} of the {size} of one file of its diff"


@pytest.mark.parametrize(
    ("params", "error"),
    [
        (
            {"offset": len(LONG_DIFF)},
            f"offset {len(LONG_DIFF)} is past the end of the {len(LONG_DIFF)} characters of the diff of PR:42",
        ),
        (
            {"file": "src/delta.cpp"},
            "the diff of PR:42 has no part that changes 'src/delta.cpp': file_chars of a read without file has the "
            "paths it holds",
        ),
    ],
    ids=["past-the-diff", "no-such-path"],
)
async def test_a_page_the_diff_does_not_hold_fails_saying_what_it_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, params: dict, error: str
) -> None:
    _github(monkeypatch, LONG_DIFF)
    ((status, record),) = await read_pages(tmp_path, **params)
    assert status is JobStatus.FAILED and record["error"] == error
