# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path

import pytest

from hyperloom.agents.framework.models import Candidate, CandidateSearchRequest
from hyperloom.agents.framework.sources import SourceConfigError
from hyperloom_multibranch.toolbox.tools import InvalidRequest, JobStatus, ToolJobs, ToolRequest, UpstreamPrSearch
from hyperloom_multibranch.toolbox.tools.tests.sinks import RecordingSink
from hyperloom_multibranch.toolbox.tools.upstream_pr import build_search

REPO = "https://github.com/sgl-project/sglang"


def req(**params: object) -> ToolRequest:
    return ToolRequest("upstream_pr_search", "s1", {"repo_url": REPO, **params})


def test_the_search_is_github_only_until_a_pr_monitor_is_named() -> None:
    search = build_search({"repo_url": REPO, "keywords": "moe, gemm", "limit": 3, "states": ["merged"]})
    assert search.search_modes == ("github",)
    assert search.keywords == ("moe", "gemm")
    assert (search.max_search_candidates, search.pr_states, search.pr_monitor) == (3, ("merged",), None)
    monitored = build_search({"repo_url": REPO, "pr_monitor_url": "http://monitor.example/"})
    assert monitored.search_modes == ("pr_monitor", "github")
    assert monitored.pr_monitor is not None


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


async def run(tmp_path: Path, enumerate_fn, request: ToolRequest) -> tuple[JobStatus, RecordingSink]:
    sink = RecordingSink()
    runner = ToolJobs([UpstreamPrSearch(enumerate_fn)], sink, tmp_path, push_delay_seconds=0)
    return await runner.wait(runner.submit(request)), sink


async def test_candidates_come_back_through_the_inbox(tmp_path: Path) -> None:
    seen: list[CandidateSearchRequest] = []

    def enumerate_fn(search: CandidateSearchRequest) -> list[Candidate]:
        seen.append(search)
        return [Candidate("PR:42", REPO, "github", "Faster MoE decode", f"{REPO}/pull/42", 0.8)]

    status, sink = await run(tmp_path, enumerate_fn, req(keywords=["moe"]))
    assert status is JobStatus.SUCCEEDED
    assert seen[0].keywords == ("moe",)
    assert sink.pointers[0].summary == f"succeeded: 1 upstream PR candidate(s) in {REPO}; top: PR:42 Faster MoE decode"
    record = json.loads(Path(sink.pointers[0].location).read_text())
    assert record["result"]["candidates"] == [
        {"ref": "PR:42", "title": "Faster MoE decode", "url": f"{REPO}/pull/42", "score": 0.8, "source": "github"}
    ]


async def test_no_match_is_a_success_with_an_empty_list(tmp_path: Path) -> None:
    status, sink = await run(tmp_path, lambda search: [], req())
    assert status is JobStatus.SUCCEEDED
    assert json.loads(Path(sink.pointers[0].location).read_text())["result"]["candidates"] == []


async def test_a_misconfigured_source_fails_with_that_reason(tmp_path: Path) -> None:
    def broken(search: CandidateSearchRequest) -> list[Candidate]:
        raise SourceConfigError("no pr_monitor config")

    status, sink = await run(tmp_path, broken, req())
    assert status is JobStatus.FAILED
    assert sink.pointers[0].summary == "failed: upstream PR search is misconfigured: no pr_monitor config"


async def test_an_unreachable_backend_fails_instead_of_crashing(tmp_path: Path) -> None:
    def down(search: CandidateSearchRequest) -> list[Candidate]:
        raise TimeoutError("pr_monitor unreachable")

    status, sink = await run(tmp_path, down, req())
    assert status is JobStatus.FAILED
    assert "TimeoutError: pr_monitor unreachable" in sink.pointers[0].summary


async def test_a_bad_repo_is_refused_at_submit(tmp_path: Path) -> None:
    runner = ToolJobs([UpstreamPrSearch(lambda s: [])], RecordingSink(), tmp_path)
    with pytest.raises(InvalidRequest, match="not a GitHub repository URL"):
        runner.submit(ToolRequest("upstream_pr_search", "s1", {"repo_url": "nope"}))
