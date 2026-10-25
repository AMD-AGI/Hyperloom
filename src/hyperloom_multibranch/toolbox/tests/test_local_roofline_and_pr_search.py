# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``roofline`` and ``upstream_pr_search`` on the toolbox the real local factory builds, over real HTTP (CPU only).

The roofline reads the trace a profile test left, with the orchestrator's own ``bypass`` analysis run for real on a small
trace: no card and no model. The PR search runs against a GitHub that answers from a fixture; a failure of that GitHub
(no network, no quota, a refused token) is the job's failure, not an empty list.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import secrets
import threading
import urllib.error
import urllib.request
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from hyperloom.hosted.mcp_server import McpServer, McpServerConfig
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.toolbox import RequestRejected
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox.local_hosted import LocalOptions, LocalSession, _runner_template
from hyperloom_multibranch.toolbox.mcp_tools import (
    McpSpecialistTokens,
    SpecialistToolOptions,
    register_specialist_tools,
)
from hyperloom_multibranch.toolbox.tests.helpers import PARENT, FakeRepo
from hyperloom_multibranch.toolbox.tests.test_analysis_route import Stack, _Analysis
from hyperloom_multibranch.toolbox.tests.test_local_analysis import LOCATION, Local, _toolbox
from hyperloom_multibranch.toolbox.tools import JobStatus

SGLANG = "https://github.com/sgl-project/sglang"
GITHUB_API = "https://api.github.com/"
_REAL_URLOPEN = urllib.request.urlopen
HOST_TOKEN = "ghp_host_token_not_for_the_result"
TRACE = [
    {"cat": "cpu_op", "name": "aten::paged_attn", "args": {"External id": 100}},
    {"cat": "cpu_op", "name": "aten::mm", "args": {"External id": 200}},
    {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 5, "External id": 100}},
    {"cat": "cuda_runtime", "name": "hipLaunchKernel", "args": {"correlation": 7, "External id": 200}},
    {"cat": "kernel", "ph": "X", "name": "paged_attention_v1", "ts": 1000, "dur": 300, "args": {"correlation": 5}},
    {"cat": "kernel", "ph": "X", "name": "Cijk_Alik_Bljk_HHS", "ts": 1300, "dur": 200, "args": {"correlation": 7}},
]
CPU_ONLY = [event for event in TRACE if event["cat"] != "kernel"]


class _Response:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = json.dumps(body).encode()

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


class Github:
    """The search endpoint of GitHub; ``failing`` is the error it raises, ``answer`` what it returns instead of the items.

    Every request is kept, with the process and thread that made it and the ``Authorization`` it carried. A request to
    anything else (the test's own calls to the MCP server) goes through.
    """

    def __init__(
        self, items: list[dict[str, Any]] | None = None, *, failing: Exception | None = None, answer: Any = None
    ) -> None:
        self.items, self.failing, self.answer = items or [], failing, answer
        self.requests: list[tuple[str, str | None, int, str]] = []

    def __call__(self, request: urllib.request.Request, timeout: float | None = None) -> Any:
        url = request.full_url
        if not url.startswith(GITHUB_API):
            return _REAL_URLOPEN(request, timeout=timeout)
        self.requests.append((url, request.get_header("Authorization"), os.getpid(), threading.current_thread().name))
        if self.failing is not None:
            raise self.failing
        return _Response(
            self.answer if self.answer is not None else {"total_count": len(self.items), "items": self.items}
        )


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://api.github.com/x", code, "refused", {}, None)  # type: ignore[arg-type]


def _pr(number: int, title: str) -> dict[str, Any]:
    return {"number": number, "title": title, "html_url": f"{SGLANG}/pull/{number}"}


@pytest.fixture(autouse=True)
def _host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A TraceLens checkout that is only a directory with a ``.git`` (the bypass analysis does not read it), and a host that
    holds a GitHub token."""
    (tmp_path / "TraceLens" / ".git").mkdir(parents=True)
    monkeypatch.setenv("TRACELENS_ROOT", str(tmp_path / "TraceLens"))
    monkeypatch.delenv("HYPERLOOM_TRACE_ANALYSIS_ROUTE", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", HOST_TOKEN)
    monkeypatch.delenv("GH_TOKEN", raising=False)


async def _serve(tmp_path: Path, **options: Any) -> AsyncIterator[Local]:
    notebook = LocalNotebook(tmp_path / "notebook", Writer(Role.TOOLBOX, "toolbox"))
    repo = FakeRepo()
    repo.heads["b1"] = PARENT
    toolbox = _toolbox(tmp_path, notebook, repo, **options)
    await toolbox.start()
    server = McpServer(McpServerConfig(host_token=secrets.token_hex(16)), campaign="w168")
    register_specialist_tools(
        server,
        toolbox=toolbox,
        notebooks=lambda b: LocalNotebook(tmp_path / "notebook", Writer(Role.SPECIALIST_RUNTIME, b, b)),
        options=SpecialistToolOptions(analysis=toolbox, toolbox_loop=asyncio.get_running_loop()),
    )
    server.start()
    try:
        stack = Stack(None, toolbox._tools, None, {}, server, McpSpecialistTokens(server))  # type: ignore[arg-type]
        yield Local(stack, toolbox, notebook, tmp_path / "session", _Analysis("tracelens"))
    finally:
        server.stop()
        await toolbox.close()


@pytest.fixture
async def served(tmp_path: Path) -> AsyncIterator[Local]:
    """The toolbox the factory builds with its default tools."""
    async for local in _serve(tmp_path):
        yield local


def _write_trace(local: Local, events: list[dict[str, Any]]) -> None:
    directory = local.session / LOCATION
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "rank0.trace.json.gz").write_bytes(gzip.compress(json.dumps({"traceEvents": events}).encode()))


async def _finish(local: Local, reply: dict[str, Any]) -> tuple[JobStatus, dict[str, Any]]:
    """The status of the job a started analysis names, and the record it left."""
    assert reply["result"]["isError"] is False, reply
    job = json.loads(reply["result"]["content"][0]["text"])["job_id"]
    status = await local.toolbox._tools.wait(job)  # type: ignore[union-attr]
    return status, json.loads((local.session / "tool-results" / job / "result.json").read_text(encoding="utf-8"))


async def _roofline(
    local: Local, events: list[dict[str, Any]] | None, **params: Any
) -> tuple[JobStatus, dict[str, Any]]:
    profile = await local.profile_test()
    if events is not None:
        _write_trace(local, events)
    return await _finish(local, await local.stack.start(tool="roofline", profile_request_id=profile, params=params))


async def _search(local: Local, **params: Any) -> tuple[JobStatus, dict[str, Any]]:
    return await _finish(
        local, await local.stack.start(tool="upstream_pr_search", params={"repo_url": SGLANG, **params})
    )


def _install(monkeypatch: pytest.MonkeyPatch, github: Github) -> Github:
    monkeypatch.setattr(urllib.request, "urlopen", github)
    return github


# --- what is advertised is what is served -------------------------------------------------------------------------


async def test_the_default_local_toolbox_serves_and_advertises_each_analysis_tool(served: Local) -> None:
    entry = served.stack.advertised()
    assert served.toolbox.analysis_tools == ("tracelens", "roofline", "kernel_tune", "upstream_pr_search")
    assert entry["inputSchema"]["properties"]["tool"]["enum"] == list(served.toolbox.analysis_tools)
    assert "upstream_pr_search" in entry["description"] and "sgl-project/sglang" in entry["description"]
    assert "required for tracelens and roofline" in json.dumps(entry["inputSchema"])


@pytest.mark.parametrize("name", ["roofline", "upstream_pr_search"])
async def test_a_tool_is_listed_only_when_the_toolbox_serves_it(tmp_path: Path, name: str) -> None:
    async for local in _serve(tmp_path, tools=[_Analysis("tracelens")]):
        entry = local.stack.advertised()
        assert entry["inputSchema"]["properties"]["tool"]["enum"] == ["tracelens"]
        assert name not in entry["description"] and name not in json.dumps(entry["inputSchema"])
        assert "must be one of ['tracelens']" in await local.stack.refused(tool=name)
        with pytest.raises(RequestRejected) as stale:
            await local.toolbox.submit_analysis("b1", name, {}, "p1")
        assert stale.value.code == "tool_unavailable_on_backend"


@pytest.mark.parametrize("name", ["roofline", "upstream_pr_search"])
async def test_a_served_stand_in_is_listed_by_its_name(tmp_path: Path, name: str) -> None:
    async for local in _serve(tmp_path, tools=[_Analysis(name)]):
        assert local.toolbox.analysis_tools == (name,)
        assert local.stack.advertised()["inputSchema"]["properties"]["tool"]["enum"] == [name]


# --- roofline -----------------------------------------------------------------------------------------------------


async def test_a_specialist_gets_the_roofline_of_the_trace_its_profile_test_left(served: Local) -> None:
    status, record = await _roofline(served, TRACE)
    result = record["result"]
    assert status is JobStatus.SUCCEEDED and record["summary"] == "roofline analysed"
    assert (result["status"], result["route"], result["target_platform"]) == ("ok", "bypass", "MI355X")
    roofline = Path(result["kernel_roofline_path"])
    assert roofline.is_file() and roofline.is_relative_to(
        served.session / "tool-results"
    )  # kept with the job, not in scratch
    kernels = json.loads(roofline.read_text(encoding="utf-8"))["kernels"]
    assert {kernel["name"]: kernel["gpu_pct"] for kernel in kernels} == {"aten::paged_attn": 60.0, "aten::mm": 40.0}


async def test_the_platform_of_the_request_is_the_one_analysed_for(served: Local) -> None:
    _, record = await _roofline(served, TRACE, target_platform="MI300X")
    assert record["result"]["target_platform"] == "MI300X"


async def test_the_roofline_needs_a_profile_test_it_does_not_profile_itself(served: Local) -> None:
    text = await served.stack.refused(tool="roofline")
    assert "profile_required" in text and "reads the trace of a profile test" in text


@pytest.mark.parametrize("route", ["agent", "report"])
async def test_a_route_other_than_bypass_fails_the_job_by_name_and_starts_no_analysis(
    served: Local, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    from hyperloom_multibranch.toolbox.tools import roofline as module

    async def never(payload: dict, *, session_dir: Path) -> dict:
        raise AssertionError("the analysis must not start")

    monkeypatch.setattr(module, "trace_analyze_handler", never)
    status, record = await _roofline(served, TRACE, analysis_route=route)
    assert status is JobStatus.FAILED and f"not {route!r}" in record["summary"]
    assert record["result"]["error_class"] == "invalid_analysis_route"


async def test_the_route_the_host_environment_names_is_not_the_rooflines(
    served: Local, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HYPERLOOM_TRACE_ANALYSIS_ROUTE", "agent")  # an agent route would start a model session
    status, record = await _roofline(served, TRACE)
    assert status is JobStatus.SUCCEEDED and record["result"]["route"] == "bypass"


@pytest.mark.parametrize(
    ("events", "error_class"),
    [(CPU_ONLY, "trace_without_gpu_kernels"), (None, "trace_input_missing")],
)
async def test_a_trace_with_no_kernel_or_no_trace_fails_the_job_instead_of_answering_ok_with_nothing(
    served: Local, events: list[dict[str, Any]] | None, error_class: str
) -> None:
    status, record = await _roofline(served, events)
    assert status is JobStatus.FAILED and record["result"]["error_class"] == error_class
    assert record["summary"].startswith(("none of the first", "trace_input does not exist", "no trace files"))


async def test_an_analysis_that_scanned_none_of_the_traces_kernels_fails_the_job_instead_of_answering_ok(
    served: Local, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What ``roofline-83e37a2e0267`` answered: ``Events scanned: 0`` over a trace with GPU kernels, and ``ok``."""
    from hyperloom_multibranch.toolbox.tools import roofline as module

    async def scanned_nothing(payload: dict, *, session_dir: Path) -> dict:
        warning = {"code": "bypass_trace_aggregation_truncated", "severity": "warning"}
        return {"status": "ok", "route": "bypass", "hot_kernels": [], "trace_health_warnings": [warning]}

    monkeypatch.setattr(module, "trace_analyze_handler", scanned_nothing)
    status, record = await _roofline(served, TRACE)

    assert status is JobStatus.FAILED and record["result"]["error_class"] == "roofline_without_kernels"
    assert "no hot kernel" in record["summary"] and "bypass_trace_aggregation_truncated" in record["result"]["error"]


# --- upstream PR search ---------------------------------------------------------------------------------------------


async def test_a_specialist_gets_the_pull_requests_a_search_finds(
    served: Local, monkeypatch: pytest.MonkeyPatch
) -> None:
    github = _install(monkeypatch, Github([_pr(42, "Faster MoE decode"), _pr(7, "Fix gemm tuning")]))
    status, record = await _search(served, keywords=["moe"], limit=2)
    assert status is JobStatus.SUCCEEDED
    assert [(c["ref"], c["title"]) for c in record["result"]["candidates"]] == [
        ("PR:42", "Faster MoE decode"),
        ("PR:7", "Fix gemm tuning"),
    ]
    assert record["summary"] == f"2 upstream PR candidate(s) in {SGLANG}; top: PR:42 Faster MoE decode"
    ((url, *_),) = github.requests
    assert "repo%3Asgl-project%2Fsglang" in url and "moe" in url and "per_page=2" in url


async def test_the_search_is_made_by_the_toolbox_process_with_its_token_and_the_token_is_not_in_the_result(
    served: Local, monkeypatch: pytest.MonkeyPatch
) -> None:
    github = _install(monkeypatch, Github([_pr(1, "x")]))
    _, record = await _search(served)
    ((_, authorization, pid, thread),) = github.requests
    assert authorization == f"Bearer {HOST_TOKEN}" and pid == os.getpid()
    assert thread != threading.main_thread().name  # a worker thread: the reactor stays free
    assert HOST_TOKEN not in json.dumps(record)


def test_a_sandbox_is_not_given_the_hosts_github_token(tmp_path: Path) -> None:
    template = _runner_template(LocalSession(tmp_path / "session", "run-1", [0]), LocalOptions())
    assert not {"GITHUB_TOKEN", "GH_TOKEN"} & set(template.envs)


async def test_no_match_on_a_github_that_answers_is_a_success_with_no_candidates(
    served: Local, monkeypatch: pytest.MonkeyPatch
) -> None:
    github = _install(monkeypatch, Github([]))
    status, record = await _search(served, keywords=["nothing-matches-this"])
    assert status is JobStatus.SUCCEEDED and record["result"]["candidates"] == []
    assert len(github.requests) == 1


@pytest.mark.parametrize(
    ("github", "reason"),
    [
        (Github(failing=urllib.error.URLError("unreachable")), "cannot be reached from this host (URLError)"),
        (Github(failing=_http_error(403)), "refused the search (HTTP 403), most likely a rate limit"),
        (Github(failing=_http_error(429)), "refused the search (HTTP 429), most likely a rate limit"),
        (Github(failing=_http_error(401)), "rejected the token"),
        (Github(failing=_http_error(422)), "answered the search with HTTP 422"),
        (Github(failing=TimeoutError("slow")), "(TimeoutError)"),
        (Github(answer={"message": "API rate limit exceeded"}), "not a search result"),
    ],
    ids=["no-network", "rate-limited", "too-many", "token-rejected", "bad-query", "timeout", "not-a-result"],
)
async def test_a_github_that_cannot_search_fails_the_job_with_why_it_is_not_an_empty_list(
    served: Local, monkeypatch: pytest.MonkeyPatch, github: Github, reason: str
) -> None:
    _install(monkeypatch, github)
    status, record = await _search(served)
    assert status is JobStatus.FAILED and record["result"] is None
    assert record["error"].startswith("upstream PR search cannot use GitHub: ") and reason in record["error"]
    assert HOST_TOKEN not in json.dumps(record)


async def test_without_a_token_the_failure_says_the_host_has_none(
    served: Local, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN")
    _install(monkeypatch, Github(failing=_http_error(403)))
    _, record = await _search(served)
    assert "the host has no GITHUB_TOKEN or GH_TOKEN" in record["error"]


async def test_a_search_with_a_token_does_not_tell_the_host_to_get_one(
    served: Local, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, Github(failing=_http_error(403)))
    _, record = await _search(served)
    assert "GITHUB_TOKEN" not in record["error"]


@pytest.mark.parametrize(
    ("params", "text"),
    [
        ({"repo_url": "https://github.com/someone/private-repo"}, "someone/private-repo is not an upstream repository"),
        ({"repo_url": "not a url"}, "not a GitHub repository URL"),
        ({"repo_url": SGLANG, "limit": 99}, "between 1 and 20"),
        ({"repo_url": SGLANG, "pr_monitor_url": "http://internal.example/"}, "server_owned_param"),
    ],
)
async def test_a_search_that_can_never_run_is_refused_before_it_starts(
    served: Local, monkeypatch: pytest.MonkeyPatch, params: dict[str, Any], text: str
) -> None:
    github = _install(monkeypatch, Github())
    assert text in await served.stack.refused(tool="upstream_pr_search", params=params)
    assert github.requests == []


async def test_a_search_names_no_profile_test(served: Local) -> None:
    profile = await served.profile_test()
    text = await served.stack.refused(
        tool="upstream_pr_search", params={"repo_url": SGLANG}, profile_request_id=profile
    )
    assert "reads no profile test" in text
