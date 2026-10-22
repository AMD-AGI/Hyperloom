# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``start_analysis``: a specialist starts tracelens or roofline on its own branch's head commit, over real HTTP."""

from __future__ import annotations

import asyncio
import json
import secrets
import urllib.request
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from hyperloom.contracts.measurement import Purpose, TestKind
from hyperloom.hosted.mcp_server import McpServer, McpServerConfig
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH
from hyperloom_multibranch.contracts.notebook.records import ToolResultPointer
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.toolbox.mcp_tools import (
    McpSpecialistTokens,
    SpecialistToolOptions,
    register_specialist_tools,
)
from hyperloom_multibranch.toolbox.tests.helpers import PARENT, Rig, build, ok, req
from hyperloom_multibranch.toolbox.tools import ToolJobs, ToolOutput, ToolRequest
from hyperloom_multibranch.toolbox.tools import op_shares as op_shares_module
from hyperloom_multibranch.toolbox.tools.op_shares import with_op_shares

SHARES = {"gemm": 61.5, "attention": 20.0}
HEAD = "head-of-b1"
FORGED = "forged-commit"


class _Analysis:
    """Stands in for the TraceLens analysis: it ends at once and fills shares the way the real tools do."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.requests: list[ToolRequest] = []

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        self.requests.append(request)
        return with_op_shares(ToolOutput(result={}, summary="analysed"), request)


class _Capture:
    """The inbox push: remembers every pointer, then writes it to the notebook as the toolbox does."""

    def __init__(self, notebook: LocalNotebook) -> None:
        self.notebook, self.pointers = notebook, []

    async def notify_tool_result(self, pointer: ToolResultPointer) -> None:
        self.pointers.append(pointer)
        await self.notebook.notify_tool_result(pointer)


@dataclass
class Stack:
    rig: Rig
    jobs: ToolJobs
    capture: _Capture
    tools: dict[str, _Analysis]
    server: McpServer
    tokens: McpSpecialistTokens

    def call(self, as_branch: str, name: str, /, **arguments: Any) -> dict[str, Any]:
        token = self.server.issue_token(f"specialist-{as_branch}", as_branch)
        message = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
        headers = {"Content-Type": "application/json", "X-Hyperloom-Token": token}
        request = urllib.request.Request(self.server.url, json.dumps(message).encode(), headers)
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())

    async def start(self, as_branch: str = "b1", /, **arguments: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self.call, as_branch, "start_analysis", **arguments)

    async def refused(self, as_branch: str = "b1", /, **arguments: Any) -> str:
        reply = await self.start(as_branch, **arguments)
        if "error" in reply:
            return str(reply["error"]["message"])
        assert reply["result"]["isError"] is True, reply
        return str(reply["result"]["content"][0]["text"])


@pytest.fixture(autouse=True)
def _shares(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(op_shares_module, "read_op_shares", lambda result: dict(SHARES))


async def _stack(tmp_path: Path, *, with_tools: bool = True) -> AsyncIterator[Stack]:
    tools = {name: _Analysis(name) for name in ("tracelens", "roofline", "profile")}
    holder: list[_Capture] = []

    def make(notebook: LocalNotebook) -> ToolJobs:
        holder.append(_Capture(notebook))
        return ToolJobs(list(tools.values()), holder[0], tmp_path / "tool-results")

    rig = build(tmp_path, tools=make if with_tools else None)
    rig.repo.heads["b1"] = HEAD
    rig.repo.recipes[HEAD] = rig.repo.recipes[PARENT]
    await rig.toolbox.start()
    server = McpServer(McpServerConfig(host_token=secrets.token_hex(16)), campaign="w32")
    root = tmp_path

    def notebooks(branch: str) -> LocalNotebook:
        return LocalNotebook(root / "notebook", Writer(Role.SPECIALIST_RUNTIME, branch, branch))

    options = SpecialistToolOptions(analysis=rig.toolbox, toolbox_loop=asyncio.get_running_loop())
    register_specialist_tools(server, toolbox=rig.toolbox, notebooks=notebooks, options=options)
    server.start()
    try:
        jobs = rig.toolbox._tools
        yield Stack(rig, jobs, holder[0] if holder else None, tools, server, McpSpecialistTokens(server))  # type: ignore[arg-type]
    finally:
        server.stop()
        await rig.toolbox.close()


@pytest.fixture
async def stack(tmp_path: Path) -> AsyncIterator[Stack]:
    async for item in _stack(tmp_path):
        yield item


@pytest.fixture
async def stack_without_tools(tmp_path: Path) -> AsyncIterator[Stack]:
    async for item in _stack(tmp_path, with_tools=False):
        yield item


def _text(reply: dict[str, Any]) -> str:
    assert reply["result"]["isError"] is False, reply
    return str(reply["result"]["content"][0]["text"])


async def _profile_test(stack: Stack, branch: str = "b1", commit: str = PARENT, **result: Any) -> str:
    """A measured PROFILE test of ``branch`` on ``commit``, as the notebook holds it."""
    rig = stack.rig
    rig.executor.script = [lambda job: ok(job, profile_location="traces/p1", **result)]
    request = replace(
        req(branch, purpose=Purpose.PROFILE, kind=TestKind.PROFILE, server_args="--p"), parent_commit=commit
    )
    request_id = await rig.toolbox.submit_test(request)
    await rig.toolbox.drain()
    return request_id


@pytest.mark.parametrize("tool", ["tracelens", "roofline"])
async def test_a_specialist_starts_each_analysis_tool_for_its_own_branch(stack: Stack, tool: str) -> None:
    profile = await _profile_test(stack, commit=HEAD)
    job = json.loads(_text(await stack.start(tool=tool, profile_request_id=profile)))["job_id"]
    assert await stack.jobs.wait(job) is not None
    (request,) = stack.tools[tool].requests
    assert (request.tool, request.branch) == (tool, "b1")
    assert request.params["trace_input"] == "traces/p1"
    (pointer,) = stack.capture.pointers
    assert (pointer.branch, pointer.job_id) == ("b1", job)


async def test_the_pointer_carries_the_server_side_head_commit_not_the_callers(stack: Stack) -> None:
    profile = await _profile_test(stack, commit=HEAD)
    job = json.loads(_text(await stack.start(tool="tracelens", profile_request_id=profile)))["job_id"]
    await stack.jobs.wait(job)
    (pointer,) = stack.capture.pointers
    assert pointer.commit_id == HEAD != PARENT
    assert dict(pointer.op_shares) == SHARES
    assert pointer.profile_request_id == profile


async def test_a_caller_supplied_commit_id_reaches_neither_the_schema_nor_the_params(stack: Stack) -> None:
    top = await stack.refused(tool="tracelens", commit_id=FORGED)
    assert "additional properties" in top.lower()
    inner = await stack.refused(tool="tracelens", params={"commit_id": FORGED})
    assert "server_owned_param" in inner and "commit_id" in inner
    nested = await stack.refused(tool="roofline", params={"profile_request_id": "r1"})
    assert "server_owned_param" in nested
    assert stack.tools["tracelens"].requests == [] and stack.capture.pointers == []


async def test_a_caller_cannot_profile_another_branch(stack: Stack) -> None:
    reply = await stack.start(tool="tracelens", branch="b2", profile_request_id="r1")
    assert "additional properties" in reply["error"]["message"].lower()
    profile = await _profile_test(stack, "b2")
    job = json.loads(_text(await stack.start("b2", tool="tracelens", profile_request_id=profile)))["job_id"]
    await stack.jobs.wait(job)
    (request,) = stack.tools["tracelens"].requests
    assert request.branch == "b2" and request.params["commit_id"] == PARENT  # b2's own head, never b1's


@pytest.mark.parametrize("tool", ["profile", "build", "nonesuch"])
async def test_any_other_tool_is_refused_with_its_reason(stack: Stack, tool: str) -> None:
    text = await stack.refused(tool=tool)
    assert "tool_not_allowed" in text and "tracelens and roofline" in text
    assert stack.tools["profile"].requests == []


async def test_a_profile_request_id_is_kept_only_for_a_profile_test_of_this_commit(stack: Stack) -> None:
    stack.rig.repo.heads["b1"] = PARENT  # the commit the profile test ran on
    mine = await _profile_test(stack)
    job = json.loads(_text(await stack.start(tool="tracelens", profile_request_id=mine)))
    await stack.jobs.wait(job["job_id"])
    (pointer,) = stack.capture.pointers
    assert (pointer.commit_id, pointer.profile_request_id) == (PARENT, mine)
    (request,) = stack.tools["tracelens"].requests
    assert request.params["trace_input"] == "traces/p1"  # the profile record's own location; the rig has no root


async def test_a_branch_whose_head_is_the_commit_integration_profiled_analyses_that_profile(stack: Stack) -> None:
    # profile_exists points a refill branch at the profile another branch ran on the same commit: it must be analysable
    stack.rig.repo.heads["b1"] = HEAD
    theirs = await _profile_test(stack, INTEGRATION_BRANCH, commit=HEAD)
    job = json.loads(_text(await stack.start(tool="tracelens", profile_request_id=theirs)))["job_id"]
    assert await stack.jobs.wait(job) is not None
    (request,) = stack.tools["tracelens"].requests
    assert (request.branch, request.params["commit_id"]) == ("b1", HEAD)
    assert request.params["trace_input"] == "traces/p1"  # integration's own trace, read off its record
    (pointer,) = stack.capture.pointers
    assert (pointer.branch, pointer.commit_id, pointer.profile_request_id) == ("b1", HEAD, theirs)


async def test_the_profile_another_branch_ran_is_that_of_the_commit_not_of_the_branch(stack: Stack) -> None:
    stack.rig.repo.heads["b1"] = PARENT
    stack.rig.repo.heads["b2"] = PARENT
    theirs = await _profile_test(stack, "b2")
    job = json.loads(_text(await stack.start(tool="roofline", profile_request_id=theirs)))["job_id"]
    await stack.jobs.wait(job)
    (request,) = stack.tools["roofline"].requests
    assert (request.branch, request.params["commit_id"]) == ("b1", PARENT)


async def test_a_profile_of_another_commit_or_an_unknown_request_is_refused(stack: Stack) -> None:
    theirs = await _profile_test(stack, "b2", commit=PARENT)
    integration = await _profile_test(stack, INTEGRATION_BRANCH, commit=HEAD)
    stack.rig.repo.heads["b1"] = "moved-on"  # a profile of PARENT or HEAD is not of b1's head, whoever ran it
    for wrong in (theirs, integration, "no-such-request"):
        text = await stack.refused(tool="tracelens", profile_request_id=wrong)
        assert "profile_request_unverified" in text
    assert stack.tools["tracelens"].requests == []


async def test_a_branch_whose_head_cannot_be_read_is_refused(stack: Stack) -> None:
    async def broken(branch: str) -> str:
        raise KeyError(branch)

    stack.rig.repo.head = broken  # type: ignore[method-assign]
    text = await stack.refused(tool="tracelens", profile_request_id="r1")
    assert "commit_unknown" in text and "b1" in text
    assert stack.tools["tracelens"].requests == []


async def test_a_toolbox_without_a_tool_executor_refuses_by_name(stack_without_tools: Stack) -> None:
    text = await stack_without_tools.refused(tool="tracelens", profile_request_id="r1")
    assert "no_tool_executor" in text


async def test_without_an_analysis_starter_the_route_refuses_by_name(tmp_path: Path) -> None:
    rig = build(tmp_path)
    server = McpServer(McpServerConfig(host_token=secrets.token_hex(16)), campaign="w32")
    register_specialist_tools(
        server,
        toolbox=rig.toolbox,
        notebooks=lambda b: LocalNotebook(tmp_path / "notebook", Writer(Role.SPECIALIST_RUNTIME, b, b)),
        options=SpecialistToolOptions(toolbox_loop=asyncio.get_running_loop()),
    )
    server.start()
    try:
        bare = Stack(rig, None, None, {}, server, McpSpecialistTokens(server))  # type: ignore[arg-type]
        assert "no_tool_executor" in await bare.refused(tool="tracelens")
    finally:
        server.stop()


@pytest.mark.parametrize("with_profile", [True, False])
async def test_a_trace_input_of_the_callers_is_refused_with_or_without_a_profile_request_id(
    stack: Stack, with_profile: bool
) -> None:
    stack.rig.repo.heads["b1"] = PARENT
    mine = await _profile_test(stack) if with_profile else ""
    for tool in ("tracelens", "roofline"):
        text = await stack.refused(tool=tool, params={"trace_input": "/etc"}, profile_request_id=mine)
        assert "server_owned_param" in text and "trace_input" in text
    assert stack.tools["tracelens"].requests == [] and stack.tools["roofline"].requests == []


async def test_tracelens_without_a_profile_request_id_is_refused_by_name(stack: Stack) -> None:
    text = await stack.refused(tool="tracelens")
    assert "profile_required" in text and "name the profile test with profile_request_id" in text
    assert stack.tools["tracelens"].requests == []


async def test_a_profile_test_without_a_trace_location_is_refused_by_name(stack: Stack) -> None:
    stack.rig.repo.heads["b1"] = PARENT
    mine = await _profile_test(stack)
    records = stack.rig.toolbox._records
    found = await records.profiles(PARENT)
    bare = tuple(replace(r, result=replace(r.result, profile_location="")) for r in found)

    async def profiles(commit: str) -> Any:
        return bare

    records.profiles = profiles  # type: ignore[method-assign]
    text = await stack.refused(tool="tracelens", profile_request_id=mine)
    assert "profile_without_trace" in text
    assert stack.tools["tracelens"].requests == []


POINT = """\
benchmark:
  framework: sglang
  envs:
    TP: '2'
    CONC: '4'
    ISL: '256'
    OSL: 64
    RANDOM_RANGE_RATIO: '1'
"""
LANDED = "tests/p1/attempt-1"
HINTS = {"split_conc": "4", "split_osl": "64", "split_r": "1", "split_isl": "256"}


async def _profile_with_points(stack: Stack, tmp_path: Path, files: dict[str, str]) -> str:
    """A PROFILE test of b1 whose landed test directory holds ``files`` (relative to it), under the toolbox's root."""
    stack.rig.repo.heads["b1"] = PARENT
    session = tmp_path / "session"
    for name, text in files.items():
        (session / LANDED / name).parent.mkdir(parents=True, exist_ok=True)
        (session / LANDED / name).write_text(text, encoding="utf-8")
    stack.rig.toolbox.trace_root = session
    return await _profile_test(stack, output_dir=LANDED)


async def _split_params(stack: Stack, tool: str, profile: str) -> dict[str, Any]:
    job = json.loads(_text(await stack.start(tool=tool, profile_request_id=profile)))["job_id"]
    await stack.jobs.wait(job)
    (request,) = stack.tools[tool].requests
    return {key: value for key, value in request.params.items() if key.startswith("split_")}


async def test_the_splitter_hints_are_the_workload_of_the_profile_tests_own_point(stack: Stack, tmp_path: Path) -> None:
    profile = await _profile_with_points(stack, tmp_path, {"variant_00_test/config.yaml": POINT})
    assert await _split_params(stack, "tracelens", profile) == HINTS
    roofline_profile = await _split_params(stack, "roofline", profile)
    assert roofline_profile == {}  # the roofline tool has no splitter


@pytest.mark.parametrize("key", sorted(HINTS))
@pytest.mark.parametrize("with_profile", [True, False])
async def test_a_hint_of_the_callers_is_refused_and_cannot_replace_the_point(
    stack: Stack, tmp_path: Path, key: str, with_profile: bool
) -> None:
    profile = await _profile_with_points(stack, tmp_path, {"variant_00_test/config.yaml": POINT})
    for tool in ("tracelens", "roofline"):
        text = await stack.refused(tool=tool, params={key: "1"}, profile_request_id=profile if with_profile else "")
        assert "server_owned_param" in text and key in text
    assert stack.tools["tracelens"].requests == [] and stack.tools["roofline"].requests == []


@pytest.mark.parametrize(
    ("files", "hints"),
    [
        ({}, {}),
        ({"variant_00_test/config.yaml": POINT, "variant_01_test/config.yaml": POINT}, {}),
        ({"variant_00_test/config.yaml": "benchmark: [not, a, mapping"}, {}),
        ({"variant_00_test/config.yaml": "benchmark:\n  envs: 7\n"}, {}),
        ({"variant_00_test/config.yaml": "- a\n- list\n"}, {}),
        (
            {"variant_00_test/config.yaml": POINT.replace("    CONC: '4'\n", "")},
            {"split_osl": "64", "split_r": "1", "split_isl": "256"},
        ),
        (
            {"variant_00_test/config.yaml": POINT.replace("    ISL: '256'\n", "")},
            {"split_conc": "4", "split_osl": "64", "split_r": "1"},
        ),
        (
            {
                "variant_00_test/config.yaml": POINT.replace("'4'", "many")
                .replace("64", "0")
                .replace("'1'", "2")
                .replace("'256'", "-1")
            },
            {},
        ),
        ({"variant_00_test/benchmark_sglang_1/config.yaml": POINT}, {}),  # the benchmark's own copy is not the point
    ],
)
async def test_a_point_that_cannot_give_a_hint_gives_none_and_the_analysis_still_starts(
    stack: Stack, tmp_path: Path, files: dict[str, str], hints: dict[str, str]
) -> None:
    profile = await _profile_with_points(stack, tmp_path, files)
    assert await _split_params(stack, "tracelens", profile) == hints


async def test_a_point_that_is_a_link_out_of_the_session_gives_no_hint(stack: Stack, tmp_path: Path) -> None:
    outside = tmp_path / "outside.yaml"
    outside.write_text(POINT, encoding="utf-8")
    profile = await _profile_with_points(stack, tmp_path, {"variant_00_test/other.yaml": ""})
    (tmp_path / "session" / LANDED / "variant_00_test" / "config.yaml").symlink_to(outside)
    assert await _split_params(stack, "tracelens", profile) == {}


async def test_without_a_session_root_there_is_no_point_to_read_and_no_hint(stack: Stack, tmp_path: Path) -> None:
    stack.rig.repo.heads["b1"] = PARENT
    profile = await _profile_test(stack, output_dir=LANDED)  # the rig has no trace_root
    assert await _split_params(stack, "tracelens", profile) == {}
