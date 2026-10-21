# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The specialist tools on a running MCP service: real HTTP requests, the local notebook behind them."""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.contracts.measurement import TestRequest
from hyperloom.hosted.mcp_server import HOST_KIND, McpServer, McpServerConfig
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.notebook.records import StartupBrief
from hyperloom_multibranch.contracts.toolbox import (
    SCOPE_SPECIALIST,
    Backpressure,
    CancelOutcome,
    QueueState,
    RequestRejected,
    TestProgress,
)
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import TTL, FakeClock, delta, intent, request, result
from hyperloom_multibranch.notebook.tests.sessions import Writers, scripted_session
from hyperloom_multibranch.toolbox import mcp_tools
from hyperloom_multibranch.toolbox.mcp_tools import (
    REPLY_INBOX_HEADER,
    REPLY_INBOX_LINES,
    POLL_INBOX_MAX_BYTES,
    POLL_INBOX_MAX_ITEMS,
    SPECIALIST_TOKEN_ENV,
    WAIT_TEST_MAX_SECONDS,
    McpSpecialistTokens,
    jsonable,
    SpecialistToolOptions,
    register_specialist_tools,
    wait_budget,
)

HOST_TOKEN = "host-token-0123456789abcdef"
CLAIM = {"parent_commit": "c1", "target": "attention", "mechanism": "flash decode", "layer": "source"}


class FakeToolbox:
    def __init__(self) -> None:
        self.submitted: list[TestRequest] = []
        self.cancelled: list[str] = []
        self.refuse: Exception | None = None
        self.progress_of: dict[str, TestProgress] = {}

    async def submit_test(self, request: TestRequest) -> str:
        if self.refuse is not None:
            raise self.refuse
        self.submitted.append(request)
        return f"req-{len(self.submitted)}"

    async def cancel(self, request_id: str) -> CancelOutcome:
        self.cancelled.append(request_id)
        return CancelOutcome.WITHDRAWN

    async def queue_state(self) -> QueueState:
        return QueueState(queued_by_priority={0: 1, 1: 2, 2: 0, 3: 0, 4: 0}, running_by_branch={"s1": 1})

    async def progress(self, request_id: str) -> TestProgress | None:
        return self.progress_of.get(request_id)


class Rig:
    def __init__(self, server: McpServer, toolbox: FakeToolbox, root: Path, clock: FakeClock) -> None:
        self.server, self.toolbox, self.root, self.clock = server, toolbox, root, clock

    def token(self, branch: str) -> str:
        return self.server.issue_token(f"specialist-{branch}", branch)

    def post(self, token: str | None, message: Any) -> tuple[int, Any]:
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["X-Hyperloom-Token"] = token
        request = urllib.request.Request(self.server.url, json.dumps(message).encode(), headers)
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read() or "null")
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read() or "null")

    def call(self, token: str | None, tool: str, **arguments: Any) -> tuple[int, Any]:
        message = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": arguments}}
        return self.post(token, message)

    def ok(self, token: str, tool: str, **arguments: Any) -> str:
        status, reply = self.call(token, tool, **arguments)
        assert status == 200 and reply["result"]["isError"] is False, reply
        return reply["result"]["content"][0]["text"]

    def failed(self, token: str, tool: str, **arguments: Any) -> str:
        status, reply = self.call(token, tool, **arguments)
        assert status == 200 and reply["result"]["isError"] is True, reply
        return reply["result"]["content"][0]["text"]

    def observer(self) -> LocalNotebook:
        return LocalNotebook(self.root, Writer(Role.OBSERVER, "observer"), clock=self.clock)


@pytest.fixture
async def seeded(tmp_path: Path) -> tuple[Path, FakeClock]:
    clock = FakeClock()
    root = tmp_path / "notebook"
    await scripted_session(root, clock)
    return root, clock


def _running(seeded: tuple[Path, FakeClock], options: SpecialistToolOptions) -> Iterator[Rig]:
    root, clock = seeded
    toolbox = FakeToolbox()
    server = McpServer(McpServerConfig(host_token=HOST_TOKEN), campaign="camp")

    def notebooks(branch: str) -> LocalNotebook:
        return LocalNotebook(root, Writer(Role.SPECIALIST_RUNTIME, branch, branch), clock=clock)

    register_specialist_tools(server, toolbox=toolbox, notebooks=notebooks, options=options)
    server.start()
    yield Rig(server, toolbox, root, clock)
    server.stop()


@pytest.fixture
def rig(seeded: tuple[Path, FakeClock]) -> Iterator[Rig]:
    yield from _running(seeded, SpecialistToolOptions(claim_ttl_seconds=60.0))


class BranchBrief:
    """A brief provider that says which branch it was made for."""

    def __init__(self, branch: str) -> None:
        self.branch = branch

    async def startup_brief(self, start_commit: str) -> StartupBrief:
        return StartupBrief(start_commit, (), lessons=(f"lesson for {self.branch}",))


@pytest.fixture
def briefed_rig(seeded: tuple[Path, FakeClock]) -> Iterator[Rig]:
    options = SpecialistToolOptions(claim_ttl_seconds=60.0, briefing_for=BranchBrief)
    yield from _running(seeded, options)


def test_the_brief_is_the_providers_for_the_branch_of_the_token(briefed_rig: Rig) -> None:
    for branch in ("s1", "s2"):
        text = briefed_rig.ok(briefed_rig.token(branch), "notebook_brief", start_commit="c1")
        assert f"- lesson for {branch}" in text
        assert text.count("lesson for") == 1


def test_a_branch_named_in_the_arguments_is_refused_not_used(briefed_rig: Rig) -> None:
    status, reply = briefed_rig.call(briefed_rig.token("s1"), "notebook_brief", start_commit="c1", branch="s2")
    assert status == 200 and "lesson for" not in json.dumps(reply)
    assert reply.get("error") is not None or reply["result"]["isError"] is True


def test_without_briefing_for_the_brief_is_the_notebooks_own(rig: Rig) -> None:
    text = rig.ok(rig.token("s1"), "notebook_brief", start_commit="c1")
    assert text.startswith("Startup brief from c1") and "lesson for" not in text


def test_a_specialist_sees_exactly_the_specialist_tools(rig: Rig) -> None:
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    _, reply = rig.post(rig.token("s1"), message)
    names = sorted(tool["name"] for tool in reply["result"]["tools"])
    assert names == sorted(
        ["submit_test", "test_result", "wait_test", "cancel_test", "queue_state", "poll_inbox", "start_analysis"]
        + [f"notebook_{name}" for name in ("lookup", "claim", "release", "search", "record", "branch_log")]
        + ["notebook_append_branch_log", "notebook_learn", "notebook_learnings", "notebook_brief", "notebook_replay"]
    )


def test_the_host_token_cannot_call_any_specialist_tool(rig: Rig) -> None:
    for tool in ("queue_state", "notebook_search", "test_result"):
        status, reply = rig.call(HOST_TOKEN, tool)
        assert (status, reply["error"]["code"]) == (200, -32001)


def test_submit_test_takes_the_branch_from_the_token(rig: Rig) -> None:
    text = rig.ok(rig.token("s1"), "submit_test", kind="test", purpose="explore", parent_commit="c1")
    assert json.loads(text) == {"request_id": "req-1"}
    (request,) = rig.toolbox.submitted
    assert (request.branch_id, request.parent_commit, request.request_id) == ("s1", "c1", None)


def test_submit_test_cannot_name_a_branch_or_an_id(rig: Rig) -> None:
    token = rig.token("s1")
    for extra in ({"branch_id": "s2"}, {"branch_id": "s1"}, {"request_id": "r9"}):
        _, reply = rig.call(token, "submit_test", kind="test", purpose="explore", parent_commit="c1", **extra)
        assert reply["error"]["code"] == -32602 and "additional properties" in reply["error"]["message"].lower()
    assert rig.toolbox.submitted == []


def test_submit_test_reports_a_contract_violation_as_invalid_params(rig: Rig) -> None:
    _, reply = rig.call(rig.token("s1"), "submit_test", kind="test", purpose="confirm", parent_commit="c1")
    assert reply["error"]["code"] == -32602 and "retest_of" in reply["error"]["message"]
    assert rig.toolbox.submitted == []


def test_a_specialist_cannot_submit_a_confirm_retest_but_can_explore(rig: Rig) -> None:
    token = rig.token("s1")
    confirm = {"kind": "test", "purpose": "confirm", "parent_commit": "c1", "retest_of": "req-0"}
    text = rig.failed(token, "submit_test", **confirm)
    assert text.startswith("rejected (confirm_not_for_specialists):") and "purpose=explore" in text
    assert rig.toolbox.submitted == []
    rig.ok(token, "submit_test", kind="test", purpose="explore", parent_commit="c1")
    assert [request.purpose.value for request in rig.toolbox.submitted] == ["explore"]


def test_toolbox_refusals_come_back_as_error_results_with_their_code(rig: Rig) -> None:
    token = rig.token("s1")
    args = {"kind": "test", "purpose": "explore", "parent_commit": "c1"}
    rig.toolbox.refuse = RequestRejected("patch does not apply", "delta_not_applicable")
    assert rig.failed(token, "submit_test", **args) == "rejected (delta_not_applicable): patch does not apply"
    rig.toolbox.refuse = Backpressure("s1", 9, 8)
    assert rig.failed(token, "submit_test", **args).startswith("backpressure:")


def test_test_result_returns_the_branchs_own_record(rig: Rig) -> None:
    body = json.loads(rig.ok(rig.token("s1"), "test_result", request_id="r1"))
    assert body["state"] == "measured"
    assert body["result"]["request_id"] == "r1" and body["result"]["validity"] == "succeeded"


def test_test_result_of_another_branch_is_forbidden(rig: Rig) -> None:
    _, reply = rig.call(rig.token("s1"), "test_result", request_id="r2")
    assert reply["error"] == {"code": -32001, "message": "a specialist reads the tests of its own branch only"}


def test_test_result_of_an_unknown_request_is_an_error_result(rig: Rig) -> None:
    assert "UnknownRequest" in rig.failed(rig.token("s1"), "test_result", request_id="nope")


PAGED_KV = intent("s1", "c1", "attention", "paged kv")


@pytest.fixture
async def waiting(seeded: tuple[Path, FakeClock]) -> Writers:
    """The scripted session plus ``r9``, a test of ``s1`` that is accepted and not over."""
    w = Writers(*seeded)
    change = delta("--block-size 16")
    claim = await w.as_role(Role.SPECIALIST_RUNTIME, "s1").acquire(PAGED_KV, change.fingerprint(), ttl_seconds=TTL)
    toolbox = w.as_role(Role.TOOLBOX)
    await toolbox.write_request(request("r9", PAGED_KV, change))
    await toolbox.attach_request(claim.key, "r9", ttl_seconds=TTL)
    return w


@pytest.fixture
def wait_rig(seeded: tuple[Path, FakeClock], waiting: Writers) -> Iterator[Rig]:
    yield from _running(seeded, SpecialistToolOptions(claim_ttl_seconds=60.0, wait_poll_seconds=0.05))


def _wait(rig: Rig, branch: str = "s1", **arguments: Any) -> dict[str, Any]:
    return json.loads(rig.ok(rig.token(branch), "wait_test", **arguments))


def test_wait_test_answers_a_measured_test_at_once_as_test_result_does(rig: Rig) -> None:
    started = time.monotonic()
    body = _wait(rig, request_id="r1")
    assert time.monotonic() - started < 5
    assert body["state"] == "measured" and body == json.loads(rig.ok(rig.token("s1"), "test_result", request_id="r1"))


async def test_wait_test_returns_as_soon_as_the_result_lands(wait_rig: Rig, waiting: Writers) -> None:
    call = asyncio.create_task(asyncio.to_thread(_wait, wait_rig, request_id="r9", wait_seconds=60))
    await asyncio.sleep(0.3)
    await waiting.as_role(Role.TOOLBOX).write_result(result("r9"))
    body = await asyncio.wait_for(call, 10)
    assert body["state"] == "measured" and body["result"]["request_id"] == "r9"


def test_wait_test_reports_how_many_tests_are_queued_ahead_of_a_waiting_one(wait_rig: Rig) -> None:
    wait_rig.toolbox.progress_of["r9"] = TestProgress("queued", queued_ahead=2)
    started = time.monotonic()
    body = _wait(wait_rig, request_id="r9", wait_seconds=1)
    assert time.monotonic() - started >= 1
    assert body == {"request_id": "r9", "state": "queued", "waited_seconds": 1, "progress": {"queued_ahead": 2}}


def test_wait_test_reports_how_long_a_running_test_ran_and_what_it_wrote_last(wait_rig: Rig) -> None:
    progress = TestProgress("running", running_seconds=125.4, last_output_seconds_ago=3.2, last_line="decode batch 8")
    wait_rig.toolbox.progress_of["r9"] = progress
    body = _wait(wait_rig, request_id="r9", wait_seconds=1)
    assert (body["state"], body["waited_seconds"]) == ("running", 1)
    assert body["progress"] == {"running_seconds": 125, "last_output_seconds_ago": 3, "last_line": "decode batch 8"}


def test_wait_test_of_a_test_the_toolbox_no_longer_holds_is_in_flight_with_no_progress(wait_rig: Rig) -> None:
    assert _wait(wait_rig, request_id="r9", wait_seconds=1) == {
        "request_id": "r9",
        "state": "in_flight",
        "waited_seconds": 1,
    }


def test_wait_test_of_another_branch_is_forbidden_as_test_result_is(wait_rig: Rig) -> None:
    for branch, request_id in (("s2", "r9"), ("s1", "r2")):
        _, reply = wait_rig.call(wait_rig.token(branch), "wait_test", request_id=request_id)
        assert reply["error"] == {"code": -32001, "message": "a specialist reads the tests of its own branch only"}


@pytest.mark.parametrize(
    ("requested", "expected"), [(None, 45), (-5, 1), (0, 1), (1, 1), (30, 30), (45, 45), (240, 45)]
)
def test_wait_seconds_is_held_under_the_cli_request_timeout(requested: int | None, expected: int) -> None:
    assert WAIT_TEST_MAX_SECONDS == 45 and wait_budget(requested) == expected


def test_a_wait_of_zero_seconds_waits_one(wait_rig: Rig) -> None:
    assert _wait(wait_rig, request_id="r9", wait_seconds=0)["waited_seconds"] == 1


def test_cancel_test_only_for_the_own_branch(rig: Rig) -> None:
    token = rig.token("s1")
    _, reply = rig.call(token, "cancel_test", request_id="r2")
    assert reply["error"]["code"] == -32001 and rig.toolbox.cancelled == []
    assert json.loads(rig.ok(token, "cancel_test", request_id="r1")) == {"outcome": "withdrawn"}
    assert rig.toolbox.cancelled == ["r1"]


def test_queue_state(rig: Rig) -> None:
    body = json.loads(rig.ok(rig.token("s1"), "queue_state"))
    assert body["queued"] == 3 and body["running"] == 1
    assert body["queued_by_priority"]["1"] == 2 and body["running_by_branch"] == {"s1": 1}


def test_poll_inbox_reads_the_callers_branch(rig: Rig) -> None:
    body = json.loads(rig.ok(rig.token("s1"), "poll_inbox"))
    assert set(body) == {"items", "cursor", "remaining"} and body["remaining"] == 0


def test_claims_are_made_for_the_branch_of_the_token(rig: Rig) -> None:
    rig.ok(rig.token("s1"), "notebook_claim", **CLAIM)
    key = {"parent_commit": "c1", "target": "attention", "mechanism": "flash decode", "layer": "source"}
    other = rig.failed(rig.token("s2"), "notebook_claim", **key)
    assert other.startswith("refused") and "claimed by s1" in other
    seen = rig.ok(rig.token("s2"), "notebook_lookup", **CLAIM)
    assert "claimed by s1" in seen


def test_a_claim_cannot_name_a_holder(rig: Rig) -> None:
    _, reply = rig.call(rig.token("s1"), "notebook_claim", branch="s2", **CLAIM)
    assert reply["error"]["code"] == -32602


def test_branch_log_is_appended_to_and_read_from_the_callers_branch(rig: Rig) -> None:
    token = rig.token("s1")
    rig.ok(token, "notebook_append_branch_log", kind="plan", content="try flash decode next")
    assert "try flash decode next" in rig.ok(token, "notebook_branch_log")
    assert "try flash decode next" not in rig.ok(rig.token("s2"), "notebook_branch_log")
    _, reply = rig.call(token, "notebook_branch_log", branch="s2")
    assert reply["error"]["code"] == -32602
    _, reply = rig.call(token, "notebook_append_branch_log", kind="plan", content="x", branch="s2")
    assert reply["error"]["code"] == -32602


def test_notebook_search_and_record_work_through_http(rig: Rig) -> None:
    token = rig.token("s1")
    assert "r1" in rig.ok(token, "notebook_record", request_id="r1")
    assert rig.ok(token, "notebook_search", query="gemm").strip()


def test_notebook_errors_are_error_results(rig: Rig) -> None:
    assert "UnknownRequest" in rig.failed(rig.token("s1"), "notebook_record", request_id="nope")


def test_missing_wrong_and_revoked_tokens(rig: Rig) -> None:
    grant = McpSpecialistTokens(rig.server).issue("s1", frozenset({SCOPE_SPECIALIST}))
    assert rig.call(grant.token, "queue_state")[0] == 200
    assert rig.call(None, "queue_state")[0] == 401
    assert rig.call("wrong-token-0123456789", "queue_state")[0] == 401


def test_unknown_tool_over_http(rig: Rig) -> None:
    _, reply = rig.call(rig.token("s1"), "drop_database")
    assert reply["error"] == {"code": -32602, "message": "unknown tool"}


def test_tokens_are_per_branch_stable_across_issue_and_gone_after_revoke(rig: Rig) -> None:
    tokens = McpSpecialistTokens(rig.server)
    scopes = frozenset({SCOPE_SPECIALIST})
    grant = tokens.issue("s1", scopes)
    assert (grant.url, grant.token_env) == (rig.server.url, SPECIALIST_TOKEN_ENV)
    assert grant.token not in repr(grant)
    assert tokens.issue("s1", scopes) == grant  # a restarted specialist keeps its token
    other = tokens.issue("s2", scopes)
    assert other.token != grant.token
    tokens.revoke("s1")
    assert rig.call(grant.token, "queue_state")[0] == 401
    assert rig.call(other.token, "queue_state")[0] == 200
    fresh = tokens.issue("s1", scopes)
    assert fresh.token != grant.token and rig.call(fresh.token, "queue_state")[0] == 200
    tokens.revoke("s1")
    tokens.revoke("s1")  # revoking twice is harmless


def test_the_token_acts_for_the_branch_it_was_issued_for(rig: Rig) -> None:
    grant = McpSpecialistTokens(rig.server).issue("s1", frozenset({SCOPE_SPECIALIST}))
    rig.ok(grant.token, "submit_test", kind="test", purpose="explore", parent_commit="c1")
    assert rig.toolbox.submitted[-1].branch_id == "s1"


def test_a_token_cannot_be_issued_with_host_scope_or_no_scope(rig: Rig) -> None:
    for scopes in (frozenset({"host"}), frozenset(), frozenset({"specialist", "host"}), frozenset({"admin"})):
        with pytest.raises(ValueError, match="issued scopes"):
            McpSpecialistTokens(rig.server).issue("s1", scopes)


def test_jsonable_turns_contract_objects_into_json_data() -> None:
    state = QueueState(queued_by_priority={0: 1}, running_by_branch={"s1": 2})
    assert json.loads(json.dumps(jsonable(state))) == {
        "queued_by_priority": {"0": 1},
        "running_by_branch": {"s1": 2},
        "queued_by_branch": {},
        "cold_start_seconds": None,
    }


def test_the_grants_heartbeat_sink_beats_during_a_blocking_call_over_http(rig: Rig) -> None:
    server = McpServer(McpServerConfig(host_token=HOST_TOKEN), campaign="camp", heartbeat_seconds=0.1)
    server.register(
        "block",
        schema={"type": "object"},
        handler=lambda arguments, ctx: time.sleep(0.65) or {"done": True},
        scopes={SCOPE_SPECIALIST},
    )
    server.start()
    try:
        grant = McpSpecialistTokens(server).issue("s1", frozenset({SCOPE_SPECIALIST}))
        beats: list[int] = []
        grant.heartbeat_sink(lambda: beats.append(1))
        request = urllib.request.Request(
            server.url,
            json.dumps(
                {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "block", "arguments": {}}}
            ).encode(),
            {"X-Hyperloom-Token": grant.token},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.status == 200
        during = len(beats)
        assert during >= 2
        time.sleep(0.4)
        assert len(beats) == during
    finally:
        server.stop()


def _post_json(request: urllib.request.Request) -> Any:
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


async def test_toolbox_calls_run_on_the_loop_the_toolbox_was_registered_from() -> None:
    """The service answers on a loop of its own per request; a test the toolbox starts there would die with it."""
    home = asyncio.get_running_loop()
    seen: list[asyncio.AbstractEventLoop] = []

    class LoopRecordingToolbox(FakeToolbox):
        async def submit_test(self, request: TestRequest) -> str:
            seen.append(asyncio.get_running_loop())
            return await super().submit_test(request)

        async def cancel(self, request_id: str) -> CancelOutcome:
            seen.append(asyncio.get_running_loop())
            return await super().cancel(request_id)

        async def queue_state(self) -> QueueState:
            seen.append(asyncio.get_running_loop())
            return await super().queue_state()

    server = McpServer(McpServerConfig(host_token=HOST_TOKEN), campaign="camp")
    toolbox = LoopRecordingToolbox()

    class Notebooks:
        async def record(self, request_id: str) -> object:
            return SimpleNamespace(request=SimpleNamespace(branch="s1"))

    register_specialist_tools(server, toolbox=toolbox, notebooks=lambda branch: Notebooks())  # type: ignore[arg-type,return-value]
    server.start()
    try:
        token = server.issue_token("specialist-s1", "s1")
        for tool, arguments in [
            ("submit_test", {"kind": "test", "purpose": "explore", "parent_commit": "c1"}),
            ("cancel_test", {"request_id": "r1"}),
            ("queue_state", {}),
        ]:
            message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool, "arguments": arguments},
            }
            request = urllib.request.Request(server.url, json.dumps(message).encode(), {"X-Hyperloom-Token": token})
            reply = await asyncio.to_thread(_post_json, request)
            assert reply["result"]["isError"] is False, reply
    finally:
        server.stop()
    assert seen == [home, home, home]


async def _land(w: Writers, count: int, first: int = 1) -> None:
    """``count`` tests of ``s1`` measured one after another: one result in its inbox each, ``x<first>`` onwards."""
    for number in range(first, first + count):
        of = intent("s1", "c1", "attention", f"mechanism {number}")
        await w.measure("s1", f"x{number}", of, delta(f"--max-num-seqs {1000 + number}"))


def _texts(reply: Any) -> list[str]:
    return [block["text"] for block in reply["result"]["content"]]


def _news(rig: Rig, token: str, tool: str = "queue_state", **arguments: Any) -> list[str]:
    """The lines of the block a reply carries after the tool's own content; none when it has no block."""
    status, reply = rig.call(token, tool, **arguments)
    assert status == 200 and "result" in reply, reply
    texts = _texts(reply)
    assert len(texts) <= 2, texts
    return texts[1].splitlines() if len(texts) == 2 else []


def _item_lines(rig: Rig, token: str) -> list[str]:
    """What ``poll_inbox`` lists, as the lines a block carries them."""
    items = json.loads(rig.ok(token, "poll_inbox"))["items"]
    return [f"{item['kind']} {item['pointer']}: {item['summary']}" for item in items]


async def test_a_reply_carries_what_arrived_in_the_branch_inbox_since_the_last_reply(
    wait_rig: Rig, waiting: Writers
) -> None:
    token = wait_rig.token("s1")
    assert _news(wait_rig, token) == []
    await waiting.as_role(Role.TOOLBOX).write_result(result("r9"))
    status, reply = wait_rig.call(token, "queue_state")
    own, block = _texts(reply)
    assert status == 200 and json.loads(own)["queued"] == 3
    assert block.splitlines() == [REPLY_INBOX_HEADER, *_item_lines(wait_rig, token)[-1:]]
    assert block.splitlines()[1].startswith("result r9: ")
    assert _news(wait_rig, token) == []


async def test_a_branch_sees_its_own_inbox_only(wait_rig: Rig, waiting: Writers) -> None:
    other = wait_rig.token("s2")
    assert _news(wait_rig, other) == []
    await _land(waiting, 1)
    assert _news(wait_rig, other) == []


async def test_the_first_call_of_a_token_starts_at_the_end_of_the_feed(wait_rig: Rig, waiting: Writers) -> None:
    await _land(waiting, 1)
    assert (await wait_rig.observer().poll("s1")).items
    early = wait_rig.token("s1")
    assert _news(wait_rig, early) == []
    await _land(waiting, 1, first=2)
    assert _news(wait_rig, early)[1].startswith("result x2: ")
    await _land(waiting, 1, first=3)
    assert _news(wait_rig, wait_rig.token("s1")) == []
    assert _news(wait_rig, early)[1].startswith("result x3: ")


async def test_what_arrives_while_the_first_call_of_a_token_runs_is_news_for_its_reply(
    wait_rig: Rig, waiting: Writers
) -> None:
    call = asyncio.create_task(
        asyncio.to_thread(wait_rig.call, wait_rig.token("s1"), "wait_test", request_id="r9", wait_seconds=60)
    )
    await asyncio.sleep(0.3)
    await _land(waiting, 1)
    await waiting.as_role(Role.TOOLBOX).write_result(result("r9"))
    status, reply = await asyncio.wait_for(call, 10)
    own, block = _texts(reply)
    assert status == 200 and json.loads(own)["state"] == "measured"
    assert [line.split(":")[0] for line in block.splitlines()] == [REPLY_INBOX_HEADER[:-1], "result x1"]


async def test_a_block_lists_at_most_twenty_items_and_counts_the_rest(wait_rig: Rig, waiting: Writers) -> None:
    token = wait_rig.token("s1")
    _news(wait_rig, token)
    await _land(waiting, REPLY_INBOX_LINES + 5)
    lines = _news(wait_rig, token)
    assert len(lines) == 1 + REPLY_INBOX_LINES + 1
    assert lines[-1] == "... 5 more, poll_inbox lists them"
    assert [line.split(":")[0] for line in lines[1:-1]][-1] == f"result x{REPLY_INBOX_LINES}"
    assert _news(wait_rig, token) == []


async def test_a_block_of_exactly_twenty_items_has_no_more_line(wait_rig: Rig, waiting: Writers) -> None:
    token = wait_rig.token("s1")
    _news(wait_rig, token)
    await _land(waiting, REPLY_INBOX_LINES)
    lines = _news(wait_rig, token)
    assert len(lines) == 1 + REPLY_INBOX_LINES and not lines[-1].startswith("...")


async def test_an_error_reply_still_gets_the_block_and_moves_the_cursor(wait_rig: Rig, waiting: Writers) -> None:
    token = wait_rig.token("s1")
    _news(wait_rig, token)
    await _land(waiting, 1)
    status, reply = wait_rig.call(token, "test_result", request_id="nope")
    assert status == 200 and reply["result"]["isError"] is True
    assert "UnknownRequest" in _texts(reply)[0] and _texts(reply)[1].splitlines()[1].startswith("result x1: ")
    assert _news(wait_rig, token) == []


async def test_a_result_reply_leaves_out_the_item_that_names_the_request_it_carries(
    wait_rig: Rig, waiting: Writers
) -> None:
    token = wait_rig.token("s1")
    _news(wait_rig, token)
    await waiting.as_role(Role.TOOLBOX).write_result(result("r9"))
    await _land(waiting, 1)
    carried = _news(wait_rig, token, "wait_test", request_id="r9")
    assert [line.split(":")[0] for line in carried] == [REPLY_INBOX_HEADER[:-1], "result x1"]
    await _land(waiting, 1, first=2)
    assert [line.split(":")[0] for line in _news(wait_rig, token, "test_result", request_id="r9")][1:] == ["result x2"]


async def test_other_replies_list_the_item_of_a_request_a_result_reply_would_leave_out(
    wait_rig: Rig, waiting: Writers
) -> None:
    token = wait_rig.token("s1")
    _news(wait_rig, token)
    await waiting.as_role(Role.TOOLBOX).write_result(result("r9"))
    assert _news(wait_rig, token, "cancel_test", request_id="r9")[1].startswith("result r9: ")


async def test_poll_inbox_keeps_its_own_cursor_apart_from_the_reply_block(wait_rig: Rig, waiting: Writers) -> None:
    token = wait_rig.token("s1")
    _news(wait_rig, token)
    await _land(waiting, 1)
    assert [line.split(":")[0] for line in _news(wait_rig, token)][1:] == ["result x1"]
    first = json.loads(wait_rig.ok(token, "poll_inbox"))
    assert any(item["pointer"] == "x1" for item in first["items"])
    await _land(waiting, 1, first=2)
    again = json.loads(wait_rig.ok(token, "poll_inbox", cursor=first["cursor"]))
    assert [item["pointer"] for item in again["items"]] == ["x2"]


async def test_a_poll_inbox_reply_carries_no_block_and_its_items_count_as_seen(wait_rig: Rig, waiting: Writers) -> None:
    token = wait_rig.token("s1")
    _news(wait_rig, token)
    await _land(waiting, 1)
    status, reply = wait_rig.call(token, "poll_inbox")
    assert status == 200 and len(_texts(reply)) == 1
    assert _news(wait_rig, token) == []


async def test_the_host_token_gets_no_block_and_leaves_the_specialists_unread(wait_rig: Rig, waiting: Writers) -> None:
    wait_rig.server.register(
        "host_ping", schema={"type": "object"}, handler=lambda arguments, ctx: {"ok": True}, scopes={HOST_KIND}
    )
    token = wait_rig.token("s1")
    _news(wait_rig, token)
    await _land(waiting, 1)
    _, reply = wait_rig.call(HOST_TOKEN, "host_ping")
    assert len(_texts(reply)) == 1
    assert _news(wait_rig, token)[1].startswith("result x1: ")


def _pages(rig: Rig, token: str, cursor: str = "") -> list[dict[str, Any]]:
    """Every page ``poll_inbox`` gives from ``cursor`` until it says nothing remains, each page as the reply read."""
    pages: list[dict[str, Any]] = []
    while True:
        page = json.loads(rig.ok(token, "poll_inbox", cursor=cursor))
        pages.append(page)
        if not page["remaining"]:
            return pages
        assert page["more"] == f"{page['remaining']} more; call poll_inbox with this cursor"
        cursor = page["cursor"]


def _pointers(pages: list[dict[str, Any]]) -> list[str]:
    return [item["pointer"] for page in pages for item in page["items"]]


async def test_a_large_inbox_is_paged_with_no_item_lost_or_repeated(wait_rig: Rig, waiting: Writers) -> None:
    await _land(waiting, 3 * POLL_INBOX_MAX_ITEMS + 7)
    observer = await wait_rig.observer().poll("s1")
    assert len(observer.items) > 3 * POLL_INBOX_MAX_ITEMS
    pages = _pages(wait_rig, wait_rig.token("s1"))
    assert all(len(page["items"]) <= POLL_INBOX_MAX_ITEMS for page in pages)
    assert [len(page["items"]) for page in pages[:-1]] == [POLL_INBOX_MAX_ITEMS] * (len(pages) - 1)
    assert _pointers(pages) == [item.pointer for item in observer.items]
    assert pages[0]["remaining"] == len(observer.items) - POLL_INBOX_MAX_ITEMS
    assert pages[-1]["cursor"] == observer.cursor and "more" not in pages[-1]
    assert json.loads(wait_rig.ok(wait_rig.token("s1"), "poll_inbox", cursor=observer.cursor))["items"] == []


async def test_a_page_holds_at_most_the_byte_budget_cursor_included(
    wait_rig: Rig, waiting: Writers, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _land(waiting, 12)
    token = wait_rig.token("s1")
    whole = _pages(wait_rig, token)
    largest = max(len(json.dumps(item)) for page in whole for item in page["items"])
    budget = len(json.dumps(whole[-1]["cursor"])) + 2 * largest + 300
    monkeypatch.setattr(mcp_tools, "POLL_INBOX_MAX_BYTES", budget)
    texts: list[str] = []
    cursor = ""
    while True:
        texts.append(wait_rig.ok(token, "poll_inbox", cursor=cursor))
        page = json.loads(texts[-1])
        if not page["remaining"]:
            break
        cursor = page["cursor"]
    assert len(texts) >= 3 and all(len(text) <= budget for text in texts)
    assert _pointers([json.loads(text) for text in texts]) == _pointers(whole)


async def test_a_full_page_stays_under_the_byte_budget_that_keeps_it_clear_of_the_cli_limit(
    wait_rig: Rig, waiting: Writers
) -> None:
    await _land(waiting, 2 * POLL_INBOX_MAX_ITEMS)
    first = wait_rig.ok(wait_rig.token("s1"), "poll_inbox")
    assert len(json.loads(first)["items"]) == POLL_INBOX_MAX_ITEMS and len(first) < POLL_INBOX_MAX_BYTES


async def test_items_that_arrive_between_pages_come_after_the_rest_and_nothing_is_repeated(
    wait_rig: Rig, waiting: Writers
) -> None:
    await _land(waiting, 2 * POLL_INBOX_MAX_ITEMS)
    token = wait_rig.token("s1")
    first = json.loads(wait_rig.ok(token, "poll_inbox"))
    await _land(waiting, 5, first=2 * POLL_INBOX_MAX_ITEMS + 1)
    rest = _pages(wait_rig, token, first["cursor"])
    every = [item.pointer for item in (await wait_rig.observer().poll("s1")).items]
    assert [item["pointer"] for item in first["items"]] + _pointers(rest) == every


async def test_a_poll_inbox_page_marks_as_seen_only_the_items_it_returned(wait_rig: Rig, waiting: Writers) -> None:
    token = wait_rig.token("s1")
    _news(wait_rig, token)
    await _land(waiting, 25)
    every = (await wait_rig.observer().poll("s1")).items
    page = json.loads(wait_rig.ok(token, "poll_inbox"))
    returned = {item["pointer"] for item in page["items"]}
    unseen = [item for item in every[-25:] if item.pointer not in returned]
    assert 0 < len(unseen) <= REPLY_INBOX_LINES
    block = _news(wait_rig, token)
    assert block == [REPLY_INBOX_HEADER, *(f"{item.kind.value} {item.pointer}: {item.summary}" for item in unseen)]
    assert _news(wait_rig, token) == []


async def test_a_poll_inbox_that_returns_every_item_still_marks_all_as_seen(wait_rig: Rig, waiting: Writers) -> None:
    token = wait_rig.token("s1")
    _news(wait_rig, token)
    await _land(waiting, 3)
    assert json.loads(wait_rig.ok(token, "poll_inbox"))["remaining"] == 0
    assert _news(wait_rig, token) == []


async def test_a_cursor_in_the_middle_of_a_page_is_refused_when_it_is_not_one(wait_rig: Rig) -> None:
    _, reply = wait_rig.call(wait_rig.token("s1"), "poll_inbox", cursor="page:x:")
    assert reply["error"]["code"] == -32602
