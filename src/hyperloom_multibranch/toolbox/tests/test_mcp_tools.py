# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The specialist tools on a running MCP service: real HTTP requests, the local notebook behind them."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from hyperloom.contracts.measurement import TestRequest
from hyperloom.hosted.mcp_server import McpServer, McpServerConfig
from hyperloom_multibranch.contracts.notebook import Role, Writer
from hyperloom_multibranch.contracts.toolbox import Backpressure, CancelOutcome, QueueState, RequestRejected
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import FakeClock
from hyperloom_multibranch.notebook.tests.sessions import scripted_session
from hyperloom_multibranch.toolbox.mcp_tools import (
    SPECIALIST_TOKEN_ENV,
    grant_access,
    jsonable,
    register_specialist_tools,
)

HOST_TOKEN = "host-token-0123456789abcdef"
CLAIM = {"parent_commit": "c1", "target": "attention", "mechanism": "flash decode", "layer": "source"}


class FakeToolbox:
    def __init__(self) -> None:
        self.submitted: list[TestRequest] = []
        self.cancelled: list[str] = []
        self.refuse: Exception | None = None

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


@pytest.fixture
def rig(seeded: tuple[Path, FakeClock]) -> Iterator[Rig]:
    root, clock = seeded
    toolbox = FakeToolbox()
    server = McpServer(McpServerConfig(host_token=HOST_TOKEN), campaign="camp")

    def notebooks(branch: str) -> LocalNotebook:
        return LocalNotebook(root, Writer(Role.SPECIALIST_RUNTIME, branch, branch), clock=clock)

    register_specialist_tools(server, toolbox=toolbox, notebooks=notebooks, claim_ttl_seconds=60.0)
    server.start()
    yield Rig(server, toolbox, root, clock)
    server.stop()


def test_a_specialist_sees_exactly_the_specialist_tools(rig: Rig) -> None:
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    _, reply = rig.post(rig.token("s1"), message)
    names = sorted(tool["name"] for tool in reply["result"]["tools"])
    assert names == sorted(
        ["submit_test", "test_result", "cancel_test", "queue_state", "poll_inbox"]
        + [f"notebook_{name}" for name in ("lookup", "claim", "release", "search", "record", "branch_log")]
        + ["notebook_append_branch_log", "notebook_brief"]
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
    assert body["result"]["request_id"] == "r1" and body["result"]["validity"] == "success"


def test_test_result_of_another_branch_is_forbidden(rig: Rig) -> None:
    _, reply = rig.call(rig.token("s1"), "test_result", request_id="r2")
    assert reply["error"] == {"code": -32001, "message": "a specialist reads the tests of its own branch only"}


def test_test_result_of_an_unknown_request_is_an_error_result(rig: Rig) -> None:
    assert "UnknownRequest" in rig.failed(rig.token("s1"), "test_result", request_id="nope")


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
    assert set(body) == {"items", "cursor"}


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
    access = grant_access(rig.server, specialist="spec-1", branch="s1")
    assert rig.call(access.token, "queue_state")[0] == 200
    assert rig.call(None, "queue_state")[0] == 401
    assert rig.call("wrong-token-0123456789", "queue_state")[0] == 401
    rig.server.revoke_specialist("spec-1")
    assert rig.call(access.token, "queue_state")[0] == 401


def test_unknown_tool_over_http(rig: Rig) -> None:
    _, reply = rig.call(rig.token("s1"), "drop_database")
    assert reply["error"] == {"code": -32602, "message": "unknown tool"}


def test_grant_access_names_the_env_var_and_hides_the_token(rig: Rig) -> None:
    access = grant_access(rig.server, specialist="spec-1", branch="s1")
    assert access.url == rig.server.url and access.token_env == SPECIALIST_TOKEN_ENV
    assert access.token not in repr(access)


def test_jsonable_turns_contract_objects_into_json_data() -> None:
    state = QueueState(queued_by_priority={0: 1}, running_by_branch={"s1": 2})
    assert json.loads(json.dumps(jsonable(state))) == {
        "queued_by_priority": {"0": 1},
        "running_by_branch": {"s1": 2},
        "queued_by_branch": {},
        "cold_start_seconds": None,
    }

