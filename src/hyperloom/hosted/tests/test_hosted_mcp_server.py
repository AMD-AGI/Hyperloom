# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The MCP service core over real HTTP: every request goes through a socket to a running server."""

from __future__ import annotations

import dataclasses
import http.client
import json
import logging
import select
import socket
import struct
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from hyperloom.hosted.host_info import Var, take_host_info
from hyperloom.hosted.mcp_server import (
    HOST_KIND,
    CallContext,
    McpServer,
    McpServerConfig,
    ReplyCall,
    ReplyNote,
    ToolError,
    ToolForbidden,
)

HOST_TOKEN = "host-token-0123456789abcdef"


class Client:
    def __init__(self, server: McpServer) -> None:
        self.port = server.port

    def send(
        self, method: str, path: str = "/mcp", body: Any = None, headers: dict[str, str] | None = None
    ) -> tuple[int, dict[str, str], Any]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        raw = body if isinstance(body, bytes) else (None if body is None else json.dumps(body).encode())
        connection.request(method, path, body=raw, headers=headers or {})
        response = connection.getresponse()
        data = response.read()
        connection.close()
        reply = json.loads(data) if data else None
        return response.status, {k.lower(): v for k, v in response.getheaders()}, reply

    def rpc(self, token: str | None, method: str, params: dict[str, Any] | None = None, *, bearer: bool = False):
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization" if bearer else "X-Hyperloom-Token"] = f"Bearer {token}" if bearer else token
        message = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        return self.send("POST", body=message, headers=headers)

    def call(self, token: str | None, tool: str, arguments: dict[str, Any] | None = None):
        return self.rpc(token, "tools/call", {"name": tool, "arguments": arguments or {}})


def _make_server(campaign: str = "camp-a") -> McpServer:
    server = McpServer(McpServerConfig(host_token=HOST_TOKEN), campaign=campaign)
    server.register(
        "whoami",
        description="Who is calling",
        schema={"type": "object", "properties": {}},
        handler=lambda args, ctx: {"kind": ctx.principal.kind, "branch": ctx.principal.branch},
        scopes={"host", "specialist"},
    )
    server.register(
        "echo",
        schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        handler=lambda args, ctx: {"text": args["text"]},
        scopes={"specialist"},
    )
    server.register("stop", schema={"type": "object"}, handler=lambda args, ctx: {"ok": True}, scopes={"host"})
    return server


@pytest.fixture
def server() -> Iterator[McpServer]:
    running = _make_server()
    running.start()
    yield running
    running.stop()


def _text(reply: Any) -> Any:
    return json.loads(reply["result"]["content"][0]["text"])


def test_valid_specialist_token_calls_a_specialist_tool(server: McpServer) -> None:
    token = server.issue_token("spec-1", "branch-7")
    status, _, reply = Client(server).call(token, "whoami")
    assert status == 200
    assert _text(reply) == {"kind": "specialist", "branch": "branch-7"}


def test_valid_host_token_via_bearer_header(server: McpServer) -> None:
    status, _, reply = Client(server).rpc(HOST_TOKEN, "tools/call", {"name": "stop"}, bearer=True)
    assert status == 200 and _text(reply) == {"ok": True}


def test_missing_and_wrong_token_get_401(server: McpServer) -> None:
    client = Client(server)
    for token in (None, "", "not-a-token-at-all-0000", HOST_TOKEN + "x"):
        status, headers, reply = client.call(token, "whoami")
        assert status == 401
        assert headers["www-authenticate"] == "Bearer"
        assert reply == {"error": "unauthorized"}


def test_wrong_scope_is_refused_and_the_handler_never_runs(server: McpServer) -> None:
    token = server.issue_token("spec-1", "b")
    status, _, reply = Client(server).call(token, "stop")
    assert status == 200
    assert reply["error"]["code"] == -32001
    assert "result" not in reply


def test_host_cannot_call_a_specialist_only_tool(server: McpServer) -> None:
    _, _, reply = Client(server).call(HOST_TOKEN, "echo", {"text": "x"})
    assert reply["error"]["code"] == -32001


def test_unknown_tool_is_invalid_params_not_forbidden(server: McpServer) -> None:
    _, _, reply = Client(server).call(server.issue_token("s", "b"), "no_such_tool")
    assert reply["error"] == {"code": -32602, "message": "unknown tool"}


def test_tools_list_shows_only_the_tools_of_the_token_kind(server: McpServer) -> None:
    client = Client(server)
    _, _, spec = client.rpc(server.issue_token("s", "b"), "tools/list")
    _, _, host = client.rpc(HOST_TOKEN, "tools/list")
    assert sorted(t["name"] for t in spec["result"]["tools"]) == ["echo", "whoami"]
    assert sorted(t["name"] for t in host["result"]["tools"]) == ["stop", "whoami"]


def test_arguments_are_validated_against_the_schema(server: McpServer) -> None:
    _, _, reply = Client(server).call(server.issue_token("s", "b"), "echo", {"text": 5})
    assert reply["error"]["code"] == -32602


def test_revoked_token_is_refused_afterwards(server: McpServer) -> None:
    client = Client(server)
    token = server.issue_token("spec-1", "b")
    assert client.call(token, "whoami")[0] == 200
    assert server.revoke_specialist("spec-1") == 1
    assert client.call(token, "whoami")[0] == 401
    assert server.revoke_specialist("spec-1") == 0


def test_revoke_specialist_kills_all_its_tokens_and_only_its_own(server: McpServer) -> None:
    client = Client(server)
    first, second = server.issue_token("spec-1", "b"), server.issue_token("spec-1", "b")
    other = server.issue_token("spec-2", "c")
    assert server.revoke_specialist("spec-1") == 2
    assert client.call(first, "whoami")[0] == 401 and client.call(second, "whoami")[0] == 401
    assert client.call(other, "whoami")[0] == 200


def test_token_of_another_campaign_is_refused() -> None:
    first, second = _make_server("camp-a"), _make_server("camp-b")
    first.start()
    second.start()
    try:
        token = first.issue_token("spec-1", "b")
        assert Client(first).call(token, "whoami")[0] == 200
        assert Client(second).call(token, "whoami")[0] == 401
    finally:
        first.stop()
        second.stop()


def test_token_whose_principal_names_another_campaign_is_refused(server: McpServer) -> None:
    from hyperloom.hosted.mcp_server import McpPrincipal

    token = server._tokens.issue(McpPrincipal("specialist", "s", "b", campaign="camp-other"))
    assert Client(server).call(token, "whoami")[0] == 401


def test_get_is_405_and_other_paths_404(server: McpServer) -> None:
    client = Client(server)
    status, headers, _ = client.send("GET")
    assert status == 405 and headers["allow"] == "POST"
    assert client.send("POST", path="/other", body={}, headers={"X-Hyperloom-Token": HOST_TOKEN})[0] == 404


def test_initialize_returns_a_session_id_header(server: McpServer) -> None:
    status, headers, reply = Client(server).rpc(HOST_TOKEN, "initialize", {"protocolVersion": "2025-03-26"})
    assert status == 200
    assert headers["mcp-session-id"]
    assert reply["result"]["protocolVersion"] == "2025-03-26"
    assert "mcp-session-id" not in Client(server).rpc(HOST_TOKEN, "ping")[1]


def test_session_id_is_not_enforced(server: McpServer) -> None:
    message = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    headers = {"X-Hyperloom-Token": HOST_TOKEN, "Mcp-Session-Id": "bogus"}
    assert Client(server).send("POST", body=message, headers=headers)[0] == 200


def test_batch_and_notification(server: McpServer) -> None:
    headers = {"X-Hyperloom-Token": HOST_TOKEN}
    batch = [
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    status, _, reply = Client(server).send("POST", body=batch, headers=headers)
    assert status == 200 and [item["id"] for item in reply] == [1, 2]
    note = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    assert Client(server).send("POST", body=note, headers=headers)[:1] == (202,)


def test_malformed_bodies(server: McpServer) -> None:
    client, headers = Client(server), {"X-Hyperloom-Token": HOST_TOKEN}
    status, _, reply = client.send("POST", body=b"{not json", headers=headers)
    assert status == 400 and reply["error"]["code"] == -32700
    _, _, reply = client.send("POST", body=[], headers=headers)
    assert reply["error"]["code"] == -32600
    _, _, reply = client.send("POST", body={"id": 1, "method": "ping"}, headers=headers)
    assert reply["error"]["code"] == -32600
    _, _, reply = client.rpc(HOST_TOKEN, "no/such")
    assert reply["error"]["code"] == -32601


def test_oversized_body_is_413_before_it_is_read(server: McpServer) -> None:
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    connection.putrequest("POST", "/mcp")
    connection.putheader("X-Hyperloom-Token", HOST_TOKEN)
    connection.putheader("Content-Length", str((1 << 20) + 1))
    connection.endheaders()
    response = connection.getresponse()
    connection.close()
    assert response.status == 413


def test_handler_receives_the_context_and_can_forbid(server: McpServer) -> None:
    seen: list[CallContext] = []

    def handler(args: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        seen.append(ctx)
        raise ToolForbidden("not your branch")

    server.register("guarded", schema={"type": "object"}, handler=handler, scopes={"specialist"})
    _, _, reply = Client(server).call(server.issue_token("spec-9", "branch-3"), "guarded")
    assert reply["error"] == {"code": -32001, "message": "not your branch"}
    assert (seen[0].principal.subject, seen[0].principal.branch, seen[0].tool) == ("spec-9", "branch-3", "guarded")


async def _async_handler(args: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
    return {"async": True}


def test_async_handlers_are_awaited(server: McpServer) -> None:
    server.register("later", schema={"type": "object"}, handler=_async_handler, scopes={"host"})
    assert _text(Client(server).call(HOST_TOKEN, "later")[2]) == {"async": True}


def test_error_messages_and_logs_never_carry_a_token(server: McpServer, caplog: pytest.LogCaptureFixture) -> None:
    token = server.issue_token("spec-1", "b")

    def leaky(args: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        raise RuntimeError(f"upstream said {token} and {HOST_TOKEN}")

    server.register("leaky", schema={"type": "object"}, handler=leaky, scopes={"specialist"})
    with caplog.at_level(logging.DEBUG):
        client = Client(server)
        _, _, reply = client.call(token, "leaky")
        client.call("wrong-" + token, "whoami")
        client.rpc(token, "tools/call", {"name": token})
    assert reply["result"]["isError"] is True
    wire = json.dumps(reply)
    assert token not in wire and HOST_TOKEN not in wire and "[redacted]" in wire
    assert token not in caplog.text and HOST_TOKEN not in caplog.text


def test_tokens_are_stored_as_digests_only(server: McpServer) -> None:
    token = server.issue_token("spec-1", "b")
    assert all(token.encode() != key for key in server._tokens._by_digest)
    assert token not in repr(server.__dict__.get("_tokens").__dict__)


def test_config_built_from_the_host_info_the_hosted_entry_takes(tmp_path) -> None:
    (tmp_path / "campaign.json").write_text("{}")
    environ = {
        Var.INPUT_FILE: str(tmp_path / "campaign.json"),
        Var.RUN_ID: "run-1",
        Var.RUN_CREATED_AT: "2026-10-01T12:30:00Z",
        Var.STATE_DIR: str(tmp_path),
        Var.MCP_PORT: "8765",
        Var.MCP_TOKEN: HOST_TOKEN,
    }
    info = take_host_info(environ)
    assert Var.MCP_TOKEN not in environ
    server = McpServer(McpServerConfig(port=info.mcp_port, host_token=info.mcp_token.reveal()), campaign="c")
    assert server.authenticate(HOST_TOKEN).kind == HOST_KIND


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="host token"):
        McpServer(McpServerConfig(host_token="short"), campaign="c")
    with pytest.raises(ValueError, match="campaign"):
        McpServer(McpServerConfig(host_token=HOST_TOKEN), campaign="")


def test_register_rejects_bad_scope_and_duplicate(server: McpServer) -> None:
    with pytest.raises(ValueError, match="scopes"):
        server.register("x", schema={"type": "object"}, handler=lambda a, c: {}, scopes={"admin"})
    with pytest.raises(ValueError, match="already registered"):
        server.register("echo", schema={"type": "object"}, handler=lambda a, c: {}, scopes={"host"})


def test_string_results_pass_through_and_tool_errors_are_error_results(server: McpServer) -> None:
    from hyperloom.hosted.mcp_server import ToolError

    def failing(args: dict[str, Any], ctx: CallContext) -> str:
        raise ToolError("refused: held by someone")

    server.register("plain", schema={"type": "object"}, handler=lambda a, c: "hello", scopes={"host"})
    server.register("bad", schema={"type": "object"}, handler=failing, scopes={"host"})
    client = Client(server)
    assert client.call(HOST_TOKEN, "plain")[2]["result"] == {
        "content": [{"type": "text", "text": "hello"}],
        "isError": False,
    }
    assert client.call(HOST_TOKEN, "bad")[2]["result"] == {
        "content": [{"type": "text", "text": "refused: held by someone"}],
        "isError": True,
    }


def test_a_result_that_cannot_be_serialized_is_an_internal_error_with_its_stack(
    server: McpServer, caplog: pytest.LogCaptureFixture
) -> None:
    def cyclic(args: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        result: dict[str, Any] = {}
        result["self"] = result
        return result

    server.register("cyclic", schema={"type": "object"}, handler=cyclic, scopes={"host"})
    with caplog.at_level(logging.WARNING, logger="hyperloom.hosted.mcp_server"):
        status, _, reply = Client(server).call(HOST_TOKEN, "cyclic")

    assert (status, reply) == (
        200,
        {"jsonrpc": "2.0", "id": 1, "error": {"code": -32603, "message": "ValueError: Circular reference detected"}},
    )
    (record,) = [r for r in caplog.records if r.getMessage().startswith("request failed: ValueError\n")]
    assert ", in _call_tool\n" in record.getMessage()


def _server_with_beat(beats: list[str], seconds: float = 0.1) -> McpServer:
    server = McpServer(McpServerConfig(host_token=HOST_TOKEN), campaign="camp", heartbeat_seconds=seconds)

    def slow(args: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        time.sleep(args.get("seconds", 0.0))
        return {"ok": True}

    schema = {"type": "object", "properties": {"seconds": {"type": "number"}}}
    server.register("slow", schema=schema, handler=slow, scopes={"host", "specialist"})
    return server


def test_beat_fires_during_a_blocking_call_and_stops_after_it() -> None:
    beats: list[str] = []
    server = _server_with_beat(beats)
    server.start()
    try:
        token = server.issue_token("spec-1", "branch-7")
        server.attach_heartbeat("spec-1", lambda: beats.append("beat"))
        assert Client(server).call(token, "slow", {"seconds": 0.75})[0] == 200
        during = len(beats)
        assert during >= 4
        time.sleep(0.4)
        assert len(beats) == during
    finally:
        server.stop()


def test_beat_belongs_to_one_specialist_and_not_to_the_host() -> None:
    beats: list[str] = []
    server = _server_with_beat(beats)
    server.start()
    try:
        server.attach_heartbeat("spec-1", lambda: beats.append("beat"))
        other = server.issue_token("spec-2", "b")
        assert Client(server).call(other, "slow", {"seconds": 0.5})[0] == 200
        assert Client(server).call(HOST_TOKEN, "slow", {"seconds": 0.5})[0] == 200
        assert Client(server).call(server.issue_token("spec-1", "b"), "slow", {"seconds": 0.0})[0] == 200
        assert beats == []
    finally:
        server.stop()


def test_revoking_the_specialist_drops_its_beat() -> None:
    beats: list[str] = []
    server = _server_with_beat(beats)
    server.start()
    try:
        server.attach_heartbeat("spec-1", lambda: beats.append("beat"))
        server.revoke_specialist("spec-1")
        token = server.issue_token("spec-1", "b")
        assert Client(server).call(token, "slow", {"seconds": 0.5})[0] == 200
        assert beats == []
    finally:
        server.stop()


def test_a_failing_beat_does_not_break_the_call() -> None:
    def broken() -> None:
        raise OSError("beat down")

    server = McpServer(McpServerConfig(host_token=HOST_TOKEN), campaign="camp", heartbeat_seconds=0.05)
    server.register(
        "slow", schema={"type": "object"}, handler=lambda a, c: time.sleep(0.3) or {"ok": 1}, scopes={"specialist"}
    )
    server.attach_heartbeat("s", broken)
    server.start()
    try:
        status, _, reply = Client(server).call(server.issue_token("s", "b"), "slow")
        assert status == 200 and reply["result"]["isError"] is False
    finally:
        server.stop()


def test_issued_scopes_decide_which_tools_open(server: McpServer) -> None:
    with pytest.raises(ValueError, match="issued scopes"):
        server.issue_token("s", "b", scopes=("host",))
    with pytest.raises(ValueError, match="issued scopes"):
        server.issue_token("s", "b", scopes=())


class Feed:
    """A fake inbox for a reply annotator: it lists what was appended after its cursor, the count of items read."""

    def __init__(self) -> None:
        self.items: list[str] = []
        self.seen: list[tuple[str, str | None]] = []
        self.broken = False

    def annotate(self, call: ReplyCall, cursor: str | None) -> ReplyNote:
        self.seen.append((call.tool, cursor))
        if self.broken:
            raise OSError("inbox down")
        return ReplyNote("\n".join(self.items[int(cursor or 0) :]), str(len(self.items)))


def _refuse(args: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
    raise ToolError("held by someone")


@pytest.fixture
def feed() -> Feed:
    return Feed()


@pytest.fixture
def noted(feed: Feed) -> Iterator[McpServer]:
    server = _make_server()
    server.register("refuse", schema={"type": "object"}, handler=_refuse, scopes={"specialist"})
    server.register(
        "arrive",
        schema={"type": "object"},
        handler=lambda args, ctx: feed.items.append("during") or {"ok": True},
        scopes={"specialist"},
    )
    server.attach_reply_annotator(feed.annotate)
    server.start()
    yield server
    server.stop()


def _blocks(reply: Any) -> list[str]:
    return [block["text"] for block in reply["result"]["content"]]


def _primed(server: McpServer, specialist: str = "spec-1") -> str:
    """A token that has made its first call, so its cursor is at the end of the feed."""
    token = server.issue_token(specialist, "branch-7")
    assert len(_blocks(Client(server).call(token, "whoami")[2])) == 1
    return token


def test_a_specialist_reply_carries_the_annotators_block_after_the_tools_own_content(
    noted: McpServer, feed: Feed
) -> None:
    token = _primed(noted)
    feed.items += ["one", "two"]
    _, _, reply = Client(noted).call(token, "whoami")
    assert _blocks(reply) == ['{"kind": "specialist", "branch": "branch-7"}', "one\ntwo"]
    assert len(_blocks(Client(noted).call(token, "whoami")[2])) == 1


def test_the_host_gets_no_annotation_and_the_annotator_is_never_asked(noted: McpServer, feed: Feed) -> None:
    token = _primed(noted)
    feed.items.append("one")
    asked = list(feed.seen)
    _, _, reply = Client(noted).call(HOST_TOKEN, "whoami")
    assert _blocks(reply) == ['{"kind": "host", "branch": ""}'] and feed.seen == asked
    assert _blocks(Client(noted).call(token, "whoami")[2])[1:] == ["one"]


def test_a_tokens_first_call_asks_where_the_feed_ends_before_the_tool_runs_and_drops_the_text(
    noted: McpServer, feed: Feed
) -> None:
    feed.items.append("history")
    token = noted.issue_token("spec-1", "b")
    assert _blocks(Client(noted).call(token, "arrive")[2])[1:] == ["during"]
    assert feed.seen == [("arrive", None), ("arrive", "1")]
    assert len(_blocks(Client(noted).call(token, "whoami")[2])) == 1


def test_each_token_has_a_cursor_of_its_own_that_dies_with_the_token(noted: McpServer, feed: Feed) -> None:
    client, first, second = Client(noted), _primed(noted, "spec-1"), _primed(noted, "spec-2")
    feed.items.append("one")
    assert _blocks(client.call(first, "whoami")[2])[1:] == ["one"]
    assert _blocks(client.call(second, "whoami")[2])[1:] == ["one"]
    assert _blocks(client.call(first, "whoami")[2])[1:] == []
    noted.revoke_specialist("spec-1")
    client.call(noted.issue_token("spec-1", "b"), "whoami")
    assert feed.seen[-2:] == [("whoami", None), ("whoami", "1")]


def test_a_failed_tool_call_still_gets_the_block_and_moves_the_cursor(noted: McpServer, feed: Feed) -> None:
    client, token = Client(noted), _primed(noted)
    feed.items.append("one")
    _, _, reply = client.call(token, "refuse")
    assert reply["result"]["isError"] is True and _blocks(reply) == ["held by someone", "one"]
    assert len(_blocks(client.call(token, "whoami")[2])) == 1


def test_the_cursor_is_kept_before_the_reply_is_written_so_the_next_call_cannot_outrun_it(
    noted: McpServer, feed: Feed
) -> None:
    token = noted.issue_token("spec-1", "branch-7")
    on_the_wire: list[bool] = []
    answer = noted.handle_post
    wire = socket.create_connection(("127.0.0.1", noted.port), timeout=10)

    def watch(principal: Any, presented: str, body: bytes) -> Any:
        reply = answer(principal, presented, body)

        def keep() -> None:
            on_the_wire.append(bool(select.select([wire], [], [], 0)[0]))
            reply.keep()

        return dataclasses.replace(reply, keep=keep)

    noted.handle_post = watch  # type: ignore[method-assign]
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "whoami"}}).encode()
    head = f"POST /mcp HTTP/1.1\r\nHost: x\r\nX-Hyperloom-Token: {token}\r\nContent-Length: {len(body)}\r\n\r\n"
    wire.sendall(head.encode() + body)
    assert wire.recv(1, socket.MSG_PEEK)
    wire.close()
    assert on_the_wire == [False]


def test_a_refused_call_has_no_result_to_carry_the_block_and_leaves_the_cursor(noted: McpServer, feed: Feed) -> None:
    client, token = Client(noted), _primed(noted)
    feed.items.append("one")
    asked = list(feed.seen)
    assert client.call(token, "echo")[2]["error"]["code"] == -32602
    assert feed.seen == asked
    assert _blocks(client.call(token, "whoami")[2])[1:] == ["one"]


def test_the_cursor_moves_only_once_the_reply_was_sent(noted: McpServer, feed: Feed) -> None:
    started, release = threading.Event(), threading.Event()

    def hold(args: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        started.set()
        release.wait(10)
        return {"ok": True}

    noted.register("hold", schema={"type": "object"}, handler=hold, scopes={"specialist"})
    token = _primed(noted)
    feed.items.append("one")
    finished = threading.Event()
    answer = noted.handle_post

    def watch(principal: Any, presented: str, body: bytes) -> Any:
        reply = answer(principal, presented, body)

        def drop() -> None:
            reply.drop()
            finished.set()

        return dataclasses.replace(reply, drop=drop)

    noted.handle_post = watch  # type: ignore[method-assign]
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "hold"}}).encode()
    head = f"POST /mcp HTTP/1.1\r\nHost: x\r\nX-Hyperloom-Token: {token}\r\nContent-Length: {len(body)}\r\n\r\n"
    gone = socket.create_connection(("127.0.0.1", noted.port), timeout=10)
    gone.sendall(head.encode() + body)
    assert started.wait(10)
    gone.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    gone.close()
    release.set()
    assert finished.wait(10)
    assert ("hold", "0") in feed.seen
    assert _blocks(Client(noted).call(token, "whoami")[2])[1:] == ["one"]
    assert len(_blocks(Client(noted).call(token, "whoami")[2])) == 1


def test_the_calls_of_one_batch_share_the_cursor_and_it_moves_when_the_batch_is_sent(
    noted: McpServer, feed: Feed
) -> None:
    client, token = Client(noted), _primed(noted)
    feed.items.append("one")
    batch = [{"jsonrpc": "2.0", "id": index, "method": "tools/call", "params": {"name": "whoami"}} for index in (1, 2)]
    _, _, replies = client.send("POST", body=batch, headers={"X-Hyperloom-Token": token})
    assert [_blocks(reply)[1:] for reply in replies] == [["one"], []]
    assert len(_blocks(client.call(token, "whoami")[2])) == 1


def test_a_failing_annotator_costs_the_reply_its_block_and_nothing_more(noted: McpServer, feed: Feed) -> None:
    client, token = Client(noted), _primed(noted)
    feed.items.append("one")
    feed.broken = True
    status, _, reply = client.call(token, "whoami")
    assert status == 200 and reply["result"]["isError"] is False and len(_blocks(reply)) == 1
    feed.broken = False
    assert _blocks(client.call(token, "whoami")[2])[1:] == ["one"]


def test_an_annotator_bug_is_answered_as_an_internal_error(noted: McpServer) -> None:
    client, token = Client(noted), _primed(noted)
    noted.attach_reply_annotator(lambda call, cursor: {}[call.tool])
    status, _, reply = client.call(token, "whoami")
    assert (status, reply) == (
        200,
        {"jsonrpc": "2.0", "id": 1, "error": {"code": -32603, "message": "KeyError: 'whoami'"}},
    )
