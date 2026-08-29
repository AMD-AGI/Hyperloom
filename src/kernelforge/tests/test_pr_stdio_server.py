"""Tests for the standalone PR Monitor stdio MCP server."""

from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys

import pytest

from kernelforge.mcp_server import pr_stdio_server as server

from kernelforge.conftest import SRC_ROOT


def _call(method, params=None):
    """Dispatch one in-process MCP request."""
    return asyncio.run(server._dispatch(method, params or {}))


def _tool(name, arguments):
    """Call one PR tool and decode its text payload."""
    reply = asyncio.run(server.handle_tool_call(name, arguments))
    return json.loads(reply["content"][0]["text"])


def test_tool_names_avoid_colliding_with_the_upstream_mcp_server():
    """Upstream exposes pr_search / pr_distill / pr_file_patch."""
    names = set(server.TOOL_NAMES)

    assert names.isdisjoint({"pr_search", "pr_distill", "pr_file_patch"})
    assert names == {"pr_find_references", "pr_get_reference", "pr_get_file_patch"}


def test_declared_names_match_the_schemas():
    assert [d["name"] for d in server.TOOL_DEFINITIONS] == list(server.TOOL_NAMES)
    for definition in server.TOOL_DEFINITIONS:
        assert definition["description"].strip()
        assert definition["inputSchema"]["type"] == "object"


def test_initialize_reports_the_server_identity():
    result = _call("initialize", {"protocolVersion": "2024-11-05"})

    assert result["serverInfo"]["name"] == server.SERVER_NAME
    assert result["protocolVersion"] == "2024-11-05"


def test_tools_list_returns_exactly_three_tools():
    assert len(_call("tools/list")["tools"]) == 3


def test_ping_and_lifecycle_methods_are_accepted():
    assert _call("ping") == {}
    assert _call("shutdown") == {}
    assert _call("resources/list") == {"resources": []}
    assert _call("prompts/list") == {"prompts": []}


def test_unsupported_method_raises():
    with pytest.raises(NotImplementedError, match="unsupported MCP method"):
        _call("does/not/exist")


def test_unknown_tool_raises():
    with pytest.raises(ValueError, match="unknown tool"):
        asyncio.run(server.handle_tool_call("nope", {}))


@pytest.mark.parametrize("arguments", [[], ["a"], "text", 0])
def test_non_object_arguments_are_rejected(arguments):
    """Reject every non-object arguments value."""
    with pytest.raises(ValueError, match="must be an object"):
        _call("tools/call", {"name": "pr_get_reference", "arguments": arguments})


def test_omitted_arguments_default_to_an_empty_object(monkeypatch):
    monkeypatch.setenv("PR_KB_REPO", "ROCm/aiter")
    monkeypatch.setattr(server, "_client", lambda: pytest.fail("must not reach the network"))

    reply = _call("tools/call", {"name": "pr_find_references"})

    assert json.loads(reply["content"][0]["text"])["results"] == []


def test_repo_defaults_to_the_campaign_environment(monkeypatch):
    monkeypatch.setenv("PR_KB_REPO", "ROCm/FlyDSL")

    assert server._resolve_repo({}) == "ROCm/FlyDSL"


def test_explicit_repo_overrides_the_default(monkeypatch):
    monkeypatch.setenv("PR_KB_REPO", "ROCm/FlyDSL")

    assert server._resolve_repo({"repo": "ROCm/aiter"}) == "ROCm/aiter"


def test_missing_repo_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("PR_KB_REPO", raising=False)

    with pytest.raises(ValueError, match="no repo configured"):
        server._resolve_repo({})


@pytest.mark.parametrize("bad", ["justname", "a/b/c", "/", "  "])
def test_malformed_repo_is_rejected(monkeypatch, bad):
    monkeypatch.delenv("PR_KB_REPO", raising=False)

    with pytest.raises(ValueError):
        server._resolve_repo({"repo": bad})


def test_find_references_without_a_query_does_not_call_the_service(monkeypatch):
    monkeypatch.setenv("PR_KB_REPO", "ROCm/aiter")
    monkeypatch.setattr(server, "_client", lambda: pytest.fail("must not reach the network"))

    result = _tool("pr_find_references", {})

    assert result["results"] == []
    assert "no file_path or keywords" in result["reason"]


def test_find_references_returns_ranked_results(monkeypatch):
    from kernelforge.knowledge.pr_monitor_search import PRReference, SearchOutcome

    monkeypatch.setenv("PR_KB_REPO", "ROCm/FlyDSL")
    monkeypatch.setattr(server, "_client", object)
    monkeypatch.setattr(
        server,
        "discover",
        lambda client, context, **kw: SearchOutcome(
            references=(
                PRReference(
                    repo="ROCm/FlyDSL",
                    number=959,
                    title="t",
                    is_merged=True,
                    worth_trying=0.6,
                    components=("moe",),
                    n_files=3,
                ),
            ),
            stats={"degraded_reason": "service_unreachable"},
        ),
    )

    result = _tool("pr_find_references", {"keywords": ["moe gemm"]})

    assert result["results"][0]["number"] == 959
    assert result["results"][0]["state"] == "merged"
    assert result["results"][0]["worth_trying"] == 0.6
    assert result["degraded_reason"] == "service_unreachable"


def test_find_references_accepts_a_bare_string_keyword(monkeypatch):
    captured = {}

    monkeypatch.setenv("PR_KB_REPO", "ROCm/FlyDSL")
    monkeypatch.setattr(server, "_client", object)

    def fake_discover(client, context, **kwargs):
        from kernelforge.knowledge.pr_monitor_search import SearchOutcome

        captured["keywords"] = context.keywords
        return SearchOutcome()

    monkeypatch.setattr(server, "discover", fake_discover)
    _tool("pr_find_references", {"keywords": "moe"})

    assert captured["keywords"] == ("moe",)


def test_get_reference_reports_a_missing_pr(monkeypatch):
    monkeypatch.setenv("PR_KB_REPO", "ROCm/aiter")
    monkeypatch.setattr(server, "_client", type("_C", (), {"get_pr": lambda s, r, n: None}))

    assert _tool("pr_get_reference", {"number": 7})["reason"] == "not_found"


def test_get_reference_counts_files_from_the_array(monkeypatch):
    """summary.changed_files is always null in practice."""
    payload = {
        "summary": {"title": "T", "is_merged": True, "changed_files": None},
        "files": [{"path": f"f{i}.py"} for i in range(5)],
        "commits": [1, 2],
        "distill": {"status": "ok", "worth_trying": 0.4},
    }
    monkeypatch.setenv("PR_KB_REPO", "ROCm/aiter")
    monkeypatch.setattr(
        server,
        "_client",
        type("_C", (), {"get_pr": lambda s, r, n: payload}),
    )

    result = _tool("pr_get_reference", {"number": 7})

    assert result["n_files"] == 5
    assert result["commits"] == 2
    assert result["distill"]["worth_trying"] == 0.4


def test_file_list_uses_the_file_path_field(monkeypatch):
    """The list field is file_path while the by-path query parameter is path."""
    payload = {
        "summary": {"title": "T", "is_merged": True},
        "files": [
            {
                "file_path": "kernels/moe/gemm2.py",
                "status": "modified",
                "additions": 204,
                "deletions": 50,
                "has_patch": True,
                "is_binary": False,
            }
        ],
    }
    monkeypatch.setenv("PR_KB_REPO", "ROCm/FlyDSL")
    monkeypatch.setattr(
        server,
        "_client",
        type("_C", (), {"get_pr": lambda s, r, n: payload}),
    )

    entry = _tool("pr_get_reference", {"number": 959})["files"][0]

    assert entry["file_path"] == "kernels/moe/gemm2.py"
    assert entry["has_patch"] is True
    assert entry["is_binary"] is False


def test_get_reference_caps_the_file_list(monkeypatch):
    payload = {
        "summary": {"title": "T", "is_merged": False},
        "files": [{"file_path": f"f{i}.py"} for i in range(200)],
    }
    monkeypatch.setenv("PR_KB_REPO", "ROCm/aiter")
    monkeypatch.setattr(
        server,
        "_client",
        type("_C", (), {"get_pr": lambda s, r, n: payload}),
    )

    result = _tool("pr_get_reference", {"number": 7})

    assert result["n_files"] == 200
    assert len(result["files"]) == server.MAX_FILES_LISTED
    assert result["files_truncated"] is True


@pytest.mark.parametrize("files", [{}, "", 0, ["not-an-object"]])
def test_get_reference_rejects_invalid_file_lists(monkeypatch, files):
    monkeypatch.setenv("PR_KB_REPO", "ROCm/aiter")
    monkeypatch.setattr(
        server,
        "_client",
        type(
            "_C",
            (),
            {"get_pr": lambda s, r, n: {"files": files}},
        ),
    )

    with pytest.raises(server.PRContractError, match="'files'"):
        _tool("pr_get_reference", {"number": 7})


def test_file_patch_absence_is_explained(monkeypatch):
    """A force-push makes an indexed path 404 at the current head."""
    monkeypatch.setenv("PR_KB_REPO", "ROCm/aiter")
    monkeypatch.setattr(
        server,
        "_client",
        type("_C", (), {"get_file_patch": lambda s, r, n, p: None}),
    )

    result = _tool("pr_get_file_patch", {"number": 7, "file_path": "a.py"})

    assert result["reason"] == "absent_at_current_head"


def test_file_patch_is_truncated_to_a_context_safe_size(monkeypatch):
    payload = {"patch": "x" * (server.MAX_PATCH_BYTES * 3)}
    monkeypatch.setenv("PR_KB_REPO", "ROCm/aiter")
    monkeypatch.setattr(
        server,
        "_client",
        type("_C", (), {"get_file_patch": lambda s, r, n, p: payload}),
    )

    result = _tool("pr_get_file_patch", {"number": 7, "file_path": "a.py"})

    assert result["truncated"] is True
    assert len(result["patch"].encode()) <= server.MAX_PATCH_BYTES


def test_small_patch_is_returned_whole(monkeypatch):
    monkeypatch.setenv("PR_KB_REPO", "ROCm/aiter")
    monkeypatch.setattr(
        server,
        "_client",
        type(
            "_C",
            (),
            {"get_file_patch": lambda s, r, n, p: {"patch": "@@ -1 +1 @@"}},
        ),
    )

    result = _tool("pr_get_file_patch", {"number": 7, "file_path": "a.py"})

    assert result["truncated"] is False
    assert result["patch"] == "@@ -1 +1 @@"


def test_file_patch_requires_the_documented_field(monkeypatch):
    monkeypatch.setenv("PR_KB_REPO", "ROCm/aiter")
    monkeypatch.setattr(
        server,
        "_client",
        type(
            "_C",
            (),
            {"get_file_patch": lambda s, r, n, p: {"diff": "legacy"}},
        ),
    )

    with pytest.raises(server.PRContractError, match="must contain 'patch'"):
        _tool("pr_get_file_patch", {"number": 7, "file_path": "a.py"})


def _backend_like_env() -> dict[str, str]:
    """Mimic a backend-spawned stdio server with a minimal PATH.

    Keep interpreter/runtime vars (e.g. LD_LIBRARY_PATH from setup-python) so the
    subprocess can actually start on self-hosted CI runners.
    """
    env = os.environ.copy()
    env["PATH"] = "/usr/bin:/bin"
    src = str(SRC_ROOT)
    prefix = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (prefix, src) if p)
    return env


def _feed_stdin(monkeypatch, lines: list[str]) -> None:
    """Install a fake stdin buffer yielding the given JSON-RPC lines."""
    stream = io.BytesIO("".join(lines).encode())
    monkeypatch.setattr(server.sys, "stdin", type("_Stdin", (), {"buffer": stream})())


def test_write_message_emits_one_compact_json_line(capsys):
    server._write_message({"jsonrpc": "2.0", "id": 1, "result": {}})

    out = capsys.readouterr().out
    assert out.count("\n") == 1
    assert " " not in out
    assert json.loads(out)["id"] == 1


def test_serve_answers_then_stops_at_end_of_input(monkeypatch, capsys):
    _feed_stdin(
        monkeypatch,
        [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n",
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}) + "\n",
        ],
    )

    asyncio.run(server._serve())

    replies = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l]
    assert [r["id"] for r in replies] == [1, 2]
    assert len(replies[1]["result"]["tools"]) == 3


def test_serve_reports_malformed_requests_and_skips_notifications(monkeypatch, capsys):
    _feed_stdin(
        monkeypatch,
        [
            "{not json}\n",
            json.dumps([1, 2, 3]) + "\n",
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n",
            json.dumps({"jsonrpc": "2.0", "id": 7, "method": "ping"}) + "\n",
        ],
    )

    asyncio.run(server._serve())

    replies = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l]
    assert [reply["id"] for reply in replies] == [None, None, 7]
    assert replies[0]["error"]["code"] == -32700
    assert replies[1]["error"]["code"] == -32600


def test_serve_rejects_non_object_params(monkeypatch, capsys):
    _feed_stdin(
        monkeypatch,
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 8,
                    "method": "ping",
                    "params": [],
                }
            )
            + "\n",
        ],
    )

    asyncio.run(server._serve())

    reply = json.loads(capsys.readouterr().out.strip())
    assert reply["error"]["code"] == -32602


def test_serve_maps_invalid_tool_arguments_to_invalid_params(
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("PR_KB_REPO", "invalid")
    _feed_stdin(
        monkeypatch,
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 8,
                    "method": "tools/call",
                    "params": {
                        "name": "pr_get_reference",
                        "arguments": {"number": 1},
                    },
                }
            )
            + "\n",
        ],
    )

    asyncio.run(server._serve())

    reply = json.loads(capsys.readouterr().out.strip())
    assert reply["error"]["code"] == -32602


def test_serve_returns_on_exit_notification(monkeypatch, capsys):
    _feed_stdin(
        monkeypatch,
        [
            json.dumps({"jsonrpc": "2.0", "method": "exit"}) + "\n",
            json.dumps({"jsonrpc": "2.0", "id": 9, "method": "ping"}) + "\n",
        ],
    )

    asyncio.run(server._serve())

    assert capsys.readouterr().out == ""


def test_serve_maps_tool_failures_to_jsonrpc_errors(monkeypatch, capsys):
    """Map internal configuration failures to server errors."""
    monkeypatch.setenv("PR_KB_REPO", "ROCm/aiter")

    def exploding_client():
        raise ValueError("invalid PR_KB_TOP_K")

    monkeypatch.setattr(server, "_client", exploding_client)
    _feed_stdin(
        monkeypatch,
        [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "pr_get_reference", "arguments": {"number": 1}},
                }
            )
            + "\n",
        ],
    )

    asyncio.run(server._serve())

    reply = json.loads(capsys.readouterr().out.strip())
    assert reply["error"]["code"] == -32603
    assert "invalid PR_KB_TOP_K" in reply["error"]["message"]


def test_main_runs_the_serve_loop(monkeypatch):
    calls = []

    async def fake_serve():
        calls.append("served")

    monkeypatch.setattr(server, "_serve", fake_serve)
    server.main()

    assert calls == ["served"]


def test_client_factory_builds_a_bounded_client():
    from kernelforge.knowledge.pr_monitor_client import PRMonitorClient

    assert isinstance(server._client(), PRMonitorClient)


def test_server_speaks_json_rpc_over_stdio():
    """End-to-end through a real subprocess, the way a backend launches it."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "kernelforge.mcp_server.pr_stdio_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=_backend_like_env(),
    )
    try:
        proc.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {},
                }
            )
            + "\n"
        )
        proc.stdin.flush()
        init = json.loads(proc.stdout.readline())

        proc.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                }
            )
            + "\n"
        )
        proc.stdin.flush()
        listed = json.loads(proc.stdout.readline())
    finally:
        proc.stdin.close()
        proc.wait(timeout=15)

    assert init["result"]["serverInfo"]["name"] == server.SERVER_NAME
    assert [t["name"] for t in listed["result"]["tools"]] == list(server.TOOL_NAMES)


def test_notifications_without_an_id_get_no_reply():
    """A JSON-RPC notification must not produce a response line."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "kernelforge.mcp_server.pr_stdio_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=_backend_like_env(),
    )
    try:
        proc.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                }
            )
            + "\n"
        )
        proc.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 9,
                    "method": "ping",
                }
            )
            + "\n"
        )
        proc.stdin.flush()
        reply = json.loads(proc.stdout.readline())
    finally:
        proc.stdin.close()
        proc.wait(timeout=15)

    assert reply["id"] == 9, "the notification must not have produced a line"
