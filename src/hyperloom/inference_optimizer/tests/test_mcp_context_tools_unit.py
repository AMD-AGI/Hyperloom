# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the in-process context-pull MCP tool surface."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer.session.session_paths import failure_evidence_path
from hyperloom.orchestrator.roles import mcp_context_tools as mct
from hyperloom.orchestrator.state.failure_packets import write_failure


def _shared_state():
    """Build a fake SharedState exposing the gaps projection."""
    return SimpleNamespace(to_gaps_summary=lambda max_attempts=0: f"gaps(max_attempts={max_attempts})")


def _provider(shared_state, session_dir=Path("/nonexistent-session"), **readers):
    """Build a ContextProvider whose readers are empty stubs unless overridden."""
    stubs = {
        "session_dir": session_dir,
        "inbox_reader": lambda since_seq: "",
        "recent_outcomes_reader": lambda top_k: "",
        "running_tasks_reader": lambda: "",
        "reference_reader": lambda name: "",
    }
    return mct.ContextProvider(shared_state=shared_state, **{**stubs, **readers})


def test_qualified():
    assert mct.qualified_context_tool_name("get_gaps") == "mcp__inference_optimizer_context__get_gaps"
    with pytest.raises(KeyError):
        mct.qualified_context_tool_name("foo")


def test_safe_handles_exception():
    def boom():
        raise RuntimeError("x")

    p = _provider(shared_state=SimpleNamespace())
    with pytest.raises(RuntimeError, match="x"):
        p._safe(boom, "lbl")


def test_an_empty_projection_reads_as_empty():
    p = _provider(shared_state=SimpleNamespace(to_gaps_summary=lambda max_attempts=0: ""), inbox_reader=lambda s: None)
    assert p.gaps() == "(gaps: empty)"
    assert p.inbox() == "(inbox: empty)"


def test_runtime_findings_reads_current_best_slot(tmp_path):
    slot = tmp_path / "variant_00_v"
    slot.mkdir()
    (slot / "runtime_findings.json").write_text(
        json.dumps({"framework": "vllm", "log_path": "/s/server.log", "findings": []}), encoding="utf-8"
    )
    state = SimpleNamespace(current_best_measurement={"launch_evidence_path": str(slot / "launch_evidence.json")})
    p = _provider(shared_state=state)

    assert p.runtime_findings() == "runtime findings for /s/server.log [vllm]"
    assert "get_runtime_findings" in mct.CONTEXT_TOOL_NAMES


# ---- _make_handler ----


async def test_make_handler_success():
    p = _provider(shared_state=_shared_state())
    handler = mct._make_handler(p, "gaps")
    out = await handler({})
    assert out["content"][0]["text"] == "gaps(max_attempts=5)"


@pytest.mark.parametrize(
    ("method_name", "args", "expected"),
    [
        ("recent_outcomes", {"top_k": 2}, "outcomes(2)"),
        ("inbox", {"since_seq": 9}, "inbox(9)"),
        ("get_failure", {"failure_id": "fail.t1.abc"}, "fail.t1.abc"),
    ],
    ids=["top_k", "since_seq", "failure_id"],
)
async def test_make_handler_forwards_kwargs(method_name, args, expected):
    p = _provider(
        shared_state=_shared_state(),
        inbox_reader=lambda s: f"inbox({s})",
        recent_outcomes_reader=lambda k: f"outcomes({k})",
    )
    out = await mct._make_handler(p, method_name)(args)
    assert expected in out["content"][0]["text"]


async def test_make_handler_non_str_result():
    provider = SimpleNamespace(foo=lambda **k: {"a": 1})
    handler = mct._make_handler(provider, "foo")
    out = await handler({})
    assert json.loads(out["content"][0]["text"]) == {"a": 1}


async def test_make_handler_exception():
    def boom(**k):
        raise RuntimeError("nope")

    provider = SimpleNamespace(foo=boom)
    handler = mct._make_handler(provider, "foo")
    out = await handler({})
    assert out["is_error"] is True


async def test_read_projection_stays_on_the_calling_thread():
    owner_thread = threading.get_ident()
    seen = []

    def read(max_attempts=0):
        seen.append(threading.get_ident())
        return "gaps"

    p = _provider(shared_state=SimpleNamespace(to_gaps_summary=read))
    out = await mct._make_handler(p, "gaps")({})
    assert seen == [owner_thread]
    assert out == {"content": [{"type": "text", "text": "gaps"}]}


# ---- build_context_tools_server ----


def test_build_server_registers_every_context_tool_with_the_sdk():
    p = _provider(shared_state=_shared_state())
    created = {}

    def tool(name, desc, schema):
        def decorator(handler):
            return (name, handler)

        return decorator

    def create_sdk_mcp_server(name, version, tools):
        created["name"] = name
        created["tools"] = tools
        return "SERVER"

    out = mct.build_context_tools_server(SimpleNamespace(tool=tool, create_sdk_mcp_server=create_sdk_mcp_server), p)
    assert out == "SERVER"
    assert created["name"] == mct.MCP_SERVER_NAME
    assert len(created["tools"]) == len(mct.CONTEXT_TOOL_SPECS)


def _mounted_handlers():
    """The handlers ``build_context_tools_server`` registers with the SDK, by tool name."""
    handlers = {}
    mct.build_context_tools_server(
        SimpleNamespace(
            tool=lambda name, _description, _schema: lambda handler: handlers.setdefault(name, handler),
            create_sdk_mcp_server=lambda _name, _version, tools: tools,
        ),
        _provider(shared_state=_shared_state()),
    )
    return handlers


async def test_get_gaps_returns_the_gaps_ledger_with_attempt_rows():
    """The pushed gaps block carries no attempt rows; the tool asks for five per gap."""
    out = await _mounted_handlers()["get_gaps"]({})
    assert out["content"][0]["text"] == "gaps(max_attempts=5)"


def test_the_mounted_tools_are_the_ones_the_prompt_does_not_push():
    """A block every turn already carries has no pull twin: the turn's prompt is its one copy."""
    assert set(_mounted_handlers()) == {
        "get_gaps",
        "get_runtime_findings",
        "get_inbox",
        "get_recent_outcomes",
        "get_running_tasks",
        "read_reference",
        "get_failure",
        "get_variant_failures",
    }


def test_spec_methods_all_exist_on_provider():
    """Every CONTEXT_TOOL_SPECS entry must name a real ContextProvider method."""
    for _, _, _, method_name in mct.CONTEXT_TOOL_SPECS:
        assert callable(getattr(mct.ContextProvider, method_name, None)), (
            f"ContextProvider missing method {method_name!r}"
        )


def _failed_variant(session_dir: Path, failure_id: str, task_id: str, **fields) -> None:
    write_failure(session_dir, {"failure_id": failure_id, "task_id": task_id, **fields})
    # Packets are listed by the time they were written; give each its own instant.
    packet = failure_evidence_path(session_dir, failure_id)
    stamp = len(list(packet.parent.iterdir()))
    os.utime(packet, ns=(stamp, stamp))


def test_get_failure_of_an_id_nothing_was_written_as_says_so(tmp_path):
    out = _provider(shared_state=_shared_state(), session_dir=tmp_path).get_failure("fail.t1.abc")
    assert out == "(get_failure: no failure evidence was written as 'fail.t1.abc')"


def test_get_failure_returns_the_packet_the_session_wrote(tmp_path):
    _failed_variant(tmp_path, "fail.t1.abc", "t1", error_class="x")
    data = json.loads(_provider(shared_state=_shared_state(), session_dir=tmp_path).get_failure("fail.t1.abc"))
    assert data == {"failure_id": "fail.t1.abc", "task_id": "t1", "error_class": "x"}
    assert json.loads((tmp_path / "reports" / "failures" / "fail.t1.abc.json").read_text()) == data


def test_get_failure_returns_the_last_packet_written_under_an_id(tmp_path):
    _failed_variant(tmp_path, "fail.t1.abc", "t1", error_class="a")
    _failed_variant(tmp_path, "fail.t1.abc", "t1", error_class="b")
    provider = _provider(shared_state=_shared_state(), session_dir=tmp_path)
    assert json.loads(provider.get_failure("fail.t1.abc"))["error_class"] == "b"
    assert len(provider.get_variant_failures().splitlines()) == 1


@pytest.mark.parametrize("failure_id", ["../state", "../../outside", "/etc/hostname", "nested/fail.t1.abc"])
def test_get_failure_reads_no_file_outside_the_failure_store(tmp_path, failure_id):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "state.json").write_text("{}")
    (tmp_path / "outside.json").write_text("{}")
    out = _provider(shared_state=_shared_state(), session_dir=tmp_path / "session").get_failure(failure_id)
    assert out.startswith("(get_failure: no failure evidence was written as ")


def test_get_failure_requires_failure_id():
    p = _provider(shared_state=_shared_state())
    out = p.get_failure("")
    assert "required" in out


def test_get_variant_failures_empty(tmp_path):
    out = _provider(shared_state=_shared_state(), session_dir=tmp_path).get_variant_failures()
    assert out == "(no failure entries)"


def test_get_variant_failures_lists_the_newest_packets_first_and_filters_by_task(tmp_path):
    for failure_id, task_id in (("fail.t1.a", "t1"), ("fail.t2.a", "t2"), ("fail.t1.b", "t1")):
        _failed_variant(tmp_path, failure_id, task_id)
    provider = _provider(shared_state=_shared_state(), session_dir=tmp_path)

    def listed(**args) -> list[str]:
        return [json.loads(line)["failure_id"] for line in provider.get_variant_failures(**args).splitlines()]

    assert listed() == ["fail.t1.b", "fail.t2.a", "fail.t1.a"]
    assert listed(task_id="t1") == ["fail.t1.b", "fail.t1.a"]
    assert listed(top_k=1) == ["fail.t1.b"]
    assert provider.get_variant_failures(task_id="t3") == "(no failure entries)"


def test_get_variant_failures_reaches_a_packet_older_than_the_thirty_newest(tmp_path):
    for index in range(35):
        _failed_variant(tmp_path, f"fail.t1.{index:04d}", "t1")
    provider = _provider(shared_state=_shared_state(), session_dir=tmp_path)
    assert len(provider.get_variant_failures(top_k=50).splitlines()) == 35
    assert json.loads(provider.get_failure("fail.t1.0000"))["failure_id"] == "fail.t1.0000"
