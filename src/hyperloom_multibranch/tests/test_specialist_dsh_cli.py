# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import io
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hyperloom_common.llm import deepseek_session
from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.specialist import dsh_cli
from hyperloom_multibranch.specialist.agent_session import TurnLog
from hyperloom_multibranch.specialist.dsh_cli import PACKAGES, READING_PATCH, UNATTENDED_OFF, WAITING_NOTE
from hyperloom_multibranch.specialist.dsh_cli import Rows, Turn, prompt_of, result_row
from hyperloom_multibranch.specialist.llm_usage import launch_usage

SID, CHILD, MODEL = "0b9c3f6e-conversation", "session-of-a-sub-agent", "deepseek-v4-flash"
GATEWAY = {"OPENAI_BASE_URL": "https://gateway.invalid/v1", "OPENAI_API_KEY": "gateway-key"}


def _event(kind: str, data: dict | None = None, session: str = SID) -> SimpleNamespace:
    """One ``session.event`` notification of the runtime, as the SDK hands it on."""
    payload = {"sessionId": session, "event": {"type": kind, "data": data or {}}}
    return SimpleNamespace(method="session.event", payload=payload)


def _call(blocks: list[dict], *, ident: str = "m1", session: str = SID, **usage: int) -> SimpleNamespace:
    """The notification of one model call that ended, with the harness's own names for its token counts."""
    message = {"id": ident, "role": "assistant", "content": blocks, "source": {"kind": "model", "model": MODEL}}
    return _event("assistant/message", {"turn": 1, "step": 1, "message": message, "usage": usage}, session)


def _tool_call(ident: str = "t1") -> dict:
    return {"type": "tool-call", "id": ident, "name": "mcp__hyperloom__submit_test", "arguments": '{"branch": "s1"}'}


def _rows(out: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in out.getvalue().splitlines()]


def _patch_rows(text: str) -> list[dict]:
    """The rows of a runtime patch, its ``!!js`` expressions read as their text."""
    return yaml.safe_load(text.replace("!!js ", ""))


def _outcome(**fields: object) -> SimpleNamespace:
    return SimpleNamespace(**{"text": "", "error": "", "finish_reason": "completed", **fields})


def test_a_model_call_is_an_assistant_row_with_its_tool_calls_and_token_counts() -> None:
    out = io.StringIO()
    turn = Turn(Rows(out), SID)

    blocks = [{"type": "reasoning", "text": "think"}, {"type": "text", "text": "I submit."}, _tool_call()]
    turn.note(_call(blocks, inputTokens=200, outputTokens=34, cacheReadTokens=1000, cacheWriteTokens=50, totalTokens=9))

    [row] = _rows(out)
    assert (row["type"], row["session_id"], "parent_tool_use_id" in row) == ("assistant", SID, False)
    assert (row["message"]["id"], row["message"]["model"]) == ("m1", MODEL)
    assert row["message"]["content"] == [
        {"type": "thinking", "thinking": "think"},
        {"type": "text", "text": "I submit."},
        {"type": "tool_use", "id": "t1", "name": "mcp__hyperloom__submit_test", "input": {"branch": "s1"}},
    ]
    assert row["message"]["usage"] == {
        "input_tokens": 200,
        "output_tokens": 34,
        "cache_read_input_tokens": 1000,
        "cache_creation_input_tokens": 50,
    }
    assert (turn.calls, turn.usage["output_tokens"]) == (1, 34)


def test_a_tool_call_whose_arguments_are_no_json_object_keeps_them_as_written() -> None:
    out = io.StringIO()
    block = {"type": "tool-call", "id": "t1", "name": "bash", "arguments": "{not json"}

    Turn(Rows(out), SID).note(_call([block]))

    assert _rows(out)[0]["message"]["content"][0]["input"] == {"arguments": "{not json"}


def test_a_tool_result_and_a_retried_model_call_are_rows_and_a_sub_agent_s_are_marked() -> None:
    out = io.StringIO()
    turn = Turn(Rows(out), SID)
    result = {"role": "tool", "toolCallId": "t1", "content": [{"type": "text", "text": "queued r7"}], "isError": True}
    failure = {"code": "RATE_LIMIT", "message": "busy", "status": 429}

    turn.note(_event("tool/result", {"message": result}))
    turn.note(_event("llm/retry", {"retry": 2, "maxRetries": 5, "delayMs": 1042.5, "failure": failure}))
    turn.note(_call([{"type": "text", "text": "from below"}], ident="m2", session=CHILD, outputTokens=5))
    turn.note(_event("request/header", {"header": {"tools": ["a very long list"]}}))
    turn.note(SimpleNamespace(method="session.status", payload={"sessionId": SID, "status": "idle"}))

    tool, retry, below = _rows(out)
    assert tool["message"]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "queued r7", "is_error": True}
    ]
    assert (retry["subtype"], retry["attempt"], retry["max_retries"]) == ("api_retry", 2, 5)
    assert (retry["error_status"], retry["error"], retry["retry_delay_ms"]) == (429, "RATE_LIMIT", 1042.5)
    assert (below["parent_tool_use_id"], below["session_id"]) == (CHILD, SID)
    assert turn.usage["output_tokens"] == 5


def test_a_turn_that_ended_well_ends_with_its_reply_its_calls_and_their_summed_tokens() -> None:
    out = io.StringIO()
    turn = Turn(Rows(out), SID)
    turn.note(_call([_tool_call()], inputTokens=200, outputTokens=34, cacheReadTokens=1000))
    turn.note(_call([{"type": "text", "text": "below"}], ident="m2", session=CHILD, inputTokens=7, outputTokens=5))
    turn.note(_call([{"type": "text", "text": "done"}], ident="m3", inputTokens=300, outputTokens=6))

    row = result_row(turn, _outcome(text="done"), 12.5)

    assert (row["type"], row["subtype"], row["is_error"], row["result"]) == ("result", "success", False, "done")
    assert (row["num_turns"], row["duration_ms"], row["session_id"]) == (3, 12500, SID)
    assert row["usage"] == {
        "input_tokens": 507,
        "output_tokens": 45,
        "cache_read_input_tokens": 1000,
        "cache_creation_input_tokens": 0,
    }
    assert "stop_reason" not in row and "api_error_status" not in row


def test_a_turn_the_gateway_refused_ends_as_an_error_with_its_status_and_counts_the_call_it_made() -> None:
    turn = Turn(Rows(io.StringIO()), SID)
    error = {"message": "invalid key", "code": "AUTH", "status": 401}
    turn.note(_event("assistant/attempt", {"turn": 1, "step": 1}))
    turn.note(_event("turn/end", {"turn": 1, "reason": {"kind": "error", "error": error}}))

    row = result_row(turn, _outcome(error="AUTH: invalid key", finish_reason="error"), 1.0)

    assert (row["subtype"], row["is_error"], row["result"]) == ("error_during_execution", True, "AUTH: invalid key")
    # A call was made, so the session does not take the failure for a conversation the CLI refused to resume.
    assert (row["num_turns"], row["api_error_status"]) == (1, 401)


def test_a_sub_agent_s_failure_is_not_the_turn_s() -> None:
    turn = Turn(Rows(io.StringIO()), SID)
    error = {"message": "busy", "code": "RATE_LIMIT", "status": 429}

    turn.note(_event("turn/end", {"reason": {"kind": "error", "error": error}}, CHILD))

    assert "api_error_status" not in result_row(turn, _outcome(text="done"), 1.0)


@pytest.mark.parametrize(
    ("error", "finish", "stop"),
    [
        ({"code": "CONTEXT_WINDOW_EXCEEDED", "message": "too long"}, "error", "model_context_window_exceeded"),
        ({}, "max-tokens", "max_tokens"),
    ],
    ids=["past-the-window", "at-the-output-cap"],
)
def test_a_turn_stopped_by_a_limit_names_it_as_a_claude_cli_does(tmp_path: Path, error, finish, stop) -> None:
    log = tmp_path / "process.log"
    with log.open("w", encoding="utf-8") as out:
        turn = Turn(Rows(out), SID)
        turn.note(_event("assistant/attempt"))
        turn.note(_event("turn/end", {"reason": {"kind": finish, "error": error}}))
        row = result_row(turn, _outcome(error="the turn stopped", finish_reason=finish), 1.0)
        turn.rows.write(row)

    assert row["stop_reason"] == stop
    turns = TurnLog(log)
    assert turns.ended() == 1 and not turns.succeeded
    assert turns.context_exceeded is (stop == "model_context_window_exceeded")


def test_a_model_call_is_in_flight_from_its_step_s_start_to_its_reply_by_session() -> None:
    clock = SimpleNamespace(now=100.0)
    turn = Turn(Rows(io.StringIO()), SID, lambda: clock.now)
    assert turn.in_flight() == {}

    turn.note(_event("step/start", {"turn": 1, "step": 1}))
    clock.now = 130.0
    turn.note(_event("step/start", {"turn": 1, "step": 1}, CHILD))
    clock.now = 190.0
    assert turn.in_flight() == {"model_request": 90.0}
    turn.note(_call([_tool_call()]))
    assert turn.in_flight() == {"model_request": 60.0}
    turn.note(_event("assistant/attempt", session=CHILD))
    assert turn.in_flight() == {}
    turn.note(_event("llm/retry-started", {"retry": 1}, CHILD))
    assert turn.in_flight() == {"model_request": 0.0}
    turn.ended()
    assert turn.in_flight() == {}


def _compaction(**usage: int) -> SimpleNamespace:
    """The event of the model call the runtime summarized dropped messages with, as a long conversation's log has it."""
    data = {"compactionId": "c1", "summary": [{"type": "text", "text": "## Primary Request"}], "llmStreamCall": True}
    return _event("compaction/summary", {**data, "shadowedTokenCount": 8511, "model": MODEL, "usage": usage})


def test_the_model_call_a_compaction_makes_is_in_flight_and_counted_with_the_turn_s(tmp_path: Path) -> None:
    out, clock = io.StringIO(), SimpleNamespace(now=100.0)
    turn = Turn(Rows(out), SID, lambda: clock.now)

    # The runtime compacts before the turn's first step, a model call of its own that takes half a minute.
    turn.note(_event("compaction/start", {"compactionId": "c1"}))
    clock.now = 135.0
    assert turn.in_flight() == {"model_request": 35.0}
    spent = {"inputTokens": 4, "outputTokens": 1837, "cacheReadTokens": 17535, "cacheWriteTokens": 11864}
    turn.note(_compaction(**spent, totalTokens=31240))
    turn.note(_event("compaction/end", {"compactionId": "c1"}))
    assert turn.in_flight() == {}
    turn.note(_call([{"type": "text", "text": "done"}], inputTokens=200, outputTokens=34, cacheReadTokens=1000))

    boundary, said = _rows(out)
    assert (boundary["type"], boundary["subtype"], boundary["model"]) == ("system", "compact_boundary", MODEL)
    assert boundary["compact_metadata"] == {"trigger": "auto", "pre_tokens": 8511}
    assert boundary["usage"]["output_tokens"] == 1837 and said["type"] == "assistant"
    row = result_row(turn, _outcome(text="done"), 40.0)
    assert (row["num_turns"], row["usage"]) == (
        2,
        {
            "input_tokens": 204,
            "output_tokens": 1871,
            "cache_read_input_tokens": 18535,
            "cache_creation_input_tokens": 11864,
        },
    )
    # A session reads the spend off the turn's result, and the conversation's size off its last reply, not the summary.
    log = tmp_path / "process.log"
    log.write_text("".join(json.dumps(item) + "\n" for item in (boundary, said, row)), encoding="utf-8")
    turns = TurnLog(log)
    assert turns.ended() == 1 and turns.context == 1200
    assert dict(launch_usage(log).tokens) == {MODEL: TokenCounts(204, 1871, 18535, 11864)}


def test_a_compaction_that_called_no_model_is_no_row_and_no_call() -> None:
    out = io.StringIO()
    turn = Turn(Rows(out), SID)

    turn.note(_event("compaction/prune", {"prunedCount": 3}))
    turn.note(_event("compaction/summary", {"compactionId": "c2", "summary": []}))

    assert (_rows(out), turn.calls, turn.usage) == ([], 0, {})


def test_a_tool_call_is_in_flight_from_its_call_to_its_result_or_the_end_of_its_session_s_turn() -> None:
    clock = SimpleNamespace(now=100.0)
    turn = Turn(Rows(io.StringIO()), SID, lambda: clock.now)

    # The runtime's own tool that waits for input blocks for as long as the model asks, an hour at most.
    turn.note(_event("tool/call", {"callId": "t1", "name": "wait_agent", "arguments": '{"timeout_ms": 900000}'}))
    clock.now = 160.0
    turn.note(_event("tool/call", {"callId": "t2", "name": "bash", "arguments": "{}"}, CHILD))
    clock.now = 200.0
    assert turn.in_flight() == {"tool_call": 100.0}
    turn.note(_event("tool/result", {"message": {"role": "tool", "toolCallId": "t1", "content": []}}))
    assert turn.in_flight() == {"tool_call": 40.0}
    # A sub-agent whose turn ended runs no tool any more, whatever became of the call.
    turn.note(_event("turn/end", {"turn": 1, "reason": {"kind": "aborted"}}, CHILD))
    assert turn.in_flight() == {}
    turn.note(_event("tool/call", {"callId": "t3", "name": "bash", "arguments": "{}"}))
    turn.note(_event("step/start", {"turn": 1, "step": 2}, CHILD))
    clock.now = 230.0
    assert turn.in_flight() == {"model_request": 30.0, "tool_call": 30.0}
    turn.ended()
    assert turn.in_flight() == {}


def test_a_stream_json_user_message_is_its_text_and_any_other_line_is_the_prompt_itself() -> None:
    blocks = [{"type": "text", "text": "first"}, {"type": "image"}, {"type": "text", "text": " second"}]
    message = json.dumps({"type": "user", "message": {"role": "user", "content": blocks}})

    assert prompt_of(message + "\n") == "first second"
    assert prompt_of(json.dumps({"type": "user", "message": {"content": "as text"}})) == "as text"
    assert prompt_of("  plain words\n") == "plain words"
    assert prompt_of('{"type": "other"}\n') == '{"type": "other"}'


class _HarnessError(Exception):
    """Stands in for ``deepseek_harness.errors.HarnessError``."""


class _Session:
    def __init__(self, harness: "_Harness", session_id: str) -> None:
        self.harness, self.id = harness, session_id

    def run(self, prompt: str, *, on_notification=None):
        harness = self.harness
        harness.prompts.append(prompt)
        number = len(harness.prompts)
        if harness.fails_at == number:
            raise _HarnessError("DeepSeek Harness runtime stdout closed")
        if harness.stalls is not None:
            on_notification(_event(harness.stalls_in, {"turn": number, "step": 1, "callId": "t1"}, self.id))
            harness.stalls.wait(5)
        events = [
            _call([{"type": "text", "text": f"reply {number}"}], ident=f"m{number}", session=self.id,
                  inputTokens=10 * number, outputTokens=number, cacheReadTokens=100),
            _event("turn/end", {"turn": number, "reason": harness.reason}, self.id),
        ]  # fmt: skip
        for notification in events:
            on_notification(notification)
        return SimpleNamespace(
            session_id=self.id,
            final_response=f"reply {number}",
            finish_reason=harness.reason["kind"],
            events=[notification.payload["event"] for notification in events],
        )


class _Harness:
    """The SDK's harness, recording how it was started and what it was asked."""

    made: list["_Harness"] = []
    known: set[str] = set()
    fails_at = 0
    reason: dict = {"kind": "completed"}
    stalls: threading.Event | None = None
    stalls_in = "step/start"

    def __init__(self, **kwargs: object) -> None:
        self.kwargs, self.prompts, self.opened, self.closed = kwargs, [], [], False
        _Harness.made.append(self)

    def start(self) -> None:
        pass

    def start_session(self, session_id: str) -> _Session:
        self.opened.append(("start", session_id))
        return _Session(self, session_id)

    def resume_session(self, session_id: str) -> _Session:
        if session_id not in _Harness.known:
            raise _HarnessError(f'session "{session_id}" not found')
        self.opened.append(("resume", session_id))
        return _Session(self, session_id)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A launcher's environment: the runtime's home in the workspace, the gateway, and the SDK stood in for."""
    sdk = SimpleNamespace(DeepSeekHarness=_Harness, errors=SimpleNamespace(HarnessError=_HarnessError))
    monkeypatch.setattr(deepseek_session, "load_harness_sdk", lambda: sdk)
    monkeypatch.setattr(dsh_cli, "load_harness_sdk", lambda: sdk)
    monkeypatch.setattr(_Harness, "made", [])
    monkeypatch.setattr(_Harness, "known", set())
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY", "DSH_MODEL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in GATEWAY.items():
        monkeypatch.setenv(name, value)
    workspace = tmp_path / "s1" / "session"
    workspace.mkdir(parents=True)
    (tmp_path / "s1" / "worktree").mkdir()
    monkeypatch.chdir(tmp_path / "s1" / "worktree")
    monkeypatch.setenv("DSH_HOME", str(workspace / ".dsh"))
    return workspace / ".dsh"


def _message(text: str) -> str:
    return json.dumps({"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": text}]}}) + "\n"


def _run(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, argv: list[str], stdin: str):
    """Run the CLI on ``stdin``: its exit code, the rows it wrote and what it said on stderr."""
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    code = dsh_cli.main(argv)
    captured = capsys.readouterr()
    return code, [json.loads(line) for line in captured.out.splitlines()], captured.err


def test_a_fresh_conversation_runs_each_message_as_a_turn_until_its_stdin_ends(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    role = tmp_path / "s1" / "session" / "system_prompt.md"
    role.write_text("You are a long-running optimization specialist.", encoding="utf-8")
    argv = ["--system-prompt-file", str(role), "--model", MODEL, "--session-id", SID]

    code, rows, _ = _run(monkeypatch, capsys, argv, _message("Startup brief") + "\n" + _message("a result arrived"))

    [harness] = _Harness.made
    assert code == 0 and harness.closed
    assert (harness.opened, harness.prompts) == ([("start", SID)], ["Startup brief", "a result arrived"])
    assert [row["type"] for row in rows] == ["system", "assistant", "result", "assistant", "result"]
    assert (rows[0]["subtype"], rows[0]["session_id"], rows[0]["model"]) == ("init", SID, MODEL)
    assert rows[0]["cwd"] == str(tmp_path / "s1" / "worktree")
    assert [row["result"] for row in rows if row["type"] == "result"] == ["reply 1", "reply 2"]
    kwargs = harness.kwargs
    assert (kwargs["profile"], kwargs["model"], kwargs["dsh_home"]) == ("sdk", MODEL, str(home))
    assert kwargs["cwd"] == kwargs["runtime_cwd"] == str(tmp_path / "s1" / "worktree")
    assert kwargs["env"]["DEEPSEEK_BASE_URL"] == GATEWAY["OPENAI_BASE_URL"]
    assert kwargs["env"]["DEEPSEEK_API_KEY"] == GATEWAY["OPENAI_API_KEY"]
    assert (kwargs["env"]["DSH_PERMISSION_MODE"], kwargs["env"]["DSH_TELEMETRY_MODE"]) == (
        "danger-full-access",
        "DISABLED",
    )
    # The role is the home's instructions, followed by what it cannot say of this harness's own waiting tool, and
    # the patch keeps the log a killed runtime leaves readable.
    instructions = (home / "AGENTS.md").read_text(encoding="utf-8")
    assert instructions == f"You are a long-running optimization specialist.\n\n{WAITING_NOTE}\n"
    assert "Never call wait_agent to wait for a test" in WAITING_NOTE and "end your turn" in WAITING_NOTE
    patch = {row["id"]: row for row in _patch_rows(Path(kwargs["patches"][-1]).read_text(encoding="utf-8"))}
    assert patch["session-persistence-jsonl"]["config"]["compression"] == "none"
    assert patch["agent-default-model"]["config"]["model"] == MODEL
    assert [patch[row]["disabled"] for row in UNATTENDED_OFF] == [True] * len(UNATTENDED_OFF)


@pytest.mark.parametrize(("others", "kept"), [(["/opt/tools"], "/opt/tools"), ([], "")])
def test_the_runtime_and_its_tools_do_not_import_from_where_the_bridge_was_pointed_at_the_app(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, others: list[str], kept: str
) -> None:
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(PACKAGES), *others]))

    _run(monkeypatch, capsys, ["--model", MODEL, "--session-id", SID], _message("a"))

    # The SDK lays this over the bridge's own environment, so only a value set here replaces the bridge's.
    assert _Harness.made[0].kwargs["env"]["PYTHONPATH"] == kept


def test_the_rows_of_a_launch_read_as_a_claude_cli_s_turns_and_spend(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    _, rows, _ = _run(monkeypatch, capsys, ["--model", MODEL, "--session-id", SID], _message("a") + _message("b"))
    log = tmp_path / "process.log"
    log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    turns, used = TurnLog(log), launch_usage(log)

    assert turns.ended() == 2 and turns.inited and turns.succeeded and not turns.refused
    assert turns.context == 20 + 100
    assert (used.session_id, dict(used.tokens)) == (SID, {MODEL: TokenCounts(30, 3, 200)})


def test_a_resumed_conversation_is_restored_before_init_and_the_home_s_sessions_are_kept(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    kept = home / "sessions" / "--worktree--" / SID / "session.v4.jsonl"
    kept.parent.mkdir(parents=True)
    kept.write_text('{"type": "session"}\n', encoding="utf-8")
    _Harness.known.add(SID)

    code, rows, _ = _run(monkeypatch, capsys, ["--model", MODEL, "--resume", SID], _message("it crashed; go on"))

    [harness] = _Harness.made
    assert code == 0 and harness.opened == [("resume", SID)] and harness.prompts == ["it crashed; go on"]
    assert [row["type"] for row in rows] == ["system", "assistant", "result"] and rows[0]["session_id"] == SID
    assert kept.read_text(encoding="utf-8") == '{"type": "session"}\n'


def test_a_conversation_that_cannot_be_restored_ends_the_cli_before_init(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    code, rows, said = _run(monkeypatch, capsys, ["--model", MODEL, "--resume", SID], _message("go on"))

    [harness] = _Harness.made
    assert (code, rows) == (1, []) and harness.closed and harness.prompts == []
    assert f'cannot resume conversation {SID}: session "{SID}" not found' in said


def test_an_sdk_that_cannot_resume_refuses_the_resume(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.delattr(_Harness, "resume_session")

    code, rows, said = _run(monkeypatch, capsys, ["--model", MODEL, "--resume", SID], _message("go on"))

    assert (code, rows) == (1, []) and "cannot resume a conversation" in said and _Harness.made[0].closed


def test_a_runtime_that_fails_in_a_turn_ends_the_cli_with_an_error_after_the_turns_it_finished(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(_Harness, "fails_at", 2)

    code, rows, said = _run(monkeypatch, capsys, ["--model", MODEL, "--session-id", SID], _message("a") + _message("b"))

    assert code == 1 and [row["type"] for row in rows] == ["system", "assistant", "result"]
    assert "the runtime failed in a turn: DeepSeek Harness runtime stdout closed" in said
    assert _Harness.made[0].closed


def test_the_mcp_servers_are_rows_of_the_patch_and_their_token_reaches_only_the_runtime_s_environment(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    token = "tok-" + "f" * 24
    monkeypatch.setenv("HYPERLOOM_SPECIALIST_MCP_TOKEN", token)
    server = {
        "type": "http",
        "url": "http://127.0.0.1:9/mcp",
        "headers": {"Authorization": "Bearer ${HYPERLOOM_SPECIALIST_MCP_TOKEN}"},
    }
    config = tmp_path / "specialist_mcp.json"
    config.write_text(json.dumps({"mcpServers": {"hyperloom": server}}), encoding="utf-8")

    _, rows, _ = _run(monkeypatch, capsys, ["--model", MODEL, "--mcp-config", str(config)], _message("a"))

    kwargs = _Harness.made[0].kwargs
    text = Path(kwargs["patches"][-1]).read_text(encoding="utf-8")
    assert 'serverName: "hyperloom"' in text and "transport: streamable-http" in text
    assert 'url: "http://127.0.0.1:9/mcp"' in text
    assert '"Authorization": !!js process.env.HYPERLOOM_DSH_MCP_ENV_0' in text
    assert kwargs["env"]["HYPERLOOM_DSH_MCP_ENV_0"] == f"Bearer {token}"
    assert token not in "".join(path.read_text(encoding="utf-8") for path in home.rglob("*") if path.is_file())
    assert rows[0]["mcp_servers"] == [{"name": "hyperloom"}]


def test_a_reading_run_boots_the_composition_that_only_reads_and_answers_one_prompt(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    prompt = "Read the checkout.\nAnswer with its manifest.\n"

    code, rows, _ = _run(monkeypatch, capsys, ["--print", "--read-only", "--model", MODEL], prompt)

    [harness] = _Harness.made
    assert code == 0 and harness.prompts == [prompt] and harness.closed
    assert [row["type"] for row in rows] == ["system", "assistant", "result"]
    kwargs = harness.kwargs
    assert (kwargs["profile"], kwargs["env"]["DSH_PERMISSION_MODE"]) == ("sdk-minimal", "read-only")
    text = Path(kwargs["patches"][-1]).read_text(encoding="utf-8")
    assert text.startswith(READING_PATCH)
    # The composition has no row of the gateway route's adapter, so the patch adds it and declares the route after.
    route = _patch_rows(text[len(READING_PATCH) :])[0]
    assert route["id"] == "llm-pi-ai" and route["config"]["providers"]["gateway"]["models"][0]["id"] == MODEL
    assert not (home / "AGENTS.md").exists() and rows[0]["mcp_servers"] == []


def test_the_reading_composition_has_no_shell_and_no_tool_that_writes() -> None:
    rows = _patch_rows(READING_PATCH)
    by_id = {row["id"]: row for row in rows if "id" in row}
    inserted = [row["name"] for row in rows[-1]["insert"]]

    assert by_id["persistent-bash"]["disabled"] is True and by_id["tool-working-directory"]["disabled"] is True
    assert by_id["tool-fs"]["config"] == {"read": True, "readImage": False, "write": False, "edit": False}
    assert by_id["sandbox-policy"]["config"]["mode"] == "read-only"
    # What is added is the tool that searches and the adapter the gateway route is declared on, and nothing else.
    assert inserted == ["@deepseek-ai/dsh-tool-fs-search", "@deepseek-ai/dsh-llm-pi-ai"]


def test_a_one_shot_run_whose_turn_failed_exits_non_zero_after_its_error_result(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    error = {"message": "DeepSeek Messages transport failed", "code": "TRANSPORT"}
    monkeypatch.setattr(_Harness, "reason", {"kind": "error", "error": error})

    code, rows, _ = _run(monkeypatch, capsys, ["--print", "--read-only", "--model", MODEL], "p")

    assert code == 1
    assert (rows[-1]["is_error"], rows[-1]["result"]) == (True, "TRANSPORT: DeepSeek Messages transport failed")


@pytest.mark.parametrize(("stalls_in", "subtype"), [("step/start", "model_request"), ("tool/call", "tool_call")])
def test_a_model_call_or_a_tool_call_in_flight_is_said_in_the_log_while_it_waits(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, stalls_in: str, subtype: str
) -> None:
    stalls = threading.Event()
    monkeypatch.setattr(_Harness, "stalls", stalls)
    monkeypatch.setattr(_Harness, "stalls_in", stalls_in)
    monkeypatch.setattr(dsh_cli, "IN_FLIGHT_ROW_S", 0.05)
    said = threading.Event()
    write = Rows.write

    def watched(self: Rows, row) -> None:
        write(self, row)
        if row.get("subtype") == subtype:
            said.set()
            stalls.set()

    monkeypatch.setattr(Rows, "write", watched)

    _, rows, _ = _run(monkeypatch, capsys, ["--model", MODEL, "--session-id", SID], _message("a"))

    assert said.is_set()
    waiting = [row for row in rows if row.get("subtype") == subtype]
    assert waiting and waiting[0]["session_id"] == SID
    # A row of a call in flight ends no turn and counts no model call.
    assert [row["type"] for row in rows if row.get("subtype") != subtype] == ["system", "assistant", "result"]


@pytest.mark.parametrize("unset", ["DSH_HOME", "OPENAI_API_KEY"])
def test_a_launch_without_its_home_or_its_gateway_is_refused_before_a_runtime_starts(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, unset: str
) -> None:
    monkeypatch.delenv(unset)

    code, rows, said = _run(monkeypatch, capsys, ["--model", MODEL], _message("a"))

    assert (code, rows, _Harness.made) == (1, [], [])
    assert ("DSH_HOME" if unset == "DSH_HOME" else "OPENAI_API_KEY") in said


def test_a_launch_that_names_no_model_runs_the_one_the_environment_names(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv("DSH_MODEL", "deepseek-v4-pro")

    _, rows, _ = _run(monkeypatch, capsys, [], _message("a"))

    assert (rows[0]["model"], _Harness.made[0].kwargs["model"]) == ("deepseek-v4-pro", "deepseek-v4-pro")
