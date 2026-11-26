# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The DeepSeek Harness as an agent CLI of a specialist's session: the stream-json protocol a Claude CLI speaks, over
one ``dsh`` runtime.

A session drives its CLI through a pipe (:mod:`~hyperloom_multibranch.specialist.agent_session`): the launch prompt and
every later message are written to the CLI's stdin as stream-json user messages, and its stdout is the log the
session reads turn ends, tool calls and token counts from. ``dsh`` holds a conversation across turns only behind
JSON-RPC (``dsh --profile sdk``), which refuses a prompt until its ``initialize`` request was answered, so nothing
written ahead into a pipe can drive it. This module is the process in between, started as
``python -m hyperloom_multibranch.specialist.dsh_cli`` on :func:`bridge_env`, which names the packages its launcher
runs: it holds one runtime through the harness SDK, runs each
message it reads as one turn of one conversation, and writes what the runtime reports as the rows a Claude CLI
writes:

- ``system``/``init`` once the runtime is up and the conversation is open (a resumed one: once it was restored);
- ``assistant`` for every model call, with its text, reasoning and tool calls, its model and its token counts, a
  sub-agent's marked by ``parent_tool_use_id``; ``user`` for every tool result;
- ``system``/``api_retry`` for every model call the runtime retries;
- ``system``/``compact_boundary`` for every model call the runtime summarizes with what it drops from a long
  conversation, with its model and token counts;
- ``result`` when the turn ends: its last reply, or the failure the runtime ended it with (``is_error``, with the
  gateway's status as ``api_error_status`` when it gave one), the model calls it made, a compaction's among them,
  and their summed token counts.

``--session-id`` names the conversation a fresh launch opens and ``--resume`` the one a launch goes on with, restored
from the session log an earlier process left in the same ``DSH_HOME``; a conversation that cannot be restored ends the
process before ``init``, as a Claude CLI ends that cannot resume. The role (``--system-prompt-file``), followed by
:data:`WAITING_NOTE`, is the home's ``AGENTS.md``, which the runtime sends as the conversation's baseline
instructions, and the MCP servers of
``--mcp-config`` are rows of the patch the runtime boots with, their header values held in variables only the runtime
sees (:func:`~hyperloom.orchestrator.specialists.dsh_launch.prepare_session_home`). The runtime runs unattended, so the
tools of its profile that wait for a person or change the runtime itself are off (:data:`UNATTENDED_OFF`).

The runtime reports a model call when it ends, not while it streams, so a long reply would show as silence. While a
model call is in flight the bridge writes a ``system``/``model_request`` row every :data:`IN_FLIGHT_ROW_S`: the
model adapter of a specialist's profile fails a call whose stream stays idle for five minutes, so a call still in
flight is one that receives. A tool call says nothing until its result either, and the runtime's own tools block
longer than a session takes silence for a hung CLI (``wait_agent`` for up to an hour, at the model's word), so a tool
call in flight is a ``system``/``tool_call`` row the same way. Past :data:`IN_FLIGHT_MAX_S` a call is vouched for no
longer.

``--print`` runs one turn on the prompt stdin holds as plain text and exits, non-zero when the turn failed.
``--read-only`` boots a composition whose only tools read: ``read``, ``glob`` and ``grep``, with no shell, no MCP
server, no sub-agent and no instruction or skill discovery in the directory it runs in (:data:`READING_PATCH` over the
closed ``sdk-minimal`` profile), for a run over a checkout that is not trusted.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from hyperloom.orchestrator.specialists.agent_cli_config import SpecialistAgentUnavailableError
from hyperloom.orchestrator.specialists.dsh_launch import DshSessionHome, dsh_sdk_version, prepare_session_home
from hyperloom_common.fs_utils import write_text_nofollow
from hyperloom_common.llm.deepseek_session import (
    DeepSeekSessionError,
    HarnessLaunch,
    child_environment,
    compaction_usage,
    load_harness_sdk,
    normalize_run_result,
    sdk_failures,
    start_harness,
    step_usage,
)
from hyperloom_common.llm.llm_config import AGENT_BACKEND_DEEPSEEK, resolve_forge_llm_model
from hyperloom_common.llm.reasoning_effort import DEFAULT_REASONING_EFFORT, harness_reasoning_effort

#: The module a session runs as its CLI: ``python -m`` this.
MODULE = "hyperloom_multibranch.specialist.dsh_cli"
#: The variable that names the runtime's home, which the launcher sets: the conversation's log is kept there.
HOME_ENV = "DSH_HOME"
#: The directory this app's packages are imported from, and the variable that names such directories to a Python.
PACKAGES = Path(__file__).resolve().parents[2]
IMPORT_PATH_ENV = "PYTHONPATH"
#: The profile a specialist's conversation boots, and the one a reading run boots.
SESSION_PROFILE, READING_PROFILE = "sdk", "sdk-minimal"
#: The rows of :data:`SESSION_PROFILE` an unattended conversation runs without: the tool that asks a person, the one
#: that installs plugins into the runtime, and the one that schedules reminders for later.
UNATTENDED_OFF = ("tool-ask-user", "tool-plugin-manager", "tool-schedule")
#: What a specialist's role is followed by. The profile's row of ``wait_agent`` cannot be switched off without the
#: tools that message a subagent, and a session sends its CLI a message only once a turn has ended, so a wait of
#: that tool for one blocks until its timeout, an hour at most, with a result of the branch already landed.
WAITING_NOTE = """\
Your harness has a tool named wait_agent. It waits for a subagent you started and for nothing else: results, \
judgements, briefs and every other message of your session reach you only after your turn has ended. Never call \
wait_agent to wait for a test or for a message of your session; end your turn instead, and what arrives wakes you."""
#: Laid over :data:`READING_PROFILE`, whose tree is closed: its shell and its directory tool off, its file tool
#: reading only, the search tool added, and its file sandbox read-only.
READING_PATCH = """\
- id: persistent-bash
  disabled: true
- id: tool-working-directory
  disabled: true
- id: tool-fs
  config:
    read: true
    readImage: false
    write: false
    edit: false
- id: sandbox-policy
  config:
    mode: read-only
    workspaceRoot: !!js process.cwd()
- insert:
    - id: tool-fs-search
      name: '@deepseek-ai/dsh-tool-fs-search'
      config:
        sampleOverCapGlobResults: false
"""
READING_PATCH_NAME = "reading_patch.yml"
#: The variable :data:`READING_PROFILE` reads its system prompt from.
SYSTEM_PROMPT_ENV = "DSH_SYSTEM_PROMPT"
#: How the runtime's file sandbox runs: a specialist writes its worktree as a Claude CLI that bypasses permissions
#: does; a reading run writes nothing.
FULL_ACCESS, READ_ONLY = "danger-full-access", "read-only"
#: How long one protocol request may wait for its answer; a turn's own length is not one request's.
REQUEST_TIMEOUT_S = 300.0
#: How often a model call or a tool call in flight is said in the log, well inside the silence a session takes its
#: CLI for lost after, and how long one call is vouched for: the longest the runtime's own waiting tool blocks.
IN_FLIGHT_ROW_S = 60.0
IN_FLIGHT_MAX_S = 3600.0
#: The rows that say them.
MODEL_REQUEST, TOOL_CALL = "model_request", "tool_call"
#: The runtime's failure code for a conversation past its model's window, and how a Claude CLI words that stop.
CONTEXT_EXCEEDED_CODE = "CONTEXT_WINDOW_EXCEEDED"
CONTEXT_EXCEEDED_STOP = "model_context_window_exceeded"
MAX_TOKENS_FINISH, MAX_TOKENS_STOP = "max-tokens", "max_tokens"
_ASSISTANT_MESSAGE, _ASSISTANT_ATTEMPT = "assistant/message", "assistant/attempt"
#: The event of the model call the runtime summarizes with what it drops from a long conversation.
_COMPACTION_SUMMARY = "compaction/summary"
#: The events of a session that open and close the wait for a model call's reply: a step's, and a compaction's.
_CALL_OPENS = frozenset({"step/start", "llm/retry-started", "compaction/start"})
_CALL_CLOSES = frozenset({_ASSISTANT_MESSAGE, _ASSISTANT_ATTEMPT, "step/end", "turn/end", "compaction/end"})
#: The events a tool call runs between, and the one that ends every tool call of its session.
_TOOL_STARTS, _TOOL_RESULT, _TURN_END = "tool/call", "tool/result", "turn/end"
#: The type of the rows that are no message of the conversation, and of a content block that is text.
_SYSTEM, _TEXT = "system", "text"
_ENCODING = "utf-8"


class Refused(RuntimeError):
    """The runtime cannot serve the conversation the command line asks for."""


@dataclass(frozen=True)
class Options:
    """One launch: the model, the role, the conversation it opens or goes on with, and how it runs."""

    model: str
    role: str = ""
    session_id: str = ""
    resume: bool = False
    mcp_config: str | None = None
    one_shot: bool = False
    reading: bool = False


class Rows:
    """The launch's stdout: one JSON row a line, each flushed as it is written, from any thread."""

    def __init__(self, out: TextIO, clock: Callable[[], float] = time.monotonic) -> None:
        self._out, self._clock, self._lock = out, clock, threading.Lock()
        self.written_at = clock()

    def write(self, row: Mapping[str, object]) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            self._out.write(line)
            self._out.flush()
            self.written_at = self._clock()


@dataclass
class Turn:
    """What one turn's notifications add up to: its rows, the model calls it made, their token counts, and what is
    still in flight: the model calls by session, and the tool calls by session and call."""

    rows: Rows
    session_id: str
    clock: Callable[[], float] = time.monotonic
    calls: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    failure: Mapping[str, object] = field(default_factory=dict)
    _in_flight: dict[str, float] = field(default_factory=dict)
    _tools: dict[tuple[str, str], float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def note(self, notification: Any) -> None:
        """Write the rows of one notification of the conversation or a sub-agent of it."""
        payload = notification.payload
        event = payload.get("event") if notification.method == "session.event" else None
        if not isinstance(event, dict):
            return
        kind, data = str(event.get("type") or ""), event.get("data")
        data = data if isinstance(data, dict) else {}
        session = str(payload.get("sessionId") or self.session_id)
        self._flight(kind, session)
        self._tool_flight(kind, session, data)
        row: dict[str, object] | None = None
        if kind == _ASSISTANT_MESSAGE:
            self.calls += 1
            row = self._assistant(event, data)
        elif kind == _ASSISTANT_ATTEMPT:
            self.calls += 1
        elif kind == _COMPACTION_SUMMARY:
            row = self._compaction(event, data)
        elif kind == "tool/result":
            row = _tool_result(data)
        elif kind == "llm/retry":
            row = _retry(data)
        elif kind == _TURN_END and session == self.session_id:
            reason = data.get("reason")
            error = reason.get("error") if isinstance(reason, dict) else None
            self.failure = error if isinstance(error, dict) else {}
        if row is not None:
            if session != self.session_id:
                row["parent_tool_use_id"] = session
            self.rows.write({**row, "session_id": self.session_id})

    def in_flight(self) -> dict[str, float]:
        """How long the oldest model call still in flight has waited for its reply (:data:`MODEL_REQUEST`), and the
        oldest tool call for its result (:data:`TOOL_CALL`); neither when none is."""
        with self._lock:
            now = self.clock()
            started = {MODEL_REQUEST: self._in_flight.values(), TOOL_CALL: self._tools.values()}
            return {what: now - min(since) for what, since in started.items() if since}

    def ended(self) -> None:
        with self._lock:
            self._in_flight.clear()
            self._tools.clear()

    def _flight(self, kind: str, session: str) -> None:
        with self._lock:
            if kind in _CALL_OPENS:
                self._in_flight[session] = self.clock()
            elif kind in _CALL_CLOSES:
                self._in_flight.pop(session, None)

    def _tool_flight(self, kind: str, session: str, data: Mapping[str, object]) -> None:
        message = data.get("message")
        with self._lock:
            if kind == _TOOL_STARTS:
                self._tools[session, str(data.get("callId") or "")] = self.clock()
            elif kind == _TOOL_RESULT and isinstance(message, dict):
                self._tools.pop((session, str(message.get("toolCallId") or "")), None)
            elif kind == _TURN_END:
                for call in [call for call in self._tools if call[0] == session]:
                    del self._tools[call]

    def _assistant(self, event: Mapping[str, object], data: Mapping[str, object]) -> dict[str, object]:
        message = data.get("message")
        message = message if isinstance(message, dict) else {}
        source = message.get("source")
        usage = self._spent(step_usage(event))
        return {
            "type": "assistant",
            "message": {
                "id": str(message.get("id") or ""),
                "type": "message",
                "role": "assistant",
                "model": str(source.get("model") or "") if isinstance(source, dict) else "",
                "content": _content(message.get("content")),
                "usage": usage,
            },
        }

    def _compaction(self, event: Mapping[str, object], data: Mapping[str, object]) -> dict[str, object] | None:
        """The row of a compaction that called a model, as a Claude CLI marks one, with what the call used: a model
        call of the turn that no reply of the conversation reports."""
        reported = compaction_usage(event)
        if reported is None:
            return None
        self.calls += 1
        return {
            "type": _SYSTEM,
            "subtype": "compact_boundary",
            "compact_metadata": {"trigger": "auto", "pre_tokens": data.get("shadowedTokenCount")},
            "model": str(data.get("model") or ""),
            "usage": self._spent(reported),
        }

    def _spent(self, reported: Mapping[str, int] | None) -> dict[str, int]:
        """Add what one model call ``reported`` to the turn's token counts; the counts a row carries of it."""
        usage = {name: count for name, count in (reported or {}).items() if name in _USAGE_NAMES}
        for name, count in usage.items():
            self.usage[name] = self.usage.get(name, 0) + count
        return usage


#: The token counts of a row, by the names an Anthropic ``usage`` block gives them.
_USAGE_NAMES = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def _content(blocks: object) -> list[dict[str, object]]:
    """The harness's content blocks as a Claude message's: text, reasoning as thinking, a tool call as a tool use."""
    out: list[dict[str, object]] = []
    for block in blocks if isinstance(blocks, list) else []:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == _TEXT:
            out.append({"type": _TEXT, "text": str(block.get("text") or "")})
        elif kind == "reasoning":
            out.append({"type": "thinking", "thinking": str(block.get("text") or "")})
        elif kind == "tool-call":
            out.append(
                {
                    "type": "tool_use",
                    "id": str(block.get("id") or ""),
                    "name": str(block.get("name") or ""),
                    "input": _arguments(block.get("arguments")),
                }
            )
    return out


def _arguments(raw: object) -> object:
    """A tool call's arguments, which the harness keeps as the JSON text the model wrote; text that is no JSON object
    stays as written."""
    if isinstance(raw, str):
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
    return {"arguments": raw}


def _text(blocks: object) -> str:
    return "".join(
        str(block.get("text") or "")
        for block in (blocks if isinstance(blocks, list) else [])
        if isinstance(block, dict) and block.get("type") == _TEXT
    )


def _tool_result(data: Mapping[str, object]) -> dict[str, object] | None:
    message = data.get("message")
    if not isinstance(message, dict):
        return None
    block = {
        "type": "tool_result",
        "tool_use_id": str(message.get("toolCallId") or ""),
        "content": _text(message.get("content")),
        "is_error": message.get("isError") is True,
    }
    return {"type": "user", "message": {"role": "user", "content": [block]}}


def _retry(data: Mapping[str, object]) -> dict[str, object]:
    failure = data.get("failure")
    failure = failure if isinstance(failure, dict) else {}
    return {
        "type": _SYSTEM,
        "subtype": "api_retry",
        "attempt": data.get("retry"),
        "max_retries": data.get("maxRetries"),
        "retry_delay_ms": data.get("delayMs"),
        "error_status": failure.get("status"),
        "error": str(failure.get("code") or ""),
    }


def result_row(turn: Turn, outcome: Any, seconds: float) -> dict[str, object]:
    """The row a turn ends with: ``outcome`` is its reply and how the runtime ended it
    (:class:`~hyperloom_common.llm.deepseek_session.DeepSeekSessionResult`), ``turn`` what its notifications added up
    to."""
    failure = turn.failure
    row: dict[str, object] = {
        "type": "result",
        "subtype": "error_during_execution" if outcome.error else "success",
        "is_error": bool(outcome.error),
        "duration_ms": int(seconds * 1000),
        "num_turns": turn.calls,
        "result": outcome.error or outcome.text,
        "session_id": turn.session_id,
        "usage": {name: turn.usage.get(name, 0) for name in _USAGE_NAMES},
    }
    if outcome.finish_reason == MAX_TOKENS_FINISH:
        row["stop_reason"] = MAX_TOKENS_STOP
    elif outcome.error and failure.get("code") == CONTEXT_EXCEEDED_CODE:
        row["stop_reason"] = CONTEXT_EXCEEDED_STOP
    status = failure.get("status")
    if outcome.error and isinstance(status, int) and not isinstance(status, bool):
        row["api_error_status"] = status
    return row


def prompt_of(line: str) -> str:
    """The text of one stream-json user message; a line that is none is the prompt as written."""
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return line.strip()
    message = row.get("message") if isinstance(row, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    return _text(content) if isinstance(content, list) else line.strip()


def _other_imports(env: Mapping[str, str]) -> list[str]:
    """The directories ``env`` names for imports besides this app's packages."""
    return [entry for entry in env.get(IMPORT_PATH_ENV, "").split(os.pathsep) if entry and Path(entry) != PACKAGES]


def bridge_env(env: Mapping[str, str]) -> dict[str, str]:
    """``env`` as the bridge is launched with it: this app's packages first among what its Python imports. A session's
    allowlisted environment does not say where they are, so a Python given ``env`` alone would run the copy it has
    installed, or none, where the app itself runs from a source tree."""
    return {**env, IMPORT_PATH_ENV: os.pathsep.join([str(PACKAGES), *_other_imports(env)])}


def _runtime_env(env: Mapping[str, str]) -> dict[str, str]:
    """``env`` for the runtime and the commands its tools run, which import as they did before :func:`bridge_env`. The
    SDK lays it over this process's environment, so the variable is set anew, to nothing when nothing else was named,
    and not left out."""
    return {**env, IMPORT_PATH_ENV: os.pathsep.join(_other_imports(env))}


def _home() -> Path:
    value = os.environ.get(HOME_ENV, "").strip()
    if not os.path.isabs(value):
        raise Refused(f"{HOME_ENV} must name the runtime's home, an absolute path the launcher sets")
    return Path(value)


def _mcp_servers(config: str | None) -> list[dict[str, str]]:
    if not config:
        return []
    with contextlib.suppress(OSError, json.JSONDecodeError):
        servers = json.loads(Path(config).read_text(encoding=_ENCODING)).get("mcpServers")
        return [{"name": str(name)} for name in servers] if isinstance(servers, dict) else []
    return []


def _launch(options: Options, home: Path, cwd: Path) -> HarnessLaunch:
    """What the runtime of ``options`` boots with, its home written: a specialist's with its role, its MCP servers
    and its session log uncompressed; a reading run's with :data:`READING_PATCH`."""
    effort = harness_reasoning_effort(DEFAULT_REASONING_EFFORT)
    try:
        if options.reading:
            home.mkdir(parents=True, mode=0o700, exist_ok=True)
            patch = write_text_nofollow(home, (READING_PATCH_NAME,), READING_PATCH, mode=0o600)
            overrides = {SYSTEM_PROMPT_ENV: options.role} if options.role else {}
            env = child_environment(overrides, READ_ONLY)
            profile = READING_PROFILE
        else:
            env = child_environment(None, FULL_ACCESS)
            instructions = f"{options.role.rstrip()}\n\n{WAITING_NOTE}\n" if options.role else ""
            request = DshSessionHome(home, options.model, instructions, options.mcp_config, UNATTENDED_OFF)
            boot = prepare_session_home(request, env)
            env.update(boot.env_additions)
            patch, profile = boot.patch, SESSION_PROFILE
    except (DeepSeekSessionError, SpecialistAgentUnavailableError, OSError) as error:
        raise Refused(str(error)) from error
    env = _runtime_env({**env, "DSH_TELEMETRY_MODE": "DISABLED"})
    return HarnessLaunch(home, cwd, options.model, effort, env, REQUEST_TIMEOUT_S, profile=profile, patches=(patch,))


class Bridge:
    """One runtime and the conversation held on it, as a CLI's turns."""

    def __init__(self, options: Options, rows: Rows, clock: Callable[[], float] = time.monotonic) -> None:
        self._options, self._rows, self._clock = options, rows, clock
        self._session_id = options.session_id or str(uuid.uuid4())
        self._harness: Any = None
        self._session: Any = None
        self._turn: Turn | None = None
        self._closed = threading.Event()

    def start(self, cwd: Path) -> None:
        """Boot the runtime and open the conversation, a resumed one restored; write ``init``.

        Raises:
            Refused: The runtime did not start, or it cannot restore the conversation to resume.
        """
        options = self._options
        launch = _launch(options, _home(), cwd)
        try:
            sdk = load_harness_sdk()
            self._harness = start_harness(sdk, launch)
            if options.resume:
                if not hasattr(self._harness, "resume_session"):
                    raise Refused("the installed DeepSeek Harness SDK cannot resume a conversation")
                self._session = self._harness.resume_session(self._session_id)
            else:
                self._session = self._harness.start_session(self._session_id)
        except DeepSeekSessionError as error:
            raise Refused(str(error)) from error
        except sdk_failures() as error:
            self.close()
            what = f"cannot resume conversation {self._session_id}" if options.resume else "the runtime did not start"
            raise Refused(f"{what}: {error}") from error
        threading.Thread(target=self._say_calls_in_flight, name="dsh-calls-in-flight", daemon=True).start()
        self._rows.write(
            {
                "type": _SYSTEM,
                "subtype": "init",
                "session_id": self._session_id,
                "cwd": str(cwd),
                "model": options.model,
                "harness": "dsh",
                "harness_version": dsh_sdk_version(),
                "mcp_servers": [] if options.reading else _mcp_servers(options.mcp_config),
            }
        )

    def turn(self, prompt: str) -> bool:
        """Run ``prompt`` as the conversation's next turn and write its rows; whether it ended without an error.

        Raises:
            Refused: The runtime failed or is gone; the conversation is no longer held here.
        """
        turn = self._turn = Turn(self._rows, self._session_id, self._clock)
        started = self._clock()
        try:
            done = self._session.run(prompt, on_notification=turn.note)
        except sdk_failures() as error:
            raise Refused(f"the runtime failed in a turn: {error}") from error
        finally:
            turn.ended()
        outcome = normalize_run_result(done, self._session_id)
        self._rows.write(result_row(turn, outcome, self._clock() - started))
        return not outcome.error

    def close(self) -> None:
        self._closed.set()
        harness, self._harness = self._harness, None
        if harness is not None:
            with contextlib.suppress(*sdk_failures()):
                harness.close()

    def _say_calls_in_flight(self) -> None:
        """Write a row for a model call or a tool call that has been in flight since the log's last row, every
        :data:`IN_FLIGHT_ROW_S`, for as long as it is vouched for."""
        while not self._closed.wait(IN_FLIGHT_ROW_S / 4):
            turn = self._turn
            if turn is None or self._clock() - self._rows.written_at < IN_FLIGHT_ROW_S:
                continue
            vouched = [(what, waited) for what, waited in turn.in_flight().items() if waited <= IN_FLIGHT_MAX_S]
            if vouched:
                what, waited = vouched[0]
                row = {"type": _SYSTEM, "subtype": what, "session_id": self._session_id, "waited_s": int(waited)}
                self._rows.write(row)


def _whole(stream: TextIO) -> Iterator[str]:
    """All that ``stream`` holds as one prompt, as a ``--print`` run takes it."""
    yield stream.read()


def _messages(stream: TextIO) -> Iterator[str]:
    """The prompt of each message ``stream`` brings, a line each."""
    for line in stream:
        if line.strip():
            yield prompt_of(line)


class Inbox:
    """The prompts stdin brings, read by a thread of its own from the start: a session writes its launch prompt
    before the runtime is up, and must not wait on a full pipe until it is."""

    def __init__(self, stream: TextIO, prompts: Callable[[TextIO], Iterator[str]]) -> None:
        self._brought: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._fill, args=(stream, prompts), name="dsh-stdin", daemon=True).start()

    def _fill(self, stream: TextIO, prompts: Callable[[TextIO], Iterator[str]]) -> None:
        # A stream closed under the reader as the process ends is the end of what it brings.
        with contextlib.suppress(OSError, ValueError):
            for prompt in prompts(stream):
                self._brought.put(prompt)
        self._brought.put(None)

    def __iter__(self) -> Iterator[str]:
        while (prompt := self._brought.get()) is not None:
            yield prompt


def parse(argv: Sequence[str] | None) -> Options:
    parser = argparse.ArgumentParser(prog=f"python -m {MODULE}", description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--model", default="", help="the harness model; DSH_MODEL's, or the default, when not given")
    parser.add_argument("--system-prompt-file", type=Path, help="the role, sent as the baseline instructions")
    conversation = parser.add_mutually_exclusive_group()
    conversation.add_argument("--session-id", default="", help="the id a fresh conversation is opened under")
    conversation.add_argument("--resume", default="", help="the id of the conversation to go on with")
    parser.add_argument("--mcp-config", help="a Claude-style MCP config whose servers the runtime connects")
    parser.add_argument("--print", dest="one_shot", action="store_true", help="run stdin as one prompt and exit")
    parser.add_argument("--read-only", dest="reading", action="store_true", help="boot the composition that only reads")
    args = parser.parse_args(argv)
    role = args.system_prompt_file.read_text(encoding=_ENCODING) if args.system_prompt_file else ""
    return Options(
        model=resolve_forge_llm_model(AGENT_BACKEND_DEEPSEEK, explicit=args.model),
        role=role,
        session_id=args.resume or args.session_id,
        resume=bool(args.resume),
        mcp_config=args.mcp_config,
        one_shot=args.one_shot,
        reading=args.reading,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the messages stdin brings as turns of one conversation until it ends, or, for ``--print``, the one prompt
    it holds; the exit code."""
    options = parse(argv)
    inbox = Inbox(sys.stdin, _whole if options.one_shot else _messages)
    bridge = Bridge(options, Rows(sys.stdout))
    try:
        bridge.start(Path.cwd())
        for prompt in inbox:
            succeeded = bridge.turn(prompt)
            if options.one_shot:
                return 0 if succeeded else 1
        return 0
    except Refused as refused:
        print(f"dsh: {refused}", file=sys.stderr, flush=True)
        return 1
    finally:
        bridge.close()


__all__ = [
    "HOME_ENV",
    "IMPORT_PATH_ENV",
    "IN_FLIGHT_MAX_S",
    "IN_FLIGHT_ROW_S",
    "MODULE",
    "PACKAGES",
    "READING_PATCH",
    "UNATTENDED_OFF",
    "WAITING_NOTE",
    "Bridge",
    "Inbox",
    "Options",
    "Refused",
    "Rows",
    "Turn",
    "bridge_env",
    "main",
    "parse",
    "prompt_of",
    "result_row",
]


if __name__ == "__main__":
    sys.exit(main())
