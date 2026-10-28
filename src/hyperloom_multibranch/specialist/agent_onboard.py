# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The onboarding model of an autonomous branch's preflight: one headless run of the specialist's Claude CLI.

The CLI reads the agent's checkout and answers; it gets no MCP server and only the tools that read: the file tools and a
few shell commands that print. A third-party checkout is untrusted, so nothing the CLI runs may write or execute what
the checkout holds, nor read its settings, whose hooks would run its code. Each run streams to a log of its own beside
the checkout, which is what its spend is read from (:mod:`llm_usage`, as a specialist launch's is) and what a reader
goes to when onboarding failed. A run that exits non-zero, outlives its timeout or answers with an error raises
:class:`~hyperloom_multibranch.specialist.agent_onboarding.AttemptFailed`, so it fails that attempt and the next one
asks again; its spend is the branch's either way, a run cut short included. Codex has no read-only headless mode this
module drives, so it is refused.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import Mapping
from pathlib import Path

from hyperloom.common.llm_config import AGENT_BACKEND_CLAUDE
from hyperloom_multibranch.contracts.blackboard.interfaces import UsageSink
from hyperloom_multibranch.contracts.blackboard.records import LlmUsage
from hyperloom_multibranch.contracts.blackboard.vocab import UsageSource
from hyperloom_multibranch.specialist.agent_onboarding import ENCODING, AttemptFailed, Onboard, tail
from hyperloom_multibranch.specialist.agent_watch import MARKER, marked_pids
from hyperloom_multibranch.specialist.agent_session import AgentCommand
from hyperloom_multibranch.specialist.llm_usage import launch_cost, launch_usage

ONBOARD_SECONDS = 900.0
LOG_PREFIX = "onboard-"
#: The built-in tools the CLI has at all; what Bash may run is :data:`ALLOWED_TOOLS`.
TOOLS = ("Read", "Glob", "Grep", "Bash")
#: Shell commands that only print. ``find`` (``-delete``, ``-exec``) and ``git log``/``git show`` (``--output``) are not
#: among them: an option of theirs writes or runs.
ALLOWED_TOOLS = (
    "Read",
    "Glob",
    "Grep",
    "Bash(ls:*)",
    "Bash(cat:*)",
    "Bash(head:*)",
    "Bash(tail:*)",
    "Bash(wc:*)",
    "Bash(git ls-files:*)",
)


def onboard_argv(command: AgentCommand) -> tuple[str, ...]:
    """The command line of one headless, read-only run; the prompt goes on stdin."""

    if command.backend != AGENT_BACKEND_CLAUDE:
        raise ValueError(f"onboarding runs the claude CLI headless and read-only, not {command.backend!r}")
    model = ("--model", command.model) if command.model else ()
    return (
        command.executable,
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        *model,
        "--tools",
        ",".join(TOOLS),
        "--allowedTools",
        ",".join(ALLOWED_TOOLS),
        # Anything not allowed above is denied, never asked: nobody answers a prompt of a headless run.
        "--permission-mode",
        "dontAsk",
        # With no --mcp-config this loads no MCP server at all, not even the user's.
        "--strict-mcp-config",
        # No settings file is read, so no hook the checkout ships runs, and no wrapper can bypass the permissions.
        "--restricted",
        "--no-session-persistence",
    )


def cli_onboard(
    command: AgentCommand, env: Mapping[str, str], branch: str, usage: UsageSink, *, seconds: float = ONBOARD_SECONDS
) -> Onboard:
    """An :data:`~hyperloom_multibranch.specialist.agent_onboarding.Onboard` that runs ``command`` once per question,
    in the checkout, with ``env``, for at most ``seconds``, and records its spend as ``branch``'s in ``usage``."""

    argv = onboard_argv(command)

    async def onboard(checkout: Path, prompt: str) -> str:
        log = _next_log(checkout.parent)
        try:
            returncode = await asyncio.wait_for(_run(argv, checkout, env, prompt, log), seconds)
        except asyncio.TimeoutError as error:
            raise AttemptFailed(f"the onboarding CLI did not answer within {seconds:.0f} s") from error
        finally:
            await _record(log, branch, command.model, usage)
        result = await asyncio.to_thread(final_result, log)
        if returncode != 0:
            raise AttemptFailed(f"the onboarding CLI exited {returncode}: {tail(_text(log))}")
        if result is None:
            raise AttemptFailed(f"the onboarding CLI printed no result: {tail(_text(log))}")
        if result.get("is_error"):
            raise AttemptFailed(f"the onboarding CLI ended with an error: {tail(str(result.get('result')))}")
        return str(result.get("result") or "")

    return onboard


def final_result(log: Path) -> Mapping[str, object] | None:
    """The result event a stream-json ``log`` ends with; ``None`` when the CLI wrote none."""

    for line in reversed(_text(log).splitlines()):
        if '"type"' not in line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("type") == "result":
            return row
    return None


async def _record(log: Path, branch: str, model: str, usage: UsageSink) -> None:
    """Record what the run that streamed to ``log`` used as ``branch``'s, a run cut short included."""

    used = await asyncio.to_thread(launch_usage, log)
    if used.tokens:
        cost = launch_cost(used.tokens, model)
        await usage.record_llm_usage(LlmUsage(branch, UsageSource.SPECIALIST, cost, used.total, used.session_id))


async def _run(argv: tuple[str, ...], cwd: Path, env: Mapping[str, str], prompt: str, log: Path) -> int:
    marked = {**env, MARKER: str(log)}
    with log.open("wb") as out:
        process = await asyncio.create_subprocess_exec(
            *argv, cwd=cwd, env=marked, stdin=asyncio.subprocess.PIPE, stdout=out, stderr=out, start_new_session=True
        )
        try:
            await process.communicate(prompt.encode(ENCODING))
        finally:
            # The CLI runs each shell command in a session of its own; a run cut short must leave none behind.
            if process.returncode is None:
                _kill(marked_pids(str(log)))
                await process.wait()
    return process.returncode or 0


def _next_log(directory: Path) -> Path:
    taken = len(list(directory.glob(f"{LOG_PREFIX}*.jsonl")))
    return directory / f"{LOG_PREFIX}{taken + 1}.jsonl"


def _text(log: Path) -> str:
    return log.read_text(encoding=ENCODING, errors="replace")


def _kill(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue


__all__ = ["ALLOWED_TOOLS", "LOG_PREFIX", "ONBOARD_SECONDS", "TOOLS", "cli_onboard", "final_result", "onboard_argv"]
