# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only agent that chooses the commit a tree-search iteration starts from."""

from __future__ import annotations

import asyncio
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass

from kernelforge.agent_backends import (
    AgentBackend,
    AgentHook,
    AgentHooks,
    AgentProviderError,
    AgentRunSpec,
    AgentToolPolicy,
    create_registered_backend,
    watchdog_timeout_sec,
)
from kernelforge.config import Config
from kernelforge.tools.git import READ_ONLY_GIT_SUBCOMMANDS
from kernelforge.orchestrator.agent_response import (
    AgentResponseIncompleteError,
    AgentResponseInfrastructureError,
    validated_agent_text,
)

START_SELECTOR_TIMEOUT_SEC = 300
START_SELECTOR_MAX_TURNS = 30
_REASON_MAX_CHARS = 2000

_START_COMMIT_PATTERN = re.compile(r"(?im)^[ \t>*`]*START_COMMIT[ \t*`]*:[ \t*`]*([0-9a-f]{7,40})\b")

_SYSTEM_PROMPT = """\
You choose where the next iteration of an autonomous GPU-kernel tree search
starts. Every node of the tree is a validated, measured kernel version kept as
a git commit; each branch is one optimization route. The iteration you start
will plan and implement one change on top of the commit you name, and its
candidate is kept only if it beats that commit's score.

Read the tree view, and when it helps, the archived attempts and lessons it
points at. Every node is a commit: inspect how solutions differ with
read-only git, one command at a time, e.g. `git log --oneline <branch>`,
`git show <commit>`, `git diff <parent> <commit>`. Weigh where measured
headroom is left: a strong node whose directions are not yet exhausted, a
route that has stopped producing nodes, a weaker node whose untried direction
could lead past the current best. Repeated failures from one node are evidence
against more of the same there; a node no one has built on is unexplored, not
unpromising.

You are read-only. Do not edit files, benchmark, or run anything but
read-only git commands.

End your answer with exactly one line naming the start, using a commit from
the tree view:
START_COMMIT: <commit>
"""


@dataclass(frozen=True)
class StartSelection:
    """The start the selector named, its whole answer and the reasoning in it, or the reason it named none."""

    commit: str = ""
    reason: str = ""
    failure: str = ""
    answer: str = ""


# Shell syntax that would let one command run another, write a file, or expand into something unread.
_SHELL_SYNTAX = re.compile(r"[;&|<>`$()\\\n]")
# Global options that may precede the subcommand; every other one (``-c``, ``--exec-path``, ...) can run code.
_GLOBAL_GIT_OPTIONS = frozenset({"--no-pager", "-P"})
# Subcommand options that write a file or run a configured external program.
_WRITING_GIT_OPTIONS = ("--output", "--ext-diff", "--textconv", "--open-files-in-pager", "-O")


def read_only_git_denial(command: str) -> str:
    """Why ``command`` is not one read-only git invocation, or \"\" when it is."""
    if _SHELL_SYNTAX.search(command):
        return "run one git command at a time, without pipes, redirection, chaining or substitution"
    try:
        tokens = shlex.split(command)
    except ValueError:
        return "the command could not be parsed"
    if not tokens or tokens[0] != "git":
        return "only read-only git commands may run in this session"
    rest = tokens[1:]
    while rest and (rest[0] in _GLOBAL_GIT_OPTIONS or rest[0] == "-C"):
        rest = rest[2:] if rest[0] == "-C" else rest[1:]
    if not rest or rest[0] not in READ_ONLY_GIT_SUBCOMMANDS:
        return f"git {rest[0] if rest else ''} is not one of: {', '.join(sorted(READ_ONLY_GIT_SUBCOMMANDS))}".strip()
    writing = [token for token in rest[1:] if token.split("=", 1)[0] in _WRITING_GIT_OPTIONS]
    if writing:
        return f"{writing[0]} writes a file or runs an external program"
    return ""


async def _allow_only_read_only_git(input_data: dict, _tool_use_id, _context) -> dict:
    if input_data.get("tool_name") != "Bash":
        return {}
    denial = read_only_git_denial(str((input_data.get("tool_input") or {}).get("command") or ""))
    if not denial:
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": f"The start selector may only run read-only git commands: {denial}.",
        }
    }


def parse_start_commit(text: str) -> str:
    """The commit token on the last ``START_COMMIT:`` line, or \"\"."""
    matches = _START_COMMIT_PATTERN.findall(text or "")
    return matches[-1].lower() if matches else ""


def _reason_of(text: str) -> str:
    kept = [line for line in text.splitlines() if not _START_COMMIT_PATTERN.match(line)]
    return "\n".join(kept).strip()[-_REASON_MAX_CHARS:]


class StartSelectorAgent:
    """One bounded, read-only selection session per iteration."""

    def __init__(
        self,
        *,
        backend: AgentBackend,
        timeout_sec: int = START_SELECTOR_TIMEOUT_SEC,
        max_turns: int = START_SELECTOR_MAX_TURNS,
    ) -> None:
        if timeout_sec <= 0 or max_turns <= 0:
            raise ValueError("start selector limits must be positive")
        self.backend = backend
        self.timeout_sec = timeout_sec
        self.max_turns = max_turns

    async def select(
        self,
        *,
        workspace: str,
        tree_view: str,
        resolve: Callable[[str], str],
        usage=None,
    ) -> StartSelection:
        """Ask for a start; ``resolve`` maps a commit token to a node of the tree, or \"\" when it names none."""
        try:
            result = await asyncio.wait_for(
                self.backend.run(self._spec(workspace, tree_view), usage=usage),
                timeout=watchdog_timeout_sec(self.timeout_sec),
            )
            text = validated_agent_text(result, role="start selector", allow_incomplete=True)
        except (
            asyncio.TimeoutError,
            AgentProviderError,
            AgentResponseInfrastructureError,
            AgentResponseIncompleteError,
        ) as error:
            return StartSelection(failure=f"{type(error).__name__}: {error}"[:_REASON_MAX_CHARS])
        token = parse_start_commit(text)
        if not token:
            return StartSelection(reason=_reason_of(text), failure="the answer named no START_COMMIT", answer=text)
        commit = resolve(token)
        if not commit:
            failure = f"START_COMMIT {token} is not a node of the tree"
            return StartSelection(reason=_reason_of(text), failure=failure, answer=text)
        return StartSelection(commit=commit, reason=_reason_of(text), answer=text)

    def _spec(self, workspace: str, tree_view: str) -> AgentRunSpec:
        """A read-only session; a provider that runs hooks gets a shell confined to read-only git by one."""
        confined_shell = self.backend.capabilities.stop_hooks
        return AgentRunSpec(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=tree_view,
            role="start selector",
            cwd=workspace,
            writable=False,
            timeout_sec=self.timeout_sec,
            tool_policy=AgentToolPolicy(
                read=True, search=True, write=False, shell=confined_shell, max_turns=self.max_turns
            ),
            protected_globs=["*"],
            hooks=(
                AgentHooks(pre_tool_use=[AgentHook(matcher="Bash", callback=_allow_only_read_only_git)])
                if confined_shell
                else None
            ),
        )


def make_start_selector(*, config: Config, usage=None) -> StartSelectorAgent:
    """The selector on the profile's own provider and model."""
    backend = create_registered_backend(config.agent_runtime(), preflight=False, usage=usage)
    return StartSelectorAgent(backend=backend)


__all__ = [
    "START_SELECTOR_MAX_TURNS",
    "START_SELECTOR_TIMEOUT_SEC",
    "StartSelection",
    "StartSelectorAgent",
    "make_start_selector",
    "parse_start_commit",
]
