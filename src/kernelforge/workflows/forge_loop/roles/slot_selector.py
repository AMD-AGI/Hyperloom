# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only agent that decides where a tree-search round expands the tree."""

from __future__ import annotations

import asyncio
import re
import shlex
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from kernelforge.agent_backends import (
    READ_ONLY_GIT_SUBCOMMANDS,
    AgentBackend,
    AgentHook,
    AgentHooks,
    AgentProviderError,
    AgentRunSpec,
    AgentToolPolicy,
    watchdog_timeout_sec,
)
from kernelforge.agent_backends.structured_output import AgentAnswer, StructuredOutputError
from kernelforge.workflows.forge_loop.roles.agent_response import (
    AgentResponseIncompleteError,
    AgentResponseInfrastructureError,
    validated_agent_answer,
)

SELECTION_LABEL = "slot selection"

_SYSTEM_PROMPT = """\
You decide where the next round of an autonomous GPU-kernel tree search expands
the tree. Every node is a validated, measured kernel version held as a git
commit, faster or slower than its parent. A direction is a named optimization
route opened at a node. This round runs up to {lanes} expansions in parallel,
one Implementer session each; you choose them.

Each expansion (a slot) is one of:
- DEEPEN: continue an existing direction from one of its nodes.
- BRANCH: open a new direction at a node. You choose the node; the planners at
  that node choose the direction from their evidence. You may give a hint,
  which they must adopt or refute with evidence.
- PORT: carry gains another node measured onto a start node (for example the
  tuning of one route onto the structure of another).

Your goal is the highest global best the campaign can reach in the time it has
left. Weigh where measured headroom remains, which routes have stopped paying,
and which slower nodes carry a structure that could lead past the best. Several
slots may share a start node. Spending every slot on the same node is allowed,
but say why.

You are read-only. Read the facts below and, when it helps, the archive,
lessons and code they point at, and inspect nodes with read-only git, one
command at a time (git log, git show, git diff).

Name every node by its full commit or an unambiguous prefix of at least seven
characters. A field a slot's type does not use stays empty: `direction` is
DEEPEN's, `hint` is BRANCH's, and `source` and `gains` are PORT's.
"""


class SlotAnswer(AgentAnswer):
    """One expansion of the round: what it does, where it starts, and why."""

    slot_type: Literal["DEEPEN", "BRANCH", "PORT"] = Field(description="What the expansion does with its start node.")
    start: str = Field(description="The node the expansion starts from.")
    reason: str = Field(description="Why this expansion, from the facts and what you read.")
    direction: str = Field(description="DEEPEN: the id of the direction to continue. Empty for any other slot.")
    hint: str = Field(
        description="BRANCH: a direction the planners must adopt or refute with evidence, when you have one. Empty otherwise."
    )
    source: str = Field(description="PORT: the node whose measured gains are carried. Empty for any other slot.")
    gains: str = Field(description="PORT: the gains to carry onto the start node. Empty for any other slot.")


class DirectionStatusAnswer(AgentAnswer):
    """One direction closed, so it is no longer deepened, or reopened."""

    direction: str = Field(description="The id of a direction of the tree.")
    status: Literal["OPEN", "CLOSED"] = Field(description="CLOSED stops the direction being deepened; OPEN allows it.")
    reason: str = Field(description="Why the direction is closed or reopened.")


class SlotSelectionAnswer(AgentAnswer):
    """Where one round expands the tree, and what it measures or reads again."""

    rationale: str = Field(
        description=(
            "Why these slots, in free-form Markdown: where measured headroom remains, which routes stopped paying, and "
            "why any node carries several slots."
        )
    )
    slots: list[SlotAnswer] = Field(description="The round's expansions, at most one per lane, in planning order.")
    remeasure: list[str] = Field(description="Nodes to measure again this round because their score is in doubt.")
    refresh_analysis: list[str] = Field(
        description="Nodes whose analysis should be refreshed before the round plans from them again."
    )
    direction_status: list[DirectionStatusAnswer] = Field(description="Directions to close or reopen.")


# Shell syntax that would let one command run another, write a file, or expand into something unread.
_SHELL_SYNTAX = re.compile(r"[;&|<>`$()\\\n]")
# Global options that may precede the subcommand; every other one (``-c``, ``--exec-path``, ...) can run code.
_GLOBAL_GIT_OPTIONS = frozenset({"--no-pager", "-P"})
# Subcommand options that write a file or run a configured external program.
_WRITING_GIT_OPTIONS = ("--output", "--ext-diff", "--textconv", "--open-files-in-pager", "-O")


def read_only_git_denial(command: str) -> str:
    """Why ``command`` is not one read-only git invocation, or "" when it is."""
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
        subcommand = rest[0] if rest else ""
        return f"git {subcommand} is not one of: {', '.join(sorted(READ_ONLY_GIT_SUBCOMMANDS))}"
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
            "permissionDecisionReason": f"The slot selector may only run read-only git commands: {denial}.",
        }
    }


@dataclass(frozen=True)
class SelectorAnswer:
    """The selector's answer, or why there is none."""

    answer: SlotSelectionAnswer | None = None
    failure: str = ""


class SlotSelector:
    """One read-only session per round that names the round's slots."""

    def __init__(self, *, backend: AgentBackend, timeout_sec: int, max_turns: int) -> None:
        self.backend = backend
        self.timeout_sec = int(timeout_sec)
        self.max_turns = int(max_turns)

    async def select(self, *, workspace: str, facts: str, lanes: int, usage=None) -> SelectorAnswer:
        """Ask for this round's slots; a failure is reported, never raised."""
        spec = self._spec(workspace=workspace, facts=facts, lanes=lanes)
        try:
            result = await asyncio.wait_for(
                self.backend.run(spec, usage=usage), timeout=watchdog_timeout_sec(self.timeout_sec)
            )
            return SelectorAnswer(answer=validated_agent_answer(result, SlotSelectionAnswer, role=SELECTION_LABEL))
        except (
            AgentProviderError,
            AgentResponseIncompleteError,
            AgentResponseInfrastructureError,
            StructuredOutputError,
            asyncio.TimeoutError,
        ) as error:
            return SelectorAnswer(failure=f"{type(error).__name__}: {error}")

    def _spec(self, *, workspace: str, facts: str, lanes: int) -> AgentRunSpec:
        runs_hooks = bool(getattr(getattr(self.backend, "capabilities", None), "stop_hooks", False))
        # A shell only where read-only git can be enforced: by hooks, or by the provider's own git guard.
        shell = runs_hooks or getattr(self.backend, "name", "") == "codex"
        return AgentRunSpec(
            system_prompt=_SYSTEM_PROMPT.format(lanes=lanes),
            user_prompt=facts,
            cwd=workspace,
            writable=False,
            timeout_sec=self.timeout_sec,
            role="slot_selector",
            tool_policy=AgentToolPolicy(read=True, search=True, write=False, shell=shell, max_turns=self.max_turns),
            protected_globs=["*"],
            hooks=self._hooks() if runs_hooks else None,
            output_schema=SlotSelectionAnswer,
        )

    @staticmethod
    def _hooks() -> AgentHooks:
        return AgentHooks(pre_tool_use=[AgentHook(matcher="Bash", callback=_allow_only_read_only_git)])


__all__ = [
    "SELECTION_LABEL",
    "DirectionStatusAnswer",
    "SelectorAnswer",
    "SlotAnswer",
    "SlotSelectionAnswer",
    "SlotSelector",
    "read_only_git_denial",
]
