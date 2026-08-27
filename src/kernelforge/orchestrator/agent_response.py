"""Validate normalized read-only agent responses before routing them."""

from __future__ import annotations

import logging
import re

from kernelforge.agent_backends import AgentRunResult
from kernelforge.agent_backends.session_resume import is_api_failure


log = logging.getLogger(__name__)

# The block a provider CLI prepends after compacting a session that ran out of
# context: a recap of everything so far, then an instruction to carry on. Both
# ends are fixed strings the CLI writes, so the block can be removed exactly
# rather than by guessing where the answer resumes.
_COMPACTION_BLOCK = re.compile(
    r"This session is being continued from a previous conversation that ran "
    r"out of context\..*?"
    r"Pick up the last task as if the break never happened\.[ \t]*\n?",
    re.DOTALL,
)


class AgentResponseInfrastructureError(RuntimeError):
    """Report a provider failure that produced no usable model answer."""


class AgentResponseIncompleteError(ValueError):
    """Report a model answer cut short by a caller-controlled limit."""


def _without_compaction_recap(text: str, *, role: str) -> str:
    """Drop a session recap the provider prepended to this answer.

    A planning session reads source across many turns, so a long one exhausts
    its context window and the CLI compacts it. What comes back is the recap
    followed by the answer, and publishing both hands the Implementer a hundred
    lines of conversation history before the plan it is meant to execute.

    Only a block with both of its fixed ends is removed. Without the terminator
    the boundary would be a guess, and a wrong guess takes the answer with it.

    Reported rather than removed quietly: everything read before the compaction
    reaches the answer only through a summary of it, which is worth knowing when
    the answer disappoints.
    """
    stripped = _COMPACTION_BLOCK.sub("", text, count=1)
    if stripped == text:
        return text
    log.warning(
        "%s ran out of context and was compacted; its recap was dropped and the answer it went on to give was kept",
        role,
    )
    return stripped.strip()


def validated_agent_text(
    result: AgentRunResult,
    *,
    role: str,
    allow_empty: bool = False,
    allow_incomplete: bool = False,
) -> str:
    """Return complete response text or classify why it is unusable."""
    if is_api_failure(result):
        detail = result.stderr_tail or result.end_reason or f"{role} backend failed before producing an answer"
        raise AgentResponseInfrastructureError(str(detail))

    text = _without_compaction_recap(str(result.text or "").strip(), role=role)
    end_reason = str(result.end_reason or "").strip()
    if end_reason in {"turn_cap", "timeout"} and not allow_incomplete:
        raise AgentResponseIncompleteError(f"{role} session ended before completing its answer: {end_reason}")
    if "[session ended with sdk error:" in text.lower() and not allow_incomplete:
        raise AgentResponseIncompleteError(f"{role} session returned a truncated SDK error transcript")
    if not text and not allow_empty:
        raise AgentResponseIncompleteError(f"{role} returned no answer")
    return text
