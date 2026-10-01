"""The orchestration agent's directive for the next macro-cycle.

A SWEEP-phase turn asks Orchestration how the next cycle should open; the reply
becomes ``SharedState.orchestration_memory``, whose ``next_cycle_directive`` is
injected into the next cycle's system prompt. Pure helpers; the SWEEP pump owns
the turn.
"""

from __future__ import annotations

from typing import Any

from hyperloom.inference_optimizer.protocol.intent import IntentType

# Max byte length for next_cycle_directive before truncation.
_DIRECTIVE_MAX_LEN: int = 1500

# Topic of the ``send_message`` intent that carries the directive when the transport only accepts intent envelopes.
DIRECTIVE_TOPIC = "cycle_directive"

# Phrases that indicate the LLM is trying to embed policy overrides in the directive.
_DIRECTIVE_POLICY_BLACKLIST: tuple[str, ...] = (
    "ignore phase",
    "bypass policy",
    "allowed actions",
    "phase contract",
    "skip phase",
    "override policy",
    "ignore policy",
)

# Appended to an ordinary Orchestration turn while SWEEP is open and another cycle is feasible.
CYCLE_DIRECTIVE_REQUEST: str = f"""\
=== MACRO-CYCLE HANDOFF ===
This macro-cycle is ending. Reply in 1-3 plain sentences with the directive the
NEXT macro-cycle should open on: which bottleneck to attack, what to
deprioritise, breadth vs depth posture, priority specialist domains. Do not
call any tool for this turn. If your output format requires an intent envelope,
send one `send_message` intent with `topic` `{DIRECTIVE_TOPIC}` and the
directive in `body`.
"""


def _sanitize_cycle_directive(raw: str) -> str:
    """Return ``raw`` if it passes safety checks, else empty string."""
    text = raw.strip()[:_DIRECTIVE_MAX_LEN]
    lower = text.lower()
    if any(phrase in lower for phrase in _DIRECTIVE_POLICY_BLACKLIST):
        return ""
    return text


def build_cycle_memory(result: Any, *, cycle: int) -> dict[str, Any]:
    """Build ``orchestration_memory`` from the handoff turn's reply (``None`` when the turn produced none)."""
    text = ""
    if result is not None:
        text = next(
            (
                str(i.payload.get("body") or "")
                for i in result.intents
                if i.type is IntentType.SEND_MESSAGE and i.payload.get("topic") == DIRECTIVE_TOPIC
            ),
            result.raw_text or "",
        )
    directive = _sanitize_cycle_directive(text)
    parse_error = "" if directive else "no usable directive in the handoff reply"
    return {"next_cycle_directive": directive, "for_cycle": int(cycle), "parse_error": parse_error}


__all__ = [
    "CYCLE_DIRECTIVE_REQUEST",
    "DIRECTIVE_TOPIC",
    "_DIRECTIVE_MAX_LEN",
    "_DIRECTIVE_POLICY_BLACKLIST",
    "_sanitize_cycle_directive",
    "build_cycle_memory",
]
