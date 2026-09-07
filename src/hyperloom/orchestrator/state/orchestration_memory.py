"""Orchestration working memory, captured at a macro-cycle boundary.

Turns the agent's own summary of the finished cycle into the structured
record on ``SharedState.orchestration_memory``, whose
``next_cycle_directive`` is injected into the next cycle's system prompt.
Pure helpers; the phase handler owns the IO.
"""

from __future__ import annotations

import functools
import json
import re
from typing import Any

from hyperloom.common.timeutil import now_iso


# List threads that carry forward when a reply omits them (``learnings``
# accumulates separately).
_MEMORY_LIST_KEYS: tuple[str, ...] = ("hypotheses", "tried_and_why", "pending")

# seconds + ``+00:00`` (canonical helper; kept importable for callers).
_now_iso = functools.partial(now_iso, "seconds")

# Max byte length for next_cycle_directive before truncation.
_DIRECTIVE_MAX_LEN: int = 1500

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

# Sent as its own turn at the macro-cycle boundary to elicit the record (parsed as JSON).
MEMORY_REQUEST_PROMPT: str = """\
=== MACRO-CYCLE HANDOFF ===
This macro-cycle is ending. From the session state above, write the working
memory the next cycle should start from. Do NOT call any tool for this turn —
reply with a single fenced JSON object and nothing else:

```json
{
  "current_plan": "<1-3 sentences: what you are driving toward right now>",
  "hypotheses": ["<open hypothesis you still want to test>", "..."],
  "tried_and_why": ["<what you tried + outcome + why it mattered>", "..."],
  "pending": ["<thread you have not closed yet>", "..."],
  "learnings": ["<durable lesson from this session so far>", "..."],
  "next_cycle_directive": "<1-3 sentences for the NEXT macro-cycle: which bottleneck to attack, what to deprioritise, breadth vs depth posture, priority specialist domains. Leave empty string if no new cycle is expected.>"
}
```

Keep it tight (a few items per list). The authoritative session facts are
re-projected every turn, so capture intent and rationale here, not raw
numbers.
"""


def _sanitize_cycle_directive(raw: str) -> str:
    """Return ``raw`` if it passes safety checks, else empty string."""
    text = raw.strip()[:_DIRECTIVE_MAX_LEN]
    lower = text.lower()
    if any(phrase in lower for phrase in _DIRECTIVE_POLICY_BLACKLIST):
        return ""
    return text


def parse_memory_reply(raw_text: str) -> dict[str, Any]:
    """Parse the agent's handoff reply into the memory schema; never raises (malformed replies carry ``parse_error``)."""
    obj = _extract_json_object(raw_text)
    if obj is None:
        return {
            "current_plan": (raw_text or "").strip()[:1000],
            "hypotheses": [],
            "tried_and_why": [],
            "pending": [],
            "learnings": [],
            "next_cycle_directive": "",
            "parse_error": "no JSON object found in memory reply",
        }
    out: dict[str, Any] = {}
    out["current_plan"] = str(obj.get("current_plan") or "").strip()
    for key in ("hypotheses", "tried_and_why", "pending", "learnings"):
        val = obj.get(key)
        if isinstance(val, list):
            out[key] = [str(x).strip() for x in val if str(x).strip()]
        elif val:
            out[key] = [str(val).strip()]
        else:
            out[key] = []
    out["next_cycle_directive"] = _sanitize_cycle_directive(str(obj.get("next_cycle_directive") or ""))
    return out


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object embedded in free-form text."""
    if not text:
        return None
    # Prefer a fenced ```json ... ``` block.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        # Fall back to the first balanced-looking span.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def build_memory_record(
    parsed: dict[str, Any],
    *,
    tick: int,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the persisted ``orchestration_memory`` record.

    ``learnings`` accumulate across cycles (deduped, capped); the other fields
    carry the prior value forward when a reply omits them, so one forgetful
    reply never blanks an in-flight plan.
    """
    prev = previous or {}
    learnings = list(prev.get("learnings") or [])
    for item in parsed.get("learnings") or []:
        if item not in learnings:
            learnings.append(item)
    learnings = learnings[-50:]  # cap so state.json stays bounded
    # Non-empty-wins: an empty field inherits the previous record's value.
    plan = str(parsed.get("current_plan") or "").strip() or prev.get("current_plan", "")
    directive = str(parsed.get("next_cycle_directive") or "").strip() or str(prev.get("next_cycle_directive") or "")
    record: dict[str, Any] = {
        "current_plan": plan,
        "learnings": learnings,
        "next_cycle_directive": directive,
        "last_capture_tick": int(tick),
        "last_capture_ts": _now_iso(),
        "capture_count": int(prev.get("capture_count", 0)) + 1,
        "parse_error": parsed.get("parse_error", ""),
    }
    for key in _MEMORY_LIST_KEYS:
        record[key] = parsed.get(key) or prev.get(key) or []
    return record


__all__ = [
    "MEMORY_REQUEST_PROMPT",
    "_DIRECTIVE_MAX_LEN",
    "_DIRECTIVE_POLICY_BLACKLIST",
    "_MEMORY_LIST_KEYS",
    "_sanitize_cycle_directive",
    "build_memory_record",
    "parse_memory_reply",
]
