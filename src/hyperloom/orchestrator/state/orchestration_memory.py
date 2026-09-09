"""Orchestration working-memory checkpoint / compaction."""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from typing import Any

from hyperloom.common.timeutil import now_iso


# Default checkpoint cadence.
DEFAULT_CHECKPOINT_EVERY_TICKS: int = 20
DEFAULT_CHECKPOINT_EVERY_MINUTES: float = 30.0
# Prompt+reply chars forcing a checkpoint regardless of cadence.
DEFAULT_CHECKPOINT_CHAR_BUDGET: int = 400_000

# Context-token guardrail, as a fraction of the model's window.
DEFAULT_CONTEXT_TOKEN_SOFT_FRACTION: float = 0.70
# Minimum ticks between two token-triggered compactions, so a re-seeded conversation is not compacted again before it
# reports a fresh level.
DEFAULT_CHECKPOINT_MIN_TICK_GAP: int = 3
# Conservative fallback window for an unknown model id.
DEFAULT_MODEL_CONTEXT_WINDOW: int = 200_000
# Keys must be lower-case with ``-`` separators; lookups are folded to that form.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-5": 200_000,
    "claude-opus-4-8": 200_000,
    "claude-opus-4-7": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
}


def context_window_for_model(model: str) -> int:
    """Context-window size (tokens) for a model id; conservative fallback if unknown."""
    key = (model or "").strip().lower().replace(".", "-").replace("_", "-")
    return MODEL_CONTEXT_WINDOWS.get(key, DEFAULT_MODEL_CONTEXT_WINDOW)


# List threads that carry forward when a checkpoint reply omits them (``learnings`` accumulates separately).
_MEMORY_LIST_KEYS: tuple[str, ...] = ("hypotheses", "tried_and_why", "pending")


# seconds + ``+00:00`` (canonical helper; kept importable for callers).
_now_iso = functools.partial(now_iso, "seconds")


@dataclass
class CheckpointPolicy:
    """When to take an orchestration-memory checkpoint."""

    every_ticks: int = DEFAULT_CHECKPOINT_EVERY_TICKS
    every_minutes: float = DEFAULT_CHECKPOINT_EVERY_MINUTES
    char_budget: int = DEFAULT_CHECKPOINT_CHAR_BUDGET
    # Context-token soft budget (absolute token count; 0 disables).
    context_token_soft: int = 0
    # Anti-thrash floor on the token trigger only (0 disables).
    min_tick_gap: int = DEFAULT_CHECKPOINT_MIN_TICK_GAP
    # Always checkpoint on a phase boundary.
    on_phase_boundary: bool = True

    def adopt_context_window(self, window: int, fraction: float) -> None:
        """Recompute the soft budget from a window the provider itself reported."""
        if window > 0:
            self.context_token_soft = int(window * fraction)

    def should_checkpoint(
        self,
        *,
        ticks_since_last: int,
        minutes_since_last: float,
        chars_since_last: int,
        phase_changed: bool,
        context_tokens_now: int = 0,
    ) -> bool:
        """Decide whether a checkpoint is due under this policy."""
        token_trigger_allowed = self.min_tick_gap <= 0 or ticks_since_last >= self.min_tick_gap
        if token_trigger_allowed and self.context_token_soft > 0 and context_tokens_now >= self.context_token_soft:
            return True
        if phase_changed and self.on_phase_boundary:
            return True
        if self.every_ticks > 0 and ticks_since_last >= self.every_ticks:
            return True
        if self.every_minutes > 0 and minutes_since_last >= self.every_minutes:
            return True
        if self.char_budget > 0 and chars_since_last >= self.char_budget:
            return True
        return False


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


# Appended as the next user turn to elicit the compact summary (parsed as JSON).
CHECKPOINT_REQUEST_PROMPT: str = """\
=== CHECKPOINT (compaction) ===
We are about to compact this conversation to keep it bounded. Summarise
YOUR OWN working memory so you can resume seamlessly from a fresh
conversation. Do NOT call any tool for this turn — reply with a single
fenced JSON object and nothing else:

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

Keep it tight (a few items per list). This snapshot — plus the
authoritative session facts — is all you will carry into the next
conversation, so capture intent and rationale, not raw numbers you can
re-pull from the context tools.
"""


def _sanitize_cycle_directive(raw: str) -> str:
    """Return ``raw`` if it passes safety checks, else empty string."""
    text = raw.strip()[:_DIRECTIVE_MAX_LEN]
    lower = text.lower()
    if any(phrase in lower for phrase in _DIRECTIVE_POLICY_BLACKLIST):
        return ""
    return text


def parse_checkpoint_reply(raw_text: str) -> dict[str, Any]:
    """Parse the agent's checkpoint reply into the memory schema."""
    obj = _extract_json_object(raw_text)
    if obj is None:
        return {
            "current_plan": (raw_text or "").strip()[:1000],
            "hypotheses": [],
            "tried_and_why": [],
            "pending": [],
            "learnings": [],
            "next_cycle_directive": "",
            "parse_error": "no JSON object found in checkpoint reply",
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


def is_degenerate_checkpoint(parsed: dict[str, Any]) -> bool:
    """True when a parsed checkpoint reply carries no usable working memory."""
    if str(parsed.get("parse_error") or "").strip():
        return True
    has_plan = bool(str(parsed.get("current_plan") or "").strip())
    has_lists = any(parsed.get(k) for k in _MEMORY_LIST_KEYS)
    return not (has_plan or has_lists)


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
    seq: int,
    tick: int,
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the persisted ``orchestration_memory`` record."""
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
        "last_checkpoint_seq": int(seq),
        "last_checkpoint_tick": int(tick),
        "last_checkpoint_ts": _now_iso(),
        "checkpoint_count": int(prev.get("checkpoint_count", 0)) + 1,
        "parse_error": parsed.get("parse_error", ""),
    }
    for key in _MEMORY_LIST_KEYS:
        record[key] = parsed.get(key) or prev.get(key) or []
    return record


def render_memory_for_seed(memory: dict[str, Any]) -> str:
    """Render an ``orchestration_memory`` record into prompt text."""
    if not memory:
        return ""
    lines: list[str] = ["=== Your working memory (recovered) ==="]
    plan = str(memory.get("current_plan") or "").strip()
    if plan:
        lines.append(f"current_plan: {plan}")

    def _block(label: str, key: str) -> None:
        """Append a labeled bullet block for a memory list field."""
        items = memory.get(key) or []
        if items:
            lines.append(f"{label}:")
            lines.extend(f"  - {str(x)}" for x in items)

    _block("hypotheses", "hypotheses")
    _block("tried_and_why", "tried_and_why")
    _block("pending", "pending")
    _block("learnings", "learnings")
    cnt = memory.get("checkpoint_count")
    if cnt:
        lines.append(f"(checkpoint #{cnt})")
    return "\n".join(lines)


@dataclass
class CheckpointTracker:
    """Mutable bookkeeping of progress since the last checkpoint."""

    last_tick: int = 0
    last_minute_mark: float = 0.0
    chars_since_last: int = 0
    last_phase: str = ""
    # Largest single request (tokens) in the latest backend turn: an absolute water level, set each turn, never
    # accumulated.
    context_tokens_now: int = 0

    def chars_add(self, n: int) -> None:
        """Accumulate characters produced since the last checkpoint."""
        self.chars_since_last += max(0, int(n))

    def set_context_tokens(self, n: int) -> None:
        """Record the current context size in tokens (absolute water level)."""
        self.context_tokens_now = max(0, int(n))

    def reset(self, *, tick: int, minute_mark: float, phase: str) -> None:
        """Reset the tracker after a checkpoint lands."""
        self.last_tick = int(tick)
        self.last_minute_mark = float(minute_mark)
        self.chars_since_last = 0
        self.last_phase = phase
        self.context_tokens_now = 0


__all__ = [
    "CHECKPOINT_REQUEST_PROMPT",
    "CheckpointPolicy",
    "CheckpointTracker",
    "DEFAULT_CHECKPOINT_CHAR_BUDGET",
    "DEFAULT_CHECKPOINT_EVERY_MINUTES",
    "DEFAULT_CHECKPOINT_EVERY_TICKS",
    "DEFAULT_CHECKPOINT_MIN_TICK_GAP",
    "DEFAULT_CONTEXT_TOKEN_SOFT_FRACTION",
    "DEFAULT_MODEL_CONTEXT_WINDOW",
    "MODEL_CONTEXT_WINDOWS",
    "_DIRECTIVE_MAX_LEN",
    "_DIRECTIVE_POLICY_BLACKLIST",
    "_sanitize_cycle_directive",
    "build_memory_record",
    "context_window_for_model",
    "is_degenerate_checkpoint",
    "parse_checkpoint_reply",
    "render_memory_for_seed",
]
