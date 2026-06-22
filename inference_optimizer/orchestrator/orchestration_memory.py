"""Orchestration working-memory checkpoint / compaction (plan Step 4).

Compresses the live ReAct conversation into a compact structured snapshot
on ``SharedState.orchestration_memory``, then resets + re-seeds from it.
Bounds context growth and drives crash recovery. Pure helpers; the
Coordinator owns the IO.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


# Default checkpoint cadence. A checkpoint fires when ANY trigger crosses its
# threshold.
DEFAULT_CHECKPOINT_EVERY_TICKS: int = 20
DEFAULT_CHECKPOINT_EVERY_MINUTES: float = 30.0
# Char budget over which we force a checkpoint regardless of cadence. Retained as
# a fallback signal for backends that do not report token usage; the authoritative
# trigger on long runs is the context-token budget below.
DEFAULT_CHECKPOINT_CHAR_BUDGET: int = 400_000

# Context-token guardrail (batch-1 #3). The real conversation size is read from
# the backend's reported usage (input + cache_read + cache_creation tokens). A
# soft trigger compacts proactively; the hard fraction is the overflow backstop
# that compacts even when the LLM summary is degenerate (see
# ``deterministic_memory_fallback`` + the Coordinator's hard path).
DEFAULT_CONTEXT_TOKEN_SOFT_FRACTION: float = 0.70
DEFAULT_CONTEXT_TOKEN_HARD_FRACTION: float = 0.85
# Conservative fallback window for an unknown model id. Claude 4.x models on the
# AMD gateway are 200k; env fractions tune the trigger without editing this map.
DEFAULT_MODEL_CONTEXT_WINDOW: int = 200_000
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-8": 200_000,
    "claude-opus-4-7": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
}


def context_window_for_model(model: str) -> int:
    """Context-window size (tokens) for a model id; conservative fallback if unknown.

    Args:
        model: The model id (e.g. ``"claude-opus-4-8"``); blank/unknown ids fall
            back to :data:`DEFAULT_MODEL_CONTEXT_WINDOW`.

    Returns:
        The window size in tokens.
    """
    return MODEL_CONTEXT_WINDOWS.get((model or "").strip(), DEFAULT_MODEL_CONTEXT_WINDOW)

# Content fields that carry forward when a checkpoint reply omits them (#1):
# scalar plan + list threads. ``learnings`` accumulates separately.
_MEMORY_LIST_KEYS: tuple[str, ...] = ("hypotheses", "tried_and_why", "pending")


def _now_iso() -> str:
    """Return the current UTC time as a second-resolution ISO-8601 string.

    Returns:
        The current UTC time formatted as a second-resolution ISO-8601 string.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class CheckpointPolicy:
    """When to take an orchestration-memory checkpoint."""

    every_ticks: int = DEFAULT_CHECKPOINT_EVERY_TICKS
    every_minutes: float = DEFAULT_CHECKPOINT_EVERY_MINUTES
    char_budget: int = DEFAULT_CHECKPOINT_CHAR_BUDGET
    # Context-token soft/hard budgets (absolute token counts; 0 disables). The
    # Coordinator derives these from the orchestration model's window × fraction.
    context_token_soft: int = 0
    context_token_hard: int = 0
    # Always checkpoint on a phase boundary (cheap + a natural seam).
    on_phase_boundary: bool = True

    def should_checkpoint(
        self,
        *,
        ticks_since_last: int,
        minutes_since_last: float,
        chars_since_last: int,
        phase_changed: bool,
        context_tokens_now: int = 0,
    ) -> bool:
        """Decide whether a checkpoint is due under this policy.

        Args:
            ticks_since_last: Ticks elapsed since the last checkpoint.
            minutes_since_last: Minutes elapsed since the last checkpoint.
            chars_since_last: Characters accumulated since the last checkpoint.
            phase_changed: Whether a phase boundary was just crossed.
            context_tokens_now: Authoritative current context size in tokens
                (input + cache_read + cache_creation); 0 when unavailable.

        Returns:
            ``True`` when any enabled trigger (context-token budget, phase
            boundary, tick, time, or char budget) is met.
        """
        if self.context_token_soft > 0 and context_tokens_now >= self.context_token_soft:
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

    def is_hard_compaction(self, context_tokens_now: int) -> bool:
        """True when context is near the window and a compaction MUST happen now.

        The hard path compacts even when the LLM summary is degenerate (using the
        deterministic fallback), so the conversation never overflows the window.

        Args:
            context_tokens_now: Current context size in tokens.

        Returns:
            ``True`` when the hard budget is set and reached.
        """
        return self.context_token_hard > 0 and context_tokens_now >= self.context_token_hard


# Appended as the next user turn on the SAME conversation to elicit the
# compact summary. The agent answers in-band; we parse the JSON block.
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
  "learnings": ["<durable lesson from this session so far>", "..."]
}
```

Keep it tight (a few items per list). This snapshot — plus the
authoritative session facts — is all you will carry into the next
conversation, so capture intent and rationale, not raw numbers you can
re-pull from the context tools.
"""


def parse_checkpoint_reply(raw_text: str) -> dict[str, Any]:
    """Parse the agent's checkpoint reply into the memory schema.

    Tolerant: accepts a fenced ```json block or bare object; missing keys
    default to empty. Never raises — malformed replies yield a best-effort
    dict (with a ``parse_error`` marker).

    Args:
        raw_text: The agent's raw checkpoint reply text.

    Returns:
        The parsed memory dict (``current_plan`` / ``hypotheses`` /
        ``tried_and_why`` / ``pending`` / ``learnings``), with a
        ``parse_error`` marker when no JSON object was found.
    """
    obj = _extract_json_object(raw_text)
    if obj is None:
        return {
            "current_plan": (raw_text or "").strip()[:1000],
            "hypotheses": [],
            "tried_and_why": [],
            "pending": [],
            "learnings": [],
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
    return out


def is_degenerate_checkpoint(parsed: dict[str, Any]) -> bool:
    """True when a parsed checkpoint reply carries no usable working memory.

    Degenerate iff parsing failed (``parse_error`` set) OR all four content
    fields are empty. The Coordinator skips compaction (preserving the live
    conversation + prior memory) on a degenerate reply unless the hard
    context-token guardrail forces a fallback compaction.

    Args:
        parsed: A parsed checkpoint reply from :func:`parse_checkpoint_reply`.

    Returns:
        ``True`` when the reply is degenerate.
    """
    if str(parsed.get("parse_error") or "").strip():
        return True
    has_plan = bool(str(parsed.get("current_plan") or "").strip())
    has_lists = any(parsed.get(k) for k in _MEMORY_LIST_KEYS)
    return not (has_plan or has_lists)


def deterministic_memory_fallback(state: Any) -> dict[str, Any]:
    """Synthesise a minimal working-memory record from authoritative SharedState facts.

    Used by the hard context-token guardrail (#3) when the LLM checkpoint reply
    is degenerate but the conversation must still be compacted to avoid window
    overflow. Pure read of ``state``; never raises on missing attributes.

    Args:
        state: The live ``SharedState`` (duck-typed; only attribute reads).

    Returns:
        A parsed-reply-shaped dict suitable for :func:`build_memory_record`.
    """
    cb = getattr(state, "current_best", {}) or {}
    stack = getattr(state, "optimization_stack", []) or []
    try:
        gain = float(getattr(state, "cumulative_gain_validated", 0.0) or 0.0)
    except (TypeError, ValueError):
        gain = 0.0
    phase = str(getattr(state, "phase", "") or "")
    cycle = int(getattr(state, "macro_cycle", 0) or 0)
    plan = (
        f"[auto] phase={phase} cycle={cycle} "
        f"best_tput={cb.get('tput')} validated_gain={gain:.2f}%"
    )
    return {
        "current_plan": plan,
        "hypotheses": [],
        "tried_and_why": [f"stack has {len(stack)} accepted change(s)"],
        "pending": ["recover plan from SharedState facts (LLM summary was degenerate)"],
        "learnings": [],
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object embedded in free-form text.

    Args:
        text: Text that may contain a fenced ```json``` block or a bare
            ``{ ... }`` span.

    Returns:
        The parsed JSON object, or ``None`` when none is found or it does not
        parse to a dict.
    """
    if not text:
        return None
    # Prefer a fenced ```json ... ``` block.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        # Fall back to the first balanced-looking { ... } span.
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
    """Build the persisted ``orchestration_memory`` record.

    ``learnings`` accumulate across checkpoints (deduped, capped) so durable
    lessons survive a later checkpoint that forgets to repeat them. The other
    content fields (``current_plan`` + the list threads) carry the prior value
    forward when the new reply omits them (#1), so one forgetful checkpoint never
    blanks an in-flight plan.

    Args:
        parsed: The parsed checkpoint reply from :func:`parse_checkpoint_reply`.
        seq: The checkpoint sequence number.
        tick: The current tick.
        previous: The prior persisted memory record, or ``None``.

    Returns:
        The persisted ``orchestration_memory`` record with accumulated,
        deduped, capped ``learnings`` and checkpoint bookkeeping.
    """
    prev = previous or {}
    learnings = list(prev.get("learnings") or [])
    for item in parsed.get("learnings") or []:
        if item not in learnings:
            learnings.append(item)
    learnings = learnings[-50:]  # cap so state.json stays bounded
    # Non-empty-wins: a new value replaces the prior one only when it carries
    # content; an empty field inherits the previous record's value.
    plan = str(parsed.get("current_plan") or "").strip() or prev.get("current_plan", "")
    record: dict[str, Any] = {
        "current_plan": plan,
        "learnings": learnings,
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
    """Render an ``orchestration_memory`` record into prompt text.

    Used for compaction re-seed and resume rebuild. Returns "" when memory
    is empty (fresh session).

    Args:
        memory: An ``orchestration_memory`` record.

    Returns:
        The rendered prompt text, or ``""`` when ``memory`` is empty.
    """
    if not memory:
        return ""
    lines: list[str] = ["=== Your working memory (recovered) ==="]
    plan = str(memory.get("current_plan") or "").strip()
    if plan:
        lines.append(f"current_plan: {plan}")

    def _block(label: str, key: str) -> None:
        """Append a labeled bullet block for a memory list field.

        Args:
            label: Section heading to render.
            key: Memory key whose list items are rendered as bullets.
        """
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
    """Mutable bookkeeping of progress since the last checkpoint.

    Lives on the Coordinator; ``reset`` is called after a checkpoint lands.
    """

    last_tick: int = 0
    last_minute_mark: float = 0.0
    chars_since_last: int = 0
    last_phase: str = ""
    # Authoritative current context size (tokens) from the latest backend turn.
    # An absolute water level, NOT an increment — set each turn, never accumulated.
    context_tokens_now: int = 0

    def chars_add(self, n: int) -> None:
        """Accumulate characters produced since the last checkpoint.

        Args:
            n: Number of characters to add (negatives are clamped to 0).
        """
        self.chars_since_last += max(0, int(n))

    def set_context_tokens(self, n: int) -> None:
        """Record the current context size in tokens (absolute water level).

        Args:
            n: Current context token count (negatives are clamped to 0).
        """
        self.context_tokens_now = max(0, int(n))

    def reset(self, *, tick: int, minute_mark: float, phase: str) -> None:
        """Reset the tracker after a checkpoint lands.

        ``context_tokens_now`` is intentionally NOT cleared: it is an absolute
        water level overwritten by the next turn's reported usage (which drops
        naturally once the conversation is reset and re-seeded).

        Args:
            tick: Current tick to record as the last checkpoint tick.
            minute_mark: Current minute mark to record.
            phase: Current phase to record.
        """
        self.last_tick = int(tick)
        self.last_minute_mark = float(minute_mark)
        self.chars_since_last = 0
        self.last_phase = phase


__all__ = [
    "CHECKPOINT_REQUEST_PROMPT",
    "CheckpointPolicy",
    "CheckpointTracker",
    "DEFAULT_CHECKPOINT_CHAR_BUDGET",
    "DEFAULT_CHECKPOINT_EVERY_MINUTES",
    "DEFAULT_CHECKPOINT_EVERY_TICKS",
    "DEFAULT_CONTEXT_TOKEN_HARD_FRACTION",
    "DEFAULT_CONTEXT_TOKEN_SOFT_FRACTION",
    "DEFAULT_MODEL_CONTEXT_WINDOW",
    "MODEL_CONTEXT_WINDOWS",
    "build_memory_record",
    "context_window_for_model",
    "deterministic_memory_fallback",
    "is_degenerate_checkpoint",
    "parse_checkpoint_reply",
    "render_memory_for_seed",
]
