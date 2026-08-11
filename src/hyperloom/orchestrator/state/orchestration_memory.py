"""Orchestration working-memory checkpoint / compaction.

Compresses the live ReAct conversation into a compact structured snapshot
on ``SharedState.orchestration_memory``, then resets + re-seeds from it.
Bounds context growth and drives crash recovery. Pure helpers; the
Coordinator owns the IO.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from typing import Any

from hyperloom.common.timeutil import now_iso


# Default checkpoint cadence. A checkpoint fires when ANY trigger crosses its
# threshold.
DEFAULT_CHECKPOINT_EVERY_TICKS: int = 20
DEFAULT_CHECKPOINT_EVERY_MINUTES: float = 30.0
# Prompt+reply chars forcing a checkpoint regardless of cadence.
DEFAULT_CHECKPOINT_CHAR_BUDGET: int = 400_000

# Context-token guardrail, as a fraction of the model's window. The compared
# quantity is one request's input side, never a per-call sum over the internal
# agentic turns (that sum can exceed the window itself).
DEFAULT_CONTEXT_TOKEN_SOFT_FRACTION: float = 0.70
# Minimum ticks between two token-triggered compactions, so a re-seeded
# conversation is not compacted again before it reports a fresh level.
DEFAULT_CHECKPOINT_MIN_TICK_GAP: int = 3
# Conservative fallback window for an unknown model id.
DEFAULT_MODEL_CONTEXT_WINDOW: int = 200_000
# Keys must be lower-case with ``-`` separators; lookups are folded to that form.
# These drive the compaction trigger (window * soft fraction), so they stay at
# the 200k every Claude model serves without an extended-window opt-in, even
# where a gateway advertises 1M. Listing a model explicitly keeps it pinned to
# that value if DEFAULT_MODEL_CONTEXT_WINDOW ever moves.
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-5": 200_000,
    "claude-opus-4-8": 200_000,
    "claude-opus-4-7": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
}


def context_window_for_model(model: str) -> int:
    """Context-window size (tokens) for a model id; conservative fallback if unknown.

    Args:
        model: The model id (e.g. ``"claude-opus-5"``); matched
            case-insensitively with ``.`` / ``_`` folded to ``-``, so gateway
            spellings such as ``"Claude-Opus-5"`` resolve. Blank/unknown ids
            fall back to :data:`DEFAULT_MODEL_CONTEXT_WINDOW`.

    Returns:
        The window size in tokens.
    """
    key = (model or "").strip().lower().replace(".", "-").replace("_", "-")
    return MODEL_CONTEXT_WINDOWS.get(key, DEFAULT_MODEL_CONTEXT_WINDOW)


# List threads that carry forward when a checkpoint reply omits them
# (``learnings`` accumulates separately).
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
        """Recompute the soft budget from a window the provider itself reported.

        :data:`MODEL_CONTEXT_WINDOWS` covers the models this project pins, and
        everything else falls back to a conservative default. A provider that
        states its own window per turn knows better than that fallback, and
        compacting against a window smaller than the real one fires early --
        which costs the conversation, because compaction resets it.

        Args:
            window: Window size the provider reported; 0 or less leaves the
                current budget untouched.
            fraction: Share of the window at which to compact.
        """
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
        """Decide whether a checkpoint is due under this policy.

        Only the token trigger is gated by :attr:`min_tick_gap`; the cadence
        triggers are unaffected by that floor.

        Args:
            ticks_since_last: Ticks elapsed since the last checkpoint.
            minutes_since_last: Minutes elapsed since the last checkpoint.
            chars_since_last: Characters accumulated since the last checkpoint.
            phase_changed: Whether a phase boundary was just crossed.
            context_tokens_now: Current size of a single request's input side in
                tokens; 0 when unavailable.

        Returns:
            ``True`` when any enabled trigger (context-token budget, phase
            boundary, tick, time, or char budget) is met.
        """
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
    """Return ``raw`` if it passes safety checks, else empty string.

    Truncates to :data:`_DIRECTIVE_MAX_LEN` bytes and rejects text that
    contains a policy-override phrase from :data:`_DIRECTIVE_POLICY_BLACKLIST`.

    Args:
        raw: The raw directive string from the LLM.

    Returns:
        The sanitized directive, or ``""`` when rejected.
    """
    text = raw.strip()[:_DIRECTIVE_MAX_LEN]
    lower = text.lower()
    if any(phrase in lower for phrase in _DIRECTIVE_POLICY_BLACKLIST):
        return ""
    return text


def parse_checkpoint_reply(raw_text: str) -> dict[str, Any]:
    """Parse the agent's checkpoint reply into the memory schema.

    Tolerant: accepts a fenced ```json block or bare object; missing keys
    default to empty. Never raises — malformed replies yield a best-effort
    dict (with a ``parse_error`` marker).

    Args:
        raw_text: The agent's raw checkpoint reply text.

    Returns:
        The parsed memory dict (``current_plan`` / ``hypotheses`` /
        ``tried_and_why`` / ``pending`` / ``learnings`` /
        ``next_cycle_directive``), with a ``parse_error`` marker when no
        JSON object was found.
    """
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
    out["next_cycle_directive"] = _sanitize_cycle_directive(
        str(obj.get("next_cycle_directive") or "")
    )
    return out


def is_degenerate_checkpoint(parsed: dict[str, Any]) -> bool:
    """True when a parsed checkpoint reply carries no usable working memory.

    Degenerate iff parsing failed (``parse_error`` set) OR all four content
    fields are empty. The Coordinator skips compaction on a degenerate reply,
    preserving the live conversation and prior memory, and counts the reply
    toward the consecutive-degenerate advisory.

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
    """Build the persisted ``orchestration_memory`` record.

    ``learnings`` accumulate across checkpoints (deduped, capped) so durable
    lessons survive a later checkpoint that forgets to repeat them. The other
    content fields (``current_plan`` + the list threads) carry the prior value
    forward when the new reply omits them, so one forgetful checkpoint never
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
    # Non-empty-wins: an empty field inherits the previous record's value.
    plan = str(parsed.get("current_plan") or "").strip() or prev.get("current_plan", "")
    directive = str(parsed.get("next_cycle_directive") or "").strip() or str(
        prev.get("next_cycle_directive") or ""
    )
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
    # Largest single request (tokens) in the latest backend turn: an absolute
    # water level, set each turn, never accumulated.
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

        Clears the water level too: the compacted conversation no longer holds
        the context that reading described.

        Args:
            tick: Current tick to record as the last checkpoint tick.
            minute_mark: Current minute mark to record.
            phase: Current phase to record.
        """
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
