"""Orchestration working-memory checkpoint / compaction (plan Step 4).

In the persistent-conversation ReAct design (plan Steps 1-3) the
Orchestration agent's plan / reasoning lives in the live Claude
conversation rather than in ``state.json``. That buys reasoning
continuity but two problems follow:

1. **Unbounded growth.** A multi-hour run would let the conversation
   transcript grow without bound, eventually blowing the context window
   and the per-call cost/latency.
2. **Resume.** A crash loses the in-memory conversation; replaying the
   full transcript is non-deterministic (it contains benchmark numbers
   that won't reproduce bit-for-bit).

The fix is a *checkpoint*: periodically ask the agent to compress its own
working memory into a compact structured snapshot, persist it on
``SharedState.orchestration_memory``, then **reset** the conversation and
re-seed it from that snapshot. The snapshot also drives crash recovery —
on resume we rebuild the conversation from ``orchestration_memory`` plus
the authoritative ``SharedState`` facts instead of replaying a transcript.

This module is pure helpers (prompt text + parsing + trigger policy); the
Coordinator owns the IO (running the summary turn, writing state, resetting
the backend).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# Default checkpoint cadence. Operators can override via the Coordinator
# constructor / env (wired in coordinator.py). A checkpoint fires when ANY
# trigger crosses its threshold; the cheapest is the per-tick counter.
DEFAULT_CHECKPOINT_EVERY_TICKS: int = 20
DEFAULT_CHECKPOINT_EVERY_MINUTES: float = 30.0
# Rough char budget over which we force a checkpoint regardless of cadence
# (a proxy for "conversation is getting large"). The Coordinator feeds the
# cumulative prompt-char estimate from backend.calls.
DEFAULT_CHECKPOINT_CHAR_BUDGET: int = 400_000


_MEMORY_KEYS: tuple[str, ...] = (
    "current_plan", "hypotheses", "tried_and_why", "pending", "learnings",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class CheckpointPolicy:
    """When to take an orchestration-memory checkpoint."""

    every_ticks: int = DEFAULT_CHECKPOINT_EVERY_TICKS
    every_minutes: float = DEFAULT_CHECKPOINT_EVERY_MINUTES
    char_budget: int = DEFAULT_CHECKPOINT_CHAR_BUDGET
    # Always checkpoint on a phase boundary (cheap + a natural seam).
    on_phase_boundary: bool = True

    def should_checkpoint(
        self,
        *,
        ticks_since_last: int,
        minutes_since_last: float,
        chars_since_last: int,
        phase_changed: bool,
    ) -> bool:
        if phase_changed and self.on_phase_boundary:
            return True
        if self.every_ticks > 0 and ticks_since_last >= self.every_ticks:
            return True
        if self.every_minutes > 0 and minutes_since_last >= self.every_minutes:
            return True
        if self.char_budget > 0 and chars_since_last >= self.char_budget:
            return True
        return False


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

    Tolerant: accepts a fenced ```json block or a bare object; missing
    keys default to empty. Never raises — a malformed reply yields a
    best-effort dict (possibly with a ``parse_error`` marker) so the
    checkpoint still advances rather than wedging the loop.
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


def _extract_json_object(text: str) -> dict[str, Any] | None:
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

    Carries the parsed fields plus checkpoint provenance. ``learnings``
    accumulate across checkpoints (deduped, capped) so durable lessons
    are not lost when a later checkpoint forgets to repeat them.
    """
    prev = previous or {}
    learnings = list(prev.get("learnings") or [])
    for item in parsed.get("learnings") or []:
        if item not in learnings:
            learnings.append(item)
    learnings = learnings[-50:]  # cap so state.json stays bounded
    return {
        "current_plan": parsed.get("current_plan", ""),
        "hypotheses": parsed.get("hypotheses", []),
        "tried_and_why": parsed.get("tried_and_why", []),
        "pending": parsed.get("pending", []),
        "learnings": learnings,
        "last_checkpoint_seq": int(seq),
        "last_checkpoint_tick": int(tick),
        "last_checkpoint_ts": _now_iso(),
        "checkpoint_count": int(prev.get("checkpoint_count", 0)) + 1,
        "parse_error": parsed.get("parse_error", ""),
    }


def render_memory_for_seed(memory: dict[str, Any]) -> str:
    """Render an ``orchestration_memory`` record into prompt text.

    Used both for compaction re-seed and resume rebuild: this is what the
    agent reads to recover its plan into a fresh conversation. Returns ""
    when memory is empty (fresh session — nothing to recover).
    """
    if not memory:
        return ""
    lines: list[str] = ["=== Your working memory (recovered) ==="]
    plan = str(memory.get("current_plan") or "").strip()
    if plan:
        lines.append(f"current_plan: {plan}")

    def _block(label: str, key: str) -> None:
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

    Lives on the Coordinator. ``note_*`` methods accumulate the deltas the
    policy needs; ``reset`` is called after a checkpoint lands.
    """

    last_tick: int = 0
    last_minute_mark: float = 0.0
    chars_since_last: int = 0
    last_phase: str = ""

    def chars_add(self, n: int) -> None:
        self.chars_since_last += max(0, int(n))

    def reset(self, *, tick: int, minute_mark: float, phase: str) -> None:
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
    "build_memory_record",
    "parse_checkpoint_reply",
    "render_memory_for_seed",
]
