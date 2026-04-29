"""Codex pseudo-continuity — explicit conversation log + restart.

Why
---

The Codex CLI has no ``--continue`` flag; every invocation is a clean
context. To approximate the "long conversation" feel of Claude's
``--continue``, we maintain ``$AGENT_DIR/conversation.jsonl`` per Codex
agent and re-inject it into the prompt header on every restart.

This module provides:

* :class:`CodexConversationLog` — append-only JSONL of role-tagged turns
  (``user``/``assistant``/``system``), with **char-budget aware
  summarisation** when the file would otherwise blow past the model's
  context window. The summarisation strategy is intentionally simple:

      1. Keep the system prompt + last K full turns intact (recency-bias).
      2. Replace older turns with a single ``role="summary"`` synthetic
         turn that records the bullet-point summary the operator (or a
         critic / sage call) provides.

  This mirrors the persona-distill cadence already in marathon mode (see
  ``orchestrator/persona.py``) — the difference is that Persona is per-agent
  *identity* whereas the conversation log is per-agent *episodic memory*.

* :class:`CodexPromptComposer` — turns the on-disk log into a single
  prompt string the launcher's ``codex --prompt-file ...`` template
  consumes. The launcher already shells out to a temp file; this composer
  is what the launcher would call from Python whenever we promote the
  pseudo-continuity from "shell-only template" to "Python-managed".

* :func:`update_after_restart` — the per-restart bookkeeping the launcher
  invokes (Phase 3+) to:
    - read the assistant turn the previous CLI run wrote;
    - append it to conversation.jsonl;
    - apply summarisation if the running char count exceeds the budget;
    - return the prompt body for the next codex invocation.

Status
------

* JSONL schema + reader/writer + size-based truncation are implemented
  and unit-tested.
* The launcher's existing codex pane template (in
  :mod:`inference_optimizer.orchestrator.multi_cli.launcher`) reads the
  raw conversation.jsonl and pastes it into the prompt as-is. When the
  launcher is upgraded to call this composer (Phase 3+), it gets the
  budget-aware version for free.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
DEFAULT_CONVERSATION_FILENAME = "conversation.jsonl"
# Codex roles run on gpt-5.4 (per ROLE_CRITIC) which today comfortably
# accommodates ~150 KB of plain prompt before refusing. We leave headroom
# for the system prompt + last assistant message, so the conversation
# budget is conservative.
DEFAULT_CHAR_BUDGET = 80_000
# Always keep this many of the most recent role=user/assistant turns
# verbatim regardless of budget (so the agent never loses recent context
# entirely to the summary).
DEFAULT_KEEP_RECENT_TURNS = 6


_VALID_ROLES = frozenset({"system", "user", "assistant", "summary"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------
@dataclass
class ConversationTurn:
    """One JSON line in the Codex conversation log.

    Attributes:
        role:      One of ``system``, ``user``, ``assistant``, ``summary``.
        content:   Text body. Long lines are fine; we only count chars
                   when budget enforcement decides whether to truncate.
        ts:        ISO-8601 timestamp. Auto-filled when omitted.
        attempt:   Optional restart attempt id, useful for grep-debugging.
    """

    role: str
    content: str
    ts: str = field(default_factory=_now_iso)
    attempt: int | None = None

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise ValueError(
                f"role {self.role!r} not in {sorted(_VALID_ROLES)}"
            )

    @classmethod
    def from_json(cls, line: str) -> "ConversationTurn":
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError("conversation turn must be a JSON object")
        return cls(
            role=str(obj["role"]),
            content=str(obj["content"]),
            ts=str(obj.get("ts") or _now_iso()),
            attempt=obj.get("attempt"),
        )

    def to_json(self) -> str:
        d: dict = {"role": self.role, "content": self.content, "ts": self.ts}
        if self.attempt is not None:
            d["attempt"] = self.attempt
        return json.dumps(d, separators=(",", ":"), ensure_ascii=False)

    def char_len(self) -> int:
        # Approximate token count via 1 char ≈ 0.25 tokens. We use chars
        # everywhere because tokenisers are model-specific and the budget
        # is itself a soft guard.
        return len(self.content) + 32  # 32 = wrapper overhead per turn


# ---------------------------------------------------------------------------
@dataclass
class CodexConversationLog:
    """Append-only JSONL of conversation turns for one Codex agent.

    The writer is single-process — each Codex CLI restart is a fresh
    subprocess that either appends one ``assistant`` turn at the end of
    its run or doesn't (when codex itself crashes); either way the file
    survives because it lives on the session NFS path.

    We do *not* try to handle concurrent writes — only the launcher's
    update path appends; codex itself only reads.
    """

    path: Path
    char_budget: int = DEFAULT_CHAR_BUDGET
    keep_recent_turns: int = DEFAULT_KEEP_RECENT_TURNS

    # ------------------------------------------------------------------
    def turns(self) -> list[ConversationTurn]:
        """Return every recorded turn (best-effort — bad lines skipped)."""
        if not self.path.is_file():
            return []
        out: list[ConversationTurn] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                out.append(ConversationTurn.from_json(line))
            except (ValueError, json.JSONDecodeError):
                continue
        return out

    def append(self, turn: ConversationTurn) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(turn.to_json() + "\n")

    def append_many(self, turns: Iterable[ConversationTurn]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            for t in turns:
                fh.write(t.to_json() + "\n")

    def char_count(self) -> int:
        return sum(t.char_len() for t in self.turns())

    # ------------------------------------------------------------------
    # Budget-aware compaction
    # ------------------------------------------------------------------
    def compact_if_over_budget(
        self, *, summariser: "Summariser | None" = None
    ) -> bool:
        """If total chars exceed ``char_budget``, fold older turns into
        a single ``role="summary"`` synthetic turn.

        Returns ``True`` when compaction occurred. Safe to call on every
        restart — it short-circuits when under budget.

        ``summariser`` is a callable ``(turns) -> str`` that produces the
        replacement text. The default :func:`naive_summariser` keeps role
        prefixes + a 3-line head/tail of every dropped turn. Production
        callers will swap in a real LLM summariser by passing one in.
        """
        all_turns = self.turns()
        if not all_turns:
            return False
        total = sum(t.char_len() for t in all_turns)
        if total <= self.char_budget:
            return False

        keep_n = max(1, self.keep_recent_turns)
        # Always keep the *first* system/summary block (so the role
        # bootstrap stays attached) and the last keep_n turns verbatim.
        head: list[ConversationTurn] = []
        if all_turns and all_turns[0].role in ("system", "summary"):
            head.append(all_turns[0])
        tail = all_turns[-keep_n:] if keep_n < len(all_turns) else all_turns[:]
        # Avoid double-counting overlap between head and tail.
        head_ids = {id(t) for t in head}
        tail_filtered = [t for t in tail if id(t) not in head_ids]
        middle = [t for t in all_turns if id(t) not in head_ids
                  and id(t) not in {id(x) for x in tail_filtered}]
        # If the only thing in `middle` is an existing summary (= we
        # already compacted before) there's nothing new to fold; bail
        # out to keep ``compact_if_over_budget`` idempotent across runs.
        non_summary_middle = [t for t in middle if t.role != "summary"]
        if not non_summary_middle:
            return False

        summary_text = (summariser or naive_summariser)(middle)
        summary_turn = ConversationTurn(
            role="summary",
            content=summary_text,
            attempt=None,
        )
        rebuilt: list[ConversationTurn] = []
        rebuilt.extend(head)
        rebuilt.append(summary_turn)
        rebuilt.extend(tail_filtered)
        self._rewrite(rebuilt)
        log.info(
            "codex_continuity: compacted %s (%d -> %d turns; budget=%d chars)",
            self.path, len(all_turns), len(rebuilt), self.char_budget,
        )
        return True

    def _rewrite(self, turns: Sequence[ConversationTurn]) -> None:
        """Atomic rewrite via tmp + rename so a concurrent reader (= the
        next codex invocation) never sees a half-written file.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for t in turns:
                fh.write(t.to_json() + "\n")
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------
# Summariser plug point
# ---------------------------------------------------------------------------
Summariser = "callable[[Sequence[ConversationTurn]], str]"  # docs hint


def naive_summariser(turns: Sequence[ConversationTurn]) -> str:
    """Default summariser: role prefix + first 240 chars of each turn.

    Useful for tests + smoke runs. Production deployments swap in a
    real LLM summariser via ``compact_if_over_budget(summariser=...)``.
    """
    lines = ["[summary of older turns:]"]
    for t in turns:
        head = t.content.strip().splitlines()[:3]
        joined = " | ".join(head)
        if len(joined) > 240:
            joined = joined[:237] + "..."
        lines.append(f"- {t.role}: {joined}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt composer (consumed by launcher's codex pane Phase 3+)
# ---------------------------------------------------------------------------
@dataclass
class CodexPromptComposer:
    """Turns ``conversation.jsonl`` + the system prompt into one text blob.

    The composer is deliberately stateless aside from the system prompt:
    every restart gets a fresh prompt by calling :meth:`compose`. The
    template matches what marathon's codex pane expected (system header
    → ``==== prior conversation ====`` block → newline) so the Python and
    bash paths stay interchangeable.
    """

    system_prompt: str
    conversation_log: CodexConversationLog
    instructions_header: str = (
        "You are a Codex agent in a multi-CLI A2A pipeline. Read your "
        "inbox.jsonl tail (after the persisted seq cursor); reply with "
        "validated_json_output intent envelopes appended to outbox.jsonl. "
        "PolicyGate enforces role permissions across processes."
    )

    def compose(self, *, attempt: int | None = None) -> str:
        turns = self.conversation_log.turns()
        sections: list[str] = [
            self.system_prompt.rstrip(),
            "",
            "==== protocol header ====",
            self.instructions_header.strip(),
            "==== end protocol header ====",
            "",
        ]
        if turns:
            sections.append("==== prior conversation (oldest -> newest) ====")
            for t in turns:
                if t.attempt is not None:
                    header = f"[{t.role} attempt={t.attempt} ts={t.ts}]"
                else:
                    header = f"[{t.role} ts={t.ts}]"
                sections.append(header)
                sections.append(t.content.rstrip())
                sections.append("")
            sections.append("==== end conversation ====")
            sections.append("")
        if attempt is not None:
            sections.append(f"==== current attempt={attempt} ====")
        return "\n".join(sections)


# ---------------------------------------------------------------------------
# Launcher integration entrypoint
# ---------------------------------------------------------------------------
def update_after_restart(
    log_path: Path,
    *,
    new_assistant_turn: str | None,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    keep_recent_turns: int = DEFAULT_KEEP_RECENT_TURNS,
    summariser: "Summariser | None" = None,
    attempt: int | None = None,
) -> bool:
    """Convenience: append the previous attempt's assistant turn (if any)
    and compact the log when over budget.

    Returns ``True`` when compaction ran. The launcher invokes this
    between codex restarts; passing ``None`` for ``new_assistant_turn``
    is fine (e.g. when the previous attempt crashed before producing
    any output).
    """
    convlog = CodexConversationLog(
        path=Path(log_path),
        char_budget=char_budget,
        keep_recent_turns=keep_recent_turns,
    )
    if new_assistant_turn:
        convlog.append(
            ConversationTurn(role="assistant",
                             content=new_assistant_turn,
                             attempt=attempt)
        )
    return convlog.compact_if_over_budget(summariser=summariser)


__all__ = [
    "CodexConversationLog",
    "CodexPromptComposer",
    "ConversationTurn",
    "DEFAULT_CHAR_BUDGET",
    "DEFAULT_CONVERSATION_FILENAME",
    "DEFAULT_KEEP_RECENT_TURNS",
    "naive_summariser",
    "update_after_restart",
]
