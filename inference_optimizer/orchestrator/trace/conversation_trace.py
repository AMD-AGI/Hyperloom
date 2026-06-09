# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Append-only writer for ``reports/trace/conversations.jsonl``.

Sibling of :mod:`.llm_trace`. Where that module owns the per-call *token*
account (deliberately small, no prompt text), this one owns the
*conversation*: the full, redacted prompt + completion for every
in-process LLM call. Both share the same identity / join keys
(``session_id`` / ``component`` / ``role`` / ``tick`` / ``phase`` /
``turn``) so the two streams line up against ``decision_trace``.

Design contract:

* **Best-effort I/O**: disk failures while appending are logged and
  swallowed; a conversation write must never break the optimization loop
  (mirrors :func:`.llm_trace.append_llm_call`).
* **Full text, redacted**: ``prompt`` and ``response`` are stored in
  full (no truncation) but passed through :func:`redact_secrets` first so
  an accidentally-logged credential value never lands on disk. The full
  text is the whole point — Langfuse export / replay needs it intact.
* **Self-contained redaction**: the redactor strips secret *values*
  (Bearer tokens, ``ak-`` / ``sk-`` / ``pk-`` keys, ``ghp_`` GitHub
  tokens, ``KEY=value`` / ``token: value`` shapes), not just env-var
  names, because we are now persisting the model's raw text.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...session_paths import conversations_path
from .llm_trace import VALID_COMPONENTS

log = logging.getLogger(__name__)


# Canonical, ordered field contract for one ``conversations.jsonl`` row.
_ROW_FIELDS: frozenset[str] = frozenset({
    "session_id",
    "ts",
    "component",
    "role",
    "task_id",
    "dyn_id",
    "tick",
    "phase",
    "turn",
    "model",
    "prompt",
    "response",
})


class ConversationRowError(ValueError):
    """Raised when a conversation row violates the closed schema."""


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------
# Each pattern captures a leading "label" group (kept) and replaces the
# trailing secret value with a placeholder. Ordered most-specific first.
_REDACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Authorization: Bearer <token>
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9\-_.=]{8,}"), r"\1[REDACTED]"),
    # Provider key prefixes: ak-, sk-, pk-, sk-lf-, pk-lf- ...
    (re.compile(r"\b((?:ak|sk|pk)-(?:lf-)?)[A-Za-z0-9\-_]{6,}"), r"\1[REDACTED]"),
    # GitHub tokens: ghp_, gho_, ghs_, ghr_, github_pat_
    (re.compile(r"\b(gh[pousr]_|github_pat_)[A-Za-z0-9_]{10,}"), r"\1[REDACTED]"),
    # KEY=value / TOKEN=value / SECRET=value / PASSWORD=value (env shape)
    (
        re.compile(
            r"(?i)\b([A-Z0-9_]*"
            r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|AUTH|CREDENTIAL)"
            r"[A-Z0-9_]*\s*[=:]\s*)"
            r"[^\s,;'\"]+"
        ),
        r"\1[REDACTED]",
    ),
)


def redact_secrets(text: str) -> str:
    """Strip obvious secret *values* from ``text`` before it hits disk.

    Conservative and idempotent: the label / prefix is preserved so the
    redacted line still reads sensibly, only the secret material is
    replaced with ``[REDACTED]``. Returns ``text`` unchanged when it
    carries no recognizable secret shape.
    """
    if not text:
        return text
    out = text
    for pattern, repl in _REDACT_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_text(value: Any) -> str:
    """Normalize a prompt / response field to a (possibly empty) string."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


@dataclass
class ConversationRecord:
    """One LLM call's full prompt + completion, plus the join keys.

    Identity (``session_id`` + ``component``) is required; everything else
    is an optional join key filled when the call site has it. ``prompt`` /
    ``response`` hold the full text and are redacted at serialization time.
    """

    session_id: str
    component: str
    role: str | None = None
    task_id: str | None = None
    dyn_id: str | None = None
    tick: int | None = None
    phase: str | None = None
    turn: int | None = None
    model: str | None = None
    prompt: str = ""
    response: str = ""

    def to_row(self) -> dict[str, Any]:
        """Serialize to the on-disk row dict, stamping ``ts`` and redacting
        the prompt / response text."""
        return {
            "session_id": str(self.session_id),
            "ts": _now_iso(),
            "component": str(self.component),
            "role": _coerce_optional_str(self.role),
            "task_id": _coerce_optional_str(self.task_id),
            "dyn_id": _coerce_optional_str(self.dyn_id),
            "tick": _coerce_optional_int(self.tick),
            "phase": _coerce_optional_str(self.phase),
            "turn": _coerce_optional_int(self.turn),
            "model": _coerce_optional_str(self.model),
            "prompt": redact_secrets(_coerce_text(self.prompt)),
            "response": redact_secrets(_coerce_text(self.response)),
        }


def _validate_row(row: dict[str, Any]) -> None:
    """Fail fast if ``row`` deviates from the closed schema."""
    keys = set(row.keys())
    extra = sorted(keys - _ROW_FIELDS)
    missing = sorted(_ROW_FIELDS - keys)
    if extra or missing:
        raise ConversationRowError(
            f"conversations row violates closed schema: "
            f"extra={extra!r} missing={missing!r}"
        )
    session_id = row.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ConversationRowError(
            f"conversations row requires a non-empty 'session_id'; got "
            f"{session_id!r}"
        )
    component = row.get("component")
    if component not in VALID_COMPONENTS:
        raise ConversationRowError(
            f"conversations row 'component'={component!r} is not one of "
            f"{sorted(VALID_COMPONENTS)!r}"
        )


def append_conversation(
    *,
    session_dir: Path,
    record: ConversationRecord,
    target: Path | None = None,
) -> None:
    """Append one validated conversation row to the conversations ledger.

    The row is serialized (which stamps ``ts`` and redacts the text),
    checked against the closed schema, then atomically appended.
    ``OSError`` while writing is logged and swallowed so a full disk or a
    permissions glitch never breaks the optimization loop.

    A schema violation (:class:`ConversationRowError`) is *not* swallowed:
    that is a programming error at the call site and must surface in tests.
    """
    row = record.to_row()
    _validate_row(row)
    dest = target if target is not None else conversations_path(session_dir)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        log.warning(
            "conversation_trace: append failed for component=%s session_id=%s: %r",
            record.component, record.session_id, exc,
        )


# Sanity guard: dataclass fields (minus the write-time ``ts``) must stay
# in lockstep with the on-disk row schema.
_DATACLASS_FIELDS: frozenset[str] = frozenset(
    f.name for f in fields(ConversationRecord)
)
assert _DATACLASS_FIELDS | {"ts"} == _ROW_FIELDS, (
    "ConversationRecord fields drifted from _ROW_FIELDS: "
    f"dataclass={sorted(_DATACLASS_FIELDS)} row={sorted(_ROW_FIELDS)}"
)


__all__ = [
    "ConversationRecord",
    "ConversationRowError",
    "append_conversation",
    "redact_secrets",
]
