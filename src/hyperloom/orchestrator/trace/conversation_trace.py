# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Append-only writer for ``reports/trace/conversations.jsonl``."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import redact_secret_values

from hyperloom.common.io import append_jsonl
from hyperloom.common.timeutil import now_iso
from hyperloom.inference_optimizer.session.session_paths import conversations_path
from ._row_utils import (
    coerce_optional_int as _coerce_optional_int,
    coerce_optional_str as _coerce_optional_str,
    validate_closed_row,
)
from .llm_trace import VALID_COMPONENTS

log = logging.getLogger(__name__)


# Canonical, ordered field contract for one ``conversations.jsonl`` row.
_ROW_FIELDS: frozenset[str] = frozenset(
    {
        "session_id",
        "ts",
        "component",
        "call_id",
        "role",
        "task_id",
        "dyn_id",
        "tick",
        "phase",
        "turn",
        "model",
        "prompt",
        "response",
    }
)


class ConversationRowError(ValueError):
    """Raised when a conversation row violates the closed schema."""


def redact_secrets(text: str) -> str:
    """Strip obvious secret *values* from ``text`` before it hits disk."""
    if not text:
        return text
    return redact_secret_values(text)


# microseconds + ``+00:00`` (canonical helper; kept importable for callers).
_now_iso = now_iso


def _coerce_text(value: Any) -> str:
    """Normalize a prompt / response field to a (possibly empty) string."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


@dataclass
class ConversationRecord:
    """One LLM call's full prompt + completion, plus the join keys."""

    session_id: str
    component: str
    # Same per-call id as the token row for this call (see ``llm_trace``); the Langfuse emitter pairs the two halves
    # on it when both carry one.
    call_id: str | None = None
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
        """Serialize to the on-disk row dict, stamping ``ts`` and redacting the prompt / response text."""
        return {
            "session_id": str(self.session_id),
            "ts": _now_iso(),
            "component": str(self.component),
            "call_id": _coerce_optional_str(self.call_id),
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


def append_conversation(
    *,
    session_dir: Path,
    record: ConversationRecord,
    target: Path | None = None,
) -> None:
    """Append one validated conversation row to the conversations ledger."""
    row = record.to_row()
    validate_closed_row(
        row,
        fields=_ROW_FIELDS,
        valid_components=VALID_COMPONENTS,
        error_cls=ConversationRowError,
        label="conversations",
    )
    dest = target if target is not None else conversations_path(session_dir)
    try:
        append_jsonl(dest, row, make_parents=True, ensure_ascii=False)
    except OSError as exc:
        log.warning(
            "conversation_trace: append failed for component=%s session_id=%s: %r",
            record.component,
            record.session_id,
            exc,
        )

    # Second sink (opt-in): mirror conversation text to Langfuse live.
    if target is None:
        try:
            from .langfuse_emitter import get_emitter

            get_emitter(session_dir).record_conversation(row)
        except Exception:  # noqa: BLE001 — Langfuse must never break the ledger
            log.debug("conversation_trace: langfuse mirror failed", exc_info=True)


# Sanity guard: dataclass fields (minus the write-time ``ts``) must stay in lockstep with the on-disk row schema.
_DATACLASS_FIELDS: frozenset[str] = frozenset(f.name for f in fields(ConversationRecord))
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
