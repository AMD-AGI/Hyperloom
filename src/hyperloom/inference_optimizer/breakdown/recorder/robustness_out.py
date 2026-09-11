# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Author-time recording of the SBD v6 ``robustness`` section.

The robustness agent's account of a turn is complete the moment its envelope
validates: the intents it raised, the parse problems it reported, and the
workdir the exchange happened in are all in hand right there. This records
them at that point.

Turns the agent could not complete are recorded too, and they are the more
valuable half: without them a session whose robustness agent was mute
throughout reads exactly like one that had nothing to report. The ``outcome``
names which of those happened.

Recording is best-effort: a failure here degrades the exported section and must
never propagate into the agent loop it is describing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

from hyperloom.common.timeutil import now_iso

from .recorder import recorder_for
from .trace import trace_skip

log = logging.getLogger(__name__)

SECTION = "robustness"
TURN_SECTION = "robustness_turn"
PRODUCER = "robustness"

#: Stable ``outcome`` codes for one robustness turn. The agent's own exit
#: decides which one applies, so the code says what happened rather than being
#: guessed later from an absence.
OUTCOME_INTENTS = "intents"
OUTCOME_INVALID_ENVELOPE = "invalid_envelope"
OUTCOME_NO_ENVELOPE = "no_envelope"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _intent_rows(intents: Iterable[Any] | None) -> list[dict[str, Any]]:
    """Project validated intents onto the recorded per-intent shape.

    Accepts either the validated intent objects (which carry a ``type`` enum
    and a payload) or plain mappings, so a caller holding either shape records
    the same row. Intents that named no type are dropped.
    """
    rows: list[dict[str, Any]] = []
    for intent in intents or ():
        if isinstance(intent, Mapping):
            kind = _text(intent.get("type"))
            payload: Mapping[str, Any] = intent.get("payload") if isinstance(intent.get("payload"), Mapping) else {}
        else:
            raw_type = getattr(intent, "type", None)
            kind = _text(getattr(raw_type, "value", raw_type))
            raw_payload = getattr(intent, "payload", None)
            payload = raw_payload if isinstance(raw_payload, Mapping) else {}
        if not kind:
            continue
        row: dict[str, Any] = {"type": kind}
        severity = _text(payload.get("severity"))
        topic = _text(payload.get("topic"))
        if severity:
            row["severity"] = severity
        if topic:
            row["topic"] = topic
        if payload:
            row["payload"] = dict(payload)
        rows.append(row)
    return rows


def record_robustness_turn(
    session_dir: Path | str | None,
    *,
    turn_idx: int,
    outcome: str,
    tick_index: Any = None,
    intents: Iterable[Any] | None = None,
    parse_warnings: Iterable[Any] | None = None,
    workdir: Path | str | None = None,
    detail: str = "",
    ts: str = "",
    producer: str = PRODUCER,
) -> None:
    """Record one robustness-agent turn, keyed by its turn index.

    Idempotent per turn: a re-recorded turn overwrites its own row and leaves
    the other turns alone. A falsy ``session_dir`` is a no-op, ``outcome`` is
    one of the ``OUTCOME_*`` codes, and ``workdir`` is kept as a provenance
    pointer rather than a data source.
    """
    if not session_dir:
        trace_skip(reason="no session_dir", section=TURN_SECTION)
        return
    try:
        row: dict[str, Any] = {
            "turn_idx": int(turn_idx),
            "outcome": _text(outcome),
            "ts": _text(ts) or now_iso(),
            "intents": _intent_rows(intents),
            "parse_warnings": [_text(w) for w in (parse_warnings or ()) if _text(w)],
        }
        if tick_index is not None:
            row["tick_index"] = tick_index
        if workdir:
            row["workdir"] = str(workdir)
        if _text(detail):
            row["detail"] = _text(detail)
        recorder_for(session_dir, producer=producer).record_upsert_item(
            TURN_SECTION,
            row,
            key=str(int(turn_idx)),
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("record_robustness_turn failed", exc_info=True)
        trace_skip(reason="writer raised", section=TURN_SECTION, error=exc)


__all__ = [
    "OUTCOME_INTENTS",
    "OUTCOME_INVALID_ENVELOPE",
    "OUTCOME_NO_ENVELOPE",
    "SECTION",
    "TURN_SECTION",
    "record_robustness_turn",
]
