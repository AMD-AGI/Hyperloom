# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Author-time recording of the SBD v6 ``critic`` section.

The critic agent reviews the session as a whole, iteration after iteration,
and each iteration is complete the moment the review comes back: what it spoke
about, how its rulings fell, and the four artifacts it left behind. This
records it there. It is the session-level channel and does not compete with the
per-proposal verdicts, which stay with the proposals they judge.

An iteration has no single verdict -- it rules on every proposal in front of it
-- so the row carries the distribution of its rulings rather than one string,
and its prose comes off the ``send_message`` intent in the emitted envelope,
which is where the agent actually speaks.

Iterations are keyed by a content-derived id rather than the process-local
iteration number, because that number is reused when a session resumes and
workdirs are pruned underneath it -- keying on it would let a later iteration
overwrite an earlier one's history.

Recording is best-effort: a failure here degrades the exported section and must
never propagate into the review loop it is describing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from hyperloom.common.jsonio import read_json

from ..critic_reviews import normalize_framework_reviews
from .recorder import recorder_for
from .trace import trace_skip

log = logging.getLogger(__name__)

SECTION = "critic"
ITERATION_SECTION = "critic_iteration"
PRODUCER = "critic"

#: How much of the critic's prose summary is kept. The verdict is the decision;
#: the summary is context for it, and an unbounded one would let one iteration
#: dominate the section.
_SUMMARY_LIMIT = 500


def _rel(path: Path, session_dir: Path | str) -> str:
    """Express ``path`` relative to the session, or as-is when outside it."""
    try:
        return Path(path).relative_to(Path(session_dir)).as_posix()
    except (ValueError, TypeError):
        return str(path)


def _stable_id(prefix: str, *parts: Any) -> str:
    """Build a readable, collision-resistant id from author-time values."""
    raw_parts: list[str] = []
    for part in parts:
        if isinstance(part, Mapping):
            text = json.dumps(dict(part), sort_keys=True, separators=(",", ":"), default=str)
        else:
            text = str(part or "")
        if text:
            raw_parts.append(text)
    raw = "|".join(raw_parts) or "unknown"
    readable = re.sub(r"[^A-Za-z0-9._:-]+", "-", raw).strip("-")[:96] or "unknown"
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{prefix}:{readable}:{digest}"


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    """The mapping rows of ``value``, ignoring anything else in the list."""
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _spoken_message(emit: Mapping[str, Any]) -> tuple[str, str]:
    """The ``(topic, body)`` of the first message the iteration spoke.

    The critic's prose does not sit on the emit itself; it travels as a
    ``send_message`` intent inside the envelope, which is also the only thing a
    turn with nothing to rule on produces.
    """
    intents = _rows(_dict(emit.get("intent_envelope")).get("intents"))
    for intent in intents:
        if str(intent.get("intent_type") or "") != "send_message":
            continue
        payload = _dict(intent.get("payload"))
        return str(payload.get("topic") or ""), str(payload.get("body_md") or "")
    return "", ""


def _verdict_counts(review: Mapping[str, Any]) -> dict[str, int]:
    """How many times each verdict was handed down this iteration.

    An iteration rules on every proposal in front of it, so it has no single
    verdict; the distribution is the honest scalar. Counted over every ruling,
    not just the framework ones, because this row describes the whole turn.
    """
    counts: dict[str, int] = {}
    for verdict_row in _rows(review.get("review_verdicts")):
        verdict = str(verdict_row.get("verdict") or "").strip().lower()
        if verdict:
            counts[verdict] = counts.get(verdict, 0) + 1
    return dict(sorted(counts.items()))


def _verdict_rollup(counts: Mapping[str, int]) -> str:
    """A one-line reading of ``counts``, e.g. ``2 approve, 1 reject``."""
    return ", ".join(f"{count} {verdict}" for verdict, count in counts.items())


def record_critic_iteration(
    session_dir: Path | str | None,
    *,
    iter_n: int,
    review: dict[str, Any] | None,
    emit: dict[str, Any] | None,
    workdir: Path | str | None,
    request: dict[str, Any] | None = None,
    judge_bundle: dict[str, Any] | None = None,
    kb_priors: dict[str, Any] | None = None,
    producer: str = PRODUCER,
) -> None:
    """Record one critic iteration under a session-unique identity.

    ``request`` and ``judge_bundle`` are read back from ``workdir`` when the
    caller does not hold them, since the agent has just written both there. A
    falsy ``session_dir`` is a no-op, and an empty ``kb_priors`` trace is
    omitted from the row rather than recorded blank.
    """
    if not session_dir:
        trace_skip(reason="no session_dir", section=ITERATION_SECTION)
        return
    try:
        review = _dict(review)
        emit = _dict(emit)
        wd = Path(workdir) if workdir else None
        request = _dict(request) or (read_json(wd / "request.json", default={}) if wd else {})
        judge_bundle = _dict(judge_bundle) or (read_json(wd / "judge_bundle.json", default={}) if wd else {})

        topic, body = _spoken_message(emit)
        counts = _verdict_counts(review)
        row: dict[str, Any] = {
            "iter": int(iter_n),
            # Author time: neither the review nor the emit carries a timestamp,
            # and this runs the moment the review comes back.
            "ts": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "verdict": _verdict_rollup(counts),
            "verdict_counts": counts,
            "summary": body[:_SUMMARY_LIMIT],
            "request_path": _rel(wd / "request.json", session_dir) if wd else None,
            "judge_bundle_path": _rel(wd / "judge_bundle.json", session_dir) if wd else None,
            "emit_path": _rel(wd / "emit.json", session_dir) if wd else None,
            "review_path": _rel(wd / "review.json", session_dir) if wd else None,
            "kb_writes": list(emit.get("kb_writes") or []) if isinstance(emit.get("kb_writes"), list) else [],
        }

        context = _dict(request.get("context"))
        phase = str(context.get("phase") or "").strip().upper()
        if phase:
            row["phase"] = phase
        try:
            row["macro_cycle"] = int(context["macro_cycle"])
        except (KeyError, TypeError, ValueError):
            pass

        framework_reviews = normalize_framework_reviews(
            request=request,
            judge_bundle=judge_bundle,
            review=review,
            emit=emit,
            review_path=row["review_path"],
        )
        if framework_reviews:
            row["framework_reviews"] = framework_reviews
        if _dict(kb_priors):
            row["kb_priors"] = dict(kb_priors or {})

        # Content only: ``ts`` is author time and ``topic`` is read off the
        # emit, so neither may seed the identity -- a wall clock in the key
        # would make every re-record a new fragment instead of an overwrite.
        row["iteration_id"] = _stable_id(
            "critic-iteration",
            iter_n,
            [r.get("proposal_msg_id") for r in framework_reviews],
            request,
            judge_bundle,
            review,
            emit,
        )
        recorder_for(session_dir, producer=producer).record_upsert_item(
            ITERATION_SECTION,
            row,
            key=row["iteration_id"],
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("record_critic_iteration failed", exc_info=True)
        trace_skip(reason="writer raised", section=ITERATION_SECTION, error=exc)


__all__ = ["ITERATION_SECTION", "SECTION", "record_critic_iteration"]
