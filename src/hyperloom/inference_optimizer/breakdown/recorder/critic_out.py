# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Author-time recording of the SBD v6 ``critic`` section.

The critic agent reviews the session as a whole, iteration after iteration,
and each iteration is complete the moment the review comes back: its topic,
its verdict, the summary it wrote and the four artifacts it left behind. This
records it there.

This is the session-level channel and it does not compete with the per-proposal
verdicts, which stay with the proposals they judge. What lives here is the
agent's own run: how many times it was asked, what it was asked about, and what
it concluded each time.

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
    caller does not hold them, since the agent has just written both there.

    Args:
        session_dir (Path | str | None): the session directory; a falsy value
            is a no-op.
        iter_n (int): the process-local critic iteration number.
        review (dict[str, Any] | None): the critic review payload.
        emit (dict[str, Any] | None): the critic emit payload.
        workdir (Path | str | None): the iteration's workdir, holding the four
            artifact files.
        request (dict[str, Any] | None): the critic request payload.
        judge_bundle (dict[str, Any] | None): the proposal bundle reviewed.
        kb_priors (dict[str, Any] | None): the iteration's historical-KB
            priors trace (whether priors were used, the request, the response,
            and whether the verdict referenced them); omitted when empty.
        producer (str): the breakdown producer label.
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

        row: dict[str, Any] = {
            "iter": int(iter_n),
            "ts": str(emit.get("ts") or review.get("ts") or ""),
            "topic": str(emit.get("topic") or review.get("topic") or ""),
            "verdict": str(review.get("verdict") or emit.get("verdict") or ""),
            "summary": str(review.get("summary") or emit.get("summary") or "")[:_SUMMARY_LIMIT],
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

        row["iteration_id"] = _stable_id(
            "critic-iteration",
            iter_n,
            row["ts"],
            [r.get("proposal_msg_id") for r in framework_reviews],
            row["topic"],
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
