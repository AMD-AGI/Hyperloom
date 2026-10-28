# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The judge's decisions, kept beside the archive, so that R2 and the offline generations read any session's.

A session's judge appends every exchange to its transcript in the session's directory, which no other session reads.
At CLOSE :func:`archive_decisions` copies each exchange, when the judge was asked, what, and what it answered, into
``<archive>/decisions/<session_id>.json``, written once, its secrets and host paths taken out as the Recipe KB's shared
knowledge is (``sanitize_shared_knowledge``). :func:`decision_calls` reads them back as the calls
:func:`~hyperloom_multibranch.blackboard.rsi.ladder.decision_points` takes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from hyperloom.common.jsonio import read_jsonl
from hyperloom.orchestrator.knowledge.remote_recipe.sanitize import sanitize_shared_knowledge
from hyperloom_multibranch.blackboard.local.files import encode_json, publish_once
from hyperloom_multibranch.blackboard.rsi.archive import TreeArchive

DECISIONS_FORMAT = "multibranch.judge-decisions"
#: What each archived exchange keeps of the transcript's line.
KEPT = ("at", "prompt", "reply")
_DECISIONS = "decisions"
_CALLS = "calls"


def _path(archive: TreeArchive, session_id: str) -> Path:
    return archive.root.parent / _DECISIONS / f"{re.sub(r'[^A-Za-z0-9._-]+', '-', session_id)}.json"


def archive_decisions(transcript: Path, archive: TreeArchive, session_id: str) -> int | None:
    """Keep the exchanges of the judge's ``transcript`` beside ``archive`` for ``session_id``; how many, ``None`` when
    the session's are kept already."""

    lines = read_jsonl(transcript) if transcript.is_file() else []
    calls = [sanitize_shared_knowledge({key: line.get(key) for key in KEPT}) for line in lines]
    document = {"format": DECISIONS_FORMAT, "session_id": session_id, _CALLS: calls}
    return len(calls) if publish_once(_path(archive, session_id), encode_json(document)) else None


def decision_calls(archive: TreeArchive, session_id: str) -> tuple[Mapping[str, object], ...]:
    """The judge's exchanges of ``session_id`` the archive keeps; none when it keeps none, or not as this module
    writes them."""

    try:
        document = json.loads(_path(archive, session_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    calls = document.get(_CALLS) if isinstance(document, dict) and document.get("format") == DECISIONS_FORMAT else ()
    return tuple(call for call in calls or () if isinstance(call, Mapping))


__all__ = ["DECISIONS_FORMAT", "KEPT", "archive_decisions", "decision_calls"]
