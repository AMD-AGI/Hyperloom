# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What an autonomous branch's agent wrote down in its ledger, read as the learnings it becomes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hyperloom_multibranch.blackboard.local.blackboard import LEARNING_MAX_CHARS
from hyperloom_multibranch.contracts.blackboard.vocab import LearningKind
from hyperloom_multibranch.specialist.agent_manifest import LedgerSource
from hyperloom_multibranch.specialist.agent_onboarding import ENCODING


@dataclass(frozen=True)
class Finding:
    """One finding of an agent's ledger, as the learning it becomes: ``content`` names the agent's kind and id."""

    id: str
    kind: LearningKind
    content: str


def read_findings(path: Path, ledger: LedgerSource) -> list[Finding]:
    """The findings of the kinds ``ledger`` maps that the ledger at ``path`` holds, each in its latest version, in the
    order they first appear; a line the agent is still writing is left for the next read."""

    try:
        lines = path.read_text(encoding=ENCODING).splitlines()
    except FileNotFoundError:
        return []
    latest: dict[str, Finding | None] = {}
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get(ledger.id), str):
            continue
        kind, text = entry.get(ledger.kind), entry.get(ledger.text)
        content = " ".join(f"{kind} {entry[ledger.id]}: {text}".split())
        if len(content) > LEARNING_MAX_CHARS:
            content = content[: LEARNING_MAX_CHARS - 1] + "\u2026"
        fits = kind in ledger.kinds and isinstance(text, str)
        latest[entry[ledger.id]] = Finding(entry[ledger.id], ledger.kinds[kind], content) if fits else None
    return [finding for finding in latest.values() if finding is not None]


__all__ = ["Finding", "read_findings"]
