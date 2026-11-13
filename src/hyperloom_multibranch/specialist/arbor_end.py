# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How an Arbor run that exited 0 ended, as its exit record classes it.

Arbor exits 0 for more than a finished run: its harness may have stopped itself (``aborted``), or it may have stopped
for a person to decide (``retraction_conflict``). Since its host interface 2.3 its exit record says which, in
``outcome``; a record without one, as an older Arbor writes it, is classed here by its reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hyperloom_multibranch.specialist.arbor_adapter import ENCODING, exit_reason

#: The outcomes of a run that exited 0: it ended as it should; it handed its next step on; its harness stopped itself,
#: so what it kept stands but it did not finish; or it stopped for a person to decide.
FINISHED, HANDED_OVER, BROKEN, NEEDS_PERSON = "finished", "handed_over", "broken", "needs_person"
#: The outcome of each exit reason, for a record that names none. A reason not listed is a finished run's.
REASON_OUTCOMES = {
    "aborted": BROKEN,
    "retraction_conflict": NEEDS_PERSON,
    "mapped": HANDED_OVER,
    "ordered": HANDED_OVER,
    "reported": HANDED_OVER,
}


@dataclass(frozen=True)
class ArborEnd:
    """An Arbor run's end: the ``reason`` its exit file gives (else what is wrong with the file), the ``outcome``
    Arbor classes it as (``""`` when the file gives no reason) and the record's ``detail``."""

    reason: str
    outcome: str = ""
    detail: str = ""

    def said(self) -> str:
        """What a branch's log says of it: that the agent finished, with its exit reason, unless the run broke or
        stopped for a person, which is said so with the record's detail: it says what to look at."""

        why = f"{self.reason} ({self.detail})" if self.detail else self.reason
        if self.outcome == BROKEN:
            return f"the agent's run broke: {why}"
        if self.outcome == NEEDS_PERSON:
            return f"the agent's run stopped for a person to decide: {why}"
        return f"the agent finished its run: {self.reason}"


def exit_end(path: Path) -> ArborEnd:
    """How Arbor says its run ended, as the exit file at ``path`` it writes before an exit 0 has it. The outcome is
    the record's own; a record without one gets its reason's (:data:`REASON_OUTCOMES`)."""

    reason = exit_reason(path)
    try:
        document = json.loads(path.read_text(encoding=ENCODING))
    except (OSError, ValueError):
        return ArborEnd(reason)
    if not isinstance(document, dict) or document.get("reason") != reason:
        return ArborEnd(reason)
    outcome, detail = document.get("outcome"), document.get("detail")
    if not isinstance(outcome, str) or not outcome:
        outcome = REASON_OUTCOMES.get(reason, FINISHED)
    return ArborEnd(reason, outcome, detail.strip() if isinstance(detail, str) else "")


__all__ = ["BROKEN", "FINISHED", "HANDED_OVER", "NEEDS_PERSON", "REASON_OUTCOMES", "ArborEnd", "exit_end"]
