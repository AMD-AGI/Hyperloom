# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The predictor's source-change channel.

An answer that describes a source edit files it as a mandate on its queue round.
The queue block offers the newest one no specialist has taken; orchestration
dispatches a patch specialist citing only the mandate id, and the router swaps
in the predictor's own text, so the work is attributed without trusting an LLM
to label its sources or to copy the prose faithfully.
"""

from __future__ import annotations

from typing import Any

from hyperloom.common.coerce import to_int
from hyperloom.common.prompt_safety import flatten_for_prompt
from hyperloom.orchestrator.predictor.client import Prediction

#: Model-authored text entering another model's prompt stays short.
MAX_MANDATE_CHARS = 4000


def patch_mandate(prediction: Prediction, *, key: str) -> dict[str, str] | None:
    """``{mandate_id, mandate}`` for the answer's source change, or ``None``; the id is stable per decision point."""
    # Flattened, because the text reaches the queue block and a specialist prompt verbatim.
    text = flatten_for_prompt(prediction.source_change)[:MAX_MANDATE_CHARS].strip()
    return {"mandate_id": f"primatune-patch-{key}", "mandate": text} if text else None


def _rounds_citing(state: Any, mandate_id: Any) -> list[dict[str, Any]]:
    wanted = str(mandate_id or "").strip()
    if not wanted:
        return []
    return [
        entry
        for entry in state.specialist_rounds or []
        if isinstance(entry, dict) and str(entry.get("mandate_id") or "") == wanted
    ]


def find_mandate(state: Any, mandate_id: Any) -> str:
    """The mandate text for ``mandate_id``, or ``""`` when no round carries it."""
    rounds = _rounds_citing(state, mandate_id)
    return str(rounds[-1].get("mandate") or "") if rounds else ""


def mark_mandate_consumed(state: Any, mandate_id: Any, task_id: str) -> bool:
    """Stop offering a mandate once a specialist task exists for it; the first task keeps the credit."""
    rounds = _rounds_citing(state, mandate_id)
    for entry in rounds:
        if not entry.get("mandate_consumed_by"):
            entry["mandate_consumed_by"] = str(task_id)
    return bool(rounds)


def open_mandate(state: Any) -> dict[str, str]:
    """The newest mandate of this macro-cycle no specialist has taken, or ``{}``.

    Older ones answered a stack the session has since moved past.
    """
    cycle = to_int(state.macro_cycle, default=0)
    for entry in reversed(state.specialist_rounds or []):
        if (
            isinstance(entry, dict)
            and to_int(entry.get("cycle"), default=0) == cycle
            and entry.get("mandate")
            and not entry.get("mandate_consumed_by")
        ):
            return {"mandate_id": str(entry.get("mandate_id") or ""), "mandate": str(entry["mandate"])}
    return {}
