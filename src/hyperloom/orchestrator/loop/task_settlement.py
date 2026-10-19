# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a finished dispatched task's result is shaped for promotion and for the settled dispatch row."""

from __future__ import annotations

import time
from collections.abc import Collection
from typing import Any, NamedTuple

from hyperloom.inference_optimizer.session.optimization_journal import Verdict

from ..state.task_registry import Task
from .sub_agent_runner import SubAgentResult


class PromotionOutcome(NamedTuple):
    """How a finished task's result settled into shared state.

    Attributes:
        verdict: The verdict its promoter reached; ``FAILED`` when nothing promoted it.
        adopted_variants: The variants the promotion adopted.
        raised: Whether promotion itself failed, leaving the rest of the bookkeeping unowed.
    """

    verdict: Verdict
    adopted_variants: Collection[str]
    raised: bool


def replay_dispatch_failed(task: Task, result: SubAgentResult) -> bool:
    """True when a warm replay failed in dispatch and still owes its promote handler a cleanup pass."""
    return task.kind == "replay_warm_recipe" and result.state == "failed"


def carry_reauthor_attempt(task: Task, result: SubAgentResult) -> None:
    """Stamp a re-authored specialist's attempt number onto the result it produced."""
    if not isinstance(result.result, dict):
        return
    reauthor_attempt = (getattr(task, "params", None) or {}).get("reauthor_attempt")
    if reauthor_attempt not in (None, ""):
        result.result.setdefault("reauthor_attempt", reauthor_attempt)


def settlement_payload(task: Task, result: SubAgentResult) -> dict[str, Any]:
    """The payload promotion reads, with a failed warm replay's dispatch error filled in."""
    payload = dict(result.result or {})
    if not replay_dispatch_failed(task, result):
        return payload
    payload.setdefault("status", "failed")
    payload.setdefault("error_class", "dispatch_failed")
    if result.error:
        payload.setdefault("error", str(result.error))
    return payload


def unpromotable_payload(result: SubAgentResult) -> dict[str, Any]:
    """The payload recorded for a result nothing promoted.

    Runner-level failures (no executor, executor raised) carry their class on the result rather
    than in the empty payload.
    """
    payload = dict(result.result or {})
    if result.error_class and not payload.get("error_class"):
        payload["error_class"] = result.error_class
    return payload


def record_action_settlement(
    task: Task,
    result: SubAgentResult,
    payload: dict[str, Any],
    verdict: Verdict,
) -> None:
    """Settle the dispatch row on the phase that ordered it, whichever way promotion ended."""
    from hyperloom.inference_optimizer.breakdown.recorder import phase_event
    from hyperloom.inference_optimizer.breakdown.recorder.phase_records import ActionSettlement

    phase_event.record_settle(
        ActionSettlement(
            task_id=str(task.task_id or ""),
            status=str(result.state or ""),
            decision=verdict.value,
            error_class=payload.get("error_class") or result.error_class,
            workspace=payload.get("workspace"),
            settled_unix=time.time(),
        )
    )
