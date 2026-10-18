# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Concluding a blocked candidate: its specialist sends it to the scheduler for a repair, or declines it.

The rule gate holds a blocked gain as ``DIAGNOSIS_PENDING`` until one of these lands through it. A repair asks for a
new branch at the candidate's parent, starting from its change and the specialist's diagnosis; one repair per intent
on a commit. A decline records the candidate once a learning of its branch cites it, so what the trade-off taught is
kept. Both are tools on the session's MCP service; the branch is the caller's token's, as for every specialist tool.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, replace
from typing import Any, Protocol, TypeVar

from hyperloom.hosted.mcp_server import CallContext, McpServer, ToolError, ToolForbidden
from hyperloom_multibranch.contracts.fork import ForkAdmission
from hyperloom_multibranch.contracts.intent import Intent
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.notebook.interfaces import RecordSearch, SnapshotQuery
from hyperloom_multibranch.contracts.notebook.peers import TestRequest
from hyperloom_multibranch.contracts.notebook.records import ClaimKey, NotebookSnapshot, TestRecord
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion
from hyperloom_multibranch.contracts.toolbox import SCOPE_SPECIALIST
from hyperloom_multibranch.rulegate.gate import RuleGate

T = TypeVar("T")

_ID = "request_id"
_CANDIDATE = "The request id of the blocked candidate's first test."


class BlockedNotebook(SnapshotQuery, RecordSearch, Protocol):
    """What concluding a blocked candidate reads; its writes go through the rule gate."""


class Refused(Exception):
    """The candidate cannot be concluded this way; the message tells the specialist why."""


class NotYours(Refused):
    """The candidate is another branch's."""


@dataclass(frozen=True)
class _Waiting:
    """A blocked candidate waiting for its diagnosis, and the notebook it was read from."""

    snapshot: NotebookSnapshot
    request: TestRequest
    judgement: Judgement


class BlockedCandidates:
    """Concludes the blocked candidates of every specialist branch, one conclusion at a time, so two calls never
    conclude one candidate twice; run it on one loop."""

    def __init__(self, notebook: BlockedNotebook, gate: RuleGate) -> None:
        self._notebook = notebook
        self._gate = gate
        self._lock = asyncio.Lock()

    async def repair(self, branch: str, request_id: str, diagnosis: str) -> tuple[TestRecord, ForkAdmission]:
        """Ask the scheduler for a branch that repairs the candidate; a rejection records it as the gate does."""
        if not diagnosis.strip():
            raise Refused("a repair request names the root cause and the repair")
        async with self._lock:
            waiting = await self._waiting(branch, request_id)
            request = waiting.request
            intent = _claimed_intent(waiting.snapshot, request)
            open_repair = next(
                (
                    other
                    for other in waiting.snapshot.records
                    if other.conclusion is Conclusion.REPAIR_REQUESTED
                    and other.request.parent_commit == request.parent_commit
                    and other.request.intent_fingerprint == request.intent_fingerprint
                ),
                None,
            )
            if open_repair is not None:
                raise Refused(
                    f"{open_repair.request_id} already asked to repair this intent on {request.parent_commit}; "
                    f"decline {request_id}"
                )
            admission = await self._gate.repair(request, waiting.judgement, intent, diagnosis)
            return await self._notebook.record(request_id), admission

    async def decline(self, branch: str, request_id: str) -> TestRecord:
        """Record the candidate as an inherent trade-off, once a learning of ``branch`` cites it."""
        async with self._lock:
            waiting = await self._waiting(branch, request_id)
            if not any(
                request_id in found.learning.request_ids and found.learning.branch == branch
                for found in waiting.snapshot.learnings
            ):
                raise Refused(
                    f"write what {request_id} taught with notebook_learn, citing it in request_ids, then decline it"
                )
            return await self._gate.decline(waiting.request, waiting.judgement)

    async def _waiting(self, branch: str, request_id: str) -> _Waiting:
        """The candidate ``request_id`` of ``branch`` while it waits for its diagnosis."""
        snapshot = await self._notebook.snapshot()
        record = next((found for found in snapshot.records if found.request_id == request_id), None)
        if record is None:
            raise Refused(f"no test {request_id}")
        if record.request.branch != branch:
            raise NotYours(f"{request_id} is a test of {record.request.branch}; conclude your own branch's")
        judgement = record.judgement
        if judgement is None:
            raise Refused(f"{request_id} is not judged; only a blocked candidate is concluded")
        if judgement.verdict is not Verdict.BLOCKED:
            raise Refused(f"{request_id} is {judgement.verdict.value}, not blocked")
        if record.conclusion is not Conclusion.DIAGNOSIS_PENDING:
            raise Refused(f"{request_id} is already concluded: {_conclusion(record)}")
        return _Waiting(snapshot, record.request, judgement)


def _claimed_intent(snapshot: NotebookSnapshot, request: TestRequest) -> Intent:
    """The intent the candidate's claim names; the scheduler cuts the repair at its parent, for its branch."""
    fingerprint = request.intent_fingerprint
    if fingerprint is None:
        raise Refused(f"{request.request_id} was submitted without an intent, so it cannot be repaired; decline it")
    key = ClaimKey(request.parent_commit, fingerprint)
    claim = next((found for found in snapshot.claims if found.key == key), None)
    if claim is None:
        raise Refused(f"the notebook holds no claim of the intent {request.request_id} tested; decline it")
    # The key's latest claim may be another branch's, of the same intent on the same commit.
    return replace(claim.intent, branch=request.branch)


def register_blocked_tools(
    server: McpServer, *, blocked: BlockedCandidates, home: asyncio.AbstractEventLoop | None = None
) -> None:
    """Register ``request_repair`` and ``decline_repair``.

    ``home`` is the loop the specialist runtime and the scheduler run on: the service answers each call on a loop of
    its own, and the scheduler's admission is safe on its own loop only. Without it, calls run on the caller's loop.
    """
    own = {"scopes": frozenset({SCOPE_SPECIALIST})}
    server.register(
        "request_repair",
        description=(
            "Send a blocked candidate of yours to the scheduler for a repair branch, cut at its parent, whose "
            "specialist starts from its change and your diagnosis; you go on on your branch."
        ),
        schema=_schema(request_id=_CANDIDATE, diagnosis="The root cause of what blocked it and the concrete repair."),
        handler=_repair(blocked, home),
        **own,
    )
    server.register(
        "decline_repair",
        description=(
            "Record a blocked candidate of yours as an inherent trade-off; write its lesson with notebook_learn, "
            "citing the candidate, first."
        ),
        schema=_schema(request_id=_CANDIDATE),
        handler=_decline(blocked, home),
        **own,
    )


def _schema(**described: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            name: {"type": "string", "minLength": 1, "description": text} for name, text in described.items()
        },
        "required": list(described),
        "additionalProperties": False,
    }


def _repair(
    blocked: BlockedCandidates, home: asyncio.AbstractEventLoop | None
) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> dict[str, str]:
        call = blocked.repair(ctx.principal.branch, arguments[_ID], arguments["diagnosis"])
        record, admission = await _at_home(home, call)
        return {
            _ID: record.request_id,
            "status": admission.status.value,
            "reason": admission.reason,
            "branch": admission.branch,
            "conclusion": _conclusion(record),
        }

    return handle


def _decline(
    blocked: BlockedCandidates, home: asyncio.AbstractEventLoop | None
) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> dict[str, str]:
        record = await _at_home(home, blocked.decline(ctx.principal.branch, arguments[_ID]))
        return {_ID: record.request_id, "conclusion": _conclusion(record)}

    return handle


def _conclusion(record: TestRecord) -> str:
    return record.conclusion.value if record.conclusion is not None else ""


async def _at_home(home: asyncio.AbstractEventLoop | None, call: Coroutine[Any, Any, T]) -> T:
    """Await ``call`` on ``home``, turning a refusal into the error the specialist reads."""
    try:
        if home is None or asyncio.get_running_loop() is home:
            return await call
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(call, home))
    except NotYours as error:
        raise ToolForbidden(str(error)) from error
    except Refused as error:
        raise ToolError(str(error)) from error


__all__ = ["BlockedCandidates", "BlockedNotebook", "NotYours", "Refused", "register_blocked_tools"]
