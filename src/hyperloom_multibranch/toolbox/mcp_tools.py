# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a specialist can call on the campaign's MCP service: tests, results, the queue, the notebook.

:func:`register_specialist_tools` puts the specialist-scoped tools on a :class:`~hyperloom.hosted.mcp_server.McpServer`
and :class:`McpSpecialistTokens` issues and revokes the token each specialist gets. Every handler takes the branch from the token and never
from the arguments, so a specialist submits, cancels, claims, logs and reads its test results for its own branch
only; the notebook's search and record reads are shared across branches, as the notebook contract has them:

* ``submit_test``, ``test_result``, ``cancel_test`` and ``queue_state`` go through the toolbox interface;
* ``wait_test`` blocks on one test of the branch until its record says it is over or the wait runs out, and then
  reports how far it has got (the toolbox's ``progress``);
* ``start_analysis`` starts ``tracelens`` or ``roofline`` on the branch's head commit (an
  :class:`~.analysis.AnalysisStarter`, the real toolbox);
* ``poll_inbox`` reads the branch's inbox;
* the ``notebook_*`` tools are the notebook's own specialist tools (lookup, claim, release, search, record, branch log,
  brief), bound per call to the notebook handle of the caller's branch.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import enum
import functools
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

from hyperloom.contracts.measurement import PRIORITY_HIGHEST, PRIORITY_LOWEST, Purpose, TestKind
from hyperloom.contracts.measurement import TestRequest as MeasuredRequest
from hyperloom.hosted.mcp_server import (
    CallContext,
    InvalidParamsError,
    McpServer,
    ToolError,
    ToolForbidden,
)
from hyperloom_multibranch.contracts.notebook.interfaces import Notebook, NotebookError, StartupBriefProvider
from hyperloom_multibranch.contracts.toolbox import (
    SCOPE_SPECIALIST,
    SpecialistGrant,
    Backpressure,
    RequestRejected,
    TestProgress,
    Toolbox,
    ToolboxError,
)
from hyperloom_multibranch.notebook.tools import ERROR, notebook_tools
from hyperloom_multibranch.toolbox.analysis import ANALYSIS_TOOLS, CODE_NO_EXECUTOR, AnalysisStarter

if TYPE_CHECKING:
    from hyperloom_multibranch.notebook.learning_search import LearningFinder

SPECIALIST_TOKEN_ENV = "HYPERLOOM_SPECIALIST_MCP_TOKEN"
_OBJECT = "object"
_STRING = "string"
_NULL = "null"
_ARRAY = "array"
_INTEGER = "integer"
_BOOLEAN = "boolean"
_SPECIALIST_SCOPES = frozenset({SCOPE_SPECIALIST})
_NO_ARGUMENTS: dict[str, Any] = {"type": _OBJECT, "properties": {}, "additionalProperties": False}
_REQUEST_ID: dict[str, Any] = {
    "type": _OBJECT,
    "properties": {"request_id": {"type": _STRING, "minLength": 1}},
    "required": ["request_id"],
    "additionalProperties": False,
}
#: The longest one ``wait_test`` call blocks; a specialist waits longer by calling it again. Claude Code aborts an
#: HTTP MCP request after 60 s, so a longer block answers nothing.
WAIT_TEST_MAX_SECONDS = 45
#: How often ``wait_test`` reads the test's record while it waits.
WAIT_POLL_SECONDS = 5.0
_WAIT_TEST: dict[str, Any] = {
    "type": _OBJECT,
    "properties": {
        "request_id": {"type": _STRING, "minLength": 1},
        "wait_seconds": {
            "type": _INTEGER,
            "description": f"Seconds to wait at most, held to 1..{WAIT_TEST_MAX_SECONDS}; the most when omitted.",
        },
    },
    "required": ["request_id"],
    "additionalProperties": False,
}
# The caller's branch is the token's; a request that names a branch or an id is refused by the schema.
_SUBMIT_TEST: dict[str, Any] = {
    "type": _OBJECT,
    "properties": {
        "kind": {"enum": [kind.value for kind in TestKind]},
        "purpose": {"enum": [purpose.value for purpose in Purpose]},
        "parent_commit": {"type": _STRING, "minLength": 1},
        "delta": {"type": _OBJECT, "description": "The recipe delta document: config, components, kernels."},
        "intent_fingerprint": {"type": [_STRING, _NULL]},
        "priority": {"type": [_INTEGER, _NULL], "minimum": PRIORITY_HIGHEST, "maximum": PRIORITY_LOWEST},
        "retest_of": {"type": [_STRING, _NULL]},
        "avoid_nodes": {"type": _ARRAY, "items": {"type": _STRING}},
        "accuracy": {"type": _BOOLEAN},
        "concurrency": {"type": _ARRAY, "items": {"type": _INTEGER, "minimum": 1}},
        "attachments": {"type": _ARRAY, "items": {"type": _OBJECT}},
        "diff_ref": {"type": _STRING},
    },
    "required": ["kind", "purpose", "parent_commit"],
    "additionalProperties": False,
}
# The commit is the branch head, read by the server; a request that names a commit or a branch is refused by the schema.
_START_ANALYSIS: dict[str, Any] = {
    "type": _OBJECT,
    "properties": {
        "tool": {"type": _STRING, "description": f"One of: {', '.join(ANALYSIS_TOOLS)}."},
        "params": {
            "type": _OBJECT,
            "description": "The tool's own parameters. trace_input, commit_id and profile_request_id are set by the server and refused here.",
        },
        "profile_request_id": {
            "type": _STRING,
            "description": "The id of a profile test of this branch whose trace is analysed; required for tracelens, checked by the server.",
        },
    },
    "required": ["tool"],
    "additionalProperties": False,
}
_POLL_INBOX: dict[str, Any] = {
    "type": _OBJECT,
    "properties": {"cursor": {"type": _STRING}},
    "additionalProperties": False,
}

# The specialist runtime submits the confirm retest of a measured win itself, through the toolbox interface.
_CONFIRM_REFUSED_CODE = "confirm_not_for_specialists"
_CONFIRM_REFUSED_REASON = (
    "a specialist does not submit purpose=confirm: the runtime retests a measured win itself, "
    "and a confirm of your own would be measured and never used; submit purpose=explore instead"
)

NotebookFor = Callable[[str], Notebook]
BriefingFor = Callable[[str], StartupBriefProvider]


class McpSpecialistTokens:
    """:class:`~hyperloom_multibranch.contracts.toolbox.SpecialistTokens` on the campaign's MCP service.

    One token per branch, kept in memory: asking again for a branch with a live token of the same scopes returns it,
    so a restarted specialist keeps working with the token its environment already has.
    """

    def __init__(self, server: McpServer) -> None:
        self._server = server
        self._lock = threading.Lock()
        self._live: dict[str, tuple[frozenset[str], SpecialistGrant]] = {}

    def issue(self, branch: str, scopes: frozenset[str]) -> SpecialistGrant:
        with self._lock:
            held = self._live.get(branch)
            if held is not None and held[0] == scopes:
                return held[1]
            self._server.revoke_specialist(branch)
            token = self._server.issue_token(branch, branch, scopes)
            sink = functools.partial(self._server.attach_heartbeat, branch)
            grant = SpecialistGrant(self._server.url, SPECIALIST_TOKEN_ENV, token, sink)
            self._live[branch] = (scopes, grant)
            return grant

    def revoke(self, branch: str) -> None:
        with self._lock:
            self._live.pop(branch, None)
            self._server.revoke_specialist(branch)


def jsonable(value: Any) -> Any:
    """``value`` as JSON data: dataclasses become objects, enums their values, mappings string-keyed objects."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    return value


async def _on(home: asyncio.AbstractEventLoop | None, call: Awaitable[Any]) -> Any:
    """Await a toolbox call on the loop the toolbox lives on.

    The service answers every request on a loop of its own that ends with the request. The toolbox starts its tests as
    tasks of the loop it was started on, so a call made on the request's loop would leave the test running on a loop
    that is about to stop: it would be cancelled the moment the reply went out.
    """
    if home is None or asyncio.get_running_loop() is home:
        return await call
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(call, home))  # type: ignore[arg-type]


async def _own_record(notebooks: NotebookFor, ctx: CallContext, request_id: str) -> Any:
    """The notebook's record of ``request_id``, when the caller's branch asked for it."""
    try:
        record = await notebooks(ctx.principal.branch).record(request_id)
    except NotebookError as exc:
        raise ToolError(f"{type(exc).__name__}: {exc}") from exc
    if record.request.branch != ctx.principal.branch:
        raise ToolForbidden("a specialist reads the tests of its own branch only")
    return record


def _submit_test(
    toolbox: Toolbox, home: asyncio.AbstractEventLoop | None
) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> dict[str, str]:
        try:
            request = MeasuredRequest.from_document({**arguments, "branch_id": ctx.principal.branch})
        except ValueError as exc:  # MeasurementError and RecipeError both are
            raise InvalidParamsError(str(exc)) from exc
        if request.purpose is Purpose.CONFIRM:
            raise ToolError(f"rejected ({_CONFIRM_REFUSED_CODE}): {_CONFIRM_REFUSED_REASON}")
        try:
            return {"request_id": await _on(home, toolbox.submit_test(request))}
        except RequestRejected as exc:
            raise ToolError(f"rejected ({exc.code}): {exc.reason}") from exc
        except Backpressure as exc:
            raise ToolError(f"backpressure: {exc}; submit again later") from exc

    return handle


def _start_analysis(
    analysis: AnalysisStarter | None, home: asyncio.AbstractEventLoop | None
) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> dict[str, str]:
        if analysis is None:
            raise ToolError(f"rejected ({CODE_NO_EXECUTOR}): this campaign runs no asynchronous tool executor")
        call = analysis.submit_analysis(
            ctx.principal.branch,
            str(arguments["tool"]),
            arguments.get("params") or {},
            str(arguments.get("profile_request_id") or ""),
        )
        try:
            return {"job_id": await _on(home, call)}
        except RequestRejected as exc:
            raise ToolError(f"rejected ({exc.code}): {exc.reason}") from exc

    return handle


def _result_document(request_id: str, record: Any) -> dict[str, Any]:
    """What ``test_result`` answers for the notebook's ``record`` of ``request_id``."""
    if record.withdrawn_at is not None:
        state = "withdrawn"
    else:
        state = "in_flight" if record.result is None else "measured"
    return {
        "request_id": request_id,
        "state": state,
        "withdrawal_reason": record.withdrawal_reason,
        "result": jsonable(record.result),
        "judgement": jsonable(record.judgement),
        "conclusion": jsonable(record.conclusion),
    }


def _test_result(notebooks: NotebookFor) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        request_id = arguments["request_id"]
        return _result_document(request_id, await _own_record(notebooks, ctx, request_id))

    return handle


def _is_over(record: Any) -> bool:
    return record.withdrawn_at is not None or record.result is not None


def wait_budget(requested: int | None) -> int:
    """The seconds a ``wait_test`` call waits at most: ``requested`` held to 1..:data:`WAIT_TEST_MAX_SECONDS`."""
    if requested is None:
        return WAIT_TEST_MAX_SECONDS
    return min(max(int(requested), 1), WAIT_TEST_MAX_SECONDS)


def _waiting_document(request_id: str, waited: float, progress: TestProgress | None) -> dict[str, Any]:
    """What ``wait_test`` answers for a test that is not over: its state and the toolbox's progress, in seconds.

    A test the toolbox neither queues nor runs (its outcome is logged but not yet in the notebook) is ``in_flight``
    with no progress.
    """
    document: dict[str, Any] = {
        "request_id": request_id,
        "state": "in_flight" if progress is None else progress.state,
        "waited_seconds": round(waited),
    }
    if progress is not None:
        document["progress"] = {
            name: round(value) if isinstance(value, float) else value
            for name, value in jsonable(progress).items()
            if name != "state" and value is not None
        }
    return document


def _wait_test(
    toolbox: Toolbox, notebooks: NotebookFor, home: asyncio.AbstractEventLoop | None, poll_seconds: float
) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        request_id = arguments["request_id"]
        budget, started = wait_budget(arguments.get("wait_seconds")), time.monotonic()
        record = await _own_record(notebooks, ctx, request_id)
        while not _is_over(record) and (left := budget - (time.monotonic() - started)) > 0:
            await asyncio.sleep(min(poll_seconds, left))
            record = await _own_record(notebooks, ctx, request_id)
        if _is_over(record):
            return _result_document(request_id, record)
        progress = await _on(home, toolbox.progress(request_id))
        return _waiting_document(request_id, time.monotonic() - started, progress)

    return handle


def _cancel_test(
    toolbox: Toolbox, notebooks: NotebookFor, home: asyncio.AbstractEventLoop | None
) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> dict[str, str]:
        await _own_record(notebooks, ctx, arguments["request_id"])
        try:
            return {"outcome": (await _on(home, toolbox.cancel(arguments["request_id"]))).value}
        except ToolboxError as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return handle


def _queue_state(
    toolbox: Toolbox, home: asyncio.AbstractEventLoop | None
) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        state = await _on(home, toolbox.queue_state())
        return {**jsonable(state), "queued": state.queued, "running": state.running}

    return handle


def _poll_inbox(notebooks: NotebookFor) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> Any:
        page = await notebooks(ctx.principal.branch).poll(ctx.principal.branch, str(arguments.get("cursor") or ""))
        return jsonable(page)

    return handle


def _notebook_handler(
    name: str,
    notebooks: NotebookFor,
    ttl_seconds: float,
    learnings: LearningFinder | None = None,
    briefing_for: BriefingFor | None = None,
) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> str:
        branch = ctx.principal.branch
        briefing = briefing_for(branch) if briefing_for is not None else None
        bound = notebook_tools(notebooks(branch), briefing, branch=branch, ttl_seconds=ttl_seconds, learnings=learnings)
        tools = {tool.name: tool for tool in bound}
        text = await tools[name].handler(arguments)
        if text.startswith(ERROR):
            raise ToolError(text[len(ERROR) :].strip())
        return text

    return handle


@dataclasses.dataclass(frozen=True)
class SpecialistToolOptions:
    """What :func:`register_specialist_tools` may be told besides the toolbox and the notebooks.

    ``claim_ttl_seconds`` is how long a claim a specialist takes lives. ``toolbox_loop`` is the loop the toolbox was started on; every call to the toolbox is made on it. When omitted it is
    the running loop of the caller, so registering from the coroutine that started the toolbox is enough; without a
    running loop the toolbox is called on the loop of each request, which suits a toolbox that keeps no tasks.
    ``learnings`` is the session's one finder of the specialists' learnings (the notebook's ``LearningRecall``, which
    adds past sessions' from the Experience KB); without it ``notebook_learnings`` finds this session's only.
    ``briefing_for`` gives the provider of the brief of a branch (the notebook's ``SessionBriefs.for_branch``), called
    with the caller's branch from the token at every ``notebook_brief``; without it the notebook's own brief is returned.
    ``analysis`` starts ``start_analysis`` jobs (the real toolbox is one); without it that tool refuses by name.
    ``wait_poll_seconds`` is how often ``wait_test`` reads the record of the test it waits for.
    """

    claim_ttl_seconds: float = 600.0
    toolbox_loop: asyncio.AbstractEventLoop | None = None
    learnings: LearningFinder | None = None
    briefing_for: BriefingFor | None = None
    analysis: AnalysisStarter | None = None
    wait_poll_seconds: float = WAIT_POLL_SECONDS


def register_specialist_tools(
    server: McpServer,
    *,
    toolbox: Toolbox,
    notebooks: NotebookFor,
    options: SpecialistToolOptions = SpecialistToolOptions(),
) -> None:
    """Register every specialist tool; ``notebooks(branch)`` is the notebook handle that writes as that branch."""
    claim_ttl_seconds, toolbox_loop, learnings = options.claim_ttl_seconds, options.toolbox_loop, options.learnings
    if toolbox_loop is None:
        with contextlib.suppress(RuntimeError):
            toolbox_loop = asyncio.get_running_loop()
    own = {"scopes": _SPECIALIST_SCOPES}
    server.register(
        "submit_test",
        description="Submit a test of a recipe change for this branch; returns the request id once accepted.",
        schema=_SUBMIT_TEST,
        handler=_submit_test(toolbox, toolbox_loop),
        **own,
    )
    server.register(
        "test_result",
        description="The state and result of one test this branch submitted.",
        schema=_REQUEST_ID,
        handler=_test_result(notebooks),
        **own,
    )
    server.register(
        "wait_test",
        description=(
            "How to wait for a test of this branch: returns as soon as its result lands, else after wait_seconds "
            "with the test's progress; call it again to keep waiting."
        ),
        schema=_WAIT_TEST,
        handler=_wait_test(toolbox, notebooks, toolbox_loop, options.wait_poll_seconds),
        **own,
    )
    server.register(
        "cancel_test",
        description="Take back a test of this branch that has not finished.",
        schema=_REQUEST_ID,
        handler=_cancel_test(toolbox, notebooks, toolbox_loop),
        **own,
    )
    server.register(
        "queue_state",
        description="Queue lengths by priority, running tests by branch, and the sandbox cold start time.",
        schema=_NO_ARGUMENTS,
        handler=_queue_state(toolbox, toolbox_loop),
        **own,
    )
    server.register(
        "start_analysis",
        description=(
            "Start tracelens (analyse a trace) or roofline (profile, then analyse) for this branch; returns the job "
            "id at once and the result arrives in poll_inbox. The commit analysed is this branch's head, set by the "
            "server; with a profile_request_id of a profile test of this branch the pointer carries its op shares."
        ),
        schema=_START_ANALYSIS,
        handler=_start_analysis(options.analysis, toolbox_loop),
        **own,
    )
    server.register(
        "poll_inbox",
        description="Pointers pushed to this branch since the cursor, such as results of asynchronous tools.",
        schema=_POLL_INBOX,
        handler=_poll_inbox(notebooks),
        **own,
    )
    # Binding to no notebook is enough to read the definitions; no handler of this probe is ever called.
    for tool in notebook_tools(cast(Notebook, None), branch="", ttl_seconds=claim_ttl_seconds):
        server.register(
            tool.name,
            description=tool.description,
            schema=tool.input_schema,
            handler=_notebook_handler(tool.name, notebooks, claim_ttl_seconds, learnings, options.briefing_for),
            **own,
        )


__all__ = [
    "SPECIALIST_TOKEN_ENV",
    "BriefingFor",
    "NotebookFor",
    "McpSpecialistTokens",
    "SpecialistToolOptions",
    "WAIT_POLL_SECONDS",
    "WAIT_TEST_MAX_SECONDS",
    "jsonable",
    "register_specialist_tools",
    "wait_budget",
]
