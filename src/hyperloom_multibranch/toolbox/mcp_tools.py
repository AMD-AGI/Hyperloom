# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a specialist can call on the campaign's MCP service: tests, results, the queue, the notebook.

:func:`register_specialist_tools` puts the specialist-scoped tools on a :class:`~hyperloom.hosted.mcp_server.McpServer`
and :func:`grant_access` issues the token one specialist gets. Every handler takes the branch from the token and never
from the arguments, so a specialist submits, reads, cancels, claims and logs for its own branch only:

* ``submit_test``, ``test_result``, ``cancel_test`` and ``queue_state`` go through the toolbox interface;
* ``poll_inbox`` reads the branch's inbox;
* the ``notebook_*`` tools are the notebook's own specialist tools (lookup, claim, release, search, record, branch log,
  brief), bound per call to the notebook handle of the caller's branch.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable, Mapping
from typing import Any, cast

from hyperloom.contracts.measurement import PRIORITY_HIGHEST, PRIORITY_LOWEST, Purpose, TestKind
from hyperloom.contracts.measurement import TestRequest as MeasuredRequest
from hyperloom.hosted.mcp_server import (
    SPECIALIST_KIND,
    CallContext,
    InvalidParamsError,
    McpServer,
    ToolError,
    ToolForbidden,
)
from hyperloom_multibranch.contracts.notebook.interfaces import Notebook, NotebookError
from hyperloom_multibranch.contracts.toolbox import (
    Backpressure,
    RequestRejected,
    Toolbox,
    ToolboxError,
)
from hyperloom_multibranch.notebook.tools import ERROR, notebook_tools

SPECIALIST_TOKEN_ENV = "HYPERLOOM_SPECIALIST_MCP_TOKEN"
_SPECIALIST_SCOPES = frozenset({SPECIALIST_KIND})
_NO_ARGUMENTS: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}
_REQUEST_ID: dict[str, Any] = {
    "type": "object",
    "properties": {"request_id": {"type": "string", "minLength": 1}},
    "required": ["request_id"],
    "additionalProperties": False,
}
# The caller's branch is the token's; a request that names a branch or an id is refused by the schema.
_SUBMIT_TEST: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"enum": [kind.value for kind in TestKind]},
        "purpose": {"enum": [purpose.value for purpose in Purpose]},
        "parent_commit": {"type": "string", "minLength": 1},
        "delta": {"type": "object", "description": "The recipe delta document: config, components, kernels."},
        "intent_fingerprint": {"type": ["string", "null"]},
        "priority": {"type": ["integer", "null"], "minimum": PRIORITY_HIGHEST, "maximum": PRIORITY_LOWEST},
        "retest_of": {"type": ["string", "null"]},
        "avoid_nodes": {"type": "array", "items": {"type": "string"}},
        "accuracy": {"type": "boolean"},
        "concurrency": {"type": "array", "items": {"type": "integer", "minimum": 1}},
        "attachments": {"type": "array", "items": {"type": "object"}},
        "diff_ref": {"type": "string"},
    },
    "required": ["kind", "purpose", "parent_commit"],
    "additionalProperties": False,
}
_POLL_INBOX: dict[str, Any] = {
    "type": "object",
    "properties": {"cursor": {"type": "string"}},
    "additionalProperties": False,
}

NotebookFor = Callable[[str], Notebook]


@dataclasses.dataclass(frozen=True)
class SpecialistAccess:
    """What the AI socket hands a specialist's CLI: the service address and the token, which travels in ``token_env``."""

    url: str
    token_env: str
    token: str = dataclasses.field(repr=False)


def grant_access(server: McpServer, *, specialist: str, branch: str) -> SpecialistAccess:
    """The access of one specialist on ``branch``; :meth:`McpServer.revoke_specialist` takes it back when it ends."""
    return SpecialistAccess(server.url, SPECIALIST_TOKEN_ENV, server.issue_token(specialist, branch))


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


async def _own_record(notebooks: NotebookFor, ctx: CallContext, request_id: str) -> Any:
    """The notebook's record of ``request_id``, when the caller's branch asked for it."""
    try:
        record = await notebooks(ctx.principal.branch).record(request_id)
    except NotebookError as exc:
        raise ToolError(f"{type(exc).__name__}: {exc}") from exc
    if record.request.branch != ctx.principal.branch:
        raise ToolForbidden("a specialist reads the tests of its own branch only")
    return record


def _submit_test(toolbox: Toolbox) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> dict[str, str]:
        try:
            request = MeasuredRequest.from_document({**arguments, "branch_id": ctx.principal.branch})
        except ValueError as exc:  # MeasurementError and RecipeError both are
            raise InvalidParamsError(str(exc)) from exc
        try:
            return {"request_id": await toolbox.submit_test(request)}
        except RequestRejected as exc:
            raise ToolError(f"rejected ({exc.code}): {exc.reason}") from exc
        except Backpressure as exc:
            raise ToolError(f"backpressure: {exc}; submit again later") from exc

    return handle


def _test_result(notebooks: NotebookFor) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        record = await _own_record(notebooks, ctx, arguments["request_id"])
        if record.withdrawn_at is not None:
            state = "withdrawn"
        else:
            state = "in_flight" if record.result is None else "measured"
        return {
            "request_id": arguments["request_id"],
            "state": state,
            "withdrawal_reason": record.withdrawal_reason,
            "result": jsonable(record.result),
            "judgement": jsonable(record.judgement),
            "conclusion": jsonable(record.conclusion),
        }

    return handle


def _cancel_test(toolbox: Toolbox, notebooks: NotebookFor) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> dict[str, str]:
        await _own_record(notebooks, ctx, arguments["request_id"])
        try:
            return {"outcome": (await toolbox.cancel(arguments["request_id"])).value}
        except ToolboxError as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return handle


def _queue_state(toolbox: Toolbox) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        state = await toolbox.queue_state()
        return {**jsonable(state), "queued": state.queued, "running": state.running}

    return handle


def _poll_inbox(notebooks: NotebookFor) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> Any:
        page = await notebooks(ctx.principal.branch).poll(ctx.principal.branch, str(arguments.get("cursor") or ""))
        return jsonable(page)

    return handle


def _notebook_handler(
    name: str, notebooks: NotebookFor, ttl_seconds: float
) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> str:
        branch = ctx.principal.branch
        tools = {tool.name: tool for tool in notebook_tools(notebooks(branch), branch=branch, ttl_seconds=ttl_seconds)}
        text = await tools[name].handler(arguments)
        if text.startswith(ERROR):
            raise ToolError(text[len(ERROR) :].strip())
        return text

    return handle


def register_specialist_tools(
    server: McpServer, *, toolbox: Toolbox, notebooks: NotebookFor, claim_ttl_seconds: float = 600.0
) -> None:
    """Register every specialist tool; ``notebooks(branch)`` is the notebook handle that writes as that branch."""
    own = {"scopes": _SPECIALIST_SCOPES}
    server.register(
        "submit_test",
        description="Submit a test of a recipe change for this branch; returns the request id once accepted.",
        schema=_SUBMIT_TEST,
        handler=_submit_test(toolbox),
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
        "cancel_test",
        description="Take back a test of this branch that has not finished.",
        schema=_REQUEST_ID,
        handler=_cancel_test(toolbox, notebooks),
        **own,
    )
    server.register(
        "queue_state",
        description="Queue lengths by priority, running tests by branch, and the sandbox cold start time.",
        schema=_NO_ARGUMENTS,
        handler=_queue_state(toolbox),
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
            handler=_notebook_handler(tool.name, notebooks, claim_ttl_seconds),
            **own,
        )


__all__ = [
    "SPECIALIST_TOKEN_ENV",
    "NotebookFor",
    "SpecialistAccess",
    "grant_access",
    "jsonable",
    "register_specialist_tools",
]
