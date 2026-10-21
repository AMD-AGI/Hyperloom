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
* ``poll_inbox`` reads the branch's inbox, one page at a time (:data:`POLL_INBOX_MAX_ITEMS` items at most);
* the ``notebook_*`` tools are the notebook's own specialist tools (lookup, claim, release, search, record, branch log,
  brief), bound per call to the notebook handle of the caller's branch.

Every reply to a specialist also carries what reached its branch's inbox since the token's last reply, as one more text
block after the tool's own content (:func:`reply_inbox`); ``poll_inbox`` keeps a cursor of its own. A cursor is the
notebook's, or, in the middle of a feed too long for one page, that cursor with the number of its items already
returned and a fingerprint of the last (``page:<n>:<fingerprint>:<notebook cursor>``); callers treat both as opaque.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import enum
import functools
import hashlib
import itertools
import json
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from hyperloom.contracts.measurement import PRIORITY_HIGHEST, PRIORITY_LOWEST, Purpose, TestKind
from hyperloom.contracts.measurement import TestRequest as MeasuredRequest
from hyperloom.contracts.recipe import KernelKind
from hyperloom.hosted.mcp_server import (
    CallContext,
    InvalidParamsError,
    McpServer,
    ReplyAnnotator,
    ReplyCall,
    ReplyNote,
    ToolError,
    ToolForbidden,
)
from hyperloom.measure.compile import LOADABLE_KERNEL_KINDS
from hyperloom_multibranch.contracts.notebook.interfaces import Notebook, NotebookError, StartupBriefProvider
from hyperloom_multibranch.contracts.notebook.records import InboxItem
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
#: How many inbox items one reply's block lists; the rest are counted.
REPLY_INBOX_LINES = 20
REPLY_INBOX_HEADER = "New for your branch since your last call:"
# These replies carry the test they were asked about, so an inbox item naming it would say it twice.
_CARRIES_REQUEST = frozenset({"wait_test", "test_result"})
#: ``poll_inbox`` lists the inbox itself; its reply carries no block, and the items it returned count as seen.
_POLL_INBOX_TOOL = "poll_inbox"
#: One ``poll_inbox`` reply holds at most this many items and this many characters of JSON; the rest is paged. Claude
#: Code warns at 10,000 tokens of MCP output and saves a text result over 50,000 characters to a file instead of
#: showing it; 16,000 characters stay near 5,000 tokens.
POLL_INBOX_MAX_ITEMS = 20
POLL_INBOX_MAX_BYTES = 16_000
#: What a page's document needs besides its items and its cursor: the keys, the counts and the marker.
_POLL_PAGE_OVERHEAD = 200
_PAGE_CURSOR = "page:"
_FINGERPRINT_CHARS = 8
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
        "delta": {
            "type": _OBJECT,
            "description": (
                "The recipe delta document: config, components, kernels. A component the recipe has takes patches "
                "only: its upstream_ref and upstream_repo are refused as upstream_label_unsupported."
            ),
        },
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
_PUT_KERNEL_ARTIFACT: dict[str, Any] = {
    "type": _OBJECT,
    "properties": {
        "kind": {"enum": sorted(kind.value for kind in LOADABLE_KERNEL_KINDS)},
        "target": {
            "type": _STRING,
            "minLength": 1,
            "description": (
                "The operator or shape the table serves, one path segment (letters, digits and ._=,+-); the merge "
                "slot of the artifact: two branches that put a table of one target compete for it."
            ),
        },
        "env": {
            "type": _STRING,
            "minLength": 1,
            "description": "The environment variable the service reads the table's path from, e.g. AITER_CONFIG_GEMM_A8W8.",
        },
        "content": {"type": _STRING, "minLength": 1, "description": "The whole table as UTF-8 text, a CSV."},
    },
    "required": ["kind", "target", "env", "content"],
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


async def _own_record(
    notebooks: NotebookFor, ctx: CallContext, request_id: str, *, profiles_shared: bool = False
) -> Any:
    """The notebook's record of ``request_id``, when the caller's branch asked for it.

    With ``profiles_shared`` a PROFILE test of any branch is also readable: ``submit_test`` refuses a second profile
    of a commit with the id of the first (``profile_exists``), which may be another branch's, and the caller waits on
    it. Taking a test back stays with its own branch.
    """
    try:
        record = await notebooks(ctx.principal.branch).record(request_id)
    except NotebookError as exc:
        raise ToolError(f"{type(exc).__name__}: {exc}") from exc
    shared = profiles_shared and record.request.kind is TestKind.PROFILE
    if record.request.branch != ctx.principal.branch and not shared:
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


def _put_kernel_artifact(
    toolbox: Toolbox, home: asyncio.AbstractEventLoop | None
) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], _ctx: CallContext) -> dict[str, object]:
        try:
            data = arguments["content"].encode()
        except UnicodeEncodeError as exc:
            raise InvalidParamsError("content is not UTF-8 text") from exc
        call = toolbox.put_kernel_artifact(KernelKind(arguments["kind"]), arguments["target"], arguments["env"], data)
        try:
            artifact = await _on(home, call)
        except RequestRejected as exc:
            raise ToolError(f"rejected ({exc.code}): {exc.reason}") from exc
        return artifact.to_document()

    return handle


def _register_put_kernel_artifact(server: McpServer, toolbox: Toolbox, home: asyncio.AbstractEventLoop | None) -> None:
    server.register(
        "put_kernel_artifact",
        description=(
            "Store a tuned table, a CSV the service loads from a file named by an environment variable, and get the "
            "kernel artifact to put in the add_kernels of the delta of submit_test. Put every shape of an operator "
            "in one table: a variable loads one file, so the first table put for an env, by any branch, binds the env "
            "to its target for the session. Any branch may then put its own table (new content, its own reference) "
            "under that target; only a table of another target name for the same env is refused, as "
            "kernel_variable_taken, and its message names the bound target. The same table put twice, by any branch, "
            "is the same artifact."
        ),
        schema=_PUT_KERNEL_ARTIFACT,
        handler=_put_kernel_artifact(toolbox, home),
        scopes=_SPECIALIST_SCOPES,
    )


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
        return _result_document(request_id, await _own_record(notebooks, ctx, request_id, profiles_shared=True))

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
        record = await _own_record(notebooks, ctx, request_id, profiles_shared=True)
        while not _is_over(record) and (left := budget - (time.monotonic() - started)) > 0:
            await asyncio.sleep(min(poll_seconds, left))
            record = await _own_record(notebooks, ctx, request_id, profiles_shared=True)
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


@dataclasses.dataclass(frozen=True)
class _Feed:
    """The inbox items of a branch not yet returned from one cursor: ``base`` is the notebook's cursor they were read
    from, ``skipped`` how many of its items earlier pages returned, ``anchor`` the fingerprint of the last of those,
    and ``end`` the notebook's cursor past all."""

    base: str
    skipped: int
    anchor: str
    items: tuple[InboxItem, ...]
    end: str

    def after(self, count: int) -> str:
        """The cursor that resumes after the first ``count`` of the items."""
        if count >= len(self.items):
            return self.end
        anchor = _fingerprint(self.items[count - 1]) if count else self.anchor
        return f"{_PAGE_CURSOR}{self.skipped + count}:{anchor}:{self.base}"


def _fingerprint(item: InboxItem) -> str:
    """A short fingerprint of ``item``, to find it again in a feed that has moved."""
    text = f"{item.kind.value}|{item.pointer}|{item.at}|{item.summary}"
    return hashlib.sha256(text.encode()).hexdigest()[:_FINGERPRINT_CHARS]


async def _read_feed(notebook: Notebook, branch: str, cursor: str) -> _Feed:
    """The feed of ``branch`` from ``cursor``. The notebook computes some items from the claims the branch holds, so
    a feed can change between two pages: a page cursor resumes after the item its page ended with, found by its
    fingerprint, and by the count only where that item is gone."""
    base, skipped, anchor = cursor, 0, ""
    if cursor.startswith(_PAGE_CURSOR):
        parts = cursor[len(_PAGE_CURSOR) :].split(":", 2)
        if len(parts) != 3 or not parts[0].isdigit() or len(parts[1]) != _FINGERPRINT_CHARS:
            raise InvalidParamsError(f"not an inbox cursor: {cursor!r}")
        skipped, anchor, base = int(parts[0]), parts[1], parts[2]
    try:
        page = await notebook.poll(branch, base)
    except ValueError as exc:
        raise InvalidParamsError(
            f"{exc}; pass the cursor of an earlier reply, or none to read the whole inbox"
        ) from None
    items = page.items
    if skipped and not (skipped <= len(items) and _fingerprint(items[skipped - 1]) == anchor):
        found = [at for at, item in enumerate(items) if _fingerprint(item) == anchor]
        skipped = min(found, key=lambda at: abs(at + 1 - skipped)) + 1 if found else min(skipped, len(items))
    return _Feed(base, skipped, anchor, items[skipped:], page.cursor)


def _item_size(item: InboxItem) -> int:
    return len(json.dumps(jsonable(item), default=str)) + 2


def _page_of(feed: _Feed) -> tuple[InboxItem, ...]:
    """The first items of ``feed`` one ``poll_inbox`` reply holds: at most :data:`POLL_INBOX_MAX_ITEMS` and
    :data:`POLL_INBOX_MAX_BYTES` with the cursor counted in, but always one so that every page moves on (the notebook
    bounds one item well under the budget)."""
    page: list[InboxItem] = []
    used = _POLL_PAGE_OVERHEAD + len(json.dumps(feed.end))
    for item in feed.items[:POLL_INBOX_MAX_ITEMS]:
        used += _item_size(item)
        if page and used > POLL_INBOX_MAX_BYTES:
            break
        page.append(item)
    return tuple(page)


def _poll_inbox(notebooks: NotebookFor) -> Callable[[dict[str, Any], CallContext], Any]:
    async def handle(arguments: dict[str, Any], ctx: CallContext) -> dict[str, Any]:
        branch = ctx.principal.branch
        feed = await _read_feed(notebooks(branch), branch, str(arguments.get("cursor") or ""))
        page = _page_of(feed)
        remaining = len(feed.items) - len(page)
        document: dict[str, Any] = {
            "items": jsonable(page),
            "cursor": feed.after(len(page)),
            "remaining": remaining,
        }
        if remaining:
            document["more"] = f"{remaining} more; call poll_inbox with this cursor"
        return document

    return handle


def _inbox_block(items: Sequence[InboxItem]) -> str:
    """The text of a reply's block for ``items``, empty for none; past :data:`REPLY_INBOX_LINES` the rest are counted."""
    if not items:
        return ""
    lines = [f"{item.kind.value} {item.pointer}: {item.summary}" for item in items[:REPLY_INBOX_LINES]]
    if len(items) > REPLY_INBOX_LINES:
        lines.append(f"... {len(items) - REPLY_INBOX_LINES} more, poll_inbox lists them")
    return "\n".join([REPLY_INBOX_HEADER, *lines])


def reply_inbox(notebooks: NotebookFor) -> ReplyAnnotator:
    """The reply annotator that adds what arrived in the caller's branch inbox since its token's last reply.

    A token's first call is asked with no cursor, so the whole feed is read and the service drops that text: a late
    first call or a restarted service starts from the end. The cursor is the notebook's own, kept by the service per
    token, apart from ``poll_inbox``'s. A ``poll_inbox`` reply carries no block, and only the items it returned count
    as seen: the cursor moves past the leading ones of the unread items that it returned, and no further.
    """

    async def annotate(call: ReplyCall, cursor: str | None) -> ReplyNote:
        branch = call.principal.branch
        notebook = notebooks(branch)
        unread = await _read_feed(notebook, branch, cursor or "")
        if call.tool == _POLL_INBOX_TOOL and cursor is not None:
            returned = set() if call.is_error else await _returned_by_poll(notebook, call)
            seen = sum(1 for _ in itertools.takewhile(returned.__contains__, unread.items))
            return ReplyNote("", unread.after(seen))
        asked = call.arguments.get("request_id") if call.tool in _CARRIES_REQUEST and not call.is_error else None
        return ReplyNote(_inbox_block([item for item in unread.items if item.pointer != asked]), unread.end)

    return annotate


async def _returned_by_poll(notebook: Notebook, call: ReplyCall) -> set[InboxItem]:
    """The items the ``poll_inbox`` call returned, read again from the cursor it was given."""
    feed = await _read_feed(notebook, call.principal.branch, str(call.arguments.get("cursor") or ""))
    return set(_page_of(feed))


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
    toolbox_loop = options.toolbox_loop
    if toolbox_loop is None:
        with contextlib.suppress(RuntimeError):
            toolbox_loop = asyncio.get_running_loop()
    own = {"scopes": _SPECIALIST_SCOPES}
    server.attach_reply_annotator(reply_inbox(notebooks))
    server.register(
        "submit_test",
        description=(
            "Submit a test of a recipe change for this branch; returns the request id once accepted. A profile of a "
            "commit that already has one queued, running or measured is refused as profile_exists and names that "
            "request: wait_test on it, or read the commit's op shares from the notebook."
        ),
        schema=_SUBMIT_TEST,
        handler=_submit_test(toolbox, toolbox_loop),
        **own,
    )
    _register_put_kernel_artifact(server, toolbox, toolbox_loop)
    server.register(
        "test_result",
        description="The state and result of one test this branch submitted, or of any branch's profile test.",
        schema=_REQUEST_ID,
        handler=_test_result(notebooks),
        **own,
    )
    server.register(
        "wait_test",
        description=(
            "How to wait for a test of this branch, or for any branch's profile test: returns as soon as its result "
            "lands, else after wait_seconds with the test's progress; call it again to keep waiting."
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
        _POLL_INBOX_TOOL,
        description=(
            "Pointers pushed to this branch since the cursor, such as results of asynchronous tools. A reply holds "
            f"at most {POLL_INBOX_MAX_ITEMS} items; when remaining is above 0, call again with the cursor it returned "
            "to read the next page."
        ),
        schema=_POLL_INBOX,
        handler=_poll_inbox(notebooks),
        **own,
    )
    _register_notebook_tools(server, notebooks, options)


def _register_notebook_tools(server: McpServer, notebooks: NotebookFor, options: SpecialistToolOptions) -> None:
    """Register the notebook's tools, whose definitions the notebook itself owns."""
    # Binding to no notebook is enough to read the definitions; no handler of this probe is ever called.
    for tool in notebook_tools(cast(Notebook, None), branch="", ttl_seconds=options.claim_ttl_seconds):
        server.register(
            tool.name,
            description=tool.description,
            schema=tool.input_schema,
            handler=_notebook_handler(
                tool.name, notebooks, options.claim_ttl_seconds, options.learnings, options.briefing_for
            ),
            scopes=_SPECIALIST_SCOPES,
        )


__all__ = [
    "POLL_INBOX_MAX_BYTES",
    "POLL_INBOX_MAX_ITEMS",
    "REPLY_INBOX_HEADER",
    "REPLY_INBOX_LINES",
    "SPECIALIST_TOKEN_ENV",
    "BriefingFor",
    "NotebookFor",
    "McpSpecialistTokens",
    "SpecialistToolOptions",
    "WAIT_POLL_SECONDS",
    "WAIT_TEST_MAX_SECONDS",
    "jsonable",
    "register_specialist_tools",
    "reply_inbox",
    "wait_budget",
]
