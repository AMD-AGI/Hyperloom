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
* ``start_analysis`` starts one of the analysis tools the toolbox serves (``tracelens``, ``roofline``, ``kernel_tune``:
  its :attr:`~.analysis.AnalysisStarter.analysis_tools`, which the tool's schema and description list, no more) on the
  branch's head commit, or on the landed output of the test ``kernel_tune`` names (an
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

from hyperloom.common.outside_facts import (
    TUNABLEOP_ENABLED_ENV,
    TUNABLEOP_TABLE_ENV,
    TUNABLEOP_TUNING_ENV,
    UPSTREAM_PR_REPOS,
)
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
from hyperloom_multibranch.toolbox.analysis import CODE_NO_EXECUTOR, AnalysisStarter
from hyperloom_multibranch.toolbox.tools.kernel_tune import KERNEL_TUNE_TOOL
from hyperloom_multibranch.toolbox.tools.roofline import ROOFLINE_TOOL, TRACELENS_TOOL
from hyperloom_multibranch.toolbox.tools.upstream_pr import DEFAULT_LIMIT, MAX_LIMIT, UPSTREAM_PR_TOOL

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
_ANALYSIS_BLURBS = {
    TRACELENS_TOOL: "analyse a trace",
    ROOFLINE_TOOL: "the kernel roofline of a trace",
    KERNEL_TUNE_TOOL: "tune GEMM shapes on one card",
    UPSTREAM_PR_TOOL: "search upstream pull requests",
}
_TEST_RESULT_GUIDE = (
    "The state and result of one test this branch submitted, or of any branch's profile test. An AgentX test's "
    "request_classes has an entry for each measured round: how many requests the client marked failed, as "
    "failed_transport (the server did not serve it) and failed_no_content (it answered and the client found nothing in "
    "the answer; failed_empty and failed_short split these by output tokens when the export says, else null with the "
    "reason), and export, the round's per-request records (aiperf's profile_export.jsonl: input length, output length "
    "and TTFT of a served request, the error of a failed one) as a path relative to the result's output_dir. A null "
    "count was not computed; it is not zero. phases says where the test's wall time went, in the order it happened "
    "(the sandbox starting, the environment rebuilt, the server booting, each round, the result signed and landed): "
    "each phase's name, seconds, gpu_seconds and, for a round's phase, the measurement point it ran for; "
    "server_boot.weights_load and its siblings are parts of server_boot. unaccounted_seconds is the part of the wall "
    "time no phase covers, so the phases and it add up to it; both are empty or null on a result recorded without them. "
    "short_prompt_consistency is null unless the test ran the short-prompt check (a change touching attention, the KV "
    "cache or prefill, a baseline, a final candidate): short prompts answered greedily by several fresh server processes "
    "of the candidate's launch. self_consistency counts, for each process, the prompts it answers differently from the "
    "candidate's other processes; reference.counts does the same against the parent's own outputs, and reference.status "
    "absent means the parent has none, not that nothing differs. first_difference gives the process, prompt and token "
    "position of the first difference: a difference in the first tokens is corruption, one late in the output may be "
    "rounding. The counts are findings, not a verdict. long_context is null unless the test ran the long-context check: "
    "long synthetic prompts (needle retrieval, multi-key retrieval, variable tracking, common-word extraction) at the "
    "stated length, answered greedily by the candidate's own server and graded by substring match. score is the share "
    "of all n prompts that passed (tasks and prompts give the per-task and per-prompt outcomes); status incomplete, "
    "with a reason, means a prompt errored or never ran (for example a launch that serves too little context) and is "
    "never a pass. The scores are findings, not a verdict."
)
_KERNEL_TUNE_PARAMS_NOTE = (
    " For kernel_tune, params.test_request_id is required: a call without it is refused (test_required)."
)
_KERNEL_TUNE_GUIDE = (
    " kernel_tune: params.test_request_id is required, with or without shapes: it names a test whose landed output "
    "the tool reads, and the test says which model, framework and precision the tuning is for. Its server log reports "
    "untuned GEMM shapes (aiter's 'not found tuned config' lines), which are what is tuned. params.shapes, a list of "
    "{env, M, N, K, ...} (env is an AITER_CONFIG_GEMM_* variable), is optional: it replaces the shapes of that log with "
    "shapes of yours, for that test's model, framework and precision; it does not stand in for test_request_id. "
    "params.timeout_sec bounds the tuning. params.retune=true tunes the shapes the tables already serve (the log's "
    "hit lines, which aiter logs only under AITER_LOG_TUNED_CONFIG) when the log reports no miss; it does not go "
    "with params.shapes, which are tuned as named. The decode "
    "CUDA-graph batch sizes the test's servers captured are added to the shapes as M. It runs Forge's tuner on one "
    "card leased for the job, so it is refused (lease_leaves_no_measurement) when no card can be spared from "
    "measurement, and its result arrives in poll_inbox: per table, the tuned rows as a rows-only CSV for "
    "put_kernel_artifact(rows_only=true), each row checked against the reference GEMM (a row that failed or could "
    "not be checked is listed with verified=false and is not in the CSV). By default a row is handed over only with "
    "a mismatch ratio of at most 0.01 (an asm kernel at a splitK above 1 may instead have a relative-norm error of at "
    "most 0.01, which its verified_rows entry shows as check=rel_norm_err). params.max_err_ratio (up to 0.05) also "
    "returns the rows above 0.01 and within "
    "it as rows_csv_relaxed, listed in relaxed_rows with their err_ratio and verified=false (reason "
    "err_ratio_above_default): they are not checked rows and are never in rows_csv; using them is your explicit "
    "choice, and only an end-to-end accuracy check (GSM8K, and the long-context check for attention or KV paths) "
    "decides whether they stand. Its microseconds are advice only; a test of the rows says whether they help."
)


_UPSTREAM_PR_GUIDE = (
    " upstream_pr_search: reads no commit and no profile test (leave profile_request_id out); params.repo_url is "
    f"required, the GitHub URL of one of {', '.join(UPSTREAM_PR_REPOS)}; params.keywords (words to look for), "
    f"params.limit (default {DEFAULT_LIMIT}, at most {MAX_LIMIT}) and params.states (open by default; merged, closed or all "
    "search pull requests of every state) are optional. The search runs in the toolbox's host process, and its result arrives in poll_inbox: the "
    "candidate pull requests with ref, title, url and score. A host that cannot reach GitHub fails the job and says why; "
    "that is not an empty answer."
)


def _start_analysis_schema(served: Sequence[str]) -> dict[str, Any]:
    """The ``start_analysis`` schema: ``tool`` is one of ``served``, the tools the toolbox serves, and nothing else.

    The commit is the branch head, read by the server; a request that names a commit or a branch is refused by the
    schema. With nothing served ``tool`` has no ``enum``, so a call reaches the toolbox's own refusal by name.
    """
    needing = " and ".join(tool for tool in (TRACELENS_TOOL, ROOFLINE_TOOL) if tool in served)
    profile_note = f"required for {needing}, checked by the server" if needing else "checked by the server"
    tool: dict[str, Any] = {"type": _STRING, "description": f"One of: {', '.join(served) or 'none served here'}."}
    if served:
        tool["enum"] = list(served)
    params_note = _KERNEL_TUNE_PARAMS_NOTE if KERNEL_TUNE_TOOL in served else ""
    return {
        "type": _OBJECT,
        "properties": {
            "tool": tool,
            "params": {
                "type": _OBJECT,
                "description": "The tool's own parameters. trace_input, commit_id, profile_request_id and landed_dir are set by the server and refused here."
                + params_note,
            },
            "profile_request_id": {
                "type": _STRING,
                "description": f"The id of a profile test of this branch's head commit, run by any branch, whose trace is analysed; {profile_note}.",
            },
        },
        "required": ["tool"],
        "additionalProperties": False,
    }


def _start_analysis_description(served: Sequence[str]) -> str:
    if not served:
        return "This toolbox serves no analysis tool: every call is refused by name."
    named = " or ".join(f"{name} ({_ANALYSIS_BLURBS[name]})" if name in _ANALYSIS_BLURBS else name for name in served)
    return (
        f"Start {named} for this branch; returns the job id at once and the result arrives in poll_inbox. The commit "
        "analysed is this branch's head, set by the server; with a profile_request_id of any branch's profile test of "
        "it the pointer carries its op shares."
        + (_KERNEL_TUNE_GUIDE if KERNEL_TUNE_TOOL in served else "")
        + (_UPSTREAM_PR_GUIDE if UPSTREAM_PR_TOOL in served else "")
    )


_PUT_KERNEL_ARTIFACT: dict[str, Any] = {
    "type": _OBJECT,
    "properties": {
        "kind": {"enum": sorted(kind.value for kind in LOADABLE_KERNEL_KINDS)},
        "target": {
            "type": _STRING,
            "minLength": 1,
            "description": (
                "The operator or shape the table serves, one path segment (letters, digits and ._=,+-); the merge "
                "slot of the artifact: two branches that put a table of one target compete for it. Optional when env "
                "is bound already in this session: leave it out to put under the bound target, which the returned "
                "artifact names. Required for an env no table has been put for yet (refused as target_required)."
            ),
        },
        "env": {
            "type": _STRING,
            "minLength": 1,
            "description": (
                "The environment variable the service reads the table's path from: an aiter table, e.g. "
                f"AITER_CONFIG_GEMM_A8W8, or a TunableOp results file, {TUNABLEOP_TABLE_ENV} (the delta must also "
                f"set {TUNABLEOP_ENABLED_ENV}=1 and {TUNABLEOP_TUNING_ENV}=0)."
            ),
        },
        "content": {
            "type": _STRING,
            "minLength": 1,
            "description": "The table as UTF-8 text, a CSV: the whole table, or with rows_only just your tuned rows.",
        },
        "rows_only": {
            "type": _BOOLEAN,
            "description": (
                "Only for an aiter table (AITER_CONFIG_*). content is just the rows you tuned, under the header of "
                "the image's table. The test loads the image's default table for the env and then your rows, and a "
                "shape in both is read from your rows, so you need not carry the default's rows. Give each shape "
                "once and keep the columns that identify it (M, N, K and the rest of the key, cu_num; gfx may be "
                "left out), else the test is refused before its server starts."
            ),
        },
    },
    "required": ["kind", "env", "content"],
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
        kind, rows_only = KernelKind(arguments["kind"]), arguments.get("rows_only", False)
        call = toolbox.put_kernel_artifact(kind, arguments.get("target"), arguments["env"], data, rows_only)
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
            "under that target, and may leave target out to take it: the returned artifact names the bound target. "
            "Only a table of another target name for the same env is refused, as kernel_variable_taken, and its "
            "message names the bound target; an env no table was put for yet needs a target (target_required). The "
            "same table put twice, by any branch, is the same artifact. With rows_only the table is just the rows you tuned: the test reads the image's "
            "default table for the env and then your rows, a shape in both from your rows, so you need not upload "
            "the default's rows."
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
    server.attach_reply_annotator(reply_inbox(notebooks))
    _register_submit_test(server, toolbox, toolbox_loop)
    _register_put_kernel_artifact(server, toolbox, toolbox_loop)
    _register_test_queries(server, toolbox, notebooks, toolbox_loop, options)
    _register_start_analysis(server, options.analysis, toolbox_loop)
    _register_poll_inbox(server, notebooks)
    _register_notebook_tools(server, notebooks, options)


def _register_submit_test(server: McpServer, toolbox: Toolbox, home: asyncio.AbstractEventLoop | None) -> None:
    server.register(
        "submit_test",
        description=(
            "Submit a test of a recipe change for this branch; returns the request id once accepted. A profile of a "
            "commit that already has one queued, running or measured is refused as profile_exists and names that "
            "request: wait_test on it, or read the commit's op shares from the notebook."
        ),
        schema=_SUBMIT_TEST,
        handler=_submit_test(toolbox, home),
        scopes=_SPECIALIST_SCOPES,
    )


def _register_test_queries(
    server: McpServer,
    toolbox: Toolbox,
    notebooks: NotebookFor,
    home: asyncio.AbstractEventLoop | None,
    options: SpecialistToolOptions,
) -> None:
    """Register the tools that read, wait for, cancel or size up the tests: ``test_result`` to ``queue_state``."""
    server.register(
        "test_result",
        description=_TEST_RESULT_GUIDE,
        schema=_REQUEST_ID,
        handler=_test_result(notebooks),
        scopes=_SPECIALIST_SCOPES,
    )
    server.register(
        "wait_test",
        description=(
            "How to wait for a test of this branch, or for any branch's profile test: returns as soon as its result "
            "lands, else after wait_seconds with the test's progress; call it again to keep waiting."
        ),
        schema=_WAIT_TEST,
        handler=_wait_test(toolbox, notebooks, home, options.wait_poll_seconds),
        scopes=_SPECIALIST_SCOPES,
    )
    server.register(
        "cancel_test",
        description="Take back a test of this branch that has not finished.",
        schema=_REQUEST_ID,
        handler=_cancel_test(toolbox, notebooks, home),
        scopes=_SPECIALIST_SCOPES,
    )
    server.register(
        "queue_state",
        description="Queue lengths by priority, running tests by branch, and the sandbox cold start time.",
        schema=_NO_ARGUMENTS,
        handler=_queue_state(toolbox, home),
        scopes=_SPECIALIST_SCOPES,
    )


def _register_start_analysis(
    server: McpServer, analysis: AnalysisStarter | None, home: asyncio.AbstractEventLoop | None
) -> None:
    served = analysis.analysis_tools if analysis is not None else ()
    server.register(
        "start_analysis",
        description=_start_analysis_description(served),
        schema=_start_analysis_schema(served),
        handler=_start_analysis(analysis, home),
        scopes=_SPECIALIST_SCOPES,
    )


def _register_poll_inbox(server: McpServer, notebooks: NotebookFor) -> None:
    server.register(
        _POLL_INBOX_TOOL,
        description=(
            "Pointers pushed to this branch since the cursor, such as results of asynchronous tools. A reply holds "
            f"at most {POLL_INBOX_MAX_ITEMS} items; when remaining is above 0, call again with the cursor it returned "
            "to read the next page."
        ),
        schema=_POLL_INBOX,
        handler=_poll_inbox(notebooks),
        scopes=_SPECIALIST_SCOPES,
    )


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
