# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The notebook as tools for a specialist's model: what the tool layer registers on the specialist's MCP server.

Each :class:`NotebookTool` has the shape of the context tools' specs (name, description, input JSON schema, handler),
as a dataclass. A handler takes the call's arguments and returns compact text; when the call fails, the text starts
with ``error:``, which the tool layer reports as an error result. The tools are bound to the notebook handle of the
specialist the server serves, so the caller is the handle's writer and never an argument.

The tools are a specialist's reads and its own writes: lookup, claim, release, search, record, branch_log,
append_branch_log and brief. Renewing a claim is the runtime's heartbeat, and what only a program writes (requests,
results, judgements, lineage, baselines, quarantine) has no tool.

An intent is named by its target, mechanism and layer, and a change by its recipe delta document, whose fingerprint
the tool computes. Output stays small: a line is at most 160 characters, a list at most 20 lines, a reply at most
4000 characters.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hyperloom.contracts.recipe import RecipeDelta, RecipeError
from hyperloom_multibranch.contracts.notebook.interfaces import (
    AlreadyTested,
    ClaimHeld,
    Notebook,
    NotebookError,
    StartupBriefProvider,
)
from hyperloom_multibranch.contracts.notebook.peers import (
    BranchLogEntry,
    BranchLogType,
    HypothesisStatus,
    Intent,
    IntentLayer,
    Validity,
    placeholder_intent_fingerprint,
)
from hyperloom_multibranch.contracts.notebook.records import (
    BranchLogRecord,
    Claim,
    ClaimKey,
    DedupAnswer,
    StartupBrief,
    TestRecord,
    TriedDirection,
)

TOOL_PREFIX = "notebook_"
LINE_CHARS = 160
MAX_LINES = 20
MAX_REPLY_CHARS = 4000
ERROR = "error:"
UNKNOWN = "unknown"

Arguments = Mapping[str, Any]
Handler = Callable[[Arguments], Awaitable[str]]


@dataclass(frozen=True)
class NotebookTool:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    handler: Handler


def clip(text: str, limit: int = LINE_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def lines(items: Sequence[str]) -> list[str]:
    """``items`` clipped, at most :data:`MAX_LINES` of them, then how many more there were."""

    shown = [clip(item) for item in items[:MAX_LINES]]
    return [*shown, f"... {len(items) - MAX_LINES} more"] if len(items) > MAX_LINES else shown


def outcome(record: TestRecord) -> str:
    """Where one test stands, in a few words."""

    if record.result is None:
        return "in flight"
    if record.result.validity is Validity.CANDIDATE_FAILURE:
        return f"the candidate failed ({record.result.failure_reason or 'no reason given'})"
    judged = record.judgement
    if judged is None:
        return "measured, not judged yet"
    verdict = f"{judged.verdict.value} {judged.gain_pct:+.2f}%"
    if record.conclusion is None:
        return verdict
    made = f" as {record.commit_id}" if record.commit_id else ""
    return f"{verdict}, {record.conclusion.value}{made}"


def render_answer(answer: DedupAnswer) -> str:
    held = f", claimed by {answer.claim.holder}" if answer.claim is not None else ""
    found = [
        f"{answer.status.value} on {answer.key.parent_commit}{held}",
        *(f"- {record.request_id}: {outcome(record)}" for record in answer.records),
        *(
            f"- elsewhere, {record.request_id} on {record.request.parent_commit}: {outcome(record)}"
            for record in answer.elsewhere
        ),
    ]
    return "\n".join(lines(found))


def render_claim(claim: Claim) -> str:
    intent = claim.intent
    return clip(
        f"claimed {intent.target} / {intent.mechanism} on {claim.key.parent_commit} for {claim.holder}, "
        f"expires in {claim.expires_at - claim.heartbeat_at:.0f}s"
    )


def render_record(record: TestRecord) -> str:
    request, judged = record.request, record.judgement
    found = [f"{record.request_id} [{request.kind.value}] on {request.branch} from {request.parent_commit}"]
    found.append(f"outcome: {outcome(record)}")
    if request.purpose:
        found.append(f"purpose: {request.purpose}")
    if judged is not None:
        found.append(f"threshold {judged.threshold_pct:.2f}%, pooled {', '.join(judged.request_ids)}")
    if record.result is not None and record.result.profile_location:
        found.append(f"profile: {record.result.profile_location}")
    return "\n".join(clip(line) for line in found)


def render_log(record: BranchLogRecord) -> str:
    entry = record.entry
    status = f"/{entry.hypothesis_status.value}" if entry.hypothesis_status is not None else ""
    return f"- {record.entry_id} [{entry.kind.value}{status}] {entry.content}"


def _tried(item: TriedDirection) -> str:
    where = f"- {item.target or UNKNOWN} / {item.mechanism or UNKNOWN} on {item.commit_id}"
    state = item.conclusion or item.verdict
    if state is None:
        return f"{where}: not judged"
    gain = f" {item.gain_pct:+.2f}%" if item.gain_pct is not None else ""
    return f"{where}: {state.value}{gain}"


def render_brief(brief: StartupBrief) -> str:
    sections = [f"Startup brief from {brief.start_commit}"]
    for title, items in (
        ("Tried:", [_tried(item) for item in brief.tried]),
        ("Lessons:", [f"- {text}" for text in brief.lessons]),
        ("Leads:", [f"- {text}" for text in brief.leads]),
    ):
        if items:
            sections += [title, *lines(items)]
    return "\n".join(sections)


def _key(arguments: Arguments) -> ClaimKey:
    layer = IntentLayer(arguments["layer"])
    fingerprint = placeholder_intent_fingerprint(str(arguments["target"]), str(arguments["mechanism"]), layer)
    return ClaimKey(str(arguments["parent_commit"]), fingerprint)


def _action(arguments: Arguments) -> str:
    return RecipeDelta.from_document(arguments.get("change") or {}).fingerprint()


@dataclass(frozen=True)
class _SpecialistTools:
    """The handlers, bound to one specialist's notebook handle and branch."""

    notebook: Notebook
    briefing: StartupBriefProvider
    branch: str
    ttl_seconds: float

    async def lookup(self, arguments: Arguments) -> str:
        return render_answer(await self.notebook.lookup(_key(arguments), _action(arguments)))

    async def claim(self, arguments: Arguments) -> str:
        key = _key(arguments)
        target, mechanism = str(arguments["target"]), str(arguments["mechanism"])
        intent = Intent(
            self.branch, key.parent_commit, target, mechanism, IntentLayer(arguments["layer"]), key.intent_fingerprint
        )
        reason = str(arguments.get("reason") or "")
        held = await self.notebook.acquire(intent, _action(arguments), ttl_seconds=self.ttl_seconds, reason=reason)
        return render_claim(held)

    async def release(self, arguments: Arguments) -> str:
        released = await self.notebook.release(_key(arguments))
        return f"released the claim on {released.key.parent_commit}"

    async def search(self, arguments: Arguments) -> str:
        limit = min(int(arguments.get("limit") or MAX_LINES), MAX_LINES)
        hits = await self.notebook.search(str(arguments["query"]), limit=limit)
        return "\n".join(lines([f"- [{hit.kind}] {hit.entry_id} ({hit.branch}): {hit.summary}" for hit in hits])) or (
            "nothing matched"
        )

    async def record(self, arguments: Arguments) -> str:
        return render_record(await self.notebook.record(str(arguments["request_id"])))

    async def branch_log(self, arguments: Arguments) -> str:
        kinds = frozenset(BranchLogType(kind) for kind in arguments.get("kinds") or ())
        found = await self.notebook.branch_log(self.branch, kinds=kinds, after=str(arguments.get("after") or ""))
        return "\n".join(lines([render_log(item) for item in found[-MAX_LINES:]])) or "no entries"

    async def append_branch_log(self, arguments: Arguments) -> str:
        status = arguments.get("hypothesis_status")
        entry = BranchLogEntry(
            self.branch,
            BranchLogType(arguments["kind"]),
            str(arguments["content"]),
            tuple(str(request_id) for request_id in arguments.get("request_ids") or ()),
            str(arguments["supersedes"]) if arguments.get("supersedes") else None,
            HypothesisStatus(status) if status else None,
        )
        return f"logged {(await self.notebook.append_branch_log(entry)).entry_id}"

    async def brief(self, arguments: Arguments) -> str:
        return render_brief(await self.briefing.startup_brief(str(arguments["start_commit"])))


def _guarded(handler: Handler) -> Handler:
    async def guarded(arguments: Arguments) -> str:
        try:
            text = await handler(arguments)
        except (ClaimHeld, AlreadyTested) as refused:
            text = f"{ERROR} refused, {render_answer(refused.answer)}"
        except (NotebookError, RecipeError, ValueError, TypeError, KeyError) as exc:
            text = f"{ERROR} {type(exc).__name__}: {exc}"
        return clip(text, MAX_REPLY_CHARS)

    return guarded


def _schema(properties: Mapping[str, Any], *required: str) -> dict[str, Any]:
    return {"type": "object", "properties": dict(properties), "required": list(required), "additionalProperties": False}


def _text(**more: Any) -> dict[str, Any]:
    return {"type": "string", **more}


_INTENT = {
    "parent_commit": _text(description="The commit the change is tried on."),
    "target": _text(description="What the change attacks: an operator, a kernel, a phase."),
    "mechanism": _text(description="How the change attacks it."),
    "layer": _text(enum=[layer.value for layer in IntentLayer]),
}
_CHANGE = {
    "change": {
        "type": "object",
        "description": "The recipe delta as a document: config, components, add_kernels and remove_kernels.",
    }
}
_KIND = _text(enum=[kind.value for kind in BranchLogType])


@dataclass(frozen=True)
class _Spec:
    name: str
    description: str
    input_schema: Mapping[str, Any]


_SPECS = (
    _Spec(
        "lookup",
        "Whether a change was tried, or is being tried, on a commit: tested, in progress, suspected or free.",
        _schema({**_INTENT, **_CHANGE}, *_INTENT),
    ),
    _Spec(
        "claim",
        "Claim an intent on a commit for this branch before testing it; refused, with the reason, when taken.",
        _schema({**_INTENT, **_CHANGE, "reason": _text()}, *_INTENT),
    ),
    _Spec("release", "Give up a claim this branch holds.", _schema(_INTENT, *_INTENT)),
    _Spec(
        "search",
        "Search the notebook: ',' separates alternatives, '&' joins terms that must all appear.",
        _schema({"query": _text(), "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LINES}}, "query"),
    ),
    _Spec(
        "record",
        "One test's request, outcome and judgement.",
        _schema({"request_id": _text()}, "request_id"),
    ),
    _Spec(
        "branch_log",
        "This branch's long-term memory, oldest first, of the given kinds, after the given entry.",
        _schema({"kinds": {"type": "array", "items": _KIND}, "after": _text()}),
    ),
    _Spec(
        "append_branch_log",
        "Add a hypothesis, attempt, result or plan to this branch's memory; supersedes names the entry it updates.",
        _schema(
            {
                "kind": _KIND,
                "content": _text(),
                "request_ids": {"type": "array", "items": _text()},
                "supersedes": _text(),
                "hypothesis_status": _text(enum=[status.value for status in HypothesisStatus]),
            },
            "kind",
            "content",
        ),
    ),
    _Spec(
        "brief",
        "The startup brief from a commit: what was tried on it and the states it builds on, lessons and leads.",
        _schema({"start_commit": _text()}, "start_commit"),
    ),
)


def notebook_tools(
    notebook: Notebook, briefing: StartupBriefProvider | None = None, *, branch: str, ttl_seconds: float
) -> tuple[NotebookTool, ...]:
    """The tools of the specialist on ``branch``; ``briefing`` gives the brief, the notebook's own by default."""

    bound = _SpecialistTools(notebook, briefing or notebook, branch, ttl_seconds)
    return tuple(
        NotebookTool(TOOL_PREFIX + spec.name, spec.description, spec.input_schema, _guarded(getattr(bound, spec.name)))
        for spec in _SPECS
    )


__all__ = [
    "ERROR",
    "LINE_CHARS",
    "MAX_LINES",
    "MAX_REPLY_CHARS",
    "TOOL_PREFIX",
    "NotebookTool",
    "notebook_tools",
    "outcome",
    "render_answer",
    "render_brief",
    "render_claim",
    "render_log",
    "render_record",
]
