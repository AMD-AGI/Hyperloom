# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The notebook as tools for a specialist's model: what the tool layer registers on the specialist's MCP server.

Each :class:`NotebookTool` has the shape of the context tools' specs (name, description, input JSON schema, handler),
as a dataclass. A handler takes the call's arguments and returns compact text; when the call fails, the text starts
with ``error:``, which the tool layer reports as an error result. The tools are bound to the notebook handle of the
specialist the server serves, so the caller is the handle's writer and never an argument.

The tools are a specialist's reads and its own writes: lookup, claim, release, search, record, branch_log,
append_branch_log, learn, learnings and brief. A granted claim also lists this session's learnings about its
direction. Renewing a claim is the runtime's heartbeat, and what only a program writes (requests, results, judgements,
lineage, baselines, quarantine) has no tool.

An intent is named by its target, mechanism and layer, and a change by its recipe delta document, whose fingerprint
the tool computes. Output stays small: a line is at most 160 characters, a list at most 20 lines, a reply at most
4000 characters.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from hyperloom.contracts.recipe import RecipeDelta, RecipeError
from hyperloom_multibranch.contracts.notebook.interfaces import (
    AlreadyTested,
    CeilingTooLow,
    ClaimHeld,
    Notebook,
    NotebookError,
    StartupBriefProvider,
)
from hyperloom_multibranch.contracts.intent import intent_fingerprint
from hyperloom_multibranch.contracts.notebook.peers import (
    BranchLogEntry,
    BranchLogType,
    HypothesisStatus,
    Intent,
    IntentLayer,
    Validity,
)
from hyperloom_multibranch.contracts.notebook.records import (
    BranchLogRecord,
    Claim,
    ClaimKey,
    DedupAnswer,
    Learning,
    OpShares,
    REPLAY_TARGET,
    StartupBrief,
    TestRecord,
    TriedDirection,
    WarmReplay,
    amdahl_ceiling_pct,
)
from hyperloom_multibranch.contracts.notebook.vocab import DedupStatus, LearningKind
from hyperloom_multibranch.notebook.changes import NO_CHANGE
from hyperloom_multibranch.notebook.learning_search import (
    LearningFinder,
    RetrievalSource,
    SessionLearnings,
    fit_lines,
    render_session_learning,
)
from hyperloom_multibranch.notebook.local.notebook import LEARNING_MAX_CHARS, LEARNING_MIN_CHARS

TOOL_PREFIX = "notebook_"
LINE_CHARS = 160
MAX_LINES = 20
MAX_REPLY_CHARS = 4000
CLAIM_LEARNINGS = 3
CEILING_OPS = 5
ERROR = "error:"
UNKNOWN = "unknown"
#: A test a branch measured with an empty delta measured its parent, so its outcome says nothing about the direction.
UNCHANGED = "measured without a change: its delta was empty"

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

    if record.withdrawn_at is not None:
        return f"withdrawn ({record.withdrawal_reason})"
    if record.result is None:
        return "in flight"
    if record.result.validity is Validity.MEASUREMENT_FAILED:
        return f"not measured ({record.result.failure_reason or 'the measurement failed'})"
    if record.result.validity is Validity.CANDIDATE_FAILED:
        return f"the candidate failed ({record.result.failure_reason or 'no reason given'})"
    judged = record.judgement
    if judged is None:
        return "measured, not judged yet"
    verdict = f"{judged.verdict.value} {judged.gain_pct:+.2f}%"
    if record.conclusion is None:
        return verdict
    made = f" as {record.commit_id}" if record.commit_id else ""
    return f"{verdict}, {record.conclusion.value}{made}"


def _testing_elsewhere(answer: DedupAnswer) -> list[str]:
    tester = answer.claim.holder if answer.claim is not None else answer.records[0].request.branch
    state = answer.claim.key.parent_commit if answer.claim is not None else answer.records[0].request.parent_commit
    return [
        f"{answer.status.value}: {tester} is testing this intent on {state}; its outcome comes to your inbox when that "
        "test ends, and the intent is free again then",
        *(
            f"- {record.request_id} on {record.request.branch} from {record.request.parent_commit}: {outcome(record)}"
            for record in answer.records
        ),
    ]


def render_answer(answer: DedupAnswer) -> str:
    if answer.status is DedupStatus.TESTING_ELSEWHERE:
        found = _testing_elsewhere(answer)
    else:
        held = f", claimed by {answer.claim.holder}" if answer.claim is not None else ""
        found = [
            f"{answer.status.value} on {answer.key.parent_commit}{held}",
            *(f"- {record.request_id}: {outcome(record)}" for record in answer.records),
        ]
    found += [
        f"- elsewhere, {record.request_id} on {record.request.parent_commit}: {outcome(record)}"
        for record in answer.elsewhere
    ]
    return "\n".join(lines(found))


def render_claim(claim: Claim) -> str:
    intent = claim.intent
    held = clip(
        f"claimed {intent.target} / {intent.mechanism} on {claim.key.parent_commit} for {claim.holder}, "
        f"expires in {claim.expires_at - claim.heartbeat_at:.0f}s"
    )
    return f"{held}\nintent fingerprint (submit tests with it): {claim.key.intent_fingerprint}"


def render_ops(shares: OpShares) -> str:
    largest = ", ".join(
        f"{op} {share:.1f}% (ceiling +{amdahl_ceiling_pct(share):.1f}%)" for op, share in shares.largest(CEILING_OPS)
    )
    return clip(f"largest ops on {shares.commit_id} ({shares.job_id}): {largest}")


def render_ceiling(intent: Intent, shares: OpShares | None) -> str:
    """What bounds the claimed intent's gain, and whether a profile of its parent commit backs it."""

    if intent.target_share_pct is None:
        return render_ops(shares) if shares is not None and intent.layer is not IntentLayer.CONFIG else ""
    ceiling = f"Amdahl ceiling +{amdahl_ceiling_pct(intent.target_share_pct):.2f}% for a target worth "
    ceiling += f"{intent.target_share_pct:.2f}% of GPU time"
    if shares is None:
        return f"{ceiling}; unverified, {intent.parent_commit} has no op shares (start_analysis profiles it)"
    return f"{ceiling}\n{render_ops(shares)}"


def render_record(record: TestRecord) -> str:
    request, judged = record.request, record.judgement
    found = [
        f"{record.request_id} [{request.kind.value}, {request.purpose.value}] on {request.branch} "
        f"from {request.parent_commit}",
        f"outcome: {outcome(record)}",
    ]
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
    gain = f" {item.gain_pct:+.2f}%" if item.gain_pct is not None else ""
    outcome = "not judged" if state is None else f"{state.value}{gain}"
    shown = UNCHANGED if item.change == NO_CHANGE else item.change
    change = f" ({shown})" if shown else ""
    return f"{where}: {outcome}{change}"


def _ceilings(shares: OpShares | None) -> list[str]:
    largest = shares.largest(CEILING_OPS) if shares is not None else ()
    return [f"- {op}: {share:.1f}% of GPU time, at most +{amdahl_ceiling_pct(share):.1f}%" for op, share in largest]


def render_brief(brief: StartupBrief) -> str:
    sections = [f"Startup brief from {brief.start_commit}"]
    for title, items in (
        ("Ceilings (largest ops; claim with target_share_pct):", _ceilings(brief.op_shares)),
        ("Tried:", [_tried(item) for item in brief.tried]),
        ("Lessons:", [f"- {text}" for text in brief.lessons]),
        ("Leads:", [f"- {text}" for text in brief.leads]),
    ):
        if items:
            sections += [title, *lines(items)]
    return "\n".join(sections)


@runtime_checkable
class WarmReplays(Protocol):
    """A briefing that replays the champions PRELUDE prepared on its branch's head, as
    ``rsi.briefing.CrossSessionBriefing`` does."""

    async def warm_replay(self, canonical_id: str) -> WarmReplay | None: ...


def render_replay(replay: WarmReplay) -> str:
    """How to test a champion's replay: the intent to claim, then the change to submit on the branch's head."""

    what = "the whole recipe" if not replay.reason else f"its launch config only: {replay.reason}"
    return "\n".join(
        (
            f"replay of {replay.canonical_id} on your head {replay.parent_commit}, {what}",
            f'claim with parent_commit "{replay.parent_commit}", target "{REPLAY_TARGET}", mechanism '
            f'"{replay.canonical_id}" and layer "{replay.layer.value}", then submit this change on that commit:',
            json.dumps(replay.delta.to_document(), sort_keys=True, separators=(",", ":")),
        )
    )


def _key(arguments: Arguments) -> ClaimKey:
    layer = IntentLayer(arguments["layer"])
    fingerprint = intent_fingerprint(str(arguments["target"]), str(arguments["mechanism"]), layer)
    return ClaimKey(str(arguments["parent_commit"]), fingerprint)


def _share(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"target_share_pct is a number, not {value!r}")
    return float(value)


def _action(arguments: Arguments) -> str:
    return RecipeDelta.from_document(arguments.get("change") or {}).fingerprint()


@dataclass(frozen=True)
class _SpecialistTools:
    """The handlers, bound to one specialist's notebook handle and branch."""

    notebook: Notebook
    briefing: StartupBriefProvider
    finder: LearningFinder
    branch: str
    ttl_seconds: float

    async def lookup(self, arguments: Arguments) -> str:
        return render_answer(await self.notebook.lookup(_key(arguments), _action(arguments)))

    async def claim(self, arguments: Arguments) -> str:
        key = _key(arguments)
        target, mechanism = str(arguments["target"]), str(arguments["mechanism"])
        share = arguments.get("target_share_pct")
        layer = IntentLayer(arguments["layer"])
        intent = Intent(self.branch, key.parent_commit, target, mechanism, layer, _share(share))
        reason = str(arguments.get("reason") or "")
        held = await self.notebook.acquire(intent, _action(arguments), ttl_seconds=self.ttl_seconds, reason=reason)
        bounded = render_ceiling(intent, await self.notebook.op_shares(key.parent_commit))
        snapshot = await self.notebook.snapshot()
        related = self.finder.here(
            snapshot, f"{target} {mechanism}", limit=CLAIM_LEARNINGS, branch=self.branch, source=RetrievalSource.CLAIM
        )
        found = [render_claim(held), *([bounded] if bounded else [])]
        if not related:
            return "\n".join(found)
        learnt = [render_session_learning(record, snapshot) for record in related]
        return "\n".join([*found, "This session's learnings about it:", *learnt])

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

    async def learn(self, arguments: Arguments) -> str:
        layer = arguments.get("layer")
        learning = Learning(
            self.branch,
            LearningKind(arguments["kind"]),
            str(arguments["content"]),
            str(arguments.get("target") or ""),
            str(arguments.get("mechanism") or ""),
            IntentLayer(layer) if layer else None,
            tuple(str(request_id) for request_id in arguments.get("request_ids") or ()),
            str(arguments["supersedes"]) if arguments.get("supersedes") else None,
        )
        return f"learnt {(await self.notebook.append_learning(learning)).entry_id}"

    async def learnings(self, arguments: Arguments) -> str:
        limit = min(int(arguments.get("limit") or MAX_LINES), MAX_LINES)
        snapshot = await self.notebook.snapshot()
        found = await self.finder.lines(snapshot, str(arguments["query"]), limit=limit, branch=self.branch)
        return "\n".join(fit_lines(found, MAX_REPLY_CHARS)) or "nothing matched"

    async def brief(self, arguments: Arguments) -> str:
        return render_brief(await self.briefing.startup_brief(str(arguments["start_commit"])))

    async def replay(self, arguments: Arguments) -> str:
        canonical_id = str(arguments["canonical_id"])
        found = await self.briefing.warm_replay(canonical_id) if isinstance(self.briefing, WarmReplays) else None
        if found is None:
            return (
                f"{ERROR} no replay of {canonical_id} on your head: the startup brief's leads name the champions "
                "that have one, and a champion your head already matches has nothing to replay"
            )
        text = render_replay(found)
        return text if len(text) <= MAX_REPLY_CHARS else f"{ERROR} the replay of {canonical_id} does not fit a reply"


def _guarded(handler: Handler) -> Handler:
    async def guarded(arguments: Arguments) -> str:
        try:
            text = await handler(arguments)
        except (ClaimHeld, AlreadyTested) as refused:
            text = f"{ERROR} refused, {render_answer(refused.answer)}"
        except CeilingTooLow as low:
            ops = f"\n{render_ops(low.profiled)}" if low.profiled is not None else ""
            text = f"{ERROR} refused, {low}; claim a larger target or several together{ops}"
        except (NotebookError, RecipeError, ValueError, TypeError, KeyError) as exc:
            text = f"{ERROR} {type(exc).__name__}: {exc}"
        return clip(text, MAX_REPLY_CHARS)

    return guarded


def _schema(properties: Mapping[str, Any], *required: str) -> dict[str, Any]:
    return {"type": "object", "properties": dict(properties), "required": list(required), "additionalProperties": False}


def _text(**more: Any) -> dict[str, Any]:
    return {"type": "string", **more}


def _list(items: Mapping[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": dict(items)}


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
_SHARE = {
    "type": "number",
    "minimum": 0,
    "maximum": 100,
    "description": (
        "For a source or kernel intent: the summed share of GPU time, in percent, of the ops the change speeds up, "
        "read from the parent commit's own profile. A claim whose ceiling cannot reach the gate's bar is refused."
    ),
}


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
        _schema({**_INTENT, **_CHANGE, "reason": _text(), "target_share_pct": _SHARE}, *_INTENT),
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
        _schema({"kinds": _list(_KIND), "after": _text()}),
    ),
    _Spec(
        "append_branch_log",
        "Add a hypothesis, attempt, result or plan to this branch's memory; supersedes names the entry it updates.",
        _schema(
            {
                "kind": _KIND,
                "content": _text(),
                "request_ids": _list(_text()),
                "supersedes": _text(),
                "hypothesis_status": _text(enum=[status.value for status in HypothesisStatus]),
            },
            "kind",
            "content",
        ),
    ),
    _Spec(
        "learn",
        "Write down what this branch learnt, for the next decision of every branch, now and in later sessions: a "
        "pitfall, an insight, how a tool or setting behaves, what a hypothesis came to. Name the direction and the "
        "tests behind it when there are; supersedes names this branch's learning it replaces.",
        _schema(
            {
                "kind": _text(enum=[kind.value for kind in LearningKind]),
                "content": _text(minLength=LEARNING_MIN_CHARS, maxLength=LEARNING_MAX_CHARS),
                "target": _text(),
                "mechanism": _text(),
                "layer": _text(enum=[layer.value for layer in IntentLayer]),
                "request_ids": _list(_text()),
                "supersedes": _text(),
            },
            "kind",
            "content",
        ),
    ),
    _Spec(
        "learnings",
        "What specialists learnt that bears on a query: this session's learnings of every branch, then past sessions' "
        "from the Experience KB, each saying whether it was learnt under this session's conditions and tested.",
        _schema({"query": _text(), "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LINES}}, "query"),
    ),
    _Spec(
        "brief",
        "The startup brief from a commit: what was tried on it and the states it builds on, lessons and leads.",
        _schema({"start_commit": _text()}, "start_commit"),
    ),
    _Spec(
        "replay",
        "A Recipe KB champion the startup brief names, replayed on your branch's head: the intent to claim and the "
        "change to submit there, its patch and kernel files already in the session's blob store.",
        _schema(
            {"canonical_id": _text(description="The champion's Recipe KB id, as the brief names it.")}, "canonical_id"
        ),
    ),
)


def notebook_tools(
    notebook: Notebook,
    briefing: StartupBriefProvider | None = None,
    *,
    branch: str,
    ttl_seconds: float,
    learnings: LearningFinder | None = None,
) -> tuple[NotebookTool, ...]:
    """The tools of the specialist on ``branch``; ``briefing`` gives the brief, the notebook's own by default.

    ``learnings`` finds the learnings for the learnings tool: the session's one ``LearningRecall`` adds past sessions'
    from the Experience KB; without it the tool finds this session's only.
    """

    bound = _SpecialistTools(notebook, briefing or notebook, learnings or SessionLearnings(), branch, ttl_seconds)
    return tuple(
        NotebookTool(TOOL_PREFIX + spec.name, spec.description, spec.input_schema, _guarded(getattr(bound, spec.name)))
        for spec in _SPECS
    )


__all__ = [
    "CEILING_OPS",
    "ERROR",
    "LINE_CHARS",
    "MAX_LINES",
    "MAX_REPLY_CHARS",
    "TOOL_PREFIX",
    "NotebookTool",
    "WarmReplays",
    "notebook_tools",
    "outcome",
    "render_answer",
    "render_brief",
    "render_ceiling",
    "render_claim",
    "render_log",
    "render_record",
    "render_replay",
]
