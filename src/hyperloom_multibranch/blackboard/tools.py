# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The blackboard as tools for a specialist's model: what the tool layer registers on the specialist's MCP server.

Each :class:`BlackboardTool` has the shape of the context tools' specs (name, description, input JSON schema, handler),
as a dataclass. A handler takes the call's arguments and returns compact text; when the call fails, the text starts
with ``error:``, which the tool layer reports as an error result. The tools are bound to the blackboard handle of the
specialist the server serves, so the caller is the handle's writer and never an argument.

The tools are a specialist's reads and its own writes: lookup, claim, release, search, record, branch_log,
append_branch_log, learn, learnings and brief. A granted claim also says what the briefing knows of its direction's
family, when it is a :class:`DirectionNotes`, and lists what the finder knows about the direction: this session's
learnings, and past sessions' lessons when it has them. A learning may correct or retire a past
session's lesson the finder knows. Renewing a claim is the runtime's heartbeat, and what only a program writes
(requests, results, judgements, lineage, baselines, quarantine) has no tool.

An intent is named by its target, mechanism and layer, and a change by its recipe delta document, whose fingerprint
the tool computes. Output stays small: a line is at most 160 characters, a list at most 20 lines, a reply at most
4000 characters.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Protocol, runtime_checkable

from hyperloom.contracts.recipe import RecipeDelta, RecipeError
from hyperloom_multibranch.contracts.blackboard.interfaces import (
    AlreadyTested,
    CeilingTooLow,
    ClaimHeld,
    DirectionNotes,
    Blackboard,
    BlackboardError,
    StartupBriefProvider,
)
from hyperloom_multibranch.contracts.intent import intent_fingerprint
from hyperloom_multibranch.contracts.blackboard.peers import (
    BranchLogEntry,
    BranchLogType,
    HypothesisStatus,
    Intent,
    IntentLayer,
    Validity,
)
from hyperloom_multibranch.contracts.blackboard.records import (
    BranchLogRecord,
    Claim,
    ClaimKey,
    DedupAnswer,
    Learning,
    OpShares,
    REPLAY_TARGET,
    REPLAY_TOOL,
    SHORT_COMMIT,
    StartupBrief,
    TestRecord,
    PeerReplay,
    TriedDirection,
    UnfinishedLead,
    WarmReplay,
    amdahl_ceiling_pct,
    is_lesson_id,
)
from hyperloom_multibranch.contracts.blackboard.vocab import DedupStatus, LearningKind, UnfinishedReason
from hyperloom_multibranch.blackboard.changes import NO_CHANGE
from hyperloom_multibranch.blackboard.families import family_of
from hyperloom_multibranch.blackboard.learning_search import LearningFinder, SessionLearnings, fit_lines
from hyperloom_multibranch.blackboard.local.blackboard import LEARNING_MAX_CHARS, LEARNING_MIN_CHARS

TOOL_PREFIX = "blackboard_"
LINE_CHARS = 160
MAX_LINES = 20
MAX_REPLY_CHARS = 4000
#: How much of a left-out patch one reply shows, leaving room in it for the lines around the text.
PATCH_PAGE_CHARS = MAX_REPLY_CHARS - 300
#: The brief's ceilings name every op of at least :data:`CEILING_MIN_SHARE_PCT` of GPU time, the largest first, up to
#: this many, so that a direction on small ops can name its targets and claim several together.
CEILING_OPS = 12
CEILING_MIN_SHARE_PCT = 1.0
#: The largest ops a one-line reply names.
LINE_OPS = 5
ERROR = "error:"
UNKNOWN = "unknown"
#: A test a branch measured with an empty delta measured its parent, so its outcome says nothing about the direction.
UNCHANGED = "measured without a change: its delta was empty"
UNFINISHED_TITLE = (
    f"Unfinished in this session (finish one: {REPLAY_TOOL} with its request_id makes its change for your head):"
)
FOCUS_LINE = "Focus, the bottleneck the scheduler opened this branch on: "
SETTINGS_TITLE = (
    "Settings this session has not tried (a draw of each family's commit odds x the gain within reach / the cost "
    "of a test):"
)
CODE_TITLE = (
    "Source and kernel changes (a draw of each family's commit odds x the ceiling of what it acts on / the cost of "
    "a test):"
)
_UNFINISHED = {
    UnfinishedReason.NOT_STACKABLE: "committed, merge not stackable",
    UnfinishedReason.NEAR_BAR: "kept accuracy, just short of its bar",
    UnfinishedReason.REPAIR_UNFINISHED: "blocked, its repair never finished",
    UnfinishedReason.WITHDRAWN: "withdrawn before its result",
}

Arguments = Mapping[str, Any]
Handler = Callable[[Arguments], Awaitable[str]]


@dataclass(frozen=True)
class BlackboardTool:
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
    if state == answer.key.parent_commit:
        what, then = "the same change", ""
    else:
        what, then = "this intent", ", and the intent is free again then"
    return [
        f"{answer.status.value}: {tester} is testing {what} on {state}; its outcome comes to your inbox when that test "
        f"ends{then}",
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
        f"{op} {share:.1f}% (ceiling +{amdahl_ceiling_pct(share):.1f}%)" for op, share in shares.largest(LINE_OPS)
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
    return [
        f"- {op}: {share:.1f}% of GPU time, at most +{amdahl_ceiling_pct(share):.1f}%"
        for op, share in largest
        if share >= CEILING_MIN_SHARE_PCT
    ]


def _unfinished(item: UnfinishedLead) -> str:
    """One lead, why it is unfinished before what it changes, so that clipping its line cuts the change, never the
    verdict."""

    gain = f" {item.gain_pct:+.2f}%" if item.gain_pct is not None else ""
    detail = f" ({item.detail})" if item.detail else ""
    shown = UNCHANGED if item.change == NO_CHANGE else item.change
    change = f" ({shown})" if shown else ""
    return (
        f"- {item.request_id} on {item.branch}, {item.measured_on[:SHORT_COMMIT]}{gain}: {_UNFINISHED[item.reason]}"
        f"{detail}; {item.target or UNKNOWN} / {item.mechanism or UNKNOWN}{change}"
    )


def _menu(menu: tuple[str, ...]) -> list[str]:
    """A list of the menu, a family's note indented under its line."""

    return [text if text.startswith(" ") else f"- {text}" for text in menu]


def render_brief(brief: StartupBrief) -> str:
    sections = [f"Startup brief from {brief.start_commit}"]
    if brief.focus:
        sections.append(f"{FOCUS_LINE}{brief.focus}")
    for title, items in (
        ("Ceilings (largest ops; claim with target_share_pct):", _ceilings(brief.op_shares)),
        ("Tried:", [_tried(item) for item in brief.tried]),
        (UNFINISHED_TITLE, [_unfinished(item) for item in brief.unfinished]),
        (SETTINGS_TITLE, _menu(brief.settings_menu)),
        (CODE_TITLE, _menu(brief.code_menu)),
        ("Lessons:", [f"- {text}" for text in brief.lessons]),
        ("Leads:", [f"- {text}" for text in brief.leads]),
    ):
        if items:
            sections += [title, *lines(items)]
    return "\n".join(sections)


@runtime_checkable
class WarmReplays(Protocol):
    """A briefing that replays the champions PRELUDE prepared on its branch's head, and reads the patches a replay
    leaves out, as ``rsi.briefing.CrossSessionBriefing`` does."""

    async def warm_replay(self, canonical_id: str) -> WarmReplay | None: ...

    async def warm_reference(self, canonical_id: str, ref: str) -> str | None: ...


@runtime_checkable
class PeerReplays(Protocol):
    """A briefing that replays another branch's tested change on its branch's head, as
    ``rsi.briefing.CrossSessionBriefing`` does."""

    async def peer_replay(self, request_id: str) -> PeerReplay: ...


def _claim_and_change(source: str, parent_commit: str, layer: IntentLayer, delta: RecipeDelta) -> tuple[str, str]:
    return (
        f'claim with parent_commit "{parent_commit}", target "{REPLAY_TARGET}", mechanism "{source}" and layer '
        f'"{layer.value}", then submit this change on that commit:',
        json.dumps(delta.to_document(), sort_keys=True, separators=(",", ":")),
    )


def render_replay(replay: WarmReplay) -> str:
    """How to test a champion's replay: where its gain was measured when that was another image, the files it leaves
    out, then the intent to claim and the change to submit on the branch's head."""

    what = "the whole recipe" if not replay.reason else f"its launch config only: {replay.reason}"
    lines = [f"replay of {replay.canonical_id} on your head {replay.parent_commit}, {what}"]
    if replay.source_image:
        lines.append(f"its gain was measured on image {replay.source_image}, not this session's; expect it to differ")
    lines.extend(f"left out: {reference}" for reference in replay.references)
    found = _claim_and_change(replay.canonical_id, replay.parent_commit, replay.layer, replay.delta)
    return "\n".join((*lines, *found))


def render_reference(canonical_id: str, ref: str, text: str, offset: int = 0) -> str:
    """The patch ``ref`` of the champion ``canonical_id``'s replay leaves out, from ``offset`` on, as much of ``text``
    as one reply holds, and where to ask for the rest."""

    start = max(offset, 0)
    page = text[start : start + PATCH_PAGE_CHARS]
    end = start + len(page)
    lines = [
        f"patch {ref} that the replay of {canonical_id} leaves out, characters {start} to {end} of {len(text)}; port "
        "what it changes to your head, it did not apply to this session's image:",
        page,
    ]
    if end < len(text):
        lines.append(f"... {len(text) - end} more characters: ask again with offset {end}")
    return "\n".join(lines)


def render_peer_replay(replay: PeerReplay) -> str:
    """How to test another branch's change on this branch's head: each kernel artifact of the head's it replaces, then
    the intent to claim and the change to submit."""

    found = _claim_and_change(replay.request_id, replay.parent_commit, replay.layer, replay.delta)
    first = f"replay of {replay.request_id}, the change {replay.branch} tested, on your head {replay.parent_commit}"
    replaced = (
        f"it replaces your head's {item.kind.value} kernel {PurePosixPath(item.ref).name} for {item.target}: a target "
        f"loads one of a kind, so this tests {replay.branch}'s in place of yours, not on top of it"
        for item in replay.replaces
    )
    return "\n".join((first, *replaced, *found))


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
    """The handlers, bound to one specialist's blackboard handle and branch."""

    blackboard: Blackboard
    briefing: StartupBriefProvider
    finder: LearningFinder
    branch: str
    ttl_seconds: float

    async def lookup(self, arguments: Arguments) -> str:
        return render_answer(await self.blackboard.lookup(_key(arguments), _action(arguments)))

    async def claim(self, arguments: Arguments) -> str:
        key = _key(arguments)
        target, mechanism = str(arguments["target"]), str(arguments["mechanism"])
        share = arguments.get("target_share_pct")
        layer = IntentLayer(arguments["layer"])
        intent = Intent(self.branch, key.parent_commit, target, mechanism, layer, _share(share))
        reason = str(arguments.get("reason") or "")
        held = await self.blackboard.acquire(intent, _action(arguments), ttl_seconds=self.ttl_seconds, reason=reason)
        bounded = render_ceiling(intent, await self.blackboard.op_shares(key.parent_commit))
        snapshot = await self.blackboard.snapshot()
        change = RecipeDelta.from_document(arguments.get("change") or {})
        note = await self.briefing.direction_note(intent, change) if isinstance(self.briefing, DirectionNotes) else ""
        family = family_of(target, mechanism, layer, change)
        learnt = await self.finder.about(snapshot, f"{target} {mechanism}", branch=self.branch, family=family)
        return "\n".join([render_claim(held), *([bounded] if bounded else []), *([note] if note else []), *learnt])

    async def release(self, arguments: Arguments) -> str:
        released = await self.blackboard.release(_key(arguments))
        return f"released the claim on {released.key.parent_commit}"

    async def search(self, arguments: Arguments) -> str:
        limit = min(int(arguments.get("limit") or MAX_LINES), MAX_LINES)
        hits = await self.blackboard.search(str(arguments["query"]), limit=limit)
        return "\n".join(lines([f"- [{hit.kind}] {hit.entry_id} ({hit.branch}): {hit.summary}" for hit in hits])) or (
            "nothing matched"
        )

    async def record(self, arguments: Arguments) -> str:
        return render_record(await self.blackboard.record(str(arguments["request_id"])))

    async def branch_log(self, arguments: Arguments) -> str:
        kinds = frozenset(BranchLogType(kind) for kind in arguments.get("kinds") or ())
        found = await self.blackboard.branch_log(self.branch, kinds=kinds, after=str(arguments.get("after") or ""))
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
        return f"logged {(await self.blackboard.append_branch_log(entry)).entry_id}"

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
            str(arguments["retires"]) if arguments.get("retires") else None,
        )
        lesson = learning.retires or (learning.supersedes if is_lesson_id(learning.supersedes) else None)
        if lesson is not None and not self.finder.knows(lesson):
            return f"{ERROR} {lesson} is no live lesson of a past session; use an id the brief or blackboard_learnings shows"
        written = (await self.blackboard.append_learning(learning)).entry_id
        if lesson is None:
            return f"learnt {written}"
        done = "retires" if learning.retires else "revises"
        return f"learnt {written}; it {done} {lesson} for the sessions after this one, and this session sees it no more"

    async def learnings(self, arguments: Arguments) -> str:
        limit = min(int(arguments.get("limit") or MAX_LINES), MAX_LINES)
        snapshot = await self.blackboard.snapshot()
        found = await self.finder.lines(snapshot, str(arguments["query"]), limit=limit, branch=self.branch)
        return "\n".join(fit_lines(found, MAX_REPLY_CHARS)) or "nothing matched"

    async def brief(self, arguments: Arguments) -> str:
        return render_brief(await self.briefing.startup_brief(str(arguments["start_commit"])))

    async def replay(self, arguments: Arguments) -> str:
        canonical_id, request_id = str(arguments.get("canonical_id") or ""), str(arguments.get("request_id") or "")
        if bool(canonical_id) == bool(request_id):
            return f"{ERROR} name a champion's canonical_id or another branch's test's request_id, one of the two"
        if request_id:
            if not isinstance(self.briefing, PeerReplays):
                return f"{ERROR} no replay of {request_id}: this specialist's brief reads no experiment repository"
            text = render_peer_replay(await self.briefing.peer_replay(request_id))
        elif arguments.get("reference"):
            return await self._reference(canonical_id, str(arguments["reference"]), int(arguments.get("offset") or 0))
        else:
            found = await self.briefing.warm_replay(canonical_id) if isinstance(self.briefing, WarmReplays) else None
            if found is None:
                return (
                    f"{ERROR} no replay of {canonical_id} on your head: the startup brief's leads name the champions "
                    "that have one, and a champion your head already matches has nothing to replay"
                )
            text = render_replay(found)
        source = request_id or canonical_id
        return text if len(text) <= MAX_REPLY_CHARS else f"{ERROR} the replay of {source} does not fit a reply"

    async def _reference(self, canonical_id: str, ref: str, offset: int) -> str:
        warm = self.briefing if isinstance(self.briefing, WarmReplays) else None
        text = await warm.warm_reference(canonical_id, ref) if warm is not None else None
        if text is None:
            return f"{ERROR} the replay of {canonical_id} leaves out no patch {ref}: name one of its 'left out: patch' lines"
        return render_reference(canonical_id, ref, text, offset)


def _guarded(handler: Handler) -> Handler:
    async def guarded(arguments: Arguments) -> str:
        try:
            text = await handler(arguments)
        except (ClaimHeld, AlreadyTested) as refused:
            text = f"{ERROR} refused, {render_answer(refused.answer)}"
        except CeilingTooLow as low:
            ops = f"\n{render_ops(low.profiled)}" if low.profiled is not None else ""
            text = f"{ERROR} refused, {low}; claim a larger target or several together{ops}"
        except (BlackboardError, RecipeError, ValueError, TypeError, KeyError) as exc:
            text = f"{ERROR} {type(exc).__name__}: {exc}"
        return clip(text, MAX_REPLY_CHARS)

    return guarded


def _schema(properties: Mapping[str, Any], *required: str) -> dict[str, Any]:
    return {"type": "object", "properties": dict(properties), "required": list(required), "additionalProperties": False}


def _text(**more: Any) -> dict[str, Any]:
    return {"type": "string", **more}


def _integer(**more: Any) -> dict[str, Any]:
    return {"type": "integer", **more}


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
        "Search the blackboard: ',' separates alternatives, '&' joins terms that must all appear.",
        _schema({"query": _text(), "limit": _integer(minimum=1, maximum=MAX_LINES)}, "query"),
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
        "Write down what this branch learnt that the tests' counts cannot say, for the next decision of every branch, "
        "now and in later sessions: how to do a change and why it works or fails, under which conditions; how a tool, "
        "a setting or the environment behaves; where the workload spends its time and what a hypothesis came to. A "
        "test's outcome alone is counted already. Name the direction and cite the tests behind it. "
        "supersedes names this branch's learning it replaces, or a past session's lesson "
        "(lsn-...) it corrects; retires names a past session's lesson this session's tests show wrong or outdated "
        "here, content saying why. Correcting or retiring a past lesson cites those tests in request_ids: one with no "
        "test of this session there, or naming a lesson that is not live, is refused.",
        _schema(
            {
                "kind": _text(enum=[kind.value for kind in LearningKind]),
                "content": _text(minLength=LEARNING_MIN_CHARS, maxLength=LEARNING_MAX_CHARS),
                "target": _text(),
                "mechanism": _text(),
                "layer": _text(enum=[layer.value for layer in IntentLayer]),
                "request_ids": _list(_text()),
                "supersedes": _text(),
                "retires": _text(),
            },
            "kind",
            "content",
        ),
    ),
    _Spec(
        "learnings",
        "What specialists learnt that bears on a query: this session's learnings of every branch, then past sessions' "
        "lessons, each saying whether it was learnt under this session's conditions and tested, with the id a "
        "learning of yours may correct or retire it by.",
        _schema({"query": _text(), "limit": _integer(minimum=1, maximum=MAX_LINES)}, "query"),
    ),
    _Spec(
        "brief",
        "The startup brief from a commit: what was tried on it and the states it builds on, lessons and leads.",
        _schema({"start_commit": _text()}, "start_commit"),
    ),
    _Spec(
        "replay",
        "A Recipe KB champion the startup brief names, or the change of another branch's test a peer_outcome inbox "
        "item or the brief's unfinished leads name, replayed on your branch's head: the intent to claim and the change "
        "to submit there. Give one of the two; a champion's patch and kernel files are already in the session's blob "
        "store. With a champion's canonical_id and reference, the text of one patch its replay leaves out, to port.",
        _schema(
            {
                "canonical_id": _text(description="The champion's Recipe KB id, as the brief names it."),
                "request_id": _text(
                    description="The other branch's test, as a peer_outcome inbox item or an unfinished lead names it."
                ),
                "reference": _text(
                    description="A patch the champion's replay leaves out, its path as a 'left out: patch' line names "
                    "it: the reply is its text, a page at a time."
                ),
                "offset": _integer(
                    minimum=0,
                    description="Where in the patch's text the page starts, as the previous page's last line says.",
                ),
            }
        ),
    ),
)


def blackboard_tools(
    blackboard: Blackboard,
    briefing: StartupBriefProvider | None = None,
    *,
    branch: str,
    ttl_seconds: float,
    learnings: LearningFinder | None = None,
) -> tuple[BlackboardTool, ...]:
    """The tools of the specialist on ``branch``; ``briefing`` gives the brief, the blackboard's own by default.

    ``learnings`` finds the learnings for the learnings tool: the session's one ``LearningRecall`` adds past sessions'
    from the Experience KB; without it the tool finds this session's only.
    """

    bound = _SpecialistTools(blackboard, briefing or blackboard, learnings or SessionLearnings(), branch, ttl_seconds)
    return tuple(
        BlackboardTool(
            TOOL_PREFIX + spec.name, spec.description, spec.input_schema, _guarded(getattr(bound, spec.name))
        )
        for spec in _SPECS
    )


__all__ = [
    "CEILING_MIN_SHARE_PCT",
    "CEILING_OPS",
    "CODE_TITLE",
    "ERROR",
    "FOCUS_LINE",
    "LINE_CHARS",
    "MAX_LINES",
    "MAX_REPLY_CHARS",
    "PATCH_PAGE_CHARS",
    "SETTINGS_TITLE",
    "TOOL_PREFIX",
    "BlackboardTool",
    "PeerReplays",
    "WarmReplays",
    "blackboard_tools",
    "outcome",
    "render_answer",
    "render_brief",
    "render_ceiling",
    "render_claim",
    "render_log",
    "render_record",
    "render_peer_replay",
    "render_reference",
    "render_replay",
]
