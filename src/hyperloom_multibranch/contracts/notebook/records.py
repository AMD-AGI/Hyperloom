# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The objects the notebook defines: claims, test records, lineage, and what its queries return."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from hyperloom.contracts.recipe import KernelArtifact, RecipeDelta
from hyperloom_multibranch.contracts.llm_prices import TokenCounts
from hyperloom_multibranch.contracts.notebook.peers import (
    BranchLogEntry,
    Intent,
    IntentLayer,
    Judgement,
    SandboxFacts,
    TestKind,
    TestRequest,
    TestResult,
    Validity,
    Verdict,
)
from hyperloom_multibranch.contracts.notebook.vocab import (
    ClaimStatus,
    CommitOrigin,
    Conclusion,
    DedupStatus,
    FrontierBasis,
    InboxKind,
    LearningKind,
    LineageKind,
    MergeOutcome,
    QuarantineAction,
    Role,
    UnfinishedReason,
    UsageSource,
)

CLAIM_TOKEN_LENGTH = 24
MAX_OP_SHARES = 50
WHOLE_PCT = 100.0
SECONDS_PER_HOUR = 3600.0
#: The facts of a node's measurement environment that change what a test there measures (#63).
ENVIRONMENT_FIELDS = ("image_digest", "gpu_model", "power_cap_w", "partition_mode")


@dataclass(frozen=True)
class Writer:
    """The identity a notebook handle writes as; a specialist runtime writes for its own ``branch``."""

    role: Role
    name: str
    branch: str = ""


@dataclass(frozen=True)
class ClaimKey:
    """The same intent on the same state: ``parent_commit`` is the state."""

    parent_commit: str
    intent_fingerprint: str

    @property
    def token(self) -> str:
        """A file-name-safe name of the key."""

        joined = f"{self.parent_commit}\x00{self.intent_fingerprint}"
        return hashlib.sha256(joined.encode()).hexdigest()[:CLAIM_TOKEN_LENGTH]


@dataclass(frozen=True)
class Claim:
    """A lease on a key, held by a branch so a restarted specialist keeps it; times are Unix seconds."""

    key: ClaimKey
    holder: str
    intent: Intent
    action_fingerprint: str
    ttl_seconds: float
    acquired_at: float
    heartbeat_at: float
    expires_at: float
    status: ClaimStatus
    request_id: str | None = None
    generation: int = 0


@dataclass(frozen=True)
class TestRecord:
    """Everything known about one test (design section 5.2): request, result, judgement, conclusion and commit."""

    __test__ = False

    request: TestRequest
    result: TestResult | None = None
    judgement: Judgement | None = None
    conclusion: Conclusion | None = None
    commit_id: str | None = None
    requested_at: float = 0.0
    measured_at: float | None = None
    judged_at: float | None = None
    withdrawn_at: float | None = None
    withdrawal_reason: str = ""

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def in_flight(self) -> bool:
        """Requested, and neither measured nor withdrawn yet."""

        return self.result is None and self.withdrawn_at is None

    @property
    def measured(self) -> bool:
        """Its result measured the candidate, a failure of the candidate included; a failed measurement did not, nor
        a result the rule gate could not re-verify."""

        return (
            self.result is not None
            and self.result.validity is not Validity.MEASUREMENT_FAILED
            and self.conclusion is not Conclusion.UNVERIFIED
        )

    @property
    def judged(self) -> bool:
        """A test the gate judged after measuring it: what a branch's stall count counts."""

        return self.request.kind is TestKind.TEST and self.measured and self.judged_at is not None


@dataclass(frozen=True)
class Withdrawal:
    """The toolbox taking back a test it will not measure, such as the queued and running tests of a stopped branch."""

    request_id: str
    reason: str


@dataclass(frozen=True)
class DedupAnswer:
    """Whether an intent on a state is free; ``records`` are this state's, ``elsewhere`` only for reference.

    ``TESTED``: the same action was measured on this state. ``IN_PROGRESS``: ``claim`` holds the key, or another claim
    on this state carries the same action. ``SUSPECTED``: the same intent was measured here through another action.
    ``TESTING_ELSEWHERE``: another branch holds ``claim`` on the same intent, or has ``records`` of it in flight, on
    another state; a replay is never refused for it, as it tests a known change on the caller's own head. It is also the
    answer while another branch has an explore of the same action in flight on this state, under whatever intent.
    """

    key: ClaimKey
    action_fingerprint: str
    status: DedupStatus
    claim: Claim | None = None
    records: tuple[TestRecord, ...] = ()
    elsewhere: tuple[TestRecord, ...] = ()


@dataclass(frozen=True)
class MergeAttempt:
    """The integrator merging source commits onto the integration head; without a test it was skipped untested, as
    ``skipped_outcome`` says: it conflicted, or its change is already on the head."""

    source_commits: tuple[str, ...]
    integration_head: str
    request_id: str | None = None
    skipped_reason: str = ""
    skipped_outcome: MergeOutcome = MergeOutcome.CONFLICT_SKIPPED


@dataclass(frozen=True)
class LineageEvent:
    """A branch starting (``from_commit``), splitting off ``from_branch`` with fix ``commit_id``, or stopping.

    ``ROOT`` records the baseline commit ``commit_id`` the integration branch starts from, and may name the digest of
    its recipe (``Recipe.digest()``); a judgement names the recipe digest of every other commit.
    """

    kind: LineageKind
    branch: str
    commit_id: str = ""
    from_commit: str = ""
    from_branch: str = ""
    reason: str = ""
    recipe_digest: str | None = None


@dataclass(frozen=True)
class BaselineRecord:
    """The PRELUDE baseline the rule gate measured, per metric; every recorded value is anchored to it."""

    root_commit: str
    values: Mapping[str, float]
    dispersion_pct: Mapping[str, float]
    threshold_pct: Mapping[str, float]
    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class RegressionCase:
    """A correctness fix's reproduction case; it applies to every descendant of ``fix_commit``."""

    case_ref: str
    fix_commit: str
    request_id: str


@dataclass(frozen=True)
class QuarantineEvent:
    """A node, or some of its GPUs, taken out of or returned to service by the toolbox."""

    node: str
    action: QuarantineAction
    reason: str
    gpu_ids: tuple[int, ...] = ()
    request_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolResultPointer:
    """Where an asynchronous tool's result is; the result itself never enters the notebook.

    The one exception is ``op_shares``, which a TraceLens analysis fills with each op's ``percent_of_total`` on
    ``commit_id``, at most :data:`MAX_OP_SHARES` of them, so the Amdahl ceiling of a claim can be read on that commit.
    ``profile_request_id`` names the profile test whose trace was analysed, when it came from one.
    """

    branch: str
    job_id: str
    location: str
    summary: str
    commit_id: str = ""
    profile_request_id: str = ""
    op_shares: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.op_shares) > MAX_OP_SHARES:
            raise ValueError(f"a pointer carries at most {MAX_OP_SHARES} op shares, not {len(self.op_shares)}")
        if self.op_shares and not self.commit_id:
            raise ValueError("op shares need the commit they were profiled on")


def amdahl_ceiling_pct(share_pct: float) -> float:
    """The most, in percent, that speeding up a region worth ``share_pct`` of the time can gain: the region made free."""

    return math.inf if share_pct >= WHOLE_PCT else WHOLE_PCT * share_pct / (WHOLE_PCT - share_pct)


@dataclass(frozen=True)
class OpShares:
    """The latest TraceLens op shares on one commit, with the tool job and the profile test they came from."""

    commit_id: str
    job_id: str
    profile_request_id: str
    shares: Mapping[str, float]

    def largest(self, count: int) -> tuple[tuple[str, float], ...]:
        return tuple(sorted(self.shares.items(), key=lambda item: (-item[1], item[0]))[:count])


@dataclass(frozen=True)
class InboxItem:
    """A pointer pushed to a branch: a request id, a tool job id, or a lineage entry id."""

    kind: InboxKind
    branch: str
    pointer: str
    summary: str
    at: float


@dataclass(frozen=True)
class InboxPage:
    """New items, and the cursor to poll from next; the cursor is opaque."""

    items: tuple[InboxItem, ...]
    cursor: str


@dataclass(frozen=True)
class BranchLogRecord:
    entry_id: str
    at: float
    entry: BranchLogEntry


#: How a lesson of the RSI lesson pool, a learning a past session left, is named where a learning refers to one.
LESSON_ID = re.compile(r"lsn-[0-9a-f]{12}")


def is_lesson_id(value: str | None) -> bool:
    """Whether ``value`` names a past session's lesson in the RSI lesson pool rather than an entry of this notebook."""

    return value is not None and LESSON_ID.fullmatch(value) is not None


@dataclass(frozen=True)
class Learning:
    """What a specialist learnt, in its own words, for the next decision of any branch, now or in a later session.

    ``target``, ``mechanism`` and ``layer`` name the direction it is about, when it is about one; ``request_ids`` the
    tests behind it. ``supersedes`` names what it replaces: an earlier learning of the same branch, or a past session's
    lesson (:data:`LESSON_ID`) it revises. ``retires`` names a past session's lesson it shows wrong or outdated here,
    ``content`` saying why. Either on a past session's lesson cites at least one of this session's tests. The conditions
    it holds under are the session's, which the notebook knows; the specialist does not repeat them.
    """

    branch: str
    kind: LearningKind
    content: str
    target: str = ""
    mechanism: str = ""
    layer: IntentLayer | None = None
    request_ids: tuple[str, ...] = ()
    supersedes: str | None = None
    retires: str | None = None


@dataclass(frozen=True)
class LearningRecord:
    entry_id: str
    at: float
    learning: Learning


@dataclass(frozen=True)
class LlmUsage:
    """What one LLM launch or call used: ``branch`` is the branch whose specialist spent it, empty for the scheduler's
    own judge, resolver or fix critic; ``tokens`` are its tokens by how they are billed; ``cost_usd`` is what they
    cost, ``None`` for a model the price sheet does not list; ``launch`` names the CLI session it came from, to trace
    it back."""

    branch: str
    source: UsageSource
    cost_usd: float | None
    tokens: TokenCounts = TokenCounts()
    launch: str = ""

    def __post_init__(self) -> None:
        if self.cost_usd is not None and (not math.isfinite(self.cost_usd) or self.cost_usd < 0):
            raise ValueError(f"an LLM cost is a finite amount of dollars, at least 0, not {self.cost_usd!r}")


def priced_usd(usages: Iterable[LlmUsage]) -> float:
    """What ``usages`` cost; an unpriced model's usage adds its tokens to the record but nothing to the dollars."""

    return sum(usage.cost_usd for usage in usages if usage.cost_usd is not None)


@dataclass(frozen=True)
class LlmUsageRecord:
    entry_id: str
    at: float
    usage: LlmUsage


@dataclass(frozen=True)
class FrontierScore:
    """Expected gain over the PRELUDE baseline from one more unit of budget, its parts, and what it rests on.

    ``scorer`` names the scorer whose estimate it took, by the label that carries its version; empty in a score from
    before scorers were named.
    """

    expected_gain_pct: float
    components: Mapping[str, float] = field(default_factory=dict)
    basis: FrontierBasis | None = None
    scorer: str = ""


@dataclass(frozen=True)
class PendingCandidate:
    """A measured explore that beats its branch's head by at least the gate's bar while its confirm is queued or running.

    ``metric`` is the one the gate's bar is on; ``value`` is the explore's, read as the gate reads it, the mean over its
    measure-phase rounds at the session's concurrency; ``gain_over_head_pct`` is over the head's recorded value.
    """

    request_id: str
    confirm_request_id: str
    parent_commit: str
    metric: str
    value: float
    gain_over_head_pct: float


@dataclass(frozen=True)
class BranchProgress:
    """One branch at a glance; ``frontier`` is filled by a scoring progress query, such as the RSI's frontier scoring.

    ``judged_since_commit`` counts the branch's tests judged since its last commit (since it opened, before any),
    ``llm_usd`` what its specialist's LLM cost, and ``pending`` its candidates waiting for their confirm.
    """

    branch: str
    running: bool
    start_commit: str
    head_commit: str
    tests: int
    commits: int
    gpu_seconds: float
    head_value: Mapping[str, float]
    frontier: FrontierScore | None = None
    judged_since_commit: int = 0
    llm_usd: float = 0.0
    pending: tuple[PendingCandidate, ...] = ()


@dataclass(frozen=True)
class CommitNode:
    """A commit of the experiment repository; a merge also names the source commits it brought in.

    ``recipe_digest`` is the digest of the recipe the repository stores for the commit, when its writer named it.
    """

    commit_id: str
    parent_commit: str
    branch: str
    origin: CommitOrigin
    request_id: str | None = None
    merged_from: tuple[str, ...] = ()
    at: float = 0.0
    recipe_digest: str | None = None


@dataclass(frozen=True)
class BranchNode:
    branch: str
    origin: LineageKind
    start_commit: str
    head_commit: str
    from_branch: str = ""
    stop_reason: str | None = None
    opened_at: float = 0.0
    stopped_at: float | None = None


@dataclass(frozen=True)
class LineageView:
    """The branch tree: every commit and every branch."""

    commits: tuple[CommitNode, ...]
    branches: tuple[BranchNode, ...]

    def commit(self, commit_id: str) -> CommitNode | None:
        return next((node for node in self.commits if node.commit_id == commit_id), None)

    def ancestry(self, commit_id: str) -> tuple[str, ...]:
        """``commit_id`` and every commit it builds on, through parents and the sources merges brought in."""

        by_id = {node.commit_id: node for node in self.commits}
        seen: list[str] = []
        pending = [commit_id]
        while pending:
            current = pending.pop()
            node = by_id.get(current)
            if current in seen or node is None:
                continue
            seen.append(current)
            pending.extend([*node.merged_from, node.parent_commit])
        return tuple(seen)


@dataclass(frozen=True)
class MergeState:
    """The latest merge attempt a commit took part in."""

    outcome: MergeOutcome
    integration_head: str
    request_id: str | None = None
    merged_commit: str | None = None


@dataclass(frozen=True)
class CommittedCommit:
    """A specialist commit as the integrator sees it; ``gain_pct`` is its own gain over its parent commit."""

    commit_id: str
    branch: str
    parent_commit: str
    request_id: str
    gain_pct: float
    fixes_correctness: bool
    foundation_fix: bool
    merged_into: str | None = None
    last_merge: MergeState | None = None


@dataclass(frozen=True)
class RecordedValue:
    """A commit's recorded value per metric, and the tests it comes from (the baseline tests for the root)."""

    commit_id: str
    values: Mapping[str, float]
    request_ids: tuple[str, ...]


@dataclass(frozen=True)
class TriedDirection:
    """One test on a state a new specialist starts from or builds on; ``change`` is its recipe delta in one line."""

    commit_id: str
    request_id: str
    target: str
    mechanism: str
    layer: IntentLayer | None
    verdict: Verdict | None
    conclusion: Conclusion | None
    gain_pct: float | None
    change: str = ""


#: The tool a specialist replays a warm-start champion with, and the target the replay's intent claims.
REPLAY_TOOL = "notebook_replay"
REPLAY_TARGET = "warm start"


@dataclass(frozen=True)
class WarmReplay:
    """The change that makes a branch's head into a Recipe KB champion's recipe, to test on ``parent_commit``, the head.

    A gain lands only on the head it was measured against, so the change is made for the caller's head, not the root.
    It replays the whole recipe when ``reason`` is empty, else the launch config only, ``reason`` saying why. ``layer``
    is the deepest layer the change reaches, the one its intent claims.
    """

    canonical_id: str
    parent_commit: str
    delta: RecipeDelta
    layer: IntentLayer
    reason: str = ""


@dataclass(frozen=True)
class PeerReplay:
    """The change another branch's test ``request_id`` tried, to test on ``parent_commit``, the caller's head.

    It is that test's own delta, not the difference between the two heads, which would carry ``branch``'s earlier
    commits too; a component new to the caller's head takes the upstream it had where the test was made. ``layer`` is
    the deepest layer the change reaches, the one its intent claims. ``replaces`` are the head's kernel artifacts the
    change's take the place of: a target loads one artifact of a kind, so the test runs without them.
    """

    request_id: str
    branch: str
    parent_commit: str
    delta: RecipeDelta
    layer: IntentLayer
    replaces: tuple[KernelArtifact, ...] = ()


@dataclass(frozen=True)
class UnfinishedLead:
    """A result of this session worth finishing, by replaying test ``request_id`` of ``branch`` on another head.

    ``target`` and ``mechanism`` are its intent's, ``change`` its recipe delta in one line, ``measured_on`` the commit it
    was tested on, ``gain_pct`` its judged gain where it had one, and ``detail`` more on ``reason``: the commit it made
    and what its merge test found, the bar it fell short of, or why it was withdrawn.
    """

    request_id: str
    branch: str
    target: str
    mechanism: str
    change: str
    measured_on: str
    reason: UnfinishedReason
    gain_pct: float | None = None
    detail: str = ""


@dataclass(frozen=True)
class StartupBrief:
    """What a fill specialist reads at start; ``leads`` and ``lessons`` come from cross-session recall (MB2).

    ``op_shares`` are the latest op shares a tool reported on the start commit, which bound what speeding each op up
    can gain; ``None`` while the commit was not profiled. ``unfinished`` are this session's results worth finishing,
    on any branch. ``focus`` is the bottleneck the scheduler opened the branch on, empty when it named none.
    """

    start_commit: str
    tried: tuple[TriedDirection, ...]
    leads: tuple[str, ...] = ()
    lessons: tuple[str, ...] = ()
    op_shares: OpShares | None = None
    unfinished: tuple[UnfinishedLead, ...] = ()
    focus: str = ""


@dataclass(frozen=True)
class SearchHit:
    entry_id: str
    kind: str
    branch: str
    summary: str


@dataclass(frozen=True)
class SessionProfile:
    """What a session optimizes: its serving identity and workload, as PRELUDE fixed them.

    The serving identity is the seven dimensions of a Recipe KB canonical id. A workload without ``isl`` and ``osl`` is
    an AgentX trace replay. ``roofline_ceiling`` is the bound PRELUDE's roofline analysis puts on each metric.
    """

    session_id: str
    model: str
    hardware: str
    framework: str
    framework_version: str
    precision: str
    tp: int
    conc: int
    model_type: str = ""
    architectures: tuple[str, ...] = ()
    isl: int | None = None
    osl: int | None = None
    kernel_optimizer: str = ""
    roofline_ceiling: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ManifestFingerprint:
    """The versions one activity ran with: measurement sockets, specialist prompts and scheduling policies."""

    activity: str
    sockets: Mapping[str, str] = field(default_factory=dict)
    prompt_versions: Mapping[str, str] = field(default_factory=dict)
    policy_versions: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentChange:
    """``node``'s measurement environment moved under ``request_id``'s result, written at ``at``: ``changed`` holds
    each :data:`ENVIRONMENT_FIELDS` field that moved, with its last known value on the node and its value now."""

    node: str
    request_id: str
    at: float
    changed: Mapping[str, tuple[str, str]]


def _environment_readings(records: Iterable[TestRecord]) -> list[tuple[float, str, SandboxFacts]]:
    found = []
    for record in records:
        result = record.result
        if result is None or record.measured_at is None or result.sandbox is None or not result.sandbox.node_name:
            continue
        found.append((record.measured_at, record.request_id, result.sandbox))
    return sorted(found, key=lambda reading: reading[0])


def _environment_changes(records: Iterable[TestRecord]) -> tuple[EnvironmentChange, ...]:
    known: dict[str, dict[str, str]] = {}
    found = []
    for at, request_id, facts in _environment_readings(records):
        node = str(facts.node_name)
        last = known.setdefault(node, {})
        changed = {}
        for name in ENVIRONMENT_FIELDS:
            value = getattr(facts, name)
            if value is None or value == "":
                continue
            now, before = str(value), last.get(name)
            if before is not None and before != now:
                changed[name] = (before, now)
            last[name] = now
        if changed:
            found.append(EnvironmentChange(node, request_id, at, changed))
    return tuple(found)


@dataclass(frozen=True)
class NotebookSnapshot:
    """Everything the notebook held at ``taken_at``, read at once: what the RSI, the dashboard and the exports use."""

    taken_at: float
    profile: SessionProfile | None
    fingerprints: tuple[ManifestFingerprint, ...]
    records: tuple[TestRecord, ...]
    lineage: LineageView
    progress: tuple[BranchProgress, ...]
    committed: tuple[CommittedCommit, ...]
    claims: tuple[Claim, ...]
    intents: Mapping[str, Intent]
    baselines: tuple[BaselineRecord, ...]
    regressions: tuple[RegressionCase, ...]
    quarantined: tuple[QuarantineEvent, ...]
    branch_logs: Mapping[str, tuple[BranchLogRecord, ...]]
    learnings: tuple[LearningRecord, ...] = ()
    op_shares: Mapping[str, OpShares] = field(default_factory=dict)
    llm_usage: tuple[LlmUsageRecord, ...] = ()

    @property
    def baseline(self) -> BaselineRecord | None:
        return self.baselines[-1] if self.baselines else None

    @property
    def gpu_hours(self) -> float:
        """The GPU time every test took, whatever its branch or purpose; time outside the notebook's tests is not in it."""

        return sum(record.result.gpu_seconds for record in self.records if record.result is not None) / SECONDS_PER_HOUR

    @property
    def llm_usd(self) -> float:
        """What every priced LLM launch and call recorded cost, the scheduler's own included."""

        return priced_usd(record.usage for record in self.llm_usage)

    @property
    def llm_unpriced_tokens(self) -> TokenCounts:
        """The tokens of every LLM launch and call whose model the price sheet does not list, which ``llm_usd`` leaves
        out, the scheduler's own included."""

        return sum((record.usage.tokens for record in self.llm_usage if record.usage.cost_usd is None), TokenCounts())

    @property
    def environment_changes(self) -> tuple[EnvironmentChange, ...]:
        """Where a node's measurement environment moved between its results, in the order they were written.

        Each field is compared with its last known value on the same node. An empty value is no reading, so a fact a
        node could not read neither hides a change nor shows one.
        """

        return _environment_changes(self.records)

    def value_of(self, commit_id: str) -> Mapping[str, float]:
        """The recorded value of a commit: the baseline for the root, else the judgement that made the commit."""

        node = self.lineage.commit(commit_id)
        if node is not None and node.origin is CommitOrigin.ROOT:
            baseline = next((row for row in reversed(self.baselines) if row.root_commit == commit_id), None)
            return baseline.values if baseline is not None else {}
        made = next((record for record in self.records if record.commit_id == commit_id), None)
        return made.judgement.recorded_value if made is not None and made.judgement is not None else {}


__all__ = [
    "CLAIM_TOKEN_LENGTH",
    "ENVIRONMENT_FIELDS",
    "LESSON_ID",
    "MAX_OP_SHARES",
    "BaselineRecord",
    "BranchLogRecord",
    "BranchNode",
    "BranchProgress",
    "Claim",
    "ClaimKey",
    "CommitNode",
    "CommittedCommit",
    "DedupAnswer",
    "EnvironmentChange",
    "FrontierScore",
    "InboxItem",
    "InboxPage",
    "Learning",
    "LearningRecord",
    "LineageEvent",
    "LineageView",
    "LlmUsage",
    "priced_usd",
    "LlmUsageRecord",
    "ManifestFingerprint",
    "MergeAttempt",
    "MergeState",
    "NotebookSnapshot",
    "OpShares",
    "PeerReplay",
    "PendingCandidate",
    "QuarantineEvent",
    "RecordedValue",
    "RegressionCase",
    "SearchHit",
    "SessionProfile",
    "REPLAY_TARGET",
    "REPLAY_TOOL",
    "StartupBrief",
    "TestRecord",
    "WarmReplay",
    "ToolResultPointer",
    "TriedDirection",
    "UnfinishedLead",
    "Withdrawal",
    "Writer",
    "amdahl_ceiling_pct",
    "is_lesson_id",
]
