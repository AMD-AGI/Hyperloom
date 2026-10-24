# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The test request a multi-branch module hands to the toolbox, and the test result it gets back.

Like the recipe contract, every type is a frozen dataclass that validates on construction and round-trips through
``to_document()`` and ``from_document()``; the notebook stores the document form. A request is what the multi-branch
modules see: the toolbox turns it into a hosted or local test call. ``request_id`` is the toolbox's: it is generated
when the request is accepted, so a request a module builds has none and the accepted copy carries it.

A result is one row per request. The tries a request needed (an infrastructure retry, a rerun of an unclear failure)
are its :class:`Attempt` records, never separate results.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Literal

from hyperloom.contracts._validate import (
    MeasurementError,
    _bool,
    _enum,
    _int,
    _items,
    _mapping,
    _number,
    _optional_bool,
    _optional_int,
    _optional_number,
    _optional_sha256,
    _optional_text,
    _relative_dir,
    _sha256,
    _strings,
    _text,
)
from hyperloom.contracts.consistency import (
    COMPARED_BY,
    REFERENCE_ABSENT,
    REFERENCE_COMPARED,
    REFERENCE_STATUSES,
    SHORT_PROMPTS_COMPLETE,
    SHORT_PROMPTS_INCOMPLETE,
    SHORT_PROMPTS_STATUSES,
    OutputCounts,
    OutputDifference,
    ShortPromptConsistency,
    ShortPromptReference,
)
from hyperloom.contracts.recipe import RecipeDelta, RecipeError

PRIORITY_HIGHEST = 0
PRIORITY_LOWEST = 4

#: A phase name is lowercase words joined by ``_``; one ``.`` ties a part to the phase it splits
#: (``server_boot.weights_load``). A runner signs the names, so they stay plain tokens readers can group by.
_PHASE_NAME = re.compile(r"[a-z][a-z0-9_]{0,47}(\.[a-z][a-z0-9_]{0,47})?")
#: What a phase can say of itself: no phase lasts a year or holds more than a thousand cards for it, and no unix time of
#: one is past the year 5000. A signed manifest states phases too, and sums of them must stay finite.
MAX_PHASE_SECONDS = 366.0 * 24 * 3600
_MAX_CARDS = 1024
_MAX_UNIX_TIME = 1.0e11


_ACCURACY = "accuracy"
_FAILURE_CLASS = "failure_class"
#: The verdicts of a ``repro`` test: the case's failure is still there, or the case ran clean.
REPRODUCED = "reproduced"
NOT_REPRODUCED = "not_reproduced"
REPRO_VERDICTS = (REPRODUCED, NOT_REPRODUCED)
#: What a test's server did with one kernel table the test loaded: some of its rows were hit, none was although hits
#: were logged, or the logs cannot say. "Not exercised" and "not observable" are different findings: the first says the
#: table changed nothing the test ran, the second says nothing about it.
CARD_CLEAN = "clean"
CARD_UNKNOWN = "unknown"
CARD_FOREIGN = "foreign"
CARD_OCCUPANCY_STATES = (CARD_CLEAN, CARD_UNKNOWN, CARD_FOREIGN)
TABLE_EXERCISED = "exercised"
TABLE_NOT_EXERCISED = "not_exercised"
TABLE_NOT_OBSERVABLE = "not_observable"
TABLE_STATUSES = (TABLE_EXERCISED, TABLE_NOT_EXERCISED, TABLE_NOT_OBSERVABLE)
#: The counts of ``RequestClasses`` that can each be unknown: a round whose export cannot be read has none of them.
_REQUEST_COUNTS = (
    "requests_total",
    "failed_total",
    "failed_transport",
    "failed_no_content",
    "failed_empty",
    "failed_short",
)


class TestKind(str, Enum):
    """The kind of test; ``SWEEP`` runs a candidate over the workload's whole concurrency range (final selection)."""

    __test__ = False

    TEST = "test"
    PROFILE = "profile"
    ACCURACY = _ACCURACY
    REPRO = "repro"
    SWEEP = "sweep"


class Purpose(str, Enum):
    """Why a test is run; the toolbox takes its default priority from it."""

    BASELINE = "baseline"
    HEALTH = "health"
    EXPLORE = "explore"
    CONFIRM = "confirm"
    INTEGRATION = "integration"
    PROFILE = "profile"
    ACCURACY = _ACCURACY
    REPRO = "repro"
    FINAL = "final"


class Validity(str, Enum):
    """Whether a test measured the candidate; ``SUCCESS`` and ``CANDIDATE_FAILURE`` are the notebook's older names."""

    SUCCEEDED = "succeeded"
    CANDIDATE_FAILED = "candidate_failed"
    MEASUREMENT_FAILED = "measurement_failed"
    SUCCESS = SUCCEEDED
    CANDIDATE_FAILURE = CANDIDATE_FAILED


class FailureClass(str, Enum):
    INFRA = "infra"
    UNCLEAR = "unclear"
    CANDIDATE = "candidate"


# --- validation --------------------------------------------------------------------------------------------------


def _delta(value: object) -> RecipeDelta:
    try:
        return RecipeDelta.from_document(value)
    except RecipeError as exc:
        raise MeasurementError(f"delta: {exc}") from exc


# --- the request -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AttachmentRef:
    """A file or directory the test needs, addressed by content: the blob store holds it under ``sha256``."""

    id: str
    kind: Literal["file", "dir"]
    sha256: str

    def __post_init__(self) -> None:
        _text(self.id, "an attachment id")
        if self.kind not in ("file", "dir"):
            raise MeasurementError(f"an attachment kind must be 'file' or 'dir', got {self.kind!r}")
        _sha256(self.sha256, f"the sha256 of attachment {self.id}")

    def to_document(self) -> dict[str, object]:
        return {"id": self.id, "kind": self.kind, "sha256": self.sha256}

    @classmethod
    def from_document(cls, document: object) -> AttachmentRef:
        fields = _mapping(document, "an attachment")
        return cls(id=fields.get("id"), kind=fields.get("kind"), sha256=fields.get("sha256"))


@dataclass(frozen=True)
class TestRequest:
    """A test a module asks for; the claim it measures is (``parent_commit``, ``intent_fingerprint``).

    ``request_id`` is empty until the toolbox accepts the request; :meth:`accepted` returns the accepted copy.
    ``priority`` of ``None`` lets the toolbox pick the default of ``purpose``. ``concurrency`` empty means the active
    main operating point. ``diff_ref`` points at the stored diff the notebook shows for the request.
    """

    __test__ = False

    kind: TestKind
    purpose: Purpose
    branch_id: str
    parent_commit: str
    delta: RecipeDelta = field(default_factory=RecipeDelta)
    intent_fingerprint: str | None = None
    priority: int | None = None
    retest_of: str | None = None
    avoid_nodes: tuple[str, ...] = ()
    accuracy: bool = False
    concurrency: tuple[int, ...] = ()
    attachments: tuple[AttachmentRef, ...] = ()
    request_id: str | None = None
    diff_ref: str = ""
    consistency_check: bool | None = None

    def __post_init__(self) -> None:
        _enum(TestKind, self.kind, "kind")
        _enum(Purpose, self.purpose, "purpose")
        _text(self.branch_id, "branch_id")
        _text(self.parent_commit, "parent_commit")
        _optional_text(self.intent_fingerprint, "intent_fingerprint")
        _optional_text(self.request_id, "request_id")
        _optional_text(self.retest_of, "retest_of")
        if not isinstance(self.delta, RecipeDelta):
            raise MeasurementError(f"delta must be a RecipeDelta, got {self.delta!r}")
        if self.consistency_check is not None:
            _bool(self.consistency_check, "consistency_check")
        self._check_priority()
        self._check_lists()
        self._check_combinations()

    def _check_priority(self) -> None:
        if self.priority is not None:
            value = _int(self.priority, "priority", minimum=PRIORITY_HIGHEST)
            if value > PRIORITY_LOWEST:
                raise MeasurementError(f"priority must be {PRIORITY_HIGHEST} to {PRIORITY_LOWEST}, got {value}")

    def _check_lists(self) -> None:
        for node in self.avoid_nodes:
            _text(node, "an avoided node")
        for value in self.concurrency:
            _int(value, "a concurrency", minimum=1)
        if len(set(self.concurrency)) != len(self.concurrency):
            raise MeasurementError(f"concurrency values must not repeat, got {self.concurrency!r}")
        ids = [attachment.id for attachment in self.attachments]
        if not all(isinstance(a, AttachmentRef) for a in self.attachments) or len(set(ids)) != len(ids):
            raise MeasurementError(f"attachments must be AttachmentRef values with distinct ids, got {ids!r}")

    def _check_combinations(self) -> None:
        if (self.purpose is Purpose.CONFIRM) != (self.retest_of is not None):
            raise MeasurementError("retest_of is required for purpose 'confirm' and allowed for no other purpose")
        if self.purpose in (Purpose.BASELINE, Purpose.HEALTH) and self.delta != RecipeDelta():
            raise MeasurementError(f"a '{self.purpose.value}' test measures the parent as it is: delta must be empty")
        if self.kind is TestKind.ACCURACY and not self.accuracy:
            raise MeasurementError("an 'accuracy' test only evaluates accuracy: accuracy must be true")
        if self.kind is TestKind.REPRO and not self.attachments:
            raise MeasurementError("a 'repro' test needs the reproduction case as an attachment")
        if len(self.concurrency) > 1 and self.kind not in (TestKind.TEST, TestKind.SWEEP):
            raise MeasurementError(f"several concurrency values are only for 'test' requests, not {self.kind.value!r}")

    @property
    def forces_consistency_check(self) -> bool:
        """Whether the caller asks for the short-prompt consistency check: ``True`` forces it; ``None`` and ``False`` leave
        it to the toolbox's rule, because a caller may add the check but never remove one the rule demands."""

        return self.consistency_check is True

    @property
    def branch(self) -> str:
        """The notebook's name for ``branch_id``."""

        return self.branch_id

    @property
    def accuracy_rounds(self) -> bool:
        """The notebook's name for ``accuracy``."""

        return self.accuracy

    @property
    def action_fingerprint(self) -> str:
        return self.delta.fingerprint()

    def accepted(self, request_id: str) -> TestRequest:
        """The copy the toolbox stores once it accepted the request under ``request_id``."""

        return replace(self, request_id=request_id)

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {
            "kind": self.kind.value,
            "purpose": self.purpose.value,
            "branch_id": self.branch_id,
            "parent_commit": self.parent_commit,
            "delta": self.delta.to_document(),
            "intent_fingerprint": self.intent_fingerprint,
            "priority": self.priority,
            "retest_of": self.retest_of,
            "avoid_nodes": list(self.avoid_nodes),
            "accuracy": self.accuracy,
            "concurrency": list(self.concurrency),
            "attachments": [attachment.to_document() for attachment in self.attachments],
            "request_id": self.request_id,
            "diff_ref": self.diff_ref,
        }
        if self.consistency_check is not None:
            document["consistency_check"] = self.consistency_check
        return document

    @classmethod
    def from_document(cls, document: object) -> TestRequest:
        f = _mapping(document, "a test request")
        return cls(
            kind=_enum(TestKind, f.get("kind"), "kind"),
            purpose=_enum(Purpose, f.get("purpose"), "purpose"),
            branch_id=f.get("branch_id"),
            parent_commit=f.get("parent_commit"),
            delta=_delta(f.get("delta", {})),
            intent_fingerprint=f.get("intent_fingerprint"),
            priority=f.get("priority"),
            retest_of=f.get("retest_of"),
            avoid_nodes=tuple(_items(f.get("avoid_nodes", ()), "avoid_nodes")),
            accuracy=_bool(f.get("accuracy", False), _ACCURACY),
            concurrency=tuple(_items(f.get("concurrency", ()), "concurrency")),
            attachments=tuple(map(AttachmentRef.from_document, _items(f.get("attachments", ()), "attachments"))),
            request_id=f.get("request_id"),
            diff_ref=f.get("diff_ref", ""),
            consistency_check=_optional_bool(f.get("consistency_check"), "consistency_check"),
        )


# --- the result --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SandboxFacts:
    """Where one attempt ran; ``backend`` tells local measurement on one host from hosted measurement.

    ``card_occupancy`` is what the runner's signed result says of who else used the attempt's cards: ``clean`` when every
    look at them found no process outside the test, ``foreign`` (with ``card_occupancy_reason`` naming the process, the card
    and when) when one was seen and the card guard only observes, ``unknown`` (with its reason) when it could not look, and
    empty when the attempt did not say (an attempt that failed before a result, a sandbox with no card, a runner that
    predates it). It is never ``clean`` for a result nobody looked at. ``card_occupancy_method`` says how the runner told a
    process outside the test from the test's own (empty when it did not say).
    """

    sandbox_id: str
    node_name: str | None = None
    gpu_model: str | None = None
    gpu_count: int = 0
    power_cap_w: float | None = None
    partition_mode: str | None = None
    image_digest: str | None = None
    same_node: bool = False
    gpu_ids: tuple[int, ...] = ()
    backend: str = ""
    card_occupancy: str = ""
    card_occupancy_reason: str = ""
    card_occupancy_method: str = ""

    def __post_init__(self) -> None:
        _text(self.sandbox_id, "a sandbox id")
        for value, what in ((self.node_name, "node_name"), (self.gpu_model, "gpu_model")):
            _optional_text(value, what)
        _optional_text(self.partition_mode, "partition_mode")
        _optional_text(self.image_digest, "image_digest")
        _int(self.gpu_count, "gpu_count", minimum=0)
        _optional_number(self.power_cap_w, "power_cap_w")
        _bool(self.same_node, "same_node")
        for gpu in self.gpu_ids:
            _int(gpu, "a gpu id", minimum=0)
        if not isinstance(self.backend, str):
            raise MeasurementError(f"backend must be a string, got {self.backend!r}")
        self._check_card_occupancy()

    def _check_card_occupancy(self) -> None:
        if self.card_occupancy not in ("", *CARD_OCCUPANCY_STATES):
            raise MeasurementError(
                f"card_occupancy must be '' or one of {CARD_OCCUPANCY_STATES}, got {self.card_occupancy!r}"
            )
        if not isinstance(self.card_occupancy_reason, str):
            raise MeasurementError(f"card_occupancy_reason must be a string, got {self.card_occupancy_reason!r}")
        if not isinstance(self.card_occupancy_method, str):
            raise MeasurementError(f"card_occupancy_method must be a string, got {self.card_occupancy_method!r}")
        if (self.card_occupancy in (CARD_UNKNOWN, CARD_FOREIGN)) != bool(self.card_occupancy_reason):
            raise MeasurementError("card_occupancy 'unknown' or 'foreign' needs its reason, and 'clean' has none")

    @property
    def node(self) -> str | None:
        """The notebook's name for ``node_name``."""

        return self.node_name

    @property
    def power_cap_watts(self) -> float | None:
        """The notebook's name for ``power_cap_w``."""

        return self.power_cap_w

    def to_document(self) -> dict[str, object]:
        return {
            "sandbox_id": self.sandbox_id,
            "node_name": self.node_name,
            "gpu_model": self.gpu_model,
            "gpu_count": self.gpu_count,
            "power_cap_w": self.power_cap_w,
            "partition_mode": self.partition_mode,
            "image_digest": self.image_digest,
            "same_node": self.same_node,
            "gpu_ids": list(self.gpu_ids),
            "backend": self.backend,
            "card_occupancy": self.card_occupancy,
            "card_occupancy_reason": self.card_occupancy_reason,
            "card_occupancy_method": self.card_occupancy_method,
        }

    @classmethod
    def from_document(cls, document: object) -> SandboxFacts:
        f = _mapping(document, "sandbox facts")
        return cls(
            sandbox_id=f.get("sandbox_id"),
            node_name=f.get("node_name"),
            gpu_model=f.get("gpu_model"),
            gpu_count=f.get("gpu_count", 0),
            power_cap_w=f.get("power_cap_w"),
            partition_mode=f.get("partition_mode"),
            image_digest=f.get("image_digest"),
            same_node=f.get("same_node", False),
            gpu_ids=tuple(_items(f.get("gpu_ids", ()), "gpu_ids")),
            backend=f.get("backend", ""),
            card_occupancy=f.get("card_occupancy", ""),
            card_occupancy_reason=f.get("card_occupancy_reason", ""),
            card_occupancy_method=f.get("card_occupancy_method", ""),
        )


@dataclass(frozen=True)
class RoundValue:
    """One benchmark round: its phase, the concurrency it ran at and the metrics of ``benchmark_report.json``.

    ``metrics`` keeps the keys of ``benchmark_report.json`` exactly as the report has them (for example
    ``output_throughput`` or ``e2e_norm_intvty_p90``), with no renaming or unit change, so a consumer reads them with
    the names of ``hyperloom.common.perf_metric``.
    """

    phase: Literal["warmup", "measure"]
    concurrency: int | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    completed_requests: int | None = None
    failed_requests: int | None = None

    def __post_init__(self) -> None:
        if self.phase not in ("warmup", "measure"):
            raise MeasurementError(f"a round phase must be 'warmup' or 'measure', got {self.phase!r}")
        _optional_int(self.concurrency, "concurrency", minimum=1)
        _optional_int(self.completed_requests, "completed_requests")
        _optional_int(self.failed_requests, "failed_requests")
        for name, value in _mapping(self.metrics, "metrics").items():
            _text(name, "a metric name")
            _number(value, f"metric {name}")

    def to_document(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "concurrency": self.concurrency,
            "metrics": dict(sorted(self.metrics.items())),
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
        }

    @classmethod
    def from_document(cls, document: object) -> RoundValue:
        f = _mapping(document, "a round")
        return cls(
            phase=f.get("phase"),
            concurrency=f.get("concurrency"),
            metrics=_mapping(f.get("metrics", {}), "metrics"),
            completed_requests=f.get("completed_requests"),
            failed_requests=f.get("failed_requests"),
        )


@dataclass(frozen=True)
class AccuracyValue:
    task: str
    metric: str
    score: float
    #: What the evaluator ran, e.g. ``lm_eval=0.4.13;gsm8k=3``; empty when it did not say.
    version: str = ""

    def __post_init__(self) -> None:
        _text(self.task, "an accuracy task")
        _text(self.metric, "an accuracy metric")
        _number(self.score, "an accuracy score")
        if not isinstance(self.version, str) or self.version != self.version.strip():
            raise MeasurementError(
                f"an accuracy version must be a string without surrounding whitespace, got {self.version!r}"
            )

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {"task": self.task, "metric": self.metric, "score": self.score}
        if self.version:
            document["version"] = self.version
        return document

    @classmethod
    def from_document(cls, document: object) -> AccuracyValue:
        f = _mapping(document, "an accuracy value")
        return cls(task=f.get("task"), metric=f.get("metric"), score=f.get("score"), version=f.get("version", ""))


def _check_failure(failure_class: object, reason: object) -> None:
    if failure_class is not None:
        _enum(FailureClass, failure_class, _FAILURE_CLASS)
    if reason is not None and not isinstance(reason, str):
        raise MeasurementError(f"failure_reason must be a string, got {reason!r}")
    if failure_class is None and reason:
        raise MeasurementError("a failure_reason needs a failure_class")


@dataclass(frozen=True)
class Attempt:
    """One try at running a test; failed tries are kept for node health and cost, never as negative results."""

    sandbox: SandboxFacts | None = None
    failure_class: FailureClass | None = None
    failure_reason: str | None = None
    started_at: float | None = None
    finished_at: float | None = None

    def __post_init__(self) -> None:
        if self.sandbox is not None and not isinstance(self.sandbox, SandboxFacts):
            raise MeasurementError(f"an attempt's sandbox must be SandboxFacts, got {self.sandbox!r}")
        _check_failure(self.failure_class, self.failure_reason)
        start, end = _optional_number(self.started_at, "started_at"), _optional_number(self.finished_at, "finished_at")
        if start is not None and end is not None and end < start:
            raise MeasurementError(f"an attempt cannot finish ({end}) before it starts ({start})")

    @property
    def failure(self) -> FailureClass | None:
        """The notebook's name for ``failure_class``."""

        return self.failure_class

    @property
    def node(self) -> str | None:
        return self.sandbox.node_name if self.sandbox is not None else None

    def to_document(self) -> dict[str, object]:
        return {
            "sandbox": None if self.sandbox is None else self.sandbox.to_document(),
            "failure_class": None if self.failure_class is None else self.failure_class.value,
            "failure_reason": self.failure_reason,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_document(cls, document: object) -> Attempt:
        f = _mapping(document, "an attempt")
        sandbox = f.get("sandbox")
        failure = f.get("failure_class")
        return cls(
            sandbox=None if sandbox is None else SandboxFacts.from_document(sandbox),
            failure_class=None if failure is None else _enum(FailureClass, failure, _FAILURE_CLASS),
            failure_reason=f.get("failure_reason"),
            started_at=f.get("started_at"),
            finished_at=f.get("finished_at"),
        )


@dataclass(frozen=True)
class TableHits:
    """How much of one kernel table a test loaded its server used, read from the server's logs.

    ``env`` is the variable that loaded the table and ``target`` and ``sha256`` name its artifact. ``rows`` is the
    table's data rows (``None`` when the table is not read as rows), ``rows_hit`` how many of them aiter resolved at
    least one lookup in, and ``hit_lines`` how many log lines said so (every tensor-parallel rank logs its own).

    ``status`` is ``exercised`` (``rows_hit`` of at least one), ``not_exercised`` (the server logged its hits and none
    was a row of this table: the table changed nothing the test ran) or ``not_observable`` (the counts are ``None``:
    the logs cannot tell, and ``reason`` says why). A table nothing was seen to read is never reported as flat.
    """

    env: str
    target: str
    sha256: str
    status: str
    rows: int | None = None
    rows_hit: int | None = None
    hit_lines: int | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        _text(self.env, "a table's env")
        _text(self.target, "a table's target")
        _sha256(self.sha256, "a table's sha256")
        if self.status not in TABLE_STATUSES:
            raise MeasurementError(f"a table's status must be one of {list(TABLE_STATUSES)}, got {self.status!r}")
        rows = _optional_int(self.rows, "a table's rows")
        rows_hit, hit_lines = _optional_int(self.rows_hit, "rows_hit"), _optional_int(self.hit_lines, "hit_lines")
        if not isinstance(self.reason, str):
            raise MeasurementError(f"a table's reason must be a string, got {self.reason!r}")
        if rows is not None and rows_hit is not None and rows_hit > rows:
            raise MeasurementError(f"a table cannot have {rows_hit} rows hit of {rows} rows")
        self._check_status(rows_hit, hit_lines)

    def _check_status(self, rows_hit: int | None, hit_lines: int | None) -> None:
        if self.status == TABLE_NOT_OBSERVABLE:
            if rows_hit is not None or hit_lines is not None or not self.reason:
                raise MeasurementError("a not_observable table has no counts and says why")
        elif rows_hit is None or hit_lines is None:
            raise MeasurementError(f"a {self.status} table counts its rows hit and its hit lines")
        elif self.status == TABLE_EXERCISED and (rows_hit < 1 or hit_lines < 1):
            raise MeasurementError("an exercised table has at least one row hit")
        elif self.status == TABLE_NOT_EXERCISED and (rows_hit or hit_lines):
            raise MeasurementError("a not_exercised table has no row hit")

    def to_document(self) -> dict[str, object]:
        return {
            "env": self.env,
            "target": self.target,
            "sha256": self.sha256,
            "status": self.status,
            "rows": self.rows,
            "rows_hit": self.rows_hit,
            "hit_lines": self.hit_lines,
            "reason": self.reason,
        }

    @classmethod
    def from_document(cls, document: object) -> TableHits:
        f = _mapping(document, "a table's hits")
        return cls(
            env=f.get("env"),
            target=f.get("target"),
            sha256=f.get("sha256"),
            status=f.get("status"),
            rows=f.get("rows"),
            rows_hit=f.get("rows_hit"),
            hit_lines=f.get("hit_lines"),
            reason=f.get("reason", ""),
        )


@dataclass(frozen=True)
class RequestClasses:
    """The requests of one measured AgentX round that its client marked failed, counted by what the server did.

    Read from the round's per-request export (aiperf's ``profile_export.jsonl``), which the signed bundle keeps:
    ``export`` is that file's path relative to the test's ``output_dir`` (``""`` when the round left none), ``point`` the
    round's measurement point and ``concurrency`` the load it ran at. ``requests_total`` counts the profiled requests (a
    warm-up request is no part of the client's own error rate and none of these counts) and ``failed_total`` those the
    client marked failed, which are exactly ``failed_transport`` (no usable answer from the server: a connection error,
    a timeout, a cancellation or an HTTP error status) plus ``failed_no_content`` (the server answered and the client
    found nothing in the answer to use).

    ``failed_no_content`` is ``failed_empty`` (the server produced zero output tokens) plus ``failed_short`` (it
    produced one or more and the client still counted the answer failed: the client's own decision, no threshold of
    ours). The client as released exports no output token count for a request it counted failed; the one the runner
    patches does, for a reply it found nothing in. The two are told apart when every no-content failure's line carries
    one; otherwise both are ``None`` and ``reason`` says so.

    A count that could not be computed is ``None`` and ``reason`` says why: never zero. A consumer that needs a count
    it finds ``None`` has no answer, not an answer of "none".
    """

    point: int
    export: str = ""
    concurrency: int | None = None
    requests_total: int | None = None
    failed_total: int | None = None
    failed_transport: int | None = None
    failed_no_content: int | None = None
    failed_empty: int | None = None
    failed_short: int | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        _int(self.point, "a request class's point", minimum=0)
        _relative_dir(self.export, "a request class's export")
        _optional_int(self.concurrency, "a request class's concurrency", minimum=1)
        if not isinstance(self.reason, str):
            raise MeasurementError(f"a request class's reason must be a string, got {self.reason!r}")
        counts = {name: getattr(self, name) for name in _REQUEST_COUNTS}
        for name, value in counts.items():
            _optional_int(value, name)
        if None in counts.values() and not self.reason:
            raise MeasurementError("a request class with a count that could not be computed says why")
        self._check_sums()

    def _check_sums(self) -> None:
        total, failed, transport, no_content = (
            self.requests_total,
            self.failed_total,
            self.failed_transport,
            self.failed_no_content,
        )
        known = [value is not None for value in (total, failed, transport, no_content)]
        if any(known) and not all(known):
            raise MeasurementError(
                "the requests, the failed ones and their transport and no-content parts come together"
            )
        if failed is not None and (failed != transport + no_content or failed > total):
            raise MeasurementError(
                f"{failed} failed requests of {total} are {transport} transport and {no_content} no-content failures"
            )
        if (self.failed_empty is None) != (self.failed_short is None):
            raise MeasurementError("the empty and the short failures come together")
        if self.failed_empty is not None and (
            no_content is None or self.failed_empty + self.failed_short != no_content
        ):
            raise MeasurementError("the empty and short failures add up to the no-content failures")

    def to_document(self) -> dict[str, object]:
        return {
            "point": self.point,
            "export": self.export,
            "concurrency": self.concurrency,
            **{name: getattr(self, name) for name in _REQUEST_COUNTS},
            "reason": self.reason,
        }

    @classmethod
    def from_document(cls, document: object) -> RequestClasses:
        f = _mapping(document, "a request class")
        return cls(
            point=f.get("point"),
            export=f.get("export", ""),
            concurrency=f.get("concurrency"),
            reason=f.get("reason", ""),
            **{name: f.get(name) for name in _REQUEST_COUNTS},
        )


@dataclass(frozen=True)
class PhaseTiming:
    """How long one phase of a test took, and where it began.

    ``started_at`` is a unix time on the clock of whoever measured the phase; a result lists its phases in the order
    they happened. ``seconds`` is the phase's length. ``gpu_seconds`` is the card time it held, its seconds times the
    cards the test holds, counted as ``TestResult.gpu_seconds`` counts the whole test (``0.0`` where nothing has put the
    cards on it). ``point`` is the index of the measurement point a round's phase belongs to, ``None`` for a phase of
    the test as a whole.
    """

    name: str
    started_at: float
    seconds: float
    gpu_seconds: float = 0.0
    point: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _PHASE_NAME.fullmatch(self.name):
            raise MeasurementError(f"a phase name is lowercase words joined by '_', got {self.name!r}")
        for value, what, most in (
            (self.started_at, "start", _MAX_UNIX_TIME),
            (self.seconds, "seconds", MAX_PHASE_SECONDS),
            (self.gpu_seconds, "gpu_seconds", MAX_PHASE_SECONDS * _MAX_CARDS),
        ):
            if _number(value, f"a phase's {what}") < 0:
                raise MeasurementError(f"a phase's {what} cannot be negative, got {value}")
            if value > most:
                raise MeasurementError(f"a phase's {what} cannot be more than {most:g}, got {value}")
        _optional_int(self.point, "a phase's point")

    def to_document(self) -> dict[str, object]:
        return {
            "name": self.name,
            "started_at": self.started_at,
            "seconds": self.seconds,
            "gpu_seconds": self.gpu_seconds,
            "point": self.point,
        }

    @classmethod
    def from_document(cls, document: object) -> PhaseTiming:
        f = _mapping(document, "a phase")
        return cls(
            name=f.get("name"),
            started_at=f.get("started_at"),
            seconds=f.get("seconds"),
            gpu_seconds=f.get("gpu_seconds", 0.0),
            point=f.get("point"),
        )


def _check_phases(phases: object, unaccounted: object) -> None:
    if not isinstance(phases, tuple) or not all(isinstance(phase, PhaseTiming) for phase in phases):
        raise MeasurementError(f"phases must be a tuple of PhaseTiming, got {phases!r}")
    if bool(phases) != (unaccounted is not None):
        raise MeasurementError("unaccounted_seconds is set exactly when the result has phases")
    _optional_number(unaccounted, "unaccounted_seconds")


@dataclass(frozen=True)
class TestResult:
    """The final outcome of one request; ``failure_class`` is the last attempt's class and empty on success.

    ``output_dir`` is relative to the session directory. The four fingerprints come from the signed result manifest.
    The timing, cost and profile fields are what the notebook asks of the toolbox beyond the measurement itself.
    ``cold_start_seconds`` is how long the test's server took from its spawn to be ready, measured by the sandbox that
    ran it; ``None`` when the test booted none or nothing recorded it.
    ``repro_verdict`` is the one outcome of a ``repro`` test: ``""`` for any other test, ``"reproduced"`` or
    ``"not_reproduced"`` for a repro test that ran its case to the end (``SUCCEEDED`` only).
    ``table_hits`` has one entry for each kernel table the test's recipe loaded, and ``runtime_findings`` says what each
    entry means in a sentence; a succeeded test that loaded none has neither. ``TableHits`` tells "the table was not
    exercised" from "the logs cannot say", which a flat result alone cannot.
    ``request_classes`` has one entry for each measured round of an AgentX test (none for any other test): how many of
    the round's requests its client marked failed, by what the server did, and where the round's per-request export is.
    A document written before it existed has none, which is not a round that failed no request.
    ``phases`` says where the test's wall time (``finished_at`` less ``started_at``) went, as the phases that followed
    one another (the sandbox starting, the environment rebuilt, the server booting, each round, the result signed and
    landed), and ``unaccounted_seconds`` is the part of that no phase covers, so the phases and it add up to the wall
    time. It is negative when the phases outrun the wall time, as the runner phases of a test this process re-attached
    to do, and it is ``None`` exactly when the result has no phases (a document written before they existed, a result
    no measurement made).
    ``short_prompt_consistency`` is the check of a test the toolbox asked it for (``None`` for any other test): a frozen set
    of short prompts answered by fresh server processes, compared with each other and with the parent's, as counts and
    positions only.
    A document written while the greedy decode pass existed may carry ``decode_outputs`` and ``decode_identity``; they
    mean nothing now and ``from_document`` leaves them out, so such a document still loads.
    """

    __test__ = False

    request_id: str
    validity: Validity
    failure_class: FailureClass | None = None
    failure_reason: str | None = None
    rounds: tuple[RoundValue, ...] = ()
    accuracy: tuple[AccuracyValue, ...] = ()
    sandbox: SandboxFacts | None = None
    attempts: tuple[Attempt, ...] = ()
    manifest_sha256: str | None = None
    signing_key_fingerprint: str | None = None
    spec_fingerprint: str | None = None
    surface_sha256: str | None = None
    output_dir: str = ""
    queued_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    gpu_seconds: float = 0.0
    profile_location: str = ""
    repro_verdict: str = ""
    nonfatal_warnings: tuple[str, ...] = ()
    cold_start_seconds: float | None = None
    table_hits: tuple[TableHits, ...] = ()
    runtime_findings: tuple[str, ...] = ()
    request_classes: tuple[RequestClasses, ...] = ()
    phases: tuple[PhaseTiming, ...] = ()
    unaccounted_seconds: float | None = None
    short_prompt_consistency: ShortPromptConsistency | None = None

    def __post_init__(self) -> None:
        _text(self.request_id, "request_id")
        _enum(Validity, self.validity, "validity")
        _check_failure(self.failure_class, self.failure_reason)
        self._check_validity()
        if self.sandbox is not None and not isinstance(self.sandbox, SandboxFacts):
            raise MeasurementError(f"sandbox must be SandboxFacts, got {self.sandbox!r}")
        _optional_sha256(self.manifest_sha256, "manifest_sha256")
        _optional_sha256(self.surface_sha256, "surface_sha256")
        _optional_text(self.signing_key_fingerprint, "signing_key_fingerprint")
        _optional_text(self.spec_fingerprint, "spec_fingerprint")
        _relative_dir(self.output_dir, "output_dir")
        if _number(self.gpu_seconds, "gpu_seconds") < 0:
            raise MeasurementError(f"gpu_seconds cannot be negative, got {self.gpu_seconds}")
        for value, what in ((self.queued_at, "queued_at"), (self.started_at, "started_at")):
            _optional_number(value, what)
        _optional_number(self.finished_at, "finished_at")
        cold_start = _optional_number(self.cold_start_seconds, "cold_start_seconds")
        if cold_start is not None and cold_start < 0:
            raise MeasurementError(f"cold_start_seconds cannot be negative, got {cold_start}")
        self._check_repro_verdict()
        _strings(self.nonfatal_warnings, "nonfatal_warnings")
        _strings(self.runtime_findings, "runtime_findings")
        if not isinstance(self.table_hits, tuple) or not all(isinstance(t, TableHits) for t in self.table_hits):
            raise MeasurementError(f"table_hits must be a tuple of TableHits, got {self.table_hits!r}")
        if not isinstance(self.request_classes, tuple) or not all(
            isinstance(c, RequestClasses) for c in self.request_classes
        ):
            raise MeasurementError(f"request_classes must be a tuple of RequestClasses, got {self.request_classes!r}")
        _check_phases(self.phases, self.unaccounted_seconds)
        consistency = self.short_prompt_consistency
        if consistency is not None and not isinstance(consistency, ShortPromptConsistency):
            raise MeasurementError(f"short_prompt_consistency must be a ShortPromptConsistency, got {consistency!r}")

    def _check_repro_verdict(self) -> None:
        if self.repro_verdict not in ("", *REPRO_VERDICTS):
            raise MeasurementError(f"repro_verdict must be '' or one of {REPRO_VERDICTS}, got {self.repro_verdict!r}")
        if self.repro_verdict and self.validity is not Validity.SUCCEEDED:
            raise MeasurementError("only a succeeded test has a repro_verdict: a case that did not run has no verdict")

    def _check_validity(self) -> None:
        if self.validity is Validity.SUCCEEDED:
            if self.failure_class is not None:
                raise MeasurementError("a succeeded test has no failure_class")
        elif self.validity is Validity.CANDIDATE_FAILED:
            if self.failure_class not in (None, FailureClass.CANDIDATE, FailureClass.UNCLEAR):
                raise MeasurementError(
                    f"a candidate_failed test fails as candidate or unclear, got {self.failure_class}"
                )
        elif self.failure_class is FailureClass.CANDIDATE:
            raise MeasurementError("a measurement_failed test is not a failure of the candidate")

    @property
    def failure(self) -> FailureClass | None:
        """The notebook's name for ``failure_class``."""

        return self.failure_class

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {
            "request_id": self.request_id,
            "validity": self.validity.value,
            "failure_class": None if self.failure_class is None else self.failure_class.value,
            "failure_reason": self.failure_reason,
            "rounds": [value.to_document() for value in self.rounds],
            "accuracy": [value.to_document() for value in self.accuracy],
            "sandbox": None if self.sandbox is None else self.sandbox.to_document(),
            "attempts": [attempt.to_document() for attempt in self.attempts],
            "manifest_sha256": self.manifest_sha256,
            "signing_key_fingerprint": self.signing_key_fingerprint,
            "spec_fingerprint": self.spec_fingerprint,
            "surface_sha256": self.surface_sha256,
            "output_dir": self.output_dir,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "gpu_seconds": self.gpu_seconds,
            "profile_location": self.profile_location,
            "repro_verdict": self.repro_verdict,
            "nonfatal_warnings": list(self.nonfatal_warnings),
            "cold_start_seconds": self.cold_start_seconds,
            "table_hits": [value.to_document() for value in self.table_hits],
            "runtime_findings": list(self.runtime_findings),
            "request_classes": [value.to_document() for value in self.request_classes],
            "phases": [value.to_document() for value in self.phases],
            "unaccounted_seconds": self.unaccounted_seconds,
        }
        if self.short_prompt_consistency is not None:
            document["short_prompt_consistency"] = self.short_prompt_consistency.to_document()
        return document

    @classmethod
    def from_document(cls, document: object) -> TestResult:
        f = _mapping(document, "a test result")
        sandbox, failure, consistency = f.get("sandbox"), f.get("failure_class"), f.get("short_prompt_consistency")
        return cls(
            request_id=f.get("request_id"),
            validity=_enum(Validity, f.get("validity"), "validity"),
            failure_class=None if failure is None else _enum(FailureClass, failure, _FAILURE_CLASS),
            failure_reason=f.get("failure_reason"),
            rounds=tuple(map(RoundValue.from_document, _items(f.get("rounds", ()), "rounds"))),
            accuracy=tuple(map(AccuracyValue.from_document, _items(f.get("accuracy", ()), _ACCURACY))),
            sandbox=None if sandbox is None else SandboxFacts.from_document(sandbox),
            attempts=tuple(map(Attempt.from_document, _items(f.get("attempts", ()), "attempts"))),
            manifest_sha256=f.get("manifest_sha256"),
            signing_key_fingerprint=f.get("signing_key_fingerprint"),
            spec_fingerprint=f.get("spec_fingerprint"),
            surface_sha256=f.get("surface_sha256"),
            output_dir=f.get("output_dir", ""),
            queued_at=f.get("queued_at"),
            started_at=f.get("started_at"),
            finished_at=f.get("finished_at"),
            gpu_seconds=f.get("gpu_seconds", 0.0),
            profile_location=f.get("profile_location", ""),
            repro_verdict=f.get("repro_verdict", ""),
            nonfatal_warnings=tuple(_items(f.get("nonfatal_warnings", ()), "nonfatal_warnings")),
            cold_start_seconds=f.get("cold_start_seconds"),
            table_hits=tuple(map(TableHits.from_document, _items(f.get("table_hits", ()), "table_hits"))),
            runtime_findings=tuple(_items(f.get("runtime_findings", ()), "runtime_findings")),
            request_classes=tuple(
                map(RequestClasses.from_document, _items(f.get("request_classes", ()), "request_classes"))
            ),
            phases=tuple(map(PhaseTiming.from_document, _items(f.get("phases", ()), "phases"))),
            unaccounted_seconds=f.get("unaccounted_seconds"),
            short_prompt_consistency=None if consistency is None else ShortPromptConsistency.from_document(consistency),
        )


__all__ = [
    "CARD_CLEAN",
    "CARD_FOREIGN",
    "CARD_OCCUPANCY_STATES",
    "CARD_UNKNOWN",
    "NOT_REPRODUCED",
    "PRIORITY_HIGHEST",
    "PRIORITY_LOWEST",
    "REPRODUCED",
    "REPRO_VERDICTS",
    "TABLE_EXERCISED",
    "TABLE_NOT_EXERCISED",
    "TABLE_NOT_OBSERVABLE",
    "TABLE_STATUSES",
    "COMPARED_BY",
    "REFERENCE_ABSENT",
    "REFERENCE_COMPARED",
    "REFERENCE_STATUSES",
    "SHORT_PROMPTS_COMPLETE",
    "SHORT_PROMPTS_INCOMPLETE",
    "SHORT_PROMPTS_STATUSES",
    "AccuracyValue",
    "AttachmentRef",
    "Attempt",
    "FailureClass",
    "MAX_PHASE_SECONDS",
    "MeasurementError",
    "OutputCounts",
    "OutputDifference",
    "PhaseTiming",
    "Purpose",
    "RequestClasses",
    "RoundValue",
    "SandboxFacts",
    "ShortPromptConsistency",
    "ShortPromptReference",
    "TableHits",
    "TestKind",
    "TestRequest",
    "TestResult",
    "Validity",
]
