# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stand-ins for the objects other multi-branch modules define, with only the fields the notebook reads.

Each gives way to its owner's type once that is merged: the intent, the judgement and the branch log entry belong to
the specialist-evolution module, the test request and result to the toolbox. Fields beyond those the multi-branch
design lists (the judgement's recorded value and fix flags, a result's attempts, timing and measurement environment, a
branch log entry's superseded entry and hypothesis status) are what the notebook asks of those owners.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from hyperloom.contracts.recipe import RecipeDelta


class IntentLayer(str, Enum):
    """Which layer of the stack an intent changes."""

    CONFIG = "config"
    SOURCE = "source"
    KERNEL = "kernel"


@dataclass(frozen=True)
class Intent:
    """What a specialist means to do before it writes the change, on the state its branch head is at."""

    branch: str
    parent_commit: str
    target: str
    mechanism: str
    layer: IntentLayer
    fingerprint: str


def placeholder_intent_fingerprint(target: str, mechanism: str, layer: IntentLayer) -> str:
    """Until the intent's owner fixes the algorithm: the structured fields, case- and space-normalized, hashed."""

    normalized = "\x1f".join(" ".join(part.lower().split()) for part in (target, mechanism, layer.value))
    return hashlib.sha256(normalized.encode()).hexdigest()


class TestKind(str, Enum):
    """The kind of test a request asks the toolbox for."""

    __test__ = False

    TEST = "test"
    PROFILE = "profile"
    ACCURACY = "accuracy"
    REPRO = "repro"


@dataclass(frozen=True)
class TestRequest:
    """A test the toolbox accepted; the claim it measures is (``parent_commit``, ``intent_fingerprint``)."""

    __test__ = False

    request_id: str
    kind: TestKind
    branch: str
    parent_commit: str
    intent_fingerprint: str
    delta: RecipeDelta
    purpose: str = ""
    priority: str = ""
    diff_ref: str = ""
    retest_of: str | None = None
    avoid_nodes: tuple[str, ...] = ()
    accuracy_rounds: bool = False

    @property
    def action_fingerprint(self) -> str:
        return self.delta.fingerprint()


class Validity(str, Enum):
    """Whether a test measured the candidate; an infrastructure failure is retried, never returned."""

    SUCCESS = "success"
    CANDIDATE_FAILURE = "candidate_failure"


class FailureClass(str, Enum):
    INFRA = "infra"
    UNCLEAR = "unclear"
    CANDIDATE = "candidate"


class RoundKind(str, Enum):
    """The rounds one test runs on one inference server."""

    WARMUP = "warmup"
    FORMAL = "formal"
    ACCURACY = "accuracy"


@dataclass(frozen=True)
class SandboxFacts:
    """Where a test ran; ``backend`` tells local measurement on one host from hosted measurement."""

    node: str
    gpu_ids: tuple[int, ...] = ()
    gpu_model: str = ""
    power_cap_watts: float | None = None
    partition_mode: str = ""
    image_digest: str = ""
    backend: str = ""


@dataclass(frozen=True)
class Attempt:
    """One try at running a test; failed tries are kept for node health and cost, never as negative results."""

    node: str
    gpu_ids: tuple[int, ...]
    failure: FailureClass | None
    started_at: float
    finished_at: float


@dataclass(frozen=True)
class RoundValues:
    kind: RoundKind
    metric: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class TestResult:
    """The final outcome of one request: one row per request, its tries in ``attempts``."""

    __test__ = False

    request_id: str
    validity: Validity
    sandbox: SandboxFacts
    manifest_sha256: str
    rounds: tuple[RoundValues, ...] = ()
    accuracy: Mapping[str, float] = field(default_factory=dict)
    failure: FailureClass | None = None
    failure_reason: str = ""
    attempts: tuple[Attempt, ...] = ()
    queued_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    gpu_seconds: float = 0.0
    profile_location: str = ""


class Verdict(str, Enum):
    IMPROVED = "improved"
    FLAT = "flat"
    REGRESSED = "regressed"


@dataclass(frozen=True)
class Judgement:
    """The rule gate's (or, for a merge, the integrator's) comparison with the parent commit's recorded value.

    ``recorded_value`` is the candidate's value per metric anchored to the PRELUDE baseline: the mean of the test and
    its confirming retest. It becomes the commit's recorded value when the candidate is committed.
    """

    verdict: Verdict
    gain_pct: float
    threshold_pct: float
    confirmed: bool
    request_ids: tuple[str, ...]
    recorded_value: Mapping[str, float] = field(default_factory=dict)
    fixes_correctness: bool = False
    foundation_fix: bool = False


class BranchLogType(str, Enum):
    HYPOTHESIS = "hypothesis"
    ATTEMPT = "attempt"
    RESULT = "result"
    PLAN = "plan"


class HypothesisStatus(str, Enum):
    OPEN = "open"
    FALSIFIED = "falsified"
    CONFIRMED = "confirmed"


@dataclass(frozen=True)
class BranchLogEntry:
    """One entry of a specialist's long-term memory; ``supersedes`` names the entry it updates."""

    branch: str
    kind: BranchLogType
    content: str
    request_ids: tuple[str, ...] = ()
    supersedes: str | None = None
    hypothesis_status: HypothesisStatus | None = None


__all__ = [
    "Attempt",
    "BranchLogEntry",
    "BranchLogType",
    "FailureClass",
    "HypothesisStatus",
    "Intent",
    "IntentLayer",
    "Judgement",
    "RoundKind",
    "RoundValues",
    "SandboxFacts",
    "TestKind",
    "TestRequest",
    "TestResult",
    "Validity",
    "Verdict",
    "placeholder_intent_fingerprint",
]
