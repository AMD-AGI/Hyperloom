# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stand-ins for the objects other multi-branch modules define, with only the fields the notebook reads.

The test request and result give way to the toolbox's types once those are merged; fields beyond those the
multi-branch design lists (a result's attempts, timing and measurement environment) are what the notebook asks of the
toolbox. The intent, the judgement and the branch log entry are their owner's types, re-exported here for the
notebook modules that still import them from this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType, HypothesisStatus
from hyperloom_multibranch.contracts.intent import Intent, IntentLayer
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict


class TestKind(str, Enum):
    """The kind of test a request asks the toolbox for; final selection sweeps each candidate over the workload."""

    __test__ = False

    TEST = "test"
    PROFILE = "profile"
    ACCURACY = "accuracy"
    REPRO = "repro"
    SWEEP = "sweep"


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
]
