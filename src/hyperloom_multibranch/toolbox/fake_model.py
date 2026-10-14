# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the fake toolbox measures: the noise model, the failure draws and one attempt's outcome.

Nothing here runs a GPU. A measurement is ``baseline * (1 + effect) * (1 + node bias) * (1 + test noise)``:

* ``effect`` is what the test is configured to be worth (by delta fingerprint or a function of the request);
* the node bias is fixed per node for the whole run (two tests on one node share it, a retest on another node does not);
* the test noise is drawn per attempt.

The defaults of the bias follow the cross-node baseline spread measured on the fleet: the median absolute bias is
about 0.4 % and the 90th percentile about 3.3 %. Every draw comes from a generator seeded by the toolbox seed and the
thing drawn for (a node, or a request and its attempt number), so a run does not depend on the order tests ran in.
"""

from __future__ import annotations

import hashlib
import math
import random
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from hyperloom.contracts.measurement import (
    AccuracyValue,
    FailureClass,
    RoundValue,
    SandboxFacts,
    TestKind,
    TestRequest,
)

#: z-score of the 90th percentile of a normal distribution.
_Z90 = 1.2815515655446004

DEFAULT_BASELINE: Mapping[str, float] = {"output_tps": 1000.0, "ttft_ms": 200.0, "tpot_ms": 20.0}
DEFAULT_CAMPAIGN = uuid.UUID("00000000-0000-5000-8000-00000000f4ce")


class Fate(str, Enum):
    """How one attempt ends."""

    OK = "ok"
    INFRA = "infra"
    UNCLEAR = "unclear"
    CANDIDATE = "candidate"
    MEASUREMENT = "measurement"  # the spec was rejected or the result did not validate: never retried


#: The reason the fake gives for each failing fate; the candidate one is a signature of ``common/failure_signature``.
FAILURE_REASONS: Mapping[Fate, str] = {
    Fate.INFRA: "sandbox_create_timeout",
    Fate.UNCLEAR: "nonzero_exit_unmatched",
    Fate.CANDIDATE: "serve_flag",
    Fate.MEASUREMENT: "result_validation_failed",
}


@dataclass(frozen=True)
class NoiseModel:
    """Relative noise: a bias fixed per node, and a standard deviation drawn anew for each attempt."""

    node_bias_median: float = 0.004
    node_bias_p90: float = 0.033
    test_noise_sigma: float = 0.003

    def __post_init__(self) -> None:
        if not 0 <= self.node_bias_median <= self.node_bias_p90 or self.test_noise_sigma < 0:
            raise ValueError("noise needs 0 <= node_bias_median <= node_bias_p90 and test_noise_sigma >= 0")

    def node_bias(self, seed: int, node: str) -> float:
        """A signed bias whose magnitude is log-normal with the configured median and 90th percentile."""

        if self.node_bias_p90 == 0:
            return 0.0
        rng = _rng(seed, "node", node)
        sigma = math.log(self.node_bias_p90 / self.node_bias_median) / _Z90 if self.node_bias_median else 0.0
        magnitude = self.node_bias_median * math.exp(sigma * rng.gauss(0.0, 1.0))
        return magnitude if rng.random() < 0.5 else -magnitude

    def test_noise(self, rng: random.Random) -> float:
        return rng.gauss(0.0, self.test_noise_sigma) if self.test_noise_sigma else 0.0


@dataclass(frozen=True)
class FailureRatios:
    """The chance an attempt ends in each failure; the rest succeed. The four add up to at most one."""

    infra: float = 0.0
    unclear: float = 0.0
    candidate: float = 0.0
    measurement: float = 0.0

    def __post_init__(self) -> None:
        ratios = (self.infra, self.unclear, self.candidate, self.measurement)
        if any(not 0 <= ratio <= 1 for ratio in ratios) or sum(ratios) > 1:
            raise ValueError(f"failure ratios must lie in [0, 1] and add up to at most 1, got {ratios}")

    def draw(self, rng: random.Random) -> Fate:
        point = rng.random()
        for fate, ratio in (
            (Fate.INFRA, self.infra),
            (Fate.UNCLEAR, self.unclear),
            (Fate.CANDIDATE, self.candidate),
            (Fate.MEASUREMENT, self.measurement),
        ):
            if point < ratio:
                return fate
            point -= ratio
        return Fate.OK


@dataclass(frozen=True)
class FakeToolboxConfig:
    """Everything configurable about the fake; the defaults give a quiet, healthy fleet."""

    seed: int = 0
    campaign_id: uuid.UUID = DEFAULT_CAMPAIGN
    nodes: tuple[str, ...] = tuple(f"node-{index}" for index in range(8))
    gpu_model: str = "mi355x"
    gpu_count: int = 1
    image_digest: str = "sha256:" + "0" * 64
    baseline: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_BASELINE))
    main_concurrency: tuple[int, ...] = (64,)
    measure_rounds: int = 3
    noise: NoiseModel = field(default_factory=NoiseModel)
    failures: FailureRatios = field(default_factory=FailureRatios)
    #: What a delta is worth, as a fraction (0.02 is two percent more throughput), by ``delta.fingerprint()``.
    effects: Mapping[str, float] = field(default_factory=dict)
    #: Overrides ``effects`` when given; the request is the accepted one.
    effect_fn: Callable[[TestRequest], float] | None = None
    default_effect: float = 0.0
    accuracy_task: str = "gsm8k"
    accuracy_metric: str = "exact_match"
    accuracy_score: float = 0.93
    #: Accuracy by delta fingerprint, for a test that should fall below the floor.
    accuracy_scores: Mapping[str, float] = field(default_factory=dict)
    max_infra_retries: int = 3
    quarantine_nodes: bool = True
    branch_queue_limit: int = 8
    default_k: int = 2
    max_parallel_tests: int = 4
    #: Virtual seconds one attempt takes; it sets the timestamps and the GPU seconds, nothing sleeps.
    attempt_seconds: float = 600.0
    start_time: float = 1_800_000_000.0
    #: Real seconds an attempt waits before it finishes, to let tests overlap in a test of the queue.
    run_delay_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.nodes or self.gpu_count < 1 or self.measure_rounds < 1 or self.max_infra_retries < 0:
            raise ValueError("the fake needs nodes, a GPU, a measure round and a non-negative retry limit")
        if self.branch_queue_limit < 1 or self.default_k < 1 or self.max_parallel_tests < 1:
            raise ValueError("queue limit, default K and parallel tests must be at least 1")
        if not self.main_concurrency or any(value < 1 for value in self.main_concurrency):
            raise ValueError("the main operating point needs positive concurrency values")

    def effect_of(self, request: TestRequest) -> float:
        if self.effect_fn is not None:
            return self.effect_fn(request)
        return self.effects.get(request.delta.fingerprint(), self.default_effect)


@dataclass(frozen=True)
class AttemptOutcome:
    """One attempt as the measurer saw it; the toolbox turns a run of them into one result."""

    fate: Fate
    sandbox: SandboxFacts | None
    rounds: tuple[RoundValue, ...] = ()
    accuracy: tuple[AccuracyValue, ...] = ()

    @property
    def failure_class(self) -> FailureClass | None:
        return {
            Fate.OK: None,
            Fate.INFRA: FailureClass.INFRA,
            Fate.UNCLEAR: FailureClass.UNCLEAR,
            Fate.CANDIDATE: FailureClass.CANDIDATE,
            Fate.MEASUREMENT: FailureClass.INFRA,
        }[self.fate]

    @property
    def failure_reason(self) -> str | None:
        return FAILURE_REASONS.get(self.fate)


def _rng(seed: int, *parts: object) -> random.Random:
    return random.Random("/".join(str(part) for part in (seed, *parts)))


class FakeMeasurer:
    """Draws the fate of an attempt, picks its node and produces its values."""

    def __init__(self, config: FakeToolboxConfig) -> None:
        self.config = config

    def run(
        self,
        request: TestRequest,
        request_id: str,
        attempt: int,
        *,
        avoid: frozenset[str],
        unavailable: frozenset[str],
        forced: Fate | None = None,
    ) -> AttemptOutcome:
        """One attempt on a fresh sandbox: ``avoid`` is best effort, ``unavailable`` (quarantined nodes) is not."""

        rng = _rng(self.config.seed, request_id, attempt)
        fate = forced if forced is not None else self.config.failures.draw(rng)
        sandbox = self._sandbox(request_id, attempt, rng, avoid, unavailable)
        if sandbox is None:
            return AttemptOutcome(Fate.INFRA, None)
        if fate is not Fate.OK:
            return AttemptOutcome(fate, sandbox)
        factor = self._factor(request, sandbox, rng)
        return AttemptOutcome(Fate.OK, sandbox, self._rounds(request, factor, rng), self._accuracy(request, rng))

    def _sandbox(
        self, request_id: str, attempt: int, rng: random.Random, avoid: frozenset[str], unavailable: frozenset[str]
    ) -> SandboxFacts | None:
        usable = sorted(set(self.config.nodes) - unavailable)
        if not usable:
            return None
        fresh = [node for node in usable if node not in avoid]
        node = rng.choice(fresh or usable)
        digest = hashlib.sha256(f"{request_id}/{attempt}".encode()).hexdigest()[:12]
        return SandboxFacts(
            sandbox_id=f"sb-{digest}",
            node_name=node,
            gpu_model=self.config.gpu_model,
            gpu_count=self.config.gpu_count,
            image_digest=self.config.image_digest,
            same_node=node in avoid,
            gpu_ids=tuple(range(self.config.gpu_count)),
            backend="fake",
        )

    def _factor(self, request: TestRequest, sandbox: SandboxFacts, rng: random.Random) -> float:
        bias = self.config.noise.node_bias(self.config.seed, sandbox.node_name or "")
        noise = self.config.noise.test_noise(rng)
        return (1 + self.config.effect_of(request)) * (1 + bias) * (1 + noise)

    def _rounds(self, request: TestRequest, factor: float, rng: random.Random) -> tuple[RoundValue, ...]:
        if request.kind is TestKind.ACCURACY:
            return ()
        rounds: list[RoundValue] = []
        for concurrency in request.concurrency or self.config.main_concurrency:
            rounds.append(self._round("warmup", concurrency, factor * 0.8, rng))
            rounds.extend(self._round("measure", concurrency, factor, rng) for _ in range(self.config.measure_rounds))
        return tuple(rounds)

    def _round(
        self, phase: Literal["warmup", "measure"], concurrency: int, factor: float, rng: random.Random
    ) -> RoundValue:
        """One round sits a little off the attempt's level: a third of the test noise, drawn per round."""

        factor *= 1 + self.config.noise.test_noise(rng) / 3
        metrics = {
            name: value * factor if name.endswith("_tps") else value / factor
            for name, value in self.config.baseline.items()
        }
        return RoundValue(phase, concurrency, metrics, completed_requests=1000, failed_requests=0)

    def _accuracy(self, request: TestRequest, rng: random.Random) -> tuple[AccuracyValue, ...]:
        if not request.accuracy:
            return ()
        score = self.config.accuracy_scores.get(request.delta.fingerprint(), self.config.accuracy_score)
        return (AccuracyValue(self.config.accuracy_task, self.config.accuracy_metric, score + rng.gauss(0, 0.001)),)
