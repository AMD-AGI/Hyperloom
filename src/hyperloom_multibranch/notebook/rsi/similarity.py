# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How much a past session says about this one: a weighted match of their task conditions.

Labels match or not; sizes (parallelism, concurrency, sequence lengths) fade with their log ratio, so 1024 against 2048
still counts two thirds; bottleneck profiles compare by cosine. A condition either side lacks is left out rather than
counted as a mismatch.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from hyperloom_multibranch.contracts.notebook.records import SessionProfile
from hyperloom_multibranch.notebook.knowledge.experience import experience_identity

#: The weight of a condition that changes little on its own: a framework release, expert parallelism.
MINOR_WEIGHT = 0.5
LABEL_WEIGHTS = {
    "gpu": 3.0,
    "framework": 2.0,
    "architecture": 2.0,
    "model_type": 1.0,
    "precision": 1.0,
    "model": 1.0,
    "framework_version": MINOR_WEIGHT,
}
SIZE_WEIGHTS = {"tp": 1.0, "conc": 1.0, "isl": 1.0, "osl": 1.0, "ep": MINOR_WEIGHT}
PROFILE_WEIGHT = 2.0
#: Below this, a past session shares too little with this one to teach it anything: one matching GPU is not enough.
MIN_SIMILARITY = 0.5
# A size eight times larger or smaller no longer counts as similar.
_LOG2_SPAN = 3.0


@dataclass(frozen=True)
class TaskConditions:
    """What a session optimizes, in the Experience KB identity vocabulary.

    ``profile`` is the baseline's GPU-time share per operator category, in percent, when PRELUDE profiled it.
    """

    labels: Mapping[str, str] = field(default_factory=dict)
    sizes: Mapping[str, float] = field(default_factory=dict)
    profile: Mapping[str, float] = field(default_factory=dict)


def task_conditions(profile: SessionProfile, bottlenecks: Mapping[str, float] | None = None) -> TaskConditions:
    """The conditions of a session with ``profile``."""

    identity = experience_identity(profile)
    return TaskConditions(
        labels={key: str(value) for key, value in identity.items() if key in LABEL_WEIGHTS},
        sizes={key: float(value) for key, value in identity.items() if key in SIZE_WEIGHTS and value is not None},
        profile=dict(bottlenecks or {}),
    )


def _closeness(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return float(left == right)
    return max(0.0, 1.0 - abs(math.log2(left / right)) / _LOG2_SPAN)


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    norm = math.sqrt(sum(value * value for value in left.values())) * math.sqrt(
        sum(value * value for value in right.values())
    )
    return dot / norm if norm else 0.0


def task_similarity(first: TaskConditions, second: TaskConditions) -> float:
    """Similarity in [0, 1]; 0 when the two share no condition."""

    total = matched = 0.0
    for key, weight in LABEL_WEIGHTS.items():
        if key in first.labels and key in second.labels:
            total += weight
            matched += weight * (first.labels[key].lower() == second.labels[key].lower())
    for key, weight in SIZE_WEIGHTS.items():
        if key in first.sizes and key in second.sizes:
            total += weight
            matched += weight * _closeness(first.sizes[key], second.sizes[key])
    if first.profile and second.profile:
        total += PROFILE_WEIGHT
        matched += PROFILE_WEIGHT * _cosine(first.profile, second.profile)
    return matched / total if total else 0.0


def effective_sessions(weights: Iterable[float]) -> float:
    """How many equally similar sessions ``weights`` are worth: the square of their sum over their sum of squares."""

    values = [weight for weight in weights if weight > 0]
    squares = sum(weight * weight for weight in values)
    return sum(values) ** 2 / squares if squares else 0.0


__all__ = [
    "LABEL_WEIGHTS",
    "MIN_SIMILARITY",
    "PROFILE_WEIGHT",
    "SIZE_WEIGHTS",
    "TaskConditions",
    "effective_sessions",
    "task_conditions",
    "task_similarity",
]
