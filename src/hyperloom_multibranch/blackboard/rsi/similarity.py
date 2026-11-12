# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How much a past session says about this one: a weighted match of their task conditions.

Labels match or not; sizes (parallelism, concurrency, sequence lengths) fade with their log ratio, so 1024 against 2048
still counts two thirds. A condition either side lacks is left out rather than counted as a mismatch, except the
workload's kind (the ``workload`` label): an AgentX trace replay is graded on interactivity, a fixed-length InferenceX
workload on throughput, so the two share nothing whatever else matches. Conditions without the label are InferenceX, as
every session was before AgentX sessions ran. An AgentX session's conditions name its objective (the ``objective``
label); without it, it was graded on P50 interactivity, as every AgentX session was before objectives.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from hyperloom.common.agentx_objective import AgentXObjective
from hyperloom.inference_optimizer.recipe_snapshot_constants import RecipeIdentity, recipe_canonical_id
from hyperloom_multibranch.contracts.blackboard.records import SessionProfile
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.blackboard.knowledge.experience import experience_identity
from hyperloom_multibranch.blackboard.knowledge.warm_start import MODEL_LABEL, labels_of

#: The weight of a condition that changes little on its own: a framework release, expert parallelism.
MINOR_WEIGHT = 0.5
LABEL_WEIGHTS = {
    "gpu": 3.0,
    "framework": 2.0,
    "architecture": 2.0,
    "model_type": 1.0,
    "precision": 1.0,
    MODEL_LABEL: 1.0,
    "framework_version": MINOR_WEIGHT,
}
SIZE_WEIGHTS = {"tp": 1.0, "conc": 1.0, "isl": 1.0, "osl": 1.0, "ep": MINOR_WEIGHT}
WORKLOAD = "workload"
OBJECTIVE = "objective"
AGENTX = WorkloadKind.AGENTX.value
INFERENCEX = WorkloadKind.INFERENCEX.value
#: Below this, a past session shares too little with this one to teach it anything: one matching GPU is not enough.
MIN_SIMILARITY = 0.5
# A size eight times larger or smaller no longer counts as similar.
_LOG2_SPAN = 3.0


@dataclass(frozen=True)
class TaskConditions:
    """What a session optimizes, in the Experience KB identity vocabulary."""

    labels: Mapping[str, str] = field(default_factory=dict)
    sizes: Mapping[str, float] = field(default_factory=dict)


def task_conditions(profile: SessionProfile) -> TaskConditions:
    """The conditions of a session with ``profile``."""

    identity = experience_identity(profile)
    kind = AGENTX if profile.isl is None else INFERENCEX
    objective = {OBJECTIVE: profile.agentx_objective.value} if kind == AGENTX else {}
    return TaskConditions(
        labels={
            **{key: str(value) for key, value in identity.items() if key in LABEL_WEIGHTS},
            WORKLOAD: kind,
            **objective,
        },
        sizes={key: float(value) for key, value in identity.items() if key in SIZE_WEIGHTS and value is not None},
    )


def _closeness(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return float(left == right)
    return max(0.0, 1.0 - abs(math.log2(left / right)) / _LOG2_SPAN)


def model_name(model: str) -> str:
    """``model`` as the Recipe KB names it, so a local path, a hub id and the bare name of one model are one name."""

    labels = labels_of(
        recipe_canonical_id(
            RecipeIdentity(model=model, hardware="", framework_name="", framework_version="", precision="")
        )
    )
    return labels[MODEL_LABEL]


def _label(key: str, value: str) -> str:
    return model_name(value) if key == MODEL_LABEL else value.lower()


def workload_kind(task: TaskConditions) -> str:
    """``agentx`` for an AgentX trace replay, ``inferencex`` for a fixed-length InferenceX workload."""

    return task.labels.get(WORKLOAD, INFERENCEX)


def agentx_objective(task: TaskConditions) -> AgentXObjective:
    """What an AgentX session with ``task`` was graded on."""

    return AgentXObjective(task.labels.get(OBJECTIVE, AgentXObjective.P50_INTERACTIVITY.value))


def task_similarity(first: TaskConditions, second: TaskConditions) -> float:
    """Similarity in [0, 1]; 0 when the two share no condition, or are workloads of different kinds."""

    if workload_kind(first) != workload_kind(second):
        return 0.0
    total = matched = 0.0
    for key, weight in LABEL_WEIGHTS.items():
        if key in first.labels and key in second.labels:
            total += weight
            matched += weight * (_label(key, first.labels[key]) == _label(key, second.labels[key]))
    for key, weight in SIZE_WEIGHTS.items():
        if key in first.sizes and key in second.sizes:
            total += weight
            matched += weight * _closeness(first.sizes[key], second.sizes[key])
    return matched / total if total else 0.0


def differing_conditions(first: TaskConditions, second: TaskConditions) -> tuple[str, ...]:
    """The labels and the sizes both hold and do not share, labels first, as the similarity weighs them."""

    labels = [
        key
        for key in LABEL_WEIGHTS
        if key in first.labels
        and key in second.labels
        and _label(key, first.labels[key]) != _label(key, second.labels[key])
    ]
    sizes = [
        key
        for key in SIZE_WEIGHTS
        if key in first.sizes and key in second.sizes and first.sizes[key] != second.sizes[key]
    ]
    return (*labels, *sizes)


__all__ = [
    "AGENTX",
    "INFERENCEX",
    "LABEL_WEIGHTS",
    "MIN_SIMILARITY",
    "OBJECTIVE",
    "SIZE_WEIGHTS",
    "WORKLOAD",
    "TaskConditions",
    "agentx_objective",
    "differing_conditions",
    "model_name",
    "task_conditions",
    "task_similarity",
    "workload_kind",
]
