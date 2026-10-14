# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Task similarity weighs matching conditions and fades with size."""

from __future__ import annotations

import dataclasses

import pytest

from hyperloom_multibranch.notebook.rsi.similarity import (
    TaskConditions,
    effective_sessions,
    task_conditions,
    task_similarity,
)
from hyperloom_multibranch.notebook.rsi.tests.trees import TASK
from hyperloom_multibranch.notebook.tests.factories import profile


def test_similarity_weighs_matching_conditions_and_fades_with_size() -> None:
    other_gpu = dataclasses.replace(TASK, labels={**TASK.labels, "gpu": "MI300X"})
    longer = dataclasses.replace(TASK, sizes={**TASK.sizes, "isl": 2048.0})

    assert task_similarity(TASK, TASK) == pytest.approx(1.0)
    assert task_similarity(TASK, other_gpu) < task_similarity(TASK, longer) < 1.0
    assert task_similarity(TASK, longer) == pytest.approx(1.0 - (1.0 / 3.0) / 14.0)
    assert task_similarity(TASK, TaskConditions(labels={"region": "x"})) == 0.0
    assert task_similarity(TaskConditions(sizes={"tp": 0.0}), TaskConditions(sizes={"tp": 0.0})) == 1.0


def test_a_profile_gives_the_labels_and_sizes_of_its_conditions() -> None:
    conditions = task_conditions(profile(), {"gemm": 60.0})

    assert conditions.labels["gpu"] == "MI355X" and conditions.labels["architecture"] == "DeepseekV3ForCausalLM"
    assert conditions.sizes == {"tp": 8.0, "conc": 64.0, "isl": 1024.0, "osl": 1024.0}
    assert dict(conditions.profile) == {"gemm": 60.0}
    assert task_conditions(profile(isl=None, osl=None)).sizes == {"tp": 8.0, "conc": 64.0}


def test_effective_sessions_count_equally_similar_sessions() -> None:
    assert effective_sessions([1.0, 1.0]) == pytest.approx(2.0)
    assert effective_sessions([1.0, 0.0]) == pytest.approx(1.0)
    assert effective_sessions([]) == 0.0
