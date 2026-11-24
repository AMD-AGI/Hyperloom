# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Task similarity weighs matching conditions and fades with size."""

from __future__ import annotations

import dataclasses

import pytest

from hyperloom_multibranch.blackboard.rsi.similarity import (
    AGENTX,
    INFERENCEX,
    TaskConditions,
    differing_conditions,
    prompt_dataset_of,
    task_conditions,
    task_similarity,
    workload_kind,
)
from hyperloom_multibranch.blackboard.rsi.tests.trees import TASK
from hyperloom_multibranch.blackboard.tests.factories import profile


def test_similarity_weighs_matching_conditions_and_fades_with_size() -> None:
    other_gpu = dataclasses.replace(TASK, labels={**TASK.labels, "gpu": "MI300X"})
    longer = dataclasses.replace(TASK, sizes={**TASK.sizes, "isl": 2048.0})

    assert task_similarity(TASK, TASK) == pytest.approx(1.0)
    assert task_similarity(TASK, other_gpu) < task_similarity(TASK, longer) < 1.0
    assert task_similarity(TASK, longer) == pytest.approx(1.0 - (1.0 / 3.0) / 12.0)
    assert task_similarity(TASK, TaskConditions(labels={"region": "x"})) == 0.0
    assert task_similarity(TaskConditions(sizes={"tp": 0.0}), TaskConditions(sizes={"tp": 0.0})) == 1.0


def test_one_model_under_a_local_path_a_hub_id_or_a_bare_name_is_one_model() -> None:
    def named(model: str) -> TaskConditions:
        return dataclasses.replace(TASK, labels={**TASK.labels, "model": model})

    local, bare, other = named("/shared/models/DeepSeek-R1"), named("deepseek-r1"), named("/shared/models/Qwen3-8B")

    assert task_similarity(TASK, local) == pytest.approx(1.0)
    assert task_similarity(local, bare) == pytest.approx(1.0)
    assert task_similarity(TASK, other) < 1.0


def test_a_profile_gives_the_labels_and_sizes_of_its_conditions() -> None:
    conditions = task_conditions(profile())

    assert conditions.labels["gpu"] == "MI355X" and conditions.labels["architecture"] == "DeepseekV3ForCausalLM"
    assert conditions.sizes == {"tp": 8.0, "conc": 64.0, "isl": 1024.0, "osl": 1024.0}
    assert task_conditions(profile(isl=None, osl=None)).sizes == {"tp": 8.0, "conc": 64.0}


def test_a_trace_replay_and_a_fixed_length_workload_share_nothing_however_alike_otherwise() -> None:
    replay = task_conditions(profile(isl=None, osl=None))
    fixed = task_conditions(profile())

    assert (workload_kind(replay), workload_kind(fixed), workload_kind(TASK)) == (AGENTX, INFERENCEX, INFERENCEX)
    assert task_similarity(fixed, replay) == task_similarity(replay, fixed) == 0.0
    assert task_similarity(replay, replay) == pytest.approx(1.0)


def test_a_session_on_a_prompt_dataset_is_a_lead_to_one_on_random_prompts_and_names_what_differs() -> None:
    random_prompts = task_conditions(profile())
    on_a_dataset = task_conditions(profile(dataset="/data/prompts.jsonl"))
    on_another = task_conditions(profile(dataset="/data/other.jsonl"))
    longer = task_conditions(profile(isl=2048))

    assert random_prompts != on_a_dataset and "dataset" not in random_prompts.labels
    assert (prompt_dataset_of(on_a_dataset), prompt_dataset_of(random_prompts)) == ("/data/prompts.jsonl", "")
    for first, second in ((on_a_dataset, random_prompts), (random_prompts, on_a_dataset), (on_a_dataset, on_another)):
        assert task_similarity(first, second) == pytest.approx(14.5 / 15.5)
        assert differing_conditions(first, second) == ("dataset",)
    assert task_similarity(on_a_dataset, on_a_dataset) == task_similarity(random_prompts, random_prompts) == 1.0
    assert differing_conditions(on_a_dataset, on_a_dataset) == ()
    assert differing_conditions(on_a_dataset, longer) == ("dataset", "isl")


def test_conditions_archived_before_the_prompt_dataset_was_recorded_read_as_random_prompts() -> None:
    assert "dataset" not in TASK.labels and prompt_dataset_of(TASK) == ""
    on_a_dataset = dataclasses.replace(TASK, labels={**TASK.labels, "dataset": "/data/prompts.jsonl"})

    assert task_similarity(TASK, on_a_dataset) < 1.0
    assert differing_conditions(on_a_dataset, TASK) == ("dataset",)
