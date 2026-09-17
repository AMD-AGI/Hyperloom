# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the shared lm-eval task-selection helpers."""

from __future__ import annotations

import pytest

from hyperloom.common.eval_tasks import (
    ACCURACY_METRIC_KEYS,
    DEFAULT_EVAL_TASKS,
    TINYBENCHMARKS_PINNED_REF,
    TINYBENCHMARKS_PINNED_SPECS,
    eval_tasks_need_tinybenchmarks,
    split_eval_tasks,
)


def test_the_default_task_is_full_gsm8k():
    """Switching the gate's task is opt-in; nothing in the tree may quietly change what a verdict means."""
    assert DEFAULT_EVAL_TASKS == "gsm8k"


@pytest.mark.parametrize(
    "tasks",
    ["tinyGSM8k", "tinyMMLU", "tinyBenchmarks", "gsm8k,tinyGSM8k", " tinygsm8k ", "hellaswag , tinyArc"],
)
def test_tiny_tasks_require_the_estimator(tasks):
    assert eval_tasks_need_tinybenchmarks(tasks) is True


@pytest.mark.parametrize("tasks", ["gsm8k", "", None, "hellaswag,mmlu", "gsm8k_cot"])
def test_non_tiny_tasks_take_on_no_extra_dependency(tasks):
    assert eval_tasks_need_tinybenchmarks(tasks) is False


def test_split_drops_blanks_and_whitespace():
    assert split_eval_tasks(" gsm8k , ,tinyGSM8k ") == ("gsm8k", "tinyGSM8k")


def test_acc_norm_is_read_but_never_preferred_over_acc():
    """A task reporting both means plain ``acc`` as its headline number."""
    assert "acc_norm,none" in ACCURACY_METRIC_KEYS
    assert ACCURACY_METRIC_KEYS.index("acc,none") < ACCURACY_METRIC_KEYS.index("acc_norm,none")


def test_strict_match_leads_the_metric_order():
    assert ACCURACY_METRIC_KEYS[0] == "exact_match,strict-match"


def test_both_tinybenchmarks_specs_are_pinned_to_the_same_commit():
    """Neither install path may drift from the other, and neither may float on a branch."""
    kinds = [kind for kind, _spec in TINYBENCHMARKS_PINNED_SPECS]
    assert kinds == ["git", "archive"]  # git first; the archive is the no-git fallback
    assert all(TINYBENCHMARKS_PINNED_REF in spec for _kind, spec in TINYBENCHMARKS_PINNED_SPECS)
