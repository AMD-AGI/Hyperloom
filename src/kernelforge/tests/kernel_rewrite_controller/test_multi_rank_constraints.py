# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a task must satisfy before it is allowed to ask for several ranks.

Both checks exist because the failure they prevent is expensive and quiet. A
rank count typed onto ordinary single-GPU work runs to the budget before the
measurement is found to describe nothing, and ranks that outnumber the visible
GPUs do not fail either -- they deadlock inside the collective and take the
budget with them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernelforge.kernel_rewrite_controller import dispatcher
from kernelforge.kernel_rewrite_controller._collective_names import (
    carries_parallelism_suffix,
    looks_like_multi_rank_operator,
)
from kernelforge.kernel_rewrite_controller.contracts import TASK_STATUS_SKIPPED, TaskContractError
from kernelforge.kernel_rewrite_controller.paths import ControllerLayout
from kernelforge.kernel_rewrite_controller.paths import operator_directory_name
from kernelforge.kernel_rewrite_controller.task import parse_task_payload
from kernelforge.knowledge.kernel_identity import KernelRecipeIdentity, kernel_recipe_canonical_id

BASE_COMMIT = "a" * 40


def _as_operator(payload: dict, name: str, world_size: int = 1) -> dict:
    return {
        **payload,
        "operator_name": name,
        "identity": {**payload["identity"], "kernel_name": name},
        "world_size": world_size,
    }


def _parse(task_dir: Path, payload: dict):
    """Parse without the directory-identity rule, which renaming would trip first."""
    return parse_task_payload(
        payload,
        task_dir=task_dir,
        expected_base_commit=BASE_COMMIT,
        enforce_directory_identity=False,
    )


def _published(tmp_path: Path, task_payload: dict, name: str, world_size: int) -> tuple[Path, ControllerLayout]:
    """A task directory named for its own identity, as the publisher writes it."""
    payload = _as_operator(task_payload, name, world_size)
    operator_id = kernel_recipe_canonical_id(KernelRecipeIdentity.from_mapping(payload["identity"]))
    layout = ControllerLayout(tmp_path / "out")
    root = layout.tasks_root / operator_directory_name(operator_id)
    root.mkdir(parents=True)
    (root / "driver.py").write_text("print('SNR: 100 dB')\n", encoding="utf-8")
    (root / "task.json").write_text(json.dumps(payload), encoding="utf-8")
    return root, layout


# --- the name check -----------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "custom_all_reduce_tp2",
        "customAllReduce",
        "two_shot_all_reduce",
        "moe_all_to_all_v",
        "reduce_scatter_gemm",
        "broadcast_weights",
        "ncclDevKernel_AllReduce",
        "_ZN5aiter18all_reduce_kernelE",
        # Named for its role rather than its collective, which is why the check
        # cannot be a list of textbook verbs.
        "mori::EpDispatchCombineOp::dispatch",
        # Abbreviations, which is how half of these are actually written.
        "custom_ar",
        "ag_gemm",
        "rs_gemm",
        "fused_ar_rmsnorm",
        # "algather" still carries "gather"; nothing here approximates.
        "algather_v2",
    ],
)
def test_a_name_that_reads_as_multi_rank_is_accepted(name: str) -> None:
    assert looks_like_multi_rank_operator(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "fused_moe",
        "gemm_a8w8_blockscale",
        "flash_attention_fwd",
        "silu_and_mul",
        "rmsnorm",
        "paged_attention_v2",
        # "reduce" alone is ordinary single-GPU work; only the compound verbs count.
        "layernorm_reduce",
        # The abbreviations only count as whole words.
        "arange_fill",
        "search_sorted",
        # A stated rank count says which shard, not that ranks compute the
        # answer; counting it here would make this test vacuous, because the
        # suffix is separately mandatory.
        "fused_moe_tp8",
    ],
)
def test_an_ordinary_single_gpu_name_is_not_mistaken_for_one(name: str) -> None:
    assert looks_like_multi_rank_operator(name) is False


def test_either_spelling_of_the_task_can_carry_the_evidence() -> None:
    """A task names itself twice and either may be the informative one."""
    assert looks_like_multi_rank_operator("fused_moe", "custom_all_reduce_tp8") is True
    assert looks_like_multi_rank_operator("", "") is False


def test_several_ranks_on_an_ordinary_operator_are_refused(task_dir: Path, task_payload: dict) -> None:
    # Carries the mandatory rank count, so only the collective test can refuse
    # it -- which is the one under test here.
    with pytest.raises(TaskContractError) as excinfo:
        _parse(task_dir, _as_operator(task_payload, "fused_moe_tp8", world_size=8))

    assert "world_size 8" in str(excinfo.value)
    assert "names one" in str(excinfo.value)


def test_several_ranks_on_a_collective_are_allowed(task_dir: Path, task_payload: dict) -> None:
    task = _parse(task_dir, _as_operator(task_payload, "custom_all_reduce_tp8", world_size=8))

    assert task.world_size == 8


def test_a_single_rank_task_is_never_asked_what_it_computes(task_dir: Path, task_payload: dict) -> None:
    """The check must not reach an operator that asked for nothing."""
    task = _parse(task_dir, _as_operator(task_payload, "fused_moe"))

    assert task.world_size == 1


# --- the GPU check ------------------------------------------------------------


def test_more_ranks_than_visible_gpus_is_skipped_before_the_repository(
    task_dir: Path,
    task_payload: dict,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Skipped, not failed: the task is sound and this host cannot run it."""
    published, layout = _published(tmp_path, task_payload, "custom_all_reduce_tp8", 8)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("the repository must not be borrowed for a task this host cannot run")

    monkeypatch.setattr(dispatcher, "create_operator_worktree", _unexpected)

    result = dispatcher.dispatch_single_task(published, layout=layout, deadline_unix=0.0)

    assert result.status == TASK_STATUS_SKIPPED
    assert "8 ranks" in result.reason
    assert "2 GPU" in result.reason


def test_enough_visible_gpus_reaches_the_worktree(
    task_dir: Path,
    task_payload: dict,
    tmp_path: Path,
    monkeypatch,
) -> None:
    published, layout = _published(tmp_path, task_payload, "custom_all_reduce_tp2", 2)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    reached: list[str] = []

    def _stop(*_args, **_kwargs):
        reached.append("worktree")
        raise RuntimeError("stop here")

    monkeypatch.setattr(dispatcher, "create_operator_worktree", _stop)

    dispatcher.dispatch_single_task(published, layout=layout, deadline_unix=0.0)

    assert reached == ["worktree"]


def test_a_count_nobody_can_answer_does_not_ground_the_task(monkeypatch) -> None:
    """Refusing on an unknown count would stop every task on such a host."""
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(dispatcher, "_visible_gpu_count", lambda: dispatcher.GPU_COUNT_UNKNOWN)

    task = type("_Task", (), {"world_size": 8})()

    assert dispatcher._insufficient_gpus(task) == ""


def test_no_gpu_at_all_is_an_answer_and_refuses(monkeypatch) -> None:
    """Zero and "could not tell" are different facts and must not share a value.

    Folding them together is what made the empty-mask answer unusable: the
    caller treated the zero as falsy and let a task through onto a dispatch
    that had told it there were no devices.
    """
    monkeypatch.setattr(dispatcher, "_visible_gpu_count", lambda: 0)

    task = type("_Task", (), {"world_size": 8})()

    assert "only 0 GPU(s)" in dispatcher._insufficient_gpus(task)


def test_a_single_rank_task_never_consults_the_gpu_count(monkeypatch) -> None:
    monkeypatch.setattr(
        dispatcher,
        "_visible_gpu_count",
        lambda: (_ for _ in ()).throw(AssertionError("must not be asked")),
    )

    assert dispatcher._insufficient_gpus(type("_Task", (), {"world_size": 1})()) == ""


def test_an_empty_mask_reports_zero_rather_than_unknown(monkeypatch) -> None:
    """No visible device is a real count, and must refuse rather than pass."""
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")

    assert dispatcher._visible_gpu_count() == 0
    assert dispatcher._visible_gpu_count() != dispatcher.GPU_COUNT_UNKNOWN


# --- the parallelism suffix ---------------------------------------------------


@pytest.mark.parametrize("name", ["custom_all_reduce_tp8", "moe_ep4_dispatch", "fused_moe_tp2", "arTp16"])
def test_a_name_stating_its_rank_count_is_accepted(name: str) -> None:
    assert carries_parallelism_suffix(name) is True


@pytest.mark.parametrize("name", ["custom_all_reduce", "all_gather", "fused_moe", "tphint", ""])
def test_a_name_without_a_rank_count_is_not(name: str) -> None:
    assert carries_parallelism_suffix(name) is False


def test_several_ranks_without_a_rank_count_in_the_name_are_refused(task_dir, task_payload) -> None:
    """The suffix keys the experience store, so it cannot stay advisory.

    world_size is deliberately outside the identity six-tuple. Left to the
    prompt, a TP-2 and a TP-8 all-reduce normalise to one operator_id: the
    scheduler keeps one task per id and drops the other, and both write the
    same recipe.
    """
    with pytest.raises(TaskContractError) as excinfo:
        _parse(task_dir, _as_operator(task_payload, "custom_all_reduce", world_size=8))

    assert "needs the rank count in the operator name" in str(excinfo.value)
    assert "_tp8" in str(excinfo.value)


def test_the_suffix_is_only_required_of_a_multi_rank_task(task_dir, task_payload) -> None:
    task = _parse(task_dir, _as_operator(task_payload, "fused_moe"))

    assert task.world_size == 1
