# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from kernelforge.kernel_rewrite_controller import (
    ControllerLayout,
    TaskStateStore,
    discover_task_dirs,
    load_task,
    parse_task_payload,
    sort_tasks,
)

BASE_COMMIT = "a" * 40


def _write_payload(task_dir: Path, payload: dict) -> None:
    (task_dir / "task.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_task_parses_identity_and_initializes_ready_state(
    task_dir: Path,
    operator_id: str,
) -> None:
    result = load_task(task_dir, expected_base_commit=BASE_COMMIT)

    assert result.ok is True
    assert result.task is not None
    assert result.task.operator_id == operator_id
    assert result.task.kernel_path == "sglang/kernels/fused_moe.py"
    assert result.task.driver_path == "driver.py"
    state = TaskStateStore(task_dir).load()
    assert state is not None
    assert state.status == "ready"


@pytest.mark.parametrize(
    "field",
    [
        "identity",
        "base_commit",
        "repo_root",
        "kernel_path",
        "operator_name",
        "driver_path",
        "priority",
    ],
)
def test_missing_required_field_skips_only_that_task(
    task_dir: Path,
    task_payload: dict,
    field: str,
) -> None:
    task_payload.pop(field)
    _write_payload(task_dir, task_payload)

    result = load_task(task_dir, expected_base_commit=BASE_COMMIT)

    assert result.ok is False
    assert field in result.reason
    state = TaskStateStore(task_dir).load()
    assert state is not None
    assert state.status == "skipped"
    assert field in state.reason


def test_invalid_json_is_recorded_as_skipped(task_dir: Path) -> None:
    (task_dir / "task.json").write_text("{", encoding="utf-8")

    result = load_task(task_dir, expected_base_commit=BASE_COMMIT)

    assert result.ok is False
    assert TaskStateStore(task_dir).load().status == "skipped"  # type: ignore[union-attr]


def test_an_unknown_task_field_is_ignored_rather_than_refused(
    task_dir: Path,
    task_payload: dict,
) -> None:
    """Every field the run acts on is required, so an extra one decides nothing."""
    task_payload["typo_field"] = True

    task = parse_task_payload(task_payload, task_dir=task_dir)

    assert task.kernel_path == "sglang/kernels/fused_moe.py"


def test_base_commit_mismatch_is_recorded_as_skipped(task_dir: Path) -> None:
    result = load_task(task_dir, expected_base_commit="b" * 40)

    assert result.ok is False
    assert "base_commit mismatch" in result.reason


def test_repo_root_must_be_an_existing_absolute_directory(
    task_dir: Path,
    task_payload: dict,
) -> None:
    task_payload["repo_root"] = "relative/repo"

    with pytest.raises(ValueError, match="repo_root must be an absolute path"):
        parse_task_payload(task_payload, task_dir=task_dir)


@pytest.mark.parametrize("kernel_path", ["/tmp/kernel.py", "../kernel.py", "a\\kernel.py"])
def test_unsafe_kernel_path_is_rejected(
    task_dir: Path,
    task_payload: dict,
    kernel_path: str,
) -> None:
    task_payload["kernel_path"] = kernel_path

    with pytest.raises(ValueError, match="kernel_path"):
        parse_task_payload(task_payload, task_dir=task_dir)


def test_driver_must_exist_inside_task_directory(task_dir: Path, task_payload: dict) -> None:
    (task_dir / "driver.py").unlink()

    with pytest.raises(ValueError, match="driver_path is not a file"):
        parse_task_payload(task_payload, task_dir=task_dir)


def test_kernel_name_is_derived_from_the_operator_name(task_dir: Path, task_payload: dict) -> None:
    """One name in, one address out.

    forge-loop resolves its knowledge-base page from ``--operator-name`` alone,
    so the dimension has to come from the same string the loop will see.
    """
    task = parse_task_payload(task_payload, task_dir=task_dir, enforce_directory_identity=False)

    assert task.operator_name == "backend::Fused.MoE-Kernel"
    assert task.identity.kernel_name == "fused_moe"


def test_an_agent_supplied_kernel_name_is_replaced_rather_than_refused(
    task_dir: Path,
    task_payload: dict,
) -> None:
    """The agent hears no refusal, so a stale draft is corrected, not thrown out."""
    task_payload["identity"] = {**task_payload["identity"], "kernel_name": "paged_attention"}

    task = parse_task_payload(task_payload, task_dir=task_dir, enforce_directory_identity=False)

    assert task.identity.kernel_name == "fused_moe"


def test_an_operator_name_that_normalizes_to_nothing_is_refused(
    task_dir: Path,
    task_payload: dict,
) -> None:
    """Every unnameable operator would otherwise share one canonical id."""
    task_payload["operator_name"] = "<<>>"

    with pytest.raises(ValueError, match="no usable kernel name"):
        parse_task_payload(task_payload, task_dir=task_dir, enforce_directory_identity=False)


def test_an_unregistered_backend_is_accepted(task_dir: Path, task_payload: dict) -> None:
    """It picks a prompt layer, and forge-loop answers an unknown one with none."""
    task_payload["identity"]["backend"] = "tilelang"

    task = parse_task_payload(task_payload, task_dir=task_dir, enforce_directory_identity=False)

    assert task.identity.backend == "tilelang"
    assert task.operator_id.split(":")[5] == "tilelang"


def test_bool_priority_is_rejected(task_dir: Path, task_payload: dict) -> None:
    task_payload["priority"] = True

    with pytest.raises(ValueError, match="priority"):
        parse_task_payload(task_payload, task_dir=task_dir)


def test_discover_task_dirs_ignores_temporary_and_incomplete_directories(
    tmp_path: Path,
    task_dir: Path,
) -> None:
    layout = ControllerLayout(tmp_path / "output")
    temporary = layout.tasks_root / ".pending"
    temporary.mkdir()
    (temporary / "task.json").write_text("{}", encoding="utf-8")
    (temporary / "driver.py").write_text("", encoding="utf-8")
    incomplete = layout.tasks_root / "incomplete"
    incomplete.mkdir()
    (incomplete / "task.json").write_text("{}", encoding="utf-8")

    assert discover_task_dirs(layout) == [task_dir.resolve()]


def test_sort_tasks_uses_priority_then_identity_and_deduplicates(task_dir: Path) -> None:
    task = load_task(task_dir, record_state=False).task
    assert task is not None
    lower_priority_duplicate = replace(task, priority=9)
    other = replace(task, operator_id=f"{task.operator_id}-other", priority=0)

    assert sort_tasks([lower_priority_duplicate, task, other]) == [other, task]


_DROP = object()


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"identity": []}, "identity must be a JSON object"),
        ({"base_commit": "abc"}, "full 40- or 64-character hexadecimal"),
        ({"driver_path": "run.py"}, "driver_path must be exactly 'driver.py'"),
        ({"operator_name": ""}, "operator_name must be a non-empty string"),
        ({"repo_root": ""}, "repo_root must be a non-empty string"),
        ({"evidence": {}}, "evidence must be a JSON list"),
        ({"reason": 7}, "reason must be a string"),
        ({"source_files": [""]}, "source_files must be a list of non-empty strings"),
    ],
)
def test_every_task_rule_refuses_its_own_violation(
    task_dir: Path,
    task_payload: dict,
    changes: dict,
    expected: str,
) -> None:
    """The parser is the only thing between an Agent's draft and a git worktree."""
    _write_payload(task_dir, {**task_payload, **changes})
    outcome = load_task(task_dir, record_state=False)
    assert outcome.task is None
    assert expected in (outcome.reason or "")


def test_an_identity_missing_a_dimension_is_refused(task_dir: Path, task_payload: dict) -> None:
    identity = dict(task_payload["identity"])
    identity.pop("backend")
    _write_payload(task_dir, {**task_payload, "identity": identity})
    outcome = load_task(task_dir, record_state=False)
    assert outcome.task is None
    assert "identity is missing fields: backend" in (outcome.reason or "")


def test_an_identity_with_an_extra_dimension_keeps_the_six_tuple(
    task_dir: Path,
    task_payload: dict,
    operator_id: str,
) -> None:
    """A seventh field names no dimension, so it cannot move the address."""
    identity = {**task_payload["identity"], "vendor": "amd"}
    _write_payload(task_dir, {**task_payload, "identity": identity})

    outcome = load_task(task_dir, record_state=False)

    assert outcome.task is not None
    assert outcome.task.operator_id == operator_id


@pytest.mark.parametrize("value", [15.3, "15.3%", None, {"decode": 15.3}])
def test_gpu_pct_is_carried_in_any_shape_the_agent_wrote(
    task_dir: Path,
    task_payload: dict,
    value: object,
) -> None:
    """Observational only: refusing over its formatting would cost an operator."""
    task_payload["gpu_pct"] = value

    task = parse_task_payload(task_payload, task_dir=task_dir)

    assert task.gpu_pct == value


def test_a_task_without_gpu_pct_is_still_valid(task_dir: Path, task_payload: dict) -> None:
    task = parse_task_payload(task_payload, task_dir=task_dir)

    assert task.gpu_pct is None


def test_shape_cases_are_carried_verbatim(task_dir: Path, task_payload: dict) -> None:
    """No code reads a case, so a shape the contract disagrees with still ships."""
    task_payload["shape_cases"] = [1, {"name": "decode"}]

    task = parse_task_payload(task_payload, task_dir=task_dir)

    assert task.shape_cases == [1, {"name": "decode"}]


def test_shape_cases_that_are_not_a_list_are_not_reshaped(task_dir: Path, task_payload: dict) -> None:
    """Wrapping a value nothing reads decides as little as refusing it would."""
    task_payload["shape_cases"] = {"name": "decode"}

    task = parse_task_payload(task_payload, task_dir=task_dir)

    assert task.shape_cases == {"name": "decode"}


def test_a_producer_other_than_forge_loop_is_refused(task_dir: Path, task_payload: dict) -> None:
    identity = {**task_payload["identity"], "producer": "fusion"}
    _write_payload(task_dir, {**task_payload, "identity": identity})
    outcome = load_task(task_dir, record_state=False)
    assert outcome.task is None
    assert "identity.producer" in (outcome.reason or "")


def test_a_task_json_that_is_not_an_object_is_refused(task_dir: Path) -> None:
    (task_dir / "task.json").write_text("[]", encoding="utf-8")
    outcome = load_task(task_dir, record_state=False)
    assert outcome.task is None
    assert "must contain a JSON object" in (outcome.reason or "")


def test_a_repo_root_that_is_not_a_directory_is_refused(
    task_dir: Path,
    task_payload: dict,
    tmp_path: Path,
) -> None:
    absent = tmp_path / "not-a-repo"
    _write_payload(task_dir, {**task_payload, "repo_root": str(absent)})
    outcome = load_task(task_dir, record_state=False)
    assert outcome.task is None
    assert "repo_root is not a directory" in (outcome.reason or "")


def test_a_directory_name_that_disagrees_with_the_identity_is_refused(
    tmp_path: Path,
    task_payload: dict,
) -> None:
    """The directory name is the published identity; a mismatch is a forged task."""
    wrong = tmp_path / "tasks" / "some-other-operator"
    wrong.mkdir(parents=True)
    (wrong / "driver.py").write_text("print('driver')\n", encoding="utf-8")
    _write_payload(wrong, task_payload)
    outcome = load_task(wrong, record_state=False)
    assert outcome.task is None
    assert "does not match canonical operator id" in (outcome.reason or "")


def test_world_size_defaults_to_one(task_dir: Path) -> None:
    result = load_task(task_dir, expected_base_commit=BASE_COMMIT)
    assert result.ok is True
    assert result.task is not None
    assert result.task.world_size == 1


def test_world_size_is_parsed_when_present(task_dir: Path, task_payload: dict) -> None:
    # Named for the collective it performs: a rank count is only accepted on an
    # operator that reads as one (see test_multi_rank_constraints).
    collective = "custom_all_reduce_tp8"
    task = parse_task_payload(
        {
            **task_payload,
            "operator_name": collective,
            "identity": {**task_payload["identity"], "kernel_name": collective},
            "world_size": 8,
        },
        task_dir=task_dir,
        expected_base_commit=BASE_COMMIT,
        enforce_directory_identity=False,
    )
    assert task.world_size == 8


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4"])
def test_invalid_world_size_skips_task(task_dir: Path, task_payload: dict, value) -> None:
    _write_payload(task_dir, {**task_payload, "world_size": value})
    result = load_task(task_dir, expected_base_commit=BASE_COMMIT)
    assert result.ok is False
    assert "world_size" in (result.reason or "")
