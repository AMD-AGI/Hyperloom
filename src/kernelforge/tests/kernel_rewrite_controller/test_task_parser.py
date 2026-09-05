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
        "schema_version",
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


def test_unknown_task_field_is_rejected(task_dir: Path, task_payload: dict) -> None:
    task_payload["typo_field"] = True

    with pytest.raises(ValueError, match="unknown fields"):
        parse_task_payload(task_payload, task_dir=task_dir)


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


def test_operator_name_must_match_identity(task_dir: Path, task_payload: dict) -> None:
    task_payload["operator_name"] = "paged_attention"

    with pytest.raises(ValueError, match="does not normalize"):
        parse_task_payload(task_payload, task_dir=task_dir)


def test_identity_backend_must_be_registered(task_dir: Path, task_payload: dict) -> None:
    task_payload["identity"]["backend"] = "rocm"

    with pytest.raises(ValueError, match="registered kernel backends"):
        parse_task_payload(task_payload, task_dir=task_dir)


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
        ({"schema_version": 99}, "unsupported task schema"),
        ({"identity": []}, "identity must be a JSON object"),
        ({"base_commit": "abc"}, "full 40- or 64-character hexadecimal"),
        ({"driver_path": "run.py"}, "driver_path must be exactly 'driver.py'"),
        ({"operator_name": ""}, "operator_name must be a non-empty string"),
        ({"repo_root": ""}, "repo_root must be a non-empty string"),
        ({"shape_cases": [1]}, "shape_cases must be a list of JSON objects"),
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


def test_an_identity_with_an_extra_dimension_is_refused(task_dir: Path, task_payload: dict) -> None:
    """The six-tuple is closed: a seventh field would silently change the id."""
    identity = {**task_payload["identity"], "vendor": "amd"}
    _write_payload(task_dir, {**task_payload, "identity": identity})
    outcome = load_task(task_dir, record_state=False)
    assert outcome.task is None
    assert "identity has unknown fields: vendor" in (outcome.reason or "")


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
