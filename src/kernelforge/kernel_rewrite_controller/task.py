# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse and validate one operator rewrite task."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from kernelforge.kernel_rewrite_controller.contracts import (
    TASK_SCHEMA_VERSION,
    KernelRewriteTask,
    TaskContractError,
    TaskParseResult,
)
from kernelforge.kernel_rewrite_controller.paths import (
    ControllerLayout,
    TaskLayout,
    safe_relative_path,
)
from kernelforge.knowledge.implementation_identity import normalize_operator_name
from kernelforge.knowledge.kernel_identity import (
    KERNEL_CANONICAL_DIMENSIONS,
    KernelRecipeIdentity,
    kernel_recipe_canonical_id,
)
from kernelforge.knowledge.loop_identity import LOOP_PRODUCER

_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_REQUIRED_TASK_FIELDS = frozenset(
    {
        "schema_version",
        "identity",
        "base_commit",
        "repo_root",
        "kernel_path",
        "operator_name",
        "driver_path",
        "priority",
    }
)
_OPTIONAL_TASK_FIELDS = frozenset(
    {
        "source_files",
        "target_functions",
        "shape_cases",
        "reason",
        "evidence",
    }
)
_TASK_FIELDS = _REQUIRED_TASK_FIELDS | _OPTIONAL_TASK_FIELDS
_IDENTITY_FIELDS = frozenset(KERNEL_CANONICAL_DIMENSIONS)


def _required_string(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise TaskContractError(f"{field_name} must be a non-empty string")
    return value.strip()


def _string_list(payload: dict[str, Any], field_name: str, *, paths: bool = False) -> tuple[str, ...]:
    value = payload.get(field_name, [])
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise TaskContractError(f"{field_name} must be a list of non-empty strings")
    if paths:
        return tuple(safe_relative_path(item, field_name=field_name) for item in value)
    return tuple(item.strip() for item in value)


def _identity(payload: Any) -> tuple[KernelRecipeIdentity, str]:
    if not isinstance(payload, dict):
        raise TaskContractError("identity must be a JSON object")
    missing = _IDENTITY_FIELDS - set(payload)
    unknown = set(payload) - _IDENTITY_FIELDS
    if missing:
        raise TaskContractError(f"identity is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise TaskContractError(f"identity has unknown fields: {', '.join(sorted(unknown))}")
    try:
        identity = KernelRecipeIdentity.from_mapping(payload)
        operator_id = kernel_recipe_canonical_id(identity)
    except ValueError as error:
        raise TaskContractError(str(error)) from error
    if identity.producer != LOOP_PRODUCER:
        raise TaskContractError(f"identity.producer must be {LOOP_PRODUCER!r}")
    return identity, operator_id


def parse_task_payload(
    payload: Any,
    *,
    task_dir: str | Path,
    expected_base_commit: str | None = None,
    enforce_directory_identity: bool = True,
) -> KernelRewriteTask:
    """Validate one decoded task payload and return its immutable contract."""
    if not isinstance(payload, dict):
        raise TaskContractError("task.json must contain a JSON object")
    missing = _REQUIRED_TASK_FIELDS - set(payload)
    unknown = set(payload) - _TASK_FIELDS
    if missing:
        raise TaskContractError(f"task.json is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise TaskContractError(f"task.json has unknown fields: {', '.join(sorted(unknown))}")

    version = payload.get("schema_version")
    if isinstance(version, bool) or version != TASK_SCHEMA_VERSION:
        raise TaskContractError(f"unsupported task schema {version!r}; expected {TASK_SCHEMA_VERSION}")

    identity, operator_id = _identity(payload.get("identity"))
    root = Path(task_dir).expanduser().resolve()
    if enforce_directory_identity and root.name != operator_id:
        raise TaskContractError(f"task directory {root.name!r} does not match canonical operator id {operator_id!r}")

    base_commit = _required_string(payload, "base_commit").lower()
    if not _COMMIT_RE.fullmatch(base_commit):
        raise TaskContractError("base_commit must be a full 40- or 64-character hexadecimal commit id")
    if expected_base_commit is not None and base_commit != str(expected_base_commit).strip().lower():
        raise TaskContractError(
            f"base_commit mismatch: task has {base_commit}, expected {str(expected_base_commit).strip().lower()}"
        )

    repo_root_raw = _required_string(payload, "repo_root")
    repo_root = Path(repo_root_raw).expanduser()
    if not repo_root.is_absolute():
        raise TaskContractError("repo_root must be an absolute path")
    repo_root = repo_root.resolve()
    if not repo_root.is_dir():
        raise TaskContractError(f"repo_root is not a directory: {repo_root}")

    kernel_path = safe_relative_path(_required_string(payload, "kernel_path"), field_name="kernel_path")
    driver_path = safe_relative_path(_required_string(payload, "driver_path"), field_name="driver_path")
    if driver_path != "driver.py":
        raise TaskContractError("driver_path must be exactly 'driver.py'")
    driver_file = (root / driver_path).resolve()
    try:
        driver_file.relative_to(root)
    except ValueError as error:
        raise TaskContractError(f"driver_path escapes task directory: {driver_path!r}") from error
    if not driver_file.is_file():
        raise TaskContractError(f"driver_path is not a file: {driver_path!r}")

    operator_name = _required_string(payload, "operator_name")
    if normalize_operator_name(operator_name) != identity.kernel_name:
        raise TaskContractError(
            "operator_name does not normalize to identity.kernel_name: "
            f"{normalize_operator_name(operator_name)!r} != {identity.kernel_name!r}"
        )

    priority = payload.get("priority")
    if isinstance(priority, bool) or not isinstance(priority, int) or priority < 0:
        raise TaskContractError("priority must be a non-negative integer")

    shape_cases = payload.get("shape_cases", [])
    if not isinstance(shape_cases, list) or any(not isinstance(case, dict) for case in shape_cases):
        raise TaskContractError("shape_cases must be a list of JSON objects")
    evidence = payload.get("evidence", [])
    if not isinstance(evidence, list):
        raise TaskContractError("evidence must be a JSON list")
    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        raise TaskContractError("reason must be a string")

    return KernelRewriteTask(
        identity=identity,
        operator_id=operator_id,
        base_commit=base_commit,
        repo_root=repo_root,
        kernel_path=kernel_path,
        operator_name=operator_name,
        driver_path=driver_path,
        priority=priority,
        source_files=_string_list(payload, "source_files", paths=True),
        target_functions=_string_list(payload, "target_functions"),
        shape_cases=tuple(copy.deepcopy(shape_cases)),
        reason=reason,
        evidence=tuple(copy.deepcopy(evidence)),
    )


def load_task(
    task_dir: str | Path,
    *,
    expected_base_commit: str | None = None,
    record_state: bool = True,
) -> TaskParseResult:
    """Load one task, recording a skipped state instead of propagating contract errors."""
    layout = TaskLayout(Path(task_dir))
    try:
        payload = json.loads(layout.task_json.read_text(encoding="utf-8"))
        task = parse_task_payload(
            payload,
            task_dir=layout.root,
            expected_base_commit=expected_base_commit,
        )
        if record_state:
            from kernelforge.kernel_rewrite_controller.state import TaskStateStore

            state_store = TaskStateStore(layout.root)
            if state_store.load() is None:
                state_store.initialize_ready()
        return TaskParseResult(task=task)
    except (OSError, json.JSONDecodeError, TaskContractError) as error:
        reason = f"could not load task: {error}"
        if record_state:
            from kernelforge.kernel_rewrite_controller.state import TaskStateStore

            TaskStateStore(layout.root).mark_skipped(reason)
        return TaskParseResult(reason=reason)


def discover_task_dirs(layout: ControllerLayout) -> list[Path]:
    """Return complete task directories in deterministic filename order."""
    root = layout.tasks_root
    if not root.is_dir():
        return []
    return sorted(
        (entry for entry in root.iterdir() if layout.is_published_task_dir(entry)),
        key=lambda entry: entry.name,
    )


def sort_tasks(tasks: list[KernelRewriteTask]) -> list[KernelRewriteTask]:
    """Sort by agent priority and canonical identity, dropping duplicate identities."""
    selected: dict[str, KernelRewriteTask] = {}
    for task in sorted(tasks, key=lambda item: (item.priority, item.operator_id)):
        selected.setdefault(task.operator_id, task)
    return list(selected.values())


__all__ = [
    "discover_task_dirs",
    "load_task",
    "parse_task_payload",
    "sort_tasks",
]
