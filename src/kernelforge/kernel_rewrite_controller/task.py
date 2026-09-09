# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse and validate one operator rewrite task."""

from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path
from typing import Any

from kernelforge.kernel_rewrite_controller.contracts import (
    KernelRewriteTask,
    TaskContractError,
    TaskParseResult,
)
from kernelforge.kernel_rewrite_controller.paths import (
    ControllerLayout,
    TaskLayout,
    operator_directory_name,
    safe_relative_path,
)
from kernelforge.knowledge.implementation_identity import normalize_operator_name
from kernelforge.knowledge.kernel_identity import (
    KERNEL_CANONICAL_DIMENSIONS,
    KernelRecipeIdentity,
    kernel_recipe_canonical_id,
)
from kernelforge.knowledge.loop_identity import LOOP_PRODUCER

log = logging.getLogger(__name__)

_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_REQUIRED_TASK_FIELDS = frozenset(
    {
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
        "gpu_pct",
    }
)
_TASK_FIELDS = _REQUIRED_TASK_FIELDS | _OPTIONAL_TASK_FIELDS
_IDENTITY_FIELDS = frozenset(KERNEL_CANONICAL_DIMENSIONS)
#: ``kernel_name`` is the address form of ``operator_name`` rather than a
#: dimension of its own, so the host derives it and the agent does not supply
#: it. A draft that still carries one is accepted and the value replaced: the
#: agent has no channel to hear a refusal on, so a rule it can only learn by
#: breaking costs the whole analysis budget.
_AGENT_IDENTITY_FIELDS = _IDENTITY_FIELDS - {"kernel_name"}
#: What ``normalize_operator_name`` answers when no usable name survives. Read
#: from the normalizer rather than spelled out so the two cannot drift. Every
#: unnameable operator would otherwise share one canonical id, and with it one
#: task directory, one patch pointer and one knowledge-base page.
_UNNAMEABLE_OPERATOR = normalize_operator_name("")


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


def _log_ignored(unknown: set[str], where: str) -> None:
    """Note the fields this contract does not read, without refusing the task.

    An extra key decides nothing: every field the run acts on is required, and
    the optional ones are read by name. Refusing over one would cost the whole
    operator, so it is dropped and recorded instead.
    """
    if unknown:
        log.info("ignoring unknown %s field(s): %s", where, ", ".join(sorted(unknown)))


def _identity(payload: Any, *, operator_name: str) -> tuple[KernelRecipeIdentity, str]:
    """Build the six-tuple, deriving ``kernel_name`` from ``operator_name``.

    Deriving rather than checking is what keeps the controller's canonical id
    and the knowledge-base page forge-loop writes to at one address. The loop
    resolves its page from ``--operator-name`` alone, so a ``kernel_name`` the
    agent chose independently could disagree with it, and the disagreement is
    silent: reads land on a page no write reached, and one operator accumulates
    two half-filled histories.
    """
    if not isinstance(payload, dict):
        raise TaskContractError("identity must be a JSON object")
    missing = _AGENT_IDENTITY_FIELDS - set(payload)
    if missing:
        raise TaskContractError(f"identity is missing fields: {', '.join(sorted(missing))}")
    _log_ignored(set(payload) - _IDENTITY_FIELDS, "identity")
    kernel_name = normalize_operator_name(operator_name)
    if kernel_name == _UNNAMEABLE_OPERATOR:
        raise TaskContractError(f"operator_name normalizes to no usable kernel name: {operator_name!r}")
    try:
        identity = KernelRecipeIdentity.from_mapping({**payload, "kernel_name": kernel_name})
        operator_id = kernel_recipe_canonical_id(identity)
    except ValueError as error:
        raise TaskContractError(str(error)) from error
    if identity.producer != LOOP_PRODUCER:
        raise TaskContractError(f"identity.producer must be {LOOP_PRODUCER!r}")
    # ``backend`` is not checked against the registered set. It selects a prompt
    # layer, and forge-loop already answers an unregistered one with no layer
    # rather than an error, so refusing here would cost a whole operator over a
    # naming choice the loop is willing to live with.
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
    if missing:
        raise TaskContractError(f"task.json is missing fields: {', '.join(sorted(missing))}")
    _log_ignored(set(payload) - _TASK_FIELDS, "task.json")

    # Read ahead of the identity: ``identity.kernel_name`` is this name's
    # address form, so the six-tuple cannot be built before it.
    operator_name = _required_string(payload, "operator_name")
    identity, operator_id = _identity(payload.get("identity"), operator_name=operator_name)
    root = Path(task_dir).expanduser().resolve()
    if enforce_directory_identity and root.name != operator_directory_name(operator_id):
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

    priority = payload.get("priority")
    if isinstance(priority, bool) or not isinstance(priority, int) or priority < 0:
        raise TaskContractError("priority must be a non-negative integer")

    # Carried verbatim rather than validated: no code reads a case, so a shape
    # the contract disagrees with is still worth more to the driver author than
    # a refused task -- and by the same argument, not reshaped either.
    shape_cases = payload.get("shape_cases", [])
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
        shape_cases=copy.deepcopy(shape_cases),
        reason=reason,
        evidence=tuple(copy.deepcopy(evidence)),
        # Unchecked by contract: it is read by people, not by the run, and a
        # refusal here would trade an operator for a number's formatting.
        gpu_pct=copy.deepcopy(payload.get("gpu_pct")),
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
    """Sort by agent priority and canonical identity, keeping one task per identity."""
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
