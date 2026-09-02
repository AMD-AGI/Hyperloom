# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-validated publication of tasks authored by the opportunity agent."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from kernelforge.durable_io import atomic_write_text, fsync_directory
from kernelforge.kernel_rewrite_controller.paths import ControllerLayout
from kernelforge.kernel_rewrite_controller.task import parse_task_payload
from kernelforge.llm.git import GitError, git


@dataclass(frozen=True)
class TaskPublicationResult:
    """Result of validating and publishing one agent-authored task."""

    source_dir: Path
    operator_id: str = ""
    published: bool = False
    reason: str = ""


def _repo_head(repo_root: Path) -> str:
    try:
        top = Path(git("rev-parse", "--show-toplevel", cwd=repo_root).stdout.strip()).resolve()
        head = git("rev-parse", "HEAD", cwd=repo_root).stdout.strip().lower()
    except GitError as error:
        raise ValueError(f"repo_root is not a Git checkout: {repo_root}: {error}") from error
    if top != repo_root.resolve():
        raise ValueError(f"repo_root must be the Git top-level directory: {repo_root}")
    return head


def _fsync_tree(root: Path) -> None:
    for directory, _subdirectories, filenames in os.walk(root):
        current = Path(directory)
        for filename in filenames:
            descriptor = os.open(str(current / filename), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        fsync_directory(current)


def publish_staged_task(
    layout: ControllerLayout,
    staged_dir: str | Path,
) -> TaskPublicationResult:
    """Validate one staged task, pin its repo HEAD, and publish it atomically."""
    source = Path(staged_dir).expanduser().resolve()
    task_json = source / "task.json"
    driver = source / "driver.py"
    if not source.is_dir() or source.is_symlink():
        return TaskPublicationResult(source_dir=source, reason="staged task is not a safe directory")
    if not task_json.is_file() or task_json.is_symlink():
        return TaskPublicationResult(source_dir=source, reason="staged task has no regular task.json")
    if not driver.is_file() or driver.is_symlink():
        return TaskPublicationResult(source_dir=source, reason="staged task has no regular driver.py")

    try:
        payload = json.loads(task_json.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("task.json must contain a JSON object")
        repo_root_raw = payload.get("repo_root")
        if not isinstance(repo_root_raw, str) or not Path(repo_root_raw).expanduser().is_absolute():
            raise ValueError("repo_root must be an absolute path")
        payload["base_commit"] = _repo_head(Path(repo_root_raw).expanduser().resolve())
        payload["driver_path"] = "driver.py"
        task = parse_task_payload(
            payload,
            task_dir=source,
            enforce_directory_identity=False,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return TaskPublicationResult(source_dir=source, reason=f"invalid staged task: {error}")

    destination = layout.task_dir(task.operator_id)
    if destination.exists() or destination.is_symlink():
        return TaskPublicationResult(
            source_dir=source,
            operator_id=task.operator_id,
            reason="operator task is already published",
        )

    layout.tasks_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            dir=str(layout.tasks_root),
            prefix=f".{task.operator_id}.",
        )
    )
    try:
        atomic_write_text(
            temporary / "task.json",
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
        shutil.copy2(driver, temporary / "driver.py")
        _fsync_tree(temporary)
        os.replace(temporary, destination)
        fsync_directory(layout.tasks_root)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    shutil.rmtree(source, ignore_errors=True)
    return TaskPublicationResult(
        source_dir=source,
        operator_id=task.operator_id,
        published=True,
    )


def publish_complete_staged_tasks(
    layout: ControllerLayout,
) -> tuple[TaskPublicationResult, ...]:
    """Publish every complete non-temporary task currently visible in staging."""
    root = layout.agent_staging_root
    if not root.is_dir():
        return ()
    results = []
    for entry in sorted(root.iterdir(), key=lambda path: path.name):
        if entry.name.startswith(".") or not entry.is_dir():
            continue
        if not (entry / "task.json").is_file() or not (entry / "driver.py").is_file():
            continue
        results.append(publish_staged_task(layout, entry))
    return tuple(results)


__all__ = [
    "TaskPublicationResult",
    "publish_complete_staged_tasks",
    "publish_staged_task",
]
