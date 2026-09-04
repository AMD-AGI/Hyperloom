# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-validated publication of tasks authored by the opportunity agent."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kernelforge.durable_io import atomic_write_text, fsync_directory, fsync_tree
from kernelforge.kernel_rewrite_controller.contracts import KernelRewriteTask
from kernelforge.kernel_rewrite_controller.paths import ControllerLayout
from kernelforge.kernel_rewrite_controller.task import parse_task_payload
from kernelforge.knowledge.implementation_identity import normalize_operator_name
from kernelforge.knowledge.kernel_identity import KERNEL_CANONICAL_DIMENSIONS
from kernelforge.llm.git import GitError, git


log = logging.getLogger(__name__)

#: How long a staged directory must stop changing before it is taken. The agent
#: writes task.json and driver.py in separate tool calls and neither write is
#: atomic, so the pair existing does not mean the pair is finished.
PUBLISH_QUIESCENT_SEC = 5.0


@dataclass(frozen=True)
class TaskPublicationResult:
    """Result of validating and publishing one agent-authored task."""

    source_dir: Path
    operator_id: str = ""
    published: bool = False
    reason: str = ""


def _normalize_agent_task_payload(payload: dict) -> dict:
    """Canonicalize harmless textual variations at the untrusted Agent boundary."""
    normalized = dict(payload)
    identity_raw = normalized.get("identity")
    if isinstance(identity_raw, dict):
        identity = dict(identity_raw)
        for field in KERNEL_CANONICAL_DIMENSIONS:
            value = identity.get(field)
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            identity[field] = normalize_operator_name(stripped) if field == "kernel_name" else stripped.lower()
        normalized["identity"] = identity
    repo_root = normalized.get("repo_root")
    if isinstance(repo_root, str):
        normalized["repo_root"] = repo_root.strip()
    return normalized


def _repo_head(repo_root: Path) -> str:
    try:
        top = Path(git("rev-parse", "--show-toplevel", cwd=repo_root).stdout.strip()).resolve()
        head = git("rev-parse", "HEAD", cwd=repo_root).stdout.strip().lower()
    except GitError as error:
        raise ValueError(f"repo_root is not a Git checkout: {repo_root}: {error}") from error
    if top != repo_root.resolve():
        raise ValueError(f"repo_root must be the Git top-level directory: {repo_root}")
    return head


def _validate_task_sources_at_base(task: KernelRewriteTask) -> None:
    for relative in dict.fromkeys((task.kernel_path, *task.source_files)):
        try:
            git(
                "cat-file",
                "-e",
                f"{task.base_commit}:{relative}",
                cwd=task.repo_root,
            )
        except GitError as error:
            raise ValueError(f"source path is not tracked in repo_root at base_commit: {relative}") from error


def publish_staged_task(
    layout: ControllerLayout,
    staged_dir: str | Path,
) -> TaskPublicationResult:
    """Validate one staged task, pin its repo HEAD, and publish it atomically."""
    unresolved_source = Path(staged_dir).expanduser()
    if unresolved_source.is_symlink():
        return TaskPublicationResult(
            source_dir=unresolved_source.absolute(),
            reason="staged task is not a safe directory",
        )
    source = unresolved_source.resolve()
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
        payload = _normalize_agent_task_payload(payload)
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
        _validate_task_sources_at_base(task)
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
            prefix=f".{destination.name}.",
        )
    )
    try:
        atomic_write_text(
            temporary / "task.json",
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
        shutil.copy2(driver, temporary / "driver.py")
        fsync_tree(temporary)
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


def _newest_mtime(root: Path) -> float:
    """Return the most recent mtime anywhere in one staged directory tree."""
    newest = 0.0
    for path in (root, *root.rglob("*")):
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return newest


def publish_complete_staged_tasks(
    layout: ControllerLayout,
    *,
    quiescent_sec: float = PUBLISH_QUIESCENT_SEC,
    now: Callable[[], float] = time.time,
) -> tuple[TaskPublicationResult, ...]:
    """Publish every staged task that is complete and no longer being written."""
    root = layout.agent_staging_root
    if not root.is_dir():
        return ()
    results = []
    for entry in sorted(root.iterdir(), key=lambda path: path.name):
        if entry.name.startswith(".") or not entry.is_dir():
            continue
        if not (entry / "task.json").is_file() or not (entry / "driver.py").is_file():
            continue
        # Both files existing is not a completion signal: this runs on a timer
        # beside the live agent, and taking a directory mid-write copies a
        # truncated driver.py -- whose contents nothing downstream validates --
        # and then deletes the agent's working copy. Publication is also
        # one-way, so a task revised right after it was written would lose the
        # revision. Wait for the tree to go quiet instead.
        if float(now()) - _newest_mtime(entry) < float(quiescent_sec):
            continue
        result = publish_staged_task(layout, entry)
        if result.published:
            log.info("published operator task %s from %s", result.operator_id, entry.name)
        else:
            # The agent gets no feedback channel, so a rejected draft is
            # otherwise invisible: it stays in staging and the run just reports
            # one fewer task than the agent believes it wrote.
            log.warning("rejected staged task %s: %s", entry.name, result.reason)
        results.append(result)
    return tuple(results)


__all__ = [
    "PUBLISH_QUIESCENT_SEC",
    "TaskPublicationResult",
    "publish_complete_staged_tasks",
    "publish_staged_task",
]
