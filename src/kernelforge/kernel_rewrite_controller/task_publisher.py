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

#: What an agent writes into ``task.json`` to take a draft back. It has no tool
#: that can delete a directory -- the analysis session runs without a shell, and
#: Write cannot remove -- so withdrawing has to be something it can write. Any
#: draft whose task.json carries this key is neither published nor counted as
#: refused, and the Stop hook stops holding the session open for it.
WITHDRAWN_KEY = "withdrawn"

#: Where a refusal is left for the agent to read, inside the draft it refused.
#: Validation runs out of process on a timer, so there is no tool result to
#: return the reason on; without a file the agent finishes the session believing
#: it published, which is the failure this whole path exists to prevent. The
#: name is stable so the prompt can point at it.
REJECTION_FILENAME = "rejection.json"


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
            identity[field] = value.strip().lower()
        # Host-owned, like base_commit and driver_path below: the parser derives
        # it too, and writing it here is what makes the published task.json state
        # the identity the controller went on to use rather than the draft's.
        operator_name = normalized.get("operator_name")
        if isinstance(operator_name, str) and operator_name.strip():
            identity["kernel_name"] = normalize_operator_name(operator_name)
        normalized["identity"] = identity
    repo_root = normalized.get("repo_root")
    if isinstance(repo_root, str):
        normalized["repo_root"] = repo_root.strip()
    return normalized


def _repo_head(repo_root: Path) -> str:
    """Pin the repository's live HEAD, refusing anything but its top level.

    A refusal here names the path to use rather than only the rule, because the
    agent revises from the reason alone: it has no shell to resolve a top level
    with, and the answer is already in hand by the time the rule can fail.
    """
    try:
        top = Path(git("rev-parse", "--show-toplevel", cwd=repo_root).stdout.strip()).resolve()
        head = git("rev-parse", "HEAD", cwd=repo_root).stdout.strip().lower()
    except GitError as error:
        raise ValueError(
            f"repo_root is not a Git checkout: {repo_root}: {error}. "
            "Pass the Git top-level of the repository that holds kernel_path."
        ) from error
    if top != repo_root.resolve():
        raise ValueError(f"repo_root must be the Git top-level directory: {repo_root} sits inside {top}; use {top}")
    return head


def _validate_task_sources_at_base(task: KernelRewriteTask) -> None:
    """Require every declared source to exist in this repository at the base.

    The common way to fail this is to name a file from the repository on the
    other side of a call chain, which reads here as an ordinary missing path, so
    the refusal says which repository was searched and where a cross-repository
    reference belongs instead.
    """
    for relative in dict.fromkeys((task.kernel_path, *task.source_files)):
        try:
            git(
                "cat-file",
                "-e",
                f"{task.base_commit}:{relative}",
                cwd=task.repo_root,
            )
        except GitError as error:
            raise ValueError(
                f"source path is not tracked in {task.repo_root} at {task.base_commit[:12]}: {relative}. "
                "One task edits one repository: a file that lives in another repo, or one Git does "
                "not track here, belongs in evidence rather than source_files."
            ) from error


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
            reason=(
                f"an operator task for {task.operator_id} is already published; "
                "drop this draft or point it at a different operator"
            ),
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


def _write_rejection(staged: Path, reason: str) -> None:
    """Leave the refusal beside the draft that earned it.

    Best-effort: a staging directory that cannot be written to is one the agent
    cannot revise either, and losing the note must not stop the scan from
    reporting the refusal through its ordinary return value.
    """
    try:
        atomic_write_text(
            staged / REJECTION_FILENAME,
            json.dumps({"draft": staged.name, "reason": reason}, indent=2, sort_keys=True) + "\n",
        )
    except OSError:
        log.warning("could not record the refusal of staged task %s", staged.name)


def _is_withdrawn(staged: Path) -> bool:
    """True when the agent has taken this draft back rather than fixed it."""
    try:
        payload = json.loads((staged / "task.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and bool(payload.get(WITHDRAWN_KEY))


def pending_rejections(staging_root: Path) -> dict[str, str]:
    """Return the refusal still standing against each staged draft.

    A published draft is deleted whole and a re-refused one is overwritten, so
    the note's presence is what says the draft is currently refused -- unless
    the agent has withdrawn it, which is the one way out it can actually take.
    """
    root = Path(staging_root)
    if not root.is_dir():
        return {}
    pending: dict[str, str] = {}
    for entry in sorted(root.iterdir(), key=lambda path: path.name):
        if entry.name.startswith(".") or not entry.is_dir():
            continue
        try:
            payload = json.loads((entry / REJECTION_FILENAME).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and not _is_withdrawn(entry):
            pending[entry.name] = str(payload.get("reason") or "")
    return pending


def publish_complete_staged_tasks(
    layout: ControllerLayout,
    *,
    quiescent_sec: float = PUBLISH_QUIESCENT_SEC,
    now: Callable[[], float] = time.time,
    refused: dict[str, float] | None = None,
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
        # Both files existing is not a completion signal: this runs on a timer beside the live agent, and taking a
        # directory mid-write copies a truncated driver.py -- whose contents nothing downstream validates -- and then
        # deletes the agent's working copy.
        newest = _newest_mtime(entry)
        if float(now()) - newest < float(quiescent_sec):
            continue
        if refused is not None and refused.get(entry.name) == newest:
            continue
        if _is_withdrawn(entry):
            # Taken back by the agent. Left in place as the record of a
            # candidate it examined and rejected, and not offered again.
            if refused is not None:
                refused[entry.name] = newest
            continue
        result = publish_staged_task(layout, entry)
        if result.published:
            log.info("published operator task %s from %s", result.operator_id, entry.name)
        else:
            log.warning("rejected staged task %s: %s", entry.name, result.reason)
            # Written before the mtime is remembered, so the note itself is part
            # of what "unchanged" means; a draft the agent then revises reads as
            # changed and is offered again.
            _write_rejection(entry, result.reason)
            if refused is not None:
                refused[entry.name] = _newest_mtime(entry)
        results.append(result)
    return tuple(results)


__all__ = [
    "PUBLISH_QUIESCENT_SEC",
    "REJECTION_FILENAME",
    "WITHDRAWN_KEY",
    "TaskPublicationResult",
    "pending_rejections",
    "publish_complete_staged_tasks",
    "publish_staged_task",
]
