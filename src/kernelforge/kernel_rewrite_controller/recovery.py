# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Recover validated forge-loop best results into controller publications."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernelforge.kernel_rewrite_controller.paths import ControllerLayout
from kernelforge.kernel_rewrite_controller.publisher import (
    PUBLICATION_FILENAME,
    publication_from_task,
    publish_operator_result,
)
from kernelforge.kernel_rewrite_controller.state import TaskStateStore
from kernelforge.kernel_rewrite_controller.task import discover_task_dirs, load_task
from kernelforge.kernel_rewrite_controller.worktree import (
    OperatorWorktree,
    export_patch_from_base,
)
from kernelforge.loop.reporting import BestResultPublisher


@dataclass(frozen=True)
class RecoveryResult:
    """Outcome of reconciling one task workspace with public results."""

    operator_id: str
    published: bool
    patch_dir: Path | None = None
    best_commit: str = ""
    reason: str = ""


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _trusted_manifest(workspace: Path) -> dict[str, Any] | None:
    publisher = BestResultPublisher(str(workspace))
    manifest = _load_json(publisher.manifest_path)
    if not manifest:
        return None
    try:
        iteration = int(manifest.get("iteration"))
    except (TypeError, ValueError):
        return None
    commit = str(manifest.get("commit_hash") or "").strip()
    try:
        complete = publisher.describes_current_best(
            iteration=iteration,
            commit_hash=commit,
        )
    except Exception:
        complete = False
    if (
        not commit
        or manifest.get("correctness_passed") is not True
        or manifest.get("total_improved") is not True
        or not complete
    ):
        return None
    return manifest


def _trusted_result_sidecar(task_dir: Path) -> dict[str, Any] | None:
    result = _load_json(task_dir / "forge-result.json")
    if not result or result.get("improved") is not True:
        return None
    best_commit = str(result.get("best_commit") or "").strip()
    if not best_commit:
        checkpoint = result.get("checkpoint")
        if isinstance(checkpoint, dict) and checkpoint.get("validation_passed") is True:
            best_commit = str(checkpoint.get("best_commit") or "").strip()
    if not best_commit:
        return None
    return {**result, "commit_hash": best_commit}


def _already_published(layout: ControllerLayout, operator_id: str, best_commit: str) -> bool:
    metadata = _load_json(layout.patch_dir(operator_id) / PUBLICATION_FILENAME)
    return bool(metadata and str(metadata.get("best_commit") or "") == best_commit)


def recover_task_result(
    layout: ControllerLayout,
    task_dir: str | Path,
    *,
    update_state: bool = True,
) -> RecoveryResult:
    """Publish the newest trusted best result from one operator workspace."""
    parsed = load_task(task_dir, record_state=False)
    if parsed.task is None:
        return RecoveryResult(operator_id=Path(task_dir).name, published=False, reason=parsed.reason)
    task = parsed.task
    workspace = layout.workspace_dir(task.operator_id)
    if not workspace.is_dir():
        return RecoveryResult(
            operator_id=task.operator_id,
            published=False,
            reason="operator workspace does not exist",
        )

    manifest = _trusted_manifest(workspace)
    sidecar = _trusted_result_sidecar(Path(task_dir))
    source = "best manifest"
    if sidecar is not None and (
        manifest is None or str(sidecar.get("commit_hash") or "") != str(manifest.get("commit_hash") or "")
    ):
        manifest = sidecar
        source = "forge result sidecar"
    if manifest is None:
        return RecoveryResult(
            operator_id=task.operator_id,
            published=False,
            reason="no trusted forge-loop best result",
        )

    best_commit = str(manifest.get("commit_hash") or "").strip().lower()
    if _already_published(layout, task.operator_id, best_commit):
        patch_dir = layout.patch_dir(task.operator_id)
        if update_state:
            TaskStateStore(task_dir).mark_recovered_success(
                result_patch_dir=str(patch_dir),
                reason="best result already published",
            )
        return RecoveryResult(
            operator_id=task.operator_id,
            published=False,
            patch_dir=patch_dir,
            best_commit=best_commit,
            reason="best result already published",
        )

    worktree = OperatorWorktree(
        repo_root=task.repo_root,
        workspace=workspace,
        branch="",
        base_commit=task.base_commit,
        kernel_path=(workspace / task.kernel_path).resolve(),
        source_files=tuple((workspace / relative).resolve() for relative in task.source_files),
    )
    try:
        patch = export_patch_from_base(worktree, best_commit=best_commit)
        if not patch.strip():
            return RecoveryResult(
                operator_id=task.operator_id,
                published=False,
                best_commit=best_commit,
                reason=f"{source} has no changes from the controller base",
            )
        patch_dir = publish_operator_result(
            layout,
            publication_from_task(
                task,
                best_commit=best_commit,
                patch=patch,
                manifest=manifest,
            ),
        )
        if update_state:
            TaskStateStore(task_dir).mark_recovered_success(
                result_patch_dir=str(patch_dir),
                reason=f"published from {source}",
            )
        return RecoveryResult(
            operator_id=task.operator_id,
            published=True,
            patch_dir=patch_dir,
            best_commit=best_commit,
        )
    except Exception as error:
        return RecoveryResult(
            operator_id=task.operator_id,
            published=False,
            best_commit=best_commit,
            reason=f"could not publish {source}: {error}",
        )


def recover_all_task_results(layout: ControllerLayout) -> tuple[RecoveryResult, ...]:
    """Reconcile every published task in deterministic operator order."""
    return tuple(recover_task_result(layout, task_dir) for task_dir in discover_task_dirs(layout))


__all__ = [
    "RecoveryResult",
    "recover_all_task_results",
    "recover_task_result",
]
