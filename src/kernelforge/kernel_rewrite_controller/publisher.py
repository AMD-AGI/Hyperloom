# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Crash-safe publication of controller patch results."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernelforge.durable_io import atomic_write_text, fsync_directory
from kernelforge.kernel_rewrite_controller.contracts import KernelRewriteTask
from kernelforge.kernel_rewrite_controller.paths import ControllerLayout

PATCH_FILENAME = "change.patch"
REPORT_FILENAME = "report.md"
PUBLICATION_FILENAME = "publication.json"
PUBLICATION_SCHEMA_VERSION = 2


class PublicationError(RuntimeError):
    """A patch result could not be published without violating durability."""


@dataclass(frozen=True)
class OperatorPublication:
    """One micro-validated operator result ready for Hyperloom."""

    operator_id: str
    identity: dict[str, str]
    base_commit: str
    best_commit: str
    repo_root: str
    kernel_path: str
    operator_name: str
    patch: str
    report: str
    manifest: dict[str, Any]

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": PUBLICATION_SCHEMA_VERSION,
            "operator_id": self.operator_id,
            "identity": self.identity,
            "base_commit": self.base_commit,
            "best_commit": self.best_commit,
            "repo_root": self.repo_root,
            "kernel_path": self.kernel_path,
            "operator_name": self.operator_name,
            "micro_validated": True,
            "manifest": self.manifest,
        }


def _version_root(layout: ControllerLayout, operator_id: str) -> Path:
    digest = hashlib.sha256(operator_id.encode("utf-8")).hexdigest()
    return layout.patch_versions_root / digest


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


def _validate_version(version: Path, publication: OperatorPublication) -> None:
    required = (PATCH_FILENAME, REPORT_FILENAME, PUBLICATION_FILENAME)
    missing = [name for name in required if not (version / name).is_file()]
    if missing:
        raise PublicationError(f"incomplete operator publication {version}: {', '.join(missing)}")
    try:
        metadata = json.loads((version / PUBLICATION_FILENAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicationError(f"invalid operator publication metadata: {version}") from error
    if metadata != publication.metadata():
        raise PublicationError(f"operator publication conflicts with existing version: {version}")
    if (version / PATCH_FILENAME).read_text(encoding="utf-8") != publication.patch:
        raise PublicationError(f"operator publication patch conflicts with existing version: {version}")
    if (version / REPORT_FILENAME).read_text(encoding="utf-8") != publication.report:
        raise PublicationError(f"operator publication report conflicts with existing version: {version}")


def publish_operator_result(
    layout: ControllerLayout,
    publication: OperatorPublication,
) -> Path:
    """Atomically point one operator directory at a complete immutable version."""
    layout.patches_root.mkdir(parents=True, exist_ok=True)
    versions = _version_root(layout, publication.operator_id)
    versions.mkdir(parents=True, exist_ok=True)
    version = versions / publication.best_commit

    if version.exists():
        _validate_version(version, publication)
    else:
        staging = Path(
            tempfile.mkdtemp(
                dir=str(versions),
                prefix=f".{publication.best_commit}.",
            )
        )
        try:
            atomic_write_text(staging / PATCH_FILENAME, publication.patch)
            atomic_write_text(staging / REPORT_FILENAME, publication.report)
            atomic_write_text(
                staging / PUBLICATION_FILENAME,
                json.dumps(publication.metadata(), indent=2, sort_keys=True) + "\n",
            )
            _fsync_tree(staging)
            os.replace(staging, version)
            fsync_directory(versions)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    destination = layout.patch_dir(publication.operator_id)
    if destination.exists() and not destination.is_symlink():
        raise PublicationError(f"operator publication path is not an atomic pointer: {destination}")
    previous_target: Path | None = None
    if destination.is_symlink():
        previous_target = (destination.parent / os.readlink(destination)).resolve()

    pointer = (
        destination.parent / f".{hashlib.sha256(publication.operator_id.encode()).hexdigest()}.tmp-{uuid.uuid4().hex}"
    )
    try:
        relative_target = os.path.relpath(version, destination.parent)
        os.symlink(relative_target, pointer)
        os.replace(pointer, destination)
        fsync_directory(destination.parent)
    finally:
        pointer.unlink(missing_ok=True)

    if previous_target is not None and previous_target != version and previous_target.is_dir():
        shutil.rmtree(previous_target, ignore_errors=True)
    return destination


def render_operator_report(
    task: KernelRewriteTask,
    *,
    best_commit: str,
    manifest: dict[str, Any] | None = None,
) -> str:
    """Render the stable human-readable report shipped beside one patch."""
    details = dict(manifest or {})
    lines = [
        "# Kernel Rewrite Result",
        "",
        f"- **Operator:** `{task.operator_id}`",
        f"- **Base commit:** `{task.base_commit}`",
        f"- **Best commit:** `{best_commit}`",
        f"- **Kernel path:** `{task.kernel_path}`",
        f"- **Correctness:** `{'passed' if details.get('correctness_passed', True) else 'failed'}`",
    ]
    if details.get("mean_case_speedup") is not None:
        lines.append(f"- **Mean case speedup:** `{float(details['mean_case_speedup']):.6f}x`")
    if details.get("iteration") is not None:
        lines.append(f"- **Best iteration:** `{int(details['iteration'])}`")
    changed_files = details.get("changed_files")
    if isinstance(changed_files, list):
        lines.extend(["", "## Changed Files", ""])
        lines.extend(f"- `{path}`" for path in changed_files)
    return "\n".join(lines) + "\n"


def publication_from_task(
    task: KernelRewriteTask,
    *,
    best_commit: str,
    patch: str,
    manifest: dict[str, Any] | None = None,
) -> OperatorPublication:
    """Build one publication with normalized metadata and report."""
    details = dict(manifest or {})
    return OperatorPublication(
        operator_id=task.operator_id,
        identity={
            "producer": task.identity.producer,
            "kernel_name": task.identity.kernel_name,
            "framework": task.identity.framework,
            "framework_version": task.identity.framework_version,
            "backend": task.identity.backend,
            "gpu": task.identity.gpu,
        },
        base_commit=task.base_commit,
        best_commit=str(best_commit).strip().lower(),
        repo_root=str(task.repo_root),
        kernel_path=task.kernel_path,
        operator_name=task.operator_name,
        patch=patch,
        report=render_operator_report(task, best_commit=best_commit, manifest=details),
        manifest=details,
    )


def published_operator_dirs(layout: ControllerLayout) -> tuple[Path, ...]:
    """Return complete public operator pointers in deterministic filename order."""
    if not layout.patches_root.is_dir():
        return ()
    published: list[Path] = []
    for entry in sorted(layout.patches_root.iterdir(), key=lambda path: path.name):
        if entry.name.startswith(".") or not entry.is_dir():
            continue
        if all((entry / filename).is_file() for filename in (PATCH_FILENAME, REPORT_FILENAME, PUBLICATION_FILENAME)):
            published.append(entry)
    return tuple(published)


__all__ = [
    "PATCH_FILENAME",
    "PUBLICATION_FILENAME",
    "PUBLICATION_SCHEMA_VERSION",
    "REPORT_FILENAME",
    "OperatorPublication",
    "PublicationError",
    "publication_from_task",
    "published_operator_dirs",
    "publish_operator_result",
    "render_operator_report",
]
