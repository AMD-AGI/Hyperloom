# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read and validate KernelForge controller patch publications."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

from kernelforge.kernel_rewrite_controller.paths import operator_directory_name

PUBLICATION_SCHEMA_VERSION = 2
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_IDENTITY_FIELDS = (
    "producer",
    "kernel_name",
    "framework",
    "framework_version",
    "backend",
    "gpu",
)
#: Every file a complete publication directory holds, in the order the contract names them.
_PUBLICATION_ARTIFACT_NAMES = ("change.patch", "report.md", "publication.json")


class ControllerPublicationError(ValueError):
    """One Controller publication is incomplete or unsafe to consume."""


@dataclass(frozen=True)
class ControllerPatchPublication:
    operator_id: str
    identity: dict[str, str]
    base_commit: str
    best_commit: str
    repo_root: Path
    kernel_path: str
    operator_name: str
    manifest: dict[str, Any]
    patch_path: Path
    report_path: Path
    publication_path: Path
    changed_files: tuple[str, ...] = ()


def discover_controller_patch_dirs(patches_root: str | Path) -> tuple[Path, ...]:
    """Return complete operator directories in deterministic filename order."""
    root = Path(patches_root).expanduser().resolve()
    if not root.is_dir():
        return ()
    complete = []
    for entry in sorted(root.iterdir(), key=lambda path: path.name):
        if entry.name.startswith(".") or not entry.is_dir():
            continue
        if all((entry / name).is_file() for name in _PUBLICATION_ARTIFACT_NAMES):
            complete.append(entry)
    return tuple(complete)


def _relative_path(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ControllerPublicationError(f"{field_name} must be a non-empty string")
    raw = value.strip()
    path = PurePosixPath(raw)
    if raw in {".", ".."} or path.is_absolute() or "\\" in raw or "\x00" in raw or ".." in path.parts:
        raise ControllerPublicationError(f"{field_name} must be a safe repo-relative path")
    return path.as_posix()


class _PublicationArtifacts(NamedTuple):
    """The files a complete publication directory holds."""

    patch_path: Path
    report_path: Path
    publication_path: Path


class _PublicationCommits(NamedTuple):
    """The Git object ids a publication was produced between."""

    base: str
    best: str


def _publication_artifacts(root: Path) -> _PublicationArtifacts:
    """The publication's artifacts, each confirmed to be a regular file."""
    paths = tuple(root / name for name in _PUBLICATION_ARTIFACT_NAMES)
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ControllerPublicationError(f"publication artifact must be a regular file: {path}")
    return _PublicationArtifacts(*paths)


def _publication_payload(publication_path: Path) -> dict[str, Any]:
    """The publication's metadata, confirmed to be a micro-validated record of a supported schema."""
    try:
        payload = json.loads(publication_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControllerPublicationError(f"could not read publication metadata: {error}") from error
    if not isinstance(payload, dict):
        raise ControllerPublicationError("publication metadata must be a JSON object")
    if payload.get("schema_version") != PUBLICATION_SCHEMA_VERSION:
        raise ControllerPublicationError(f"unsupported publication schema {payload.get('schema_version')!r}")
    if payload.get("micro_validated") is not True:
        raise ControllerPublicationError("publication is not micro-validated")
    return payload


def _publication_identity(payload: dict[str, Any]) -> dict[str, str]:
    """The canonical identity dimensions the publication names."""
    identity_raw = payload.get("identity")
    if not isinstance(identity_raw, dict) or set(identity_raw) != set(_IDENTITY_FIELDS):
        raise ControllerPublicationError("publication identity must contain the six canonical dimensions")
    identity: dict[str, str] = {}
    for field in _IDENTITY_FIELDS:
        value = identity_raw.get(field)
        if not isinstance(value, str) or not value:
            raise ControllerPublicationError(f"identity.{field} must be a non-empty string")
        identity[field] = value
    return identity


def _publication_operator_id(payload: dict[str, Any], identity: dict[str, str], directory_name: str) -> str:
    """The operator id, confirmed to agree with both the identity and the directory holding it."""
    operator_id = str(payload.get("operator_id") or "")
    expected_id = ":".join(["kernel", *(identity[field] for field in _IDENTITY_FIELDS)])
    if operator_id != expected_id or directory_name != operator_directory_name(operator_id):
        raise ControllerPublicationError("operator_id does not match identity or directory name")
    return operator_id


def _publication_commits(payload: dict[str, Any]) -> _PublicationCommits:
    """The base and best commits, confirmed to be full hexadecimal object ids."""
    base_commit = str(payload.get("base_commit") or "").lower()
    best_commit = str(payload.get("best_commit") or "").lower()
    if not _COMMIT_RE.fullmatch(base_commit) or not _COMMIT_RE.fullmatch(best_commit):
        raise ControllerPublicationError("publication commits must be full hexadecimal object ids")
    return _PublicationCommits(base_commit, best_commit)


def _publication_repo_root(payload: dict[str, Any]) -> Path:
    """The existing absolute repository the patch was exported against."""
    repo_root_raw = payload.get("repo_root")
    if not isinstance(repo_root_raw, str) or not Path(repo_root_raw).expanduser().is_absolute():
        raise ControllerPublicationError("repo_root must be an absolute path")
    repo_root = Path(repo_root_raw).expanduser().resolve()
    if not repo_root.is_dir():
        raise ControllerPublicationError(f"repo_root does not exist: {repo_root}")
    return repo_root


def load_controller_publication(patch_dir: str | Path) -> ControllerPatchPublication:
    """Validate one publication directory and return its integration contract."""
    root = Path(patch_dir).expanduser()
    if not root.is_dir():
        raise ControllerPublicationError(f"patch directory does not exist: {root}")
    artifacts = _publication_artifacts(root)
    payload = _publication_payload(artifacts.publication_path)
    identity = _publication_identity(payload)
    operator_id = _publication_operator_id(payload, identity, root.name)
    commits = _publication_commits(payload)
    repo_root = _publication_repo_root(payload)
    kernel_path = _relative_path(payload.get("kernel_path"), "kernel_path")
    operator_name = str(payload.get("operator_name") or "").strip()
    if not operator_name:
        raise ControllerPublicationError("operator_name must be a non-empty string")
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise ControllerPublicationError("manifest must be a JSON object")
    # The scope the Controller says its patch has.
    changed_files_raw = payload.get("changed_files", [])
    if not isinstance(changed_files_raw, list):
        raise ControllerPublicationError("changed_files must be a JSON list")
    changed_files = tuple(_relative_path(value, "changed_files entry") for value in changed_files_raw)
    return ControllerPatchPublication(
        operator_id=operator_id,
        identity=identity,
        base_commit=commits.base,
        best_commit=commits.best,
        repo_root=repo_root,
        kernel_path=kernel_path,
        operator_name=operator_name,
        manifest=manifest,
        patch_path=artifacts.patch_path.resolve(),
        report_path=artifacts.report_path.resolve(),
        publication_path=artifacts.publication_path.resolve(),
        changed_files=changed_files,
    )


__all__ = [
    "PUBLICATION_SCHEMA_VERSION",
    "ControllerPatchPublication",
    "ControllerPublicationError",
    "discover_controller_patch_dirs",
    "load_controller_publication",
]
