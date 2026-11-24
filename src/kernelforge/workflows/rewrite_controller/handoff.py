# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read the handoff produced by Hyperloom: three Markdown documents and the source repositories it declares."""

from __future__ import annotations

import json
from pathlib import Path

from kernelforge.workflows.rewrite_controller.contracts import (
    HandoffBundle,
    HandoffContractError,
    REPOSITORIES_FILENAME,
    SERVING_CONTEXT_FILENAME,
    TRACE_EVIDENCE_FILENAME,
    WORKLOAD_FILENAME,
)


def _read_document(root: Path, filename: str) -> str:
    path = root / filename
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise HandoffContractError(f"could not read handoff document {path}: {error}") from error


def _read_repositories(root: Path) -> tuple[tuple[Path, ...], dict[Path, Path]]:
    """The repositories the host declares, at least one, and the installed package each one's checkout stands in for.

    ``{"repositories": ["<absolute path>", ...], "served_trees": {"<repository>": "<absolute path>", ...}}``;
    ``served_trees`` is optional, names declared repositories only, and need not name every one of them.
    """
    path = root / REPOSITORIES_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HandoffContractError(f"could not read handoff repositories {path}: {error}") from error
    entries = payload.get("repositories") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        raise HandoffContractError(f"{path} must list at least one source repository under 'repositories'")
    if not all(isinstance(entry, str) and Path(entry).is_absolute() for entry in entries):
        raise HandoffContractError(f"every repository in {path} must be an absolute path")
    served = payload.get("served_trees", {})
    if not isinstance(served, dict) or not all(
        repository in entries and isinstance(tree, str) and Path(tree).is_absolute()
        for repository, tree in served.items()
    ):
        raise HandoffContractError(f"'served_trees' in {path} must map declared repositories to absolute paths")
    repositories = tuple(Path(entry).resolve() for entry in entries)
    return repositories, {Path(repository).resolve(): Path(tree) for repository, tree in served.items()}


def read_handoff(handoff_dir: str | Path) -> HandoffBundle:
    """Read all required handoff documents without interpreting their content, and the declared repositories."""
    root = Path(handoff_dir).expanduser().resolve()
    if not root.is_dir():
        raise HandoffContractError(f"handoff directory does not exist: {root}")
    repositories, served_trees = _read_repositories(root)
    return HandoffBundle(
        root=root,
        workload=_read_document(root, WORKLOAD_FILENAME),
        serving_context=_read_document(root, SERVING_CONTEXT_FILENAME),
        trace_evidence=_read_document(root, TRACE_EVIDENCE_FILENAME),
        repositories=repositories,
        served_trees=served_trees,
    )


__all__ = ["read_handoff"]
