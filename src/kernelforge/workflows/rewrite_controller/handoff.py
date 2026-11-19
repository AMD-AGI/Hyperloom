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


def _read_repositories(root: Path) -> tuple[Path, ...]:
    """The repositories the host declares: ``{"repositories": ["<absolute path>", ...]}``, at least one."""
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
    return tuple(Path(entry).resolve() for entry in entries)


def read_handoff(handoff_dir: str | Path) -> HandoffBundle:
    """Read all required handoff documents without interpreting their content, and the declared repositories."""
    root = Path(handoff_dir).expanduser().resolve()
    if not root.is_dir():
        raise HandoffContractError(f"handoff directory does not exist: {root}")
    return HandoffBundle(
        root=root,
        workload=_read_document(root, WORKLOAD_FILENAME),
        serving_context=_read_document(root, SERVING_CONTEXT_FILENAME),
        trace_evidence=_read_document(root, TRACE_EVIDENCE_FILENAME),
        repositories=_read_repositories(root),
    )


__all__ = ["read_handoff"]
