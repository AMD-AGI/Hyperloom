# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read the Markdown handoff produced by Hyperloom."""

from __future__ import annotations

from pathlib import Path

from kernelforge.kernel_rewrite_controller.contracts import (
    HandoffBundle,
    HandoffContractError,
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


def read_handoff(handoff_dir: str | Path) -> HandoffBundle:
    """Read all required handoff documents without interpreting their content."""
    root = Path(handoff_dir).expanduser().resolve()
    if not root.is_dir():
        raise HandoffContractError(f"handoff directory does not exist: {root}")
    return HandoffBundle(
        root=root,
        workload=_read_document(root, WORKLOAD_FILENAME),
        serving_context=_read_document(root, SERVING_CONTEXT_FILENAME),
        trace_evidence=_read_document(root, TRACE_EVIDENCE_FILENAME),
    )


__all__ = ["read_handoff"]
