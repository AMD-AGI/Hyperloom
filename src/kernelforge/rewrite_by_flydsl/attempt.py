# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Attempt-scoped producer state inside the caller's workspace."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from kernelforge.rewrite_by_flydsl.protocol import ATTEMPT_ROOT_DIR


@dataclass(frozen=True)
class AttemptWorkspace:
    """One rewrite attempt's private directory inside the caller's workspace."""

    workspace: Path
    attempt_id: str

    @property
    def relative_root(self) -> str:
        return f"{ATTEMPT_ROOT_DIR}/{self.attempt_id}"

    @property
    def root(self) -> Path:
        return self.workspace / ATTEMPT_ROOT_DIR / self.attempt_id

    @property
    def temporary_paths(self) -> list[str]:
        """Workspace-relative paths the consumer may reclaim."""
        return [self.relative_root]

    def candidate_path(self, name: str) -> Path:
        """Resolve the candidate kernel inside this attempt's directory."""
        cleaned = str(name).strip()
        if not cleaned:
            raise ValueError("the FlyDSL kernel name must not be empty")
        candidate = (self.root / cleaned).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"the FlyDSL kernel name escapes the attempt directory: {name}")
        return candidate


def create_attempt_workspace(
    workspace: str | Path,
    *,
    attempt_id: str = "",
) -> AttemptWorkspace:
    """Create this attempt's private directory under the caller's workspace."""
    attempt = AttemptWorkspace(
        workspace=Path(workspace).resolve(),
        attempt_id=(attempt_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"),
    )
    attempt.root.mkdir(parents=True, exist_ok=True)
    return attempt


def export_import_path(attempt: AttemptWorkspace) -> None:
    """Make the attempt directory importable by every driver forge launches."""
    entries = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    root = str(attempt.root)
    if root in entries:
        return
    os.environ["PYTHONPATH"] = os.pathsep.join([root, *entries])
