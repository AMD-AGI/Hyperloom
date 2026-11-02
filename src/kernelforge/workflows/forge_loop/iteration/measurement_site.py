# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The checkout one candidate is built, validated and benchmarked in, with the task's paths inside it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kernelforge.evaluation.jit_rebuild import tracked_source_changes


@dataclass(frozen=True)
class MeasurementSite:
    """The workspace one candidate is built, validated and benchmarked in, with its paths inside it."""

    workspace_dir: str
    driver_script: str
    build_dir: str | None
    declared_sources: tuple[str, ...]

    def rebased(self, workspace_dir: str) -> "MeasurementSite":
        """The same task paths inside another checkout of the same repository."""
        return MeasurementSite(
            workspace_dir=str(workspace_dir),
            driver_script=self.relocate(self.driver_script, workspace_dir),
            build_dir=self.relocate(self.build_dir, workspace_dir) if self.build_dir else None,
            declared_sources=tuple(self.relocate(path, workspace_dir) for path in self.declared_sources),
        )

    def jit_sources(self) -> list[str]:
        """Declared hints plus actual tracked edits that may require recompilation."""
        return list(dict.fromkeys([*self.declared_sources, *tracked_source_changes(self.workspace_dir)]))

    def relocate(self, path: str, workspace_dir: str) -> str:
        """The path ``path`` of this site names inside another checkout ``workspace_dir``."""
        root = Path(self.workspace_dir).resolve()
        candidate = Path(path)
        absolute = (candidate if candidate.is_absolute() else root / candidate).resolve()
        try:
            relative = absolute.relative_to(root)
        except ValueError:
            # A path outside the workspace (a system driver, a shared build tree) is the same file from every checkout.
            return str(path)
        return str(Path(workspace_dir) / relative)
