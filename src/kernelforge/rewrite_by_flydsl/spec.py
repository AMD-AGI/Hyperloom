# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rewrite specification — the normalized description of one cross-language"""

from __future__ import annotations

from kernelforge.loop.scoring import DEFAULT_SNR_THRESHOLD_DB

from dataclasses import dataclass, field
from pathlib import Path

from kernelforge.rewrite_by_flydsl import protocol


@dataclass
class RewriteSpec:
    """Everything the rewrite pipeline needs, resolved once at ingest."""

    # Stable logical identity of the operation, as the caller names it; it may carry a namespace or punctuation.
    op_name: str

    # Source (to rewrite) — absolute paths inside the workspace.
    source_kernel: str  # e.g. /ws/softmax.py or /ws/attention.hip
    target_functions: list[str]  # kernel entry names, e.g. ["softmax_kernel_online"]
    # Host callable in the source that runs the kernel (a hint shown to the port agent).
    source_entry: str = ""

    # One of ``protocol.SUPPORTED_SOURCE_LANGUAGES``, resolved at ingest.
    source_language: str = ""

    # Produced file (this layer only rewrites into FlyDSL).
    flydsl_kernel: str = ""  # e.g. /ws/kernel.py (the file the agent writes)

    # Shapes driving correctness (vs the source oracle) and benchmark.
    shapes: list[dict] = field(default_factory=list)

    # Correctness gate.
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB

    # Workspace root (git repo the loop keep/reverts in).
    workspace: str = "."

    @property
    def operator_slug(self) -> str:
        """Legal identifier fragment derived from the logical operator name."""
        return protocol.operator_slug(self.op_name)

    @property
    def builder_symbol(self) -> str:
        """The FlyDSL factory symbol the ported kernel must expose."""
        return protocol.builder_symbol(self.op_name)

    @property
    def source_kernel_name(self) -> str:
        return Path(self.source_kernel).name

    @property
    def flydsl_kernel_name(self) -> str:
        return Path(self.flydsl_kernel).name

    @property
    def flydsl_kernel_relpath(self) -> str:
        """Candidate path relative to the workspace, for prompts and logs."""
        try:
            return Path(self.flydsl_kernel).resolve().relative_to(Path(self.workspace).resolve()).as_posix()
        except ValueError:
            return self.flydsl_kernel_name
