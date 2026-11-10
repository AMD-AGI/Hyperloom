# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One forge-rewrite invocation, grouped by the stage that reads each part."""

from __future__ import annotations

from dataclasses import dataclass

from kernelforge.workflows.flydsl_rewrite.ingest import RewriteSubject
from kernelforge.workflows.flydsl_rewrite.optimize import OptimizeOptions


@dataclass(frozen=True)
class DriverSetup:
    """The measurement driver, whether a non-conforming one is prepared, and the evidence preparation reads."""

    driver: str
    prepare_driver: bool = True
    invocation_spec_file: str = ""


@dataclass(frozen=True)
class RewriteWorkspace:
    """The git workspace and where the forge experiments are written."""

    workspace_dir: str
    experiments_dir: str


@dataclass(frozen=True)
class RewriteBudget:
    """The rewrite's runtime budget and its absolute deadline."""

    max_hours: float = 1.0
    deadline_unix: float | None = None


@dataclass(frozen=True)
class ApplybackSwitch:
    """Whether the apply-back stage runs, and how many integration sessions it may use."""

    applyback: bool = True
    max_applyback_attempts: int = 2


@dataclass(frozen=True)
class RewriteRun:
    """One rewrite: what it ports, its driver and workspace, its budget, and how PORT, OPTIMIZE and apply-back run."""

    subject: RewriteSubject
    driver: DriverSetup
    workspace: RewriteWorkspace
    budget: RewriteBudget = RewriteBudget()
    optimize: OptimizeOptions = OptimizeOptions()
    roofline_ceiling: bool = False
    applyback: ApplybackSwitch = ApplybackSwitch()
    applyback_import_modules: list[str] | tuple[str, ...] = ()
    rewrite_kb_enabled: bool = True
    flydsl_kernel_name: str = "kernel.py"
    max_port_attempts: int = 3
    permission_mode: str | None = None
    result_json: str | None = None
