# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Attempt-local state for one specialist integration, never shared by tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...enablement.runtime.stack_actions import EnablementStackAction, ProvisionResult
    from ...state.shared_state import SharedState


@dataclass(slots=True)
class IntegrateAttempt:
    """Own the resolved inputs and acquired resources of one executor invocation."""

    task_id: str
    specialist_task_id: str = ""
    specialist_workspace: Path | None = None
    shared_state: SharedState | None = None
    done_payload: dict[str, Any] | None = field(default_factory=dict)
    provision_result: ProvisionResult | None = None
    stack_action: EnablementStackAction | None = None
    localization_patches: list[Path] = field(default_factory=list)
    localization_touched: list[str] = field(default_factory=list)
    base_sha_by_root: dict[str, str] = field(default_factory=dict)
    output_root: Path | None = None
    framework_root: Path | None = None
    applied: list[Path] = field(default_factory=list)
    applied_artifacts: list[dict[str, Any]] = field(default_factory=list)
    extra_envs_applied: dict[str, str] = field(default_factory=dict)
    extra_server_args_applied: str = ""
    dropped_env_overrides: list[str] = field(default_factory=list)
    setup_result: dict[str, Any] = field(default_factory=dict)
    switch_manifest: list[dict[str, Any]] = field(default_factory=list)
    switch_problems: list[str] = field(default_factory=list)
    nogit_patch_backups: list[dict[str, Any]] | None = None
    nogit_backup_root: Path | None = None
    pending: dict[str, Any] = field(default_factory=dict)

    @property
    def attempt_venv_root(self) -> str:
        """The acquired runtime, or no runtime when this attempt did not provision."""
        if self.provision_result is None:
            return ""
        return self.provision_result.runtime.venv_root
