# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Base class for Coordinator collaborators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from hyperloom.orchestrator.bus.message_bus import MessageBus
    from hyperloom.orchestrator.bus.resource_lock import ResourceLockManager
    from hyperloom.orchestrator.bus.storage.connection import SqliteConnection
    from hyperloom.orchestrator.bus.gpu_pool import SpecialistGpuPool
    from hyperloom.orchestrator.loop.sub_agent_runner import SubAgentRunner
    from hyperloom.orchestrator.policy.gate import PolicyGate
    from hyperloom.orchestrator.state.round_store import RoundStore
    from hyperloom.orchestrator.state.shared_state import SharedState
    from hyperloom.orchestrator.state.task_registry import TaskRegistry
    from hyperloom.orchestrator.loop.coordinator import Coordinator, CoordinatorState


class OrchestrationPrompt:
    """Owns the orchestration system-prompt overrides, rebuild closure, and snapshot writes."""

    def __init__(
        self,
        overrides: dict[str, str],
        *,
        is_user_supplied: bool = False,
        rebuild: Callable[..., str] | None = None,
    ) -> None:
        self.overrides: dict[str, str] = overrides
        self.is_user_supplied: bool = is_user_supplied
        self.rebuild: Callable[..., str] | None = rebuild

    def get(self, agent_name: str) -> str | None:
        """Return the override for *agent_name*, or None when absent."""
        return self.overrides.get(agent_name)

    def set(self, agent_name: str, prompt: str) -> None:
        """Install a new prompt for *agent_name*."""
        self.overrides[agent_name] = prompt


class CoordinatorCollaborator:
    """An object that borrows the Coordinator's infrastructure for its own methods."""

    def __init__(self, coordinator: "Coordinator") -> None:
        self._coord = coordinator

    # ------------------------------------------------------------------ #
    # Persistent session state and core infrastructure                     #
    # ------------------------------------------------------------------ #

    @property
    def shared_state(self) -> "SharedState":
        return self._coord.shared_state

    @property
    def session_dir(self):
        return self._coord.session_dir

    @property
    def tasks(self) -> "TaskRegistry":
        return self._coord.tasks

    @property
    def bus(self) -> "MessageBus":
        return self._coord.bus

    @property
    def locks(self) -> "ResourceLockManager":
        return self._coord.locks

    @property
    def db(self) -> "SqliteConnection":
        return self._coord.db

    @property
    def sub(self) -> "SubAgentRunner":
        return self._coord.sub

    @property
    def rounds(self) -> "RoundStore":
        return self._coord.rounds

    @property
    def policy(self) -> "PolicyGate":
        return self._coord.policy

    @property
    def gpu_specialist_pool(self) -> "SpecialistGpuPool":
        return self._coord.gpu_specialist_pool

    @property
    def framework_gpu_pool(self) -> "SpecialistGpuPool":
        return self._coord.framework_gpu_pool

    @property
    def knowledge_plane(self):
        return self._coord.knowledge_plane

    @property
    def backends(self):
        return self._coord.backends

    @property
    def state(self) -> "CoordinatorState":
        return self._coord.state

    @property
    def orch_prompt(self) -> "OrchestrationPrompt":
        return self._coord.orch_prompt

    @property
    def recipe_kb(self):
        return self._coord.recipe_kb

    @property
    def role_registry(self):
        return self._coord.role_registry

    @property
    def action_registry(self):
        return self._coord.action_registry

    @property
    def cursors(self):
        return self._coord.cursors
