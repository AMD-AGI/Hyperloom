# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Base class for Coordinator collaborators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

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

    # ------------------------------------------------------------------ #
    # Private coordinator state read by specific collaborators             #
    # ------------------------------------------------------------------ #

    @property
    def _journal(self):
        return getattr(self._coord, "_journal", None)

    @_journal.setter
    def _journal(self, value):
        self._coord._journal = value

    @property
    def _resumed_from(self):
        return self._coord._resumed_from

    @_resumed_from.setter
    def _resumed_from(self, value):
        self._coord._resumed_from = value

    @property
    def _dispatcher_poll_sec(self):
        return self._coord._dispatcher_poll_sec

    # ------------------------------------------------------------------ #
    # Collaborator objects (other extracted subsystems)                    #
    # ------------------------------------------------------------------ #

    @property
    def dispatcher(self):
        return self._coord.dispatcher

    @property
    def writeback(self):
        return self._coord.writeback

    @property
    def phase_kernel(self):
        return self._coord.phase_kernel

    @property
    def phase_framework(self):
        return self._coord.phase_framework

    @property
    def phase_machine(self):
        return self._coord.phase_machine

    @property
    def phase_prelude(self):
        return self._coord.phase_prelude

    @property
    def phase_close(self):
        return self._coord.phase_close

    @property
    def phase_sweep(self):
        return self._coord.phase_sweep

    @property
    def phase_internal(self):
        return self._coord.phase_internal

    @property
    def phase_kernel_stack(self):
        return self._coord.phase_kernel_stack

    @property
    def phase_macro_cycle(self):
        return self._coord.phase_macro_cycle

    @property
    def specialist_dispatch(self):
        return self._coord.specialist_dispatch

    @property
    def conversation(self):
        return self._coord.conversation

    @property
    def router(self):
        return self._coord.router

    @property
    def proposals(self):
        return self._coord.proposals

    @property
    def maintenance(self):
        return self._coord.maintenance

    @property
    def build_lifecycle(self):
        return self._coord.build_lifecycle

    @property
    def enablement_lane(self):
        return self._coord.enablement_lane

    @property
    def enablement_params(self):
        return self._coord.enablement_params

    @property
    def enablement_build(self):
        return self._coord.enablement_build

    @property
    def enablement_revalidation(self):
        return self._coord.enablement_revalidation

    @property
    def gpu_lanes(self):
        return self._coord.gpu_lanes

    @property
    def gap_refresh(self):
        return self._coord.gap_refresh

    @property
    def cycle_memory(self):
        return self._coord.cycle_memory

    # ------------------------------------------------------------------ #
    # Coordinator helper methods used across collaborators                 #
    # ------------------------------------------------------------------ #

    @property
    def _kb_hardware_slug(self):
        return self._coord._kb_hardware_slug

    @property
    def _current_objective(self):
        return self._coord._current_objective

    @_current_objective.setter
    def _current_objective(self, value):
        self._coord._current_objective = value

    @property
    def _run_deadline(self):
        return self._coord._run_deadline

    @_run_deadline.setter
    def _run_deadline(self, value):
        self._coord._run_deadline = value

    @property
    def _record_coordinator_exception(self):
        return self._coord._record_coordinator_exception

    @property
    def _run_started_monotonic(self):
        return self._coord._run_started_monotonic

    @property
    def _proposal_scorer(self):
        return self._coord._proposal_scorer

    @property
    def _stop(self):
        return self._coord._stop

    @property
    def _cycle_soft_restart(self) -> bool:
        return self._coord._cycle_soft_restart

    @property
    def reconciler(self):
        return self._coord.reconciler

    @property
    def stop_classification(self):
        return self._coord.stop_classification
