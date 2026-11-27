# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CYCLE_END phase handler: the stack work that closes a macro-cycle, and the next cycle's handoff."""

from __future__ import annotations
import logging as _logging
from typing import TYPE_CHECKING, Any

from ..kernel import kernel_decisions
from ..state.orchestration_memory import CYCLE_DIRECTIVE_REQUEST, build_cycle_memory
from ..collaborator import CoordinatorCollaborator
from . import machine_state as _phase_state
from . import phase_transitions

if TYPE_CHECKING:
    from .machine import Transition

log = _logging.getLogger(__name__)

#: The ``from_phase`` and reason of the entry a resumed session replays, as KERNEL's resume re-entry names it.
_RESUME = "resume"


class CycleEndPhase(CoordinatorCollaborator):
    """CYCLE_END phase handler: settles the KEPT stack at the end of each macro-cycle."""

    #: Set once this process has checked a resumed CYCLE_END for entry work that never finished.
    _resume_checked: bool = False

    async def pump(self) -> None:
        """Ask Orchestration, once per macro-cycle, how the next cycle should open."""
        state = self.shared_state
        if (
            "orchestration" not in self.backends
            or self.orch_prompt.is_user_supplied
            or state.orchestration_memory.get("for_cycle") == state.macro_cycle
        ):
            return
        feasible, _ = phase_transitions.cycle_reloop_decision(state)
        if not feasible:
            return
        replies: list[Any] = []

        async def _directive_turn() -> None:
            replies.append(await self._coord.reactor_pass("orchestration", request=CYCLE_DIRECTIVE_REQUEST))

        await self._coord.await_within_session_bound(_directive_turn, stage="reactor:orchestration")
        state.orchestration_memory = build_cycle_memory(replies[0] if replies else None, cycle=state.macro_cycle)
        log.info(
            "cycle %d handoff: directive=%r parse_error=%r",
            state.macro_cycle,
            state.orchestration_memory["next_cycle_directive"][:80],
            state.orchestration_memory["parse_error"],
        )

    async def on_enter_cycle_end(self, tr: "Transition") -> None:
        """Settle the stack, mark this cycle's end settled, and hand the cycle off."""
        await self._settle_stack()
        state = self.shared_state
        state.cycle_end_settled_cycle = int(state.macro_cycle or 0)
        state.save(self.session_dir)
        log.info("CYCLE_END entry (from=%s): cycle %d settled.", tr.from_phase or "<unknown>", state.macro_cycle)
        # A settled CYCLE_END is left on the next advance, before this phase's pump runs again.
        await self.pump()

    async def _settle_stack(self) -> None:
        """Integrate the kernel KEEPs this cycle left queued."""
        if kernel_decisions.has_keep_pending_integrate(self.shared_state):
            await self._coord.phase_kernel_stack.drain_pending_keep_integrates()

    async def resume_unsettled_cycle_end(self) -> None:
        """Re-fire the CYCLE_END entry hook, once per resumed process, when this cycle's end never settled.

        A resume never re-enters the current phase, and CYCLE_END has no budget of its own, so a leg that stopped
        inside the entry hook would otherwise wait in CYCLE_END for a settle nothing makes. Acts on the process's first
        phase decision only, before it is taken; a session that reaches CYCLE_END later in the process enters it
        through the transition's own hook.
        """
        if self._resume_checked:
            return
        self._resume_checked = True
        state = self.shared_state
        if not self._coord.writeback.resumed_from.get("is_resume"):
            return
        if (state.phase or "").strip().upper() != _phase_state.PHASE_CYCLE_END:
            return
        if int(state.cycle_end_settled_cycle) == int(state.macro_cycle or 0):
            return
        log.info("resume: CYCLE_END of cycle %d never settled; re-entering CYCLE_END.", state.macro_cycle)
        from .machine import Transition

        await self.on_enter_cycle_end(
            Transition(
                from_phase=_RESUME,
                to_phase=_phase_state.PHASE_CYCLE_END,
                reason=_RESUME,
                evidence={},
            )
        )
