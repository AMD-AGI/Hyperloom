# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Orchestration-memory I/O: capture and per-cycle prompt reseeding."""

from __future__ import annotations

import logging as _logging

from ..collaborator import CoordinatorCollaborator
from ..prompts import write_prompt_snapshot as _write_prompt_snapshot
from ..state.orchestration_memory import MEMORY_REQUEST_PROMPT, build_memory_record, parse_memory_reply

log = _logging.getLogger(__name__)

__all__ = ["CycleMemoryCollaborator"]


class CycleMemoryCollaborator(CoordinatorCollaborator):
    """Orchestration-memory capture, directive fallback, and cycle prompt reseeding."""

    async def _capture_cycle_memory(self) -> bool:
        """Ask Orchestration for the finished cycle's working memory and persist it.

        The turn is stateless, so it carries the same full state projection a
        reactor pass does; the reply is the only place
        ``orchestration_memory.next_cycle_directive`` is produced.

        Returns:
            ``True`` when a record was persisted, ``False`` when no
            orchestration backend is configured.
        """
        backend = self.backends.get("orchestration")
        if backend is None:
            return False
        result = await backend.run(
            prompt=f"{await self._coord.conversation._compose_prompt('orchestration')}\n\n{MEMORY_REQUEST_PROMPT}",
            system_prompt=await self._coord.conversation._load_system_prompt("orchestration"),
            tools=[],
            max_turns=0,
            allow_no_intent=True,
        )
        state = self.shared_state
        record = build_memory_record(
            parse_memory_reply(getattr(result, "raw_text", "") or ""),
            tick=int(state.tick or 0),
            previous=dict(state.orchestration_memory or {}),
        )
        if record.get("parse_error"):
            log.warning(
                "_capture_cycle_memory: %s; carrying the previous cycle's memory forward",
                record["parse_error"],
            )
        state.orchestration_memory = record
        return True

    def _reseed_orch_prompt_for_cycle(self) -> bool:
        """Rebuild the orchestration system prompt for the new macro-cycle.

        Injects the freshly-captured ``next_cycle_directive`` and the deterministic
        cycle strategy into a rebuilt prompt and snapshots the installed scope.
        Skips a user-supplied ``--orch-prompt``. Returns True when reseeded.
        """
        orch_prompt = self._coord.orch_prompt
        if orch_prompt.is_user_supplied:
            return False
        rebuild = orch_prompt.rebuild
        if rebuild is None:
            return False
        state = self.shared_state
        cycle = int(state.macro_cycle or 0)
        directive = str((dict(state.orchestration_memory or {})).get("next_cycle_directive", "") or "")
        log_rows = list(state.cycle_strategy_log or [])
        prior_cycles = [r for r in log_rows if isinstance(r, dict) and int(r.get("cycle", -1) or -1) != cycle]
        focus_plan = self._coord.phase_macro_cycle._plan_cycle_focus()
        focus_plan["prior_cycles"] = prior_cycles[-5:]
        new_prompt = rebuild(
            macro_cycle=cycle,
            cycle_directive=directive,
            cycle_strategy=focus_plan,
            phase=state.phase,
        )
        orch_prompt.set("orchestration", new_prompt)
        _write_prompt_snapshot(
            self.session_dir, "orchestration", new_prompt, phase=state.phase, macro_cycle=int(state.macro_cycle or 0)
        )
        return True
