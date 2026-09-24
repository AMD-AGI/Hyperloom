# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Orchestration-memory I/O: capture, directive fallback, and per-cycle prompt reseeding."""

from __future__ import annotations

import logging as _logging
from datetime import datetime, timezone

from ..prompts import write_prompt_snapshot as _write_prompt_snapshot
from ..state.orchestration_memory import MEMORY_REQUEST_PROMPT, build_memory_record, parse_memory_reply

log = _logging.getLogger(__name__)

__all__ = ["CycleMemoryCollaborator"]


class CycleMemoryCollaborator:
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
            prompt=f"{await self._compose_prompt('orchestration')}\n\n{MEMORY_REQUEST_PROMPT}",
            system_prompt=await self._load_system_prompt("orchestration"),
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
        state.orchestration_memory_history = [*(state.orchestration_memory_history or []), record][-10:]
        return True

    def _cycle_directive_fallback(self) -> str:
        """Render a deterministic cycle focus from ``_plan_cycle_focus``.

        Used when the memory capture produced no ``next_cycle_directive``; keeps
        every cycle's CYCLE DIRECTIVE section grounded in real telemetry.
        """
        planned = self._plan_cycle_focus()
        focus = str(planned.get("focus") or "").strip()
        if not focus:
            return ""
        parts = [f"focus={focus}"]
        rationale = str(planned.get("rationale") or "").strip()
        if rationale:
            parts.append(rationale)
        bottleneck = str(planned.get("bottleneck_at_start") or "").strip()
        if bottleneck:
            parts.append(f"bottleneck={bottleneck}")
        saturated = planned.get("saturated_at_start") or []
        if saturated:
            parts.append(f"deprioritize saturated={list(saturated)}")
        return "; ".join(parts)

    def _reseed_orch_prompt_for_cycle(self) -> bool:
        """Rebuild the orchestration system prompt for the new macro-cycle.

        Injects the freshly-captured ``next_cycle_directive`` (or a deterministic
        fallback) into a rebuilt prompt, mutates ``system_prompt_overrides``,
        snapshots the installed scope, and records the directive in the
        ``cycle_directive_history`` ring. Skips a user-supplied
        ``--orch-prompt``. Best-effort; returns True when reseeded.
        """
        if getattr(self, "_orch_prompt_is_user_supplied", False):
            return False
        rebuild = getattr(self, "_rebuild_orch_prompt", None)
        if rebuild is None:
            return False
        state = self.shared_state
        cycle = int(getattr(state, "macro_cycle", 0) or 0)
        directive = str((dict(getattr(state, "orchestration_memory", {}) or {})).get("next_cycle_directive", "") or "")
        source = "llm"
        if not directive:
            directive = self._cycle_directive_fallback()
            source = "deterministic"
        new_prompt = rebuild(macro_cycle=cycle, cycle_directive=directive, phase=state.phase)
        overrides = getattr(self, "system_prompt_overrides", None)
        if not isinstance(overrides, dict):
            return False
        overrides["orchestration"] = new_prompt
        _write_prompt_snapshot(self.session_dir, "orchestration", new_prompt, phase=state.phase)
        history = list(getattr(state, "cycle_directive_history", []) or [])
        history.append(
            {
                "cycle": cycle,
                "directive": directive,
                "source": source,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        state.cycle_directive_history = history[-10:]
        return True
