# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Orchestration-memory I/O: capture and per-cycle prompt reseeding."""

from __future__ import annotations

import logging as _logging

from ..collaborator import CoordinatorCollaborator
from ..state.orchestration_memory import MEMORY_REQUEST_PROMPT, build_memory_record, parse_memory_reply

log = _logging.getLogger(__name__)

__all__ = ["CycleMemoryCollaborator"]


class CycleMemoryCollaborator(CoordinatorCollaborator):
    """Orchestration-memory capture and cycle prompt reseeding."""

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

