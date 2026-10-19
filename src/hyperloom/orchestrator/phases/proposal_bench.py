# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FRAMEWORK_AGENT's bench of untested specialist proposals and the operator work the Coordinator owns there."""

from __future__ import annotations

import logging as _logging
from collections.abc import Iterable, Mapping
from hashlib import sha1
from typing import Any

from ..collaborator import CoordinatorCollaborator
from ..specialists.domains import CUSTOM_SPECIALIST_KEY
from .dispatch_order import is_operator_work

log = _logging.getLogger(__name__)

__all__ = ["CUSTOM_PROPOSAL_DOMAIN", "ProposalBench", "kb_exposure_of"]

# ``untested_proposal_rows`` names a round's domain without the ``_specialist`` suffix.
CUSTOM_PROPOSAL_DOMAIN: str = CUSTOM_SPECIALIST_KEY.removesuffix("_specialist")


def kb_exposure_of(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The KB exposure of a grid assembled from several specialists' proposals, for its one proposal row.

    The row holds every Experience any of those reads showed. It names a read only when one read fed every proposal.
    """
    rows = list(rows)
    read_ids = {str(row.get("kb_read_id") or "") for row in rows}
    refs: dict[str, Any] = {}
    for row in rows:
        for ref in row.get("kb_rendered_refs") or []:
            if isinstance(ref, Mapping) and str(ref.get("id") or "") not in refs:
                refs[str(ref.get("id") or "")] = dict(ref)
    return {"kb_read_id": read_ids.pop() if len(read_ids) == 1 else "", "kb_rendered_refs": list(refs.values())}


class ProposalBench(CoordinatorCollaborator):
    """Benches untested proposals, the operator-defined specialist's first, and tracks that operator work."""

    # Cap on specialist proposal rows benched per automatic explore grid.
    _AUTO_EXPLORE_GRID_CAP: int = 4

    @staticmethod
    def _has_operator_work(tasks: Iterable[Any]) -> bool:
        """Whether any of ``tasks`` is the operator-defined specialist or the bench of its proposals."""
        return any(is_operator_work(t.kind, t.params) for t in tasks)

    def _custom_proposal_rows(self) -> list[dict[str, Any]]:
        """Untested proposals of the operator-defined specialist, capped at one grid."""
        rows = [r for r in self.shared_state.untested_proposal_rows() if r["domain"] == CUSTOM_PROPOSAL_DOMAIN]
        return rows[: self._AUTO_EXPLORE_GRID_CAP]

    def _bench_key(self, rows: list[dict[str, Any]], *, operator_priority: bool) -> str:
        """Idempotency key naming the fingerprint set, so a grid that recorded nothing is not re-enqueued."""
        fp_hash = sha1(",".join(sorted(row["fingerprint"] for row in rows)).encode(), usedforsecurity=False)
        prefix = "auto-explore-custom" if operator_priority else "auto-explore"
        return f"{prefix}-c{int(self.shared_state.macro_cycle or 0)}-{fp_hash.hexdigest()[:12]}"

    async def operator_work_pending(self) -> bool:
        """Whether the operator-defined specialist or the bench of its proposals has not run yet."""
        if self._has_operator_work((*await self.tasks.queued(), *await self.tasks.running())):
            return True
        custom = self._custom_proposal_rows()
        if not custom:
            return False
        return await self.tasks.find_by_idempotency_key(self._bench_key(custom, operator_priority=True)) is None

    async def _maybe_finish_custom_only_cycle(self) -> None:
        """In custom-specialist-only mode, hint ``skip_to_sweep`` once this cycle's custom work has all run."""
        from ..specialists.domains import custom_specialist_key
        from ..state.shared_state import ESCALATE_HINT_SKIP_TO_SWEEP
        from ..state.task_registry import TERMINAL_STATES

        state = self.shared_state
        key = custom_specialist_key(int(state.macro_cycle or 0), str(state.custom_specialist_prompt or "").strip())
        custom = await self.tasks.find_by_idempotency_key(key)
        # A cycle whose dispatch admission refused has nothing left to run.
        refused = custom is None and key == self._coord.specialist_dispatch.refused_custom_key
        if not refused and (custom is None or custom.state not in TERMINAL_STATES):
            return
        # A succeeded task's round, which carries its proposals, may not be booked yet.
        if (
            custom is not None
            and custom.state == "succeeded"
            and not any(
                isinstance(r, dict) and r.get("task_id") == custom.task_id for r in state.specialist_rounds or []
            )
        ):
            return
        if await self.operator_work_pending():
            return
        if any(t.kind == "integrate_patch" for t in (*await self.tasks.queued(), *await self.tasks.running())):
            return
        if any(p.action_name == "integrate_patch" for p in self.state.pending_proposals.values()):
            return
        if state.set_pending_escalate_hint(ESCALATE_HINT_SKIP_TO_SWEEP):
            log.info("custom-specialist-only: cycle %s done; leaving FRAMEWORK_AGENT", state.macro_cycle)
