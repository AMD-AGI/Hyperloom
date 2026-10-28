# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FRAMEWORK_AGENT's bench of untested specialist proposals and the operator work the Coordinator owns there."""

from __future__ import annotations

import logging as _logging
from collections.abc import Iterable, Mapping
from hashlib import sha1
from typing import TYPE_CHECKING, Any

from hyperloom.inference_optimizer.session.optimization_journal import Verdict

from ..collaborator import CoordinatorCollaborator
from ..specialists.domains import CUSTOM_SPECIALIST_KEY
from ..state.shared_state import inject_stack_base_params
from ..state.task_registry import TaskSpec
from .dispatch_order import LEVER_BUNDLE_REASON, is_operator_work
from .machine_state import PHASE_FRAMEWORK_AGENT

if TYPE_CHECKING:
    from ..state.task_registry import Task

log = _logging.getLogger(__name__)

_COORDINATOR_SOURCE = "coordinator_internal"

#: A reference batch cancelled (a recovered pod) or failed (a server that did not boot) is enqueued again, this often.
_CYCLE_REFERENCE_ATTEMPTS = 3
_UNRUN_STATES = frozenset({"cancelled", "failed"})


def _never_ran(task: Any) -> bool:
    """Whether a reference batch's task ended without a batch: cancelled, failed, or settled on a failed result."""
    if str(task.state) in _UNRUN_STATES:
        return True
    last = (list(getattr(task, "history", None) or []) or [{}])[-1]
    result = ((last.get("evidence") or {}).get("outcome") or {}).get("result") or {}
    return str(result.get("status") or "") == "failed"


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


def _holds_cycle_batch(tasks: Iterable[Any]) -> bool:
    """Whether a cycle-bench batch is among *tasks*; a reference batch carries an empty arm list."""
    from ..actions.executors.cycle_bench_arm import CYCLE_ARMS_PARAM

    return any(CYCLE_ARMS_PARAM in (getattr(task, "params", None) or {}) for task in tasks)


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

    async def _enqueue_coordinator_explore(self, params: Mapping[str, Any], *, idempotency_key: str) -> "Task | None":
        """Enqueue an explore the Coordinator owns, on the accepted stack; the new task, or ``None`` if the key exists."""
        state = self.shared_state
        task_params = dict(params)
        if state.baseline_config_path:
            task_params["config_path"] = state.baseline_config_path
        inject_stack_base_params(task_params, state, anchor=True)
        benchmark_script = state.accepted_baseline_script()
        if benchmark_script:
            task_params["benchmark_script"] = benchmark_script
        lanes, ttl = self._coord.dispatcher.registry_lanes_ttl("explore")
        task, was_existing = await self.tasks.create_or_return_existing(
            TaskSpec(
                kind="explore",
                params=task_params,
                idempotency_key=idempotency_key,
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
                dispatch_class="coordinator",
            )
        )
        return None if was_existing else task

    async def enqueue_dormant_lever_bundle(self) -> bool:
        """Enqueue one round benching every dormant rewrite lever together; whether a new round was enqueued.

        Keyed on the set of switches, so a set is benched once and a grown set again. Dispatched ahead of queued
        integrates (see :mod:`.dispatch_order`): left to the orchestration it waited behind them, and no explore fit
        the phase they had already filled. Nothing is enqueued into a phase on its way out, which would cancel it.
        """
        from ..actions.executors.explore import dormant_lever_bundle

        variant = dormant_lever_bundle(self.shared_state)
        if variant is None or self._coord.dispatcher.admission_frozen:
            return False
        switches = sha1(",".join(variant["framework_lever_bundle"]).encode(), usedforsecurity=False).hexdigest()[:12]
        task = await self._enqueue_coordinator_explore(
            {"source": _COORDINATOR_SOURCE, "reason": LEVER_BUNDLE_REASON, "grid": [variant]},
            idempotency_key=f"lever-bundle-c{int(self.shared_state.macro_cycle or 0)}-{switches}",
        )
        if task is not None:
            log.info(
                "lever bundle: enqueued explore task_id=%s over %s", task.task_id, variant["framework_lever_bundle"]
            )
        return task is not None

    async def on_task_settled(self, task: Task, result: Any, verdict: Verdict) -> None:
        """Hand a promoted settlement to FRAMEWORK: an integrate's outcome, and a moved stack's cycle batch."""
        if task.kind == "integrate_patch":
            await self.on_integrate_patch_settled(task, result, verdict)
        if verdict is Verdict.ADOPTED:
            await self.on_stack_adopted()

    async def on_stack_adopted(self) -> None:
        """Queue the moved stack's cycle batch before the dispatcher's next pass hands the lanes to an integrate."""
        if str(self.shared_state.phase or "").upper() == PHASE_FRAMEWORK_AGENT:
            await self.enqueue_cycle_batch()

    async def enqueue_cycle_batch(self) -> bool:
        """Enqueue one cycle-bench batch over the patches whose levers await the screen; whether one was enqueued.

        One batch at a time, so a grown set does not re-bench the arms already queued. A lone patch waits for a second
        while the serving lanes have other work, and is screened alone once they would otherwise idle.
        """
        from ..actions.executors import cycle_bench_arm as cba
        from ..actions.executors._integrate_vocabulary import ATTRIBUTION_CYCLE_INCONCLUSIVE

        arms = cba.cycle_arms(list(self.shared_state.authored_framework_levers or []))
        if not arms:
            return await self._enqueue_cycle_reference()
        if self._coord.dispatcher.admission_frozen:
            # A batch queued now would be cancelled by the transition; the levers go back to replay attribution.
            for switch in (s for arm in arms for s in arm["envs"]):
                self.shared_state.record_framework_lever_attribution(
                    switch, gain_pct=None, source=ATTRIBUTION_CYCLE_INCONCLUSIVE
                )
            return False
        live = [*await self.tasks.queued(), *await self.tasks.running()]
        if _holds_cycle_batch(live):
            return False
        serving = ("server_lifecycle", "benchmark_lane")
        if len(arms) < 2 and any(lane in serving for t in live for lane in (getattr(t, "requires_lanes", ()) or ())):
            return False
        names = sha1(",".join(arm["name"] for arm in arms).encode(), usedforsecurity=False).hexdigest()[:12]
        params = {
            "source": _COORDINATOR_SOURCE,
            "reason": cba.CYCLE_BATCH_REASON,
            cba.CYCLE_ARMS_PARAM: arms,
        }
        task = await self._enqueue_coordinator_explore(
            params, idempotency_key=f"cycle-batch-c{int(self.shared_state.macro_cycle or 0)}-{names}"
        )
        if task is not None:
            log.info("cycle bench: enqueued batch task_id=%s over %s", task.task_id, [arm["name"] for arm in arms])
        return task is not None

    async def _enqueue_cycle_reference(self) -> bool:
        """Enqueue the two bases alone for an AgentX stack no batch has measured, so the patches admitted dormant
        on it are held to the tree before them; whether one was enqueued."""
        from hyperloom.common.perf_metric import agentx_active

        from ..actions.executors import cycle_bench_arm as cba

        state = self.shared_state
        stack_len = len(state.optimization_stack or [])
        if (
            not cba.cycle_bench_enabled()
            or not agentx_active(benchmark_mode=state.benchmark_mode)
            or state.cycle_screen_refused
            or (state.cycle_reference or {}).get("stack_len") == stack_len
            or self._coord.dispatcher.admission_frozen
        ):
            return False
        if _holds_cycle_batch([*await self.tasks.queued(), *await self.tasks.running()]):
            return False
        key = await self._cycle_reference_key(f"cycle-reference-c{int(state.macro_cycle or 0)}-s{stack_len}")
        if key is None:
            return False
        params = {"source": _COORDINATOR_SOURCE, "reason": cba.CYCLE_BATCH_REASON, cba.CYCLE_ARMS_PARAM: []}
        task = await self._enqueue_coordinator_explore(params, idempotency_key=key)
        if task is not None:
            log.info(
                "cycle bench: enqueued the reference batch task_id=%s for stack length %d", task.task_id, stack_len
            )
        return task is not None

    async def _cycle_reference_key(self, base: str) -> str | None:
        """The key of this stack's next reference batch: a new attempt only once the last ended without running."""
        for attempt in range(_CYCLE_REFERENCE_ATTEMPTS):
            key = base if attempt == 0 else f"{base}-a{attempt}"
            task = await self.tasks.find_by_idempotency_key(key)
            if task is None or not _never_ran(task):
                return key
        return None

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
