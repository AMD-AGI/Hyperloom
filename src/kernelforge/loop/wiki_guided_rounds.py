# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the iteration loop does with a ``--wiki-guided`` campaign's knowledge base round.

``wiki_guidance`` holds the state and its rules; this is the loop's side of it: planning the round (building the batch,
then the parts of the groups still open), noting the plan the iteration measures, reading what its session reported,
crediting the measurement, and ending the round, after which every round plans as KernelForge does without the flag.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from kernelforge.knowledge.wiki_tool import KnowledgeWikiTool, withdraw_knowledge_wiki
from kernelforge.loop.run_state import make_event
from kernelforge.loop.wiki_guidance import (
    DECISION_KEEP,
    MAX_PLAYBOOK_ATTEMPTS,
    REASON_NO_BATCH_PLAN,
    REASON_NO_USEFUL_DIRECTION,
    REASON_PLAYBOOK_UNAVAILABLE,
    REASON_ROUND_DONE,
    UNSETTLED_RETRIES,
    GuidanceState,
    WikiGuidanceStore,
    batch_plan,
    cited_items,
    latest_report,
    read_citations,
    record_outcome,
    render_for_implementer,
    render_for_planning,
    settles_nothing,
)
from kernelforge.loop.wiki_playbook import SUPPORT_PROFILING, declined, parse_playbook
from kernelforge.orchestrator.contracts import OrchestrationContext, OrchestrationRunResult
from kernelforge.orchestrator.orchestration import OrchestrationInfrastructureError, OrchestrationOutputError
from kernelforge.orchestrator.wiki_batch import NO_BATCH, OPEN_STATUSES, PHASE_FREE, PHASE_GUIDED, BatchSession

log = logging.getLogger(__name__)

_NOTHING = "nothing"


def _say(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class GuidanceSinks:
    """Where guided planning records what the campaign keeps: its durable events, and a state file it could not write."""

    append_event: Callable[[dict], None]
    persistence_failed: Callable[[str], None]


@dataclass(frozen=True)
class PlannedRound:
    """The context a round planned from and what its planning produced."""

    context: OrchestrationContext
    result: OrchestrationRunResult


class WikiGuidedRounds:
    """A ``--wiki-guided`` campaign's knowledge base round and what each round does with it."""

    def __init__(
        self, workspace_dir: str | Path, *, knowledge_wiki: KnowledgeWikiTool | None, resume: bool, sinks: GuidanceSinks
    ) -> None:
        self.store = WikiGuidanceStore(workspace_dir)
        # A fresh campaign starts its round whatever an earlier campaign left in the workspace; a resumed one continues
        # where it was.
        self.state: GuidanceState = self.store.load() if resume else GuidanceState()
        self.knowledge_wiki = knowledge_wiki
        self.sinks = sinks
        # The plan the current iteration measures, read when its outcome credits the ledger, the batch it implements in
        # its one session, and the parts that session was given.
        self.plan_text = ""
        self.batch: list[str] = []
        self.session = NO_BATCH
        if self.state.phase == PHASE_FREE:
            self._withdraw()
        if not resume:
            # Written now, so a resume after an early crash cannot pick up an earlier campaign's file.
            self._save()

    def start_iteration(self) -> None:
        self.plan_text = ""
        self.batch = []
        self.session = NO_BATCH

    def implementer_note(self) -> str:
        return render_for_implementer(self.state)

    def result_fields(self) -> dict[str, Any]:
        """What an ``iteration_result`` event carries: the planning phase, and the batch the iteration implemented."""
        return {"guidance_phase": self.state.phase, "wiki_batch": list(self.batch) or None}

    def with_guidance(self, context: OrchestrationContext) -> OrchestrationContext:
        """Attach the round's playbook, or once it is over what it settled, to a planning context."""
        guidance = render_for_planning(self.state)
        return context if guidance is None else replace(context, wiki_guidance=guidance)

    async def plan_round(
        self, iteration: int, service, context: OrchestrationContext, *, usage, lanes: int
    ) -> PlannedRound:
        """Plan this round: the knowledge base batch while the round is open, else as KernelForge plans it."""
        batch = await self._open_batch(iteration, service, context, usage=usage)
        if batch:
            guided = self.with_guidance(context)
            # One part per group whatever the configured lane width: the whole batch is one session.
            result = await service.run(guided, usage=usage, lanes=len(batch))
            plans = [str(plan) for plan in result.optimization_plans if str(plan).strip()]
            if result.optimization_plan_executable and plans:
                plan = self._compose(iteration, plans, batch)
                return PlannedRound(guided, replace(result, optimization_plans=(plan,)))
            if "probe_device_hazard" in (result.structured_output_diagnostics or {}):
                # The device is held, so this iteration measures nothing and the round is planned again after it.
                return PlannedRound(guided, result)
            self._end_round(iteration, REASON_NO_BATCH_PLAN)
        context = self.with_guidance(context)
        return PlannedRound(context, await service.run(context, usage=usage, lanes=lanes))

    def note_session_report(self, iteration: int, final_text: str) -> None:
        """Credit a batch from the items its session reported landing and dropping, when it reported them."""
        state = self.state
        if not self.batch or state.playbook is None:
            return
        report = latest_report(str(final_text or ""))
        cites = read_citations(report, state.ledger)
        if cites.any:
            self._note_plan(iteration, report)
        self.sinks.append_event(
            make_event(
                "wiki_batch_report",
                iteration,
                landed=list(cites.claimed) or None,
                dropped=list(cites.skipped) or None,
                untried=list(cites.untried) or None,
                source="session" if cites.any else "plan",
            )
        )
        if cites.any:
            landed, dropped, untried = (
                ", ".join(ids) or _NOTHING for ids in (cites.claimed, cites.skipped, cites.untried)
            )
            _say(f"  [batch] session landed {landed}; dropped {dropped}; did not try {untried}")
        else:
            _say("  [batch] session reported no item lines; the plan's citations are credited")

    def record(self, iteration: int, decision: str, *, detail: str = "") -> None:
        """Credit the measured batch, and end the round unless a kept first batch left items to complete or the
        session's outcome said nothing about the batch for the first time."""
        state = self.state
        if state.phase != PHASE_GUIDED or not self.batch or iteration in state.recorded_iterations:
            return
        changed = record_outcome(state, iteration=iteration, plan_text=self.plan_text, decision=decision)
        kept = decision == DECISION_KEEP
        state.outcomes.append({"iteration": iteration, "decision": decision, "detail": "" if kept else detail[:300]})
        self._save()
        if changed:
            self.sinks.append_event(
                make_event(
                    "wiki_playbook_ledger",
                    iteration,
                    decision=decision,
                    items={item_id: state.ledger[item_id].status for item_id in changed},
                    counts=state.status_counts(),
                )
            )
            _say(
                "  [wiki-guided] ledger: "
                + ", ".join(f"{item_id} {state.ledger[item_id].status}" for item_id in changed)
            )
        settled = [outcome for outcome in state.outcomes if not settles_nothing(outcome["decision"])]
        if settles_nothing(decision) and len(state.outcomes) - len(settled) <= UNSETTLED_RETRIES:
            _say(
                f"  [wiki-guided] the batch's session ended {decision} without measuring it; the next round runs it again"
            )
        elif not (kept and len(settled) == 1 and state.open_groups()):
            self._end_round(iteration, REASON_ROUND_DONE)

    def recover_keep(self, iteration: int) -> None:
        """Credit a KEEP the process that measured it died before crediting; the plan and batch it measured are on
        record."""
        state = self.state
        if state.inflight_iteration != iteration:
            return
        self.plan_text = state.inflight_plan
        self.batch = list(state.batch)
        self.record(iteration, DECISION_KEEP)

    async def _open_batch(self, iteration: int, service, context: OrchestrationContext, *, usage) -> list[str]:
        """The groups this round implements: the builder's batch first, then the groups still open after it."""
        state = self.state
        if state.phase != PHASE_GUIDED:
            return []
        if state.playbook is None and not await self._build(iteration, service, context, usage=usage):
            return []
        state.batch = state.open_groups()
        if not state.batch:
            self._end_round(iteration, REASON_ROUND_DONE)
        self._save()
        return list(state.batch)

    async def _build(self, iteration: int, service, context: OrchestrationContext, *, usage) -> bool:
        """Ask the builder for the batch, once more after an outage of its backend; whether the round has one."""
        builder = getattr(service, "build_wiki_playbook", None)
        if builder is None:
            self._unavailable(iteration, "the orchestration service cannot build a playbook", attempt=0, settled=True)
            return False
        _say("  [wiki-guided] building the batch from the knowledge base and the evidence...")
        for attempt in range(1, MAX_PLAYBOOK_ATTEMPTS + 1):
            try:
                text = await builder(context, usage=usage)
            except OrchestrationInfrastructureError as outage:
                # The backend refused, not the answer: asking again may get one.
                detail = f"{type(outage).__name__}: {outage}"
                self._unavailable(iteration, detail, attempt=attempt, settled=attempt == MAX_PLAYBOOK_ATTEMPTS)
                continue
            except (OrchestrationOutputError, ValueError) as failure:
                # The backend answered and the answer is unusable; asking again would buy the same answer.
                self._unavailable(iteration, f"{type(failure).__name__}: {failure}", attempt=attempt, settled=True)
                return False
            return self._add_build(iteration, text, attempt=attempt)
        return False

    def _add_build(self, iteration: int, text: str, *, attempt: int) -> bool:
        try:
            self.store.save_response(text)
        except OSError:
            log.debug("wiki playbook response not saved", exc_info=True)
        build, notes = parse_playbook(text)
        if build is None:
            if not declined(text):
                self._unavailable(iteration, "; ".join(notes), attempt=attempt, settled=True)
                return False
            self.sinks.append_event(make_event("wiki_playbook_declined", iteration))
            _say("  [wiki-guided] the knowledge base and the evidence support no direction worth a session")
            self._end_round(iteration, REASON_NO_USEFUL_DIRECTION)
            return False
        if notes:
            _say("  [wiki-guided] playbook notes: " + "; ".join(notes))
        self.state.add_build(build)
        groups = [group.group_id for group in build.groups]
        profiling = [item.item_id for item in build.items if item.support == SUPPORT_PROFILING]
        self.sinks.append_event(
            make_event(
                "wiki_playbook_built",
                iteration,
                items=list(build.item_ids()),
                groups=groups,
                profiling_only=profiling or None,
            )
        )
        _say(
            f"  [wiki-guided] batch: {len(build.items)} item(s) in {len(groups)} group(s) ({', '.join(groups)})"
            + (f"; supported by profiling only: {', '.join(profiling)}" if profiling else "")
        )
        return True

    def _compose(self, iteration: int, plans: Sequence[str], batch: Sequence[str]) -> str:
        """Turn the round's part plans into this iteration's one session plan; ``batch`` is the round's groups."""
        groups = [self._plan_group(plan) for plan in plans]
        if len(plans) == len(batch) and not all(groups):
            # One part per group, in batch order: a part that cites no item still has its group.
            groups = [group or batch_group for group, batch_group in zip(groups, batch)]
        # The batch the measurement decides about is the round's, whatever its part plans happen to cite.
        self.batch = list(batch)
        self.session = BatchSession(tuple((group, self._open_items(group)) for group in groups if group))
        self.sinks.append_event(
            make_event("wiki_batch_planned", iteration, parts=len(plans), groups=[group or None for group in groups])
        )
        _say(f"  [batch] {len(plans)} part(s) in one session: " + ", ".join(group or "?" for group in groups))
        plan = batch_plan([(self._group_label(group), plan) for group, plan in zip(groups, plans)])
        self._note_plan(iteration, plan)
        return plan

    def _note_plan(self, iteration: int, text: str) -> None:
        """Remember the plan this iteration measures, durably, so a KEEP recovery can credit it."""
        self.plan_text = text or ""
        self.state.inflight_iteration = iteration
        self.state.inflight_plan = self.plan_text
        self._save()

    def _unavailable(self, iteration: int, error: str, *, attempt: int, settled: bool) -> None:
        self.sinks.append_event(
            make_event("wiki_playbook_unavailable", iteration, error=error[:500], attempt=attempt, settled=settled)
        )
        _say(f"  [wiki-guided] playbook unavailable ({error})")
        if settled:
            self._end_round(iteration, REASON_PLAYBOOK_UNAVAILABLE)

    def _end_round(self, iteration: int, reason: str) -> None:
        state = self.state
        state.phase = PHASE_FREE
        state.reason_codes = [reason]
        state.batch = []
        self._save()
        self._withdraw()
        self.sinks.append_event(
            make_event(
                "wiki_guidance_phase",
                iteration,
                phase=PHASE_FREE,
                reason_codes=[reason],
                items=state.status_counts() or None,
            )
        )
        _say(f"  [wiki-guided] knowledge base round over ({reason}); planning as KernelForge does from here")

    def _withdraw(self) -> None:
        if self.knowledge_wiki is not None:
            withdraw_knowledge_wiki(self.knowledge_wiki)

    def _plan_group(self, plan: str) -> str:
        """The playbook group a part plan claims items of, or ``""``."""
        playbook = self.state.playbook
        if playbook is None:
            return ""
        groups = {item.item_id: item.group for item in playbook.items}
        claimed, _ = cited_items(plan, self.state.ledger)
        return groups[claimed[0]] if claimed else ""

    def _open_items(self, group_id: str) -> tuple[str, ...]:
        playbook = self.state.playbook
        if playbook is None:
            return ()
        return tuple(
            item.item_id
            for item in playbook.items
            if item.group == group_id and self.state.ledger[item.item_id].status in OPEN_STATUSES
        )

    def _group_label(self, group_id: str) -> str:
        playbook = self.state.playbook
        if not group_id or playbook is None:
            return ""
        title = next((group.title for group in playbook.groups if group.group_id == group_id), "")
        return f"playbook group {group_id}" + (f" ({title})" if title else "")

    def _save(self) -> None:
        """Persist the guidance state; a failure degrades persistence and never the iteration."""
        try:
            self.store.save(self.state)
        except OSError as error:
            self.sinks.persistence_failed(f"persist wiki guidance: {error}")
            log.warning("wiki guidance state not durable: %s", error)
