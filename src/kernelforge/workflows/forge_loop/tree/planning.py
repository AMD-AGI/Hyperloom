# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Plan one tree round: every start node is planned in its own checkout, on each profile its slots run on, all side by side."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from hyperloom_common.io import DURABLE, atomic_write_text
from kernelforge.utils.git import git
from kernelforge.workflows.forge_loop.roles.orchestration import OrchestrationInfrastructureError
from kernelforge.workflows.forge_loop.search.search_policy import (
    OBJECTIVE_DISCOVER_NEW_MECHANISM,
    OBJECTIVE_IMMEDIATE_CANONICAL_GAIN,
    SEARCH_MODE_DIVERSIFY,
    SEARCH_MODE_EXPLOIT,
)
from kernelforge.workflows.forge_loop.tree.model import DirectionProposal, Round, SearchTree, Slot, SlotType, TreeNode
from kernelforge.workflows.forge_loop.tree.profiles import SlotActors

_DIRECTION_HEADING = "## direction"
# Markdown emphasis a planner may wrap a record's keys and values in.
_EMPHASIS = "*_` "
_DIRECTION_KEYS = {
    "name": "name",
    "hypothesis": "hypothesis",
    "expected ceiling": "expected_ceiling",
    "fails if": "failure_criterion",
    "hint": "hint",
}


@dataclass(frozen=True)
class SlotPlan:
    """The plan one slot's Implementer is given, and the direction record a BRANCH plan carries."""

    slot: Slot
    text: str
    path: str
    proposal: DirectionProposal | None = None
    hint_answer: str = ""
    # How the round planner had this slot's lane differ from the start's other lanes, and the cases it aims at.
    approach: str = ""
    focus_cases: tuple[str, ...] = ()

    @property
    def title(self) -> str:
        """What the plan sets out to do: its direction's name, else its first heading, else its first line."""
        if self.proposal is not None:
            return self.proposal.name
        lines = [line.strip() for line in self.text.splitlines() if line.strip()]
        headings = [line.lstrip("#").strip() for line in lines if line.startswith("#")]
        return next((item for item in headings if item.lower() != "direction"), lines[0] if lines else "")


@dataclass(frozen=True)
class UnplannedSlot:
    """A slot that will not run this round, and why."""

    slot: Slot
    reason: str


@dataclass(frozen=True)
class RoundPlans:
    """Everything planning bought for a round."""

    plans: tuple[SlotPlan, ...]
    unplanned: tuple[UnplannedSlot, ...]


@dataclass(frozen=True)
class PlanningUnit:
    """The slots of one start that one Agent profile plans together, and the directory their records go to."""

    start: str
    profile_id: str
    slots: tuple[Slot, ...]
    records: Path


def planning_units(slots: Sequence[Slot], round_dir: Path) -> list[PlanningUnit]:
    """The round's slots grouped by start and then by profile, in the order the slots name them.

    A start planned on one profile keeps its records in ``start_<commit>``; a start whose slots run on several gives
    each profile a directory of its own.
    """
    by_start: dict[str, dict[str, list[Slot]]] = {}
    for slot in slots:
        by_start.setdefault(slot.start, {}).setdefault(slot.agent_profile_id, []).append(slot)
    units = []
    for start, by_profile in by_start.items():
        for profile_id, members in by_profile.items():
            name = f"start_{start[:12]}" + (f"_{profile_id}" if len(by_profile) > 1 else "")
            units.append(PlanningUnit(start, profile_id, tuple(members), round_dir / name))
    return units


def read_direction_record(plan: str) -> tuple[DirectionProposal | None, str]:
    """The ``## Direction`` section of a BRANCH plan, and its answer to the selector's hint."""
    fields: dict[str, str] = {}
    inside = False
    for line in plan.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            inside = stripped.lower().startswith(_DIRECTION_HEADING)
            continue
        if inside and stripped.startswith("-") and ":" in stripped:
            key, value = stripped.lstrip("- ").split(":", 1)
            target = _DIRECTION_KEYS.get(key.strip(_EMPHASIS).strip().lower())
            if target and target not in fields:
                fields[target] = value.strip().strip(_EMPHASIS).strip()
    if not fields.get("name"):
        return None, fields.get("hint", "")
    proposal = DirectionProposal(
        name=fields["name"],
        hypothesis=fields.get("hypothesis", ""),
        expected_ceiling=fields.get("expected_ceiling", ""),
        failure_criterion=fields.get("failure_criterion", ""),
    )
    return proposal, fields.get("hint", "")


def start_brief(tree: SearchTree, start: TreeNode, slots: Sequence[Slot]) -> str:
    """What the planners at one start are asked to produce, lane by lane."""
    lines = [
        f"## This round, at start node {start.commit} ({start.score:.6f}x)",
        f"Plan {len(slots)} lane(s) at this node, one per slot below and in this order. Each lane's Implementer "
        "works in its own copy of this node's code. A correct but slower first step of a structural change is "
        "kept in the search tree and can be continued later, so do not trade a promising route for a small "
        "immediate gain.",
    ]
    for position, assignment in enumerate(slot_assignments(tree, start, slots), start=1):
        lines.append(f"{position}. {assignment}")
    if any(slot.slot_type is SlotType.BRANCH for slot in slots):
        lines.append(
            'Right after its Summary line, every BRANCH lane\'s plan carries a "## Direction" section of lines '
            '"- name:", "- hypothesis:", "- expected ceiling:" (with its evidence), "- fails if:" and, when the slot '
            'carries a hint, "- hint:" saying whether it was adopted or, if refuted, the evidence that refutes it.'
        )
    return "\n".join(lines)


def slot_assignments(tree: SearchTree, start: TreeNode, slots: Sequence[Slot]) -> tuple[str, ...]:
    """What each slot planned at ``start`` asks its lane for, in slot order."""
    opened = ", ".join(f'{item.direction_id} "{item.name}"' for item in tree.directions_from(start.commit))
    return tuple(_slot_task(tree, slot, opened) for slot in slots)


def _slot_task(tree: SearchTree, slot: Slot, opened: str) -> str:
    if slot.slot_type is SlotType.DEEPEN:
        direction = tree.direction(slot.direction)
        return (
            f'DEEPEN direction {slot.direction} "{direction.name}": take its next step. Hypothesis: '
            f"{direction.hypothesis} Expected ceiling: {direction.expected_ceiling}."
        )
    hint = f' Selector hint: "{slot.hint}" -- adopt it or refute it with evidence.' if slot.hint else ""
    return (
        f"BRANCH: open a new direction at this node, materially different from those already opened here "
        f"({opened or 'none'}).{hint}"
    )


def port_plan(tree: SearchTree, slot: Slot, diff_path: str) -> str:
    """The plan a PORT slot's Implementer is given: what to carry, from where, measured how."""
    source = tree.require_node(slot.source)
    target = tree.require_node(slot.start)
    return "\n".join(
        [
            "# Port plan",
            "",
            f"Carry gains measured on node {source.commit} ({source.score:.6f}x) onto this checkout of node "
            f"{target.commit} ({target.score:.6f}x).",
            "",
            f"Gains to carry: {slot.gains}",
            f"Why: {slot.reason}",
            "",
            f"The source's change since the two nodes' common ancestor is at {diff_path}. Do not paste it: the "
            "structures differ. Re-express each gain on this node's code, keep the node's own structure, and check "
            "each carried change with the driver; drop a gain that does not hold here.",
        ]
    )


class TreePlanningMixin:
    """Buy one round's plans: one planning unit per start and profile, all of them side by side.

    The host supplies the iteration loop's context builders, ``ic``, ``_usage``, ``_round_planning_sec``,
    ``_checkpoint_llm_usage``, ``_record_probe_hazard``, ``_check_profile_again``, ``_start_worktrees`` and
    ``_experiments``.
    """

    async def _plan_round_slots(self, tree: SearchTree, round_: Round, actors: SlotActors) -> RoundPlans:
        """Plan every slot of ``round_`` on its own profile; a unit whose planning failed leaves its slots unplanned."""
        started_at = time.time()
        try:
            units = planning_units(round_.slots, self._experiments.tree_round(round_.iteration))
            # One after another: concurrent ``git worktree add`` calls contend for the repository's own locks.
            for start in dict.fromkeys(unit.start for unit in units):
                self._start_worktrees.ensure(start)
            outcomes = await asyncio.gather(
                *(
                    self._plan_unit(tree, round_.iteration, unit, actors.of(unit.slots[0]).orchestration_factory)
                    for unit in units
                )
            )
        finally:
            self._round_planning_sec += max(0.0, time.time() - started_at)
        plans = tuple(item for outcome in outcomes for item in outcome.plans)
        unplanned = tuple(item for outcome in outcomes for item in outcome.unplanned)
        return RoundPlans(plans, unplanned)

    async def _plan_unit(
        self, tree: SearchTree, iteration: int, unit: PlanningUnit, orchestration_factory
    ) -> RoundPlans:
        ports = [slot for slot in unit.slots if slot.slot_type is SlotType.PORT]
        planned = [slot for slot in unit.slots if slot.slot_type is not SlotType.PORT]
        port_plans = tuple(self._write_port_plan(tree, unit, slot) for slot in ports)
        if not planned:
            return RoundPlans(port_plans, ())
        if orchestration_factory is None:
            return RoundPlans(
                port_plans, tuple(UnplannedSlot(slot, "no planning service is configured") for slot in planned)
            )
        context = self._start_context(tree, tree.require_node(unit.start), planned)
        try:
            result = await orchestration_factory().run(context, usage=self._usage, lanes=len(planned))
        except OrchestrationInfrastructureError as error:
            # The profile's line may be down: the next round's boundary checks it before it is given slots again.
            self._check_profile_again(unit.profile_id)
            reason = f"planning failed: {type(error).__name__}: {error}"
            return RoundPlans(port_plans, tuple(UnplannedSlot(slot, reason) for slot in planned))
        finally:
            self._checkpoint_llm_usage()
        self._record_probe_hazard(iteration, result)
        mapped = self._map_plans(unit, planned, result)
        return RoundPlans((*port_plans, *mapped.plans), mapped.unplanned)

    def _start_context(self, tree: SearchTree, start: TreeNode, slots: Sequence[Slot]):
        """The campaign's planning context, re-seated on one start node's checkout and measurements."""
        base = self._with_ceiling_standing(
            self._with_case_config_coverage(self._active_analysis_context or self._build_orchestration_context())
        )
        worktree = str(self._start_worktrees.path_of(start.commit))
        site = self._canonical_site()
        diff = self._analysis_cumulative_diff(
            evidence_commit=self.run_state.analysis.evidence_commit, canonical_commit=start.commit
        )
        branching = any(slot.slot_type is SlotType.BRANCH for slot in slots)
        return replace(
            base,
            workspace=worktree,
            analysis_commit=start.commit,
            canonical_commit=start.commit,
            source_map_path=site.relocate(base.source_map_path, worktree),
            editable_sources=tuple(site.relocate(path, worktree) for path in base.editable_sources),
            cases=tuple(
                replace(case, latency_ms=start.case_times.get(case.case_id, case.latency_ms)) for case in base.cases
            ),
            current_mean_case_speedup=start.score,
            evidence_stale=bool(base.evidence_commit and base.evidence_commit != start.commit),
            cumulative_diff_path=diff.path,
            cumulative_diff_error=diff.error,
            program_context=f"{base.program_context}\n\n{start_brief(tree, start, slots)}",
            search_mode=SEARCH_MODE_DIVERSIFY if branching else SEARCH_MODE_EXPLOIT,
            search_objective=OBJECTIVE_DISCOVER_NEW_MECHANISM if branching else OBJECTIVE_IMMEDIATE_CANONICAL_GAIN,
            supervisor_guidance="",
            last_critic_verdict="",
            last_critic_review="",
            # Every start is a planning unit of its own, with no earlier round of its own whose KEEP could stand in
            # for a review.
            review_plan=True,
            lane_assignments=slot_assignments(tree, start, slots),
        )

    def _map_plans(self, unit: PlanningUnit, slots: list[Slot], result) -> RoundPlans:
        """Give each slot the plan published for its lane; a slot whose lane has none says why.

        Slot k is planning lane k, and each published plan names the lane it was made for, so a lane lost anywhere
        in planning costs exactly its own slot.
        """
        diagnostics = dict(getattr(result, "structured_output_diagnostics", None) or {})
        published = {
            lane.lane_id: (lane, text) for lane, text in zip(result.published_lanes, result.optimization_plans)
        }
        dropped = (diagnostics.get("lane_narrowing") or {}).get("dropped_lanes") or {}
        losses = {
            **(diagnostics.get("lane_losses") or {}),
            **{lane: f"the plan critic dropped its lane: {reason}" for lane, reason in dropped.items()},
        }
        unavailable = (diagnostics.get("synthesis") or {}).get("message") or "planning published no plan for its lane"
        plans: list[SlotPlan] = []
        unplanned: list[UnplannedSlot] = []
        for lane_id, slot in enumerate(slots, start=1):
            lane, text = published.get(lane_id, (None, ""))
            if lane is not None and text.strip():
                plans.append(_slot_plan(unit, slot, text.strip(), approach=lane.approach, focus_cases=lane.focus_cases))
            else:
                unplanned.append(UnplannedSlot(slot, losses.get(str(lane_id)) or unavailable))
        _write_record(unit.records, "orchestration.json", diagnostics)
        return RoundPlans(tuple(plans), tuple(unplanned))

    def _write_port_plan(self, tree: SearchTree, unit: PlanningUnit, slot: Slot) -> SlotPlan:
        ancestor = git("merge-base", slot.start, slot.source, cwd=self.ic.workspace_dir).stdout.strip()
        diff = git("diff", ancestor, slot.source, cwd=self.ic.workspace_dir).stdout
        diff_path = _write_record(unit.records, f"slot_{slot.index:02d}_source.diff", diff)
        return _slot_plan(unit, slot, port_plan(tree, slot, str(diff_path)))


def _slot_plan(
    unit: PlanningUnit, slot: Slot, text: str, *, approach: str = "", focus_cases: tuple[str, ...] = ()
) -> SlotPlan:
    proposal, hint_answer = read_direction_record(text) if slot.slot_type is SlotType.BRANCH else (None, "")
    path = _write_record(unit.records, f"slot_{slot.index:02d}.md", text)
    return SlotPlan(slot, text, str(path), proposal, hint_answer, approach=approach, focus_cases=focus_cases)


def _write_record(directory: Path, name: str, content) -> Path:
    path = directory / name
    text = content if isinstance(content, str) else json.dumps(content, indent=2, sort_keys=True, default=str)
    atomic_write_text(path, text, options=DURABLE)
    return path


__all__ = [
    "PlanningUnit",
    "RoundPlans",
    "SlotPlan",
    "TreePlanningMixin",
    "UnplannedSlot",
    "planning_units",
    "port_plan",
    "read_direction_record",
    "slot_assignments",
    "start_brief",
]
