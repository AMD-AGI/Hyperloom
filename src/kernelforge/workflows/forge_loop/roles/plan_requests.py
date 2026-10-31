# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The user-turn requests the orchestration planner sends: dispatch, single-plan synthesis, partition and revision."""

from __future__ import annotations

from collections.abc import Mapping

from kernelforge.workflows.forge_loop.roles.contracts import (
    OrchestrationContext,
    PlanCriticOutcome,
    SpecialistDefinition,
    SpecialistRound,
    SynthesizedPlan,
)
from kernelforge.workflows.forge_loop.roles.plan_prompts import REVISION_SYSTEM_PROMPT

PARTITION_OUTPUT_SCHEMA = {
    "lanes": [
        {
            "ground": ("The files, functions and mechanisms this lane owns and may edit"),
            "reason": "Why this is one session's worth of work",
            "joint": (
                "true when this lane owns a change together with "
                "the launch configuration that serves it, or a "
                "move no single region contains"
            ),
            "fallback": (
                "For a joint lane: the smaller change inside the same ground to land if the main step is abandoned"
            ),
        }
    ],
    "cross_cutting_move": {
        "move": ("The largest move the evidence supports that no single region contains"),
        "lane_id": ("The 1-based lane that owns it, or 0 if no lane does"),
        "unassigned_reason": ("Why no lane owns it, when no lane does"),
    },
}


def dispatch_payload(context: OrchestrationContext, definitions: Mapping[str, SpecialistDefinition]) -> dict:
    """The dispatch request: the task under the round's guidance, the context and the registered roles."""
    ruling_guidance = (
        "The latest Supervisor Ruling is the current planning authority and "
        "overrides subjective recommendations or conclusions in historical "
        "lesson records; objective measurements remain authoritative. "
        if context.supervisor_guidance
        else ""
    )
    mode_guidance = (
        "A latest Supervisor Ruling is present. Use that ruling to decide "
        "whether this planning cycle should persist or diversify; the "
        f"recorded search mode ({context.search_mode}) is background state."
        if context.supervisor_guidance
        else (
            "Seek materially different mechanisms across scored cases."
            if context.search_mode == "DIVERSIFY"
            else "Target the strongest immediate canonical gain."
        )
    )
    return {
        "task": (
            "Select the specialist roles needed for the supplied cases. "
            "Assignments may overlap cases when independent expertise is "
            "useful. Return role/case intent only; the framework binds "
            "assignment IDs and exact evidence paths. " + ruling_guidance + mode_guidance
        ),
        "context": context.to_prompt_dict(),
        "available_specialists": [definitions[role_id].to_dict() for role_id in sorted(definitions)],
        "output_schema": {
            "assignments": [
                {
                    "role_id": "registered-role-id",
                    "target_case_ids": ["case-id"],
                    "reason": "Why this specialist is required",
                }
            ],
        },
    }


def synthesis_payload(context: OrchestrationContext, specialists: SpecialistRound, analyses: list[dict]) -> dict:
    """The single-plan synthesis request: the task under the round's guidance and every specialist's answer."""
    failures = [outcome.to_dict() for outcome in specialists.outcomes if outcome.failure is not None]
    search_guidance = (
        "Use the latest Supervisor Ruling to choose whether immediate gain "
        "or mechanism diversity is appropriate for this planning cycle."
        if context.supervisor_guidance
        else (
            "Prioritize mechanisms that can produce the strongest immediate canonical gain."
            if context.search_mode == "EXPLOIT"
            else (
                "Prioritize meaningful mechanism diversity and cross-case "
                "headroom while keeping the resulting plan feasible."
            )
        )
    )
    ruling_guidance = (
        "Honor the latest Supervisor Ruling over subjective conclusions in "
        "historical lesson records, while preserving objective validation "
        "and measurement facts. "
        if context.supervisor_guidance
        else ""
    )
    return {
        "task": (
            "Produce the optimization plan that the Implementer should execute. "
            + ruling_guidance
            + f"{search_guidance} Reconcile all specialist analyses into one "
            "decision: select and sequence the most valuable compatible "
            "work, explain critical trade-offs, and omit ideas that do not "
            "justify their cost or risk. Do not merely summarize each "
            "specialist in turn."
        ),
        "context": context.to_prompt_dict(),
        "dispatch_plan": specialists.dispatch_plan.to_dict(),
        "specialist_coverage": dict(specialists.coverage),
        "specialist_analyses": analyses,
        "specialist_failures": failures,
    }


def revision_payloads(
    context: OrchestrationContext,
    specialists: SpecialistRound,
    draft: SynthesizedPlan,
    critic: PlanCriticOutcome,
) -> tuple[dict, dict]:
    """The revision request for a resumed synthesis session, and the self-contained one for a fresh session."""
    revision_task = (
        "Revise the draft into the final optimization plan. Address the critic's substantive concerns exactly once."
    )
    # A resume re-enters the lane's own synthesis session, which already holds the dispatch, every specialist
    # analysis and the whole synthesis conversation.
    resumed_payload = {
        "task": revision_task,
        "draft_plan": draft.text,
        "critic_verdict": critic.verdict,
        "critic_review": critic.review,
    }
    # The fresh-session fallback holds no prior context, so it genuinely needs the whole planning bundle to revise
    # from.
    fresh_payload = {
        **resumed_payload,
        "revision_instructions": REVISION_SYSTEM_PROMPT.strip(),
        "context": context.to_prompt_dict(),
        "dispatch_plan": specialists.dispatch_plan.to_dict(),
        "specialist_coverage": dict(specialists.coverage),
        "specialist_outcomes": [outcome.to_dict() for outcome in specialists.outcomes],
    }
    return resumed_payload, fresh_payload
