# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Planning contexts and scripted agent backends the orchestration test modules share."""

from __future__ import annotations

import json

from kernelforge.agent_backends import AgentRunResult
from kernelforge.workflows.forge_loop.roles.contracts import (
    CaseEvidence,
    DispatchPlan,
    EvidenceRef,
    OrchestrationContext,
    SpecialistRound,
    SpecialistAssignment,
    SpecialistDefinition,
)
from kernelforge.workflows.forge_loop.roles.orchestration import OrchestrationAgent, OrchestrationService
from kernelforge.workflows.forge_loop.roles.specialists import SpecialistAgent, SpecialistPool


def _context() -> OrchestrationContext:
    return OrchestrationContext(
        analysis_commit="abc123",
        workspace="/workspace",
        gpu_target="gfx942",
        objective="equal-weight mean case speedup",
        program_context="Optimize the operator.",
        source_map_path="analysis/abc123/source_map.json",
        knowledge_index="Curated local knowledge index.",
        cases=(
            CaseEvidence(
                case_id="case-a",
                latency_ms=1.0,
                bottleneck="memory",
                profile_summary_path=("analysis/abc123/profiles/case-a/summary.json"),
            ),
            CaseEvidence(
                case_id="case-b",
                latency_ms=4.0,
                bottleneck="compute",
                profile_summary_path=("analysis/abc123/profiles/case-b/summary.json"),
            ),
        ),
        evidence_refs=(
            EvidenceRef(
                kind="history",
                path="handoffs/iter_001.json",
                summary="Previous iteration outcome",
            ),
        ),
    )


def _definitions() -> dict[str, SpecialistDefinition]:
    return {
        "compute": SpecialistDefinition(
            role_id="compute",
            description="Compute optimization specialist",
            instructions="Analyze instruction throughput and scheduling.",
            capabilities=("compute", "scheduling"),
        ),
        "memory": SpecialistDefinition(
            role_id="memory",
            description="Memory optimization specialist",
            instructions="Analyze memory layout and cache behavior.",
            capabilities=("memory", "cache"),
        ),
    }


def _assignment(
    *,
    assignment_id: str = "memory-1",
    role_id: str = "memory",
    case_id: str = "case-a",
) -> SpecialistAssignment:
    return SpecialistAssignment(
        assignment_id=assignment_id,
        role_id=role_id,
        target_case_ids=(case_id,),
        evidence_refs=(
            EvidenceRef(
                kind="profile",
                path=f"analysis/abc123/profiles/{case_id}/summary.json",
                summary=f"Profile for {case_id}",
            ),
        ),
        reason="The case is bottlenecked in this specialist domain.",
    )


def _dispatch_payload(*, both_roles: bool = False) -> str:
    assignments = [
        {
            "assignment_id": "memory-1",
            "role_id": "memory",
            "target_case_ids": ["case-a"],
            "evidence_refs": [
                {
                    "kind": "measurement",
                    "path": "case:case-a",
                    "summary": "Memory case",
                }
            ],
            "reason": "Inspect memory behavior.",
        }
    ]
    if both_roles:
        assignments.append(
            {
                "assignment_id": "compute-1",
                "role_id": "compute",
                "target_case_ids": ["case-b"],
                "evidence_refs": [
                    {
                        "kind": "measurement",
                        "path": "case:case-b",
                        "summary": "Compute case",
                    }
                ],
                "reason": "Inspect compute behavior.",
            }
        )
    return json.dumps({"assignments": assignments})


class _StaticBackend:
    def __init__(
        self,
        result: str = "",
        error: Exception | None = None,
        end_reason: str = "agent_stopped",
        stderr_tail: str = "",
    ) -> None:
        self.result = result
        self.error = error
        self.end_reason = end_reason
        self.stderr_tail = stderr_tail
        self.calls = 0
        self.specs = []

    async def run(self, spec, usage=None) -> AgentRunResult:
        self.calls += 1
        self.specs.append(spec)
        if self.error is not None:
            raise self.error
        return AgentRunResult(
            text=self.result,
            end_reason=self.end_reason,
            stderr_tail=self.stderr_tail,
        )


class _QueuedBackend:
    def __init__(
        self,
        results: list[str | AgentRunResult],
    ) -> None:
        self.results = list(results)
        self.specs = []

    async def run(self, spec, usage=None) -> AgentRunResult:
        self.specs.append(spec)
        result = self.results.pop(0)
        return result if isinstance(result, AgentRunResult) else AgentRunResult(text=result)


class _PerCallBackend:
    """A backend whose reply differs per call, so lanes are distinguishable."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.specs: list = []

    async def run(self, spec, usage=None) -> AgentRunResult:
        self.specs.append(spec)
        index = min(len(self.specs) - 1, len(self.replies) - 1)
        return AgentRunResult(
            text=self.replies[index],
            end_reason="agent_stopped",
            stderr_tail="",
        )


async def _outcome(role: str, analysis: str):
    return await SpecialistAgent(
        definition=_definitions()[role],
        backend=_StaticBackend(analysis),
        timeout_sec=1,
        max_turns=2,
    ).run(_assignment(assignment_id=f"{role}-1", role_id=role), _context())


def _coverage(roles: list[str]) -> dict:
    return {
        "successful_roles": roles,
        "covered_cases": ["case-a"],
        "missing_cases": [],
        "failed_roles": [],
    }


async def _compute_and_memory_round() -> SpecialistRound:
    """The round most lane tests divide: one compute and one memory analysis of one dispatch."""
    outcomes = (
        await _outcome("compute", "Instruction issue is the limit."),
        await _outcome("memory", "The scale stream is re-read."),
    )
    dispatch = DispatchPlan(analysis_commit="abc123", assignments=(_assignment(),))
    return SpecialistRound(dispatch, outcomes, _coverage(["compute", "memory"]))


def _partition(*grounds: str) -> str:
    """The round partition a lane round now buys before it synthesizes."""
    return json.dumps({"lanes": [{"ground": ground, "reason": "one session's worth"} for ground in grounds]})


GROUND_A = "chunk_intra.py: the MFMA issue schedule"
GROUND_B = "chunk.py: the scale stream's staging"


JOINT_GROUND = "chunk_intra.py: the BLOCK_H tile and the num_warps/waves_per_eu/num_stages the launch passes with it"


def _partition_lanes(lanes: list[dict], move: object | None = None) -> str:
    """A partition answer that can carry joint lanes and its widest move."""
    payload: dict = {"lanes": lanes}
    if move is not None:
        payload["cross_cutting_move"] = move
    return json.dumps(payload)


def _lane(ground: str, **extra) -> dict:
    return {"ground": ground, "reason": "one session's worth", **extra}


def _single_lane_service() -> OrchestrationService:
    """One memory specialist, and a planner that dispatches it and then writes one plan with no critic."""
    return OrchestrationService(
        agent=OrchestrationAgent(
            backend=_QueuedBackend([_dispatch_payload(), "# Optimization plan\nUse vector loads."]),
            timeout_sec=1,
            max_turns=2,
            min_assignments=1,
        ),
        specialist_pool=SpecialistPool(
            {
                "memory": SpecialistAgent(
                    definition=_definitions()["memory"],
                    backend=_StaticBackend("Memory analysis"),
                    timeout_sec=1,
                    max_turns=2,
                )
            },
            max_parallel=1,
        ),
        definitions={"memory": _definitions()["memory"]},
    )
