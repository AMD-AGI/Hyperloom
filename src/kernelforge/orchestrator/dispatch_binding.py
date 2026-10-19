# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bind a round's resolved role/case intents to assignment ids and the exact evidence each assignment reads."""

from __future__ import annotations

from collections.abc import Sequence

from kernelforge.orchestrator.contracts import (
    CaseEvidence,
    DispatchIntent,
    DispatchPlan,
    EvidenceRef,
    OrchestrationContext,
    SpecialistAssignment,
)

# Evidence every assignment reads, whatever cases it targets.
_GLOBAL_EVIDENCE_KINDS = frozenset(
    {
        "analysis_artifact_catalog",
        "analysis_bundle",
        "analysis_cumulative_diff",
        "analysis_summary",
        "analysis_source_map",
        "analysis_workflow",
        "latest_lesson",
        "supervisor_guidance",
    }
)
# Analysis flags that make a case's profile summary a measured profile rather than an interpretation.
_PROFILED_FLAGS = frozenset(
    {
        "analysis_profiled",
        "analysis_profile_incremental",
        "analysis_checkpoint_profile_interpretation",
        "analysis_checkpoint_normalized_only",
        "analysis_checkpoint_raw_profile_only",
    }
)


def _case_refs(case: CaseEvidence) -> list[EvidenceRef]:
    refs = [
        EvidenceRef(
            kind="measurement",
            path=f"case:{case.case_id}",
            summary=f"Canonical timing and Analysis flags for {case.case_id}.",
        )
    ]
    if case.profile_summary_path:
        profiled = any(flag in _PROFILED_FLAGS for flag in case.flags)
        refs.append(
            EvidenceRef(
                kind=("profile" if profiled else "analysis_interpretation"),
                path=case.profile_summary_path,
                summary=("Measured profile" if profiled else "Analysis interpretation")
                + f" for {case.case_id}. Flags: "
                + f"{', '.join(case.flags) or '(none)'}.",
            )
        )
    return refs


def bind_dispatch(
    context: OrchestrationContext,
    intents: Sequence[DispatchIntent],
    notes: Sequence[str],
) -> DispatchPlan:
    """One assignment per intent, in order; each intent's cases must be scored cases of ``context``."""
    global_refs = [reference for reference in context.evidence_refs if reference.kind in _GLOBAL_EVIDENCE_KINDS]
    source_ref = EvidenceRef(
        kind="source_map",
        path=context.source_map_path,
        summary="Current source map or anchor source.",
    )
    case_by_id = {case.case_id: case for case in context.cases}
    assignments = []
    for index, intent in enumerate(intents, start=1):
        refs = {reference.path: reference for reference in global_refs}
        refs[source_ref.path] = source_ref
        for case_id in intent.target_case_ids:
            refs.update((reference.path, reference) for reference in _case_refs(case_by_id[case_id]))
        assignments.append(
            SpecialistAssignment(
                assignment_id=f"{intent.role_id}-{index}",
                role_id=intent.role_id,
                target_case_ids=tuple(intent.target_case_ids),
                evidence_refs=tuple(refs.values()),
                reason=intent.reason,
            )
        )
    return DispatchPlan(
        analysis_commit=context.analysis_commit,
        assignments=tuple(assignments),
        normalization_notes=tuple(notes),
    )
