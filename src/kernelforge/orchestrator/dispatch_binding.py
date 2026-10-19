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
_GLOBAL_EVIDENCE_KINDS = {
    "analysis_artifact_catalog",
    "analysis_bundle",
    "analysis_cumulative_diff",
    "analysis_summary",
    "analysis_source_map",
    "analysis_workflow",
    "latest_lesson",
    "supervisor_guidance",
}
_PROFILED_CASE_FLAGS = {
    "analysis_profiled",
    "analysis_profile_incremental",
    "analysis_checkpoint_profile_interpretation",
    "analysis_checkpoint_normalized_only",
    "analysis_checkpoint_raw_profile_only",
}


def _case_evidence_refs(case: CaseEvidence) -> dict[str, EvidenceRef]:
    """The evidence one scored case hands every specialist assigned to it, keyed by path."""
    measurement = EvidenceRef(
        kind="measurement",
        path=f"case:{case.case_id}",
        summary=(f"Canonical timing and Analysis flags for {case.case_id}."),
    )
    refs = {measurement.path: measurement}
    if case.profile_summary_path:
        if _PROFILED_CASE_FLAGS.isdisjoint(case.flags):
            kind, label = "analysis_interpretation", "Analysis interpretation"
        else:
            kind, label = "profile", "Measured profile"
        refs[case.profile_summary_path] = EvidenceRef(
            kind=kind,
            path=case.profile_summary_path,
            summary=label + f" for {case.case_id}. Flags: " + f"{', '.join(case.flags) or '(none)'}.",
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
            refs.update(_case_evidence_refs(case_by_id[case_id]))
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
