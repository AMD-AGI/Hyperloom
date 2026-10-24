# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Checking the correctness fix a candidate declares with its repro case, before the gate settles the candidate.

The case runs on the candidate's parent, on the candidate and on the session's root (``prove_fix``). A proven fix the
gate would otherwise only record (flat or regressed) also goes to the critic: only a fix whose trigger is reachable
under the workload counts, and lands or forks as a fix. A proven gain lands on its gain and keeps its case as a
regression case. A fix that is not proven, or that the critic refuses, leaves the candidate judged on its performance
alone, and says why.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from hyperloom.contracts.measurement import AttachmentRef, TestRequest
from hyperloom.measure.compile import BlobError, BlobStore
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.notebook.interfaces import CommitQuery
from hyperloom_multibranch.rulegate.critic import FixReview, FixReviewer, ReproVerdicts
from hyperloom_multibranch.rulegate.fix_case import FIX_CASE_ID, CaseError, case_of, read_case
from hyperloom_multibranch.rulegate.regression import FixNotProven, FixProof, ReproRunner, prove_fix

#: Characters of one patch the critic is shown; a longer patch is cut, and says so.
PATCH_CHARS = 20_000
#: A fix is checked only on a final verdict the gate would land or record; the rest wait for their retest.
CHECKED = (Verdict.FLAT, Verdict.REGRESSED, Verdict.IMPROVED)


@dataclass(frozen=True)
class FixChecking:
    """How a session checks a declared fix: the runner its repro tests go through, the critic, the blob store its case
    and patches are read from, and the workload the critic judges reachability under."""

    runner: ReproRunner
    critic: FixReviewer
    blobs: BlobStore
    workload: str


@dataclass(frozen=True)
class FixCheck:
    """What the check settles the candidate with: the judgement (a fix when it counts), the proof the gate registers
    the case from, and why the candidate is judged on its performance alone, empty when it is not."""

    judgement: Judgement
    proof: FixProof | None
    alone: str = ""


def declares_fix(request: TestRequest, judgement: Judgement) -> bool:
    """Whether the candidate carries a case, or an attachment under the case's id, and its judgement is one a fix is
    checked on."""
    declared = any(attachment.id == FIX_CASE_ID for attachment in request.attachments)
    return declared and judgement.verdict in CHECKED and judgement.accuracy_ok


async def check_fix(
    checking: FixChecking, notebook: CommitQuery, request: TestRequest, judgement: Judgement
) -> FixCheck:
    """Prove the fix ``request`` declares and, unless it gained, have the critic review it."""
    try:
        case = case_of(request)
    except CaseError as error:
        return FixCheck(judgement, None, f"its case is unusable: {error}")
    if case is None:
        raise ValueError(f"{request.request_id} declares no fix")
    try:
        proof = await prove_fix(checking.runner, notebook, request, (case,))
    except FixNotProven as error:
        return FixCheck(judgement, None, f"its fix is not proven: {error.reason}")
    if judgement.verdict is Verdict.IMPROVED:
        return FixCheck(judgement, proof)
    try:
        review = _review(checking, case, request, judgement, proof)
    except (BlobError, CaseError) as error:
        return FixCheck(judgement, None, f"its fix cannot be reviewed: {error}")
    verdict = await checking.critic.review(review)
    if not verdict.reachable:
        return FixCheck(judgement, None, f"the reviewer refused its fix: {verdict.reason}")
    return FixCheck(replace(judgement, fixes_correctness=True, foundation_fix=proof.foundation), proof)


def _review(
    checking: FixChecking, case: AttachmentRef, request: TestRequest, judgement: Judgement, proof: FixProof
) -> FixReview:
    patches = {
        patch.path: _clipped(checking.blobs.get(patch.sha256).decode("utf-8", errors="replace"))
        for change in request.delta.components
        for patch in change.add_patches
    }
    root = proof.root.outcome if proof.root.request_id != proof.parent.request_id else ""
    verdicts = ReproVerdicts(proof.parent.outcome, proof.candidate.outcome, root)
    text = read_case(checking.blobs.get(case.sha256))
    return FixReview(request.delta, patches, text, checking.workload, judgement, verdicts)


def _clipped(text: str) -> str:
    return text if len(text) <= PATCH_CHARS else f"{text[:PATCH_CHARS]}\n[cut at {PATCH_CHARS} characters]"


__all__ = ["CHECKED", "PATCH_CHARS", "FixCheck", "FixChecking", "check_fix", "declares_fix"]
