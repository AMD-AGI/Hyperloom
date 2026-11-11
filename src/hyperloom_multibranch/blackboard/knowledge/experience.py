# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publish a session's tested candidates to the Experience KB, so what they taught reaches later sessions.

Every specialist test of kind ``test`` that the rule gate judged, or that failed by the candidate's own fault, becomes
one Experience through the packaged ``multibranch-notebook-v1`` mapping; the other kinds (profile, accuracy, repro,
sweep) examine a state rather than try a change, so they are not published, nor is a landed retraction, which undoes
a change its agent found invalid:

* conditions: the session profile under the names the ``inference-recipe-v1`` declaration gives them, plus the
  intent's target (as ``bottleneck``), mechanism and layer;
* action: the recipe delta, a config change fingerprinted as the single-lineage producer fingerprints one;
* consequence: the parent commit's recorded value and the candidate's, the judged gain plus or minus the threshold as
  the interval;
* evidence: the request and the requests its judgement pooled. A test without evidence never reaches the KB: the
  mapping refuses a candidate without reasoning, a recorded parent value, or an outcome.

The reasoning is the hypothesis or attempt the specialist logged citing the request; the reflection is its latest
learning citing the request, else the result it logged, else the hypothesis it settled. A test whose measurement failed taught nothing and is not published. The measured-against state is the parent commit,
the experiment repository commit that pins the exact recipe. A test that brought in a change a landed retraction later
undid is still published, its reflection naming the retraction.

A judged record changes only while its split is pending, so pending splits wait; everything else is published once and
reads ``unchanged`` afterwards. Merges stack commits already published, so they are not.

The shared ``inference-recipe-v1`` declaration holds an InferenceX session's configuration and source patch changes,
which the single-lineage flow reads too. What it cannot hold, a change that ships a kernel artifact and every change of
an AgentX session, graded on its objective, goes through the ``multibranch-recipe-v1`` mapping to the blackboard's own
declaration of that name (:func:`publish_multibranch_experiences`). Each mapping refuses the other's candidates.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hyperloom.common.perf_metric import TOTAL_TOKEN_THROUGHPUT
from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_common.timeutil import epoch_iso
from hyperloom_kb.collect import CollectionTarget, CollectReport, collect
from hyperloom_multibranch.contracts.judgement import AccuracyVerdict
from hyperloom_multibranch.contracts.blackboard import INTEGRATION_BRANCH, NotRecorded
from hyperloom_multibranch.contracts.blackboard.peers import BranchLogType, HypothesisStatus, Intent, TestKind, Validity
from hyperloom_multibranch.contracts.blackboard.records import (
    BranchLogRecord,
    LearningRecord,
    BlackboardSnapshot,
    SessionProfile,
    TestRecord,
)
from hyperloom_multibranch.contracts.blackboard.vocab import COMMITTING, Conclusion
from hyperloom_multibranch.blackboard.knowledge.lessons import direction, later_retractions
from hyperloom_multibranch.rulegate.agentx import partial_window

MAPPING_PATH = Path(__file__).with_name("mappings") / "multibranch-notebook-v1.yaml"
MULTIBRANCH_MAPPING_PATH = Path(__file__).with_name("mappings") / "multibranch-recipe-v1.yaml"
EXPORT_FORMAT = "multibranch.notebook.export.v1"
THROUGHPUT_OBJECTIVE = "output_throughput"
AGENTX_INTERACTIVITY_OBJECTIVE = "agentx_interactivity"
AGENTX_TOTAL_OBJECTIVE = "agentx_total_throughput"
SUMMARY_CHARS = 160
_PENDING = frozenset({Conclusion.SPLIT_REQUESTED, Conclusion.SPLIT_PENDING, Conclusion.DIAGNOSIS_PENDING})
_REASONS = frozenset({BranchLogType.HYPOTHESIS, BranchLogType.ATTEMPT})
_SETTLED = frozenset({HypothesisStatus.CONFIRMED, HypothesisStatus.FALSIFIED})


def experience_identity(profile: SessionProfile) -> dict[str, object]:
    """The session's conditions, keyed as the ``inference-recipe-v1`` declaration names them."""

    identity: dict[str, object | None] = {
        "model": profile.model,
        "gpu": profile.hardware,
        "framework": profile.framework,
        "model_type": profile.model_type,
        "architecture": profile.architectures[0] if profile.architectures else "",
        "framework_version": profile.framework_version,
        "precision": profile.precision,
        "tp": profile.tp,
        "conc": profile.conc,
        "isl": profile.isl,
        "osl": profile.osl,
    }
    return {key: value for key, value in identity.items() if value not in (None, "")}


def intent_conditions(intent: Intent | None) -> dict[str, object]:
    """What an intent adds to the conditions: its target (the bottleneck it attacks), mechanism and layer."""

    if intent is None:
        return {}
    return {"bottleneck": intent.target, "mechanism": intent.mechanism, "layer": intent.layer.value}


def change_document(delta: RecipeDelta) -> dict[str, object]:
    """The delta as the mapping reads it: a family, a one-line summary, and the content the fingerprint covers."""

    config = delta.config
    patches = [patch.to_document() for change in delta.components for patch in change.add_patches]
    if delta.add_kernels or delta.remove_kernels:
        family = "kernel"
    else:
        family = "patch" if delta.components else "config"
    parts = [
        config.server_args,
        *(f"{name}={value}" for name, value in sorted(config.envs.items())),
        *(f"drop {flag}" for flag in config.remove_args),
        *(f"unset {name}" for name in config.unset_envs),
        *(str(patch["path"]) for patch in patches),
        *(kernel.ref for kernel in delta.add_kernels),
    ]
    return {
        "family": family,
        "summary": " ".join(part for part in parts if part)[:SUMMARY_CHARS],
        "content": {
            "extra_server_args": config.server_args,
            "extra_envs": dict(config.envs),
            "remove_args": list(config.remove_args),
            "unset_envs": list(config.unset_envs),
            "args_mode": config.args_mode.value,
            "patches": patches,
            "add_kernels": [kernel.to_document() for kernel in delta.add_kernels],
            "remove_kernels": list(delta.remove_kernels),
        },
    }


def branch_notes(log: Sequence[BranchLogRecord], request_id: str) -> tuple[str, str]:
    """The reasoning behind a test and what the specialist learnt from it, from the log entries citing it."""

    citing = [record.entry for record in log if request_id in record.entry.request_ids]
    reasons = [entry.content for entry in citing if entry.kind in _REASONS]
    results = [entry.content for entry in citing if entry.kind is BranchLogType.RESULT]
    settled = [
        f"{entry.hypothesis_status.value}: {entry.content}"
        for entry in citing
        if entry.kind is BranchLogType.HYPOTHESIS and entry.hypothesis_status in _SETTLED
    ]
    learnt = results or settled
    return (reasons[0] if reasons else ""), (learnt[-1] if learnt else "")


def cited_learning(learnings: Sequence[LearningRecord], branch: str, request_id: str) -> str:
    """The latest learning of ``branch`` citing ``request_id`` that no later one superseded; empty when there is none."""

    superseded = {record.learning.supersedes for record in learnings if record.learning.supersedes}
    citing = [
        record.learning.content
        for record in learnings
        if record.learning.branch == branch
        and request_id in record.learning.request_ids
        and record.entry_id not in superseded
    ]
    return citing[-1] if citing else ""


def _learn(learnt: str, logged: str, undone: Sequence[str]) -> tuple[str, str]:
    """What a test taught and where that came from: the specialist's learning, else the result it logged, then each
    landed retraction that later undid a change the test brought in."""

    said = "; ".join(part for part in (learnt or logged, *undone) if part)
    source = "specialist_learning" if learnt else "specialist_branch_log" if logged else "retraction" if undone else ""
    return said, source


def _decision(record: TestRecord) -> str:
    """A failed candidate the gate judged all the same, one the AgentX checkpoint stopped, was reverted, not broken."""
    if record.result is not None and record.result.validity is Validity.CANDIDATE_FAILED and record.judgement is None:
        return "failed"
    return "keep" if record.conclusion in COMMITTING else "revert"


def _accuracy(record: TestRecord, others: Sequence[TestRecord] = ()) -> dict[str, object] | None:
    """The accuracy of the tests ``record``'s judgement rests on, its own reading first, the rest found among
    ``others``: a confirm's eval stands for an explore that skipped its own. A test the gate did not judge, such as a
    candidate that failed first, did not pass it."""

    judged = record.judgement
    behind = set(judged.request_ids) - {record.request_id} if judged is not None else set()
    readings = [record, *(other for other in others if other.request_id in behind)]
    scores = [value.score for reading in readings if reading.result is not None for value in reading.result.accuracy]
    asked = any(reading.request.accuracy for reading in readings)
    if not (asked or scores or record.conclusion is Conclusion.ACCURACY_FAILED):
        return None
    passed = (
        judged is not None
        and judged.accuracy is AccuracyVerdict.PASSED
        and record.conclusion is not Conclusion.ACCURACY_FAILED
    )
    return {"required": True, "passed": passed, "value": next(iter(scores), None)}


def _measured(record: TestRecord, metric: str) -> dict[str, object]:
    judged = record.judgement
    if judged is None or partial_window(judged):
        return {"outcome_value": None, "interval": None, "interval_text": "", "tests": [record.request_id]}
    low, high = judged.gain_pct - judged.threshold_pct, judged.gain_pct + judged.threshold_pct
    return {
        "outcome_value": judged.recorded_value.get(metric),
        "interval": {"mean": judged.gain_pct, "low": low, "high": high},
        "interval_text": f"{judged.gain_pct:+.2f}% within [{low:+.2f}%, {high:+.2f}%] over {len(judged.request_ids)} tests",
        "tests": list(judged.request_ids),
    }


def _state(snapshot: BlackboardSnapshot, commit_id: str) -> str:
    """The parent's recipe digest, the same in every session that measured that recipe; else its commit id."""

    node = snapshot.lineage.commit(commit_id)
    return node.recipe_digest if node is not None and node.recipe_digest else commit_id


def candidate_row(
    record: TestRecord, snapshot: BlackboardSnapshot, profile: SessionProfile, metric: str
) -> dict[str, object] | None:
    """One export row; ``None`` for a merge, another kind than ``test``, a test in flight or not measured, a
    judgement waiting on its split, or a landed retraction, which tried no optimisation and has no decision the
    declarations' ``keep``, ``revert`` and ``failed`` name."""

    result = record.result
    if (
        record.request.branch == INTEGRATION_BRANCH
        or record.request.kind is not TestKind.TEST
        or result is None
        or not record.measured
        or record.landed_retraction
    ):
        return None
    failed = result.validity is Validity.CANDIDATE_FAILED
    if not failed and (record.judgement is None or record.conclusion is None or record.conclusion in _PENDING):
        return None
    request = record.request
    reasoning, logged = branch_notes(snapshot.branch_logs.get(request.branch, ()), record.request_id)
    learnt = cited_learning(snapshot.learnings, request.branch, record.request_id)
    learn, learn_source = _learn(learnt, logged, later_retractions(record, snapshot))
    return {
        "request_id": record.request_id,
        "branch": request.branch,
        "parent_commit": request.parent_commit,
        "parent_state": _state(snapshot, request.parent_commit),
        "commit_id": record.commit_id or "",
        "created_at": epoch_iso(record.requested_at),
        "completed_at": epoch_iso(record.judged_at or record.measured_at or record.requested_at),
        "identity": {
            **experience_identity(profile),
            **intent_conditions(snapshot.intents.get(request.intent_fingerprint)),
        },
        "direction": direction(record, snapshot),
        "change": change_document(request.delta),
        "reasoning": reasoning,
        "learn": learn,
        "learn_source": learn_source,
        "decision": _decision(record),
        "conclusion": record.conclusion.value if record.conclusion is not None else "",
        "baseline_value": snapshot.value_of(request.parent_commit).get(metric),
        **_measured(record, metric),
        "accuracy": _accuracy(record, snapshot.records),
        "error_class": (result.failure_reason or "") if failed else "",
        "gpu_seconds": result.gpu_seconds,
    }


def export_document(snapshot: BlackboardSnapshot, metric: str) -> dict[str, object]:
    """The ``multibranch.notebook.export.v1`` document, the mapping's input; ``metric`` is the metric the session is
    graded on, which names an AgentX session's objective."""

    profile = snapshot.profile
    if profile is None:
        raise NotRecorded("PRELUDE has not written the session profile yet")
    rows = (candidate_row(record, snapshot, profile, metric) for record in snapshot.records)
    agentx = AGENTX_TOTAL_OBJECTIVE if metric == TOTAL_TOKEN_THROUGHPUT else AGENTX_INTERACTIVITY_OBJECTIVE
    return {
        "format": EXPORT_FORMAT,
        "metadata": {
            "session_id": profile.session_id,
            "objective": THROUGHPUT_OBJECTIVE if profile.isl is not None else agentx,
            "metric": metric,
        },
        "candidates": [row for row in rows if row is not None],
    }


def publish_experiences(
    snapshot: BlackboardSnapshot,
    metric: str,
    *,
    kb: CollectionTarget | None = None,
    receipt: Path | None = None,
) -> CollectReport:
    """Write the session's Experiences the shared declaration holds to ``kb``, or to the service ``HYPERLOOM_KB_URL``
    names."""

    return collect(MAPPING_PATH, export_document(snapshot, metric), kb=kb, receipt=receipt)


def publish_multibranch_experiences(
    snapshot: BlackboardSnapshot,
    metric: str,
    *,
    kb: CollectionTarget | None = None,
    receipt: Path | None = None,
) -> CollectReport:
    """Write the session's Experiences the shared declaration cannot hold, kernel changes and an AgentX session's,
    under ``multibranch-recipe-v1`` to ``kb``, or to the service ``HYPERLOOM_KB_URL`` names."""

    return collect(MULTIBRANCH_MAPPING_PATH, export_document(snapshot, metric), kb=kb, receipt=receipt)


__all__ = [
    "AGENTX_INTERACTIVITY_OBJECTIVE",
    "AGENTX_TOTAL_OBJECTIVE",
    "EXPORT_FORMAT",
    "MAPPING_PATH",
    "MULTIBRANCH_MAPPING_PATH",
    "THROUGHPUT_OBJECTIVE",
    "branch_notes",
    "candidate_row",
    "change_document",
    "cited_learning",
    "experience_identity",
    "export_document",
    "intent_conditions",
    "publish_experiences",
    "publish_multibranch_experiences",
]
