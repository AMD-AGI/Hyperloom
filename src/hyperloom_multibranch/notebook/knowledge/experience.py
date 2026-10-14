# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publish a session's tested candidates to the Experience KB, so what they taught reaches later sessions.

Every specialist test the rule gate judged, or that failed by the candidate's own fault, becomes one Experience through
the packaged ``multibranch-notebook-v1`` mapping:

* conditions: the session profile under the names the ``inference-recipe-v1`` declaration gives them, plus the
  intent's target (as ``bottleneck``), mechanism and layer;
* action: the recipe delta, a config change fingerprinted as the single-lineage producer fingerprints one;
* consequence: the parent commit's recorded value and the candidate's, the judged gain plus or minus the threshold as
  the interval;
* evidence: the request and the requests its judgement pooled. A test without evidence never reaches the KB: the
  mapping refuses a candidate without reasoning, a recorded parent value, or an outcome.

The reasoning is the hypothesis or attempt the specialist logged citing the request, else the request's purpose; the
reflection is the result it logged, else the hypothesis it settled. The measured-against state is the parent commit,
the experiment repository commit that pins the exact recipe.

A judged record changes only while its split is pending, so pending splits wait; everything else is published once and
reads ``unchanged`` afterwards. Merges stack commits already published, so they are not.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom_kb.collect import CollectionTarget, CollectReport, collect
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, NotRecorded
from hyperloom_multibranch.contracts.notebook.peers import BranchLogType, HypothesisStatus, Intent, Validity
from hyperloom_multibranch.contracts.notebook.records import (
    BranchLogRecord,
    NotebookSnapshot,
    SessionProfile,
    TestRecord,
)
from hyperloom_multibranch.contracts.notebook.vocab import COMMITTING, Conclusion
from hyperloom_multibranch.notebook.knowledge.lessons import direction

MAPPING_PATH = Path(__file__).with_name("mappings") / "multibranch-notebook-v1.yaml"
EXPORT_FORMAT = "multibranch.notebook.export.v1"
THROUGHPUT_OBJECTIVE = "output_throughput"
SUMMARY_CHARS = 160
_PENDING = frozenset({Conclusion.SPLIT_REQUESTED, Conclusion.SPLIT_PENDING})
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


def _iso(at: float) -> str:
    return datetime.fromtimestamp(at, timezone.utc).isoformat()


def _decision(record: TestRecord) -> str:
    if record.result is not None and record.result.validity is Validity.CANDIDATE_FAILURE:
        return "failed"
    return "keep" if record.conclusion in COMMITTING else "revert"


def _accuracy(record: TestRecord) -> dict[str, object] | None:
    accuracy = record.result.accuracy if record.result is not None else {}
    if not (record.request.accuracy_rounds or accuracy or record.conclusion is Conclusion.ACCURACY_FAILED):
        return None
    judged = record.judgement
    passed = record.conclusion is not Conclusion.ACCURACY_FAILED and (judged is None or judged.accuracy_ok)
    return {"required": True, "passed": passed, "value": next(iter(accuracy.values()), None)}


def _measured(record: TestRecord, metric: str) -> dict[str, object]:
    judged = record.judgement
    if judged is None:
        return {"outcome_value": None, "interval": None, "interval_text": "", "tests": [record.request_id]}
    low, high = judged.gain_pct - judged.threshold_pct, judged.gain_pct + judged.threshold_pct
    return {
        "outcome_value": judged.recorded_value.get(metric),
        "interval": {"mean": judged.gain_pct, "low": low, "high": high},
        "interval_text": f"{judged.gain_pct:+.2f}% within [{low:+.2f}%, {high:+.2f}%] over {len(judged.request_ids)} tests",
        "tests": list(judged.request_ids),
    }


def candidate_row(
    record: TestRecord, snapshot: NotebookSnapshot, profile: SessionProfile, metric: str
) -> dict[str, object] | None:
    """One export row; ``None`` for a merge, a test in flight, or a judgement waiting on its split."""

    result = record.result
    if record.request.branch == INTEGRATION_BRANCH or result is None:
        return None
    failed = result.validity is Validity.CANDIDATE_FAILURE
    if not failed and (record.judgement is None or record.conclusion is None or record.conclusion in _PENDING):
        return None
    request = record.request
    reasoning, learn = branch_notes(snapshot.branch_logs.get(request.branch, ()), record.request_id)
    return {
        "request_id": record.request_id,
        "branch": request.branch,
        "parent_commit": request.parent_commit,
        "parent_state": request.parent_commit,
        "commit_id": record.commit_id or "",
        "created_at": _iso(record.requested_at),
        "completed_at": _iso(record.judged_at or record.measured_at or record.requested_at),
        "identity": {
            **experience_identity(profile),
            **intent_conditions(snapshot.intents.get(request.intent_fingerprint)),
        },
        "direction": direction(record, snapshot),
        "change": change_document(request.delta),
        "reasoning": reasoning or request.purpose,
        "learn": learn,
        "decision": _decision(record),
        "conclusion": record.conclusion.value if record.conclusion is not None else "",
        "baseline_value": snapshot.value_of(request.parent_commit).get(metric),
        **_measured(record, metric),
        "accuracy": _accuracy(record),
        "error_class": result.failure_reason if failed else "",
        "gpu_seconds": result.gpu_seconds,
    }


def export_document(snapshot: NotebookSnapshot, metric: str) -> dict[str, object]:
    """The ``multibranch.notebook.export.v1`` document, the mapping's input; ``metric`` is the throughput metric."""

    profile = snapshot.profile
    if profile is None:
        raise NotRecorded("PRELUDE has not written the session profile yet")
    rows = (candidate_row(record, snapshot, profile, metric) for record in snapshot.records)
    return {
        "format": EXPORT_FORMAT,
        "metadata": {
            "session_id": profile.session_id,
            "objective": THROUGHPUT_OBJECTIVE if profile.isl is not None else "agentx_interactivity",
            "metric": metric,
        },
        "candidates": [row for row in rows if row is not None],
    }


def publish_experiences(
    snapshot: NotebookSnapshot,
    metric: str,
    *,
    kb: CollectionTarget | None = None,
    dry_run: bool = False,
    receipt: Path | None = None,
) -> CollectReport:
    """Write the session's Experiences to ``kb``, or to the service ``HYPERLOOM_KB_URL`` names."""

    return collect(MAPPING_PATH, export_document(snapshot, metric), kb=kb, dry_run=dry_run, receipt=receipt)


__all__ = [
    "EXPORT_FORMAT",
    "MAPPING_PATH",
    "THROUGHPUT_OBJECTIVE",
    "branch_notes",
    "candidate_row",
    "change_document",
    "experience_identity",
    "export_document",
    "intent_conditions",
    "publish_experiences",
]
