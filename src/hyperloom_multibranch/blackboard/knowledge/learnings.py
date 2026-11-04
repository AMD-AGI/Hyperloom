# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Specialists' learnings in the Experience KB: published at CLOSE, so later sessions' specialists can retrieve them.

Every learning a specialist wrote and did not later supersede becomes one Experience of the ``multibranch-learning-v1``
schema, which this package declares itself, through the packaged ``multibranch-learning-v1`` mapping:

* conditions: the session profile under the names the ``inference-recipe-v1`` declaration gives them, plus the layer
  of the direction the learning is about, when it names one;
* the learning: its text (what retrieval matches first), kind, target and mechanism;
* evidence: each test it cites and how that test came out; a learning some cited test measured is ``measured``, any
  other ``unmeasured``.

The KB registers the schema the first time a write carries it, so nothing changes in the KB itself.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from hyperloom_kb.collect import CollectionTarget, CollectReport, collect
from hyperloom_multibranch.contracts.blackboard import NotRecorded
from hyperloom_multibranch.contracts.blackboard.peers import Validity
from hyperloom_multibranch.contracts.blackboard.records import LearningRecord, BlackboardSnapshot, TestRecord
from hyperloom_multibranch.blackboard.knowledge.experience import experience_identity
from hyperloom_multibranch.blackboard.learning_search import current_learnings, outcome_of

MAPPING_PATH = Path(__file__).with_name("mappings") / "multibranch-learning-v1.yaml"
EXPORT_FORMAT = "multibranch.notebook.learnings.v1"


def measures(record: TestRecord) -> bool:
    """Whether ``record`` measured what a learning citing it says: a judgement, or a candidate that failed."""

    return record.judgement is not None or (
        record.result is not None and record.result.validity is Validity.CANDIDATE_FAILED
    )


def _iso(at: float) -> str:
    return datetime.fromtimestamp(at, timezone.utc).isoformat()


def learning_row(record: LearningRecord, snapshot: BlackboardSnapshot) -> dict[str, object]:
    """One export row: the learning, the session's conditions, and the tests it cites."""

    learning = record.learning
    profile = snapshot.profile
    if profile is None:
        raise NotRecorded("PRELUDE has not written the session profile yet")
    by_id = {test.request_id: test for test in snapshot.records}
    cited = [by_id[request_id] for request_id in learning.request_ids if request_id in by_id]
    layer = {"layer": learning.layer.value} if learning.layer is not None else {}
    return {
        "entry_id": record.entry_id,
        "branch": learning.branch,
        "kind": learning.kind.value,
        "content": learning.content,
        "target": learning.target,
        "mechanism": learning.mechanism,
        "request_ids": list(learning.request_ids),
        "created_at": _iso(record.at),
        "measured": any(measures(test) for test in cited),
        "evidence": [f"{test.request_id} on {test.request.parent_commit}: {outcome_of(test)}" for test in cited],
        "identity": {**experience_identity(profile), **layer},
    }


def export_document(snapshot: BlackboardSnapshot) -> dict[str, object]:
    """The ``multibranch.notebook.learnings.v1`` document, the mapping's input."""

    profile = snapshot.profile
    if profile is None:
        raise NotRecorded("PRELUDE has not written the session profile yet")
    return {
        "format": EXPORT_FORMAT,
        "metadata": {"session_id": profile.session_id},
        "learnings": [learning_row(record, snapshot) for record in current_learnings(snapshot)],
    }


def publish_learnings(
    snapshot: BlackboardSnapshot,
    *,
    kb: CollectionTarget | None = None,
    dry_run: bool = False,
    receipt: Path | None = None,
) -> CollectReport:
    """CLOSE: write the session's learnings to ``kb``, or to the service ``HYPERLOOM_KB_URL`` names."""

    return collect(MAPPING_PATH, export_document(snapshot), kb=kb, dry_run=dry_run, receipt=receipt)


__all__ = [
    "EXPORT_FORMAT",
    "MAPPING_PATH",
    "export_document",
    "learning_row",
    "measures",
    "publish_learnings",
]
