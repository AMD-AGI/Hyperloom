# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``session_breakdown.json`` for a multi-branch session: the notebook's facts, laid out for people and the RSI.

The single-lineage breakdown describes one lineage of phases. A multi-branch session is a tree of branches, so this
export has its own schema version and its own sections:

* ``metadata``: the session profile, the manifest fingerprint of each activity, and ``session``, the block Pulse reads
  for both flows (the session id, the source revision the session ran on, and the Claw session, framework image and
  node a hosted run is given). In a hosted run it also holds, as written, what the run leaves in the session directory:
  ``plugs`` and ``hosted_measurement``, and the records ``measurement_sandboxes`` (one per test, keyed by
  ``request_id`` as ``tests`` are). From the agent runner's samples it holds ``specialist_resources``, one record per
  specialist, and ``container_resources``, the container's last reading.
* ``baseline``: the root's recorded values, dispersion and thresholds.
* ``lineage``: every commit and branch, with when each was made, opened or stopped, and a commit's recipe digest when
  its writer named it.
* ``branches`` and ``committed``: each branch's progress, and each commit with its gain and merge state.
* ``tests``: every test record: request, result with its attempts and GPU time, judgement, conclusion, commit, times.
* ``measurement_nodes``: per node, GPU model, backend and image, the tests run there, their GPU time and failed
  attempts; ``environment_changes``: where a node's image, GPU model, power cap or partition mode moved between its
  results.
* ``agent_manifests``: by autonomous branch, the manifest of the third-party agent that drove it, credentials dropped.
* ``close``: the commit final selection chose and its recipe, how the scheduler chose it, and the Recipe KB
  write-back, as the single-lineage export reports it; when the scheduler passes them, what CLOSE's knowledge steps
  did (``knowledge``: the Experiences, the learnings and the archived search tree) and how the session's merges went
  (``merges``: the integrator's conflict statistics); given the session's metric, each branch whose lead over
  integration was stranded, for how many minutes and at what peak (``stranded``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from hyperloom.common.io import atomic_write_json
from hyperloom.orchestrator.knowledge.remote_recipe.models import RemoteWriteResult
from hyperloom.orchestrator.specialists.resources import (
    CONTAINER_RECORD_KIND,
    RESOURCES_RELATIVE_PATH,
    SPECIALIST_RECORD_KIND,
)
from hyperloom_multibranch.contracts.notebook.peers import SandboxFacts
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot, TestRecord
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.notebook.knowledge.recipe_writeback import FinalRecipe
from hyperloom_multibranch.notebook.knowledge.stranded_spans import stranded_spans

SCHEMA_VERSION = "multibranch-2"
EXPORTER_VERSION = "hyperloom-multibranch-breakdown-2"
FILENAME = "session_breakdown.json"
#: The environment variables a hosted run's ``metadata.session`` reads, by key; a run Claw did not host leaves them null.
#: ``image`` is the framework image the CampaignSpec names, which Pulse keys its leaderboard on.
SESSION_VARIABLES = {
    "claw_session_id": "CLAW_SESSION_ID",
    "image": "HYPERLOOM_IMAGE",
    "node_name": "NODE_NAME",
}
#: The records a hosted run appends in the session directory, by the ``metadata`` key the breakdown copies them under.
HOSTED_RECORDS = {"measurement_sandboxes": Path("hosted/measurement_sandboxes.jsonl")}
#: The documents a hosted run writes in the session directory, by the ``metadata`` key the breakdown copies them under.
HOSTED_DOCUMENTS = {
    "plugs": Path("plugs.resolved.json"),
    "hosted_measurement": Path("hosted/measurement_stats.json"),
}


@dataclass
class MeasurementUsage:
    """What one measurement environment ran: tests, their GPU time, and the attempts that failed and were retried."""

    node: str
    gpu_model: str
    backend: str
    image_digest: str
    tests: int = 0
    gpu_seconds: float = 0.0
    failed_attempts: int = 0


def _environment(sandbox: SandboxFacts | None) -> tuple[str, str, str, str]:
    """Node, GPU model, backend and image; all empty for a test that never reached a sandbox."""

    if sandbox is None:
        return ("", "", "", "")
    return (sandbox.node_name or "", sandbox.gpu_model or "", sandbox.backend, sandbox.image_digest or "")


def measurement_usage(records: Iterable[TestRecord]) -> tuple[MeasurementUsage, ...]:
    """Usage per measurement environment, sorted by node, GPU model, backend and image."""

    usage: dict[tuple[str, str, str, str], MeasurementUsage] = {}
    for record in records:
        if record.result is None:
            continue
        key = _environment(record.result.sandbox)
        row = usage.setdefault(key, MeasurementUsage(*key))
        row.tests += 1
        row.gpu_seconds += record.result.gpu_seconds
        row.failed_attempts += sum(attempt.failure is not None for attempt in record.result.attempts)
    return tuple(usage[key] for key in sorted(usage))


def session_spend(snapshot: NotebookSnapshot) -> dict[str, object]:
    """GPU hours and LLM dollars so far, and the tokens of the LLM calls the dollars leave out for want of a price."""

    return {
        "gpu_hours": snapshot.gpu_hours,
        "llm_usd": snapshot.llm_usd,
        "llm_unpriced_tokens": encode(snapshot.llm_unpriced_tokens),
    }


def session_metadata(
    snapshot: NotebookSnapshot, environ: Mapping[str, str], code_revision: str | None = None
) -> dict[str, object]:
    """The ``metadata.session`` block: the session id, the source revision the session ran on as it recorded it at
    start, and what :data:`SESSION_VARIABLES` name; null where unset."""

    session_id = snapshot.profile.session_id if snapshot.profile is not None else None
    named = {key: environ.get(name) or None for key, name in SESSION_VARIABLES.items()}
    return {"session_id": session_id, "code_revision": code_revision or None, **named}


def _rows(path: Path) -> list[object] | None:
    """The rows of the JSON lines file ``path``, without a row that is no JSON, as a half-written last line is;
    ``None`` when there is no such file."""

    if not path.is_file():
        return None
    rows: list[object] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def hosted_records(session_dir: Path | None) -> dict[str, object]:
    """What a hosted run left in ``session_dir``, as written, by its ``metadata`` key: each :data:`HOSTED_DOCUMENTS`
    file's document, and each :data:`HOSTED_RECORDS` file's rows. A file that is not there gives no key; a document
    that is no JSON gives none either, and a row that is no JSON, as a half-written last line is, is left out."""

    if session_dir is None:
        return {}
    found: dict[str, object] = {}
    for key, relative in HOSTED_DOCUMENTS.items():
        try:
            found[key] = json.loads((session_dir / relative).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    for key, relative in HOSTED_RECORDS.items():
        rows = _rows(session_dir / relative)
        if rows is not None:
            found[key] = rows
    return found


def agent_resources(runner_dir: Path | None) -> dict[str, object]:
    """What the agent runner sampled under ``runner_dir``, the session directory it was given, by its ``metadata``
    key: each specialist's record as ``specialist_resources``, and as ``container_resources`` the container's last
    reading, whose peaks and kill count cover the run up to it. No samples give neither key, and no container reading
    gives no ``container_resources``."""

    rows = _rows(runner_dir / RESOURCES_RELATIVE_PATH) if runner_dir is not None else None
    if rows is None:
        return {}
    records = [row for row in rows if isinstance(row, dict)]
    found: dict[str, object] = {
        "specialist_resources": [row for row in records if row.get("kind") == SPECIALIST_RECORD_KIND]
    }
    readings = [row for row in records if row.get("kind") == CONTAINER_RECORD_KIND]
    if readings:
        found["container_resources"] = readings[-1]
    return found


def _final(final: FinalRecipe | None) -> dict[str, object] | None:
    if final is None:
        return None
    return {"commit_id": final.commit_id, "recipe_digest": final.recipe.digest(), "recipe": final.recipe.to_document()}


@dataclass(frozen=True)
class CloseOutcome:
    """What CLOSE did, for the breakdown's ``close`` section.

    ``final`` is the commit final selection chose; ``selection`` is how the scheduler chose it, such as its candidates
    and their measurements, as JSON-ready data. ``knowledge`` is what CLOSE's knowledge steps did, as the
    ``rsi.session`` close reports them (``KnowledgeClose.document()``), and ``merges`` how the session's merges went,
    as the integrator counts them, JSON-ready; ``close`` has each only when given. ``metric`` is the session's: given
    it, ``close`` holds the stranded leads' spans on it (:func:`stranded_spans`).
    """

    final: FinalRecipe | None = None
    selection: Mapping[str, object] | None = None
    kb_write_back: RemoteWriteResult | None = None
    knowledge: Mapping[str, object] | None = None
    merges: Mapping[str, object] | None = None
    metric: str = ""


@dataclass(frozen=True)
class SessionPlace:
    """Where the breakdown reads a session's other records, and what the session ran under.

    ``session_dir`` is where a hosted run's records are read from, ``runner_dir`` the agent runner's session directory,
    and ``environ`` the run's variables (the process's by default); ``code_revision`` is the source revision the
    session's manifest recorded at start, which a process that exports another run's session, such as ``final-only``,
    does not have in its environment.
    """

    session_dir: Path | None = None
    runner_dir: Path | None = None
    environ: Mapping[str, str] | None = None
    code_revision: str | None = None


def build_breakdown(
    snapshot: NotebookSnapshot, outcome: CloseOutcome | None = None, place: SessionPlace | None = None
) -> dict[str, object]:
    """The breakdown document of ``snapshot``, stamped with the time the snapshot was taken; ``outcome`` is what CLOSE
    did and ``place`` where the session's other records are, each empty by default."""

    outcome, place = outcome or CloseOutcome(), place or SessionPlace()
    close: dict[str, object] = {
        "final": _final(outcome.final),
        "selection": dict(outcome.selection or {}),
        "kb_write_back": encode(outcome.kb_write_back),
    }
    if outcome.knowledge is not None:
        close["knowledge"] = dict(outcome.knowledge)
    if outcome.merges is not None:
        close["merges"] = dict(outcome.merges)
    if outcome.metric:
        close["stranded"] = encode(stranded_spans(snapshot, outcome.metric))
    return {
        "schema_version": SCHEMA_VERSION,
        "exporter_version": EXPORTER_VERSION,
        "exported_at_utc": datetime.fromtimestamp(snapshot.taken_at, timezone.utc).isoformat(),
        "metadata": {
            "session_profile": encode(snapshot.profile),
            "manifest_fingerprints": encode(snapshot.fingerprints),
            "session": session_metadata(
                snapshot, os.environ if place.environ is None else place.environ, place.code_revision
            ),
            **hosted_records(place.session_dir),
            **agent_resources(place.runner_dir),
        },
        "baseline": encode(snapshot.baseline),
        "lineage": encode(snapshot.lineage),
        "branches": encode(snapshot.progress),
        "spend": session_spend(snapshot),
        "committed": encode(snapshot.committed),
        "tests": encode(snapshot.records),
        "measurement_nodes": encode(measurement_usage(snapshot.records)),
        "environment_changes": encode(snapshot.environment_changes),
        "agent_manifests": encode(snapshot.agent_manifests),
        "close": close,
    }


def write_breakdown(directory: Path, breakdown: Mapping[str, object]) -> Path:
    """Write ``breakdown`` atomically as ``directory / FILENAME`` and return the path."""

    path = directory / FILENAME
    atomic_write_json(path, dict(breakdown))
    return path


__all__ = [
    "EXPORTER_VERSION",
    "FILENAME",
    "HOSTED_DOCUMENTS",
    "HOSTED_RECORDS",
    "SCHEMA_VERSION",
    "SESSION_VARIABLES",
    "CloseOutcome",
    "MeasurementUsage",
    "SessionPlace",
    "agent_resources",
    "build_breakdown",
    "hosted_records",
    "measurement_usage",
    "session_metadata",
    "session_spend",
    "write_breakdown",
]
