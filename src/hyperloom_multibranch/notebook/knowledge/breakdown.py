# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``session_breakdown.json`` for a multi-branch session: the notebook's facts, laid out for people and the RSI.

The single-lineage breakdown describes one lineage of phases. A multi-branch session is a tree of branches, so this
export has its own schema version and its own sections:

* ``metadata``: the session profile, and the manifest fingerprint of each activity.
* ``baseline``: the root's recorded values, dispersion and thresholds.
* ``lineage``: every commit and branch, with when each was made, opened or stopped, and a commit's recipe digest when
  its writer named it.
* ``branches`` and ``committed``: each branch's progress, and each commit with its gain and merge state.
* ``tests``: every test record: request, result with its attempts and GPU time, judgement, conclusion, commit, times.
* ``measurement_sandboxes``: per node, GPU model, backend and image, the tests run there, their GPU time and failed
  attempts.
* ``close``: the commit final selection chose and its recipe, how the scheduler chose it, and the Recipe KB
  write-back, as the single-lineage export reports it; when the scheduler passes them, what CLOSE's knowledge steps
  did (``knowledge``: the Experiences, the learnings and the archived search tree).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from hyperloom.common.io import atomic_write_json
from hyperloom.orchestrator.knowledge.remote_recipe.models import RemoteWriteResult
from hyperloom_multibranch.contracts.notebook.peers import SandboxFacts
from hyperloom_multibranch.contracts.notebook.records import NotebookSnapshot, TestRecord
from hyperloom_multibranch.notebook.codec import encode
from hyperloom_multibranch.notebook.knowledge.recipe_writeback import FinalRecipe

SCHEMA_VERSION = "multibranch-1"
EXPORTER_VERSION = "hyperloom-multibranch-breakdown-1"
FILENAME = "session_breakdown.json"


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


def _final(final: FinalRecipe | None) -> dict[str, object] | None:
    if final is None:
        return None
    return {"commit_id": final.commit_id, "recipe_digest": final.recipe.digest(), "recipe": final.recipe.to_document()}


def build_breakdown(
    snapshot: NotebookSnapshot,
    *,
    final: FinalRecipe | None = None,
    selection: Mapping[str, object] | None = None,
    kb_write_back: RemoteWriteResult | None = None,
    knowledge: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """The breakdown document of ``snapshot``, stamped with the time the snapshot was taken.

    ``final`` is the commit final selection chose; ``selection`` is how the scheduler chose it, such as its candidates
    and their measurements, as JSON-ready data. ``knowledge`` is what CLOSE's knowledge steps did, as the
    ``rsi.session`` close reports them (``KnowledgeClose.document()``); ``close`` has it only when given.
    """

    close: dict[str, object] = {
        "final": _final(final),
        "selection": dict(selection or {}),
        "kb_write_back": encode(kb_write_back),
    }
    if knowledge is not None:
        close["knowledge"] = dict(knowledge)
    return {
        "schema_version": SCHEMA_VERSION,
        "exporter_version": EXPORTER_VERSION,
        "exported_at_utc": datetime.fromtimestamp(snapshot.taken_at, timezone.utc).isoformat(),
        "metadata": {
            "session_profile": encode(snapshot.profile),
            "manifest_fingerprints": encode(snapshot.fingerprints),
        },
        "baseline": encode(snapshot.baseline),
        "lineage": encode(snapshot.lineage),
        "branches": encode(snapshot.progress),
        "spend": {"gpu_hours": snapshot.gpu_hours, "llm_usd": snapshot.llm_usd},
        "committed": encode(snapshot.committed),
        "tests": encode(snapshot.records),
        "measurement_sandboxes": encode(measurement_usage(snapshot.records)),
        "close": close,
    }


def write_breakdown(directory: Path, breakdown: Mapping[str, object]) -> Path:
    """Write ``breakdown`` atomically as ``directory / FILENAME`` and return the path."""

    path = directory / FILENAME
    atomic_write_json(path, dict(breakdown), indent=2, sort_keys=True, make_parents=True)
    return path


__all__ = [
    "EXPORTER_VERSION",
    "FILENAME",
    "SCHEMA_VERSION",
    "MeasurementUsage",
    "build_breakdown",
    "measurement_usage",
    "write_breakdown",
]
