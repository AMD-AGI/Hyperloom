# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""KB writeback adapters for specialist outcomes.

Appends structured records as JSON-Lines under
``framework-agent/kb/framework_optimization/lessons.jsonl`` (the ``fa`` CLI
reads them to skip already-integrated PRs). :data:`KB_ROOT` is
monkeypatchable in tests.

Gbrain outcomes write-back: Hyperloom's role is
LOCAL-ONLY — it never talks to gbrain directly. ``lessons.jsonl`` is a
fixed, shared (not session-scoped) append-only file that a separate
``Primus-Claw/knowledge/pr`` worker task tails and mirrors into gbrain's
``pr-kb-outcomes/`` pages (see that repo's ``pr_kb/outcomes_sync.py`` — the
sole owner of the gbrain write path for this data). This module's only job
is to keep appending here reliably.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path


#: Default KB root for framework-PR lessons; override via
#: ``INFERENCE_OPTIMIZER_FA_KB_PATH``.
def _default_kb_root() -> Path:
    """Resolve the default KB root for framework-PR lessons.

    Honours the ``INFERENCE_OPTIMIZER_FA_KB_PATH`` override when set;
    otherwise derives a repo-relative path so ``framework-agent`` sits
    next to ``src/hyperloom/inference_optimizer/``.

    Returns:
        Path: The ``framework_optimization`` directory under the resolved
            KB root.
    """
    override = os.environ.get("INFERENCE_OPTIMIZER_FA_KB_PATH", "").strip()
    if override:
        return Path(override) / "framework_optimization"
    return Path(__file__).parents[2] / "framework-agent" / "kb" / "framework_optimization"


KB_ROOT: Path = _default_kb_root()

#: Filename for the JSONL append log; stable so the fa CLI can hard-code it
#: (single POSIX append is atomic).
LESSONS_FILE: str = "lessons.jsonl"

#: Allowed ``outcome`` values; keep stable (downstream readers match exact strings).
OUTCOME_INTEGRATED: str = "integrated"
OUTCOME_REVERTED_SMOKE_FAIL: str = "reverted_smoke_fail"
OUTCOME_REJECTED_APPLY_FAIL: str = "rejected_apply_fail"
# Candidate skipped because the semantic audit found it already present
# in the live tree (cross-session dedup so later runs don't re-audit it).
OUTCOME_ALREADY_PRESENT: str = "already_present"
ALLOWED_OUTCOMES: frozenset[str] = frozenset(
    {
        OUTCOME_INTEGRATED,
        OUTCOME_REVERTED_SMOKE_FAIL,
        OUTCOME_REJECTED_APPLY_FAIL,
        OUTCOME_ALREADY_PRESENT,
    }
)


def _str_list(values: list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize an optional string list for JSONL storage."""
    return [str(v).strip() for v in (values or []) if str(v).strip()]


def _record(
    *,
    pr_url: str,
    pr_sha: str,
    patch_path: str,
    outcome: str,
    tps_delta_pct: float,
    session_id: str,
    framework: str = "",
    gap_canonical_id: str = "",
    gap_keywords: list[str] | None = None,
    model_class: str = "",
    gpu_type: str = "",
    precision: str = "",
    applicability: str = "",
    provenance: str = "",
    accuracy_delta_pct: float = 0.0,
    changed_files: list[str] | None = None,
    source_framework: str = "",
    target_framework: str = "",
) -> dict:
    """Build the canonical framework-PR outcome record dict.

    Sync helper kept separate from the async writer for testability.

    Args:
        pr_url (str): URL of the upstream PR the patch came from.
        pr_sha (str): Commit SHA of the PR head (cross-session dedup key).
        patch_path (str): Local snapshot path of the applied patch.
        outcome (str): One of :data:`ALLOWED_OUTCOMES`.
        tps_delta_pct (float): %-throughput delta vs. the pre-integrate
            baseline.
        session_id (str): Orchestrator session id (audit hook).
        framework (str): Source framework of the PR (rating prior signal).
        gap_canonical_id (str): Canonical gap id (rating prior exact match key).
        gap_keywords (list[str] | None): Gap keywords for fuzzy association.
        model_class (str): Model class or family for parameter-PR association.
        gpu_type (str): GPU type for workload-aware ranking.
        precision (str): Precision mode for workload-aware ranking.
        applicability (str): Audit/apply route classification.
        provenance (str): Source path of the record.
        accuracy_delta_pct (float): Accuracy delta when available.
        changed_files (list[str] | None): PR changed files used for ranking.
        source_framework (str): Cross-framework source framework, when known.
        target_framework (str): Cross-framework target framework, when known.

    Returns:
        dict: The record with a ``ts`` timestamp plus the stringified /
            coerced input fields.
    """
    return {
        "ts": time.time(),
        "session_id": str(session_id or ""),
        "pr_url": str(pr_url or ""),
        "pr_sha": str(pr_sha or ""),
        "patch_path": str(patch_path or ""),
        "outcome": str(outcome or ""),
        "tps_delta_pct": float(tps_delta_pct or 0.0),
        "framework": str(framework or ""),
        "gap_canonical_id": str(gap_canonical_id or ""),
        "gap_keywords": _str_list(gap_keywords),
        "model_class": str(model_class or ""),
        "gpu_type": str(gpu_type or ""),
        "precision": str(precision or ""),
        "applicability": str(applicability or ""),
        "provenance": str(provenance or ""),
        "accuracy_delta_pct": float(accuracy_delta_pct or 0.0),
        "changed_files": _str_list(changed_files),
        "source_framework": str(source_framework or ""),
        "target_framework": str(target_framework or ""),
    }


def _append_record_sync(record: dict) -> Path:
    """Append a single JSONL record under :data:`KB_ROOT`.

    Creates the KB root directory if needed, then appends one JSON line.

    Args:
        record (dict): The record dict to serialise (see :func:`_record`).

    Returns:
        Path: The on-disk path of the ``lessons.jsonl`` file so callers
            can log / surface it.
    """
    KB_ROOT.mkdir(parents=True, exist_ok=True)
    path = KB_ROOT / LESSONS_FILE
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return path


async def write_framework_record(
    *,
    pr_url: str,
    pr_sha: str,
    patch_path: str,
    outcome: str,
    tps_delta_pct: float,
    session_id: str,
    framework: str = "",
    gap_canonical_id: str = "",
    gap_keywords: list[str] | None = None,
    model_class: str = "",
    gpu_type: str = "",
    precision: str = "",
    applicability: str = "",
    provenance: str = "",
    accuracy_delta_pct: float = 0.0,
    changed_files: list[str] | None = None,
    source_framework: str = "",
    target_framework: str = "",
) -> Path:
    """Append a framework-PR outcome record to ``lessons.jsonl``.

    Parameters:

    * ``pr_url`` / ``pr_sha`` — cross-session dedup keys.
    * ``patch_path`` — local snapshot path.
    * ``outcome`` — must be one of :data:`ALLOWED_OUTCOMES`.
    * ``tps_delta_pct`` — %-throughput delta vs. pre-integrate baseline.
    * ``session_id`` — orchestrator session id.
    * ``framework`` — source framework of the PR (rating prior signal).
    * ``gap_canonical_id`` — canonical gap id (rating prior exact match key).
    * ``gap_keywords`` — gap keywords (rating prior fuzzy Jaccard match).
    * ``model_class`` / ``gpu_type`` / ``precision`` — workload parameters.
    * ``applicability`` / ``provenance`` — apply route and source context.
    * ``accuracy_delta_pct`` — optional accuracy delta.
    * ``changed_files`` — PR file paths used for fuzzy association.
    * ``source_framework`` / ``target_framework`` — cross-framework context.
    """
    if outcome not in ALLOWED_OUTCOMES:
        raise ValueError(f"write_framework_record: outcome={outcome!r} must be one of {sorted(ALLOWED_OUTCOMES)!r}")
    record = _record(
        pr_url=pr_url,
        pr_sha=pr_sha,
        patch_path=patch_path,
        outcome=outcome,
        tps_delta_pct=tps_delta_pct,
        session_id=session_id,
        framework=framework,
        gap_canonical_id=gap_canonical_id,
        gap_keywords=gap_keywords,
        model_class=model_class,
        gpu_type=gpu_type,
        precision=precision,
        applicability=applicability,
        provenance=provenance,
        accuracy_delta_pct=accuracy_delta_pct,
        changed_files=changed_files,
        source_framework=source_framework,
        target_framework=target_framework,
    )
    return await asyncio.to_thread(_append_record_sync, record)


__all__ = [
    "ALLOWED_OUTCOMES",
    "OUTCOME_ALREADY_PRESENT",
    "KB_ROOT",
    "LESSONS_FILE",
    "OUTCOME_INTEGRATED",
    "OUTCOME_REJECTED_APPLY_FAIL",
    "OUTCOME_REVERTED_SMOKE_FAIL",
    "write_framework_record",
]
