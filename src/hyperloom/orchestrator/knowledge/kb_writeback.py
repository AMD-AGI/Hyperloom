# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KB writeback adapters for specialist outcomes.

Appends structured records as JSON-Lines to ``<KB_ROOT>/lessons.jsonl``
(the ``fa`` CLI reads them to skip already-integrated PRs). The root comes
from :func:`hyperloom.agents.framework.kb.mutable_kb_root`, the same owner the
reader uses, so both halves move together; resolving it here independently is
what left written lessons unreadable by the next session. Local append only.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path

from hyperloom.common.io import append_jsonl


def _default_kb_root() -> Path:
    """Resolve the framework-PR lessons directory.

    Delegates the root to the reader's owner so a deployment that redirects
    the KB redirects both halves. Resolved per call rather than at import,
    because the environment is not fully settled at import time.

    Returns:
        Path: The ``framework_optimization`` directory under the resolved
            KB root.
    """
    from hyperloom.agents.framework.kb import framework_optimization_root

    return framework_optimization_root()

#: Filename for the JSONL append log; stable so the fa CLI can hard-code it
#: (single POSIX append is atomic).
LESSONS_FILE: str = "lessons.jsonl"

#: Allowed ``outcome`` values; keep stable (downstream readers match exact strings).
OUTCOME_INTEGRATED: str = "integrated"
OUTCOME_REVERTED_SMOKE_FAIL: str = "reverted_smoke_fail"
OUTCOME_REJECTED_APPLY_FAIL: str = "rejected_apply_fail"
# Candidate skipped: semantic audit found it already in the live tree.
OUTCOME_ALREADY_PRESENT: str = "already_present"
# An env-gated rewrite whose switch-off parity leg measured a behavioural change:
# with every switch unset the patch did not reproduce the base. A property of the
# patch, and a genuine lesson for later sessions.
OUTCOME_REVERTED_SWITCH_OFF_PARITY: str = "reverted_switch_off_parity"
# The parity leg produced no usable measurement, so the invariant was never tested.
# Kept separate from the verdict above on purpose: recording an unmeasured run as a
# parity violation would teach every later session that a rewrite breaks when
# disabled on the strength of a failed measurement.
OUTCOME_REVERTED_PARITY_INCONCLUSIVE: str = "reverted_parity_inconclusive"
ALLOWED_OUTCOMES: frozenset[str] = frozenset(
    {
        OUTCOME_INTEGRATED,
        OUTCOME_REVERTED_SMOKE_FAIL,
        OUTCOME_REJECTED_APPLY_FAIL,
        OUTCOME_ALREADY_PRESENT,
        OUTCOME_REVERTED_SWITCH_OFF_PARITY,
        OUTCOME_REVERTED_PARITY_INCONCLUSIVE,
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
    """Append a single JSONL record under the resolved KB root.

    Creates the KB root directory if needed, then appends one JSON line.

    Args:
        record (dict): The record dict to serialise (see :func:`_record`).

    Returns:
        Path: The on-disk path of the ``lessons.jsonl`` file so callers
            can log / surface it.
    """
    path = _default_kb_root() / LESSONS_FILE
    append_jsonl(path, record, make_parents=True, sort_keys=True)
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
    session_dir: Path | str | None = None,
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
    * ``session_dir`` — when set, the KB write is additionally instrumented
      into the session breakdown; telemetry failures are swallowed and never
      fail the write.

    Returns:
        Path: The ``lessons.jsonl`` file the record was appended to.

    Raises:
        ValueError: If ``outcome`` is not in :data:`ALLOWED_OUTCOMES`.
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
    path = await asyncio.to_thread(_append_record_sync, record)
    if session_dir:
        try:
            from hyperloom.inference_optimizer.breakdown.recorder import instrument

            identity = "|".join((pr_sha, pr_url, gap_canonical_id, outcome))
            operation_key = hashlib.sha256(identity.encode("utf-8", errors="replace")).hexdigest()[:16]
            operation_id = f"op:framework-kb:{operation_key}"
            artifact_id = f"artifact:framework-kb:{operation_key}"
            measurement_id = f"measurement:framework-kb:{operation_key}:gain"
            instrument.record_measurement(
                session_dir,
                measurement_id=measurement_id,
                operation_id=operation_id,
                kind="gain",
                name="throughput_delta",
                value=float(tps_delta_pct or 0.0),
                unit="percent",
                status="succeeded",
                producer="framework-kb",
                metric_basis="framework_candidate",
                source={"field": "tps_delta_pct"},
            )
            instrument.record_artifact(
                session_dir,
                artifact_id=artifact_id,
                operation_id=operation_id,
                producer_operation_id=operation_id,
                kind="kb_record",
                path=str(path),
                present=True,
                status="available",
                producer="framework-kb",
            )
            instrument.record_operation(
                session_dir,
                operation_id=operation_id,
                root_operation_id=operation_id,
                kind="kb_write",
                name="framework optimization lesson",
                status="succeeded",
                source="framework_kb_writeback",
                executor_class="deterministic",
                purpose="knowledge_write",
                strategy_group="framework",
                strategy=str(provenance or "framework"),
                producer="framework-kb",
                inputs=record,
                outputs={"path": str(path), "outcome": outcome},
                measurement_refs=[measurement_id],
                artifact_refs=[artifact_id],
            )
            instrument.record_trace_event(
                session_dir,
                trace_event_id=f"trace:{operation_id}:finalized",
                operation_id=operation_id,
                kind="kb_write_finalized",
                status="succeeded",
                outcome=outcome,
                producer="framework-kb",
            )
        except Exception:  # noqa: BLE001 -- KB persistence must not depend on telemetry
            pass
    return path


__all__ = [
    "ALLOWED_OUTCOMES",
    "OUTCOME_ALREADY_PRESENT",
    "LESSONS_FILE",
    "OUTCOME_INTEGRATED",
    "OUTCOME_REJECTED_APPLY_FAIL",
    "OUTCOME_REVERTED_PARITY_INCONCLUSIVE",
    "OUTCOME_REVERTED_SMOKE_FAIL",
    "OUTCOME_REVERTED_SWITCH_OFF_PARITY",
    "write_framework_record",
]
