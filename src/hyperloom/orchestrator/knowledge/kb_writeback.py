# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KB writeback adapters for specialist outcomes."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from hyperloom.common.io import append_jsonl


def _default_kb_root() -> Path:
    """Resolve the framework-PR lessons directory."""
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
# An env-gated rewrite whose switch-off parity leg measured a behavioural change: with every switch unset the patch
# did not reproduce the base.
OUTCOME_REVERTED_SWITCH_OFF_PARITY: str = "reverted_switch_off_parity"
# The parity leg produced no usable measurement, so the invariant was never tested.
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
    """Build the canonical framework-PR outcome record dict."""
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
    """Append a single JSONL record under the resolved KB root."""
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
    """Append a framework-PR outcome record to ``lessons.jsonl``."""
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
