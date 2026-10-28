# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KB writeback adapters for specialist outcomes."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from hyperloom.agents.framework.kb import LESSONS_FILE, framework_optimization_root
from hyperloom.common.io import JsonLineFormat, append_jsonl


def _default_kb_root() -> Path:
    """Resolve the framework-PR lessons directory."""
    return framework_optimization_root()


#: Allowed ``outcome`` values; keep stable (downstream readers match exact strings).
OUTCOME_INTEGRATED: str = "integrated"
OUTCOME_REVERTED_SMOKE_FAIL: str = "reverted_smoke_fail"
OUTCOME_REJECTED_APPLY_FAIL: str = "rejected_apply_fail"
# Not benchmarked: the live tree already held every change the candidate makes.
OUTCOME_ALREADY_PRESENT: str = "already_present"
# An env-gated rewrite whose switch-off parity leg measured a behavioural change: with every switch unset the patch
# did not reproduce the base.
OUTCOME_REVERTED_SWITCH_OFF_PARITY: str = "reverted_switch_off_parity"
# The parity leg produced no usable measurement, so the invariant was never tested.
OUTCOME_REVERTED_PARITY_INCONCLUSIVE: str = "reverted_parity_inconclusive"
# An env-gated rewrite benched below the KEEP threshold and kept in the tree with every switch off, its switches
# registered as levers to bench together with others.
OUTCOME_KEPT_INERT: str = "kept_inert_levers_registered"
ALLOWED_OUTCOMES: frozenset[str] = frozenset(
    {
        OUTCOME_INTEGRATED,
        OUTCOME_REVERTED_SMOKE_FAIL,
        OUTCOME_REJECTED_APPLY_FAIL,
        OUTCOME_ALREADY_PRESENT,
        OUTCOME_REVERTED_SWITCH_OFF_PARITY,
        OUTCOME_REVERTED_PARITY_INCONCLUSIVE,
        OUTCOME_KEPT_INERT,
    }
)


def _str_list(values: list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize an optional string list for JSONL storage."""
    return [str(v).strip() for v in (values or []) if str(v).strip()]


@dataclass(frozen=True)
class FrameworkOutcome:
    """The fields of one framework-PR outcome record, before normalization."""

    pr_url: str
    pr_sha: str
    patch_path: str
    outcome: str
    tps_delta_pct: float
    session_id: str
    framework: str = ""
    gap_canonical_id: str = ""
    gap_keywords: list[str] | None = None
    model_class: str = ""
    gpu_type: str = ""
    precision: str = ""
    applicability: str = ""
    provenance: str = ""
    accuracy_delta_pct: float = 0.0
    changed_files: list[str] | None = None
    source_framework: str = ""
    target_framework: str = ""


def _record(fields: FrameworkOutcome) -> dict:
    """Build the canonical framework-PR outcome record dict."""
    return {
        "ts": time.time(),
        "session_id": str(fields.session_id or ""),
        "pr_url": str(fields.pr_url or ""),
        "pr_sha": str(fields.pr_sha or ""),
        "patch_path": str(fields.patch_path or ""),
        "outcome": str(fields.outcome or ""),
        "tps_delta_pct": float(fields.tps_delta_pct or 0.0),
        "framework": str(fields.framework or ""),
        "gap_canonical_id": str(fields.gap_canonical_id or ""),
        "gap_keywords": _str_list(fields.gap_keywords),
        "model_class": str(fields.model_class or ""),
        "gpu_type": str(fields.gpu_type or ""),
        "precision": str(fields.precision or ""),
        "applicability": str(fields.applicability or ""),
        "provenance": str(fields.provenance or ""),
        "accuracy_delta_pct": float(fields.accuracy_delta_pct or 0.0),
        "changed_files": _str_list(fields.changed_files),
        "source_framework": str(fields.source_framework or ""),
        "target_framework": str(fields.target_framework or ""),
    }


def _append_record_sync(record: dict) -> Path:
    """Append a single JSONL record under the resolved KB root."""
    path = _default_kb_root() / LESSONS_FILE
    append_jsonl(path, record, make_parents=True, fmt=JsonLineFormat(sort_keys=True))
    return path


async def write_framework_record(fields: FrameworkOutcome) -> Path:
    """Append a framework-PR outcome record to ``lessons.jsonl``."""
    if fields.outcome not in ALLOWED_OUTCOMES:
        raise ValueError(
            f"write_framework_record: outcome={fields.outcome!r} must be one of {sorted(ALLOWED_OUTCOMES)!r}"
        )
    record = _record(fields)
    return await asyncio.to_thread(_append_record_sync, record)


__all__ = [
    "ALLOWED_OUTCOMES",
    "FrameworkOutcome",
    "OUTCOME_ALREADY_PRESENT",
    "LESSONS_FILE",
    "OUTCOME_INTEGRATED",
    "OUTCOME_KEPT_INERT",
    "OUTCOME_REJECTED_APPLY_FAIL",
    "OUTCOME_REVERTED_PARITY_INCONCLUSIVE",
    "OUTCOME_REVERTED_SMOKE_FAIL",
    "OUTCOME_REVERTED_SWITCH_OFF_PARITY",
    "write_framework_record",
]
