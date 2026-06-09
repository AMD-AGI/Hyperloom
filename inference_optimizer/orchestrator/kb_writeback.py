# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""F2-5 — KB writeback adapters for specialist outcomes.

Appends structured records as JSON-Lines under
``framework-agent/kb/framework_optimization/lessons.jsonl`` (the ``fa`` CLI
reads them to skip already-integrated PRs). :data:`KB_ROOT` is
monkeypatchable in tests.
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
ALLOWED_OUTCOMES: frozenset[str] = frozenset({
    OUTCOME_INTEGRATED,
    OUTCOME_REVERTED_SMOKE_FAIL,
    OUTCOME_REJECTED_APPLY_FAIL,
})


def _record(
    *,
    pr_url: str,
    pr_sha: str,
    patch_path: str,
    outcome: str,
    tps_delta_pct: float,
    session_id: str,
) -> dict:
    """Build the canonical record dict (sync helper for testability)."""
    return {
        "ts": time.time(),
        "session_id": str(session_id or ""),
        "pr_url": str(pr_url or ""),
        "pr_sha": str(pr_sha or ""),
        "patch_path": str(patch_path or ""),
        "outcome": str(outcome or ""),
        "tps_delta_pct": float(tps_delta_pct or 0.0),
    }


def _append_record_sync(record: dict) -> Path:
    """Append a single JSONL record under :data:`KB_ROOT`.

    Returns the on-disk path so callers can log / surface it.
    """
    KB_ROOT.mkdir(parents=True, exist_ok=True)
    path = KB_ROOT / LESSONS_FILE
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return path


async def write_framework_pr_record(
    *,
    pr_url: str,
    pr_sha: str,
    patch_path: str,
    outcome: str,
    tps_delta_pct: float,
    session_id: str,
) -> Path:
    """Append a framework-PR outcome record to ``lessons.jsonl`` (F2 design).

    Parameters:

    * ``pr_url`` / ``pr_sha`` — cross-session dedup keys.
    * ``patch_path`` — local snapshot path.
    * ``outcome`` — must be one of :data:`ALLOWED_OUTCOMES`.
    * ``tps_delta_pct`` — %-throughput delta vs. pre-integrate baseline.
    * ``session_id`` — orchestrator session id.
    """
    if outcome not in ALLOWED_OUTCOMES:
        raise ValueError(
            f"write_framework_pr_record: outcome={outcome!r} must be one of "
            f"{sorted(ALLOWED_OUTCOMES)!r}"
        )
    record = _record(
        pr_url=pr_url,
        pr_sha=pr_sha,
        patch_path=patch_path,
        outcome=outcome,
        tps_delta_pct=tps_delta_pct,
        session_id=session_id,
    )
    return await asyncio.to_thread(_append_record_sync, record)


__all__ = [
    "ALLOWED_OUTCOMES",
    "KB_ROOT",
    "LESSONS_FILE",
    "OUTCOME_INTEGRATED",
    "OUTCOME_REJECTED_APPLY_FAIL",
    "OUTCOME_REVERTED_SMOKE_FAIL",
    "write_framework_pr_record",
]
