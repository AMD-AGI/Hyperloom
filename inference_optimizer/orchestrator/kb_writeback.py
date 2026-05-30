"""F2-5 — KB writeback adapters for specialist outcomes.

Each helper appends a structured record to a known KB sub-graph so
later sessions can read the trail as priors. The first writer is
:func:`write_framework_pr_record`, called by the IntegratePatchExecutor
when the integrated patch carried a ``specialist:serving:framework_pr``
provenance.

Records are JSON-Lines (one record per line) under
``framework-agent/kb/framework_optimization/lessons.jsonl``. The path
is intentionally inside the framework-agent KB tree because the
records describe upstream-PR outcomes, not orchestrator state; the
``fa`` CLI reads from the same tree on subsequent
``fa phase-discover`` calls to filter out PRs we already integrated.

Module-level :data:`KB_ROOT` is monkeypatchable in tests so the writer
can target a tmp_path without polluting the workspace KB.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

#: Default KB root for framework-PR lessons. Resolved relative to the
#: repo so ``framework-agent`` lives next to ``inference_optimizer/``.
#: Operators can override via ``INFERENCE_OPTIMIZER_FA_KB_PATH`` to
#: point at a shared mount.
def _default_kb_root() -> Path:
    override = os.environ.get("INFERENCE_OPTIMIZER_FA_KB_PATH", "").strip()
    if override:
        return Path(override) / "framework_optimization"
    return Path(__file__).parents[2] / "framework-agent" / "kb" / "framework_optimization"


KB_ROOT: Path = _default_kb_root()

#: Filename for the JSONL append log. Stable so the fa CLI / dashboards
#: can hard-code it; one file per KB sub-graph keeps the writer atomic
#: (single ``open(..., 'a')`` is append-safe under POSIX).
LESSONS_FILE: str = "lessons.jsonl"

#: Allowed ``outcome`` values. Keep stable — downstream readers
#: (fa CLI / breakdown) match on these exact strings.
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
    """Append a framework-PR outcome record to ``lessons.jsonl``.

    Async-friendly: the underlying file write is sync (POSIX append
    is atomic enough for our scale; we are not adding aiofile just
    for one append per KEPT integrate). The function is declared
    async so callers in the IntegratePatchExecutor can ``await`` it
    inline without spawning a thread for one syscall.

    Parameters mirror the F2 design:

    * ``pr_url`` / ``pr_sha`` — keys for cross-session deduplication.
    * ``patch_path`` — local snapshot path (workspace-relative usually
      survives across sessions when the worktree is preserved).
    * ``outcome`` — must be one of :data:`ALLOWED_OUTCOMES`.
    * ``tps_delta_pct`` — %-throughput delta vs. the pre-integrate
      baseline as reported by IntegratePatchExecutor's bench step.
    * ``session_id`` — orchestrator session id (audit hook).
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
    # Run the blocking file write off the event loop so a slow shared
    # filesystem cannot stall the integrate task.
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
