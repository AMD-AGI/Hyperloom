#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Batch-index rotation for the gte100 dispatcher (optimize-gte100.yml).

The pool is larger than one batch, so each scheduled fire runs ONE batch and the
batch index advances over time to sweep the whole pool, wrapping at the end.
This is the single source of truth for that arithmetic so it can be unit-tested
instead of living only inside a workflow's shell heredoc.

The rotation is paced by ``max_hours`` (the per-task optimizer budget), mirroring
optimize-submit's generate_hf_matrix._cron_batch_index: one batch advances per
``max_hours`` of wall-clock time since the anchor.

INVARIANT (must hold): the schedule cron PERIOD must equal ``max_hours``.
``slot = floor(elapsed / max_hours)`` only advances exactly one batch per fire
when each fire is ``max_hours`` apart:
  * cron faster than max_hours (e.g. 4x/day but max_hours=12 -> step 12h = 2
    fires/slot): two consecutive fires land on the SAME slot -> the SAME batch
    is dispatched twice.
  * cron slower than max_hours: some slots are never hit -> batches are skipped.
This is NOT robust to arbitrary cron changes — it is robust to the ANCHOR
shifting, not to the period. gte100 deliberately pairs a 12h cron with
max_hours=12 (and optimize-submit a 6h cron with max_hours=6); if you change one,
change the other to match. NOTE: gte100 dispatches with
exclude_active_workflows=false, so it has NO de-dup safety net — a period/
max_hours mismatch will actually double-submit a batch.

CLI (used by the workflow):
    python3 batch_rotate.py --count N --batch-size B --max-hours H \
        [--batch-index I] [--event schedule|workflow_dispatch] \
        [--now-iso 2026-06-25T03:07:00+00:00]
prints a single TSV line: "<batch_index>\\t<count>\\t<batches>\\t<start>\\t<end>"
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone

# Anchor instant: the first scheduled fire that maps to batch 0. 2026-06-25
# 03:07 UTC (Beijing 11:07) — the 06-24 fires never ran (that day's dispatches
# all failed with HTTP 422), so the rotation is anchored here to actually start
# the sweep at batch 0 instead of skipping ahead by the calendar clock.
ROTATE_ANCHOR_UTC = datetime(2026, 6, 25, 3, 7, tzinfo=timezone.utc)

# Fallback optimizer budget (hours) used as the rotation step size when
# --max-hours is unset/invalid. Keep in sync with the gte100 dispatcher default.
DEFAULT_MAX_HOURS = 12.0


def num_batches(count: int, batch_size: int) -> int:
    """Number of batches needed to cover ``count`` items at ``batch_size``.

    Always at least 1 (an empty pool still has a single, empty batch 0).
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return max(math.ceil(max(count, 0) / batch_size), 1)


def _step_hours(max_hours: float | None) -> float:
    """Rotation step size in hours; falls back to the default when not positive."""
    try:
        h = float(max_hours) if max_hours is not None else 0.0
    except (TypeError, ValueError):
        h = 0.0
    return h if h > 0 else DEFAULT_MAX_HOURS


def scheduled_slot(now: datetime, *, max_hours: float | None = None,
                   anchor: datetime = ROTATE_ANCHOR_UTC) -> int:
    """Monotonic batch slot for ``now``, advancing one per ``max_hours``.

    slot = floor((now - anchor) / max_hours), clamped at 0 for instants before
    the anchor (so the sweep never wraps to the tail before batch 0).
    """
    now = now.astimezone(timezone.utc)
    elapsed_h = (now - anchor).total_seconds() / 3600.0
    return max(int(elapsed_h // _step_hours(max_hours)), 0)


def resolve_batch_index(count: int, batch_size: int, *, event: str,
                        batch_index: str | None, now: datetime | None = None,
                        max_hours: float | None = None) -> int:
    """Resolve the batch index to run.

    On ``schedule`` (or when no explicit index is given) the index auto-rotates
    from the max_hours-paced slot; on manual dispatch the provided index is used.
    The result is always wrapped into ``[0, batches)``.
    """
    batches = num_batches(count, batch_size)
    raw = (batch_index or "").strip()
    if event == "schedule" or not raw:
        slot = scheduled_slot(now or datetime.now(timezone.utc), max_hours=max_hours)
        return slot % batches
    return int(raw) % batches


def slice_bounds(batch_index: int, batch_size: int, count: int) -> tuple[int, int]:
    """Return the ``[start, end)`` item bounds for ``batch_index``."""
    start = batch_index * batch_size
    end = min(start + batch_size, count)
    return start, end


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--batch-index", default="")
    p.add_argument("--event", default="workflow_dispatch")
    p.add_argument("--now-iso", default="")
    p.add_argument("--max-hours", default="", help="Rotation step size in hours (default 12)")
    args = p.parse_args(argv)

    now = datetime.fromisoformat(args.now_iso) if args.now_iso else None
    max_hours = float(args.max_hours) if args.max_hours else None
    bi = resolve_batch_index(args.count, args.batch_size, event=args.event,
                             batch_index=args.batch_index, now=now,
                             max_hours=max_hours)
    batches = num_batches(args.count, args.batch_size)
    start, end = slice_bounds(bi, args.batch_size, args.count)
    print(f"{bi}\t{args.count}\t{batches}\t{start}\t{end}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
