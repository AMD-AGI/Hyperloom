#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Batch-index rotation for the gte100 dispatcher (optimize-gte100.yml).

The pool is larger than one batch, so each scheduled fire runs ONE batch and the
batch index advances over time to sweep the whole pool, wrapping at the end.
This is the single source of truth for that arithmetic so it can be unit-tested
instead of living only inside a workflow's shell heredoc.

CLI (used by the workflow):
    python3 batch_rotate.py --count N --batch-size B \
        [--batch-index I] [--event schedule|workflow_dispatch] \
        [--now-iso 2026-06-24T04:07:00+00:00]
prints a single TSV line: "<batch_index>\\t<count>\\t<batches>\\t<start>\\t<end>"
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone

# Anchor day for the scheduled rotation. Fires happen twice a day (UTC 04 & 16);
# slot advances by 2 each day and by 1 at the 12:00 UTC boundary.
ROTATE_EPOCH = datetime(2026, 6, 24, tzinfo=timezone.utc).date()
FIRES_PER_DAY = 2


def num_batches(count: int, batch_size: int) -> int:
    """Number of batches needed to cover ``count`` items at ``batch_size``.

    Always at least 1 (an empty pool still has a single, empty batch 0).
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return max(math.ceil(max(count, 0) / batch_size), 1)


def scheduled_slot(now: datetime, epoch=ROTATE_EPOCH) -> int:
    """Monotonic fire slot for ``now`` given two fires/day from ``epoch``.

    slot = days_since_epoch * 2 + (0 if before 12:00 UTC else 1). Days before
    the epoch clamp to 0 so the slot is never negative.
    """
    now = now.astimezone(timezone.utc)
    days = max((now.date() - epoch).days, 0)
    return days * FIRES_PER_DAY + (0 if now.hour < 12 else 1)


def resolve_batch_index(count: int, batch_size: int, *, event: str,
                        batch_index: str | None, now: datetime | None = None) -> int:
    """Resolve the batch index to run.

    On ``schedule`` (or when no explicit index is given) the index auto-rotates
    from the current fire slot; on manual dispatch the provided index is used.
    The result is always wrapped into ``[0, batches)``.
    """
    batches = num_batches(count, batch_size)
    raw = (batch_index or "").strip()
    if event == "schedule" or not raw:
        slot = scheduled_slot(now or datetime.now(timezone.utc))
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
    args = p.parse_args(argv)

    now = datetime.fromisoformat(args.now_iso) if args.now_iso else None
    bi = resolve_batch_index(args.count, args.batch_size, event=args.event,
                             batch_index=args.batch_index, now=now)
    batches = num_batches(args.count, args.batch_size)
    start, end = slice_bounds(bi, args.batch_size, args.count)
    print(f"{bi}\t{args.count}\t{batches}\t{start}\t{end}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
