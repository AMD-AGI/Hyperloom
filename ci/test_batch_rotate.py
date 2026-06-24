# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ci/batch_rotate.py — the gte100 dispatcher batch rotation."""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

_CI_DIR = Path(__file__).resolve().parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

import batch_rotate as br  # noqa: E402


def _utc(y, m, d, h):
    return datetime(y, m, d, h, tzinfo=timezone.utc)


# ── num_batches ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("count,bs,expected", [
    (900, 60, 15),
    (899, 60, 15),
    (840, 60, 14),
    (60, 60, 1),
    (1, 60, 1),
    (0, 60, 1),     # empty pool still has a single batch 0
    (61, 60, 2),
])
def test_num_batches(count, bs, expected):
    assert br.num_batches(count, bs) == expected


def test_num_batches_invalid_size():
    with pytest.raises(ValueError):
        br.num_batches(100, 0)


# ── scheduled_slot ──────────────────────────────────────────────────────────


def test_slot_epoch_morning_is_zero():
    # Epoch day, before 12:00 UTC -> slot 0.
    assert br.scheduled_slot(_utc(2026, 6, 24, 4), epoch=br.ROTATE_EPOCH) == 0


def test_slot_epoch_afternoon_is_one():
    assert br.scheduled_slot(_utc(2026, 6, 24, 16), epoch=br.ROTATE_EPOCH) == 1


def test_slot_advances_two_per_day():
    # Next day morning -> slot 2; next day afternoon -> slot 3.
    assert br.scheduled_slot(_utc(2026, 6, 25, 4), epoch=br.ROTATE_EPOCH) == 2
    assert br.scheduled_slot(_utc(2026, 6, 25, 16), epoch=br.ROTATE_EPOCH) == 3


def test_slot_noon_boundary():
    # Exactly 12:00 UTC counts as the afternoon fire.
    assert br.scheduled_slot(_utc(2026, 6, 24, 12), epoch=br.ROTATE_EPOCH) == 1
    assert br.scheduled_slot(_utc(2026, 6, 24, 11), epoch=br.ROTATE_EPOCH) == 0


def test_slot_before_epoch_clamps_to_zero():
    assert br.scheduled_slot(_utc(2026, 6, 1, 4), epoch=br.ROTATE_EPOCH) == 0


def test_slot_naive_or_other_tz_is_normalized():
    # +08:00 13:00 == 05:00 UTC (morning) -> slot 0 on epoch day.
    tz8 = timezone(timedelta(hours=8))
    assert br.scheduled_slot(datetime(2026, 6, 24, 13, tzinfo=tz8), epoch=br.ROTATE_EPOCH) == 0


# ── resolve_batch_index ─────────────────────────────────────────────────────


def test_resolve_schedule_rotates_and_wraps():
    # 15 batches; slot 0 -> 0, and 15 days later (slot 30) wraps to 0.
    assert br.resolve_batch_index(900, 60, event="schedule", batch_index=None,
                                  now=_utc(2026, 6, 24, 4)) == 0
    assert br.resolve_batch_index(900, 60, event="schedule", batch_index=None,
                                  now=_utc(2026, 6, 24, 16)) == 1
    # day +7 afternoon -> slot 15 -> wraps to 0
    assert br.resolve_batch_index(900, 60, event="schedule", batch_index=None,
                                  now=_utc(2026, 7, 1, 16)) == 0


def test_resolve_schedule_ignores_input_index():
    # On schedule, a stray input index is ignored in favour of the slot.
    assert br.resolve_batch_index(900, 60, event="schedule", batch_index="9",
                                  now=_utc(2026, 6, 24, 4)) == 0


def test_resolve_dispatch_uses_input():
    assert br.resolve_batch_index(900, 60, event="workflow_dispatch",
                                  batch_index="3") == 3


def test_resolve_dispatch_wraps_oversized_index():
    # 15 batches; index 17 wraps to 2.
    assert br.resolve_batch_index(900, 60, event="workflow_dispatch",
                                  batch_index="17") == 2


def test_resolve_dispatch_empty_input_falls_back_to_slot():
    # Manual run with no index -> behaves like schedule (slot-based).
    assert br.resolve_batch_index(900, 60, event="workflow_dispatch",
                                  batch_index="", now=_utc(2026, 6, 24, 16)) == 1


# ── slice_bounds ────────────────────────────────────────────────────────────


def test_slice_bounds_full_batch():
    assert br.slice_bounds(0, 60, 900) == (0, 60)
    assert br.slice_bounds(1, 60, 900) == (60, 120)


def test_slice_bounds_last_partial_batch():
    # 900 models, batch 14 -> 840..900 (full); 901 models -> batch 15 -> 900..901.
    assert br.slice_bounds(14, 60, 900) == (840, 900)
    assert br.slice_bounds(15, 60, 901) == (900, 901)
