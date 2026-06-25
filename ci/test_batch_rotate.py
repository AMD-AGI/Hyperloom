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


def _utc2(y, m, d, h, mi):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


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


# ── scheduled_slot (max_hours-paced) ────────────────────────────────────────
# Anchor = 2026-06-25 03:07 UTC; gte100 step = 12h.


def test_slot_anchor_is_zero():
    assert br.scheduled_slot(_utc2(2026, 6, 25, 3, 7), max_hours=12) == 0


def test_slot_advances_one_per_max_hours():
    # +12h -> slot 1; +24h -> slot 2.
    assert br.scheduled_slot(_utc2(2026, 6, 25, 15, 7), max_hours=12) == 1
    assert br.scheduled_slot(_utc2(2026, 6, 26, 3, 7), max_hours=12) == 2


def test_slot_within_window_stays():
    # Just under +12h is still slot 0.
    assert br.scheduled_slot(_utc2(2026, 6, 25, 15, 6), max_hours=12) == 0


def test_slot_step_scales_with_max_hours():
    # 6h step: +6h -> slot 1, +12h -> slot 2.
    assert br.scheduled_slot(_utc2(2026, 6, 25, 9, 7), max_hours=6) == 1
    assert br.scheduled_slot(_utc2(2026, 6, 25, 15, 7), max_hours=6) == 2


def test_slot_default_step_when_max_hours_unset():
    # Unset/invalid -> DEFAULT_MAX_HOURS (12h).
    assert br.scheduled_slot(_utc2(2026, 6, 25, 15, 7), max_hours=None) == 1
    assert br.scheduled_slot(_utc2(2026, 6, 25, 15, 7), max_hours=0) == 1


def test_slot_before_anchor_clamps_to_zero():
    assert br.scheduled_slot(_utc(2026, 6, 1, 4), max_hours=12) == 0


def test_slot_other_tz_is_normalized():
    # +08:00 11:07 == 03:07 UTC == anchor -> slot 0.
    tz8 = timezone(timedelta(hours=8))
    assert br.scheduled_slot(datetime(2026, 6, 25, 11, 7, tzinfo=tz8), max_hours=12) == 0


# ── resolve_batch_index ─────────────────────────────────────────────────────


def test_resolve_schedule_rotates_and_wraps():
    # 15 batches, 12h step. anchor -> batch 0, +12h -> batch 1.
    assert br.resolve_batch_index(900, 60, event="schedule", batch_index=None,
                                  now=_utc2(2026, 6, 25, 3, 7), max_hours=12) == 0
    assert br.resolve_batch_index(900, 60, event="schedule", batch_index=None,
                                  now=_utc2(2026, 6, 25, 15, 7), max_hours=12) == 1
    # +15 steps (15*12h = 7.5 days) -> slot 15 -> wraps to 0
    assert br.resolve_batch_index(900, 60, event="schedule", batch_index=None,
                                  now=_utc2(2026, 7, 2, 15, 7), max_hours=12) == 0


def test_resolve_schedule_ignores_input_index():
    # On schedule, a stray input index is ignored in favour of the slot.
    assert br.resolve_batch_index(900, 60, event="schedule", batch_index="9",
                                  now=_utc2(2026, 6, 25, 3, 7), max_hours=12) == 0


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
                                  batch_index="", now=_utc2(2026, 6, 25, 15, 7),
                                  max_hours=12) == 1


# ── slice_bounds ────────────────────────────────────────────────────────────


def test_slice_bounds_full_batch():
    assert br.slice_bounds(0, 60, 900) == (0, 60)
    assert br.slice_bounds(1, 60, 900) == (60, 120)


def test_slice_bounds_last_partial_batch():
    # 900 models, batch 14 -> 840..900 (full); 901 models -> batch 15 -> 900..901.
    assert br.slice_bounds(14, 60, 900) == (840, 900)
    assert br.slice_bounds(15, 60, 901) == (900, 901)
