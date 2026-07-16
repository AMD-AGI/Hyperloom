# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
    (0, 60, 1),     # empty pool still has batch 0
    (61, 60, 2),
])
def test_num_batches(count, bs, expected):
    assert br.num_batches(count, bs) == expected


def test_num_batches_invalid_size():
    with pytest.raises(ValueError):
        br.num_batches(100, 0)


# ── scheduled_slot (max_hours-paced): anchor = 2026-06-25 03:07 UTC, step 12h ──


def test_slot_anchor_is_zero():
    assert br.scheduled_slot(_utc2(2026, 6, 25, 3, 7), max_hours=12) == 0


def test_slot_advances_one_per_max_hours():
    assert br.scheduled_slot(_utc2(2026, 6, 25, 15, 7), max_hours=12) == 1
    assert br.scheduled_slot(_utc2(2026, 6, 26, 3, 7), max_hours=12) == 2


def test_slot_within_window_stays():
    assert br.scheduled_slot(_utc2(2026, 6, 25, 15, 6), max_hours=12) == 0


def test_slot_step_scales_with_max_hours():
    assert br.scheduled_slot(_utc2(2026, 6, 25, 9, 7), max_hours=6) == 1
    assert br.scheduled_slot(_utc2(2026, 6, 25, 15, 7), max_hours=6) == 2


def test_slot_default_step_when_max_hours_unset():
    assert br.scheduled_slot(_utc2(2026, 6, 25, 15, 7), max_hours=None) == 1
    assert br.scheduled_slot(_utc2(2026, 6, 25, 15, 7), max_hours=0) == 1


def test_slot_before_anchor_clamps_to_zero():
    assert br.scheduled_slot(_utc(2026, 6, 1, 4), max_hours=12) == 0


def test_slot_other_tz_is_normalized():
    tz8 = timezone(timedelta(hours=8))
    assert br.scheduled_slot(datetime(2026, 6, 25, 11, 7, tzinfo=tz8), max_hours=12) == 0


def test_slot_custom_anchor_overrides_default():
    # A caller-supplied anchor restarts the sweep at that instant (slot 0).
    anchor = _utc2(2026, 6, 26, 13, 0)
    assert br.scheduled_slot(_utc2(2026, 6, 26, 13, 7), max_hours=6, anchor=anchor) == 0
    assert br.scheduled_slot(_utc2(2026, 6, 26, 19, 7), max_hours=6, anchor=anchor) == 1
    assert br.scheduled_slot(_utc2(2026, 6, 26, 13, 7), max_hours=6) > 0


def test_resolve_batch_index_custom_anchor():
    anchor = _utc2(2026, 6, 26, 13, 0)
    assert br.resolve_batch_index(2121, 240, event="schedule", batch_index=None,
                                  now=_utc2(2026, 6, 26, 13, 7), max_hours=6,
                                  anchor=anchor) == 0
    assert br.resolve_batch_index(2121, 240, event="schedule", batch_index=None,
                                  now=_utc2(2026, 6, 26, 19, 7), max_hours=6,
                                  anchor=anchor) == 1


# ── resolve_batch_index ─────────────────────────────────────────────────────


def test_resolve_schedule_rotates_and_wraps():
    assert br.resolve_batch_index(900, 60, event="schedule", batch_index=None,
                                  now=_utc2(2026, 6, 25, 3, 7), max_hours=12) == 0
    assert br.resolve_batch_index(900, 60, event="schedule", batch_index=None,
                                  now=_utc2(2026, 6, 25, 15, 7), max_hours=12) == 1
    # slot 15 wraps to 0
    assert br.resolve_batch_index(900, 60, event="schedule", batch_index=None,
                                  now=_utc2(2026, 7, 2, 15, 7), max_hours=12) == 0


def test_resolve_schedule_ignores_input_index():
    assert br.resolve_batch_index(900, 60, event="schedule", batch_index="9",
                                  now=_utc2(2026, 6, 25, 3, 7), max_hours=12) == 0


def test_resolve_dispatch_uses_input():
    assert br.resolve_batch_index(900, 60, event="workflow_dispatch",
                                  batch_index="3") == 3


def test_resolve_dispatch_wraps_oversized_index():
    assert br.resolve_batch_index(900, 60, event="workflow_dispatch",
                                  batch_index="17") == 2


def test_resolve_dispatch_empty_input_falls_back_to_slot():
    assert br.resolve_batch_index(900, 60, event="workflow_dispatch",
                                  batch_index="", now=_utc2(2026, 6, 25, 15, 7),
                                  max_hours=12) == 1


# ── slice_bounds ────────────────────────────────────────────────────────────


def test_slice_bounds_full_batch():
    assert br.slice_bounds(0, 60, 900) == (0, 60)
    assert br.slice_bounds(1, 60, 900) == (60, 120)


def test_slice_bounds_last_partial_batch():
    assert br.slice_bounds(14, 60, 900) == (840, 900)
    assert br.slice_bounds(15, 60, 901) == (900, 901)


# ── _step_hours error branch + main() CLI ───────────────────────────────────


def test_step_hours_invalid_falls_back():
    assert br._step_hours("not-a-number") == br.DEFAULT_MAX_HOURS
    assert br._step_hours(None) == br.DEFAULT_MAX_HOURS
    assert br._step_hours(6) == 6.0


def test_main_emits_tsv(capsys):
    rc = br.main([
        "--count", "900",
        "--batch-size", "60",
        "--event", "schedule",
        "--now-iso", "2026-06-25T15:07:00+00:00",
        "--max-hours", "12",
    ])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out == "1\t900\t15\t60\t120"


def test_main_with_custom_anchor(capsys):
    rc = br.main([
        "--count", "120",
        "--batch-size", "60",
        "--event", "schedule",
        "--now-iso", "2026-06-26T13:07:00+00:00",
        "--max-hours", "6",
        "--anchor", "2026-06-26T13:00:00+00:00",
    ])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "0\t120\t2\t0\t60"
