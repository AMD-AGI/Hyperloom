# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json

from hyperloom_multibranch.specialist.exhausted import EXHAUSTED_PREFIX, Exhausted, ExhaustedWhy, read_exhausted

RECORD = {
    "direction": "Fusion",
    "head": "a" * 40,
    "tried": ["r1", "r2"],
    "why": "no_ceiling_above_bar",
    "detail": "no elementwise op above 1% of GPU time is left",
    "reopen_if": "the head's profile changes",
}


def test_a_record_is_read_back_from_its_branch_log_entry():
    assert read_exhausted(EXHAUSTED_PREFIX + json.dumps(RECORD)) == Exhausted(
        "Fusion",
        "a" * 40,
        ("r1", "r2"),
        ExhaustedWhy.NO_CEILING_ABOVE_BAR,
        "no elementwise op above 1% of GPU time is left",
        "the head's profile changes",
    )


def test_an_entry_that_is_no_record_or_is_malformed_reads_as_none():
    assert read_exhausted("Tried fusing the rope kernel: +0.4%.") is None
    assert read_exhausted(EXHAUSTED_PREFIX + "{not json") is None
    assert read_exhausted(EXHAUSTED_PREFIX + json.dumps({**RECORD, "why": "bored"})) is None
    assert read_exhausted(EXHAUSTED_PREFIX + json.dumps({k: v for k, v in RECORD.items() if k != "head"})) is None


def test_a_record_that_cites_no_test_is_no_record():
    assert read_exhausted(EXHAUSTED_PREFIX + json.dumps({**RECORD, "tried": []})) is None
