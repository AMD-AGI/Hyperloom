# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json

import pytest

from hyperloom_multibranch.specialist.directions import DIRECTIONS
from hyperloom_multibranch.specialist.exhausted import (
    EXHAUSTED_PREFIX,
    Exhausted,
    ExhaustedWhy,
    exhausted_refusal,
    read_exhausted,
    session_direction,
)

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


def test_a_record_after_a_note_in_the_same_entry_is_read():
    content = "77e75ef6 regressed -8.29%, so triton loses. " + EXHAUSTED_PREFIX + json.dumps(RECORD) + " Next: Fusion."

    assert read_exhausted(content) == read_exhausted(EXHAUSTED_PREFIX + json.dumps(RECORD))


def test_a_record_names_a_session_direction_by_its_name_or_the_name_and_a_qualifier():
    assert session_direction("GEMM tuning") == next(d for d in DIRECTIONS if d.startswith("GEMM tuning:"))
    assert session_direction(" gemm TUNING (decode blockscale CK table)") == session_direction("GEMM tuning")
    assert session_direction("GEMM tunings") is None
    assert session_direction("server args/env settings") is None


# Details of blocked_by_tool records from session 4aef3318.
NAMES_NO_TRIED_PATH = "Remaining Memset needs aiter C++ change; other leads already on head or flat."
NAMES_A_CALL_BUT_NO_ERROR = (
    "The Memset (7.2%) is hipMemsetAsync inside aiter's asm splitK blockscale GEMM (aiter C++, not reachable from an "
    "sglang patch); table swaps to ck splitK0 were flat or worse (b14, b16, b10)."
)
NAMES_A_CALL_AND_ITS_ERROR = (
    "PTPC bpreshuffle replay fails with missing aiter.jit.module_gemm_a8w8_bpreshuffle; other fast-path leads (fp8 KV, "
    "unified attn, memset splitK, fused rope/quant) already covered by other branches."
)
NAMES_A_FILE_AND_ITS_ERROR = (
    "Memset (7.8%) is hipMemsetAsync at csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu:328 inside aiter C++ "
    "(splitK>1); the fused qknorm_rope replay IMA-crashes on this head."
)


def _blocked(detail: str, why: str = "blocked_by_tool") -> str:
    return EXHAUSTED_PREFIX + json.dumps({**RECORD, "why": why, "detail": detail})


@pytest.mark.parametrize("why", ["blocked_by_tool", "needs_capability"])
@pytest.mark.parametrize("detail", [NAMES_A_CALL_AND_ITS_ERROR, NAMES_A_FILE_AND_ITS_ERROR])
def test_a_blocked_record_whose_detail_names_the_path_tried_and_its_error_stands(why, detail):
    assert read_exhausted(_blocked(detail, why)).detail == detail
    assert exhausted_refusal(_blocked(detail, why)) == ""


@pytest.mark.parametrize("why", ["blocked_by_tool", "needs_capability"])
@pytest.mark.parametrize(
    ("detail", "lacks"),
    [
        (NAMES_NO_TRIED_PATH, "names no file or call you tried and no error it gave"),
        (NAMES_A_CALL_BUT_NO_ERROR, "names no error it gave"),
        ("ck_gemm_a8w8.cu was read; the kernel has no split-K variant", "names no error it gave"),
        ("the patch failed with a compile error", "names no file or call you tried"),
    ],
)
def test_a_blocked_record_whose_detail_names_no_tried_path_is_refused_saying_what_it_lacks(why, detail, lacks):
    assert read_exhausted(_blocked(detail, why)) is None

    refusal = exhausted_refusal(_blocked(detail, why))
    assert "Fusion" in refusal and why in refusal and lacks in refusal and detail in refusal


def test_other_reasons_need_no_tried_path_and_an_entry_with_no_record_is_not_refused():
    record = EXHAUSTED_PREFIX + json.dumps({**RECORD, "detail": NAMES_NO_TRIED_PATH})

    assert read_exhausted(record).detail == NAMES_NO_TRIED_PATH
    assert exhausted_refusal(record) == ""
    assert exhausted_refusal("Tried fusing the rope kernel: +0.4%.") == ""
    assert exhausted_refusal(EXHAUSTED_PREFIX + "{not json") == ""
