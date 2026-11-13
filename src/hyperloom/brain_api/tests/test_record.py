# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A recorded call reads back exactly as it was written; the sample file is the format an evaluator reads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.brain_api import (
    CONTRACT_VERSION,
    BrainCall,
    CallSite,
    Candidate,
    ConfigDelta,
    Proposal,
    ProposeAsk,
    Review,
    ReviewAsk,
    SearchState,
    WireError,
    Workload,
    append_call,
    read_calls,
)

_SAMPLES = Path(__file__).parent / "samples" / "calls.jsonl"


def _state() -> SearchState:
    return SearchState(Workload("m", "vllm"), baseline_tput=1.0, current_tput=1.0, keep_threshold_pct=1.0)


def test_the_samples_cover_every_call_site_and_read_back_byte_for_byte() -> None:
    calls = list(read_calls(_SAMPLES))
    assert sorted(c.call_site for c in calls) == sorted(CallSite)
    assert {c.contract_version for c in calls} == {CONTRACT_VERSION}
    lines = [json.loads(line) for line in _SAMPLES.read_text().splitlines() if line.strip()]
    assert [c.to_wire() for c in calls] == lines


def test_append_then_read_returns_the_same_calls(tmp_path) -> None:
    path = tmp_path / "brain_calls.jsonl"
    propose = BrainCall(CallSite.PROPOSE, CONTRACT_VERSION, "v0", _state(), ProposeAsk(task="t"), Proposal())
    failed = BrainCall(
        CallSite.REVIEW,
        CONTRACT_VERSION,
        "v0",
        _state(),
        ReviewAsk((Candidate("a", ConfigDelta("--a")),)),
        error="timeout",
        duration_sec=3.0,
    )
    append_call(path, propose)
    append_call(path, failed)
    assert list(read_calls(path)) == [propose, failed]


def test_a_call_must_pair_its_ask_and_answer_and_hold_one_of_answer_or_error() -> None:
    with pytest.raises(WireError, match="takes a ProposeAsk"):
        BrainCall(
            CallSite.PROPOSE,
            CONTRACT_VERSION,
            "v0",
            _state(),
            ReviewAsk((Candidate("a", ConfigDelta("--a")),)),
            Review(),
        )
    with pytest.raises(WireError, match="answers a Proposal"):
        BrainCall(CallSite.PROPOSE, CONTRACT_VERSION, "v0", _state(), ProposeAsk(task="t"), Review())
    with pytest.raises(WireError, match="exactly one"):
        BrainCall(CallSite.PROPOSE, CONTRACT_VERSION, "v0", _state(), ProposeAsk(task="t"))
    with pytest.raises(WireError, match="exactly one"):
        BrainCall(CallSite.PROPOSE, CONTRACT_VERSION, "v0", _state(), ProposeAsk(task="t"), Proposal(), error="x")


def test_a_bad_line_is_named(tmp_path) -> None:
    path = tmp_path / "calls.jsonl"
    good = _SAMPLES.read_text().splitlines()[0]
    path.write_text(good + "\n\n" + json.dumps({"call_site": "plan"}) + "\n")
    with pytest.raises(WireError, match=r"calls.jsonl:3"):
        list(read_calls(path))
