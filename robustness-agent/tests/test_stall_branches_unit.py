# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Branch coverage for robustness_agent.signals.stall."""

from __future__ import annotations

from robustness_agent.role.prompt_inputs import InboxItem, ReactorContext
from robustness_agent.signals.stall import (
    StallConfig,
    _coerce_unix,
    _collect_last_seen,
    evaluate_stall_signals,
)
from robustness_agent.sources.base import SourceData
from robustness_agent.signals.symptom import SymptomSeverity


def _item(from_agent: str, ts=None) -> InboxItem:
    payload = {"ts": ts} if ts is not None else {}
    return InboxItem(seq=1, msg_id="m", from_agent=from_agent, topic="heartbeat", payload=payload)


def test_coerce_unix_variants() -> None:
    assert _coerce_unix(None) is None
    assert _coerce_unix(5) == 5.0
    assert _coerce_unix("2026-01-01T00:00:00Z") > 0  # ISO parse (line 157)
    assert _coerce_unix("123.5") == 123.5  # float fallback (lines 159-160)
    assert _coerce_unix("not-a-time") is None  # lines 161-162


def test_collect_last_seen_branches() -> None:
    inbox = [
        _item("user", ts=999.0),  # untracked -> skip (line 113)
        _item("kernel", ts=50.0),  # tracked -> set (lines 117-119)
        _item("kernel"),  # no ts -> skip
    ]
    events = [
        {"agent": "critic"},  # no ts -> continue (line 127)
        {"agent": "orchestration", "ts": 200.0},
        {"agent": "user", "ts": 5.0},  # untracked
    ]
    last = _collect_last_seen(inbox, events)
    assert last == {"kernel": 50.0, "orchestration": 200.0}


def test_evaluate_stall_emits_high() -> None:
    ctx = ReactorContext(inbox=[_item("orchestration", ts=100.0)], now_unix=10_000.0)
    data = SourceData(coordinator_events=[])
    symptoms = evaluate_stall_signals(ctx, data, config=StallConfig(stall_timeout_s=300.0, severity_high_after_s=900.0))
    assert len(symptoms) == 1
    assert symptoms[0].name == "agent_stall"
    assert symptoms[0].severity == SymptomSeverity.HIGH


def test_evaluate_stall_no_activity_no_symptom() -> None:
    # No ground truth for any agent -> no accusations.
    ctx = ReactorContext(inbox=[], now_unix=10_000.0)
    data = SourceData(coordinator_events=[])
    assert evaluate_stall_signals(ctx, data) == []
