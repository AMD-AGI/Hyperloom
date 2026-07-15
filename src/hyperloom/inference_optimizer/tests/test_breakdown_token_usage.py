# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the promoted ``token_usage`` breakdown section.

Covers the pure projection in :func:`collectors.collect_token_usage`:
convenience totals, by-component / by-phase passthrough, decision attribution
split, and the action_timeline correlation on ``task_id``.
"""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.inference_optimizer.breakdown import collectors as col
from hyperloom.inference_optimizer.breakdown.collectors.decision import (
    _token_convenience,
    aggregate_session_cache_tokens,
)


def test_token_convenience_adds_cache_hit_rate():
    b = _token_convenience({"total_cache_creation": 100, "total_cache_read": 300})
    assert b["cache_hit_rate"] == 0.75


def test_token_convenience_cache_hit_rate_zero_without_cache():
    assert _token_convenience({})["cache_hit_rate"] == 0.0


def test_aggregate_session_cache_tokens_reads_ledger(tmp_path: Path):
    trace = tmp_path / "reports" / "trace"
    trace.mkdir(parents=True)
    (trace / "llm_calls.jsonl").write_text(
        json.dumps({"cache_creation_input_tokens": 100, "cache_read_input_tokens": 300})
        + "\n"
        + json.dumps({"cache_creation_input_tokens": 0, "cache_read_input_tokens": 100})
        + "\n",
        encoding="utf-8",
    )
    assert aggregate_session_cache_tokens(tmp_path) == (100, 400)


def test_aggregate_session_cache_tokens_missing_ledger(tmp_path: Path):
    assert aggregate_session_cache_tokens(tmp_path) == (0, 0)


def _bucket(ti, to, cc, cr, calls):
    return {
        "total_in": ti,
        "total_out": to,
        "total_cache_creation": cc,
        "total_cache_read": cr,
        "calls": calls,
    }


def _decision_trace_fixture():
    """A decision_trace dict shaped like collect_decision_trace's output.

    Session total: 3 calls. One specialist call is attributed to a decision
    (task_id=spec-1); the other two (orchestration + kernel) are unattributed.
    """
    session_total = _bucket(100, 200, 10, 20, 3)
    unattributed = _bucket(90, 50, 6, 8, 2)
    return {
        "decision_trace": [
            {
                "phase": "EXPLORE",
                "tick": 3,
                "ts": "2026-06-10T16:41:06Z",
                "decision": {
                    "component": "orchestration",
                    "change": "specialist",
                    "outcome": "KEEP",
                    "gain_pct": None,
                    "task_id": "spec-1",
                },
                "tokens": {
                    "by_component": {"specialist": _bucket(10, 150, 4, 12, 1)},
                    "total_in": 10,
                    "total_out": 150,
                    "total_cache": 16,
                    "calls": 1,
                },
            },
            {
                "phase": "EXPLORE",
                "tick": 4,
                "ts": "2026-06-10T16:56:28Z",
                "decision": {
                    "component": "orchestration",
                    "change": "noop",
                    "outcome": "KEEP",
                    "gain_pct": None,
                    "task_id": "noop-1",
                },
                "tokens": {
                    "by_component": {},
                    "total_in": 0,
                    "total_out": 0,
                    "total_cache": 0,
                    "calls": 0,
                },
            },
        ],
        "token_rollup": {
            "by_phase": {"EXPLORE": session_total},
            "by_component": {
                "specialist": _bucket(10, 150, 4, 12, 1),
                "orchestration": _bucket(80, 40, 4, 6, 1),
                "kernel_agent": _bucket(10, 10, 2, 2, 1),
            },
            "session_total": session_total,
        },
        "unattributed_tokens": unattributed,
    }


def _timeline_fixture():
    return [
        {
            "action": "specialist",
            "change": "specialist",
            "decision": "KEEP",
            "phase": "EXPLORE",
            "task_id": "spec-1",
            "ts": "2026-06-10T16:41:06Z",
        },
        {
            "action": "sglang-use-aiter-global",
            "change": "sglang-use-aiter-global",
            "decision": "REVERT",
            "phase": "EXPLORE",
            "task_id": "explore-99",
            "ts": "2026-06-10T16:51:58Z",
        },
        {
            "action": "baseline",
            "change": "baseline",
            "decision": "KEEP",
            "phase": "PRELUDE",
            "task_id": None,
            "ts": "2026-06-10T16:03:18Z",
        },
    ]


class TestTokenConvenience:
    def test_split_cache_bucket(self):
        out = col._token_convenience(_bucket(100, 200, 10, 20, 3))
        assert out["total_in_out"] == 300
        assert out["grand_total"] == 330

    def test_combined_cache_bucket(self):
        out = col._token_convenience({"total_in": 10, "total_out": 150, "total_cache": 16, "calls": 1})
        assert out["total_in_out"] == 160
        assert out["grand_total"] == 176

    def test_none_is_zeroed(self):
        out = col._token_convenience(None)
        assert out["total_in_out"] == 0
        assert out["grand_total"] == 0


class TestCollectTokenUsage:
    def test_session_total_has_convenience_figures(self):
        out = col.collect_token_usage(_decision_trace_fixture(), _timeline_fixture(), [])
        st = out["session_total"]
        assert st["calls"] == 3
        assert st["total_in_out"] == 300  # 100 + 200
        assert st["grand_total"] == 330  # + 10 + 20

    def test_by_component_and_by_phase_passthrough(self):
        out = col.collect_token_usage(_decision_trace_fixture(), _timeline_fixture(), [])
        assert set(out["by_component"]) == {"specialist", "orchestration", "kernel_agent"}
        assert out["by_component"]["specialist"]["grand_total"] == 176
        assert out["by_phase"]["EXPLORE"]["grand_total"] == 330

    def test_attribution_split(self):
        out = col.collect_token_usage(_decision_trace_fixture(), _timeline_fixture(), [])
        attr = out["attribution"]
        # attributed = session_total - unattributed = (100-90, 200-50, 10-6, 20-8, 3-2)
        assert attr["attributed_to_decisions"]["calls"] == 1
        assert attr["attributed_to_decisions"]["total_in"] == 10
        assert attr["unattributed"]["calls"] == 2
        assert attr["attributed_calls_pct"] == round(100.0 / 3, 2)

    def test_timeline_correlates_tokens_by_task_id(self):
        out = col.collect_token_usage(_decision_trace_fixture(), _timeline_fixture(), [])
        rows = {r["action"]: r for r in out["timeline"]}
        # spec-1 has tokens; the others (no matching decision tokens) are null.
        assert rows["specialist"]["tokens"] is not None
        assert rows["specialist"]["tokens"]["total_in_out"] == 160
        assert rows["sglang-use-aiter-global"]["tokens"] is None
        assert rows["baseline"]["task_id"] is None
        assert rows["baseline"]["tokens"] is None

    def test_attribution_splits_overhead_from_unattributed(self):
        # Add an overhead bucket so the three-way split reconciles to session_total.
        dt = _decision_trace_fixture()
        dt["overhead_tokens"] = _bucket(80, 40, 4, 6, 1)
        dt["unattributed_tokens"] = _bucket(10, 10, 2, 2, 1)
        out = col.collect_token_usage(dt, _timeline_fixture(), [])
        attr = out["attribution"]
        # attributed = session_total - unattributed - overhead
        assert attr["attributed_to_decisions"]["calls"] == 1
        assert attr["attributed_to_decisions"]["total_in"] == 10
        assert attr["overhead"]["calls"] == 1
        assert attr["overhead"]["total_in"] == 80
        assert attr["unattributed"]["calls"] == 1
        assert attr["overhead_calls_pct"] == round(100.0 / 3, 2)

    def test_empty_decision_trace_is_safe(self):
        out = col.collect_token_usage({}, [], [])
        assert out["session_total"]["calls"] == 0
        assert out["session_total"]["grand_total"] == 0
        assert out["by_component"] == {}
        assert out["timeline"] == []

    def test_zero_call_decision_not_in_timeline_tokens(self):
        # A calls=0 decision must stay tokens=null (no zero-bucket injection).
        tl = [
            {"action": "noop", "change": "noop", "decision": "KEEP", "phase": "EXPLORE", "task_id": "noop-1", "ts": "x"}
        ]
        out = col.collect_token_usage(_decision_trace_fixture(), tl, [])
        assert out["timeline"][0]["tokens"] is None
