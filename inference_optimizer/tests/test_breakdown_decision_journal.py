"""Unit tests for decision_journal collector (v1.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.breakdown import build


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.fixture
def params_round_session(tmp_path: Path) -> Path:
    sd = tmp_path / "session"
    sd.mkdir()
    _write_json(sd / "manifest.json", {"schema_version": 1, "session_id": "dj"})
    _write_json(sd / "state.json", {
        "session_id": "dj",
        "baseline_tput": 1000.0,
        "current_best": {"tput": 1070.0},
        "params_attempts": [{
            "ts": "2026-05-15T10:00:00+00:00",
            "task_id": "p-task-1",
            "status": "succeeded",
            "decision": "discarded",
            "extras": {
                "round_id": "params-001",
                "best_variant_name": "ncds_16",
                "gain_vs_cb": 0.56,
                "best_gain_pct_vs_base": 0.56,
                "promotion_rule": "below_threshold",
                "promotion_rule_detail": (
                    "gain_vs_cb=0.56% < single_shot_threshold=0.2% "
                    "and no cross_round_consistent winner"
                ),
                "keep_threshold_pct": 0.2,
                "accuracy_gate_passed": None,
                "variants_tested_count": 2,
            },
        }],
        "backend_winners_history": [{
            "action": "params",
            "round_id": "params-001",
            "base_tput": 1000.0,
            "ts": "2026-05-15T10:00:00+00:00",
            "winners": [{
                "name": "ncds_16",
                "fingerprint": "fp-win",
                "gain_pct": 0.56,
                "tput": 1005.6,
                "extra_sglang_args": "--foo 1",
                "extra_envs": {"SGLANG_NUM_CONTINUOUS_DECODE_STEPS": "16"},
            }],
            "best": {"name": "ncds_16", "fingerprint": "fp-win", "gain_pct": 0.56},
        }],
        "params_search": {
            "schema_version": 2,
            "tested": {
                "fp-win": {
                    "name": "ncds_16",
                    "fingerprint": "fp-win",
                    "gain_pct": 0.56,
                    "extra_sglang_args": "--foo 1",
                    "extra_envs": {"SGLANG_NUM_CONTINUOUS_DECODE_STEPS": "16"},
                    "result": {"status": "succeeded", "output_throughput": 1005.6},
                },
                "fp-lose": {
                    "name": "bad_knob",
                    "fingerprint": "fp-lose",
                    "gain_pct": -2.0,
                    "extra_sglang_args": "--bad",
                    "result": {"status": "succeeded", "output_throughput": 980.0},
                },
            },
            "rejected": [{
                "name": "bad_knob",
                "fingerprint": "fp-lose",
                "reason": "not_keep",
                "gain_pct": -2.0,
            }],
            "last_round": {
                "base_tput": 1000.0,
                "tested_fp": ["fp-win", "fp-lose"],
                "round_winners": ["ncds_16"],
                "selected_new": [],
            },
        },
    })
    return sd


def test_decision_journal_from_winners_history(params_round_session: Path) -> None:
    journal = build(params_round_session)["decision_journal"]
    assert len(journal) >= 1
    entry = journal[0]
    assert entry["phase"] == "params"
    assert entry["round_id"] == "params-001"
    assert entry["baseline_ref_tput"] == pytest.approx(1000.0)
    names = {v["name"] for v in entry["variants"]}
    assert "ncds_16" in names
    assert "bad_knob" in names
    rejected = [v for v in entry["variants"] if v["name"] == "bad_knob"][0]
    assert rejected["outcome"] == "rejected"
    assert rejected["reject_reason"] == "not_keep"
    assert entry["round_decision"]["outcome"] == "discarded"
    assert entry["round_decision"]["best_variant_name"] == "ncds_16"
    assert entry["round_decision"]["promotion_rule"] == "below_threshold"
    assert entry["round_decision"]["keep_threshold_pct"] == pytest.approx(0.2)
    assert entry["round_decision"]["variants_tested_count"] == 2


def test_decision_journal_detail_level_verbose(params_round_session: Path) -> None:
    b = build(params_round_session, detail_level="verbose")
    assert b["detail_level"] == "verbose"
    assert len(b["decision_journal"][0]["variants"]) == 2
