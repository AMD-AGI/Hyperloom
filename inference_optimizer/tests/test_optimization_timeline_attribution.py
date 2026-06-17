# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for proposer / operation_kind / per-variant attribution in the
optimization timeline (phase_timeline + decision_trace).

Verifies the augmentation that threads "who proposed", "what kind of change"
(operation_kind), and "how it measured" (per-variant metrics + proposal scores)
from the explore executor through the journal into the breakdown timeline.
"""

from __future__ import annotations

import json
from pathlib import Path

from inference_optimizer.breakdown import collectors


def test_journal_event_carries_proposer_and_operation_kind() -> None:
    from inference_optimizer.breakdown.collectors import _journal_entry_to_event

    ev = _journal_entry_to_event({
        "ts": "2026-06-12T00:00:01Z",
        "kind": "backend",
        "change": "--attention-backend AITER",
        "outcome": "KEEP",
        "phase": "EXPLORE",
        "gain_pct": 4.2,
        "variant_name": "v01",
        "provenance": "specialist:serving_specialist",
        "scope": "domain",
        "fingerprint": "fp1",
        "metrics": {"runtime_sec": 30.0},
    })
    extras = ev["extras"]
    assert extras["operation_kind"] == "backend"
    assert extras["provenance"] == "specialist:serving_specialist"
    assert extras["proposer"] == "specialist:serving_specialist"
    assert extras["scope"] == "domain"
    assert extras["fingerprint"] == "fp1"
    assert extras["metrics"] == {"runtime_sec": 30.0}


def test_decision_trace_resolves_proposer_kind_and_scores(tmp_path: Path) -> None:
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "optimization_journal.json").write_text(
        json.dumps({"entries": [
            {"ts": "2026-06-12T00:00:01Z", "kind": "backend",
             "change": "--attention-backend AITER", "outcome": "KEEP",
             "phase": "EXPLORE", "gain_pct": 4.2, "task_id": "t1",
             "variant_name": "v01", "provenance": "specialist:serving_specialist",
             "scope": "domain", "fingerprint": "fp1",
             "metrics": {"runtime_sec": 30.0, "wall_clock_ratio_vs_baseline": 1.1}},
            {"ts": "2026-06-12T00:00:05Z", "kind": "param",
             "change": "--max-num-batched-tokens 8192", "outcome": "REVERT",
             "phase": "EXPLORE", "task_id": "t2", "variant_name": "v02",
             "provenance": "default_grid"},
        ]}),
        encoding="utf-8",
    )
    state = {
        "specialist_rounds": [{
            "round_id": "r1", "domain": "serving",
            "ensemble_scores": {"scale": "0-10", "models": {
                "modelA": {"v01": {"score": 8.5, "reason": "strong"}},
                "modelB": {"v01": {"score": 7.0, "reason": "ok"}},
            }},
        }],
    }
    out = collectors.collect_decision_trace(tmp_path, state, [])
    rows = out["decision_trace"]
    by_task = {r["decision"].get("task_id"): r["decision"] for r in rows}

    keep = by_task["t1"]
    assert keep["component"] == "specialist:serving_specialist"  # resolved proposer
    assert keep["operation_kind"] == "backend"
    assert keep["provenance"] == "specialist:serving_specialist"
    assert keep["scope"] == "domain"
    assert keep["fingerprint"] == "fp1"
    assert keep["metrics"]["runtime_sec"] == 30.0
    # proposal_scorer signal joined by variant_name.
    raters = {s["rater"] for s in keep["proposal_scores"]}
    assert raters == {"modelA", "modelB"}

    revert = by_task["t2"]
    assert revert["component"] == "grid"  # default_grid -> grid
    assert revert["operation_kind"] == "param"


def test_shape_ledger_entry_tags_operation_kind_and_proposer() -> None:
    from inference_optimizer.breakdown.collectors import _shape_ledger

    ledger = {
        "schema_version": 1,
        "tested": {
            "fpA": {"name": "v01", "fingerprint": "fpA",
                    "extra_server_args": "--attention-backend AITER",
                    "gain_pct": 3.0, "provenance": "specialist:serving",
                    "scope": "domain"},
        },
        "accepted": [], "rejected": [],
    }
    shaped = _shape_ledger(ledger)
    entry = shaped["top_by_gain"][0]
    assert entry["operation_kind"] == "backend"
    assert entry["provenance"] == "specialist:serving"
    assert entry["proposer"] == "specialist:serving"
    assert entry["scope"] == "domain"
