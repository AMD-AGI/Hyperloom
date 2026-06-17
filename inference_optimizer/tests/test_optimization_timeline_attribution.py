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

    ev = _journal_entry_to_event(
        {
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
        }
    )
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
        json.dumps(
            {
                "entries": [
                    {
                        "ts": "2026-06-12T00:00:01Z",
                        "kind": "backend",
                        "change": "--attention-backend AITER",
                        "outcome": "KEEP",
                        "phase": "EXPLORE",
                        "gain_pct": 4.2,
                        "task_id": "t1",
                        "variant_name": "v01",
                        "provenance": "specialist:serving_specialist",
                        "scope": "domain",
                        "fingerprint": "fp1",
                        "predicted_gain_pct": 9.0,
                        "metrics": {"runtime_sec": 30.0, "wall_clock_ratio_vs_baseline": 1.1},
                    },
                    {
                        "ts": "2026-06-12T00:00:05Z",
                        "kind": "param",
                        "change": "--max-num-batched-tokens 8192",
                        "outcome": "REVERT",
                        "phase": "EXPLORE",
                        "task_id": "t2",
                        "variant_name": "v02",
                        "provenance": "default_grid",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    state = {
        "specialist_rounds": [
            {
                "round_id": "r1",
                "domain": "serving",
                "ensemble_scores": {
                    "scale": "0-10",
                    "models": {
                        "modelA": {"v01": {"score": 8.5, "reason": "strong"}},
                        "modelB": {"v01": {"score": 7.0, "reason": "ok"}},
                    },
                },
            }
        ],
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
    # Predicted gain surfaced for calibration (predicted 9.0 vs realized 4.2).
    assert keep["predicted_gain_pct"] == 9.0
    # proposal_scorer signal joined by variant_name.
    raters = {s["rater"] for s in keep["proposal_scores"]}
    assert raters == {"modelA", "modelB"}

    revert = by_task["t2"]
    assert revert["component"] == "grid"  # default_grid -> grid
    assert revert["operation_kind"] == "param"


def test_decision_trace_routes_overhead_vs_unattributed(tmp_path: Path) -> None:
    # A specialist + scorer call both keyed to t1 (attributed); an
    # orchestration + critic call with no key (overhead); a stray keyless
    # 'kernel' call (genuinely unattributed).
    trace = tmp_path / "reports" / "trace"
    trace.mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "optimization_journal.json").write_text(
        json.dumps({"entries": [
            {"ts": "2026-06-12T00:00:01Z", "kind": "backend", "change": "x",
             "outcome": "KEEP", "phase": "EXPLORE", "gain_pct": 1.0,
             "task_id": "t1", "provenance": "specialist:perf"},
        ]}),
        encoding="utf-8",
    )
    rows = [
        {"component": "specialist", "task_id": "t1", "input_tokens": 10,
         "output_tokens": 5, "ts": "2026-06-12T00:00:00Z"},
        {"component": "proposal_scorer", "task_id": "t1", "input_tokens": 3,
         "output_tokens": 2, "ts": "2026-06-12T00:00:00Z"},
        {"component": "orchestration", "input_tokens": 20, "output_tokens": 7,
         "ts": "2026-06-12T00:00:00Z"},
        {"component": "critic", "input_tokens": 8, "output_tokens": 4,
         "ts": "2026-06-12T00:00:00Z"},
        {"component": "kernel", "input_tokens": 100, "output_tokens": 50,
         "ts": "2026-06-12T00:00:00Z"},
    ]
    (trace / "llm_calls.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8",
    )
    out = collectors.collect_decision_trace(tmp_path, {}, [])
    # specialist + scorer attach to t1's decision.
    t1 = out["decision_trace"][0]
    assert t1["tokens"]["calls"] == 2
    assert set(t1["tokens"]["by_component"]) == {"specialist", "proposal_scorer"}
    # orchestration + critic -> overhead; keyless kernel -> unattributed.
    assert out["overhead_tokens"]["calls"] == 2
    assert out["overhead_tokens"]["total_in"] == 28      # 20 + 8
    assert out["unattributed_tokens"]["calls"] == 1
    assert out["unattributed_tokens"]["total_in"] == 100


def test_decision_trace_attributes_critic_review_to_decision(tmp_path: Path) -> None:
    # A critic call that reviewed proposal msg "m1" should attribute to the
    # decision whose task the proposal became (via proposal_task_map).
    trace = tmp_path / "reports" / "trace"
    trace.mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "optimization_journal.json").write_text(
        json.dumps({"entries": [
            {"ts": "2026-06-12T00:00:02Z", "kind": "backend", "change": "x",
             "outcome": "KEEP", "phase": "EXPLORE", "gain_pct": 3.0,
             "task_id": "task-A", "provenance": "specialist:perf"},
        ]}),
        encoding="utf-8",
    )
    (trace / "proposal_task_map.jsonl").write_text(
        json.dumps({"proposal_msg_id": "m1", "task_id": "task-A"}) + "\n",
        encoding="utf-8",
    )
    rows = [
        {"component": "critic", "reviewed_msg_ids": ["m1"], "input_tokens": 40,
         "output_tokens": 9, "ts": "2026-06-12T00:00:00Z"},
        # A critic call reviewing two distinct materialized proposals stays in
        # overhead (ambiguous attribution).
        {"component": "critic", "reviewed_msg_ids": ["m1", "m2"],
         "input_tokens": 5, "output_tokens": 1, "ts": "2026-06-12T00:00:00Z"},
    ]
    (trace / "proposal_task_map.jsonl").write_text(
        "\n".join(json.dumps(r) for r in [
            {"proposal_msg_id": "m1", "task_id": "task-A"},
            {"proposal_msg_id": "m2", "task_id": "task-B"},
        ]) + "\n",
        encoding="utf-8",
    )
    (trace / "llm_calls.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8",
    )
    out = collectors.collect_decision_trace(tmp_path, {}, [])
    dec = out["decision_trace"][0]
    # Single-target critic review folded into task-A's decision.
    assert dec["tokens"]["calls"] == 1
    assert "critic" in dec["tokens"]["by_component"]
    assert dec["tokens"]["by_component"]["critic"]["total_in"] == 40
    # The ambiguous (m1+m2) critic call stays in overhead, not unattributed.
    assert out["overhead_tokens"]["calls"] == 1
    assert out["overhead_tokens"]["total_in"] == 5


def test_shape_ledger_entry_tags_operation_kind_and_proposer() -> None:
    from inference_optimizer.breakdown.collectors import _shape_ledger

    ledger = {
        "schema_version": 1,
        "tested": {
            "fpA": {
                "name": "v01",
                "fingerprint": "fpA",
                "extra_server_args": "--attention-backend AITER",
                "gain_pct": 3.0,
                "provenance": "specialist:serving",
                "scope": "domain",
            },
        },
        "accepted": [],
        "rejected": [],
    }
    shaped = _shape_ledger(ledger)
    entry = shaped["top_by_gain"][0]
    assert entry["operation_kind"] == "backend"
    assert entry["provenance"] == "specialist:serving"
    assert entry["proposer"] == "specialist:serving"
    assert entry["scope"] == "domain"
