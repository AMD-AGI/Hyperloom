# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a ``decision_trace.jsonl`` row records for each decision source: a KEEP/REVERT journal entry and a
dynamic_action dispatch."""

from __future__ import annotations

import json

from hyperloom.inference_optimizer.session.optimization_journal import Journal, JournalEntry
from hyperloom.inference_optimizer.session.session_paths import decision_trace_path
from hyperloom.inference_optimizer.trace.decision_trace import write_decision_trace


def _write_jsonl(path, *rows: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _trace_rows(session_dir) -> list[dict]:
    return [json.loads(line) for line in decision_trace_path(session_dir).read_text(encoding="utf-8").splitlines()]


def test_a_keep_revert_row_carries_its_proposer_attribution_and_optional_measurements(tmp_path):
    entry = JournalEntry(
        phase="EXPLORE",
        iter=2,
        kind="config",
        change="tp=4",
        outcome="KEEP",
        gain_pct=3.5,
        predicted_gain_pct=5.0,
        task_id="t1",
        variant_name="v-tp4",
        ts="2026-06-09T15:00:00Z",
        provenance="default_grid",
        scope="tp",
        fingerprint="fp-1",
        metrics={"throughput": 120.0},
    )
    Journal.load_or_create(tmp_path, Journal(session_id="SID", model="m", hardware="h")).append_entry(entry)
    scores = {"models": {"rater-a": {"v-tp4": {"score": 0.8, "reason": "fits"}}}}
    warnings: list[str] = []

    write_decision_trace(tmp_path, {"specialist_rounds": [{"ensemble_scores": scores}]}, warnings)

    (row,) = _trace_rows(tmp_path)
    assert row["phase"] == "EXPLORE"
    assert row["decision"] == {
        "component": "grid",
        "provenance": "default_grid",
        "scope": "tp",
        "fingerprint": "fp-1",
        "change": "tp=4",
        "outcome": "KEEP",
        "gain_pct": 3.5,
        "task_id": "t1",
        "operation_kind": "config",
        "predicted_gain_pct": 5.0,
        "kind": "config",
        "metrics": {"throughput": 120.0},
        "variant_name": "v-tp4",
        "proposal_scores": [{"rater": "rater-a", "score": 0.8, "reason": "fits"}],
    }
    assert warnings == []


def test_a_dynamic_action_dispatch_is_a_row_owning_the_calls_keyed_by_its_dyn_id(tmp_path):
    dispatch = {
        "event": "integrate",
        "ts": "2026-06-09T16:00:00Z",
        "tick": 4,
        "verdict": "keep",
        "integrate_status": "integrated",
        "delta_pct": 2.5,
    }
    _write_jsonl(
        tmp_path / "agents" / "orchestration" / "dynamic_actions" / "dyn-7" / "dispatch_history.jsonl", dispatch
    )
    call = {"dyn_id": "dyn-7", "component": "orchestration", "input_tokens": 10, "output_tokens": 2}
    _write_jsonl(tmp_path / "reports" / "trace" / "llm_calls.jsonl", call)

    write_decision_trace(tmp_path, {}, [])

    (row,) = _trace_rows(tmp_path)
    assert row["tick"] == 4
    assert row["decision"] == {
        "component": "dynamic_action",
        "operation_kind": "dynamic_action",
        "event": "integrate",
        "dyn_id": "dyn-7",
        "verdict": "keep",
        "outcome": "integrated",
        "gain_pct": 2.5,
    }
    assert (row["tokens"]["calls"], row["tokens"]["total_in"], row["tokens"]["total_out"]) == (1, 10, 2)
