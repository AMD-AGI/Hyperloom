"""Coverage completion tests for the experiment tracker and schema."""

from __future__ import annotations

import json

from kernelforge.tracker import Experiment, ExperimentTracker
from kernelforge.tracker.schema import Iteration


# ─── ExperimentTracker ───


def test_on_complete_callback_fires(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    exp = tracker.create(task_id="t", backend="ck")
    seen = []
    tracker.on_complete(lambda e: seen.append(e.experiment_id))
    # A raising callback must be swallowed (contextlib.suppress).
    tracker.on_complete(lambda e: (_ for _ in ()).throw(RuntimeError("boom")))
    returned = tracker.mark_complete(exp.experiment_id)
    assert returned.experiment_id == exp.experiment_id
    assert seen == [exp.experiment_id]


def test_set_llm_usage_and_kb_experience(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    exp = tracker.create(task_id="t", backend="ck")
    # Empty payloads are no-ops.
    tracker.set_llm_usage(exp.experiment_id, {})
    tracker.set_kb_experience(exp.experiment_id, {})
    assert tracker.get(exp.experiment_id).llm_usage == {}

    tracker.set_llm_usage(exp.experiment_id, {"input_tokens": 5})
    tracker.set_kb_experience(exp.experiment_id, {"selected": "sol_a"})
    loaded = tracker.get(exp.experiment_id)
    assert loaded.llm_usage == {"input_tokens": 5}
    assert loaded.kb_experience == {"selected": "sol_a"}


def test_set_baseline_precedence(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    exp = tracker.create(task_id="t", backend="ck")
    tracker.set_baseline(exp.experiment_id, 2.0)
    assert tracker.get(exp.experiment_id).baseline_wall_ms == 2.0
    # Existing baseline is not overwritten.
    tracker.set_baseline(exp.experiment_id, 9.0)
    assert tracker.get(exp.experiment_id).baseline_wall_ms == 2.0


def test_list_experiments_skips_malformed(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    tracker.create(task_id="good", backend="ck")
    (tmp_path / "broken.json").write_text("{ not valid json")
    exps = tracker.list_experiments()
    assert len(exps) == 1


def test_list_experiments_skips_json_that_is_not_an_experiment_record(tmp_path):
    """Valid JSON is not enough: a listed row must carry identity and a birth time."""
    tracker = ExperimentTracker(tmp_path)
    good = tracker.create(task_id="good", backend="ck")
    (tmp_path / "sidecar.json").write_text(json.dumps({"note": "not an experiment"}))
    (tmp_path / "headless.json").write_text(json.dumps({"created_at": "2026-01-01"}))
    (tmp_path / "array.json").write_text(json.dumps([{"experiment_id": "x"}]))

    listed = tracker.list_experiments()

    assert [exp.experiment_id for exp in listed] == [good.experiment_id]


def test_get_missing_raises(tmp_path):
    import pytest

    tracker = ExperimentTracker(tmp_path)
    with pytest.raises(FileNotFoundError):
        tracker.get("does_not_exist")


def test_summary_delegates(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    exp = tracker.create(task_id="t", backend="ck", target_wall_ms=1.0)
    tracker.log_iteration(exp.experiment_id, snr_db=35.0, wall_ms=0.8)
    summary = tracker.summary(exp.experiment_id)
    assert "Iter" in summary


def test_iteration_roundtrip_dict():
    it = Iteration(iteration_id=1, config={"BLOCK_M": 128}, snr_db=35.0, wall_ms=1.2)
    d = it.to_dict()
    assert "iteration_id" in d
    # Falsy/empty fields are dropped for compactness.
    assert "notes" not in d
    restored = Iteration.from_dict({**d, "unknown": 1})
    assert restored.snr_db == 35.0


def test_is_gate_met_no_target():
    assert Experiment(experiment_id="e").is_gate_met() is False


def test_effective_baseline_none():
    exp = Experiment(experiment_id="e")
    assert exp.effective_baseline_ms() is None
    assert exp.best_mean_case_speedup() is None


def test_consecutive_reverts():
    exp = Experiment(experiment_id="e")
    exp.add_iteration(snr_db=35.0, wall_ms=1.0, decision="KEEP")
    exp.add_iteration(snr_db=35.0, wall_ms=1.1, decision="REVERT")
    exp.add_iteration(snr_db=35.0, wall_ms=1.2, decision="REVERT")
    assert exp.consecutive_reverts() == 2
    # A KEEP breaks the streak.
    exp.add_iteration(snr_db=35.0, wall_ms=0.9, decision="KEEP")
    assert exp.consecutive_reverts() == 0


def test_summary_table_legacy_data_is_unscored_and_not_plateaued():
    exp = Experiment(experiment_id="e", target_wall_ms=0.5)
    for wall in (1.00, 0.99, 0.995):
        exp.add_iteration(snr_db=35.0, wall_ms=wall)
    table = exp.summary_table()
    assert "Gate (" not in table
    assert "PLATEAUED" not in table


def test_uses_authoritative_scoring_requires_a_per_case_score():
    """Raw wall time alone is display-only history, never an authoritative score."""
    scored = Experiment(experiment_id="scored")
    scored.add_iteration(snr_db=35.0, wall_ms=1.0, mean_case_speedup=1.2)
    legacy = Experiment(experiment_id="legacy")
    legacy.add_iteration(snr_db=35.0, wall_ms=1.0)

    assert scored.uses_authoritative_scoring() is True
    assert legacy.uses_authoritative_scoring() is False
    # The legacy raw best is offered only while scoring is unauthoritative.
    assert scored.legacy_best_iteration() is None
    assert legacy.legacy_best_iteration() is not None


def test_summary_table_announces_plateau_and_reverted_changes():
    exp = Experiment(experiment_id="e", baseline_wall_ms=2.0)
    for speedup, wall in ((1.100, 1.00), (1.110, 0.99), (1.105, 0.995)):
        exp.add_iteration(snr_db=35.0, wall_ms=wall, mean_case_speedup=speedup)
    exp.changes_reverted = ["unrolled epilogue", "widened lds tile"]

    table = exp.summary_table()

    assert exp.is_plateaued() is True
    assert "PLATEAUED" in table
    assert "Reverted: unrolled epilogue, widened lds tile" in table


def test_experiment_to_from_dict_roundtrip():
    exp = Experiment(experiment_id="e", backend="ck", baseline_wall_ms=2.0)
    exp.add_iteration(snr_db=35.0, wall_ms=1.0)
    d = exp.to_dict()
    json.dumps(d)
    restored = Experiment.from_dict({**d, "unknown_field": 1})
    assert restored.experiment_id == "e"
    assert len(restored.iterations) == 1
    assert restored.iterations[0].wall_ms == 1.0
