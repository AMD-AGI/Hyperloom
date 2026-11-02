"""Coverage completion tests for the experiment tracker and schema."""

from __future__ import annotations

import json

from kernelforge.workflows.forge_loop.campaign.schema import Experiment, ExperimentSubject
from kernelforge.workflows.forge_loop.campaign.experiment import ExperimentTracker
from kernelforge.workflows.forge_loop.campaign.schema import Iteration


# ─── ExperimentTracker ───


def test_on_complete_callback_fires(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    exp = tracker.create(ExperimentSubject(task_id="t", backend="ck"))
    seen = []
    tracker.on_complete(lambda e: seen.append(e.experiment_id))
    # A raising callback must be swallowed (contextlib.suppress).
    tracker.on_complete(lambda e: (_ for _ in ()).throw(RuntimeError("boom")))
    returned = tracker.mark_complete(exp.experiment_id)
    assert returned.experiment_id == exp.experiment_id
    assert seen == [exp.experiment_id]


def test_set_llm_usage_and_kb_experience(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    exp = tracker.create(ExperimentSubject(task_id="t", backend="ck"))
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
    exp = tracker.create(ExperimentSubject(task_id="t", backend="ck"))
    tracker.set_baseline(exp.experiment_id, 2.0)
    assert tracker.get(exp.experiment_id).baseline_wall_ms == 2.0
    # Existing baseline is not overwritten.
    tracker.set_baseline(exp.experiment_id, 9.0)
    assert tracker.get(exp.experiment_id).baseline_wall_ms == 2.0


def test_get_missing_raises(tmp_path):
    import pytest

    tracker = ExperimentTracker(tmp_path)
    with pytest.raises(FileNotFoundError):
        tracker.get("does_not_exist")


def test_iteration_roundtrip_dict():
    it = Iteration(iteration_id=1, config={"BLOCK_M": 128}, snr_db=35.0, wall_ms=1.2)
    d = it.to_dict()
    assert "iteration_id" in d
    # Falsy/empty fields are dropped for compactness.
    assert "notes" not in d
    restored = Iteration.from_dict({**d, "unknown": 1})
    assert restored.snr_db == 35.0


def test_experiment_to_from_dict_roundtrip():
    exp = Experiment(experiment_id="e", backend="ck", baseline_wall_ms=2.0)
    exp.add_iteration(snr_db=35.0, wall_ms=1.0)
    d = exp.to_dict()
    json.dumps(d)
    restored = Experiment.from_dict({**d, "unknown_field": 1})
    assert restored.experiment_id == "e"
    assert len(restored.iterations) == 1
    assert restored.iterations[0].wall_ms == 1.0
