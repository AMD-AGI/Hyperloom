"""Coverage completion tests for the postmortem and auto-evolution pipeline."""

from __future__ import annotations

import json


from kernelforge.learning.auto_evolve import AutoEvolver
from kernelforge.learning.postmortem import PostMortem
from kernelforge.learning.tuning_db import TuningDatabase
from kernelforge.tracker.schema import Experiment


# ─── PostMortem ───


def test_postmortem_empty_experiment(tmp_path):
    pm = PostMortem(tmp_path)
    assert pm.analyze(Experiment(experiment_id="e")) == []


def test_postmortem_snr_failure_lesson(tmp_path):
    exp = Experiment(experiment_id="e", backend="ck")
    exp.add_iteration(snr_db=10.0, wall_ms=1.0, config={"BLOCK_M": 64})
    pm = PostMortem(tmp_path)
    lessons = pm.analyze(exp)
    assert any("Correctness failure" in l.title for l in lessons)


def test_postmortem_plateau_lesson(tmp_path):
    exp = Experiment(experiment_id="e", backend="ck")
    for wall, speedup in (
        (1.00, 1.200),
        (0.99, 1.205),
        (0.995, 1.210),
    ):
        exp.add_iteration(
            snr_db=35.0,
            wall_ms=wall,
            mean_case_speedup=speedup,
        )
    pm = PostMortem(tmp_path)
    lessons = pm.analyze(exp)
    assert any(l.category == "methodology" for l in lessons)


def test_postmortem_summary(tmp_path):
    exp = Experiment(experiment_id="e", backend="ck")
    exp.add_iteration(snr_db=35.0, wall_ms=2.0)
    exp.add_iteration(snr_db=33.0, wall_ms=2.5)  # regression
    pm = PostMortem(tmp_path)
    lessons = pm.analyze(exp)
    summary = pm.summary(lessons)
    assert "Lessons Learned" in summary
    assert "Pitfall" in summary
    assert pm.summary([]) == "No lessons extracted from this experiment."


def _evolver(tmp_path) -> AutoEvolver:
    return AutoEvolver(
        tuning_db=TuningDatabase(tmp_path / "tuning"),
        postmortem=PostMortem(tmp_path / "kb"),
    )


def test_on_experiment_complete_logs_and_discovers(tmp_path):
    evolver = _evolver(tmp_path)
    exp = Experiment(experiment_id="e", backend="ck", task_id="attention_bwd")
    exp.add_iteration(snr_db=35.0, wall_ms=2.0, config={"BLOCK_M": 64})
    exp.add_iteration(snr_db=34.0, wall_ms=1.5, config={"BLOCK_M": 128})
    results = evolver.on_experiment_complete(exp)
    assert "lessons" in results
    assert "transfer_rules" in results


def _seed_tuning_entries(db: TuningDatabase) -> None:
    """Seed the entries file directly (log() is a no-op with persistence off)."""
    db.db_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for op in ["attention_fwd", "attention_bwd", "sla_fwd"]:
        entries.append(
            dict(
                operation=op,
                backend="ck",
                gpu_target="gfx950",
                dtype="bf16",
                shape={"seq_len": 4096},
                config={"wpe": 2},
                wall_ms=8.0,
                passed_correctness=True,
            )
        )
        entries.append(
            dict(
                operation=op,
                backend="ck",
                gpu_target="gfx950",
                dtype="bf16",
                shape={"seq_len": 4096},
                config={"wpe": 3},
                wall_ms=12.0,
                passed_correctness=True,
            )
        )
    with open(db._entries_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    db._rules_path.write_text("[]")


def test_on_experiment_complete_applies_discovered_rules(tmp_path):
    evolver = _evolver(tmp_path)
    _seed_tuning_entries(evolver.tuning_db)
    exp = Experiment(experiment_id="e", backend="ck", task_id="attention_bwd")
    exp.add_iteration(snr_db=35.0, wall_ms=2.0, config={"BLOCK_M": 64})
    results = evolver.on_experiment_complete(exp)
    assert results["transfer_rules"]
