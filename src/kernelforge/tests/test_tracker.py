"""Tests for experiment tracker."""

import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from kernelforge.tracker import ExperimentTracker, Experiment

from kernelforge.conftest import SRC_ROOT


def _log_iterations_worker(
    experiments_dir,
    experiment_id,
    worker_index,
    count,
    ready_queue,
    start_event,
):
    tracker = ExperimentTracker(experiments_dir)
    save = tracker._save

    def delayed_save(experiment):
        time.sleep(0.01)
        save(experiment)

    tracker._save = delayed_save
    ready_queue.put(True)
    start_event.wait()
    for index in range(count):
        tracker.log_iteration(
            experiment_id,
            wall_ms=float(worker_index * count + index),
            notes=f"worker-{worker_index}-iteration-{index}",
        )


def _create_segment_worker(
    experiments_dir,
    campaign_id,
    segment_index,
    parent_experiment_id,
    ready_queue,
    start_event,
    result_queue,
):
    tracker = ExperimentTracker(experiments_dir)
    save = tracker._save

    def delayed_save(experiment):
        if experiment.segment_index == segment_index:
            time.sleep(0.05)
        save(experiment)

    tracker._save = delayed_save
    ready_queue.put(True)
    start_event.wait()
    segment = tracker.create_segment(
        campaign_id=campaign_id,
        segment_index=segment_index,
        parent_experiment_id=parent_experiment_id,
        task_id="gemm",
    )
    result_queue.put(segment.experiment_id)


def test_create_experiment():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(tmpdir)
        exp = tracker.create(
            task_id="test_gemm",
            backend="triton",
            kernel_backend="triton",
            description="Test GEMM experiment",
            target_wall_ms=1.0,
            baseline_wall_ms=2.0,
        )
        assert exp.experiment_id
        assert exp.task_id == "test_gemm"
        assert exp.backend == "triton"
        assert (Path(tmpdir) / f"{exp.experiment_id}.json").exists()


def test_create_experiment_with_caller_owned_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(tmpdir)

        exp = tracker.create(task_id="test", experiment_id="hyperloom")

        assert exp.experiment_id == "hyperloom"
        assert (Path(tmpdir) / "hyperloom.json").exists()


def test_set_checkpoint_persists_best_commit():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(tmpdir)
        exp = tracker.create(task_id="test", experiment_id="hyperloom")
        checkpoint = {
            "state": "best_committed",
            "best_commit": "abc123",
            "best_ms": 0.9,
        }

        tracker.set_checkpoint(exp.experiment_id, checkpoint)

        assert tracker.get(exp.experiment_id).checkpoint == checkpoint


def test_create_experiment_rejects_path_like_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(tmpdir)

        with pytest.raises(ValueError):
            tracker.create(task_id="test", experiment_id="../escape")


def test_log_iteration():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(tmpdir)
        exp = tracker.create(task_id="test", backend="ck")

        it = tracker.log_iteration(
            exp.experiment_id,
            config={"BLOCK_M": 128, "BLOCK_N": 128},
            snr_db=35.0,
            wall_ms=1.5,
            wait_mfma_ratio=3.2,
            vgpr=240,
            decision="Try BLOCK_K=128",
        )
        assert it.iteration_id == 1
        assert it.snr_db == 35.0

        # Log another iteration
        it2 = tracker.log_iteration(
            exp.experiment_id,
            config={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128},
            snr_db=34.5,
            wall_ms=1.2,
            decision="Improved. Try num_stages=3",
        )
        assert it2.iteration_id == 2

        # Verify persistence
        loaded = tracker.get(exp.experiment_id)
        assert len(loaded.iterations) == 2
        assert loaded.iterations[1].wall_ms == 1.2


def test_best_iteration():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(tmpdir)
        exp = tracker.create(task_id="test", backend="ck", target_wall_ms=1.0)

        # Iteration 1: passes SNR, mediocre perf
        tracker.log_iteration(
            exp.experiment_id,
            snr_db=35.0,
            wall_ms=2.0,
            mean_case_speedup=1.0,
        )
        # Iteration 2: fails SNR — should be excluded from best
        tracker.log_iteration(
            exp.experiment_id,
            snr_db=15.0,
            wall_ms=0.5,
            mean_case_speedup=4.0,
        )
        # Iteration 3: passes SNR, best perf
        tracker.log_iteration(
            exp.experiment_id,
            snr_db=32.0,
            wall_ms=1.1,
            mean_case_speedup=2.0,
        )

        best = tracker.get_best(exp.experiment_id)
        assert best is not None
        assert best.iteration_id == 3
        assert best.wall_ms == 1.1


def test_reverted_iteration_cannot_become_best_or_meet_gate():
    """Exclude a failed confirmation result from best-performance reporting."""
    exp = Experiment(experiment_id="test", target_wall_ms=1.0)
    exp.add_iteration(
        snr_db=35.0,
        wall_ms=1.1,
        mean_case_speedup=1.0,
        decision="KEEP",
    )
    exp.add_iteration(
        snr_db=35.0,
        wall_ms=0.9,
        mean_case_speedup=2.0,
        decision="REVERT",
    )
    assert exp.best_iteration().wall_ms == 1.1
    assert exp.is_gate_met() is False


def test_plateau_detection():
    exp = Experiment(experiment_id="test")
    # Add 3 authoritative scores with <2% variance.
    exp.add_iteration(snr_db=30.0, wall_ms=1.00, mean_case_speedup=1.20)
    exp.add_iteration(snr_db=31.0, wall_ms=0.99, mean_case_speedup=1.21)
    exp.add_iteration(snr_db=30.5, wall_ms=0.995, mean_case_speedup=1.205)
    assert exp.is_plateaued(n=3, threshold=0.02)

    # Add iteration with significant improvement → no longer plateaued
    exp.add_iteration(snr_db=30.0, wall_ms=0.80, mean_case_speedup=1.40)
    assert not exp.is_plateaued(n=3, threshold=0.02)


def test_gate_check():
    exp = Experiment(experiment_id="test", target_wall_ms=1.0)
    assert not exp.is_gate_met()

    exp.add_iteration(
        snr_db=35.0,
        wall_ms=1.5,
        mean_case_speedup=1.0,
    )  # above target
    assert not exp.is_gate_met()

    exp.add_iteration(
        snr_db=35.0,
        wall_ms=0.9,
        mean_case_speedup=2.0,
    )  # below target
    assert exp.is_gate_met()
    assert exp.is_gate_met(exp.scoring_view())


def test_mean_case_speedup():
    exp = Experiment(experiment_id="test", baseline_wall_ms=2.0)
    exp.add_iteration(snr_db=30.0, wall_ms=1.0, mean_case_speedup=2.0)
    assert exp.best_mean_case_speedup() == 2.0


def test_legacy_history_is_display_only():
    exp = Experiment(
        experiment_id="legacy",
        baseline_wall_ms=2.0,
        target_wall_ms=1.0,
    )
    legacy = exp.add_iteration(snr_db=35.0, wall_ms=1.0)

    assert exp.best_iteration() is None
    assert exp.legacy_best_iteration() is legacy
    assert exp.display_best_iteration() is legacy
    assert exp.display_speedup() == (2.0, "legacy raw ratio")
    assert exp.is_gate_met() is False


def test_display_prefers_authoritative_score_over_faster_legacy_raw_wall():
    exp = Experiment(experiment_id="mixed", baseline_wall_ms=2.0)
    exp.add_iteration(snr_db=35.0, wall_ms=0.5)
    scored = exp.add_iteration(
        snr_db=35.0,
        wall_ms=1.5,
        mean_case_speedup=1.4,
    )

    assert exp.display_best_iteration() is scored
    assert exp.display_speedup() == (1.4, "mean case speedup")


def test_plateau_uses_authoritative_score_instead_of_raw_wall():
    exp = Experiment(experiment_id="scored")
    for wall_ms, speedup in (
        (3.0, 1.200),
        (1.0, 1.205),
        (2.0, 1.210),
    ):
        exp.add_iteration(
            snr_db=35.0,
            wall_ms=wall_ms,
            mean_case_speedup=speedup,
        )

    assert exp.is_plateaued()


def test_plateau_does_not_fall_back_to_raw_wall_with_too_few_scores():
    exp = Experiment(experiment_id="partially-scored")
    exp.add_iteration(snr_db=35.0, wall_ms=1.000)
    exp.add_iteration(
        snr_db=35.0,
        wall_ms=1.001,
        mean_case_speedup=1.20,
    )
    exp.add_iteration(
        snr_db=35.0,
        wall_ms=1.002,
        mean_case_speedup=1.21,
    )

    assert not exp.is_plateaued()


def test_summary_table():
    exp = Experiment(
        experiment_id="test",
        target_wall_ms=1.0,
        baseline_wall_ms=2.0,
    )
    exp.add_iteration(
        snr_db=35.0,
        wall_ms=1.5,
        mean_case_speedup=1.0,
        decision="initial",
    )
    exp.add_iteration(
        snr_db=33.0,
        wall_ms=0.9,
        mean_case_speedup=2.0,
        decision="optimized",
    )

    table = exp.summary_table()
    assert "Iter" in table
    assert "0.900" in table
    assert "Gate (1.0 ms): MET" in table


def test_list_experiments():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(tmpdir)
        tracker.create(task_id="exp1", backend="ck")
        tracker.create(task_id="exp2", backend="triton")

        exps = tracker.list_experiments()
        assert len(exps) == 2


def test_legacy_experiment_json_remains_readable(tmp_path):
    legacy = {
        "experiment_id": "legacy-1",
        "task_id": "legacy-task",
        "created_at": "2026-07-20T10:00:00",
        "iterations": [],
    }
    (tmp_path / "legacy-1.json").write_text(json.dumps(legacy))

    exp = ExperimentTracker(tmp_path).get("legacy-1")

    assert exp.task_id == "legacy-task"
    assert exp.campaign_id == ""
    assert exp.segment_index == 0
    assert exp.parent_experiment_id == ""
    assert exp.status == ""
    assert exp.ended_at == ""


def test_create_segment_persists_directly_in_experiments_root(tmp_path):
    tracker = ExperimentTracker(tmp_path)

    segment = tracker.create_segment(
        campaign_id="campaign-1",
        segment_index=1,
        task_id="gemm",
        backend="triton",
    )

    assert segment.campaign_id == "campaign-1"
    assert segment.segment_index == 1
    assert segment.parent_experiment_id == ""
    assert segment.status == "running"
    assert segment.started_at
    assert (tmp_path / f"{segment.experiment_id}.json").is_file()
    assert list(tmp_path.glob("*.json")) == [tmp_path / f"{segment.experiment_id}.json"]


def test_create_child_segment_links_parent_and_interrupts_abandoned_run(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    parent = tracker.create_segment(
        campaign_id="campaign-1",
        segment_index=1,
        task_id="gemm",
    )

    child = tracker.create_segment(
        campaign_id="campaign-1",
        segment_index=2,
        parent_experiment_id=parent.experiment_id,
        task_id="gemm",
    )

    reloaded_parent = tracker.get(parent.experiment_id)
    assert reloaded_parent.status == "interrupted"
    assert reloaded_parent.ended_at
    assert child.parent_experiment_id == parent.experiment_id
    assert child.segment_index == 2
    assert child.status == "running"


def test_create_segment_crash_retry_reuses_persisted_child(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    parent = tracker.create_segment(
        campaign_id="campaign-1",
        segment_index=1,
        task_id="gemm",
    )
    child = tracker.create_segment(
        campaign_id="campaign-1",
        segment_index=2,
        parent_experiment_id=parent.experiment_id,
        task_id="gemm",
    )
    parent_ended_at = tracker.get(parent.experiment_id).ended_at
    (tmp_path / "unrelated.json").write_text("{not valid json")

    retried = tracker.create_segment(
        campaign_id="campaign-1",
        segment_index=2,
        parent_experiment_id=parent.experiment_id,
        task_id="gemm",
    )

    assert retried.experiment_id == child.experiment_id
    assert tracker.get(parent.experiment_id).ended_at == parent_ended_at
    matching_segments = [
        experiment
        for experiment in tracker.list_experiments()
        if experiment.campaign_id == "campaign-1" and experiment.segment_index == 2
    ]
    assert [experiment.experiment_id for experiment in matching_segments] == [child.experiment_id]


def test_create_segment_retry_rejects_parent_mismatch(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    parent = tracker.create_segment(
        campaign_id="campaign-1",
        segment_index=1,
    )
    child = tracker.create_segment(
        campaign_id="campaign-1",
        segment_index=2,
        parent_experiment_id=parent.experiment_id,
    )

    with pytest.raises(ValueError, match="parent_experiment_id mismatch"):
        tracker.create_segment(
            campaign_id="campaign-1",
            segment_index=2,
            parent_experiment_id="different-parent",
        )

    matching_segments = [
        experiment
        for experiment in tracker.list_experiments()
        if experiment.campaign_id == "campaign-1" and experiment.segment_index == 2
    ]
    assert [experiment.experiment_id for experiment in matching_segments] == [child.experiment_id]


def test_create_child_segment_rejects_broken_lineage(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    parent = tracker.create_segment(
        campaign_id="campaign-1",
        segment_index=1,
    )

    with pytest.raises(ValueError, match="campaign mismatch"):
        tracker.create_segment(
            campaign_id="campaign-2",
            segment_index=2,
            parent_experiment_id=parent.experiment_id,
        )

    with pytest.raises(ValueError, match="segment index"):
        tracker.create_segment(
            campaign_id="campaign-1",
            segment_index=3,
            parent_experiment_id=parent.experiment_id,
        )


def test_mark_complete_is_idempotent_and_fires_callbacks_once(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    completed = []
    tracker.on_complete(lambda exp: completed.append(exp.experiment_id))
    exp = tracker.create(task_id="gemm")

    first = tracker.mark_complete(exp.experiment_id)
    second = tracker.mark_complete(exp.experiment_id)

    assert first.status == "completed"
    assert first.ended_at
    assert second.ended_at == first.ended_at
    assert completed == [exp.experiment_id]


def test_mark_interrupted_is_idempotent_and_does_not_overwrite_completion(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    exp = tracker.create(task_id="gemm")

    first = tracker.mark_interrupted(exp.experiment_id)
    second = tracker.mark_interrupted(exp.experiment_id)
    assert first.status == "interrupted"
    assert second.ended_at == first.ended_at

    completed = tracker.create(task_id="completed")
    tracker.mark_complete(completed.experiment_id)
    tracker.mark_interrupted(completed.experiment_id)
    assert tracker.get(completed.experiment_id).status == "completed"


def test_experiment_save_is_atomic(tmp_path, monkeypatch):
    tracker = ExperimentTracker(tmp_path)
    exp = tracker.create(task_id="gemm")

    def fail_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        tracker.set_baseline(exp.experiment_id, 1.25)

    assert tracker.get(exp.experiment_id).baseline_wall_ms is None
    assert list(tmp_path.glob(".experiment.*.tmp")) == []


def test_concurrent_processes_preserve_all_logged_iterations(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    exp = tracker.create(task_id="gemm")
    process_count = 4
    iterations_per_process = 8
    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    start_event = context.Event()
    processes = [
        context.Process(
            target=_log_iterations_worker,
            args=(
                tmp_path,
                exp.experiment_id,
                worker_index,
                iterations_per_process,
                ready_queue,
                start_event,
            ),
        )
        for worker_index in range(process_count)
    ]

    for process in processes:
        process.start()
    for _ in processes:
        ready_queue.get(timeout=10)
    start_event.set()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    loaded = tracker.get(exp.experiment_id)
    expected_count = process_count * iterations_per_process
    assert len(loaded.iterations) == expected_count
    assert [iteration.iteration_id for iteration in loaded.iterations] == list(range(1, expected_count + 1))
    assert len({iteration.notes for iteration in loaded.iterations}) == expected_count


def test_concurrent_processes_create_one_campaign_segment(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    parent = tracker.create_segment(
        campaign_id="campaign-1",
        segment_index=1,
        task_id="gemm",
    )
    process_count = 4
    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    start_event = context.Event()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_create_segment_worker,
            args=(
                tmp_path,
                "campaign-1",
                2,
                parent.experiment_id,
                ready_queue,
                start_event,
                result_queue,
            ),
        )
        for _ in range(process_count)
    ]

    for process in processes:
        process.start()
    for _ in processes:
        ready_queue.get(timeout=10)
    start_event.set()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    experiment_ids = [result_queue.get(timeout=5) for _ in processes]
    assert len(set(experiment_ids)) == 1
    matching_segments = [
        experiment
        for experiment in tracker.list_experiments()
        if experiment.campaign_id == "campaign-1" and experiment.segment_index == 2
    ]
    assert [experiment.experiment_id for experiment in matching_segments] == [experiment_ids[0]]


def test_concurrent_subprocess_updates_preserve_fields_and_single_transition(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    exp = tracker.create_segment(
        campaign_id="campaign-1",
        segment_index=1,
        task_id="gemm",
    )
    gate = tmp_path / "start"
    callback_log = tmp_path / "callbacks.log"
    script = """
import sys
import time
from pathlib import Path

from kernelforge.tracker import ExperimentTracker

experiments_dir, experiment_id, operation, ready_path, gate_path, callback_path = sys.argv[1:]
tracker = ExperimentTracker(experiments_dir)
save = tracker._save

def delayed_save(experiment):
    time.sleep(0.1)
    save(experiment)

tracker._save = delayed_save

def record_completion(_experiment):
    with open(callback_path, "a") as callback_file:
        callback_file.write("completed\\n")

tracker.on_complete(record_completion)
Path(ready_path).touch()
gate = Path(gate_path)
while not gate.exists():
    time.sleep(0.001)

if operation == "usage":
    tracker.set_llm_usage(experiment_id, {"input_tokens": 17})
elif operation == "kb":
    tracker.set_kb_experience(experiment_id, {"read": "hit"})
elif operation == "baseline":
    tracker.set_baseline(experiment_id, 1.25)
elif operation == "complete":
    tracker.mark_complete(experiment_id)
elif operation == "segment":
    tracker.create_segment(
        campaign_id="campaign-1",
        segment_index=2,
        parent_experiment_id=experiment_id,
    )
else:
    raise AssertionError(f"unknown operation: {operation}")
"""
    operations = ["usage", "kb", "baseline", "complete", "complete", "complete", "segment"]
    env = os.environ.copy()
    src_dir = SRC_ROOT
    env["PYTHONPATH"] = os.pathsep.join(path for path in (str(src_dir), env.get("PYTHONPATH", "")) if path)
    ready_paths = [tmp_path / f"ready-{index}" for index in range(len(operations))]
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(tmp_path),
                exp.experiment_id,
                operation,
                str(ready_paths[index]),
                str(gate),
                str(callback_log),
            ],
            cwd=tmp_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index, operation in enumerate(operations)
    ]

    deadline = time.monotonic() + 10
    while not all(path.exists() for path in ready_paths):
        exited = [process for process in processes if process.poll() is not None]
        if exited:
            stdout, stderr = exited[0].communicate()
            pytest.fail(f"subprocess exited before synchronization:\n{stdout}\n{stderr}")
        if time.monotonic() >= deadline:
            pytest.fail("subprocesses did not reach the synchronization point")
        time.sleep(0.01)
    gate.touch()

    for process in processes:
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, f"{stdout}\n{stderr}"

    loaded = tracker.get(exp.experiment_id)
    assert loaded.llm_usage == {"input_tokens": 17}
    assert loaded.kb_experience == {"read": "hit"}
    assert loaded.baseline_wall_ms == 1.25
    assert loaded.status in {"completed", "interrupted"}
    callbacks = callback_log.read_text().splitlines() if callback_log.exists() else []
    assert len(callbacks) <= 1


def test_create_segment_rejects_an_unusable_campaign_or_index(tmp_path):
    tracker = ExperimentTracker(tmp_path)

    with pytest.raises(ValueError, match="campaign_id is required"):
        tracker.create_segment(campaign_id="   ", segment_index=1)

    with pytest.raises(ValueError, match="segment index must be at least 1"):
        tracker.create_segment(campaign_id="campaign-1", segment_index=0)

    with pytest.raises(ValueError, match="must be 1 when no parent"):
        tracker.create_segment(campaign_id="campaign-1", segment_index=2)

    # A rejected request must leave no half-created segment behind.
    assert list(tmp_path.glob("*.json")) == []


def test_segment_lookup_ignores_unreadable_and_non_experiment_json(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    (tmp_path / "corrupt.json").write_text("{not valid json")
    (tmp_path / "not_an_experiment.json").write_text(json.dumps([{"campaign_id": "campaign-1", "segment_index": 1}]))

    segment = tracker.create_segment(campaign_id="campaign-1", segment_index=1)

    # A stray file that merely looks like a match must not be adopted as one.
    assert segment.campaign_id == "campaign-1"
    assert segment.segment_index == 1
    assert (tmp_path / f"{segment.experiment_id}.json").is_file()


def test_set_checkpoint_ignores_an_empty_payload(tmp_path):
    tracker = ExperimentTracker(tmp_path)
    exp = tracker.create(task_id="test", experiment_id="hyperloom")
    checkpoint = {"state": "best_committed", "best_commit": "abc123"}
    tracker.set_checkpoint(exp.experiment_id, checkpoint)

    tracker.set_checkpoint(exp.experiment_id, {})

    # An empty payload must not erase the recovery anchor an owner reads back.
    assert tracker.get(exp.experiment_id).checkpoint == checkpoint
