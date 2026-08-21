# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Warm replay must not promote a config that broke the model.

Warm replay promoted on throughput alone. Across the retained session pool, 45
of 241 promoted replays (19%) were promoted while the replayed config was
producing garbage — in the worst case a +23.95% "gain" on a config scoring
0.0000 on gsm8k against a 0.9014 baseline. The promoted config then becomes the
base every later measurement in that session is taken against.

Two properties this file pins down, both learned from the recorded sessions:

* The score lives in the *warmup* round. The cold-start guard evaluates once,
  in the warmup round, and decides on the measure round, so a gate that reads
  the deciding round's own workspace finds nothing: across 852 recorded replays
  the score sat in ``warmup_round`` 320 times and in ``measure_round`` never.
* A replay with no score is admitted, not rejected. Rejecting on absent
  evidence would have blocked every double-run replay, since the deciding
  round never carries a score of its own.
"""

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator

from .test_warm_replay import _make_coord, _StubTask, _warm_recipe_t1

RISKY_ENVS = {"VLLM_ROCM_USE_AITER": "1"}
SAFE_ARGS = "--max-num-seqs 2048"
BASELINE_ACC = 0.90


def _coord_with_baseline(tmp_path: Path, accuracy: float) -> Coordinator:
    coord = _make_coord(tmp_path, warm_start_recipe=_warm_recipe_t1())
    coord.shared_state.baseline_accuracy = accuracy
    coord.shared_state.warm_replay_outcome = {
        "status": "in_flight",
        "warm_recipe_tier": "exact",
        "warm_recipe_conf": 0.85,
        "expected_gain_pct": 25.0,
        "replay_task_id": "task-warm-replay-prelude",
    }
    return coord


def _risky_task() -> _StubTask:
    return _StubTask(
        params={
            "extra_server_args": "--attention-backend AITER",
            "extra_envs": dict(RISKY_ENVS),
        }
    )


def _safe_task() -> _StubTask:
    return _StubTask(params={"extra_server_args": SAFE_ARGS, "extra_envs": {}})


def _promoted(coord: Coordinator) -> bool:
    return bool(coord.shared_state.optimization_stack)


def _double_run_dirs(tmp_path: Path, warmup_score: float | None) -> dict:
    """Build a replay task directory shaped like a real cold-start double run.

    Returns the result envelope the executor hands back: it decides on the
    measure round, so ``output_dir`` and ``workspace`` both point there while
    any score sits under the sibling warmup round.
    """
    root = tmp_path / "runs" / "replay_warm_recipe" / "task-warm-replay-prelude"
    warm_bench = root / "warmup_round" / "benchmark_vllm_1"
    measure_bench = root / "measure_round" / "benchmark_vllm_2"
    warm_bench.mkdir(parents=True, exist_ok=True)
    measure_bench.mkdir(parents=True, exist_ok=True)
    if warmup_score is not None:
        (warm_bench / "results_2026-08-19T00-00-00.json").write_text(
            json.dumps(
                {"results": {"gsm8k": {"exact_match,strict-match": warmup_score}}}
            ),
            encoding="utf-8",
        )
    return {
        "status": "succeeded",
        "output_throughput": 738.0,
        "output_dir": str(root / "measure_round"),
        "workspace": str(measure_bench),
    }


class TestScoreIsFoundWhereTheDoubleRunWritesIt:
    """The deciding round carries no score of its own; the warmup round does."""

    def test_a_warmup_round_score_is_read(self, tmp_path):
        coord = _coord_with_baseline(tmp_path, BASELINE_ACC)
        result = _double_run_dirs(tmp_path, warmup_score=0.89)
        coord._promote_warm_replay(result, task=_risky_task())

        outcome = coord.shared_state.warm_replay_outcome
        assert outcome["eval_ran"] is True
        assert outcome["replay_accuracy"] == pytest.approx(0.89)
        assert _promoted(coord) is True

    def test_a_collapsed_warmup_round_score_blocks_promotion(self, tmp_path):
        coord = _coord_with_baseline(tmp_path, BASELINE_ACC)
        result = _double_run_dirs(tmp_path, warmup_score=0.0076)
        coord._promote_warm_replay(result, task=_risky_task())

        assert _promoted(coord) is False
        outcome = coord.shared_state.warm_replay_outcome
        assert outcome["status"] == "accuracy_failed"
        assert outcome["replay_accuracy"] == pytest.approx(0.0076)

    def test_no_results_file_anywhere_records_that_no_eval_ran(self, tmp_path):
        coord = _coord_with_baseline(tmp_path, BASELINE_ACC)
        result = _double_run_dirs(tmp_path, warmup_score=None)
        coord._promote_warm_replay(result, task=_risky_task())

        outcome = coord.shared_state.warm_replay_outcome
        assert outcome["eval_ran"] is False
        assert outcome["replay_accuracy"] is None
        assert "no results" in outcome["eval_error"]


class TestWarmReplayRejectsBrokenConfigs:
    def test_a_collapsed_score_blocks_promotion(self, tmp_path):
        """The case observed 45 times: big throughput win, ruined accuracy."""
        coord = _coord_with_baseline(tmp_path, BASELINE_ACC)
        coord._promote_warm_replay(
            {"status": "succeeded", "output_throughput": 738.0, "accuracy": 0.20},
            task=_risky_task(),
        )
        assert _promoted(coord) is False
        outcome = coord.shared_state.warm_replay_outcome
        assert outcome["status"] != "reproduced"
        assert "accuracy" in str(outcome.get("reason") or "").lower()

    def test_an_intact_score_still_promotes(self, tmp_path):
        """The gate must not cost a genuine win."""
        coord = _coord_with_baseline(tmp_path, BASELINE_ACC)
        coord._promote_warm_replay(
            {"status": "succeeded", "output_throughput": 738.0, "accuracy": 0.89},
            task=_risky_task(),
        )
        assert _promoted(coord) is True
        assert coord.shared_state.warm_replay_outcome["status"] == "reproduced"

    @pytest.mark.parametrize("drop", [0.0, 0.04])
    def test_a_drop_within_tolerance_is_not_a_regression(self, tmp_path, drop):
        """Healthy run-to-run spread reaches 0.037 in the observed pool, so the
        0.05 tolerance must survive it."""
        coord = _coord_with_baseline(tmp_path, BASELINE_ACC)
        coord._promote_warm_replay(
            {
                "status": "succeeded",
                "output_throughput": 738.0,
                "accuracy": BASELINE_ACC - drop,
            },
            task=_risky_task(),
        )
        assert _promoted(coord) is True


class TestEveryReplayIsJudged:
    """A KB recipe is another machine's evidence, so reproducing its throughput
    says nothing about whether it still computes correctly here. The high-risk
    trigger the other lanes use is deliberately not applied."""

    def test_a_config_with_no_high_risk_knob_is_still_judged(self, tmp_path):
        coord = _coord_with_baseline(tmp_path, BASELINE_ACC)
        coord._promote_warm_replay(
            {"status": "succeeded", "output_throughput": 738.0, "accuracy": 0.20},
            task=_safe_task(),
        )
        assert _promoted(coord) is False
        assert coord.shared_state.warm_replay_outcome["status"] == "accuracy_failed"

    def test_a_sound_config_with_no_high_risk_knob_promotes(self, tmp_path):
        coord = _coord_with_baseline(tmp_path, BASELINE_ACC)
        coord._promote_warm_replay(
            {"status": "succeeded", "output_throughput": 738.0, "accuracy": 0.89},
            task=_safe_task(),
        )
        assert _promoted(coord) is True


class TestAbsentEvidenceDoesNotBlock:
    """A failed measurement never stops the run. It is not evidence the config
    broke the model, and rejecting on it would block every double-run replay,
    since the deciding round never carries a score of its own."""

    def test_a_missing_verdict_still_promotes_and_is_marked(self, tmp_path):
        coord = _coord_with_baseline(tmp_path, BASELINE_ACC)
        coord._promote_warm_replay(
            {"status": "succeeded", "output_throughput": 738.0},
            task=_risky_task(),
        )
        assert _promoted(coord) is True
        outcome = coord.shared_state.warm_replay_outcome
        assert outcome["eval_ran"] is False
        assert outcome["replay_accuracy"] is None
        assert outcome["eval_error"]

    def test_an_unscorable_results_file_promotes_and_records_why(self, tmp_path):
        """The eval ran and produced a file with no metric this parser knows —
        a different state from an eval that never ran, and still not a reason
        to stop."""
        coord = _coord_with_baseline(tmp_path, BASELINE_ACC)
        result = _double_run_dirs(tmp_path, warmup_score=None)
        bench = Path(result["output_dir"]).parent / "warmup_round" / "benchmark_vllm_1"
        (bench / "results_2026-08-19T00-00-00.json").write_text(
            json.dumps({"results": {"gsm8k": {"unknown_metric": 1.0}}}),
            encoding="utf-8",
        )
        coord._promote_warm_replay(result, task=_risky_task())

        assert _promoted(coord) is True
        outcome = coord.shared_state.warm_replay_outcome
        assert outcome["eval_ran"] is True
        assert outcome["replay_accuracy"] is None
        assert "no recognized metric" in outcome["eval_error"]

    def test_an_undecodable_results_file_is_an_eval_that_ran(self, tmp_path):
        """A file the parser could not decode still proves the eval ran."""
        coord = _coord_with_baseline(tmp_path, BASELINE_ACC)
        result = _double_run_dirs(tmp_path, warmup_score=None)
        bench = Path(result["output_dir"]).parent / "warmup_round" / "benchmark_vllm_1"
        (bench / "results_2026-08-19T00-00-00.json").write_text(
            "{ this is not json",
            encoding="utf-8",
        )
        coord._promote_warm_replay(result, task=_risky_task())

        assert _promoted(coord) is True
        outcome = coord.shared_state.warm_replay_outcome
        assert outcome["eval_ran"] is True
        assert outcome["replay_accuracy"] is None
        assert "parse error" in outcome["eval_error"]

    def test_a_parser_crash_is_not_an_eval_that_ran(self, tmp_path, monkeypatch):
        """A parser that raised read no file, so nothing says the eval ran.

        Recording this as "ran" reads as a model that answered nothing, which
        is the one state an operator must be able to tell it apart from: the
        first is a broken config, the second is broken infrastructure.
        """
        coord = _coord_with_baseline(tmp_path, BASELINE_ACC)
        result = _double_run_dirs(tmp_path, warmup_score=None)

        from hyperloom.orchestrator.actions.executors import _accuracy_gate

        def _raise(*_args, **_kwargs):
            raise OSError("results directory vanished mid-read")

        monkeypatch.setattr(_accuracy_gate, "parse_eval_results", _raise)
        coord._promote_warm_replay(result, task=_risky_task())

        assert _promoted(coord) is True
        outcome = coord.shared_state.warm_replay_outcome
        assert outcome["eval_ran"] is False
        assert outcome["replay_accuracy"] is None
        assert "eval parse raised" in outcome["eval_error"]

    def test_a_passing_replay_records_no_eval_error(self, tmp_path):
        coord = _coord_with_baseline(tmp_path, BASELINE_ACC)
        coord._promote_warm_replay(
            {"status": "succeeded", "output_throughput": 738.0, "accuracy": 0.89},
            task=_risky_task(),
        )
        assert coord.shared_state.warm_replay_outcome["eval_error"] is None

    def test_no_baseline_missing_verdict_still_promotes(self, tmp_path):
        coord = _coord_with_baseline(tmp_path, 0.0)
        coord._promote_warm_replay(
            {"status": "succeeded", "output_throughput": 738.0},
            task=_risky_task(),
        )
        assert _promoted(coord) is True
        assert coord.shared_state.warm_replay_outcome["baseline_accuracy"] is None

    def test_no_baseline_collapsed_score_rejected_by_absolute_floor(self, tmp_path):
        """``--no-eval`` sessions carry no baseline reference; a collapsed replay
        must still be caught by the enablement absolute floor."""
        coord = _coord_with_baseline(tmp_path, 0.0)
        coord._promote_warm_replay(
            {"status": "succeeded", "output_throughput": 738.0, "accuracy": 0.20},
            task=_risky_task(),
        )
        assert _promoted(coord) is False
        outcome = coord.shared_state.warm_replay_outcome
        assert outcome["status"] == "accuracy_failed"
        assert "absolute floor" in str(outcome.get("reason") or "").lower()

    def test_no_baseline_sound_score_passes_absolute_floor(self, tmp_path):
        coord = _coord_with_baseline(tmp_path, 0.0)
        coord._promote_warm_replay(
            {"status": "succeeded", "output_throughput": 738.0, "accuracy": 0.89},
            task=_risky_task(),
        )
        assert _promoted(coord) is True


class TestAccuracyIsRecordedOnSuccess:
    """A promotion that was checked and passed is not the same record as one
    that was never checked."""

    def test_a_passing_replay_records_both_scores(self, tmp_path):
        coord = _coord_with_baseline(tmp_path, BASELINE_ACC)
        coord._promote_warm_replay(
            {"status": "succeeded", "output_throughput": 738.0, "accuracy": 0.89},
            task=_risky_task(),
        )
        outcome = coord.shared_state.warm_replay_outcome
        assert outcome["eval_ran"] is True
        assert outcome["replay_accuracy"] == pytest.approx(0.89)
        assert outcome["baseline_accuracy"] == pytest.approx(BASELINE_ACC)

    def test_the_promoted_stack_entry_carries_the_score(self, tmp_path):
        coord = _coord_with_baseline(tmp_path, BASELINE_ACC)
        coord._promote_warm_replay(
            {"status": "succeeded", "output_throughput": 738.0, "accuracy": 0.89},
            task=_risky_task(),
        )
        entry = coord.shared_state.optimization_stack[-1]
        assert entry["action"] == "replay_warm_recipe"
        assert entry["accuracy"] == pytest.approx(0.89)
