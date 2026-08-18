# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Warm replay must not promote a config that broke the model.

Every other lane that can promote a serving config checks accuracy when the
config touches a high-risk knob: ``ExploreExecutor`` calls
``is_high_accuracy_risk`` + ``accuracy_passed``, the framework agent and the
kernel integrate handler call ``accuracy_keep_block``. Warm replay promotes on
throughput alone.

That gap is not theoretical. Across the retained session pool, 45 of 241
promoted warm replays (19%) were promoted while the replayed config was
producing garbage — in the worst case a +23.95% "gain" on a config scoring
0.0000 on gsm8k against a 0.9014 baseline. The score was measured and written
into the same round directory; the promotion path simply never read it.

The promoted config then becomes the base every later measurement in that
session is taken against, and is written back to the recipe KB on throughput,
so the next session replays it again.
"""

from pathlib import Path

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator

from .test_warm_replay import _make_coord, _StubTask, _warm_recipe_t1

# A knob on the high-risk list; the same one the real cases carried.
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


def _promoted(coord: Coordinator) -> bool:
    return bool(coord.shared_state.optimization_stack)


class TestWarmReplayRejectsBrokenConfigs:
    def test_a_collapsed_score_blocks_promotion(self, tmp_path):
        """The case observed 45 times: big throughput win, ruined accuracy."""
        coord = _coord_with_baseline(tmp_path, BASELINE_ACC)
        # 600 -> 738 is +23%, comfortably over the bar; accuracy is destroyed.
        coord._promote_warm_replay(
            {"status": "succeeded", "output_throughput": 738.0, "accuracy": 0.20},
            task=_risky_task(),
        )
        assert _promoted(coord) is False
        outcome = coord.shared_state.warm_replay_outcome
        assert outcome["status"] != "reproduced"
        assert "accuracy" in str(outcome.get("reason") or "").lower()

    def test_a_missing_verdict_blocks_promotion(self, tmp_path):
        """Fail closed. A high-risk config with no accuracy evidence is exactly
        the state that produced the 45 bad promotions."""
        coord = _coord_with_baseline(tmp_path, BASELINE_ACC)
        coord._promote_warm_replay(
            {"status": "succeeded", "output_throughput": 738.0},
            task=_risky_task(),
        )
        assert _promoted(coord) is False
        assert coord.shared_state.warm_replay_outcome["status"] != "reproduced"

    def test_an_intact_score_still_promotes(self, tmp_path):
        """The gate must not cost a genuine win: the two KB champions traced in
        this investigation both held accuracy while gaining >90%."""
        coord = _coord_with_baseline(tmp_path, BASELINE_ACC)
        coord._promote_warm_replay(
            {"status": "succeeded", "output_throughput": 738.0, "accuracy": 0.89},
            task=_risky_task(),
        )
        assert _promoted(coord) is True
        assert coord.shared_state.warm_replay_outcome["status"] == "reproduced"


class TestWarmReplayGateScope:
    """Same trigger condition as ExploreExecutor, so the two lanes agree."""

    def test_a_config_with_no_high_risk_knob_is_not_gated(self, tmp_path):
        coord = _coord_with_baseline(tmp_path, BASELINE_ACC)
        coord._promote_warm_replay(
            {"status": "succeeded", "output_throughput": 738.0},
            task=_StubTask(params={"extra_server_args": SAFE_ARGS, "extra_envs": {}}),
        )
        assert _promoted(coord) is True

    def test_no_baseline_accuracy_means_no_reference_to_judge_against(self, tmp_path):
        """``accuracy_passed`` already treats a non-positive baseline as "skip";
        the gate must not invent a verdict it cannot support."""
        coord = _coord_with_baseline(tmp_path, 0.0)
        coord._promote_warm_replay(
            {"status": "succeeded", "output_throughput": 738.0},
            task=_risky_task(),
        )
        assert _promoted(coord) is True

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
