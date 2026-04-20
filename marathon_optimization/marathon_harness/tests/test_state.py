"""Unit tests for MarathonState — scoring, pop_action, kernel tracking, save/load."""

import json
import tempfile
import time
from pathlib import Path

from marathon_harness.state import MarathonState, compute_score


def test_compute_score_basic():
    score = compute_score(expected_gain=5.0, cost_minutes=30)
    assert score > 0
    # Higher gain, same cost → higher score
    score2 = compute_score(expected_gain=10.0, cost_minutes=30)
    assert score2 > score


def test_compute_score_risk_penalty():
    clean = compute_score(5.0, 30, accuracy_risk=0.0, crash_risk=0.0)
    risky = compute_score(5.0, 30, accuracy_risk=0.5, crash_risk=0.5)
    assert risky < clean


def test_compute_score_gap_multiplier():
    small_gap = compute_score(5.0, 30, target_gap_pct=10)
    big_gap = compute_score(5.0, 30, target_gap_pct=200)
    assert big_gap > small_gap


def test_pop_action_returns_highest_score():
    state = MarathonState(session_dir="/tmp/test")
    state.push_action({"id": "low", "score": 2})
    state.push_action({"id": "high", "score": 9})
    state.push_action({"id": "mid", "score": 5})

    popped = state.pop_action()
    assert popped["id"] == "high"
    assert len(state.action_stack) == 2


def test_pop_action_empty_returns_none():
    state = MarathonState(session_dir="/tmp/test")
    assert state.pop_action() is None


def test_pop_action_stability():
    """Multiple pops should return in descending score order."""
    state = MarathonState(session_dir="/tmp/test")
    state.push_action({"id": "a", "score": 3})
    state.push_action({"id": "b", "score": 7})
    state.push_action({"id": "c", "score": 5})

    ids = []
    while True:
        a = state.pop_action()
        if a is None:
            break
        ids.append(a["id"])

    assert ids == ["b", "c", "a"]


def test_kernel_fingerprinting():
    state = MarathonState(session_dir="/tmp/test")

    is_new = state.register_kernel("flash_attn_v2", "/sgl-workspace/sglang/kernels/attn.py",
                                   ["M=2048,N=128,K=128"])
    assert is_new is True
    assert state.unique_kernel_count == 1
    assert state.kernel_attempt_count == 1

    # Same kernel again → not new
    is_new = state.register_kernel("flash_attn_v2", "/sgl-workspace/sglang/kernels/attn.py",
                                   ["M=2048,N=128,K=128"])
    assert is_new is False
    assert state.unique_kernel_count == 1
    assert state.kernel_attempt_count == 2

    # Different shapes → new fingerprint
    is_new = state.register_kernel("flash_attn_v2", "/sgl-workspace/sglang/kernels/attn.py",
                                   ["M=4096,N=128,K=128"])
    assert is_new is True
    assert state.unique_kernel_count == 2


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        state = MarathonState(
            session_dir=td,
            model_name="TestModel",
            model_class="moe_mla",
            current_tput_per_gpu=42.5,
            baseline_tput_per_gpu=40.0,
        )
        state.push_action({"id": "test_action", "score": 7, "action": "dispatch-fix"})
        state.register_kernel("test_kernel", "/tmp/k.py")
        saved = state.save()

        loaded = MarathonState.load(saved)
        assert loaded.model_name == "TestModel"
        assert loaded.model_class == "moe_mla"
        assert loaded.current_tput_per_gpu == 42.5
        assert len(loaded.action_stack) == 1
        assert loaded.action_stack[0]["id"] == "test_action"
        assert loaded.unique_kernel_count == 1


def test_checkpoint_creates_symlink():
    with tempfile.TemporaryDirectory() as td:
        state = MarathonState(session_dir=td, model_name="Ckpt")
        ckpt_path = state.checkpoint("test")
        assert ckpt_path.exists()
        latest = Path(td) / "checkpoints" / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == ckpt_path.resolve()


def test_update_tier():
    state = MarathonState(session_dir="/tmp/test")
    state.start_time = time.time()
    state.current_time_tier = "tier1"

    # Simulate 4 hours elapsed
    state.start_time = time.time() - (4 * 3600)
    new = state.update_tier()
    assert new == "tier2"
    assert state.current_time_tier == "tier2"

    # Call again without time change → no new tier
    assert state.update_tier() is None


def test_apply_update_rules_post_crash():
    state = MarathonState(session_dir="/tmp/test")
    state.push_action({"id": "crash_action", "score": 8})
    state.apply_update_rules("post_crash", {"action_id": "crash_action"})
    assert state.action_stack[0]["score"] < 8  # should be penalized


def test_handoff_boosts():
    state = MarathonState(session_dir="/tmp/test")
    action = {"tags": ["marathon-candidate", "register-pressure-fixable"]}
    boost = state.apply_handoff_boosts(action)
    assert boost == 6  # 3 + 3
