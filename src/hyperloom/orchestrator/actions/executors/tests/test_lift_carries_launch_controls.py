# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A KEEP that declares no launch controls keeps the previous current_best's controls."""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.shared_state import SharedState

HICACHE_FLAGS = ["--enable-hierarchical-cache", "--hicache-ratio"]


def _coord(tmp_path: Path, previous: dict) -> Coordinator:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        baseline_tput=1000.0,
        current_best={"tput": 1100.0, "extra_server_args": "--tp 8", **previous},
        model_path="/models/m",
    )
    return coord


def test_append_winner_keeps_replace_mode_and_removals(tmp_path):
    coord = _coord(
        tmp_path,
        {"args_mode": "replace", "remove_args": HICACHE_FLAGS, "unset_envs": ["FOO"]},
    )
    winner = {"name": "src-patch", "output_throughput": 1200.0, "extra_server_args": "--tp 8"}

    assert coord.writeback.lift_to_current_best("integrate_patch", 1200.0, winner) is True

    best = coord.shared_state.current_best
    assert best["args_mode"] == "replace"
    assert best["remove_args"] == HICACHE_FLAGS
    assert best["unset_envs"] == ["FOO"]


def test_winner_controls_override_previous(tmp_path):
    coord = _coord(tmp_path, {"args_mode": "replace", "remove_args": HICACHE_FLAGS})
    winner = {
        "name": "v",
        "output_throughput": 1200.0,
        "extra_server_args": "--tp 8",
        "args_mode": "replace",
        "remove_args": ["--other"],
    }

    assert coord.writeback.lift_to_current_best("explore", 1200.0, winner) is True
    assert coord.shared_state.current_best["remove_args"] == ["--other"]


def test_launch_config_after_an_append_winner_still_removes_the_flags(tmp_path):
    coord = _coord(tmp_path, {"args_mode": "replace", "remove_args": HICACHE_FLAGS})
    winner = {"name": "src-patch", "output_throughput": 1200.0, "extra_server_args": "--tp 8"}

    assert coord.writeback.lift_to_current_best("integrate_patch", 1200.0, winner) is True

    launch = coord.writeback.current_best_launch_config()
    assert launch["args_mode"] == "replace"
    assert launch["remove_args"] == HICACHE_FLAGS


def test_nothing_is_carried_when_the_previous_best_declares_no_controls(tmp_path):
    coord = _coord(tmp_path, {})
    winner = {"name": "src-patch", "output_throughput": 1200.0, "extra_server_args": "--tp 8"}

    assert coord.writeback.lift_to_current_best("integrate_patch", 1200.0, winner) is True

    best = coord.shared_state.current_best
    assert not any(best.get(key) for key in ("args_mode", "remove_args", "unset_envs"))
