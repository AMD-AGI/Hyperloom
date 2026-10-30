# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A KEEP that declares no launch controls keeps the previous current_best's controls."""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.shared_state import SharedState

HICACHE_FLAGS = ["--enable-hierarchical-cache", "--hicache-ratio"]


AGENTX_MEASUREMENT = {"benchmark_valid": True, "diagnostic_only": False, "workload_identity": "agentx:workload"}


def _coord(tmp_path: Path, previous: dict, *, agentx_backend: str = "") -> Coordinator:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        baseline_tput=1000.0,
        current_best={"tput": 1100.0, "extra_server_args": "--tp 8", **previous},
        model_path="/models/m",
        agentx_backend=agentx_backend,
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


def _removal_winner() -> dict:
    return {
        "name": "nohicache_spf",
        "output_throughput": 1200.0,
        "extra_server_args": "--schedule-policy shortest-prefill-first",
        "remove_args": HICACHE_FLAGS,
    }


def test_a_magpie_agentx_removal_keep_stays_relative_to_the_recipe(tmp_path):
    """Replace mode makes Magpie drop every recipe option, the speculative-decoding ones the golden gate pins too."""
    coord = _coord(tmp_path, {"workload_identity": "agentx:workload"}, agentx_backend="magpie")
    winner = {**_removal_winner(), **AGENTX_MEASUREMENT}

    assert coord.writeback.lift_to_current_best("explore", 1200.0, winner) is True

    launch = coord.writeback.current_best_launch_config()
    assert launch["args_mode"] == "append"
    assert launch["remove_args"] == HICACHE_FLAGS


def test_a_removal_keep_outside_magpie_agentx_replaces_the_inherited_args(tmp_path):
    coord = _coord(tmp_path, {})

    assert coord.writeback.lift_to_current_best("explore", 1200.0, _removal_winner()) is True

    assert coord.writeback.current_best_launch_config()["args_mode"] == "replace"


def test_nothing_is_carried_when_the_previous_best_declares_no_controls(tmp_path):
    coord = _coord(tmp_path, {})
    winner = {"name": "src-patch", "output_throughput": 1200.0, "extra_server_args": "--tp 8"}

    assert coord.writeback.lift_to_current_best("integrate_patch", 1200.0, winner) is True

    best = coord.shared_state.current_best
    assert not any(best.get(key) for key in ("args_mode", "remove_args", "unset_envs"))
