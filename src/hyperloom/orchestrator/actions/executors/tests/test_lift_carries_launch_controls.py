# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A KEEP that declares no launch controls keeps the previous current_best's controls."""

from __future__ import annotations

from pathlib import Path

from hyperloom.inference_optimizer.grid_server_args import compose_server_args
from hyperloom.orchestrator.actions.executors._workload_envs import _apply_caller_removals
from hyperloom.orchestrator.actions.executors.profile import _merge_profile_params
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.shared_state import SharedState, inject_stack_base_params

HICACHE_FLAGS = ["--enable-hierarchical-cache", "--hicache-ratio"]

EAGLE_ARGS = "--tp 8 --speculative-algorithm EAGLE3 --speculative-num-steps 3 --speculative-num-draft-tokens 4"


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


def _eagle_retune_winner() -> dict:
    """A winner whose recipe drops the inherited speculative flags and then asserts its own values."""
    return {
        "name": "eagle3_steps3",
        "output_throughput": 1200.0,
        "extra_server_args": EAGLE_ARGS,
        "remove_args": [*HICACHE_FLAGS, "--speculative-num-steps", "--speculative-num-draft-tokens"],
    }


def test_a_removal_the_winner_reasserts_is_not_carried_onto_the_snapshot(tmp_path):
    """``extra_server_args`` is the composed launch state, so a removal naming it is no delta against it."""
    coord = _coord(tmp_path, {})

    assert coord.writeback.lift_to_current_best("explore", 1200.0, _eagle_retune_winner()) is True

    best = coord.shared_state.current_best
    assert "--speculative-num-steps 3 --speculative-num-draft-tokens 4" in best["extra_server_args"]
    # The hicache flags the recipe suppresses are absent from the snapshot, so those removals still mean something.
    assert best["remove_args"] == HICACHE_FLAGS


def test_a_task_seeded_from_current_best_launches_the_speculative_flags_it_measured(tmp_path):
    coord = _coord(tmp_path, {})
    assert coord.writeback.lift_to_current_best("explore", 1200.0, _eagle_retune_winner()) is True

    params: dict = {}
    inject_stack_base_params(params, coord.shared_state)
    _merge_profile_params(params)
    composed = compose_server_args(
        base_extra_args=params.get("extra_server_args"),
        remove_args=params.get("remove_args"),
        args_mode=params.get("args_mode", "append"),
    )

    # SGLang's EAGLE path asserts both are set; composing them away leaves the server unlaunchable.
    assert "--speculative-num-steps 3" in composed
    assert "--speculative-num-draft-tokens 4" in composed


def _hicache_retune_winner() -> dict:
    """A winner whose recipe clears the inherited hicache knob and then sets its own value."""
    return {
        "name": "hicache_1_6",
        "output_throughput": 1200.0,
        "extra_server_args": "--tp 8",
        "extra_envs": {"SGLANG_HICACHE_RATIO": "1.6"},
        "unset_envs": ["SGLANG_HICACHE_RATIO", "SGLANG_MOE_PADDING"],
    }


def test_an_unset_the_winner_reasserts_is_not_carried_onto_the_snapshot(tmp_path):
    """``extra_envs`` is the composed launch state, so an unset naming it is no delta against it."""
    coord = _coord(tmp_path, {"extra_envs": {"SGLANG_HICACHE_RATIO": "0.8", "SGLANG_MOE_PADDING": "1"}})

    assert coord.writeback.lift_to_current_best("explore", 1200.0, _hicache_retune_winner()) is True

    best = coord.shared_state.current_best
    assert best["extra_envs"]["SGLANG_HICACHE_RATIO"] == "1.6"
    # The padding knob the recipe clears is absent from the snapshot, so that unset still means something.
    assert best["unset_envs"] == ["SGLANG_MOE_PADDING"]


def test_a_task_seeded_from_current_best_launches_the_env_value_it_measured(tmp_path):
    coord = _coord(tmp_path, {"extra_envs": {"SGLANG_HICACHE_RATIO": "0.8", "SGLANG_MOE_PADDING": "1"}})
    assert coord.writeback.lift_to_current_best("explore", 1200.0, _hicache_retune_winner()) is True

    params: dict = {}
    inject_stack_base_params(params, coord.shared_state)
    _merge_profile_params(params)
    envs = dict(params.get("extra_envs") or {})
    _apply_caller_removals(envs, "EXTRA_SGLANG_ARGS", params.get("remove_args"), params.get("unset_envs"), [])

    # The relaunch applies the carried unsets over the envs it just laid down, so a stale one wipes the winner's value.
    assert envs["SGLANG_HICACHE_RATIO"] == "1.6"
    assert "SGLANG_MOE_PADDING" not in envs


def test_nothing_is_carried_when_the_previous_best_declares_no_controls(tmp_path):
    coord = _coord(tmp_path, {})
    winner = {"name": "src-patch", "output_throughput": 1200.0, "extra_server_args": "--tp 8"}

    assert coord.writeback.lift_to_current_best("integrate_patch", 1200.0, winner) is True

    best = coord.shared_state.current_best
    assert not any(best.get(key) for key in ("args_mode", "remove_args", "unset_envs"))
