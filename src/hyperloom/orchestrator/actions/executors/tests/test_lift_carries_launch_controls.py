# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A KEEP that declares no launch controls keeps the previous current_best's controls."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom.inference_optimizer.grid_server_args import compose_server_args
from hyperloom.orchestrator.actions.executors._workload_envs import _apply_caller_removals
from hyperloom.orchestrator.actions.executors.agentx_point import LaunchChanges, apply_launch_changes
from hyperloom.orchestrator.actions.executors.profile import _merge_profile_params
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.shared_state import SharedState, inject_stack_base_params

HICACHE_FLAGS = ["--enable-hierarchical-cache", "--hicache-ratio"]
CAPTURE_CAP = "--max-cudagraph-capture-size"
#: The AgentX recipe a session base holds, which no current_best restates.
RECIPE = (
    f"--tensor-parallel-size 4 --max-model-len 65536 --enable-hierarchical-cache --hicache-ratio 2 {CAPTURE_CAP} 512"
)

EAGLE_ARGS = "--tp 8 --speculative-algorithm EAGLE3 --speculative-num-steps 3 --speculative-num-draft-tokens 4"


AGENTX_MEASUREMENT = {"valid_measurement": True, "diagnostic_only": False, "workload_identity": "agentx:workload"}


def _coord(tmp_path: Path, previous: dict, *, agentx_backend: str = "") -> Coordinator:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        baseline_tput=1000.0,
        current_best={"tput": 1100.0, "extra_server_args": "--tp 8", **previous},
        model_path="/models/m",
        agentx_backend=agentx_backend,
        benchmark_mode="agentx" if agentx_backend else "",
        # The winners here carry throughput only; the launch-control rules under test do not depend on the axis.
        grading={"objective": GRADED_OUTPUT},
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


def test_a_removal_keep_outside_agentx_replaces_the_inherited_args(tmp_path):
    coord = _coord(tmp_path, {})

    assert coord.writeback.lift_to_current_best("explore", 1200.0, _removal_winner()) is True

    launch = coord.writeback.current_best_launch_config()
    assert launch["args_mode"] == "replace"
    assert launch["remove_args"] == HICACHE_FLAGS


@pytest.mark.parametrize("agentx_backend", ["inferencex", "mlperf"])
def test_an_agentx_removal_keep_stays_relative_to_the_recipe(tmp_path, agentx_backend):
    """The KEEP holds only its delta; replace mode would make every later round drop the recipe's other flags."""
    coord = _coord(tmp_path, {"workload_identity": "agentx:workload"}, agentx_backend=agentx_backend)
    winner = {**_removal_winner(), **AGENTX_MEASUREMENT}

    assert coord.writeback.lift_to_current_best("explore", 1200.0, winner) is True

    launch = coord.writeback.current_best_launch_config()
    assert launch["args_mode"] != "replace"
    assert launch["remove_args"] == HICACHE_FLAGS
    # An append KEEP stacks on the flags earlier KEEPs accepted rather than standing in for them.
    assert launch["extra_server_args"].split() == ["--tp", "8", "--schedule-policy", "shortest-prefill-first"]


def test_an_agentx_keep_that_only_removes_an_accepted_flag_keeps_it_removed(tmp_path):
    """Explore records such a KEEP with no args of its own and the stack after its removal; stacking it on the
    previous best must not put the removed flag back, nor drop the removal as one the args contradict."""
    previous = {"workload_identity": "agentx:workload", "extra_server_args": f"{CAPTURE_CAP} 512 --tp 8"}
    coord = _coord(tmp_path, previous, agentx_backend="inferencex")
    winner = {
        "name": "no_capture_cap",
        "output_throughput": 1200.0,
        "candidate_extra_server_args": "",
        "extra_server_args": "--tp 8",
        "remove_args": [CAPTURE_CAP],
        **AGENTX_MEASUREMENT,
    }

    assert coord.writeback.lift_to_current_best("explore", 1200.0, winner) is True

    launch = coord.writeback.current_best_launch_config()
    assert launch["extra_server_args"].split() == ["--tp", "8"]
    assert launch["remove_args"] == [CAPTURE_CAP]
    assert launch["args_mode"] != "replace"


def _agentx_append_best(tmp_path: Path, **controls) -> Coordinator:
    """An AgentX session whose current_best is an explore KEEP stored as a delta on the recipe."""
    previous = {"workload_identity": "agentx:workload", "extra_server_args": "", **controls}
    return _coord(tmp_path, previous, agentx_backend="inferencex")


def _integrate_winner(**fields) -> dict:
    """An integrate KEEP as the lift receives one: the round's own launch controls only, not the stack's."""
    return {"name": "integrate", "output_throughput": 1200.0, **AGENTX_MEASUREMENT, **fields}


def _next_round_argv(coord: Coordinator) -> list[str]:
    """The argv the next AgentX round launches: the recipe laid down, then the accepted launch composed on it."""
    launch = coord.writeback.current_best_launch_config()
    benchmark: dict = {"agentx": {"enabled": True}}
    apply_launch_changes(benchmark, LaunchChanges(RECIPE, {}, replace_args=True), accepted=True)
    apply_launch_changes(
        benchmark,
        LaunchChanges(
            launch["extra_server_args"],
            launch["extra_envs"],
            launch["remove_args"],
            launch["unset_envs"],
            launch["args_mode"] == "replace",
        ),
        accepted=True,
    )
    return benchmark["agentx"]["launch_overrides"]["append_args"]


def test_a_later_agentx_keep_with_removals_of_its_own_keeps_the_earlier_removals(tmp_path):
    coord = _agentx_append_best(tmp_path, remove_args=HICACHE_FLAGS)
    winner = _integrate_winner(extra_server_args="--schedule-policy lpm", remove_args=[CAPTURE_CAP])

    assert coord.writeback.lift_to_current_best("integrate_patch", 1200.0, winner) is True

    assert coord.writeback.current_best_launch_config()["remove_args"] == [*HICACHE_FLAGS, CAPTURE_CAP]
    argv = _next_round_argv(coord)
    assert argv[argv.index("--tensor-parallel-size") + 1] == "4"
    assert not {*HICACHE_FLAGS, CAPTURE_CAP} & set(argv)
    assert "--schedule-policy" in argv


def test_a_later_agentx_keep_that_only_unsets_an_env_keeps_the_earlier_removals_and_unsets(tmp_path):
    coord = _agentx_append_best(
        tmp_path, remove_args=HICACHE_FLAGS, unset_envs=["SGLANG_MOE_PADDING"], extra_envs={"KEEP_ME": "1"}
    )
    winner = _integrate_winner(extra_server_args="", unset_envs=["SGLANG_HICACHE_RATIO"])

    assert coord.writeback.lift_to_current_best("integrate_patch", 1200.0, winner) is True

    launch = coord.writeback.current_best_launch_config()
    assert launch["remove_args"] == HICACHE_FLAGS
    assert launch["unset_envs"] == ["SGLANG_MOE_PADDING", "SGLANG_HICACHE_RATIO"]
    assert launch["extra_envs"] == {"KEEP_ME": "1"}


def test_a_later_agentx_keep_that_restores_an_earlier_removal_drops_that_removal(tmp_path):
    """The winner was measured with the flag and the env back, so neither earlier removal may suppress them now."""
    coord = _agentx_append_best(tmp_path, remove_args=HICACHE_FLAGS, unset_envs=["SGLANG_MOE_PADDING"])
    winner = _integrate_winner(
        extra_server_args="--hicache-ratio 1.5",
        extra_envs={"SGLANG_MOE_PADDING": "1"},
        remove_args=[CAPTURE_CAP],
    )

    assert coord.writeback.lift_to_current_best("integrate_patch", 1200.0, winner) is True

    launch = coord.writeback.current_best_launch_config()
    assert launch["remove_args"] == ["--enable-hierarchical-cache", CAPTURE_CAP]
    assert not launch["unset_envs"]
    argv = _next_round_argv(coord)
    assert argv[argv.index("--hicache-ratio") + 1] == "1.5"
    assert not {"--enable-hierarchical-cache", CAPTURE_CAP} & set(argv)


def test_an_agentx_round_on_a_removal_keep_still_launches_the_recipe(tmp_path):
    """A round composes the session base, then the accepted launch: the recipe's TP must survive the KEEP's removal."""
    coord = _coord(tmp_path, {"workload_identity": "agentx:workload"}, agentx_backend="inferencex")
    winner = {**_removal_winner(), **AGENTX_MEASUREMENT}
    assert coord.writeback.lift_to_current_best("explore", 1200.0, winner) is True

    argv = _next_round_argv(coord)
    assert argv[argv.index("--tensor-parallel-size") + 1] == "4"
    assert "--max-model-len" in argv
    assert not set(HICACHE_FLAGS) & set(argv)
    assert "--schedule-policy" in argv


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


def test_an_agentx_removal_the_winner_reasserts_is_not_carried_onto_the_snapshot(tmp_path):
    coord = _coord(tmp_path, {"workload_identity": "agentx:workload"}, agentx_backend="inferencex")
    winner = {**_eagle_retune_winner(), **AGENTX_MEASUREMENT}

    assert coord.writeback.lift_to_current_best("explore", 1200.0, winner) is True

    assert coord.shared_state.current_best["remove_args"] == HICACHE_FLAGS


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
    _apply_caller_removals(envs, "EXTRA_SGLANG_ARGS", params.get("remove_args"), params.get("unset_envs"))

    # The relaunch applies the carried unsets over the envs it just laid down, so a stale one wipes the winner's value.
    assert envs["SGLANG_HICACHE_RATIO"] == "1.6"
    assert "SGLANG_MOE_PADDING" not in envs


def test_nothing_is_carried_when_the_previous_best_declares_no_controls(tmp_path):
    coord = _coord(tmp_path, {})
    winner = {"name": "src-patch", "output_throughput": 1200.0, "extra_server_args": "--tp 8"}

    assert coord.writeback.lift_to_current_best("integrate_patch", 1200.0, winner) is True

    best = coord.shared_state.current_best
    assert not any(best.get(key) for key in ("args_mode", "remove_args", "unset_envs"))
