# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A native AgentX GEAK proposal reproduces the measured round's tunables and leaves GEAK its own topology."""

import pytest

from hyperloom.inference_optimizer.agentx.geak_proxy import seed_native_geak_proxy


def _handoff(launch_json: str) -> dict:
    return {
        "accepted_flags": "--accepted 1",
        "raw_baseline_tput": 140.0,
        "baseline_env_spec": {
            "launch_json": launch_json,
            "config": {
                "server_launch_flags": "--tp 8 --port 30000 --block-size 128 --enable-prefix-caching",
                "extra_server_args": "--block-size 128",
                "extra_envs": {"SGLANG_USE_AITER": "1", "MALLOC_CONF": "background_thread:true"},
                "unset_envs": ["AITER_REMOVED"],
            },
        },
    }


def test_native_geak_keeps_the_rounds_tunables_and_env_and_drops_what_geak_owns() -> None:
    handoff = _handoff("/runs/r/run/launch.json")
    original = handoff["baseline_env_spec"]

    seed_native_geak_proxy(handoff)

    config = handoff["baseline_env_spec"]["config"]
    assert config["server_launch_flags"] == "--block-size 128 --enable-prefix-caching"
    assert config["extra_server_args"] == ""
    assert config["extra_envs"] == {"SGLANG_USE_AITER": "1", "MALLOC_CONF": "background_thread:true"}
    assert config["unset_envs"] == ["AITER_REMOVED"]
    assert handoff["accepted_flags"] == ""
    assert handoff["accepted_env"] == "SGLANG_USE_AITER=1 MALLOC_CONF=background_thread:true"
    assert (handoff["raw_baseline_tput"], handoff["orchestrator_best_tput_same_config"]) == (0.0, 0.0)
    assert handoff["same_config_reference_verification_status"] == "unverified_workload"
    assert original["config"]["server_launch_flags"].startswith("--tp 8"), "the caller's env spec is not mutated"


def test_native_geak_refuses_a_round_that_left_no_launch_json() -> None:
    with pytest.raises(ValueError, match="launch.json"):
        seed_native_geak_proxy(_handoff(""))
