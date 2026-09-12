# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The GEAK handoff's shell environment preserves current-best values."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["two words", '{"key": "a b", "eq": "a=b"}', "user's config", r"a\b", "", "plain"])
async def test_handoff_preserves_environment_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    expected = {"SGLANG_USE_AITER": "1", "CUSTOM_CONFIG": value}
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        model_path="/models/example",
        current_best={"extra_envs": expected, "optimization_stack": []},
    )
    coord.phase_kernel._record_geak_kernel_journey = lambda _result: None
    monkeypatch.setenv("FRAMEWORK", "sglang")

    def stop_after_handoff(_name: str) -> Path:
        raise RuntimeError("stop after handoff write")

    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers._kernel_agent_tool_path",
        stop_after_handoff,
    )
    await coord._run_geak_kernel_phase(from_phase="KERNEL")

    handoff = json.loads((tmp_path / "geak" / "handoff.json").read_text(encoding="utf-8"))
    assert handoff["baseline_env_spec"]["config"]["extra_envs"] == expected
    assert dict(token.split("=", 1) for token in shlex.split(handoff["accepted_env"])) == expected


@pytest.fixture
def capture_handoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def capture(current, *, observed_flags="", framework="sglang", baseline_flags=""):
        recipe = tmp_path / "baseline.yaml"
        recipe.write_text(
            yaml.safe_dump(
                {
                    "benchmark": {
                        "framework": framework,
                        "envs": {
                            "TP": "1",
                            "EXTRA_VLLM_ARGS" if framework == "vllm" else "EXTRA_SGLANG_ARGS": baseline_flags,
                            "SGLANG_AITER_MLA_PERSIST": "1",
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        coord = Coordinator.__new__(Coordinator)
        coord.session_dir = tmp_path
        coord.shared_state = SharedState(
            model_path="/models/example",
            baseline_config_path=str(recipe),
            last_baseline={"extras": {"materialized_config": str(recipe)}},
            current_best={**current, "measurement": {"resolved_server_launch_flags": observed_flags}},
        )
        coord.phase_kernel._record_geak_kernel_journey = lambda _result: None
        monkeypatch.setenv("FRAMEWORK", framework)
        monkeypatch.delenv("MAX_MODEL_LEN", raising=False)
        monkeypatch.delenv("GPU_MEMORY_UTILIZATION", raising=False)

        def stop_after_handoff(_name: str) -> Path:
            raise RuntimeError("stop after handoff write")

        monkeypatch.setattr(
            "hyperloom.orchestrator.kernel.request_handlers._kernel_agent_tool_path",
            stop_after_handoff,
        )
        await coord._run_geak_kernel_phase(from_phase="KERNEL")
        return json.loads((tmp_path / "geak" / "handoff.json").read_text(encoding="utf-8"))

    return capture


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("current", "expected_mode", "expected_remove", "expected_unset", "observed_flags"),
    [
        ({}, "append", [], [], ""),
        ({"unset_envs": " SGLANG_AITER_MLA_PERSIST "}, "append", [], ["SGLANG_AITER_MLA_PERSIST"], ""),
        ({"remove_args": [" --disable-cuda-graph ", ""]}, "append", ["--disable-cuda-graph"], [], ""),
        ({"extra_server_args": "", "args_mode": " REPLACE "}, "replace", [], [], ""),
        (
            {"extra_server_args": "--context-length 4096", "args_mode": "replace"},
            "replace",
            [],
            [],
            "--context-length 4096 --mem-fraction-static 0.8",
        ),
        (
            {"extra_envs": {"SGLANG_AITER_MLA_PERSIST": "3"}, "unset_envs": ["SGLANG_AITER_MLA_PERSIST"]},
            "append",
            [],
            ["SGLANG_AITER_MLA_PERSIST"],
            "",
        ),
    ],
    ids=["legacy", "unset_env", "remove_arg", "empty_replacement", "observed_replacement", "explicit_env_readd"],
)
async def test_handoff_transports_current_best_launch_controls(
    capture_handoff, current, expected_mode, expected_remove, expected_unset, observed_flags
) -> None:
    handoff = await capture_handoff(current, observed_flags=observed_flags)
    config = handoff["baseline_env_spec"]["config"]

    assert config["args_mode"] == expected_mode
    assert config["remove_args"] == expected_remove
    assert config["unset_envs"] == expected_unset
    assert config["server_launch_flags"] == observed_flags
    assert config["extra_server_args"] == handoff["accepted_flags"] == current.get("extra_server_args", "")
    assert config["extra_envs"] == current.get("extra_envs", {})
    assert dict(token.split("=", 1) for token in shlex.split(handoff["accepted_env"])) == config["extra_envs"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("current", "observed_flags", "expected_max_len", "expected_mem"),
    [
        ({"extra_server_args": "", "args_mode": "replace"}, "", 0, 0.0),
        ({"remove_args": ["--max-model-len", "--gpu-memory-utilization"]}, "", 0, 0.0),
        (
            {"extra_server_args": "--max-model-len 4096 --gpu-memory-utilization 0.7", "args_mode": "replace"},
            "",
            4096,
            0.7,
        ),
        (
            {"extra_server_args": "", "args_mode": "replace"},
            "--max-model-len 8192 --gpu-memory-utilization 0.8",
            8192,
            0.8,
        ),
        ({}, "", 2248, 0.9),
    ],
    ids=["empty_replacement", "removed_knobs", "declared_replacement", "observed_snapshot", "legacy_recipe"],
)
async def test_handoff_serving_fidelity_respects_current_best_launch_controls(
    capture_handoff, current, observed_flags, expected_max_len, expected_mem
) -> None:
    handoff = await capture_handoff(
        current,
        framework="vllm",
        baseline_flags="--max-model-len 2248 --gpu-memory-utilization 0.9",
        observed_flags=observed_flags,
    )

    assert handoff["max_model_len"] == expected_max_len
    assert handoff["mem_fraction"] == pytest.approx(expected_mem)
