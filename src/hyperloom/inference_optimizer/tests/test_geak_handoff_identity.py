"""Regression tests for GEAK handoff/current-best measurement identity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.writeback import WritebackCollaborator
from hyperloom.orchestrator.state.shared_state import SharedState


def _writeback(tmp_path: Path, state: SharedState) -> WritebackCollaborator:
    writer = WritebackCollaborator.__new__(WritebackCollaborator)
    writer.session_dir = tmp_path
    writer.shared_state = state
    return writer


def _verified_current_best(tmp_path: Path) -> SharedState:
    state = SharedState(
        baseline_tput=100.0,
        current_best={
            "action": "explore",
            "tput": 1403.43,
            "extra_server_args": "--dsa-prefill-backend aiter --dsa-decode-backend tilelang",
            "extra_envs": {"SGLANG_USE_AITER": "1"},
            "optimization_stack": [],
            "measurement": {
                "tput": 1403.43,
                "resolved_server_launch_flags": "--mem-fraction-static 0.8",
                "benchmark_workspace": "runs/explore/dsa/stack_rebench",
            },
        },
    )
    writer = _writeback(tmp_path, state)
    spec = writer.build_env_spec()
    state.current_best["measurement"]["launch_identity"] = spec["launch_identity"]
    return state


def test_env_spec_uses_embedded_current_best_stack_not_global_stack(tmp_path: Path) -> None:
    state = _verified_current_best(tmp_path)
    state.current_best["optimization_stack"] = [
        {
            "scope": "source_patch",
            "variant_name": "current",
            "source_snapshot": str(tmp_path / "current"),
            "source_snapshot_complete": True,
        }
    ]
    state.optimization_stack = [
        *state.current_best["optimization_stack"],
        {
            "scope": "source_patch",
            "variant_name": "stale",
            "source_snapshot": str(tmp_path / "stale"),
            "source_snapshot_complete": True,
        },
    ]

    spec = _writeback(tmp_path, state).build_env_spec()

    assert [entry["id"] for entry in spec["source_snapshots"]] == ["current"]


def test_env_spec_never_recovers_server_flags_from_unrelated_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _verified_current_best(tmp_path)
    state.current_best["measurement"] = {"tput": 1403.43}
    run = tmp_path / "runs" / "wrong-variant"
    run.mkdir(parents=True)
    (run / "inferencex_result.json").write_text(json.dumps({"output_throughput": 1403.43}), encoding="utf-8")
    (run / "server.log").write_text(
        "+ python -m sglang.launch_server --model-path /model --speculative-algorithm NEXTN\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FRAMEWORK", "sglang")

    spec = _writeback(tmp_path, state).build_env_spec()

    assert spec["config"]["server_launch_flags"] == ""
    assert "--speculative-algorithm" not in json.dumps(spec)


def test_measurement_identity_invalidates_when_current_best_config_changes(tmp_path: Path) -> None:
    recipe = tmp_path / "baseline.yaml"
    recipe.write_text("benchmark: {model: /models/a}\n", encoding="utf-8")
    state = SharedState(
        baseline_config_path=str(recipe),
        current_best={
            "action": "explore",
            "tput": 100.0,
            "extra_server_args": "--block-size 32",
            "extra_envs": {"A": "1"},
            "optimization_stack": [],
        }
    )
    writer = _writeback(tmp_path, state)
    writer._stamp_current_best_measurement()
    identity = state.current_best["measurement"]["launch_identity"]

    assert state.current_best_measurement["launch_identity"] == identity
    state.current_best["extra_server_args"] = "--block-size 64"

    assert writer.build_env_spec()["launch_identity"] != identity


def test_measurement_identity_invalidates_when_recipe_content_changes(tmp_path: Path) -> None:
    recipe = tmp_path / "baseline.yaml"
    recipe.write_text("benchmark: {model: /models/a}\n", encoding="utf-8")
    state = SharedState(
        baseline_config_path=str(recipe),
        current_best={"tput": 100.0, "optimization_stack": []},
    )
    writer = _writeback(tmp_path, state)
    writer._stamp_current_best_measurement()
    identity = state.current_best_measurement["launch_identity"]

    recipe.write_text("benchmark: {model: /models/b}\n", encoding="utf-8")

    assert writer.build_env_spec()["launch_identity"] != identity


@pytest.mark.asyncio
async def test_handoff_rejects_stale_tput_without_matching_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        baseline_tput=17981.31,
        current_best={
            "action": "explore",
            "tput": 20845.60,
            "extra_server_args": "--kv-cache-dtype fp8",
            "extra_envs": {"VLLM_ROCM_USE_AITER": "1"},
            "optimization_stack": [],
        },
        model_path="/models/qwen",
        gpu_type="mi355x",
        isl=1024,
        osl=1024,
        conc=64,
    )
    coord.phase_kernel._record_geak_kernel_journey = lambda _result: None
    monkeypatch.setenv("FRAMEWORK", "vllm")

    def _stop_after_handoff(_name: str) -> Path:
        raise RuntimeError("stop after handoff write")

    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers._kernel_agent_tool_path",
        _stop_after_handoff,
    )

    await coord._run_geak_kernel_phase(from_phase="KERNEL")

    handoff = json.loads((tmp_path / "geak" / "handoff.json").read_text(encoding="utf-8"))
    assert handoff["accepted_flags"] == "--kv-cache-dtype fp8"
    assert handoff["same_config_reference_status"] == "unverified"
    assert handoff["orchestrator_best_tput_same_config"] == 0.0


@pytest.mark.asyncio
async def test_handoff_uses_only_matching_current_best_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _verified_current_best(tmp_path)
    state.model_path = "/models/glm"
    state.gpu_type = "mi355x"
    state.isl = 8192
    state.osl = 1024
    state.conc = 64
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = state
    coord.phase_kernel._record_geak_kernel_journey = lambda _result: None
    monkeypatch.setenv("FRAMEWORK", "sglang")

    def _stop_after_handoff(_name: str) -> Path:
        raise RuntimeError("stop after handoff write")

    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers._kernel_agent_tool_path",
        _stop_after_handoff,
    )

    await coord._run_geak_kernel_phase(from_phase="KERNEL")

    handoff = json.loads((tmp_path / "geak" / "handoff.json").read_text(encoding="utf-8"))
    assert handoff["same_config_reference_status"] == "verified"
    assert handoff["orchestrator_best_tput_same_config"] == pytest.approx(1403.43)
    assert handoff["baseline_env_spec"]["launch_identity"] == handoff["same_config_reference_identity"]
