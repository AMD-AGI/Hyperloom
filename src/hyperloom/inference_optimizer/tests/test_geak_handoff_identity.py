"""Regression tests for GEAK handoff/current-best measurement identity."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.common.launch_log_evidence import observed_sglang_server_identity_from_log
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
        },
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
    assert handoff["same_config_reference_verification_status"] == "verified_observed"
    assert handoff["orchestrator_best_tput_same_config"] == pytest.approx(1403.43)
    assert handoff["baseline_env_spec"]["launch_identity"] == handoff["same_config_reference_identity"]


@pytest.mark.asyncio
async def test_handoff_marks_declared_only_identity_without_faking_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _verified_current_best(tmp_path)
    measurement = state.current_best["measurement"]
    measurement["resolved_server_launch_flags"] = ""
    measurement["launch_evidence"] = {
        "schema_version": 1,
        "framework": "sglang",
        "requested_server_args": "--mem-fraction-static 0.8",
        "requested_server_env": {"TP": "1"},
        "recipe_digest": "sha256:declared",
        "actual_server_log_path": "",
        "observed_server_launch_flags": "",
    }
    measurement["launch_identity"] = _writeback(tmp_path, state).build_env_spec()["launch_identity"]
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
    assert handoff["same_config_reference_status"] == "unverified"
    assert handoff["same_config_reference_verification_status"] == "verified_declared_only"
    assert handoff["same_config_reference_observed_identity"] == ""
    assert handoff["measurement_evidence"]["requested_server_args"] == "--mem-fraction-static 0.8"
    assert handoff["orchestrator_best_tput_same_config"] == 0.0


@pytest.mark.asyncio
async def test_handoff_does_not_verify_matching_identity_without_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _verified_current_best(tmp_path)
    measurement = state.current_best["measurement"]
    measurement["resolved_server_launch_flags"] = ""
    measurement.pop("launch_evidence", None)
    measurement["launch_identity"] = _writeback(tmp_path, state).build_env_spec()["launch_identity"]
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
    assert handoff["same_config_reference_status"] == "unverified"
    assert handoff["same_config_reference_verification_status"] == "unverified"
    assert handoff["orchestrator_best_tput_same_config"] == 0.0


@pytest.mark.asyncio
async def test_handoff_exposes_archived_sglang_observed_identity_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _verified_current_best(tmp_path)
    measurement = state.current_best["measurement"]
    measurement["resolved_server_launch_flags"] = ""
    measurement["launch_evidence"] = {
        "schema_version": 1,
        "framework": "sglang",
        "requested_server_args": "--mem-fraction-static 0.8",
        "requested_server_env": {"TP": "8"},
        "recipe_digest": "sha256:declared",
        "observed_server_identity": {
            "context_length": 8192,
            "mem_fraction_static": 0.8,
            "model_path": "/models/qwen",
            "tp_size": 8,
        },
    }
    measurement["launch_identity"] = _writeback(tmp_path, state).build_env_spec()["launch_identity"]
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = state
    coord.phase_kernel._record_geak_kernel_journey = lambda _result: None
    monkeypatch.setenv("FRAMEWORK", "sglang")

    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.request_handlers._kernel_agent_tool_path",
        lambda _name: (_ for _ in ()).throw(RuntimeError("stop after handoff write")),
    )
    await coord._run_geak_kernel_phase(from_phase="KERNEL")

    handoff = json.loads((tmp_path / "geak" / "handoff.json").read_text(encoding="utf-8"))
    expected = measurement["launch_evidence"]["observed_server_identity"]
    assert handoff["same_config_reference_verification_status"] == "verified_observed"
    assert handoff["same_config_observed_identity"] == expected
    assert handoff["observed_server_identity"] == expected


def test_sglang_server_args_fallback_records_declared_resolved_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeServerArgs:
        @classmethod
        def add_cli_args(cls, parser):
            parser.add_argument("--model-path")
            parser.add_argument("--mem-fraction-static", type=float, default=0.9)
            parser.add_argument("--context-length", type=int, default=4096)

        def __init__(self, **kwargs):
            self.model_path = kwargs["model_path"]
            self.mem_fraction_static = kwargs["mem_fraction_static"]
            self.context_length = kwargs["context_length"]

    sglang = ModuleType("sglang")
    srt = ModuleType("sglang.srt")
    server_args = ModuleType("sglang.srt.server_args")
    server_args.ServerArgs = FakeServerArgs
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)

    resolved = WritebackCollaborator._resolved_sglang_server_config(
        {
            "framework": "sglang",
            "model_path": "/models/qwen",
            "requested_server_args": "--mem-fraction-static 0.8 --context-length 8192",
        }
    )

    assert resolved["model_path"] == "/models/qwen"
    assert resolved["mem_fraction_static"] == pytest.approx(0.8)
    assert resolved["context_length"] == 8192


def test_sglang_server_args_fallback_contains_parser_system_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeServerArgs:
        @classmethod
        def add_cli_args(cls, parser):
            parser.add_argument("--tp-size", type=int)

    sglang = ModuleType("sglang")
    srt = ModuleType("sglang.srt")
    server_args = ModuleType("sglang.srt.server_args")
    server_args.ServerArgs = FakeServerArgs
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)

    assert (
        WritebackCollaborator._resolved_sglang_server_config(
            {"framework": "sglang", "requested_server_args": "--tp-size not-an-integer"}
        )
        == {}
    )


def test_empty_observed_identity_reparses_measurement_log(tmp_path: Path) -> None:
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "server_args=ServerArgs(model_path='/models/qwen', tp_size=8, context_length=8192)\n",
        encoding="utf-8",
    )
    identity = WritebackCollaborator._measurement_observed_server_identity(
        {
            "server_log_path": str(server_log),
            "launch_evidence": {"framework": "sglang", "observed_server_identity": {}},
        }
    )

    assert identity == {"context_length": 8192, "model_path": "/models/qwen", "tp_size": 8}


def test_env_spec_does_not_probe_measurement_workspace_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = _verified_current_best(tmp_path)
    workspace = tmp_path / "measurement" / "benchmark"
    workspace.mkdir(parents=True)
    (workspace.parent / "server.log").write_text(
        "+ python -m sglang.launch_server --model-path /wrong --speculative-algorithm NEXTN\n",
        encoding="utf-8",
    )
    state.current_best["measurement"] = {
        "tput": 1403.43,
        "benchmark_workspace": str(workspace),
    }
    monkeypatch.setenv("FRAMEWORK", "sglang")

    spec = _writeback(tmp_path, state).build_env_spec()

    assert spec["config"]["server_launch_flags"] == ""


def test_archived_sglang_server_args_log_yields_stable_observed_identity(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text(
        "INFO server_args=ServerArgs(model_path='/models/qwen', tp_size=8, "
        "mem_fraction_static=0.8, context_length=8192, kv_cache_dtype='fp8', "
        "attention_backend='aiter', prefill_attention_backend='fa3', "
        "decode_attention_backend='triton', disable_radix_cache=True, "
        "trust_remote_code=False)\n",
        encoding="utf-8",
    )

    identity = observed_sglang_server_identity_from_log(str(log))

    assert identity == {
        "attention_backend": "aiter",
        "context_length": 8192,
        "decode_attention_backend": "triton",
        "disable_radix_cache": True,
        "kv_cache_dtype": "fp8",
        "mem_fraction_static": 0.8,
        "model_path": "/models/qwen",
        "prefill_attention_backend": "fa3",
        "tp_size": 8,
        "trust_remote_code": False,
    }


def test_archived_sglang_server_args_after_legacy_line_cap_is_observed(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text(
        "".join(f"startup noise {index}\n" for index in range(300))
        + "server_args=ServerArgs(model_path='/models/qwen', tp_size=8)\n",
        encoding="utf-8",
    )

    assert observed_sglang_server_identity_from_log(str(log)) == {
        "model_path": "/models/qwen",
        "tp_size": 8,
    }


def test_archived_sglang_server_args_rejects_executable_log_values(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text("server_args=ServerArgs(model_path=__import__('os').getcwd())\n", encoding="utf-8")

    assert observed_sglang_server_identity_from_log(str(log)) == {}
