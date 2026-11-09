# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK replays the AgentX workload Hyperloom declares; Hyperloom owns canonical AgentX validation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
import yaml

from hyperloom.orchestrator.actions.executors._server_lifecycle import GpuHandoff
from hyperloom.orchestrator.actions.executors._workload_envs import geak_acceptance
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.shared_state import SharedState

from ._geak_helpers import stop_geak_before_launch


@pytest.fixture(autouse=True)
def _isolate_workload_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tuple(os.environ):
        if key.startswith(("AGENTX_", "HYPERLOOM_PROFILE_")) or key in {
            "AIPERF_BIN",
            "HYPERLOOM_AGENTX",
            "HYPERLOOM_PERF_METRIC",
            "E2E_METRIC",
            "FRAMEWORK",
            "MODEL_PATH",
            "GPU_TYPE",
            "INFERENCEX_PATH",
            "INFERENCE_OPTIMIZER_EXTRA_ENV",
            "INFERENCE_OPTIMIZER_SERVER_ARGS",
            "CONC",
            "ISL",
            "OSL",
            "NUM_PROMPTS",
            "NUM_WARMUPS",
            "RANDOM_RANGE_RATIO",
            "SEED",
            "TP",
            "MAX_MODEL_LEN",
            "GPU_MEMORY_UTILIZATION",
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        }:
            monkeypatch.delenv(key, raising=False)


#: GEAK's ``E2E_METRIC`` value -> the ``metric_basis`` that must ride with it.
_BASIS_OF = {
    "output": "aggregate_output_tok_s",
    "e2e_norm_intvty_p50": "e2e_norm_intvty_p50",
}

#: A corpus other than GEAK's default, so a declared one can only have come from the round.
_ROUND_CORPUS = "semianalysis_cc_traces_weka_062126_256k"


def _accepted_round_report(workspace: Path) -> None:
    """Leave the validated report Magpie writes for the accepted round in ``workspace``."""
    workspace.mkdir(parents=True, exist_ok=True)
    report = {
        "scenario": "agentx",
        "success": True,
        "benchmark_valid": True,
        "publishable": True,
        "agentx_metrics": {
            "execution_owner": "magpie",
            "server_launch": {"effective_argv": ["python", "-m", "sglang.launch_server"]},
            "dataset": {"loader": _ROUND_CORPUS, "num_dataset_entries": 512},
            "requests": {"successful": 600},
            "throughput": {
                "input_tokens_per_second": 30000.0,
                "output_tokens_per_second": 120.0,
                "duration_seconds": 3600.0,
            },
        },
    }
    (workspace / "benchmark_report.json").write_text(json.dumps(report), encoding="utf-8")


def _round_replay_client(workspace: Path, *, interpreter: str) -> Path:
    """Leave the aiperf venv InferenceX's AgentX client builds in the round's workspace."""
    bin_dir = workspace / ".agentx-runtime" / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(interpreter)
    client = bin_dir / "aiperf"
    client.write_text("#!/bin/sh\n", encoding="utf-8")
    client.chmod(0o755)
    return client


def _only_on_path(monkeypatch: pytest.MonkeyPatch, bin_dir: Path, *names: str) -> None:
    """Make ``bin_dir`` the whole ``PATH``, holding an executable stub for each of ``names``."""
    bin_dir.mkdir()
    for name in names:
        stub = bin_dir / name
        stub.write_text("#!/bin/sh\n", encoding="utf-8")
        stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))


def _coord(
    tmp_path: Path, *, framework: str = "sglang", agentx: bool = True, metric: str = "e2e_norm_intvty_p50"
) -> Coordinator:
    benchmarks = tmp_path / "InferenceX" / "benchmarks"
    benchmarks.mkdir(parents=True)
    for name in ("benchmark_lib.sh", f"{framework}_mi355x.sh"):
        (benchmarks / name).write_text("# stub\n", encoding="utf-8")
    recipe = tmp_path / "baseline.yaml"
    recipe.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": framework,
                    "model": "/models/accepted",
                    "runner_type": "mi355x",
                    "inferencex_path": str(benchmarks.parent),
                    **({"agentx": {"enabled": True}} if agentx else {"benchmark_script": f"{framework}_mi355x.sh"}),
                    "envs": {
                        "CONC": 6,
                        "ISL": 2048,
                        "OSL": 1536,
                        "NUM_PROMPTS": 73,
                        "NUM_WARMUPS": 3,
                        "RANDOM_RANGE_RATIO": 0.25,
                        "SEED": 41,
                        "TP": 2,
                        "ROCR_VISIBLE_DEVICES": "6,7",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        baseline_config_path=str(recipe),
        benchmark_mode="agentx" if agentx else "synthetic",
        agentx_backend="magpie" if agentx else "",
        framework=framework,
        model_path="/models/accepted",
        gpu_type="mi355x",
        tp=2,
        isl=2048,
        osl=1536,
        conc=6,
        baseline_tput=100.0,
        current_best={
            "action": "explore",
            "tput": 140.0,
            "extra_server_args": "--accepted-flag 1",
            "extra_envs": {"ACCEPTED_SETTING": "1"},
            "remove_args": ["--obsolete-flag"],
            "unset_envs": ["OBSOLETE_SETTING"],
            "args_mode": "replace",
            "optimization_stack": [],
        },
        current_best_measurement={
            "tput": 140.0,
            "resolved_server_launch_flags": "--accepted-flag 1",
            "benchmark_workspace": str(tmp_path / "accepted_run"),
        },
    )
    coord.shared_state.current_best_measurement["declared_launch_identity"] = coord.writeback.build_env_spec()[
        "launch_identity"
    ]
    if agentx:
        prefix = (
            ["python", "-m", "sglang.launch_server"]
            if framework == "sglang"
            else ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", "/models/accepted"]
        )
        coord.shared_state.current_best_measurement["launch_evidence"] = {
            "magpie_server_launch": {
                "effective_argv": prefix + ["--tensor-parallel-size", "2", "--accepted-flag", "1"],
                "runtime_environment": {"ACCEPTED_SETTING": "1", "HIP_VISIBLE_DEVICES": "6,7"},
            }
        }
        _accepted_round_report(tmp_path / "accepted_run")
    coord.phase_kernel._record_geak_kernel_journey = lambda _result: None
    return coord


async def _handoff(coord: Coordinator, monkeypatch: pytest.MonkeyPatch) -> dict:
    stop_geak_before_launch(monkeypatch)
    await coord.phase_kernel._run_geak_kernel_phase(from_phase="KERNEL")
    return json.loads((coord.session_dir / "geak" / "handoff.json").read_text(encoding="utf-8"))


_GPUS_FREE = GpuHandoff(reaped=[], busy=[], holders=[], unverified="")


def _gpus_at_handoff(
    monkeypatch: pytest.MonkeyPatch, answer: GpuHandoff | Exception
) -> list[tuple[Path, list[int] | None]]:
    """Answer the hand-off's GPU check with ``answer``; return the (session dir, cards) it was asked about."""
    asked: list[tuple[Path, list[int] | None]] = []

    def _check(session_dir: Path, cards: list[int] | None, *, timeout_sec: float) -> GpuHandoff:
        asked.append((Path(session_dir), cards))
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._server_lifecycle.free_gpus_for_handoff", _check)
    return asked


def _launch_mock_runner(coord: Coordinator, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict]:
    """Let the phase reach the runner launch; return the env of each launch."""
    coord.phase_kernel._geak_timeouts = lambda: (60, 90, False)
    launches: list[dict] = []

    def _start_runner(_cmd, *, env, **_kwargs):
        launches.append(dict(env))
        process = Mock(returncode=0)
        process.communicate.return_value = ("", "")
        return process

    monkeypatch.setattr("hyperloom.orchestrator.phases.kernel_geak_dispatch.subprocess.Popen", _start_runner)
    return launches


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
@pytest.mark.asyncio
async def test_native_handoff_keeps_supported_schema_and_seeds_geaks_own_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, framework: str
) -> None:
    coord = _coord(tmp_path, framework=framework)
    monkeypatch.setenv("FRAMEWORK", "vllm" if framework == "sglang" else "sglang")
    monkeypatch.setenv("MODEL_PATH", "/models/wrong")
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    monkeypatch.setenv("TP", "8")
    monkeypatch.setenv("CONC", "99")

    handoff = await _handoff(coord, monkeypatch)

    assert handoff["schema_version"] == 3
    assert handoff["bench_client"] == "auto"
    assert handoff["bench_launcher"] == "native"
    assert not {"bench_client_config", "benchmark_mode", "workload_identity", "launch_server_script"} & handoff.keys()
    assert handoff["workload_spec"] == {
        "kind": "agentx_trace_replay",
        "client": "aiperf",
        "corpus": _ROUND_CORPUS,
        "canonical_corpus": _ROUND_CORPUS,
        "num_entries": 512,
        "canonical_num_entries": 512,
        "observed_isl": 180000,
        "observed_osl": 720,
        "observed_requests": 600,
        "concurrency": 6,
        "metric_basis": "e2e_norm_intvty_p50",
        "intvty_p90_veto_pct": 5.0,
        "acceptance": geak_acceptance("e2e_norm_intvty_p50", 5.0),
    }
    assert handoff["e2e_metric"] == "e2e_norm_intvty_p50"
    assert handoff["framework"] == framework
    assert handoff["model_path"] == "/models/accepted"
    assert handoff["gpu_type"] == "mi355x"
    assert handoff["launch_recipe"] == coord.shared_state.baseline_config_path
    assert handoff["workload"] == {"isl": 2048, "osl": 1536, "conc": 6}
    assert handoff["gpu_pin"]["var"] == "ROCR_VISIBLE_DEVICES"
    assert handoff["gpu_pin"]["ids"] == [6, 7]
    assert handoff["gpu_ids"] == "0,1"
    assert handoff["gpu_ids_space"] == "logical"
    assert handoff["tp"] == 2
    assert handoff["accepted_flags"] == ""
    assert handoff["baseline_env_spec"]["config"]["server_launch_flags"] == "--accepted-flag 1"
    assert "ACCEPTED_SETTING=1" in handoff["accepted_env"]
    assert handoff["same_config_reference_status"] == "unverified"
    assert handoff["same_config_reference_verification_status"] == "unverified_workload"
    assert handoff["orchestrator_best_tput_same_config"] == 0.0
    assert handoff["raw_baseline_tput"] == 0.0
    spec = handoff["baseline_env_spec"]
    assert spec["launch_identity"] == handoff["same_config_reference_identity"]
    assert spec["config"]["remove_args"] == ["--obsolete-flag"]
    assert spec["config"]["unset_envs"] == ["OBSOLETE_SETTING"]
    assert spec["config"]["args_mode"] == "replace"


@pytest.mark.asyncio
async def test_native_handoff_uses_verified_launch_without_searching_server_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coord = _coord(tmp_path)
    receipt = coord.shared_state.current_best_measurement["launch_evidence"]["magpie_server_launch"]
    receipt["effective_argv"] += ["--mem-fraction-static", "0.95", "--trust-remote-code"]
    receipt["runtime_environment"]["AITER_SETTING"] = "enabled"
    handoff = await _handoff(coord, monkeypatch)
    config = handoff["baseline_env_spec"]["config"]
    assert config["server_launch_flags"] == "--accepted-flag 1 --mem-fraction-static 0.95 --trust-remote-code"
    assert config["extra_envs"] == {"ACCEPTED_SETTING": "1", "AITER_SETTING": "enabled"}
    assert handoff["native_server_launch"] == receipt
    assert handoff["bench_launcher"] == "native"


@pytest.mark.asyncio
async def test_native_handoff_does_not_depend_on_inferencex_shell_launchers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coord = _coord(tmp_path)
    (tmp_path / "InferenceX" / "benchmarks" / "sglang_mi355x.sh").unlink()
    handoff = await _handoff(coord, monkeypatch)
    assert "launch_server_script" not in handoff
    assert handoff["bench_launcher"] == "native"


@pytest.mark.asyncio
async def test_native_handoff_without_the_rounds_report_still_declares_agentx_and_leaves_the_corpus_to_geak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    coord = _coord(tmp_path)
    (tmp_path / "accepted_run" / "benchmark_report.json").unlink()

    with caplog.at_level("WARNING", logger="hyperloom.orchestrator.phases.kernel_geak_dispatch"):
        handoff = await _handoff(coord, monkeypatch)

    spec = handoff["workload_spec"]
    assert spec["kind"] == "agentx_trace_replay"
    assert (spec["concurrency"], spec["metric_basis"]) == (6, "e2e_norm_intvty_p50")
    replayed_trace_keys = {
        "corpus",
        "canonical_corpus",
        "num_entries",
        "canonical_num_entries",
        "observed_isl",
        "observed_osl",
    }
    assert not replayed_trace_keys & spec.keys()
    assert "GEAK replays its canonical corpus" in caplog.text


@pytest.mark.parametrize("interpreter_runs", [True, False], ids=["runs-here", "interpreter-gone-after-resume"])
@pytest.mark.asyncio
async def test_geak_gets_the_aiperf_the_accepted_round_replayed_with_only_when_it_runs_here(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interpreter_runs: bool
) -> None:
    coord = _coord(tmp_path)
    interpreter = sys.executable if interpreter_runs else str(tmp_path / "gone" / "python3.11")
    client = _round_replay_client(tmp_path / "accepted_run", interpreter=interpreter)
    # GEAK falls back to the aiperf on PATH, so a round client that cannot run still lets it launch.
    _only_on_path(monkeypatch, tmp_path / "bin", "aiperf")
    launches = _launch_mock_runner(coord, monkeypatch, tmp_path)
    _gpus_at_handoff(monkeypatch, _GPUS_FREE)

    await coord.phase_kernel._run_geak_kernel_phase(from_phase="KERNEL")

    [env] = launches
    assert env.get("AIPERF_BIN") == (str(client) if interpreter_runs else None)


@pytest.mark.parametrize("interpreter_runs", [True, False], ids=["round-client-runs", "round-client-gone"])
@pytest.mark.asyncio
async def test_an_operators_aiperf_bin_outranks_the_rounds_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interpreter_runs: bool
) -> None:
    coord = _coord(tmp_path)
    interpreter = sys.executable if interpreter_runs else str(tmp_path / "gone" / "python3.11")
    _round_replay_client(tmp_path / "accepted_run", interpreter=interpreter)
    _only_on_path(monkeypatch, tmp_path / "bin")
    monkeypatch.setenv("AIPERF_BIN", "/opt/operator/aiperf")
    launches = _launch_mock_runner(coord, monkeypatch, tmp_path)
    _gpus_at_handoff(monkeypatch, _GPUS_FREE)

    await coord.phase_kernel._run_geak_kernel_phase(from_phase="KERNEL")

    [env] = launches
    assert env["AIPERF_BIN"] == "/opt/operator/aiperf"


@pytest.mark.asyncio
async def test_geak_is_skipped_with_a_reason_when_no_aiperf_can_run_here(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coord = _coord(tmp_path)
    _round_replay_client(tmp_path / "accepted_run", interpreter=str(tmp_path / "gone" / "python3.11"))
    _only_on_path(monkeypatch, tmp_path / "bin")
    launches = _launch_mock_runner(coord, monkeypatch, tmp_path)
    _gpus_at_handoff(monkeypatch, _GPUS_FREE)

    await coord.phase_kernel._run_geak_kernel_phase(from_phase="KERNEL")

    assert launches == []
    result = coord.shared_state.geak_result
    assert (result["status"], result["error_class"]) == ("skipped", "agentx_client_unavailable")
    assert str(tmp_path / "accepted_run") in result["error"]


@pytest.mark.asyncio
async def test_a_synthetic_handoff_launches_geak_without_any_aiperf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coord = _coord(tmp_path, agentx=False)
    _only_on_path(monkeypatch, tmp_path / "bin")
    launches = _launch_mock_runner(coord, monkeypatch, tmp_path)
    _gpus_at_handoff(monkeypatch, _GPUS_FREE)

    await coord.phase_kernel._run_geak_kernel_phase(from_phase="KERNEL")

    assert len(launches) == 1


@pytest.mark.parametrize(
    ("metric_override", "expected_metric"),
    [
        (None, "e2e_norm_intvty_p50"),
        ("intvty_v1", "e2e_norm_intvty_p50"),
        ("composite_v1", "output"),
        ("output", "output"),
    ],
)
@pytest.mark.asyncio
async def test_agentx_geak_metric_aligned_result_is_only_a_proposal_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    metric_override: str | None,
    expected_metric: str,
) -> None:
    if metric_override is not None:
        monkeypatch.setenv("HYPERLOOM_PERF_METRIC", metric_override)
    coord = _coord(tmp_path, metric=expected_metric)
    _round_replay_client(tmp_path / "accepted_run", interpreter=sys.executable)
    # Seed a DIFFERENT axis so the assertion below proves the handoff overrides the
    # inherited env rather than happening to agree with it.
    monkeypatch.setenv("E2E_METRIC", "total")
    launches = _launch_mock_runner(coord, monkeypatch, tmp_path)
    _gpus_at_handoff(monkeypatch, _GPUS_FREE)
    with caplog.at_level("INFO", logger="hyperloom.orchestrator.phases.kernel_geak_dispatch"):
        await coord.phase_kernel._run_geak_kernel_phase(from_phase="KERNEL")

    handoff = json.loads((tmp_path / "geak" / "handoff.json").read_text(encoding="utf-8"))
    [captured_env] = launches
    assert (handoff["e2e_metric"], captured_env["E2E_METRIC"]) == (expected_metric, expected_metric)
    assert handoff["workload_spec"]["metric_basis"] == _BASIS_OF[expected_metric]
    assert handoff["workload_spec"].get("acceptance") == geak_acceptance(_BASIS_OF[expected_metric], 5.0)
    assert handoff["same_config_reference_status"] == "unverified"
    assert handoff["same_config_reference_verification_status"] == "unverified_workload"
    assert handoff["orchestrator_best_tput_same_config"] == 0.0
    assert handoff["raw_baseline_tput"] == 0.0
    assert "canonical AgentX validation remains in Hyperloom" in caplog.text
    assert coord.shared_state.benchmark_mode == "agentx"
    assert coord.shared_state.current_best["tput"] == 140.0
    assert coord.shared_state.geak_result["error_class"] == "no_result_json"


@pytest.mark.asyncio
async def test_gpus_a_leftover_server_still_holds_are_not_handed_to_geak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coord = _coord(tmp_path)
    launches = _launch_mock_runner(coord, monkeypatch, tmp_path)
    holder = "pgid=274559 python3 -m sglang.launch_server"
    asked = _gpus_at_handoff(
        monkeypatch,
        GpuHandoff(reaped=[274559], busy=[(6, 281000.0, 288000.0)], holders=[holder], unverified=""),
    )

    await coord.phase_kernel._run_geak_kernel_phase(from_phase="KERNEL")

    # The recipe pins ROCR_VISIBLE_DEVICES=6,7, so GEAK's logical 0,1 are physical cards 6 and 7.
    assert asked == [(tmp_path, [6, 7])]
    assert launches == []
    result = coord.shared_state.geak_result
    assert (result["status"], result["error_class"]) == ("error", "gpus_busy_at_handoff")
    assert "GPU6 281000/288000 MiB" in result["error"]
    assert holder in result["error"]


@pytest.mark.asyncio
async def test_gpus_that_cannot_be_checked_are_still_handed_to_geak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    coord = _coord(tmp_path)
    _round_replay_client(tmp_path / "accepted_run", interpreter=sys.executable)
    launches = _launch_mock_runner(coord, monkeypatch, tmp_path)
    _gpus_at_handoff(
        monkeypatch, GpuHandoff(reaped=[], busy=[], holders=[], unverified="rocm-smi could not report VRAM")
    )

    with caplog.at_level("WARNING", logger="hyperloom.orchestrator.phases.kernel_geak_dispatch"):
        await coord.phase_kernel._run_geak_kernel_phase(from_phase="KERNEL")

    assert len(launches) == 1
    assert coord.shared_state.geak_result["error_class"] == "no_result_json"
    assert "cannot confirm GPUs [6, 7] are free" in caplog.text


def _geak_returns_a_win(coord: Coordinator, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Let the runner leave an ``ok`` result.json."""
    coord.phase_kernel._geak_timeouts = lambda: (60, 90, False)

    def _start_runner(_cmd, **_kwargs):
        (tmp_path / "geak" / "result.json").write_text(
            json.dumps({"status": "ok", "throughput_speedup": 1.39, "accepted_config": {"flags": "", "env": ""}}),
            encoding="utf-8",
        )
        process = Mock(returncode=0)
        process.communicate.return_value = ("", "")
        return process

    monkeypatch.setattr("hyperloom.orchestrator.phases.kernel_geak_dispatch.subprocess.Popen", _start_runner)
    coord.phase_kernel._record_geak_candidate = lambda _result: True
    coord.bus = Mock(append_and_seq=AsyncMock())


def _gpu_checks(monkeypatch: pytest.MonkeyPatch, answers: list[GpuHandoff]) -> list[list[int] | None]:
    """Answer successive GPU checks from ``answers``; return the cards each one was asked about."""
    asked: list[list[int] | None] = []

    def _check(_session_dir: Path, cards: list[int] | None, *, timeout_sec: float) -> GpuHandoff:
        asked.append(cards)
        return answers[min(len(asked), len(answers)) - 1]

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._server_lifecycle.free_gpus_for_handoff", _check)
    return asked


@pytest.mark.asyncio
async def test_the_rebench_gets_the_gpus_geak_hands_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    coord = _coord(tmp_path, agentx=False)
    _geak_returns_a_win(coord, monkeypatch, tmp_path)
    asked = _gpu_checks(monkeypatch, [_GPUS_FREE])
    rebenches: list[tuple[str, list[int] | None]] = []

    async def _rebench(*, reason: str, handoff_cards: list[int] | None) -> None:
        rebenches.append((reason, handoff_cards))

    coord.phase_kernel._revalidate_geak_candidate = _rebench

    await coord.phase_kernel._run_geak_kernel_phase(from_phase="KERNEL")

    assert asked == [[6, 7]]
    assert rebenches == [("geak_e2e_win", [6, 7])]


@pytest.mark.asyncio
async def test_a_measured_rebench_waits_for_its_gpus_and_runs_even_when_they_stay_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    coord = _coord(tmp_path, agentx=False)
    holder = "pgid=8812 python3 -m sglang.launch_server"
    asked = _gpu_checks(
        monkeypatch, [GpuHandoff(reaped=[], busy=[(6, 120000.0, 288000.0)], holders=[holder], unverified="")]
    )
    coord.writeback.geak_rebench_params = lambda *, reason: {"skipped": True, "reason": "geak_overlay_unloadable"}
    replays: list[tuple[str, int]] = []

    async def _replay(*, decline_reason: str) -> None:
        replays.append((decline_reason, len(asked)))

    coord.phase_kernel._revalidate_on_geak_harness = _replay

    with caplog.at_level("ERROR", logger="hyperloom.orchestrator.phases.kernel_geak_dispatch"):
        await coord.phase_kernel._revalidate_geak_candidate(reason="geak_e2e_win", handoff_cards=[6, 7])

    assert asked == [[6, 7]]
    assert replays == [("geak_overlay_unloadable", 1)]
    assert "geak rebench: GPUs still hold VRAM after GEAK returned: GPU6 120000/288000 MiB" in caplog.text
    assert holder in caplog.text


@pytest.mark.asyncio
async def test_a_rebench_settled_without_measuring_does_not_wait_for_gpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coord = _coord(tmp_path, agentx=False)
    asked = _gpu_checks(monkeypatch, [_GPUS_FREE])
    coord.writeback.geak_rebench_params = lambda *, reason: {"skipped": True, "reason": "geak_no_material"}
    settled: list[bool] = []
    coord.writeback.settle_geak_no_material = lambda: settled.append(True)

    await coord.phase_kernel._revalidate_geak_candidate(reason="geak_e2e_win", handoff_cards=[6, 7])

    assert settled == [True]
    assert asked == []


@pytest.mark.asyncio
async def test_a_defect_in_the_gpu_check_raises_instead_of_handing_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coord = _coord(tmp_path)
    launches = _launch_mock_runner(coord, monkeypatch, tmp_path)
    _gpus_at_handoff(monkeypatch, TypeError("defect"))

    with pytest.raises(TypeError, match="defect"):
        await coord.phase_kernel._run_geak_kernel_phase(from_phase="KERNEL")

    assert launches == []


@pytest.mark.parametrize(
    ("metric_override", "expected_metric"),
    [(None, "output"), ("intvty_v1", "e2e_norm_intvty_p50"), ("composite_v1", "output"), ("output", "output")],
)
@pytest.mark.asyncio
async def test_synthetic_handoff_keeps_existing_protocol_and_metric_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metric_override: str | None, expected_metric: str
) -> None:
    coord = _coord(tmp_path, agentx=False)
    if metric_override is not None:
        monkeypatch.setenv("HYPERLOOM_PERF_METRIC", metric_override)

    handoff = await _handoff(coord, monkeypatch)

    assert handoff["bench_client"] == "auto"
    assert "bench_client_config" not in handoff
    assert "bench_launcher" not in handoff
    assert "launch_server_script" not in handoff
    assert "workload_spec" not in handoff
    assert handoff["e2e_metric"] == expected_metric
    assert handoff["same_config_reference_status"] == "verified"
    assert handoff["same_config_reference_verification_status"] == "verified_observed"
    assert handoff["orchestrator_best_tput_same_config"] == pytest.approx(140.0)
    assert handoff["raw_baseline_tput"] == pytest.approx(100.0)
    assert handoff["bench_protocol"] == {
        "random_range_ratio": 0.25,
        "num_prompts": 73,
        "num_warmups": 3,
        "seed": 41,
    }
