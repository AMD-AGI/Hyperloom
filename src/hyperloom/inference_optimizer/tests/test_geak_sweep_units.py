# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the GEAK post-optimization sweep and the sweep/kernel helpers it shares with the native concurrency sweep."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest

from hyperloom.common.visible_devices import VISIBLE_DEVICE_VARS
from hyperloom.orchestrator.actions.executors import _geak_sweep
from hyperloom.orchestrator.actions.executors._geak_sweep import sweep_via_geak
from hyperloom.orchestrator.actions.executors._grid_base import coerce_extra_envs
from hyperloom.orchestrator.actions.executors._grid_runner import VariantResult
from hyperloom.orchestrator.kernel.attempt_summary import _backend_results_dir
from hyperloom.orchestrator.kernel.conc_sweep import (
    _budget_limited_without_valid_pair,
    _point_from_variant,
)
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.coordinator_helpers import (
    _parse_server_arg_value,
    _resolve_gpu_pin,
    _resolve_handoff_gpu_ids,
    _resolve_handoff_gpu_ids_space,
)
from hyperloom.orchestrator.state.shared_state import SharedState


def _bench_script(tmp_path: Path) -> Path:
    """A ``bench_e2e.sh`` stub that records NUM_PROMPTS and reports throughput."""
    bench = tmp_path / "bench_e2e.sh"
    bench.parent.mkdir(parents=True, exist_ok=True)
    bench.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_DIR"])
summary = {
    "output_throughput_tok_s_median": 200.0,
    "ttft_ms_median": 10.0,
    "tpot_ms_median": 3.0,
    "e2el_ms_median": 50.0,
}
(out / "bench_summary.json").write_text(json.dumps(summary), encoding="utf-8")
(out / "env.json").write_text(
    json.dumps({"NUM_PROMPTS": os.environ.get("NUM_PROMPTS")}), encoding="utf-8"
)
PY
""",
        encoding="utf-8",
    )
    return bench


@pytest.mark.asyncio
async def test_sweep_via_geak_uses_validated_regimes_and_pins_num_prompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``bench_protocol`` -> fall back to the first ``validated_regimes`` entry; ``pin_num_prompts`` forwards that
    regime's NUM_PROMPTS onto the point.
    """
    bench = _bench_script(tmp_path)
    monkeypatch.setenv("MODEL_PATH", "/models/x")
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("TP", "1")

    def _fake_run(_cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        env = kwargs["env"]
        out = Path(env["OUT_DIR"])
        summary = {
            "output_throughput_tok_s_median": 200.0,
            "ttft_ms_median": 10.0,
            "tpot_ms_median": 3.0,
            "e2el_ms_median": 50.0,
        }
        (out / "bench_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (out / "env.json").write_text(json.dumps({"NUM_PROMPTS": env.get("NUM_PROMPTS")}), encoding="utf-8")
        (out / "server.log").write_text(
            "server_args=ServerArgs(model_path='/models/x', tp_size=1, context_length=4096)\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(_cmd, 0, "", "")

    monkeypatch.setattr(_geak_sweep.subprocess, "run", _fake_run)

    result = await sweep_via_geak(
        result={
            "bench_script": str(bench),
            "output_dir": str(tmp_path),
            # No ``bench_protocol`` -> validated_regimes[0] fallback.
            "validated_regimes": [
                {"num_warmups": 3, "seed": 7, "num_prompts": 64},
                {"num_warmups": 9},
            ],
            "accepted_config": {"flags": "", "env": ""},
        },
        conc_values=[1],
        isl_osl_configs=["16:16"],
        output_root=tmp_path / "sweep",
        variant_timeout_sec=30,
        repeats=1,
        pin_num_prompts=True,
    )

    assert result["status"] == "succeeded"
    out_dir = tmp_path / "sweep" / "variant_0_conc1_isl16_osl16"
    env = json.loads((out_dir / "env.json").read_text(encoding="utf-8"))
    assert env["NUM_PROMPTS"] == "64"
    evidence = result["promotion_measurement"]["launch_evidence"]
    assert result["promotion_measurement"]["server_log_path"] == str(out_dir / "server.log")
    assert evidence["actual_server_log_path"] == str(out_dir / "server.log")
    assert evidence["observed_server_identity"] == {
        "context_length": 4096,
        "model_path": "/models/x",
        "tp_size": 1,
    }


@pytest.mark.asyncio
async def test_sweep_via_geak_prefers_executable_final_launch_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2a runs the GEAK final script without changing its single-point protocol."""
    bench = _bench_script(tmp_path)
    final = tmp_path / "final_launch.sh"
    final.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    final.chmod(0o755)
    monkeypatch.setenv("MODEL_PATH", "/models/x")
    captured: dict[str, Any] = {}

    def _fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        captured["command"] = command
        captured["env"] = kwargs["env"]
        out = Path(command[2])
        (out / "bench_summary.json").write_text(
            json.dumps({"output_throughput_tok_s_median": 200.0}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(_geak_sweep.subprocess, "run", _fake_run)
    result = await sweep_via_geak(
        result={
            "bench_script": str(bench),
            "final_launch_script": str(final),
            "accepted_config": {},
        },
        conc_values=[1],
        isl_osl_configs=["16:16"],
        output_root=tmp_path / "sweep",
        variant_timeout_sec=30,
        repeats=3,
    )

    out_dir = tmp_path / "sweep" / "variant_0_conc1_isl16_osl16"
    assert result["replay_mode"] == "final_launch_script"
    assert captured["command"] == ["bash", str(final), str(out_dir)]
    assert captured["env"]["REPLICAS"] == "3"


@pytest.mark.asyncio
@pytest.mark.parametrize("requirement", ["live_tree_files", "cache_invalidation"])
@pytest.mark.parametrize("launcher_state", ["absent", "missing", "not_executable"])
async def test_tuning_cannot_fall_back_without_its_deployment(
    tmp_path: Path, requirement: str, launcher_state: str
) -> None:
    marker = tmp_path / "bench_started"
    bench = tmp_path / "bench.sh"
    bench.write_text("#!/bin/bash\ntouch " + shlex.quote(str(marker)) + "\n", encoding="utf-8")
    final = tmp_path / "final.sh"
    if launcher_state == "not_executable":
        final.write_text("#!/bin/bash\n", encoding="utf-8")
        final.chmod(0o644)
    result = await sweep_via_geak(
        result={
            "bench_script": str(bench),
            "final_launch_script": "" if launcher_state == "absent" else str(final),
            "accepted_config": {"flags": "--tp 1"},
            "tuning_skillset": {"gate": "accepted", requirement: ["required-data"]},
        },
        conc_values=[1],
        isl_osl_configs=["16:16"],
        output_root=tmp_path / "sweep",
        variant_timeout_sec=10,
    )
    assert result["status"] == "failed"
    assert result["error_class"] == "missing_deployment_launcher"
    assert not result.get("promotion_measurement")
    assert not marker.exists()


@pytest.mark.asyncio
async def test_flags_only_can_still_replay_without_a_final_launcher(tmp_path: Path) -> None:
    marker = tmp_path / "bench_started"
    bench = tmp_path / "bench.sh"
    bench.write_text("#!/bin/bash\ntouch " + shlex.quote(str(marker)) + "\n", encoding="utf-8")
    result = await sweep_via_geak(
        result={"bench_script": str(bench), "final_launch_script": str(tmp_path / "missing.sh")},
        conc_values=[1],
        isl_osl_configs=["16:16"],
        output_root=tmp_path / "sweep",
        variant_timeout_sec=10,
        repeats=1,
    )
    assert marker.is_file()
    assert result["replay_mode"] == "bench_e2e_fallback"
    assert result["status"] == "failed"
    assert not result.get("promotion_measurement")


@pytest.mark.asyncio
async def test_sweep_via_geak_marks_variant_failed_on_subprocess_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising ``subprocess.run`` is caught and recorded as a failed variant."""
    bench = _bench_script(tmp_path)
    monkeypatch.setenv("MODEL_PATH", "/models/x")

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise OSError("cannot spawn bench process")

    monkeypatch.setattr(_geak_sweep.subprocess, "run", _boom)

    result = await sweep_via_geak(
        result={"bench_script": str(bench), "output_dir": str(tmp_path), "accepted_config": {}},
        conc_values=[1],
        isl_osl_configs=["16:16"],
        output_root=tmp_path / "sweep",
        variant_timeout_sec=30,
    )

    assert result["status"] == "failed"
    entry = result["points"][0]
    assert entry["status"] == "failed"
    assert "cannot spawn bench process" in entry["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("identity_source", ["handoff", "baseline_recipe"])
@pytest.mark.parametrize("pin_var", ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"])
async def test_geak_harness_replay_uses_run_gpu_pin_and_recipe_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_source: str,
    pin_var: str,
) -> None:
    """Replay keeps the run's devices and serving identity after ambient drift."""
    for name in VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.setenv("MODEL_PATH", "/models/stale-shell-model")
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("TP", "8")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

    recipe = tmp_path / "baseline.yaml"
    recipe_env = {"TP": 2, pin_var: "4,5"}
    recipe.write_text(
        json.dumps({"benchmark": {"framework": "vllm", "model": "/models/validated", "envs": recipe_env}}),
        encoding="utf-8",
    )
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        benchmark_mode="synthetic",
        framework="vllm",
        model_path="/models/validated",
        tp=2,
        baseline_config_path=str(recipe),
        baseline_tput=100.0,
        current_best={"action": "explore", "tput": 150.0},
        isl=16,
        osl=16,
        conc=1,
    )
    pin = _resolve_gpu_pin(recipe_envs=recipe_env, environ={})
    gpu_ids = _resolve_handoff_gpu_ids(gpu_pin=pin, tp=2)
    gpu_ids_space = _resolve_handoff_gpu_ids_space(gpu_pin=pin)
    assert gpu_ids == ("0,1" if pin_var == "ROCR_VISIBLE_DEVICES" else "4,5")
    assert gpu_ids_space == ("logical" if pin_var == "ROCR_VISIBLE_DEVICES" else "absolute")
    geak_dir = tmp_path / "geak"
    bench = _bench_script(geak_dir)
    if identity_source == "handoff":
        handoff = {
            "schema_version": 3,
            "model_path": "/models/validated",
            "framework": "vllm",
            "tp": 2,
            "gpu_pin": pin,
            "gpu_ids": gpu_ids,
            "gpu_ids_space": gpu_ids_space,
            "launch_recipe": str(recipe),
            "baseline_env_spec": coord.build_env_spec(),
            "bench_client": "native",
            "workload": {"isl": 16, "osl": 16, "conc": 1},
        }
        (geak_dir / "handoff.json").write_text(json.dumps(handoff), encoding="utf-8")
    coord.shared_state.geak_result = {
        "status": "ok",
        "bench_script": str(bench),
        "output_dir": str(geak_dir),
        "throughput_speedup": 2.0,
        "bench_client": "native",
        "accepted_config": {"flags": "--max-num-batched-tokens 4096", "env": ""},
        "validated_regimes": [{"isl": 16, "osl": 16, "conc": 1}],
    }
    captured: dict[str, str] = {}

    def _fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        captured.update(kwargs["env"])
        out = Path(kwargs["env"]["OUT_DIR"])
        (out / "bench_summary.json").write_text(json.dumps({"output_throughput_tok_s_median": 200.0}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(_geak_sweep.subprocess, "run", _fake_run)
    outcome = await coord._validate_geak_via_geak_harness(reason="unit")

    assert captured["GPU"] == gpu_ids
    if gpu_ids_space == "logical":
        assert captured["ROCR_VISIBLE_DEVICES"] == "4,5"
    else:
        assert "ROCR_VISIBLE_DEVICES" not in captured
    assert (captured["MODEL"], captured["BACKEND"], captured["TP"]) == ("/models/validated", "vllm", "2")
    assert outcome["validated"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("bench_client", ["auto", "native", "inferencex"])
async def test_geak_replay_uses_existing_client_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bench_client: str,
) -> None:
    bench = _bench_script(tmp_path)
    captured: dict[str, str] = {}

    def _fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        captured.update(kwargs["env"])
        out = Path(captured["OUT_DIR"])
        (out / "bench_summary.json").write_text(json.dumps({"output_throughput_tok_s_median": 200.0}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(_geak_sweep.subprocess, "run", _fake_run)
    outcome = await sweep_via_geak(
        result={"status": "ok", "bench_script": str(bench), "bench_client": bench_client, "accepted_config": {}},
        handoff={
            "model_path": "/models/validated",
            "framework": "sglang",
            "tp": 1,
            "gpu_ids": "4",
            "gpu_ids_space": "absolute",
            "bench_client": bench_client,
        },
        conc_values=[4],
        isl_osl_configs=["16:16"],
        output_root=tmp_path / "sweep",
        variant_timeout_sec=30,
        repeats=3,
    )

    assert captured["BENCH_CLIENT"] == bench_client
    assert captured["REPEATS"] == "3"
    assert outcome["status"] == "succeeded"
    assert outcome["promotion_measurement"]["output_throughput"] == 200.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "accepted_config,expected_env,expected_flags",
    [
        (
            {"env_map": {"AITER_CONFIG": '{"path": "/tmp/a (b); c"}'}, "env": "AITER_CONFIG=wrong"},
            {"AITER_CONFIG": '{"path": "/tmp/a (b); c"}'},
            "",
        ),
        ({"env_map": {}, "env": "AITER_CONFIG=wrong --trust-remote-code"}, {}, ""),
        (
            {"env_map": {"AITER_CONFIG": "", "LD_PRELOAD": "/untrusted.so", "PYTHONPATH": "/untrusted"}},
            {"AITER_CONFIG": ""},
            "",
        ),
        ({"env": "SGLANG_USE_AITER=1 --trust-remote-code"}, {"SGLANG_USE_AITER": "1"}, "--trust-remote-code"),
    ],
)
async def test_sweep_replays_normalized_return_environment_and_fresh_accuracy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accepted_config, expected_env, expected_flags
) -> None:
    bench = _bench_script(tmp_path)
    captured = {}

    def _fake_run(command, **kwargs):
        captured.update(kwargs["env"])
        out = Path(kwargs["env"]["OUT_DIR"])
        (out / "bench_summary.json").write_text(json.dumps({"output_throughput_tok_s_median": 200.0}), encoding="utf-8")
        (out / "results.json").write_text(
            json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.8}}}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(_geak_sweep.subprocess, "run", _fake_run)
    result = await sweep_via_geak(
        result={"bench_script": str(bench), "accepted_config": accepted_config, "accuracy": 0.99},
        conc_values=[1],
        isl_osl_configs=["16:16"],
        output_root=tmp_path / "sweep",
        variant_timeout_sec=30,
    )

    assert dict(token.split("=", 1) for token in shlex.split(captured["EXTRA_ENV"])) == expected_env
    assert captured["EXTRA_SERVER_ARGS"] == expected_flags
    point = result["promotion_measurement"]
    assert point["accuracy"] == 0.8
    assert point["accuracy_source"] == str(Path(point["workspace"]) / "results.json")
    assert point["launch_evidence"]["requested_server_env"] == expected_env


@pytest.mark.asyncio
@pytest.mark.parametrize("env_map", [None, [], {"AITER_CONFIG": 1}, {1: "value"}])
async def test_sweep_rejects_malformed_structured_environment_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env_map
) -> None:
    bench = _bench_script(tmp_path)

    def _must_not_launch(*_args, **_kwargs):
        pytest.fail("malformed structured environment must not launch a replay")

    monkeypatch.setattr(_geak_sweep.subprocess, "run", _must_not_launch)
    result = await sweep_via_geak(
        result={"bench_script": str(bench), "accepted_config": {"env_map": env_map, "env": "VALID=1"}},
        conc_values=[1],
        isl_osl_configs=["16:16"],
        output_root=tmp_path / "sweep",
        variant_timeout_sec=30,
    )
    assert result["status"] == "failed"
    assert result["error_class"] == "invalid_accepted_config"


@pytest.mark.asyncio
@pytest.mark.parametrize("native_accuracy", [0.8, 0.2, None])
async def test_sweep_reads_only_fresh_native_quality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_accuracy
) -> None:
    bench = _bench_script(tmp_path)
    (tmp_path / "results_stale.json").write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.99}}}), encoding="utf-8"
    )

    def _fake_run(command, **kwargs):
        out = Path(kwargs["env"]["OUT_DIR"])
        (out / "bench_summary.json").write_text(
            json.dumps({"output_throughput_tok_s_median": 200.0, "accuracy": 0.99}), encoding="utf-8"
        )
        if native_accuracy is not None:
            (out / "results.json").write_text(
                json.dumps({"results": {"gsm8k": {"exact_match,strict-match": native_accuracy}}}), encoding="utf-8"
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(_geak_sweep.subprocess, "run", _fake_run)
    result = await sweep_via_geak(
        result={"bench_script": str(bench), "accuracy": 0.99, "eval_dir": str(tmp_path)},
        conc_values=[1],
        isl_osl_configs=["16:16"],
        output_root=tmp_path / "fresh_sweep",
        variant_timeout_sec=30,
    )
    assert result["promotion_measurement"]["accuracy"] == native_accuracy


def test_point_from_variant_defaults_conc_zero_on_bad_env() -> None:
    """A non-numeric CONC env coerces the row's ``conc`` to 0 rather than raising."""
    v = VariantResult(
        name="optimized_concX",
        extra_server_args="",
        extra_envs={"CONC": "not-a-number"},
        status="failed",
    )
    point = _point_from_variant(v, arm="optimized")
    assert point["conc"] == 0
    assert point["arm"] == "optimized"
    assert point["status"] == "failed"


def test_budget_limited_without_valid_pair_paths() -> None:
    """Empty-points guard, a genuine failure short-circuit, and the budget-skip case."""
    summary_no_pairs = {"successful_pairs": 0}
    # No points at all -> not attributable to budget gating.
    assert (
        _budget_limited_without_valid_pair(
            budget_exhausted=True,
            summary=summary_no_pairs,
            baseline_points=[],
            optimized_points=[],
        )
        is False
    )
    # A genuinely-failed (non budget) point -> not budget gating.
    assert (
        _budget_limited_without_valid_pair(
            budget_exhausted=True,
            summary=summary_no_pairs,
            baseline_points=[{"status": "failed", "error_class": "crash"}],
            optimized_points=[],
        )
        is False
    )
    # Every remaining point was a budget skip -> budget-limited.
    assert (
        _budget_limited_without_valid_pair(
            budget_exhausted=True,
            summary=summary_no_pairs,
            baseline_points=[{"status": "skipped", "error_class": "budget_exhausted"}],
            optimized_points=[],
        )
        is True
    )


def test_backend_results_dir_keyed_and_single_subdir(tmp_path: Path) -> None:
    """Resolve the results dir by session key, and via the lone-subdir fallback."""
    from hyperloom.inference_optimizer.session.session_paths import (
        kernel_agent_runs_root,
    )

    # Keyed by ``session_dir.name``.
    sd = tmp_path / "sess-A"
    runs = kernel_agent_runs_root(sd)
    (runs / sd.name / "results").mkdir(parents=True)
    assert _backend_results_dir(sd, "") == runs / sd.name / "results"

    # Migrated-key recovery: a single subdir under the runs root.
    sd2 = tmp_path / "sess-B"
    runs2 = kernel_agent_runs_root(sd2)
    (runs2 / "migrated-key" / "results").mkdir(parents=True)
    assert _backend_results_dir(sd2, "") == runs2 / "migrated-key" / "results"

    # Ambiguous (no keyed match, not exactly one subdir) -> None.
    sd3 = tmp_path / "sess-C"
    runs3 = kernel_agent_runs_root(sd3)
    (runs3 / "one").mkdir(parents=True)
    (runs3 / "two").mkdir(parents=True)
    assert _backend_results_dir(sd3, "") is None


def test_coerce_extra_envs_skips_malformed_tokens() -> None:
    """The GEAK/sweep env coercion drops empty tokens and empty keys in both the shell-string and token-list shapes rather than emitting junk keys."""
    # Shell-string shape: leading separator -> empty token; ``=v`` -> empty key.
    assert coerce_extra_envs("; =v FOO=1") == {"FOO": "1"}
    # Token-list shape: dict item with a None key, a token without ``=``, a non-string item, and an empty-key ``=v``
    # are all skipped.
    assert coerce_extra_envs([{None: "x", "A": "1"}, "noeq", 123, "=v", "B=2"]) == {"A": "1", "B": "2"}


def test_parse_server_arg_value_falls_back_on_unbalanced_quotes() -> None:
    """The GEAK handoff recovers a flag value even when the server-args string is not shlex-parseable (unbalanced quote -> plain ``str.split`` fallback)."""
    got = _parse_server_arg_value('--max-model-len 4096 "unbalanced', "--max-model-len")
    assert got == "4096"
    # ``--flag=value`` form is also handled.
    assert _parse_server_arg_value("--gpu-memory-utilization=0.9", "--gpu-memory-utilization") == "0.9"
    # Absent flag -> None.
    assert _parse_server_arg_value("--tp 8", "--max-model-len") is None
