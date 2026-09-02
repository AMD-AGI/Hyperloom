# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Baseline parameter override tests."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from hyperloom.orchestrator.actions.executors._grid_runner import (
    sanitize_result_dir,
    sanitize_script_name,
)
from hyperloom.orchestrator.actions.executors._workload_envs import (
    materialize_config_with_envs,
)
from hyperloom.orchestrator.actions.executors.baseline import (
    BaselineExecutor,
)
from hyperloom.orchestrator.state.shared_state import SharedState


_CLI_STUB = SimpleNamespace(
    _load_model_max_position_embeddings=lambda _model: 32768,
    _model_has_dual_chunk_attention=lambda _model: False,
    _model_is_moe=lambda _model: False,
    _model_moe_runner_requires_aiter=lambda _model: False,
    _resolve_amd_gpu_type=lambda gpu: str(gpu or "").lower(),
    # Supported, i.e. leave the server args alone: these tests assert what the
    # trust/tokenizer path writes, not what the aiter MoE shape gate strips.
    model_supports_aiter_ck_fused_moe=lambda _model, _tp: True,
)


@pytest.fixture(autouse=True)
def _isolate_leak_root(tmp_path_factory, monkeypatch):
    """Pin ``INFERENCE_OPTIMIZER_LEAK_ROOTS`` to an empty sandbox so the artifact harvest does not scrape the host's ``/workspace``."""
    sandbox = tmp_path_factory.mktemp("isolated_leak_root")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(sandbox))


# Sanitization
def test_sanitize_script_name_accepts_bare_filename():
    assert sanitize_script_name("sglang_mi300x.sh") == "sglang_mi300x.sh"
    assert sanitize_script_name(" sglang_mi300x.sh ") == "sglang_mi300x.sh"
    assert sanitize_script_name("DSR1_FP8.sh") == "DSR1_FP8.sh"
    assert sanitize_script_name("a-b_c.0.sh") == "a-b_c.0.sh"


def test_sanitize_script_name_empty_returns_none():
    assert sanitize_script_name(None) is None
    assert sanitize_script_name("") is None
    assert sanitize_script_name("   ") is None


@pytest.mark.parametrize(
    "bad",
    [
        "../etc/passwd.sh",
        "scripts/sglang.sh",
        "no_extension",
        "with space.sh",
        "trailing.SH",  # case-sensitive *.sh
        "../sglang_mi300x.sh",
        "sglang_mi300x.sh; rm -rf /",
        "$(evil).sh",
    ],
)
def test_sanitize_script_name_rejects_unsafe(bad):
    with pytest.raises(ValueError):
        sanitize_script_name(bad)


def test_sanitize_result_dir_accepts_paths():
    assert sanitize_result_dir("/workspace/hyperloom/runs/baseline/t1") == ("/workspace/hyperloom/runs/baseline/t1")
    assert sanitize_result_dir("runs/baseline/t1") == "runs/baseline/t1"
    assert sanitize_result_dir(" /tmp/leak ") == "/tmp/leak"


def test_sanitize_result_dir_empty_returns_none():
    assert sanitize_result_dir(None) is None
    assert sanitize_result_dir("") is None
    assert sanitize_result_dir("   ") is None


@pytest.mark.parametrize(
    "bad",
    [
        "/tmp/with space",
        "/tmp/leak;rm -rf /",
        "/tmp/$(evil)",
        "/tmp/leak`whoami`",
        "/tmp/leak\nrm",
        "/tmp/leak|rm",
        "/tmp/leak&rm",
        "/tmp/leak<other",
    ],
)
def test_sanitize_result_dir_rejects_unsafe(bad):
    with pytest.raises(ValueError):
        sanitize_result_dir(bad)


# materialize_config_with_envs honors benchmark_script after gpu_type pop
def _write_yaml(
    path: Path,
    *,
    benchmark_script: str | None = None,
    framework: str = "sglang",
    model: str = "/path/models/Qwen-Qwen3-8B",
) -> None:
    cfg: dict = {
        "benchmark": {
            "framework": framework,
            "model": model,
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 1, "CONC": 8, "ISL": 256, "OSL": 256},
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        }
    }
    if benchmark_script is not None:
        cfg["benchmark"]["benchmark_script"] = benchmark_script
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def test_materialize_config_with_envs_pins_benchmark_script_after_gpu_pop(
    tmp_path,
):
    base = tmp_path / "base.yaml"
    _write_yaml(base, benchmark_script="dsr1_fp8_mi300x.sh")
    out = tmp_path / "out"
    out.mkdir()

    materialized = materialize_config_with_envs(
        base,
        out,
        model_path="/path/models/DeepSeek-R1",
        gpu_type="mi300x",
        benchmark_script="sglang_mi300x.sh",
    )
    cfg = yaml.safe_load(materialized.read_text())
    # Override re-pins benchmark_script after the gpu_type pop.
    assert cfg["benchmark"]["benchmark_script"] == "sglang_mi300x.sh"
    assert cfg["benchmark"]["runner_type"] == "mi300x"


def test_materialize_config_with_envs_forces_generic_without_override(tmp_path):
    """Without an override, gpu_type force-pins the generic ``{framework}_{gpu_type}.sh`` so Magpie never falls through to the InferenceX native script."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, benchmark_script="dsr1_fp8_mi300x.sh")
    out = tmp_path / "out"
    out.mkdir()

    materialized = materialize_config_with_envs(
        base,
        out,
        gpu_type="mi300x",
    )
    cfg = yaml.safe_load(materialized.read_text())
    # framework in the _write_yaml fixture is "sglang".
    assert cfg["benchmark"]["benchmark_script"] == "sglang_mi300x.sh"


def test_materialize_writes_operator_extra_env_into_vllm_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_EXTRA_ENV",
        '{"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"}',
    )
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm")
    out = tmp_path / "out"
    out.mkdir()
    envs = yaml.safe_load(materialize_config_with_envs(base, out, gpu_type="mi355x").read_text())["benchmark"]["envs"]
    assert envs["VLLM_USE_BREAKABLE_CUDAGRAPH"] == "0"


def test_kimi_materialize_enables_remote_client_trust(tmp_path):
    base = tmp_path / "base.yaml"
    _write_yaml(base, model="/path/models/moonshotai-Kimi-K2.6")
    out = tmp_path / "out"
    out.mkdir()

    with patch.dict("sys.modules", {"hyperloom.inference_optimizer.cli.model_gate": _CLI_STUB}):
        materialized = materialize_config_with_envs(
            base,
            out,
            model_path="/path/models/moonshotai-Kimi-K2.6",
            gpu_type="mi300x",
        )
    envs = yaml.safe_load(materialized.read_text())["benchmark"]["envs"]

    assert envs["SGLANG_ROCM_FUSED_DECODE_MLA"] == "0"
    assert envs["MAGPIE_TRUST_REMOTE_CODE"] == "1"
    assert "--trust-remote-code" in envs["EXTRA_SGLANG_ARGS"]


def test_custom_tokenizer_auto_map_enables_client_and_server_trust(tmp_path):
    model = tmp_path / "custom-tokenizer-model"
    model.mkdir()
    (model / "config.json").write_text(
        """
        {
          "model_type": "custom_text",
          "architectures": ["CustomForCausalLM"],
          "auto_map": {"AutoTokenizer": ["tokenization_custom.CustomTokenizer", null]}
        }
        """,
        encoding="utf-8",
    )
    base = tmp_path / "base.yaml"
    _write_yaml(base, model=str(model))
    out = tmp_path / "out"
    out.mkdir()

    with patch.dict("sys.modules", {"hyperloom.inference_optimizer.cli.model_gate": _CLI_STUB}):
        materialized = materialize_config_with_envs(
            base,
            out,
            model_path=str(model),
            gpu_type="mi300x",
        )
    envs = yaml.safe_load(materialized.read_text())["benchmark"]["envs"]

    assert envs["MAGPIE_TRUST_REMOTE_CODE"] == "1"
    assert envs["BENCH_TRUST_REMOTE_CODE"] == "1"
    assert envs["HF_HUB_TRUST_REMOTE_CODE"] == "1"
    assert "--trust-remote-code" in envs["EXTRA_SGLANG_ARGS"]


def test_qwen36_materialize_enables_client_and_server_trust(tmp_path):
    base = tmp_path / "base.yaml"
    _write_yaml(
        base,
        framework="vllm",
        model="/path/models/Qwen-Qwen3.6-35B-A3B",
    )
    out = tmp_path / "out"
    out.mkdir()

    with patch.dict("sys.modules", {"hyperloom.inference_optimizer.cli.model_gate": _CLI_STUB}):
        materialized = materialize_config_with_envs(
            base,
            out,
            model_path="/path/models/Qwen-Qwen3.6-35B-A3B",
            gpu_type="mi300x",
        )
    envs = yaml.safe_load(materialized.read_text())["benchmark"]["envs"]

    assert envs["MAGPIE_TRUST_REMOTE_CODE"] == "1"
    assert "--trust-remote-code" in envs["EXTRA_VLLM_ARGS"]


def _write_yaml_with_server_args(
    path: Path,
    *,
    framework: str,
    env_key: str,
    server_args: str,
) -> None:
    """Like ``_write_yaml`` but seeds a framework server-args env in the YAML."""
    cfg: dict = {
        "benchmark": {
            "framework": framework,
            "model": "/path/models/Qwen-Qwen3-8B",
            "precision": "bf16",
            "run_mode": "local",
            "envs": {
                "TP": 1,
                "CONC": 8,
                "ISL": 256,
                "OSL": 256,
                env_key: server_args,
            },
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        }
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def test_materialize_dedups_duplicate_vllm_attention_backend(tmp_path):
    """A YAML EXTRA_VLLM_ARGS base + a variant's extra_server_args must not yield a duplicate --attention-backend in the materialized config (vLLM crashes EngineCoreProc on a duplicate)."""
    base = tmp_path / "base.yaml"
    _write_yaml_with_server_args(
        base,
        framework="vllm",
        env_key="EXTRA_VLLM_ARGS",
        server_args="--attention-backend ROCM_AITER_FA",
    )
    out = tmp_path / "out"
    out.mkdir()

    materialized = materialize_config_with_envs(
        base,
        out,
        model_path="/path/models/Qwen-Qwen3-8B",
        gpu_type="mi300x",
        extra_server_args="--attention-backend ROCM_FLASH",
    )
    vllm_args = yaml.safe_load(materialized.read_text())["benchmark"]["envs"]["EXTRA_VLLM_ARGS"]

    assert vllm_args.count("--attention-backend") == 1, vllm_args
    # last-wins: the variant override survives, the YAML base is dropped.
    assert "ROCM_FLASH" in vllm_args
    assert "ROCM_AITER_FA" not in vllm_args


def test_materialize_keeps_sglang_repeated_attention_backend(tmp_path):
    """SGLang tolerates a repeated flag (last-wins at the server), so materialize must NOT dedup it."""
    base = tmp_path / "base.yaml"
    _write_yaml_with_server_args(
        base,
        framework="sglang",
        env_key="EXTRA_SGLANG_ARGS",
        server_args="--attention-backend aiter",
    )
    out = tmp_path / "out"
    out.mkdir()

    materialized = materialize_config_with_envs(
        base,
        out,
        model_path="/path/models/Qwen-Qwen3-8B",
        gpu_type="mi300x",
        extra_server_args="--attention-backend triton",
    )
    sglang_args = yaml.safe_load(materialized.read_text())["benchmark"]["envs"]["EXTRA_SGLANG_ARGS"]

    assert sglang_args.count("--attention-backend") == 2, sglang_args


# TP auto-clamp against visible GPU count
def _write_yaml_with_tp(path: Path, tp: int) -> None:
    """Like ``_write_yaml`` but lets the test pin the YAML's default TP."""
    cfg: dict = {
        "benchmark": {
            "framework": "sglang",
            "model": "/path/models/Qwen-Qwen3-8B",
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": tp, "CONC": 8, "ISL": 256, "OSL": 256},
            "timeout_seconds": 600,
            "profiler": {
                "torch_profiler": {"enabled": False},
                "system_profiler": {"enabled": False},
                "tracelens": {"enabled": False},
            },
            "gpu_selection": {"auto": False},
        }
    }
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def test_materialize_config_with_envs_clamps_tp_to_visible_gpus(
    tmp_path,
    monkeypatch,
    caplog,
):
    """A 4-GPU pod must not launch sglang/vllm with ``TP=8``; the materializer clamps TP to the visible GPU count."""
    base = tmp_path / "base.yaml"
    _write_yaml_with_tp(base, tp=8)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.delenv("TP", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", raising=False)

    with patch(
        "hyperloom.orchestrator.actions.executors._workload_envs._visible_gpu_count",
        return_value=4,
    ):
        materialized = materialize_config_with_envs(base, out)

    cfg = yaml.safe_load(materialized.read_text())
    assert cfg["benchmark"]["envs"]["TP"] == 4
    assert cfg["benchmark"]["envs"]["ROCR_VISIBLE_DEVICES"] == "0,1,2,3"


def test_materialize_config_with_envs_clamp_respects_env_override(
    tmp_path,
    monkeypatch,
):
    """When the operator sets ``$TP`` the clamp still fires; ``INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP=1`` is the documented bypass for a deliberate oversubscribed launch."""
    base = tmp_path / "base.yaml"
    _write_yaml_with_tp(base, tp=1)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("TP", "8")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")

    with patch(
        "hyperloom.orchestrator.actions.executors._workload_envs._visible_gpu_count",
        return_value=4,
    ):
        materialized = materialize_config_with_envs(base, out)

    cfg = yaml.safe_load(materialized.read_text())
    # Bypass keeps the operator-requested TP=8.
    assert cfg["benchmark"]["envs"]["TP"] == 8
    assert cfg["benchmark"]["envs"]["ROCR_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"


def test_materialize_config_with_envs_no_clamp_when_visible_zero(
    tmp_path,
    monkeypatch,
):
    """When ``_visible_gpu_count`` returns 0 (CPU-only / rocm-smi failure) the materializer must NOT clamp to 0; it leaves the YAML TP intact."""
    base = tmp_path / "base.yaml"
    _write_yaml_with_tp(base, tp=2)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.delenv("TP", raising=False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", raising=False)

    with patch(
        "hyperloom.orchestrator.actions.executors._workload_envs._visible_gpu_count",
        return_value=0,
    ):
        materialized = materialize_config_with_envs(base, out)

    cfg = yaml.safe_load(materialized.read_text())
    assert cfg["benchmark"]["envs"]["TP"] == 2
    # ROCR was unset upstream, so it is derived from TP.
    assert cfg["benchmark"]["envs"]["ROCR_VISIBLE_DEVICES"] == "0,1"


# BaselineExecutor.__call__ end-to-end (subprocess mocked)
def _fake_workspace(slot: Path, *, tput: float = 1500.0) -> Path:

    import json

    ws = slot / "benchmark_sglang_20260513_010101"
    ws.mkdir(parents=True)
    (ws / "benchmark_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "framework": "sglang",
                "model": "/path/models/Qwen-Qwen3-8B",
                "throughput": {
                    "request_throughput": tput / 256,
                    "output_throughput": tput,
                    "total_token_throughput": tput * 2,
                    "completed_requests": 64,
                    "duration_seconds": 25.0,
                },
                "latency": {
                    "ttft": {"mean_ms": 100.0, "p99_ms": 120.0},
                    "e2el": {"mean_ms": 2000.0, "p99_ms": 2300.0},
                },
            }
        )
    )
    return ws


def _make_ctx(params: dict) -> SimpleNamespace:
    task = SimpleNamespace(task_id="t-baseline-1", params=params)
    return SimpleNamespace(task=task, extra={})


def _run(coro):
    return asyncio.run(coro)


def test_baseline_executor_forwards_override_to_yaml_and_env(tmp_path):
    base = tmp_path / "base.yaml"
    _write_yaml(base, benchmark_script="dsr1_fp8_mi300x.sh")
    output_dir = tmp_path / "ws"
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        cfg_idx = cmd.index("--benchmark-config")
        out_idx = cmd.index("--output-dir")
        cfg_path = Path(cmd[cfg_idx + 1])
        slot = Path(cmd[out_idx + 1])
        captured["cfg"] = yaml.safe_load(cfg_path.read_text())
        captured["env"] = dict(kwargs.get("env") or {})
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "model_path": "/path/models/DeepSeek-R1",
            "gpu_type": "mi300x",
            "benchmark_script": "sglang_mi300x.sh",
            "result_dir": str(tmp_path / "redirect_leak"),
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    bench = captured["cfg"]["benchmark"]
    assert bench["benchmark_script"] == "sglang_mi300x.sh"
    assert bench["runner_type"] == "mi300x"
    assert captured["env"]["RESULT_DIR"] == str(tmp_path / "redirect_leak")


def test_baseline_eager_fallback_records_effective_task_args(tmp_path):
    base = tmp_path / "base.yaml"
    _write_yaml(
        base,
        benchmark_script="vllm_mi300x.sh",
        framework="vllm",
    )
    output_dir = tmp_path / "ws"
    shared_state = SharedState(
        baseline_eager_fallback=True,
        model_path="/models/qwen",
    )
    shared_state.save(tmp_path)

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        _fake_workspace(Path(cmd[out_idx + 1]))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
        shared_state=shared_state,
    )
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "framework": "vllm",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert "--enforce-eager" in ctx.task.params["extra_server_args"]
    assert shared_state.baseline_eager_fallback is False


def test_baseline_executor_falls_back_to_shared_state_model_path(tmp_path, monkeypatch):
    # params has no model_path and MODEL_PATH is unset: without the SharedState
    # fallback the bare YAML model name leaks into --model-path.
    monkeypatch.delenv("MODEL_PATH", raising=False)
    base = tmp_path / "base.yaml"
    _write_yaml(base, model="PrimeIntellect-Qwen3-1.7B")  # bare name in YAML
    output_dir = tmp_path / "ws"
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        cfg_idx = cmd.index("--benchmark-config")
        out_idx = cmd.index("--output-dir")
        captured["cfg"] = yaml.safe_load(Path(cmd[cfg_idx + 1]).read_text())
        _fake_workspace(Path(cmd[out_idx + 1]))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
        shared_state=SimpleNamespace(
            model_path="/path/models/PrimeIntellect-Qwen3-1.7B",
        ),
    )
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert captured["cfg"]["benchmark"]["model"] == "/path/models/PrimeIntellect-Qwen3-1.7B"


def test_baseline_executor_falls_back_to_ctx_extra_shared_state_model_path(tmp_path, monkeypatch):
    # Production form: the executor is a module-level singleton and live state
    # arrives via ctx.extra; without reading it the bare YAML model name leaks.
    monkeypatch.delenv("MODEL_PATH", raising=False)
    base = tmp_path / "base.yaml"
    _write_yaml(base, model="PrimeIntellect-Qwen3-1.7B")  # bare name in YAML
    output_dir = tmp_path / "ws"
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        cfg_idx = cmd.index("--benchmark-config")
        out_idx = cmd.index("--output-dir")
        captured["cfg"] = yaml.safe_load(Path(cmd[cfg_idx + 1]).read_text())
        _fake_workspace(Path(cmd[out_idx + 1]))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(  # no shared_state — singleton form
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    task = SimpleNamespace(
        task_id="t-baseline-extra",
        params={
            "output_dir": str(output_dir),
            "timeout_sec": 10,
        },
    )
    ctx = SimpleNamespace(
        task=task,
        extra={
            "shared_state": SimpleNamespace(
                model_path="/path/models/PrimeIntellect-Qwen3-1.7B",
            ),
        },
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert captured["cfg"]["benchmark"]["model"] == "/path/models/PrimeIntellect-Qwen3-1.7B"


def test_baseline_executor_defaults_result_dir_to_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-benchmark")
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    output_dir = tmp_path / "ws"
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        captured["env"] = dict(kwargs.get("env") or {})
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    # Default RESULT_DIR is the per-task workspace: this fixture omits
    # benchmark_script so the cold-start double-run is not eligible and the
    # single-round path runs directly in output_dir.
    assert captured["env"]["RESULT_DIR"] == str(output_dir)
    assert "OPENAI_API_KEY" not in captured["env"]


def test_baseline_executor_pins_magpie_inferencex_path(tmp_path, monkeypatch):
    """The baseline executor's Magpie subprocess must inherit ``MAGPIE_INFERENCEX_PATH=$INFERENCEX_PATH`` so Magpie loads the patched checkout.

    The mirror is disabled here so the asserted path is exactly the configured
    ``$INFERENCEX_PATH`` rather than a hash-named local mirror dir.
    """
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_LOCAL_INFERENCEX", "1")
    monkeypatch.setenv("INFERENCEX_PATH", "/path/hyperloom/InferenceX")
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    output_dir = tmp_path / "ws"
    captured: dict = {}

    def fake_run(cmd, *args, **kwargs):
        out_idx = cmd.index("--output-dir")
        slot = Path(cmd[out_idx + 1])
        captured["env"] = dict(kwargs.get("env") or {})
        _fake_workspace(slot)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx({"output_dir": str(output_dir), "timeout_sec": 10})

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "succeeded"
    assert captured["env"].get("MAGPIE_INFERENCEX_PATH") == ("/path/hyperloom/InferenceX"), (
        "MAGPIE_INFERENCEX_PATH must equal $INFERENCEX_PATH so Magpie loads the patched checkout (#210 root cause)"
    )


def test_baseline_executor_rejects_bad_param(tmp_path):
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    output_dir = tmp_path / "ws"

    def fake_run(cmd, *args, **kwargs):  # pragma: no cover — must not run
        raise AssertionError("subprocess.run should not be invoked")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "benchmark_script": "../etc/passwd.sh",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed"
    assert result["error_class"] == "bad_param"
    assert "benchmark_script" in result["error"]


def test_baseline_executor_rejects_bad_result_dir(tmp_path):
    base = tmp_path / "base.yaml"
    _write_yaml(base)
    output_dir = tmp_path / "ws"

    def fake_run(cmd, *args, **kwargs):  # pragma: no cover
        raise AssertionError("subprocess.run should not be invoked")

    executor = BaselineExecutor(
        magpie_python="/opt/venv/bin/python",
        default_config_path=base,
        session_dir=tmp_path,
    )
    ctx = _make_ctx(
        {
            "output_dir": str(output_dir),
            "timeout_sec": 10,
            "result_dir": "/tmp/leak;rm -rf /",
        }
    )

    with patch(
        "hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill",
        side_effect=fake_run,
    ):
        result = _run(executor(ctx))

    assert result["status"] == "failed"
    assert result["error_class"] == "bad_param"
    assert "result_dir" in result["error"]


# reference-script base layer (precedence + 0-degrade)
def _fw_args(materialized: Path, env_name: str = "EXTRA_VLLM_ARGS") -> str:
    cfg = yaml.safe_load(materialized.read_text())
    return str(cfg["benchmark"]["envs"].get(env_name, ""))


def _seed_reference(tmp_path, monkeypatch, *, server_args: str = "", envs: dict | None = None) -> None:
    """Pin a session whose SharedState carries the reference recipe.

    ``materialize_config_with_envs`` reads the recipe from the session rather than
    from its arguments, so every caller gets the same base.
    """
    from hyperloom.orchestrator.state.shared_state import SharedState

    sd = tmp_path / "session"
    sd.mkdir(parents=True, exist_ok=True)
    SharedState(
        session_id="ref-test",
        reference_server_args=server_args,
        reference_envs=dict(envs or {}),
    ).save(sd)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", str(sd))


def test_reference_base_seeds_lowest_priority(tmp_path, monkeypatch):
    """Reference flags appear in the framework env at lowest priority."""
    _seed_reference(tmp_path, monkeypatch, server_args="--block-size 128 --attention-backend TRITON_ATTN")
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm", model="/path/models/X")
    out = tmp_path / "out"
    out.mkdir()
    materialized = materialize_config_with_envs(
        base,
        out,
        model_path="/path/models/X",
        gpu_type="mi300x",
    )
    args = _fw_args(materialized)
    assert "--block-size 128" in args
    assert "TRITON_ATTN" in args


def test_reference_base_reaches_a_caller_that_renders_its_own_yaml(tmp_path, monkeypatch):
    """Every caller gets the recipe, not just the one that used to forward it.

    ``explore`` / ``sweep`` / ``conc_sweep`` / ``integrate_patch`` /
    ``framework_agent`` / ``rebench`` render their own YAML and pass no recipe;
    they must still benchmark on the operator's base.
    """
    _seed_reference(
        tmp_path,
        monkeypatch,
        server_args="--enable-prefix-caching",
        envs={"AITER_SITUV2_A8W4": "1", "AITER_BF16_FP8_MOE_BOUND": "0"},
    )
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm", model="/path/models/X")
    out = tmp_path / "out"
    out.mkdir()

    materialized = materialize_config_with_envs(
        base,
        out,
        model_path="/path/models/X",
        gpu_type="mi300x",
        out_name="explore_base.with_envs.yaml",
    )

    assert "--enable-prefix-caching" in _fw_args(materialized)
    envs = yaml.safe_load(materialized.read_text())["benchmark"]["envs"]
    assert envs["AITER_SITUV2_A8W4"] == "1"
    assert envs["AITER_BF16_FP8_MOE_BOUND"] == "0"


def test_reference_base_extra_args_override_wins(tmp_path, monkeypatch):
    """A per-task extra_server_args override beats the reference base, deduped once."""
    _seed_reference(tmp_path, monkeypatch, server_args="--block-size 128 --attention-backend TRITON_ATTN")
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm", model="/path/models/X")
    out = tmp_path / "out"
    out.mkdir()
    materialized = materialize_config_with_envs(
        base,
        out,
        model_path="/path/models/X",
        gpu_type="mi300x",
        extra_server_args="--attention-backend ROCM_FLASH",
    )
    args = _fw_args(materialized)
    # reference-only flag survives
    assert args.count("--block-size") == 1
    assert "--block-size 128" in args
    # override flag wins and is not doubled
    assert args.count("--attention-backend") == 1
    assert "ROCM_FLASH" in args
    assert "TRITON_ATTN" not in args


def test_reference_and_extra_single_value_flag_deduped_recipe_wins(tmp_path, monkeypatch):
    """A single-value flag in both reference and extra args collapses to last-wins."""
    _seed_reference(tmp_path, monkeypatch, server_args="--gpu-memory-utilization 0.95")
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm", model="/path/models/X")
    out = tmp_path / "out"
    out.mkdir()
    materialized = materialize_config_with_envs(
        base,
        out,
        model_path="/path/models/X",
        gpu_type="mi300x",
        extra_server_args="--kv-cache-dtype fp8 --gpu-memory-utilization 0.97",
    )
    args = _fw_args(materialized)
    assert args.count("--gpu-memory-utilization") == 1
    assert "0.97" in args and "0.95" not in args
    assert "--kv-cache-dtype fp8" in args


def test_reference_envs_do_not_clobber_existing(tmp_path, monkeypatch):
    """reference_envs use setdefault — never override a YAML/CLI-set env."""
    _seed_reference(tmp_path, monkeypatch, envs={"VLLM_ROCM_USE_AITER": "0", "VLLM_FP8_PADDING": "1"})
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm", model="/path/models/X")
    out = tmp_path / "out"
    out.mkdir()
    materialized = materialize_config_with_envs(
        base,
        out,
        model_path="/path/models/X",
        gpu_type="mi300x",
        extra_envs={"VLLM_ROCM_USE_AITER": "1"},
    )
    envs = yaml.safe_load(materialized.read_text())["benchmark"]["envs"]
    # extra_envs (CLI) wins over reference; the new reference key still lands.
    assert envs["VLLM_ROCM_USE_AITER"] == "1"
    assert envs["VLLM_FP8_PADDING"] == "1"


def test_no_reference_recipe_is_a_no_op(tmp_path, monkeypatch):
    """A session without a reference recipe renders exactly the plain YAML."""
    _seed_reference(tmp_path, monkeypatch)
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework="vllm", model="/path/models/X")
    out = tmp_path / "out"
    out.mkdir()

    materialized = materialize_config_with_envs(
        base,
        out,
        model_path="/path/models/X",
        gpu_type="mi300x",
        out_name="x.yaml",
    )

    assert _fw_args(materialized) == ""
