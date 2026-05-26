"""Unit tests for ``orchestrator.pmc_workload_params``.

Exercises the YAML-driven derivation of pmc_roofline task parameters for
both sglang and vLLM frameworks, plus the error branches when the
materialised config is missing or unreadable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.orchestrator.pmc_workload_params import (
    derive_pmc_roofline_params_from_config,
)


def _write_config(
    tmp_path: Path,
    *,
    framework: str = "sglang",
    model: str = "/wekafs/models/Qwen-Qwen3-8B",
    precision: str = "bf16",
    envs: dict | None = None,
) -> Path:
    cfg_path = tmp_path / "magpie.yaml"
    envs = envs or {"TP": 1, "CONC": 8, "ISL": 1024, "OSL": 256}
    cfg_path.write_text(
        "benchmark:\n"
        f"  framework: {framework}\n"
        f"  model: {model}\n"
        f"  precision: {precision}\n"
        "  envs:\n"
        + "".join(f"    {k}: {v}\n" for k, v in envs.items())
    )
    return cfg_path


def test_returns_none_when_config_missing(tmp_path):
    assert derive_pmc_roofline_params_from_config(tmp_path / "ghost.yaml") is None


def test_returns_none_when_yaml_unreadable(tmp_path, monkeypatch):
    bad = tmp_path / "bad.yaml"
    bad.write_text("{ this is not yaml: : ::: ::")

    # Force read_text to raise to exercise the OSError branch.
    def boom(self, **kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(Path, "read_text", boom)
    assert derive_pmc_roofline_params_from_config(bad) is None


def test_returns_none_when_no_model(tmp_path):
    cfg = _write_config(tmp_path, model="")
    assert derive_pmc_roofline_params_from_config(cfg) is None


def test_sglang_default_command_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_PMC_ROOFLINE_PORT", "30002")
    monkeypatch.delenv("GPU_TYPE", raising=False)
    monkeypatch.delenv("HYPERLOOM_PMC_ROOFLINE_GPU_TYPE", raising=False)
    monkeypatch.setenv("INFERENCEX_PATH", "/inferencex")
    cfg = _write_config(tmp_path)
    out = derive_pmc_roofline_params_from_config(cfg, gpu_type="MI300X")
    assert out is not None
    server = out["server_cmd"]
    assert server[0:3] == ["python", "-m", "sglang.launch_server"]
    assert "--port" in server and server[server.index("--port") + 1] == "30002"
    # benchmark command points at the InferenceX serving harness.
    assert any("benchmark_serving.py" in part for part in out["benchmark_cmd"])
    # GPU type normalised lower-case from the explicit override.
    assert out["gpu_type"] == "mi300x"


def test_vllm_branch_translates_precision(tmp_path):
    cfg = _write_config(tmp_path, framework="vllm", precision="bf16")
    out = derive_pmc_roofline_params_from_config(cfg)
    assert out is not None
    server = out["server_cmd"]
    assert server[0:2] == ["vllm", "serve"]
    # bf16 maps to bfloat16 for vLLM dtype.
    assert "bfloat16" in server


def test_passthrough_extra_args(tmp_path):
    cfg = _write_config(
        tmp_path,
        framework="vllm",
        envs={
            "TP": 1, "CONC": 4, "ISL": 256, "OSL": 256,
            "EXTRA_VLLM_ARGS": "--max-num-seqs 64 --quantization fp8",
        },
    )
    out = derive_pmc_roofline_params_from_config(cfg)
    assert out is not None
    server = out["server_cmd"]
    assert "--max-num-seqs" in server
    assert "fp8" in " ".join(server)


def test_environment_fallbacks(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_MODEL_LEN", "4096")
    monkeypatch.setenv("PRECISION", "fp16")
    monkeypatch.setenv("CONC", "32")
    monkeypatch.setenv("ISL", "128")
    cfg_path = tmp_path / "magpie.yaml"
    cfg_path.write_text(
        "benchmark:\n"
        "  framework: sglang\n"
        "  model: /weights/m\n"
        "  envs: {}\n"
    )
    out = derive_pmc_roofline_params_from_config(cfg_path)
    assert out is not None
    assert "4096" in out["server_cmd"]
    # precision falls back from env when not set in YAML.
    assert out["precision"] == "fp16"
