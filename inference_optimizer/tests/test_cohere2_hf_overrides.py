"""Command-A ``--hf-overrides`` injection tests.

Command-A checkpoints declare ``Cohere2VisionForConditionalGeneration`` at
the top level. vLLM resolves that path and hits
``CONFIG_MAPPING["cohere2_moe"]`` KeyError on transformers builds that do
not register ``cohere2_moe``. Hyperloom injects ``--hf-overrides`` to pin
``Cohere2MoeForCausalLM`` unless the operator already supplied one.
Exercised at both the pure-helper and ``materialize_config_with_envs`` layers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from inference_optimizer.orchestrator.action_executors._grid_runner import (
    inject_vllm_cohere2_hf_overrides,
)
from inference_optimizer.orchestrator.action_executors._workload_envs import (
    materialize_config_with_envs,
)

_COMMAND_A_MODEL = "/wekafs/models/CohereLabs-command-a-plus-05-2026-fp8"
_EXPECTED_OVERRIDES = (
    '--hf-overrides {"architectures":["Cohere2MoeForCausalLM"],'
    '"model_type":"cohere2"}'
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Clear workload env knobs so rendered YAML is hermetic."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    for key in (
        "SGLANG_CONTEXT_HEADROOM_TOKENS", "SGLANG_CONTEXT_FLOOR_TOKENS",
        "SGLANG_WATCHDOG_TIMEOUT",
        "CONC", "ISL", "OSL", "MAX_MODEL_LEN", "TP", "RANDOM_RANGE_RATIO",
        "ROCR_VISIBLE_DEVICES", "PRECISION", "RUN_EVAL", "FRAMEWORK",
    ):
        monkeypatch.delenv(key, raising=False)


def _write_yaml(path: Path, *, framework: str, model: str) -> None:
    cfg: dict = {
        "benchmark": {
            "framework": framework,
            "model": model,
            "precision": "bf16",
            "run_mode": "local",
            "envs": {"TP": 8, "CONC": 32, "ISL": 256, "OSL": 256},
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


def _materialize_envs(
    tmp_path: Path,
    *,
    framework: str,
    model: str,
    extra_server_args: str = "",
) -> dict:
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework=framework, model=model)
    out = tmp_path / "out"
    out.mkdir()
    materialized = materialize_config_with_envs(
        base, out, extra_server_args=extra_server_args,
    )
    cfg = yaml.safe_load(materialized.read_text())
    return cfg["benchmark"]["envs"]


# inject_vllm_cohere2_hf_overrides (pure helper)


def test_injects_for_command_a_vllm():
    out = inject_vllm_cohere2_hf_overrides("", "vllm", _COMMAND_A_MODEL)
    assert _EXPECTED_OVERRIDES in out


def test_noop_for_non_command_a():
    out = inject_vllm_cohere2_hf_overrides("", "vllm", "/wekafs/models/Qwen-Qwen3-8B")
    assert out == ""


def test_noop_for_sglang():
    out = inject_vllm_cohere2_hf_overrides("", "sglang", _COMMAND_A_MODEL)
    assert out == ""


def test_respects_operator_hf_overrides():
    pinned = '--hf-overrides {"architectures":["OtherModel"]}'
    out = inject_vllm_cohere2_hf_overrides(pinned, "vllm", _COMMAND_A_MODEL)
    assert out == pinned
    assert _EXPECTED_OVERRIDES not in out


def test_merges_with_existing_args():
    out = inject_vllm_cohere2_hf_overrides(
        "--gpu-memory-utilization 0.9", "vllm", _COMMAND_A_MODEL,
    )
    assert "--gpu-memory-utilization 0.9" in out
    assert _EXPECTED_OVERRIDES in out


# materialize_config_with_envs (production choke point)


def test_materialize_injects_for_command_a_vllm(tmp_path):
    envs = _materialize_envs(tmp_path, framework="vllm", model=_COMMAND_A_MODEL)
    assert _EXPECTED_OVERRIDES in envs["EXTRA_VLLM_ARGS"]


def test_materialize_noop_for_qwen_vllm(tmp_path):
    envs = _materialize_envs(
        tmp_path, framework="vllm", model="/wekafs/models/Qwen-Qwen3-8B",
    )
    assert "hf-overrides" not in envs.get("EXTRA_VLLM_ARGS", "")


def test_materialize_respects_extra_server_args(tmp_path):
    pinned = '--hf-overrides {"architectures":["OtherModel"]}'
    envs = _materialize_envs(
        tmp_path,
        framework="vllm",
        model=_COMMAND_A_MODEL,
        extra_server_args=pinned,
    )
    assert pinned in envs["EXTRA_VLLM_ARGS"]
    assert _EXPECTED_OVERRIDES not in envs["EXTRA_VLLM_ARGS"]
