# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""CK block-scale patch wiring tests at the env-materialization choke point.

The ``SGLANG_FP8_BLOCKSCALE_CK_MAX_M`` env only yields a speedup on a
KernelForge-patched sglang ``fp8_utils.py`` (M-aware CK routing). Hyperloom is
responsible for ensuring that patch is applied whenever it injects the env, by
calling ``ensure_sglang_patched_for_ck_blockscale`` from
``materialize_config_with_envs``. A prior incident showed this wiring can
silently drift across deploys (the env was injected but the patcher was never
invoked, so the env no-op'd on an unpatched tree). These tests pin the wiring:
the patcher must be invoked exactly when (sglang framework + env present +
``HYPERLOOM_ENABLE_PATCH`` on), and never otherwise, and a fail-soft patch
result must not break materialization.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hyperloom.inference_optimizer.cli import model_gate as cli_model_gate
from hyperloom.orchestrator.actions.executors import _workload_envs
from hyperloom.orchestrator.actions.executors._workload_envs import (
    materialize_config_with_envs,
)

_CK_ENV = "SGLANG_FP8_BLOCKSCALE_CK_MAX_M"


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Deterministic gating: kill switch on, no host GPU autodetect leakage."""
    monkeypatch.delenv("HYPERLOOM_ENABLE_PATCH", raising=False)
    monkeypatch.delenv("GPU_TYPE", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setattr(cli_model_gate, "_autodetect_gpu_type", lambda: None)
    for key in ("CONC", "ISL", "OSL", "MAX_MODEL_LEN", "TP", "PRECISION",
                "RUN_EVAL", "FRAMEWORK", _CK_ENV):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def patcher_calls(monkeypatch):
    """Record (and stub) the CK patcher invoked from the materialize hook."""
    calls: list[bool] = []

    def _fake(*_args, **_kwargs) -> bool:
        calls.append(True)
        return True

    monkeypatch.setattr(
        _workload_envs, "ensure_sglang_patched_for_ck_blockscale", _fake
    )
    return calls


def _write_yaml(path: Path, *, framework: str) -> None:
    cfg = {
        "benchmark": {
            "framework": framework,
            "model": "/models/whatever",
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
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)


def _materialize(
    tmp_path: Path, *, framework: str = "sglang", extra_envs: dict | None = None
) -> dict:
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework=framework)
    out = tmp_path / "out"
    out.mkdir()
    materialized = materialize_config_with_envs(
        base,
        out,
        model_path="/models/whatever",
        gpu_type=None,
        extra_envs=extra_envs,
    )
    return yaml.safe_load(materialized.read_text())["benchmark"]["envs"]


def test_patcher_invoked_when_ck_env_present_on_sglang(tmp_path, patcher_calls):
    envs = _materialize(tmp_path, extra_envs={_CK_ENV: "256"})
    assert patcher_calls == [True]
    # The env survives materialization so the (now patched) tree honours it.
    assert envs.get(_CK_ENV) == "256"


def test_patcher_not_invoked_without_ck_env(tmp_path, patcher_calls):
    _materialize(tmp_path)
    assert patcher_calls == []


def test_patcher_not_invoked_for_non_sglang(tmp_path, patcher_calls):
    _materialize(tmp_path, framework="vllm", extra_envs={_CK_ENV: "256"})
    assert patcher_calls == []


def test_patcher_not_invoked_when_kill_switch_off(tmp_path, patcher_calls, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_ENABLE_PATCH", "0")
    _materialize(tmp_path, extra_envs={_CK_ENV: "256"})
    assert patcher_calls == []


def test_materialize_survives_failsoft_patch(tmp_path, monkeypatch):
    # A failed patch must not break materialization; the env is left to no-op.
    monkeypatch.setattr(
        _workload_envs,
        "ensure_sglang_patched_for_ck_blockscale",
        lambda *a, **k: False,
    )
    envs = _materialize(tmp_path, extra_envs={_CK_ENV: "256"})
    assert envs.get(_CK_ENV) == "256"
