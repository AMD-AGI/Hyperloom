# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""sglang ``--watchdog-timeout`` injection tests.

The first aiter request JIT-compiles a kernel that can exceed sglang's default
300s watchdog and kill the server during warmup. Hyperloom injects a longer
``--watchdog-timeout`` into ``EXTRA_SGLANG_ARGS`` unless the user pinned one.
Exercised at both the pure-helper and ``materialize_config_with_envs`` layers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from inference_optimizer.orchestrator.action_executors._grid_runner import (
    DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC,
    inject_sglang_watchdog_timeout,
    resolve_sglang_watchdog_timeout,
)
from inference_optimizer.orchestrator.action_executors._workload_envs import (
    materialize_config_with_envs,
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Clear the watchdog env knob and disable the TP clamp so the rendered YAML is hermetic."""
    monkeypatch.delenv("SGLANG_WATCHDOG_TIMEOUT", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    for key in (
        "CONC", "ISL", "OSL", "MAX_MODEL_LEN", "TP", "RANDOM_RANGE_RATIO",
        "ROCR_VISIBLE_DEVICES", "PRECISION", "RUN_EVAL", "FRAMEWORK",
    ):
        monkeypatch.delenv(key, raising=False)


def _write_yaml(path: Path, *, framework: str = "sglang") -> None:
    cfg: dict = {
        "benchmark": {
            "framework": framework,
            "model": "/wekafs/models/Qwen-Qwen3-8B",
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


def _materialize_envs(
    tmp_path: Path,
    *,
    framework: str = "sglang",
    extra_server_args: str = "",
    extra_envs: dict | None = None,
) -> dict:
    """Render a YAML via the production choke point and return its envs map."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, framework=framework)
    out = tmp_path / "out"
    out.mkdir()
    materialized = materialize_config_with_envs(
        base, out,
        extra_server_args=extra_server_args,
        extra_envs=extra_envs,
    )
    cfg = yaml.safe_load(materialized.read_text())
    return cfg["benchmark"]["envs"]


# resolve_sglang_watchdog_timeout
def test_resolve_defaults_to_1800(monkeypatch):
    monkeypatch.delenv("SGLANG_WATCHDOG_TIMEOUT", raising=False)
    assert DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SEC == 1800
    assert resolve_sglang_watchdog_timeout() == 1800


def test_resolve_reads_env_override(monkeypatch):
    monkeypatch.setenv("SGLANG_WATCHDOG_TIMEOUT", "900")
    assert resolve_sglang_watchdog_timeout() == 900


@pytest.mark.parametrize("bad", ["", "   ", "abc", "1.5", "0", "-5"])
def test_resolve_bad_value_falls_back_to_default(monkeypatch, bad):
    monkeypatch.setenv("SGLANG_WATCHDOG_TIMEOUT", bad)
    assert resolve_sglang_watchdog_timeout() == 1800


# inject_sglang_watchdog_timeout (pure helper)
def test_inject_appends_default_for_sglang(monkeypatch):
    monkeypatch.delenv("SGLANG_WATCHDOG_TIMEOUT", raising=False)
    assert (
        inject_sglang_watchdog_timeout("--foo bar", "sglang")
        == "--foo bar --watchdog-timeout 1800"
    )


def test_inject_appends_when_args_empty(monkeypatch):
    monkeypatch.delenv("SGLANG_WATCHDOG_TIMEOUT", raising=False)
    assert (
        inject_sglang_watchdog_timeout("", "sglang") == "--watchdog-timeout 1800"
    )
    # None coerces to empty and is treated identically.
    assert (
        inject_sglang_watchdog_timeout(None, "sglang")
        == "--watchdog-timeout 1800"
    )


def test_inject_uses_env_override(monkeypatch):
    monkeypatch.setenv("SGLANG_WATCHDOG_TIMEOUT", "900")
    assert (
        inject_sglang_watchdog_timeout("", "sglang") == "--watchdog-timeout 900"
    )


@pytest.mark.parametrize("existing", [
    "--watchdog-timeout 600",
    "--watchdog-timeout=600",
    "--foo 1 --watchdog-timeout 600 --bar 2",
])
def test_inject_does_not_double_user_value(existing):
    out = inject_sglang_watchdog_timeout(existing, "sglang")
    assert out == existing
    assert out.count("--watchdog-timeout") == 1
    assert "1800" not in out


def test_inject_empty_or_unknown_framework_treated_as_sglang(monkeypatch):
    """An empty/unknown framework routes to EXTRA_SGLANG_ARGS, so the watchdog is injected there too."""
    monkeypatch.delenv("SGLANG_WATCHDOG_TIMEOUT", raising=False)
    assert inject_sglang_watchdog_timeout("", "") == "--watchdog-timeout 1800"
    assert inject_sglang_watchdog_timeout("", None) == "--watchdog-timeout 1800"


@pytest.mark.parametrize("framework", ["vllm", "atom"])
def test_inject_noop_for_non_sglang(framework):
    assert inject_sglang_watchdog_timeout("--foo", framework) == "--foo"
    assert inject_sglang_watchdog_timeout("", framework) == ""
    assert "--watchdog-timeout" not in inject_sglang_watchdog_timeout(
        "--gpu-memory-utilization 0.9", framework,
    )


# materialize_config_with_envs (the production choke point)
def test_materialize_sglang_injects_default_watchdog(tmp_path):
    envs = _materialize_envs(tmp_path, framework="sglang")
    assert "--watchdog-timeout 1800" in envs["EXTRA_SGLANG_ARGS"]


def test_materialize_sglang_respects_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_WATCHDOG_TIMEOUT", "900")
    envs = _materialize_envs(tmp_path, framework="sglang")
    assert "--watchdog-timeout 900" in envs["EXTRA_SGLANG_ARGS"]
    assert "1800" not in envs["EXTRA_SGLANG_ARGS"]


def test_materialize_sglang_does_not_double_user_watchdog(tmp_path):
    envs = _materialize_envs(
        tmp_path, framework="sglang",
        extra_server_args="--watchdog-timeout 600",
    )
    sglang_args = envs["EXTRA_SGLANG_ARGS"]
    assert sglang_args.count("--watchdog-timeout") == 1
    assert "600" in sglang_args
    assert "1800" not in sglang_args


def test_materialize_honors_user_watchdog_via_extra_envs(tmp_path):
    """A user pinning the flag into EXTRA_SGLANG_ARGS via extra_envs must not be doubled."""
    envs = _materialize_envs(
        tmp_path, framework="sglang",
        extra_envs={"EXTRA_SGLANG_ARGS": "--watchdog-timeout 600"},
    )
    sglang_args = envs["EXTRA_SGLANG_ARGS"]
    assert sglang_args.count("--watchdog-timeout") == 1
    assert "600" in sglang_args


def test_materialize_sglang_preserves_existing_args(tmp_path):
    """Injection appends; it must not clobber a user-supplied ``--context-length``."""
    envs = _materialize_envs(
        tmp_path, framework="sglang",
        extra_server_args="--context-length 6144",
    )
    sglang_args = envs["EXTRA_SGLANG_ARGS"]
    assert "--context-length 6144" in sglang_args
    assert "--watchdog-timeout 1800" in sglang_args


def test_materialize_vllm_does_not_inject_watchdog(tmp_path):
    envs = _materialize_envs(tmp_path, framework="vllm")
    # A vllm run creates no sglang env and no watchdog flag.
    assert "EXTRA_SGLANG_ARGS" not in envs
    assert "--watchdog-timeout" not in envs.get("EXTRA_VLLM_ARGS", "")
