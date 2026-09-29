# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Native AgentX selection preserves the legacy CLI and persisted sessions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from hyperloom.common.agentx_mode import config_enables_native_agentx, native_agentx_session
from hyperloom.inference_optimizer.cli.preflight import _native_agentx_preflight_requested


@pytest.mark.parametrize(
    "selector, expected", [("enable", True), ({"enabled": True}, True), (False, False), (None, False)]
)
def test_fresh_native_selection_requires_explicit_source_yaml(tmp_path, selector, expected):
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump({"benchmark": {"agentx": selector}}), encoding="utf-8")
    env = {"HYPERLOOM_BENCHMARK_CONFIG": str(path), "HYPERLOOM_AGENTX": "1"}
    assert native_agentx_session(env=env) is expected
    assert native_agentx_session(env={"HYPERLOOM_AGENTX": "1", "AGENTX_SERVER_SCRIPT": "native.sh"}) is False


@pytest.mark.parametrize("raw", ["[", "[]", "benchmark: []", "benchmark: {}"])
def test_invalid_or_generic_yaml_does_not_select_native(tmp_path, raw):
    path = tmp_path / "source.yaml"
    path.write_text(raw, encoding="utf-8")
    assert config_enables_native_agentx(path) is False
    assert config_enables_native_agentx(tmp_path / "missing.yaml") is False


@pytest.mark.parametrize("mapping", [True, False])
def test_saved_legacy_or_synthetic_state_outranks_ambient_native_config(tmp_path, mapping):
    source = tmp_path / "native.yaml"
    source.write_text("benchmark:\n  agentx: enable\n", encoding="utf-8")
    env = {"HYPERLOOM_BENCHMARK_CONFIG": str(source), "HYPERLOOM_AGENTX": "1"}
    for mode in ("agentx", "synthetic"):
        fields = {"benchmark_mode": mode, "agentx_epoch": 1}
        state = fields if mapping else SimpleNamespace(**fields)
        assert native_agentx_session(state, env=env) is False


def test_profile_template_cannot_downgrade_native_session_identity(tmp_path):
    source = tmp_path / "source.yaml"
    source.write_text("benchmark:\n  agentx: enable\n", encoding="utf-8")
    profile = tmp_path / "profile.yaml"
    profile.write_text("benchmark:\n  profiling: enable\n", encoding="utf-8")
    state = SimpleNamespace(
        benchmark_mode="agentx", baseline_config_path=str(profile), benchmark_source_config_path=str(source)
    )
    assert native_agentx_session(state, env={"HYPERLOOM_BENCHMARK_CONFIG": str(profile)}) is True
    state.benchmark_source_config_path = ""
    state.agentx_epoch = 2
    assert native_agentx_session(state, env={}) is True


@pytest.mark.parametrize("epoch, expected", [(1, False), (2, True)])
def test_preflight_uses_saved_identity_before_exporting_native_pins(tmp_path, monkeypatch, epoch, expected):
    source = tmp_path / "source.yaml"
    source.write_text("benchmark:\n  agentx: enable\n", encoding="utf-8")
    monkeypatch.setenv("HYPERLOOM_BENCHMARK_CONFIG", str(source))
    monkeypatch.setenv("MAGPIE_REF", "")
    monkeypatch.setenv("INFERENCEX_REF", "")
    pins = {"MAGPIE_REF": "a" * 40, "INFERENCEX_REF": "b" * 40, "AGENTX_SERVER_SCRIPT": "native.sh"}
    (tmp_path / "state.json").write_text(
        json.dumps({"benchmark_mode": "agentx", "agentx_epoch": epoch, "agentx_runtime_pins": pins}), encoding="utf-8"
    )
    assert _native_agentx_preflight_requested(argparse.Namespace(resume_from=str(tmp_path))) is expected
    assert os.environ["MAGPIE_REF"] == (pins["MAGPIE_REF"] if expected else "")


def test_preflight_does_not_promote_legacy_env_or_unreadable_resume(monkeypatch, tmp_path):
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("AGENTX_SERVER_SCRIPT", "native.sh")
    assert _native_agentx_preflight_requested(argparse.Namespace()) is False
    assert _native_agentx_preflight_requested(argparse.Namespace(resume_from=str(tmp_path))) is False


@pytest.mark.parametrize("selector", [None, False])
def test_generic_config_keeps_native_package_lazy_in_fresh_process(tmp_path, selector):
    config = tmp_path / "generic.yaml"
    benchmark = {"framework": "sglang"}
    if selector is not None:
        benchmark["agentx"] = selector
    config.write_text(yaml.safe_dump({"benchmark": benchmark}), encoding="utf-8")
    code = """
import sys
from hyperloom.common.agentx_mode import native_agentx_session
from hyperloom.orchestrator.actions.executors._ray_backend import _should_use_ray_backend
assert not native_agentx_session()
_should_use_ray_backend()
assert not any(name.startswith('hyperloom.inference_optimizer.agentx') for name in sys.modules)
"""
    env = dict(os.environ, HYPERLOOM_BENCHMARK_CONFIG=str(config), HYPERLOOM_AGENTX="0")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
