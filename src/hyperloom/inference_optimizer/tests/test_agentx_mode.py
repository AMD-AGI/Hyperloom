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

from hyperloom.common.agentx_mode import (
    config_enables_native_agentx,
    managed_native_agentx_session,
    native_agentx_session,
    native_agentx_optimization_session,
)
from hyperloom.inference_optimizer.cli.preflight import _native_agentx_preflight_requested


@pytest.mark.parametrize(
    "selector, expected", [("enable", True), ({"enabled": True}, True), (False, False), (None, False)]
)
def test_fresh_native_selection_accepts_source_yaml_or_public_switch(tmp_path, selector, expected):
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump({"benchmark": {"agentx": selector}}), encoding="utf-8")
    env = {"HYPERLOOM_BENCHMARK_CONFIG": str(path)}
    assert native_agentx_session(env=env) is expected
    assert native_agentx_session(env={"HYPERLOOM_AGENTX": "1"}) is True


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


def test_preflight_selects_native_for_fresh_switch_but_not_unreadable_resume(monkeypatch, tmp_path):
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("AGENTX_SERVER_SCRIPT", "native.sh")
    assert _native_agentx_preflight_requested(argparse.Namespace()) is True
    assert _native_agentx_preflight_requested(argparse.Namespace(resume_from=str(tmp_path))) is False


@pytest.mark.parametrize("selector", [None, False, "disable", "off", {"enabled": False}, {"enabled": "false"}])
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


@pytest.mark.parametrize("epoch,native,optimize", [(1, False, False), (2, True, False), (3, True, True), (4, True, True)])
def test_saved_epoch_controls_subprocess_routing(tmp_path, epoch, native, optimize):
    state = {"benchmark_mode": "agentx", "agentx_epoch": epoch}
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    env = {"INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR": str(tmp_path), "HYPERLOOM_AGENTX": "1"}
    assert native_agentx_session(env=env) is native
    assert native_agentx_optimization_session(env=env) is optimize
    assert native_agentx_optimization_session(state, env={"HYPERLOOM_AGENTX": "1"}) is optimize
    assert managed_native_agentx_session(env=env) is (epoch == 4)
    assert managed_native_agentx_session(state, env={"HYPERLOOM_AGENTX": "1"}) is (epoch == 4)


def test_epoch_one_outranks_saved_foreign_config(tmp_path):
    source = tmp_path / "foreign.yaml"
    source.write_text("benchmark:\n  agentx: enable\n", encoding="utf-8")
    state = {"benchmark_mode": "agentx", "agentx_epoch": 1, "baseline_config_path": str(source)}
    assert not native_agentx_session(state, env={"HYPERLOOM_AGENTX": "1"})


def test_fresh_agentx_optimization_has_no_extra_switch():
    assert native_agentx_optimization_session(env={"HYPERLOOM_AGENTX": "1"})
    assert not native_agentx_optimization_session(env={"HYPERLOOM_AGENTX": "0"})
    assert managed_native_agentx_session(env={"HYPERLOOM_AGENTX": "1"})
    assert not managed_native_agentx_session(env={"HYPERLOOM_AGENTX": "0"})


@pytest.mark.parametrize(
    "epoch,warm_replay,backend", [(1, True, "legacy"), (2, False, "native"), (3, True, "native"), (4, True, "native")]
)
def test_persisted_epoch_keeps_backend_and_optimization_contract(epoch, warm_replay, backend):
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState.from_dict({"benchmark_mode": "agentx", "agentx_epoch": epoch, "warm_replay_enabled": True})
    assert state.agentx_epoch == epoch
    assert state.agentx_backend == backend
    assert state.warm_replay_enabled is warm_replay
    restored = SharedState.from_dict(state.to_dict())
    assert restored.agentx_backend == backend
    assert restored.agentx_epoch == epoch
    assert restored.warm_replay_enabled is warm_replay
