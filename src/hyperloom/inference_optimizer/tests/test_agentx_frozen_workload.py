# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Candidate materialization preserves the operator's AgentX replay protocol."""

from __future__ import annotations

import json
import os

import pytest
import yaml

from hyperloom.orchestrator.actions.executors._grid_base import GridVariant
from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml
from hyperloom.orchestrator.actions.executors._workload_envs import materialize_config_with_envs
from hyperloom.orchestrator.loop.coordinator_helpers import _accepted_config_as_variant


@pytest.fixture
def operator_config(tmp_path, monkeypatch):
    for key in os.environ:
        if key.startswith(("AGENTX_", "AIPERF_", "INFERENCE_OPTIMIZER_")) or key == "WEKA_LOADER_OVERRIDE":
            monkeypatch.delenv(key)
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "sessions"))
    monkeypatch.setenv("CONC", "32")
    monkeypatch.setenv("TP", "4")
    monkeypatch.setenv("AGENTX_WARMUP_REQUESTS_PER_LANE", "10")
    monkeypatch.setenv("AGENTX_LIVE_ASSISTANT", "0")
    source = tmp_path / "base.yaml"
    source.write_text(yaml.safe_dump({"benchmark": {"framework": "vllm", "model": "MiniMax-M3-MXFP4", "envs": {}}}))
    return source


def _candidate(source, output, path, extra_envs=None, unset_envs=None):
    flags, accepted = _accepted_config_as_variant({"env_map": extra_envs or {}})
    variant = GridVariant("geak_revalidate", flags, accepted, unset_envs=unset_envs)
    if path == "direct":
        return materialize_config_with_envs(
            source, output, extra_envs=variant.extra_envs, unset_envs=variant.unset_envs, agentx_mode=True
        )
    baseline = materialize_config_with_envs(source, output.parent / "baseline", agentx_mode=True)
    return _build_variant_yaml(baseline, "", variant, output_subdir=output)


@pytest.mark.parametrize("path", ["direct", "grid"])
@pytest.mark.parametrize(
    "override",
    [
        {"AGENTX_WARMUP_REQUESTS_PER_LANE": "20"},
        {"AGENTX_LIVE_ASSISTANT": "1"},
        {"AGENTX_WARMUP_REQUESTS_PER_LANE": "20", "AGENTX_LIVE_ASSISTANT": "1"},
        {"AGENTX_DURATION": "900"},
        {"AGENTX_MAX_CTX": "8192"},
        {"WEKA_LOADER_OVERRIDE": "other-corpus"},
        {"AGENTX_SERVER_SCRIPT": "other-client.sh"},
        {"AGENTX_REALTIME_METRICS": "false"},
    ],
)
def test_candidate_cannot_change_or_introduce_replay_controls(operator_config, tmp_path, path, override):
    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match="frozen AgentX workload controls") as error:
        _candidate(operator_config, output, path, override)
    assert all(key in str(error.value) for key in override)
    assert not output.exists()


@pytest.mark.parametrize("path", ["direct", "grid"])
def test_candidate_cannot_unset_operator_protocol(operator_config, tmp_path, path):
    with pytest.raises(ValueError, match="AGENTX_WARMUP_REQUESTS_PER_LANE"):
        _candidate(operator_config, tmp_path / "candidate", path, unset_envs=["AGENTX_WARMUP_REQUESTS_PER_LANE"])


@pytest.mark.parametrize("path", ["direct", "grid"])
def test_identical_inherited_values_and_engine_tuning_remain_valid(operator_config, tmp_path, path):
    baseline = materialize_config_with_envs(operator_config, tmp_path / "reference", agentx_mode=True)
    before = yaml.safe_load(baseline.read_text())["benchmark"]
    extra = {
        "AGENTX_WARMUP_REQUESTS_PER_LANE": "10",
        "AGENTX_LIVE_ASSISTANT": "0",
        "VLLM_USE_TRITON_FLASH_ATTN": "0",
        "AGENTX_CAPTURE_ID": "profile-capture",
    }
    candidate = _candidate(operator_config, tmp_path / "candidate", path, extra)
    after = yaml.safe_load(candidate.read_text())["benchmark"]
    assert after["workload_spec"] == before["workload_spec"]
    assert {key: after["envs"][key] for key in extra} == extra
    assert after["workload_spec"]["warmup_requests_per_lane"] == 10


@pytest.mark.parametrize("path", ["direct", "grid"])
@pytest.mark.parametrize("source", ["yaml", "process", "cli_extra_env"])
def test_initial_operator_protocol_is_preserved(operator_config, tmp_path, monkeypatch, path, source):
    pins = {"AGENTX_WARMUP_REQUESTS_PER_LANE": "20", "AGENTX_LIVE_ASSISTANT": "1"}
    if source == "yaml":
        for key in pins:
            monkeypatch.delenv(key)
        config = yaml.safe_load(operator_config.read_text())
        config["benchmark"]["envs"].update(pins)
        operator_config.write_text(yaml.safe_dump(config))
    elif source == "process":
        for key, value in pins.items():
            monkeypatch.setenv(key, value)
    else:
        monkeypatch.setenv("INFERENCE_OPTIMIZER_EXTRA_ENV", json.dumps(pins))
    candidate = _candidate(operator_config, tmp_path / "candidate", path, pins)
    bench = yaml.safe_load(candidate.read_text())["benchmark"]
    assert {key: bench["envs"][key] for key in pins} == pins
    assert bench["workload_spec"]["warmup_requests_per_lane"] == 20
    with pytest.raises(ValueError, match="AGENTX_LIVE_ASSISTANT"):
        _candidate(operator_config, tmp_path / "changed", path, {"AGENTX_LIVE_ASSISTANT": "0"})


def test_grid_base_cannot_replace_frozen_protocol(operator_config, tmp_path):
    baseline = materialize_config_with_envs(operator_config, tmp_path / "baseline", agentx_mode=True)
    with pytest.raises(ValueError, match="AGENTX_LIVE_ASSISTANT"):
        _build_variant_yaml(
            baseline,
            "",
            GridVariant("inherited"),
            output_subdir=tmp_path / "candidate",
            base_extra_envs={"AGENTX_LIVE_ASSISTANT": "1"},
        )


def test_cli_operator_warmup_grace_is_scaled_once(operator_config, tmp_path, monkeypatch):
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_EXTRA_ENV",
        json.dumps({"AGENTX_WARMUP_GRACE_PERIOD": "1800", "AGENTX_WARMUP_GRACE_CONC": "8"}),
    )
    candidate = _candidate(operator_config, tmp_path / "candidate", "grid")
    bench = yaml.safe_load(candidate.read_text())["benchmark"]
    assert bench["envs"]["AGENTX_WARMUP_GRACE_PERIOD"] == "7200"
    assert bench["workload_spec"]["warmup_grace_period_s"] == 7200
