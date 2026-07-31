# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the AgentX execution-boundary helper maybe_prepare_agentx.

The in-place _run_magpie hook self-disables under pytest, so the deploy +
preflight logic is factored here and tested directly.
"""

from __future__ import annotations

import pytest
import yaml

from hyperloom.inference_optimizer.agentx import runtime
from hyperloom.inference_optimizer.agentx.preflight import AgentXPreflightError

_DEPLOY = "hyperloom.inference_optimizer.agentx.deploy.deploy_agentx_assets"
_RESOLVE = "hyperloom.inference_optimizer.agentx.preflight.resolve_aiperf_bin"
_CHECK = "hyperloom.inference_optimizer.agentx.preflight.check_aiperf_capability"


@pytest.fixture(autouse=True)
def _clear_memo():
    runtime._PREFLIGHTED_BINS.clear()
    yield
    runtime._PREFLIGHTED_BINS.clear()


def _cfg(tmp_path, script):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        yaml.safe_dump({"benchmark": {"framework": "vllm", "benchmark_script": script}}),
        encoding="utf-8",
    )
    return p


def test_noop_when_not_aiperf_script(tmp_path, monkeypatch):
    calls = {"deploy": 0, "preflight": 0}
    monkeypatch.setattr(_DEPLOY, lambda d: calls.__setitem__("deploy", calls["deploy"] + 1))
    monkeypatch.setattr(_CHECK, lambda b, **k: calls.__setitem__("preflight", calls["preflight"] + 1))
    cfg = _cfg(tmp_path, "vllm_mi300x.sh")
    assert runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=cfg) is False
    assert calls == {"deploy": 0, "preflight": 0}


def test_deploy_before_preflight_with_resolved_bin(tmp_path, monkeypatch):
    order = []
    monkeypatch.setattr(_DEPLOY, lambda d: order.append("deploy"))
    monkeypatch.setattr(_RESOLVE, lambda env: "/venv/bin/aiperf")
    monkeypatch.setattr(_CHECK, lambda b, **k: order.append(("preflight", b)))
    cfg = _cfg(tmp_path, "aiperf_client.sh")
    assert runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=cfg) is True
    assert order == ["deploy", ("preflight", "/venv/bin/aiperf")]


def test_preflight_memoized_per_bin(tmp_path, monkeypatch):
    n = {"p": 0}
    monkeypatch.setattr(_DEPLOY, lambda d: None)
    monkeypatch.setattr(_RESOLVE, lambda env: "/b/aiperf")
    monkeypatch.setattr(_CHECK, lambda b, **k: n.__setitem__("p", n["p"] + 1))
    cfg = _cfg(tmp_path, "aiperf_client.sh")
    runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=cfg)
    runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=cfg)
    assert n["p"] == 1  # second call reuses the memoized capability result


def test_incapable_bin_not_memoized(tmp_path, monkeypatch):
    n = {"p": 0}

    def _raise(b, **k):
        n["p"] += 1
        raise AgentXPreflightError("no weka-trace")

    monkeypatch.setattr(_DEPLOY, lambda d: None)
    monkeypatch.setattr(_RESOLVE, lambda env: "/b/aiperf")
    monkeypatch.setattr(_CHECK, _raise)
    cfg = _cfg(tmp_path, "aiperf_client.sh")
    with pytest.raises(AgentXPreflightError):
        runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=cfg)
    with pytest.raises(AgentXPreflightError):
        runtime.maybe_prepare_agentx(env={}, inferencex_path=str(tmp_path), config_path=cfg)
    assert n["p"] == 2  # a failed preflight is re-checked, not memoized
