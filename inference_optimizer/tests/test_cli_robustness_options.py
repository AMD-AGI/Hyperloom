# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ``inference_optimizer.cli`` robustness backend wiring.

Covers ``_build_robustness_options`` (multi-node ``--nodes >= 2`` cluster
policy: disable local probe, enable cluster pod metrics, turn off the
127.0.0.1:8888 inference probe, lift the no_levers floor to 60 min) and
``_resolve_robustness_choice`` (multi-node auto-downgrade to mock).
"""

from __future__ import annotations

import argparse

import pytest

from inference_optimizer.cli import (
    _build_robustness_options,
    _resolve_robustness_choice,
)


_WORKLOAD_ENV_KEYS = (
    "ROBUSTNESS_WORKLOAD_UID",
    "CLAW_WORKLOAD_UID",
    "WORKLOAD_UID",
    "KUBE_WORKLOAD_UID",
    "RAY_JOB_ID",
)


@pytest.fixture(autouse=True)
def _clear_workload_env(monkeypatch):
    """Keep workload-uid + server-url env discovery deterministic."""
    for key in _WORKLOAD_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("ROBUSTNESS_SERVER_URL", raising=False)


def _ns(**overrides) -> argparse.Namespace:
    """Build a namespace matching what the argparse parser would produce."""
    base = dict(
        nodes=1,
        robustness_server_url=None,
        robustness_llm_rca=None,
        robustness_backend=None,
        robustness_workload_uid=None,
        robustness_disable_local_probe=None,
        robustness_enable_cluster_pod_metrics=None,
        robustness_pod_metrics_categories=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# _build_robustness_options — multi-node cluster policy

def test_single_node_emits_no_multi_node_options():
    """Default 1-node call passes nothing extra into request.options."""
    options = _build_robustness_options(_ns())
    assert options == {}


def test_single_node_passes_server_url_and_llm_rca():
    """Existing operator-supplied flags still propagate verbatim."""
    options = _build_robustness_options(_ns(
        nodes=1,
        robustness_server_url="http://robustness.svc:8080",
        robustness_llm_rca=True,
    ))
    assert options == {
        "robustness_server_url": "http://robustness.svc:8080",
        "llm_rca_enabled": True,
    }


def test_multi_node_auto_enables_disable_local_probe_and_pod_metrics():
    """``--nodes >= 2`` auto-enables the cluster-only policy."""
    options = _build_robustness_options(_ns(nodes=4))
    assert options["nodes"] == 4
    assert options["disable_local_probe"] is True
    assert options["enable_cluster_pod_metrics"] is True


def test_multi_node_disables_inference_probe_and_bumps_floor():
    """``nodes >= 2`` → inference probe off plus the full cluster policy and the 60 min no_levers floor."""
    options = _build_robustness_options(_ns(nodes=2))
    assert options == {
        "nodes": 2,
        "disable_local_probe": True,
        "enable_cluster_pod_metrics": True,
        "auto_probe_inference_server": False,
        "progress_no_levers_min_minutes": 60.0,
    }


def test_multi_node_bumps_no_levers_floor_to_60_minutes():
    """``nodes >= 2`` → progress_no_levers_min_minutes=60.0; single-node keeps the default (key absent)."""
    multi = _build_robustness_options(_ns(nodes=2))
    assert multi["progress_no_levers_min_minutes"] == 60.0
    single = _build_robustness_options(_ns(nodes=1))
    assert "progress_no_levers_min_minutes" not in single


def test_multi_node_respects_explicit_opt_out():
    """Operator can override the multi-node cluster defaults explicitly."""
    options = _build_robustness_options(
        _ns(
            nodes=2,
            robustness_disable_local_probe=False,
            robustness_enable_cluster_pod_metrics=False,
        )
    )
    assert options["disable_local_probe"] is False
    assert options["enable_cluster_pod_metrics"] is False


def test_multi_node_preserves_operator_flags():
    """Multi-node auto-disable must coexist with explicit operator flags."""
    options = _build_robustness_options(_ns(
        nodes=4,
        robustness_server_url="http://robustness.svc:8080",
    ))
    assert options == {
        "robustness_server_url": "http://robustness.svc:8080",
        "nodes": 4,
        "disable_local_probe": True,
        "enable_cluster_pod_metrics": True,
        "auto_probe_inference_server": False,
        "progress_no_levers_min_minutes": 60.0,
    }


def test_workload_uid_cli_flag_wins_over_env(monkeypatch):
    monkeypatch.setenv("CLAW_WORKLOAD_UID", "env-uid")
    options = _build_robustness_options(
        _ns(nodes=2, robustness_workload_uid="cli-uid"),
    )
    assert options["workload_uid"] == "cli-uid"


def test_workload_uid_env_fallback(monkeypatch):
    monkeypatch.setenv("CLAW_WORKLOAD_UID", "env-uid")
    options = _build_robustness_options(_ns(nodes=2))
    assert options["workload_uid"] == "env-uid"


def test_pod_metrics_categories_csv_is_split():
    options = _build_robustness_options(
        _ns(robustness_pod_metrics_categories=" gpu, memory ,  ,disk"),
    )
    assert options["pod_metrics_categories"] == ["gpu", "memory", "disk"]


def test_missing_nodes_attr_treated_as_single_node():
    """A Namespace without ``nodes`` must not crash and defaults to single-node semantics."""
    ns = argparse.Namespace(
        robustness_server_url=None,
        robustness_llm_rca=None,
    )
    options = _build_robustness_options(ns)
    assert "auto_probe_inference_server" not in options
    assert "progress_no_levers_min_minutes" not in options
    assert "disable_local_probe" not in options


def test_nodes_zero_or_none_treated_as_single_node():
    """``nodes=0`` and ``nodes=None`` safely degrade to single-node."""
    for options in (
        _build_robustness_options(_ns(nodes=0)),
        _build_robustness_options(_ns(nodes=None)),
    ):
        assert "auto_probe_inference_server" not in options
        assert "progress_no_levers_min_minutes" not in options
        assert "disable_local_probe" not in options


# _resolve_robustness_choice — multi-node auto-downgrade to mock

def test_resolve_choice_single_node_default_keeps_agent():
    """Default path on single-node must stay ``"agent"`` so the real
    LocalProbe coverage is preserved on hosts where the inference server
    / ray actually live in the sandbox container."""
    ns = _ns(nodes=1, robustness_backend=None)
    assert _resolve_robustness_choice(ns) == "agent"


def test_resolve_choice_single_node_explicit_mock_kept():
    """Explicit ``--robustness-mock`` on single-node passes through."""
    ns = _ns(nodes=1, robustness_backend="mock")
    assert _resolve_robustness_choice(ns) == "mock"


def test_resolve_choice_multi_node_no_server_default_downgrades_to_mock(capsys):
    """``nodes >= 2`` + default agent + no robustness-server → mock,
    silently (no cluster source available, so the agent would fall back
    to the noisy sandbox-local LocalProbe)."""
    ns = _ns(nodes=2, robustness_backend=None)
    chosen = _resolve_robustness_choice(ns)
    assert chosen == "mock"
    captured = capsys.readouterr()
    assert "WARN" not in captured.err
    assert "WARN" not in captured.out


def test_resolve_choice_multi_node_no_server_explicit_agent_warns(capsys):
    """``nodes >= 2`` + explicit ``--robustness-agent`` + no server →
    mock with a WARNING that points operators at the SKILL section and
    tells them to configure a server to keep the agent."""
    ns = _ns(nodes=2, robustness_backend="agent")
    chosen = _resolve_robustness_choice(ns)
    assert chosen == "mock"
    captured = capsys.readouterr()
    assert "WARN" in captured.err
    assert "auto-downgrad" in captured.err.lower()
    assert "multi_node/SKILL.md" in captured.err


def test_resolve_choice_multi_node_with_server_url_keeps_agent(capsys):
    """``nodes >= 2`` + explicit agent + ``--robustness-server-url`` →
    stays ``agent``: the cluster source replaces the sandbox-local
    probes, so no downgrade and no warning."""
    ns = _ns(
        nodes=2,
        robustness_backend="agent",
        robustness_server_url="http://robustness.svc:8080",
    )
    chosen = _resolve_robustness_choice(ns)
    assert chosen == "agent"
    captured = capsys.readouterr()
    assert "WARN" not in captured.err


def test_resolve_choice_multi_node_default_with_server_keeps_agent():
    """``nodes >= 2`` + default backend + configured server → ``agent``."""
    ns = _ns(
        nodes=2,
        robustness_backend=None,
        robustness_server_url="http://robustness.svc:8080",
    )
    assert _resolve_robustness_choice(ns) == "agent"


def test_resolve_choice_multi_node_server_via_env_keeps_agent(monkeypatch):
    """A server configured via ``ROBUSTNESS_SERVER_URL`` env also keeps
    the agent backend on multi-node."""
    monkeypatch.setenv("ROBUSTNESS_SERVER_URL", "http://robustness.svc:8080")
    ns = _ns(nodes=2, robustness_backend="agent")
    assert _resolve_robustness_choice(ns) == "agent"


def test_resolve_choice_multi_node_explicit_mock_no_warning(capsys):
    """Operators who anticipate the auto-downgrade and pass
    ``--robustness-mock`` explicitly must NOT see the WARNING."""
    ns = _ns(nodes=4, robustness_backend="mock")
    chosen = _resolve_robustness_choice(ns)
    assert chosen == "mock"
    captured = capsys.readouterr()
    assert "WARN" not in captured.err


def test_resolve_choice_missing_nodes_attr_treated_as_single_node():
    """Legacy entry points that omit ``nodes`` keep the agent default."""
    ns = argparse.Namespace(robustness_backend=None)
    assert _resolve_robustness_choice(ns) == "agent"


def test_resolve_choice_nodes_zero_or_none_treated_as_single_node():
    """``nodes=0`` / ``nodes=None`` must NOT trip the multi-node downgrade."""
    for ns in (
        _ns(nodes=0, robustness_backend="agent"),
        _ns(nodes=None, robustness_backend="agent"),
    ):
        assert _resolve_robustness_choice(ns) == "agent"
