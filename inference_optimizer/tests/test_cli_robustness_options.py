"""Unit tests for the multi-node robustness CLI option plumbing.

Exercises :func:`inference_optimizer.cli._build_robustness_options` —
the small function that translates CLI flags + env vars into the
``request.options`` payload sent to ``robustness-agent``'s runtime.
The multi-node policy lives here: when ``--nodes >= 2`` the agent
must disable its local sandbox probe and lean on robustness-server
for cluster pod metrics; this test pins that contract so a future
refactor cannot silently regress to the false-positive M1 behaviour.
"""

from __future__ import annotations

import argparse

import pytest

from inference_optimizer.cli import _build_robustness_options


def _ns(**overrides) -> argparse.Namespace:
    """Build a namespace matching what the argparse parser would produce."""
    base = dict(
        robustness_server_url=None,
        robustness_llm_rca=None,
        nodes=1,
        robustness_workload_uid=None,
        robustness_disable_local_probe=None,
        robustness_enable_cluster_pod_metrics=None,
        robustness_pod_metrics_categories=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_single_node_emits_no_multi_node_options(monkeypatch):
    """Default 1-node call passes nothing extra into request.options."""
    for key in (
        "ROBUSTNESS_WORKLOAD_UID",
        "CLAW_WORKLOAD_UID",
        "WORKLOAD_UID",
        "KUBE_WORKLOAD_UID",
        "RAY_JOB_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    options = _build_robustness_options(_ns())
    assert "disable_local_probe" not in options
    assert "enable_cluster_pod_metrics" not in options
    assert "nodes" not in options
    assert "workload_uid" not in options


def test_multi_node_auto_enables_disable_local_probe_and_pod_metrics(monkeypatch):
    """``--nodes >= 2`` auto-enables the cluster-only policy."""
    for key in (
        "ROBUSTNESS_WORKLOAD_UID",
        "CLAW_WORKLOAD_UID",
        "WORKLOAD_UID",
        "KUBE_WORKLOAD_UID",
        "RAY_JOB_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    options = _build_robustness_options(_ns(nodes=4))
    assert options["nodes"] == 4
    assert options["disable_local_probe"] is True
    assert options["enable_cluster_pod_metrics"] is True


def test_multi_node_respects_explicit_opt_out(monkeypatch):
    """Operator can override the multi-node defaults explicitly."""
    for key in (
        "ROBUSTNESS_WORKLOAD_UID",
        "CLAW_WORKLOAD_UID",
        "WORKLOAD_UID",
        "KUBE_WORKLOAD_UID",
        "RAY_JOB_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    options = _build_robustness_options(
        _ns(
            nodes=2,
            robustness_disable_local_probe=False,
            robustness_enable_cluster_pod_metrics=False,
        )
    )
    assert options["disable_local_probe"] is False
    assert options["enable_cluster_pod_metrics"] is False


def test_workload_uid_cli_flag_wins_over_env(monkeypatch):
    monkeypatch.setenv("CLAW_WORKLOAD_UID", "env-uid")
    options = _build_robustness_options(
        _ns(nodes=2, robustness_workload_uid="cli-uid"),
    )
    assert options["workload_uid"] == "cli-uid"


def test_workload_uid_env_fallback(monkeypatch):
    monkeypatch.delenv("ROBUSTNESS_WORKLOAD_UID", raising=False)
    monkeypatch.setenv("CLAW_WORKLOAD_UID", "env-uid")
    monkeypatch.delenv("WORKLOAD_UID", raising=False)
    monkeypatch.delenv("KUBE_WORKLOAD_UID", raising=False)
    monkeypatch.delenv("RAY_JOB_ID", raising=False)
    options = _build_robustness_options(_ns(nodes=2))
    assert options["workload_uid"] == "env-uid"


def test_pod_metrics_categories_csv_is_split(monkeypatch):
    for key in (
        "ROBUSTNESS_WORKLOAD_UID",
        "CLAW_WORKLOAD_UID",
        "WORKLOAD_UID",
        "KUBE_WORKLOAD_UID",
        "RAY_JOB_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    options = _build_robustness_options(
        _ns(robustness_pod_metrics_categories=" gpu, memory ,  ,disk"),
    )
    assert options["pod_metrics_categories"] == ["gpu", "memory", "disk"]


def test_existing_server_url_and_llm_rca_pass_through():
    """Pre-existing option keys still ride through unchanged."""
    options = _build_robustness_options(
        _ns(
            robustness_server_url="http://example.invalid:8000",
            robustness_llm_rca=True,
        )
    )
    assert options["robustness_server_url"] == "http://example.invalid:8000"
    assert options["llm_rca_enabled"] is True
