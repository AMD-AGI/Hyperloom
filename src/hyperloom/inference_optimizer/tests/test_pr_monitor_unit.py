# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the slimmed PRMonitorClient stub."""

from __future__ import annotations


from hyperloom.orchestrator.knowledge import pr_monitor as pm


# ---- from_args / config ---------------------------------------------------
def test_from_args_env_resolution(monkeypatch):
    monkeypatch.delenv("PR_MONITOR_URL", raising=False)
    monkeypatch.setenv("PRIMUS_CORTEX_PR_URL", "http://env-host/v1/")
    c = pm.PRMonitorClient.from_args()
    assert c.base_url == "http://env-host/v1"  # trailing slash stripped


def test_from_args_explicit_url():
    c = pm.PRMonitorClient.from_args(url="http://x/v1")
    assert c.base_url == "http://x/v1"


def test_from_args_timeout_sec_ignored():
    # timeout_sec is accepted for compat but not stored
    c = pm.PRMonitorClient.from_args(url="http://x", timeout_sec=2.5)
    assert c.base_url == "http://x"


def test_from_args_disabled():
    c = pm.PRMonitorClient.from_args(enabled=False)
    assert c.enabled is False


def test_enabled_true_by_default():
    c = pm.PRMonitorClient.from_args(url="http://x")
    assert c.enabled is True
