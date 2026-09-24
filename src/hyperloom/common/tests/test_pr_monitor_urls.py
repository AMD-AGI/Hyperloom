import pytest

from hyperloom.common.env import EnvValueError
from hyperloom.common.pr_monitor_urls import (
    DEFAULT_KB_STORE_URL,
    pr_monitor_base_url,
    pr_monitor_enabled,
    pr_monitor_mcp_url,
    pr_monitor_rest_url,
)


def test_pr_monitor_urls_derive_from_kb_store_url() -> None:
    env = {"KB_STORE_URL": "https://global.example/knowledge-base/"}

    assert pr_monitor_base_url(env=env) == "https://global.example/knowledge-base/pr-monitor"
    assert pr_monitor_rest_url(env=env) == "https://global.example/knowledge-base/pr-monitor/v1"
    assert pr_monitor_mcp_url(env=env) == "https://global.example/knowledge-base/pr-monitor/mcp/"


def test_local_mode_uses_default_kb_store_url() -> None:
    assert pr_monitor_base_url(env={}) == f"{DEFAULT_KB_STORE_URL}/pr-monitor"
    assert pr_monitor_rest_url(env={}) == f"{DEFAULT_KB_STORE_URL}/pr-monitor/v1"
    assert pr_monitor_mcp_url(env={}) == f"{DEFAULT_KB_STORE_URL}/pr-monitor/mcp/"


def test_remote_mode_does_not_default_missing_kb_store_url() -> None:
    env = {"KNOWLEDGE_STORE_MODE": "remote"}
    assert pr_monitor_base_url(env=env) == ""


def test_runtime_disable_marker_is_separate_from_url_derivation() -> None:
    env = {
        "KB_STORE_URL": "https://kb.example/knowledge-base",
        "HYPERLOOM_PR_MONITOR_ENABLED": "0",
    }
    assert pr_monitor_enabled(env) is False
    assert pr_monitor_base_url(env=env) == "https://kb.example/knowledge-base/pr-monitor"


def test_an_unreadable_disable_marker_is_not_read_as_enabled() -> None:
    """The marker is written by preflight; a value it cannot spell means the two sides disagree on the run."""
    with pytest.raises(EnvValueError, match="HYPERLOOM_PR_MONITOR_ENABLED"):
        pr_monitor_enabled({"HYPERLOOM_PR_MONITOR_ENABLED": "ture"})
