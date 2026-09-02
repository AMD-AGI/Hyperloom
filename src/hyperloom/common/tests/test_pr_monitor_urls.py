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
