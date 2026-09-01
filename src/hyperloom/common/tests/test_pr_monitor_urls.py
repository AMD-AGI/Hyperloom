from hyperloom.common.pr_monitor_urls import (
    pr_monitor_base_url,
    pr_monitor_mcp_url,
    pr_monitor_rest_url,
)


def test_pr_monitor_urls_derive_from_kb_store_url() -> None:
    env = {"KB_STORE_URL": "https://global.example/knowledge-base/"}

    assert pr_monitor_base_url(env=env) == "https://global.example/knowledge-base/pr-monitor"
    assert pr_monitor_rest_url(env=env) == "https://global.example/knowledge-base/pr-monitor/v1"
    assert pr_monitor_mcp_url(env=env) == "https://global.example/knowledge-base/pr-monitor/mcp/"


def test_pr_monitor_urls_are_empty_without_kb_store() -> None:
    assert pr_monitor_base_url(env={}) == ""
    assert pr_monitor_rest_url(env={}) == ""
    assert pr_monitor_mcp_url(env={}) == ""
