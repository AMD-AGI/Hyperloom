# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for :mod:`runtime.web_tools.config`: env normalization, provider chain, ``has_search_api_key`` guard."""

from __future__ import annotations

import pytest

from runtime.web_tools.config import WebToolsConfig, _normalize_provider


def test_defaults_disable_everything(monkeypatch):
    for var in (
        "CRITIC_WEB_TOOLS_ENABLED",
        "WEB_SEARCH_PROVIDER",
        "WEB_SEARCH_FALLBACK",
        "WEB_FETCH_ENABLED",
        "TAVILY_API_KEY",
        "SERPER_API_KEY",
        "BRAVE_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = WebToolsConfig.from_env()
    assert cfg.critic_web_tools_enabled is False
    assert cfg.search_provider == "disabled"
    assert cfg.search_fallback == ()
    assert cfg.fetch_enabled is False
    assert cfg.search_provider_chain() == ()


def test_provider_chain_drops_dupes_and_disabled(monkeypatch):
    monkeypatch.setenv("CRITIC_WEB_TOOLS_ENABLED", "true")
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("WEB_SEARCH_FALLBACK", "tavily, serper, disabled,brave")
    cfg = WebToolsConfig.from_env()
    assert cfg.search_provider_chain() == ("tavily", "serper", "brave")


def test_provider_chain_only_fallback_when_primary_disabled(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "disabled")
    monkeypatch.setenv("WEB_SEARCH_FALLBACK", "tavily,serper")
    cfg = WebToolsConfig.from_env()
    assert cfg.search_provider == "disabled"
    assert cfg.search_provider_chain() == ("tavily", "serper")


def test_unknown_provider_becomes_disabled(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_PROVIDER", "DuckDuckGo")
    monkeypatch.setenv("WEB_SEARCH_FALLBACK", "bing,tavily")
    cfg = WebToolsConfig.from_env()
    assert cfg.search_provider == "disabled"
    # Unknown fallback entries are dropped; only tavily survives.
    assert cfg.search_provider_chain() == ("tavily",)


def test_has_search_api_key_checks_each_provider(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tk")
    monkeypatch.setenv("SERPER_API_KEY", "")
    cfg = WebToolsConfig.from_env()
    assert cfg.has_search_api_key("tavily") is True
    assert cfg.has_search_api_key("serper") is False
    assert cfg.has_search_api_key("brave") is False
    assert cfg.has_search_api_key("unknown") is False


def test_csv_normalization_lowercases_and_trims(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_DOMAIN_DENYLIST", "  Example.COM, , spam.io ")
    monkeypatch.setenv("WEB_FETCH_DOMAIN_DENYLIST", "INTERNAL.local")
    cfg = WebToolsConfig.from_env()
    assert cfg.search_domain_denylist == ("example.com", "spam.io")
    assert cfg.fetch_domain_denylist == ("internal.local",)


def test_bool_and_int_parsing(monkeypatch):
    monkeypatch.setenv("CRITIC_WEB_TOOLS_ENABLED", "Yes")
    monkeypatch.setenv("WEB_FETCH_ENABLED", "on")
    monkeypatch.setenv("CRITIC_WEB_MAX_TOOL_TURNS", "0")  # clamped to 1
    monkeypatch.setenv("WEB_FETCH_MAX_BYTES", "not-an-int")
    cfg = WebToolsConfig.from_env()
    assert cfg.critic_web_tools_enabled is True
    assert cfg.fetch_enabled is True
    assert cfg.critic_web_max_tool_turns == 1
    # invalid int -> default
    assert cfg.fetch_max_bytes == 10 * 1024 * 1024


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("tavily", "tavily"),
        ("  TAVILY ", "tavily"),
        ("anthropic", "disabled"),  # critic-agent does not ship anthropic native
        ("", "disabled"),
    ],
)
def test_normalize_provider_table(raw, expected):
    assert _normalize_provider(raw) == expected
