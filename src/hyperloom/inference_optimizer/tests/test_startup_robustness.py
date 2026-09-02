# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the startup-robustness preflight + launch-info wire format (cli.py)."""

from __future__ import annotations

import argparse
import json
import os
import time

import pytest

from hyperloom.inference_optimizer import cli
from hyperloom.inference_optimizer.cli import credentials as cli_credentials
from hyperloom.inference_optimizer.cli import model_gate as cli_model_gate
from hyperloom.inference_optimizer.cli.parser import _build_parser

_OAUTH_ENV = "_".join(("CLAUDE", "CODE", "OAUTH", "TOKEN"))


# _validate_credentials
@pytest.fixture
def clean_creds_env(monkeypatch):
    for var in (
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "_".join(("OPENAI", "API", "KEY")),
        "_".join(("ANTHROPIC", "API", "KEY")),
        "_".join(("ANTHROPIC", "AUTH", "TOKEN")),
        "_".join(("DEEPSEEK", "API", "KEY")),
        "DEEPSEEK_BASE_URL",
        _OAUTH_ENV,
        "HYPERLOOM_AGENT_BACKEND",
        "HYPERLOOM_HERMES_PROFILE",
        "HYPERLOOM_HERMES_BIN",
        "HERMES_HOME",
    ):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_validate_credentials_passes_single_gateway(clean_creds_env):
    """Single-gateway pair (OPENAI_API_KEY + OPENAI_BASE_URL) passes."""
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "fake-token")
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    cli_credentials._validate_credentials()


def test_validate_credentials_passes_anthropic_only_entrypoint(clean_creds_env):
    """Split entrypoint: only the Anthropic side configured is enough."""
    clean_creds_env.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-fake-token")
    cli_credentials._validate_credentials()


def test_validate_credentials_rejects_openai_key_with_anthropic_url(clean_creds_env, capsys):
    """A URL on one side paired with only the other side's key is a mispairing:
    the OpenAI key would be sent to the Anthropic host."""
    clean_creds_env.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "openai-fake-token")
    with pytest.raises(SystemExit) as exc_info:
        cli_credentials._validate_credentials()
    assert exc_info.value.code == 2
    assert "Conflicting LLM credentials" in capsys.readouterr().err


def test_validate_credentials_passes_anthropic_auth_token_only(clean_creds_env):
    """ANTHROPIC_AUTH_TOKEN alone satisfies the check."""
    clean_creds_env.setenv("_".join(("ANTHROPIC", "AUTH", "TOKEN")), "anthropic-fake-token")
    cli_credentials._validate_credentials()


def test_validate_credentials_passes_official_anthropic_key_only(clean_creds_env):
    """Official Anthropic SDK default endpoint works without ANTHROPIC_BASE_URL."""
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-fake-token")
    cli_credentials._validate_credentials()


def test_validate_credentials_passes_official_openai_key_only(clean_creds_env):
    """Official OpenAI SDK default endpoint works without OPENAI_BASE_URL."""
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "openai-fake-token")
    cli_credentials._validate_credentials()


def test_validate_credentials_passes_oauth_token_only(clean_creds_env):
    """A Max/Pro subscription token alone is a complete Anthropic-side setup."""
    clean_creds_env.setenv(_OAUTH_ENV, "sk-ant-oat01-fake")
    cli_credentials._validate_credentials()


def test_validate_credentials_passes_explicit_hermes_profile(clean_creds_env, tmp_path, monkeypatch):
    profile = tmp_path / "profiles" / "faithful"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    clean_creds_env.setenv("HYPERLOOM_AGENT_BACKEND", "hermes")
    clean_creds_env.setenv("HYPERLOOM_HERMES_PROFILE", "faithful")
    clean_creds_env.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cli_credentials, "resolve_hermes_executable", lambda: "/usr/bin/hermes")
    cli_credentials._validate_credentials()


def test_validate_credentials_rejects_missing_hermes_profile(clean_creds_env, tmp_path, monkeypatch):
    clean_creds_env.setenv("HYPERLOOM_AGENT_BACKEND", "hermes")
    clean_creds_env.setenv("HYPERLOOM_HERMES_PROFILE", "missing")
    clean_creds_env.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cli_credentials, "resolve_hermes_executable", lambda: "/usr/bin/hermes")
    with pytest.raises(SystemExit) as exc_info:
        cli_credentials._validate_credentials()
    assert exc_info.value.code == 2


def test_resolve_llm_endpoints_oauth_only_implies_official_anthropic(clean_creds_env):
    """OAuth-only derives the official endpoint and leaves the OpenAI side empty."""
    clean_creds_env.setenv(_OAUTH_ENV, "sk-ant-oat01-fake")
    anthropic_url, openai_url = cli_credentials._resolve_llm_endpoints()
    assert anthropic_url == "https://api.anthropic.com"
    assert openai_url == ""


def test_validate_credentials_accepts_oauth_alongside_configured_openai_side(clean_creds_env):
    """Subscription Anthropic side + fully configured OpenAI side is a legal pair."""
    clean_creds_env.setenv(_OAUTH_ENV, "sk-ant-oat01-fake")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "ak-openai")
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gw.example.com/v1")
    cli_credentials._validate_credentials()


def test_validate_credentials_rejects_oauth_with_bare_openai_base_url(clean_creds_env, capsys):
    """OAuth never vouches for the OpenAI side: its base URL still needs its own key."""
    clean_creds_env.setenv(_OAUTH_ENV, "sk-ant-oat01-fake")
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gw.example.com/v1")
    with pytest.raises(SystemExit) as exc_info:
        cli_credentials._validate_credentials()
    assert exc_info.value.code == 2
    assert "Conflicting LLM credentials" in capsys.readouterr().err


@pytest.mark.parametrize(
    "anthropic_env",
    [
        pytest.param(("_".join(("ANTHROPIC", "API", "KEY")), "sk-ant-api"), id="api-key"),
        pytest.param((_OAUTH_ENV, "sk-ant-oat01-fake"), id="oauth-token"),
        pytest.param(("_".join(("DEEPSEEK", "API", "KEY")), "sk-deepseek"), id="deepseek-key"),
    ],
)
def test_validate_credentials_accepts_implied_endpoints_on_both_sides(clean_creds_env, anthropic_env):
    """A key that implies its own official endpoint never borrows the other
    side's, so it pairs with a bare OPENAI_API_KEY."""
    clean_creds_env.setenv(*anthropic_env)
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "ak-openai")
    cli_credentials._validate_credentials()


def test_validate_credentials_rejects_gateway_anthropic_url_with_bare_openai_key(clean_creds_env, capsys):
    """An explicit ANTHROPIC_BASE_URL marks a gateway deploy, where a bare
    OPENAI_API_KEY is a gateway key that lost its OPENAI_BASE_URL."""
    clean_creds_env.setenv("ANTHROPIC_BASE_URL", "https://gw.example.com/anthropic")
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "gw-key")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "gw-key")
    with pytest.raises(SystemExit) as exc_info:
        cli_credentials._validate_credentials()
    assert exc_info.value.code == 2
    assert "Conflicting LLM credentials" in capsys.readouterr().err


def test_validate_credentials_warns_when_api_key_shadows_oauth(clean_creds_env, capsys):
    """The Claude CLI prefers the API key, so the subscription would go unused."""
    clean_creds_env.setenv(_OAUTH_ENV, "sk-ant-oat01-fake")
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "sk-ant-api")
    cli_credentials._validate_credentials()
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert _OAUTH_ENV in err
    assert "_".join(("ANTHROPIC", "API", "KEY")) in err


def test_validate_credentials_warns_when_oauth_widens_an_openai_only_deploy(clean_creds_env, capsys):
    """A stray token silently moves orchestration off the configured gateway."""
    clean_creds_env.setenv(_OAUTH_ENV, "sk-ant-oat01-fake")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "ak-openai")
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gw.example.com/v1")
    cli_credentials._validate_credentials()
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "OPENAI_BASE_URL" in err


def test_validate_credentials_does_not_warn_when_anthropic_side_was_intended(clean_creds_env, capsys):
    """An explicit Anthropic key means the dual-sided shape was deliberate."""
    clean_creds_env.setenv(_OAUTH_ENV, "sk-ant-oat01-fake")
    clean_creds_env.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "sk-ant-api")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "ak-openai")
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gw.example.com/v1")
    cli_credentials._validate_credentials()
    assert "orchestration runs on the Claude subscription" not in capsys.readouterr().err


def test_validate_credentials_oauth_only_is_not_warned_about(clean_creds_env, capsys):
    clean_creds_env.setenv(_OAUTH_ENV, "sk-ant-oat01-fake")
    cli_credentials._validate_credentials()
    assert "WARNING" not in capsys.readouterr().err


def test_validate_credentials_exits_2_when_no_key(clean_creds_env, capsys):
    """A base URL without any key is rejected."""
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    with pytest.raises(SystemExit) as exc_info:
        cli_credentials._validate_credentials()
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "API key" in err


def test_validate_credentials_lists_both_missing(clean_creds_env, capsys):
    with pytest.raises(SystemExit):
        cli_credentials._validate_credentials()
    err = capsys.readouterr().err
    missing_line = err.split("Missing required credential(s):")[1].split("\n")[0]
    assert "usable endpoint/key pair" in missing_line
    assert "API key" in missing_line


def test_validate_credentials_no_bypass_paths(clean_creds_env):
    """HYPERLOOM_SKIP_CREDS_CHECK does NOT bypass — the bypass path was removed."""
    clean_creds_env.setenv("HYPERLOOM_SKIP_CREDS_CHECK", "1")
    with pytest.raises(SystemExit) as exc_info:
        cli_credentials._validate_credentials()
    assert exc_info.value.code == 2


# _resolve_llm_endpoints
def test_resolve_llm_endpoints_openai_only_leaves_anthropic_unset(clean_creds_env):
    """Only the OpenAI side is configured: the Anthropic side stays empty rather
    than being derived from the OpenAI gateway."""
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    anthropic_url, openai_url = cli_credentials._resolve_llm_endpoints()
    assert openai_url == "https://gateway.example/v1"
    assert anthropic_url == ""


def test_resolve_llm_endpoints_anthropic_only_leaves_openai_unset(clean_creds_env):
    """Only the Anthropic side is configured: the OpenAI/Codex side stays empty
    rather than being derived from the Anthropic gateway."""
    clean_creds_env.setenv("ANTHROPIC_BASE_URL", "https://gateway.example/anthropic")
    anthropic_url, openai_url = cli_credentials._resolve_llm_endpoints()
    assert anthropic_url == "https://gateway.example/anthropic"
    assert openai_url == ""


def test_resolve_llm_endpoints_official_anthropic_key_only(clean_creds_env):
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-fake-token")
    anthropic_url, openai_url = cli_credentials._resolve_llm_endpoints()
    assert anthropic_url == "https://api.anthropic.com"
    assert openai_url == ""


def test_resolve_llm_endpoints_official_openai_key_only(clean_creds_env):
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "openai-fake-token")
    anthropic_url, openai_url = cli_credentials._resolve_llm_endpoints()
    assert anthropic_url == ""
    assert openai_url == "https://api.openai.com/v1"


def test_resolve_llm_endpoints_one_gateway_under_both_names(clean_creds_env):
    """One gateway serving both providers is configured explicitly on both sides;
    each side then resolves to its own value with no derivation involved."""
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "ak-gw")
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "ak-gw")
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gw.example.com/api/v1/llm-proxy/v1")
    clean_creds_env.setenv("ANTHROPIC_BASE_URL", "https://gw.example.com/api/v1/llm-proxy")
    anthropic_url, openai_url = cli_credentials._resolve_llm_endpoints()
    assert anthropic_url == "https://gw.example.com/api/v1/llm-proxy"
    assert openai_url == "https://gw.example.com/api/v1/llm-proxy/v1"


def test_validate_credentials_rejects_openai_gateway_with_foreign_anthropic_key(clean_creds_env, capsys):
    """A gateway URL on the OpenAI side must not be paired with only an Anthropic
    key. The check reads the raw env, before any endpoint resolution."""
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gw.example.com/v1")
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "sk-ant-real")
    with pytest.raises(SystemExit) as exc_info:
        cli_credentials._validate_credentials()
    assert exc_info.value.code == 2
    assert "Conflicting LLM credentials" in capsys.readouterr().err


def test_validate_credentials_rejects_anthropic_gateway_with_foreign_openai_key(clean_creds_env, capsys):
    """Mirror image: an Anthropic-side gateway must not be paired with a
    different OpenAI key."""
    clean_creds_env.setenv("ANTHROPIC_BASE_URL", "https://gw.example.com/anthropic")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "sk-openai-real")
    with pytest.raises(SystemExit) as exc_info:
        cli_credentials._validate_credentials()
    assert exc_info.value.code == 2
    assert "Conflicting LLM credentials" in capsys.readouterr().err


def test_validate_credentials_rejects_gateway_key_plus_foreign_anthropic_key(clean_creds_env, capsys):
    """The gateway having its own key does not excuse a second, different
    provider key riding along without its own base URL."""
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gw.example.com/v1")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "ak-gw")
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "sk-ant-real")
    with pytest.raises(SystemExit) as exc_info:
        cli_credentials._validate_credentials()
    assert exc_info.value.code == 2


def test_validate_credentials_rejects_mirrored_key_without_its_own_base_url(clean_creds_env, capsys):
    """An Anthropic-side key still needs ANTHROPIC_BASE_URL, even when its value
    matches the OpenAI-side key."""
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gw.example.com/v1")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "ak-gw")
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "ak-gw")
    with pytest.raises(SystemExit) as exc_info:
        cli_credentials._validate_credentials()
    assert exc_info.value.code == 2
    assert "Conflicting LLM credentials" in capsys.readouterr().err


def test_validate_credentials_accepts_one_gateway_configured_on_both_sides(clean_creds_env):
    """The hosted sandbox points both sides at the same gateway and sets both
    keys, so each side is self-consistent."""
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gw.example.com/v1")
    clean_creds_env.setenv("ANTHROPIC_BASE_URL", "https://gw.example.com")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "ak-gw")
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "ak-gw")
    cli_credentials._validate_credentials()


def test_validate_credentials_accepts_dual_entry_with_distinct_keys(clean_creds_env):
    """Two explicit base URLs are self-consistent: each side keeps its own key."""
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "sk-openai")
    clean_creds_env.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "sk-ant")
    cli_credentials._validate_credentials()


def test_openai_key_only_makes_claude_follow_codex_before_preflight(clean_creds_env):
    """Key-only official OpenAI must select Codex orchestration before preflight writes OPENAI_BASE_URL."""
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "openai-fake-token")
    assert cli._claude_model_should_follow_codex() is True


def test_anthropic_only_critic_agent_runtime_needed(clean_creds_env):
    """Official Anthropic-only now keeps the full critic-agent (native Anthropic
    review path), so its KB prepare/commit runtime IS required."""
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-fake-token")
    anthropic_url, openai_url = cli_credentials._resolve_llm_endpoints()
    clean_creds_env.setenv("ANTHROPIC_BASE_URL", anthropic_url)
    if openai_url:
        clean_creds_env.setenv("OPENAI_BASE_URL", openai_url)
    assert cli._critic_agent_runtime_needed("agent") is True


def test_critic_agent_runtime_always_needed_for_agent_choice(clean_creds_env):
    """Preflight may add stale/runtime OpenAI env, but the runtime is required
    either way: there is no longer a degraded critic that skips it."""
    clean_creds_env.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-fake-token")
    assert cli._codex_model_should_follow_claude() is True

    clean_creds_env.setenv("OPENAI_BASE_URL", "https://api.anthropic.com")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "stale-openai-token")

    assert cli._critic_agent_runtime_needed("agent") is True


def test_build_backends_keeps_anthropic_critic_when_codex_follows_claude(
    clean_creds_env,
    monkeypatch,
    tmp_path,
):
    """Stale OpenAI env after preflight must not force critic-agent onto Codex."""
    from hyperloom.inference_optimizer.cli import backends as cli_backends

    clean_creds_env.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-fake-token")
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://api.anthropic.com")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "stale-openai-token")

    class _FakeClaude:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeCodex:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeCriticAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(cli_backends, "ClaudeBackend", _FakeClaude)
    monkeypatch.setattr(cli_backends, "CodexBackend", _FakeCodex)
    monkeypatch.setattr(cli_backends, "CriticAgentBackend", _FakeCriticAgent)

    built = cli_backends._build_backends(
        claude_model="claude-opus-4-8",
        codex_model="stale-codex-model",
        critic_choice="agent",
        session_dir=tmp_path,
        critic_agent_root=tmp_path,
        codex_follows_claude=True,
    )

    critic = built["critic"]
    assert isinstance(critic, _FakeCriticAgent)
    assert critic.kwargs["protocol"] == "anthropic"
    assert critic.kwargs["claude_model"] == "claude-opus-4-8"


def test_openai_only_critic_agent_runtime_needed(clean_creds_env):
    """Official OpenAI-only keeps the critic-agent path."""
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "openai-fake-token")
    openai_url = cli_credentials._resolve_llm_endpoints()[1]
    clean_creds_env.setenv("OPENAI_BASE_URL", openai_url)
    assert cli._critic_agent_runtime_needed("agent") is True


def test_resolve_llm_endpoints_ignores_retired_deepseek_key(clean_creds_env):
    """``_resolve_llm_endpoints`` no longer knows DeepSeek; the shim does."""
    clean_creds_env.setenv("_".join(("DEEPSEEK", "API", "KEY")), "deepseek-fake-token")
    assert cli_credentials._resolve_llm_endpoints() == ("", "")


def test_resolve_llm_endpoints_dual_protocol_gateway_keeps_both_sides(clean_creds_env):
    """A normalized DeepSeek config is an ordinary two-sided gateway."""
    clean_creds_env.setenv("ANTHROPIC_BASE_URL", "https://deepseek.example/anthropic")
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://deepseek.example/v1")
    anthropic_url, openai_url = cli_credentials._resolve_llm_endpoints()
    assert anthropic_url == "https://deepseek.example/anthropic"
    assert openai_url == "https://deepseek.example/v1"


def test_resolve_llm_endpoints_deepseek_anthropic_only_leaves_openai_unset(clean_creds_env):
    """A DeepSeek Anthropic endpoint no longer implies anything about the other side.

    Endpoint derivation across sides was removed; when the OpenAI side matters
    the caller goes through ``derive_openai_base_url``, which knows DeepSeek
    serves ``/v1`` and not AMD's ``/Unified/v1``.
    """
    clean_creds_env.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "deepseek-fake-token")
    anthropic_url, openai_url = cli_credentials._resolve_llm_endpoints()
    assert anthropic_url == "https://api.deepseek.com/anthropic"
    assert openai_url == ""


def test_resolve_llm_endpoints_both_official_keys_no_urls(clean_creds_env):
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-fake-token")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "openai-fake-token")
    anthropic_url, openai_url = cli_credentials._resolve_llm_endpoints()
    assert anthropic_url == "https://api.anthropic.com"
    assert openai_url == "https://api.openai.com/v1"


def test_resolve_llm_endpoints_both_kept_distinct(clean_creds_env):
    """Both set: each side is respected as-is (true dual entrypoint)."""
    clean_creds_env.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    anthropic_url, openai_url = cli_credentials._resolve_llm_endpoints()
    assert anthropic_url == "https://api.anthropic.com"
    assert openai_url == "https://api.openai.com/v1"


def test_resolve_llm_endpoints_neither_set_returns_empty(clean_creds_env):
    anthropic_url, openai_url = cli_credentials._resolve_llm_endpoints()
    assert anthropic_url == ""
    assert openai_url == ""


# _validate_and_resolve_claude_model catalog candidates
def _record_probed_urls(monkeypatch, catalog: set[str]) -> list[str]:
    probed: list[str] = []

    def fake_probe(*, base_url: str, api_key: str):
        probed.append(base_url)
        return catalog if api_key else None

    monkeypatch.setattr(cli, "_probe_llm_catalog", fake_probe)
    return probed


def test_catalog_probe_skips_keyless_anthropic_side_and_uses_the_gateway(clean_creds_env):
    """A stray subscription token must not cost the gateway its verification."""
    clean_creds_env.setenv(_OAUTH_ENV, "sk-ant-oat01-fake")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "ak-openai")
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gw.example.com/v1")
    probed = _record_probed_urls(clean_creds_env, {"claude-opus-5"})
    args = argparse.Namespace(claude_model="claude-opus-5")

    cli._validate_and_resolve_claude_model(args, ("https://api.anthropic.com", "https://gw.example.com/v1"))

    assert probed == ["https://gw.example.com/v1"]


def test_catalog_probe_keeps_anthropic_side_when_it_has_its_own_key(clean_creds_env):
    """A keyed Anthropic side still owns the Claude catalog, gateway untouched."""
    clean_creds_env.setenv("_".join(("ANTHROPIC", "API", "KEY")), "sk-ant-api")
    clean_creds_env.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    clean_creds_env.setenv("_".join(("OPENAI", "API", "KEY")), "ak-openai")
    clean_creds_env.setenv("OPENAI_BASE_URL", "https://gw.example.com/v1")
    probed = _record_probed_urls(clean_creds_env, {"claude-opus-5"})
    args = argparse.Namespace(claude_model="claude-opus-5")

    cli._validate_and_resolve_claude_model(args, ("https://api.anthropic.com", "https://gw.example.com/v1"))

    assert probed == ["https://api.anthropic.com"]


# _resolve_gpu_type
def test_resolve_gpu_type_probe_only():
    """No --gpu-type passed; probe wins."""
    gpu, warns = cli_model_gate._resolve_gpu_type(user_specified="", probed="mi355x")
    assert gpu == "mi355x"
    assert warns == []


def test_resolve_gpu_type_user_only():
    """Probe failed (CPU sandbox); user value is used as-is, no warn."""
    gpu, warns = cli_model_gate._resolve_gpu_type(user_specified="mi300x", probed="")
    assert gpu == "mi300x"
    assert warns == []


def test_resolve_gpu_type_agreement_silent():
    gpu, warns = cli_model_gate._resolve_gpu_type(user_specified="mi355x", probed="mi355x")
    assert gpu == "mi355x"
    assert warns == []


def test_resolve_gpu_type_disagreement_probe_always_wins():
    """On disagreement the probe wins unconditionally and warns loudly."""
    gpu, warns = cli_model_gate._resolve_gpu_type(
        user_specified="mi300x",
        probed="mi355x",
    )
    assert gpu == "mi355x"
    assert len(warns) == 1
    assert "mi300x" in warns[0]
    assert "mi355x" in warns[0]


def test_resolve_gpu_type_no_inputs_returns_empty():
    """No probe, no user value → empty gpu_type."""
    gpu, warns = cli_model_gate._resolve_gpu_type(user_specified="", probed="")
    assert gpu == ""
    assert warns == []


# _emit_launch_info
def test_emit_launch_info_prints_kv_sentinel(tmp_path, capsys):
    session_dir = tmp_path / "model" / "20260101T000000Z"
    session_dir.mkdir(parents=True)
    info = cli._emit_launch_info(
        pid=12345,
        session_dir=session_dir,
        session_id="sess-xyz",
        run_log="/tmp/run.log",
        gpu_type="mi355x",
        framework="sglang",
        model="/models/qwen3",
        launch_info_file=None,
    )
    out = capsys.readouterr().out
    assert "HYPERLOOM_LAUNCH " in out
    line = [ln for ln in out.splitlines() if ln.startswith("HYPERLOOM_LAUNCH")][0]
    body = line[len("HYPERLOOM_LAUNCH ") :]
    parsed = dict(token.split("=", 1) for token in body.split(" "))
    assert parsed["pid"] == "12345"
    assert parsed["session_dir"] == str(session_dir)
    assert parsed["session_id"] == "sess-xyz"
    assert parsed["gpu_type"] == "mi355x"
    assert parsed["framework"] == "sglang"
    assert parsed["model"] == "/models/qwen3"
    assert info["event"] == "launch"


def test_emit_launch_info_writes_json_file(tmp_path, capsys):
    session_dir = tmp_path / "sess"
    session_dir.mkdir()
    out_file = tmp_path / "subdir" / "launch.json"
    cli._emit_launch_info(
        pid=7777,
        session_dir=session_dir,
        session_id="sid",
        run_log="",
        gpu_type="mi300x",
        framework="vllm",
        model="m",
        launch_info_file=str(out_file),
    )
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["pid"] == 7777
    assert data["session_dir"] == str(session_dir)
    assert data["session_id"] == "sid"
    assert data["framework"] == "vllm"
    assert data["manifest"] == str(session_dir / "manifest.json")
    out = capsys.readouterr().out
    assert "Launch info file" in out
    assert str(out_file) in out


def test_emit_launch_info_no_file_no_extra_print(tmp_path, capsys):
    session_dir = tmp_path / "sess"
    session_dir.mkdir()
    cli._emit_launch_info(
        pid=1,
        session_dir=session_dir,
        session_id="s",
        run_log="",
        gpu_type="",
        framework="",
        model="",
        launch_info_file=None,
    )
    out = capsys.readouterr().out
    assert "Launch info file" not in out


# CLI flag wiring (parser end-to-end)
def test_parser_accepts_launch_info_file():
    parser = _build_parser()
    ns = parser.parse_args(
        [
            "optimize",
            "--model",
            "/models/test",
            "--launch-info-file",
            "/tmp/launch.json",
        ]
    )
    assert ns.launch_info_file == "/tmp/launch.json"


def test_parser_default_launch_info_file_is_none():
    parser = _build_parser()
    ns = parser.parse_args(["optimize", "--model", "/models/test"])
    assert ns.launch_info_file is None


def test_parser_does_not_expose_removed_bypass_flags():
    """Regression guard: --no-creds-check and --gpu-type-force must stay removed."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "optimize",
                "--model",
                "/m",
                "--no-creds-check",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "optimize",
                "--model",
                "/m",
                "--gpu-type-force",
            ]
        )


# clean_stale_aiter_locks
def _make_aiter_tree(root):
    """Build a minimal aiter jit/build/ layout with mixed lock ages."""
    stale_mtime = time.time() - 30 * 60

    (root / "module_moe" / "build").mkdir(parents=True)
    (root / "module_other" / "build").mkdir(parents=True)

    top_stale = root / "lock_module_moe_stale"
    top_fresh = root / "lock_module_moe_fresh"
    inner_stale_lock = root / "module_moe" / "build" / "lock"
    inner_stale_ninja = root / "module_moe" / "build" / ".ninja_lock"
    inner_non_lock = root / "module_moe" / "build" / "compile_commands.json"
    inner_fresh = root / "module_other" / "build" / "lock"
    bare_so = root / "some_random.so"

    for p, content in (
        (top_stale, "stale"),
        (top_fresh, "fresh"),
        (inner_stale_lock, "stale"),
        (inner_stale_ninja, "stale"),
        (inner_non_lock, "{}"),
        (inner_fresh, "fresh"),
        (bare_so, "fake binary"),
    ):
        p.write_text(content)

    for p in (top_stale, inner_stale_lock, inner_stale_ninja):
        os.utime(p, (stale_mtime, stale_mtime))

    return {
        "stale_top": top_stale,
        "fresh_top": top_fresh,
        "stale_inner_lock": inner_stale_lock,
        "stale_inner_ninja": inner_stale_ninja,
        "non_lock": inner_non_lock,
        "fresh_inner": inner_fresh,
        "bare_so": bare_so,
    }


def test_clean_stale_aiter_locks_deletes_stale_keeps_fresh(tmp_path):
    layout = _make_aiter_tree(tmp_path)
    stats = cli.clean_stale_aiter_locks(
        aiter_jit_dir=tmp_path,
        stale_minutes=5,
    )
    assert stats["deleted"] == 3
    assert stats["skipped_fresh"] == 2
    assert stats["errors"] == 0
    assert not layout["stale_top"].exists()
    assert not layout["stale_inner_lock"].exists()
    assert not layout["stale_inner_ninja"].exists()
    assert layout["fresh_top"].exists()
    assert layout["fresh_inner"].exists()
    assert layout["non_lock"].exists()
    assert layout["bare_so"].exists()


def test_clean_stale_aiter_locks_handles_missing_dir():
    """When aiter cannot be located, return empty stats — never raise."""
    stats = cli.clean_stale_aiter_locks(
        aiter_jit_dir=type("X", (), {"is_dir": lambda self: False})(),  # noqa: E731
    )
    assert stats["scanned"] == 0
    assert stats["deleted"] == 0


def test_clean_stale_aiter_locks_respects_stale_minutes(tmp_path):
    """Bumping the threshold up keeps moderately-old locks alive."""
    (tmp_path / "lock_module_x").write_text("x")
    moderately_old = time.time() - 4 * 60
    os.utime(tmp_path / "lock_module_x", (moderately_old, moderately_old))
    stats = cli.clean_stale_aiter_locks(
        aiter_jit_dir=tmp_path,
        stale_minutes=10,
    )
    assert stats["deleted"] == 0
    assert stats["skipped_fresh"] == 1
    assert (tmp_path / "lock_module_x").exists()


def test_clean_stale_aiter_locks_auto_discovers_via_env_override(
    tmp_path,
    monkeypatch,
):
    """``$INFERENCE_OPTIMIZER_AITER_JIT_DIR`` resolves when no explicit dir is passed."""
    (tmp_path / "build").mkdir()
    stale_lock = tmp_path / "build" / "lock_module_z"
    stale_lock.write_text("x")
    stale_mtime = time.time() - 30 * 60
    os.utime(stale_lock, (stale_mtime, stale_mtime))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_AITER_JIT_DIR", str(tmp_path))
    stats = cli.clean_stale_aiter_locks(stale_minutes=5)
    assert stats["dir"] in {str(tmp_path), str(tmp_path / "build")}
    assert stats["deleted"] == 1
    assert not stale_lock.exists()
