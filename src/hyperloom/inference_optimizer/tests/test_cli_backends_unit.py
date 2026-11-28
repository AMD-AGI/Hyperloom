# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coverage for ``cli_backends``: per-role backend construction (which agent CLI runs orchestration, validation errors)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom_common.llm import llm_config
from hyperloom.inference_optimizer.cli import backends as clib


def _clear_provider_env(monkeypatch) -> None:
    for key in (
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "_".join(("CLAUDE", "CODE", "OAUTH", "TOKEN")),
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_API_KEY",
        "LLM_GATEWAY_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def _set_dual_protocol_gateway(monkeypatch) -> None:
    """One gateway (e.g. DeepSeek) serving both protocols, each side with its own URL and key."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-deepseek-key")


@pytest.fixture(autouse=True)
def _stub_backends(monkeypatch):
    """Replace heavy backend classes with lightweight stubs (no SDK / network)."""
    monkeypatch.setattr(clib, "ClaudeBackend", lambda **kw: ("claude", kw))
    monkeypatch.setattr(clib, "CodexBackend", lambda **kw: ("codex", kw))
    monkeypatch.setattr(clib, "DeepSeekBackend", lambda **kw: ("deepseek", kw))


def _build(**over):
    kwargs = dict(
        claude_model="claude-x",
        codex_model="codex-y",
        session_dir=Path("/tmp/s"),
    )
    kwargs.update(over)
    models = clib.ModelRouting(
        claude_model=kwargs.pop("claude_model"),
        codex_model=kwargs.pop("codex_model"),
        codex_follows_claude=kwargs.pop("codex_follows_claude", False),
        claude_follows_codex=kwargs.pop("claude_follows_codex", False),
    )
    session_dir = kwargs.pop("session_dir")
    assert not kwargs, kwargs
    return clib._build_backends(models, session_dir=session_dir)


@pytest.fixture(autouse=True)
def _isolated_provider_env(monkeypatch):
    """Every case in this module resolves backends from the environment, so the machine running the suite must not be able to change the answer."""
    _clear_provider_env(monkeypatch)
    # An operator override outranks the credential rule every case here exercises.
    monkeypatch.delenv(llm_config.AGENT_BACKEND_ENV, raising=False)
    # Orchestration selection ranks the installed SDKs after the credential, and which extras the suite happens to
    # run with is exactly the kind of machine state this fixture exists to hold still.
    monkeypatch.setattr(llm_config, "claude_agent_sdk_installed", lambda: True)
    monkeypatch.setattr(llm_config, "codex_agent_sdk_installed", lambda: True)
    monkeypatch.setattr(llm_config, "deepseek_harness_sdk_installed", lambda: True)


def test_build_backends_defaults_to_claude_orchestration() -> None:
    b = _build()
    assert b["orchestration"][0] == "claude"
    assert set(b) == {"orchestration"}


def test_build_backends_anthropic_only_uses_claude_for_orchestration(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    name, kwargs = _build()["orchestration"]
    assert name == "claude"
    assert kwargs["model"] == "claude-x"


def test_build_backends_openai_only_uses_codex_for_orchestration(monkeypatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    name, kwargs = _build()["orchestration"]
    assert name == "codex"
    assert kwargs["model"] == "codex-y"


def test_build_backends_dual_config_keeps_claude_for_orchestration(monkeypatch) -> None:
    """Both sides authenticate and both SDKs import, so the tie goes to Claude."""
    _clear_provider_env(monkeypatch)
    _set_dual_protocol_gateway(monkeypatch)

    assert _build()["orchestration"][0] == "claude"


def test_an_operator_override_outranks_the_credential_rule(monkeypatch) -> None:
    """The harness returns unenforced intents, so only an explicit ask selects it."""
    _clear_provider_env(monkeypatch)
    _set_dual_protocol_gateway(monkeypatch)
    monkeypatch.setenv(llm_config.AGENT_BACKEND_ENV, "deepseek")

    name, kwargs = _build()["orchestration"]
    assert name == "deepseek"
    # The role only reads state and answers with intents.
    assert kwargs["permission_mode"] == "read-only"


def test_an_explicit_codex_pins_orchestration_over_the_credential_ranking(monkeypatch) -> None:
    """Both sides authenticate, which ranks Claude first; naming Codex must win."""
    _clear_provider_env(monkeypatch)
    _set_dual_protocol_gateway(monkeypatch)
    monkeypatch.setenv(llm_config.AGENT_BACKEND_ENV, "codex")

    assert _build()["orchestration"][0] == "codex"


def test_the_harness_is_never_selected_by_credentials_alone(monkeypatch) -> None:
    """A DeepSeek gateway configures Claude's and Codex's protocols, not this route."""
    _clear_provider_env(monkeypatch)
    _set_dual_protocol_gateway(monkeypatch)

    assert _build()["orchestration"][0] == "claude"


def test_asking_for_the_harness_without_its_sdk_fails_at_launch(monkeypatch) -> None:
    """Naming a backend whose SDK cannot import must not be answered hours in."""
    _clear_provider_env(monkeypatch)
    _set_dual_protocol_gateway(monkeypatch)
    monkeypatch.setenv(llm_config.AGENT_BACKEND_ENV, "deepseek")
    monkeypatch.setattr(llm_config, "deepseek_harness_sdk_installed", lambda: False)

    with pytest.raises(ValueError, match="requires the DeepSeek Harness SDK"):
        _build()


def test_an_unknown_orchestration_backend_is_rejected(monkeypatch) -> None:
    """Silently falling back would hide a typo behind a whole run on the wrong CLI."""
    _clear_provider_env(monkeypatch)
    _set_dual_protocol_gateway(monkeypatch)
    monkeypatch.setenv(llm_config.AGENT_BACKEND_ENV, "clause")

    with pytest.raises(ValueError, match="not in"):
        _build()


def test_build_backends_dual_config_drops_claude_when_its_sdk_is_absent(monkeypatch) -> None:
    """Same credentials, so only the importable SDK separates them -- and a ClaudeBackend that cannot import is not an answer."""
    _clear_provider_env(monkeypatch)
    _set_dual_protocol_gateway(monkeypatch)
    monkeypatch.setattr(llm_config, "claude_agent_sdk_installed", lambda: False)

    assert _build()["orchestration"][0] == "codex"


def test_orchestration_follows_the_openai_only_intent_after_preflight_fills_anthropic(monkeypatch):
    """Preflight may publish ANTHROPIC_BASE_URL; the intent captured before it still puts orchestration on Codex."""
    monkeypatch.setattr(llm_config, "preferred_agent_backend", lambda: llm_config.AGENT_BACKEND_CLAUDE)
    assert clib.orchestration_runs_on_codex(codex_follows_claude=False, claude_follows_codex=True) is True
    assert clib.orchestration_runs_on_codex(codex_follows_claude=False, claude_follows_codex=False) is False
    assert clib.orchestration_runs_on_codex(codex_follows_claude=True, claude_follows_codex=True) is False
