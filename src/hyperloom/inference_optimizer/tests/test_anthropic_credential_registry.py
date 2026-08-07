# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Consistency tests binding every Anthropic-credential consumer to the single
source of truth in ``ANTHROPIC_CREDENTIAL_ENV_ORDER``.

Detection sites must recognize a newly registered credential form on their own;
the two controlled subsets (values that may be materialized into an API-key slot,
and secrets that may cross into a subprocess) must not follow it silently."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.common import llm_config as lc
from hyperloom.common.llm_config import (
    ANTHROPIC_CREDENTIAL_ENV_ORDER,
    ANTHROPIC_SYNTHESIZABLE_KEY_ENVS,
    CLAUDE_GATEWAY_SIGNAL_KEYS,
    CLAUDE_OAUTH_TOKEN_ENV,
)
from hyperloom.inference_optimizer.cli import credentials as cr
from hyperloom.inference_optimizer.cli import preflight as pf
from hyperloom.orchestrator.specialists.subprocess_ import _SPECIALIST_SECRET_ENV_ALLOWLIST

_FAKE_CREDENTIAL_ENV = "ANTHROPIC_FAKE_CREDENTIAL_FOR_TEST"

_PROVIDER_ENV = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_API_KEY",
    "LLM_GATEWAY_KEY",
    *ANTHROPIC_CREDENTIAL_ENV_ORDER,
)


@pytest.fixture
def registered_fake_credential(monkeypatch):
    """Register one extra credential form and export it, nothing else set."""
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        lc,
        "ANTHROPIC_CREDENTIAL_ENV_ORDER",
        ANTHROPIC_CREDENTIAL_ENV_ORDER + (_FAKE_CREDENTIAL_ENV,),
    )
    monkeypatch.setenv(_FAKE_CREDENTIAL_ENV, "fake-credential-value")
    return _FAKE_CREDENTIAL_ENV


def test_detection_sites_recognize_a_newly_registered_form(registered_fake_credential):
    """Adding a form to the tuple must be all it takes for detection to follow."""
    assert lc.has_anthropic_credential() is True
    assert cr._has_explicit_anthropic_key() is True
    # The endpoint resolver keys off the same predicate.
    anthropic_url, _ = cr._resolve_llm_endpoints()
    assert anthropic_url == "https://api.anthropic.com"
    # Cross-provider pairing sees an Anthropic side, so it stays silent.
    cr._reject_cross_provider_pairing()
    # has_key / has_usable_endpoint both accept it: no SystemExit.
    cr._validate_credentials()


def test_materializing_subsets_ignore_a_newly_registered_form(registered_fake_credential):
    """The billing-sensitive subsets are reviewed lists, not tuple followers."""
    assert registered_fake_credential not in ANTHROPIC_SYNTHESIZABLE_KEY_ENVS
    assert lc.anthropic_synthesizable_key() == ""
    # claude_primary_key (~/.claude/config.json) and fallback_key (request-level
    # credential synthesis) both resolve to nothing for an unreviewed form.
    assert lc.claude_sdk_env_options().get("env", {}).get("ANTHROPIC_API_KEY") is None


def test_specialist_allowlist_registers_every_credential_form():
    """Explicit registration, per the minimum-privilege boundary: this fails
    when a form joins the tuple without a decision on subprocess exposure."""
    missing = [name for name in ANTHROPIC_CREDENTIAL_ENV_ORDER if name not in _SPECIALIST_SECRET_ENV_ALLOWLIST]
    assert missing == []


def test_gateway_signal_keys_cover_every_credential_form():
    missing = [name for name in ANTHROPIC_CREDENTIAL_ENV_ORDER if name not in CLAUDE_GATEWAY_SIGNAL_KEYS]
    assert missing == []


def test_supersets_keep_their_non_anthropic_entries():
    """Both consumers are proper supersets; collapsing either onto the tuple
    would drop OpenAI-side env isolation and AWS credentials respectively."""
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "LLM_GATEWAY_KEY"):
        assert name in CLAUDE_GATEWAY_SIGNAL_KEYS
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_CONFIG_FILE",
        "AWS_PROFILE",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SHARED_CREDENTIALS_FILE",
        "OPENAI_API_KEY",
    ):
        assert name in _SPECIALIST_SECRET_ENV_ALLOWLIST


def test_synthesizable_subset_is_a_reviewed_subset():
    assert set(ANTHROPIC_SYNTHESIZABLE_KEY_ENVS) <= set(ANTHROPIC_CREDENTIAL_ENV_ORDER)
    assert CLAUDE_OAUTH_TOKEN_ENV not in ANTHROPIC_SYNTHESIZABLE_KEY_ENVS


def _install_scripts() -> list[Path]:
    root = Path(pf.__file__).resolve().parents[1]
    return [
        root / "assets" / "install.sh",
        root / "assets" / "install_baremetal.sh",
        root.parents[0] / "agents" / "kernel" / "scripts" / "install.sh",
    ]


def _executable_lines(script: Path) -> list[str]:
    """Script lines with comments and blank lines dropped, so a name mentioned
    only in prose cannot satisfy a reference check."""
    lines = []
    for raw in script.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


@pytest.mark.parametrize("script", _install_scripts(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_shell_credential_checks_know_every_credential_form(script: Path):
    """Shell entrypoints cannot import the tuple, so assert textually that each
    registered form is read or assigned by their credential handling."""
    code = _executable_lines(script)
    missing = [
        name
        for name in ANTHROPIC_CREDENTIAL_ENV_ORDER
        if not any(f"${{{name}" in line or f"${name}" in line or f"{name}=" in line for line in code)
    ]
    assert missing == [], f"{script.name} never reads or assigns {missing}"


def _config_json_writers() -> list[Path]:
    return [s for s in _install_scripts() if "primaryApiKey" in s.read_text(encoding="utf-8")]


def test_some_installer_still_writes_the_claude_config_key():
    """Guards the check below from going vacuous if the write is renamed."""
    assert _config_json_writers(), "no installer writes ~/.claude/config.json primaryApiKey"


@pytest.mark.parametrize("script", _config_json_writers(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_shell_config_json_key_excludes_the_subscription_token(script: Path):
    """The variable feeding ~/.claude/config.json primaryApiKey must be built
    from the synthesizable subset only."""
    code = _executable_lines(script)
    sources = [line for line in code if line.startswith("local _claude_key=")]
    assert sources, f"{script.name} writes primaryApiKey from an unrecognized variable"
    roots = [line for line in code if line.startswith("_ANTHROPIC_KEY_VAL=")]
    assert roots, f"{script.name} no longer feeds _claude_key from _ANTHROPIC_KEY_VAL"
    for line in sources + roots:
        assert CLAUDE_OAUTH_TOKEN_ENV not in line, line
    for line in roots:
        assert any(name in line for name in ANTHROPIC_SYNTHESIZABLE_KEY_ENVS), line
