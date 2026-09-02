# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for direct-gateway auth setup in ``_preflight``.

Pins the direct-gateway contract: base URLs are resolved for split/single
entrypoints and key aliases are fanned out from the provider key. Also covers
the surrounding preflight steps — dependency ensures (SDKs, Ray, bench-serving),
orchestration-model validation against the gateway catalog, the IR-3 KB/PR
probe, and the framework env guard.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from typing import Any

import pytest

from hyperloom.common.llm_config import deepseek_compat_env, parse_custom_headers
from hyperloom.inference_optimizer import cli
from hyperloom.inference_optimizer.cli import credentials as cli_credentials
from hyperloom.inference_optimizer.cli import preflight as cli_preflight
from hyperloom.inference_optimizer.cli.parser import _build_parser


@pytest.fixture
def stub_install_steps(monkeypatch, tmp_path):
    """Stub out heavyweight install steps so _preflight() is fast."""
    monkeypatch.setattr(cli_preflight, "_load_dotenv_fallback", lambda: None)
    # Stub the kernel-agent env fallback (it hard-fails when missing).
    monkeypatch.setattr(cli_preflight, "_load_kernel_agent_env_fallback", lambda: None)

    # InferenceX setup is orthogonal to the auth block under test. Point
    # INFERENCEX_PATH at a writable dir so detection short-circuits, and stub
    # the clone as a fallback.
    inferencex_dir = tmp_path / "InferenceX"
    (inferencex_dir / "benchmarks").mkdir(parents=True)
    (inferencex_dir / "benchmarks" / "benchmark_lib.sh").write_text("# stub", encoding="utf-8")
    monkeypatch.setenv("INFERENCEX_PATH", str(inferencex_dir))
    monkeypatch.setattr(cli_preflight, "_clone_inferencex", lambda dest: str(inferencex_dir))

    def _fake_which(name: str):
        return f"/usr/bin/{name}"

    monkeypatch.setattr(cli_preflight.shutil, "which", _fake_which)

    class _FakeCompleted:
        def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def _fake_run(cmd, *args, **kwargs):
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(cli_preflight.subprocess, "run", _fake_run)
    return None


@pytest.fixture(autouse=True)
def _restore_environ():
    """Roll back direct ``os.environ`` writes after every test in this module.

    Several tests here exercise the real `_load_dotenv_fallback`, which writes
    straight into ``os.environ`` — monkeypatch cannot undo that. One of them clears
    ``REPO_ROOT`` on purpose, and the fallback then walks up to the repository root
    and finds the `.env` that `install.sh` generates there. On a machine where the
    installer has run, that leaked `ANTHROPIC_BASE_URL` into the environment and
    four later auth tests in this file failed — a real defect in the suite's
    hermeticity that only appears after a real deployment step, which is the worst
    time to be chasing a phantom failure.
    """
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


@pytest.fixture
def clean_url_env(monkeypatch):
    """Strip URL env vars and fully restore os.environ afterwards.

    ``_preflight`` writes alias vars directly into ``os.environ``; monkeypatch
    cannot roll those back, so snapshot and restore the whole environ.
    """
    import os

    snapshot = dict(os.environ)
    for var in (
        "ANTHROPIC_BASE_URL",
        "OPENAI_BASE_URL",
        "GEAK_BASE_URL",
        "LLM_API_BASE",
        "_".join(("ANTHROPIC", "API", "KEY")),
        "_".join(("ANTHROPIC", "AUTH", "TOKEN")),
        "_".join(("DEEPSEEK", "API", "KEY")),
        "DEEPSEEK_BASE_URL",
        "_".join(("OPENAI", "API", "KEY")),
        "_".join(("GEAK", "API", "KEY")),
        "_".join(("LLM", "API", "KEY")),
        "_".join(("AMD_LLM", "API", "KEY")),
        "INFERENCEX_PATH",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        yield monkeypatch
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


def test_dotenv_fallback_ignores_arbitrary_cwd_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("REPO_ROOT", raising=False)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "LD_PRELOAD=/tmp/evil.so\nOPENAI_BASE_URL=https://evil.example/v1\n",
        encoding="utf-8",
    )
    cli_preflight._load_dotenv_fallback()
    assert "LD_PRELOAD" not in os.environ
    assert os.environ.get("OPENAI_BASE_URL") != "https://evil.example/v1"


def test_dotenv_fallback_filters_explicit_repo_env(tmp_path, monkeypatch):
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    (tmp_path / ".env").write_text(
        "LD_PRELOAD=/tmp/evil.so\nPYTHONSTARTUP=/tmp/pwn.py\nOPENAI_BASE_URL=https://gateway.example/v1\n",
        encoding="utf-8",
    )
    cli_preflight._load_dotenv_fallback()
    assert "LD_PRELOAD" not in os.environ
    assert "PYTHONSTARTUP" not in os.environ
    assert os.environ["OPENAI_BASE_URL"] == "https://gateway.example/v1"


def test_dotenv_fallback_parses_safe_lines_and_preserves_env_wins(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://operator.example/v1")
    monkeypatch.delenv("HYPERLOOM_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("TRACELENS_ROOT", raising=False)
    (tmp_path / ".env").write_text(
        "\n"
        "# comment\n"
        "export HYPERLOOM_RUNTIME_DIR='/runtime from env'\n"
        "OPENAI_BASE_URL=https://from-file.example/v1\n"
        "TRACELENS_ROOT=/path/to/your/TraceLens\n"
        "NO_EQUALS_LINE\n"
        "BAD-NAME=drop\n",
        encoding="utf-8",
    )

    cli_preflight._load_dotenv_fallback()

    assert os.environ["HYPERLOOM_RUNTIME_DIR"] == "/runtime from env"
    assert os.environ["OPENAI_BASE_URL"] == "https://operator.example/v1"
    assert "TRACELENS_ROOT" not in os.environ
    err = capsys.readouterr().err
    assert "BAD-NAME" in err


def test_dotenv_fallback_loads_gateway_custom_headers(tmp_path, monkeypatch):
    """A header-authenticated gateway survives .env -> environment -> parsing.

    setup writes these headers into ``.env``, so the loader has to accept them.
    It strips the quotes and leaves ``${VAR}`` intact; the expansion belongs to
    ``parse_custom_headers``, which is what lets the secret live in one place.
    """
    anthropic_key_var = "_".join(("ANTHROPIC", "API", "KEY"))
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))
    for var in ("ANTHROPIC_CUSTOM_HEADERS", "OPENAI_CUSTOM_HEADERS", anthropic_key_var):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / ".env").write_text(
        f"{anthropic_key_var}=ak-gateway-token\n"
        f'ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: ${{{anthropic_key_var}}}"\n'
        'OPENAI_CUSTOM_HEADERS="X-Tenant: acme"\n',
        encoding="utf-8",
    )

    cli_preflight._load_dotenv_fallback()

    assert os.environ["ANTHROPIC_CUSTOM_HEADERS"] == f"Ocp-Apim-Subscription-Key: ${{{anthropic_key_var}}}"
    assert os.environ["OPENAI_CUSTOM_HEADERS"] == "X-Tenant: acme"
    assert parse_custom_headers(os.environ["ANTHROPIC_CUSTOM_HEADERS"]) == {
        "Ocp-Apim-Subscription-Key": "ak-gateway-token"
    }
    assert parse_custom_headers(os.environ["OPENAI_CUSTOM_HEADERS"]) == {"X-Tenant": "acme"}


def test_preflight_does_not_export_a_derived_url_for_a_subscription_token(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    """The Claude CLI resolves its own endpoint, and all three installers keep
    this URL a local variable. Exporting a derived one would diverge from them
    and hand every child a gateway signal the operator never set."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _oauth_only_env(monkeypatch, base_url="")

    resolved = cli_preflight._preflight()

    # Still resolved for the decisions downstream of it...
    assert resolved[0] == "https://api.anthropic.com"
    # ...but never published into the environment.
    assert "ANTHROPIC_BASE_URL" not in cli.os.environ


def test_preflight_still_exports_an_explicit_url_for_a_subscription_token(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    """Only a *derived* URL is withheld; an operator who set one keeps it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _oauth_only_env(monkeypatch, base_url="https://gw.example/anthropic")

    cli_preflight._preflight()

    assert cli.os.environ["ANTHROPIC_BASE_URL"] == "https://gw.example/anthropic"


def test_preflight_resolves_urls_and_fans_out_auth_aliases(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("_".join(("OPENAI", "API", "KEY")), "new-gateway-key")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://gateway.example/api/v1/llm-proxy/v1",
    )
    # ANTHROPIC_BASE_URL unset -> the Anthropic side stays disabled, never derived.
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    # Derived aliases start unset so the provider key fills them.
    for name in (
        "_".join(("ANTHROPIC", "AUTH", "TOKEN")),
        "_".join(("ANTHROPIC", "API", "KEY")),
        "_".join(("GEAK", "API", "KEY")),
        "_".join(("LLM", "API", "KEY")),
        "_".join(("AMD_LLM", "API", "KEY")),
    ):
        monkeypatch.delenv(name, raising=False)
    # Base-url aliases start unset -> default to the resolved gateway.
    for name in ("GEAK_BASE_URL", "LLM_API_BASE"):
        monkeypatch.delenv(name, raising=False)

    config_dir = tmp_path / ".claude"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        '{"primaryApiKey":"old-key","customApiUrl":"https://old.example/v1"}',
        encoding="utf-8",
    )

    resolved = cli_preflight._preflight()

    # OpenAI side only: the Anthropic endpoint is left unset rather than derived.
    assert resolved == ("", "https://gateway.example/api/v1/llm-proxy/v1")
    assert "ANTHROPIC_BASE_URL" not in cli.os.environ
    assert cli.os.environ["OPENAI_BASE_URL"] == resolved[1]
    # The OpenAI key fills its own name plus the internal LLM aliases.
    for name in (
        "_".join(("OPENAI", "API", "KEY")),
        "_".join(("LLM", "API", "KEY")),
        "_".join(("AMD_LLM", "API", "KEY")),
    ):
        assert cli.os.environ[name] == "new-gateway-key"
    # The Anthropic-side keys are never cross-filled from the OpenAI key.
    assert "_".join(("ANTHROPIC", "API", "KEY")) not in cli.os.environ
    assert "_".join(("ANTHROPIC", "AUTH", "TOKEN")) not in cli.os.environ
    assert cli.os.environ["LLM_API_BASE"] == resolved[1]
    # GEAK runs on the Anthropic side, so its aliases are never derived here.
    assert "_".join(("GEAK", "API", "KEY")) not in cli.os.environ
    assert "GEAK_BASE_URL" not in cli.os.environ
    assert "_".join(("legacy backend", "API", "KEY")) not in cli.os.environ
    assert "_".join(("legacy backend", "BASE", "URL")) not in cli.os.environ

    # No Anthropic side: nothing is written for Claude.
    config_text = (config_dir / "config.json").read_text(encoding="utf-8")
    assert "new-gateway-key" not in config_text
    assert "gateway.example" not in config_text


def test_preflight_keeps_explicit_provider_keys(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    """Explicit provider keys are preserved; only the internal GEAK/LLM aliases are gap-filled."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("_".join(("OPENAI", "API", "KEY")), "openai-user-token")
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-user-token")
    for name in (
        "_".join(("ANTHROPIC", "AUTH", "TOKEN")),
        "_".join(("GEAK", "API", "KEY")),
        "_".join(("LLM", "API", "KEY")),
        "_".join(("AMD_LLM", "API", "KEY")),
    ):
        monkeypatch.delenv(name, raising=False)

    resolved = cli_preflight._preflight()

    # Both base URLs are kept distinct.
    assert resolved == ("https://api.anthropic.com", "https://api.openai.com/v1")
    # Explicit provider keys are preserved.
    assert cli.os.environ["_".join(("OPENAI", "API", "KEY"))] == "openai-user-token"
    assert cli.os.environ["_".join(("ANTHROPIC", "API", "KEY"))] == "anthropic-user-token"
    # GEAK runs on the Anthropic side, so its key alias is never derived.
    assert "_".join(("GEAK", "API", "KEY")) not in cli.os.environ
    # The Anthropic auth-token alias is never cross-filled from another key.
    assert "_".join(("ANTHROPIC", "AUTH", "TOKEN")) not in cli.os.environ


def test_preflight_keeps_anthropic_side_supplied_by_dotenv(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    """``.env`` is operator configuration: with the OpenAI side exported in the
    shell and the Anthropic side coming from ``.env``, both sides survive."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gw.example.com/v1")
    monkeypatch.setenv("_".join(("OPENAI", "API", "KEY")), "ak-gw")

    def _dotenv_supplies_anthropic_side():
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gw.example.com")
        monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "ak-gw")

    monkeypatch.setattr(cli_preflight, "_load_dotenv_fallback", _dotenv_supplies_anthropic_side)

    resolved = cli_preflight._preflight()

    assert resolved == ("https://gw.example.com", "https://gw.example.com/v1")
    assert cli.os.environ["_".join(("ANTHROPIC", "API", "KEY"))] == "ak-gw"


def test_preflight_rejects_half_configured_side_from_dotenv(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    """A key in ``.env`` whose own base URL is absent is a mispaired shape and is
    rejected, not silently dropped, even though the shell side is complete."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gw.example.com/v1")
    monkeypatch.setenv("_".join(("OPENAI", "API", "KEY")), "ak-openai")

    def _dotenv_has_stale_anthropic_key():
        monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "old-anthropic-key")

    monkeypatch.setattr(cli_preflight, "_load_dotenv_fallback", _dotenv_has_stale_anthropic_key)

    with pytest.raises(SystemExit) as excinfo:
        cli_preflight._preflight()

    assert excinfo.value.code == 2


def test_preflight_openai_only_drops_anthropic_creds_from_installer_env(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    """A stale installer env file must not inject an Anthropic-side key into an
    OpenAI-only run: that would turn a valid config into a rejected one."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("_".join(("OPENAI", "API", "KEY")), "openai-user-token")

    def _stale_kernel_env_loader():
        monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "stale-anthropic-key")

    monkeypatch.setattr(cli_preflight, "_load_kernel_agent_env_fallback", _stale_kernel_env_loader)

    resolved = cli_preflight._preflight()

    assert resolved == ("", "https://gateway.example/v1")
    assert "_".join(("ANTHROPIC", "API", "KEY")) not in cli.os.environ


def test_preflight_claude_config_uses_explicit_anthropic_key(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    """Dual entry: ~/.claude/config.json primaryApiKey is the explicit ANTHROPIC API key, not the OpenAI-side key."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("_".join(("OPENAI", "API", "KEY")), "openai-user-token")
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-user-token")

    cli_preflight._preflight()

    config_text = (tmp_path / ".claude" / "config.json").read_text(encoding="utf-8")
    assert '"primaryApiKey": "anthropic-user-token"' in config_text
    assert "openai-user-token" not in config_text
    assert '"customApiUrl": "https://api.anthropic.com"' in config_text


def test_preflight_anthropic_only_leaves_openai_protocol_aliases_unset(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    """The GEAK / LLM aliases address OpenAI-protocol endpoints, so an
    Anthropic-only entry leaves both the URL and the key side unset."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-user-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    resolved = cli_preflight._preflight()

    # Official Anthropic is not OpenAI-compatible; only Anthropic side resolved.
    assert resolved == ("https://api.anthropic.com", "")
    for name in (
        "GEAK_BASE_URL",
        "LLM_API_BASE",
        "_".join(("GEAK", "API", "KEY")),
        "_".join(("LLM", "API", "KEY")),
        "_".join(("AMD_LLM", "API", "KEY")),
    ):
        assert name not in cli.os.environ, name
    assert "_".join(("legacy backend", "BASE", "URL")) not in cli.os.environ


def test_preflight_official_anthropic_key_only_uses_default_endpoint(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-user-token")

    resolved = cli._preflight()

    assert resolved == ("https://api.anthropic.com", "")
    assert cli.os.environ["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
    assert "OPENAI_BASE_URL" not in cli.os.environ
    config_text = (tmp_path / ".claude" / "config.json").read_text(encoding="utf-8")
    assert '"primaryApiKey": "anthropic-user-token"' in config_text
    assert '"customApiUrl": "https://api.anthropic.com"' in config_text


def test_preflight_anthropic_only_ignores_stale_kernel_env_openai_fallback(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-user-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    def _stale_kernel_env_loader():
        monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example.invalid/Unified/v1")
        monkeypatch.setenv("_".join(("SAFE", "API", "KEY")), "old-gateway-key")
        monkeypatch.setenv("LLM_API_BASE", "https://llm.example.invalid/Unified/v1")

    monkeypatch.setattr(cli_preflight, "_load_kernel_agent_env_fallback", _stale_kernel_env_loader)

    resolved = cli._preflight()

    assert resolved == ("https://api.anthropic.com", "")
    assert cli.os.environ["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
    assert "OPENAI_BASE_URL" not in cli.os.environ
    assert "_".join(("OPENAI", "API", "KEY")) not in cli.os.environ
    # Defense-in-depth: a stray legacy gateway key from the installer env is
    # stripped too, so it never reaches child processes.
    assert "_".join(("SAFE", "API", "KEY")) not in cli.os.environ
    # The stale OpenAI-side LLM_API_BASE is dropped, not rewritten to Anthropic.
    assert "LLM_API_BASE" not in cli.os.environ


def test_preflight_official_openai_key_only_uses_default_endpoint(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("_".join(("OPENAI", "API", "KEY")), "openai-user-token")

    resolved = cli._preflight()

    assert resolved == ("", "https://api.openai.com/v1")
    assert "ANTHROPIC_BASE_URL" not in cli.os.environ
    assert cli.os.environ["OPENAI_BASE_URL"] == "https://api.openai.com/v1"


def test_preflight_both_official_keys_without_urls_uses_default_endpoints(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-user-token")
    monkeypatch.setenv("_".join(("OPENAI", "API", "KEY")), "openai-user-token")

    resolved = cli._preflight()

    assert resolved == ("https://api.anthropic.com", "https://api.openai.com/v1")
    assert cli.os.environ["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
    assert cli.os.environ["OPENAI_BASE_URL"] == "https://api.openai.com/v1"


def test_preflight_preserves_operator_geak_tunnel_url(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    """An operator-pinned GEAK tunnel URL survives preflight.

    Preflight must NOT clobber GEAK_BASE_URL back to the direct gateway URL,
    while still defaulting the unset LLM_API_BASE to the gateway.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("_".join(("OPENAI", "API", "KEY")), "gateway-key")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://gateway.example/api/v1/llm-proxy/v1",
    )
    tunnel = "https://127.0.0.1:18444/api/v1/llm-proxy/v1"
    monkeypatch.setenv("GEAK_BASE_URL", tunnel)
    # LLM_API_BASE left unset → should default to the gateway.

    resolved = cli_preflight._preflight()

    gateway = resolved[1]
    assert cli.os.environ["GEAK_BASE_URL"] == tunnel
    assert cli.os.environ["LLM_API_BASE"] == gateway
    assert "_".join(("legacy backend", "BASE", "URL")) not in cli.os.environ


def test_preflight_rewrites_stale_proxy_even_when_operator_set(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    """A leftover 127.0.0.1:4002 value is force-rewritten, not preserved."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("_".join(("OPENAI", "API", "KEY")), "gateway-key")
    monkeypatch.setenv(
        "OPENAI_BASE_URL",
        "https://gateway.example/api/v1/llm-proxy/v1",
    )
    monkeypatch.setenv(
        "LLM_API_BASE",
        "http://127.0.0.1:4002/api/v1/llm-proxy/v1",
    )

    resolved = cli_preflight._preflight()

    assert cli.os.environ["LLM_API_BASE"] == resolved[1]
    assert "127.0.0.1:4002" not in cli.os.environ["LLM_API_BASE"]


def test_preflight_keeps_official_anthropic_endpoint_despite_leftover_deepseek_key(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    """Regression: a forgotten DEEPSEEK_API_KEY must not hijack a real Anthropic key.

    Half-adopting the retired gateway would resolve ANTHROPIC_BASE_URL to
    DeepSeek's host while the operator's own Anthropic key is what gets sent
    there, and would add an OpenAI side they never configured.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "sk-real-anthropic")
    monkeypatch.setenv("_".join(("DEEPSEEK", "API", "KEY")), "sk-legacy-deepseek")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("_".join(("ANTHROPIC", "AUTH", "TOKEN")), raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("_".join(("OPENAI", "API", "KEY")), raising=False)

    cli_preflight._preflight()

    assert cli.os.environ["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
    assert cli.os.environ.get("OPENAI_BASE_URL", "") == ""
    assert cli.os.environ.get("_".join(("OPENAI", "API", "KEY")), "") == ""


def test_provider_fallback_keys_strip_retired_deepseek_vars_in_either_mode():
    """A stale .env must not hand a single-provider run the other side.

    The retired variables normalize to BOTH protocol sides, so an Anthropic-only
    shell has to drop them just like an OpenAI-only one does.
    """
    for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"):
        assert key in cli_preflight._PROVIDER_FALLBACK_KEYS, key
        assert key in cli_preflight._ANTHROPIC_FALLBACK_KEYS, key


def test_is_stale_proxy_url_matches_legacy_only():
    assert cli_credentials._is_stale_proxy_url("http://127.0.0.1:4002/api/v1/llm-proxy/v1")
    assert not cli_credentials._is_stale_proxy_url("https://127.0.0.1:18444/api/v1/llm-proxy/v1")
    assert not cli_credentials._is_stale_proxy_url("https://gateway.example/v1")
    assert not cli_credentials._is_stale_proxy_url("")
    assert not cli_credentials._is_stale_proxy_url(None)


_GEAK_CFG_TEMPLATE = """model:
  model_class: litellm
  model_name: openai/claude-opus-4-7
  api_key: test-token
  base_url: {url}
  model_kwargs:
    max_tokens: 16384
run:
  mode: full
"""


def test_sync_geak_config_rewrites_stale_base_url(tmp_path):
    """An install-time gateway URL is rewritten to the operator tunnel."""
    cfg = tmp_path / "geak.yaml"
    cfg.write_text(
        _GEAK_CFG_TEMPLATE.format(
            url="https://old-gateway.example.invalid/api/v1/llm-proxy/v1",
        ),
        encoding="utf-8",
    )
    tunnel = "https://127.0.0.1:18444/api/v1/llm-proxy/v1"

    changed = cli_credentials._sync_geak_config_base_url(str(cfg), tunnel)

    assert changed is True
    text = cfg.read_text(encoding="utf-8")
    assert f"base_url: {tunnel}" in text
    assert "old-gateway.example.invalid" not in text
    assert "model_class: litellm" in text
    assert "api_key: test-token" in text


def test_sync_geak_config_noop_when_already_in_sync(tmp_path):
    cfg = tmp_path / "geak.yaml"
    url = "https://127.0.0.1:18444/api/v1/llm-proxy/v1"
    cfg.write_text(_GEAK_CFG_TEMPLATE.format(url=url), encoding="utf-8")
    before = cfg.read_text(encoding="utf-8")

    assert cli_credentials._sync_geak_config_base_url(str(cfg), url) is False
    assert cfg.read_text(encoding="utf-8") == before


def test_sync_geak_config_missing_file_is_safe(tmp_path):
    missing = tmp_path / "nope.yaml"
    assert cli_credentials._sync_geak_config_base_url(str(missing), "https://x/v1") is False


def test_sync_geak_config_no_base_url_line_is_safe(tmp_path):
    cfg = tmp_path / "geak.yaml"
    cfg.write_text("model:\n  model_class: litellm\n", encoding="utf-8")
    assert cli_credentials._sync_geak_config_base_url(str(cfg), "https://x/v1") is False


def test_preflight_anthropic_only_sets_geak_v4_claude_model(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    """GEAKv4/main uses Claude Code Workflow and reads GEAK_CLAUDE_MODEL."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-user-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-4-6")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("_".join(("OPENAI", "API", "KEY")), raising=False)
    monkeypatch.delenv("GEAK_CLAUDE_MODEL", raising=False)

    cli._preflight()

    assert cli.os.environ["GEAK_CLAUDE_MODEL"] == "claude-opus-4-6"


def test_preflight_migrates_retired_deepseek_env_to_both_sides(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    """A retired DEEPSEEK_* config resolves to both protocol sides plus its model."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("_".join(("DEEPSEEK", "API", "KEY")), "deepseek-token")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("_".join(("ANTHROPIC", "API", "KEY")), raising=False)
    monkeypatch.delenv("_".join(("ANTHROPIC", "AUTH", "TOKEN")), raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("_".join(("OPENAI", "API", "KEY")), raising=False)
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    monkeypatch.delenv("GEAK_CLAUDE_MODEL", raising=False)

    cli._preflight()

    assert cli.os.environ["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"
    assert cli.os.environ["OPENAI_BASE_URL"] == "https://api.deepseek.com/v1"
    assert cli.os.environ["GEAK_CLAUDE_MODEL"] == "deepseek-v4-pro"


def test_preflight_retired_deepseek_env_exports_claude_cli_auth_aliases(
    monkeypatch,
    tmp_path,
    clean_url_env,
    stub_install_steps,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("_".join(("DEEPSEEK", "API", "KEY")), "deepseek-token")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("_".join(("ANTHROPIC", "API", "KEY")), raising=False)
    monkeypatch.delenv("_".join(("ANTHROPIC", "AUTH", "TOKEN")), raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("_".join(("OPENAI", "API", "KEY")), raising=False)

    cli._preflight()

    assert cli.os.environ["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"
    assert cli.os.environ["_".join(("ANTHROPIC", "API", "KEY"))] == "deepseek-token"
    assert cli.os.environ["_".join(("ANTHROPIC", "AUTH", "TOKEN"))] == "deepseek-token"


def test_sync_geak_config_empty_args_are_safe(tmp_path):
    cfg = tmp_path / "geak.yaml"
    cfg.write_text(_GEAK_CFG_TEMPLATE.format(url="https://x/v1"), encoding="utf-8")
    assert cli_credentials._sync_geak_config_base_url("", "https://y/v1") is False
    assert cli_credentials._sync_geak_config_base_url(str(cfg), "") is False


def test_sync_geak_config_preserves_url_with_special_chars(tmp_path):
    """A replacement URL with regex-special chars must land verbatim."""
    cfg = tmp_path / "geak.yaml"
    cfg.write_text(
        _GEAK_CFG_TEMPLATE.format(url="https://old/v1"),
        encoding="utf-8",
    )
    weird = r"https://host/api\g<0>/v1"

    assert cli_credentials._sync_geak_config_base_url(str(cfg), weird) is True
    assert f"base_url: {weird}" in cfg.read_text(encoding="utf-8")


class _RecordingRun:
    """Test double for subprocess.run that records calls and replays a script."""

    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.calls: list[list[str]] = []

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(list(cmd))
        if not self.script:
            raise AssertionError(f"unexpected subprocess call: {cmd}")
        item = self.script.pop(0)
        if callable(item):
            return item(cmd, *args, **kwargs)
        return item


class _Completed:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_ensure_python_sdks_skips_when_all_present(monkeypatch, capsys):
    """All four import-checks return rc=0 → no pip install fires."""
    runner = _RecordingRun([_Completed(returncode=0) for _ in range(4)])
    monkeypatch.setattr(cli_preflight.subprocess, "run", runner)

    cli_preflight._ensure_python_sdks("/opt/venv/bin/python", [])

    assert len(runner.calls) == 4
    for call in runner.calls:
        assert call[0] == "/opt/venv/bin/python"
        assert call[1] == "-c"
        assert call[2].startswith("import ")
    captured = capsys.readouterr().out
    assert "claude_agent_sdk OK" in captured
    assert "openai_codex OK" in captured
    assert "openai OK" in captured
    assert "httpx OK" in captured


def test_ensure_python_sdks_explicit_hermes_skips_unused_agent_sdks(monkeypatch, capsys):
    runner = _RecordingRun([_Completed(returncode=0) for _ in range(2)])
    monkeypatch.setattr(cli_preflight.subprocess, "run", runner)
    monkeypatch.setenv("HYPERLOOM_AGENT_BACKEND", "hermes")

    result = cli_preflight._ensure_python_sdks("/opt/venv/bin/python", [])

    assert [call[2] for call in runner.calls] == ["import openai", "import httpx"]
    assert "claude-agent-sdk" not in result["target"]
    assert "openai-codex" not in result["target"]
    captured = capsys.readouterr().out
    assert "openai OK" in captured and "httpx OK" in captured


def test_ray_version_tracks_python_wheel_support():
    assert cli_preflight._ray_version_for_python((3, 12)) == "2.44.1"
    assert cli_preflight._ray_version_for_python((3, 13)) == "2.44.1"
    assert cli_preflight._ray_version_for_python((3, 14)) == "2.55.1"


def test_ensure_python_sdks_installs_missing_openai_codex(monkeypatch, capsys):
    """Both agent runtimes are provisioned: a missing codex SDK is installed too.

    Without it an OpenAI-only deployment reaches the TraceLens skill runner and the
    forge kernel backend with no runtime to execute them.
    """
    runner = _RecordingRun(
        [
            _Completed(returncode=0),
            _Completed(returncode=1),
            _Completed(returncode=0),
            _Completed(returncode=0),
            _Completed(returncode=0),
        ]
    )
    monkeypatch.setattr(cli_preflight.subprocess, "run", runner)

    cli_preflight._ensure_python_sdks("/opt/venv/bin/python", [])

    install_call = runner.calls[2]
    assert install_call[1:4] == ["-m", "pip", "install"]
    assert any(arg.startswith("openai-codex") for arg in install_call)
    captured = capsys.readouterr().out
    assert "installed openai-codex" in captured


def test_ensure_python_sdks_installs_missing_claude_agent_sdk(monkeypatch, capsys):
    """When `import claude_agent_sdk` fails, pip install runs with the SAME interpreter."""
    runner = _RecordingRun(
        [
            _Completed(returncode=1),
            _Completed(returncode=0),
            _Completed(returncode=0),
            _Completed(returncode=0),
            _Completed(returncode=0),
        ]
    )
    monkeypatch.setattr(cli_preflight.subprocess, "run", runner)

    cli_preflight._ensure_python_sdks("/opt/venv/bin/python", ["--break-system-packages"])

    assert len(runner.calls) == 5
    install_call = runner.calls[1]
    assert install_call[0] == "/opt/venv/bin/python"
    assert install_call[1:4] == ["-m", "pip", "install"]
    assert "--break-system-packages" in install_call
    assert any(arg.startswith("claude-agent-sdk") for arg in install_call)
    captured = capsys.readouterr().out
    assert "installing claude-agent-sdk" in captured
    assert "installed claude-agent-sdk" in captured


# _ensure_ray
def test_ensure_ray_skips_when_smoke_passes(monkeypatch, capsys):
    """Ray/click smoke passes -> no pip install fires (probe uses the interpreter)."""
    runner = _RecordingRun([_Completed(returncode=0)])
    monkeypatch.setattr(cli_preflight.subprocess, "run", runner)

    cli_preflight._ensure_ray("/opt/venv/bin/python", [])

    assert len(runner.calls) == 1
    probe = runner.calls[0]
    # Probe validates Ray with the SAME interpreter, not shutil.which on PATH.
    assert probe[0:2] == ["/opt/venv/bin/python", "-c"]
    assert "ray.__version__" in probe[2]
    assert "ray.scripts.scripts" in probe[2]
    assert "ray OK" in capsys.readouterr().out


def test_ensure_ray_installs_when_smoke_fails(monkeypatch, capsys):
    """When Ray/click smoke fails, pip install runs with the SAME interpreter.

    Guards the bypass-only regression: a stray ``ray`` on PATH must not stop
    the install when the active interpreter cannot run Ray correctly.
    """
    runner = _RecordingRun(
        [
            _Completed(returncode=1, stderr="click version incompatible with Ray CLI: 8.4.2 >= 8.3.0"),
            _Completed(returncode=0),
            _Completed(returncode=0),
        ]
    )
    monkeypatch.setattr(cli_preflight.subprocess, "run", runner)

    cli_preflight._ensure_ray("/opt/venv/bin/python", ["--break-system-packages"])

    assert len(runner.calls) == 3
    install_call = runner.calls[1]
    assert install_call[0] == "/opt/venv/bin/python"
    assert install_call[1:4] == ["-m", "pip", "install"]
    assert "--break-system-packages" in install_call
    assert any(arg.startswith("ray[default]==") for arg in install_call)
    assert "click<8.3.0" in install_call
    captured = capsys.readouterr().out
    assert "ray/click invalid" in captured
    assert "ray installed OK" in captured


# _ensure_bench_serving_deps
def test_ensure_bench_serving_deps_skips_when_all_present(monkeypatch, capsys):
    """find_spec probe reports nothing missing -> no pip install fires."""
    runner = _RecordingRun([_Completed(returncode=0, stdout="")])
    monkeypatch.setattr(cli_preflight.subprocess, "run", runner)

    cli_preflight._ensure_bench_serving_deps("/opt/venv/bin/python", [])

    assert len(runner.calls) == 1
    probe = runner.calls[0]
    assert probe[0] == "/opt/venv/bin/python"
    assert probe[1] == "-c"
    # every dep name is handed to the probe as argv (checked via find_spec).
    for dep in cli_preflight._BENCH_SERVING_DEPS:
        assert dep in probe
    assert "benchmark_serving client deps OK" in capsys.readouterr().out


def test_ensure_bench_serving_deps_installs_only_missing_subset(monkeypatch, capsys):
    """Only the modules the probe reports missing are pip-installed (same interp)."""
    runner = _RecordingRun(
        [
            _Completed(returncode=0, stdout="transformers\ndatasets\n"),
            _Completed(returncode=0),
        ]
    )
    monkeypatch.setattr(cli_preflight.subprocess, "run", runner)

    cli_preflight._ensure_bench_serving_deps("/opt/venv/bin/python", ["--break-system-packages"])

    assert len(runner.calls) == 2
    install = runner.calls[1]
    assert install[0] == "/opt/venv/bin/python"
    assert install[1:4] == ["-m", "pip", "install"]
    assert "--break-system-packages" in install
    assert "transformers" in install and "datasets" in install
    assert "aiohttp" not in install  # present deps are not reinstalled
    assert "installing benchmark_serving client deps" in capsys.readouterr().out


def test_ensure_bench_serving_deps_probe_failure_installs_all(monkeypatch):
    """A crashed probe falls back to attempting the full dep set."""
    runner = _RecordingRun([_Completed(returncode=2, stdout=""), _Completed(returncode=0)])
    monkeypatch.setattr(cli_preflight.subprocess, "run", runner)

    cli_preflight._ensure_bench_serving_deps("/opt/venv/bin/python", [])

    assert len(runner.calls) == 2
    install = runner.calls[1]
    for dep in cli_preflight._BENCH_SERVING_DEPS:
        assert dep in install


# _unset_hip_visible_devices
def test_unset_hip_visible_devices_pops_when_rocr_present(monkeypatch, capsys):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2,3")

    cli_preflight._unset_hip_visible_devices()

    import os as _os

    assert "HIP_VISIBLE_DEVICES" not in _os.environ
    assert _os.environ["ROCR_VISIBLE_DEVICES"] == "0,1,2,3"
    assert "WARNING" in capsys.readouterr().out


def test_unset_hip_visible_devices_keeps_hip_when_rocr_unset(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)

    cli_preflight._unset_hip_visible_devices()

    import os as _os

    assert _os.environ["HIP_VISIBLE_DEVICES"] == "0,1,2,3"


def test_visibility_normalizer_prefers_hip_when_explicit(monkeypatch, capsys):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("HYPERLOOM_PREFER_HIP_VISIBLE_DEVICES", "1")

    cli_preflight._unset_hip_visible_devices()

    import os as _os

    assert _os.environ["HIP_VISIBLE_DEVICES"] == "0,1"
    assert "ROCR_VISIBLE_DEVICES" not in _os.environ
    assert "explicit HIP visibility" in capsys.readouterr().out


def _make_args(**overrides) -> argparse.Namespace:
    """Build a minimal Namespace; translates legacy ``critic_mock`` into ``critic_backend``."""
    base = dict(
        claude_model="claude-opus-4-7",
        codex_model="gpt-5.4",
        critic_backend="mock",
        no_kernel=False,
    )
    if "critic_mock" in overrides:
        base["critic_backend"] = "mock" if overrides.pop("critic_mock") else "agent"
    base.update(overrides)
    return argparse.Namespace(**base)


def test_resolve_robustness_choice_defaults_to_agent():
    args = _make_args(robustness_backend=None)

    assert cli.DEFAULT_ROBUSTNESS_BACKEND == "agent"
    assert cli._resolve_robustness_choice(args) == "agent"


def test_resolve_robustness_choice_explicit_mock_wins():
    args = _make_args(robustness_backend="mock")

    assert cli._resolve_robustness_choice(args) == "mock"


def test_resolve_robustness_choice_keeps_the_agent_on_multi_node():
    """Multi-node runs the agent on its node-agnostic signals; the local probe
    is what gets disabled, not the whole backend."""
    args = _make_args(robustness_backend=None, nodes=4)

    assert cli._resolve_robustness_choice(args) == "agent"


def test_resolve_robustness_choice_env_override_still_works(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DEFAULT_ROBUSTNESS_BACKEND", "mock")
    reloaded_cli = importlib.reload(cli)
    try:
        args = _make_args(robustness_backend=None)
        assert reloaded_cli.DEFAULT_ROBUSTNESS_BACKEND == "mock"
        assert reloaded_cli._resolve_robustness_choice(args) == "mock"
    finally:
        monkeypatch.delenv(
            "INFERENCE_OPTIMIZER_DEFAULT_ROBUSTNESS_BACKEND",
            raising=False,
        )
        importlib.reload(cli)


def test_validate_claude_model_rejects_unsupported_arg(monkeypatch, capsys):
    """With custom models explicitly disabled, unsupported ids abort before the catalog probe."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", "0")
    probe_calls: list[str] = []

    def _no_probe(**kwargs):
        probe_calls.append(kwargs.get("base_url", ""))
        raise AssertionError("probe should not be reached on static-gate fail")

    monkeypatch.setattr(cli, "_probe_llm_catalog", _no_probe)

    args = _make_args(claude_model="claude-opus-4-5")
    with pytest.raises(SystemExit) as exc_info:
        cli._validate_and_resolve_claude_model(args, None)
    assert exc_info.value.code == 2
    assert probe_calls == []
    err = capsys.readouterr().err
    assert "claude-opus-4-5" in err
    assert "claude-opus-4-7" in err
    assert "claude-opus-4-6" in err


def test_validate_claude_model_custom_allowed_when_optout_set(monkeypatch, capsys):
    """Opt-out lets a non-AMD orchestration model pass when in catalog."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", "1")
    # Pin a probe URL so the stubbed catalog probe runs.
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "https://gw.example/v1")
    monkeypatch.setattr(
        cli,
        "_probe_llm_catalog",
        lambda **kw: {"my-org/custom-claude", "gpt-5.4"},
    )
    args = _make_args(claude_model="my-org/custom-claude")
    catalog = cli._validate_and_resolve_claude_model(args, None)

    assert args.claude_model == "my-org/custom-claude"
    assert "my-org/custom-claude" in catalog
    out = capsys.readouterr().out
    assert "confirmed in gateway catalog" in out


def test_validate_claude_model_custom_optout_no_amd_fallback(monkeypatch, capsys):
    """Under opt-out a catalog miss errors with no silent opus-4-6 fallback."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "https://gw.example/v1")
    monkeypatch.setattr(
        cli,
        "_probe_llm_catalog",
        lambda **kw: {"claude-opus-4-6", "gpt-5.4"},
    )
    args = _make_args(claude_model="my-org/custom-claude")
    with pytest.raises(SystemExit) as exc_info:
        cli._validate_and_resolve_claude_model(args, None)
    assert exc_info.value.code == 2
    assert args.claude_model == "my-org/custom-claude"
    err = capsys.readouterr().err
    assert "not present in gateway catalog" in err


def test_validate_claude_model_custom_optout_rejects_empty(monkeypatch, capsys):
    """Opt-out with an empty model id aborts before the probe."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", "1")

    def _no_probe(**kwargs):
        raise AssertionError("probe should not run on empty-model abort")

    monkeypatch.setattr(cli, "_probe_llm_catalog", _no_probe)
    args = _make_args(claude_model="")
    with pytest.raises(SystemExit) as exc_info:
        cli._validate_and_resolve_claude_model(args, None)
    assert exc_info.value.code == 2
    assert "is empty" in capsys.readouterr().err


def test_validate_claude_model_custom_allowed_by_default(monkeypatch, capsys):
    """Default (env unset): custom orchestration models pass when the catalog contains them."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "https://gw.example/v1")
    monkeypatch.setattr(
        cli,
        "_probe_llm_catalog",
        lambda **kw: {"my-org/custom-claude", "gpt-5.4"},
    )
    args = _make_args(claude_model="my-org/custom-claude")
    catalog = cli._validate_and_resolve_claude_model(args, None)

    assert args.claude_model == "my-org/custom-claude"
    assert "my-org/custom-claude" in catalog
    assert "confirmed in gateway catalog" in capsys.readouterr().out


def test_validate_claude_model_deepseek_allowed_by_default(monkeypatch, capsys):
    """DeepSeek-style orchestration models are valid when the configured gateway serves them."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "https://api.deepseek.com/anthropic")
    monkeypatch.setattr(
        cli,
        "_probe_llm_catalog",
        lambda **kw: {"deepseek-v4-pro"},
    )
    args = _make_args(claude_model="deepseek-v4-pro")
    catalog = cli._validate_and_resolve_claude_model(args, None)

    assert args.claude_model == "deepseek-v4-pro"
    assert "deepseek-v4-pro" in catalog
    assert "confirmed in gateway catalog" in capsys.readouterr().out


def test_validate_claude_model_custom_explicitly_disabled_still_hard_gates(monkeypatch, capsys):
    """Explicit opt-out disable: custom model is rejected by the static gate."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", "0")

    def _no_probe(**kwargs):
        raise AssertionError("probe should not run on static-gate fail")

    monkeypatch.setattr(cli, "_probe_llm_catalog", _no_probe)
    args = _make_args(claude_model="my-org/custom-claude")
    with pytest.raises(SystemExit) as exc_info:
        cli._validate_and_resolve_claude_model(args, None)
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL=1" in err


def test_validate_claude_model_opus_5_in_catalog_keeps_choice(monkeypatch, capsys):
    """The new default passes the gateway gate; the gateway spelling also resolves."""
    # The AMD gateway lists it as "Claude-Opus-5"; the probe folds that form.
    assert cli._catalog_compare_model_id("Claude-Opus-5") == "claude-opus-5"
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "https://gw.example/v1")
    monkeypatch.setattr(
        cli,
        "_probe_llm_catalog",
        lambda **kw: {"claude-opus-5", "claude-opus-4-6", "gpt-5.4"},
    )
    args = _make_args(claude_model="claude-opus-5")
    cli._validate_and_resolve_claude_model(args, None)

    assert args.claude_model == "claude-opus-5"
    assert "confirmed in gateway catalog" in capsys.readouterr().out


def test_validate_claude_model_4_7_in_catalog_keeps_choice(monkeypatch, capsys):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "https://gw.example/v1")
    monkeypatch.setattr(
        cli,
        "_probe_llm_catalog",
        lambda **kw: {"claude-opus-4-7", "claude-opus-4-6", "gpt-5.4"},
    )
    args = _make_args(claude_model="claude-opus-4-7")
    catalog = cli._validate_and_resolve_claude_model(args, None)

    assert args.claude_model == "claude-opus-4-7"
    assert "claude-opus-4-7" in catalog


def test_validate_claude_model_probes_anthropic_url_in_dual_entry(monkeypatch):
    """Dual entry: the Claude catalog probe targets the Anthropic URL + key, not OpenAI."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-user-token")
    monkeypatch.delenv("_".join(("OPENAI", "API", "KEY")), raising=False)

    seen: dict[str, str] = {}

    def _capture(**kw):
        seen["base_url"] = kw.get("base_url", "")
        seen["api_key"] = kw.get("api_key", "")
        return {"claude-opus-4-7", "gpt-5.4"}

    monkeypatch.setattr(cli, "_probe_llm_catalog", _capture)
    args = _make_args(claude_model="claude-opus-4-7")
    cli._validate_and_resolve_claude_model(args, None)

    assert seen["base_url"] == "https://api.anthropic.com"
    assert seen["api_key"] == "anthropic-user-token"


def test_validate_claude_model_falls_back_to_openai_url_single_gateway(monkeypatch):
    """Single gateway: with no ANTHROPIC_BASE_URL, the probe uses the OpenAI URL."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.delenv("_".join(("ANTHROPIC", "API", "KEY")), raising=False)
    monkeypatch.delenv("_".join(("ANTHROPIC", "AUTH", "TOKEN")), raising=False)
    monkeypatch.setenv("_".join(("OPENAI", "API", "KEY")), "gateway-key")

    seen: dict[str, str] = {}

    def _capture(**kw):
        seen["base_url"] = kw.get("base_url", "")
        seen["api_key"] = kw.get("api_key", "")
        return {"claude-opus-4-7", "gpt-5.4"}

    monkeypatch.setattr(cli, "_probe_llm_catalog", _capture)
    args = _make_args(claude_model="claude-opus-4-7")
    cli._validate_and_resolve_claude_model(args, None)

    assert seen["base_url"] == "https://gateway.example/v1"
    assert seen["api_key"] == "gateway-key"


def test_validate_claude_model_skips_probe_for_oauth_only(monkeypatch, capsys):
    """The catalog probe is bearer-authenticated; a subscription token has nothing
    to send, so probing would only fail with a misleading auth error."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("_".join(("CLAUDE", "CODE", "OAUTH", "TOKEN")), "sk-ant-oat01-fake")
    for var in (
        "_".join(("ANTHROPIC", "API", "KEY")),
        "_".join(("ANTHROPIC", "AUTH", "TOKEN")),
        "_".join(("DEEPSEEK", "API", "KEY")),
        "_".join(("OPENAI", "API", "KEY")),
    ):
        monkeypatch.delenv(var, raising=False)

    def _no_probe(**kw):
        raise AssertionError("catalog probe should not run without a bearer credential")

    monkeypatch.setattr(cli, "_probe_llm_catalog", _no_probe)
    args = _make_args(claude_model="claude-opus-5")
    result = cli._validate_and_resolve_claude_model(args, None)

    assert result is None
    assert args.claude_model == "claude-opus-5"
    out = capsys.readouterr().out
    assert "catalog probe skipped: oauth-only credential" in out


def _oauth_only_env(monkeypatch, *, base_url: str) -> None:
    """An oauth-only shell pointed at ``base_url``."""
    for var in (
        "_".join(("ANTHROPIC", "API", "KEY")),
        "_".join(("ANTHROPIC", "AUTH", "TOKEN")),
        "_".join(("DEEPSEEK", "API", "KEY")),
        "_".join(("OPENAI", "API", "KEY")),
        "OPENAI_BASE_URL",
        "LLM_GATEWAY_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("_".join(("CLAUDE", "CODE", "OAUTH", "TOKEN")), "sk-ant-oat01-fake")
    if base_url:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)
    else:
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)


def test_preflight_warns_when_a_subscription_token_targets_a_foreign_endpoint(monkeypatch, capsys):
    """A subscription token only authenticates against Anthropic itself, so a
    third-party gateway both fails and puts the credential on the wire to a
    host that was never meant to see it. Every other gate reads this shape as
    a valid Anthropic side and stays silent."""
    _oauth_only_env(monkeypatch, base_url="https://gateway.internal.example/anthropic")

    cli_credentials._validate_credentials()

    err = capsys.readouterr().err
    assert "https://gateway.internal.example/anthropic" in err
    assert "only valid against https://api.anthropic.com" in err


@pytest.mark.parametrize("base_url", ["", "https://api.anthropic.com", "https://api.anthropic.com/"])
def test_preflight_accepts_a_subscription_token_on_the_official_endpoint(monkeypatch, capsys, base_url):
    """The supported shape, with or without an explicit official URL."""
    _oauth_only_env(monkeypatch, base_url=base_url)

    cli_credentials._validate_credentials()

    assert "only valid against" not in capsys.readouterr().err


def test_provider_only_mode_reads_a_subscription_token_as_anthropic_only(monkeypatch):
    """Without this the token yields no provider-only mode, so a stale OpenAI
    side from the kernel-agent env file is never suppressed."""
    _oauth_only_env(monkeypatch, base_url="")

    assert cli_preflight._provider_only_mode() == "anthropic"
    # Nothing else in this shell carries the verdict: drop the token and the
    # mode collapses, which is what the pre-registry code returned all along.
    monkeypatch.delenv("_".join(("CLAUDE", "CODE", "OAUTH", "TOKEN")))
    assert cli_preflight._provider_only_mode() == ""


def test_claude_config_json_is_left_alone_in_subscription_mode(monkeypatch, tmp_path, capsys):
    """customApiUrl would point the CLI away from the only endpoint that accepts
    the token, so subscription mode must not touch the operator's config."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _oauth_only_env(monkeypatch, base_url="")
    config_path = tmp_path / ".claude" / "config.json"

    cli_credentials._reset_claude_config_to_upstream("", "https://gateway.internal.example/anthropic")

    assert not config_path.exists()
    assert "config.json left alone" in capsys.readouterr().out


def test_claude_config_json_still_written_for_a_gateway_bearer_token(monkeypatch, tmp_path):
    """The skip must key off "is this run on the subscription", not off an empty
    primaryApiKey: this host authenticates through ANTHROPIC_AUTH_TOKEN, so it
    arrives with no ANTHROPIC_API_KEY while genuinely needing the gateway URL."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _oauth_only_env(monkeypatch, base_url="https://gateway.internal.example/anthropic")
    monkeypatch.setenv("_".join(("ANTHROPIC", "AUTH", "TOKEN")), "gateway-bearer")
    config_path = tmp_path / ".claude" / "config.json"

    cli_credentials._reset_claude_config_to_upstream("", "https://gateway.internal.example/anthropic")

    written = json.loads(config_path.read_text(encoding="utf-8"))
    assert written["customApiUrl"] == "https://gateway.internal.example/anthropic"
    assert written["primaryApiKey"] == "", "the subscription token must never reach primaryApiKey"


def test_validate_claude_model_still_probes_when_oauth_accompanies_an_api_key(monkeypatch):
    """An API key alongside the token can authenticate the probe, so it still runs."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("_".join(("CLAUDE", "CODE", "OAUTH", "TOKEN")), "sk-ant-oat01-fake")
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-user-token")
    monkeypatch.delenv("_".join(("OPENAI", "API", "KEY")), raising=False)

    seen: dict[str, str] = {}

    def _capture(**kw):
        seen["api_key"] = kw.get("api_key", "")
        return {"claude-opus-5"}

    monkeypatch.setattr(cli, "_probe_llm_catalog", _capture)
    args = _make_args(claude_model="claude-opus-5")
    cli._validate_and_resolve_claude_model(args, None)

    assert seen["api_key"] == "anthropic-user-token"


def test_validate_claude_model_split_entry_no_models_route_proceeds(monkeypatch, capsys):
    """Dual entry: Anthropic side returns 404/405 for /models (no catalog route)
    → proceed without probing the OpenAI side. The OpenAI catalog must never
    gate a Claude model."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-user-token")
    monkeypatch.setenv("_".join(("OPENAI", "API", "KEY")), "openai-user-token")

    probed: list[str] = []

    def _probe(**kw):
        url = kw.get("base_url", "")
        probed.append(url)
        if "anthropic" in url or "deepseek" in url:
            return cli._CATALOG_NO_MODELS_ENDPOINT
        return {"gpt-5.4"}

    monkeypatch.setattr(cli, "_probe_llm_catalog", _probe)
    args = _make_args(claude_model="claude-opus-4-7")
    result = cli._validate_and_resolve_claude_model(args, None)

    # Only the Anthropic side is probed.
    assert probed == ["https://api.deepseek.com/anthropic"]
    assert result is None
    assert args.claude_model == "claude-opus-4-7"
    assert "no /models route" in capsys.readouterr().out.lower()


def test_validate_claude_model_split_entry_auth_error_refuses(monkeypatch):
    """Dual entry: Anthropic catalog probe fails with auth/network (None, not the
    404 sentinel) and custom models disabled → refuse to start."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", "0")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-user-token")
    monkeypatch.setenv("_".join(("OPENAI", "API", "KEY")), "openai-user-token")

    # None models a 401/403/network/5xx failure, not "no route".
    monkeypatch.setattr(cli, "_probe_llm_catalog", lambda **kw: None)
    args = _make_args(claude_model="claude-opus-4-7")
    with pytest.raises(SystemExit) as exc:
        cli._validate_and_resolve_claude_model(args, None)
    assert exc.value.code == 2


def test_catalog_probe_retries_the_other_side_only_when_a_route_is_missing(monkeypatch, capsys):
    """A dual-protocol gateway lists its models on the OpenAI side only.

    The Anthropic side answers 404 for /models, which is the sentinel rather
    than None -- so the candidate loop has to keep going on the sentinel, or the
    catalog is never read and every model stays unverified until the first call.

    Both halves of the stub mirror what api.deepseek.com actually answers:
    /anthropic/models is a 404 and /v1/models lists exactly deepseek-v4-pro and
    deepseek-v4-flash. That correspondence is what makes reading the catalog an
    improvement rather than a regression -- the default model has to be in the
    ids the gateway really serves, or resolution below exits 2 on the miss
    instead of proceeding on the old "no /models route" warning.
    """
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", "1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "sk-ds")
    monkeypatch.setenv("_".join(("OPENAI", "API", "KEY")), "sk-ds")

    probed: list[str] = []

    def _probe(*, base_url, api_key):
        probed.append(base_url)
        if base_url.endswith("/anthropic"):
            return cli._CATALOG_NO_MODELS_ENDPOINT
        return {"deepseek-v4-pro", "deepseek-v4-flash"}

    monkeypatch.setattr(cli, "_probe_llm_catalog", _probe)
    args = _make_args(claude_model="deepseek-v4-pro")

    catalog = cli._validate_and_resolve_claude_model(args, None)

    assert probed == ["https://api.deepseek.com/anthropic", "https://api.deepseek.com/v1"]
    assert catalog == {"deepseek-v4-pro", "deepseek-v4-flash"}
    assert args.claude_model == "deepseek-v4-pro"
    # Confirmed against a real catalog rather than waved through with a warning.
    out = capsys.readouterr().out.lower()
    assert "confirmed in gateway catalog" in out
    assert "cannot verify" not in out


def test_catalog_probe_does_not_retry_the_other_side_when_a_gateway_is_flaky(monkeypatch, capsys):
    """An unreachable Anthropic side must degrade, not get answered by OpenAI.

    A 5xx / auth / timeout returns None, not the missing-route sentinel. Probing
    the OpenAI side then answers a Claude question with an OpenAI catalog, and
    every allowlisted Claude id would fail against it -- turning a transient
    gateway blip into a hard exit.
    """
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", "1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://llm.amd.example/Anthropic")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.amd.example/Unified/v1")
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "ak-x")
    monkeypatch.setenv("_".join(("OPENAI", "API", "KEY")), "ak-x")

    probed: list[str] = []

    def _probe(*, base_url, api_key):
        probed.append(base_url)
        if base_url.endswith("/Anthropic"):
            return None
        return {"gpt-5.6-sol"}

    monkeypatch.setattr(cli, "_probe_llm_catalog", _probe)
    args = _make_args(claude_model="claude-opus-5")

    assert cli._validate_and_resolve_claude_model(args, None) is None
    assert probed == ["https://llm.amd.example/Anthropic"]
    assert args.claude_model == "claude-opus-5"
    assert "cannot verify" in capsys.readouterr().out.lower()


def test_validate_claude_model_custom_model_warns_when_catalog_unreachable(monkeypatch, capsys):
    """ALLOW_CUSTOM=1 + catalog unreachable → WARN and proceed (no sys.exit)."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", "1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(cli, "_probe_llm_catalog", lambda **kw: None)

    args = _make_args(claude_model="my-org/custom-claude")
    result = cli._validate_and_resolve_claude_model(args, None)

    assert result is None
    assert args.claude_model == "my-org/custom-claude"
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "cannot verify" in out.lower()


def test_validate_claude_model_4_7_missing_falls_back_to_4_6(monkeypatch, capsys):
    """Catalog has 4-6 only → arg rewritten with a WARN."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "https://gw.example/v1")
    monkeypatch.setattr(
        cli,
        "_probe_llm_catalog",
        lambda **kw: {"claude-opus-4-6", "claude-opus-4-5", "gpt-5.4"},
    )
    args = _make_args(claude_model="claude-opus-4-7")
    cli._validate_and_resolve_claude_model(args, None)

    assert args.claude_model == "claude-opus-4-6"
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "falling back" in out
    assert "claude-opus-4-6" in out


def test_validate_claude_model_opus_5_missing_falls_back_to_next_rung(monkeypatch, capsys):
    """A gateway that predates opus-5 must land on 4-8, not skip to the last rung.

    This is the common transition-period catalog: every older allowlist entry is
    present but the new default is not.
    """
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "https://gw.example/v1")
    monkeypatch.setattr(
        cli,
        "_probe_llm_catalog",
        lambda **kw: {"claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "gpt-5.4"},
    )
    args = _make_args(claude_model="claude-opus-5")
    cli._validate_and_resolve_claude_model(args, None)

    assert args.claude_model == "claude-opus-4-8"
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "falling back" in out


def test_validate_claude_model_fallback_walks_the_allowlist_order(monkeypatch, capsys):
    """The ladder skips rungs the gateway lacks and stops at the first it serves."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "https://gw.example/v1")
    monkeypatch.setattr(
        cli,
        "_probe_llm_catalog",
        lambda **kw: {"claude-opus-4-7", "claude-opus-4-6", "gpt-5.4"},
    )
    args = _make_args(claude_model="claude-opus-5")
    cli._validate_and_resolve_claude_model(args, None)

    assert args.claude_model == "claude-opus-4-7"


def test_validate_claude_model_neither_in_catalog_aborts(monkeypatch, capsys):
    """Catalog missing all allowed Claude models -> sys.exit(2)."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "https://gw.example/v1")
    monkeypatch.setattr(
        cli,
        "_probe_llm_catalog",
        lambda **kw: {"claude-opus-4-5", "claude-haiku-4-5-20251001", "gpt-5.4"},
    )
    args = _make_args(claude_model="claude-opus-4-7")

    with pytest.raises(SystemExit) as exc_info:
        cli._validate_and_resolve_claude_model(args, None)
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "claude-opus-5" in err
    assert "claude-opus-4-8" in err
    assert "claude-opus-4-7" in err
    assert "claude-opus-4-6" in err
    assert "claude-opus-4-5" in err


def test_validate_claude_model_aborts_when_catalog_unreachable(monkeypatch, capsys):
    """With custom models explicitly disabled, catalog probe returned None → sys.exit(2)."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_ALLOW_CUSTOM_ORCH_MODEL", "0")
    monkeypatch.setattr(cli, "_probe_llm_catalog", lambda **kw: None)
    args = _make_args(claude_model="claude-opus-4-7")

    with pytest.raises(SystemExit) as exc_info:
        cli._validate_and_resolve_claude_model(args, None)
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "gateway catalog unreachable" in err
    assert "Refusing to start" in err


class _FakeResp:
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


def test_probe_llm_catalog_retries_on_transient_error_then_succeeds(monkeypatch):
    """First two attempts raise, third returns 200 → set returned + 2 sleeps."""
    sleeps: list[float] = []

    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    attempt: list[int] = [0]

    def _flaky_get(url, **kwargs):
        attempt[0] += 1
        if attempt[0] <= 2:
            raise RuntimeError(f"transient {attempt[0]}")
        return _FakeResp(200, {"data": [{"id": "claude-opus-4-7"}]})

    fake_httpx = type("FakeHttpx", (), {"get": staticmethod(_flaky_get)})
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    ids = cli._probe_llm_catalog(
        base_url="https://gateway/v1",
        api_key="test-token",
    )
    assert ids == {"claude-opus-4-7"}
    assert attempt[0] == 3
    assert sleeps[:2] == list(cli._CATALOG_RETRY_DELAYS_SEC[:2])


def test_probe_llm_catalog_passes_anthropic_custom_headers(monkeypatch):
    """Direct catalog probes must use the same gateway headers as Anthropic SDK calls."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: sub-key")
    seen: dict[str, Any] = {}

    def _get(url, **kwargs):
        seen["url"] = url
        seen["headers"] = kwargs.get("headers") or {}
        return _FakeResp(200, {"data": [{"id": "claude-opus-4-6"}]})

    fake_httpx = type("FakeHttpx", (), {"get": staticmethod(_get)})
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    ids = cli._probe_llm_catalog(
        base_url="https://llm.example.invalid/anthropic",
        api_key="dummy",
    )
    assert ids == {"claude-opus-4-6"}
    assert seen["url"] == "https://llm.example.invalid/anthropic/models"
    assert seen["headers"]["Ocp-Apim-Subscription-Key"] == "sub-key"
    assert seen["headers"]["Authorization"] == "Bearer dummy"


def test_probe_llm_catalog_uses_openai_custom_headers_for_openai_side(monkeypatch):
    """Strict per-side: probing the OpenAI base uses OPENAI_CUSTOM_HEADERS (not
    ANTHROPIC_CUSTOM_HEADERS)."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example.invalid/Unified/v1")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: openai-key")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: anthropic-key")
    seen: dict[str, Any] = {}

    def _get(url, **kwargs):
        seen["headers"] = kwargs.get("headers") or {}
        return _FakeResp(200, {"data": [{"id": "gpt-5.4"}]})

    fake_httpx = type("FakeHttpx", (), {"get": staticmethod(_get)})
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    ids = cli._probe_llm_catalog(base_url="https://llm.example.invalid/Unified/v1", api_key="dummy")
    assert ids == {"gpt-5.4"}
    assert seen["headers"]["Ocp-Apim-Subscription-Key"] == "openai-key"


def test_probe_llm_catalog_normalizes_claude_catalog_ids(monkeypatch):
    """AMD gateway may return title-case dot-version Claude IDs."""
    monkeypatch.setattr("time.sleep", lambda s: None)

    def _get(url, **kwargs):
        return _FakeResp(200, {"data": [{"id": "Claude-Opus-4.6"}]})

    fake_httpx = type("FakeHttpx", (), {"get": staticmethod(_get)})
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    ids = cli._probe_llm_catalog(base_url="https://llm.example.invalid/anthropic", api_key="dummy")

    assert "Claude-Opus-4.6" in ids
    assert "claude-opus-4-6" in ids


def test_probe_llm_catalog_returns_none_when_all_attempts_fail(monkeypatch, capsys):
    """All 4 attempts fail → returns None."""
    monkeypatch.setattr("time.sleep", lambda s: None)

    def _always_500(url, **kwargs):
        return _FakeResp(500, None)

    fake_httpx = type("FakeHttpx", (), {"get": staticmethod(_always_500)})
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    ids = cli._probe_llm_catalog(
        base_url="https://gateway/v1",
        api_key="test-token",
    )
    assert ids is None
    out = capsys.readouterr().out
    expected_attempts = 1 + len(cli._CATALOG_RETRY_DELAYS_SEC)
    assert out.count("catalog probe attempt") == expected_attempts
    assert "exhausted" in out


def test_probe_llm_catalog_returns_none_for_empty_base_url(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert cli._probe_llm_catalog(base_url="", api_key="test-token") is None


def test_probe_llm_catalog_returns_sentinel_on_404_without_retry(monkeypatch):
    """404 (no /models route) → return the no-catalog sentinel immediately, no retry."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    attempt: list[int] = [0]

    def _get_404(url, **kwargs):
        attempt[0] += 1
        return _FakeResp(404, None)

    fake_httpx = type("FakeHttpx", (), {"get": staticmethod(_get_404)})
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    result = cli._probe_llm_catalog(base_url="https://api.deepseek.com/anthropic", api_key="test-token")
    assert result is cli._CATALOG_NO_MODELS_ENDPOINT
    assert attempt[0] == 1


def test_probe_llm_catalog_returns_none_on_401(monkeypatch):
    """401 is an auth failure, not "no route" → None (retries), never the sentinel."""
    monkeypatch.setattr("time.sleep", lambda s: None)

    def _get_401(url, **kwargs):
        return _FakeResp(401, None)

    fake_httpx = type("FakeHttpx", (), {"get": staticmethod(_get_401)})
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    result = cli._probe_llm_catalog(base_url="https://gateway/v1", api_key="bad-key")
    assert result is None


def test_smoke_test_codex_model_warns_when_missing(monkeypatch, capsys):
    args = _make_args(codex_model="gpt-99.9", critic_mock=False)
    monkeypatch.setattr(cli, "_probe_llm_catalog", lambda **kw: {"claude-opus-4-7", "gpt-5.4", "gpt-4.1"})
    cli._smoke_test_codex_model(args, ("https://anthropic", "https://openai/v1"))
    out = capsys.readouterr().out
    assert "WARNING" in out


def test_smoke_test_codex_model_probes_openai_side(monkeypatch, capsys):
    """Dual entry: Codex smoke probes the OpenAI URL, not the Anthropic one."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    seen: dict[str, str] = {}

    def _capture(**kw):
        seen["base_url"] = kw.get("base_url", "")
        return {"gpt-5.4"}

    monkeypatch.setattr(cli, "_probe_llm_catalog", _capture)
    args = _make_args(codex_model="gpt-5.4", critic_mock=False)
    cli._smoke_test_codex_model(args, ("https://api.anthropic.com", "https://api.openai.com/v1"))
    assert seen["base_url"] == "https://api.openai.com/v1"


def test_smoke_test_codex_model_skipped_when_unused(monkeypatch, capsys):
    """--critic-mock → no probe / no warn (Codex is only needed for the critic-agent path)."""
    args = _make_args(
        codex_model="gpt-totally-fake",
        critic_mock=True,
    )

    def _no_probe(**kw):
        raise AssertionError("probe should not run when Codex is unused")

    monkeypatch.setattr(cli, "_probe_llm_catalog", _no_probe)
    cli._smoke_test_codex_model(args, ("https://anthropic", "https://openai/v1"))
    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert "gpt-totally-fake" not in out


def test_smoke_test_codex_model_confirms_when_present(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_probe_llm_catalog", lambda **kw: {"claude-opus-4-7", "gpt-5.4"})
    args = _make_args(codex_model="gpt-5.4", critic_mock=False)
    cli._smoke_test_codex_model(args, ("https://anthropic", "https://openai/v1"))
    out = capsys.readouterr().out
    assert "confirmed in gateway catalog" in out
    assert "WARNING" not in out


def test_smoke_test_codex_model_falls_back_to_next_rung(monkeypatch, capsys):
    """A gateway that predates the default Codex model degrades at preflight.

    The Codex side is WARN-only, so without a ladder this would sail past
    preflight and fail on the first Codex turn instead.
    """
    monkeypatch.setattr(cli, "_probe_llm_catalog", lambda **kw: {"gpt-5.5", "gpt-5.4"})
    args = _make_args(codex_model="gpt-5.6-sol", critic_mock=False)
    cli._smoke_test_codex_model(args, ("https://anthropic", "https://openai/v1"))

    assert args.codex_model == "gpt-5.5"
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "falling back" in out


def test_smoke_test_codex_model_ladder_skips_missing_rungs(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_probe_llm_catalog", lambda **kw: {"gpt-5.4", "gpt-4.1"})
    args = _make_args(codex_model="gpt-5.6-sol", critic_mock=False)
    cli._smoke_test_codex_model(args, ("https://anthropic", "https://openai/v1"))

    assert args.codex_model == "gpt-5.4"


def test_smoke_test_codex_model_leaves_custom_ids_alone(monkeypatch, capsys):
    """An operator-chosen id outside the ladder is reported, never rewritten."""
    monkeypatch.setattr(cli, "_probe_llm_catalog", lambda **kw: {"gpt-5.5", "gpt-5.4"})
    args = _make_args(codex_model="my-org/custom-gpt", critic_mock=False)
    cli._smoke_test_codex_model(args, ("https://anthropic", "https://openai/v1"))

    assert args.codex_model == "my-org/custom-gpt"
    out = capsys.readouterr().out
    assert "WARNING" in out
    # The warning names known-good ids so the operator has something to pass.
    assert "gpt-5.6-sol" in out


def test_openai_only_deploy_walks_the_codex_ladder_before_deriving_claude(monkeypatch, capsys):
    """OpenAI-only: CODEX_MODEL also drives orchestration, so its ladder must run first.

    Otherwise ``args.claude_model`` is derived from a codex id the gateway does
    not serve, and the Claude gate hard-aborts on a model the operator never
    chose -- with the Codex ladder never getting a turn.
    """
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gw.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    for var in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "https://gw.example/v1")
    # A gateway that has not picked up the new default yet.
    monkeypatch.setattr(cli, "_probe_llm_catalog", lambda **kw: {"gpt-5.5", "gpt-5.4"})

    assert cli._claude_model_should_follow_codex() is True
    args = _make_args(codex_model="gpt-5.6-sol", critic_mock=True)
    cli._resolve_models_for_run(args, None)

    assert args.codex_model == "gpt-5.5"
    assert args.claude_model == "gpt-5.5"


def test_openai_only_ladder_runs_even_with_a_mock_critic(monkeypatch, capsys):
    """The ladder cannot be gated on the critic here: codex_model drives orchestration."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gw.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    for var in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CATALOG_PROBE_URL", "https://gw.example/v1")
    monkeypatch.setattr(cli, "_probe_llm_catalog", lambda **kw: {"gpt-5.4"})

    args = _make_args(codex_model="gpt-5.6-sol", critic_mock=True)
    cli._resolve_models_for_run(args, None)

    assert args.codex_model == "gpt-5.4"
    assert args.claude_model == "gpt-5.4"


def test_smoke_test_codex_model_warns_on_probe_failure(monkeypatch, capsys):
    """OpenAI-side catalog unreachable → WARN-only (does not block startup)."""
    monkeypatch.setattr(cli, "_probe_llm_catalog", lambda **kw: None)
    args = _make_args(codex_model="gpt-5.4", critic_mock=False)
    cli._smoke_test_codex_model(args, ("https://anthropic", "https://openai/v1"))
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "unreachable" in out


def test_smoke_test_codex_model_skips_for_anthropic_only_fallback(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    def _no_probe(**kw):
        raise AssertionError("Anthropic-only fallback does not use CodexBackend")

    monkeypatch.setattr(cli, "_probe_llm_catalog", _no_probe)
    args = _make_args(codex_model="claude-sonnet-4-5-20250929", critic_mock=False)
    cli._smoke_test_codex_model(args, ("https://api.anthropic.com", ""))

    assert capsys.readouterr().out == ""


def test_parser_anthropic_only_empty_codex_model_uses_claude_model(monkeypatch):
    """With only Anthropic configured, an empty CODEX_MODEL follows CLAUDE_MODEL."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://llm.example.invalid/anthropic")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-4-6")
    monkeypatch.setenv("CODEX_MODEL", "")

    args = _build_parser().parse_args(["optimize", "--model", "/m", "--framework", "vllm"])

    assert args.claude_model == "claude-opus-4-6"
    assert args.codex_model == "claude-opus-4-6"
    assert cli._codex_model_should_follow_claude() is True


def test_parser_dual_protocol_gateway_empty_codex_model_uses_gateway_model(monkeypatch):
    """An empty CODEX_MODEL is filled by the shim, not left to the GPT default."""
    monkeypatch.setenv("_".join(("DEEPSEEK", "API", "KEY")), "deepseek-token")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("CLAUDE_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("CODEX_MODEL", "")

    for key, value in deepseek_compat_env().items():
        monkeypatch.setenv(key, value)
    args = cli._build_parser().parse_args(["optimize", "--model", "/m", "--framework", "vllm"])

    assert args.claude_model == "deepseek-v4-pro"
    assert args.codex_model == "deepseek-v4-pro"
    # Both protocol sides now resolve, so Codex no longer has to follow Claude.
    assert cli._codex_model_should_follow_claude() is False


def test_parser_retired_deepseek_key_only_defaults_to_gateway_model(monkeypatch):
    """A key-only legacy config must not inherit the Claude Opus / GPT defaults.

    The parser runs BEFORE ``_preflight`` normalizes the environment, so it has
    to resolve this on its own -- relying on preflight to export CLAUDE_MODEL
    would leave ``args.claude_model`` on the AMD default.
    """
    monkeypatch.setenv("_".join(("DEEPSEEK", "API", "KEY")), "deepseek-token")
    for name in (
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "ANTHROPIC_BASE_URL",
        "OPENAI_BASE_URL",
        "CLAUDE_MODEL",
        "CODEX_MODEL",
        "INFERENCE_OPTIMIZER_CLAUDE_FOLLOWS_CODEX",
    ):
        monkeypatch.delenv(name, raising=False)

    args = cli._build_parser().parse_args(["optimize", "--model", "/m", "--framework", "vllm"])

    assert args.claude_model == "deepseek-v4-pro"
    assert args.codex_model == "deepseek-v4-pro"


def test_parser_standard_dual_protocol_config_defaults_to_gateway_model(monkeypatch):
    """The configuration the docs recommend resolves its own models.

    Without this both sides would be handed ``claude-opus-5`` / ``gpt-5.6-sol``
    and fail on the first call, since DeepSeek serves neither.
    """
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "sk-deepseek")
    monkeypatch.setenv("_".join(("OPENAI", "API", "KEY")), "sk-deepseek")
    for name in (
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "CLAUDE_MODEL",
        "CODEX_MODEL",
        "INFERENCE_OPTIMIZER_CLAUDE_FOLLOWS_CODEX",
    ):
        monkeypatch.delenv(name, raising=False)

    args = cli._build_parser().parse_args(["optimize", "--model", "/m", "--framework", "vllm"])

    assert args.claude_model == "deepseek-v4-pro"
    assert args.codex_model == "deepseek-v4-pro"


@pytest.mark.parametrize(
    ("anthropic_url", "openai_url", "expected"),
    [
        ("https://api.deepseek.com/anthropic", "https://api.deepseek.com/v1", True),
        ("https://gw.example/x", "https://gw.example/x", True),
        ("https://llm.amd.example/Anthropic", "https://llm.amd.example/Unified/v1", True),
        ("https://api.anthropic.com", "https://api.openai.com/v1", False),
        ("", "https://api.openai.com/v1", False),
    ],
)
def test_same_gateway_recognizes_one_host_serving_both_protocols(anthropic_url, openai_url, expected):
    """The catalog probe may retry the OpenAI side only within one gateway.

    A dual-protocol gateway lists its models on the OpenAI side only, so a
    string-equality check would leave its catalog permanently unreadable.
    """
    assert cli._same_gateway(anthropic_url, openai_url) is expected


def test_parser_anthropic_only_generated_codex_default_uses_claude_model(monkeypatch):
    """Generated setup env defaults must not force GPT on an Anthropic-only run."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://llm.example.invalid/anthropic")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-4-6")
    monkeypatch.setenv("CODEX_MODEL", "gpt-5.4")

    args = _build_parser().parse_args(["optimize", "--model", "/m", "--framework", "vllm"])

    assert args.claude_model == "claude-opus-4-6"
    assert args.codex_model == "claude-opus-4-6"


def test_preflight_does_not_clear_cached_anthropic_only_codex_follow(
    monkeypatch, tmp_path, clean_url_env, stub_install_steps
):
    """An Anthropic-only deploy stays Anthropic-only across preflight: the OpenAI
    side is never populated from the Anthropic gateway, so Codex keeps following
    the Claude model."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://llm.example.invalid/anthropic")
    monkeypatch.setenv("_".join(("ANTHROPIC", "API", "KEY")), "anthropic-user-token")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("_".join(("OPENAI", "API", "KEY")), raising=False)
    monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-5")
    args = cli._build_parser().parse_args(
        [
            "optimize",
            "--model",
            "/m",
            "--framework",
            "vllm",
            "--codex-model",
            "gpt-5.5",
        ]
    )

    codex_follows_before = cli._codex_model_should_follow_claude()
    resolved = cli._preflight()
    if codex_follows_before:
        args.codex_model = args.claude_model

    # The OpenAI/Codex side stays unset, so Codex follows the Claude model.
    assert resolved == ("https://llm.example.invalid/anthropic", "")
    assert cli._codex_model_should_follow_claude() is True
    assert args.codex_model == "claude-sonnet-5"


def test_parser_openai_only_empty_claude_model_uses_codex_model(monkeypatch):
    """With only OpenAI configured, an empty CLAUDE_MODEL follows CODEX_MODEL."""
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example.invalid/Unified/v1")
    monkeypatch.setenv("CODEX_MODEL", "GPT-5.4")
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)

    args = _build_parser().parse_args(["optimize", "--model", "/m", "--framework", "vllm"])

    assert args.claude_model == "GPT-5.4"
    assert args.codex_model == "GPT-5.4"
    assert cli._claude_model_should_follow_codex() is True


def test_parser_marker_forces_claude_model_to_follow_codex(monkeypatch):
    """Launchers may pre-derive ANTHROPIC_BASE_URL while preserving OpenAI-only model semantics."""
    monkeypatch.setenv("INFERENCE_OPTIMIZER_CLAUDE_FOLLOWS_CODEX", "1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example.invalid/Unified/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://llm.example.invalid/Unified")
    monkeypatch.setenv("CODEX_MODEL", "GPT-5.5")
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)

    args = _build_parser().parse_args(["optimize", "--model", "/m", "--framework", "vllm"])

    assert args.claude_model == "GPT-5.5"
    assert args.codex_model == "GPT-5.5"


def test_validate_claude_model_openai_only_accepts_codex_model(monkeypatch):
    """OpenAI-only runs validate the followed orchestration model against the OpenAI catalog."""
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://llm.example.invalid/Unified/v1")
    monkeypatch.setenv("_".join(("OPENAI", "API", "KEY")), "openai-token")
    monkeypatch.setenv("CODEX_MODEL", "GPT-5.4")
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    seen: dict[str, str] = {}

    def _capture(**kw):
        seen["base_url"] = kw.get("base_url", "")
        seen["api_key"] = kw.get("api_key", "")
        return {"GPT-5.4"}

    monkeypatch.setattr(cli, "_probe_llm_catalog", _capture)
    args = _build_parser().parse_args(["optimize", "--model", "/m", "--framework", "vllm"])
    cli._validate_and_resolve_claude_model(
        args, ("https://llm.example.invalid/Unified", "https://llm.example.invalid/Unified/v1")
    )

    assert seen == {"base_url": "https://llm.example.invalid/Unified/v1", "api_key": "openai-token"}
    assert args.claude_model == "GPT-5.4"


"""preflight soft-degrade tests."""


import argparse
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


def _ns(**overrides) -> argparse.Namespace:
    defaults: dict = {
        "degraded_kb": False,
        "degraded_pr": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _write_marker(
    marker_path: Path,
    *,
    kb_reachable: bool,
    pr_reachable: bool,
    kb_skipped: bool = False,
    pr_skipped: bool = False,
    kb_failure_reason: str | None = None,
    pr_failure_reason: str | None = None,
) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "kb_reachable": kb_reachable,
        "pr_reachable": pr_reachable,
        "kb_skipped": kb_skipped,
        "pr_skipped": pr_skipped,
    }
    if kb_failure_reason is not None:
        payload["kb_failure_reason"] = kb_failure_reason
    if pr_failure_reason is not None:
        payload["pr_failure_reason"] = pr_failure_reason
    marker_path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_run_writes_marker(marker_path: Path, **marker_kwargs):
    """Return a ``subprocess.run`` stub that writes ``marker_path`` and returns an appropriate rc."""

    def _runner(cmd, env=None, check=False, timeout=None):
        _write_marker(marker_path, **marker_kwargs)
        kb_skipped = marker_kwargs.get("kb_skipped", False)
        pr_skipped = marker_kwargs.get("pr_skipped", False)
        kb_ok = marker_kwargs.get("kb_reachable", False)
        pr_ok = marker_kwargs.get("pr_reachable", False)
        rc = 0
        if not kb_skipped and not kb_ok:
            rc = 1
        if not pr_skipped and not pr_ok:
            rc = 1
        return subprocess.CompletedProcess(cmd, rc)

    return _runner


@pytest.fixture
def marker_path(tmp_path, monkeypatch) -> Path:
    user_data = tmp_path / "user_data"
    monkeypatch.setenv("USER_DATA_PATH", str(user_data))
    monkeypatch.delenv("HYPERLOOM_PR_MONITOR_ENABLED", raising=False)
    return user_data / "runtime" / "recipe_kb" / ".kb_preflight.json"


def test_ir3_kb_ok_pr_ok(marker_path):
    args = _ns()
    with patch.object(
        cli_preflight.subprocess,
        "run",
        side_effect=_fake_run_writes_marker(marker_path, kb_reachable=True, pr_reachable=True),
    ):
        cli_preflight._run_ir3_preflight(args)
    assert args.recipe_kb_enabled is True
    assert args.pr_monitor_enabled is True
    assert args.kb_degraded_reason is None
    assert args.pr_degraded_reason is None


def test_ir3_kb_probe_unreachable_does_not_auto_degrade_recipe_kb(marker_path):
    """Recipe KB has no remote IR-3 probe; unreachable kb marker must not flip recipe_kb_enabled."""
    args = _ns()
    with patch.object(
        cli_preflight.subprocess,
        "run",
        side_effect=_fake_run_writes_marker(
            marker_path,
            kb_reachable=False,
            pr_reachable=True,
            kb_failure_reason="500",
        ),
    ):
        cli_preflight._run_ir3_preflight(args)
    assert args.recipe_kb_enabled is True
    assert args.pr_monitor_enabled is True
    assert args.kb_degraded_reason is None
    assert args.pr_degraded_reason is None


def test_ir3_kb_explicit_flag(marker_path):
    args = _ns(degraded_kb=True)
    seen_env: dict = {}

    def _runner(cmd, env=None, check=False, timeout=None):
        seen_env.update(env or {})
        _write_marker(marker_path, kb_reachable=False, pr_reachable=True, kb_skipped=True)
        return subprocess.CompletedProcess(cmd, 0)

    with patch.object(cli_preflight.subprocess, "run", side_effect=_runner):
        cli_preflight._run_ir3_preflight(args)
    assert "SKIP_KB_PROBE" not in seen_env
    assert "SKIP_PR_PROBE" not in seen_env
    assert args.recipe_kb_enabled is False
    assert args.kb_degraded_reason == "explicit_flag"
    assert args.pr_monitor_enabled is True
    assert args.pr_degraded_reason is None


def test_ir3_kb_401_with_token(marker_path, monkeypatch):
    monkeypatch.setenv("KB_SERVICE_TOKEN", "tok-abc")
    args = _ns()
    with patch.object(
        cli_preflight.subprocess,
        "run",
        side_effect=_fake_run_writes_marker(marker_path, kb_reachable=True, pr_reachable=True),
    ):
        cli_preflight._run_ir3_preflight(args)
    assert args.recipe_kb_enabled is True
    assert args.kb_degraded_reason is None


def test_ir3_kb_401_missing_token_does_not_auto_degrade_recipe_kb(marker_path, monkeypatch):
    monkeypatch.delenv("KB_SERVICE_TOKEN", raising=False)
    args = _ns()
    with patch.object(
        cli_preflight.subprocess,
        "run",
        side_effect=_fake_run_writes_marker(
            marker_path,
            kb_reachable=False,
            pr_reachable=True,
            kb_failure_reason="missing_token",
        ),
    ):
        cli_preflight._run_ir3_preflight(args)
    assert args.recipe_kb_enabled is True
    assert args.kb_degraded_reason is None


def test_ir3_pr_explicit_flag_kb_ok(marker_path):
    args = _ns(degraded_pr=True)
    seen_env: dict = {}

    def _runner(cmd, env=None, check=False, timeout=None):
        seen_env.update(env or {})
        _write_marker(marker_path, kb_reachable=True, pr_reachable=False, pr_skipped=True)
        return subprocess.CompletedProcess(cmd, 0)

    with patch.object(cli_preflight.subprocess, "run", side_effect=_runner):
        cli_preflight._run_ir3_preflight(args)
    assert seen_env.get("SKIP_PR_PROBE") == "1"
    assert args.recipe_kb_enabled is True
    assert args.kb_degraded_reason is None
    assert args.pr_monitor_enabled is False
    assert args.pr_degraded_reason == "explicit_flag"


def test_ir3_both_flags_short_circuit(marker_path):
    args = _ns(degraded_kb=True, degraded_pr=True)
    with patch.object(cli_preflight.subprocess, "run") as run_mock:
        cli_preflight._run_ir3_preflight(args)
        run_mock.assert_not_called()
    assert args.recipe_kb_enabled is False
    assert args.pr_monitor_enabled is False
    assert args.kb_degraded_reason == "explicit_flag"
    assert args.pr_degraded_reason == "explicit_flag"


def test_ir3_preflight_does_not_inject_recipe_kb_url(marker_path, monkeypatch):
    monkeypatch.delenv("RECIPE_KB_KB_URL", raising=False)
    args = _ns()
    seen_env: dict = {}

    def _runner(cmd, env=None, check=False, timeout=None):
        seen_env.update(env or {})
        _write_marker(
            marker_path,
            kb_reachable=False,
            pr_reachable=True,
            kb_skipped=True,
        )
        return subprocess.CompletedProcess(cmd, 0)

    with patch.object(cli_preflight.subprocess, "run", side_effect=_runner):
        cli_preflight._run_ir3_preflight(args)
    assert "RECIPE_KB_KB_URL" not in seen_env
    assert args.recipe_kb_enabled is True
    assert args.kb_degraded_reason is None


def test_ir3_unreachable_local_default_disables_pr_monitor(marker_path, monkeypatch):
    from hyperloom.common.pr_monitor_urls import DEFAULT_KB_STORE_URL

    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "local")
    monkeypatch.delenv("KB_STORE_URL", raising=False)
    args = _ns()
    seen_env: dict = {}

    def _runner(cmd, env=None, check=False, timeout=None):
        seen_env.update(env or {})
        _write_marker(
            marker_path,
            kb_reachable=False,
            pr_reachable=False,
            kb_skipped=True,
            pr_failure_reason="timeout",
        )
        return subprocess.CompletedProcess(cmd, 1)

    with patch.object(cli_preflight.subprocess, "run", side_effect=_runner):
        outcome = cli_preflight._run_ir3_preflight(args)

    assert seen_env["KB_STORE_URL"] == DEFAULT_KB_STORE_URL
    assert args.recipe_kb_enabled is True
    assert args.pr_monitor_enabled is False
    assert args.pr_degraded_reason == "ir3_auto"
    assert "HYPERLOOM_PR_MONITOR_ENABLED" not in os.environ
    assert outcome["detail"]["pr_monitor"] == {
        "enabled": False,
        "reason": "ir3_unreachable",
    }


def test_ir3_real_shell_probes_local_default_url(marker_path, tmp_path, monkeypatch):
    from hyperloom.common.pr_monitor_urls import DEFAULT_KB_STORE_URL

    monkeypatch.setenv("KNOWLEDGE_STORE_MODE", "local")
    monkeypatch.delenv("KB_STORE_URL", raising=False)
    curl_log = tmp_path / "curl.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_curl = fake_bin / "curl"
    fake_curl.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CURL_LOG"\nprintf \'{"ok":true}\\n__HTTP_CODE__:200\\n\'\n',
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    monkeypatch.setenv("CURL_LOG", str(curl_log))
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")

    args = _ns()
    outcome = cli_preflight._run_ir3_preflight(args)

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert f"{DEFAULT_KB_STORE_URL}/pr-monitor/v1/healthz" in curl_log.read_text(encoding="utf-8")
    assert marker["pr_reachable"] is True
    assert marker["pr_skipped"] is False
    assert args.pr_monitor_enabled is True
    assert outcome["status"] == "applied"


def test_cli_parser_exposes_degraded_flags():
    parser = _build_parser()
    args = parser.parse_args(["optimize", "--model", "/x", "--degraded-kb"])
    assert args.degraded_kb is True
    assert args.degraded_pr is False
    args = parser.parse_args(["optimize", "--model", "/x", "--degraded-pr"])
    assert args.degraded_pr is True
    assert args.degraded_kb is False
    args = parser.parse_args(["optimize", "--model", "/x"])
    assert args.degraded_kb is False
    assert args.degraded_pr is False


# Framework guard


def test_expected_framework_guard_rejects_mismatch(monkeypatch, capsys):
    monkeypatch.setenv("EXPECTED_FRAMEWORK", "vllm")
    with pytest.raises(SystemExit) as exc:
        cli._enforce_expected_framework("sglang")
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "EXPECTED_FRAMEWORK" in err
    assert "vllm" in err
    assert "sglang" in err


def test_expected_framework_guard_accepts_match(monkeypatch):
    monkeypatch.setenv("EXPECTED_FRAMEWORK", "VLLM")
    cli._enforce_expected_framework("vllm")


def test_expected_framework_guard_namespaced_env_var(monkeypatch):
    monkeypatch.delenv("EXPECTED_FRAMEWORK", raising=False)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXPECTED_FRAMEWORK", "vllm")
    with pytest.raises(SystemExit) as exc:
        cli._enforce_expected_framework("sglang")
    assert exc.value.code == 2


def test_expected_framework_guard_namespaced_takes_precedence(monkeypatch):
    # Namespaced var wins over the compact one.
    monkeypatch.setenv("INFERENCE_OPTIMIZER_EXPECTED_FRAMEWORK", "vllm")
    monkeypatch.setenv("EXPECTED_FRAMEWORK", "sglang")
    with pytest.raises(SystemExit):
        cli._enforce_expected_framework("sglang")


def test_expected_framework_guard_unset_is_noop(monkeypatch):
    monkeypatch.delenv("EXPECTED_FRAMEWORK", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_EXPECTED_FRAMEWORK", raising=False)
    # No env pins -> guard is a no-op.
    cli._enforce_expected_framework("sglang")
    cli._enforce_expected_framework("anything")
