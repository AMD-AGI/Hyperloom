# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import re
import subprocess

from pathlib import Path

import pytest

from hyperloom.common.llm_config import deepseek_compat_env
from hyperloom.inference_optimizer import setup


class _Completed:
    returncode = 7


def test_setup_cli_forwards_flags_and_workspace_env(tmp_path: Path, monkeypatch):
    installer = tmp_path / "install_baremetal.sh"
    installer.write_text("#!/bin/sh\n", encoding="utf-8")
    seen: dict[str, object] = {}

    def _fake_run(cmd, *, env):
        seen["cmd"] = cmd
        seen["env"] = env
        return _Completed()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup, "_INSTALL_BAREMETAL_SH", installer)
    monkeypatch.setattr(setup, "_PACKAGE_SKILL", tmp_path / "SKILL.md")
    monkeypatch.setattr(setup.subprocess, "run", _fake_run)

    rc = setup.main(["--check-only", "--dry-run", "--", "--install-framework", "none"])

    assert rc == 7
    assert seen["cmd"] == [
        "bash",
        str(installer),
        "--check-only",
        "--dry-run",
        "--install-framework",
        "none",
    ]
    env = seen["env"]
    assert env["REPO_ROOT"] == str(tmp_path)
    assert "HYPERLOOM_ENV_FILE" not in env
    assert env["HYPERLOOM_SKILL_PATH"] == str(tmp_path / "SKILL.md")


def test_setup_cli_scrubs_ambient_llm_env_when_dotenv_exists(tmp_path: Path, monkeypatch):
    installer = tmp_path / "install_baremetal.sh"
    installer.write_text("#!/bin/sh\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "ANTHROPIC_API_KEY=<PLEASE_FILL_IN>",
                "ANTHROPIC_BASE_URL=https://api.anthropic.com",
                "CLAUDE_MODEL=claude-opus-4-8",
                "HYPERLOOM_RUN_MODE=baremetal",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def _fake_run(cmd, *, env):
        seen["cmd"] = cmd
        seen["env"] = env
        return _Completed()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REPO_ROOT", "/stale/root")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://llm.example.invalid/anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stale-anthropic-key")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: stale")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "stale-openai-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "stale-deepseek-key")
    monkeypatch.setenv("LLM_GATEWAY_KEY", "stale-gateway-key")
    monkeypatch.setenv("CLAUDE_MODEL", "stale-claude-model")
    monkeypatch.setenv("CODEX_MODEL", "stale-codex-model")
    monkeypatch.setattr(setup, "_INSTALL_BAREMETAL_SH", installer)
    monkeypatch.setattr(setup, "_PACKAGE_SKILL", tmp_path / "SKILL.md")
    monkeypatch.setattr(setup.subprocess, "run", _fake_run)

    rc = setup.main(["--", "--install-framework", "none"])

    assert rc == 7
    env = seen["env"]
    assert env["REPO_ROOT"] == str(tmp_path)
    assert "HYPERLOOM_ENV_FILE" not in env
    assert env["HYPERLOOM_SKILL_PATH"] == str(tmp_path / "SKILL.md")
    assert env["HYPERLOOM_SETUP_ENV_AUTHORITATIVE"] == "1"
    for key in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_CUSTOM_HEADERS",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "LLM_GATEWAY_KEY",
        "CLAUDE_MODEL",
        "CODEX_MODEL",
    ):
        assert key not in env


def test_setup_cli_scrubs_stale_workspace_runtime_env_when_dotenv_exists(tmp_path: Path, monkeypatch):
    installer = tmp_path / "install_baremetal.sh"
    installer.write_text("#!/bin/sh\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "ANTHROPIC_API_KEY=<PLEASE_FILL_IN>",
                "ANTHROPIC_BASE_URL=https://api.anthropic.com",
                "CLAUDE_MODEL=claude-opus-4-8",
                "USER_DATA_PATH=/new/workspace/session",
                "HYPERLOOM_RUN_MODE=baremetal",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def _fake_run(cmd, *, env):
        seen["cmd"] = cmd
        seen["env"] = env
        return _Completed()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USER_DATA_PATH", "/old/workspace/session")
    monkeypatch.setenv("HYPERLOOM_RUNTIME_DIR", "/old/workspace/session/runtime")
    monkeypatch.setenv("KERNEL_AGENT_ENV", "/old/workspace/session/runtime/kernel-agent.env.sh")
    monkeypatch.setenv("HYPERLOOM_ROOT", "/old/workspace/session/runtime/source-mirrors")
    monkeypatch.setenv("KERNEL_AGENT_ROOT", "/old/workspace/hyperloom/agents/kernel")
    monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", "/old/workspace/hyperloom/agents/kernel")
    monkeypatch.setenv("FRAMEWORK_AGENT_ROOT", "/old/workspace/hyperloom/agents/framework")
    monkeypatch.setenv("HYPERLOOM_SKILL_PATH", "/old/workspace/hyperloom/inference_optimizer/SKILL.md")
    monkeypatch.setenv("PYTHONPATH", "/old/workspace:/old/site-packages")
    monkeypatch.setattr(setup, "_INSTALL_BAREMETAL_SH", installer)
    monkeypatch.setattr(setup, "_PACKAGE_SKILL", tmp_path / "SKILL.md")
    monkeypatch.setattr(setup.subprocess, "run", _fake_run)

    rc = setup.main(["--", "--install-framework", "none"])

    assert rc == 7
    env = seen["env"]
    assert env["REPO_ROOT"] == str(tmp_path)
    assert env["HYPERLOOM_SKILL_PATH"] == str(tmp_path / "SKILL.md")
    assert env["HYPERLOOM_SETUP_ENV_AUTHORITATIVE"] == "1"
    for key in (
        "USER_DATA_PATH",
        "HYPERLOOM_RUNTIME_DIR",
        "KERNEL_AGENT_ENV",
        "HYPERLOOM_ROOT",
        "KERNEL_AGENT_ROOT",
        "HYPERLOOM_KERNEL_AGENT_ROOT",
        "FRAMEWORK_AGENT_ROOT",
        "PYTHONPATH",
    ):
        assert key not in env


def test_baremetal_setup_authoritative_anthropic_env_removes_openai_keys(tmp_path: Path):
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    script_text = install_script.read_text(encoding="utf-8")
    start = script_text.index("read_dotenv_var() {")
    end = script_text.index("\nwrite_runtime_dotenv() {")
    credential_functions = script_text[start:end]
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "ANTHROPIC_API_KEY=<PLEASE_FILL_IN>",
                "ANTHROPIC_BASE_URL=https://api.anthropic.com",
                "CLAUDE_MODEL=claude-opus-4-8",
                "HYPERLOOM_RUN_MODE=baremetal",
                "OPENAI_BASE_URL=https://api.anthropic.com",
                "OPENAI_API_KEY=stale-openai-key",
                "OPENAI_CUSTOM_HEADERS=stale-header: stale",
                "LLM_GATEWAY_KEY=stale-gateway-key",
                "SAFE_API_KEY=stale-safe-key",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runner = tmp_path / "run.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"DOTENV={dotenv}",
                "CHECK_ONLY=0",
                "DRY_RUN=0",
                "OPENAI_BASE_URL_ARG=",
                "log() { :; }",
                "warn() { :; }",
                'die() { echo "$*" >&2; exit 99; }',
                "is_interactive() { return 1; }",
                credential_functions,
                "unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL ANTHROPIC_CUSTOM_HEADERS",
                "unset DEEPSEEK_API_KEY DEEPSEEK_BASE_URL",
                "HYPERLOOM_SETUP_ENV_AUTHORITATIVE=1",
                "OPENAI_BASE_URL=https://api.anthropic.com",
                "OPENAI_API_KEY=ambient-openai-key",
                "OPENAI_CUSTOM_HEADERS='ambient-header: stale'",
                "LLM_GATEWAY_KEY=ambient-gateway-key",
                "resolve_credentials",
                f"printf 'OPENAI_BASE_URL=%s\n' \"${{OPENAI_BASE_URL-}}\" > {tmp_path / 'resolved-env.txt'}",
                f"printf 'OPENAI_API_KEY=%s\n' \"${{OPENAI_API_KEY-}}\" >> {tmp_path / 'resolved-env.txt'}",
                f"printf 'OPENAI_CUSTOM_HEADERS=%s\n' \"${{OPENAI_CUSTOM_HEADERS-}}\" >> {tmp_path / 'resolved-env.txt'}",
                f"printf 'LLM_GATEWAY_KEY=%s\n' \"${{LLM_GATEWAY_KEY-}}\" >> {tmp_path / 'resolved-env.txt'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(["bash", str(runner)], check=True)

    text = dotenv.read_text(encoding="utf-8")
    assert "ANTHROPIC_BASE_URL=https://api.anthropic.com" in text
    assert "ANTHROPIC_API_KEY=<PLEASE_FILL_IN>" in text
    assert "OPENAI_BASE_URL=" not in text
    assert "OPENAI_API_KEY=" not in text
    assert "OPENAI_CUSTOM_HEADERS=" not in text
    assert "LLM_GATEWAY_KEY=" not in text
    assert "SAFE_API_KEY=" not in text
    resolved_env = (tmp_path / "resolved-env.txt").read_text(encoding="utf-8")
    assert resolved_env == (
        "OPENAI_BASE_URL=\nOPENAI_API_KEY=\nOPENAI_CUSTOM_HEADERS=\nLLM_GATEWAY_KEY=\n"
    )


def test_baremetal_setup_migrates_retired_deepseek_env_to_both_sides(tmp_path: Path):
    """A legacy DEEPSEEK_* .env is rewritten into the standard two-sided form."""
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    script_text = install_script.read_text(encoding="utf-8")
    start = script_text.index("read_dotenv_var() {")
    end = script_text.index("\nwrite_runtime_dotenv() {")
    credential_functions = script_text[start:end]
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "HYPERLOOM_LLM_MODE=deepseek",
                "DEEPSEEK_API_KEY=deepseek-test-key",
                "DEEPSEEK_BASE_URL=https://api.deepseek.com/anthropic",
                "OPENAI_BASE_URL=https://gateway.example/v1",
                "OPENAI_API_KEY=stale-openai-key",
                "LLM_GATEWAY_KEY=stale-gateway-key",
                "SAFE_API_KEY=stale-safe-key",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runner = tmp_path / "deepseek-run.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                # The credentials in this scenario must come from .env alone, so
                # don't inherit any provider variable from the pytest process.
                "unset OPENAI_BASE_URL OPENAI_API_KEY OPENAI_CUSTOM_HEADERS",
                "unset ANTHROPIC_BASE_URL ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN",
                "unset CLAUDE_MODEL CODEX_MODEL GEAK_CLAUDE_MODEL",
                "unset DEEPSEEK_API_KEY DEEPSEEK_BASE_URL DEEPSEEK_MODEL",
                f"DOTENV={dotenv}",
                "CHECK_ONLY=0",
                "DRY_RUN=0",
                "log() { :; }",
                "warn() { :; }",
                'die() { echo "$*" >&2; exit 99; }',
                "is_interactive() { return 1; }",
                credential_functions,
                "HYPERLOOM_SETUP_ENV_AUTHORITATIVE=1",
                "SAFE_API_KEY=ambient-safe-key",
                "LLM_GATEWAY_KEY=ambient-gateway-key",
                "resolve_credentials",
                f"printf 'OPENAI_BASE_URL=%s\n' \"${{OPENAI_BASE_URL-}}\" > {tmp_path / 'deepseek-env.txt'}",
                f"printf 'OPENAI_API_KEY=%s\n' \"${{OPENAI_API_KEY-}}\" >> {tmp_path / 'deepseek-env.txt'}",
                f"printf 'ANTHROPIC_BASE_URL=%s\n' \"${{ANTHROPIC_BASE_URL-}}\" >> {tmp_path / 'deepseek-env.txt'}",
                f"printf 'ANTHROPIC_API_KEY=%s\n' \"${{ANTHROPIC_API_KEY-}}\" >> {tmp_path / 'deepseek-env.txt'}",
                f"printf 'CLAUDE_MODEL=%s\n' \"${{CLAUDE_MODEL-}}\" >> {tmp_path / 'deepseek-env.txt'}",
                f"printf 'DEEPSEEK_API_KEY=%s\n' \"${{DEEPSEEK_API_KEY-}}\" >> {tmp_path / 'deepseek-env.txt'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(["bash", str(runner)], check=True)

    text = dotenv.read_text(encoding="utf-8")
    # Retired variables are gone; both protocol sides now point at DeepSeek.
    assert "DEEPSEEK_API_KEY=" not in text
    assert "DEEPSEEK_BASE_URL=" not in text
    assert "ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic" in text
    assert "ANTHROPIC_API_KEY=deepseek-test-key" in text
    assert "OPENAI_BASE_URL=https://api.deepseek.com/v1" in text
    assert "OPENAI_API_KEY=deepseek-test-key" in text
    assert "SAFE_API_KEY=" not in text
    resolved_env = (tmp_path / "deepseek-env.txt").read_text(encoding="utf-8")
    assert resolved_env == (
        "OPENAI_BASE_URL=https://api.deepseek.com/v1\n"
        "OPENAI_API_KEY=deepseek-test-key\n"
        "ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic\n"
        "ANTHROPIC_API_KEY=deepseek-test-key\n"
        "CLAUDE_MODEL=deepseek-v4-pro\n"
        "DEEPSEEK_API_KEY=\n"
    )


def _baremetal_credential_functions() -> str:
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    script_text = install_script.read_text(encoding="utf-8")
    start = script_text.index("read_dotenv_var() {")
    end = script_text.index("\nwrite_runtime_dotenv() {")
    return script_text[start:end]


_CLEAN_PROVIDER_ENV = [
    "unset OPENAI_BASE_URL OPENAI_API_KEY OPENAI_CUSTOM_HEADERS",
    "unset ANTHROPIC_BASE_URL ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN",
    # Now that the shell reads the subscription token too, a developer machine
    # exporting one would otherwise change what these runs resolve.
    "unset CLAUDE_CODE_OAUTH_TOKEN",
    "unset CLAUDE_MODEL CODEX_MODEL GEAK_CLAUDE_MODEL",
    "unset DEEPSEEK_API_KEY DEEPSEEK_BASE_URL DEEPSEEK_MODEL",
]


def test_baremetal_setup_keeps_hand_written_dual_protocol_openai_side(tmp_path: Path):
    """The configuration the docs now recommend must survive a setup run.

    Both sides are on one host, so the OpenAI side is part of the same gateway
    credential rather than a second provider -- and that has to be decided from
    the URLs, not from whether a legacy DEEPSEEK_* migration happened to run.
    """
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic",
                "ANTHROPIC_API_KEY=deepseek-test-key",
                'ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: deepseek-test-key"',
                "OPENAI_BASE_URL=https://api.deepseek.com/v1",
                "OPENAI_API_KEY=deepseek-test-key",
                'OPENAI_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: deepseek-test-key"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runner = tmp_path / "dual-run.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                *_CLEAN_PROVIDER_ENV,
                f"DOTENV={dotenv}",
                "CHECK_ONLY=0",
                "DRY_RUN=0",
                "log() { :; }",
                "warn() { :; }",
                'die() { echo "$*" >&2; exit 99; }',
                "is_interactive() { return 1; }",
                _baremetal_credential_functions(),
                "HYPERLOOM_SETUP_ENV_AUTHORITATIVE=1",
                "resolve_credentials",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(["bash", str(runner)], check=True)

    text = dotenv.read_text(encoding="utf-8")
    assert "ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic" in text
    assert "OPENAI_BASE_URL=https://api.deepseek.com/v1" in text
    assert "OPENAI_API_KEY=deepseek-test-key" in text
    # A gateway that authenticates on a header needs both sides' headers to
    # survive: this deployment shape is exactly the one that cannot re-derive
    # them from the keys.
    assert 'ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: deepseek-test-key"' in text
    assert 'OPENAI_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: deepseek-test-key"' in text


def test_baremetal_setup_drops_openai_side_of_a_separate_provider(tmp_path: Path):
    """A different host is a second provider, and is still scrubbed."""
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "ANTHROPIC_BASE_URL=https://api.anthropic.com",
                "ANTHROPIC_API_KEY=anthropic-test-key",
                "OPENAI_BASE_URL=https://api.openai.com/v1",
                "OPENAI_API_KEY=openai-test-key",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runner = tmp_path / "split-run.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                *_CLEAN_PROVIDER_ENV,
                f"DOTENV={dotenv}",
                "CHECK_ONLY=0",
                "DRY_RUN=0",
                "log() { :; }",
                "warn() { :; }",
                'die() { echo "$*" >&2; exit 99; }',
                "is_interactive() { return 1; }",
                _baremetal_credential_functions(),
                "HYPERLOOM_SETUP_ENV_AUTHORITATIVE=1",
                "resolve_credentials",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(["bash", str(runner)], check=True)

    text = dotenv.read_text(encoding="utf-8")
    assert "ANTHROPIC_BASE_URL=https://api.anthropic.com" in text
    assert "OPENAI_BASE_URL=" not in text
    assert "OPENAI_API_KEY=" not in text


def test_baremetal_setup_keeps_legacy_env_when_validation_fails(tmp_path: Path):
    """A failed run must not consume the operator's only copy of the config."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("DEEPSEEK_BASE_URL=https://api.deepseek.com/anthropic\n", encoding="utf-8")
    runner = tmp_path / "fail-run.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                *_CLEAN_PROVIDER_ENV,
                f"DOTENV={dotenv}",
                "CHECK_ONLY=0",
                "DRY_RUN=0",
                "log() { :; }",
                "warn() { :; }",
                'die() { echo "$*" >&2; exit 99; }',
                "is_interactive() { return 1; }",
                _baremetal_credential_functions(),
                "resolve_credentials",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(["bash", str(runner)], capture_output=True, text=True)

    assert proc.returncode == 99, f"expected the credential die(), got {proc.returncode}: {proc.stderr}"
    assert "DEEPSEEK_BASE_URL=https://api.deepseek.com/anthropic" in dotenv.read_text(encoding="utf-8")


_SHIM_REPORTED_VARS = (
    "ANTHROPIC_BASE_URL",
    "_".join(("ANTHROPIC", "API", "KEY")),
    "OPENAI_BASE_URL",
    "_".join(("OPENAI", "API", "KEY")),
    "CLAUDE_MODEL",
    "CODEX_MODEL",
    "GEAK_CLAUDE_MODEL",
)

_SHIM_CASES = {
    "key only": {"_".join(("DEEPSEEK", "API", "KEY")): "sk-ds"},
    "explicit anthropic key": {
        "_".join(("DEEPSEEK", "API", "KEY")): "sk-ds",
        "_".join(("ANTHROPIC", "API", "KEY")): "sk-real",
    },
    "explicit anthropic url": {
        "_".join(("DEEPSEEK", "API", "KEY")): "sk-ds",
        "ANTHROPIC_BASE_URL": "https://gw.example/anthropic",
    },
    "foreign openai side": {
        "_".join(("DEEPSEEK", "API", "KEY")): "sk-ds",
        "OPENAI_BASE_URL": "https://gw.example/v1",
        "_".join(("OPENAI", "API", "KEY")): "sk-gw",
    },
    "explicit models": {
        "_".join(("DEEPSEEK", "API", "KEY")): "sk-ds",
        "CLAUDE_MODEL": "claude-opus-5",
        "CODEX_MODEL": "gpt-5.6-sol",
    },
    "uppercase anthropic segment": {
        "_".join(("DEEPSEEK", "API", "KEY")): "sk-ds",
        "DEEPSEEK_BASE_URL": "https://gw.example/Anthropic",
    },
    "bare deepseek host": {
        "_".join(("DEEPSEEK", "API", "KEY")): "sk-ds",
        "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
    },
    # A subscription token is a configured Anthropic side, so a stale DEEPSEEK_*
    # leftover must be ignored whole rather than pointing the run at DeepSeek's
    # host -- where the migrated API key would also outrank the token and move
    # the run onto API billing.
    "subscription token already configured": {
        "_".join(("DEEPSEEK", "API", "KEY")): "sk-ds",
        "_".join(("CLAUDE", "CODE", "OAUTH", "TOKEN")): "sk-ant-oat01-fake",
    },
}


def _optimizer_shim() -> tuple[str, str]:
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install.sh"
    script_text = install_script.read_text(encoding="utf-8")
    start = script_text.index("normalize_legacy_deepseek_env() {")
    end = script_text.index("\npreflight_validate_credentials() {")
    return script_text[start:end], "normalize_legacy_deepseek_env"


def _baremetal_shim() -> tuple[str, str]:
    # Carries read_dotenv_var along, which this shim consults for each retired
    # key. DOTENV points at a missing file below, so the .env lookups come back
    # empty and only the exported values drive the comparison.
    return _baremetal_credential_functions(), "migrate_legacy_deepseek_env"


_SHIM_INSTALLERS = {
    "install.sh": _optimizer_shim,
    "install_baremetal.sh": _baremetal_shim,
}


@pytest.mark.parametrize("installer", sorted(_SHIM_INSTALLERS))
@pytest.mark.parametrize("case", sorted(_SHIM_CASES))
def test_shell_shim_matches_python_deepseek_compat_env(tmp_path: Path, case: str, installer: str):
    """Every shell shim and ``deepseek_compat_env`` must resolve identically.

    Three copies of this translation exist (Python plus the two installers that
    own an entrypoint); a divergence would send credentials to a different
    endpoint depending on which one the operator happened to use.

    ``ANTHROPIC_AUTH_TOKEN`` is excluded from the compared set on purpose and
    pinned separately below: an installer's job ends at a ``.env``, which only
    ever carries the API-key spelling, while the Python shim resolves in-process
    where offering the bearer spelling as well costs nothing.
    """
    env = _SHIM_CASES[case]
    shim_text, entry_point = _SHIM_INSTALLERS[installer]()

    runner = tmp_path / "shim.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                # The Python side is given an explicit mapping, so the shell has
                # to start from the same blank slate rather than inherit pytest's.
                *_CLEAN_PROVIDER_ENV,
                "warn() { :; }",
                "log() { :; }",
                f'DOTENV="{tmp_path / "absent.env"}"',
                *(f'export {name}="{value}"' for name, value in env.items()),
                shim_text,
                entry_point,
                *(
                    f'printf "%s=%s\\n" {name} "${{{name}-}}"'
                    for name in (*_SHIM_REPORTED_VARS, "ANTHROPIC_AUTH_TOKEN")
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(["bash", str(runner)], capture_output=True, text=True, check=True)
    from_shell = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
    from_python = {**env, **deepseek_compat_env(env)}

    for name in _SHIM_REPORTED_VARS:
        assert from_shell.get(name, "") == from_python.get(name, ""), name
    # The documented exception, asserted rather than assumed: no installer may
    # start writing the bearer spelling without this test being updated.
    assert from_shell.get("ANTHROPIC_AUTH_TOKEN", "") == env.get("ANTHROPIC_AUTH_TOKEN", "")


def test_install_preflights_accept_dual_protocol_gateway(tmp_path: Path):
    script_paths = [
        (
            "install",
            Path(setup.__file__).resolve().parent / "assets" / "install.sh",
            # The gate is what's under test here; the legacy shim has its own
            # coverage and is defined outside the extracted slice.
            ["preflight_load_dotenv() { :; }", "normalize_legacy_deepseek_env() { :; }"],
        ),
        (
            "kernel",
            Path(setup.__file__).resolve().parents[1] / "agents" / "kernel" / "scripts" / "install.sh",
            [],
        ),
    ]
    for name, script_path, stubs in script_paths:
        script_text = script_path.read_text(encoding="utf-8")
        # Start at the cross-provider check so the extracted slice is the whole
        # credential preflight, including the rejection helper it calls.
        start = script_text.index("preflight_reject_cross_provider() {")
        end = script_text.index(
            "\npreflight_validate_credentials", script_text.index("preflight_validate_credentials() {")
        )
        runner = tmp_path / f"{name}-dual-protocol-preflight.sh"
        runner.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    f"REPO_ROOT={tmp_path}",
                    "CHECK_ONLY=0",
                    "DRY_RUN=0",
                    "ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic",
                    "ANTHROPIC_API_KEY=deepseek-test-key",
                    "OPENAI_BASE_URL=https://api.deepseek.com/v1",
                    "OPENAI_API_KEY=deepseek-test-key",
                    "log() { :; }",
                    "warn() { :; }",
                    'die() { echo "$*" >&2; exit 99; }',
                    *stubs,
                    script_text[start:end],
                    "preflight_validate_credentials",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        subprocess.run(["bash", str(runner)], check=True)


def test_install_preflights_reject_cross_provider_pairing(tmp_path: Path):
    """Both installers reject a mispaired config at install time."""
    script_paths = [
        (
            "install",
            Path(setup.__file__).resolve().parent / "assets" / "install.sh",
            ["preflight_load_dotenv() { :; }"],
        ),
        (
            "kernel",
            Path(setup.__file__).resolve().parents[1] / "agents" / "kernel" / "scripts" / "install.sh",
            [],
        ),
    ]
    for name, script_path, stubs in script_paths:
        script_text = script_path.read_text(encoding="utf-8")
        start = script_text.index("preflight_reject_cross_provider() {")
        end = script_text.index(
            "\npreflight_validate_credentials", script_text.index("preflight_validate_credentials() {")
        )
        runner = tmp_path / f"{name}-cross-provider.sh"
        runner.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -uo pipefail",
                    f"REPO_ROOT={tmp_path}",
                    "CHECK_ONLY=0",
                    "DRY_RUN=0",
                    # OpenAI-side gateway URL paired with only an Anthropic key.
                    "OPENAI_BASE_URL=https://gw.example.com/v1",
                    "ANTHROPIC_API_KEY=sk-ant-real",
                    "log() { :; }",
                    "warn() { echo \"$*\" >&2; }",
                    'die() { echo "$*" >&2; exit 99; }',
                    *stubs,
                    script_text[start:end],
                    "preflight_validate_credentials",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        proc = subprocess.run(["bash", str(runner)], capture_output=True, text=True, env=_oauth_only_env())

        assert proc.returncode != 0, f"{name}: mispaired credentials were accepted"
        assert "Conflicting LLM credentials" in proc.stderr, proc.stderr


def test_baremetal_setup_rejects_cross_provider_pairing(tmp_path: Path):
    """install_baremetal.sh rejects a mispaired config during setup, like the CLI
    preflight and the other two installers."""
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    script_text = install_script.read_text(encoding="utf-8")
    start = script_text.index("read_dotenv_var() {")
    end = script_text.index("\nwrite_runtime_dotenv() {")
    credential_functions = script_text[start:end]
    dotenv = tmp_path / ".env"
    dotenv.write_text("HYPERLOOM_RUN_MODE=baremetal\n", encoding="utf-8")
    runner = tmp_path / "cross-provider-run.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -uo pipefail",
                f"DOTENV={dotenv}",
                "CHECK_ONLY=0",
                "DRY_RUN=0",
                "OPENAI_BASE_URL_ARG=",
                "log() { :; }",
                "warn() { echo \"$*\" >&2; }",
                'die() { echo "$*" >&2; exit 99; }',
                "is_interactive() { return 1; }",
                credential_functions,
                # OpenAI-side gateway URL paired with only an Anthropic key.
                "OPENAI_BASE_URL=https://gw.example.com/v1",
                "ANTHROPIC_API_KEY=sk-ant-real",
                "resolve_credentials",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(["bash", str(runner)], capture_output=True, text=True, env=_oauth_only_env())

    assert proc.returncode != 0, "mispaired credentials were accepted"
    assert "Conflicting LLM credentials" in proc.stderr, proc.stderr


def _oauth_only_env() -> dict[str, str]:
    """Ambient provider credentials would mask an oauth-only preflight."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "OPENAI_", "DEEPSEEK_"))}
    env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    return env


def test_install_preflights_accept_oauth_only_credentials(tmp_path: Path):
    """A Claude subscription token alone is a self-consistent Anthropic side."""
    script_paths = [
        (
            "install",
            Path(setup.__file__).resolve().parent / "assets" / "install.sh",
            ["preflight_load_dotenv() { :; }"],
        ),
        (
            "kernel",
            Path(setup.__file__).resolve().parents[1] / "agents" / "kernel" / "scripts" / "install.sh",
            [],
        ),
    ]
    for name, script_path, stubs in script_paths:
        script_text = script_path.read_text(encoding="utf-8")
        start = script_text.index("preflight_reject_cross_provider() {")
        end = script_text.index(
            "\npreflight_validate_credentials", script_text.index("preflight_validate_credentials() {")
        )
        runner = tmp_path / f"{name}-oauth-preflight.sh"
        runner.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -uo pipefail",
                    f"REPO_ROOT={tmp_path}",
                    "CHECK_ONLY=0",
                    "DRY_RUN=0",
                    "CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-test",
                    "log() { :; }",
                    'warn() { echo "$*" >&2; }',
                    'die() { echo "$*" >&2; exit 99; }',
                    *stubs,
                    script_text[start:end],
                    "preflight_validate_credentials",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        proc = subprocess.run(["bash", str(runner)], capture_output=True, text=True, env=_oauth_only_env())

        assert proc.returncode == 0, f"{name}: oauth-only rejected: {proc.stderr}"


def test_baremetal_setup_accepts_oauth_only_without_mirroring_it(tmp_path: Path):
    """Subscription token resolves on its own and stays out of API-key slots."""
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    script_text = install_script.read_text(encoding="utf-8")
    start = script_text.index("read_dotenv_var() {")
    end = script_text.index("\nwrite_runtime_dotenv() {")
    credential_functions = script_text[start:end]
    dotenv = tmp_path / ".env"
    dotenv.write_text("HYPERLOOM_RUN_MODE=baremetal\n", encoding="utf-8")
    dump = tmp_path / "resolved.txt"
    runner = tmp_path / "oauth-only-run.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -uo pipefail",
                f"DOTENV={dotenv}",
                "CHECK_ONLY=0",
                "DRY_RUN=0",
                "log() { :; }",
                'warn() { echo "$*" >&2; }',
                'die() { echo "$*" >&2; exit 99; }',
                "is_interactive() { return 1; }",
                credential_functions,
                "CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-test",
                "resolve_credentials",
                f'printf "%s|%s|%s\\n" "${{CLAUDE_CODE_OAUTH_TOKEN:-}}" "${{ANTHROPIC_API_KEY:-}}"'
                f' "${{ANTHROPIC_AUTH_TOKEN:-}}" > {dump}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(["bash", str(runner)], capture_output=True, text=True, env=_oauth_only_env())

    assert proc.returncode == 0, f"oauth-only rejected: {proc.stderr}"
    assert dump.read_text(encoding="utf-8").strip() == "sk-ant-oat01-test||"
    persisted = dotenv.read_text(encoding="utf-8")
    assert "CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-test" in persisted
    assert "ANTHROPIC_API_KEY=" not in persisted
    assert "ANTHROPIC_AUTH_TOKEN=" not in persisted


def test_install_preflights_accept_oauth_alongside_bare_openai_key(tmp_path: Path):
    """Mirrors the CLI: both keys imply their own official endpoint, so neither
    borrows the other's and the pair is legal."""
    script_paths = [
        (
            "install",
            Path(setup.__file__).resolve().parent / "assets" / "install.sh",
            ["preflight_load_dotenv() { :; }"],
        ),
        (
            "kernel",
            Path(setup.__file__).resolve().parents[1] / "agents" / "kernel" / "scripts" / "install.sh",
            [],
        ),
    ]
    for name, script_path, stubs in script_paths:
        script_text = script_path.read_text(encoding="utf-8")
        start = script_text.index("preflight_reject_cross_provider() {")
        end = script_text.index(
            "\npreflight_validate_credentials", script_text.index("preflight_validate_credentials() {")
        )
        runner = tmp_path / f"{name}-oauth-openai-pair.sh"
        runner.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -uo pipefail",
                    f"REPO_ROOT={tmp_path}",
                    "CHECK_ONLY=0",
                    "DRY_RUN=0",
                    "CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-test",
                    "OPENAI_API_KEY=sk-openai-official",
                    "log() { :; }",
                    'warn() { echo "$*" >&2; }',
                    'die() { echo "$*" >&2; exit 99; }',
                    *stubs,
                    script_text[start:end],
                    "preflight_validate_credentials",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        proc = subprocess.run(["bash", str(runner)], capture_output=True, text=True, env=_oauth_only_env())

        assert proc.returncode == 0, f"{name}: oauth + official OpenAI key rejected: {proc.stderr}"


def test_install_preflights_still_reject_gateway_url_with_bare_openai_key(tmp_path: Path):
    """An explicit ANTHROPIC_BASE_URL keeps flagging an OpenAI key that lost its
    own base URL."""
    script_paths = [
        (
            "install",
            Path(setup.__file__).resolve().parent / "assets" / "install.sh",
            ["preflight_load_dotenv() { :; }"],
        ),
        (
            "kernel",
            Path(setup.__file__).resolve().parents[1] / "agents" / "kernel" / "scripts" / "install.sh",
            [],
        ),
    ]
    for name, script_path, stubs in script_paths:
        script_text = script_path.read_text(encoding="utf-8")
        start = script_text.index("preflight_reject_cross_provider() {")
        end = script_text.index(
            "\npreflight_validate_credentials", script_text.index("preflight_validate_credentials() {")
        )
        runner = tmp_path / f"{name}-gateway-bare-openai.sh"
        runner.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -uo pipefail",
                    f"REPO_ROOT={tmp_path}",
                    "CHECK_ONLY=0",
                    "DRY_RUN=0",
                    "ANTHROPIC_BASE_URL=https://gw.example.com/anthropic",
                    "ANTHROPIC_API_KEY=gw-key",
                    "OPENAI_API_KEY=gw-key",
                    "log() { :; }",
                    'warn() { echo "$*" >&2; }',
                    'die() { echo "$*" >&2; exit 99; }',
                    *stubs,
                    script_text[start:end],
                    "preflight_validate_credentials",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        proc = subprocess.run(["bash", str(runner)], capture_output=True, text=True, env=_oauth_only_env())

        assert proc.returncode != 0, f"{name}: gateway URL with bare OpenAI key was accepted"
        assert "Conflicting LLM credentials" in proc.stderr, proc.stderr


def test_baremetal_setup_accepts_oauth_alongside_bare_openai_key(tmp_path: Path):
    """install_baremetal.sh mirrors the same relaxed pairing rule."""
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    script_text = install_script.read_text(encoding="utf-8")
    start = script_text.index("read_dotenv_var() {")
    end = script_text.index("\nwrite_runtime_dotenv() {")
    credential_functions = script_text[start:end]
    dotenv = tmp_path / ".env"
    dotenv.write_text("HYPERLOOM_RUN_MODE=baremetal\n", encoding="utf-8")
    runner = tmp_path / "oauth-openai-pair-run.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -uo pipefail",
                f"DOTENV={dotenv}",
                "CHECK_ONLY=0",
                "DRY_RUN=0",
                "OPENAI_BASE_URL_ARG=",
                "log() { :; }",
                'warn() { echo "$*" >&2; }',
                'die() { echo "$*" >&2; exit 99; }',
                "is_interactive() { return 1; }",
                credential_functions,
                "CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-test",
                "OPENAI_API_KEY=sk-openai-official",
                "resolve_credentials",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(["bash", str(runner)], capture_output=True, text=True, env=_oauth_only_env())

    assert proc.returncode == 0, f"oauth + official OpenAI key rejected: {proc.stderr}"


def test_baremetal_install_no_longer_accepts_openai_safe_credential_flags():
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    script_text = install_script.read_text(encoding="utf-8")
    credential_start = script_text.index("resolve_credentials() {")
    credential_end = script_text.index("\nwrite_runtime_dotenv() {")
    credential_text = script_text[credential_start:credential_end]

    assert "--safe-api-key" not in script_text
    assert "--openai-base-url" not in script_text
    assert "SAFE_API_KEY_ARG" not in script_text
    assert "OPENAI_BASE_URL_ARG" not in script_text
    assert "safe_key" not in credential_text
    assert "openai_key" not in credential_text
    assert "openai_url" not in credential_text
    assert "dv_safe" not in credential_text
    assert "dv_openai" not in credential_text


def test_baremetal_runtime_deps_probe_vllm_venv_for_isolated_vllm(tmp_path: Path):
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    script_text = install_script.read_text(encoding="utf-8")
    loop_start = script_text.index("  local m", script_text.index('  if [ "$any_fw" -eq 0 ]; then'))
    loop_end = script_text.index('\n\n  [ "$rc" -ne 0 ]', loop_start)
    runtime_dep_loop = script_text[loop_start:loop_end]

    host_py = tmp_path / "host-python"
    vllm_root = tmp_path / "vllm-venv"
    vllm_py = vllm_root / "bin" / "python"
    host_py.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    vllm_py.parent.mkdir(parents=True)
    vllm_py.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    host_py.chmod(0o755)
    vllm_py.chmod(0o755)

    runner = tmp_path / "runtime-deps.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"py={host_py}",
                f"VLLM_VENV_ROOT={vllm_root}",
                "FRAMEWORK_ENV=isolated",
                "FRAMEWORKS=sglang,vllm",
                # sgl_kernel is only probed once SGLang itself imported.
                "sglang_ok=1",
                "INSTALL_FRAMEWORK=none",
                'log() { :; }',
                'warn() { :; }',
                '_py_has() { printf "probe %s %s\\n" "$1" "$2"; return 0; }',
                "run_probe() {",
                runtime_dep_loop,
                "}",
                "run_probe",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = subprocess.run(
        ["bash", str(runner)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()

    assert out == [
        f"probe {vllm_py} triton",
        f"probe {vllm_py} aiter",
        f"probe {host_py} sgl_kernel",
    ]


def _runtime_dep_loop(install_script: Path) -> str:
    script_text = install_script.read_text(encoding="utf-8")
    loop_start = script_text.index("  local m", script_text.index('  if [ "$any_fw" -eq 0 ]; then'))
    loop_end = script_text.index('\n\n  [ "$rc" -ne 0 ]', loop_start)
    return script_text[loop_start:loop_end]


def test_baremetal_runtime_deps_skip_sgl_kernel_without_sglang(tmp_path: Path):
    """An atom-only host must not be told sgl_kernel is missing.

    atom images ship neither SGLang nor its sgl_kernel companion, so probing for
    it there only produces a scary 'missing' line about a dependency the run
    will never use.
    """
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"

    host_py = tmp_path / "host-python"
    host_py.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    host_py.chmod(0o755)

    runner = tmp_path / "runtime-deps-atom.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"py={host_py}",
                f"VLLM_VENV_ROOT={tmp_path}/absent",
                "FRAMEWORK_ENV=shared",
                "FRAMEWORKS=sglang,vllm,atom",
                "sglang_ok=0",
                "INSTALL_FRAMEWORK=none",
                "log() { :; }",
                "warn() { :; }",
                '_py_has() { printf "probe %s\\n" "$2"; return 0; }',
                "run_probe() {",
                _runtime_dep_loop(install_script),
                "}",
                "run_probe",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = subprocess.run(
        ["bash", str(runner)], check=True, text=True, stdout=subprocess.PIPE
    ).stdout.splitlines()

    assert out == ["probe triton", "probe aiter"]


def test_baremetal_preflight_probes_atom_by_default():
    """Phase 1 must accept an atom-only host without extra flags.

    The default probe list gated on sglang/vllm alone, so setup inside
    rocm/atom:latest died with 'no serving framework importable' even though
    atom was installed and is a registered framework.
    """
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    text = install_script.read_text(encoding="utf-8")

    m = re.search(r'^FRAMEWORKS="\$\{FRAMEWORKS:-([^}]+)\}"', text, re.MULTILINE)
    assert m, "install_baremetal.sh must define an overridable FRAMEWORKS default"
    assert "atom" in m.group(1).split(","), (
        "atom must be in the default Phase 1 probe list; otherwise setup inside "
        "rocm/atom:latest fails preflight"
    )

    assert "rocm/atom:latest" in text, (
        "the missing-framework hint must point atom users at their image"
    )


def test_baremetal_triton_pin_is_advisory_when_installing_nothing():
    """A prebuilt image may carry its own triton.

    The torch/triton pin exists to protect a framework build. When
    --install-framework is none there is nothing to build, so a drifting triton
    must warn rather than fail the whole preflight -- rocm/atom:latest ships a
    triton newer than torch's pin.
    """
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    text = install_script.read_text(encoding="utf-8")

    guard = re.search(
        r'if \[ "\$INSTALL_FRAMEWORK" = "none" \]; then\s*\n'
        r'\s*check_torch_triton_alignment "\$py" \|\| true\s*\n'
        r"\s*else\s*\n"
        r'\s*check_torch_triton_alignment "\$py" \|\| rc=1',
        text,
    )
    assert guard, (
        "the triton alignment check must be advisory when --install-framework "
        "is none and fatal only when a framework layer will be built"
    )


def test_baremetal_aiter_install_preserves_system_triton_and_rechecks_alignment(tmp_path: Path):
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    script_text = install_script.read_text(encoding="utf-8")
    start = script_text.index("install_aiter_ref_with_constraints() {")
    end = script_text.index("\n}\n\ninstall_compatible_aiter() {", start) + 3
    install_aiter = script_text[start:end]

    fake_py = tmp_path / "python"
    calls_file = tmp_path / "calls.txt"
    fake_py.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "env=%s args=%s\\n" "${AITER_USE_SYSTEM_TRITON-__unset__}" "$*" >> "$CALLS_FILE"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_py.chmod(0o755)

    runner = tmp_path / "aiter-install.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"export CALLS_FILE={calls_file}",
                "checkout_aiter_ref() { printf 'checkout %s %s\\n' \"$1\" \"$2\" >> \"$CALLS_FILE\"; }",
                "check_torch_triton_alignment() { printf 'align %s\\n' \"$1\" >> \"$CALLS_FILE\"; }",
                install_aiter,
                f"install_aiter_ref_with_constraints {fake_py} {tmp_path / 'aiter'} v0.1.18 {tmp_path / 'constraints.txt'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(["bash", str(runner)], check=True)

    assert calls_file.read_text(encoding="utf-8").splitlines() == [
        f"checkout {tmp_path / 'aiter'} v0.1.18",
        f"env=1 args=-m pip install --constraint {tmp_path / 'constraints.txt'} --config-settings editable_mode=compat -e {tmp_path / 'aiter'}",
        "env=__unset__ args=-c import aiter",
        f"align {fake_py}",
    ]


def test_baremetal_aiter_install_fails_when_import_fails(tmp_path: Path):
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    script_text = install_script.read_text(encoding="utf-8")
    start = script_text.index("install_aiter_ref_with_constraints() {")
    end = script_text.index("\n}\n\ninstall_compatible_aiter() {", start) + 3
    install_aiter = script_text[start:end]

    fake_py = tmp_path / "python"
    calls_file = tmp_path / "calls.txt"
    fake_py.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "args=%s\\n" "$*" >> "$CALLS_FILE"',
                '[ "$*" = "-c import aiter" ] && exit 42',
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_py.chmod(0o755)

    runner = tmp_path / "aiter-import-fail.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"export CALLS_FILE={calls_file}",
                "checkout_aiter_ref() { printf 'checkout %s %s\\n' \"$1\" \"$2\" >> \"$CALLS_FILE\"; }",
                "check_torch_triton_alignment() { printf 'align %s\\n' \"$1\" >> \"$CALLS_FILE\"; return 0; }",
                install_aiter,
                f"if install_aiter_ref_with_constraints {fake_py} {tmp_path / 'aiter'} v0.1.18 {tmp_path / 'constraints.txt'}; then",
                "  echo RESULT=success",
                "else",
                "  echo RESULT=failure",
                "fi",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = subprocess.run(["bash", str(runner)], check=True, text=True, stdout=subprocess.PIPE).stdout

    assert "RESULT=failure" in out
    assert calls_file.read_text(encoding="utf-8").splitlines() == [
        f"checkout {tmp_path / 'aiter'} v0.1.18",
        f"args=-m pip install --constraint {tmp_path / 'constraints.txt'} --config-settings editable_mode=compat -e {tmp_path / 'aiter'}",
        "args=-c import aiter",
    ]


def test_baremetal_sglang_installs_aiter_when_find_spec_succeeds_but_import_fails(tmp_path: Path):
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    script_text = install_script.read_text(encoding="utf-8")
    start = script_text.index("install_sglang_framework() {")
    end = script_text.index("\n}\n\n# Verify that the installed vLLM package resolves", start) + 3
    install_sglang_framework = script_text[start:end]

    fake_py = tmp_path / "python"
    calls_file = tmp_path / "calls.txt"
    import_flag = tmp_path / "aiter-ok"
    fake_py.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "py args=%s\\n" "$*" >> "$CALLS_FILE"',
                'if [ "$*" = "-c import aiter" ]; then',
                f"  [ -f {import_flag} ] && exit 0 || exit 42",
                "fi",
                'if [ "$*" = "-c import sglang" ] || [ "$*" = "-c import sgl_kernel" ]; then exit 0; fi',
                "if [ \"$1\" = \"-\" ]; then echo 3.12; exit 0; fi",
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fake_py.chmod(0o755)

    runner = tmp_path / "sglang-aiter-reinstall.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"export CALLS_FILE={calls_file}",
                "CHECK_ONLY=0",
                "DRY_RUN=0",
                "AITER_REF=",
                "SGLANG_ROCM_EXTRA=rocm720",
                "SGLANG_ROCM_PYPI_VERSION=7.2.0",
                "SGLANG_REPO=https://example.invalid/sglang.git",
                "SGLANG_REF=main",
                "AITER_REPO=https://example.invalid/aiter.git",
                f"AITER_ROOT={tmp_path / 'aiter'}",
                f"resolve_python() {{ printf '%s\\n' {fake_py}; }}",
                f"framework_deps_root() {{ printf '%s\\n' {tmp_path}; }}",
                "log() { :; }",
                "warn() { :; }",
                'die() { echo "$*" >&2; exit 99; }',
                "_py_has() { [ \"$2\" = aiter ] && return 0; return 0; }",
                "install_sglang_from_wheel() { :; }",
                "install_sglang_from_source() { :; }",
                f"install_compatible_aiter() {{ printf 'install_compatible_aiter %s %s\\n' \"$1\" \"$2\" >> \"$CALLS_FILE\"; touch {import_flag}; }}",
                install_sglang_framework,
                "install_sglang_framework",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(["bash", str(runner)], check=True)

    assert f"install_compatible_aiter {fake_py} {tmp_path / 'aiter'}" in calls_file.read_text(encoding="utf-8")


def test_kernel_install_no_longer_exports_openai_safe_credentials():
    install_script = Path(setup.__file__).resolve().parents[1] / "agents" / "kernel" / "scripts" / "install.sh"
    script_text = install_script.read_text(encoding="utf-8")
    write_start = script_text.index("write_env_file() {")
    write_end = script_text.index("\nensure_geak()", write_start)
    write_text = script_text[write_start:write_end]

    assert "_OPENAI_BASE_URL_VAL" not in script_text
    assert "_OPENAI_KEY_VAL" not in script_text
    assert "_snap_safe" not in script_text
    assert "_snap_openai" not in script_text
    # The kernel-agent drives Claude Code, so kernel-agent.env.sh stays
    # Anthropic-only regardless of what the gateway serves.
    assert "export OPENAI_BASE_URL" not in write_text
    assert "export OPENAI_API_KEY" not in write_text

    # The gateway credentials may be *read* in memory -- the single-gateway
    # branch derives ANTHROPIC_BASE_URL/ANTHROPIC_API_KEY from
    # OPENAI_BASE_URL + OPENAI_API_KEY, mirroring the CLI's
    # _resolve_llm_endpoints(). What must never happen is persisting them:
    # neither exported into kernel-agent.env.sh nor written back to .env.
    assert "export OPENAI_API_KEY" not in write_text
    assert "upsert_dotenv_var OPENAI_API_KEY" not in write_text
    # The legacy gateway key is never read, exported or persisted -- but a stale
    # value left in a migrating .env is still scrubbed.
    assert "export SAFE_API_KEY" not in script_text
    assert "upsert_dotenv_var SAFE_API_KEY" not in script_text
    assert "remove_dotenv_var SAFE_API_KEY" in write_text
    # ... and it does not touch the OpenAI side, which it never resolves.
    assert "remove_dotenv_var OPENAI_BASE_URL" not in write_text
    assert "remove_dotenv_var OPENAI_API_KEY" not in write_text
    # Retired provider variables are scrubbed on every re-install.
    assert "remove_dotenv_var DEEPSEEK_API_KEY" in write_text
    assert "remove_dotenv_var DEEPSEEK_BASE_URL" in write_text


def test_packaged_install_sh_resolves_target_workspace_root(tmp_path: Path):
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install.sh"
    script_text = install_script.read_text(encoding="utf-8")
    start = script_text.index("resolve_repo_root() {")
    end = script_text.index('\nREPO_ROOT="$(resolve_repo_root)"', start)
    resolve_repo_root = script_text[start:end]

    source_root = tmp_path / "source"
    source_assets = source_root / "src" / "hyperloom" / "inference_optimizer" / "assets"
    source_assets.mkdir(parents=True)
    (source_root / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")

    target_root = tmp_path / "target"
    packaged_assets = target_root / "hyperloom" / "inference_optimizer" / "assets"
    packaged_assets.mkdir(parents=True)

    runner = tmp_path / "resolve-root.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                resolve_repo_root,
                f"_script_dir={source_assets}",
                "unset REPO_ROOT",
                "printf 'source=%s\n' \"$(resolve_repo_root)\"",
                f"_script_dir={packaged_assets}",
                "unset REPO_ROOT",
                "printf 'packaged=%s\n' \"$(resolve_repo_root)\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = subprocess.run(
        ["bash", str(runner)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout

    assert f"source={source_root}\n" in out
    assert f"packaged={target_root}\n" in out


def test_install_sh_scrubs_stale_runtime_env_for_setup_dotenv(tmp_path: Path):
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install.sh"
    script_text = install_script.read_text(encoding="utf-8")
    start = script_text.index("setup_dotenv_is_authoritative() {")
    end = script_text.index("\nload_dotenv_no_clobber() {", start)
    helpers = script_text[start:end]
    load_start = script_text.index("load_dotenv_no_clobber() {")
    load_end = script_text.index("\n# Load .env before deriving", load_start)
    loader = script_text[load_start:load_end]

    workspace = tmp_path / "target"
    workspace.mkdir()
    (workspace / ".env").write_text(
        "\n".join(
            [
                "HYPERLOOM_RUN_MODE=baremetal",
                f"USER_DATA_PATH={workspace}/session",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    runner = tmp_path / "install-scrub.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"REPO_ROOT={workspace}",
                helpers,
                loader,
                "USER_DATA_PATH=/old/workspace/session",
                "HYPERLOOM_RUNTIME_DIR=/old/workspace/session/runtime",
                "KERNEL_AGENT_ENV=/old/workspace/session/runtime/kernel-agent.env.sh",
                "HYPERLOOM_ROOT=/old/workspace/session/runtime/source-mirrors",
                "KERNEL_AGENT_ROOT=/old/workspace/hyperloom/agents/kernel",
                "HYPERLOOM_KERNEL_AGENT_ROOT=/old/workspace/hyperloom/agents/kernel",
                "FRAMEWORK_AGENT_ROOT=/old/workspace/hyperloom/agents/framework",
                "HYPERLOOM_SKILL_PATH=/old/workspace/hyperloom/inference_optimizer/SKILL.md",
                "PYTHONPATH=/old/workspace",
                "scrub_stale_workspace_env_for_setup_dotenv",
                "load_dotenv_no_clobber",
                'HYPERLOOM_RUNTIME_DIR="${HYPERLOOM_RUNTIME_DIR:-${USER_DATA_PATH}/runtime}"',
                'KERNEL_AGENT_ENV="${KERNEL_AGENT_ENV:-${HYPERLOOM_RUNTIME_DIR}/kernel-agent.env.sh}"',
                "printf 'USER_DATA_PATH=%s\n' \"${USER_DATA_PATH-}\"",
                "printf 'HYPERLOOM_RUNTIME_DIR=%s\n' \"${HYPERLOOM_RUNTIME_DIR-}\"",
                "printf 'KERNEL_AGENT_ENV=%s\n' \"${KERNEL_AGENT_ENV-}\"",
                "printf 'KERNEL_AGENT_ROOT=%s\n' \"${KERNEL_AGENT_ROOT-}\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = subprocess.run(
        ["bash", str(runner)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout

    assert f"USER_DATA_PATH={workspace}/session\n" in out
    assert f"HYPERLOOM_RUNTIME_DIR={workspace}/session/runtime\n" in out
    assert f"KERNEL_AGENT_ENV={workspace}/session/runtime/kernel-agent.env.sh\n" in out
    assert "KERNEL_AGENT_ROOT=\n" in out
    assert "/old/workspace" not in out


def test_kernel_env_authoritative_anthropic_mode_does_not_emit_openai_aliases(tmp_path: Path):
    install_script = Path(setup.__file__).resolve().parents[1] / "agents" / "kernel" / "scripts" / "install.sh"
    script_text = install_script.read_text(encoding="utf-8")
    upsert_start = script_text.index("upsert_dotenv_var() {")
    upsert_end = script_text.index("\n# In --check-only mode")
    compose_start = script_text.index("_compose_pythonpath() {")
    compose_end = script_text.index("\n# Keep REPO_ROOT on PYTHONPATH")
    env_start = script_text.index("write_env_file() {")
    env_end = script_text.index("\nensure_geak() {")
    dotenv_helpers = script_text[upsert_start:upsert_end]
    compose_pythonpath = script_text[compose_start:compose_end]
    write_env_file = script_text[env_start:env_end]
    dotenv = tmp_path / ".env"
    kernel_env = tmp_path / "runtime" / "kernel-agent.env.sh"
    dotenv.write_text(
        "\n".join(
            [
                "ANTHROPIC_API_KEY=<PLEASE_FILL_IN>",
                "ANTHROPIC_BASE_URL=https://api.anthropic.com",
                "HYPERLOOM_RUN_MODE=baremetal",
                "OPENAI_BASE_URL=https://api.anthropic.com",
                "OPENAI_API_KEY=stale-openai-key",
                "LLM_GATEWAY_KEY=stale-gateway-key",
                "SAFE_API_KEY=stale-safe-key",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runner = tmp_path / "kernel-run.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"DOTENV={dotenv}",
                f"KERNEL_AGENT_ENV={kernel_env}",
                f"USER_DATA_PATH={tmp_path / 'session'}",
                f"HYPERLOOM_RUNTIME_DIR={tmp_path / 'runtime'}",
                "CHECK_ONLY=0",
                "DRY_RUN=0",
                "HYPERLOOM_SETUP_ENV_AUTHORITATIVE=1",
                "HYPERLOOM_SETUP_LLM_MODE=anthropic",
                "_ANTHROPIC_BASE_URL_VAL=https://api.anthropic.com",
                "_ANTHROPIC_KEY_VAL='<PLEASE_FILL_IN>'",
                "_OPENAI_BASE_URL_VAL=",
                "_OPENAI_KEY_VAL=",
                "LLM_GATEWAY_KEY=",
                "LLM_API_KEY=",
                "HYPERLOOM_KERNEL_AGENT_ROOT=",
                "KERNEL_AGENT_ROOT=",
                "MAGPIE_PATH=",
                "MAGPIE_PYTHON=",
                "PYTHONPATH=",
                "INFERENCEX_PATH=",
                "TRACELENS_ROOT=",
                "TRACELENS_INTERNAL_ROOT=",
                "HYPERLOOM_ROOT=",
                "GEAK_E2E_RUNNER=",
                "GEAK_ROOT=",
                "GEAK_CLAUDE_MODEL_VAL=",
                "GEAK_RUN_MODE_VAL=",
                "GEAK_SCORE_TARGET=",
                "GEAK_SKIP_PROFILE=",
                "GEAK_MAX_BENCHMARK_SHAPES=",
                "log() { :; }",
                "warn() { :; }",
                dotenv_helpers,
                compose_pythonpath,
                write_env_file,
                "write_env_file",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(["bash", str(runner)], check=True)

    kernel_text = kernel_env.read_text(encoding="utf-8")
    dotenv_text = dotenv.read_text(encoding="utf-8")
    assert "export ANTHROPIC_BASE_URL='https://api.anthropic.com'" in kernel_text
    assert "export OPENAI_BASE_URL=" not in kernel_text
    assert "export OPENAI_API_KEY=" not in kernel_text
    # The OpenAI side is not resolved by this installer, so its .env entries stay.
    assert "OPENAI_BASE_URL=https://api.anthropic.com" in dotenv_text
    assert "OPENAI_API_KEY=stale-openai-key" in dotenv_text
    # Gateway aliases it does own are still purged.
    assert "LLM_GATEWAY_KEY=" not in dotenv_text
    assert "SAFE_API_KEY=" not in dotenv_text


def test_kernel_env_keeps_anthropic_creds_in_dotenv(tmp_path: Path):
    """Writing kernel-agent env must NOT wipe the Anthropic creds the operator
    put in .env (an Anthropic-only setup must keep ANTHROPIC_API_KEY /
    ANTHROPIC_BASE_URL after install)."""
    install_script = Path(setup.__file__).resolve().parents[1] / "agents" / "kernel" / "scripts" / "install.sh"
    script_text = install_script.read_text(encoding="utf-8")
    upsert_start = script_text.index("upsert_dotenv_var() {")
    upsert_end = script_text.index("\n# In --check-only mode")
    compose_start = script_text.index("_compose_pythonpath() {")
    compose_end = script_text.index("\n# Keep REPO_ROOT on PYTHONPATH")
    env_start = script_text.index("write_env_file() {")
    env_end = script_text.index("\nensure_geak() {")
    dotenv_helpers = script_text[upsert_start:upsert_end]
    compose_pythonpath = script_text[compose_start:compose_end]
    write_env_file = script_text[env_start:env_end]
    dotenv = tmp_path / ".env"
    kernel_env = tmp_path / "runtime" / "kernel-agent.env.sh"
    dotenv.write_text(
        "\n".join(
            [
                "ANTHROPIC_API_KEY=anthropic-real-key",
                "ANTHROPIC_BASE_URL=https://api.anthropic.com",
                "HYPERLOOM_RUN_MODE=baremetal",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runner = tmp_path / "kernel-run.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"DOTENV={dotenv}",
                f"KERNEL_AGENT_ENV={kernel_env}",
                f"USER_DATA_PATH={tmp_path / 'session'}",
                f"HYPERLOOM_RUNTIME_DIR={tmp_path / 'runtime'}",
                "CHECK_ONLY=0",
                "DRY_RUN=0",
                "_ANTHROPIC_BASE_URL_VAL=https://api.anthropic.com",
                "_ANTHROPIC_KEY_VAL=anthropic-real-key",
                "_OPENAI_BASE_URL_VAL=",
                "_OPENAI_KEY_VAL=",
                "LLM_GATEWAY_KEY=",
                "LLM_API_KEY=",
                "HYPERLOOM_KERNEL_AGENT_ROOT=",
                "KERNEL_AGENT_ROOT=",
                "MAGPIE_PATH=",
                "MAGPIE_PYTHON=",
                "PYTHONPATH=",
                "INFERENCEX_PATH=",
                "TRACELENS_ROOT=",
                "TRACELENS_INTERNAL_ROOT=",
                "HYPERLOOM_ROOT=",
                "GEAK_E2E_RUNNER=",
                "GEAK_ROOT=",
                "GEAK_CLAUDE_MODEL_VAL=",
                "GEAK_RUN_MODE_VAL=",
                "GEAK_SCORE_TARGET=",
                "GEAK_SKIP_PROFILE=",
                "GEAK_MAX_BENCHMARK_SHAPES=",
                "log() { :; }",
                "warn() { :; }",
                dotenv_helpers,
                compose_pythonpath,
                write_env_file,
                "write_env_file",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(["bash", str(runner)], check=True)

    kernel_text = kernel_env.read_text(encoding="utf-8")
    dotenv_text = dotenv.read_text(encoding="utf-8")
    # .env must still carry the Anthropic creds after install.
    assert "ANTHROPIC_API_KEY=anthropic-real-key" in dotenv_text
    assert "ANTHROPIC_BASE_URL=https://api.anthropic.com" in dotenv_text
    # kernel-agent env mirrors the same Anthropic values and no OpenAI leak.
    assert "export ANTHROPIC_API_KEY='anthropic-real-key'" in kernel_text
    assert "export ANTHROPIC_BASE_URL='https://api.anthropic.com'" in kernel_text
    assert "export OPENAI_API_KEY=" not in kernel_text
    assert "OPENAI_API_KEY=" not in dotenv_text


def test_kernel_env_persists_geak_claude_model_to_dotenv(tmp_path: Path):
    """Fresh-shell CLI starts from .env, so GEAK_CLAUDE_MODEL must be persisted
    there in addition to kernel-agent.env.sh."""

    def bash_path(path: Path) -> str:
        text = str(path)
        if path.drive:
            rest = text[len(path.drive) :].replace("\\", "/")
            return f"/mnt/{path.drive[0].lower()}{rest}"
        return text

    install_script = Path(setup.__file__).resolve().parents[1] / "agents" / "kernel" / "scripts" / "install.sh"
    script_text = install_script.read_text(encoding="utf-8")
    upsert_start = script_text.index("upsert_dotenv_var() {")
    upsert_end = script_text.index("\n# In --check-only mode")
    compose_start = script_text.index("_compose_pythonpath() {")
    compose_end = script_text.index("\n# Keep REPO_ROOT on PYTHONPATH")
    env_start = script_text.index("write_env_file() {")
    env_end = script_text.index("\nensure_geak() {")
    dotenv_helpers = script_text[upsert_start:upsert_end]
    compose_pythonpath = script_text[compose_start:compose_end]
    write_env_file = script_text[env_start:env_end]
    dotenv = tmp_path / ".env"
    kernel_env = tmp_path / "runtime" / "kernel-agent.env.sh"
    dotenv.write_text(
        f"HYPERLOOM_KERNEL_AGENT_ROOT={tmp_path / 'kernel-agent'}\n",
        encoding="utf-8",
    )
    runner = tmp_path / "kernel-run.sh"
    runner_text = (
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"DOTENV={bash_path(dotenv)}",
                f"KERNEL_AGENT_ENV={bash_path(kernel_env)}",
                f"USER_DATA_PATH={bash_path(tmp_path / 'session')}",
                f"HYPERLOOM_RUNTIME_DIR={bash_path(tmp_path / 'runtime')}",
                "CHECK_ONLY=0",
                "DRY_RUN=0",
                "_ANTHROPIC_BASE_URL_VAL=",
                "_ANTHROPIC_KEY_VAL=",
                "_OPENAI_BASE_URL_VAL=",
                "_OPENAI_KEY_VAL=",
                "LLM_GATEWAY_KEY=",
                "LLM_API_KEY=",
                "HYPERLOOM_KERNEL_AGENT_ROOT=",
                "KERNEL_AGENT_ROOT=",
                "MAGPIE_PATH=",
                "MAGPIE_PYTHON=",
                "PYTHONPATH=",
                "INFERENCEX_PATH=",
                "TRACELENS_ROOT=",
                "TRACELENS_INTERNAL_ROOT=",
                "HYPERLOOM_ROOT=",
                "GEAK_E2E_RUNNER=",
                "GEAK_ROOT=",
                "GEAK_CLAUDE_MODEL_VAL=claude-opus-4-8",
                "GEAK_RUN_MODE_VAL=",
                "GEAK_SCORE_TARGET=",
                "GEAK_SKIP_PROFILE=",
                "GEAK_MAX_BENCHMARK_SHAPES=",
                "log() { :; }",
                "warn() { :; }",
                dotenv_helpers,
                compose_pythonpath,
                write_env_file,
                "write_env_file",
            ]
        )
        + "\n"
    )
    with runner.open("w", encoding="utf-8", newline="\n") as f:
        f.write(runner_text)

    subprocess.run(["bash", bash_path(runner)], check=True)

    kernel_text = kernel_env.read_text(encoding="utf-8")
    dotenv_text = dotenv.read_text(encoding="utf-8")
    assert "export GEAK_CLAUDE_MODEL='claude-opus-4-8'" in kernel_text
    assert "GEAK_CLAUDE_MODEL=claude-opus-4-8" in dotenv_text


def test_setup_cli_reports_missing_installer(tmp_path: Path, monkeypatch, capsys):
    missing = tmp_path / "missing.sh"
    monkeypatch.setattr(setup, "_INSTALL_BAREMETAL_SH", missing)

    rc = setup.main([])

    assert rc == 1
    assert str(missing) in capsys.readouterr().err
