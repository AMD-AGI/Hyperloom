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
    assert resolved_env == ("OPENAI_BASE_URL=\nOPENAI_API_KEY=\nOPENAI_CUSTOM_HEADERS=\nLLM_GATEWAY_KEY=\n")


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
                'warn() { echo "$*" >&2; }',
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
                "log() { :; }",
                "warn() { :; }",
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

    out = subprocess.run(["bash", str(runner)], check=True, text=True, stdout=subprocess.PIPE).stdout.splitlines()

    assert out == ["probe triton", "probe aiter"]


def test_baremetal_preflight_probes_atom_by_default():
    """Phase 1 must accept an atom-only host without extra flags.

    The default probe list gated on sglang/vllm alone, so setup inside an
    atom-only image died with 'no serving framework importable' even though
    atom was installed and is a registered framework.
    """
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    text = install_script.read_text(encoding="utf-8")

    m = re.search(r'^FRAMEWORKS="\$\{FRAMEWORKS:-([^}]+)\}"', text, re.MULTILINE)
    assert m, "install_baremetal.sh must define an overridable FRAMEWORKS default"
    assert "atom" in m.group(1).split(","), (
        "atom must be in the default Phase 1 probe list; otherwise setup inside an atom-only image fails preflight"
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


def _sourceable_installer(install_script: Path, tmp_path: Path) -> Path:
    """The real installer with its ``main`` invocation stripped.

    Lets a test source the script and drive its actual functions against a fake
    host, so behaviour is asserted end to end rather than by matching source
    text.
    """
    text = install_script.read_text(encoding="utf-8")
    marker = '\nmain "$@"\n'
    assert marker in text, "install_baremetal.sh must end by invoking main"
    lib = tmp_path / "installer_lib.sh"
    lib.write_text(text.replace(marker, "\n"), encoding="utf-8")
    return lib


def _fake_python(tmp_path: Path, importable: set[str], hip: str = "7.2.0") -> Path:
    """A python stub that answers ``_py_has`` probes and the torch.version.hip read."""
    cases = "\n".join(f"""  *"find_spec('{name}')"*) exit 0 ;;""" for name in sorted(importable))
    stub = tmp_path / "fake-python"
    stub.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'case "$*" in',
                cases,
                "  *find_spec*) exit 1 ;;",
                f"  -) printf '{hip}\\n'; exit 0 ;;",
                "esac",
                "exit 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


def _drive_installer(
    tmp_path: Path,
    *,
    importable: set[str],
    dotenv: Path,
    body: str,
    install_framework: str = "none",
) -> subprocess.CompletedProcess:
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    lib = _sourceable_installer(install_script, tmp_path)
    py = _fake_python(tmp_path, importable)
    runner = tmp_path / "runner.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"source {lib}",
                f'DOTENV="{dotenv}"',
                f'USER_DATA_PATH="{tmp_path}/data"',
                f'INSTALL_FRAMEWORK="{install_framework}"',
                "FRAMEWORK_ENV=shared",
                f'VLLM_VENV_ROOT="{tmp_path}/absent"',
                "DRY_RUN=0",
                "CHECK_ONLY=0",
                f'resolve_python() {{ printf "%s" "{py}"; }}',
                body,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return subprocess.run(["bash", str(runner)], text=True, capture_output=True)


def _dotenv_lines(dotenv: Path) -> list[str]:
    return dotenv.read_text(encoding="utf-8").splitlines()


def test_baremetal_atom_only_host_writes_framework_atom(tmp_path: Path):
    """The whole point of probing atom: the .env downstream skills read must say so.

    Preflight accepting atom is not enough — resolution used to return only
    sglang/vllm, so an atom-only host finished setup with no FRAMEWORK at all
    and every downstream skill defaulted to the wrong engine.
    """
    dotenv = tmp_path / ".env"
    res = _drive_installer(tmp_path, importable={"atom"}, dotenv=dotenv, body="write_runtime_dotenv")

    assert res.returncode == 0, res.stderr
    assert "FRAMEWORK=atom" in _dotenv_lines(dotenv)


def test_baremetal_clears_stale_framework_when_none_importable(tmp_path: Path):
    """A re-imaged host must not keep pointing at an engine that is gone."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("FRAMEWORK=sglang\nKEEP_ME=1\n", encoding="utf-8")

    res = _drive_installer(tmp_path, importable=set(), dotenv=dotenv, body="write_runtime_dotenv")

    assert res.returncode == 0, res.stderr
    lines = _dotenv_lines(dotenv)
    assert not [ln for ln in lines if ln.startswith("FRAMEWORK=")], lines
    assert "KEEP_ME=1" in lines


def test_baremetal_framework_resolution_keeps_sglang_precedence(tmp_path: Path):
    """Adding atom must not change what an existing sglang host resolves to."""
    dotenv = tmp_path / ".env"
    res = _drive_installer(tmp_path, importable={"sglang", "atom"}, dotenv=dotenv, body="write_runtime_dotenv")

    assert res.returncode == 0, res.stderr
    assert "FRAMEWORK=sglang" in _dotenv_lines(dotenv)


def test_baremetal_next_steps_names_the_detected_framework(tmp_path: Path):
    """The closing prompt hardcoded sglang, sending atom hosts down a dead path."""
    res = _drive_installer(
        tmp_path,
        importable={"atom"},
        dotenv=tmp_path / ".env",
        body="print_next_steps",
    )

    assert res.returncode == 0, res.stderr
    assert "- Framework: atom" in res.stdout


def test_baremetal_profiler_hotfix_accepts_an_atom_only_host(tmp_path: Path):
    """The hotfix patches ROCm profiler libs, which torch.profiler uses on any engine."""
    res = _drive_installer(
        tmp_path,
        importable={"atom"},
        dotenv=tmp_path / ".env",
        body="rocm_profiler_hotfix_compatible && echo HOTFIX_ELIGIBLE",
    )

    assert "HOTFIX_ELIGIBLE" in res.stdout, res.stderr
    assert "neither sglang nor vllm" not in res.stderr


def test_baremetal_profiler_hotfix_still_skipped_without_any_framework(tmp_path: Path):
    res = _drive_installer(
        tmp_path,
        importable=set(),
        dotenv=tmp_path / ".env",
        body="rocm_profiler_hotfix_compatible && echo HOTFIX_ELIGIBLE",
    )

    assert "HOTFIX_ELIGIBLE" not in res.stdout
    assert "no serving framework importable" in res.stderr


_HOTFIX_HIP_SONAME = "libamdhip64.so.7.2.53211-35e8c7bf89"
_HOTFIX_TRACER_SONAME = "libroctracer64.so.4.1.70202"


_BACKUP_DIRNAME = ".profiler_hotfix_backup"


def _fake_python_for_torch_lib(
    dest: Path,
    torch_lib: Path,
    *,
    importable: frozenset[str] = frozenset({"sglang"}),
    runtime_ok: bool = True,
) -> Path:
    probe_cases = "\n".join(f"""  *"find_spec('{fw}')"*) exit 0 ;;""" for fw in sorted(importable))
    runtime_case = '  *torch.version.hip*) printf "7.2.0\\n" ;;'
    if not runtime_ok:
        runtime_case = (
            "  *torch.version.hip*)"
            f' grep -q hotfix "{torch_lib}/libamdhip64.so" 2>/dev/null && exit 1;'
            ' printf "7.2.0\\n" ;;'
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'case "$*" in',
                probe_cases,
                "  *find_spec*) exit 1 ;;",
                "esac",
                'probe="$(cat)"',
                'case "$probe" in',
                f'  *__file__*) printf "%s\\n" "{torch_lib}" ;;',
                runtime_case,
                '  *) printf "7.2.0\\n" ;;',
                "esac",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dest.chmod(0o755)
    return dest


def _fake_rocm_lib(tmp_path: Path) -> Path:
    rocm_lib = tmp_path / "rocm" / "lib"
    rocm_lib.mkdir(parents=True)
    (rocm_lib / _HOTFIX_HIP_SONAME).write_bytes(b"hotfix-hip-bytes")
    (rocm_lib / _HOTFIX_TRACER_SONAME).write_bytes(b"hotfix-tracer-bytes")
    (rocm_lib / "libamdhip64.so").symlink_to(_HOTFIX_HIP_SONAME)
    (rocm_lib / "libroctracer64.so").symlink_to(_HOTFIX_TRACER_SONAME)
    return rocm_lib


def _drive_torch_lib_sync(
    tmp_path: Path,
    *,
    body: str,
    torch_hip: bytes | None = None,
    torch_tracer: bytes | None = None,
    runtime_ok: bool = True,
    extra_lines: tuple[str, ...] = (),
) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """Drive the torch-lib helpers against a fake /opt/rocm and a fake torch/lib."""
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    lib = _sourceable_installer(install_script, tmp_path)

    rocm_lib = _fake_rocm_lib(tmp_path)
    torch_lib = tmp_path / "torch" / "lib"
    torch_lib.mkdir(parents=True)
    if torch_hip is not None:
        (torch_lib / "libamdhip64.so").write_bytes(torch_hip)
    if torch_tracer is not None:
        (torch_lib / "libroctracer64.so").write_bytes(torch_tracer)

    py = _fake_python_for_torch_lib(tmp_path / "fake-python-torch", torch_lib, runtime_ok=runtime_ok)
    runner = tmp_path / "runner.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"source {lib}",
                "DRY_RUN=0",
                "CHECK_ONLY=0",
                f'ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR="{rocm_lib}"',
                f'resolve_python() {{ printf "%s" "{py}"; }}',
                *extra_lines,
                body,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    res = subprocess.run(["bash", str(runner)], text=True, capture_output=True)
    return res, rocm_lib, torch_lib


_SYNC_BODY = 'sync_rocm_profiler_libs_to_torch_lib "$ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR"'
_SYNC_BODY_SOFT = f"{_SYNC_BODY} || true"


def test_baremetal_hotfix_syncs_resolved_libs_into_torch_lib(tmp_path: Path):
    """torch/lib copies carry DT_RPATH=$ORIGIN, so a /opt/rocm-only overlay is
    invisible to torch.profiler and decode traces stay empty."""
    res, _rocm_lib, torch_lib = _drive_torch_lib_sync(
        tmp_path,
        body=_SYNC_BODY,
        torch_hip=b"stock-hip-bytes",
        torch_tracer=b"stock-tracer-bytes",
    )

    assert res.returncode == 0, res.stderr
    assert (torch_lib / "libamdhip64.so").read_bytes() == b"hotfix-hip-bytes"
    assert (torch_lib / "libroctracer64.so").read_bytes() == b"hotfix-tracer-bytes"
    backup = torch_lib / _BACKUP_DIRNAME / "libamdhip64.so"
    assert backup.is_file(), "the replaced torch copy must be recoverable"
    assert backup.read_bytes() == b"stock-hip-bytes"


def test_hotfix_torch_lib_sync_skips_a_second_run(tmp_path: Path):
    """Re-running used to re-copy unconditionally and stamp a fresh backup dir
    holding the already-hotfixed copy, piling large .so files into site-packages."""
    res, _rocm_lib, torch_lib = _drive_torch_lib_sync(
        tmp_path,
        body=f"{_SYNC_BODY}\n{_SYNC_BODY}",
        torch_hip=b"stock-hip-bytes",
        torch_tracer=b"stock-tracer-bytes",
    )

    assert res.returncode == 0, res.stderr
    assert "already in sync" in res.stdout
    backup_dirs = sorted(p.name for p in torch_lib.glob(".profiler_hotfix_backup*"))
    assert backup_dirs == [_BACKUP_DIRNAME], backup_dirs
    # The one backup must still hold the vendor bytes, not the first hotfix.
    assert (torch_lib / _BACKUP_DIRNAME / "libamdhip64.so").read_bytes() == b"stock-hip-bytes"


def test_hotfix_torch_lib_sync_rolls_back_when_torch_breaks(tmp_path: Path):
    """cmp only proves the bytes landed; the new libamdhip64 still has to work
    against the other ROCm libs torch bundles under $ORIGIN."""
    res, _rocm_lib, torch_lib = _drive_torch_lib_sync(
        tmp_path,
        body=_SYNC_BODY_SOFT,
        torch_hip=b"stock-hip-bytes",
        torch_tracer=b"stock-tracer-bytes",
        runtime_ok=False,
    )

    assert res.returncode == 0, res.stderr
    assert "torch did not come up after the sync" in res.stderr
    assert (torch_lib / "libamdhip64.so").read_bytes() == b"stock-hip-bytes"
    assert (torch_lib / "libroctracer64.so").read_bytes() == b"stock-tracer-bytes"


def test_hotfix_torch_lib_sync_refreshes_a_snapshot_after_a_torch_upgrade(tmp_path: Path):
    """A torch upgrade between runs left the snapshot describing the previous
    install, so the new vendor libs were overwritten with no copy kept and a
    rollback would have pushed the old ABI into the new torch."""
    torch_lib = tmp_path / "torch" / "lib"
    res, _rocm_lib, _torch_lib = _drive_torch_lib_sync(
        tmp_path,
        body=(
            f"{_SYNC_BODY}\n"
            # Stand in for `pip install -U torch`: fresh vendor bytes in place.
            f'printf vendor-hip-v2 > "{torch_lib}/libamdhip64.so"\n'
            f'printf vendor-tracer-v2 > "{torch_lib}/libroctracer64.so"\n'
            f"{_SYNC_BODY}"
        ),
        torch_hip=b"vendor-hip-v1",
        torch_tracer=b"vendor-tracer-v1",
    )

    assert res.returncode == 0, res.stderr
    assert "vendor libs changed" in res.stdout
    backup = torch_lib / _BACKUP_DIRNAME
    assert (backup / "libamdhip64.so").read_bytes() == b"vendor-hip-v2"
    assert (backup / "libroctracer64.so").read_bytes() == b"vendor-tracer-v2"


def test_sync_torch_profiler_libs_does_not_warn_when_verify_is_off(tmp_path: Path):
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    lib = _sourceable_installer(install_script, tmp_path)
    rocm_lib = _fake_rocm_lib(tmp_path)
    torch_lib = tmp_path / "torch" / "lib"
    torch_lib.mkdir(parents=True)
    (torch_lib / "libamdhip64.so").write_bytes(b"stock-hip-bytes")
    (torch_lib / "libroctracer64.so").write_bytes(b"stock-tracer-bytes")
    py = _fake_python_for_torch_lib(tmp_path / "fake-python-torch", torch_lib)

    runner = tmp_path / "runner.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"source {lib}",
                "DRY_RUN=0",
                "CHECK_ONLY=0",
                f'ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR="{rocm_lib}"',
                f'resolve_python() {{ printf "%s" "{py}"; }}',
                'sync_torch_profiler_libs "$ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    res = subprocess.run(["bash", str(runner)], text=True, capture_output=True)

    assert res.returncode == 0, res.stderr
    assert "verification reported issues" not in res.stderr
    assert "synced into" in res.stdout


def test_hotfix_asset_change_keeps_the_vendor_backup(tmp_path: Path):
    hotfix_hip = tmp_path / "rocm" / "lib" / _HOTFIX_HIP_SONAME
    res, _rocm_lib, torch_lib = _drive_torch_lib_sync(
        tmp_path,
        body=(
            f"{_SYNC_BODY}\n"
            f'printf hotfix-hip-v2 > "{hotfix_hip}"\n'
            f"{_SYNC_BODY}"
        ),
        torch_hip=b"vendor-hip-bytes",
        torch_tracer=b"vendor-tracer-bytes",
    )

    assert res.returncode == 0, res.stderr
    assert "vendor libs changed" not in res.stdout
    assert (torch_lib / _BACKUP_DIRNAME / "libamdhip64.so").read_bytes() == b"vendor-hip-bytes"
    assert (torch_lib / "libamdhip64.so").read_bytes() == b"hotfix-hip-v2"


def test_partial_hotfix_asset_update_with_absent_tracer_keeps_vendor_backup(tmp_path: Path):
    """When torch never shipped libroctracer64.so, a single-library hotfix bump
    must not treat the injected tracer as a vendor refresh trigger."""
    hotfix_hip = tmp_path / "rocm" / "lib" / _HOTFIX_HIP_SONAME
    res, _rocm_lib, torch_lib = _drive_torch_lib_sync(
        tmp_path,
        body=(
            f"{_SYNC_BODY}\n"
            f'printf hotfix-hip-v2 > "{hotfix_hip}"\n'
            f"{_SYNC_BODY}"
        ),
        torch_hip=b"VENDOR-hip",
        torch_tracer=None,
    )
    backup = torch_lib / _BACKUP_DIRNAME

    assert res.returncode == 0, res.stderr
    assert "vendor libs changed" not in res.stdout
    assert (backup / "libamdhip64.so").read_bytes() == b"VENDOR-hip"
    assert not (backup / "libroctracer64.so").exists()
    assert "vendor:libroctracer64.so:absent" in (backup / ".fingerprint").read_text()


def test_refresh_preserves_vendor_when_torch_still_carries_hotfix(tmp_path: Path):
    """A truncated fingerprint can force a refresh, but the rebuild must not
    promote hotfix bytes already sitting in torch/lib into the vendor snapshot."""
    fp = f'"{tmp_path}/torch/lib/{_BACKUP_DIRNAME}/.fingerprint"'
    res, _rocm_lib, torch_lib = _drive_torch_lib_sync(
        tmp_path,
        body=(
            f"{_SYNC_BODY}\n"
            f"head -n 1 > {fp}.tmp {fp} && mv -f {fp}.tmp {fp}\n"
            f"{_SYNC_BODY}"
        ),
        torch_hip=b"vendor-hip-bytes",
        torch_tracer=b"vendor-tracer-bytes",
    )
    backup = torch_lib / _BACKUP_DIRNAME

    assert res.returncode == 0, res.stderr
    assert (backup / "libamdhip64.so").read_bytes() == b"vendor-hip-bytes"
    assert (backup / "libroctracer64.so").read_bytes() == b"vendor-tracer-bytes"
    assert not (backup / "libroctracer64.so").read_bytes().startswith(b"hotfix")


def test_sync_warns_when_framework_imports_but_torch_lib_is_missing(tmp_path: Path):
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    lib = _sourceable_installer(install_script, tmp_path)
    rocm_lib = _fake_rocm_lib(tmp_path)
    missing_torch_lib = tmp_path / "missing" / "torch" / "lib"
    py = _fake_python_for_torch_lib(tmp_path / "fake-python-torch", missing_torch_lib)

    runner = tmp_path / "runner.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"source {lib}",
                "DRY_RUN=0",
                "CHECK_ONLY=0",
                f'ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR="{rocm_lib}"',
                f'resolve_python() {{ printf "%s" "{py}"; }}',
                f'{_SYNC_BODY_SOFT}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    res = subprocess.run(["bash", str(runner)], text=True, capture_output=True)

    assert "torch/lib not resolved" in res.stderr


def test_hotfix_survives_a_runtime_probe_failing_for_unrelated_reasons(tmp_path: Path):
    """A busy GPU or HIP OOM fails the probe both before and after the restore.
    Once the vendor bytes are back, that must degrade to "not applied", not die."""
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    lib = _sourceable_installer(install_script, tmp_path)
    rocm_lib = _fake_rocm_lib(tmp_path)
    torch_lib = tmp_path / "torch" / "lib"
    torch_lib.mkdir(parents=True)
    (torch_lib / "libamdhip64.so").write_bytes(b"vendor-hip-bytes")
    (torch_lib / "libroctracer64.so").write_bytes(b"vendor-tracer-bytes")
    py = _fake_python_for_torch_lib(tmp_path / "fake-python-torch", torch_lib)

    runner = tmp_path / "runner.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"source {lib}",
                "DRY_RUN=0",
                "CHECK_ONLY=0",
                f'ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR="{rocm_lib}"',
                f'resolve_python() {{ printf "%s" "{py}"; }}',
                # Fails regardless of which bytes are in place.
                "verify_torch_runtime() { return 1; }",
                f"{_SYNC_BODY} || echo SYNC_REPORTED_FAILURE",
                "echo REACHED_NEXT_PHASE",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    res = subprocess.run(["bash", str(runner)], text=True, capture_output=True)

    assert "REACHED_NEXT_PHASE" in res.stdout, res.stderr
    assert "SYNC_REPORTED_FAILURE" in res.stdout
    assert "vendor libraries restored" in res.stderr
    assert (torch_lib / "libamdhip64.so").read_bytes() == b"vendor-hip-bytes"
    assert (torch_lib / "libroctracer64.so").read_bytes() == b"vendor-tracer-bytes"


def test_hotfix_phase_survives_a_failed_torch_lib_sync(tmp_path: Path):
    """The installer runs under set -e: a bare sync call would end setup instead
    of degrading to "hotfix not applied"."""
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    lib = _sourceable_installer(install_script, tmp_path)
    rocm_lib = _fake_rocm_lib(tmp_path)
    torch_lib = tmp_path / "torch" / "lib"
    torch_lib.mkdir(parents=True)
    (torch_lib / "libamdhip64.so").write_bytes(b"stock-hip-bytes")
    (torch_lib / "libroctracer64.so").write_bytes(b"stock-tracer-bytes")
    py = _fake_python_for_torch_lib(tmp_path / "fake-python-torch", torch_lib, runtime_ok=False)
    # apply_rocm_profiler_hotfix deletes the extract dir, so hand it a copy.
    extract = tmp_path / "extract"
    extract.mkdir()
    (extract / _HOTFIX_HIP_SONAME).write_bytes(b"hotfix-hip-bytes")
    (extract / _HOTFIX_TRACER_SONAME).write_bytes(b"hotfix-tracer-bytes")

    runner = tmp_path / "runner.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"source {lib}",
                "DRY_RUN=0",
                "CHECK_ONLY=0",
                f'ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR="{rocm_lib}"',
                f'resolve_python() {{ printf "%s" "{py}"; }}',
                # Stand in for the /opt/rocm half, which needs a real download.
                "rocm_profiler_hotfix_compatible() { return 0; }",
                "rocm_profiler_hotfix_applied() { return 0; }",
                "verify_rocm_profiler_hotfix() { return 0; }",
                f'download_rocm_profiler_hotfix_libs() {{ printf "%s" "{extract}"; }}',
                "apply_rocm_profiler_hotfix",
                "echo REACHED_NEXT_PHASE",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    res = subprocess.run(["bash", str(runner)], text=True, capture_output=True)

    assert "REACHED_NEXT_PHASE" in res.stdout, res.stderr
    assert "torch lib sync reported issues" in res.stderr
    assert (torch_lib / "libamdhip64.so").read_bytes() == b"stock-hip-bytes"


def test_hotfix_syncs_the_isolated_vllm_venv_torch_not_the_shared_one(tmp_path: Path):
    """vLLM defaults to FRAMEWORK_ENV=isolated with its own ROCm torch under
    $VLLM_VENV_ROOT, so a shared-interpreter lookup patched a torch that never
    runs the benchmark and left the profiler unfixed."""
    install_script = Path(setup.__file__).resolve().parent / "assets" / "install_baremetal.sh"
    lib = _sourceable_installer(install_script, tmp_path)
    rocm_lib = _fake_rocm_lib(tmp_path)

    shared_torch_lib = tmp_path / "shared" / "torch" / "lib"
    venv_torch_lib = tmp_path / "vllm-venv" / "torch" / "lib"
    for d in (shared_torch_lib, venv_torch_lib):
        d.mkdir(parents=True)
        (d / "libamdhip64.so").write_bytes(b"stock-hip-bytes")
        (d / "libroctracer64.so").write_bytes(b"stock-tracer-bytes")

    venv_root = tmp_path / "vllm-venv"
    shared_py = _fake_python_for_torch_lib(tmp_path / "shared-python", shared_torch_lib, importable=frozenset())
    _fake_python_for_torch_lib(venv_root / "bin" / "python", venv_torch_lib, importable=frozenset({"vllm"}))

    runner = tmp_path / "runner.sh"
    runner.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"source {lib}",
                "DRY_RUN=0",
                "CHECK_ONLY=0",
                "FRAMEWORK_ENV=isolated",
                f'VLLM_VENV_ROOT="{venv_root}"',
                f'ROCM_PROFILER_HOTFIX_TARGET_LIB_DIR="{rocm_lib}"',
                f'resolve_python() {{ printf "%s" "{shared_py}"; }}',
                _SYNC_BODY,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    res = subprocess.run(["bash", str(runner)], text=True, capture_output=True)

    assert res.returncode == 0, res.stderr
    assert (venv_torch_lib / "libamdhip64.so").read_bytes() == b"hotfix-hip-bytes"
    assert (shared_torch_lib / "libamdhip64.so").read_bytes() == b"stock-hip-bytes", (
        "the shared torch has no importable framework and must be left alone"
    )


def _drive_hotfix_gate(
    tmp_path: Path,
    *,
    run_mode: str,
    importable: set[str],
    in_container: bool = False,
):
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"HYPERLOOM_RUN_MODE={run_mode}\n", encoding="utf-8")
    return _drive_installer(
        tmp_path,
        importable=importable,
        dotenv=dotenv,
        body=(
            # Pinned so the result does not depend on whether the test host
            # itself is a container.
            f"running_in_container() {{ return {0 if in_container else 1}; }}\n"
            "rocm_profiler_hotfix_compatible && echo HOTFIX_ELIGIBLE"
        ),
    )


def test_docker_run_mode_skips_the_hotfix_for_a_vllm_image(tmp_path: Path):
    """vLLM ROCm images carry their own kineto profiler workaround, so overlaying
    /opt/rocm into their torch/lib trades a vendor-validated pair for an untested one."""
    res = _drive_hotfix_gate(tmp_path, run_mode="docker", importable={"vllm"})

    assert "HOTFIX_ELIGIBLE" not in res.stdout
    assert "container run without sglang" in res.stderr


def test_docker_run_mode_applies_the_hotfix_for_an_sglang_image(tmp_path: Path):
    res = _drive_hotfix_gate(tmp_path, run_mode="docker", importable={"sglang"})

    assert "HOTFIX_ELIGIBLE" in res.stdout, res.stderr


def test_baremetal_run_mode_keeps_the_hotfix_for_vllm(tmp_path: Path):
    """The framework split is docker-only: a bare-metal vLLM host profiles through
    the same ROCm libs the overlay fixes, so gating it there would lose profiling."""
    res = _drive_hotfix_gate(tmp_path, run_mode="baremetal", importable={"vllm"})

    assert "HOTFIX_ELIGIBLE" in res.stdout, res.stderr


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
                'checkout_aiter_ref() { printf \'checkout %s %s\\n\' "$1" "$2" >> "$CALLS_FILE"; }',
                'check_torch_triton_alignment() { printf \'align %s\\n\' "$1" >> "$CALLS_FILE"; }',
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
                'checkout_aiter_ref() { printf \'checkout %s %s\\n\' "$1" "$2" >> "$CALLS_FILE"; }',
                'check_torch_triton_alignment() { printf \'align %s\\n\' "$1" >> "$CALLS_FILE"; return 0; }',
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
                'if [ "$1" = "-" ]; then echo 3.12; exit 0; fi',
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
                '_py_has() { [ "$2" = aiter ] && return 0; return 0; }',
                "install_sglang_from_wheel() { :; }",
                "install_sglang_from_source() { :; }",
                f'install_compatible_aiter() {{ printf \'install_compatible_aiter %s %s\\n\' "$1" "$2" >> "$CALLS_FILE"; touch {import_flag}; }}',
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
                "_ANTHROPIC_CUSTOM_HEADERS_VAL='Ocp-Apim-Subscription-Key: ${ANTHROPIC_API_KEY}'",
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
    # A header-authenticated gateway needs its header here too, and the single
    # quotes must keep ${VAR} intact for parse_custom_headers to expand.
    assert "export ANTHROPIC_CUSTOM_HEADERS='Ocp-Apim-Subscription-Key: ${ANTHROPIC_API_KEY}'" in kernel_text
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
