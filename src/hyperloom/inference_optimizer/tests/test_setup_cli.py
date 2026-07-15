# Copyright Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import subprocess

from pathlib import Path

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
    assert env["HYPERLOOM_ENV_FILE"] == str(tmp_path / ".env")
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
    monkeypatch.setenv("HYPERLOOM_ENV_FILE", "/stale/.env")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://llm-api.amd.com/anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stale-anthropic-key")
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "Ocp-Apim-Subscription-Key: stale")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "stale-openai-key")
    monkeypatch.setenv("SAFE_API_KEY", "stale-safe-key")
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
    assert env["HYPERLOOM_ENV_FILE"] == str(tmp_path / ".env")
    assert env["HYPERLOOM_SKILL_PATH"] == str(tmp_path / "SKILL.md")
    assert env["HYPERLOOM_SETUP_ENV_AUTHORITATIVE"] == "1"
    for key in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_CUSTOM_HEADERS",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "SAFE_API_KEY",
        "DEEPSEEK_API_KEY",
        "LLM_GATEWAY_KEY",
        "CLAUDE_MODEL",
        "CODEX_MODEL",
    ):
        assert key not in env


def test_baremetal_setup_authoritative_anthropic_env_removes_openai_keys(tmp_path: Path):
    install_script = (
        Path(setup.__file__).resolve().parent
        / "assets"
        / "install_baremetal.sh"
    )
    script_text = install_script.read_text(encoding="utf-8")
    start = script_text.index("read_dotenv_var() {")
    end = script_text.index("\nwrite_combined_env() {")
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
                "SAFE_API_KEY=stale-safe-key",
                "LLM_GATEWAY_KEY=stale-gateway-key",
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
                "SAFE_API_KEY_PLACEHOLDER=ak-your-api-key-here",
                "CHECK_ONLY=0",
                "DRY_RUN=0",
                "SAFE_API_KEY_ARG=",
                "OPENAI_BASE_URL_ARG=",
                "log() { :; }",
                "warn() { :; }",
                "die() { echo \"$*\" >&2; exit 99; }",
                "is_interactive() { return 1; }",
                credential_functions,
                "HYPERLOOM_SETUP_ENV_AUTHORITATIVE=1",
                "OPENAI_BASE_URL=https://api.anthropic.com",
                "OPENAI_API_KEY=ambient-openai-key",
                "OPENAI_CUSTOM_HEADERS='ambient-header: stale'",
                "SAFE_API_KEY=ambient-safe-key",
                "LLM_GATEWAY_KEY=ambient-gateway-key",
                "resolve_credentials",
                f"printf 'OPENAI_BASE_URL=%s\n' \"${{OPENAI_BASE_URL-}}\" > {tmp_path / 'resolved-env.txt'}",
                f"printf 'OPENAI_API_KEY=%s\n' \"${{OPENAI_API_KEY-}}\" >> {tmp_path / 'resolved-env.txt'}",
                f"printf 'OPENAI_CUSTOM_HEADERS=%s\n' \"${{OPENAI_CUSTOM_HEADERS-}}\" >> {tmp_path / 'resolved-env.txt'}",
                f"printf 'SAFE_API_KEY=%s\n' \"${{SAFE_API_KEY-}}\" >> {tmp_path / 'resolved-env.txt'}",
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
    assert "SAFE_API_KEY=" not in text
    assert "LLM_GATEWAY_KEY=" not in text
    resolved_env = (tmp_path / "resolved-env.txt").read_text(encoding="utf-8")
    assert resolved_env == (
        "OPENAI_BASE_URL=\n"
        "OPENAI_API_KEY=\n"
        "OPENAI_CUSTOM_HEADERS=\n"
        "SAFE_API_KEY=\n"
        "LLM_GATEWAY_KEY=\n"
    )


def test_kernel_env_authoritative_anthropic_mode_does_not_emit_openai_aliases(tmp_path: Path):
    install_script = (
        Path(setup.__file__).resolve().parents[1]
        / "agents"
        / "kernel"
        / "scripts"
        / "install.sh"
    )
    script_text = install_script.read_text(encoding="utf-8")
    upsert_start = script_text.index("upsert_dotenv_var() {")
    upsert_end = script_text.index("\n# In --check-only mode")
    env_start = script_text.index("write_env_file() {")
    env_end = script_text.index("\nensure_geak() {")
    dotenv_helpers = script_text[upsert_start:upsert_end]
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
                "SAFE_API_KEY=stale-safe-key",
                "LLM_GATEWAY_KEY=stale-gateway-key",
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
                "GEAK_API_KEY_VAL=",
                "LLM_GATEWAY_KEY=",
                "LLM_API_KEY=",
                "GEAK_BASE_URL_VAL=",
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
    assert "export SAFE_API_KEY=" not in kernel_text
    assert "OPENAI_BASE_URL=" not in dotenv_text
    assert "OPENAI_API_KEY=" not in dotenv_text
    assert "SAFE_API_KEY=" not in dotenv_text
    assert "LLM_GATEWAY_KEY=" not in dotenv_text


def test_kernel_env_keeps_anthropic_creds_in_dotenv(tmp_path: Path):
    """Writing kernel-agent env must NOT wipe the Anthropic creds the operator
    put in .env (an Anthropic-only setup must keep ANTHROPIC_API_KEY /
    ANTHROPIC_BASE_URL after install)."""
    install_script = (
        Path(setup.__file__).resolve().parents[1]
        / "agents"
        / "kernel"
        / "scripts"
        / "install.sh"
    )
    script_text = install_script.read_text(encoding="utf-8")
    upsert_start = script_text.index("upsert_dotenv_var() {")
    upsert_end = script_text.index("\n# In --check-only mode")
    env_start = script_text.index("write_env_file() {")
    env_end = script_text.index("\nensure_geak() {")
    dotenv_helpers = script_text[upsert_start:upsert_end]
    write_env_file = script_text[env_start:env_end]
    dotenv = tmp_path / ".env"
    kernel_env = tmp_path / "runtime" / "kernel-agent.env.sh"
    dotenv.write_text(
        "\n".join(
            [
                "ANTHROPIC_API_KEY=sk-ant-real-key",
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
                "_ANTHROPIC_KEY_VAL=sk-ant-real-key",
                "_OPENAI_BASE_URL_VAL=",
                "_OPENAI_KEY_VAL=",
                "GEAK_API_KEY_VAL=",
                "LLM_GATEWAY_KEY=",
                "LLM_API_KEY=",
                "GEAK_BASE_URL_VAL=",
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
    assert "ANTHROPIC_API_KEY=sk-ant-real-key" in dotenv_text
    assert "ANTHROPIC_BASE_URL=https://api.anthropic.com" in dotenv_text
    # kernel-agent env mirrors the same Anthropic values and no OpenAI leak.
    assert "export ANTHROPIC_API_KEY='sk-ant-real-key'" in kernel_text
    assert "export ANTHROPIC_BASE_URL='https://api.anthropic.com'" in kernel_text
    assert "export OPENAI_API_KEY=" not in kernel_text
    assert "OPENAI_API_KEY=" not in dotenv_text


def test_setup_cli_reports_missing_installer(tmp_path: Path, monkeypatch, capsys):
    missing = tmp_path / "missing.sh"
    monkeypatch.setattr(setup, "_INSTALL_BAREMETAL_SH", missing)

    rc = setup.main([])

    assert rc == 1
    assert str(missing) in capsys.readouterr().err
