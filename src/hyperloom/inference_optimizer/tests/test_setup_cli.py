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
    assert "HYPERLOOM_ENV_FILE" not in env
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
    install_script = (
        Path(setup.__file__).resolve().parent
        / "assets"
        / "install_baremetal.sh"
    )
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
                "unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL ANTHROPIC_CUSTOM_HEADERS",
                "unset DEEPSEEK_API_KEY DEEPSEEK_BASE_URL",
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


def test_baremetal_setup_authoritative_deepseek_env_does_not_require_openai(tmp_path: Path):
    install_script = (
        Path(setup.__file__).resolve().parent
        / "assets"
        / "install_baremetal.sh"
    )
    script_text = install_script.read_text(encoding="utf-8")
    start = script_text.index("read_dotenv_var() {")
    end = script_text.index("\nwrite_runtime_dotenv() {")
    credential_functions = script_text[start:end]
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "HYPERLOOM_LLM_MODE=deepseek",
                "DEEPSEEK_API_KEY=sk-deepseek",
                "DEEPSEEK_BASE_URL=https://api.deepseek.com/anthropic",
                "OPENAI_BASE_URL=https://gateway.example/v1",
                "OPENAI_API_KEY=stale-openai-key",
                "ANTHROPIC_BASE_URL=https://api.anthropic.com",
                "ANTHROPIC_API_KEY=stale-anthropic-key",
                "SAFE_API_KEY=stale-safe-key",
                "LLM_GATEWAY_KEY=stale-gateway-key",
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
                "OPENAI_BASE_URL=https://gateway.example/v1",
                "OPENAI_API_KEY=ambient-openai-key",
                "ANTHROPIC_BASE_URL=https://api.anthropic.com",
                "ANTHROPIC_API_KEY=ambient-anthropic-key",
                "SAFE_API_KEY=ambient-safe-key",
                "LLM_GATEWAY_KEY=ambient-gateway-key",
                "resolve_credentials",
                f"printf 'OPENAI_BASE_URL=%s\n' \"${{OPENAI_BASE_URL-}}\" > {tmp_path / 'deepseek-env.txt'}",
                f"printf 'OPENAI_API_KEY=%s\n' \"${{OPENAI_API_KEY-}}\" >> {tmp_path / 'deepseek-env.txt'}",
                f"printf 'ANTHROPIC_BASE_URL=%s\n' \"${{ANTHROPIC_BASE_URL-}}\" >> {tmp_path / 'deepseek-env.txt'}",
                f"printf 'ANTHROPIC_API_KEY=%s\n' \"${{ANTHROPIC_API_KEY-}}\" >> {tmp_path / 'deepseek-env.txt'}",
                f"printf 'SAFE_API_KEY=%s\n' \"${{SAFE_API_KEY-}}\" >> {tmp_path / 'deepseek-env.txt'}",
                f"printf 'DEEPSEEK_API_KEY=%s\n' \"${{DEEPSEEK_API_KEY-}}\" >> {tmp_path / 'deepseek-env.txt'}",
                f"printf 'DEEPSEEK_BASE_URL=%s\n' \"${{DEEPSEEK_BASE_URL-}}\" >> {tmp_path / 'deepseek-env.txt'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(["bash", str(runner)], check=True)

    text = dotenv.read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY=sk-deepseek" in text
    assert "DEEPSEEK_BASE_URL=https://api.deepseek.com/anthropic" in text
    assert "OPENAI_BASE_URL=" not in text
    assert "OPENAI_API_KEY=" not in text
    assert "ANTHROPIC_BASE_URL=" not in text
    assert "ANTHROPIC_API_KEY=" not in text
    assert "SAFE_API_KEY=" not in text
    resolved_env = (tmp_path / "deepseek-env.txt").read_text(encoding="utf-8")
    assert resolved_env == (
        "OPENAI_BASE_URL=\n"
        "OPENAI_API_KEY=\n"
        "ANTHROPIC_BASE_URL=\n"
        "ANTHROPIC_API_KEY=\n"
        "SAFE_API_KEY=\n"
        "DEEPSEEK_API_KEY=sk-deepseek\n"
        "DEEPSEEK_BASE_URL=https://api.deepseek.com/anthropic\n"
    )


def test_install_preflights_accept_deepseek_only_without_openai(tmp_path: Path):
    script_paths = [
        (
            "install",
            Path(setup.__file__).resolve().parent / "assets" / "install.sh",
            ["preflight_load_dotenv() { :; }"],
        ),
        (
            "kernel",
            Path(setup.__file__).resolve().parents[1]
            / "agents"
            / "kernel"
            / "scripts"
            / "install.sh",
            [],
        ),
    ]
    for name, script_path, stubs in script_paths:
        script_text = script_path.read_text(encoding="utf-8")
        start = script_text.index("preflight_validate_credentials() {")
        end = script_text.index("\npreflight_validate_credentials", start)
        runner = tmp_path / f"{name}-deepseek-preflight.sh"
        runner.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    f"REPO_ROOT={tmp_path}",
                    "CHECK_ONLY=0",
                    "DRY_RUN=0",
                    "DEEPSEEK_API_KEY=sk-deepseek",
                    "log() { :; }",
                    "warn() { :; }",
                    "die() { echo \"$*\" >&2; exit 99; }",
                    *stubs,
                    script_text[start:end],
                    "preflight_validate_credentials",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        subprocess.run(["bash", str(runner)], check=True)


def test_baremetal_install_no_longer_accepts_openai_safe_credential_flags():
    install_script = (
        Path(setup.__file__).resolve().parent
        / "assets"
        / "install_baremetal.sh"
    )
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


def test_kernel_install_no_longer_exports_openai_safe_credentials():
    install_script = (
        Path(setup.__file__).resolve().parents[1]
        / "agents"
        / "kernel"
        / "scripts"
        / "install.sh"
    )
    script_text = install_script.read_text(encoding="utf-8")
    write_start = script_text.index("write_env_file() {")
    write_end = script_text.index("\nensure_geak()", write_start)
    write_text = script_text[write_start:write_end]

    assert "_OPENAI_BASE_URL_VAL" not in script_text
    assert "_OPENAI_KEY_VAL" not in script_text
    assert "_snap_safe" not in script_text
    assert "_snap_openai" not in script_text
    assert "export OPENAI_BASE_URL" not in write_text
    assert "export OPENAI_API_KEY" not in write_text
    assert "upsert_dotenv_var OPENAI_BASE_URL" not in write_text
    assert "upsert_dotenv_var OPENAI_API_KEY" not in write_text
    assert "SAFE_API_KEY:-" not in script_text


def test_packaged_install_sh_resolves_target_workspace_root(tmp_path: Path):
    install_script = (
        Path(setup.__file__).resolve().parent
        / "assets"
        / "install.sh"
    )
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
    install_script = (
        Path(setup.__file__).resolve().parent
        / "assets"
        / "install.sh"
    )
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
    assert "ANTHROPIC_API_KEY=sk-ant-real-key" in dotenv_text
    assert "ANTHROPIC_BASE_URL=https://api.anthropic.com" in dotenv_text
    # kernel-agent env mirrors the same Anthropic values and no OpenAI leak.
    assert "export ANTHROPIC_API_KEY='sk-ant-real-key'" in kernel_text
    assert "export ANTHROPIC_BASE_URL='https://api.anthropic.com'" in kernel_text
    assert "export OPENAI_API_KEY=" not in kernel_text
    assert "OPENAI_API_KEY=" not in dotenv_text


def test_kernel_env_persists_geak_claude_model_to_dotenv(tmp_path: Path):
    """Fresh-shell CLI starts from .env, so GEAK_CLAUDE_MODEL must be persisted
    there in addition to kernel-agent.env.sh."""
    def bash_path(path: Path) -> str:
        text = str(path)
        if path.drive:
            rest = text[len(path.drive):].replace("\\", "/")
            return f"/mnt/{path.drive[0].lower()}{rest}"
        return text

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
