# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Security and executable-resolution contract for Coordinator Hermes transport."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer.cli import backends as cli_backends
from hyperloom.inference_optimizer.protocol.intent import IntentType
from hyperloom.orchestrator.roles.hermes import HermesBackend, LLMCallFailed


@pytest.mark.asyncio
async def test_coordinator_hermes_is_tool_confined_without_yolo(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[list[str], dict]] = []
    binary = tmp_path / "custom-hermes"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)

    def fake_run(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        return SimpleNamespace(returncode=0, stdout="no intent", stderr="")

    monkeypatch.setattr("hyperloom.orchestrator.roles.hermes.subprocess.run", fake_run)
    backend = HermesBackend(
        allowed_intents=frozenset({IntentType.ALERT}),
        hermes_bin=str(binary),
        cwd=tmp_path,
    )

    await backend.run("status", allow_no_intent=True)

    argv, kwargs = calls[0]
    assert argv[0] == str(binary.resolve())
    assert "--safe-mode" in argv
    assert argv[argv.index("--toolsets") + 1] == "todo"
    assert "--yolo" not in argv
    assert kwargs["cwd"] == tmp_path.resolve()


def test_build_backends_propagates_custom_hermes_binary(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "custom-hermes"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("HYPERLOOM_AGENT_BACKEND", "hermes")
    monkeypatch.setenv("HYPERLOOM_HERMES_BIN", str(binary))

    built = cli_backends._build_backends(
        claude_model="claude-opus-5",
        codex_model="gpt-5.6",
        critic_choice="mock",
        session_dir=tmp_path,
        robustness_choice="mock",
    )

    assert built["orchestration"].hermes_bin == str(binary.resolve())


@pytest.mark.asyncio
async def test_coordinator_hermes_redacts_configured_secret_value(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "custom-hermes"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)

    def fake_run(argv, **kwargs):
        return SimpleNamespace(returncode=2, stdout="", stderr="provider rejected top-secret-value")

    monkeypatch.setattr("hyperloom.orchestrator.roles.hermes.subprocess.run", fake_run)
    backend = HermesBackend(
        allowed_intents=frozenset({IntentType.ALERT}),
        hermes_bin=str(binary),
        cwd=tmp_path,
        env={"PRIVATE_TOKEN": "top-secret-value"},
    )

    with pytest.raises(LLMCallFailed) as exc_info:
        await backend.run("status", allow_no_intent=True)

    message = str(exc_info.value)
    assert "top-secret-value" not in message
    assert "[REDACTED]" in message
