# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the bounded cold-start validation tool."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from hyperloom.inference_optimizer import experience_v1
from hyperloom.inference_optimizer.tools import cold_start_check as check
from hyperloom.orchestrator.roles.base import BackendTurnResult


def test_dotenv_rejects_active_placeholders_without_printing_values(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "OPENAI_BASE_URL=https://<your-gateway-host>/v1\nOPENAI_API_KEY=ak-your-api-key-here\n",
        encoding="utf-8",
    )

    result = check._check_dotenv(tmp_path)

    assert result.status == "failed"
    assert result.detail == "active placeholder values: OPENAI_API_KEY, OPENAI_BASE_URL"
    assert "ak-your" not in result.detail


def test_dotenv_allows_commented_placeholders(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "# OPENAI_BASE_URL=https://<your-gateway-host>/v1\n# OPENAI_API_KEY=ak-your-api-key-here\n",
        encoding="utf-8",
    )

    assert check._check_dotenv(tmp_path).status == "passed"


def test_required_experience_kb_fails_when_collection_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr(experience_v1, "enabled", lambda: False)

    result = check._check_experience_kb(require_experience_kb=True)

    assert result.status == "failed"
    assert result.detail == "HYPERLOOM_KB_ENABLE is false"


def test_fleet_experience_kb_cold_start_checks_remote_health(monkeypatch) -> None:
    seen = {"health": 0}

    class Client:
        def health(self):
            seen["health"] += 1
            return {"status": "ok"}

    module = ModuleType("hyperloom_kb")
    module.experience_kb_from_env = lambda: SimpleNamespace(
        enabled=True,
        client=Client(),
    )
    monkeypatch.setitem(sys.modules, "hyperloom_kb", module)
    monkeypatch.setattr(experience_v1, "enabled", lambda: True)
    monkeypatch.setattr(experience_v1, "validate_experience_config", lambda: None)

    result = check._check_experience_kb(require_experience_kb=True)

    assert result.status == "passed"
    assert "health check succeeded" in result.detail
    assert seen["health"] == 1


def test_llm_round_trip_uses_production_claude_backend(monkeypatch) -> None:
    import hyperloom.orchestrator.roles.claude as claude_module

    seen: dict[str, object] = {}

    class FakeClaudeBackend:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        async def run(self, prompt, **kwargs):
            seen["prompt"] = prompt
            seen["run_kwargs"] = kwargs
            return BackendTurnResult(raw_text="HYPERLOOM_COLD_START_OK")

    monkeypatch.setattr(check.llm_config, "is_openai_only", lambda: False)
    monkeypatch.setattr(claude_module, "ClaudeBackend", FakeClaudeBackend)

    result = asyncio.run(
        check._check_llm_round_trip(
            claude_model="claude-test",
            codex_model="unused",
            timeout=10,
        )
    )

    assert result.status == "passed"
    assert seen["model"] == "claude-test"
    assert seen["raw_completion"] is True
    assert seen["run_kwargs"]["tools"] == []


def test_process_failure_redacts_credentials(monkeypatch) -> None:
    class Completed:
        returncode = 1
        stdout = ""
        stderr = "request failed with ak-YB-secret-value"

    monkeypatch.setattr(check.subprocess, "run", lambda *args, **kwargs: Completed())

    result = check._run_process("probe", ["false"], timeout=1)

    assert result.status == "failed"
    assert "ak-YB-secret-value" not in result.detail
    assert "ak-[REDACTED]" in result.detail
