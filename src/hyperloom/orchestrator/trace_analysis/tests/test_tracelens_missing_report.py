# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A TraceLens run that wrote no analysis.md reports what the agent said."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom_common.llm import llm_config
from hyperloom.common.codex_session import CodexSessionResult
from hyperloom.orchestrator.trace_analysis import tracelens_skill_runner as tlr


def _skill_request(tmp_path: Path) -> tlr.TraceLensSkillRequest:
    return tlr.TraceLensSkillRequest(
        skill_path=tmp_path / "skill.md",
        output_dir=tmp_path / "out",
        tracelens_root=tmp_path,
        tracelens_internal_root=None,
        capture_folder=None,
    )


async def _run_codex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    async def _turn(**_kwargs: object) -> CodexSessionResult:
        return CodexSessionResult(text=text)

    monkeypatch.setattr(llm_config, "resolve_agent_backend", lambda: llm_config.AGENT_BACKEND_CODEX)
    await tlr.run_tracelens_skill(
        _skill_request(tmp_path), budget_minutes=1.0, sdk=tlr.SdkEntryPoints(codex_turn_runner=_turn)
    )


async def test_a_missing_report_carries_the_agents_reply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent's own explanation is the only record of why it wrote nothing."""
    reply = "bwrap: Can't mkdir /checkout/.agents: Permission denied"

    with pytest.raises(RuntimeError) as excinfo:
        await _run_codex(tmp_path, monkeypatch, reply)

    message = str(excinfo.value)
    assert "did not write" in message
    assert reply in message


async def test_a_missing_report_without_a_reply_keeps_the_plain_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With nothing to quote, the message ends at the report path."""
    with pytest.raises(RuntimeError) as excinfo:
        await _run_codex(tmp_path, monkeypatch, "  \n")

    assert str(excinfo.value).endswith("analysis.md")


async def test_a_long_reply_is_cut_to_a_bounded_excerpt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The message lands in a status file and a log line, so it must stay short."""
    with pytest.raises(RuntimeError) as excinfo:
        await _run_codex(tmp_path, monkeypatch, "x" * 10_000)

    assert len(str(excinfo.value)) < 2_000
