# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A DeepSeek TraceLens session runs within the caller's budget."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.common import deepseek_task, llm_config
from hyperloom.orchestrator.trace_analysis import tracelens_deepseek
from hyperloom.orchestrator.trace_analysis import tracelens_skill_runner as tlr


def _skill_request(tmp_path: Path) -> tlr.TraceLensSkillRequest:
    return tlr.TraceLensSkillRequest(
        skill_path=tmp_path / "skill.md",
        output_dir=tmp_path / "out",
        tracelens_root=tmp_path,
        tracelens_internal_root=tmp_path / "TraceLens-internal",
        capture_folder=None,
    )


async def test_tracelens_forwards_its_budget_to_the_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A TraceLens budget of a few minutes must not leave the harness session running for its two-hour default."""
    seen: dict[str, Any] = {}

    async def _fake_drive(
        request: deepseek_task.DeepSeekTaskRequest, owed: deepseek_task.OwedArtifact | None = None
    ) -> deepseek_task.DeepSeekTaskResult:
        seen["request"], seen["owed"] = request, owed
        (tmp_path / "out" / "analysis.md").write_text("done")
        return deepseek_task.DeepSeekTaskResult(text="report written")

    monkeypatch.setattr(llm_config, "resolve_agent_backend", lambda: llm_config.AGENT_BACKEND_DEEPSEEK)
    monkeypatch.setattr(tracelens_deepseek, "run_deepseek_task", _fake_drive)

    result = await tlr.run_tracelens_skill(_skill_request(tmp_path), budget_minutes=10.0)

    assert seen["request"].timeout_sec == 600.0
    assert seen["request"].cwd == tmp_path
    # The session owes analysis.md, which the fake wrote.
    assert seen["owed"] is not None and seen["owed"].done()
    assert result.runner == "deepseek"


async def test_the_harness_runs_the_system_prompt_every_runtime_shares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The harness adds its own tool mapping, so the prompt it is handed must carry none of another runtime's."""
    seen: dict[str, Any] = {}

    async def _fake_drive(
        request: deepseek_task.DeepSeekTaskRequest, owed: deepseek_task.OwedArtifact | None = None
    ) -> deepseek_task.DeepSeekTaskResult:
        seen["system_prompt"] = request.system_prompt
        (tmp_path / "out" / "analysis.md").write_text("done")
        return deepseek_task.DeepSeekTaskResult()

    monkeypatch.setattr(llm_config, "resolve_agent_backend", lambda: llm_config.AGENT_BACKEND_DEEPSEEK)
    monkeypatch.setattr(tracelens_deepseek, "run_deepseek_task", _fake_drive)

    await tlr.run_tracelens_skill(_skill_request(tmp_path), budget_minutes=10.0, model="claude-opus-5")

    assert seen["system_prompt"] == tlr._TRACELENS_SYSTEM_PROMPT
    assert "runtime mapping" not in seen["system_prompt"]
    assert tlr._TRACELENS_SYSTEM_PROMPT in tlr._CODEX_DEVELOPER_INSTRUCTIONS
