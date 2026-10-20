# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A DeepSeek TraceLens session runs within the caller's budget."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.common import agent_sdk_runner, llm_config
from hyperloom.orchestrator.trace_analysis import tracelens_deepseek
from hyperloom.orchestrator.trace_analysis import tracelens_skill_runner as tlr


def _skill_request(tmp_path: Path) -> tlr.TraceLensSkillRequest:
    return tlr.TraceLensSkillRequest(
        skill_path=tmp_path / "skill.md",
        trace_path=tmp_path / "trace.json.gz",
        output_dir=tmp_path / "out",
        tracelens_root=tmp_path,
        tracelens_internal_root=tmp_path / "TraceLens-internal",
        platform="MI300X",
        framework="sglang",
        analysis_mode="default",
        capture_folder=None,
    )


async def test_tracelens_forwards_its_budget_to_the_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A TraceLens budget of a few minutes must not leave the harness session running for its two-hour default."""
    seen: dict[str, Any] = {}

    async def _fake_drive(
        request: agent_sdk_runner.AgentSessionRequest, owed: agent_sdk_runner.OwedArtifact | None = None
    ) -> agent_sdk_runner.AgentSdkTurnResult:
        seen["request"], seen["owed"] = request, owed
        (tmp_path / "out" / "analysis.md").write_text("done")
        return agent_sdk_runner.AgentSdkTurnResult(text="report written")

    monkeypatch.setattr(llm_config, "resolve_agent_backend", lambda: llm_config.AGENT_BACKEND_DEEPSEEK)
    monkeypatch.setattr(tracelens_deepseek, "drive_agent_session", _fake_drive)

    result = await tlr.run_tracelens_skill(_skill_request(tmp_path), budget_minutes=10.0)

    assert seen["request"].timeout_sec == 600.0
    assert seen["request"].cwd == tmp_path
    # The session owes analysis.md, which the fake wrote.
    assert seen["owed"] is not None and seen["owed"].done()
    assert result.runner == "deepseek"
    assert result.raw_text == "report written"
