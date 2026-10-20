# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A DeepSeek TraceLens session runs within the caller's budget."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.common import agent_sdk_runner
from hyperloom.orchestrator.trace_analysis import tracelens_skill_runner


async def test_tracelens_forwards_its_budget_to_the_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def _fake_drive(**kwargs: Any) -> agent_sdk_runner.AgentSdkTurnResult:
        seen.update(kwargs)
        (tmp_path / "analysis.md").write_text("done")
        return agent_sdk_runner.AgentSdkTurnResult(text="", error="", chunks=[])

    monkeypatch.setattr(agent_sdk_runner, "drive_agent_session", _fake_drive)

    await tracelens_skill_runner._run_tracelens_skill_deepseek(
        prompt="p",
        output_dir=tmp_path,
        prefix_path=tmp_path / "prefix",
        tracelens_root=tmp_path,
        model="deepseek-v4-flash",
        timeout_sec=600.0,
        log=None,
    )

    assert seen["timeout_sec"] == 600.0
