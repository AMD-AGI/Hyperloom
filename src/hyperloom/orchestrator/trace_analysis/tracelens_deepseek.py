# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the standalone TraceLens skill's session on the DeepSeek Harness."""

from __future__ import annotations

from pathlib import Path

from hyperloom.common.agent_sdk_runner import AgentSdkTurnResult, AgentSessionRequest, OwedArtifact, drive_agent_session


async def drive_tracelens_skill(request: AgentSessionRequest, report_path: Path) -> AgentSdkTurnResult:
    """Run the skill's session, continued until it has written ``report_path``.

    The harness ends a turn on any reply without a tool call, including one
    that writes a tool call out as text, so a session can stop mid-workflow.
    """
    owed = OwedArtifact(
        done=report_path.exists,
        continuation_prompt=(
            f"{report_path} has not been written yet, so the workflow is not finished. A tool call "
            "written out as text in a reply is not executed. Continue from the step you stopped at, "
            f"calling tools directly, and write {report_path} before you stop."
        ),
    )
    return await drive_agent_session(request, owed)
