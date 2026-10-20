# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the standalone TraceLens skill on the DeepSeek Harness."""

from __future__ import annotations

from hyperloom.common.agent_sdk_runner import AgentSessionRequest, OwedArtifact, drive_agent_session

from .tracelens_skill_runner import (
    _CODEX_DEVELOPER_INSTRUCTIONS,
    DEFAULT_ALLOWED_TOOLS,
    TraceLensSkillRunResult,
    _SkillRun,
)

_MAX_TURNS = 300


async def run_tracelens_skill_deepseek(
    run: _SkillRun,
    prompt: str,
    *,
    model: str,
    timeout_sec: float,
) -> TraceLensSkillRunResult:
    """Run the TraceLens skill on the DeepSeek Harness, continued until ``analysis.md`` exists.

    Args:
        run (_SkillRun): The run's request, command-prefix cache path and logging callback.
        prompt (str): The orchestrator prompt.
        model (str): The harness model id.
        timeout_sec (float): Wall-clock budget of the whole session.

    Returns:
        TraceLensSkillRunResult: The artifacts produced by the run.

    Raises:
        RuntimeError: If ``analysis.md`` was not written.
    """
    output_dir = run.request.output_dir
    report_path = output_dir / "analysis.md"
    request = AgentSessionRequest(
        prompt=prompt,
        system_prompt=_CODEX_DEVELOPER_INSTRUCTIONS.replace("Codex", "DeepSeek Harness"),
        cwd=run.request.tracelens_root,
        model=model,
        max_turns=_MAX_TURNS,
        allowed_tools=DEFAULT_ALLOWED_TOOLS,
        log=run.log,
        timeout_sec=timeout_sec,
    )
    owed = OwedArtifact(
        done=report_path.exists,
        continuation_prompt=(
            f"{report_path} has not been written yet, so the workflow is not finished. A tool call "
            "written out as text in a reply is not executed. Continue from the step you stopped at, "
            f"calling tools directly, and write {report_path} before you stop."
        ),
    )
    result = await drive_agent_session(request, owed)

    if not report_path.exists():
        if result.error:
            raise RuntimeError(f"TraceLens DeepSeek runner failed before writing {report_path}: {result.error}")
        raise RuntimeError(f"TraceLens DeepSeek runner did not write {report_path}")
    artifact_paths = {
        "tracelens_agent_report": str(report_path),
        "tracelens_cmd_prefix": str(run.prefix_path),
    }
    if result.error:
        artifact_paths["tracelens_agent_sdk_error"] = result.error
    return TraceLensSkillRunResult(
        output_dir=output_dir,
        report_path=report_path,
        runner="deepseek",
        raw_text=result.text,
        artifact_paths=artifact_paths,
    )
