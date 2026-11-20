# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Each tool runs the orchestrator's existing implementation, and what that returns reaches the inbox."""

from pathlib import Path

import pytest

from hyperloom_multibranch.toolbox.tools import (
    InvalidRequest,
    JobStatus,
    ToolJobs,
    ToolRequest,
    tracelens_tool,
)
from hyperloom_multibranch.toolbox.tools.tests.sinks import RecordingSink


def test_tracelens_needs_a_trace() -> None:
    with pytest.raises(InvalidRequest, match="tracelens needs trace_input"):
        tracelens_tool().validate(ToolRequest("tracelens", "s1", {}))


async def test_a_handler_tool_gets_its_payload_and_the_job_directory(tmp_path: Path) -> None:
    seen: list[tuple[dict, Path]] = []

    async def handler(payload: dict, *, session_dir: Path) -> dict:
        seen.append((payload, session_dir))
        return {"status": "completed", "hot_kernels": ["gemm"]}

    tool = tracelens_tool()
    tool._handler = handler
    sink = RecordingSink()
    runner = ToolJobs([tool], sink, tmp_path, push_delay_seconds=0)
    job_id = runner.submit(ToolRequest("tracelens", "s3", {"trace_input": "/t"}), job_id="j9")
    assert await runner.wait(job_id) is JobStatus.SUCCEEDED
    assert [p for p, _ in seen] == [{"trace_input": "/t"}]
    assert seen[0][1] != tmp_path / "j9"  # run on local scratch, not the (possibly shared) job directory
    assert sink.pointers[0].summary == "succeeded: trace analysed"
