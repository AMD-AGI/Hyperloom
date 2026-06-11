# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""CI-collected tests for TraceLens agent transcript persistence (#266).

The broader ``kernel-agent/tools/test_tracelens_csv.py`` file exercises many
TraceLens integration paths that require repo-local TraceLens assets. These
tests isolate the small SDK-runner contract added by #266 so CI covers the
transcript feature without pulling in those environment-dependent cases.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_TOOL_DIR = Path(__file__).resolve().parents[2] / "kernel-agent" / "tools"
if str(_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOL_DIR))

import tracelens_skill_runner as tlr  # noqa: E402


class _FakeOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_run_tracelens_skill_writes_agent_transcript(tmp_path):
    @dataclass
    class TextBlock:
        text: str

    @dataclass
    class ToolUseBlock:
        name: str
        input: dict
        id: str = "tu_1"

    @dataclass
    class AssistantMessage:
        content: list[Any]

    @dataclass
    class ResultMessage:
        content: list[Any] = field(default_factory=list)
        result: str = ""
        usage: dict = field(default_factory=dict)

    output_dir = tmp_path / "out"

    async def _fake_query(*, prompt, options):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "analysis.md").write_text("# report\n", encoding="utf-8")
        yield AssistantMessage(content=[TextBlock("starting analysis")])
        yield AssistantMessage(content=[
            ToolUseBlock(name="Bash", input={"command": "ls"}, id="tu_42"),
        ])
        yield ResultMessage(result="done", usage={"input_tokens": 10})

    res = asyncio.run(tlr.run_tracelens_skill(
        skill_path=tmp_path / "skill.md",
        trace_path=tmp_path / "trace.json.gz",
        output_dir=output_dir,
        tracelens_root=tmp_path,
        tracelens_internal_root=tmp_path / "TraceLens-internal",
        platform="MI300X",
        framework="sglang",
        analysis_mode="default",
        capture_folder=None,
        budget_minutes=1,
        sdk_query_factory=_fake_query,
        sdk_options_cls=_FakeOptions,
    ))

    transcript_path = output_dir / "agent_transcript.jsonl"
    assert transcript_path.exists()
    assert res.artifact_paths["tracelens_agent_transcript"] == str(transcript_path)

    lines = [
        json.loads(line)
        for line in transcript_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 3
    tool_blocks = [
        block
        for record in lines
        for block in record.get("content", [])
        if block.get("block") in ("ToolUseBlock", "ServerToolUseBlock")
    ]
    assert any(
        block.get("name") == "Bash"
        and block.get("input", {}).get("command") == "ls"
        for block in tool_blocks
    )
    assert any(record.get("usage", {}).get("input_tokens") == 10 for record in lines)


def test_transcript_failure_never_aborts_run(tmp_path):
    class _Unserializable:
        @property
        def text(self):
            raise RuntimeError("boom")

    class _Message:
        def __init__(self, content):
            self.content = content

    output_dir = tmp_path / "out"

    async def _fake_query(*, prompt, options):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "analysis.md").write_text("# report\n", encoding="utf-8")
        yield _Message(content=[_Unserializable()])

    res = asyncio.run(tlr.run_tracelens_skill(
        skill_path=tmp_path / "skill.md",
        trace_path=tmp_path / "trace.json.gz",
        output_dir=output_dir,
        tracelens_root=tmp_path,
        tracelens_internal_root=tmp_path / "TraceLens-internal",
        platform="MI300X",
        framework="sglang",
        analysis_mode="default",
        capture_folder=None,
        budget_minutes=1,
        sdk_query_factory=_fake_query,
        sdk_options_cls=_FakeOptions,
    ))

    assert res.report_path.exists()

