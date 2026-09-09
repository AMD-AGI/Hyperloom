# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.trace import geak_harvest

WORKFLOW_DIR = "/opt/geak/e2e_workflow"
EXP_ROOT = "/shared/runs/session-under-test"
SESSION_ID = "geak-session-1"


def _assistant(msg_id: str, agent_model: str = "claude-opus-5", **usage: int) -> dict:
    return {
        "type": "assistant",
        "timestamp": "2026-01-01T00:00:10.000Z",
        "message": {
            "id": msg_id,
            "model": agent_model,
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": usage.get("input_tokens", 100),
                "output_tokens": usage.get("output_tokens", 30),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                "output_tokens_details": {"thinking_tokens": usage.get("thinking_tokens", 0)},
            },
        },
    }


@pytest.fixture
def claude_home(tmp_path: Path) -> Path:
    """A minimal Claude Code home holding one GEAK driving transcript."""
    home = tmp_path / "claude"
    proj = home / "projects" / "-opt-geak"
    proj.mkdir(parents=True)
    lines = [
        {"type": "summary", "cwd": WORKFLOW_DIR, "sessionId": SESSION_ID},
        {"type": "user", "cwd": WORKFLOW_DIR, "message": {"content": f"exp_root={EXP_ROOT}"}},
        _assistant("msg_a", input_tokens=1000, output_tokens=100, thinking_tokens=40),
        _assistant("msg_b", input_tokens=2000, output_tokens=50),
    ]
    (proj / f"{SESSION_ID}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return home


def _shard_rows(session_dir: Path, suffix: str) -> list[dict]:
    ext = session_dir / "reports" / "trace" / "ext"
    rows: list[dict] = []
    for path in sorted(ext.glob(f"*{suffix}")):
        rows += [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return rows


def _harvest(session_dir: Path, home: Path, **kw):
    return geak_harvest.harvest_geak_calls(
        session_dir=session_dir,
        session_id="hl-session",
        exp_root=kw.pop("exp_root", EXP_ROOT),
        workflow_dir=kw.pop("workflow_dir", WORKFLOW_DIR),
        phase="KERNEL_AGENT",
        claude_home=home,
        pid=4242,
        **kw,
    )


def test_harvest_emits_one_turn_row_and_one_detail_row_per_api_call(tmp_path, claude_home):
    sd = tmp_path / "session"
    result = _harvest(sd, claude_home)
    assert (result.transcripts, result.turns, result.api_calls) == (1, 1, 2)

    turns = _shard_rows(sd, "-4242.jsonl")
    details = _shard_rows(sd, ".detail.jsonl")
    assert len(turns) == 1 and len(details) == 2
    assert turns[0]["component"] == "geak"
    assert turns[0]["api_calls"] == 2
    # The join key is what makes the two ledgers one story.
    assert {d["call_id"] for d in details} == {turns[0]["call_id"]}
    assert [d["api_call_index"] for d in details] == [0, 1]


def test_thinking_is_moved_out_of_output_tokens_not_added_to_them(tmp_path, claude_home):
    sd = tmp_path / "session"
    _harvest(sd, claude_home)
    first = sorted(_shard_rows(sd, ".detail.jsonl"), key=lambda r: r["api_call_index"])[0]
    # Anthropic bills 100 output tokens of which 40 are thinking.
    assert first["output_tokens"] == 60
    assert first["reasoning_output_tokens"] == 40
    assert first["osl"] == 100


def test_rows_land_under_the_geak_task_path(tmp_path, claude_home):
    sd = tmp_path / "session"
    _harvest(sd, claude_home)
    turn = _shard_rows(sd, "-4242.jsonl")[0]
    assert turn["task_path"].split("/")[0] == geak_harvest.ROOT_SEGMENT
    assert turn["phase"] == "KERNEL_AGENT"


def test_a_second_harvest_of_an_unchanged_transcript_emits_nothing(tmp_path, claude_home):
    sd = tmp_path / "session"
    assert _harvest(sd, claude_home).api_calls == 2
    again = _harvest(sd, claude_home)
    assert (again.turns, again.api_calls) == (0, 0)
    assert len(_shard_rows(sd, ".detail.jsonl")) == 2


def test_a_grown_transcript_yields_only_its_new_calls(tmp_path, claude_home):
    sd = tmp_path / "session"
    _harvest(sd, claude_home)
    transcript = next((claude_home / "projects").rglob("*.jsonl"))
    with transcript.open("a") as fh:
        fh.write(json.dumps(_assistant("msg_c")) + "\n")
    grown = _harvest(sd, claude_home)
    assert grown.api_calls == 1
    assert len(_shard_rows(sd, ".detail.jsonl")) == 3


def test_a_transcript_for_another_run_is_not_harvested(tmp_path, claude_home):
    sd = tmp_path / "session"
    result = _harvest(sd, claude_home, exp_root="/shared/runs/some-other-session")
    assert (result.transcripts, result.turns, result.api_calls) == (0, 0, 0)
    assert _shard_rows(sd, ".jsonl") == []


def test_a_transcript_from_another_cwd_is_not_harvested(tmp_path, claude_home):
    sd = tmp_path / "session"
    result = _harvest(sd, claude_home, workflow_dir="/somewhere/else/e2e_workflow")
    assert result.api_calls == 0


def test_not_before_skips_a_transcript_older_than_the_subprocess(tmp_path, claude_home):
    sd = tmp_path / "session"
    result = _harvest(sd, claude_home, not_before=4_000_000_000.0)
    assert result.api_calls == 0
