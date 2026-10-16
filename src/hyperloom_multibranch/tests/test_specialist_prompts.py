# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest

from hyperloom.orchestrator.prompts.specialist_prompt_builder import BASH_KILL_SAFETY_PREAMBLE
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.notebook.records import BranchLogRecord
from hyperloom_multibranch.specialist.prompts import ENTRY_CHARS, RECENT_ENTRIES, launch_prompt, role_prompt

WORKLOAD = "qwen3-8b on vllm, TP 1, ISL 1024, OSL 1024, concurrency 64, InferenceX"
TOOLS = (
    "notebook_branch_log",
    "notebook_claim",
    "worktree_sync",
    "worktree_patch",
    "submit_test",
    "poll_inbox",
    "test_result",
    "notebook_append_branch_log",
    "notebook_learn",
)


def _record(n: int, content: str = "", request_ids: tuple[str, ...] = ()) -> BranchLogRecord:
    entry = BranchLogEntry("s1", BranchLogType.RESULT, content or f"entry {n}", request_ids)
    return BranchLogRecord(f"e{n}", float(n), entry)


def _launch(**overrides) -> str:
    fields = {"branch": "s1", "head": "c7", "k": 3, "worktree": "/nfs/s1/worktree", "brief": "", "recent": ()}
    return launch_prompt(**{**fields, **overrides})


def test_the_role_names_the_workload_every_step_s_tool_and_the_rules():
    prompt = role_prompt(WORKLOAD.replace(", ", ",\n  "))

    assert f"The session optimizes one workload: {WORKLOAD}." in prompt
    assert all(tool in prompt for tool in TOOLS)
    assert 'Never submit purpose "confirm"' in prompt
    assert BASH_KILL_SAFETY_PREAMBLE in prompt
    with pytest.raises(ValueError, match="names the workload"):
        role_prompt(" \n")


def test_a_launch_opens_with_the_brief_then_says_where_the_branch_stands():
    prompt = _launch(brief="leads: fp8 KV cache measured +4% on this model")

    assert prompt.startswith("Startup brief:\nleads: fp8 KV cache")
    assert prompt.index("Your branch: s1. Its head: c7. Tests you may have running at once: 3.") > prompt.index("leads")
    assert "first session on your branch" in prompt


def test_a_launch_shows_the_latest_entries_cut_to_size_and_how_the_last_session_ended():
    entries = [_record(n) for n in range(RECENT_ENTRIES + 4)] + [_record(99, "x" * 1000, ("s1-17",))]

    prompt = _launch(recent=entries, last_end="the agent CLI exited 1")

    assert prompt.startswith("Your branch: s1.")
    assert "[e4] result: entry 4" not in prompt and "[e5] result: entry 5" in prompt
    assert f"[e99] result (s1-17): {'x' * (ENTRY_CHARS - 3)}...\n" in prompt
    assert "Your previous session ended: the agent CLI exited 1." in prompt
