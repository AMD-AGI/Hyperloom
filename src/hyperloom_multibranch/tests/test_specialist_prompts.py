# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json

import pytest

from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom.orchestrator.prompts.specialist_prompt_builder import BASH_KILL_SAFETY_PREAMBLE
from hyperloom_multibranch.contracts.branch_log import BranchLogEntry, BranchLogType
from hyperloom_multibranch.contracts.fork import RepairRequest
from hyperloom_multibranch.contracts.intent import Intent, IntentLayer
from hyperloom_multibranch.contracts.judgement import Judgement, Verdict
from hyperloom_multibranch.contracts.notebook.records import BranchLogRecord
from hyperloom_multibranch.specialist import prompts
from hyperloom_multibranch.specialist.prompts import (
    ENTRY_CHARS,
    RECENT_ENTRIES,
    VERSION_HEX,
    launch_prompt,
    prompt_versions,
    role_prompt,
)
from hyperloom_multibranch.specialist.runtime import SessionStart

WORKLOAD = "qwen3-8b on vllm, TP 1, ISL 1024, OSL 1024, concurrency 64, InferenceX"
TOOLS = (
    "notebook_branch_log",
    "notebook_claim",
    "start_analysis",
    "worktree_sync",
    "worktree_check",
    "worktree_patch",
    "submit_test",
    "poll_inbox",
    "test_result",
    "notebook_append_branch_log",
    "notebook_learn",
    "request_repair",
    "decline_repair",
)


def _record(n: int, content: str = "", request_ids: tuple[str, ...] = ()) -> BranchLogRecord:
    entry = BranchLogEntry("s1", BranchLogType.RESULT, content or f"entry {n}", request_ids)
    return BranchLogRecord(f"e{n}", float(n), entry)


def _launch(*, brief: str = "", **overrides) -> str:
    fields = {"head": "c7", "worktree": "/nfs/s1/worktree", "recent": ()}
    return launch_prompt(SessionStart("s1", "c0", brief, 3), **{**fields, **overrides})


def test_the_role_names_the_workload_every_step_s_tool_and_the_rules():
    prompt = role_prompt(WORKLOAD.replace(", ", ",\n  "))

    assert f"The session optimizes one workload: {WORKLOAD}." in prompt
    assert all(tool in prompt for tool in TOOLS)
    assert 'Never submit purpose "confirm"' in prompt
    assert "target_share_pct" in prompt and "w/(1-w)" in prompt
    assert BASH_KILL_SAFETY_PREAMBLE in prompt
    assert "A fixable defect" in prompt and "An inherent trade-off" in prompt
    assert prompt.index("notebook_learn, citing the candidate") < prompt.index("then call decline_repair")
    with pytest.raises(ValueError, match="names the workload"):
        role_prompt(" \n")


def test_a_launch_opens_with_the_brief_then_says_where_the_branch_stands():
    prompt = _launch(brief="leads: fp8 KV cache measured +4% on this model")

    assert prompt.startswith("Startup brief:\nleads: fp8 KV cache")
    facts = "Your branch: s1, started at c0. Its head: c7. Tests you may have running at once: 3."
    assert prompt.index(facts) > prompt.index("leads")
    assert "first session on your branch" in prompt


def test_every_launch_of_a_repair_branch_opens_with_the_candidate_s_change_and_its_diagnosis():
    blocked = Judgement(Verdict.BLOCKED, 3.25, 2.0, ("s0-4", "s0-5"), {"output_throughput": 1030.0}, False)
    kv = Intent("s0", "c3", "kv cache", "fp8 kv cache", IntentLayer.CONFIG)
    change = RecipeDelta(config=ConfigDelta(server_args="--kv-cache-dtype fp8"))
    repair = RepairRequest(kv, change, blocked, "per-tensor KV scales; scale per head\n")
    start = SessionStart("s1", "c3", "leads: fp8 KV cache measured +4% on this model", 3, repair)

    prompt = launch_prompt(start, head="c3", worktree="/nfs/s1/worktree", recent=(_record(1),))

    assert prompt.startswith(
        "Your branch repairs s0-4, a blocked candidate of s0: on c3 it measured +3.25% and was held back by the "
        "accuracy gate. Your branch starts at that commit. Its intent: kv cache / fp8 kv cache (config)."
    )
    assert json.dumps(change.to_document(), sort_keys=True) in prompt
    assert "the root cause and the repair:\nper-tensor KV scales; scale per head\nStart from that change" in prompt
    assert prompt.index("fp8 kv cache (config)") < prompt.index("Startup brief:") < prompt.index("Your branch: s1,")
    assert "Your branch repairs" not in _launch(brief="leads: none")


def test_each_prompt_is_versioned_by_its_own_text(monkeypatch):
    versions = prompt_versions()

    assert set(versions) == {"specialist_role", "specialist_launch"}
    assert all(len(version) == VERSION_HEX and int(version, 16) >= 0 for version in versions.values())
    monkeypatch.setattr(prompts, "_ROLE", prompts._ROLE + "- One more rule.\n")
    edited = prompt_versions()
    assert edited["specialist_role"] != versions["specialist_role"]
    assert edited["specialist_launch"] == versions["specialist_launch"]


def test_a_launch_shows_the_latest_entries_cut_to_size_and_how_the_last_session_ended():
    entries = [_record(n) for n in range(RECENT_ENTRIES + 4)] + [_record(99, "x" * 1000, ("s1-17",))]

    prompt = _launch(recent=entries, last_end="the agent CLI exited 1")

    assert prompt.startswith("Your branch: s1,")
    assert "[e4] result: entry 4" not in prompt and "[e5] result: entry 5" in prompt
    assert f"[e99] result (s1-17): {'x' * (ENTRY_CHARS - 3)}...\n" in prompt
    assert "Your previous session ended: the agent CLI exited 1." in prompt
