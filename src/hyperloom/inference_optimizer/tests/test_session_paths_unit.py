# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the per-session path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.session import session_paths as sp


SD = Path("/tmp/sess")


def test_top_level_files():
    assert sp.manifest_path(SD) == SD / "manifest.json"
    assert sp.state_path(SD) == SD / "state.json"


def test_runs_root_and_dir():
    assert sp.runs_root(SD) == SD / "runs"
    p = sp.runs_dir(SD, "baseline", "t1")
    assert p == SD / "runs" / "baseline" / "t1"


def test_runs_dir_rejects_unknown_action():
    with pytest.raises(ValueError):
        sp.runs_dir(SD, "definitely-not-an-action", "t1")


@pytest.mark.parametrize(
    "bad_task_id",
    [
        "",
        "   ",
        "../escape",
        "..",
        ".",
        "a/../../etc",
        "sub/dir",
        "/abs/path",
        "x/y",
    ],
)
def test_runs_dir_rejects_bad_task_id(bad_task_id):
    # Blank or path-like ids would relocate the sandbox onto a shared path.
    with pytest.raises(ValueError):
        sp.runs_dir(SD, "baseline", bad_task_id)


def test_runs_dir_accepts_uuid_hex_task_id():
    tid = "0123456789abcdef0123456789abcdef"
    assert sp.runs_dir(SD, "baseline", tid) == SD / "runs" / "baseline" / tid


def test_kernel_agent_runs_dir_rejects_bad_id():
    for bad in ("", "   ", "../x", ".", "a/b", "/abs"):
        with pytest.raises(ValueError):
            sp.kernel_agent_runs_dir(SD, bad)


def test_patches_dir_rejects_bad_id():
    for bad in ("", "   ", "../x", ".", "a/b", "/abs"):
        with pytest.raises(ValueError):
            sp.patches_dir(SD, bad)


def test_validate_action_strips():
    assert sp._validate_action("  baseline  ") == "baseline"


def test_kernel_and_patch_paths():
    assert sp.kernel_agent_runs_dir(SD, "s1") == SD / "kernel-agent" / "runs" / "s1"
    assert sp.patches_dir(SD, "k1") == SD / "patches" / "k1"


def test_reports_dir():
    assert sp.reports_dir(SD) == SD / "reports"


def test_trace_paths():
    assert sp.trace_dir(SD) == SD / "reports" / "trace"
    assert sp.llm_calls_path(SD).name == "llm_calls.jsonl"
    assert sp.decision_trace_path(SD).name == "decision_trace.jsonl"
    assert sp.conversations_path(SD).name == "conversations.jsonl"
    assert sp.proposal_task_map_path(SD).name == "proposal_task_map.jsonl"


def test_enablement_paths():
    assert sp.enablement_dir(SD) == SD / "reports" / "enablement"
    assert sp.enablement_round_dir(SD, "abc123") == SD / "reports" / "enablement" / "abc123"


@pytest.mark.parametrize("task_id", ["../evil", ""])
def test_enablement_round_dir_refuses_an_unusable_id(task_id):
    """A blank id would put every round in one directory; the caller skips those."""
    with pytest.raises(ValueError):
        sp.enablement_round_dir(SD, task_id)


def test_research_and_competitor_paths():
    assert sp.research_hints_md(SD).name == "research_hints.md"
    assert sp.research_hints_json(SD).name == "research_hints.json"
    assert sp.competitor_target_json(SD).name == "competitor_target.json"


def test_agent_paths():
    assert sp.agent_dir(SD, "critic") == SD / "agents" / "critic"
    assert sp.agent_prompt_snapshot(SD, "critic").name == "system_prompt.snapshot.md"
    assert sp.agent_mcp_setup_path(SD, "orchestration").name == "mcp_setup.json"


def test_target_analysis_paths():
    assert sp.target_analysis_dir(SD) == SD / "target_analysis"
    assert sp.target_baseline_json(SD).name == "target_baseline.json"
    assert sp.target_analysis_report_md(SD).name == "target_analysis_report.md"


def test_recipe_kb_paths():
    assert sp.recipe_kb_dir(SD) == SD / "runtime" / "recipe_kb"
    assert sp.recipe_kb_warm_json(SD).name == ".kb_warm.json"
    assert sp.recipe_kb_pitfalls_json(SD).name == ".kb_pitfalls.json"
    assert sp.recipe_kb_pending_ndjson(SD).name == ".kb_pending.ndjson"
    assert sp.recipe_kb_flushed_ndjson(SD).name == ".kb_flushed.ndjson"
    assert sp.recipe_kb_dead_letter_ndjson(SD).name == ".kb_dead_letter.ndjson"
    assert sp.recipe_kb_audit_jsonl(SD).name == ".kb_audit.jsonl"
    assert sp.recipe_kb_flusher_pid(SD).name == ".kb_flusher.pid"
    assert sp.recipe_kb_flusher_status_json(SD).name == ".kb_flusher_status.json"
    assert sp.pr_monitor_status_json(SD).name == ".pr_monitor_status.json"


def test_recipe_snapshot_paths():
    assert sp.recipe_snapshot_dir(SD) == SD / "runtime" / "recipe_snapshot"
    assert sp.recipe_snapshot_audit_jsonl(SD).name == ".audit.jsonl"


def test_fs_safe_id_folds_a_fusion_operator_name():
    # The id that crashed integrate: legal as an id, illegal as a directory.
    assert sp.fs_safe_id("llm:ar_residual_rmsnorm_fp8quant_wiring") == "llm-ar_residual_rmsnorm_fp8quant_wiring"


def test_fs_safe_id_preserves_ordinary_ids():
    # uuid hex / k001 / dotted names round-trip untouched, so existing run dirs
    # keep their names and no in-flight session sees its workspace move.
    for value in ("k001", "abc123", "kernel.name-v2", "9f3c1a2b4d5e"):
        assert sp.fs_safe_id(value) == value


def test_fs_safe_id_collapses_runs_and_trims_edges():
    assert sp.fs_safe_id("a::  ::b") == "a-b"
    assert sp.fs_safe_id(":::lead-and-trail:::") == "lead-and-trail"


def test_fs_safe_id_falls_back_when_nothing_survives():
    assert sp.fs_safe_id("") == "anon"
    assert sp.fs_safe_id(":::") == "anon"
    assert sp.fs_safe_id(None) == "anon"
    assert sp.fs_safe_id(":::", fallback="unnamed") == "unnamed"


def test_fs_safe_id_output_is_always_a_usable_single_segment(tmp_path):
    for raw in ("llm:x", "a/b", "..", "  ", "sp ace", "*glob?"):
        seg = sp.fs_safe_id(raw)
        (tmp_path / seg).mkdir(exist_ok=True)
        assert (tmp_path / seg).is_dir()
        # Never escapes its parent.
        assert (tmp_path / seg).resolve().parent == tmp_path.resolve()
