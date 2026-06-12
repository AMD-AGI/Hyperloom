# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the per-session path helpers (single source of truth for
every path inside a session directory)."""
from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer import session_paths as sp


SD = Path("/tmp/sess")


def test_top_level_files():
    assert sp.manifest_path(SD) == SD / "manifest.json"
    assert sp.state_path(SD) == SD / "state.json"


def test_runs_root_and_dir():
    assert sp.runs_root(SD) == SD / "runs"
    # baseline is in the runs-workspace action set (fallback or registry)
    p = sp.runs_dir(SD, "baseline", "t1")
    assert p == SD / "runs" / "baseline" / "t1"
    # blank task_id falls back to "unknown"
    assert sp.runs_dir(SD, "baseline", "").name == "unknown"


def test_runs_dir_rejects_unknown_action():
    with pytest.raises(ValueError):
        sp.runs_dir(SD, "definitely-not-an-action", "t1")


def test_validate_action_strips():
    assert sp._validate_action("  baseline  ") == "baseline"


def test_kernel_and_patch_paths():
    assert sp.kernel_workspace(SD, "k1") == SD / "kernel-agent-workspace" / "k1"
    assert sp.kernel_workspace(SD, "").name == "unknown"
    assert sp.kernel_agent_runs_dir(SD, "s1") == SD / "kernel-agent" / "runs" / "s1"
    assert sp.kernel_agent_runs_dir(SD, "").name == "unknown"
    assert sp.patches_dir(SD, "k1") == SD / "patches" / "k1"
    assert sp.patches_dir(SD, "").name == "unknown"


def test_reports_and_report_file():
    assert sp.reports_dir(SD) == SD / "reports"
    assert sp.report_file(SD, "20260101") == SD / "reports" / "20260101_final.md"
    assert sp.report_file(SD, "20260101", "json").name == "20260101_final.json"


def test_trace_paths():
    assert sp.trace_dir(SD) == SD / "reports" / "trace"
    assert sp.llm_calls_path(SD).name == "llm_calls.jsonl"
    assert sp.trace_ext_dir(SD) == SD / "reports" / "trace" / "ext"
    assert sp.ext_trace_path(SD, "geak", 1234).name == "geak-1234.jsonl"
    assert sp.ext_trace_path(SD, "", 1).name == "unknown-1.jsonl"
    assert sp.decision_trace_path(SD).name == "decision_trace.jsonl"
    assert sp.conversations_path(SD).name == "conversations.jsonl"


def test_research_and_competitor_paths():
    assert sp.research_hints_md(SD).name == "research_hints.md"
    assert sp.research_hints_json(SD).name == "research_hints.json"
    assert sp.competitor_target_json(SD).name == "competitor_target.json"


def test_logs_and_agent_paths():
    assert sp.logs_dir(SD) == SD / "logs"
    assert sp.agent_log(SD, "orchestration").name == "orchestration.log"
    assert sp.agent_dir(SD, "critic") == SD / "agents" / "critic"
    assert sp.agent_inbox(SD, "critic").name == "inbox.jsonl"
    assert sp.agent_outbox(SD, "critic").name == "outbox.jsonl"
    assert sp.agent_persona(SD, "critic").name == "persona.md"
    assert sp.agent_prompt_snapshot(SD, "critic").name == "system_prompt.snapshot.md"


def test_optimizer_run_paths():
    assert sp.optimizer_runs_dir(SD) == SD / "optimizer_runs"
    assert sp.optimizer_run_log(SD, "tag").name == "run_tag.log"
    assert sp.optimizer_run_log(SD, "").name == "run_unknown.log"
    assert sp.optimizer_run_pidfile(SD, "tag").name == "run_tag.pid"
    assert sp.optimizer_run_pidfile(SD, "").name == "run_unknown.pid"


def test_target_analysis_paths():
    assert sp.target_analysis_dir(SD) == SD / "target_analysis"
    assert sp.target_baseline_json(SD).name == "target_baseline.json"
    assert sp.target_analysis_report_md(SD).name == "target_analysis_report.md"


def test_cortex_paths():
    assert sp.cortex_dir(SD) == SD / "runtime" / "cortex"
    assert sp.cortex_sid_file(SD).name == ".kb_sid"
    assert sp.cortex_warm_json(SD).name == ".kb_warm.json"
    assert sp.cortex_pitfalls_json(SD).name == ".kb_pitfalls.json"
    assert sp.cortex_pending_ndjson(SD).name == ".kb_pending.ndjson"
    assert sp.cortex_flushed_ndjson(SD).name == ".kb_flushed.ndjson"
    assert sp.cortex_dead_letter_ndjson(SD).name == ".kb_dead_letter.ndjson"
    assert sp.cortex_audit_jsonl(SD).name == ".kb_audit.jsonl"
    assert sp.cortex_flusher_pid(SD).name == ".kb_flusher.pid"
    assert sp.cortex_flusher_status_json(SD).name == ".kb_flusher_status.json"
    assert sp.pr_monitor_status_json(SD).name == ".pr_monitor_status.json"


def test_recipe_snapshot_paths():
    assert sp.recipe_snapshot_dir(SD) == SD / "runtime" / "recipe_snapshot"
    assert sp.recipe_snapshot_audit_jsonl(SD).name == ".audit.jsonl"
