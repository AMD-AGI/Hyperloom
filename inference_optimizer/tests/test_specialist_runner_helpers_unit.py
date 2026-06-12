# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for SpecialistRunner pure helpers + workspace file protocol:
failure classification, empty-done synthesis, redaction, path resolution, and
the prompt/transcript/heartbeat/done writers (including the no-workspace
no-ops)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from inference_optimizer.orchestrator import specialist_runner as sr
from inference_optimizer.orchestrator.specialist_runner import (
    SpecialistFailureType,
    SpecialistRunResult,
    SpecialistRunner,
    build_empty_specialist_done,
    classify_specialist_failure,
)


def _runner(**over):
    kwargs = dict(backend_factory=lambda *a, **k: None)
    kwargs.update(over)
    return SpecialistRunner(**kwargs)


# ---- _now_iso / _safe_redact ----------------------------------------------
def test_now_iso():
    assert "T" in sr._now_iso()


def test_safe_redact():
    line = "export ANTHROPIC_API_KEY=sk-123 and GITHUB_TOKEN=ghp_x"
    out = sr._safe_redact(line)
    assert "ANTHROPIC_API_KEY[REDACTED]" in out
    assert "GITHUB_TOKEN[REDACTED]" in out
    # no secret -> unchanged
    assert sr._safe_redact("plain line") == "plain line"


# ---- _extra_focus_tags ----------------------------------------------------
def test_extra_focus_tags(monkeypatch):
    monkeypatch.setattr(sr, "normalize_dispatch_tags",
                        lambda params: ["anchor", "extra1", "extra2", ""])
    domain = SimpleNamespace(kb_anchor="anchor")
    tags = sr._extra_focus_tags({"dispatch_tags": []}, domain)
    assert "anchor" not in tags
    assert set(tags) == {"extra1", "extra2"}


# ---- classify_specialist_failure ------------------------------------------
def test_classify_succeeded():
    assert classify_specialist_failure("succeeded", "") == (
        SpecialistFailureType.NONE, False)


def test_classify_tool_violation():
    assert classify_specialist_failure("tool_violation", "x") == (
        SpecialistFailureType.TOOL_VIOLATION, False)


def test_classify_stale_variants():
    assert classify_specialist_failure("stale", "request timeout")[0] == \
        SpecialistFailureType.TIMEOUT
    assert classify_specialist_failure("stale", "stale_heartbeat 200s")[0] == \
        SpecialistFailureType.STALE_HEARTBEAT
    ftype, retry = classify_specialist_failure("stale", "subprocess_error")
    assert ftype == SpecialistFailureType.CRASH and retry is True


def test_classify_empty_synthesised():
    assert classify_specialist_failure("empty_synthesised", "unknown_domain")[0] == \
        SpecialistFailureType.CONFIG
    assert classify_specialist_failure("empty_synthesised", "no_workspace")[0] == \
        SpecialistFailureType.CONFIG
    assert classify_specialist_failure("empty_synthesised", "ran out")[0] == \
        SpecialistFailureType.NO_OUTPUT


def test_classify_unknown():
    assert classify_specialist_failure("weird_status", "")[0] == \
        SpecialistFailureType.UNKNOWN


# ---- build_empty_specialist_done ------------------------------------------
def test_build_empty_specialist_done():
    out = build_empty_specialist_done(
        gap_canonical_id="g1", domain="kernel", reason="no idea",
        confidence=2.0)  # clamped to 1.0
    assert out["empty"] is True
    assert out["proposal_set"] == []
    assert out["confidence"] == 1.0
    assert out["summary"] == "no idea"


def test_build_empty_specialist_done_default_reason():
    out = build_empty_specialist_done(
        gap_canonical_id="g", domain="d", reason="")
    assert out["summary"] == "specialist exited empty"


# ---- to_sub_agent_result --------------------------------------------------
def _run_result(**over):
    kwargs = dict(task_id="t1", domain="kernel", gap_canonical_id="g1",
                  status="succeeded", specialist_done={})
    kwargs.update(over)
    return SpecialistRunResult(**kwargs)


def test_to_sub_agent_result_succeeded():
    out = SpecialistRunner.to_sub_agent_result(
        _run_result(task_id="t1", status="empty_synthesised"))
    assert out.state == "succeeded"
    assert out.task_id == "t1"


def test_to_sub_agent_result_failed():
    out = SpecialistRunner.to_sub_agent_result(
        _run_result(task_id="t2", status="stale", error="boom"))
    assert out.state == "failed"
    assert out.error == "boom"


# ---- path helpers ---------------------------------------------------------
def test_path_helpers_none_workspace():
    r = _runner()
    assert r._prompt_path(None) is None
    assert r._transcript_path(None) is None
    assert r._heartbeat_path(None) is None
    assert r._done_path(None) is None


def test_path_helpers_with_workspace(tmp_path):
    r = _runner()
    assert r._prompt_path(tmp_path).name == "prompt.md"
    assert r._transcript_path(tmp_path).name == "transcript.jsonl"
    assert r._heartbeat_path(tmp_path).name == "heartbeat.json"
    assert r._done_path(tmp_path).name == "specialist_done.json"


# ---- _resolve_workspace ---------------------------------------------------
def test_resolve_workspace_from_extra(tmp_path):
    r = _runner()
    ws = tmp_path / "explicit"
    ctx = SimpleNamespace(extra={"workspace": str(ws)},
                          task=SimpleNamespace(task_id="t1"))
    out = r._resolve_workspace(ctx)
    assert out == ws and out.exists()


def test_resolve_workspace_no_session_dir():
    r = _runner()
    ctx = SimpleNamespace(extra={}, task=SimpleNamespace(task_id="t1"))
    assert r._resolve_workspace(ctx) is None


def test_resolve_workspace_session_dir(tmp_path):
    r = _runner(session_dir=tmp_path)
    ctx = SimpleNamespace(extra=None, task=SimpleNamespace(task_id="task-9"))
    out = r._resolve_workspace(ctx)
    assert out is not None and out.exists()
    assert "task-9" in str(out)


# ---- write helpers (no-op + real) -----------------------------------------
def test_write_helpers_noop_none_workspace():
    r = _runner()
    # all must no-op without raising when workspace is None
    r._write_prompt(None, "sys", "user")
    r._append_transcript(None, 0, {"k": "v"})
    r._write_heartbeat(None, turn=0, max_turns=1, status="x")
    r._write_specialist_done(None, {"a": 1})


def test_write_prompt(tmp_path):
    r = _runner()
    r._write_prompt(tmp_path, "SYS", "USER")
    text = (tmp_path / "prompt.md").read_text(encoding="utf-8")
    assert "SYS" in text and "USER" in text


def test_append_transcript(tmp_path):
    r = _runner()
    r._append_transcript(tmp_path, 1, {"event": "turn"})
    r._append_transcript(tmp_path, 2, {"event": "turn2"})
    lines = (tmp_path / "transcript.jsonl").read_text(
        encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["turn"] == 1


def test_write_heartbeat(tmp_path):
    r = _runner()
    r._write_heartbeat(tmp_path, turn=3, max_turns=10, status="working")
    payload = json.loads((tmp_path / "heartbeat.json").read_text(encoding="utf-8"))
    assert payload["turn"] == 3 and payload["status"] == "working"


def test_write_specialist_done(tmp_path):
    r = _runner()
    r._write_specialist_done(tmp_path, {"empty": True})
    payload = json.loads(
        (tmp_path / "specialist_done.json").read_text(encoding="utf-8"))
    assert payload["empty"] is True and "ts" in payload


# ---- _maybe_setup_worktree ------------------------------------------------
def test_maybe_setup_worktree_in_process_mode(tmp_path):
    r = _runner()  # no subprocess_config
    ctx = SimpleNamespace(task=SimpleNamespace(task_id="t", params={}))
    assert r._maybe_setup_worktree(ctx, workspace=tmp_path) == (None, None, "")


def test_maybe_setup_worktree_readonly(tmp_path):
    cfg = sr.SpecialistSubprocessConfig()
    r = _runner(backend_factory=None, subprocess_config=cfg)
    ctx = SimpleNamespace(task=SimpleNamespace(
        task_id="t", params={"readonly": True}))
    assert r._maybe_setup_worktree(ctx, workspace=tmp_path) == (None, None, "")
