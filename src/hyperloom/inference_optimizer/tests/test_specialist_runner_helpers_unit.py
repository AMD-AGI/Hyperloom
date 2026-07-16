# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coverage for SpecialistRunner pure helpers + workspace file protocol:
failure classification, empty-done synthesis, redaction, path resolution, and
the prompt/transcript/heartbeat/done writers."""

from __future__ import annotations

import json
from types import SimpleNamespace

from hyperloom.orchestrator.specialists import runner as sr
from hyperloom.orchestrator.specialists.runner import (
    SpecialistFailureType,
    SpecialistRunner,
    build_empty_specialist_done,
    classify_specialist_failure,
)


def _runner(**over):
    kwargs = dict(backend_factory=lambda *a, **k: None)
    kwargs.update(over)
    return SpecialistRunner(**kwargs)


def test_now_iso():
    assert "T" in sr._now_iso()


def test_safe_redact():
    line = "export ANTHROPIC_API_KEY=sk-123 and GITHUB_TOKEN=ghp_x"
    out = sr._safe_redact(line)
    assert "ANTHROPIC_API_KEY[REDACTED]" in out
    assert "GITHUB_TOKEN[REDACTED]" in out
    assert sr._safe_redact("plain line") == "plain line"


def test_extra_focus_tags(monkeypatch):
    monkeypatch.setattr(sr, "normalize_dispatch_tags", lambda params: ["anchor", "extra1", "extra2", ""])
    domain = SimpleNamespace(kb_anchor="anchor")
    tags = sr._extra_focus_tags({"dispatch_tags": []}, domain)
    assert "anchor" not in tags
    assert set(tags) == {"extra1", "extra2"}


def test_classify_succeeded():
    assert classify_specialist_failure("succeeded", "") == (SpecialistFailureType.NONE, False)


def test_classify_tool_violation():
    assert classify_specialist_failure("tool_violation", "x") == (SpecialistFailureType.TOOL_VIOLATION, False)


def test_classify_stale_variants():
    assert classify_specialist_failure("stale", "request timeout")[0] == SpecialistFailureType.TIMEOUT
    assert classify_specialist_failure("stale", "stale_heartbeat 200s")[0] == SpecialistFailureType.STALE_HEARTBEAT
    ftype, retry = classify_specialist_failure("stale", "subprocess_error")
    assert ftype == SpecialistFailureType.CRASH and retry is True


def test_classify_empty_synthesised():
    assert classify_specialist_failure("empty_synthesised", "unknown_domain")[0] == SpecialistFailureType.CONFIG
    assert classify_specialist_failure("empty_synthesised", "no_workspace")[0] == SpecialistFailureType.CONFIG
    assert classify_specialist_failure("empty_synthesised", "ran out")[0] == SpecialistFailureType.NO_OUTPUT


def test_classify_unknown():
    assert classify_specialist_failure("weird_status", "")[0] == SpecialistFailureType.UNKNOWN


def test_build_empty_specialist_done():
    out = build_empty_specialist_done(
        gap_canonical_id="g1", domain="kernel_agent", reason="no idea", confidence=2.0
    )
    assert out["empty"] is True
    assert out["proposal_set"] == []
    assert out["confidence"] == 1.0
    assert out["summary"] == "no idea"


def test_build_empty_specialist_done_default_reason():
    out = build_empty_specialist_done(gap_canonical_id="g", domain="d", reason="")
    assert out["summary"] == "specialist exited empty"


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
    assert r._partial_done_path(tmp_path).name == "specialist_done.partial.json"
    assert r._partial_done_path(None) is None


def test_resolve_workspace_from_extra(tmp_path):
    r = _runner()
    ws = tmp_path / "explicit"
    ctx = SimpleNamespace(extra={"workspace": str(ws)}, task=SimpleNamespace(task_id="t1"))
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


def test_write_helpers_noop_none_workspace():
    r = _runner()
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
    lines = (tmp_path / "transcript.jsonl").read_text(encoding="utf-8").strip().splitlines()
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
    payload = json.loads((tmp_path / "specialist_done.json").read_text(encoding="utf-8"))
    assert payload["empty"] is True and "ts" in payload


def test_write_specialist_done_atomic_leaves_no_tmp(tmp_path):
    # Final write goes through temp + os.replace; no .tmp residue, valid JSON.
    r = _runner()
    r._write_specialist_done(tmp_path, {"empty": True})
    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads((tmp_path / "specialist_done.json").read_text(encoding="utf-8"))["empty"] is True


def test_write_specialist_done_partial(tmp_path):
    # Partial lands at its own path, is flagged, and does not create the final file.
    r = _runner()
    r._write_specialist_done_partial(tmp_path, {"proposal_set": [{"name": "x"}]})
    partial = tmp_path / "specialist_done.partial.json"
    assert partial.exists()
    assert not (tmp_path / "specialist_done.json").exists()
    payload = json.loads(partial.read_text(encoding="utf-8"))
    assert payload["_recovered_from_partial"] is True
    assert payload["proposal_set"] == [{"name": "x"}]
    assert "ts" in payload


def test_write_specialist_done_partial_noop_none_workspace():
    _runner()._write_specialist_done_partial(None, {"a": 1})  # must not raise


def test_write_specialist_done_partial_rewrite_is_atomic(tmp_path):
    r = _runner()
    for i in range(5):
        r._write_specialist_done_partial(tmp_path, {"turns_used": i})
    assert not list(tmp_path.glob("*.tmp"))
    payload = json.loads((tmp_path / "specialist_done.partial.json").read_text(encoding="utf-8"))
    assert payload["turns_used"] == 4


def test_maybe_setup_worktree_in_process_mode(tmp_path):
    r = _runner()  # no subprocess_config
    ctx = SimpleNamespace(task=SimpleNamespace(task_id="t", params={}))
    assert r._maybe_setup_worktree(ctx, workspace=tmp_path) == (None, None, "")


def test_maybe_setup_worktree_readonly(tmp_path):
    cfg = sr.SpecialistSubprocessConfig()
    r = _runner(backend_factory=None, subprocess_config=cfg)
    ctx = SimpleNamespace(task=SimpleNamespace(task_id="t", params={"readonly": True}))
    assert r._maybe_setup_worktree(ctx, workspace=tmp_path) == (None, None, "")


def test_patch_path_within_bases_accepts_sandbox_paths(tmp_path):
    # SWSPLAT-33372: legitimate patch paths inside the worktree/workspace.
    from pathlib import Path

    worktree = tmp_path / "worktree"
    workspace = tmp_path / "workspace"
    worktree.mkdir()
    workspace.mkdir()
    bases = [worktree, workspace]

    inside = worktree / "patches" / "x.patch"
    inside.parent.mkdir(parents=True)
    inside.write_text("diff", encoding="utf-8")
    assert sr._patch_path_within_bases(inside, bases) is True
    assert sr._patch_path_within_bases(workspace / "y.patch", bases) is True


def test_patch_path_within_bases_rejects_outside_paths(tmp_path):
    # SWSPLAT-33372: absolute / traversal paths outside the sandbox are refused.
    from pathlib import Path

    worktree = tmp_path / "worktree"
    workspace = tmp_path / "workspace"
    worktree.mkdir()
    workspace.mkdir()
    bases = [worktree, workspace]

    assert sr._patch_path_within_bases(Path("/etc/passwd"), bases) is False
    assert sr._patch_path_within_bases(worktree / ".." / "escape.patch", bases) is False
    assert sr._patch_path_within_bases(tmp_path / "sibling.patch", bases) is False
