# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
    line = "export ANTHROPIC_API_KEY=redact_me and GITHUB_TOKEN=redact_me_too"
    out = sr._safe_redact(line)
    assert "ANTHROPIC_API_KEY=[REDACTED]" in out
    assert "GITHUB_TOKEN=[REDACTED]" in out
    assert "redact_me" not in out
    assert "redact_me_too" not in out
    assert sr._safe_redact("plain line") == "plain line"


def test_safe_redact_headers():
    line = "Authorization: Bearer redactable-header-value"
    out = sr._safe_redact(line)
    assert "Authorization: Bearer [REDACTED]" in out
    assert "redactable-header-value" not in out


def test_safe_redact_bare_gateway_token_shapes():
    """ak-/pk- shaped tokens are masked even without their env-var name."""
    line = "curl --header x ak-abc123def456 and pk-lf-98765 and sk-zyx987"
    out = sr._safe_redact(line)
    assert "ak-abc123def456" not in out
    assert "pk-lf-98765" not in out
    assert "sk-zyx987" not in out
    assert out.count("[REDACTED]") == 3


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
    out = build_empty_specialist_done(gap_canonical_id="g1", domain="kernel_agent", reason="no idea", confidence=2.0)
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


def test_append_transcript_redacts_nested_metadata(tmp_path):
    r = _runner()
    r._append_transcript(
        tmp_path,
        1,
        {
            "metadata": {
                "prompt": "OPENAI_API_KEY=redact_me",
                "headers": ["Authorization: Bearer redactable-header-value"],
            }
        },
    )

    text = (tmp_path / "transcript.jsonl").read_text(encoding="utf-8")
    assert "redact_me" not in text
    assert "redactable-header-value" not in text
    assert "[REDACTED]" in text


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


def _finalize(r, tmp_path, payload):
    """Drive ``_finalize`` far enough to inspect the artifact it writes."""
    prep = sr._PreparedRun(
        domain=SimpleNamespace(key="serving_specialist"),
        gap="gap-1",
        workspace=tmp_path,
    )
    ctx = SimpleNamespace(task=SimpleNamespace(task_id="t1", params={}), extra={})
    result = r._finalize(
        ctx=ctx,
        prep=prep,
        specialist_done_payload=payload,
        turns_used=1,
        tool_violations=[],
        backend_error="",
        extra_notes=[],
        patches_written=[],
    )
    return result, json.loads((tmp_path / "specialist_done.json").read_text(encoding="utf-8"))


def test_finalize_strips_forbidden_fields_before_the_critic_can_see_them(tmp_path):
    """The Critic is told to reject a proposal_set carrying self-reported gain
    fields, which costs the round every idea in it. Dropping them makes that
    verdict unreachable; the audit note still records what was there."""
    result, written = _finalize(
        _runner(),
        tmp_path,
        {
            "proposal_set": [{"name": "v1", "reason": "why", "expected_gain": 8.0, "score": 2}],
            "summary": "s",
        },
    )

    proposal = written["proposal_set"][0]
    assert "expected_gain" not in proposal
    assert "score" not in proposal
    assert proposal["name"] == "v1" and proposal["reason"] == "why"
    joined = "\n".join(result.notes)
    assert "patch_safety_forbidden_fields" in joined
    assert "expected_gain" in joined and "score" in joined


def test_finalize_keeps_the_round_level_confidence_the_audit_records(tmp_path):
    _, written = _finalize(
        _runner(),
        tmp_path,
        {"proposal_set": [{"name": "v1"}], "confidence": 0.6},
    )

    assert written["confidence"] == 0.6


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


def test_maybe_setup_worktree_bases_on_the_framework_being_optimised(tmp_path, monkeypatch):
    """A framework specialist must get a worktree of the framework it patches.

    ``framework_source_roots`` is the source-file allowlist, and its order is
    arbitrary with respect to the session: on a pod that ships aiter as a git
    checkout, aiter sorts first. A WorldPlay session then handed its specialist
    an aiter worktree, the specialist authored correct patches against
    ``hyvideo/`` paths that are absent from it, and patch-safety dropped every
    one as ``missing_target`` — leaving an env-only proposal that toggled a
    switch with no code behind it and measured 0.0% five rounds running.
    """
    aiter = tmp_path / "aiter"
    aiter.mkdir()
    (aiter / ".git").mkdir()
    worldplay = tmp_path / "HY-WorldPlay"
    worldplay.mkdir()
    (worldplay / ".git").mkdir()
    monkeypatch.setenv("WORLDPLAY_REPO_PATH", str(worldplay))

    cfg = sr.SpecialistSubprocessConfig(
        framework_source_roots=(str(aiter), str(worldplay)),
    )
    r = _runner(backend_factory=None, subprocess_config=cfg)
    seen: dict = {}

    def _fake_setup(base, worktree_path, branch):
        seen["base"] = base
        return worktree_path, ""

    monkeypatch.setattr(sr, "_setup_worktree", _fake_setup)
    ctx = SimpleNamespace(
        task=SimpleNamespace(
            task_id="t",
            params={"framework": "worldplay", "domain": "framework_rewrite_specialist"},
        )
    )

    _wt, base, err = r._maybe_setup_worktree(ctx, workspace=tmp_path)

    assert err == ""
    assert base == worldplay, f"specialist would patch {seen.get('base')}, not the framework"


def test_maybe_setup_worktree_falls_back_when_the_framework_is_not_a_checkout(tmp_path, monkeypatch):
    """A pip-installed framework must not cost the specialist its isolation."""
    aiter = tmp_path / "aiter"
    aiter.mkdir()
    (aiter / ".git").mkdir()
    monkeypatch.setenv("WORLDPLAY_REPO_PATH", str(tmp_path / "not-a-checkout"))

    cfg = sr.SpecialistSubprocessConfig(framework_source_roots=(str(aiter),))
    r = _runner(backend_factory=None, subprocess_config=cfg)
    monkeypatch.setattr(sr, "_setup_worktree", lambda base, path, branch: (path, ""))
    ctx = SimpleNamespace(task=SimpleNamespace(task_id="t", params={"framework": "worldplay"}))

    _wt, base, err = r._maybe_setup_worktree(ctx, workspace=tmp_path)

    assert err == ""
    assert base == aiter


def test_patch_path_within_bases_accepts_sandbox_paths(tmp_path):
    # Legitimate patch paths stay inside the worktree/workspace.
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
    # Absolute / traversal paths outside the sandbox are refused.
    from pathlib import Path

    worktree = tmp_path / "worktree"
    workspace = tmp_path / "workspace"
    worktree.mkdir()
    workspace.mkdir()
    bases = [worktree, workspace]

    assert sr._patch_path_within_bases(Path("/etc/passwd"), bases) is False
    assert sr._patch_path_within_bases(worktree / ".." / "escape.patch", bases) is False
    assert sr._patch_path_within_bases(tmp_path / "sibling.patch", bases) is False
