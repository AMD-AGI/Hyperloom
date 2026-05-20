"""kb_write KEEP-lesson append tests (P3 PR-I)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from framework_agent.agent.kb_priors import read_priors
from framework_agent.agent.kb_write import append_keep_lesson


def _kb_root(tmp_path: Path) -> Path:
    root = tmp_path / "kb"
    (root / "framework_optimization" / "seeds").mkdir(parents=True)
    return root


def test_append_keep_lesson_writes_block(tmp_path: Path):
    root = _kb_root(tmp_path)
    out = append_keep_lesson(
        framework="sglang",
        patch_id="fw-20260520-deadbeef",
        summary="block_manager refactor",
        rationale="reduces fragmentation at high CONC",
        gain_pct=4.2,
        session_id="session-test",
        kb_root=root,
    )
    assert out is not None
    body = out.read_text(encoding="utf-8")
    assert "KEEP: sglang block_manager refactor" in body
    assert "Framework: sglang" in body
    assert "Patch: fw-20260520-deadbeef" in body
    assert "Gain: 4.20%" in body
    assert "reduces fragmentation at high CONC" in body


def test_append_keep_lesson_appends_multiple_blocks(tmp_path: Path):
    root = _kb_root(tmp_path)
    append_keep_lesson(
        framework="sglang", patch_id="fw-1", summary="a", rationale="a-body",
        gain_pct=5.0, kb_root=root,
    )
    append_keep_lesson(
        framework="vllm", patch_id="fw-2", summary="b", rationale="b-body",
        gain_pct=8.5, kb_root=root,
    )
    body = (root / "framework_optimization" / "empirical_kb.md").read_text()
    # Two distinct entry blocks.
    headers = re.findall(r"^# fw-keep-", body, re.MULTILINE)
    assert len(headers) == 2
    assert "Patch: fw-1" in body
    assert "Patch: fw-2" in body


def test_append_keep_lesson_returns_none_when_kb_root_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """No env, no fa_root -> append silently returns None."""
    monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
    monkeypatch.delenv("FRAMEWORK_AGENT_ROOT", raising=False)
    out = append_keep_lesson(
        framework="sglang", patch_id="fw-x", summary="s",
        rationale="r", gain_pct=1.0,
    )
    assert out is None


def test_appended_lesson_is_readable_by_kb_priors(tmp_path: Path):
    """Round-trip: write a lesson then read_priors picks it up as a
    KbEntry with category='lesson'."""
    root = _kb_root(tmp_path)
    append_keep_lesson(
        framework="sglang", patch_id="fw-roundtrip",
        summary="round-trip check",
        rationale="lesson body",
        gain_pct=3.5,
        kb_root=root,
    )
    priors = read_priors("sglang", kb_root=root)
    lessons = [p for p in priors if p.category == "lesson"]
    assert len(lessons) == 1
    assert "fw-roundtrip" in lessons[0].body or lessons[0].target_framework == "sglang"
