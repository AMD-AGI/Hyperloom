"""KB priors reader tests (P3 PR-G)."""

from __future__ import annotations

from pathlib import Path

import pytest

from framework_agent.agent.kb_priors import (
    KbEntry,
    read_priors,
    resolve_kb_root,
)


def _seed(name: str, body: str) -> tuple[str, str]:
    return name, body


def _write_kb(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "kb" / "framework_optimization"
    (root / "seeds").mkdir(parents=True)
    for name, content in files.items():
        if name == "_lessons":
            (root / "empirical_kb.md").write_text(content, encoding="utf-8")
        else:
            (root / "seeds" / name).write_text(content, encoding="utf-8")
    return tmp_path / "kb"


# ---------------------------------------------------------------------------
def test_resolve_kb_root_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FRAMEWORK_AGENT_KB_DIR", str(tmp_path))
    assert resolve_kb_root() == tmp_path.resolve()


def test_resolve_kb_root_fa_root_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    (tmp_path / "kb").mkdir()
    monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
    monkeypatch.setenv("FRAMEWORK_AGENT_ROOT", str(tmp_path))
    assert resolve_kb_root() == (tmp_path / "kb").resolve()


def test_resolve_kb_root_returns_none_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
    monkeypatch.setenv("FRAMEWORK_AGENT_ROOT", str(tmp_path / "nonexistent"))
    assert resolve_kb_root() is None


# ---------------------------------------------------------------------------
def test_read_priors_returns_empty_when_no_kb_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("FRAMEWORK_AGENT_KB_DIR", raising=False)
    monkeypatch.delenv("FRAMEWORK_AGENT_ROOT", raising=False)
    assert read_priors("sglang") == []


def test_read_priors_parses_seed_with_framework_and_tags(tmp_path: Path):
    kb_root = _write_kb(tmp_path, {
        "fw-perf-001.md": (
            "# fw-perf-001 vllm chunked prefill split granularity\n"
            "Framework: vllm\n"
            "Tags: prefill, chunked\n"
            "Body line 1\n"
            "Body line 2\n"
        ),
    })
    entries = read_priors("vllm", kb_root=kb_root)
    assert len(entries) == 1
    e = entries[0]
    assert e.entry_id == "fw-perf-001"
    assert e.title == "vllm chunked prefill split granularity"
    assert e.category == "perf"
    assert e.target_framework == "vllm"
    assert e.tags == ("prefill", "chunked")
    assert "Body line 1" in e.body


def test_read_priors_orders_pitfall_then_boundary_then_perf(tmp_path: Path):
    kb_root = _write_kb(tmp_path, {
        "fw-perf-001.md": "# fw-perf-001 perf entry\nFramework: vllm\n",
        "fw-boundary-001.md": "# fw-boundary-001 boundary entry\nFramework: vllm\n",
        "fw-pitfall-001.md": "# fw-pitfall-001 pitfall entry\nFramework: vllm\n",
    })
    entries = read_priors("vllm", kb_root=kb_root)
    cats = [e.category for e in entries]
    assert cats == ["pitfall", "boundary", "perf"]


def test_read_priors_matching_framework_before_cross_framework(
    tmp_path: Path,
):
    kb_root = _write_kb(tmp_path, {
        "fw-perf-001.md": "# fw-perf-001 cross fw entry\n",
        "fw-perf-002.md": "# fw-perf-002 sglang entry\nFramework: sglang\n",
        "fw-perf-003.md": "# fw-perf-003 vllm entry\nFramework: vllm\n",
    })
    entries = read_priors("sglang", kb_root=kb_root)
    # All 3 are 'perf' so secondary sort by framework-match kicks in.
    # sglang entry first, then cross-fw (empty target), then vllm.
    assert entries[0].entry_id == "fw-perf-002"


def test_read_priors_loads_lessons_file(tmp_path: Path):
    kb_root = _write_kb(tmp_path, {
        "_lessons": (
            "# fw-keep-20260520-abcd1234 KEEP: sglang block_manager refactor\n"
            "Framework: sglang\n"
            "Source: session-001\n"
            "Lesson body.\n"
            "\n"
            "# fw-keep-20260520-efgh5678 KEEP: vllm scheduler tweak\n"
            "Framework: vllm\n"
            "Source: session-002\n"
            "Another lesson.\n"
        ),
    })
    entries = read_priors("sglang", kb_root=kb_root)
    lessons = [e for e in entries if e.category == "lesson"]
    assert len(lessons) == 2
    # sglang-targeted lesson ranks above vllm one for sglang target.
    assert lessons[0].target_framework == "sglang"


def test_read_priors_ignores_malformed_seed_file(tmp_path: Path):
    kb_root = _write_kb(tmp_path, {
        "good.md": "# fw-perf-001 good entry\nFramework: vllm\n",
        "bad.md": "no header here, just body lines",
    })
    entries = read_priors("vllm", kb_root=kb_root)
    # Bad file silently dropped; good one parsed.
    assert len(entries) == 1
    assert entries[0].entry_id == "fw-perf-001"
