"""Tests for ``orchestrator.kb`` — IMPL-CHECKLIST §5.1‒5.12."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.kb import (
    Conflict,
    KBEntry,
    KBError,
    KnowledgeBase,
    _model_family,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _write_entry(path: Path, **kwargs) -> None:
    rec = {
        "category": "test",
        "user_id": "default",
        "model": "x",
        "model_family": "x",
        "action": "y",
        "lesson": "lesson body",
        "tags": [],
        "gain": 0.0,
        "status": "keep",
        "ts": 1700_000_000.0,
    }
    rec.update(kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# _model_family
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model,expected",
    [
        ("deepseek-ai/DeepSeek-V3", "deepseek"),
        ("meta-llama/Llama-3-8b", "llama"),
        ("Qwen/Qwen2-MoE-A2.7B", "qwen"),
        ("microsoft/phi-3-mini", "phi"),
        ("openai/gpt-oss-20b", "gpt-oss"),
        ("moonshot-ai/kimi-plus", "kimi"),
        ("mistralai/Mistral-7B", "mistral"),
        ("mistralai/Mixtral-8x7B", "mixtral"),
        ("", "unknown"),
        ("/srv/models/some-future-thing", "some"),
    ],
)
def test_model_family_classification(model: str, expected: str):
    assert _model_family(model) == expected


# ---------------------------------------------------------------------------
# cold start
# ---------------------------------------------------------------------------
def test_count_entries_empty(tmp_path: Path):
    kb = KnowledgeBase(tmp_path)
    assert kb.count_entries("llama") == 0


def test_count_entries_filters_user_id(tmp_path: Path):
    kb = KnowledgeBase(tmp_path, user_id="alice")
    _write_entry(kb.entries_path, user_id="alice", model_family="llama")
    _write_entry(kb.entries_path, user_id="bob", model_family="llama")
    _write_entry(kb.entries_path, user_id="alice", model_family="qwen")
    assert kb.count_entries("llama") == 1


def test_warm_start_after_first_ingest(tmp_path: Path):
    kb = KnowledgeBase(tmp_path)
    assert kb.is_warm_start_eligible("llama") is False
    kb.ingest("test", "llama-3-8b", "backends", "lesson", [], 5.0, "keep")
    assert kb.is_warm_start_eligible("llama") is True


def test_warm_start_does_not_match_other_family(tmp_path: Path):
    kb = KnowledgeBase(tmp_path)
    kb.ingest("test", "llama-3-8b", "backends", "lesson", [], 5.0, "keep")
    assert kb.is_warm_start_eligible("deepseek") is False


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------
def test_ingest_appends_jsonl(tmp_path: Path):
    kb = KnowledgeBase(tmp_path)
    rec = kb.ingest(
        "model_class_lesson",
        "deepseek-V3",
        "backends",
        "vllm > sglang for MLA",
        ["mla", "moe"],
        9.5,
        "keep",
    )
    assert isinstance(rec, KBEntry)
    rows = kb.entries_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    decoded = json.loads(rows[0])
    assert decoded["model_family"] == "deepseek"
    assert decoded["gain"] == 9.5
    assert decoded["tags"] == ["mla", "moe"]


def test_ingest_two_records(tmp_path: Path):
    kb = KnowledgeBase(tmp_path)
    kb.ingest("c1", "llama-3", "a", "l1", [], 1.0, "keep")
    kb.ingest("c2", "llama-3", "b", "l2", [], 2.0, "fail")
    rows = kb.entries_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# personas
# ---------------------------------------------------------------------------
def test_persona_round_trip(tmp_path: Path):
    kb = KnowledgeBase(tmp_path)
    kb.append_persona("executor", "I learned to always check GPU memory.")
    body = kb.read_persona("executor")
    assert "always check GPU memory" in body
    # multiple appends accumulate
    kb.append_persona("executor", "I will use vllm for dense.")
    body2 = kb.read_persona("executor")
    assert "vllm for dense" in body2
    # empty notes are no-ops
    kb.append_persona("executor", "   ")
    assert kb.read_persona("executor") == body2


def test_read_persona_missing_returns_empty(tmp_path: Path):
    kb = KnowledgeBase(tmp_path)
    assert kb.read_persona("nobody") == ""


# ---------------------------------------------------------------------------
# cross-run synthesis
# ---------------------------------------------------------------------------
def test_cross_run_synthesize_writes_summary(tmp_path: Path):
    kb = KnowledgeBase(tmp_path)
    kb.ingest("c", "llama-3", "a", "l1", [], 4.0, "keep")
    kb.ingest("c", "llama-3", "a", "l2", [], 0.0, "fail")
    kb.ingest("c", "deepseek", "a", "l3", [], 7.0, "keep")
    summary = kb.cross_run_synthesize()
    assert summary["kind"] == "cross_run_synthesis"
    assert summary["samples"] == 3
    fams = summary["by_family"]
    assert fams["llama"]["count"] == 2
    assert fams["llama"]["kept_count"] == 1
    assert fams["deepseek"]["mean_gain"] == pytest.approx(7.0)
    # also persisted to insights.jsonl
    assert kb.insights_path.is_file()


# ---------------------------------------------------------------------------
# conflict detection
# ---------------------------------------------------------------------------
def test_detect_conflicts_flags_keep_vs_revert(tmp_path: Path):
    kb = KnowledgeBase(tmp_path)
    kb.ingest("c", "llama-3", "backends", "vllm wins", [], 9.0, "keep")
    kb.ingest("c", "llama-3-old", "backends", "sglang wins", [], 0.0, "revert")
    cs = kb.detect_conflicts()
    assert cs and isinstance(cs[0], Conflict)
    # mirrored to file
    body = kb.conflicts_path.read_text(encoding="utf-8")
    assert "kb_conflict" in body


def test_detect_conflicts_nothing_when_all_kept(tmp_path: Path):
    kb = KnowledgeBase(tmp_path)
    kb.ingest("c", "llama-3", "backends", "x", [], 1.0, "keep")
    kb.ingest("c", "llama-3", "backends", "y", [], 2.0, "keep")
    assert kb.detect_conflicts() == []


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------
def test_recall_empty_when_cold_start(tmp_path: Path):
    kb = KnowledgeBase(tmp_path)
    assert kb.recall_for_model("llama-3", "executor") == ""


def test_recall_with_stub_kb_query(tmp_path: Path, monkeypatch):
    kb = KnowledgeBase(tmp_path)
    kb.ingest("c", "llama-3", "a", "lesson", [], 0.0, "keep")

    captured: dict = {}

    def fake_check_output(argv, timeout=None, text=True):
        captured["argv"] = argv
        return "- (model_class_lesson, llama-3, ...) lesson\n"

    monkeypatch.setattr(
        "inference_optimizer.orchestrator.kb.subprocess.check_output",
        fake_check_output,
    )
    out = kb.recall_for_model(
        "llama-3", "executor",
        kb_query_argv=["pretend-binary"],  # forces fake path
    )
    assert "lesson" in out
    assert captured["argv"][0] == "pretend-binary"


def test_recall_silent_on_subprocess_failure(tmp_path: Path, monkeypatch):
    kb = KnowledgeBase(tmp_path)
    kb.ingest("c", "llama-3", "a", "lesson", [], 0.0, "keep")

    def fake_check_output(argv, timeout=None, text=True):
        raise FileNotFoundError("no kb_query")

    monkeypatch.setattr(
        "inference_optimizer.orchestrator.kb.subprocess.check_output",
        fake_check_output,
    )
    assert kb.recall_for_model("llama-3", "executor") == ""
