"""Tests for the KB CLI scripts — IMPL-CHECKLIST §5.26‒5.30."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_optimizer.kb import kb_ingest, kb_query
from inference_optimizer.kb.kb_query import score_record, tokenize


# ---------------------------------------------------------------------------
# tokenize / scoring
# ---------------------------------------------------------------------------
def test_tokenize_lowercases_and_strips_symbols():
    assert tokenize("FOO bar 123!?") == ["foo", "bar", "123"]


def test_tokenize_empty():
    assert tokenize("") == []


def test_score_zero_when_nothing_matches():
    rec = {"category": "x", "model": "x", "lesson": "z"}
    assert score_record(["foobar"], rec) == 0


def test_score_keep_status_boost():
    rec_keep = {
        "category": "lesson", "model": "llama", "lesson": "vllm",
        "status": "keep", "gain": 9.0,
    }
    rec_revert = dict(rec_keep, status="revert")
    s_keep = score_record(["llama", "vllm"], rec_keep)
    s_revert = score_record(["llama", "vllm"], rec_revert)
    assert s_keep > s_revert


# ---------------------------------------------------------------------------
# end-to-end ingest -> query
# ---------------------------------------------------------------------------
def _ingest(tmp_path: Path, **kw) -> int:
    args = [
        "--kb-dir", str(tmp_path / "kb"),
        "--category", kw.pop("category", "test"),
        "--model", kw.pop("model", "llama-3-8b"),
        "--action", kw.pop("action", "backends"),
        "--lesson", kw.pop("lesson", "vllm beats sglang on dense"),
        "--gain", str(kw.pop("gain", 9.0)),
        "--status", kw.pop("status", "keep"),
    ]
    if "tags" in kw:
        args += ["--tags", kw.pop("tags")]
    if kw:
        raise AssertionError(f"unused kwargs: {kw}")
    return kb_ingest.main(args)


def test_ingest_creates_jsonl(tmp_path: Path):
    rc = _ingest(tmp_path)
    assert rc == 0
    out = tmp_path / "kb" / "entries.jsonl"
    assert out.is_file()
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert rec["model_family"] == "llama"
    assert rec["status"] == "keep"


def test_ingest_supports_json_tags(tmp_path: Path):
    _ingest(tmp_path, tags='["dense","backend_choice"]')
    out = tmp_path / "kb" / "entries.jsonl"
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert rec["tags"] == ["dense", "backend_choice"]


def test_ingest_supports_csv_tags(tmp_path: Path):
    _ingest(tmp_path, tags="dense, backend_choice")
    out = tmp_path / "kb" / "entries.jsonl"
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert rec["tags"] == ["dense", "backend_choice"]


def test_query_returns_top_k(tmp_path: Path, capsys):
    _ingest(tmp_path, model="llama-3", lesson="vllm wins for dense")
    _ingest(tmp_path, model="deepseek", lesson="sglang wins for MLA",
            tags="moe,mla")
    _ingest(
        tmp_path, model="qwen-moe", lesson="params --max-running 256 +5%",
        action="params",
    )

    rc = kb_query.main(
        [
            "vllm dense",
            "--kb-dir", str(tmp_path / "kb"),
            "--top-k", "2",
            "--compact",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "vllm wins for dense" in out


def test_query_json_output(tmp_path: Path, capsys):
    _ingest(tmp_path, model="llama-3", lesson="hello world")
    capsys.readouterr()  # drop the ingest banner
    rc = kb_query.main(
        [
            "hello",
            "--kb-dir", str(tmp_path / "kb"),
            "--top-k", "5",
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out.strip()
    # the JSON line is the last (and only) line printed by kb_query
    data = json.loads(out)
    assert isinstance(data, list)
    assert any("hello world" in str(r.get("lesson")) for r in data)


def test_query_no_matches(tmp_path: Path, capsys):
    (tmp_path / "kb").mkdir(parents=True, exist_ok=True)
    rc = kb_query.main(
        [
            "nothing",
            "--kb-dir", str(tmp_path / "kb"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "(no matches)" in out


def test_query_empty_query(tmp_path: Path, capsys):
    (tmp_path / "kb").mkdir(parents=True, exist_ok=True)
    rc = kb_query.main(
        [
            "...",
            "--kb-dir", str(tmp_path / "kb"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "empty query" in out
