"""Tests for `fa kb <op>` CLI surface.

Hermetic - each test points FRAMEWORK_AGENT_KB_DIR at a tmp_path so
the KB layout under test is local to the test process.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import framework_agent.runtime.cli as cli


@pytest.fixture
def kb_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the KB root to a clean per-test tmp directory."""
    monkeypatch.setenv("FRAMEWORK_AGENT_KB_DIR", str(tmp_path))
    monkeypatch.delenv("FRAMEWORK_AGENT_ROOT", raising=False)
    return tmp_path


def test_kb_list_empty(kb_root: Path, capsys: pytest.CaptureFixture) -> None:
    """fa kb list on a clean KB returns an empty domains array."""
    rc = cli.main(["kb", "list"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["domains"] == []
    assert payload["kb_root"] == str(kb_root)


def test_kb_list_after_contribute(kb_root: Path, capsys: pytest.CaptureFixture) -> None:
    """A successful contribute makes the new domain show up in list."""
    rc = cli.main([
        "kb", "contribute",
        "--domain", "framework",
        "--body", "hello",
        "--source", "test", "--session-id", "s1",
    ])
    assert rc == 0
    capsys.readouterr()  # discard contribute output
    rc = cli.main(["kb", "list"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["domains"] == ["framework"]


def test_kb_show_unknown_domain_exit_two(kb_root: Path, capsys: pytest.CaptureFixture) -> None:
    """fa kb show on a non-existent domain exits 2 with a clear error."""
    rc = cli.main(["kb", "show", "--domain", "nope"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not found" in err


def test_kb_search_returns_hits(kb_root: Path, capsys: pytest.CaptureFixture) -> None:
    """fa kb search returns the file hits whose content matches the query."""
    d = kb_root / "framework"
    d.mkdir()
    (d / "empirical_kb.md").write_text("FlashInfer NVFP4 winner")
    rc = cli.main(["kb", "search", "--query", "flashinfer"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["hits"][0]["domain"] == "framework"


def test_kb_contribute_requires_body(kb_root: Path, capsys: pytest.CaptureFixture) -> None:
    """contribute with neither --body nor --body-file should rc=2."""
    rc = cli.main(["kb", "contribute", "--domain", "framework"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--body" in err


def test_kb_synthesize_pure_python_smoke(kb_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """fa kb synthesize without --with-llm emits a deterministic digest."""
    findings = [
        {
            "title": "winner PR:1",
            "body": "explanation",
            "source": "fa explore --execute",
            "session_id": "s1",
            "candidate_ref": "PR:1",
            "metrics": {"throughput": 1234.5, "throughput_ratio": 1.10},
        }
    ]
    findings_path = tmp_path / "findings.json"
    findings_path.write_text(json.dumps(findings), encoding="utf-8")
    rc = cli.main([
        "kb", "synthesize",
        "--domain", "framework",
        "--findings", str(findings_path),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "## Synthesised findings - framework" in out
    assert "### winner PR:1" in out
    assert "1234.5" in out


def test_kb_synthesize_with_llm_missing_sdk_exit_two(
    kb_root: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--with-llm but claude_agent_sdk absent must surface as rc=2 with hint."""
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    rc = cli.main(["kb", "synthesize", "--domain", "framework", "--with-llm"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "claude_agent_sdk not installed" in err
