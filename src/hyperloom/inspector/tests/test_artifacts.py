# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from hyperloom.inspector import artifacts
from hyperloom.inspector.tests.conftest import FAKE_KEY


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("rank0.pt.trace.json.gz", "trace"),
        ("coordinator.db", "binary"),
        ("transcript.jsonl", "jsonl"),
        ("state.json", "json"),
        ("server.log", "log"),
        ("prompt.md", "text"),
        ("baseline_config.with_envs.yaml", "config"),
        ("README", "other"),
    ],
)
def test_classify(name, kind):
    assert artifacts.classify(name) == kind


def test_index_never_lists_env_files(session_dir):
    (session_dir / ".env").write_text("A=1", encoding="utf-8")
    (session_dir / "runs" / ".env.local").write_text("A=1", encoding="utf-8")
    env_dir = session_dir / ".envs"
    env_dir.mkdir()
    (env_dir / "x.txt").write_text("A=1", encoding="utf-8")
    paths = [r["path"] for r in artifacts.index(session_dir, {})]
    assert paths and not any(".env" in p for p in paths)


def test_view_modes(session_dir):
    j = artifacts.view(session_dir, "state.json")
    assert j["mode"] == "json" and isinstance(j["data"], dict)
    (session_dir / "big.log").write_text("x" * 10 + f"\n{FAKE_KEY}\n", encoding="utf-8")
    t = artifacts.view(session_dir, "big.log")
    assert t["mode"] == "text" and FAKE_KEY not in t["text"] and not t["truncated"]
    (session_dir / "r.pt.trace.json.gz").write_bytes(b"\x1f\x8b")
    assert artifacts.view(session_dir, "r.pt.trace.json.gz")["mode"] == "meta"
    (session_dir / "bad.json").write_text("{nope", encoding="utf-8")
    assert artifacts.view(session_dir, "bad.json")["mode"] == "text"


def test_view_jsonl_pages_and_redacts(session_dir):
    lines = [f'{{"n": {i}, "api_key": "k{i}"}}' for i in range(7)] + ["torn"]
    (session_dir / "t.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    page = artifacts.view(session_dir, "t.jsonl", offset=5, limit=10)
    assert page["total"] == 8 and page["offset"] == 5
    assert [r.get("n") for r in page["rows"]] == [5, 6, None]
    assert page["rows"][0]["api_key"] == "<redacted>"
    assert "_unparsed" in page["rows"][2]


def test_view_refuses_escape_and_missing(session_dir):
    with pytest.raises(PermissionError):
        artifacts.view(session_dir, "../x")
    with pytest.raises(FileNotFoundError):
        artifacts.view(session_dir, "nope.txt")


def test_diff_rows_and_identical(session_dir):
    (session_dir / "a.yaml").write_text("tp: 4\nenv:\n  A: '1'\n", encoding="utf-8")
    (session_dir / "b.yaml").write_text("tp: 4\nenv:\n  A: '2'\n  B: x\n", encoding="utf-8")
    out = artifacts.diff(session_dir, "a.yaml", "b.yaml")
    tags = [r[0] for r in out["rows"]]
    assert tags[0] == "hunk" and "replace" in tags and not out["identical"]
    assert ["replace", 3, "  A: '1'", 3, "  A: '2'"] in out["rows"]
    assert any(r[0] == "replace" and r[1] is None and r[4] == "  B: x" for r in out["rows"])
    assert artifacts.diff(session_dir, "a.yaml", "a.yaml")["identical"]


def test_diff_refuses_binary_and_env(session_dir):
    (session_dir / "x.db").write_bytes(b"\0")
    with pytest.raises(ValueError):
        artifacts.diff(session_dir, "x.db", "state.json")
    with pytest.raises(PermissionError):
        artifacts.diff(session_dir, ".env", "state.json")


def test_index_marks_large_subtrees_as_bulk(session_dir, monkeypatch):
    monkeypatch.setattr(artifacts, "BULK_MIN_FILES", 3)
    task = session_dir / "runs" / "specialist" / "tbig"
    for i in range(5):
        (task / "worktree" / "pkg").mkdir(parents=True, exist_ok=True)
        (task / "worktree" / "pkg" / f"m{i}.py").write_text("x = 1\n")
    (task / "prompt.md").write_text("hi\n")
    for i in range(5):
        (session_dir / "reports" / "many").mkdir(parents=True, exist_ok=True)
        (session_dir / "reports" / "many" / f"r{i}.json").write_text("{}")
    by_path = {a["path"]: a for a in artifacts.index(session_dir, {"tbig": "spec:static_recon"})}
    assert by_path["runs/specialist/tbig/worktree/pkg/m0.py"]["bulk"] == "runs/specialist/tbig/worktree"
    assert "bulk" not in by_path["runs/specialist/tbig/prompt.md"]
    # Outside a task the first segment is the root, so reports/many (not reports) is the tree.
    assert by_path["reports/many/r0.json"]["bulk"] == "reports/many"


def test_index_says_when_it_stops_at_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "MAX_FILES", 3)
    for i in range(3):
        (tmp_path / f"f{i}.txt").write_text("x", encoding="utf-8")
    full = artifacts.index(tmp_path, {})
    assert len(full) == 3 and not full.truncated
    (tmp_path / "f3.txt").write_text("x", encoding="utf-8")
    cut = artifacts.index(tmp_path, {})
    assert [a["path"] for a in cut] == ["f0.txt", "f1.txt", "f2.txt"] and cut.truncated
