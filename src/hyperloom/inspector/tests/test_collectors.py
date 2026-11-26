# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os

import pytest

from hyperloom.inspector.collectors import (
    ChangedFiles,
    CoordinatorDb,
    JsonlTail,
    first_exception,
    newest_mtime,
    parse_ts,
    read_json,
    resolve_inside,
    specialist_turn,
    tail_file,
)


def test_parse_ts_forms():
    assert parse_ts(12) == 12.0
    assert parse_ts("1970-01-01T00:00:10Z") == 10.0
    assert parse_ts("1970-01-01T00:00:10+00:00") == 10.0
    assert parse_ts("") is None and parse_ts(None) is None and parse_ts("nope") is None


def test_read_json_missing_and_torn(tmp_path):
    assert read_json(tmp_path / "none.json") is None
    (tmp_path / "torn.json").write_text('{"a": ', encoding="utf-8")
    assert read_json(tmp_path / "torn.json") is None
    (tmp_path / "ok.json").write_text('{"a": 1}', encoding="utf-8")
    assert read_json(tmp_path / "ok.json") == {"a": 1}


def test_jsonl_tail_holds_a_torn_line_until_it_completes(tmp_path):
    path = tmp_path / "x.jsonl"
    tail = JsonlTail(path)
    assert tail.read_new() == []
    path.write_bytes(b'{"n": 1}\n{"n": ')
    assert tail.read_new() == [{"n": 1}]
    with path.open("ab") as fh:
        fh.write(b"2}\nnot json\n[1]\n")
    assert tail.read_new() == [{"n": 2}], "invalid lines and non-objects are skipped"
    assert tail.read_new() == []


def test_jsonl_tail_restarts_when_the_file_shrinks(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"n": 1}\n{"n": 2}\n', encoding="utf-8")
    tail = JsonlTail(path)
    assert len(tail.read_new()) == 2
    path.write_text('{"n": 3}\n', encoding="utf-8")
    assert tail.read_new() == [{"n": 3}]


def test_changed_files_reports_only_new_or_modified(tmp_path):
    (tmp_path / "a.json").write_text('{"v": 1}', encoding="utf-8")
    watch = ChangedFiles(tmp_path, "*.json")
    assert [d for _, d in watch.read_changed()] == [{"v": 1}]
    assert watch.read_changed() == []
    (tmp_path / "a.json").write_text('{"v": 2}', encoding="utf-8")
    os.utime(tmp_path / "a.json", (1, 1))
    assert [d for _, d in watch.read_changed()] == [{"v": 2}]


def test_coordinator_db_reads_incrementally(session_dir):
    db = CoordinatorDb(session_dir / "storage" / "coordinator.db")
    first = db.new_events()
    assert first and db.new_events() == []
    assert {t["task_id"] for t in db.tasks()} >= {"t-base", "t-scout"}
    assert isinstance(db.tasks()[0]["history"], list)


def test_coordinator_db_missing_file_is_empty(tmp_path):
    db = CoordinatorDb(tmp_path / "none.db")
    assert db.new_events() == [] and db.tasks() == []


@pytest.mark.parametrize("rel", ["../x", "/etc/passwd", ".env", "runs/.env.local", "runs/../../x"])
def test_resolve_inside_refuses(session_dir, rel):
    assert resolve_inside(session_dir, rel) is None


def test_resolve_inside_refuses_symlinks_out(session_dir, tmp_path):
    (tmp_path / "outside.txt").write_text("x", encoding="utf-8")
    (session_dir / "link.txt").symlink_to(tmp_path / "outside.txt")
    assert resolve_inside(session_dir, "link.txt") is None
    assert resolve_inside(session_dir, "state.json") == (session_dir / "state.json").resolve()


def test_tail_file_reads_the_end_of_a_file(session_dir):
    big = session_dir / "runs" / "big.log"
    big.write_text("first\n" + "y" * 100 + "\nlast\n", encoding="utf-8")
    text = tail_file(big, max_bytes=20)
    assert text.endswith("last\n") and "first" not in text
    assert tail_file(session_dir / "missing.log") == ""


def test_newest_mtime(session_dir, tmp_path):
    assert newest_mtime(tmp_path / "missing") is None
    f = session_dir / "runs" / "late.txt"
    f.write_text("x", encoding="utf-8")
    os.utime(f, (4_000_000_000, 4_000_000_000))
    assert newest_mtime(session_dir) == 4_000_000_000


def test_first_exception_is_the_root_cause_without_the_logger_prefix(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        "(Worker_TP0 pid=1) INFO loading\n"
        "(Worker_TP0 pid=1) ERROR 09-25 15:57:21 [multiproc_executor.py:1053] ValueError: Mixed FP8 tensors\n"
        "(EngineCore pid=2) RuntimeError: Worker failed with error 'Mixed FP8 tensors'\n"
    )
    assert first_exception(log) == "ValueError: Mixed FP8 tensors"
    assert first_exception(log, max_bytes=10) is None
    assert first_exception(tmp_path / "missing.log") is None


def _stream_json(path, lines):
    path.write_text("".join(json.dumps(x) + "\n" for x in lines) + '{"type": "assist')


def test_specialist_turn_groups_an_assistant_message_and_its_tool_results(tmp_path):
    def asst(mid, block):
        return {
            "type": "assistant",
            "message": {"id": mid, "model": "m", "usage": {"output_tokens": 1}, "content": [block]},
        }

    log = tmp_path / "process.log"
    _stream_json(
        log,
        [
            {"type": "system", "subtype": "init"},
            asst("a", {"type": "text", "text": "hello"}),
            asst("b", {"type": "thinking", "thinking": "", "signature": "sig"}),
            asst("b", {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}),
            {"type": "user", "message": {"content": [{"type": "tool_result", "content": "x" * 5000}]}},
            asst("c", {"type": "text", "text": "done"}),
        ],
    )
    turn = specialist_turn(log, 2)
    assert (turn["turn"], turn["model"]) == (2, "m")
    assert turn["response"] == [
        {"type": "thinking", "thinking": ""},
        {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
    ]
    assert turn["tool_results"][0]["content"].endswith("[1000 more chars]")
    assert specialist_turn(log, 3)["response"] == [{"type": "text", "text": "done"}]
    assert specialist_turn(log, 4) is None
    assert specialist_turn(tmp_path / "missing.log", 1) is None
