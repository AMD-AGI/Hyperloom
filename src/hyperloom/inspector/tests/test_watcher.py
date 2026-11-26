# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import time

from hyperloom.inspector import artifacts, session_layout
from hyperloom.inspector.tests.conftest import FAKE_KEY, FAKE_SECRET, T0, iso
from hyperloom.inspector.watcher import Stream, Watcher


def _live(session_dir):
    w = Watcher(session_dir)
    w.poll()
    return w


def test_crashed_specialists_that_landed_as_succeeded_are_flagged(session_dir):
    tasks = {t["task_id"]: t for t in _live(session_dir).snapshot()["tasks"]}
    assert tasks["t-scout"]["state"] == "succeeded"
    assert tasks["t-scout"]["health"] == "crashed"
    assert tasks["t-recon"]["health"] == "crashed"
    assert tasks["t-perf"]["health"] == "ok"
    assert tasks["t-roof"]["health"] in ("running", "stalled")
    assert tasks["t-recon"]["node"] == "spec:static_recon"


def test_snapshot_budget(session_dir):
    snap = _live(session_dir).snapshot()
    assert snap["budget"]["pct"] == 25.0
    assert (snap["summary"]["tick"], snap["budget"]["used_sec"]) == (42, 1800)
    assert snap["llm"]["orchestration"]["errors"] == 1


def test_snapshot_says_where_the_artifact_list_is_cut(session_dir, monkeypatch):
    assert _live(session_dir).snapshot()["artifacts_cut_at"] is None
    monkeypatch.setattr(artifacts, "MAX_FILES", 2)
    snap = _live(session_dir).snapshot()
    assert (snap["artifact_count"], snap["artifacts_cut_at"]) == (2, 2)


def test_torn_jsonl_line_is_held_back(session_dir):
    snap = _live(session_dir).snapshot()
    assert [c["call_id"] for c in snap["streams"]["llm_call"]] == ["c1", "c2", "c3"]


def test_snapshot_is_redacted_and_never_lists_env_files(session_dir):
    w = _live(session_dir)
    blob = json.dumps(w.snapshot())
    assert FAKE_KEY not in blob
    assert not any(a["path"].startswith(".env") for a in w.artifacts_at())


def test_an_optimizer_on_this_host_is_alive_while_its_process_runs(session_dir):
    lock = {"hostname": socket.gethostname(), "pid": os.getpid()}
    (session_dir / "runtime" / "optimizer.lock").write_text(json.dumps(lock), encoding="utf-8")
    assert _live(session_dir).optimizer_status() == {"alive": True, "pid": os.getpid()}


def test_stream_keeps_time_order_for_late_items():
    s = Stream()
    s.add({"n": 2}, 2.0)
    s.add({"n": 1}, 1.0)
    s.add({"n": 3}, None)
    assert [i["n"] for i in s.items] == [1, 2, 3]
    assert s.released(1.5) == 1


def test_conversation_lookup_is_redacted(session_dir):
    w = _live(session_dir)
    rec = w.conversation("c1")
    assert rec["response"] == "ok"
    assert FAKE_KEY not in rec["prompt"]
    assert w.conversation("missing") is None


def test_conversation_without_call_id_matches_task_and_turn(session_dir):
    w = _live(session_dir)
    assert w.conversation(task_id="spec1", turn="1")["response"] == "spec"
    assert w.conversation(task_id="spec1")["response"] == "spec"
    assert w.conversation(task_id="spec1", turn="2") is None
    assert w.conversation() is None


def test_trimmed_stream_keeps_absolute_cursors():
    s = Stream()
    for i in range(5):
        s.add({"n": i}, float(i))
    s.trim(3)
    assert s.dropped == 2 and [i["n"] for i in s.items] == [2, 3, 4]
    assert s.released(10.0) == 5
    # A client holding cursor 4 gets exactly the one item it hasn't seen.
    assert [i["n"] for i in s.between(4, 5)] == [4]
    s.add({"n": 5}, 5.0)
    s.trim(3)
    assert s.released(10.0) == 6 and [i["n"] for i in s.between(5, 6)] == [5]
    assert [i["n"] for i in s.upto(4)] == [3]


def test_delta_sends_each_released_item_once(session_dir):
    w = _live(session_dir)
    cursors: dict[str, int] = {}
    first = w.delta(cursors)["items"]
    assert first["event"] and cursors["event"] == len(first["event"])
    assert w.delta(cursors)["items"] == {}
    w.streams["event"].add({"type": "late"}, time.time())
    assert [e["type"] for e in w.delta(cursors)["items"]["event"]] == ["late"]


def test_best_throughput_falls_back_to_results_when_state_has_none(session_dir):
    state = json.loads((session_dir / "state.json").read_text())
    state["current_best"] = {"action": "explore", "hot_tput": None}
    (session_dir / "state.json").write_text(json.dumps(state))
    best = _live(session_dir).snapshot()["summary"]["current_best"]
    assert best["hot_tput"] == 1685.16 and best["source"] == "journal"


def test_running_task_shows_its_step_and_failed_variants_their_root_cause(session_dir):
    notes = [
        {"progress": {"unit": "variant", "label": "k7", "status": "failed"}, "ts": iso(T0 + 720)},
        {"progress": {"unit": "variant_step", "label": "q4:benchmark", "status": "started"}, "ts": iso(T0 + 730)},
    ]
    db = sqlite3.connect(session_dir / "storage" / "coordinator.db")
    (history,) = db.execute("SELECT history FROM tasks WHERE task_id='t-roof'").fetchone()
    db.execute("UPDATE tasks SET history=? WHERE task_id='t-roof'", (json.dumps(json.loads(history) + notes),))
    db.commit()
    db.close()
    log = session_dir / "runs" / "roofline" / "t-roof" / "v00_k7" / "warmup_round" / "server.log"
    log.parent.mkdir(parents=True)
    log.write_text("ERROR [x.py:1] ValueError: Mixed FP8 tensors\nRuntimeError: wrapped\n")
    tasks = {t["task_id"]: t for t in _live(session_dir).snapshot()["tasks"]}
    progress = tasks["t-roof"]["progress"]
    assert progress["current"]["label"] == "q4:benchmark"
    assert progress["units"][0]["error"] == "ValueError: Mixed FP8 tensors"


def test_specialist_turns_without_a_conversation_come_from_process_log(session_dir):
    log = session_dir / "runs" / "specialist" / "t-scout" / "process.log"
    msg = {"id": "m1", "content": [{"type": "text", "text": f"key {FAKE_KEY}"}]}
    log.write_text(json.dumps({"type": "assistant", "message": msg}) + "\n")
    w = _live(session_dir)
    rec = w.conversation(task_id="t-scout", turn="1")
    assert rec["source"] == "runs/specialist/t-scout/process.log"
    assert rec["response"][0]["type"] == "text" and FAKE_KEY not in json.dumps(rec)
    assert w.conversation(task_id="t-scout", turn="2") is None
    assert w.conversation(task_id="../t-scout", turn="1") is None


def test_snapshot_masks_what_hyperloom_masks(session_dir):
    calls = session_dir / "reports" / "trace" / "llm_calls.jsonl"
    text = calls.read_text(encoding="utf-8")
    torn = text.rindex("{")
    secret = FAKE_SECRET
    leak = {"call_id": "c9", "ts": iso(T0 + 5), "error": f"Authorization: Basic {secret} AWS_SESSION_TOKEN={secret}"}
    calls.write_text(text[:torn] + json.dumps(leak) + "\n" + text[torn:], encoding="utf-8")
    blob = json.dumps(_live(session_dir).snapshot())
    assert "c9" in blob and secret not in blob


def test_a_private_ledger_another_account_keeps_is_reported_not_read(session_dir, tmp_path, monkeypatch):
    monkeypatch.setenv(session_layout.ENV_LEDGER_DIR, str(tmp_path / "ledgers"))
    (session_dir / session_layout.LEDGER_MARK).parent.mkdir(exist_ok=True)
    (session_dir / session_layout.LEDGER_MARK).write_text("{}", encoding="utf-8")
    snap = _live(session_dir).snapshot(include_streams=False)
    assert snap["tasks"] == [], "the database left in the session is stale once the ledger moved out"
    assert snap["run"]["coordinator_db"]["problem"] and snap["badges"][0]["level"] == "error"
    assert "private ledger" in snap["badges"][0]["text"] and "HYPERLOOM_LEDGER_DIR" in snap["badges"][0]["text"]


def test_the_ledger_is_followed_when_it_moves_out_of_the_session(session_dir, tmp_path, monkeypatch):
    monkeypatch.setenv(session_layout.ENV_LEDGER_DIR, str(tmp_path / "ledgers"))
    w = _live(session_dir)
    shared = session_dir / session_layout.SHARED_DB
    assert w.db.path == shared and w.tasks
    private = session_layout.private_db_path(session_dir)
    private.parent.mkdir(parents=True)
    shutil.copyfile(shared, private)
    conn = sqlite3.connect(shared)
    conn.execute("DELETE FROM tasks")
    conn.commit()
    conn.close()
    (session_dir / session_layout.LEDGER_MARK).parent.mkdir(exist_ok=True)
    (session_dir / session_layout.LEDGER_MARK).write_text("{}", encoding="utf-8")
    w.poll()
    snap = w.snapshot(include_streams=False)
    assert w.db.path == private and snap["tasks"] and snap["run"]["coordinator_db"]["problem"] is None
