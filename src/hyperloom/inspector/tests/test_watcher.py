# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import time

from hyperloom.inspector import artifacts, digest, session_layout
from hyperloom.inspector.tests.conftest import FAKE_KEY, FAKE_SECRET, T0, iso
from hyperloom.inspector.watcher import Clock, Stream, Watcher, WatchOptions


def _live(session_dir):
    w = Watcher(session_dir, WatchOptions(sample_metrics=False))
    w.poll()
    return w


def _replay_at(session_dir, t):
    w = Watcher(session_dir, WatchOptions(replay=True))
    w.poll()
    w.clock.seek(t, paused=True)
    return w


def test_crashed_specialists_that_landed_as_succeeded_are_flagged(session_dir):
    tasks = {t["task_id"]: t for t in _live(session_dir).snapshot()["tasks"]}
    assert tasks["t-scout"]["state"] == "succeeded"
    assert tasks["t-scout"]["health"] == "crashed"
    assert tasks["t-recon"]["health"] == "crashed"
    assert tasks["t-perf"]["health"] == "ok"
    assert tasks["t-roof"]["health"] in ("running", "stalled")
    assert tasks["t-recon"]["node"] == "spec:static_recon"


def test_snapshot_badges_graph_and_budget(session_dir):
    snap = _live(session_dir).snapshot()
    assert "2 task(s) crashed" in snap["badges"][0]["text"]
    edges = {(e["source"], e["target"], e["label"]): e["count"] for e in snap["graph"]["edges"]}
    assert edges[("critic", "orchestration", "review_verdict")] == 1
    assert edges[("coordinator", "bus", "delegated_result")] == 4
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


def test_phase_bands_skip_marker_rows(session_dir):
    phases = _live(session_dir).snapshot()["timeline"]["phases"]
    assert [(p["phase"], p["start"], p["end"]) for p in phases] == [
        ("PRELUDE", T0 + 5, T0 + 650),
        ("SYSTEM", T0 + 650, None),
    ]


def test_replay_chapters_cover_the_recording_past_the_clock(session_dir):
    timeline = _replay_at(session_dir, T0 + 305).snapshot()["timeline"]
    assert [(p["phase"], p["end"]) for p in timeline["phases"]] == [("PRELUDE", None)]
    assert [(p["phase"], p["start"], p["end"]) for p in timeline["chapters"]] == [
        ("PRELUDE", T0 + 5, T0 + 650),
        ("SYSTEM", T0 + 650, None),
    ]
    assert "chapters" not in _live(session_dir).snapshot()["timeline"]


def test_a_live_run_gone_quiet_ends_the_timeline_and_token_chart_at_its_last_activity(session_dir, monkeypatch):
    w = _live(session_dir)
    snap = w.snapshot()
    end = snap["timeline"]["end"]
    assert T0 < end < w.clock.now() - 600
    assert snap["tokens"]["series"]["end"] == end
    monkeypatch.setattr(w, "optimizer_status", lambda: {"alive": True})
    snap = w.snapshot()
    assert snap["timeline"]["end"] is None
    assert snap["tokens"]["series"]["end"] > w.clock.now() - 5
    assert _replay_at(session_dir, T0 + 305).snapshot()["timeline"]["end"] is None


def test_a_finished_live_run_s_quiet_badges_do_not_grow_with_the_clock(session_dir, monkeypatch):
    w = _live(session_dir)
    now = w.clock.now()

    def ages_at(t):
        monkeypatch.setattr(w.clock, "now", lambda: t)
        return {a["agent"]: a["age_sec"] for a in w.snapshot()["agents"]}

    end = w.snapshot()["timeline"]["end"]
    quiet = {"orchestration": round(max(0.0, end - (T0 + 720))), "critic": round(max(0.0, end - (T0 + 640)))}
    assert ages_at(now) == ages_at(now + 86_400) == quiet
    monkeypatch.setattr(w, "optimizer_status", lambda: {"alive": True})
    assert ages_at(now + 60)["critic"] == round(now + 60 - (T0 + 640))


def test_an_optimizer_on_this_host_is_alive_while_its_process_runs(session_dir):
    lock = {"hostname": socket.gethostname(), "pid": os.getpid()}
    (session_dir / "runtime" / "optimizer.lock").write_text(json.dumps(lock), encoding="utf-8")
    assert _live(session_dir).optimizer_status() == {"alive": True, "pid": os.getpid()}


def test_replay_rolls_views_back_to_the_clock(session_dir):
    snap = _replay_at(session_dir, T0 + 305).snapshot()
    assert snap["summary"]["phase"] == "PRELUDE"
    states = {t["task_id"]: (t["state"], t["health"]) for t in snap["tasks"]}
    assert states["t-scout"] == ("succeeded", "crashed")
    assert states["t-recon"] == ("running", "running")
    assert "t-roof" not in states
    assert len(snap["streams"]["event"]) == 2
    assert snap["streams"]["llm_call"] == []
    assert all(a["mtime"] <= T0 + 305 for a in _replay_at(session_dir, T0 + 305).artifacts_at())


def test_replay_shows_the_run_state_of_the_clock_not_of_the_end(session_dir):
    snap = _replay_at(session_dir, T0 + 305).snapshot()
    s = snap["summary"]
    assert s["baseline_tput"] is None
    assert snap["budget"]["used_sec"] == 305
    assert (s["tick"], s["crash_count"]) == (None, None)
    assert "baseline null tok/s" in digest.build(snap)
    assert "tick 42" not in digest.build(snap)
    assert _replay_at(session_dir, T0 + 2).snapshot()["summary"]["phase"] is None
    later = _replay_at(session_dir, T0 + 700).snapshot()
    assert later["summary"]["baseline_tput"] == 1685.16
    assert later["budget"]["used_sec"] == 700


def test_replay_shows_the_stop_and_validated_gain_once_the_clock_passes_the_stop(session_dir):
    path = session_dir / "state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    stop = {"stop_ts": iso(T0 + 650), "stop_reason": "target_reached", "cumulative_gain_validated": 12.5}
    path.write_text(json.dumps({**state, **stop, "elapsed_charged_sec": 500}), encoding="utf-8")
    before = _replay_at(session_dir, T0 + 640).snapshot()
    assert [before["summary"][k] for k in stop] == [None, None, None]
    after = _replay_at(session_dir, T0 + 700).snapshot()
    assert [after["summary"][k] for k in stop] == list(stop.values())
    assert after["budget"]["used_sec"] == 500


def test_replay_delta_releases_items_and_resets_on_rewind(session_dir):
    w = _replay_at(session_dir, T0 + 305)
    cursors: dict[str, int] = {}
    first = w.delta(cursors)
    assert len(first["items"]["event"]) == 2
    w.clock.seek(T0 + 700)
    second = w.delta(cursors)
    assert [c["call_id"] for c in second["items"]["llm_call"]] == ["c1", "c2"]
    w.clock.seek(T0 + 100)
    assert w.delta(cursors)["reset"] is True


def _bench_file(session_dir, rel, end, tput, ttft):
    path = session_dir / "runs" / rel / "inferencex_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"output_throughput": tput, "mean_ttft_ms": ttft, "p99_ttft_ms": ttft * 3, "benchmark_end_time_unix": end}
        ),
        encoding="utf-8",
    )


def test_measured_benchmark_runs_become_points_released_by_the_clock(session_dir):
    _bench_file(session_dir, "explore/t-exp/v00_a/variant_00_a/benchmark_1", T0 + 200, 1000.0, 80.0)
    _bench_file(session_dir, "explore/t-exp/v00_a/warmup_round/variant_00_a/benchmark_0", T0 + 100, 900.0, 95.0)
    _bench_file(session_dir, "explore/t-exp/v01_b/variant_00_b/benchmark_2", T0 + 500, 1200.0, 70.0)
    (session_dir / "runs" / "explore" / "t-exp" / "v01_b" / "stray.json").write_text("{}", encoding="utf-8")

    live = _live(session_dir).snapshot()["streams"]["benchmark"]
    assert [(p["variant"], p["output_throughput"], p["latency"]["ttft"]) for p in live] == [
        ("a", 1000.0, {"mean": 80.0, "median": None, "p99": 240.0}),
        ("b", 1200.0, {"mean": 70.0, "median": None, "p99": 210.0}),
    ]
    assert {p["task_id"] for p in live} == {"t-exp"}

    early = _replay_at(session_dir, T0 + 250).snapshot()["streams"]["benchmark"]
    assert [p["variant"] for p in early] == ["a"]


def test_baseline_point_is_released_once_from_its_timeline_record_with_the_raw_file_percentiles(session_dir):
    path = session_dir / "reports" / "sbd_v6" / "timeline" / "000001-baseline.json"
    assert _live(session_dir).snapshot()["streams"]["benchmark"] == []

    action = {
        "task_id": "t-base",
        "end_time": iso(T0 + 600),
        "measurement": {"output_throughput": 1685.2, "ttft_mean_ms": 60.2},
    }
    path.write_text(json.dumps({"id": 1, "kind": "baseline", "ext": {"actions": [action]}}), encoding="utf-8")
    raw = session_dir / "runs" / "baseline" / "t-base" / "measure_round" / "benchmark_1" / "inferencex_result.json"
    raw.parent.mkdir(parents=True)
    raw.write_text(json.dumps({"mean_ttft_ms": 60.2, "median_ttft_ms": 51.0, "p99_ttft_ms": 190.0}), encoding="utf-8")
    w = _live(session_dir)
    path.write_text(
        json.dumps({"id": 1, "kind": "baseline", "ext": {"actions": [action], "again": 1}}), encoding="utf-8"
    )
    os.utime(path, (T0 + 900, T0 + 900))
    w.poll()
    points = w.snapshot()["streams"]["benchmark"]
    assert [(p["kind"], p["task_id"], p["output_throughput"]) for p in points] == [("baseline", "t-base", 1685.2)]
    assert points[0]["latency"]["ttft"] == {"mean": 60.2, "median": 51.0, "p99": 190.0}
    assert points[0]["_t"] == T0 + 600

    raw.unlink()
    assert _live(session_dir).snapshot()["streams"]["benchmark"][0]["latency"]["ttft"]["median"] is None


def test_clock_range_speed_and_pause():
    clock = Clock(replay=True, speed=10.0)
    clock.set_range(100.0, 200.0)
    clock.seek(150.0, paused=True)
    assert clock.now() == 150.0
    clock.seek(500.0)
    assert clock.now() == 200.0
    assert Clock().info()["mode"] == "live"


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


def test_replay_end_shows_artifacts_written_after_the_last_event(session_dir):
    w = _replay_at(session_dir, T0 + 10_000)
    late = [a for a in w.artifact_index if a["mtime"] > w.clock.end]
    assert late, "fixture should have a file newer than the last event"
    assert {a["path"] for a in w.artifacts_at()} == {a["path"] for a in w.artifact_index}


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
    assert _replay_at(session_dir, T0 + 599).snapshot()["summary"]["current_best"]["hot_tput"] is None
    assert _replay_at(session_dir, T0 + 601).snapshot()["summary"]["current_best"]["hot_tput"] == 1685.16


def test_accuracy_from_state_appears_once_the_clock_passes_it(session_dir):
    state = json.loads((session_dir / "state.json").read_text())
    state["baseline_accuracy"] = 0.968
    accepted = [{"name": "k7", "accuracy": 0.962, "ts": iso(T0 + 700)}, {"name": "ungated", "accuracy": None}]
    state["explore_search"] = {"accepted": accepted}
    (session_dir / "state.json").write_text(json.dumps(state))
    live = _live(session_dir).snapshot()["summary"]["accuracy"]
    assert live == {"baseline": 0.968, "variants": [{"variant": "k7", "value": 0.962, "t": T0 + 700}]}
    assert _replay_at(session_dir, T0 + 699).snapshot()["summary"]["accuracy"]["variants"] == []
    assert _replay_at(session_dir, T0 + 701).snapshot()["summary"]["accuracy"] == live
    assert _replay_at(session_dir, T0 + 1).snapshot()["summary"]["accuracy"]["baseline"] is None


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
    replayed = {t["task_id"]: t for t in _replay_at(session_dir, T0 + 725).snapshot()["tasks"]}
    assert replayed["t-roof"]["progress"]["current"]["label"] == "k7"


def test_resource_history_backfills_and_records_outside_the_session(session_dir, tmp_path, monkeypatch):
    from hyperloom.inspector import watcher as watcher_mod
    from hyperloom.inspector.history import MetricsHistory

    monkeypatch.setattr(watcher_mod, "container_memory", lambda: {"used_bytes": 2, "limit_bytes": 4})
    monkeypatch.setattr(watcher_mod, "gpu_metrics", lambda: [{"gfx_pct": 50.0}])
    monkeypatch.setattr(watcher_mod.socket, "gethostname", lambda: "pod-a")  # the optimizer's host
    path = tmp_path / "hist" / "s.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps({"t": T0 + 5, "memory": {"used_bytes": 1, "limit_bytes": 4}, "gpus": []}) + "\n")
    w = Watcher(session_dir, history=MetricsHistory(path))
    w.poll()
    w.poll()  # within the sample interval: no second sample
    metrics = w.streams["metrics"].items
    assert [m["memory"]["used_bytes"] for m in metrics] == [1, 2]
    assert metrics[1]["gpus"] == [{"gfx_pct": 50.0}]
    assert len(path.read_text().splitlines()) == 2
    replay = Watcher(session_dir, WatchOptions(replay=True), history=MetricsHistory(path))
    replay.poll()
    # History samples, including the one written just now, don't stretch the replay range.
    assert replay.clock.end < metrics[1]["_t"]
    assert len(path.read_text().splitlines()) == 2


def test_no_resources_are_sampled_while_the_optimizer_runs_on_another_host(session_dir, monkeypatch):
    from hyperloom.inspector import watcher as watcher_mod

    monkeypatch.setattr(watcher_mod, "container_memory", lambda: {"used_bytes": 2, "limit_bytes": 4})
    monkeypatch.setattr(watcher_mod, "gpu_metrics", list)
    w = Watcher(session_dir)
    w.poll()
    assert w.streams["metrics"].items == []
    assert w.snapshot()["run"]["optimizer"]["host"] == "pod-a"
    monkeypatch.setattr(watcher_mod.socket, "gethostname", lambda: "pod-a")
    w._last_sample = 0.0
    w.poll()
    assert [m["memory"]["used_bytes"] for m in w.streams["metrics"].items] == [2]
    assert "host" not in w.snapshot()["run"]["optimizer"]


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
