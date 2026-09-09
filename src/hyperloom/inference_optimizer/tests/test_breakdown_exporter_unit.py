# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the session_breakdown.json exporter."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hyperloom.common.timeutil import iso_z
from hyperloom.inference_optimizer.breakdown import exporter as ex
from hyperloom.inference_optimizer.breakdown.collectors import sessions

# ---- _load_session_json ----


def test_load_state_missing(tmp_path):
    warnings = []
    assert ex._load_session_json(tmp_path / "state.json", "state.json", warnings) == {}
    assert any("state.json missing" in w for w in warnings)


def test_load_state_valid(tmp_path):
    (tmp_path / "state.json").write_text('{"session_id": "s"}', encoding="utf-8")
    assert ex._load_session_json(tmp_path / "state.json", "state.json", [])["session_id"] == "s"


def test_load_state_parse_error(tmp_path):
    (tmp_path / "state.json").write_text("{bad", encoding="utf-8")
    warnings = []
    assert ex._load_session_json(tmp_path / "state.json", "state.json", warnings) == {}
    assert any("failed to parse state.json" in w for w in warnings)


def test_load_manifest_missing(tmp_path):
    warnings = []
    assert ex._load_session_json(tmp_path / "manifest.json", "manifest.json", warnings) == {}
    assert any("manifest.json missing" in w for w in warnings)


def test_load_manifest_parse_error(tmp_path):
    (tmp_path / "manifest.json").write_text("{bad", encoding="utf-8")
    warnings = []
    assert ex._load_session_json(tmp_path / "manifest.json", "manifest.json", warnings) == {}
    assert any("failed to parse manifest.json" in w for w in warnings)


# ---- _safe_collect ----


def test_safe_collect_success():
    assert ex._safe_collect("x", lambda: 42, []) == 42


def test_safe_collect_exception_default_dict():
    warnings = []
    out = ex._safe_collect("x", lambda: (_ for _ in ()).throw(ValueError("e")), warnings)
    assert out == {}
    assert any("collector:x failed" in w for w in warnings)


def test_safe_collect_exception_with_default():
    def boom():
        raise RuntimeError("e")

    assert ex._safe_collect("x", boom, [], default=[]) == []


# ---- _json_default ----


def test_json_default_path():
    assert ex._json_default(Path("/x")) == "/x"


def test_json_default_set():
    assert ex._json_default({3, 1, 2}) == [1, 2, 3]


def test_json_default_typeerror():
    with pytest.raises(TypeError):
        ex._json_default(object())


# ---- build ----


def test_build_empty_session(tmp_path):
    out = ex.build(tmp_path)
    assert out["exporter_version"] == ex.EXPORTER_VERSION
    assert "warnings" in out
    assert "session" in out["metadata"]
    assert any("missing" in w for w in out["warnings"])




# ---- write_breakdown_json ----


def test_write_breakdown_json(tmp_path):
    target = ex.write_breakdown_json(tmp_path)
    assert target.name == ex.BREAKDOWN_FILENAME
    assert target.is_file()
    data = json.loads(target.read_text())
    assert data["exporter_version"] == ex.EXPORTER_VERSION


def test_write_breakdown_json_custom_output(tmp_path):
    out = tmp_path / "sub" / "bd.json"
    target = ex.write_breakdown_json(tmp_path, output_path=out)
    assert target == out.resolve()
    assert out.is_file()


# ---- patch_breakdown_langfuse ----


def test_patch_breakdown_langfuse_no_breakdown(tmp_path):
    assert ex.patch_breakdown_langfuse(tmp_path) is False


# ---- write_minimal_final_report ----


def test_write_minimal_final_report_creates(tmp_path):
    target = ex.write_minimal_final_report(tmp_path)
    assert target.name == "final.md"
    assert target.is_file()
    text = target.read_text()
    assert "emergency final report" in text


def test_write_minimal_final_report_idempotent(tmp_path):
    target = ex.write_minimal_final_report(tmp_path)
    target.write_text("PRESERVED", encoding="utf-8")
    again = ex.write_minimal_final_report(tmp_path)
    assert again.read_text() == "PRESERVED"


def test_write_minimal_final_report_with_attempts(tmp_path):
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState.load_or_init(tmp_path)
    state.last_baseline = {"tput": 50.0, "ts": "t0"}
    state.save(tmp_path)

    target = ex.write_minimal_final_report(tmp_path)
    text = target.read_text()
    assert "last_baseline" in text


# ---- write_minimal_final_json ----


def test_write_minimal_final_json_creates(tmp_path):
    target = ex.write_minimal_final_json(tmp_path)
    assert target.name == "final.json"
    assert target.is_file()
    data = json.loads(target.read_text(encoding="utf-8"))
    # Crash-safe fallback marker distinguishing this from full ReportExecutor output.
    assert data["safety_net"] is True
    assert data["report_complete"] is False
    # Headline fields the downstream stats pipeline keys off must be present.
    for key in ("session_id", "model_name", "stop_reason", "baseline_tput"):
        assert key in data


def test_write_minimal_final_json_idempotent(tmp_path):
    # A pre-existing final.json must never be clobbered by the minimal fallback.
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "final.json").write_text('{"full_report": true}', encoding="utf-8")
    again = ex.write_minimal_final_json(tmp_path)
    assert json.loads(again.read_text(encoding="utf-8")) == {"full_report": True}


def test_write_minimal_final_json_refreshes_stale_fallback(tmp_path):
    # A prior crash-safe fallback is stale after a resume and must be
    # overwritten with the current state, NOT preserved.
    from hyperloom.orchestrator.state.shared_state import SharedState

    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "final.json").write_text(
        '{"safety_net": true, "stop_reason": "time_exhausted", "baseline_tput": 1.0}',
        encoding="utf-8",
    )

    state = SharedState.load_or_init(tmp_path)
    state.set_stop_reason("signal")
    state.baseline_tput = 42.0
    state.save(tmp_path)

    again = ex.write_minimal_final_json(tmp_path)
    data = json.loads(again.read_text(encoding="utf-8"))
    assert data["stop_reason"] == "signal"
    assert data["baseline_tput"] == 42.0


def test_write_minimal_final_json_recovers_corrupt(tmp_path):
    # A non-empty but invalid final.json must be backed up and replaced with a
    # consumable fallback, not left as garbled JSON downstream can't read.
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "final.json").write_text('{"baseline_tput": 35.83, "trunc', encoding="utf-8")

    target = ex.write_minimal_final_json(tmp_path)
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["safety_net"] is True
    # Original (corrupt) bytes preserved for forensics.
    corrupt = reports / "final.json.corrupt"
    assert corrupt.is_file()
    assert corrupt.read_text(encoding="utf-8") == '{"baseline_tput": 35.83, "trunc'


def test_write_minimal_final_json_fields(tmp_path):
    from hyperloom.orchestrator.state.shared_state import SharedState

    state = SharedState.load_or_init(tmp_path)
    state.session_id = "sess-464"
    state.model_name = "command-a-plus"
    state.set_stop_reason("time_exhausted")
    state.baseline_tput = 35.83
    state.current_best = {"action": "baseline", "tput": 35.83}
    state.save(tmp_path)

    target = ex.write_minimal_final_json(tmp_path)
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["session_id"] == "sess-464"
    assert data["model_name"] == "command-a-plus"
    assert data["stop_reason"] == "time_exhausted"
    assert data["baseline_tput"] == 35.83
    assert data["current_best"] == {"action": "baseline", "tput": 35.83}


def test_patch_breakdown_langfuse_success(tmp_path):
    from hyperloom.orchestrator.trace.langfuse_emitter import _receipt_path

    ex.write_breakdown_json(tmp_path)
    receipt_path = _receipt_path(tmp_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps({"enabled": True, "counts_final": True}), encoding="utf-8")

    assert ex.patch_breakdown_langfuse(tmp_path) is True
    bd = json.loads((tmp_path / ex.BREAKDOWN_FILENAME).read_text())
    assert bd["metadata"]["langfuse"]["enabled"] is True
    assert ex.patch_breakdown_langfuse(tmp_path) is False


# ---- outcome.final over the close-out's recipe ----


def test_final_is_projected_from_the_recipe_the_close_out_settled(tmp_path):
    """``outcome.final`` reports the recipe the close recorded, not a state re-read.

    The export used to rebuild the headline from ``current_best`` and recover
    the launch by trying candidate run directories. The close-out states the
    terminal configuration instead, and this is the only source of it.
    """
    import json

    sd = tmp_path
    (sd / "state.json").write_text(
        json.dumps({"optimization_stack": [], "framework": "sglang"}),
        encoding="utf-8",
    )
    (sd / "manifest.json").write_text(
        json.dumps({"schema_version": 3, "session_id": "s", "model_name": "m", "framework": "sglang"}),
        encoding="utf-8",
    )
    parts = sd / "runtime" / "breakdown" / "parts"
    parts.mkdir(parents=True)
    (parts / "close__coordinator.json").write_text(
        json.dumps(
            {
                "kind": "singleton",
                "section": "close",
                "producer": "coordinator",
                "seq": 1,
                "ts": "2026-01-01T00:00:00Z",
                "payload": {
                    "status": "succeeded",
                    "final_recipe": {
                        "throughput": 123.0,
                        "action_path": ["baseline"],
                        "extra_server_args": "",
                        "extra_envs": {},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    bd = ex.build(sd)
    final_sec = bd.get("outcome", {}).get("final", {})
    assert final_sec.get("throughput_tok_s_per_gpu") == pytest.approx(123.0), "recorded recipe lost"
    assert final_sec.get("action_path") == ["baseline"]
    invocation = final_sec.get("invocation")
    assert invocation is not None, "the final block must always carry an invocation"
    # A session that never settled a validation has no recorded launch, and the
    # block says so rather than leaving the reader to guess at an empty string.
    assert invocation["framework_args_source"] == "unrecorded"


# ---- telemetry.orchestration_context ----


def _write_checkpoint_events(session_dir: Path, levels: list[int], *, degenerate: int = 0) -> None:
    """Seed a coordinator DB with orchestration checkpoint events."""
    import sqlite3

    db_dir = session_dir / "storage"
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_dir / "coordinator.db")
    try:
        conn.execute("CREATE TABLE events (seq INTEGER PRIMARY KEY, topic TEXT, payload TEXT)")
        for i, level in enumerate(levels):
            payload = {"kind": "orchestration_checkpoint", "tick": i + 1, "context_tokens": level}
            conn.execute(
                "INSERT INTO events (topic, payload) VALUES (?, ?)",
                ("observation", json.dumps(payload)),
            )
        for _ in range(degenerate):
            conn.execute(
                "INSERT INTO events (topic, payload) VALUES (?, ?)",
                ("observation", json.dumps({"kind": "orchestration_checkpoint_degraded"})),
            )
            # The repeat-degeneracy advisory duplicates the kind with a severity.
            conn.execute(
                "INSERT INTO events (topic, payload) VALUES (?, ?)",
                ("observation", json.dumps({"kind": "orchestration_checkpoint_degraded", "severity": "medium"})),
            )
        conn.commit()
    finally:
        conn.close()






def test_recorder_snapshot_leaves_the_task_config_contract_intact(tmp_path):
    """An unset knob must stay unset; a recorder snapshot used to coerce it to 0."""
    from types import SimpleNamespace

    from hyperloom.inference_optimizer.breakdown.recorder import instrument

    (tmp_path / "state.json").write_text(
        json.dumps({"session_id": "s", "framework": "sglang", "model_name": "qwen3-8b"}),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"session_id": "s", "framework_version": "0.4.1", "workload": {"conc": 64}}),
        encoding="utf-8",
    )
    instrument.snapshot_state_sections(
        tmp_path,
        SimpleNamespace(framework="sglang", model_name="qwen3-8b", model_path="", session_id="s"),
    )

    task_config = ex.build(tmp_path)["metadata"]["task_config"]
    assert task_config["framework_name"] == "sglang"
    assert task_config["framework_version"] == "0.4.1"
    assert task_config["conc"] == 64
    assert task_config["tp"] is None


# ---- session elapsed time ----


def _freeze_now(monkeypatch, instant: datetime) -> None:
    """Pin the session collector's clock to *instant*.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        instant (datetime): The UTC instant every ``datetime.now`` call returns.
    """

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz) if tz else instant

    monkeypatch.setattr(sessions, "datetime", _FrozenDatetime)


def test_a_session_that_has_not_stopped_yet_is_still_measured_up_to_now(tmp_path, monkeypatch):
    _freeze_now(monkeypatch, datetime(2026, 8, 8, 1, 37, 27, tzinfo=timezone.utc))
    section = sessions.collect_session(tmp_path, {"session_id": "s", "start_ts": "2026-08-08T00:37:27+00:00"}, {}, [])
    assert section["elapsed_minutes"] == 60.0


def test_a_stopped_session_measures_the_same_however_late_it_is_exported(tmp_path, monkeypatch):
    state = {
        "session_id": "s",
        "start_ts": "2026-08-08T00:37:27+00:00",
        "stop_reason": "time_exhausted",
        "stop_ts": "2026-08-08T02:55:27+00:00",
    }
    _freeze_now(monkeypatch, datetime(2026, 8, 8, 3, 0, 0, tzinfo=timezone.utc))
    first = sessions.collect_session(tmp_path, state, {}, [])["elapsed_minutes"]
    _freeze_now(monkeypatch, datetime(2026, 10, 20, 3, 0, 0, tzinfo=timezone.utc))
    second = sessions.collect_session(tmp_path, state, {}, [])["elapsed_minutes"]
    assert first == second == 138.0


def _stopped_session(session_dir: Path, *, ran_for: timedelta, stopped_ago: timedelta = timedelta(0)):
    """Write a stopped session's state so the recorder fragment is spooled.

    Args:
        session_dir (Path): The session directory to write into.
        ran_for (timedelta): How long the session ran before it stopped.
        stopped_ago (timedelta): How long before now it stopped, so an export
            measured to the recorded end can be told from one measured to now.

    Returns:
        SharedState: The saved state.
    """
    from hyperloom.orchestrator.state.shared_state import SharedState

    # Whole seconds: the exported end is canonicalised to second precision.
    stopped_at = (datetime.now(timezone.utc) - stopped_ago).replace(microsecond=0)
    state = SharedState.load_or_init(session_dir)
    state.session_id = "sess-1178"
    state.start_ts = (stopped_at - ran_for).isoformat(timespec="microseconds")
    state.set_stop_reason("time_exhausted")
    state.stop_ts = stopped_at.isoformat(timespec="microseconds")
    state.save(session_dir)
    return state


def test_a_recorded_session_exports_the_time_it_actually_ran(tmp_path):
    """A session that stopped days ago still ran for two hours, however late it is exported."""
    state = _stopped_session(tmp_path, ran_for=timedelta(hours=2), stopped_ago=timedelta(days=3))

    bd = ex.build(tmp_path)
    assert bd["metadata"]["session"]["start_ts"] == state.start_ts
    assert bd["metadata"]["session"]["ended_at_utc"] == iso_z(state.stop_ts)
    assert bd["metadata"]["session"]["elapsed_minutes"] == pytest.approx(120.0, abs=0.02)


def test_the_human_report_reads_the_same_elapsed_time_as_the_machine_field(tmp_path):
    """``elapsed_minutes`` is the key the rendered report prints; the fragment used to drop it."""
    _stopped_session(tmp_path, ran_for=timedelta(hours=2))

    bd = ex.build(tmp_path)
    assert 119.0 <= bd["metadata"]["session"]["elapsed_minutes"] <= 121.0


def test_the_recorder_path_keeps_the_fields_only_the_collector_can_resolve(tmp_path, monkeypatch):
    """The image comes from the manifest / environment, which the live state never sees."""
    monkeypatch.setenv("HYPERLOOM_IMAGE", "registry.example/hyperloom:test")
    _stopped_session(tmp_path, ran_for=timedelta(minutes=5))

    bd = ex.build(tmp_path)
    assert bd["metadata"]["session"]["image"] == "registry.example/hyperloom:test"
    assert bd["metadata"]["session"]["image_id"] == "hyperloom:test"
    assert bd["metadata"]["session"]["session_dir"] == str(tmp_path.resolve())


def test_a_clean_stop_resume_keeps_measuring_from_the_original_start(tmp_path):
    """--max-hours still counts from there after a clean stop, so the elapsed time does too."""
    now = datetime.now(timezone.utc)
    state = {
        "session_id": "sess-1178",
        "start_ts": (now - timedelta(hours=5)).isoformat(),
        "resumed_ts": (now - timedelta(hours=1)).isoformat(),
        "max_minutes": 360,
    }

    section = sessions.collect_session(tmp_path, state, {}, [])
    # The four hours the session was not running are charged to the budget too.
    assert 299.0 <= section["elapsed_minutes"] <= 301.0


def test_elapsed_time_is_measured_from_the_resumed_start_not_the_first_launch(tmp_path):
    """A resume after a stop re-anchors ``start_ts``, so elapsed restarts with the budget."""
    (tmp_path / "manifest.json").write_text(
        json.dumps({"session_id": "sess-1178", "created_at_utc": "2026-08-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    _stopped_session(tmp_path, ran_for=timedelta(minutes=30))

    bd = ex.build(tmp_path)
    assert 29.0 <= bd["metadata"]["session"]["elapsed_minutes"] <= 31.0
    # The first launch is still on record, so the gap before the resume is visible.
    assert bd["metadata"]["session"]["created_at_utc"] == "2026-08-01T00:00:00+00:00"


def test_a_resumed_session_is_not_reported_as_stopped_by_the_previous_legs_close(tmp_path):
    """A resume clears the reason in state, but the old CLOSE row stays in ``phase_history``."""
    from hyperloom.orchestrator.state.shared_state import SharedState

    now = datetime.now(timezone.utc)
    state = SharedState.load_or_init(tmp_path)
    state.session_id = "sess-1178"
    state.record_phase_transition(
        to_phase="CLOSE",
        reason="time_exhausted",
        ts=(now - timedelta(days=6)).isoformat(timespec="seconds"),
    )
    state.start_ts = (now - timedelta(minutes=30)).isoformat(timespec="microseconds")
    state.record_phase_transition(to_phase="PRELUDE", reason="resumed", ts=state.start_ts)
    state.save(tmp_path)

    bd = ex.build(tmp_path)
    assert bd["outcome"]["stop_reason"] == ""
    assert bd["metadata"]["session"]["ended_at_utc"] == ""
    assert 29.0 <= bd["metadata"]["session"]["elapsed_minutes"] <= 31.0


def test_a_session_resumed_after_a_clean_stop_is_still_reported_as_running(tmp_path):
    """A clean stop keeps ``start_ts``, so the previous leg's CLOSE sits inside the window."""
    now = datetime.now(timezone.utc)
    state = {
        "session_id": "sess-1178",
        "start_ts": (now - timedelta(hours=3)).isoformat(),
        "stop_reason": "",
        "phase": "CLOSE",
        "resumed_ts": (now - timedelta(hours=2)).isoformat(),
        "phase_history": [
            {
                "to_phase": "CLOSE",
                "reason": "time_exhausted",
                "ts": (now - timedelta(hours=2, minutes=30)).isoformat(),
            }
        ],
    }

    section = sessions.collect_session(tmp_path, state, {}, [])
    assert section["stop_reason"] == ""
    assert section["ended_at_utc"] == ""
    assert 179.0 <= section["elapsed_minutes"] <= 181.0


def test_a_close_the_state_file_never_recorded_still_supplies_the_end(tmp_path):
    """The fallback's own case: the run stopped and only ``phase_history`` knows why."""
    now = datetime.now(timezone.utc)
    state = {
        "session_id": "sess-1178",
        "start_ts": (now - timedelta(hours=2)).isoformat(),
        "phase_history": [
            {
                "to_phase": "CLOSE",
                "reason": "target_reached",
                "ts": (now - timedelta(minutes=5)).isoformat(),
            }
        ],
    }

    section = sessions.collect_session(tmp_path, state, {}, [])
    assert section["stop_reason"] == "target_reached"
    assert section["ended_at_utc"] != ""
    assert 114.0 <= section["elapsed_minutes"] <= 116.0


@pytest.mark.parametrize(
    "start_ts, close_ts",
    [
        ("", "2026-08-01T00:00:00+00:00"),
        ("2026-08-01T00:00:00+00:00", ""),
        ("2026-08-01T00:00:00+00:00", "not-a-timestamp"),
        # A CLOSE at the boundary belongs to the leg that started there.
        ("2026-08-01T00:00:00+00:00", "2026-08-01T00:00:00+00:00"),
    ],
)
def test_a_close_stands_when_there_is_nothing_comparable_to_disqualify_it(tmp_path, start_ts, close_ts):
    """Only two parseable timestamps can place a CLOSE in a previous leg."""
    state = {
        "session_id": "sess-1178",
        "start_ts": start_ts,
        "phase_history": [{"to_phase": "CLOSE", "reason": "target_reached", "ts": close_ts}],
    }

    assert sessions.collect_session(tmp_path, state, {}, [])["stop_reason"] == "target_reached"


def test_the_newest_close_of_the_leg_supplies_the_end(tmp_path):
    """A cyclic run reaches CLOSE more than once; the last word is the session's."""
    now = datetime.now(timezone.utc)
    state = {
        "session_id": "sess-1178",
        "start_ts": (now - timedelta(hours=4)).isoformat(),
        "phase_history": [
            {"to_phase": "CLOSE", "reason": "sweep_done", "ts": (now - timedelta(hours=3)).isoformat()},
            {"to_phase": "EXPLORE", "reason": "cycle_reloop", "ts": (now - timedelta(hours=2)).isoformat()},
            {"to_phase": "CLOSE", "reason": "target_reached", "ts": (now - timedelta(minutes=10)).isoformat()},
        ],
    }

    section = sessions.collect_session(tmp_path, state, {}, [])
    assert section["stop_reason"] == "target_reached"
    assert 229.0 <= section["elapsed_minutes"] <= 231.0


def test_a_history_written_out_of_order_still_supplies_the_end(tmp_path):
    """The scan walks back from the newest row, so a stale one must not end the search."""
    now = datetime.now(timezone.utc)
    state = {
        "session_id": "sess-1178",
        "start_ts": (now - timedelta(hours=2)).isoformat(),
        "resumed_ts": (now - timedelta(hours=2)).isoformat(),
        "phase_history": [
            {"to_phase": "CLOSE", "reason": "target_reached", "ts": (now - timedelta(minutes=5)).isoformat()},
            {"to_phase": "CLOSE", "reason": "time_exhausted", "ts": (now - timedelta(days=3)).isoformat()},
        ],
    }

    assert sessions.collect_session(tmp_path, state, {}, [])["stop_reason"] == "target_reached"


def test_an_unreadable_stop_time_does_not_become_the_session_end(tmp_path):
    """Pre-``stop_ts`` this branch stamped the export clock; a bad value must not read as an end."""
    now = datetime.now(timezone.utc)
    state = {
        "session_id": "sess-1178",
        "start_ts": (now - timedelta(hours=2)).isoformat(),
        "stop_reason": "target_reached",
        "stop_ts": "not-a-timestamp",
    }

    section = sessions.collect_session(tmp_path, state, {}, [])
    assert section["ended_at_utc"] != "not-a-timestamp"
    assert 119.0 <= section["elapsed_minutes"] <= 121.0


def test_an_unreadable_stop_time_falls_back_to_the_close_transition(tmp_path):
    now = datetime.now(timezone.utc)
    closed_at = (now - timedelta(minutes=30)).isoformat()
    state = {
        "session_id": "sess-1178",
        "start_ts": (now - timedelta(hours=2)).isoformat(),
        "stop_reason": "target_reached",
        "stop_ts": "not-a-timestamp",
        "phase_history": [{"to_phase": "CLOSE", "reason": "target_reached", "ts": closed_at}],
    }

    section = sessions.collect_session(tmp_path, state, {}, [])
    assert section["ended_at_utc"] == iso_z(closed_at)
    assert 89.0 <= section["elapsed_minutes"] <= 91.0


# ---- _merge_session ----


def test_the_recorder_fragment_overlays_the_collected_section():
    merged = ex._merge_session(
        {"session_id": "sess-1178", "stop_reason": "target_reached"},
        {"session_id": "", "stop_reason": "", "image": "registry.example/hyperloom:test"},
    )
    assert merged["session_id"] == "sess-1178"
    assert merged["stop_reason"] == "target_reached"
    assert merged["image"] == "registry.example/hyperloom:test"


def test_a_section_with_no_fragment_is_returned_untouched():
    section = {"session_id": "sess-1178"}
    assert ex._merge_session(None, section) is section
    assert ex._merge_session({}, section) is section


def test_an_unrecorded_budget_does_not_erase_the_collected_one():
    """The snapshot writes every key on every save, so an unset int arrives as 0."""
    merged = ex._merge_session(
        {"max_minutes": 0, "tick_count": 0},
        {"max_minutes": 360, "tick_count": 12},
    )
    assert merged["max_minutes"] == 360
    assert merged["tick_count"] == 12


def test_the_live_phase_stays_in_the_section_even_when_blank():
    """Only the recorder knows the phase; the key was always present before the merge."""
    merged = ex._merge_session({"phase": ""}, {"session_id": "sess-1178"})
    assert merged["phase"] == ""


def test_the_merged_section_measures_its_own_elapsed_time():
    merged = ex._merge_session(
        {
            "start_ts": "2026-08-08T00:00:00+00:00",
            "ended_at_utc": "2026-08-08T02:00:00+00:00",
            "stop_reason": "target_reached",
        },
        {"elapsed_minutes": 0.0},
    )
    assert merged["elapsed_minutes"] == 120.0


# ---- every collector is isolated ----




def test_collector_arguments_are_evaluated_inside_the_isolation():
    """A collector whose arguments raise is caught like one whose body does.

    The arguments are built by the zero-argument closure the wrapper calls, so
    a section that drifted to an unexpected shape cannot abort the build while
    another collector's call is being assembled.
    """
    warnings: list[str] = []
    drifted = "not-a-mapping"

    out = ex._safe_collect("probe", lambda: {"seen": drifted.get("key")}, warnings, default={})

    assert out == {}
    assert any("collector:probe failed: AttributeError" in w for w in warnings)
