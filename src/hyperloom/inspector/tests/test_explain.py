# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import sqlite3

import pytest

from hyperloom.inspector import explain
from hyperloom.inspector.tests.conftest import FAKE_KEY, T0, iso
from hyperloom.inspector.watcher import Watcher, WatchOptions

GRID = [
    {
        "name": "ep",
        "extra_args": "--enable-expert-parallel",
        "extra_envs": {},
        "provenance": "llm",
        "kb_evidence": ["kb: EP helps MoE at conc 32"],
    },
    {"name": "mtp1", "extra_args": "--speculative-config mtp1", "extra_envs": {"MTP": "1"}, "provenance": "specialist"},
]
OUTCOMES = [
    {
        "variant_name": "ep",
        "outcome": "REVERT",
        "stage": "decision",
        "reason": "below keep threshold",
        "error_excerpt": None,
        "metrics": {"tput": 1650.0, "gain_pct": -2.1},
        "variant": {"name": "ep", "extra_server_args": "--enable-expert-parallel", "extra_envs": {}},
    },
    {
        "variant_name": "mtp1",
        "outcome": "KEEP",
        "stage": "decision",
        "metrics": {"tput": 1900.0, "gain_pct": 12.7},
        "variant": {"name": "mtp1", "extra_server_args": "--speculative-config mtp1", "extra_envs": {"MTP": "1"}},
    },
]
RESULT = {
    "status": "succeeded",
    "base_tput": 1685.16,
    "output_throughput": 1900.0,
    "gain_pct": 12.7,
    "best_variant": {"name": "mtp1", "gain_pct": 12.7},
    "winners": [{"name": "mtp1", "gain_pct": 12.7, "tput": 1900.0}],
    "losers": [{"name": "ep", "gain_pct": -2.1, "tput": 1650.0, "reason": "below keep threshold"}],
    "per_variant_outcomes": OUTCOMES,
}


def _progress(label, status, tput, t):
    return {"progress": {"unit": "variant", "label": label, "status": status, "output_throughput": tput}, "ts": iso(t)}


def _add_explore(session_dir):
    """An explore task with a two-candidate grid, its progress rows, its result and two journal entries."""
    params = {"grid": GRID, "base_tput": 1685.16, "keep_threshold_pct": 1.0}
    history = [
        {"from": "queued", "to": "running", "ts": iso(T0 + 660)},
        _progress("ep", "done", 1650.0, T0 + 680),
        _progress("mtp1", "done", 1900.0, T0 + 700),
        {"from": "running", "to": "succeeded", "ts": iso(T0 + 750)},
    ]
    db = sqlite3.connect(session_dir / "storage" / "coordinator.db")
    db.execute(
        "INSERT INTO tasks (task_id, kind, state, params, idempotency_key, requires_lanes, "
        "lease_ttl_sec, history, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("t-expl", "explore", "succeeded", json.dumps(params), "t-expl", "[]", 0, json.dumps(history))
        + (iso(T0 + 655), iso(T0 + 750)),
    )
    payload = {"task_id": "t-expl", "state": "succeeded", "result": RESULT}
    db.execute(
        "INSERT INTO events (msg_id, from_agent, to_agent, topic, payload, ts) VALUES (?,?,?,?,?,?)",
        ("m-expl", "coordinator", "*", "delegated_result", json.dumps(payload), iso(T0 + 750)),
    )
    db.commit()
    db.close()
    path = session_dir / "reports" / "optimization_journal.json"
    journal = json.loads(path.read_text())
    for change, outcome, tput, gain, t in (
        ("--enable-expert-parallel", "REVERT", 1650.0, -2.1, T0 + 751),
        ("--speculative-config mtp1 | env: MTP=1", "KEEP", 1900.0, 12.7, T0 + 752),
    ):
        journal["entries"].append(
            {
                "ts": iso(t),
                "phase": "SYSTEM",
                "kind": "param",
                "change": change,
                "outcome": outcome,
                "throughput_after": tput,
                "gain_pct": gain,
                "task_id": "t-expl",
            }
        )
    path.write_text(json.dumps(journal))
    log = session_dir / "runs" / "explore" / "t-expl" / "v01_mtp1" / "server.log"
    log.parent.mkdir(parents=True)
    log.write_text("".join(f"INFO line {i}\n" for i in range(60)) + f"export HF_TOKEN={FAKE_KEY}\n")


def _live(session_dir):
    w = Watcher(session_dir, WatchOptions(sample_metrics=False))
    w.poll()
    return w


def _replay_at(session_dir, t):
    w = Watcher(session_dir, WatchOptions(replay=True, sample_metrics=False))
    w.poll()
    w.clock.seek(t, paused=True)
    return w


def _section(w, ref):
    view = w.snapshot()
    return explain.item_section(w, view, ref, view["clock"]["now"])


# ----- one section per kind ----------------------------------------------------------------------------------


def test_node_lists_health_traffic_and_its_tasks(session_dir):
    label, text = _section(_live(session_dir), {"kind": "node", "id": "spec:static_recon"})
    assert label == "static_recon"
    assert "health crashed" in text and "Traffic in: coordinator dispatch ×1" in text
    assert "static_recon (specialist, t-recon) succeeded, health crashed" in text


def test_task_has_its_history_result_and_redacted_log_tail(session_dir):
    label, text = _section(_live(session_dir), {"kind": "task", "id": "t-scout"})
    assert label == "research_scout"
    assert "State succeeded, health crashed" in text and "queued→running" in text
    assert "Result: status" not in text and "runner_status=stale" in text
    assert "runs/specialist/t-scout/process.log" in text and "Error: model refused" in text
    assert FAKE_KEY not in text and "<redacted>" in text


def test_explore_task_shows_params_without_the_grid_body_variants_and_newest_log(session_dir):
    _add_explore(session_dir)
    label, text = _section(_live(session_dir), {"kind": "task", "id": "t-expl"})
    assert label == "explore"
    assert "Grid (2 candidates): ep, mtp1" in text and "kb_evidence" not in text
    assert 'Params: {"base_tput":1685.2,"keep_threshold_pct":1.0}' in text
    assert "Best variant: mtp1" in text and "Losers: ep" in text
    assert "variant ep: status=done output_throughput=1650.0" in text
    assert "Last 40 lines of runs/explore/t-expl/v01_mtp1/server.log" in text
    assert "INFO line 21" in text and "INFO line 20\n" not in text
    assert FAKE_KEY not in text


def test_event_shows_payload_and_the_result_of_a_delegated_result(session_dir):
    w = _live(session_dir)
    seq = next(e["seq"] for e in w.snapshot()["streams"]["event"] if (e["payload"] or {}).get("task_id") == "t-perf")
    label, text = _section(w, {"kind": "event", "seq": seq})
    assert label == "delegated_result from coordinator to *"
    assert f"Bus event #{seq}: delegated_result" in text and "About task t-perf" in text
    assert "Result: runner_status=ok" in text and '"proposals":2' in text


def test_call_has_metrics_and_the_redacted_conversation(session_dir):
    label, text = _section(_live(session_dir), {"kind": "call", "call_id": "c1"})
    assert label == "orchestration call c1"
    assert "model claude-opus-5-5" in text and "input_tokens=1000" in text
    assert "Prompt:\nuse key <redacted>" in text and "Response:\nok" in text
    _, text = _section(_live(session_dir), {"kind": "call", "call_id": "c3"})
    assert "status=error" in text and "No prompt or response was recorded" in text


def test_journal_entry_has_the_best_before_neighbours_and_its_recipe(session_dir):
    _add_explore(session_dir)
    ref = {"kind": "journal", "task_id": "t-expl", "change": "--enable-expert-parallel", "ts": iso(T0 + 751)}
    label, text = _section(_live(session_dir), ref)
    assert label == "--enable-expert-parallel"
    assert "outcome=REVERT" in text and "keep threshold 1.0%" in text
    assert "best so far 1685.2 tok/s (baseline)" in text
    assert "Entries of the same task after it:" in text and "outcome=KEEP" in text
    assert "Candidate ep: recipe --enable-expert-parallel" in text and "kb: EP helps MoE" in text
    assert "Outcome of ep: outcome=REVERT" in text
    ref = {"kind": "journal", "task_id": "t-base", "change": "baseline", "ts": iso(T0 + 600)}
    label, text = _section(_live(session_dir), ref)
    assert label == "baseline" and "best so far none yet" in text


def test_variant_has_recipe_progress_outcome_and_task_settings(session_dir):
    _add_explore(session_dir)
    label, text = _section(_live(session_dir), {"kind": "variant", "task_id": "t-expl", "label": "mtp1"})
    assert label == "mtp1"
    assert "recipe --speculative-config mtp1 | env: MTP=1" in text and "provenance=specialist" in text
    assert "Progress row (variant): status=done output_throughput=1900.0" in text
    assert "Outcome of mtp1: outcome=KEEP" in text
    assert "base_tput=1685.2 keep_threshold_pct=1.0" in text
    assert "Before it ran: Baseline 1685.2 tok/s; best so far 1685.2 tok/s (baseline); keep threshold 1.0%" in text


# ----- refusals, redaction, cap ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ref",
    [
        {"kind": "node", "id": "spec:nope"},
        {"kind": "task", "id": "t-nope"},
        {"kind": "event", "seq": 9999},
        {"kind": "call", "call_id": "nope"},
        {"kind": "journal", "task_id": "t-base", "change": "baseline", "ts": iso(T0 + 1)},
        {"kind": "variant", "task_id": "t-base", "label": "nope"},
    ],
)
def test_unknown_items_raise_lookup_error(session_dir, ref):
    with pytest.raises(LookupError):
        _section(_live(session_dir), ref)


def test_replay_refuses_items_past_the_clock(session_dir):
    _add_explore(session_dir)
    w = _replay_at(session_dir, T0 + 305)
    for ref in (
        {"kind": "task", "id": "t-roof"},
        {"kind": "call", "call_id": "c1"},
        {"kind": "variant", "task_id": "t-expl", "label": "ep"},
        {"kind": "journal", "task_id": "t-expl", "change": "--enable-expert-parallel", "ts": iso(T0 + 751)},
        {"kind": "node", "id": "exec:roofline"},
    ):
        with pytest.raises(LookupError):
            _section(w, ref)
    # Mid-way through the explore task its outcome hasn't landed and later rows are hidden.
    w.clock.seek(T0 + 690, paused=True)
    _, text = _section(w, {"kind": "variant", "task_id": "t-expl", "label": "ep"})
    assert "No outcome recorded yet." in text and "1650.0" in text
    _, text = _section(w, {"kind": "task", "id": "t-expl"})
    assert "mtp1: status" not in text and "Result:" not in text and "server.log" not in text


def test_section_is_capped(session_dir):
    log = session_dir / "runs" / "specialist" / "t-scout" / "process.log"
    log.write_text("".join(f"{i:03d} " + "y" * 290 + "\n" for i in range(60)))
    _, text = _section(_live(session_dir), {"kind": "task", "id": "t-scout"})
    assert len(text) == explain.MAX_CHARS and text.endswith("[section truncated]")


# ----- reference validation ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("obj", "message"),
    [
        ("task", "JSON object"),
        ({"kind": "artifact", "path": "x"}, "unknown item kind"),
        ({"id": "x"}, "unknown item kind"),
        ({"kind": "task"}, "item.id is required"),
        ({"kind": "task", "id": ""}, "item.id is required"),
        ({"kind": "task", "id": 7}, "must be a string"),
        ({"kind": "event", "seq": "3"}, "must be an integer"),
        ({"kind": "event", "seq": True}, "must be an integer"),
        ({"kind": "call", "turn": 1}, "call_id or item.task_id"),
        ({"kind": "call", "task_id": "t", "turn": 1.5}, "must be an integer"),
        ({"kind": "journal", "task_id": "t", "change": "c"}, "item.ts is required"),
        ({"kind": "variant", "task_id": "t", "label": "x" * 201}, "over 200 characters"),
    ],
)
def test_bad_references_raise_value_error(obj, message):
    with pytest.raises(ValueError, match=message):
        explain.parse_ref(obj)


def test_references_keep_only_their_kinds_fields():
    assert explain.parse_ref({"kind": "task", "id": "t1", "extra": "dropped"}) == {"kind": "task", "id": "t1"}
    assert explain.parse_ref({"kind": "call", "task_id": "t1", "turn": 2}) == {
        "kind": "call",
        "task_id": "t1",
        "turn": 2,
    }
    assert explain.parse_ref({"kind": "variant", "task_id": "t", "label": "x" * 200})["label"] == "x" * 200
    # A journal row written outside a task has no task id; the page sends "".
    assert explain.parse_ref({"kind": "journal", "task_id": "", "change": "c", "ts": "t"})["task_id"] == ""
