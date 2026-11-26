# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime

import pytest

from hyperloom.inspector import tokens as tokens_mod
from hyperloom.inspector.history import history_path
from hyperloom.inspector.price_sheet import SHEET
from hyperloom.inspector.tests import producers
from hyperloom.inspector.tests.conftest import T0, iso
from hyperloom.inspector.token_terms import TERMS
from hyperloom.inspector.tokens import KINDS, NO_MODEL, StreamTurns, TokenTracker
from hyperloom.inspector.watcher import Watcher, WatchOptions
from hyperloom.inference_optimizer.trace.parse_usage import (
    parse_claude_stream_json_turn_usages,
    parse_claude_stream_json_usage,
)
from hyperloom_common.llm.llm_prices import TokenCounts, price_of

_USAGE_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def _assistant(mid, inp, cache_write, cache_read, out, parent=None, model="claude-opus-5"):
    usage = {
        "input_tokens": inp,
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cache_read,
        "output_tokens": out,
    }
    message = {"id": mid, "model": model, "usage": usage, "content": [{"type": "text", "text": "x"}]}
    return json.dumps({"type": "assistant", "message": message, "parent_tool_use_id": parent})


def _tool_result(t):
    return json.dumps({"type": "user", "message": {"content": [{"type": "tool_result"}]}, "timestamp": iso(t)})


def _result(output, usage_output=None, cost=None, duration_ms=None, **model_usage):
    return json.dumps(
        {
            "type": "result",
            "usage": {"output_tokens": usage_output if usage_output is not None else output},
            "modelUsage": {"claude-opus-5": {"outputTokens": output, **model_usage}},
            **({"total_cost_usd": cost} if cost is not None else {}),
            **({"duration_ms": duration_ms} if duration_ms is not None else {}),
        }
    )


LEDGER_USD = (1000 * 4 + 200 * 20 + 800 * 4 + 100 * 20 + 500 * 4 + 50 * 20) / 1e6
# t-perf's turns at $5/$25, cache writes at 1.25x and reads at 0.1x the input price, output 1500.
SPEC_USD = ((2000 + 2 + 3) * 5 + (10000 + 500 + 300) * 6.25 + (10000 + 10500) * 0.5 + 1500 * 25) / 1e6


# t-perf (running from T0 + 23): three turns, the first split over two lines, then a subagent turn.
TURNS = [
    json.dumps({"type": "system", "subtype": "init"}),
    _assistant("msg_a", 2000, 10000, 0, 3),
    _assistant("msg_a", 2000, 10000, 0, 8),
    _tool_result(T0 + 40),
    _assistant("msg_b", 2, 500, 10000, 1),
    _tool_result(T0 + 60),
    _assistant("msg_c", 3, 300, 10500, 2, parent="toolu_1"),
]


def _log(session_dir, lines, end="\n"):
    path = session_dir / "runs" / "specialist" / "t-perf" / "process.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + end, encoding="utf-8")
    return path


def _append(path, text):
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)


def _ledger_rows(outputs):
    usage = [(2000, 10000, 0), (2, 500, 10000), (3, 300, 10500)]
    return [
        {
            "component": "specialist",
            "role": "specialist",
            "model": "claude-opus-5",
            "task_id": "t-perf",
            "turn": i + 1,
            "input_tokens": inp,
            "cache_creation_input_tokens": cw,
            "cache_read_input_tokens": cr,
            "output_tokens": outputs[i],
            "latency_ms": 377000 if i == 2 else None,
            "status": "ok",
            "ts": iso(T0 + 400),
        }
        for i, (inp, cw, cr) in enumerate(usage)
    ]


def _add_ledger(session_dir, rows):
    # The fixture's ledger ends on a torn line; close it (an unparsable line) before appending.
    _append(session_dir / "reports" / "trace" / "llm_calls.jsonl", "\n" + "".join(json.dumps(r) + "\n" for r in rows))


def _set_state(session_dir, task_id, state):
    with sqlite3.connect(session_dir / "storage" / "coordinator.db") as db:
        db.execute("UPDATE tasks SET state=? WHERE task_id=?", (state, task_id))


def _live(session_dir, tokens=None):
    w = Watcher(session_dir, WatchOptions(sample_metrics=False), tokens=tokens or TokenTracker(session_dir))
    w.poll()
    return w


def _multibranch_usd(ev: dict) -> float:
    """What multi-branch records as the spend of *ev*'s tokens: the sheet's rates, reasoning billed as output."""
    counts = TokenCounts(
        input=ev["input"] or 0,
        output=(ev["output"] or 0) + (ev["reasoning"] or 0),
        cache_read=ev["cache_read"] or 0,
        cache_write=ev["cache_write"] or 0,
    )
    return price_of(ev["model"]).cost_usd(counts)


def test_each_call_costs_what_multi_branch_records(session_dir):
    w = Watcher(session_dir, WatchOptions(sample_metrics=False))
    w.poll()
    tokens, doc = w.snapshot()["tokens"], w.token_document()
    assert tokens["totals"]["calls"] == doc["totals"]["calls"] == 3 and doc["terms"] == TERMS
    assert tokens["pricing"]["source"] == doc["pricing"]["source"] == SHEET.source
    for ev in doc["events"]:
        assert ev["missing"] == [] and ev["cost_usd"] == pytest.approx(_multibranch_usd(ev), abs=1e-8)
    assert doc["totals"]["cost_usd"] == tokens["totals"]["cost_usd"] == pytest.approx(LEDGER_USD)
    assert tokens_mod.UNPRICED == ("no_model", "no_price", "no_search_price")
    w.stop()


@pytest.mark.parametrize(
    ("result", "outputs"),
    [
        (_result(1500, usage_output=1400), [None, None, 1500]),  # placeholders: the total goes on the last turn
        (_result(6), [3, 1, 2]),  # first-line counts that add up to the total are kept
    ],
)
def test_turns_match_the_runner_parser(tmp_path, result, outputs):
    path = tmp_path / "process.log"
    path.write_text("\n".join([*TURNS, result]) + "\n", encoding="utf-8")
    turns = StreamTurns(path)
    assert turns.read()
    ours = [{k: u[k] for k in _USAGE_KEYS} for u in turns.usages()]
    assert ours == parse_claude_stream_json_turn_usages(path)
    assert [u["output_tokens"] for u in ours] == outputs


def test_turns_are_read_incrementally_and_restart_on_rewrite(tmp_path):
    path = tmp_path / "process.log"
    path.write_text("\n".join(TURNS[:4]) + "\n" + TURNS[4][:30], encoding="utf-8")
    turns = StreamTurns(path)
    assert turns.read()
    assert [t["t"] for t in turns.turns] == [None]
    assert not turns.read()
    _append(path, TURNS[4][30:] + "\n" + "\n".join(TURNS[5:]) + "\n")
    assert turns.read()
    assert [t["t"] for t in turns.turns] == [None, T0 + 40, T0 + 60]
    assert [u["output_tokens"] for u in turns.usages()] == [None, None, None]
    path.write_text(TURNS[1] + "\n", encoding="utf-8")
    assert turns.read()
    assert len(turns.turns) == 1


def test_a_line_longer_than_a_chunk_is_dropped_not_stalled(tmp_path, monkeypatch):
    monkeypatch.setattr(tokens_mod, "_CHUNK", 400)
    assert len(TURNS[1]) < 400
    path = tmp_path / "process.log"
    path.write_text(json.dumps({"type": "user", "pad": "x" * 1000}) + "\n" + TURNS[1] + "\n", encoding="utf-8")
    turns = StreamTurns(path)
    for _ in range(6):
        turns.read()
    assert [t["input_tokens"] for t in turns.turns] == [2000]


def test_ledger_rows_sit_at_request_time_in_their_phase(session_dir):
    events = _live(session_dir).tokens.events()
    got = [(e["call_id"], e["t"], e["node"], e["phase"], e["total"], e["status"]) for e in events]
    assert got == [
        ("c1", T0 + 606, "orchestration", "PRELUDE", 1200, "ok"),
        ("c2", T0 + 632, "critic", "PRELUDE", 900, "ok"),
        ("c3", T0 + 718, "orchestration", "SYSTEM", 550, "error"),
    ]
    assert events[-1]["cumulative"]["total"] == 2650
    assert all(e["source"] == "ledger" for e in events)


def test_a_specialist_s_first_turn_no_log_line_matches_sits_at_its_start(session_dir):
    _add_ledger(session_dir, _ledger_rows([None, None, 1500]))  # a Codex specialist: no stream-JSON turns to match
    turns = [e for e in _live(session_dir).tokens.events() if e["task_id"] == "t-perf"]
    assert [(e["turn"], e["t"]) for e in turns] == [(1, T0 + 23), (2, T0 + 400), (3, T0 + 400)]


def test_live_specialist_turns_then_the_ledger_replaces_them(session_dir):
    _set_state(session_dir, "t-perf", "running")
    path = _log(session_dir, TURNS)
    w = _live(session_dir)
    live = [e for e in w.tokens.events() if e["task_id"] == "t-perf"]
    # Turn 1 at the task's start, later turns at the tool result they answer; output pending.
    assert [(e["t"], e["turn"], e["source"], e["output_pending"]) for e in live] == [
        (T0 + 23, 1, "live", True),
        (T0 + 40, 2, "live", True),
        (T0 + 60, 3, "live", True),
    ]
    assert [e["output"] for e in live] == [None, None, None]
    assert {e["node"] for e in live} == {"spec:perf_model"}
    version = w.version
    view = w.snapshot()["tokens"]
    assert view["totals"]["live"] == 3
    assert view["pending_tasks"] == ["t-perf"]

    _append(path, _result(1500) + "\n")
    w.poll()
    assert w.version > version
    live = [e for e in w.tokens.events() if e["task_id"] == "t-perf"]
    assert [e["output"] for e in live] == [None, None, 1500]
    assert not any(e["output_pending"] for e in live)

    _add_ledger(session_dir, _ledger_rows([None, None, 1500]))
    w.poll()
    events = w.tokens.events()
    spec = [e for e in events if e["task_id"] == "t-perf"]
    assert [(e["t"], e["source"]) for e in spec] == [(T0 + 23, "ledger"), (T0 + 40, "ledger"), (T0 + 60, "ledger")]
    assert len(events) == 6
    totals = w.snapshot()["tokens"]["totals"]
    spec_total = 2000 + 10000 + 2 + 500 + 10000 + 3 + 300 + 10500 + 1500
    assert totals["total"] == 2650 + spec_total
    assert (totals["calls"], totals["errors"], totals["live"], totals["output_pending"]) == (6, 1, 0, 0)
    assert totals["output_unreported"] == 0  # the earlier turns' outputs are in the last one's


def test_view_series_adds_up_and_replay_cuts_at_the_clock(session_dir):
    _log(session_dir, [*TURNS, _result(1500)])
    view = _live(session_dir).snapshot()["tokens"]
    series = view["series"]
    assert series["start"] == T0
    assert series["kinds"] == list(KINDS)
    assert (series["end"] - series["start"]) / series["step"] <= tokens_mod.MAX_BUCKETS
    for j, kind in enumerate(KINDS):
        assert sum(r[2 + j] for r in series["rows"]) == view["totals"][kind]
    assert sum(r[2 + len(KINDS)] for r in series["rows"]) == view["totals"]["calls"] == 6
    assert all(len(r) == 3 + 2 * len(KINDS) for r in series["rows"])
    assert view["by_node"]["spec:perf_model"]["calls"] == 3
    assert view["by_component"]["orchestration"] == {
        **dict.fromkeys(KINDS, 0),
        "input": 1500,
        "output": 250,
        "total": 1750,
        "calls": 2,
        "errors": 1,
        "web_searches": 0,
        "cost_usd": round((1500 * 4 + 250 * 20) / 1e6, 6),
        "unpriced": 0,
        "no_model": 0,
        "no_price": 0,
        "no_search_price": 0,
    }
    assert view["pricing"]["source"] == SHEET.source and view["totals"]["unpriced"] == 0

    w = Watcher(session_dir, WatchOptions(replay=True), tokens=TokenTracker(session_dir))
    w.poll()
    w.clock.seek(T0 + 620, paused=True)
    cut = w.snapshot()["tokens"]
    assert cut["totals"]["calls"] == 4
    assert cut["by_node"]["orchestration"]["calls"] == 1
    assert w.token_document()["upto"] == T0 + 620
    assert len(w.token_document()["events"]) == 4


def test_a_replay_reaches_the_turns_an_interrupted_specialist_logged_after_the_rest(session_dir):
    _set_state(session_dir, "t-perf", "running")  # the session stopped with it running, its log going on
    _log(session_dir, [*TURNS, _tool_result(T0 + 5000), _assistant("msg_d", 4, 0, 10800, 1)])
    w = Watcher(session_dir, WatchOptions(replay=True), tokens=TokenTracker(session_dir))
    w.poll()
    assert w.clock.end == T0 + 5000
    w.clock.seek(T0 + 6000, paused=True)
    turns = [e["turn"] for e in w.token_document()["events"] if e["task_id"] == "t-perf"]
    assert turns == [1, 2, 3, 4]


def test_a_quiet_live_run_ends_its_charts_at_a_specialist_s_turns_after_the_coordinator_stopped(session_dir):
    _set_state(session_dir, "t-perf", "running")  # as the coordinator left it; the specialist went on and finished
    _log(session_dir, [*TURNS, _tool_result(T0 + 5000), _assistant("msg_d", 4, 0, 10800, 1), _result(1500)])
    snap = _live(session_dir).snapshot()
    assert snap["timeline"]["end"] == snap["tokens"]["series"]["end"] == T0 + 5000


def test_every_figure_has_one_name_and_a_definition(session_dir):
    terms = _live(session_dir).snapshot()["tokens"]["terms"]
    assert terms == TERMS
    shown = {"total", "calls", "errors", "cache_hit", "cost_usd", "unpriced", "output_pending", "recent"}
    billed = (*KINDS, "web_search")
    assert set(billed) | shown <= set(terms)
    assert all(t["group"] and t["label"] and t["definition"] for t in terms.values())
    assert len({t["label"] for t in terms.values()}) == len(terms)
    # Short, as a hover hint shows them: one line each, but a billed category answers what it is, how it arises and
    # how it is billed, a line each.
    asks = ["What it is: ", "How it arises: ", "How it is billed: "]
    for key, t in terms.items():
        lines = t["definition"].split("\n")
        assert set(t) == {"group", "label", "definition"}, key
        if key in billed:
            assert [line[: len(ask)] for line, ask in zip(lines, asks)] == asks and len(lines) == 3, key
            assert all(len(line) <= 260 for line in lines), key
        else:
            assert len(lines) == 1 and len(t["definition"]) <= 120, key
    # No price is assumed: a definition says where one comes from, as the built-in sheet prices it.
    assert not re.search(r"\$\d|\d times for|by default", json.dumps(terms)), "a term names a price"
    assert not re.search(r"per_web_search|price list", json.dumps(terms)), "a term names what the sheet does not have"
    assert "is unpriced" in terms["web_search"]["definition"].split("\nHow it is billed: ")[1]
    assert "at the input price" in terms["cache_write"]["definition"]
    assert "no price per search" in terms["web_search"]["definition"]
    assert _live(session_dir).snapshot()["tokens"]["terms"] == TERMS
    assert "built-in price sheet" in terms["source"]["definition"]


def test_timeline_file_is_written_outside_the_session(session_dir, tmp_path):
    _set_state(session_dir, "t-perf", "running")
    _log(session_dir, TURNS)
    before = sorted((p.relative_to(session_dir), p.stat().st_mtime) for p in session_dir.rglob("*") if p.is_file())
    path = history_path(session_dir, tmp_path / "state", suffix=".tokens.json")
    w = _live(session_dir, tokens=TokenTracker(session_dir, path))
    doc = json.loads(path.read_text())
    assert doc["schema"] == tokens_mod.SCHEMA
    assert doc["session_id"] == session_dir.name
    first = doc["events"][0]
    assert first["ts"].endswith(".000Z")
    assert datetime.fromisoformat(first["ts"].replace("Z", "+00:00")).timestamp() == first["t"] == T0 + 23
    assert doc["events"][-1]["cumulative"]["total"] == doc["totals"]["total"]
    assert doc["totals"]["output_pending"] == 3
    assert not path.with_name(path.name + ".tmp").exists()
    w.stop()
    after = sorted((p.relative_to(session_dir), p.stat().st_mtime) for p in session_dir.rglob("*") if p.is_file())
    assert after == before
    with pytest.raises(ValueError):
        history_path(session_dir, session_dir / "state", suffix=".tokens.json")


def test_saves_are_throttled_and_an_empty_session_still_gets_a_file(tmp_path):
    session = tmp_path / "s"
    session.mkdir()
    tracker = TokenTracker(session, tmp_path / "out" / "s.tokens.json", write_every=60)
    assert tracker.save()
    assert json.loads(tracker.path.read_text())["events"] == []
    tracker.add_ledger({"component": "critic", "input_tokens": 5, "status": "ok", "ts": iso(T0)})
    tracker.poll([])
    assert not tracker.save()
    assert tracker.save(force=True)
    assert json.loads(tracker.path.read_text())["totals"]["input"] == 5
    assert not tracker.save(force=True)


def test_every_event_is_priced_and_the_costs_add_up(session_dir):
    _log(session_dir, [*TURNS, _result(1500)])
    w = _live(session_dir)
    events = w.tokens.events()
    assert [e["cost_usd"] for e in events if e["call_id"]] == pytest.approx([0.008, 0.0052, 0.003])
    view = w.snapshot()["tokens"]
    assert view["totals"]["cost_usd"] == pytest.approx(LEDGER_USD + SPEC_USD)
    assert events[-1]["cumulative"]["cost_usd"] == pytest.approx(LEDGER_USD + SPEC_USD)
    assert view["by_node"]["spec:perf_model"]["cost_usd"] == pytest.approx(SPEC_USD)
    assert view["totals"]["unpriced"] == 0
    rows, k = view["series"]["rows"], len(KINDS)
    assert sum(sum(r[3 + k :]) for r in rows) == pytest.approx(LEDGER_USD + SPEC_USD)
    cache_read = sum(r[3 + k + KINDS.index("cache_read")] for r in rows)
    assert cache_read == pytest.approx((10000 + 10500) * 0.5 / 1e6)
    pricing = view["pricing"]
    assert pricing["models"]["claude-opus-5"] == {
        "id": "claude-opus-5",
        "input": 5.0,
        "cache_write": 6.25,
        "cache_read": 0.5,
        "output": 25.0,
        "cache_basis": "listed",
        "spent_usd": pytest.approx(
            {
                "input": (2000 + 2 + 3) * 5 / 1e6,
                "cache_write": (10000 + 500 + 300) * 6.25 / 1e6,
                "cache_read": (10000 + 10500) * 0.5 / 1e6,
                "output": 1500 * 25 / 1e6,
                "reasoning": 0,
            }
        ),
    }
    assert pricing["models"]["claude-opus-5-5"]["id"] == "claude-opus-5-5"
    assert (pricing["source"], pricing["unpriced"], view["check"]) == (SHEET.source, {}, None)
    assert list(pricing) == ["source", "built_from", "unit", "models", "unpriced"]


@pytest.mark.parametrize("side_in_usage", [False, True], ids=["usage-of-the-turns", "usage-with-side-calls"])
def test_a_single_turn_specialist_counts_the_same_live_and_from_the_ledger(session_dir, side_in_usage):
    # One turn, then a side call on another model. Whether the CLI's result usage also counts the side call or only
    # the turns, the single cumulative row the runner writes for it must not change the totals.
    usage = {"input_tokens": 100 + 50 * side_in_usage, "cache_read_input_tokens": 1000, "output_tokens": 40}
    result = json.loads(_result(40, inputTokens=100, cacheReadInputTokens=1000))
    result["usage"] = usage
    result["modelUsage"]["claude-haiku-4-5"] = {"inputTokens": 50, "outputTokens": 5}
    log = _log(session_dir, [_assistant("msg_a", 100, 0, 1000, 1), json.dumps(result)])
    w = _live(session_dir)
    live = w.snapshot()["tokens"]["by_node"]["spec:perf_model"]
    row = producers.specialist_turn_record("t-perf", 1, parse_claude_stream_json_usage(log))
    _add_ledger(session_dir, [{**row.to_row(), "model": "claude-opus-5", "ts": iso(T0 + 400)}])
    w.poll()
    ledger = w.snapshot()["tokens"]["by_node"]["spec:perf_model"]
    assert [e["source"] for e in w.tokens.events() if e["task_id"] == "t-perf"] == ["ledger", "result"]
    assert {k: ledger[k] for k in (*KINDS, "total")} == {k: live[k] for k in (*KINDS, "total")}
    assert (ledger["input"], ledger["cache_read"], ledger["output"]) == (150, 1000, 45)


def test_a_log_whose_turns_report_zero_usage_counts_its_result_once(session_dir):
    # Some gateways stream every turn with zeroed usage and report the real counts only on the result line. The
    # runner then writes one cumulative row, and that row must be counted once, not the turns and the result as well.
    result = json.loads(_result(40, inputTokens=900, cacheReadInputTokens=1000))
    result["usage"] = {"input_tokens": 900, "cache_read_input_tokens": 1000, "output_tokens": 40}
    log = _log(session_dir, [_assistant("msg_a", 0, 0, 0, 0), _assistant("msg_b", 0, 0, 0, 0), json.dumps(result)])
    turns = StreamTurns(log)
    turns.read()
    assert parse_claude_stream_json_turn_usages(log) == turns.usages() == []
    w = _live(session_dir)
    assert not [e for e in w.tokens.events() if e["task_id"] == "t-perf"]
    row = producers.specialist_turn_record("t-perf", 1, parse_claude_stream_json_usage(log))
    _add_ledger(session_dir, [{**row.to_row(), "model": "claude-opus-5", "ts": iso(T0 + 400)}])
    w.poll()
    assert [e["source"] for e in w.tokens.events() if e["task_id"] == "t-perf"] == ["ledger"]
    node = w.snapshot()["tokens"]["by_node"]["spec:perf_model"]
    assert (node["input"], node["cache_read"], node["output"]) == (900, 1000, 40)


@pytest.mark.parametrize("state", ["cancelled", "failed", "succeeded"])
def test_a_specialist_that_ended_without_reporting_its_output_is_not_pending(session_dir, state):
    _set_state(session_dir, "t-perf", state)
    _log(session_dir, TURNS)  # its turns, and no result line: the ledger never gets them
    w = _live(session_dir)
    turns = [e for e in w.tokens.events() if e["task_id"] == "t-perf"]
    assert [(e["source"], e["output_pending"], e["output_unreported"]) for e in turns] == [("live", False, True)] * 3
    view = w.snapshot()["tokens"]
    assert (view["totals"]["output_pending"], view["totals"]["output_unreported"]) == (0, 3)
    assert (view["pending_tasks"], view["unreported_tasks"]) == ([], ["t-perf"])
    assert view["by_node"]["spec:perf_model"]["output"] == 0  # unknown, not counted


def test_a_specialist_s_ledger_rows_without_its_result_line_leave_its_output_unreported(session_dir):
    _set_state(session_dir, "t-perf", "running")
    log = _log(session_dir, TURNS)
    w = _live(session_dir)
    assert {e["output_pending"] for e in w.tokens.events() if e["task_id"] == "t-perf"} == {True}
    # Timed out: the runner writes the turns its log has, the outputs none, as the result line never came.
    outputs = [u["output_tokens"] for u in parse_claude_stream_json_turn_usages(log)]
    assert outputs == [None, None, None]
    _add_ledger(session_dir, _ledger_rows(outputs))
    w.poll()
    turns = [e for e in w.tokens.events() if e["task_id"] == "t-perf"]
    assert [(e["source"], e["output_pending"], e["output_unreported"]) for e in turns] == [("ledger", False, True)] * 3
    view = w.snapshot()["tokens"]
    assert (view["totals"]["output_unreported"], view["unreported_tasks"]) == (3, ["t-perf"])


def test_a_killed_specialist_s_recovered_ledger_rows_keep_their_counts(session_dir):
    # A gateway streamed every turn with zeroed usage and the run was killed before its result line; the runner
    # recovered each turn's counts, output too, from the CLI's own session record and wrote them to the ledger.
    _set_state(session_dir, "t-perf", "cancelled")
    _log(session_dir, [_assistant(f"msg_{i}", 0, 0, 0, 0) for i in "abc"])
    w = _live(session_dir)
    assert {e["output_unreported"] for e in w.tokens.events() if e["task_id"] == "t-perf"} == {True}
    _add_ledger(session_dir, _ledger_rows([20, 30, 40]))
    w.poll()
    turns = [e for e in w.tokens.events() if e["task_id"] == "t-perf"]
    assert [(e["source"], e["input"], e["cache_write"], e["output"], e["output_unreported"]) for e in turns] == [
        ("ledger", 2000, 10000, 20, False),
        ("ledger", 2, 500, 30, False),
        ("ledger", 3, 300, 40, False),
    ]
    assert w.snapshot()["tokens"]["unreported_tasks"] == []


def test_ledger_rows_that_name_no_model_take_it_from_the_log(session_dir):
    # Older runners wrote no model on a specialist's rows: its cost must not drop when they land.
    _log(session_dir, [*TURNS, _result(1500)])
    w = _live(session_dir)
    _add_ledger(session_dir, [{**r, "model": None} for r in _ledger_rows([None, None, 1500])])
    w.poll()
    spec = [e for e in w.tokens.events() if e["task_id"] == "t-perf"]
    assert [(e["source"], e["model"]) for e in spec] == [("ledger", "claude-opus-5")] * 3
    view = w.snapshot()["tokens"]
    assert view["by_node"]["spec:perf_model"]["cost_usd"] == pytest.approx(SPEC_USD)
    assert view["totals"]["unpriced"] == 0


def test_a_total_with_no_priced_call_has_no_cost(session_dir):
    _add_ledger(session_dir, [{**r, "model": None} for r in _ledger_rows([None, None, 1500])])  # and no log
    w = _live(session_dir)
    view = w.snapshot()["tokens"]
    nothing = view["by_model"][NO_MODEL]
    assert (nothing["calls"], nothing["unpriced"], nothing["no_model"], nothing["cost_usd"]) == (3, 3, 3, None)
    assert (view["totals"]["no_model"], view["totals"]["no_price"]) == (3, 0)
    assert view["by_node"]["spec:perf_model"]["cost_usd"] is None
    assert view["pricing"]["unpriced"] == {NO_MODEL: {"calls": 3, "total": nothing["total"]}}
    assert (view["totals"]["cost_usd"], view["totals"]["unpriced"]) == (pytest.approx(LEDGER_USD), 3)
    empty = TokenTracker(session_dir).view(T0)
    assert (empty["totals"]["cost_usd"], empty["recent"]["cost_usd"]) == (0.0, 0.0)


def test_openai_rows_take_cached_input_and_reasoning_out(tmp_path):
    tracker = TokenTracker(tmp_path)
    row = {"component": "specialist", "input_tokens": 24099, "cache_read_input_tokens": 11648, "status": "ok"}
    row |= {"output_tokens": 44, "reasoning_output_tokens": 10, "ts": iso(T0)}
    tracker.add_ledger({**row, "model": "gpt-5.6-terra"})
    tracker.add_ledger({**row, "model": "claude-opus-5", "reasoning_output_tokens": None, "ts": iso(T0 + 1)})
    tracker.poll([])
    gpt, claude = tracker.events()
    assert [gpt[k] for k in ("input", "cache_read", "output", "reasoning")] == [24099 - 11648, 11648, 34, 10]
    assert gpt["total"] == 24099 + 44
    assert gpt["cost_usd"] == pytest.approx((12451 * 1.36 + 11648 * 0.136 + 44 * 8.16) / 1e6)  # reasoning as output
    assert (claude["input"], claude["output"], claude["total"]) == (24099, 44, 24099 + 11648 + 44)


KIMI = "dvue-aoai-005-Kimi-K2.6"


@pytest.mark.parametrize(
    ("row", "split"),
    [
        # The scorer only talks the OpenAI protocol: 100 completion tokens, 80 of them reasoning, are 100 in all.
        ({"component": "proposal_scorer", "model": KIMI, "reasoning_output_tokens": 80}, True),
        ({"component": "critic", "model": KIMI, "reasoning_output_tokens": 80}, True),
        # The critic over the Anthropic protocol reports no reasoning, and its input excludes the cached prefix.
        ({"component": "critic", "model": "claude-opus-5", "cache_read_input_tokens": 300}, False),
        ({"component": "orchestration", "model": KIMI, "cache_read_input_tokens": 300}, False),
        ({"component": "specialist", "model": "gpt-5.4", "cache_read_input_tokens": 300}, True),
    ],
    ids=["scorer", "critic-openai", "critic-anthropic", "role-on-claude-protocol", "codex-backend"],
)
def test_the_writer_tells_whether_a_row_counts_by_the_openai_protocol(tmp_path, row, split):
    tracker = TokenTracker(tmp_path)
    tracker.add_ledger({"input_tokens": 1000, "output_tokens": 100, "status": "ok", "ts": iso(T0), **row})
    tracker.poll([])
    (ev,) = tracker.events()
    cached, reasoning = row.get("cache_read_input_tokens") or 0, row.get("reasoning_output_tokens") or 0
    if split:
        assert (ev["input"], ev["output"]) == (1000 - cached, 100 - reasoning)
        assert ev["total"] == 1000 + 100
    else:
        assert (ev["input"], ev["output"]) == (1000, 100)
        assert ev["total"] == 1000 + cached + 100 + reasoning


def test_a_model_with_no_price_is_unpriced_and_the_rest_still_cost(session_dir):
    _add_ledger(session_dir, [{"component": "critic", "model": "Qwen3.8-27B", "input_tokens": 70, "ts": iso(T0 + 640)}])
    w = _live(session_dir)
    view = w.snapshot()["tokens"]
    assert view["totals"]["cost_usd"] == pytest.approx(LEDGER_USD)
    assert view["totals"]["unpriced"] == 1
    qwen = view["by_model"]["Qwen3.8-27B"]
    assert (qwen["cost_usd"], qwen["no_price"], qwen["no_model"]) == (None, 1, 0)
    critic = view["by_node"]["critic"]
    assert (critic["cost_usd"], critic["unpriced"]) == (pytest.approx((800 * 4 + 100 * 20) / 1e6), 1)
    assert view["totals"]["no_price"] == 1 and view["totals"]["no_model"] == 0
    assert view["pricing"]["unpriced"] == {"Qwen3.8-27B": {"calls": 1, "total": 70}}
    assert w.token_document()["events"][-1]["cumulative"]["cost_usd"] == pytest.approx(LEDGER_USD)


def test_a_call_that_searched_is_unpriced_as_the_sheet_has_no_price_per_search(tmp_path):
    tracker = TokenTracker(tmp_path)
    row = {"component": "critic", "model": "claude-opus-5", "input_tokens": 10, "cache_read_input_tokens": 5}
    tracker.add_ledger({**row, "web_searches": 2, "status": "ok", "ts": iso(T0)})
    tracker.add_ledger({**row, "status": "ok", "ts": iso(T0 + 1)})
    tracker.poll([])
    searched, plain = tracker.events()
    assert (searched["missing"], searched["cost_usd"]) == (["web_search"], None)
    assert (plain["missing"], plain["cost_usd"]) == ([], pytest.approx((10 * 5 + 5 * 0.5) / 1e6))
    totals = tracker.view(T0 + 2)["totals"]
    assert [totals[k] for k in ("unpriced", "no_search_price", "web_searches")] == [1, 1, 2]
    assert tokens_mod.missing({**searched, "model": "claude-opus-9"}) == ["price"]
    assert tokens_mod.missing({**searched, "model": None}) == ["model"]
    assert tokens_mod.call_cost(searched) is None


def test_side_calls_on_the_result_line_are_counted_once_and_match_the_cli_cost(session_dir):
    side_usd = (400 * 5 + 1000 * 0.5) / 1e6
    result = _result(
        1500,
        cost=SPEC_USD + side_usd,
        duration_ms=50_000,
        inputTokens=2005 + 400,
        cacheCreationInputTokens=10800,
        cacheReadInputTokens=20500 + 1000,
    )
    _log(session_dir, [*TURNS, result])
    w = _live(session_dir)
    side = [e for e in w.tokens.events() if e["source"] == "result"]
    assert len(side) == 1
    assert (side[0]["task_id"], side[0]["turn"], side[0]["node"]) == ("t-perf", None, "spec:perf_model")
    assert [side[0][k] for k in ("input", "cache_write", "cache_read", "web_searches")] == [400, 0, 1000, 0]
    assert side[0]["t"] == T0 + 23 + 50  # the task's start plus the CLI's duration, never a file time
    assert side[0]["cost_usd"] == pytest.approx(side_usd)
    view = w.snapshot()["tokens"]
    assert view["totals"]["calls"] == 6  # the side calls are not one call more
    assert view["check"]["tasks"] == 1
    assert view["check"]["claude_cli_usd"] == pytest.approx(SPEC_USD + side_usd)
    assert view["check"]["priced_usd"] == pytest.approx(view["check"]["claude_cli_usd"])
    assert sum(r[2 + len(KINDS)] for r in view["series"]["rows"]) == 6

    _add_ledger(session_dir, _ledger_rows([None, None, 1500]))
    w.poll()
    spec = [e for e in w.tokens.events() if e["task_id"] == "t-perf"]
    assert [(e["source"], e["t"]) for e in spec][-1] == ("result", T0 + 400)  # the ledger's completion stamp
    assert [e["source"] for e in spec] == ["ledger", "ledger", "ledger", "result"]
    assert w.snapshot()["tokens"]["by_node"]["spec:perf_model"]["cost_usd"] == pytest.approx(SPEC_USD + side_usd)


def test_unpriced_side_usage_counts_toward_its_specialists_call_not_as_one_more(session_dir):
    # One turn on a model the sheet does not list, then searches on it: both events are unpriced, one call.
    usage = {"inputTokens": 100, "cacheReadInputTokens": 1000, "outputTokens": 1, "webSearchRequests": 2}
    result = {"type": "result", "modelUsage": {"Qwen3.8-27B": usage}}
    _log(session_dir, [_assistant("msg_a", 100, 0, 1000, 1, model="Qwen3.8-27B"), json.dumps(result)])
    w = _live(session_dir)
    assert [e["source"] for e in w.tokens.events() if e["task_id"] == "t-perf"] == ["live", "result"]
    spec = w.snapshot()["tokens"]["by_node"]["spec:perf_model"]
    assert {k: spec[k] for k in ("calls", "unpriced", "no_price", "web_searches", "cost_usd")} == {
        "calls": 1,
        "unpriced": 1,
        "no_price": 1,
        "web_searches": 2,
        "cost_usd": None,
    }


def test_side_calls_on_another_model_are_priced_at_its_own_rates(session_dir):
    # t-perf's turns ran on Opus 5, its side calls on Haiku 5.5, which wrote 600 of the 1500 output tokens.
    opus_usd, haiku_usd = SPEC_USD - 600 * 25 / 1e6, (400 * 0.1 + 600 * 0.5) / 1e6
    opus = {"inputTokens": 2005, "cacheCreationInputTokens": 10800, "cacheReadInputTokens": 20500, "outputTokens": 900}
    result = {
        "type": "result",
        "modelUsage": {"claude-opus-5": opus, "claude-haiku-5-5": {"inputTokens": 400, "outputTokens": 600}},
        "total_cost_usd": opus_usd + haiku_usd,
        "duration_ms": 50_000,
    }
    _log(session_dir, [*TURNS, json.dumps(result)])
    w = _live(session_dir)

    def check(sources):
        spec = [e for e in w.tokens.events() if e["task_id"] == "t-perf"]
        assert [(e["source"], e["model"], e["output"]) for e in spec] == [
            *zip(sources, ["claude-opus-5"] * 3, [None, None, 900]),
            ("result", "claude-haiku-5-5", 600),
        ]
        view = w.snapshot()["tokens"]
        assert view["by_model"]["claude-haiku-5-5"]["cost_usd"] == pytest.approx(haiku_usd)
        assert view["by_node"]["spec:perf_model"]["cost_usd"] == pytest.approx(opus_usd + haiku_usd)
        assert view["by_node"]["spec:perf_model"]["output"] == 1500
        assert view["check"]["priced_usd"] == pytest.approx(view["check"]["claude_cli_usd"])

    check(["live"] * 3)
    _add_ledger(session_dir, _ledger_rows([None, None, 1500]))  # the runner's session total on the last turn
    w.poll()
    check(["ledger"] * 3)


def test_a_subagent_turn_on_another_model_is_priced_at_its_rates_though_the_ledger_names_the_specialists(session_dir):
    # Turn 2 is a subagent's on Haiku 5.5, which also wrote 100 of the 1500 output tokens; the ledger puts the
    # specialist's claude-opus-5 on every turn.
    turns = [
        *TURNS[:4],
        _assistant("msg_b", 2, 500, 10000, 1, parent="toolu_1", model="claude-haiku-5-5"),
        _tool_result(T0 + 60),
        _assistant("msg_c", 3, 300, 10500, 2),
    ]
    opus_usd = ((2000 + 3) * 5 + (10000 + 300) * 6.25 + 10500 * 0.5 + 1400 * 25) / 1e6
    haiku_usd = (2 * 0.1 + 500 * 0.125 + 10000 * 0.01 + 100 * 0.5) / 1e6
    opus = {"inputTokens": 2003, "cacheCreationInputTokens": 10300, "cacheReadInputTokens": 10500, "outputTokens": 1400}
    small = {"inputTokens": 2, "cacheCreationInputTokens": 500, "cacheReadInputTokens": 10000, "outputTokens": 100}
    result = {
        "type": "result",
        "modelUsage": {"claude-opus-5": opus, "claude-haiku-5-5": small},
        "total_cost_usd": opus_usd + haiku_usd,
        "duration_ms": 50_000,
    }
    _log(session_dir, [*turns, json.dumps(result)])
    w = _live(session_dir)

    def check(source):
        spec = [e for e in w.tokens.events() if e["task_id"] == "t-perf"]
        assert [(e["source"], e["model"], e["output"]) for e in spec] == [
            (source, "claude-opus-5", None),
            (source, "claude-haiku-5-5", None),
            (source, "claude-opus-5", 1400),
            ("result", "claude-haiku-5-5", 100),
        ]
        view = w.snapshot()["tokens"]
        assert view["by_model"]["claude-opus-5"]["cost_usd"] == pytest.approx(opus_usd)
        assert view["by_model"]["claude-haiku-5-5"]["cost_usd"] == round(haiku_usd, 6)  # to a millionth of a dollar
        assert view["check"]["priced_usd"] == pytest.approx(view["check"]["claude_cli_usd"])

    check("live")
    _add_ledger(session_dir, _ledger_rows([None, None, 1500]))
    w.poll()
    check("ledger")


def test_the_running_cost_is_null_until_a_call_is_priced_and_a_call_costing_nothing_counts(tmp_path):
    tracker = TokenTracker(tmp_path)
    for i, (model, tokens) in enumerate((("Qwen3.8-27B", 1000), ("claude-opus-5", 0), ("claude-opus-5", 1000))):
        tracker.add_ledger({"component": "critic", "model": model, "input_tokens": tokens, "ts": iso(T0 + i)})
    tracker.poll([])
    unpriced, nothing, priced = tracker.events()
    assert (unpriced["cost_usd"], unpriced["cumulative"]["cost_usd"]) == (None, None)
    assert (nothing["cost_usd"], nothing["cumulative"]["cost_usd"]) == (0, 0)
    assert priced["cumulative"]["cost_usd"] == pytest.approx(1000 * 5 / 1e6)


def test_a_model_s_cost_by_kind_is_its_priced_calls_and_adds_up_to_its_cost(tmp_path):
    tracker = TokenTracker(tmp_path)  # the call that searched is unpriced: the sheet has no price per search
    for i, searches in enumerate((0, 1)):
        row = {"component": "critic", "model": "claude-opus-5", "input_tokens": 1000, "output_tokens": 100}
        tracker.add_ledger({**row, "web_searches": searches, "ts": iso(T0 + i)})
    tracker.poll([])
    for doc in (tracker.view(T0 + 10), tracker.document()):
        spent, model = doc["pricing"]["models"]["claude-opus-5"]["spent_usd"], doc["by_model"]["claude-opus-5"]
        assert spent == {"input": 0.005, "cache_write": 0, "cache_read": 0, "output": 0.0025, "reasoning": 0}
        assert (sum(spent.values()), model["unpriced"]) == (pytest.approx(model["cost_usd"]), 1)


def test_a_write_error_disables_the_file_not_the_watcher(session_dir, tmp_path, caplog):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    tracker = TokenTracker(session_dir, blocker / "s.tokens.json")
    w = _live(session_dir, tokens=tracker)
    assert tracker.path is None
    assert "token timeline" in caplog.text
    assert w.snapshot()["tokens"]["totals"]["calls"] == 3
