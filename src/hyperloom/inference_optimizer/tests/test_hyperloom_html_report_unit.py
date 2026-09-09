# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the Hyperloom session HTML report.

The report's value is that a reader can trust a number on it, so the tests are
mostly about honesty rather than layout: an unmeasured outcome must not render
as zero, an unpriced call must not be counted as free, and gain must reach a
phase only through the declared source map.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.tools import render_hyperloom_html_report as H
from hyperloom.inference_optimizer.tools.dump_llm_call_report import build_tree
from hyperloom.inference_optimizer.tools.render_hyperloom_html_report import call_rows


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _call(**over) -> dict:
    row = {
        "call_id": "c1",
        "phase": "FRAMEWORK_AGENT",
        "task_path": "explore/attempt-1",
        "model": "claude-opus-5",
        "isl": 1000,
        "osl": 100,
        "input_tokens": 900,
        "output_tokens": 100,
        "cache_read_input_tokens": 100,
        "cost_usd": 0.5,
        "cost_source": "priced",
        "latency_ms": 2000,
        "api_call_index": 1,
        "turn": 1,
        "tool_call_count": 1,
    }
    row.update(over)
    return row


@pytest.fixture()
def session(tmp_path: Path) -> Path:
    """A minimal but realistic session: two phases, a detailed turn and a bare one."""
    sd = tmp_path / "model" / "20260101T000000Z-abcd1234"
    trace = sd / "reports" / "trace"
    _write(
        trace / "llm_calls.jsonl",
        [
            _call(call_id="t1"),
            _call(call_id="t2", phase="PRELUDE", task_path="baseline", cost_usd=0.25),
        ],
    )
    # t1 expanded into two API calls; t2 did not, so it is counted from its turn row.
    _write(
        trace / "llm_calls_detail.jsonl",
        [
            _call(call_id="t1", api_call_index=1, cost_usd=0.30),
            _call(call_id="t1", api_call_index=2, cost_usd=0.40, isl=2000),
        ],
    )
    (sd / H.BREAKDOWN_FILENAME).write_text(
        json.dumps(
            {
                "metadata": {"session": {"session_id": "sess-42", "code_revision": "abc1234"}},
                "model_info": {"model_type": "qwen3"},
                "outcome": {
                    "baseline": {"throughput_tok_s_per_gpu": 100.0},
                    "final": {"throughput_tok_s_per_gpu": 120.0, "gain_pct": 20.0, "action_path": ["explore:x"]},
                    "status": "completed",
                    "stop_reason": "target_reached",
                    "stage_reached": "close",
                    "validation": {
                        "unattributed_gain_pct": 0.0,
                        "reconciliation_gap_pct": 0.0,
                        "attribution": {
                            "available": True,
                            "by_source": {"framework_agent": {"keep_count": 1, "total_gain_pct": 20.0}},
                        },
                    },
                },
                "phase_timeline": [
                    {"phase": "FRAMEWORK_AGENT", "action": "env", "decision": "KEEP", "key_metric": 120.0},
                    {"phase": "PRELUDE", "action": "baseline", "decision": "KEEP", "key_metric": 100.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    return sd


def test_spend_is_counted_once_per_call_not_once_per_turn(session: Path):
    """A detailed turn is counted from its detail rows only; a bare turn from itself."""
    turns, details = H.load_ledgers(session)
    rows = H.call_rows(turns, details)
    assert len(rows) == 3  # two detail rows for t1, plus t2 which had none
    assert round(sum(H.num(r["cost_usd"]) for r in rows), 6) == 0.95


def test_a_phase_the_map_does_not_cover_is_unattributed_not_credited(session: Path):
    """PRELUDE spends real money and is credited with no gain, and the page says so."""
    page = H.render(session)
    turns, details = H.load_ledgers(session)
    root, _ = H.build_tree(turns, details)
    joined = H.join_gain(H.phase_records(root), H.outcome_ladder(H.load_breakdown(session / H.BREAKDOWN_FILENAME)))
    by_phase = {r["phase"]: r for r in joined}
    assert by_phase["FRAMEWORK_AGENT"]["gain_pct"] == 20.0
    assert by_phase["FRAMEWORK_AGENT"]["source"] == "framework_agent"
    assert by_phase["PRELUDE"]["gain_pct"] is None
    assert "not attributed" in page


def test_a_missing_breakdown_says_not_recorded_rather_than_zero(session: Path):
    """No outcome file must never render as a 0% gain."""
    (session / H.BREAKDOWN_FILENAME).unlink()
    page = H.render(session)
    assert "not recorded" in page
    assert "+0.00%" not in page


def test_unpriced_calls_are_declared_a_floor(session: Path):
    """A call with no price is excluded from the total and confessed in the banner."""
    trace = session / "reports" / "trace"
    _write(
        trace / "llm_calls.jsonl",
        [_call(call_id="t1"), _call(call_id="t3", cost_source="unavailable", cost_usd=0.0)],
    )
    turns, details = H.load_ledgers(session)
    _, cov = H.build_tree(turns, details)
    assert cov["calls_unpriced"] >= 1
    assert "left out of the USD total" in H.render(session)


def test_growth_curve_drops_buckets_too_small_to_have_a_median():
    """A single call is not a median, so its bucket is not plotted."""
    rows = [_call(api_call_index=1, isl=10) for _ in range(3)] + [_call(api_call_index=2, isl=99)]
    curve = H.growth_curve(rows)
    assert [point["index"] for point in curve] == [1]
    assert curve[0]["isl_median"] == 10


def test_model_mix_orders_by_spend(session: Path):
    """The dearest model is first, and an unrecorded model is named, not dropped."""
    mix = H.model_mix([_call(model="a", cost_usd=1.0), _call(model=None, cost_usd=2.0)])
    assert [entry["model"] for entry in mix] == ["unrecorded", "a"]


def test_identity_prefers_what_the_run_recorded(session: Path):
    """The recorded session id wins over the directory name it happens to sit in."""
    ident = H.session_identity(session, H.load_breakdown(session / H.BREAKDOWN_FILENAME))
    assert ident["session_id"] == "sess-42"
    assert ident["model"] == "qwen3"
    assert H.session_identity(session, None)["session_id"] == session.name


def test_the_page_is_pure_ascii_and_self_contained(session: Path):
    """It is read off shared storage through viewers that ignore the charset."""
    page = H.render(session)
    assert page.isascii()
    assert "http://" not in page and "https://" not in page
    assert page.startswith("<!doctype html>")


def test_every_phase_of_the_session_appears_not_just_the_kernel_one(session: Path):
    """The point of this report: a session's bill is not spent in one phase."""
    page = H.render(session)
    assert "FRAMEWORK_AGENT" in page and "PRELUDE" in page


def test_main_refuses_a_session_with_no_ledger(tmp_path: Path, capsys):
    """Exit 2 and say why, rather than writing a confident empty page."""
    assert H.main(["--session-dir", str(tmp_path), "-o", str(tmp_path / "x.html")]) == 2
    assert "no llm_calls.jsonl" in capsys.readouterr().err


def test_main_writes_the_page(session: Path, tmp_path: Path):
    """The CLI writes where it is told and reports the size."""
    out = tmp_path / "out" / "report.html"
    assert H.main(["--session-dir", str(session), "-o", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_turn_and_detail_without_call_ids_are_one_call_not_two() -> None:
    """A producer that wrote no call id must not have its calls counted twice.

    The specialist and critic backends recorded neither the turn row's
    ``call_id`` nor the detail row's. Joining on the raw field made the detail
    row look orphaned and its turn look undetailed, so the same API call landed
    in the totals twice -- observed on a real session as 11,898,833 input
    tokens counted a second time.
    """
    turn = {
        "session_id": "s1",
        "task_path": "specialist/abc/turn-1",
        "turn": 1,
        "phase": "PRELUDE",
        "input_tokens": 100,
        "output_tokens": 10,
        "cost_usd": None,
    }
    detail = dict(turn)
    detail["api_call_index"] = None

    root, cov = build_tree([turn], [detail])
    totals = root.roll_up()

    assert totals.calls == 1, "the turn and its detail row are one API call"
    assert totals.isl == 100
    assert totals.osl == 10
    assert cov["detail_rows_orphaned"] == 0
    assert cov["turns_with_detail"] == 1
    assert cov["turns_without_detail"] == 0

    assert len(call_rows([turn], [detail])) == 1, "the flat row list must agree with the tree"


def test_unrelated_rows_without_call_ids_still_count_separately() -> None:
    """The fallback join must key on the turn, not lump every id-less row together."""
    a = {"session_id": "s1", "task_path": "specialist/abc/turn-1", "turn": 1, "input_tokens": 100}
    b = {"session_id": "s1", "task_path": "specialist/xyz/turn-1", "turn": 1, "input_tokens": 7}

    root, _ = build_tree([a, b], [])
    assert root.roll_up().calls == 2
    assert root.roll_up().isl == 107
