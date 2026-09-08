# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the structured GEAK HTML report.

The contract these pin is mostly about honesty: a quantity that was not recorded
must render as "not measured" and never as zero, a role that cannot be read must
render as unlabelled and never as a guess, and the coverage banner must always
appear so a reader knows what the numbers exclude.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.tools import render_geak_html_report as R


def _call(**kw):
    row = {
        "phase": "P8 HeadKernel h0",
        "agent": "a1",
        "call_index": 0,
        "ts": "2026-09-08T10:00:00.000Z",
        "model": "claude-opus-5",
        "isl": 1000,
        "osl": 10,
        "usd": 0.5,
        "tools": ["Bash"],
        "prompt": "You are Engineer r1_d2 (specialty=memory) for round 1.",
        "output": "",
        "thinking": 0,
        "dt_s": 1.0,
    }
    row.update(kw)
    return row


@pytest.fixture
def reports(tmp_path: Path) -> Path:
    d = tmp_path / "reports"
    d.mkdir()
    rows = [_call(call_index=i, isl=1000 + 100 * i) for i in range(12)]
    rows += [_call(phase="P4 ConfigSweep", agent="a2", call_index=i, usd=0.1) for i in range(3)]
    (d / R.CALLS_FILENAME).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / R.OUTCOME_FILENAME).write_text(
        json.dumps(
            {
                "run_id": "e2e_test_run",
                "stages": [
                    {"kind": "reference", "phase": "P1 Setup+Baseline", "after_tok_s": 100.0, "present": True},
                    {
                        "kind": "delta",
                        "phase": "P4 ConfigSweep",
                        "before_tok_s": 100.0,
                        "after_tok_s": 150.0,
                        "delta_pct": 50.0,
                        "present": True,
                    },
                    {
                        "kind": "kernel",
                        "phase": "HeadKernel",
                        "task": "h0_gemm_task",
                        "present": True,
                        "isolated_speedup": 1.0,
                        "amdahl_ceiling_e2e_pct": 0.0,
                        "pct_gpu_time": 23.13,
                    },
                ],
                "summary": {"observed_delta_pct_first_to_last": 50.0, "compounded_is_estimate": False},
            }
        )
    )
    return d


def test_load_calls_skips_junk_lines(tmp_path: Path):
    """A truncated or malformed line must not take the whole report down."""
    path = tmp_path / "c.jsonl"
    path.write_text('{"a":1}\nnot json\n\n[1,2]\n{"b":2}\n')
    assert R.load_calls(path) == [{"a": 1}, {"b": 2}]


def test_load_outcome_returns_none_when_absent(tmp_path: Path):
    """Absent is None, which the report renders as 'not measured'."""
    assert R.load_outcome(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{{{")
    assert R.load_outcome(bad) is None


def test_num_rejects_bools_and_non_finite():
    """A bool is not a token count and NaN is not a cost."""
    assert R._num(True) == 0.0
    assert R._num(float("nan")) == 0.0
    assert R._num(float("inf")) == 0.0
    assert R._num(3) == 3.0


def test_derive_role_reads_a_role_and_admits_when_it_cannot():
    """Roles are derived, and an unreadable one is named unlabelled, never guessed."""
    role = R.derive_role("You are Engineer r2_d1 (specialty=compute) for round 2.")
    assert role["role"] == "Engineer r2_d1"
    assert role["specialty"] == "compute"
    assert role["round"] == 2
    assert R.derive_role("## PROCESS SAFETY ...")["role"] == R.UNLABELLED
    assert R.derive_role("")["role"] == R.UNLABELLED


def test_agent_records_fold_calls_and_order_by_spend(reports: Path):
    agents = R.agent_records(R.load_calls(reports / R.CALLS_FILENAME))
    assert [a["agent"] for a in agents] == ["a1", "a2"]
    top = agents[0]
    assert top["calls"] == 12
    assert top["usd"] == pytest.approx(6.0)
    assert top["first_isl"] == 1000 and top["last_isl"] == 2100
    assert top["tools"] == {"Bash": 12}


def test_cost_curve_excludes_conversations_too_short_to_bucket(reports: Path):
    """Three calls cannot be split into ten positions, so that agent is left out."""
    rows = R.load_calls(reports / R.CALLS_FILENAME)
    agents = R.agent_records(rows)
    phases = {p["phase"]: p for p in R.phase_records(rows, agents)}
    assert len(phases["P8 HeadKernel h0"]["cost_curve"]) == R.DECILES
    assert phases["P4 ConfigSweep"]["cost_curve"] == []


def test_concentration_is_cumulative_and_ends_at_100(reports: Path):
    rows = R.load_calls(reports / R.CALLS_FILENAME)
    phases = R.phase_records(rows, R.agent_records(rows))
    for phase in phases:
        points = phase["concentration"]
        assert points[-1]["cum_pct"] == pytest.approx(100.0)
        assert all(a["cum_pct"] <= b["cum_pct"] + 1e-9 for a, b in zip(points, points[1:]))


def test_concentration_of_a_free_phase_is_empty():
    """No spend means no share to attribute — an empty list, not a divide by zero."""
    assert R.concentration([{"usd": 0.0}]) == []


def test_join_outcome_matches_kernels_by_head_token(reports: Path):
    rows = R.load_calls(reports / R.CALLS_FILENAME)
    joined = R.join_outcome(R.phase_records(rows, R.agent_records(rows)), R.load_outcome(reports / R.OUTCOME_FILENAME))
    by = {p["phase"]: p["outcome"] for p in joined}
    assert by["P8 HeadKernel h0"]["kind"] == "kernel"
    assert by["P8 HeadKernel h0"]["tasks"][0]["task"] == "h0_gemm_task"
    assert by["P4 ConfigSweep"]["kind"] == "delta"
    assert by["P4 ConfigSweep"]["gain_pct"] == 50.0


def test_join_outcome_without_an_outcome_file_is_none_not_zero(reports: Path):
    """The distinction the whole report rests on: unmeasured is not zero."""
    rows = R.load_calls(reports / R.CALLS_FILENAME)
    joined = R.join_outcome(R.phase_records(rows, R.agent_records(rows)), None)
    assert all(p["outcome"] is None for p in joined)


def test_pct_says_not_measured_for_none():
    assert "not measured" in R._pct(None)
    assert R._pct(1.5) == "+1.50%"


def test_sparkline_of_no_data_says_so():
    assert "no data" in R._sparkline([])
    assert "<svg" in R._sparkline([1.0, 2.0, 3.0])


def test_render_produces_a_self_contained_document(reports: Path):
    """No external fetches: the page must stand alone on shared storage."""
    doc = R.render(reports / R.CALLS_FILENAME, reports / R.OUTCOME_FILENAME)
    assert doc.startswith("<!doctype html>")
    assert "http://" not in doc and "https://" not in doc
    assert "<script src" not in doc and "<link" not in doc
    assert "Coverage" in doc
    assert "e2e_test_run" in doc


def test_render_without_an_outcome_says_the_half_is_missing(reports: Path):
    (reports / R.OUTCOME_FILENAME).unlink()
    doc = R.render(reports / R.CALLS_FILENAME, reports / R.OUTCOME_FILENAME)
    assert "Not available" in doc
    assert "what each phase cost but not what it bought" in doc


def test_unlabelled_agents_are_flagged_in_coverage(tmp_path: Path):
    d = tmp_path / "reports"
    d.mkdir()
    rows = [_call(prompt="## PROCESS SAFETY", call_index=i) for i in range(3)]
    (d / R.CALLS_FILENAME).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    doc = R.render(d / R.CALLS_FILENAME, d / R.OUTCOME_FILENAME)
    assert R.UNLABELLED in doc
    assert "could not be given a role" in doc


def test_coverage_counts_unpriced_and_untimed_calls(tmp_path: Path):
    rows = [_call(usd=0, dt_s=0), _call(call_index=1)]
    agents = R.agent_records(rows)
    cov = R.coverage(rows, agents)
    assert cov["unpriced_calls"] == 1 and cov["untimed_calls"] == 1


def test_agent_key_is_dom_safe(reports: Path):
    rows = R.load_calls(reports / R.CALLS_FILENAME)
    for agent in R.agent_records(rows):
        key = R.agent_key(agent)
        assert key.replace("_", "").replace("-", "").isalnum()


def test_call_payload_is_capped_and_reports_the_remainder(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(R, "MAX_CALLS_EMBEDDED", 2)
    rows = [_call(call_index=i) for i in range(5)]
    agents = R.agent_records(rows)
    assert len(agents[0]["rows"]) == 2
    assert agents[0]["rows_truncated"] == 3
    assert '"more":3' in R._agent_payload(agents)


def test_main_writes_the_file(reports: Path, tmp_path: Path, capsys):
    out = tmp_path / "out" / "r.html"
    assert R.main(["--reports-dir", str(reports), "-o", str(out)]) == 0
    assert out.is_file() and out.stat().st_size > 1000
    assert "wrote" in capsys.readouterr().out


def test_main_refuses_a_directory_with_no_ledger(tmp_path: Path, capsys):
    assert R.main(["--reports-dir", str(tmp_path)]) == 2
    assert R.CALLS_FILENAME in capsys.readouterr().err
