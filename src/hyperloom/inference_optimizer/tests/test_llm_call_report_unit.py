# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.tools import dump_llm_call_report as rpt


def _turn(call_id: str, phase: str, task_path: str, **kw) -> dict:
    row = {
        "session_id": "s1",
        "component": "orchestration",
        "call_id": call_id,
        "phase": phase,
        "task_path": task_path,
        "model": "m",
        "input_tokens": 100,
        "output_tokens": 20,
        "reasoning_output_tokens": 5,
        "latency_ms": 1000,
        "cost_usd": 1.0,
        "cost_input_usd": 0.6,
        "cost_output_usd": 0.3,
        "cost_thinking_usd": 0.1,
        "cost_source": "derived",
        "status": "ok",
    }
    row.update(kw)
    return row


def _detail(call_id: str, index: int, phase: str, task_path: str, **kw) -> dict:
    row = {
        "session_id": "s1",
        "component": "orchestration",
        "call_id": call_id,
        "api_call_index": index,
        "phase": phase,
        "task_path": task_path,
        "model": "m",
        "isl": 100,
        "osl": 25,
        "input_tokens": 100,
        "output_tokens": 20,
        "reasoning_output_tokens": 5,
        "latency_ms": 500,
        "thinking_ms": 100,
        "output_ms": 400,
        "cost_usd": 0.5,
        "cost_source": "derived",
        "tool_calls": [{"tool": "Bash"}],
        "status": "ok",
    }
    row.update(kw)
    return row


def _write(session_dir: Path, turns: list[dict], details: list[dict], ext: dict | None = None) -> None:
    trace = session_dir / "reports" / "trace"
    (trace / "ext").mkdir(parents=True, exist_ok=True)
    (trace / "llm_calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in turns))
    (trace / "llm_calls_detail.jsonl").write_text("".join(json.dumps(r) + "\n" for r in details))
    for name, rows in (ext or {}).items():
        (trace / "ext" / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_a_detailed_turn_is_counted_from_its_detail_rows_only(tmp_path: Path):
    sd = tmp_path / "s"
    _write(
        sd,
        [_turn("c1", "PROFILE", "profile/plan")],
        [_detail("c1", 0, "PROFILE", "profile/plan"), _detail("c1", 1, "PROFILE", "profile/plan")],
    )
    root, coverage = rpt.build_tree(*rpt.load_ledgers(sd))
    assert root.total.calls == 2
    assert root.total.turns == 1
    # 2 x 0.5, not 2 x 0.5 + the turn's own 1.0.
    assert root.total.usd_total == pytest.approx(1.0)
    assert coverage["turns_with_detail"] == 1
    assert coverage["turns_without_detail"] == 0


def test_an_undetailed_turn_is_counted_once_from_the_turn_row(tmp_path: Path):
    sd = tmp_path / "s"
    _write(sd, [_turn("c1", "PROFILE", "profile/plan")], [])
    root, coverage = rpt.build_tree(*rpt.load_ledgers(sd))
    assert root.total.calls == 1
    assert root.total.usd_total == pytest.approx(1.0)
    # ISL/OSL are absent from a turn row and derived on the detail writer's rule.
    assert root.total.isl == 100
    assert root.total.osl == 25
    assert coverage["turns_without_detail"] == 1


def test_the_tree_nests_by_phase_then_by_task_path_segment(tmp_path: Path):
    sd = tmp_path / "s"
    _write(sd, [_turn("c1", "KERNEL_AGENT", "geak/HeadKernel/lane-a")], [])
    root, _ = rpt.build_tree(*rpt.load_ledgers(sd))
    node = root.children["KERNEL_AGENT"].children["geak"].children["HeadKernel"].children["lane-a"]
    assert node.total.calls == 1
    assert root.children["KERNEL_AGENT"].total.usd_total == pytest.approx(1.0)


def test_a_parent_total_is_the_sum_of_its_children(tmp_path: Path):
    sd = tmp_path / "s"
    _write(
        sd,
        [_turn("c1", "EXPLORE", "explore/a"), _turn("c2", "EXPLORE", "explore/b")],
        [],
    )
    root, _ = rpt.build_tree(*rpt.load_ledgers(sd))
    explore = root.children["EXPLORE"].children["explore"]
    assert explore.total.usd_total == pytest.approx(
        sum(k.total.usd_total for k in explore.children.values())
    )


def test_rows_with_no_phase_or_path_get_their_own_labelled_node(tmp_path: Path):
    sd = tmp_path / "s"
    _write(sd, [_turn("c1", "", "")], [])
    root, _ = rpt.build_tree(*rpt.load_ledgers(sd))
    assert root.children[rpt.UNPHASED].children[rpt.UNPATHED].total.calls == 1


def test_an_unpriced_call_is_excluded_from_cost_and_flagged_in_coverage(tmp_path: Path):
    sd = tmp_path / "s"
    _write(
        sd,
        [
            _turn("c1", "PROFILE", "p"),
            _turn("c2", "PROFILE", "p", cost_source="unavailable", cost_usd=None),
        ],
        [],
    )
    root, coverage = rpt.build_tree(*rpt.load_ledgers(sd))
    assert root.total.calls == 2
    assert root.total.calls_priced == 1
    assert root.total.usd_total == pytest.approx(1.0)
    assert coverage["calls_unpriced"] == 1


def test_ext_shards_are_split_into_turn_and_detail_streams(tmp_path: Path):
    sd = tmp_path / "s"
    _write(
        sd,
        [],
        [],
        ext={
            "geak-7.jsonl": [_turn("g1", "KERNEL_AGENT", "geak/HeadKernel", component="geak")],
            "geak-7.detail.jsonl": [
                _detail("g1", 0, "KERNEL_AGENT", "geak/HeadKernel", component="geak")
            ],
        },
    )
    root, coverage = rpt.build_tree(*rpt.load_ledgers(sd))
    # One turn, expanded into one API call -- not two calls.
    assert root.total.calls == 1
    assert root.total.turns == 1
    assert coverage["turns_with_detail"] == 1
    assert rpt._has_node(root, "geak")


def test_a_detail_row_with_no_turn_row_is_counted_and_reported_as_an_orphan(tmp_path: Path):
    sd = tmp_path / "s"
    _write(sd, [], [_detail("gone", 0, "PROFILE", "p")])
    root, coverage = rpt.build_tree(*rpt.load_ledgers(sd))
    assert root.total.calls == 1
    assert coverage["detail_rows_orphaned"] == 1


def test_markdown_names_the_missing_geak_subtree(tmp_path: Path):
    sd = tmp_path / "s"
    _write(sd, [_turn("c1", "PROFILE", "p")], [])
    root, coverage = rpt.build_tree(*rpt.load_ledgers(sd))
    md = rpt.render_markdown("s", root, coverage)
    assert "no `geak` subtree is present" in md


def test_trimming_the_tree_does_not_change_any_total(tmp_path: Path):
    sd = tmp_path / "s"
    _write(sd, [_turn("c1", "KERNEL_AGENT", "geak/HeadKernel/lane-a/kernel-3")], [])
    root, _ = rpt.build_tree(*rpt.load_ledgers(sd))
    before = root.total.as_dict()
    rpt._trim(root, 0, 2)
    root.roll_up()
    assert root.total.as_dict() == before
    assert root.children["KERNEL_AGENT"].children["geak"].children == {}


def test_main_writes_both_artifacts(tmp_path: Path, monkeypatch):
    sd = tmp_path / "s"
    _write(sd, [_turn("c1", "PROFILE", "p")], [])
    monkeypatch.delenv("USER_DATA_PATH", raising=False)
    assert rpt.main(["-s", str(sd)]) == 0
    payload = json.loads((sd / "reports" / "llm_call_report.json").read_text())
    assert payload["tree"]["totals"]["calls"] == 1
    assert (sd / "reports" / "llm_call_report.md").read_text().startswith("# LLM call report")


def test_main_refuses_a_session_with_no_ledger(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    assert rpt.main(["-s", str(tmp_path / "empty")]) == 2


def test_a_truncated_final_line_is_skipped_not_fatal(tmp_path: Path):
    sd = tmp_path / "s"
    _write(sd, [_turn("c1", "PROFILE", "p")], [])
    path = sd / "reports" / "trace" / "llm_calls.jsonl"
    path.write_text(path.read_text() + '{"session_id": "s1", "compo')
    root, _ = rpt.build_tree(*rpt.load_ledgers(sd))
    assert root.total.calls == 1
