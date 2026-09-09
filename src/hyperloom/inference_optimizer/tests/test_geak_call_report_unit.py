# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.tools import dump_geak_call_report as rpt

EVAL_DIR = "/runs/sess-1/geak/e2e_cycle0"


def _assistant(msg_id: str, block: int, ts: str, *, out: int, think: int, tools=()) -> dict:
    content: list[dict] = [{"type": "text", "text": "x"}]
    content += [{"type": "tool_use", "name": name, "id": f"{msg_id}-{i}"} for i, name in enumerate(tools)]
    return {
        "type": "assistant",
        "timestamp": ts,
        "apiBlockIndex": block,
        "message": {
            "id": msg_id,
            "model": "claude-opus-5",
            "content": content,
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 900,
                "cache_creation_input_tokens": 90,
                "output_tokens": out,
                "output_tokens_details": {"thinking_tokens": think},
            },
        },
    }


def _write_run(root: Path, *, eval_dir: str = EVAL_DIR, transcripts: bool = True) -> Path:
    session = root / "projects" / "-slug" / "sess-uuid"
    wf_dir = session / "workflows"
    wf_dir.mkdir(parents=True)
    record = {
        "runId": "wf_test-001",
        "name": "e2e",
        "timestamp": "2026-09-05T00:00:00.000Z",
        "defaultModel": "claude-opus-5",
        "args": {"eval_dir": eval_dir},
        "totalToolCalls": 3,
        "agentCount": 2,
        "durationMs": 5000,
        "workflowProgress": [
            {"type": "workflow_phase", "index": 0, "title": "Setup"},
            {"type": "workflow_phase", "index": 1, "title": "HeadKernel"},
            {
                "type": "workflow_agent",
                "agentId": "a1",
                "label": "director:setup",
                "phaseTitle": "Setup",
                "model": "claude-opus-5",
                "tokens": 123,
                "toolCalls": 2,
                "durationMs": 4000,
                "state": "done",
            },
            {
                "type": "workflow_agent",
                "agentId": "a2",
                "label": "extract_op k",
                "phaseTitle": "HeadKernel",
                "model": "mystery-model",
                "tokens": 45,
                "toolCalls": 1,
                "durationMs": 1000,
                "state": "done",
            },
        ],
    }
    (wf_dir / "wf_test-001.json").write_text(json.dumps(record), encoding="utf-8")
    if transcripts:
        td = session / "subagents" / "workflows" / "wf_test-001"
        td.mkdir(parents=True)
        (td / "agent-a1.jsonl").write_text(
            "\n".join(
                json.dumps(r)
                for r in (
                    _assistant("m1", 0, "2026-09-05T00:00:01.000Z", out=1, think=0),
                    _assistant("m1", 1, "2026-09-05T00:00:02.000Z", out=100, think=40, tools=("Bash",)),
                    {"type": "user", "timestamp": "2026-09-05T00:00:03.000Z"},
                    _assistant("m2", 0, "2026-09-05T00:00:05.000Z", out=50, think=10, tools=("Read",)),
                )
            ),
            encoding="utf-8",
        )
    return wf_dir / "wf_test-001.json"


def _tree(tmp_path: Path, **kw):
    path = _write_run(tmp_path, **kw)
    record = json.loads(path.read_text())
    return rpt.build_tree(record, rpt.transcript_dir(path, "wf_test-001"))


def test_resolve_matches_eval_dir_and_rejects_others(tmp_path: Path) -> None:
    _write_run(tmp_path)
    homes = [tmp_path]
    assert rpt.resolve_runs(homes=homes, eval_dir=EVAL_DIR)
    # The session root is a prefix of the recorded eval dir, so it must match.
    assert rpt.resolve_runs(homes=homes, eval_dir="/runs/sess-1/geak")
    assert rpt.resolve_runs(homes=homes, session_dir=Path("/runs/sess-1"))
    assert not rpt.resolve_runs(homes=homes, eval_dir="/no/such/path")
    assert not rpt.resolve_runs(homes=homes, run_id="wf_other")


def test_a_record_without_an_eval_dir_never_matches(tmp_path: Path) -> None:
    _write_run(tmp_path, eval_dir="")
    assert not rpt.resolve_runs(homes=[tmp_path], eval_dir="/anything")


def test_continuation_blocks_count_as_one_call(tmp_path: Path) -> None:
    root, _ = _tree(tmp_path)
    setup = root.children["Setup"].children["director:setup"]
    # Three assistant rows, two message ids: the split blocks are one call.
    assert setup.total.calls == 2
    assert set(setup.children) == {"call 0000", "call 0001"}
    # Usage is cumulative across blocks, so the last block wins outright.
    assert setup.children["call 0000"].total.osl == 100


def test_thinking_is_a_subset_of_output(tmp_path: Path) -> None:
    root, _ = _tree(tmp_path)
    call = root.children["Setup"].children["director:setup"].children["call 0000"].total
    assert call.osl == 100
    assert call.tokens_thinking == 40
    assert call.tokens_out == 60
    assert call.isl == 10 + 900 + 90


def test_tool_calls_are_counted_once_at_their_own_leaf(tmp_path: Path) -> None:
    root, _ = _tree(tmp_path)
    call = root.children["Setup"].children["director:setup"].children["call 0000"]
    assert call.own.tool_calls == 0
    assert call.children["tool:Bash"].own.tool_calls == 1
    assert call.total.tool_calls == 1


def test_agent_wallclock_comes_from_the_record(tmp_path: Path) -> None:
    root, _ = _tree(tmp_path)
    assert root.children["Setup"].total.ms_total == pytest.approx(4000.0)
    assert root.total.ms_total == pytest.approx(5000.0)


def test_an_unpriced_model_is_excluded_not_zeroed(tmp_path: Path) -> None:
    root, coverage = _tree(tmp_path)
    head = root.children["HeadKernel"].children["extract_op k"].total
    assert head.osl == 45
    assert head.calls_priced == 0
    assert head.usd_total == 0.0
    assert coverage["calls_unpriced"] >= 1
    assert coverage["agents_estimated_from_summary"] == ["HeadKernel/extract_op k"]


def test_missing_transcripts_still_report_the_declared_agents(tmp_path: Path) -> None:
    root, coverage = _tree(tmp_path, transcripts=False)
    assert coverage["agents_with_transcript"] == 0
    assert coverage["agents_without_transcript"] == 2
    assert root.total.tool_calls == 3
    assert root.total.usd_total == 0.0


def test_declared_phases_survive_with_no_agents(tmp_path: Path) -> None:
    root, _ = _tree(tmp_path)
    assert {"Setup", "HeadKernel"} <= set(root.children)


def test_markdown_leads_with_coverage(tmp_path: Path) -> None:
    root, coverage = _tree(tmp_path)
    md = rpt.render_markdown(root, coverage)
    assert md.index("## Coverage") < md.index("## Run totals")
    assert "Reconciliation against the run record" in md
    assert "wf_test-001" in md


def test_main_writes_both_files(tmp_path: Path) -> None:
    _write_run(tmp_path)
    out = tmp_path / "out"
    code = rpt.main(
        ["--eval-dir", EVAL_DIR, "--claude-home", str(tmp_path), "-o", str(out)]
    )
    assert code == 0
    assert (out / "geak_call_report.md").is_file()
    payload = json.loads((out / "geak_call_report.json").read_text())
    assert payload["tree"]["totals"]["calls"] == 3


def test_main_reports_a_missing_run(tmp_path: Path) -> None:
    _write_run(tmp_path)
    assert rpt.main(["--eval-dir", "/nope", "--claude-home", str(tmp_path)]) == 2


def test_exp_root_is_a_selection_fallback(tmp_path: Path) -> None:
    session = tmp_path / "projects" / "-slug" / "sess-uuid"
    (session / "workflows").mkdir(parents=True)
    (session / "workflows" / "wf_standalone.json").write_text(
        json.dumps(
            {
                "runId": "wf_standalone",
                "startTime": 1,
                "args": {"exp_root": "/exp/run-7"},
                "workflowProgress": [],
            }
        ),
        encoding="utf-8",
    )
    assert rpt.resolve_runs(homes=[tmp_path], eval_dir="/exp/run-7")
    assert not rpt.resolve_runs(homes=[tmp_path], eval_dir="/exp/run-8")


def test_list_all_ignores_the_selectors(tmp_path: Path) -> None:
    _write_run(tmp_path)
    assert rpt.resolve_runs(homes=[tmp_path], list_all=True)
    assert rpt.main(["--list", "--claude-home", str(tmp_path)]) == 0


def test_captured_text_carries_prompt_and_response(tmp_path: Path) -> None:
    path = _write_run(tmp_path)
    calls = rpt.read_agent_calls(
        rpt.transcript_dir(path, "wf_test-001") / "agent-a1.jsonl",
        capture_text=True,
    )
    assert calls[0]["output_text"] == "x\n\nx"
    assert calls[0]["prompt_text"] == ""
    assert "prompt_text" not in rpt.read_agent_calls(
        rpt.transcript_dir(path, "wf_test-001") / "agent-a1.jsonl"
    )[0]


def test_text_capture_respects_the_character_cap(tmp_path: Path) -> None:
    session = tmp_path / "projects" / "-slug" / "s" / "subagents" / "workflows" / "w"
    session.mkdir(parents=True)
    row = _assistant("m1", 0, "2026-09-05T00:00:01.000Z", out=5, think=0)
    row["message"]["content"] = [{"type": "text", "text": "y" * 50}]
    (session / "agent-a1.jsonl").write_text(json.dumps(row), encoding="utf-8")
    call = rpt.read_agent_calls(session / "agent-a1.jsonl", capture_text=True, text_chars=10)[0]
    assert call["output_text"].startswith("y" * 10)
    assert "[+40 chars]" in call["output_text"]


def test_the_orchestrator_turn_is_opt_in(tmp_path: Path) -> None:
    path = _write_run(tmp_path)
    session = path.parent.parent
    (session.parent / f"{session.name}.jsonl").write_text(
        json.dumps(_assistant("orch-1", 0, "2026-09-05T00:00:00.500Z", out=7, think=2)),
        encoding="utf-8",
    )
    record = json.loads(path.read_text())
    td = rpt.transcript_dir(path, "wf_test-001")

    plain, plain_cov = rpt.build_tree(record, td)
    assert "(orchestrator)" not in plain.children
    assert "orchestrator_calls" not in plain_cov

    joined, cov = rpt.build_tree(record, td, orchestrator=rpt.session_transcript(path))
    assert cov["orchestrator_calls"] == 1
    assert joined.children["(orchestrator)"].total.osl == 7
    assert joined.total.calls == plain.total.calls + 1


def test_nested_workflows_join_by_time_containment(tmp_path: Path) -> None:
    path = _write_run(tmp_path)
    record = json.loads(path.read_text())
    record["startTime"] = 1_000_000
    record["durationMs"] = 10_000
    path.write_text(json.dumps(record), encoding="utf-8")
    for name, start in (("wf_inside", 1_005_000), ("wf_outside", 9_000_000)):
        (path.parent / f"{name}.json").write_text(
            json.dumps({"runId": name, "startTime": start, "workflowProgress": []}),
            encoding="utf-8",
        )
    found = rpt.nested_records(path, record)
    assert [c["runId"] for c in found] == ["wf_inside"]

    root, cov = rpt.build_tree(
        record,
        rpt.transcript_dir(path, "wf_test-001"),
        nested=[(found[0], tmp_path / "nowhere")],
    )
    assert "nested workflow wf_inside" in root.children
    assert cov["nested_runs"][0]["join"] == "time containment"


def test_the_sidecar_tags_every_call_with_its_position(tmp_path: Path) -> None:
    path = _write_run(tmp_path)
    sink: list[dict] = []
    rpt.build_tree(
        json.loads(path.read_text()),
        rpt.transcript_dir(path, "wf_test-001"),
        capture_text=True,
        sink=sink,
    )
    assert [r["phase"] for r in sink] == ["Setup", "Setup"]
    assert sink[0]["agent"] == "director:setup"
    assert sink[0]["tool_calls"] == ["Bash"]
    assert sink[0]["isl"] == 1000
