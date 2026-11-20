# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel targets from the tuning agent's top configurations: which are taken, and how their reports merge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.trace_analysis import primus_search as ps
from hyperloom.orchestrator.trace_analysis import primus_targets as pt
from hyperloom.orchestrator.trace_analysis.tests.test_primus_targets import (  # noqa: F401
    _task_groups_without_tracelens,
    fake_primus,
)


def _record(idx: int, tps: float, *, legal: bool = True, config: str = "", mbs: int = 8) -> dict:
    cmd = ["python", "-m", "primus.cli.main", "projection", "performance", "--config", config or f"t{idx}.yaml"]
    knobs = {"tp": 1, "pp": 1, "ep": 8, "cp": 1, "mbs": mbs, "vpp": 1, "recompute_granularity": None}
    return {"idx": idx, "config": knobs, "result": {"legal": legal, "tokens_per_s_per_gpu": tps, "cmd": cmd}}


def _write_trials(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_trials_rank_legal_ones_best_first_once_per_config(tmp_path):
    records = [
        _record(0, 100.0),
        _record(1, 300.0, mbs=16),
        _record(2, 900.0, legal=False),
        _record(3, 200.0),
        _record(4, 150.0, config="t1.yaml"),
    ]
    ranked = ps.ranked_trials(_write_trials(tmp_path / "trials.jsonl", records))
    assert [(t.idx, t.config) for t in ranked] == [(1, "t1.yaml"), (3, "t3.yaml"), (0, "t0.yaml")]
    assert ranked[0].label == "#1 TP1 PP1 EP8 CP1 MBS16"


def test_the_best_comes_first_and_stands_alone_when_the_trials_miss_it(tmp_path):
    ranked = ps.ranked_trials(_write_trials(tmp_path / "t.jsonl", [_record(i, 10.0 * (5 - i)) for i in range(5)]))
    assert [t.idx for t in ps.chosen(ranked, "t2.yaml", top_k=3)] == [2, 0, 1]
    assert [t.idx for t in ps.chosen(ranked, "t2.yaml", top_k=0)] == [2]
    alone = ps.chosen([], "/x/best.yaml", top_k=5)
    assert [(t.config, t.label) for t in alone] == [("/x/best.yaml", "best")]


def _row(op: str, ms: float, *, stage: int = 0, m: int = 4096, calls: int = 10) -> dict:
    dims = {"m": m, "n": 512, "k": 2880, "dtype": "bf16", "layout": "NT"}
    return {"stage": stage, "scope": "moe", "op": op, "phase": "fwd", "kind": "gemm", "dims": dims,
            "est_ms": ms / calls, "calls": calls, "est_total_ms": ms}  # fmt: skip


def _kernel_report(rows: list[dict], ms: float) -> dict:
    return {"schema": pt.REPORT_SCHEMA, "iteration": {"iteration_time_ms": ms}, "deployment": {}, "kernels": rows}


def test_reports_merge_with_each_configuration_weighted_equally():
    a = _kernel_report([_row("attn_q", 60.0), _row("lm_head", 40.0)], 100.0)
    b = _kernel_report([_row("attn_q", 100.0, m=8192), _row("router", 100.0)], 200.0)
    merged = ps.merge_reports([a, b], ["a", "b"])
    rows = {(r["op"], r["dims"]["m"]): r for r in merged["kernels"]}
    # b is rescaled to a's 100 ms, then each counts half.
    assert {k: r["est_total_ms"] for k, r in rows.items()} == pytest.approx(
        {("attn_q", 4096): 30.0, ("attn_q", 8192): 25.0, ("lm_head", 4096): 20.0, ("router", 4096): 25.0}
    )
    assert rows[("attn_q", 8192)]["configs"] == ["b (stage 0)"] and rows[("attn_q", 8192)]["calls"] == 2
    assert merged["iteration"] == a["iteration"] and merged["provenance"]["merged_configs"] == ["a", "b"]
    attn = next(c for c in pt.candidates_from_report(merged) if c["operation"] == "attn_q.fwd")
    assert attn["gpu_pct"] == pytest.approx(55.0)
    assert [c["dims"]["m"] for c in attn["primus_task"]["cases"]] == [4096, 8192]
    with pytest.raises(ValueError):
        ps.merge_reports([], [])


def test_pipeline_stages_merge_by_where_each_term_costs_the_most():
    pp1 = _kernel_report([_row("attn_q", 60.0), _row("lm_head", 40.0)], 100.0)
    pp2 = _kernel_report([_row("attn_q", 100.0), _row("attn_q", 50.0, stage=1), _row("lm_head", 50.0, stage=1)], 200.0)
    merged = ps.merge_reports([pp1, pp2], ["pp1", "pp2"])
    assert {r["stage"] for r in merged["kernels"]} == {0}
    shares = {c["operation"]: c["gpu_pct"] for c in pt.candidates_from_report(merged)}
    # Mean shares 80 % (60, 100) and 45 % (40, 50 on the last stage), over their sum.
    assert shares == pytest.approx({"attn_q.fwd": 64.0, "lm_head.fwd": 36.0})


def test_a_failed_projection_is_left_out_unless_it_is_the_best():
    trials = [ps.Trial(i, f"t{i}.yaml", 1.0, None, None, {}) for i in range(3)]

    def project(trial, i):
        if i == 1:
            raise RuntimeError("primus projection failed (rc=1)")
        return {"iteration": {"iteration_time_ms": 10.0 + i}}

    done = ps.project_top(trials, project)
    assert done.labels == ["t0", "t2"] and len(done.reports) == 2
    assert "rc=1" in done.configs[1]["error"] and done.configs[2]["projected_iteration_ms"] == 12.0

    def first_fails(trial, i):
        raise RuntimeError("no")

    with pytest.raises(RuntimeError):
        ps.project_top(trials, first_fails)


_FAKE_AGENT = """
import json, os, sys
argv = sys.argv[1:]
out = argv[argv.index("--out-dir") + 1]
os.makedirs(os.path.join(out, "trials"), exist_ok=True)
records = []
for idx, (name, tps) in enumerate((("slow", 50.0), ("best", 300.0), ("next", 200.0), ("third", 100.0))):
    path = os.path.join(out, "trials", name + ".yaml")
    open(path, "w").write(name + "\\n")
    cmd = ["python", "-m", "primus.cli.main", "--config", path]
    knobs = {"tp": 1, "pp": 1, "ep": 8, "cp": 1, "mbs": idx + 1}
    records.append({"idx": idx, "config": knobs, "result": {"legal": True, "tokens_per_s_per_gpu": tps, "cmd": cmd}})
open(os.path.join(out, "trials.jsonl"), "w").write("".join(json.dumps(r) + "\\n" for r in records))
best = next(r for r in records if r["result"]["tokens_per_s_per_gpu"] == 300.0)
json.dump({"best": best, "trials_total": len(records)}, open(os.path.join(out, "summary.json"), "w"))
"""


def test_tune_takes_its_targets_from_the_top_configurations(fake_primus, tmp_path, capsys):
    (Path(fake_primus.root) / "primus/agents/tuning_agent.py").write_text(_FAKE_AGENT)
    args = ["tune", "--workload", str(tmp_path / "w.yaml"), "--top-k", "2", "--workspace-path", str(tmp_path / "ws")]
    assert pt.main(args + ["--primus-root", fake_primus.root, "--primus-python", fake_primus.python]) == 0
    result = json.loads(capsys.readouterr().out)
    configs = result["tuning"]["configs"]
    assert [Path(c["config"]).stem for c in configs] == ["best", "next"]
    assert result["tuning"]["winner_config"] == configs[0]["config"]
    paths = result["artifact_paths"]
    assert Path(paths["kernel_report"]).name == "targets_report.json"
    merged = json.loads(Path(paths["kernel_report"]).read_text())
    assert merged["provenance"]["merged_configs"] == [c["label"] for c in configs]
    assert json.loads(Path(paths["config_search"]).read_text())["top_k"] == 2
    assert "merged over 2 configurations" in Path(result["analysis_report_path"]).read_text()
