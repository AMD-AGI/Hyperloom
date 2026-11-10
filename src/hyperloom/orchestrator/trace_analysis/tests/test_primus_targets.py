# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Primus' projected training iteration stands in for a profiled one, and measured wins flow back to it."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from hyperloom.orchestrator.trace_analysis import primus_targets as pt
from hyperloom.orchestrator.trace_analysis.tests.test_primus_calibration import training_report, write_training_trace


@pytest.fixture(autouse=True)
def _task_groups_without_tracelens(monkeypatch):
    """Task grouping parses sources through TraceLens; without it installed, group nothing."""
    if importlib.util.find_spec("TraceLens") is not None:
        return
    name = "hyperloom.orchestrator.trace_analysis.tracelens_skill_runner"
    stub = ModuleType(name)
    stub.aggregate_by_source_function = lambda rows: []
    monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.delitem(sys.modules, "hyperloom.orchestrator.trace_analysis._analysis_reports", raising=False)


def _report() -> dict:
    report = training_report()
    grouped = {"m": 8192, "n": 2880, "k": 2880, "batch": 16, "dtype": "fp8", "layout": "NT"}
    for row in report["kernels"]:
        if row["kind"] == "grouped_gemm":
            row["dims"] = grouped | {"layout": {"fwd": "NT", "dgrad": "NN", "wgrad": "TN"}[row["phase"]]}
    return report


def test_candidate_names_round_trip():
    row = {"scope": "moe", "kind": "grouped_gemm", "op": "moe_expert_gate", "phase": "dgrad"}
    name = pt.candidate_name(row)
    assert name == "primus::moe::grouped_gemm::moe_expert_gate.dgrad"
    assert pt.parse_candidate_name(name) == row
    assert pt.parse_candidate_name("Cijk_Alik_Bljk") is None


def test_candidates_carry_shapes_and_their_primus_term():
    rows = pt.candidates_from_report(_report())
    assert rows[0]["name"] == "primus::output::gemm::lm_head.fwd" and rows[0]["duration_us"] == pytest.approx(5000.0)
    q = next(r for r in rows if r["operation"] == "attn_q.fwd")
    assert q["shapes"] == ["(4096,2880) bf16", "(512,2880) bf16"]
    assert q["primus_task"]["cases"][0]["dims"]["layout"] == "NT"
    assert q["shape_provenance"] == pt.SHAPE_PROVENANCE
    with pytest.raises(ValueError):
        pt.candidates_from_report({"schema": "other"})


def test_gemm_shapes_split_dense_and_grouped_by_layout():
    dense, grouped = pt.gemm_shapes(_report())
    assert dense == [
        {
            "M": 4096,
            "N": 512,
            "K": 2880,
            "q_dtype_w": "bf16",
            "layout": "NT",
            "calls": 3,
            "est_total_ms": 3.0,
            "ops": ["attn_q.fwd", "attn_k.fwd", "attn_v.fwd"],
        }
    ]
    assert [(g["G"], g["layout"], g["ops"]) for g in grouped] == [
        (16, "NT", ["moe_expert_gate.fwd"]), (16, "NN", ["moe_expert_gate.dgrad"]), (16, "TN", ["moe_expert_gate.wgrad"]),
    ]  # fmt: skip


def test_write_run_lays_out_a_tracelens_run(tmp_path):
    subject = pt.Subject(workspace_path=str(tmp_path), session_id="s1", framework="megatron")
    result = pt.write_run(_report(), subject, report_path=str(tmp_path / "kernel_report.json"))
    assert result["status"] == "ok" and result["trace_input_type"] == "primus"
    run_dir = Path(result["analysis_report_path"]).parent
    assert {"kernel_candidates.json", "shapes.json", "grouped_shapes.json", "analysis.md"} <= {
        p.name for p in run_dir.iterdir()
    }
    assert json.loads((run_dir / "grouped_shapes.json").read_text())[0]["G"] == 16
    assert result["run_meta"]["route_ext"]["dense_gemm_shapes"] == 1


def test_adopted_kernels_become_speedups_and_strangers_are_reported():
    summary = {
        "by_kernel": [
            {
                "kernel_name": "primus::output::engine::cross_entropy.fwd",
                "claimed": True,
                "speedup": 2.0,
                "lane": "triton",
            },
            {"kernel_name": "primus::moe::comm::moe_a2a_exposed.fwd", "claimed": False, "speedup": 3.0},
            {"kernel_name": "primus::moe::gemm::attn_o.fwd", "claimed": True, "speedup": 0.9},
            {"kernel_name": "Cijk_foo", "claimed": True, "speedup": 1.5},
        ]
    }
    entries, unplaced = pt.speedups_from_summary(summary)
    assert entries == [
        {"op": "cross_entropy", "phase": "fwd", "scope": "output", "speedup": 2.0, "source": "kernel-opt:triton"}
    ]
    assert unplaced == ["Cijk_foo"]
    assert pt._manual_speedup("moe_expert_up.wgrad=1.25") == {
        "op": "moe_expert_up",
        "phase": "wgrad",
        "speedup": 1.25,
        "source": "manual",
    }


_FAKE_MAIN = """
import json, os, sys
argv = sys.argv[1:]
arg = lambda flag: argv[argv.index(flag) + 1] if flag in argv else None
out = arg("--kernel-report")
assert os.path.isabs(out) and os.path.isabs(arg("--config")), argv
report = {"schema": "primus.kernel_report", "iteration": {}, "kernels": [],
          "provenance": {"config": arg("--config"), "calibration": arg("--kernel-calibration"), "speedups": arg("--kernel-speedups")}}
json.dump(report, open(out, "w"))
print("Iteration Time: 1,234.5 ms" if not arg("--kernel-speedups") else "Iteration Time: 1000.0 ms")
"""

_FAKE_AGENT = """
import json, os, sys
argv = sys.argv[1:]
out = argv[argv.index("--out-dir") + 1]
os.makedirs(out, exist_ok=True)
cmd = ["python", "-m", "primus.cli.main", "--config", os.path.join(out, "trials", "best.yaml")]
best = {"result": {"cmd": cmd, "iteration_ms": 10.0}, "calibration": os.environ.get("PRIMUS_KERNEL_CALIBRATION")}
json.dump({"best": best, "trials_total": 1}, open(os.path.join(out, "summary.json"), "w"))
"""


@pytest.fixture
def fake_primus(tmp_path):
    root = tmp_path / "primus_root"
    for pkg, body in (("primus/cli/main.py", _FAKE_MAIN), ("primus/agents/tuning_agent.py", _FAKE_AGENT)):
        path = root / pkg
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    for pkg in ("primus", "primus/cli", "primus/agents"):
        (root / pkg / "__init__.py").write_text("")
    return pt.Primus(root=str(root), python=sys.executable)


def test_project_passes_absolute_paths_and_reads_the_iteration_time(fake_primus, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("cfg.yaml").write_text("x: 1\n")
    Path("cal.json").write_text("{}")
    report = pt.project(fake_primus, "cfg.yaml", Path("out/report.json"), calibration="cal.json")
    assert report["iteration"]["iteration_time_ms"] == pytest.approx(1234.5)
    assert report["provenance"]["config"] == str(tmp_path / "cfg.yaml")
    assert (
        report["provenance"]["calibration"] == str(tmp_path / "cal.json") and report["provenance"]["speedups"] is None
    )


def test_project_failure_says_what_ran(fake_primus, tmp_path):
    (Path(fake_primus.root) / "primus/cli/main.py").write_text("import sys; print('boom'); sys.exit(3)")
    with pytest.raises(RuntimeError, match=r"rc=3(.|\n)*boom"):
        pt.project(fake_primus, str(tmp_path / "cfg.yaml"), tmp_path / "report.json")


def test_tune_hands_the_calibration_to_the_agent_and_returns_its_winner(fake_primus, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("cal.json").write_text("{}")
    found = pt.tune(fake_primus, "workload.yaml", Path("tune"), calibration="cal.json", seed_budget=2)
    assert found["winner_config"] == str(tmp_path / "tune" / "tuning" / "trials" / "best.yaml")
    assert found["best"]["calibration"] == str(tmp_path / "cal.json")
    assert "num_nodes: 1" in (tmp_path / "tune" / "target_cluster.yaml").read_text()


def test_project_command_reports_the_gain(fake_primus, tmp_path, capsys):
    args = [
        "project",
        "--config",
        str(tmp_path / "cfg.yaml"),
        "--kernel-speedups",
        str(tmp_path / "fast.json"),
        "--out-dir",
        str(tmp_path / "p"),
    ]
    args += ["--primus-root", fake_primus.root, "--primus-python", fake_primus.python]
    assert pt.main(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["baseline_iteration_ms"], out["projected_iteration_ms"], out["gain_pct"]) == (
        1234.5,
        1000.0,
        pytest.approx(18.996, abs=1e-3),
    )


def test_validate_compares_time_and_class_mix(tmp_path):
    trace = write_training_trace(tmp_path / "exp-rank[0].pt.trace.json.gz")
    result = pt.validate(training_report(), pt._traces(str(tmp_path)), measured_ms=25.0)
    assert result["error_pct"] == pytest.approx(-20.0)
    assert [m["trace"] for m in result["mix"]] == [trace]
    assert 0.0 < result["mix"][0]["total_variation"] < 1.0
