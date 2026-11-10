# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A training step's kernel becomes a KernelForge task, and the kept kernel's phases flow back as speedups."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hyperloom.orchestrator.trace_analysis import primus_forge as pf
from hyperloom.orchestrator.trace_analysis import primus_targets as pt


def _report() -> dict:
    lm_fwd = {"m": 65536, "n": 201088, "k": 2880, "batch": 1, "dtype": "bf16", "layout": "NT"}
    return {
        "deployment": {"tp": 1, "pp": 1, "ep": 8, "cp": 1, "micro_batch_size": 16, "seq_length": 4096},
        "iteration": {"iteration_time_ms": 900.0},
        "kernels": [
            {"op": "lm_head", "phase": "fwd", "kind": "gemm", "dims": lm_fwd, "est_total_ms": 60.0},
            {"op": "cross_entropy", "phase": "fwd", "scope": "output", "kind": "engine", "est_total_ms": 70.0},
            {"op": "cross_entropy", "phase": "bwd", "scope": "output", "kind": "engine", "est_total_ms": 20.0},
        ],
    }


def test_task_spec_reads_the_steps_shapes_and_share():
    spec = pf.task_spec(_report(), "cross_entropy")
    assert spec["cases"] == [{"tokens": 65536, "vocab": 201088}]
    assert spec["phases_ms"] == {"bwd": 20.0, "fwd": 70.0}
    assert spec["share_pct"] == pytest.approx(10.0)
    assert {c["tokens"] for c in spec["check_cases"]} == {4096, 65536}
    with pytest.raises(ValueError, match="no forge task"):
        pf.task_spec(_report(), "attn_sdpa")


def test_write_task_is_a_committed_workspace_with_loop_scripts(tmp_path):
    spec = pf.task_spec(_report(), "cross_entropy", tokens=[65536, 32768], config="cfg.yaml")
    files = pf.write_task(spec, tmp_path / "run")
    ws = Path(files["workspace"])
    assert {p.name for p in ws.iterdir()} >= {"kernel.py", "driver.py", "program.md", "task.json"}
    assert json.loads((ws / "task.json").read_text())["cases"][1]["tokens"] == 32768
    assert "forge_experiments/" in (ws / ".gitignore").read_text()
    tracked = subprocess.run(["git", "-C", str(ws), "ls-files"], capture_output=True, text=True, check=True)
    assert ".gitignore" in tracked.stdout.split()
    program = (ws / "program.md").read_text()
    assert "cfg.yaml" in program and "10.0% of the iteration" in program and "{" not in program
    log = subprocess.run(["git", "-C", str(ws), "log", "--oneline"], capture_output=True, text=True, check=True)
    assert "the task" in log.stdout
    loop = Path(files["forge_loop.sh"]).read_text()
    assert "prepare.sh" in loop and "--baseline-json" in loop and "--kernel-backend triton" in loop
    resume = Path(files["forge_loop_resume.sh"]).read_text()
    assert all("--roofline-ceiling on" in s for s in (loop, resume))
    with pytest.raises(ValueError, match="not empty"):
        pf.write_task(spec, tmp_path / "run")


def _phases(fwd: tuple[float, float], bwd: tuple[float, float]) -> dict:
    def rec(x):
        return {"baseline_ms": x[0], "kernel_ms": x[1]}

    return {
        "op": "cross_entropy",
        "scope": "output",
        "cases": {
            "T=65536": {"fwd": rec(fwd), "bwd": rec(bwd)},
            "T=32768": {"fwd": rec((fwd[0] / 2, fwd[1])), "bwd": rec(bwd)},
        },
    }


def test_speedups_from_phases_per_phase_at_the_steps_tokens():
    entries = pf.speedups_from_phases(_phases((70.0, 7.0), (40.0, 50.0)))
    by_phase = {e["phase"]: e for e in entries}
    assert by_phase["fwd"]["speedup"] == pytest.approx(10.0)
    # a slower phase is fed back as measured
    assert by_phase["bwd"]["speedup"] == pytest.approx(0.8)
    assert by_phase["fwd"]["scope"] == "output" and by_phase["fwd"]["op"] == "cross_entropy"
    small = {
        e["phase"]: e["speedup"] for e in pf.speedups_from_phases(_phases((70.0, 7.0), (40.0, 50.0)), tokens=32768)
    }
    assert small["fwd"] == pytest.approx(5.0)


def test_speedups_against_transformer_engine():
    phases = _phases((70.0, 7.0), (40.0, 8.0))
    phases["cases"]["T=65536"]["fwd"]["te_ms"] = 14.0
    phases["cases"]["T=65536"]["bwd"]["te_ms"] = 2.0
    by_phase = {e["phase"]: e["speedup"] for e in pf.speedups_from_phases(phases, against="te")}
    # TE does most of its work in the forward, so its backward can beat the kernel's
    assert by_phase == {"fwd": pytest.approx(2.0), "bwd": pytest.approx(0.25)}
    del phases["cases"]["T=65536"]["bwd"]["te_ms"]
    assert [e["phase"] for e in pf.speedups_from_phases(phases, against="te")] == ["fwd"]


def test_feedback_reads_phases_and_run_state(tmp_path):
    phases = tmp_path / "phases.json"
    phases.write_text(json.dumps(_phases((70.0, 7.0), (40.0, 20.0))))
    run = tmp_path / "run"
    pf.write_task(pf.task_spec(_report(), "cross_entropy"), run)
    state = {"baseline_case_times": {"T=65536": 110.0}, "best_case_times": {"T=65536": 22.0}}
    (run / "workspace" / "forge_experiments").mkdir()
    (run / "workspace" / "forge_experiments" / "run_state.json").write_text(json.dumps(state))
    out = tmp_path / "speedups.json"
    assert pt.main(["feedback", "--forge-phases", str(phases), "--forge-run", str(run), "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["schema"] == pf.SPEEDUPS_SCHEMA
    got = [(e["phase"], e["speedup"], e["source"]) for e in doc["speedups"]]
    assert ("fwd", 10.0, "forge:cross_entropy") in got and ("bwd", 2.0, "forge:cross_entropy") in got
    assert ("fwd", 5.0, "forge:cross_entropy:run_state") in got
