# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Cover TuneReport.to_dict, write_report, and BaseTuner.execute paths."""

from __future__ import annotations

import json

from kernelforge.gemm_tune.model_analyzer import ModelProfile
from kernelforge.gemm_tune.report import TuneReport, build_report, write_report
from kernelforge.gemm_tune.tuners.base import BaseTuner, TuneContext, TuneResult


def _profile():
    return ModelProfile(model_path="/m", hidden_size=4096, intermediate_size=11008)


# ── TuneReport.to_dict / write_report ────────────────────────────────────────
def test_report_to_dict_includes_skipped_and_error():
    report = TuneReport(
        status="failed",
        micro_decision="failed",
        tuners_skipped=[{"tuner": "x", "skip_reason": "no"}],
        error="boom",
        error_class="RuntimeError",
    )
    d = report.to_dict()
    assert d["tuners_skipped"] == [{"tuner": "x", "skip_reason": "no"}]
    assert d["error"] == "boom" and d["error_class"] == "RuntimeError"


def test_report_to_dict_omits_empty_optionals():
    report = TuneReport(status="ok", micro_decision="no_improvement")
    d = report.to_dict()
    assert "tuners_skipped" not in d and "error" not in d


def test_write_report_creates_json(tmp_path):
    report = build_report(
        results=[],
        skipped=[("fmoe_ck", "not MoE")],
        profile=_profile(),
        framework="sglang",
        precision="bf16",
        quant_type="none",
        gpu_type="mi300x",
        tp=1,
        conc=64,
        tokens=[64],
        started_at="2026-01-01T00:00:00Z",
        total_elapsed_s=1.0,
    )
    out = tmp_path / "nested" / "dir"
    path = write_report(report, out)
    assert path == out / "result.json"
    data = json.loads(path.read_text())
    assert data["status"] == "skipped"


# ── TuneResult.to_dict extra branches ────────────────────────────────────────
def test_tune_result_to_dict_full():
    r = TuneResult(
        tuner_name="t",
        status="ok",
        artifact_path="/a",
        env_var="V",
        env_value="/a",
        total_shapes=3,
        improved_shapes=2,
        best_micro_speedup=1.2,
        avg_micro_speedup=1.1,
        shape_results=[{"M": 4, "speedup": 1.2}],
        error="e",
        error_class="C",
        skip_reason="sr",
    )
    d = r.to_dict()
    assert d["artifact"] == "/a" and d["env_var"] == "V"
    assert d["shape_results"] == [{"M": 4, "speedup": 1.2}]
    assert d["error"] == "e" and d["skip_reason"] == "sr"


def test_has_improvement_variants():
    assert TuneResult("t", "ok", candidate=True).has_improvement is True
    assert TuneResult("t", "ok", improved_shapes=1, best_micro_speedup=1.1).has_improvement is True
    assert TuneResult("t", "ok", improved_shapes=0).has_improvement is False


# ── BaseTuner.execute ────────────────────────────────────────────────────────
def _ctx(tmp_path):
    return TuneContext(
        profile=_profile(),
        framework="sglang",
        precision="bf16",
        quant_type="none",
        gpu_type="mi300x",
        tp=1,
        conc=64,
        tokens=[],
        mp=1,
        output_dir=tmp_path,
        iters=10,
        warmup=2,
        min_improvement_pct=3.0,
        timeout_s=60,
    )


class _OkTuner(BaseTuner):
    name = "ok_tuner"

    def validate(self):
        return None

    def run(self):
        return TuneResult(tuner_name=self.name, status="ok", improved_shapes=1, best_micro_speedup=1.2)


class _ValidateFailTuner(BaseTuner):
    name = "vf_tuner"

    def validate(self):
        return "missing input"

    def run(self):  # pragma: no cover - never reached
        raise AssertionError("run should not be called")


class _RunRaisesTuner(BaseTuner):
    name = "boom_tuner"

    def validate(self):
        return None

    def run(self):
        raise ValueError("kernel crash")


def test_execute_success_sets_elapsed(tmp_path):
    res = _OkTuner(_ctx(tmp_path)).execute()
    assert res.status == "ok"
    assert res.elapsed_s >= 0.0


def test_execute_validation_error(tmp_path):
    res = _ValidateFailTuner(_ctx(tmp_path)).execute()
    assert res.status == "failed"
    assert res.error == "missing input"
    assert res.error_class == "validation_error"


def test_execute_run_exception_captured(tmp_path):
    res = _RunRaisesTuner(_ctx(tmp_path)).execute()
    assert res.status == "failed"
    assert res.error_class == "ValueError"
    assert "kernel crash" in res.error


def test_base_tuner_creates_work_dir(tmp_path):
    t = _OkTuner(_ctx(tmp_path))
    assert t.work_dir == tmp_path / "tuners" / "ok_tuner"
    assert t.work_dir.is_dir()
