# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for rocprof_roofline.py (parsing, classification, CLI, enrich)."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parent


def _load_module():
    """Load rocprof_roofline.py as an isolated module."""
    spec = importlib.util.spec_from_file_location(
        "rocprof_roofline_under_test", _TOOLS_DIR / "rocprof_roofline.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


rr = _load_module()


def _content(*, mfma_actual=900.0, hbm_actual=1000.0, real_peak=3000.0,
             include_fp=True, ai_hbm=2.5):
    """Build synthetic rocprof-compute analyze text for one kernel."""
    fp_row = (
        f"│ 4.1.1 │ MFMA FLOPs (F16) │ {mfma_actual} │ GFLOPs │ 1000.0 │ x │\n"
        if include_fp else ""
    )
    return (
        "Kernel 0: my_gemm_kernel (87.50%)\n"
        "4.1 Roofline Rate Metrics\n"
        f"│ 4.1.0 │ HBM Bandwidth │ {hbm_actual} │ GB/s │ 2000.0 │ x │\n"
        f"{fp_row}"
        "╘═══╛\n"
        "4.2 Roofline AI Plot Points\n"
        f"│ 4.2.0 │ AI HBM │ {ai_hbm} │ Flops/Byte │\n"
        "│ 4.2.1 │ Performance │ 5000.0 │ Gflop/s │\n"
        "╘═══╛\n"
        f"│ 17.1.5 │ Peak HBM │ {real_peak} │ GB/s │\n"
        "4.3 Roofline Plot\n"
    )


# ---- pure helpers ----

def test_safe_float():
    assert rr._safe_float("3.5") == 3.5
    assert rr._safe_float("nan-ish") is None
    assert rr._safe_float(None) is None


@pytest.mark.parametrize("text,expected", [
    ("memory-bound xyz", "memory"),
    ("compute-bound xyz", "compute"),
    ("latency-bound xyz", "latency"),
    ("weird", "unknown"),
    ("", "unknown"),
])
def test_bound_type(text, expected):
    assert rr._bound_type(text) == expected


def test_recommended_actions():
    assert rr.recommended_actions("memory")
    assert rr.recommended_actions("compute")
    assert rr.recommended_actions("latency")
    assert rr.recommended_actions("unknown") == []


# ---- parsing ----

def test_parse_blocks_and_ai_and_real_peak():
    a = rr.RocprofRooflineAnalyzer()
    a.content = _content()
    blocks = a.parse_roofline_blocks()
    assert blocks[0][0] == "my_gemm_kernel"
    assert blocks[0][1]["HBM Bandwidth"] == (1000.0, 2000.0, "GB/s")
    ai = a.parse_roofline_ai()
    assert ai[0]["AI HBM"] == (2.5, "Flops/Byte")
    assert ai[0]["Performance (TFLOPs)"] == (5.0, "TFLOPS")
    assert a.parse_real_hbm_peak() == 3000.0


def test_parse_real_hbm_peak_fallback_metric():
    a = rr.RocprofRooflineAnalyzer()
    a.content = "│ 2.1.23 │ HBM │ x │ y │ 4242.0 │\n"
    assert a.parse_real_hbm_peak() == 4242.0


def test_parse_real_hbm_peak_none():
    a = rr.RocprofRooflineAnalyzer()
    a.content = "no peak here\n"
    assert a.parse_real_hbm_peak() is None


# ---- compute_efficiency branches ----

def test_efficiency_compute_bound():
    a = rr.RocprofRooflineAnalyzer()
    a.content = _content(mfma_actual=900.0)
    payload = a.analyze_structured()
    row = payload["results"][0]
    assert row["bottleneck"] == "compute"
    assert "compute-bound" in row["rocprof_roofline"]["bound"]


def test_efficiency_memory_bound():
    a = rr.RocprofRooflineAnalyzer()
    a.content = _content(mfma_actual=50.0, hbm_actual=2900.0, real_peak=3000.0)
    eff = a.analyze_structured()["results"][0]["rocprof_roofline"]
    assert eff["bound_type"] == "memory"


def test_efficiency_latency_bound():
    a = rr.RocprofRooflineAnalyzer()
    a.content = _content(mfma_actual=50.0, hbm_actual=100.0, real_peak=3000.0)
    eff = a.analyze_structured()["results"][0]["rocprof_roofline"]
    assert eff["bound_type"] == "latency"


def test_efficiency_latency_no_fp():
    a = rr.RocprofRooflineAnalyzer()
    a.content = _content(include_fp=False, ai_hbm=0.0)
    eff = a.analyze_structured()["results"][0]["rocprof_roofline"]
    assert eff["bound_type"] == "latency"
    assert "no FP work" in eff["bound"]


# ---- text report ----

def test_build_text_report():
    a = rr.RocprofRooflineAnalyzer()
    a.content = _content()
    report = rr.build_text_report(a.analyze_structured())
    assert "kernel function name:" in report
    assert "my_gemm_kernel" in report
    assert "Kernel bound:" in report


# ---- row projection / matching ----

def test_kernel_name_matches():
    row = {"matched_kernel_name": "foo", "name": "bar"}
    assert rr._kernel_name_matches(row, "")
    assert rr._kernel_name_matches(row, "foo")
    assert rr._kernel_name_matches(row, "bar")
    assert not rr._kernel_name_matches(row, "nope")


def test_project_payload_empty():
    assert rr._project_payload_to_row({}, "")["status"] == "failed"
    assert rr._project_payload_to_row({"results": []})["status"] == "failed"


def test_project_payload_first_row():
    payload = {"results": [{"name": "k", "rocprof_roofline": {"bound_type": "compute"}}]}
    out = rr._project_payload_to_row(payload)
    assert out["bound_type"] == "compute"
    assert out["matched_kernel_name"] == "k"


def test_project_payload_target_not_matched():
    payload = {"results": [{"name": "k", "rocprof_roofline": {}}]}
    out = rr._project_payload_to_row(payload, target_kernel="other")
    assert out["status"] == "skipped"
    assert out["reason"] == "target_kernel_not_matched"


def test_project_payload_target_matched():
    payload = {"results": [{"name": "k", "rocprof_roofline": {"ai_hbm": 1.0}}]}
    out = rr._project_payload_to_row(payload, target_kernel="k")
    assert out["target_kernel"] == "k"


# ---- workdir / atomic write ----

def test_profile_workdir_prefers_existing(tmp_path):
    f = tmp_path / "src.py"
    f.write_text("x", encoding="utf-8")
    assert rr._profile_workdir({"source_file": str(f)}, tmp_path) == tmp_path
    assert rr._profile_workdir({}, tmp_path) == tmp_path


def test_atomic_write_json(tmp_path):
    p = tmp_path / "out" / "x.json"
    rr._atomic_write_json(p, {"a": 1})
    assert json.loads(p.read_text())["a"] == 1


# ---- rocprof-compute resolution / version ----

def test_resolve_rocprof_compute_from_env(tmp_path, monkeypatch):
    tool = tmp_path / "rocprof-compute"
    tool.write_text("#!/bin/sh\n", encoding="utf-8")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("HYPERLOOM_ROCPROF_COMPUTE_PATH", str(tool))
    assert rr._resolve_rocprof_compute() == str(tool)


def test_resolve_rocprof_compute_none(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_ROCPROF_COMPUTE_PATH", raising=False)
    monkeypatch.delenv("ROCPROF_COMPUTE_PATH", raising=False)
    monkeypatch.setattr(rr.shutil, "which", lambda _x: None)
    monkeypatch.setattr(rr.Path, "is_file", lambda self: False)
    assert rr._resolve_rocprof_compute() is None


def test_check_rocprof_compute_version(monkeypatch):
    monkeypatch.setattr(rr, "_resolve_rocprof_compute", lambda: "/x/tool")

    class P:
        returncode = 0
        stdout = "rocprofiler-compute version: 3.1.0\n"

    monkeypatch.setattr(rr.subprocess, "run", lambda *a, **k: P())
    assert rr._check_rocprof_compute() == "3.1.0"


def test_check_rocprof_compute_missing(monkeypatch):
    monkeypatch.setattr(rr, "_resolve_rocprof_compute", lambda: None)
    assert rr._check_rocprof_compute() is None


def test_check_rocprof_compute_nonzero(monkeypatch):
    monkeypatch.setattr(rr, "_resolve_rocprof_compute", lambda: "/x/tool")

    class P:
        returncode = 1
        stdout = "boom"

    monkeypatch.setattr(rr.subprocess, "run", lambda *a, **k: P())
    assert rr._check_rocprof_compute() is None


# ---- run() ----

def test_run_missing_tool(monkeypatch, tmp_path):
    monkeypatch.setattr(rr, "_resolve_rocprof_compute", lambda: None)
    a = rr.RocprofRooflineAnalyzer(tmp_path)
    ok, err = a.run(workdir=str(tmp_path), cmd="echo hi")
    assert ok is False
    assert "not installed" in err


def test_run_success(monkeypatch, tmp_path):
    monkeypatch.setattr(rr, "_resolve_rocprof_compute", lambda: "/x/tool")
    monkeypatch.setattr(rr, "_check_rocprof_compute", lambda: "3.1.0")

    calls = {"n": 0}

    class P:
        def __init__(self):
            self.returncode = 0
            self.stdout = _content() if calls["n"] == 2 else "profiled"

    def fake_run(*a, **k):
        calls["n"] += 1
        return P()

    monkeypatch.setattr(rr.subprocess, "run", fake_run)
    a = rr.RocprofRooflineAnalyzer(tmp_path)
    ok, err = a.run(workdir=str(tmp_path), cmd="echo hi", target_kernel="my_gemm_kernel")
    assert ok is True
    assert err is None
    assert "Roofline Rate Metrics" in a.content


def test_run_profile_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(rr, "_resolve_rocprof_compute", lambda: "/x/tool")
    monkeypatch.setattr(rr, "_check_rocprof_compute", lambda: "3.1.0")

    class P:
        returncode = 2
        stdout = "profile error"

    monkeypatch.setattr(rr.subprocess, "run", lambda *a, **k: P())
    a = rr.RocprofRooflineAnalyzer(tmp_path)
    ok, err = a.run(workdir=str(tmp_path), cmd="echo hi")
    assert ok is False
    assert err == "profile error"


# ---- main() ----

def test_main_success(monkeypatch, tmp_path):
    def fake_run(self, **kwargs):
        self.content = _content()
        return True, None

    monkeypatch.setattr(rr.RocprofRooflineAnalyzer, "run", fake_run)
    out_json = tmp_path / "k.json"
    out_txt = tmp_path / "k.txt"
    rc = rr.main([
        "--workdir", str(tmp_path), "--cmd", "echo hi",
        "--out-json", str(out_json), "--out-txt", str(out_txt),
        "--raw-txt", str(tmp_path / "raw.txt"),
    ])
    assert rc == 0
    assert json.loads(out_json.read_text())["status"] == "ok"
    assert (tmp_path / "raw.txt").is_file()


def test_main_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        rr.RocprofRooflineAnalyzer, "run", lambda self, **k: (False, "boom"),
    )
    out_json = tmp_path / "k.json"
    out_txt = tmp_path / "k.txt"
    rc = rr.main([
        "--workdir", str(tmp_path), "--cmd", "echo hi",
        "--out-json", str(out_json), "--out-txt", str(out_txt),
    ])
    assert rc == 1
    assert json.loads(out_json.read_text())["status"] == "failed"


# ---- enrich_kernel_roofline_sidecar ----

def test_enrich_missing_inputs(tmp_path):
    out = rr.enrich_kernel_roofline_sidecar(
        sidecar_path=tmp_path / "nope.json", candidates_path=tmp_path / "no2.json",
    )
    assert out["status"] == "missing_inputs"


def test_enrich_json_load_error(tmp_path):
    sc = tmp_path / "s.json"
    cd = tmp_path / "c.json"
    sc.write_text("{bad", encoding="utf-8")
    cd.write_text("{}", encoding="utf-8")
    out = rr.enrich_kernel_roofline_sidecar(sidecar_path=sc, candidates_path=cd)
    assert out["status"].startswith("json_load_error")


def test_enrich_sidecar_missing_kernels(tmp_path):
    sc = tmp_path / "s.json"
    cd = tmp_path / "c.json"
    sc.write_text(json.dumps({"not_kernels": 1}), encoding="utf-8")
    cd.write_text(json.dumps({"hot_kernels": []}), encoding="utf-8")
    out = rr.enrich_kernel_roofline_sidecar(sidecar_path=sc, candidates_path=cd)
    assert out["status"] == "sidecar_missing_kernels"


def test_enrich_skips_non_reusable(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "_check_rocprof_compute", lambda: "3.1.0")
    sc = tmp_path / "s.json"
    cd = tmp_path / "c.json"
    sc.write_text(json.dumps({"kernels": [{"kernel_id": "1", "name": "k"}]}), encoding="utf-8")
    cd.write_text(json.dumps({"hot_kernels": [{"kernel_id": "1", "reusable_native_kernel": False}]}), encoding="utf-8")
    out = rr.enrich_kernel_roofline_sidecar(sidecar_path=sc, candidates_path=cd)
    assert out["skipped"] == 1
    assert out["status"] == "ok"


def _install_fake_harness_generator(monkeypatch, impl):
    """Inject a fake harness_generator module exposing maybe_generate_harness."""
    import sys
    import types
    mod = types.ModuleType("harness_generator")
    mod.maybe_generate_harness = impl
    monkeypatch.setitem(sys.modules, "harness_generator", mod)


def test_generate_harness_no_benchmark_files(tmp_path):
    cmd, err = rr._generate_harness_for_candidate({}, out_dir=tmp_path, log_fn=lambda m: None)
    assert cmd == ""
    assert err == "no_benchmark_files"


def test_generate_harness_no_resolvable_file(tmp_path):
    cand = {"benchmark_files": [str(tmp_path / "missing.py")]}
    cmd, err = rr._generate_harness_for_candidate(cand, out_dir=tmp_path, log_fn=lambda m: None)
    assert err == "no_resolvable_benchmark_file"


def test_generate_harness_success(tmp_path, monkeypatch):
    bench = tmp_path / "b.py"
    bench.write_text("x = 1\n", encoding="utf-8")
    import types as _t
    _install_fake_harness_generator(
        monkeypatch,
        lambda **k: _t.SimpleNamespace(test_command="python h.py --correctness"),
    )
    cand = {"benchmark_files": [str(bench)]}
    cmd, err = rr._generate_harness_for_candidate(cand, out_dir=tmp_path, log_fn=lambda m: None)
    assert err is None
    assert cmd == "python h.py --profile"


def test_generate_harness_unavailable(tmp_path, monkeypatch):
    bench = tmp_path / "b.py"
    bench.write_text("x = 1\n", encoding="utf-8")
    _install_fake_harness_generator(monkeypatch, lambda **k: None)
    cand = {"benchmark_files": [str(bench)]}
    cmd, err = rr._generate_harness_for_candidate(cand, out_dir=tmp_path, log_fn=lambda m: None)
    assert err == "harness_unavailable"


def test_generate_harness_error(tmp_path, monkeypatch):
    bench = tmp_path / "b.py"
    bench.write_text("x = 1\n", encoding="utf-8")

    def boom(**k):
        raise RuntimeError("nope")

    _install_fake_harness_generator(monkeypatch, boom)
    cand = {"benchmark_files": [str(bench)]}
    cmd, err = rr._generate_harness_for_candidate(cand, out_dir=tmp_path, log_fn=lambda m: None)
    assert err.startswith("harness_generator_error")


def test_enrich_full_matched_path(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "_check_rocprof_compute", lambda: "3.1.0")
    monkeypatch.setattr(
        rr, "_generate_harness_for_candidate",
        lambda cand, **k: ("python h.py --profile", None),
    )

    def fake_run(self, **kwargs):
        self.content = _content()
        return True, None

    monkeypatch.setattr(rr.RocprofRooflineAnalyzer, "run", fake_run)
    # Sidecar lives two levels under a reports root so out_dir resolves.
    root = tmp_path / "session" / "reports"
    root.mkdir(parents=True)
    sc = root / "kernel_roofline.json"
    cd = root / "candidates.json"
    sc.write_text(json.dumps({"kernels": [{"kernel_id": "1", "name": "my_gemm_kernel"}]}), encoding="utf-8")
    cd.write_text(json.dumps({"hot_kernels": [
        {"kernel_id": "1", "reusable_native_kernel": True, "name": "my_gemm_kernel"},
    ]}), encoding="utf-8")
    out = rr.enrich_kernel_roofline_sidecar(sidecar_path=sc, candidates_path=cd)
    assert out["matched"] == 1
    assert out["status"] == "ok"


def test_enrich_failed_run_path(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "_check_rocprof_compute", lambda: "3.1.0")
    monkeypatch.setattr(
        rr, "_generate_harness_for_candidate",
        lambda cand, **k: ("python h.py --profile", None),
    )
    monkeypatch.setattr(
        rr.RocprofRooflineAnalyzer, "run", lambda self, **k: (False, "profile boom"),
    )
    root = tmp_path / "session" / "reports"
    root.mkdir(parents=True)
    sc = root / "kernel_roofline.json"
    cd = root / "candidates.json"
    sc.write_text(json.dumps({"kernels": [{"kernel_id": "1", "name": "k"}]}), encoding="utf-8")
    cd.write_text(json.dumps({"hot_kernels": [
        {"kernel_id": "1", "reusable_native_kernel": True, "name": "k"},
    ]}), encoding="utf-8")
    out = rr.enrich_kernel_roofline_sidecar(sidecar_path=sc, candidates_path=cd)
    assert out["failed"] == 1


def test_enrich_no_test_command_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "_check_rocprof_compute", lambda: "3.1.0")
    monkeypatch.setattr(
        rr, "_generate_harness_for_candidate",
        lambda cand, **k: ("", "no_benchmark_files"),
    )
    root = tmp_path / "session" / "reports"
    root.mkdir(parents=True)
    sc = root / "kernel_roofline.json"
    cd = root / "candidates.json"
    sc.write_text(json.dumps({"kernels": [{"kernel_id": "1", "name": "k"}]}), encoding="utf-8")
    cd.write_text(json.dumps({"hot_kernels": [
        {"kernel_id": "1", "reusable_native_kernel": True, "name": "k"},
    ]}), encoding="utf-8")
    out = rr.enrich_kernel_roofline_sidecar(sidecar_path=sc, candidates_path=cd)
    assert out["skipped"] == 1


def test_enrich_skips_when_rocprof_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "_check_rocprof_compute", lambda: None)
    sc = tmp_path / "s.json"
    cd = tmp_path / "c.json"
    sc.write_text(json.dumps({"kernels": [{"kernel_id": "1", "name": "k"}]}), encoding="utf-8")
    cd.write_text(json.dumps({"hot_kernels": [{"kernel_id": "1", "reusable_native_kernel": True}]}), encoding="utf-8")
    logs = []
    out = rr.enrich_kernel_roofline_sidecar(
        sidecar_path=sc, candidates_path=cd, log_fn=logs.append,
    )
    assert out["skipped"] == 1
