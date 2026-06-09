from __future__ import annotations

import json
from pathlib import Path

import kernel_optimization as ko
from rocprof_roofline import RocprofRooflineAnalyzer, build_text_report


SAMPLE_ROOFLINE = """
Kernel 0: triton_my_kernel (100%)
4.1 Roofline Rate Metrics
│ 4.1.1 │ HBM Bandwidth │ 300 │ GB/s │ 200 │
│ 4.1.2 │ MFMA FLOPs (F16) │ 1000 │ Gflop/s │ 2000 │
╘══════════════════════════════════════════════════════════════════════
4.2 Roofline AI Plot Points
│ 4.2.1 │ AI HBM │ 4 │ FLOPs/byte │
│ 4.2.3 │ Performance │ 500 │ Gflop/s │
╘══════════════════════════════════════════════════════════════════════
17. L2 Cache
│ 17.1.5 │ HBM Bandwidth │ 400 │ GB/s │
""".strip()


def test_rocprof_roofline_mapper_exposes_efficiency_percentage():
    analyzer = RocprofRooflineAnalyzer()
    analyzer.content = SAMPLE_ROOFLINE

    payload = analyzer.analyze_structured()
    row = payload["results"][0]
    roof = row["rocprof_roofline"]

    assert row["name"] == "triton_my_kernel"
    assert roof["bound_type"] == "memory"
    assert roof["ai_hbm"] == 4.0
    assert roof["perf_gflops"] == 500.0
    assert roof["compute_utilization_pct"] == 50.0
    assert roof["bandwidth_utilization_pct"] == 75.0
    assert roof["roofline_efficiency_pct"] == 62.5
    assert roof["roofline_efficiency_basis"] == "empirical_peak"
    assert "Roofline Eff (empirical peak): 62.5%" in build_text_report(payload)


def test_rocprof_compute_resolver_uses_configured_absolute_path(tmp_path: Path, monkeypatch):
    import rocprof_roofline as rr

    tool = tmp_path / "rocprof-compute"
    tool.write_text("#!/bin/sh\n", encoding="utf-8")
    tool.chmod(0o755)
    monkeypatch.setenv("HYPERLOOM_ROCPROF_COMPUTE_PATH", str(tool))
    monkeypatch.setattr(rr.shutil, "which", lambda _name: None)

    seen = {}

    def fake_run(cmd, **_kwargs):
        seen["cmd"] = cmd
        return type("Proc", (), {
            "returncode": 0,
            "stdout": "rocprofiler-compute version: 1.2.3\n",
        })()

    monkeypatch.setattr(rr.subprocess, "run", fake_run)

    assert rr._resolve_rocprof_compute() == str(tool)
    assert rr._check_rocprof_compute() == "1.2.3"
    assert seen["cmd"] == [str(tool), "--version"]


def test_rocprof_run_uses_resolved_absolute_path(tmp_path: Path, monkeypatch):
    import rocprof_roofline as rr

    monkeypatch.setattr(rr, "_resolve_rocprof_compute", lambda: "/opt/rocm/bin/rocprof-compute")
    monkeypatch.setattr(rr, "_check_rocprof_compute", lambda: "1.2.3")
    commands = []

    def fake_run(cmd, **_kwargs):
        commands.append(cmd[0])
        stdout = SAMPLE_ROOFLINE if " analyze " in cmd[0] else ""
        return type("Proc", (), {"returncode": 0, "stdout": stdout})()

    monkeypatch.setattr(rr.subprocess, "run", fake_run)

    ok, error = RocprofRooflineAnalyzer(tmp_path / "out").run(
        workdir=str(tmp_path),
        cmd="python harness.py --profile",
        target_kernel="triton_my_kernel",
    )

    assert ok is True
    assert error is None
    assert commands[0].startswith("/opt/rocm/bin/rocprof-compute profile")
    assert commands[1].startswith("/opt/rocm/bin/rocprof-compute analyze")


def _make_sidecar(path: Path, kernel_id: str = "k001") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "source": "tracelens_analysis",
        "kernels": [{"kernel_id": kernel_id, "name": "fused_kernel", "bottleneck": "unknown", "reusable_native_kernel": True}],
    }), encoding="utf-8")


def _make_rocprof_json(path: Path) -> None:
    path.write_text(json.dumps({
        "status": "ok",
        "results": [{
            "name": "triton_my_kernel",
            "matched_kernel_name": "triton_my_kernel",
            "status": "matched",
            "rocprof_roofline": {
                "bound_type": "memory",
                "ai_hbm": 4.0,
                "roofline_efficiency_pct": 62.5,
                "compute_utilization_pct": 50.0,
                "bandwidth_utilization_pct": 75.0,
            },
        }],
    }), encoding="utf-8")


def test_kernel_roofline_sidecar_before_kernel_opt_schema(tmp_path: Path):
    """_update_kernel_roofline_sidecar writes into before_kernel_opt sub-key."""
    workspace = tmp_path / "session"
    sidecar = workspace / "reports" / "kernel_roofline.json"
    _make_sidecar(sidecar)
    rocprof_json = tmp_path / "before.json"
    rocprof_txt = tmp_path / "before.txt"
    rocprof_txt.write_text("ROOFLINE CLASSIFICATION\n", encoding="utf-8")
    _make_rocprof_json(rocprof_json)

    ko._update_kernel_roofline_sidecar(
        workspace_path=str(workspace),
        kernel_id="k001",
        rocprof_json_path=str(rocprof_json),
        rocprof_txt_path=str(rocprof_txt),
        log_path=None,
        phase="before_kernel_opt",
    )

    updated = json.loads(sidecar.read_text(encoding="utf-8"))
    row = updated["kernels"][0]
    assert updated["source"] == "tracelens_analysis+rocprof_roofline"
    rr = row["rocprof_roofline"]
    assert "before_kernel_opt" in rr
    assert "after_kernel_opt" in rr
    assert rr["after_kernel_opt"] is None
    assert rr["before_kernel_opt"]["matched_kernel_name"] == "triton_my_kernel"
    assert row["efficiency_percent"] == 62.5
    assert row["compute_utilization_pct"] == 50.0
    assert row["bandwidth_utilization_pct"] == 75.0


def test_sidecar_kernel_name_without_metrics_does_not_tag_source(tmp_path: Path):
    """A matched row carrying only a kernel name (no numeric roofline metrics)
    must NOT flip source to +rocprof_roofline."""
    workspace = tmp_path / "session"
    sidecar = workspace / "reports" / "kernel_roofline.json"
    _make_sidecar(sidecar)
    rocprof_json = tmp_path / "before.json"
    rocprof_txt = tmp_path / "before.txt"
    rocprof_txt.write_text("no metrics\n", encoding="utf-8")
    rocprof_json.write_text(json.dumps({
        "status": "ok",
        "results": [{
            "name": "triton_my_kernel",
            "matched_kernel_name": "triton_my_kernel",
            "status": "matched",
            "rocprof_roofline": {},
        }],
    }), encoding="utf-8")

    ko._update_kernel_roofline_sidecar(
        workspace_path=str(workspace),
        kernel_id="k001",
        rocprof_json_path=str(rocprof_json),
        rocprof_txt_path=str(rocprof_txt),
        log_path=None,
        phase="before_kernel_opt",
    )

    updated = json.loads(sidecar.read_text(encoding="utf-8"))
    assert updated["source"] == "tracelens_analysis"
    rr = updated["kernels"][0]["rocprof_roofline"]
    assert rr["before_kernel_opt"]["matched_kernel_name"] == "triton_my_kernel"


def test_kernel_roofline_sidecar_after_kernel_opt_schema(tmp_path: Path):
    """after_kernel_opt is written independently without overwriting before."""
    workspace = tmp_path / "session"
    sidecar = workspace / "reports" / "kernel_roofline.json"
    workspace_reports = workspace / "reports"
    workspace_reports.mkdir(parents=True)
    # Pre-populate with existing before_kernel_opt
    sidecar.write_text(json.dumps({
        "schema_version": 1,
        "source": "tracelens_analysis+rocprof_roofline",
        "kernels": [{
            "kernel_id": "k001",
            "reusable_native_kernel": True,
            "rocprof_roofline": {
                "before_kernel_opt": {"status": "matched", "roofline_efficiency_pct": 62.5},
                "after_kernel_opt": None,
            },
        }],
    }), encoding="utf-8")

    rocprof_json = tmp_path / "after.json"
    rocprof_txt = tmp_path / "after.txt"
    rocprof_txt.write_text("after roofline\n", encoding="utf-8")
    _make_rocprof_json(rocprof_json)

    ko._update_kernel_roofline_sidecar(
        workspace_path=str(workspace),
        kernel_id="k001",
        rocprof_json_path=str(rocprof_json),
        rocprof_txt_path=str(rocprof_txt),
        log_path=None,
        phase="after_kernel_opt",
    )

    updated = json.loads(sidecar.read_text(encoding="utf-8"))
    row = updated["kernels"][0]
    rr = row["rocprof_roofline"]
    # before is preserved
    assert rr["before_kernel_opt"]["roofline_efficiency_pct"] == 62.5
    # after is filled in
    assert rr["after_kernel_opt"] is not None
    assert rr["after_kernel_opt"]["matched_kernel_name"] == "triton_my_kernel"


def test_generated_harness_uses_profile_mode_for_rocprof():
    cmd = "python /tmp/run/unittest/harness_moe.py --correctness"
    assert ko._rocprof_profile_command(cmd) == "python /tmp/run/unittest/harness_moe.py --profile"
    user_cmd = "python /tmp/custom.py --correctness"
    assert ko._rocprof_profile_command(user_cmd) == user_cmd


def test_sidecar_writes_skipped_status_without_json(tmp_path: Path):
    """k001/k002-style skipped attempts (no benchmark_files -> no rocprof JSON)
    must tag before_kernel_opt with status/reason instead of leaving null."""
    workspace = tmp_path / "session"
    sidecar = workspace / "reports" / "kernel_roofline.json"
    _make_sidecar(sidecar, kernel_id="k002")

    ko._update_kernel_roofline_sidecar(
        workspace_path=str(workspace),
        kernel_id="k002",
        rocprof_json_path="",
        rocprof_txt_path="",
        log_path=None,
        rocprof_status="skipped",
        rocprof_reason="missing_test_command",
        phase="before_kernel_opt",
    )

    updated = json.loads(sidecar.read_text(encoding="utf-8"))
    row = updated["kernels"][0]
    assert updated["source"] == "tracelens_analysis"
    rr = row["rocprof_roofline"]
    assert rr["before_kernel_opt"] == {"status": "skipped", "reason": "missing_test_command"}
    assert rr["after_kernel_opt"] is None


def test_enrich_marks_no_benchmark_files_skipped(tmp_path: Path, monkeypatch):
    """``enrich_kernel_roofline_sidecar`` must label candidates without
    benchmark files as ``status=skipped reason=no_benchmark_files`` under
    ``before_kernel_opt``."""
    from rocprof_roofline import enrich_kernel_roofline_sidecar
    import rocprof_roofline as rr

    monkeypatch.setattr(rr, "_check_rocprof_compute", lambda: "stub")

    sidecar = tmp_path / "reports" / "kernel_roofline.json"
    cands = tmp_path / "kernel_candidates.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps({
        "schema_version": 1,
        "kernels": [
            {"kernel_id": "k001", "name": "ck_moe_stage1", "reusable_native_kernel": True},
            {"kernel_id": "k042", "name": "aten::mm", "reusable_native_kernel": False},
        ],
    }), encoding="utf-8")
    cands.write_text(json.dumps({
        "hot_kernels": [
            {
                "kernel_id": "k001",
                "name": "ck_moe_stage1",
                "reusable_native_kernel": True,
                "benchmark_files": [],
                "source_file": "/sgl-workspace/aiter/csrc/foo.cu",
            },
            {
                "kernel_id": "k042",
                "name": "aten::mm",
                "reusable_native_kernel": False,
                "benchmark_files": [],
            },
        ],
    }), encoding="utf-8")

    summary = enrich_kernel_roofline_sidecar(
        sidecar_path=sidecar,
        candidates_path=cands,
        workdir=tmp_path,
    )

    updated = json.loads(sidecar.read_text(encoding="utf-8"))
    rows = {r["kernel_id"]: r for r in updated["kernels"]}
    assert rows["k001"]["rocprof_roofline"]["before_kernel_opt"] == {
        "status": "skipped",
        "reason": "no_benchmark_files",
    }
    assert rows["k001"]["rocprof_roofline"]["after_kernel_opt"] is None
    assert rows["k042"]["rocprof_roofline"]["before_kernel_opt"] == {
        "status": "skipped",
        "reason": "not_reusable_native_kernel",
    }
    assert summary["skipped"] == 2
    assert summary["matched"] == 0
    assert summary["status"] == "ok"
    assert updated.get("source") is None


def test_enrich_marks_all_skipped_when_rocprof_unavailable(tmp_path: Path, monkeypatch):
    """When rocprof-compute is missing, every reusable row gets a clear
    ``rocprof_compute_unavailable`` reason under before_kernel_opt."""
    from rocprof_roofline import enrich_kernel_roofline_sidecar
    import rocprof_roofline as rr
    monkeypatch.setattr(rr, "_check_rocprof_compute", lambda: None)

    sidecar = tmp_path / "reports" / "kernel_roofline.json"
    cands = tmp_path / "kernel_candidates.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps({
        "kernels": [{"kernel_id": "k005", "reusable_native_kernel": True}],
    }), encoding="utf-8")
    cands.write_text(json.dumps({
        "hot_kernels": [{
            "kernel_id": "k005",
            "reusable_native_kernel": True,
            "benchmark_files": ["/tmp/test.py"],
        }],
    }), encoding="utf-8")

    enrich_kernel_roofline_sidecar(
        sidecar_path=sidecar,
        candidates_path=cands,
        workdir=tmp_path,
    )
    updated = json.loads(sidecar.read_text(encoding="utf-8"))
    rr_row = updated["kernels"][0]["rocprof_roofline"]
    assert rr_row["before_kernel_opt"]["reason"] == "rocprof_compute_unavailable"
    assert rr_row["after_kernel_opt"] is None
    assert updated.get("source") is None
