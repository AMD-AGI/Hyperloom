# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for breakdown.session_package.package_session_artifacts.

Builds a synthetic session dir mixing curated artifacts with bulky
``runs/`` traces + per-turn agent dumps, then asserts the zip contains
exactly the curated set (plus the manifest log) and excludes the noise.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from inference_optimizer.breakdown.session_package import (
    ENV_PACKAGE_LOOSE,
    MANIFEST_JSON_NAME,
    MANIFEST_TXT_NAME,
    PACKAGE_SUBDIR,
    package_session_artifacts,
)


def _write(p: Path, content: str = "x") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _build_session(sd: Path) -> None:
    # curated (must be included)
    _write(sd / "session_breakdown.json", '{"a":1}')
    _write(sd / "state.json", "{}")
    _write(sd / "manifest.json", "{}")
    _write(sd / "reports" / "final.json", "{}")
    _write(sd / "reports" / "final.md", "# final")
    _write(sd / "reports" / "optimization_journal.json", "[]")
    _write(sd / "reports" / "kernel_optimization_summary.json", "{}")
    _write(sd / "reports" / "kernel_roofline.json", "{}")
    _write(sd / "reports" / "trace" / "decision_trace.jsonl", "{}\n")
    _write(sd / "reports" / "trace" / "llm_calls.jsonl", "{}\n")
    _write(sd / "target_analysis" / "target_baseline.json", "{}")
    _write(sd / "target_analysis" / "target_analysis_report.md", "# t")
    _write(sd / "storage" / "coordinator.db", "sqlite")
    # tracelens family under a dynamic <ts>/<tl-id> path
    tl = sd / "kernel-agent" / "runs" / "20260609T010022Z" / "20260609T012416Z_tl-abc" / "tracelens"
    _write(tl / "analysis.md", "# analysis")
    _write(tl / "tracelens_report.json", "{}")
    _write(tl / "summary.json", "{}")
    _write(tl / "priority_data.json", "{}")
    _write(tl.parent / "kernel_candidates.json", "[]")
    _write(tl.parent / "trace_input_manifest.json", "{}")
    _write(tl / "category_findings" / "gemm_findings.md", "# g")
    _write(tl / "system_findings" / "cpu_idle_findings.md", "# c")
    _write(tl / "perf_report_csvs" / "GEMM.csv", "a,b\n1,2\n")
    # per-run reports (curated)
    run = sd / "runs" / "baseline" / "abc" / "measure_round" / "benchmark_sglang_x"
    _write(run / "benchmark_report.json", "{}")
    _write(run / "summary.txt", "ok")
    _write(run / "inferencex_result.json", "{}")
    _write(sd / "runs" / "gemm_tuning" / "k" / "logs" / "final_report.json", "{}")
    _write(sd / "runs" / "gemm_tuning" / "k" / "logs" / "best_results.json", "{}")
    _write(sd / "runs" / "specialist" / "s1" / "specialist_done.json", "{}")
    _write(sd / "runs" / "recover" / "r1" / "result.json", "{}")

    # NOISE — must be excluded
    _write(run / "torch_trace" / "x.trace.json.gz", "BLOB")
    _write(tl / "trace_split" / "y.trace.json.gz", "BLOB")
    _write(sd / "critic-workdir" / "000001" / "request.json", "{}")
    _write(sd / "robustness-workdir" / "000001" / "request.json", "{}")
    _write(sd / "hands.log", "log")
    _write(sd / "runs" / "specialist" / "s1" / "prompt.md", "prompt")  # not in spec


def _zip_names(zip_path: Path) -> set[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return set(zf.namelist())


def test_package_includes_curated_excludes_noise(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    _build_session(sd)
    dest = tmp_path / "workspace"

    out = package_session_artifacts(sd, session_id="sid-123", dest_root=dest)
    assert out is not None
    assert out == dest / PACKAGE_SUBDIR / "sid-123.zip"
    assert out.exists()

    names = _zip_names(out)

    # curated present
    for rel in (
        "session_breakdown.json", "state.json", "manifest.json",
        "reports/final.json", "reports/final.md",
        "reports/optimization_journal.json",
        "reports/kernel_optimization_summary.json",
        "reports/kernel_roofline.json",
        "reports/trace/decision_trace.jsonl",
        "reports/trace/llm_calls.jsonl",
        "target_analysis/target_baseline.json",
        "target_analysis/target_analysis_report.md",
        "storage/coordinator.db",
        "kernel-agent/runs/20260609T010022Z/20260609T012416Z_tl-abc/tracelens/analysis.md",
        "kernel-agent/runs/20260609T010022Z/20260609T012416Z_tl-abc/tracelens/tracelens_report.json",
        "kernel-agent/runs/20260609T010022Z/20260609T012416Z_tl-abc/tracelens/summary.json",
        "kernel-agent/runs/20260609T010022Z/20260609T012416Z_tl-abc/tracelens/priority_data.json",
        "kernel-agent/runs/20260609T010022Z/20260609T012416Z_tl-abc/kernel_candidates.json",
        "kernel-agent/runs/20260609T010022Z/20260609T012416Z_tl-abc/trace_input_manifest.json",
        "kernel-agent/runs/20260609T010022Z/20260609T012416Z_tl-abc/tracelens/category_findings/gemm_findings.md",
        "kernel-agent/runs/20260609T010022Z/20260609T012416Z_tl-abc/tracelens/system_findings/cpu_idle_findings.md",
        "kernel-agent/runs/20260609T010022Z/20260609T012416Z_tl-abc/tracelens/perf_report_csvs/GEMM.csv",
        "runs/baseline/abc/measure_round/benchmark_sglang_x/benchmark_report.json",
        "runs/baseline/abc/measure_round/benchmark_sglang_x/summary.txt",
        "runs/baseline/abc/measure_round/benchmark_sglang_x/inferencex_result.json",
        "runs/gemm_tuning/k/logs/final_report.json",
        "runs/gemm_tuning/k/logs/best_results.json",
        "runs/specialist/s1/specialist_done.json",
        "runs/recover/r1/result.json",
    ):
        assert rel in names, f"missing curated: {rel}"

    # noise excluded
    for rel in (
        "runs/baseline/abc/measure_round/benchmark_sglang_x/torch_trace/x.trace.json.gz",
        "kernel-agent/runs/20260609T010022Z/20260609T012416Z_tl-abc/tracelens/trace_split/y.trace.json.gz",
        "critic-workdir/000001/request.json",
        "robustness-workdir/000001/request.json",
        "hands.log",
        "runs/specialist/s1/prompt.md",
    ):
        assert rel not in names, f"noise leaked in: {rel}"

    # manifest log present + inside the zip
    assert MANIFEST_JSON_NAME in names
    assert MANIFEST_TXT_NAME in names


def test_manifest_records_included_and_missing(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    _build_session(sd)
    # remove one curated file so its glob shows up as unmatched
    (sd / "reports" / "kernel_roofline.json").unlink()
    dest = tmp_path / "workspace"

    out = package_session_artifacts(sd, session_id="sid-x", dest_root=dest)
    assert out is not None
    with zipfile.ZipFile(out) as zf:
        manifest = json.loads(zf.read(MANIFEST_JSON_NAME))

    assert manifest["session_id"] == "sid-x"
    assert manifest["included_count"] == len(manifest["included_files"])
    incl_paths = {e["path"] for e in manifest["included_files"]}
    assert "session_breakdown.json" in incl_paths
    assert "reports/kernel_roofline.json" not in incl_paths
    # the removed file's glob is reported as unmatched
    assert "reports/kernel_roofline.json" in manifest["unmatched_globs"]
    # conc_sweep_summary was never created → also unmatched (audit signal)
    assert "reports/conc_sweep_summary.json" in manifest["unmatched_globs"]


def test_empty_session_returns_none(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    sd.mkdir()
    (sd / "runs").mkdir()  # only noise-ish empty dirs
    out = package_session_artifacts(sd, session_id="empty", dest_root=tmp_path / "ws")
    assert out is None


def test_missing_session_dir_returns_none(tmp_path: Path) -> None:
    out = package_session_artifacts(
        tmp_path / "does-not-exist", session_id="x", dest_root=tmp_path / "ws",
    )
    assert out is None


def test_session_id_falls_back_to_dirname(tmp_path: Path) -> None:
    sd = tmp_path / "Qwen_20260609T010022Z"
    _write(sd / "session_breakdown.json", "{}")
    out = package_session_artifacts(sd, dest_root=tmp_path / "ws")
    assert out is not None
    assert out.name == "Qwen_20260609T010022Z.zip"


def test_atomic_no_tmp_left_behind(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    _write(sd / "session_breakdown.json", "{}")
    dest = tmp_path / "ws"
    package_session_artifacts(sd, session_id="sid", dest_root=dest)
    leftovers = list((dest / PACKAGE_SUBDIR).glob(".*tmp"))
    assert leftovers == []


def test_loose_files_dropped_at_dest_root(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    _build_session(sd)
    dest = tmp_path / "workspace"

    out = package_session_artifacts(sd, session_id="sid-123", dest_root=dest)
    assert out is not None
    assert out.exists()  # zip still under the package subdir
    assert out == dest / PACKAGE_SUBDIR / "sid-123.zip"

    # loose files land DIRECTLY under the dest root (not under <sid>/),
    # preserving each file's original relative path
    for rel in (
        "session_breakdown.json",
        "state.json",
        "reports/final.json",
        "reports/trace/decision_trace.jsonl",
        "kernel-agent/runs/20260609T010022Z/20260609T012416Z_tl-abc/tracelens/analysis.md",
        "runs/baseline/abc/measure_round/benchmark_sglang_x/benchmark_report.json",
    ):
        assert (dest / rel).is_file(), f"missing loose file: {rel}"

    # noise stays out of the loose tree too
    assert not (dest / "hands.log").exists()
    assert not (dest / "critic-workdir" / "000001" / "request.json").exists()

    # manifest also written loose at the root
    assert (dest / MANIFEST_JSON_NAME).is_file()
    assert (dest / MANIFEST_TXT_NAME).is_file()

    # loose content matches source byte-for-byte
    assert (dest / "session_breakdown.json").read_text(encoding="utf-8") == '{"a":1}'


def test_loose_can_be_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_PACKAGE_LOOSE, "0")
    sd = tmp_path / "session"
    _build_session(sd)
    dest = tmp_path / "workspace"

    out = package_session_artifacts(sd, session_id="sid-123", dest_root=dest)
    assert out is not None and out.exists()  # zip still written
    assert not (dest / "session_breakdown.json").exists()  # no loose copies


def test_truncation_is_flagged_in_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import inference_optimizer.breakdown.session_package as sp

    # Force a tiny byte cap so the bundle truncates after the first file.
    monkeypatch.setattr(sp, "_MAX_TOTAL_BYTES", 5)
    sd = tmp_path / "session"
    _build_session(sd)
    dest = tmp_path / "workspace"

    out = sp.package_session_artifacts(sd, session_id="sid-trunc", dest_root=dest)
    assert out is not None
    with zipfile.ZipFile(out) as zf:
        manifest = json.loads(zf.read(MANIFEST_JSON_NAME))

    assert manifest["truncated"] is True
    assert len(manifest["dropped_files"]) > 0  # what got left out is recorded
    # the dropped set and the included set are disjoint
    incl = {e["path"] for e in manifest["included_files"]}
    assert not (incl & set(manifest["dropped_files"]))


def test_no_truncation_flag_when_under_cap(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    _build_session(sd)
    out = package_session_artifacts(sd, session_id="sid-ok", dest_root=tmp_path / "ws")
    assert out is not None
    with zipfile.ZipFile(out) as zf:
        manifest = json.loads(zf.read(MANIFEST_JSON_NAME))
    assert manifest["truncated"] is False
    assert manifest["dropped_files"] == []


def test_loose_does_not_wipe_dest_root(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    _build_session(sd)
    dest = tmp_path / "workspace"
    dest.mkdir()
    # a pre-existing unrelated file at the dest root must survive
    keep = dest / "unrelated-other-session.txt"
    keep.write_text("keep me", encoding="utf-8")

    package_session_artifacts(sd, session_id="sid-123", dest_root=dest)

    assert keep.is_file()  # loose copy must never blow away the root
    assert keep.read_text(encoding="utf-8") == "keep me"
    assert (dest / "session_breakdown.json").is_file()  # loose copy still landed
