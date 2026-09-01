# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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

from hyperloom.inference_optimizer.breakdown.session_package import (
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
    _write(sd / "reports" / "sbd_v6" / "timeline" / "000001-install.json", "{}")
    _write(sd / "reports" / "sbd_v6" / "timeline" / "000002-model_gate.json", "{}")
    _write(sd / "reports" / "sbd_v6" / "write_warnings.jsonl", "{}\n")
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
    _write(sd / "runs" / "specialist" / "s1" / "prompt.md", "prompt")


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
        "session_breakdown.json",
        "state.json",
        "manifest.json",
        "reports/final.json",
        "reports/final.md",
        "reports/optimization_journal.json",
        "reports/kernel_optimization_summary.json",
        "reports/kernel_roofline.json",
        "reports/sbd_v6/timeline/000001-install.json",
        "reports/sbd_v6/timeline/000002-model_gate.json",
        "reports/sbd_v6/write_warnings.jsonl",
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
    (sd / "runs").mkdir()
    out = package_session_artifacts(sd, session_id="empty", dest_root=tmp_path / "ws")
    assert out is None


def test_missing_session_dir_returns_none(tmp_path: Path) -> None:
    out = package_session_artifacts(
        tmp_path / "does-not-exist",
        session_id="x",
        dest_root=tmp_path / "ws",
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

    # loose files land directly under the dest root, keeping their relative path
    for rel in (
        "session_breakdown.json",
        "state.json",
        "reports/final.json",
        "reports/sbd_v6/timeline/000001-install.json",
        "reports/sbd_v6/timeline/000002-model_gate.json",
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
    from hyperloom.inference_optimizer.breakdown import session_package as sp

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
    keep = dest / "unrelated-other-session.txt"
    keep.write_text("keep me", encoding="utf-8")

    package_session_artifacts(sd, session_id="sid-123", dest_root=dest)

    assert keep.is_file()  # loose copy must never blow away the root
    assert keep.read_text(encoding="utf-8") == "keep me"
    assert (dest / "session_breakdown.json").is_file()  # loose copy still landed


def test_current_setting_sh_is_included(tmp_path: Path) -> None:
    """current_setting.sh at session root is collected."""
    sd = tmp_path / "session"
    _write(sd / "session_breakdown.json", "{}")
    _write(sd / "current_setting.sh", "#!/usr/bin/env bash\nvllm serve $MODEL\n")
    dest = tmp_path / "ws"

    out = package_session_artifacts(sd, session_id="cs-sid", dest_root=dest)
    assert out is not None
    assert "current_setting.sh" in _zip_names(out)


def test_enablement_artifacts_are_included(tmp_path: Path) -> None:
    """reports/enablement/** covers round.json, patches, and the setting script."""
    sd = tmp_path / "session"
    _write(sd / "session_breakdown.json", "{}")
    _write(sd / "reports" / "enablement" / "tid-abc" / "round.json", '{"status":"kept"}')
    _write(sd / "reports" / "enablement" / "tid-abc" / "patches" / "001_fix.patch", "diff\n")
    _write(sd / "reports" / "enablement" / "enablement_setting.sh", "#!/usr/bin/env bash\n")
    dest = tmp_path / "ws"

    out = package_session_artifacts(sd, session_id="en-sid", dest_root=dest)
    assert out is not None
    names = _zip_names(out)
    assert "reports/enablement/tid-abc/round.json" in names
    assert "reports/enablement/tid-abc/patches/001_fix.patch" in names
    assert "reports/enablement/enablement_setting.sh" in names


# ---- session boundary ------------------------------------------------------
def _manifest(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read(MANIFEST_JSON_NAME))


def test_symlink_out_of_the_session_is_not_bundled(tmp_path: Path) -> None:
    """A link planted in the session must not pull outside content into the bundle."""
    sd = tmp_path / "session"
    _write(sd / "session_breakdown.json", "{}")
    outside = tmp_path / "outside" / "secret.json"
    _write(outside, '{"secret":"OUTSIDE"}')
    (sd / "reports").mkdir(parents=True, exist_ok=True)
    (sd / "reports" / "final.json").symlink_to(outside)
    dest = tmp_path / "workspace"

    out = package_session_artifacts(sd, session_id="sid", dest_root=dest)

    assert out is not None
    names = _zip_names(out)
    assert "reports/final.json" not in names
    with zipfile.ZipFile(out) as zf:
        for name in names:
            assert b"OUTSIDE" not in zf.read(name)
    assert not (dest / "reports" / "final.json").exists()

    manifest = _manifest(out)
    assert "reports/final.json" in manifest["refused_files"]
    assert manifest["complete"] is False


def test_symlink_staying_inside_the_session_is_still_bundled(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    _write(sd / "session_breakdown.json", "{}")
    real = sd / "runs" / "b1" / "benchmark_report.json"
    _write(real, '{"tput":1}')
    (sd / "reports").mkdir(parents=True, exist_ok=True)
    (sd / "reports" / "final.json").symlink_to(real)

    out = package_session_artifacts(sd, session_id="sid", dest_root=tmp_path / "ws")

    assert out is not None
    assert "reports/final.json" in _zip_names(out)
    assert _manifest(out)["refused_files"] == []


def test_non_regular_files_are_refused(tmp_path: Path) -> None:
    """Sockets/FIFOs are not artifacts and cannot be archived."""
    import os

    sd = tmp_path / "session"
    _write(sd / "session_breakdown.json", "{}")
    (sd / "reports").mkdir(parents=True, exist_ok=True)
    os.mkfifo(sd / "reports" / "final.json")

    out = package_session_artifacts(sd, session_id="sid", dest_root=tmp_path / "ws")

    assert out is not None
    assert "reports/final.json" not in _zip_names(out)
    assert "reports/final.json" in _manifest(out)["refused_files"]


# ---- manifest describes what was written ----------------------------------
def test_manifest_never_claims_a_file_the_zip_lacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sd = tmp_path / "session"
    _build_session(sd)
    real_write = zipfile.ZipFile.write

    def _flaky(self, filename, arcname=None, **kw):  # noqa: ANN001, ANN202
        if arcname == "reports/final.json":
            raise OSError("disk gone")
        return real_write(self, filename, arcname=arcname, **kw)

    monkeypatch.setattr(zipfile.ZipFile, "write", _flaky)

    out = package_session_artifacts(sd, session_id="sid", dest_root=tmp_path / "ws")

    assert out is not None
    names = _zip_names(out)
    manifest = _manifest(out)
    claimed = {e["path"] for e in manifest["included_files"]}
    assert claimed <= names
    assert "reports/final.json" not in claimed
    assert manifest["failed_files"] == ["reports/final.json"]
    assert manifest["complete"] is False
    assert manifest["included_count"] == len(manifest["included_files"])


def test_loose_manifest_describes_the_loose_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A loose copy failure is reported in the loose manifest, not the zip's."""
    from hyperloom.inference_optimizer.breakdown import session_package as sp

    sd = tmp_path / "session"
    _build_session(sd)
    dest = tmp_path / "workspace"
    real_copy = sp.shutil.copy2

    def _flaky(src, dst, **kw):  # noqa: ANN001, ANN202
        if Path(dst).name == "final.json":
            raise OSError("disk gone")
        return real_copy(src, dst, **kw)

    monkeypatch.setattr(sp.shutil, "copy2", _flaky)

    out = package_session_artifacts(sd, session_id="sid", dest_root=dest)

    assert out is not None
    assert "reports/final.json" in _zip_names(out)  # zip was fine
    assert _manifest(out)["failed_files"] == []
    loose = json.loads((dest / MANIFEST_JSON_NAME).read_text(encoding="utf-8"))
    assert loose["failed_files"] == ["reports/final.json"]
    assert loose["complete"] is False
    assert "reports/final.json" not in {e["path"] for e in loose["included_files"]}


def test_complete_flag_set_on_a_clean_bundle(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    _build_session(sd)

    out = package_session_artifacts(sd, session_id="sid", dest_root=tmp_path / "ws")

    assert out is not None
    manifest = _manifest(out)
    assert manifest["complete"] is True
    assert manifest["failed_files"] == []
    assert manifest["refused_files"] == []
    assert manifest["dropped_files"] == []


def test_truncated_bundle_is_not_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hyperloom.inference_optimizer.breakdown import session_package as sp

    monkeypatch.setattr(sp, "_MAX_TOTAL_BYTES", 5)
    sd = tmp_path / "session"
    _build_session(sd)

    out = sp.package_session_artifacts(sd, session_id="sid", dest_root=tmp_path / "ws")

    assert out is not None
    assert _manifest(out)["complete"] is False


def test_manifest_text_names_missing_files_for_a_human_reader(tmp_path: Path) -> None:
    sd = tmp_path / "session"
    _write(sd / "session_breakdown.json", "{}")
    outside = tmp_path / "outside" / "secret.json"
    _write(outside, "{}")
    (sd / "reports").mkdir(parents=True, exist_ok=True)
    (sd / "reports" / "final.json").symlink_to(outside)

    out = package_session_artifacts(sd, session_id="sid", dest_root=tmp_path / "ws")

    assert out is not None
    with zipfile.ZipFile(out) as zf:
        text = zf.read(MANIFEST_TXT_NAME).decode("utf-8")
    assert "complete     : False" in text
    assert "REFUSED" in text
    assert "reports/final.json" in text
