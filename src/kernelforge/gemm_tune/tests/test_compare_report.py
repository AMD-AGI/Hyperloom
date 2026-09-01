# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for aiter compare-report lookup when stdout has no Pre/Post table.

When aiter tunes >30 shapes it writes the --compare table to
``/tmp/aiter_compare/tuned_<tuner>.<pid>.compare.txt`` instead of stdout.
"""

from __future__ import annotations

import os
import time
import types
from pathlib import Path

import kernelforge.gemm_tune.tuners._aiter_dense_common as ac
from kernelforge.gemm_tune.tuners.base import TuneContext

_HDR = "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio"

# Verbatim compare table (same format as stdout table).
_COMPARE_TABLE = """\
--- Would update (2 shapes) ---
Shape                                    |    Pre(us) |   Post(us) |   Improve |             Action
(8192, 5120, 5120)                       |    1037.74 |     269.71 |    74.01% |             UPDATE
(8192, 7168, 5120)                       |    1446.45 |     336.56 |    76.73% |             UPDATE
Re-run with --update_improved to apply.
"""


def _ctx(tmp_path):
    return TuneContext(
        profile=types.SimpleNamespace(),
        framework="vllm-aiter",
        precision="fp8",
        quant_type="blockscale",
        gpu_type="mi355x",
        tp=1,
        conc=64,
        tokens=[64],
        mp=1,
        output_dir=tmp_path,
        iters=1,
        warmup=0,
        min_improvement_pct=3.0,
        timeout_s=60,
        untuned_csv=tmp_path / "in.csv",
    )


def _prep(tmp_path, monkeypatch, *, compare_dir: Path | None = None):
    monkeypatch.setenv("FORGE_SPLITK_TRIAL", "0")
    (tmp_path / "in.csv").write_text("M,N,K\n8192,5120,5120\n")
    row = "gfx950,256,8192,5120,5120,ck,8,0,269.71,knl,100,1000,0.0\n"
    (tmp_path / "tuned_a8w8_blockscale.csv").write_text(_HDR + "\n" + row)
    (tmp_path / "profile_a8w8_blockscale.csv").write_text(_HDR + "\n" + row)
    monkeypatch.setattr(ac, "find_tuner_script", lambda k: tmp_path / "script.py")
    monkeypatch.setattr(ac, "_resolve_input_csv", lambda ctx, wd, needs_q_dtype_w=False: tmp_path / "in.csv")
    monkeypatch.setattr(ac, "resolve_aiter_root", lambda: str(tmp_path))
    monkeypatch.setattr(ac._tr, "is_isolation_enabled", lambda: False)
    monkeypatch.setattr(ac._tr, "with_task_timeout", lambda cmd: cmd)
    monkeypatch.setattr(ac, "run_subprocess", lambda cmd, **k: (0, "Successfully tuned 2 shapes\n", ""))
    monkeypatch.setattr(ac, "_find_latest_candidate", lambda name, t: None)
    if compare_dir is not None:
        monkeypatch.setattr(
            ac,
            "_find_latest_compare_report",
            lambda name, t: ac._find_latest_compare_report_impl(name, t, compare_dir),
        )


def _run(tmp_path):
    return ac.run_aiter_dense_tuner(
        tuner_name="a8w8_blockscale",
        script_key="a8w8_blockscale",
        env_var="AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
        ctx=_ctx(tmp_path),
        work_dir=tmp_path,
        extra_args=["--libtype", "all"],
    )


class TestFindLatestCompareReport:
    def test_rejects_stale_report(self, tmp_path):
        compare_dir = tmp_path / "aiter_compare"
        compare_dir.mkdir()
        stale = compare_dir / "tuned_a8w8_blockscale.99999.compare.txt"
        stale.write_text(_COMPARE_TABLE)
        os.utime(stale, (1000, 1000))
        start_time = time.time()
        assert ac._find_latest_compare_report_impl("a8w8_blockscale", start_time, compare_dir) is None

    def test_rejects_sibling_tuner(self, tmp_path):
        compare_dir = tmp_path / "aiter_compare"
        compare_dir.mkdir()
        start_time = time.time() - 1
        sibling = compare_dir / "tuned_a8w8_blockscale_bpreshuffle.12345.compare.txt"
        sibling.write_text(_COMPARE_TABLE)
        assert ac._find_latest_compare_report_impl("a8w8_blockscale", start_time, compare_dir) is None

    def test_rejects_concurrent_wrong_stem(self, tmp_path):
        compare_dir = tmp_path / "aiter_compare"
        compare_dir.mkdir()
        start_time = time.time() - 1
        other = compare_dir / "tuned_a4w4_blockscale.54321.compare.txt"
        other.write_text(_COMPARE_TABLE)
        assert ac._find_latest_compare_report_impl("a8w8_blockscale", start_time, compare_dir) is None

    def test_accepts_matching_report(self, tmp_path):
        compare_dir = tmp_path / "aiter_compare"
        compare_dir.mkdir()
        start_time = time.time() - 1
        ours = compare_dir / "tuned_a8w8_blockscale.11111.compare.txt"
        ours.write_text(_COMPARE_TABLE)
        found = ac._find_latest_compare_report_impl("a8w8_blockscale", start_time, compare_dir)
        assert found == ours


class TestCompareReportWiring:
    def test_stdout_empty_compare_file_gives_default_us_and_improved(self, tmp_path, monkeypatch):
        compare_dir = tmp_path / "aiter_compare"
        compare_dir.mkdir()
        start_time = time.time() - 1
        report = compare_dir / "tuned_a8w8_blockscale.22222.compare.txt"
        report.write_text(_COMPARE_TABLE)
        _prep(tmp_path, monkeypatch, compare_dir=compare_dir)
        # Force run_start_time to be before the report was written.
        monkeypatch.setattr("time.time", lambda: start_time + 0.5)
        result = _run(tmp_path)
        assert result.status == "ok"
        assert result.improved_shapes == 2
        assert result.unverified_shapes == 0
        assert all(not r.get("tuned_unverified") for r in result.shape_results)
        first = result.shape_results[0]
        assert first["default_us"] == 1037.74
        assert first["tuned_us"] == 269.71
        assert first["speedup"] > 1.0
        assert first["improved"] is True

    def test_report_persisted_to_work_dir(self, tmp_path, monkeypatch):
        compare_dir = tmp_path / "aiter_compare"
        compare_dir.mkdir()
        start_time = time.time() - 1
        report = compare_dir / "tuned_a8w8_blockscale.33333.compare.txt"
        report.write_text(_COMPARE_TABLE)
        _prep(tmp_path, monkeypatch, compare_dir=compare_dir)
        monkeypatch.setattr("time.time", lambda: start_time + 0.5)
        _run(tmp_path)
        persisted = tmp_path / "compare_a8w8_blockscale.txt"
        assert persisted.is_file()
        assert "1037.74" in persisted.read_text()

    def test_stale_report_not_used_falls_back_to_candidate(self, tmp_path, monkeypatch):
        compare_dir = tmp_path / "aiter_compare"
        compare_dir.mkdir()
        stale = compare_dir / "tuned_a8w8_blockscale.44444.compare.txt"
        stale.write_text(_COMPARE_TABLE)
        os.utime(stale, (1000, 1000))
        _prep(tmp_path, monkeypatch, compare_dir=compare_dir)
        cand = compare_dir / "tuned_a8w8_blockscale.55555.candidate.csv"
        cand.write_text(_HDR + "\ngfx950,256,8192,5120,5120,ck,8,0,269.71,knl,100,1000,0.0\n")
        monkeypatch.setattr(ac, "_find_latest_candidate", lambda name, t: cand)
        result = _run(tmp_path)
        assert result.unverified_shapes == 1
        assert result.shape_results[0]["tuned_unverified"] is True
        assert result.shape_results[0]["default_us"] is None
