# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A tuner whose candidates were never timed (us 0, tflops inf in every row, as on the vLLM image) fails, instead of
reporting a table and a speedup."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

import kernelforge.workflows.gemm_tune.tuners._aiter_dense_common as ac
from kernelforge.tests.workflows.gemm_tune._sglang_dense_bf16_run import _csv, _prep, _run
from kernelforge.workflows.gemm_tune.tuners.base import TuneContext

_BF16_ROW = "gfx950,256,{m},4096,4096,False,torch.bfloat16,torch.bfloat16,False,False,{lib},1,0,{us},knl,0.0,{tflops},0"
_DENSE_HDR = "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio"
_IMPROVED = """\
--- Would update (1 shapes) ---
Shape                                    |    Pre(us) |   Post(us) |   Improve |             Action
(8192, 5120, 5120)                       |    1037.74 |     269.71 |    74.01% |             UPDATE
"""


def _timings():
    from kernelforge.workflows.gemm_tune.tuners._timings import unmeasured_timings

    return unmeasured_timings


class TestTheCheck:
    @pytest.mark.parametrize("tflops, verdict", [("inf", "never timed"), (700, "")], ids=["all-untimed", "one-timed"])
    def test_unmeasured_unless_one_row_is_timed(self, tmp_path, tflops, verdict):
        path = tmp_path / "p.csv"
        rows = [_BF16_ROW.format(m=1, lib="torch", us=0, tflops="inf")]
        path.write_text(_csv([*rows, _BF16_ROW.format(m=1, lib="hipblaslt", us=5.0, tflops=tflops)]))
        result = _timings()(path)
        assert verdict in result if verdict else result == ""

    def test_missing_or_empty_files_say_nothing(self, tmp_path):
        (tmp_path / "empty.csv").write_text(_csv([]))
        assert _timings()(tmp_path / "nope.csv", tmp_path / "empty.csv") == ""


def test_bf16_run_with_untimed_candidates_fails(tmp_path, monkeypatch):
    rows = [_BF16_ROW.format(m=m, lib=lib, us=0, tflops="inf") for m in (1, 512) for lib in ("torch", "hipblaslt")]
    _prep(tmp_path, monkeypatch, tuned_rows=rows[1::2], profile_rows=rows)
    result = _run(tmp_path)
    assert result.status == "failed"
    assert result.error_class == "unmeasured_timings"


def test_dense_run_with_untimed_candidates_fails_despite_its_compare_table(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_SPLITK_TRIAL", "0")
    (tmp_path / "in.csv").write_text("M,N,K\n8192,5120,5120\n")
    row = "gfx950,256,8192,5120,5120,ck,8,0,0,knl,inf,0,0.0\n"
    (tmp_path / "profile_a8w8_blockscale.csv").write_text(_DENSE_HDR + "\n" + row)
    candidate = tmp_path / "found.candidate.csv"
    candidate.write_text(_DENSE_HDR + "\n" + row)
    monkeypatch.setattr(ac, "find_tuner_script", lambda k: tmp_path / "script.py")
    monkeypatch.setattr(ac, "_resolve_input_csv", lambda ctx, wd, needs_q_dtype_w=False: tmp_path / "in.csv")
    monkeypatch.setattr(ac, "resolve_aiter_root", lambda: str(tmp_path))
    monkeypatch.setattr(ac._tr, "is_isolation_enabled", lambda: False)
    monkeypatch.setattr(ac._tr, "with_task_timeout", lambda cmd: cmd)
    monkeypatch.setattr(ac, "run_subprocess", lambda cmd, **k: (0, _IMPROVED, ""))
    monkeypatch.setattr(ac, "_find_latest_candidate", lambda name, t: candidate)
    ctx = TuneContext(
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
    spec = ac.DenseTunerSpec("a8w8_blockscale", "a8w8_blockscale", "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE")

    result = ac.run_aiter_dense_tuner(spec, ctx, tmp_path)

    assert result.status == "failed"
    assert result.error_class == "unmeasured_timings"
    assert not Path(result.artifact_path or tmp_path / "none").is_file()
