# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""force_candidate wiring: run_aiter_dense_tuner must set TuneResult.candidate
from the deployed CSV's split-K content, so a split-K artifact is still promoted
to e2e even when the microbench reports no improvement. Guards the exact
regression the split-K fix exists to prevent (a refactor dropping the wiring).
"""

from __future__ import annotations

import types

import kernelforge.gemm_tune.tuners._aiter_dense_common as ac
from kernelforge.gemm_tune.tuners.base import TuneContext

_HDR = "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio"


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


def _prep_and_mock(tmp_path, monkeypatch, splitk):
    # The serve-safe splitK cap trials the real a8w8_blockscale CK kernel when a
    # GPU is present, which JIT-builds aiter modules and turns this pure wiring
    # test into a multi-minute (observed: hanging) build. Use the tuner's own
    # escape hatch so _shape_max stays on the static cap.
    monkeypatch.setenv("FORGE_SPLITK_TRIAL", "0")
    (tmp_path / "in.csv").write_text("M,N,K\n64,5120,5120\n")
    row = f"gfx950,256,64,5120,5120,ck,8,{splitk},16.0,knl,100,1000,0.0\n"
    # the tuned artifact the tuner "produced" + its full-candidate profile
    (tmp_path / "tuned_a8w8.csv").write_text(_HDR + "\n" + row)
    (tmp_path / "profile_a8w8.csv").write_text(_HDR + "\n" + row)
    monkeypatch.setattr(ac, "find_tuner_script", lambda k: tmp_path / "script.py")
    monkeypatch.setattr(ac, "_resolve_input_csv", lambda ctx, wd, needs_q_dtype_w=False: tmp_path / "in.csv")
    monkeypatch.setattr(ac, "resolve_aiter_root", lambda: str(tmp_path))
    monkeypatch.setattr(ac._tr, "is_isolation_enabled", lambda: False)
    monkeypatch.setattr(ac._tr, "with_task_timeout", lambda cmd: cmd)
    monkeypatch.setattr(ac, "run_subprocess", lambda cmd, **k: (0, "", ""))
    monkeypatch.setattr(ac, "_find_latest_candidate", lambda name, t: None)


def _run(tmp_path):
    return ac.run_aiter_dense_tuner(
        tuner_name="a8w8",
        script_key="a8w8_blockscale",
        env_var="AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
        ctx=_ctx(tmp_path),
        work_dir=tmp_path,
        extra_args=["--libtype", "all", "--splitK"],
    )


def test_splitk_csv_sets_candidate_true(tmp_path, monkeypatch):
    _prep_and_mock(tmp_path, monkeypatch, splitk=2)
    assert _run(tmp_path).candidate is True


def test_splitk0_csv_leaves_candidate_false(tmp_path, monkeypatch):
    _prep_and_mock(tmp_path, monkeypatch, splitk=0)
    assert _run(tmp_path).candidate is False


def _run_no_splitk(tmp_path):
    # Same driver, but no --splitK, so a forced candidate can only come from the
    # new-shape path -- isolating the fix from the split-K force_candidate.
    return ac.run_aiter_dense_tuner(
        tuner_name="a8w8",
        script_key="a8w8_blockscale",
        env_var="AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
        ctx=_ctx(tmp_path),
        work_dir=tmp_path,
        extra_args=["--libtype", "all"],
    )


def test_all_new_shapes_force_candidate(tmp_path, monkeypatch):
    # aiter reports every shape as NEW (no prior baseline). status=ok with
    # unverified_shapes>0 (bf16-aligned); candidate=True sends configs to E2E.
    _prep_and_mock(tmp_path, monkeypatch, splitk=0)
    new_table = "--- Would update (1 shapes) ---\n(64, 5120, 5120) | N/A | 16.0 | N/A | NEW\n"
    monkeypatch.setattr(ac, "run_subprocess", lambda cmd, **k: (0, new_table, ""))
    result = _run_no_splitk(tmp_path)
    assert result.candidate is True
    assert result.status == "ok"
    assert result.improved_shapes == 0 and result.unverified_shapes == 1
    assert any(r.get("is_new") for r in result.shape_results)


def test_candidate_csv_fallback_forces_candidate(tmp_path, monkeypatch):
    # aiter printed only a "Successfully tuned shapes" summary, so there is no
    # per-shape Pre/Post table and every row recovered from the candidate CSV is
    # tuned_unverified. Those rows can never show a micro speedup, so leaving
    # them out of the force path discarded a real tuned artifact as
    # no_improvement -- the reporting artefact behind fp8 bpreshuffle's "0/44".
    _prep_and_mock(tmp_path, monkeypatch, splitk=0)
    (tmp_path / "candidate_a8w8.csv").write_text(_HDR + "\ngfx950,256,64,5120,5120,ck,8,0,16.0,knl,100,1000,0.0\n")
    monkeypatch.setattr(ac, "run_subprocess", lambda cmd, **k: (0, "Successfully tuned 1 shapes\n", ""))
    result = _run_no_splitk(tmp_path)
    assert result.candidate is True
    assert result.status == "ok"
    assert result.improved_shapes == 0 and result.unverified_shapes == 1
    assert all(r.get("tuned_unverified") for r in result.shape_results)


def test_all_update_shapes_do_not_force_candidate(tmp_path, monkeypatch):
    # A normal comparison with real speedups is promoted through has_improvement,
    # NOT the new-shape force path -- guards against over-forcing.
    _prep_and_mock(tmp_path, monkeypatch, splitk=0)
    upd_table = "--- Would update (1 shapes) ---\n(64, 5120, 5120) | 32.0 | 16.0 | 50.0% | UPDATE\n"
    monkeypatch.setattr(ac, "run_subprocess", lambda cmd, **k: (0, upd_table, ""))
    result = _run_no_splitk(tmp_path)
    assert result.candidate is False  # not forced; promoted via micro improvement
    assert result.status == "ok"
