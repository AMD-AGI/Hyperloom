# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the CK MoE (fmoe_ck) tuner input generation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time

import pytest

from kernelforge.gemm_tune.model_analyzer import ModelProfile
from kernelforge.gemm_tune.tuners import fmoe_ck as fm
from kernelforge.gemm_tune.tuners.base import TuneContext


def _moe_ctx(tmp_path, **overrides) -> TuneContext:
    profile = ModelProfile(
        model_path="/fake",
        hidden_size=4096,
        intermediate_size=14336,
        moe_intermediate_size=1536,
        num_attention_heads=32,
        num_key_value_heads=8,
        is_moe=True,
        num_experts=128,
        num_experts_per_tok=8,
    )
    base = dict(
        profile=profile,
        framework="vllm-aiter",
        precision="fp8",
        quant_type="blockscale",
        gpu_type="mi355x",
        tp=1,
        conc=64,
        tokens=[16, 64],
        mp=1,
        output_dir=tmp_path,
        iters=5,
        warmup=2,
        min_improvement_pct=1.0,
        timeout_s=60,
    )
    base.update(overrides)
    return TuneContext(**base)


def _column(csv_path, name: str) -> list[str]:
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    idx = lines[0].split(",").index(name)
    return [row.split(",")[idx] for row in lines[1:]]


def test_fmoe_validate_rejects_indivisible_tp(tmp_path, monkeypatch):
    """A width that does not shard evenly cannot yield the runtime's key."""
    monkeypatch.setattr(fm, "find_tuner_script", lambda _name: "/fake/gemm_moe_tune.py")
    err = fm.FmoeCKTuner(_moe_ctx(tmp_path, tp=5)).validate()
    assert err is not None
    assert "not divisible by tp 5" in err


def test_fmoe_validate_requires_a_runtime_observed_key(tmp_path, monkeypatch):
    """Without observed evidence, tuning an inferred key wastes hours to learn nothing."""
    monkeypatch.setattr(fm, "find_tuner_script", lambda _name: "/fake/gemm_moe_tune.py")
    err = fm.FmoeCKTuner(_moe_ctx(tmp_path)).validate()
    assert err is not None
    assert "no runtime-observed MoE miss" in err


def test_fmoe_does_not_run_for_a_hit_only_dispatch_key(tmp_path, monkeypatch):
    """A dispatch identifies the key, but only a miss makes it tuning demand."""
    demand = tmp_path / "demand.json"
    demand.write_text(
        json.dumps(
            {
                "demands": [],
                "dispatch": {
                    "moe": {
                        "keys": [
                            {
                                "miss_count": 0,
                                "tokens": [32],
                                "untuned_tokens": [],
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(fm, "find_tuner_script", lambda _name: "/fake/gemm_moe_tune.py")
    tuner = fm.FmoeCKTuner(_moe_ctx(tmp_path, demand_json=demand))
    monkeypatch.setattr(tuner, "run", lambda: pytest.fail("zero-miss key was tuned"))

    result = tuner.execute()

    assert result.error_class == "validation_error"
    assert "no runtime-observed MoE miss" in result.error


def test_fmoe_validate_passes_with_a_runtime_observed_key(tmp_path, monkeypatch):
    monkeypatch.setattr(fm, "find_tuner_script", lambda _name: "/fake/gemm_moe_tune.py")
    ctx = _moe_ctx(tmp_path, moe_untuned_csv=_write_runtime_csv(tmp_path))
    assert fm.FmoeCKTuner(ctx).validate() is None


def _write_runtime_csv(tmp_path, *, inter_dim="512", q_a="torch.float8_e4m3fn", q_w="torch.float4_e2m1fn_x2"):
    """A CSV shaped like one built from an observed aiter dispatch tuple."""
    path = tmp_path / "runtime_key.csv"
    rows = [fm._FMOE_CSV_HEADER]
    for token in (4, 512):
        rows.append(
            f"{token},4096,{inter_dim},256,6,ActivationType.Silu,torch.bfloat16,{q_a},{q_w},QuantType.per_1x32,1,0"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_caller_supplied_csv_is_tuned_verbatim(tmp_path):
    """The observed dispatch key is authoritative; nothing here may override it."""
    external = _write_runtime_csv(tmp_path)
    ctx = _moe_ctx(tmp_path, precision="bf16", quant_type="", moe_untuned_csv=external)

    resolved, source = fm.FmoeCKTuner(ctx)._resolve_untuned_csv()

    assert resolved == external
    assert source == "runtime_observed"
    assert _column(resolved, "q_dtype_a") == ["torch.float8_e4m3fn"] * 2
    assert _column(resolved, "q_dtype_w") == ["torch.float4_e2m1fn_x2"] * 2
    assert set(_column(resolved, "inter_dim")) == {"512"}


def test_no_runtime_key_is_refused_rather_than_derived_from_the_config(tmp_path):
    with pytest.raises(ValueError, match="no runtime-observed MoE miss"):
        fm.FmoeCKTuner(_moe_ctx(tmp_path))._resolve_untuned_csv()


def test_dense_untuned_csv_is_not_consumed_as_moe_shapes(tmp_path):
    """The dense field carries an M,N,K table and is already set in production."""
    dense = tmp_path / "a8w8_blockscale_untuned_gemm.csv"
    dense.write_text("M,N,K\n256,1536,4096\n", encoding="utf-8")
    ctx = _moe_ctx(tmp_path, untuned_csv=dense)

    with pytest.raises(ValueError, match="no runtime-observed MoE miss"):
        fm.FmoeCKTuner(ctx)._resolve_untuned_csv()


def test_missing_caller_csv_raises_rather_than_deriving(tmp_path):
    ctx = _moe_ctx(tmp_path, moe_untuned_csv=tmp_path / "absent.csv")
    with pytest.raises(FileNotFoundError):
        fm.FmoeCKTuner(ctx)._resolve_untuned_csv()


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("", "empty"),
        ("token,model_dim\n1,2\n", "missing required column"),
        (fm._FMOE_CSV_HEADER + "\n", "no shape rows"),
        (fm._FMOE_CSV_HEADER + "\n4,4096,512\n", "expected 12"),
    ],
)
def test_malformed_caller_csv_raises_rather_than_deriving(tmp_path, content, expected):
    """Silently deriving a different key is what makes a tuned table unreachable."""
    bad = tmp_path / "bad.csv"
    bad.write_text(content, encoding="utf-8")
    ctx = _moe_ctx(tmp_path, moe_untuned_csv=bad)

    with pytest.raises(ValueError, match=expected):
        fm.FmoeCKTuner(ctx)._resolve_untuned_csv()


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("temp_var", ["TMPDIR", "TEMP", "TMP", None])
@pytest.mark.parametrize("improved", [False, True])
def test_compare_candidate_uses_child_tempdir(tmp_path, monkeypatch, isolated, temp_var, improved):
    parent_tmp = tmp_path / "cached"
    parent_tmp.mkdir()
    child_tmp = tmp_path / "child"
    child_tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(parent_tmp))
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.delenv(name, raising=False)
    if temp_var:
        monkeypatch.setenv(temp_var, str(child_tmp))
    expected_tmp = parent_tmp
    external = _write_runtime_csv(tmp_path)
    tuner = fm.FmoeCKTuner(_moe_ctx(tmp_path, moe_untuned_csv=external))
    monkeypatch.setattr(fm, "find_tuner_script", lambda _: Path("/fake/tune.py"))
    monkeypatch.setattr(fm, "resolve_aiter_root", lambda: None)
    monkeypatch.setattr(fm._tr, "is_isolation_enabled", lambda: isolated)
    monkeypatch.setattr(fm._tr.FaultBlocklist, "save", lambda self: None)
    produced = []

    def run(cmd, **kwargs):
        assert "--compare" in cmd
        assert "--update_improved" not in cmd
        assert Path(kwargs["env_override"]["TMPDIR"]) == expected_tmp
        compare_dir = expected_tmp / "aiter_compare"
        compare_dir.mkdir(exist_ok=True)
        stem = Path(cmd[cmd.index("-o") + 1]).stem
        candidate = compare_dir / f"{stem}.37386.candidate.csv"
        lines = Path(cmd[cmd.index("-i") + 1]).read_text(encoding="utf-8").splitlines()
        candidate.write_text(
            lines[0] + ",kernelName1,kernelName2\n" + "\n".join(row + ",stage1,stage2" for row in lines[1:]) + "\n",
            encoding="utf-8",
        )
        os.utime(candidate, (time.time() + 1, time.time() + 1))
        unrelated = compare_dir / "other_tuned_fmoe.123.candidate.csv"
        unrelated.write_text("wrong table\n", encoding="utf-8")
        os.utime(unrelated, (time.time() + 2, time.time() + 2))
        produced.append(candidate)
        action = "UPDATE" if improved else "KEEP"
        return 0, f"(4, shape) | 10 | 8 | 20% | {action}", ""

    monkeypatch.setattr(fm, "run_subprocess", run)
    monkeypatch.setattr(fm._tr, "run_subprocess", run)
    result = tuner.run()
    assert result.status == ("ok" if improved else "no_improvement")
    artifact = Path(result.artifact_path)
    assert artifact == tuner.work_dir / "candidate_fmoe.csv"
    assert result.env_value == str(artifact)
    for candidate in produced:
        candidate.unlink()
    assert _column(artifact, "token") == ["4", "512"]
    assert _column(artifact, "kernelName1") == ["stage1", "stage1"]
    if temp_var:
        assert os.environ[temp_var] == str(child_tmp)
    else:
        assert "TMPDIR" not in os.environ


@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("content", [None, "", "token,model_dim\n4,4096\n"])
@pytest.mark.parametrize("improved", [False, True])
def test_compare_without_valid_candidate_never_exports_path(tmp_path, monkeypatch, isolated, content, improved):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    external = _write_runtime_csv(tmp_path)
    tuner = fm.FmoeCKTuner(_moe_ctx(tmp_path, moe_untuned_csv=external))
    monkeypatch.setattr(fm, "find_tuner_script", lambda _: Path("/fake/tune.py"))
    monkeypatch.setattr(fm, "resolve_aiter_root", lambda: None)
    monkeypatch.setattr(fm._tr, "is_isolation_enabled", lambda: isolated)
    monkeypatch.setattr(fm._tr.FaultBlocklist, "save", lambda self: None)

    def run(cmd, **kwargs):
        if content is not None:
            compare_dir = tmp_path / "aiter_compare"
            compare_dir.mkdir(exist_ok=True)
            stem = Path(cmd[cmd.index("-o") + 1]).stem
            candidate = compare_dir / f"{stem}.123.candidate.csv"
            candidate.write_text(content, encoding="utf-8")
            os.utime(candidate, (time.time() + 1, time.time() + 1))
        action = "UPDATE" if improved else "KEEP"
        return 0, f"(4, shape) | 10 | 8 | 20% | {action}", ""

    monkeypatch.setattr(fm, "run_subprocess", run)
    monkeypatch.setattr(fm._tr, "run_subprocess", run)
    result = tuner.run()
    assert result.status == ("failed" if improved else "no_improvement")
    assert not result.artifact_path
    assert not result.env_value
    assert not result.env_var
    if improved:
        assert result.error_class == "missing_artifact"


def test_candidate_collection_ignores_old_and_other_tuners(tmp_path):
    for name in ("tuned_fmoe.1.candidate.csv", "tuned_fmoe_other.2.candidate.csv", "other_tuned_fmoe.3.candidate.csv"):
        candidate = tmp_path / name
        candidate.write_text("old", encoding="utf-8")
        stamp = 1 if name == "tuned_fmoe.1.candidate.csv" else 20
        os.utime(candidate, (stamp, stamp))
    assert fm._tr._latest_candidate(tmp_path, "tuned_fmoe", 10) is None


def test_untuned_shapes_are_not_valid_tuned_artifacts(tmp_path):
    assert fm._validate_fmoe_csv(_write_runtime_csv(tmp_path), tuned=True) is not None


@pytest.mark.parametrize("cached", [False, True])
def test_compare_temp_env_matches_real_child_with_relative_tmpdir(tmp_path, monkeypatch, cached):
    import subprocess
    import sys

    selected = tmp_path / "selected"
    selected.mkdir()
    child_cwd = tmp_path / "child_cwd"
    child_cwd.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TMPDIR", "selected")
    monkeypatch.setattr(tempfile, "tempdir", "selected" if cached else None)
    env_override = fm._tr.compare_temp_env()
    actual = subprocess.check_output(
        [sys.executable, "-c", "import tempfile; print(tempfile.gettempdir())"],
        cwd=child_cwd,
        env={**os.environ, **env_override},
        text=True,
    ).strip()
    assert Path(actual) == selected
    assert env_override == {"TMPDIR": str(selected)}
    assert os.environ["TMPDIR"] == "selected"
