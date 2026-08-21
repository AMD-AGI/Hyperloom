# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""forge GEMM tuned-CSV durability: KEEP must persist the CSV into the serving
aiter config dir + snapshot it (recipe-portable), not reference the ephemeral
tuner workspace path.
"""
from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import hyperloom.orchestrator.kernel.request_handlers as rh


def _fake_aiter(monkeypatch, tmp_path: Path) -> Path:
    aiter_pkg = tmp_path / "aiter"
    aiter_pkg.mkdir()
    (aiter_pkg / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: types.SimpleNamespace(origin=str(aiter_pkg / "__init__.py"))
        if name == "aiter"
        else None,
    )
    return aiter_pkg


def _durable(aiter_pkg: Path, name: str) -> Path:
    """Where a persisted CSV lands: below model_configs/, not inside it.

    aiter auto-merges everything its non-recursive ``model_configs/*.csv`` glob
    finds when the env var is unset, so the copy has to sit one level down.
    """
    return aiter_pkg / "configs" / "model_configs" / "hyperloom" / name


def test_persist_copies_into_aiter_config_and_snapshots(tmp_path, monkeypatch):
    aiter_pkg = _fake_aiter(monkeypatch, tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    src = ws / "tuned.csv"
    src.write_text("gfx,cu_num,M,N,K,splitK\ngfx950,256,64,5120,5120,2\n", encoding="utf-8")

    extra = {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": str(src)}
    out, snap = rh._persist_forge_gemm_csv_durably(
        extra, model_path="/models/Qwen3-14B-FP8", session_dir=ws
    )

    dst = _durable(aiter_pkg, "a8w8_blockscale_tuned_gemm_qwen3-14b-fp8.csv")
    assert dst.is_file()  # copied where the env var can reach it
    assert out["AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"] == str(dst)  # env repointed to durable path
    assert snap and Path(snap).is_dir()  # durable snapshot dir
    assert (Path(snap) / "manifest.json").is_file()
    assert (
        Path(snap) / "files" / "configs" / "model_configs" / "hyperloom" / dst.name
    ).is_file()
    # snapshot must live under the DURABLE optimization_stack/src (survives run
    # cleanup), NOT the ephemeral runs/gemm_tuning workspace (#2 recipe-portable).
    assert "optimization_stack" in Path(snap).parts and "src" in Path(snap).parts
    assert "runs" not in Path(snap).parts


def test_persist_keeps_the_copy_out_of_aiters_auto_merge_scan(tmp_path, monkeypatch):
    """The copy must be invisible to aiter's env-less table scan.

    ``jit/core.py::get_config_file`` globs ``model_configs/*{table}*.csv`` and
    merges everything it finds whenever the env var is unset -- which is the
    common case for a plain server start. A candidate persisted before E2E has
    ruled on it would reach every later server that way, so a REVERT would read
    as reverted while the table stayed silently in effect. Observed for real: a
    V4-Flash run merged dsv3's table, so the scan does not even discriminate by
    model.

    One level down is enough: the glob is not recursive, and the env var still
    points at the file.
    """
    aiter_pkg = _fake_aiter(monkeypatch, tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    src = ws / "tuned.csv"
    src.write_text("gfx,cu_num,M,N,K,splitK\ngfx950,256,64,5120,5120,2\n", encoding="utf-8")

    out, _snap = rh._persist_forge_gemm_csv_durably(
        {"AITER_CONFIG_GEMM_BF16": str(src)},
        model_path="/models/Qwen3-14B-FP8",
        session_dir=ws,
    )

    dst = Path(out["AITER_CONFIG_GEMM_BF16"])
    assert dst.is_file(), "the copy still has to exist for the env var to reach"
    model_configs = aiter_pkg / "configs" / "model_configs"
    assert list(model_configs.glob("*bf16_tuned_gemm*.csv")) == []
    assert dst.parent != model_configs


def test_persist_missing_source_csv_is_noop(tmp_path, monkeypatch):
    _fake_aiter(monkeypatch, tmp_path)
    extra = {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": str(tmp_path / "nope.csv")}
    out, snap = rh._persist_forge_gemm_csv_durably(extra, model_path="/m/x", session_dir=tmp_path)
    assert out == extra and snap == ""


def test_persist_no_env_is_noop(tmp_path):
    out, snap = rh._persist_forge_gemm_csv_durably({}, model_path="/m/x", session_dir=tmp_path)
    assert out == {} and snap == ""


def test_persist_aiter_not_importable_falls_back(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    src = ws / "tuned.csv"
    src.write_text("gfx,M,N,K,splitK\ngfx950,64,5120,5120,2\n", encoding="utf-8")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    extra = {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": str(src)}
    out, snap = rh._persist_forge_gemm_csv_durably(extra, model_path="/m/x", session_dir=ws)
    assert out == extra and snap == ""  # unchanged; never breaks KEEP


def test_persist_snapshot_failure_keeps_copy_and_repoint(tmp_path, monkeypatch):
    # A snapshot failure must NOT discard the durable copy + env repoint that
    # were already committed (that is what makes the KEEP survive replay).
    aiter_pkg = _fake_aiter(monkeypatch, tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    src = ws / "tuned.csv"
    src.write_text("gfx,M,N,K,splitK\ngfx950,64,5120,5120,2\n", encoding="utf-8")

    import hyperloom.orchestrator.source_snapshot as ss

    def _boom(**kwargs):
        raise RuntimeError("snapshot dest not writable")

    monkeypatch.setattr(ss, "snapshot_source_layer", _boom)

    extra = {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": str(src)}
    out, snap = rh._persist_forge_gemm_csv_durably(
        extra, model_path="/models/Qwen3-14B-FP8", session_dir=ws
    )

    dst = _durable(aiter_pkg, "a8w8_blockscale_tuned_gemm_qwen3-14b-fp8.csv")
    assert dst.is_file()  # copy committed despite the snapshot failure
    assert out["AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"] == str(dst)  # repoint SURVIVES
    assert snap == ""  # snapshot dir empty (it failed), but durability is kept


def test_persist_fmoe_csv_uses_tuned_fmoe_stem(tmp_path, monkeypatch):
    aiter_pkg = _fake_aiter(monkeypatch, tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    src = ws / "tuned_fmoe.csv"
    src.write_text("cu_num,token,model_dim,inter_dim,quantType\n304,16,4096,512,14\n", encoding="utf-8")

    extra = {"AITER_CONFIG_FMOE": str(src)}
    out, snap = rh._persist_forge_gemm_csv_durably(
        extra, model_path="/models/DeepSeek-V4-Flash", session_dir=ws
    )

    dst = _durable(aiter_pkg, "tuned_fmoe_deepseek-v4-flash.csv")
    assert dst.is_file()
    assert out["AITER_CONFIG_FMOE"] == str(dst)
    assert snap and Path(snap).is_dir()


def test_persist_copies_dense_and_fmoe_together(tmp_path, monkeypatch):
    aiter_pkg = _fake_aiter(monkeypatch, tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    dense = ws / "dense.csv"
    dense.write_text("gfx,M,N,K,splitK\ngfx950,64,5120,5120,2\n", encoding="utf-8")
    fmoe = ws / "fmoe.csv"
    fmoe.write_text("cu_num,token\n304,16\n", encoding="utf-8")

    extra = {
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": str(dense),
        "AITER_CONFIG_FMOE": str(fmoe),
    }
    out, snap = rh._persist_forge_gemm_csv_durably(
        extra, model_path="/models/Qwen3-14B-FP8", session_dir=ws
    )

    assert out["AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"].endswith(
        "a8w8_blockscale_tuned_gemm_qwen3-14b-fp8.csv"
    )
    assert out["AITER_CONFIG_FMOE"].endswith("tuned_fmoe_qwen3-14b-fp8.csv")
    assert snap and (Path(snap) / "manifest.json").is_file()
