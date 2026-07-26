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


def test_persist_copies_into_aiter_config_and_snapshots(tmp_path, monkeypatch):
    aiter_pkg = _fake_aiter(monkeypatch, tmp_path)
    ws = tmp_path / "ws"
    ws.mkdir()
    src = ws / "tuned.csv"
    src.write_text("gfx,cu_num,M,N,K,splitK\ngfx950,256,64,5120,5120,2\n", encoding="utf-8")

    extra = {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": str(src)}
    out, snap = rh._persist_forge_gemm_csv_durably(
        extra, model_path="/models/Qwen3-14B-FP8", snapshot_root=ws
    )

    dst = aiter_pkg / "configs" / "model_configs" / "a8w8_blockscale_tuned_gemm_qwen3-14b-fp8.csv"
    assert dst.is_file()  # copied where aiter reads it
    assert out["AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"] == str(dst)  # env repointed to durable path
    assert snap and Path(snap).is_dir()  # durable snapshot dir
    assert (Path(snap) / "manifest.json").is_file()
    assert (Path(snap) / "files" / "configs" / "model_configs" / dst.name).is_file()


def test_persist_missing_source_csv_is_noop(tmp_path, monkeypatch):
    _fake_aiter(monkeypatch, tmp_path)
    extra = {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": str(tmp_path / "nope.csv")}
    out, snap = rh._persist_forge_gemm_csv_durably(extra, model_path="/m/x", snapshot_root=tmp_path)
    assert out == extra and snap == ""


def test_persist_no_env_is_noop(tmp_path):
    out, snap = rh._persist_forge_gemm_csv_durably({}, model_path="/m/x", snapshot_root=tmp_path)
    assert out == {} and snap == ""


def test_persist_aiter_not_importable_falls_back(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    src = ws / "tuned.csv"
    src.write_text("gfx,M,N,K,splitK\ngfx950,64,5120,5120,2\n", encoding="utf-8")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    extra = {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": str(src)}
    out, snap = rh._persist_forge_gemm_csv_durably(extra, model_path="/m/x", snapshot_root=ws)
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
        extra, model_path="/models/Qwen3-14B-FP8", snapshot_root=ws
    )

    dst = aiter_pkg / "configs" / "model_configs" / "a8w8_blockscale_tuned_gemm_qwen3-14b-fp8.csv"
    assert dst.is_file()  # copy committed despite the snapshot failure
    assert out["AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"] == str(dst)  # repoint SURVIVES
    assert snap == ""  # snapshot dir empty (it failed), but durability is kept
