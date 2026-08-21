# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""forge GEMM tuned-CSV durability: KEEP must persist the CSV into the session
directory and snapshot it (recipe-portable), not reference the ephemeral
tuner workspace path and not write into the shared installed aiter package.
"""
from __future__ import annotations

from pathlib import Path

import hyperloom.orchestrator.kernel.request_handlers as rh


def test_persist_copies_into_session_and_snapshots(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    src = ws / "tuned.csv"
    src.write_text("gfx,cu_num,M,N,K,splitK\ngfx950,256,64,5120,5120,2\n", encoding="utf-8")

    extra = {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": str(src)}
    out, snap = rh._persist_forge_gemm_csv_durably(
        extra, model_path="/models/Qwen3-14B-FP8", session_dir=ws
    )

    slug = "qwen3-14b-fp8"
    staged_root = ws / "optimization_stack" / "src" / f"forge_gemm_{slug}" / "staged"
    dst = staged_root / "configs" / "model_configs" / f"a8w8_blockscale_tuned_gemm_{slug}.csv"

    assert dst.is_file()
    assert out["AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"] == str(dst)
    assert snap and Path(snap).is_dir()
    assert (Path(snap) / "manifest.json").is_file()
    assert (Path(snap) / "files" / "configs" / "model_configs" / dst.name).is_file()
    assert "optimization_stack" in Path(snap).parts and "src" in Path(snap).parts
    assert "runs" not in Path(snap).parts


def test_persist_missing_source_csv_is_noop(tmp_path):
    extra = {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": str(tmp_path / "nope.csv")}
    out, snap = rh._persist_forge_gemm_csv_durably(extra, model_path="/m/x", session_dir=tmp_path)
    assert out == extra and snap == ""


def test_persist_no_env_is_noop(tmp_path):
    out, snap = rh._persist_forge_gemm_csv_durably({}, model_path="/m/x", session_dir=tmp_path)
    assert out == {} and snap == ""


def test_persist_snapshot_failure_keeps_copy_and_repoint(tmp_path, monkeypatch):
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

    slug = "qwen3-14b-fp8"
    staged_root = ws / "optimization_stack" / "src" / f"forge_gemm_{slug}" / "staged"
    dst = staged_root / "configs" / "model_configs" / f"a8w8_blockscale_tuned_gemm_{slug}.csv"

    assert dst.is_file()
    assert out["AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"] == str(dst)
    assert snap == ""
