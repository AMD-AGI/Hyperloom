# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the forge shapes-JSON alignment hook in the GEMM tuning handler."""

from __future__ import annotations

import json

from hyperloom.orchestrator.kernel import request_handlers as krh
from hyperloom.orchestrator.kernel.gemm_shape_coverage import load_shapes_json


def _write_shapes(tmp_path, shapes):
    path = tmp_path / "forge_shapes.json"
    path.write_text(
        json.dumps([{"M": m, "N": n, "K": k} for m, n, k in shapes]),
        encoding="utf-8",
    )
    return str(path)


class TestAlignForgeShapesForAiter:
    def test_aligns_for_aiter_tuner_families(self, tmp_path):
        source = _write_shapes(tmp_path, [(1076, 5120, 17408)])
        out, report = krh._align_forge_shapes_for_aiter(
            source,
            forge_framework="vllm-aiter",
            workspace=tmp_path / "ws",
        )
        assert out != source
        assert out.endswith("forge_shapes.aiter_aligned.json")
        assert report["applied"] is True
        aligned = set(load_shapes_json(out))
        # The raw prefill M is gone; its fine-padded key plus a full power-of-two
        # ladder up to the observed maximum take its place.
        assert (1076, 5120, 17408) not in aligned
        assert (1088, 5120, 17408) in aligned
        assert {m for m, _, _ in aligned} == {16, 32, 64, 128, 256, 512, 1024, 1088, 2048}

    def test_sglang_also_uses_the_aiter_csv_lookup(self, tmp_path):
        source = _write_shapes(tmp_path, [(2087, 7168, 5120)])
        out, report = krh._align_forge_shapes_for_aiter(
            source,
            forge_framework="sglang",
            workspace=tmp_path / "ws",
        )
        assert report["applied"] is True
        assert (2112, 7168, 5120) in load_shapes_json(out)

    def test_non_aiter_framework_is_untouched(self, tmp_path):
        source = _write_shapes(tmp_path, [(1076, 5120, 17408)])
        out, report = krh._align_forge_shapes_for_aiter(
            source,
            forge_framework="vllm",
            workspace=tmp_path / "ws",
        )
        assert out == source
        assert report is None

    def test_opt_out_via_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HYPERLOOM_GEMM_ALIGN_SHAPES", "0")
        source = _write_shapes(tmp_path, [(1076, 5120, 17408)])
        out, report = krh._align_forge_shapes_for_aiter(
            source,
            forge_framework="vllm-aiter",
            workspace=tmp_path / "ws",
        )
        assert out == source
        assert report is None

    def test_shape_budget_is_configurable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HYPERLOOM_GEMM_ALIGN_MAX_SHAPES", "2")
        source = _write_shapes(
            tmp_path,
            [(1076, 5120, 17408), (4142, 5120, 17408), (7211, 5120, 17408)],
        )
        out, report = krh._align_forge_shapes_for_aiter(
            source,
            forge_framework="vllm-aiter",
            workspace=tmp_path / "ws",
        )
        assert report["applied"] is True
        # One (N, K) pair, so the budget floor keeps a single covering row.
        assert len(load_shapes_json(out)) <= 2

    def test_tuning_budget_caps_the_ladder(self, tmp_path):
        """A short tuner window must not produce more shapes than it can finish."""
        source = _write_shapes(tmp_path, [(4142, 5120, 17408), (4142, 7168, 5120)])
        wide, _ = krh._align_forge_shapes_for_aiter(
            source,
            forge_framework="vllm-aiter",
            workspace=tmp_path / "wide",
            budget_sec=18000,
        )
        narrow, report = krh._align_forge_shapes_for_aiter(
            source,
            forge_framework="vllm-aiter",
            workspace=tmp_path / "narrow",
            budget_sec=900,
        )
        assert len(load_shapes_json(narrow)) < len(load_shapes_json(wide))
        # Every projection keeps at least one reachable row.
        assert {(n, k) for _m, n, k in load_shapes_json(narrow)} == {(5120, 17408), (7168, 5120)}
        assert report["applied"] is True

    def test_more_tuning_workers_afford_more_shapes(self, tmp_path):
        source = _write_shapes(tmp_path, [(4142, 5120, 17408), (4142, 7168, 5120)])
        one, _ = krh._align_forge_shapes_for_aiter(
            source, forge_framework="vllm-aiter", workspace=tmp_path / "mp1", budget_sec=900, mp=1
        )
        eight, _ = krh._align_forge_shapes_for_aiter(
            source, forge_framework="vllm-aiter", workspace=tmp_path / "mp8", budget_sec=900, mp=8
        )
        assert len(load_shapes_json(eight)) > len(load_shapes_json(one))

    def test_missing_shapes_file_is_a_no_op(self, tmp_path):
        out, report = krh._align_forge_shapes_for_aiter(
            str(tmp_path / "absent.json"),
            forge_framework="vllm-aiter",
            workspace=tmp_path / "ws",
        )
        assert out == str(tmp_path / "absent.json")
        assert report is None

    def test_alignment_is_idempotent(self, tmp_path):
        source = _write_shapes(tmp_path, [(1076, 5120, 17408)])
        first, _ = krh._align_forge_shapes_for_aiter(
            source,
            forge_framework="vllm-aiter",
            workspace=tmp_path / "ws1",
        )
        second, report = krh._align_forge_shapes_for_aiter(
            first,
            forge_framework="vllm-aiter",
            workspace=tmp_path / "ws2",
        )
        assert set(load_shapes_json(second)) == set(load_shapes_json(first))
        assert report["applied"] is False
