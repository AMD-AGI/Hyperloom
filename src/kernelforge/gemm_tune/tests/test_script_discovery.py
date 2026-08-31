# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for aiter tuner script discovery.

The hardcoded-path design is what broke: aiter moved the bf16 dense tuner from
``gradlib/`` to ``csrc/gemm_a16w16/`` and the constant kept pointing at the old
location. These tests pin that a move is survivable, that the preference order
is honoured, and that "aiter ships no such script" stays distinguishable from
"we have not wired it up" -- only the former justifies dropping a tier.
"""

from __future__ import annotations

import pytest

from kernelforge.gemm_tune import script_discovery as sd


@pytest.fixture(autouse=True)
def _clear_cache():
    sd._INVENTORY_CACHE.clear()
    yield
    sd._INVENTORY_CACHE.clear()


def _make(csrc, rel):
    path = csrc / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# stub tuner", encoding="utf-8")
    return path


class TestHintedResolution:
    def test_finds_script_at_hinted_path(self, tmp_path):
        want = _make(tmp_path, "ck_gemm_a8w8/gemm_a8w8_tune.py")
        assert sd.discover_tuner_script("a8w8", tmp_path) == want

    def test_prefers_direct_tuner_over_shim(self, tmp_path):
        direct = _make(tmp_path, "gemm_a16w16/gemm_a16w16_tune.py")
        _make(tmp_path, "gemm_a16w16/gemm_tuner.py")
        # The shim rewrites the tuner's exit code, so the direct script wins.
        assert sd.discover_tuner_script("sglang_dense_bf16", tmp_path) == direct

    def test_falls_back_to_shim_when_direct_missing(self, tmp_path):
        shim = _make(tmp_path, "gemm_a16w16/gemm_tuner.py")
        assert sd.discover_tuner_script("sglang_dense_bf16", tmp_path) == shim


class TestSearchSurvivesRelocation:
    def test_finds_script_moved_to_a_new_directory(self, tmp_path):
        # Exactly the failure mode that started this work, in the other
        # direction: the hinted path is empty, the file lives elsewhere.
        moved = _make(tmp_path, "some_new_layout/v2/gemm_a8w8_tune.py")
        assert sd.discover_tuner_script("a8w8", tmp_path) == moved

    def test_search_does_not_confuse_batched_variant(self, tmp_path):
        _make(tmp_path, "elsewhere/batched_gemm_a8w8_tune.py")
        # batched_* is a different tuner; matching it here would silently tune
        # the wrong operator.
        assert sd.discover_tuner_script("a8w8", tmp_path) is None

    def test_search_does_not_confuse_blockscale_variant(self, tmp_path):
        _make(tmp_path, "elsewhere/gemm_a8w8_blockscale_tune.py")
        assert sd.discover_tuner_script("a8w8", tmp_path) is None

    def test_missing_script_returns_none(self, tmp_path):
        assert sd.discover_tuner_script("a8w8", tmp_path) is None

    def test_unknown_tuner_returns_none(self, tmp_path):
        _make(tmp_path, "ck_gemm_a8w8/gemm_a8w8_tune.py")
        assert sd.discover_tuner_script("not_a_tuner", tmp_path) is None

    def test_search_result_is_deterministic(self, tmp_path):
        # Two candidates, filesystem order unspecified -> sorted pick.
        a = _make(tmp_path, "aaa/gemm_a16w16_tune.py")
        _make(tmp_path, "zzz/gemm_a16w16_tune.py")
        assert sd.discover_tuner_script("sglang_dense_bf16", tmp_path) == a


class TestInventory:
    def test_lists_every_tune_script(self, tmp_path):
        _make(tmp_path, "ck_gemm_a8w8/gemm_a8w8_tune.py")
        _make(tmp_path, "batched/batched_gemm_bf16_tune.py")
        _make(tmp_path, "opus/opus_gemm_tune.py")
        _make(tmp_path, "ck_gemm_a8w8/helper.py")  # not a tuner
        assert set(sd.inventory(tmp_path)) == {
            "gemm_a8w8_tune",
            "batched_gemm_bf16_tune",
            "opus_gemm_tune",
        }

    def test_missing_csrc_is_empty_not_an_error(self, tmp_path):
        assert sd.inventory(tmp_path / "nope") == {}

    def test_unwired_scripts_excludes_the_ones_we_drive(self, tmp_path):
        _make(tmp_path, "ck_gemm_a8w8/gemm_a8w8_tune.py")  # wired
        _make(tmp_path, "batched/batched_gemm_a8w8_tune.py")  # Tier-1 stock
        _make(tmp_path, "opus/opus_gemm_tune.py")  # Tier-1 stock
        unwired = sd.unwired_scripts(tmp_path)
        assert set(unwired) == {"batched_gemm_a8w8_tune", "opus_gemm_tune"}
        assert "gemm_a8w8_tune" not in unwired
