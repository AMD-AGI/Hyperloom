# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the Triton MoE search space and the precision its written config claims to have tuned."""

from __future__ import annotations

import json
from pathlib import Path

from kernelforge.gemm_tune.model_analyzer import ModelProfile
from kernelforge.gemm_tune.tuners import vllm_moe_triton as mt
from kernelforge.gemm_tune.tuners.base import TuneContext


def test_cap_below_the_seed_count_still_keeps_every_seed(monkeypatch):
    # Seeds are ordered first so a capped run keeps the configs already measured to work.
    monkeypatch.setenv(mt._THOROUGH_CAP_ENV, "2")
    space = mt.build_search_space(True)
    assert space == [dict(c) for c in mt._SEED_CONFIGS]


def test_cap_above_the_seed_count_is_honoured(monkeypatch):
    monkeypatch.setenv(mt._THOROUGH_CAP_ENV, "20")
    space = mt.build_search_space(True)
    assert len(space) == 20
    assert space[: len(mt._SEED_CONFIGS)] == [dict(c) for c in mt._SEED_CONFIGS]


def _key(cfg):
    return tuple(sorted(cfg.items()))


class TestFastSpace:
    def test_fast_is_the_trusted_seed_set(self):
        assert mt.build_search_space(False) == mt._SEED_CONFIGS

    def test_fast_is_a_copy_not_the_module_list(self):
        space = mt.build_search_space(False)
        space[0]["BLOCK_SIZE_M"] = 999
        assert mt._SEED_CONFIGS[0]["BLOCK_SIZE_M"] != 999


class TestThoroughActuallyWidens:
    def test_thorough_is_larger_than_fast(self):
        # The original bug: --thorough was inert.
        assert len(mt.build_search_space(True)) > len(mt.build_search_space(False))

    def test_thorough_keeps_every_seed_first(self):
        space = mt.build_search_space(True)
        assert space[: len(mt._SEED_CONFIGS)] == mt._SEED_CONFIGS

    def test_thorough_has_no_duplicates(self):
        space = mt.build_search_space(True)
        assert len({_key(c) for c in space}) == len(space)

    def test_every_config_has_all_axes_and_split_k(self):
        expected = set(mt._AXES) | {"SPLIT_K"}
        assert all(set(c) == expected for c in mt.build_search_space(True))


class TestBlockSizeK256:
    def test_bk256_exists_in_the_grid(self):
        assert any(c["BLOCK_SIZE_K"] == 256 for c in mt._grid_configs())

    def test_thorough_actually_searches_bk256(self):
        # The measured winners all sat here; a capped thorough run must still reach it rather than spend the whole
        # budget on BK=64/128.
        assert any(c["BLOCK_SIZE_K"] == 256 for c in mt.build_search_space(True))

    def test_the_default_search_reaches_bk256_too(self):
        # Widening --thorough was not enough on its own: Hyperloom only asks for thorough at session_max_min >= 1440
        # and mp >= 4, so almost every session runs the default list.
        assert any(c["BLOCK_SIZE_K"] == 256 for c in mt.build_search_space(False))

    def test_the_grid_still_covers_more_than_the_seeded_points(self):
        # The seeds pin three measured winners; the grid is what finds the next one, so promoting them must not turn
        # --thorough back into the seeds.
        seeded = {tuple(sorted(c.items())) for c in mt._SEED_CONFIGS}
        grid_only = [c for c in mt.build_search_space(True) if tuple(sorted(c.items())) not in seeded]
        assert len(grid_only) > len(mt._SEED_CONFIGS)
        assert any(c["BLOCK_SIZE_K"] == 256 for c in grid_only)


def _moe_ctx(tmp_path) -> TuneContext:
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
    return TuneContext(
        profile=profile,
        framework="vllm",
        precision="bf16",
        quant_type="none",
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


class TestTheConfigNameStatesTheMeasuredDtype:
    """vLLM loads a tuned config by the dtype in its filename, so the name is a claim about the measurement.

    Which precisions may be swept at all is the router's decision (see
    ``test_router.TestOnlyTheMeasuredPrecisionIsTuned``); the tuner only has to name what it measured.
    """

    def test_the_written_config_is_named_for_the_dtype_that_was_benchmarked(self, tmp_path, monkeypatch):
        tuner = mt.VllmMoeTritonTuner(_moe_ctx(tmp_path))

        def fake_sweep(cmd, *, timeout_s, log_file):
            (tuner.work_dir / "sweep_results.json").write_text(
                json.dumps({"16": {"BLOCK_SIZE_M": 64}}), encoding="utf-8"
            )
            return 0, json.dumps({"status": "ok", "shape_details": [], "best_speedup": 1.2}), ""

        monkeypatch.setattr(mt, "run_subprocess", fake_sweep)

        result = tuner.run()

        written = [p.name for p in Path(result.artifact_path).iterdir()]
        assert written == ["E=128,N=1536,device_name=AMD_Instinct_MI355X,dtype=bfloat16.json"]


class TestCap:
    def test_default_cap_matches_the_measured_sample(self, monkeypatch):
        monkeypatch.delenv(mt._THOROUGH_CAP_ENV, raising=False)
        assert len(mt.build_search_space(True)) == mt._DEFAULT_THOROUGH_CAP

    def test_env_override_widens_the_budget(self, monkeypatch):
        monkeypatch.setenv(mt._THOROUGH_CAP_ENV, "12")
        assert len(mt.build_search_space(True)) == 12

    def test_garbage_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(mt._THOROUGH_CAP_ENV, "not-a-number")
        assert len(mt.build_search_space(True)) == mt._DEFAULT_THOROUGH_CAP

    def test_non_positive_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(mt._THOROUGH_CAP_ENV, "0")
        assert len(mt.build_search_space(True)) == mt._DEFAULT_THOROUGH_CAP
