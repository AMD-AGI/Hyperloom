# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the Triton MoE search space.

Two defects are pinned here. The tuner used to pass one fixed eight-entry list
in both modes, so ``--thorough`` changed nothing at all. And that list stopped
at ``BLOCK_SIZE_K=128``, while the measured best config at M=32/256/1024 used
``BK=256`` every time (1.0975x-1.1235x better) -- an axis a fixed list can never
be wrong about, because it never contains the value.
"""

from __future__ import annotations

from kernelforge.gemm_tune.tuners import vllm_moe_triton as mt


def test_cap_below_the_seed_count_still_keeps_every_seed(monkeypatch):
    # Seeds are ordered first so a capped run keeps the configs already measured
    # to work. Slicing inside that prefix would make --thorough search LESS than
    # the default does, which is the opposite of what the flag promises.
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
        # The measured winners all sat here; a capped thorough run must still
        # reach it rather than spend the whole budget on BK=64/128.
        assert any(c["BLOCK_SIZE_K"] == 256 for c in mt.build_search_space(True))

    def test_the_default_search_reaches_bk256_too(self):
        # Widening --thorough was not enough on its own: Hyperloom only asks for
        # thorough at session_max_min >= 1440 and mp >= 4, so almost every
        # session runs the default list. With the measured winners absent from
        # it, the axis that decided those measurements stayed unreachable in
        # practice however wide the thorough grid became.
        assert any(c["BLOCK_SIZE_K"] == 256 for c in mt.build_search_space(False))

    def test_the_grid_still_covers_more_than_the_seeded_points(self):
        # The seeds pin three measured winners; the grid is what finds the next
        # one, so promoting them must not turn --thorough back into the seeds.
        seeded = {tuple(sorted(c.items())) for c in mt._SEED_CONFIGS}
        grid_only = [c for c in mt.build_search_space(True) if tuple(sorted(c.items())) not in seeded]
        assert len(grid_only) > len(mt._SEED_CONFIGS)
        assert any(c["BLOCK_SIZE_K"] == 256 for c in grid_only)


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
