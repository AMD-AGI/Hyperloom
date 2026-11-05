# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernels a session recompiled per call shape, told apart from kernels that are merely tile-tuned."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.orchestrator.measurement import jit_variants as jv


def _variant(cache: Path, index: int, kernel: str, suffix: str) -> None:
    """Write one triton cache entry in the layout a real run leaves: one hash dir per compiled variant."""
    entry = cache / f"HASH{index:032d}"
    entry.mkdir(parents=True)
    (entry / f"{kernel}.json").write_text(
        json.dumps({"name": f"{kernel}_{suffix}", "num_warps": 4, "num_stages": 2}),
        encoding="utf-8",
    )
    # Real entries carry a sibling group file and the compiled artifacts; neither should be read as a variant.
    (entry / f"__grp__{kernel}.json").write_text("{}", encoding="utf-8")
    (entry / f"{kernel}.hsaco").write_text("", encoding="utf-8")


def _grid_leak(cache: Path, kernel: str, count: int, *, start: int = 0) -> None:
    """A constant tracking a runtime quantity: one new value per call shape, everything else fixed."""
    for i in range(count):
        _variant(cache, start + i, kernel, f"BLOCK_M_64_GRID_MN_{i * 7 + 1}")


def _tuned_table(cache: Path, kernel: str, start: int) -> None:
    """A tile table: several constants move together, none of them per call."""
    combos = [(m, n, w) for m in (32, 64, 128) for n in (64, 128) for w in (2, 4, 8)]
    for i, (m, n, w) in enumerate(combos):
        _variant(cache, start + i, kernel, f"BLOCK_M_{m}_BLOCK_N_{n}_WARPS_{w}")


def test_a_constant_tracking_a_runtime_quantity_is_called_a_leak(tmp_path):
    _grid_leak(tmp_path, "_fused_fp8_bmm_rope_cat_and_cache_mla_kernel", 40)

    (row,) = jv.variant_counts(tmp_path)

    assert row["kernel"] == "_fused_fp8_bmm_rope_cat_and_cache_mla_kernel"
    assert row["variants"] == 40
    assert row["single_constant_leak"] is True
    widest = row["varying"][0]
    assert widest["after"] == "GRID_MN"
    assert widest["distinct_values"] == 40


def test_a_tile_table_with_many_variants_is_not_called_a_leak(tmp_path):
    _tuned_table(tmp_path, "_gemm_a8w8_blockscale_kernel", start=0)

    (row,) = jv.variant_counts(tmp_path)

    assert row["variants"] == 18
    # No single constant carries the spread: the widest holds 3 of 18.
    assert row["single_constant_leak"] is False
    assert row["varying"][0]["distinct_values"] < row["variants"] * jv.LEAK_VALUE_SHARE


def test_kernels_below_the_floor_are_not_reported(tmp_path):
    _grid_leak(tmp_path, "_small_kernel", jv.MIN_VARIANTS - 1)

    assert jv.variant_counts(tmp_path) == []


def test_the_worst_offender_comes_first(tmp_path):
    _grid_leak(tmp_path, "_quiet_kernel", 20)
    _grid_leak(tmp_path, "_loud_kernel", 60, start=100)

    rows = jv.variant_counts(tmp_path)

    assert [r["kernel"] for r in rows] == ["_loud_kernel", "_quiet_kernel"]
    assert rows[0]["variants"] == 60


def test_an_absent_or_empty_cache_reports_nothing(tmp_path):
    assert jv.variant_counts(tmp_path / "absent") == []
    assert jv.variant_counts(None) == []
    assert jv.variant_counts("") == []
    (tmp_path / "empty").mkdir()
    assert jv.variant_counts(tmp_path / "empty") == []


def test_a_cache_this_process_cannot_stat_reports_nothing(tmp_path):
    """Raising here sank the whole hand-off document, which several other censuses also write."""
    denied = tmp_path / "denied"
    denied.mkdir()
    (tmp_path).chmod(0o000)
    try:
        assert jv.variant_counts(denied) == []
    finally:
        (tmp_path).chmod(0o755)


def test_an_unreadable_entry_does_not_sink_the_census(tmp_path):
    _grid_leak(tmp_path, "_ok_kernel", 20)
    broken = tmp_path / "HASHbroken"
    broken.mkdir()
    (broken / "_ok_kernel.json").write_text("{ not json", encoding="utf-8")

    (row,) = jv.variant_counts(tmp_path)

    assert row["variants"] == 20


def test_the_rendering_separates_a_leak_from_a_table(tmp_path):
    _grid_leak(tmp_path, "_leaky", 40)
    _tuned_table(tmp_path, "_tuned", start=200)

    body = "\n".join(jv.render_jit_variants(jv.variant_counts(tmp_path)))

    assert "`_leaky`" in body and "`_tuned`" in body
    assert "tracks a runtime quantity" in body
    assert "consistent with a tuning table" in body
    assert "GRID_MN" in body
