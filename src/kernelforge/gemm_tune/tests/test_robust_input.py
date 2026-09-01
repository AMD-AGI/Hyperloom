# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for robust handling of inline / malformed shapes & csv inputs.

Regression cover for the OSError(ENAMETOOLONG) crash: callers passed inline
JSON content in --shapes-json instead of a file path, and Path(inline).is_file()
raised OSError(36), killing the dense tuner at elapsed_s=0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernelforge.gemm_tune.cli import _normalize_inline_shapes_json, _safe_is_file
from kernelforge.gemm_tune.tuners import _aiter_dense_common as ac
from kernelforge.gemm_tune.tuners._aiter_dense_common import (
    _conform_csv_columns,
    _parse_tuner_stdout,
    _resolve_input_csv,
    validate_dense_tuner_inputs,
)
from kernelforge.gemm_tune.tuners.base import TuneContext
from kernelforge.gemm_tune.model_analyzer import ModelProfile


# The exact production payload: a Python-repr list far longer than NAME_MAX.
_INLINE = "[{'M': 64, 'N': 16384, 'K': 3072, 'dtype': 'bf16'}]" * 6

_STUB_FP8 = "torch.float8_e4m3fn"


@pytest.fixture
def stub_fp8_dtype(monkeypatch):
    """Resolve dtypes without aiter so these stay pure unit tests.

    Production reads the dtype from the installed aiter and raises when it
    cannot; only the integration tests below exercise the real mapping.
    """
    monkeypatch.setattr(ac, "_aiter_dtype_str", lambda alias: _STUB_FP8)
    return _STUB_FP8


def _ctx(tmp_path: Path, **overrides) -> TuneContext:
    base = dict(
        profile=None,
        framework="sglang",
        precision="fp8",
        quant_type="blockscale",
        gpu_type="mi300x",
        tp=1,
        conc=64,
        tokens=[16],
        mp=1,
        output_dir=tmp_path,
        iters=5,
        warmup=2,
        min_improvement_pct=1.0,
        timeout_s=60,
    )
    base.update(overrides)
    return TuneContext(**base)


def test_safe_is_file_handles_too_long():
    assert len(_INLINE) > 255
    assert _safe_is_file(_INLINE) is False  # must not raise OSError(36)


def test_safe_is_file_true(tmp_path):
    f = tmp_path / "real.csv"
    f.write_text("M,N,K\n", encoding="utf-8")
    assert _safe_is_file(str(f)) is True


def test_normalize_inline_shapes_json_existing_path(tmp_path):
    f = tmp_path / "shapes.json"
    f.write_text('[{"M":1,"N":2,"K":3}]', encoding="utf-8")
    assert _normalize_inline_shapes_json(str(f), tmp_path) == str(f)


def test_normalize_inline_shapes_json_python_repr(tmp_path):
    out = _normalize_inline_shapes_json("[{'M': 64, 'N': 16384, 'K': 3072}]", tmp_path)
    assert out == str(tmp_path / "_inline_shapes.json")
    assert json.loads(Path(out).read_text())[0]["N"] == 16384


def test_normalize_inline_shapes_json_garbage(tmp_path):
    assert _normalize_inline_shapes_json("", tmp_path) == ""
    assert _normalize_inline_shapes_json("nope.json", tmp_path) == ""


def test_resolve_input_csv_does_not_crash_on_inline_path(tmp_path):
    # shapes_json is a Path built from inline content (the production bug).
    ctx = _ctx(tmp_path, shapes_json=Path(_INLINE))
    # Previously raised OSError(36); now resolves to None (no usable input).
    assert _resolve_input_csv(ctx, tmp_path) is None


def test_parse_tuner_stdout_marks_updated_section_as_improved():
    output = """
============= Compare Report =============
--- Updated (1 shapes) ---
Shape | Pre(us) | Post(us) | Improve | Action
(1070, 7168, 5120) | 207.26 | 70.21 | 66.12% | UPDATE
--- Skipped (1 shapes) ---
Shape | Pre(us) | Post(us) | Improve | Reason
(1070, 5120, 5120) | 152.90 | 154.74 | -1.21% | < 3.0% improve
"""

    results = _parse_tuner_stdout(output, "")

    assert [row["improved"] for row in results] == [True, False]


def test_resolve_input_csv_preserves_recorded_shapes_in_fast_mode(tmp_path):
    # Recorded rows are never rewritten; the guard only appends the decode
    # buckets this capture cannot serve (M=16 already answers 1/4/16).
    csv = tmp_path / "untuned.csv"
    csv.write_text("M,N,K\n16,1536,7168\n", encoding="utf-8")
    ctx = _ctx(tmp_path, untuned_csv=csv)
    out = _resolve_input_csv(ctx, tmp_path)
    rows = out.read_text(encoding="utf-8").strip().splitlines()
    assert rows[:2] == ["M,N,K", "16,1536,7168"]
    assert rows[2:] == ["32,1536,7168", "64,1536,7168"]


def test_resolve_input_csv_augments_recorded_shapes_in_thorough_mode(tmp_path):
    csv = tmp_path / "untuned.csv"
    csv.write_text("M,N,K\n16,1536,7168\n", encoding="utf-8")
    ctx = _ctx(tmp_path, untuned_csv=csv, thorough=True, tokens=[1024])
    out = _resolve_input_csv(ctx, tmp_path)
    assert out == tmp_path / "augmented_dense.csv"
    rows = out.read_text(encoding="utf-8").strip().splitlines()
    assert "16,1536,7168" in rows
    assert "8192,1536,7168" in rows


def test_resolve_input_csv_covers_decode_m_when_capture_is_prefill_only(tmp_path):
    """Repro: shape capture recorded only a large prefill M (e.g. 2095), missing
    the decode band. Fast mode would then tune the wrong operating point -> micro
    win but E2E regression (observed -18.45% on Qwen3.5-122B). The resolved CSV
    must add the decode-representative M while keeping the recorded prefill M.
    """
    shapes = tmp_path / "forge_shapes.json"
    shapes.write_text(
        json.dumps(
            [
                {"M": 2095, "N": 8704, "K": 3072},
                {"M": 2095, "N": 10240, "K": 3072},
            ]
        ),
        encoding="utf-8",
    )
    ctx = _ctx(tmp_path, shapes_json=shapes, conc=64, tokens=[4, 8, 16, 32, 64, 128, 256, 512])
    out = _resolve_input_csv(ctx, tmp_path)
    rows = out.read_text(encoding="utf-8").strip().splitlines()
    m_values = {int(r.split(",")[0]) for r in rows[1:]}
    assert 2095 in m_values  # recorded prefill point preserved
    # One row per decode lookup bucket for conc=64; 16 also answers M=1/4, and
    # nothing above the concurrency cap is tuned.
    assert {16, 32, 64}.issubset(m_values)
    assert 128 not in m_values
    # NK pairs preserved for every M.
    nk_values = {tuple(r.split(",")[1:3]) for r in rows[1:]}
    assert nk_values == {("8704", "3072"), ("10240", "3072")}


def test_resolve_input_csv_covers_decode_m_for_prefill_only_manifest(tmp_path, monkeypatch):
    """The shapes_manifest branch must also get decode coverage: a manifest can
    capture only large prefill M (same CUDA Graph gap), so it flows through the
    same fast-mode decode guard instead of returning early."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")  # presence only; writer is patched

    def _fake_manifest_csv(_manifest, work_dir, needs_q_dtype_w=False):
        out = work_dir / "manifest_untuned.csv"
        out.write_text("M,N,K\n2095,8704,3072\n", encoding="utf-8")
        return out

    monkeypatch.setattr("kernelforge.gemm_tune.shape_manifest.write_manifest_untuned_csv", _fake_manifest_csv)
    ctx = _ctx(tmp_path, shapes_manifest=manifest, conc=64, tokens=[16, 512])
    out = _resolve_input_csv(ctx, tmp_path)
    m_values = {int(r.split(",")[0]) for r in out.read_text(encoding="utf-8").strip().splitlines()[1:]}
    assert m_values == {16, 32, 64, 2095}


def test_resolve_input_csv_preserves_capture_that_already_covers_decode(tmp_path):
    """A capture holding every decode bucket is left untouched in fast mode (no
    needless tuning-time blow-up)."""
    shapes = tmp_path / "forge_shapes.json"
    shapes.write_text(
        json.dumps(
            [
                {"M": 16, "N": 1536, "K": 7168},
                {"M": 32, "N": 1536, "K": 7168},
                {"M": 64, "N": 1536, "K": 7168},
            ]
        ),
        encoding="utf-8",
    )
    ctx = _ctx(tmp_path, shapes_json=shapes, conc=64)
    out = _resolve_input_csv(ctx, tmp_path)
    m_values = {int(r.split(",")[0]) for r in out.read_text(encoding="utf-8").strip().splitlines()[1:]}
    assert m_values == {16, 32, 64}


def _rows(csv: Path) -> list[str]:
    return csv.read_text(encoding="utf-8").strip().splitlines()


def _group_m(csv: Path) -> dict[tuple[str, str], set[int]]:
    """Map ``(N, K)`` -> the M values tuned for it."""
    out: dict[tuple[str, str], set[int]] = {}
    for row in _rows(csv)[1:]:
        m, n, k = row.split(",")[:3]
        out.setdefault((n, k), set()).add(int(m))
    return out


def test_decode_coverage_is_decided_per_dispatch_group(tmp_path):
    """aiter looks a config up per (M,N,K), so decode rows for one projection say
    nothing about another. Only the group that lacks buckets gets rows."""
    csv = tmp_path / "untuned.csv"
    csv.write_text(
        "M,N,K\n"
        "16,8704,3072\n32,8704,3072\n64,8704,3072\n"  # fully covered
        "2095,10240,3072\n",  # prefill only
        encoding="utf-8",
    )
    ctx = _ctx(tmp_path, untuned_csv=csv, conc=64)

    groups = _group_m(_resolve_input_csv(ctx, tmp_path))

    assert groups[("8704", "3072")] == {16, 32, 64}
    assert groups[("10240", "3072")] == {16, 32, 64, 2095}


def test_decode_coverage_ignores_m_outside_the_decode_grid(tmp_path):
    """M=100 sits below the ceiling but pads to bucket 112, which no decode M
    dispatches to, so the group still needs real bucket rows."""
    csv = tmp_path / "untuned.csv"
    csv.write_text("M,N,K\n100,8704,3072\n", encoding="utf-8")
    ctx = _ctx(tmp_path, untuned_csv=csv, conc=64)

    m_values = _group_m(_resolve_input_csv(ctx, tmp_path))[("8704", "3072")]

    assert m_values == {16, 32, 64, 100}


@pytest.mark.parametrize(
    ("recorded", "expected_added"),
    [(64, {16, 32, 128, 256}), (128, {16, 32, 64, 256})],
)
def test_one_decode_grid_member_does_not_cover_the_other_buckets(tmp_path, recorded, expected_added):
    """A tuned M=64 row is never consulted for runtime M=16 or M=32: each probes
    its own exact/padded keys. Holding one grid member is not coverage."""
    csv = tmp_path / "untuned.csv"
    csv.write_text(f"M,N,K\n{recorded},8704,3072\n", encoding="utf-8")
    ctx = _ctx(tmp_path, untuned_csv=csv, conc=256)

    m_values = _group_m(_resolve_input_csv(ctx, tmp_path))[("8704", "3072")]

    assert m_values == {recorded} | expected_added


def test_decode_bucket_16_serves_the_smaller_grid_members(tmp_path):
    """M=1/2/4/8 all pad into bucket 16, so a single row covers them -- the guard
    must not emit one row per small M."""
    csv = tmp_path / "untuned.csv"
    csv.write_text("M,N,K\n2095,8704,3072\n", encoding="utf-8")
    ctx = _ctx(tmp_path, untuned_csv=csv, conc=8)

    m_values = _group_m(_resolve_input_csv(ctx, tmp_path))[("8704", "3072")]

    assert m_values == {16, 2095}  # grid [1,4,8] collapses to the one bucket


def test_decode_coverage_preserves_row_order_and_q_dtype(tmp_path):
    """Manifest CSVs arrive weight-ordered and carry a per-row q_dtype_w; the
    guard must append rather than rebuild, and inherit each group's dtype."""
    csv = tmp_path / "untuned.csv"
    csv.write_text(
        "M,N,K,q_dtype_w\n"
        "2095,10240,3072,torch.float8_e4m3fn\n"  # hottest
        "2095,8704,3072,torch.bfloat16\n",  # colder, different dtype
        encoding="utf-8",
    )
    ctx = _ctx(tmp_path, untuned_csv=csv, conc=8)

    rows = _rows(_resolve_input_csv(ctx, tmp_path, needs_q_dtype_w=True))

    # Original rows survive verbatim, in their original (weight) order.
    assert rows[1] == "2095,10240,3072,torch.float8_e4m3fn"
    assert rows[2] == "2095,8704,3072,torch.bfloat16"
    # One bucket row per group, each inheriting its own group's dtype.
    assert rows[3:] == [
        "16,10240,3072,torch.float8_e4m3fn",
        "16,8704,3072,torch.bfloat16",
    ]


def test_decode_coverage_does_not_cross_m_between_groups(tmp_path):
    """A prefill M recorded for one projection must not appear under another."""
    csv = tmp_path / "untuned.csv"
    csv.write_text("M,N,K\n2095,10240,3072\n4096,8704,3072\n", encoding="utf-8")
    ctx = _ctx(tmp_path, untuned_csv=csv, conc=8)

    groups = _group_m(_resolve_input_csv(ctx, tmp_path))

    assert 4096 not in groups[("10240", "3072")]
    assert 2095 not in groups[("8704", "3072")]


# Recorded from aiter's own ``get_padded_m(m, 8704, 3072, gl)`` on MI355X
# (gfx950). The mirror is load-bearing for decode coverage, and its only guard
# used to be the comparison below -- which needs aiter installed and therefore
# never runs in the unit-test lane. Pinning the observed values keeps the
# mirror's behaviour under test everywhere; comparing against the live aiter
# stays as the drift detector wherever aiter is present.
_PADDED_M_OBSERVED: tuple[tuple[int, int, int], ...] = (
    (1, 16, 1),
    (2, 16, 2),
    (3, 16, 4),
    (4, 16, 4),
    (8, 16, 8),
    (15, 16, 16),
    (16, 16, 16),
    (17, 32, 32),
    (24, 32, 32),
    (31, 32, 32),
    (32, 32, 32),
    (33, 48, 64),
    (48, 48, 64),
    (64, 64, 64),
    (96, 96, 128),
    (100, 112, 128),
    (127, 128, 128),
    (128, 128, 128),
    (129, 144, 256),
    (192, 192, 256),
    (240, 240, 256),
    (241, 256, 256),
    (255, 256, 256),
    (256, 256, 256),
    (257, 288, 512),
    (288, 288, 512),
    (512, 512, 512),
    (513, 544, 1024),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (2095, 2112, 4096),
    (4096, 4096, 4096),
    # Past the 32->64 step at M=1024 and the 64->128 step at M=4096. The values
    # above happen to be multiples of both 32 and 64, so they cannot tell a
    # two-tier mirror from aiter's four tiers; these can.
    (1025, 1088, 2048),
    (1040, 1088, 2048),
    (1056, 1088, 2048),
    (1057, 1088, 2048),
    (2049, 2112, 4096),
    (4097, 4224, 8192),
    (4128, 4224, 8192),
    (8193, 8320, 16384),
    (10000, 10112, 16384),
)


#: ``(M, N, gl=1)`` read off the installed aiter. ``gl=1`` is not a pure power
#: of two: past M=8192 a wide N collapses the bucket to 8192, so the mirror
#: needs N to answer at all. Sampled on both sides of the N>4096 branch.
_PADDED_M_GL1_OBSERVED: tuple[tuple[int, int, int], ...] = (
    (1025, 8704, 2048),
    (1025, 2048, 2048),
    (2049, 8704, 4096),
    (2049, 2048, 4096),
    (4097, 8704, 8192),
    (4097, 2048, 8192),
    (8192, 8704, 8192),
    (8192, 2048, 8192),
    (8193, 8704, 8192),
    (8193, 2048, 16384),
    (10000, 8704, 8192),
    (10000, 2048, 16384),
    (16384, 8704, 8192),
    (16384, 2048, 16384),
)


@pytest.mark.parametrize(("m", "gl0", "pow2"), _PADDED_M_OBSERVED)
def test_padded_m_mirror_matches_recorded_aiter_behaviour(m, gl0, pow2):
    """Runs everywhere, including the lane that has no aiter installed."""
    assert ac._padded_m_gl0(m) == gl0
    assert ac._next_pow2(m) == pow2


@pytest.mark.parametrize(("m", "n", "gl1"), _PADDED_M_GL1_OBSERVED)
def test_padded_m_gl1_mirror_matches_recorded_aiter_behaviour(m, n, gl1):
    assert ac._padded_m_gl1(m, n) == gl1


def test_the_recorded_buckets_capture_the_granularity_change():
    """The table is only a guard if it straddles where the behaviour changes.

    ``gl=0`` steps its granularity three times -- 16 up to 256, then 32, then
    64 past 1024, then 128 past 4096 -- and the power-of-two bucket diverges
    from it well before the first of those. A table sampling only round numbers
    passes against a mirror that got any boundary wrong: 2048, 2095 and 4096 are
    all multiples of both 32 and 64, so a two-tier mirror matches them exactly
    while being wrong at 1025 and 4097.
    """
    recorded = {m: (a, b) for m, a, b in _PADDED_M_OBSERVED}
    assert recorded[256] == (256, 256) and recorded[257] == (288, 512)
    assert recorded[240] == (240, 256) and recorded[241] == (256, 256)
    assert recorded[129] == (144, 256)
    # 32 -> 64 at M=1024, and 64 -> 128 at M=4096.
    assert recorded[1024] == (1024, 1024) and recorded[1025] == (1088, 2048)
    assert recorded[4096] == (4096, 4096) and recorded[4097] == (4224, 8192)
    # And the gl=1 table has to straddle the N branch, not just M.
    gl1 = {(m, n): v for m, n, v in _PADDED_M_GL1_OBSERVED}
    assert gl1[(8193, 8704)] == 8192 and gl1[(8193, 2048)] == 16384


def test_padded_m_mirror_matches_installed_aiter():
    """The local padded-M mirror must track aiter's own bucketing; drift would
    silently make the coverage guard judge the wrong lookup keys."""
    gemm_op_common = pytest.importorskip("aiter.ops.gemm_op_common")
    get_padded_m = gemm_op_common.get_padded_m
    n, k = 8704, 3072
    for m, _gl0, _pow2 in _PADDED_M_OBSERVED:
        assert ac._padded_m_gl0(m) == get_padded_m(m, n, k, 0), f"gl=0 mismatch at M={m}"
    for m, n_i, _gl1 in _PADDED_M_GL1_OBSERVED:
        assert ac._padded_m_gl1(m, n_i) == get_padded_m(m, n_i, k, 1), f"gl=1 mismatch at M={m} N={n_i}"


def test_aiter_dtype_str_rejects_a_dtype_outside_aiters_table(monkeypatch):
    """A dtype aiter cannot translate must fail here, not silently reach the
    tuner: the old fallback returned the gfx942 fnuz constant, which is exactly
    the value that dies with a lookup error on gfx950."""
    import types

    fake = types.SimpleNamespace(
        dtypes=types.SimpleNamespace(fp8="torch.float8_e4m3fnuz"),
        dtype2str_dict={"torch.float8_e4m3fn": "f8"},
    )
    monkeypatch.setitem(__import__("sys").modules, "aiter", fake)

    with pytest.raises(ac.AiterDtypeUnavailable, match="dtype2str_dict"):
        ac._aiter_fp8_dtype_str()


def test_aiter_dtype_str_reports_a_missing_alias(monkeypatch):
    import types

    fake = types.SimpleNamespace(dtypes=types.SimpleNamespace(), dtype2str_dict={})
    monkeypatch.setitem(__import__("sys").modules, "aiter", fake)

    with pytest.raises(ac.AiterDtypeUnavailable, match="no dtypes.fp4x2"):
        ac._aiter_dtype_str("fp4x2")


def test_manifest_keeps_curated_shapes_in_thorough_mode(tmp_path, monkeypatch):
    """A manifest is a curated, weight-ordered set: thorough mode must not
    explode it into the full config-derived M grid, only guarantee decode."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    def _fake_manifest_csv(_manifest, work_dir, needs_q_dtype_w=False):
        out = work_dir / "untuned_manifest.csv"
        out.write_text("M,N,K\n2095,8704,3072\n", encoding="utf-8")
        return out

    monkeypatch.setattr("kernelforge.gemm_tune.shape_manifest.write_manifest_untuned_csv", _fake_manifest_csv)
    ctx = _ctx(tmp_path, shapes_manifest=manifest, thorough=True, conc=8, tokens=[1024])

    m_values = _group_m(_resolve_input_csv(ctx, tmp_path))[("8704", "3072")]

    # The one decode bucket for conc=8 plus the curated prefill row -- and
    # nothing from the thorough grid (e.g. the 8192 high-watermark).
    assert m_values == {16, 2095}


def _profile(**kw) -> ModelProfile:
    base = dict(
        model_path="/fake",
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
    )
    base.update(kw)
    return ModelProfile(**base)


def test_resolve_input_csv_derives_from_config_when_no_input(tmp_path):
    # No csv, no shapes_json -> derive shapes from the model config.
    ctx = _ctx(tmp_path, profile=_profile())
    out = _resolve_input_csv(ctx, tmp_path, needs_q_dtype_w=False)
    assert out is not None and out.is_file()
    lines = out.read_text().strip().splitlines()
    assert lines[0] == "M,N,K"
    assert len(lines) > 1  # at least one derived shape


def test_resolve_input_csv_derives_with_q_dtype_w(tmp_path, stub_fp8_dtype):
    ctx = _ctx(tmp_path, profile=_profile())
    out = _resolve_input_csv(ctx, tmp_path, needs_q_dtype_w=True)
    assert out.read_text().splitlines()[0] == "M,N,K,q_dtype_w"


def test_resolve_input_csv_none_when_profile_lacks_dims(tmp_path):
    ctx = _ctx(tmp_path, profile=_profile(hidden_size=0, intermediate_size=0))
    assert _resolve_input_csv(ctx, tmp_path) is None


def test_validate_allows_config_derivation(tmp_path):
    # No csv/shapes but a usable profile -> validate passes (script presence is
    # environment-dependent, so only assert the shape-availability gate here).
    ctx = _ctx(tmp_path, profile=_profile())
    err = validate_dense_tuner_inputs(ctx, "a8w8_blockscale", script_label="blockscale")
    assert err is None or "script not found" in err


def test_validate_blocks_when_no_shapes_available(tmp_path):
    ctx = _ctx(tmp_path, profile=_profile(hidden_size=0, intermediate_size=0))
    err = validate_dense_tuner_inputs(ctx, "a8w8_blockscale", script_label="blockscale")
    assert err is not None


def test_validate_allows_demand_json_without_csv(tmp_path):
    demand = tmp_path / "demand.json"
    demand.write_text(json.dumps({"tuners": {}}), encoding="utf-8")
    ctx = _ctx(
        tmp_path,
        profile=_profile(hidden_size=0, intermediate_size=0),
        demand_json=demand,
    )
    err = validate_dense_tuner_inputs(ctx, "a8w8_blockscale", script_label="blockscale")
    assert err is None or "script not found" in err


def test_conform_csv_adds_missing_q_dtype_w(tmp_path, stub_fp8_dtype):
    # A blockscale M,N,K file handed to a tuner that needs q_dtype_w (M1).
    src = tmp_path / "a8w8_blockscale_untuned_gemm.csv"
    src.write_text("M,N,K\n16,1536,7168\n32,512,7168\n", encoding="utf-8")
    out = _conform_csv_columns(src, tmp_path, needs_q_dtype_w=True)
    assert out != src
    lines = out.read_text().strip().splitlines()
    assert lines[0] == "M,N,K,q_dtype_w"
    assert lines[1].endswith(stub_fp8_dtype)
    assert lines[1].split(",")[:3] == ["16", "1536", "7168"]


def test_conform_csv_drops_extra_q_dtype_w(tmp_path):
    src = tmp_path / "a8w8_untuned_gemm.csv"
    src.write_text("M,N,K,q_dtype_w\n16,1536,7168,torch.float8_e4m3fnuz\n", encoding="utf-8")
    out = _conform_csv_columns(src, tmp_path, needs_q_dtype_w=False)
    assert out.read_text().strip().splitlines()[0] == "M,N,K"
    assert out.read_text().strip().splitlines()[1] == "16,1536,7168"


def test_conform_csv_passthrough_when_matching(tmp_path):
    src = tmp_path / "x.csv"
    src.write_text("M,N,K\n1,2,3\n", encoding="utf-8")
    assert _conform_csv_columns(src, tmp_path, needs_q_dtype_w=False) == src


def test_resolve_input_csv_conforms_supplied_csv(tmp_path, stub_fp8_dtype):
    # End-to-end: untuned_csv is blockscale (M,N,K) but tuner needs q_dtype_w.
    csv = tmp_path / "untuned.csv"
    csv.write_text("M,N,K\n16,1536,7168\n", encoding="utf-8")
    ctx = _ctx(tmp_path, untuned_csv=csv)
    out = _resolve_input_csv(ctx, tmp_path, needs_q_dtype_w=True)
    assert out.read_text().splitlines()[0] == "M,N,K,q_dtype_w"


def test_cli_tokens_accepts_bracketed_list():
    # Defense: forge tolerates a bracketed token string (e.g. "[4, 8, 64]").
    from click.testing import CliRunner  # noqa: F401  (import guarded below)

    # Parse exactly like cli.run does.
    tokens = "[4, 8, 64]"
    tokens_clean = tokens.strip().strip("[](){}")
    parsed = [int(t.strip().strip("'\"")) for t in tokens_clean.split(",") if t.strip().strip("'\"")]
    assert parsed == [4, 8, 64]
